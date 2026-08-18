"""The safety core: allowlist-by-AST validation of model-generated SQL.

Role in architecture: nothing reaches sqlite3 without passing `SqlGuard.check()`.
This is an ALLOWLIST over a parsed syntax tree, not a regex denylist, because a
denylist loses to trivial evasion (`DR/**/OP`, `dRoP`, unicode, nested
statements) while a parser sees the real structure. Layered with the read-only
connection in sql/execute.py: a guard bypass still cannot write.

The checks are individual `GuardRule` objects run in order, so adding a rule is
adding a class - and each rule is independently testable.

Rejects: parse failures, multiple statements, any root that is not SELECT, any
DDL/DML/PRAGMA/ATTACH node anywhere in the tree, unknown tables, a qualified
column reference that does not exist on the table its alias names, functions
outside the allowlist. Injects LIMIT when absent.

In:  candidate SQL string + SchemaInfo.
Out: `GuardResult(ok, sql, reason)` - `sql` is the rewritten, LIMIT-bounded query.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from core import config
from sql.schema import SchemaInfo

# Statement types that must never appear anywhere in the tree - not at the root,
# not inside a CTE, not inside a subquery.
_FORBIDDEN_NODE_NAMES = (
    "Insert", "Update", "Delete", "Drop", "Alter", "Create", "Attach", "Detach",
    "Pragma", "Command", "Transaction", "Commit", "Rollback", "Vacuum", "Analyze",
    "Set", "Use", "Merge", "Copy", "Grant", "Revoke", "TruncateTable", "Reindex",
)
# Some node classes only exist in certain sqlglot versions; resolve defensively.
FORBIDDEN_NODES = tuple(
    cls for cls in (getattr(exp, n, None) for n in _FORBIDDEN_NODE_NAMES) if cls is not None
)

_CALL_RE = re.compile(r"\s*([A-Za-z_][A-Za-z_0-9]*)\s*\(")


@dataclass
class GuardResult:
    ok: bool
    sql: str  # rewritten SQL when ok, original otherwise
    reason: str = ""

    @property
    def verdict(self) -> str:
        """Compact string for the trace."""
        return "PASS" if self.ok else f"REJECT: {self.reason}"


def iter_nodes(tree: exp.Expression):
    """Yield every node in the tree.

    sqlglot changed `walk()` from yielding (node, parent, key) tuples to yielding
    bare nodes across versions; normalise so the guard is version-tolerant.
    """
    for item in tree.walk():
        yield item[0] if isinstance(item, tuple) else item


def function_names(node: exp.Expression) -> set[str]:
    """Every lowercase name a function node could legitimately be called by.

    The check is anchored on what SQLite will actually receive, because sqlglot's
    internal node names are not SQLite names:

      * `strftime('%Y', d)` parses to `TimeToStr` (sql_name "TIME_TO_STR") and
        wraps its argument in `TsOrDsToTimestamp` - an implicit conversion node
        that renders back to a bare column and is not a call at all.
      * `CASE WHEN a THEN b END` is `exp.Case` holding `exp.If` children;
        `exp.If` is a Func subclass that renders as syntax, not a call.

    So: render the node to SQLite, and treat it as a function only if the output
    actually looks like `NAME(...)`. That identifier - plus sqlglot's registered
    aliases - is matched against the allowlist. Unknown functions still parse to
    `Anonymous` and are caught by their literal name, so nothing is weakened:
    anything that reaches sqlite3 as a call is checked as a call.
    """
    if isinstance(node, exp.Anonymous):
        return {(node.name or "").lower()}
    if not isinstance(node, exp.Func):
        return set()
    if isinstance(node, exp.If) and isinstance(node.parent, exp.Case):
        return set()

    try:
        match = _CALL_RE.match(node.sql(dialect="sqlite"))
    except Exception:  # pragma: no cover - unrenderable node
        match = None
    if match is None:
        # Not emitted as a call (implicit cast/conversion inserted by the
        # parser). Its children are still walked separately.
        return set()

    names: set[str] = {match.group(1).lower()}
    try:
        names.update(n.lower() for n in node.sql_names())
    except Exception:  # pragma: no cover - node class without registered names
        pass
    return {n for n in names if n}


# --------------------------------------------------------------------------
# rules
# --------------------------------------------------------------------------
class GuardRule(ABC):
    """One validation check over the parsed statement.

    `violation()` returns a rejection reason, or None if the rule is satisfied.
    """

    @abstractmethod
    def violation(self, tree: exp.Expression, schema: SchemaInfo) -> str | None: ...


class SelectOnlyRule(GuardRule):
    """The root must be a SELECT (or a UNION/subquery of SELECTs)."""

    def violation(self, tree: exp.Expression, schema: SchemaInfo) -> str | None:
        if isinstance(tree, (exp.Select, exp.Union, exp.Subquery)):
            return None
        return f"only SELECT statements are allowed (got {tree.key.upper()})"


class NoWriteNodeRule(GuardRule):
    """No DDL/DML/PRAGMA node anywhere - not even nested inside a CTE."""

    def violation(self, tree: exp.Expression, schema: SchemaInfo) -> str | None:
        for node in iter_nodes(tree):
            if isinstance(node, FORBIDDEN_NODES):
                return f"disallowed statement type '{node.key.upper()}'"
        return None


class KnownTablesRule(GuardRule):
    """Every referenced table must exist in the LIVE introspected schema."""

    @staticmethod
    def _cte_names(tree: exp.Expression) -> set[str]:
        names: set[str] = set()
        for cte in tree.find_all(exp.CTE):
            alias = cte.alias_or_name
            if alias:
                names.add(alias.lower())
        return names

    def violation(self, tree: exp.Expression, schema: SchemaInfo) -> str | None:
        allowed = self._cte_names(tree)
        for table in tree.find_all(exp.Table):
            name = table.name
            if not name or name.lower() in allowed:
                continue
            if not schema.has(name):
                return f"unknown table '{name}'"
            if table.catalog or (table.db and table.db.lower() != "main"):
                # Blocks cross-database reads via ATTACH-style qualified names.
                return f"qualified database access is not allowed ('{name}')"
        return None


class KnownColumnsRule(GuardRule):
    """Every qualified column reference must exist on the table its alias names.

    Runs after KnownTablesRule (whose tables are assumed valid here). The model
    keeps aliasing a column onto the wrong table (`a.owner_role` when `a` is
    `accounts` and `owner_role` lives on `opportunities`/`activities`) - SQLite
    catches this too, but only after a wasted generation call, and its raw
    error names the failing token, not which table the column actually lives
    on. This rejects before execution and tells the repair pass exactly where
    to move the reference.

    Deliberately does not check UNQUALIFIED columns (bare `stage`, no `o.`
    prefix): with more than one table in scope that would require resolving
    which one SQLite means, and getting it wrong would reject a valid query.
    Also skips CTE and derived-subquery aliases - their "columns" are a
    computed SELECT list, not live schema columns, so there is nothing in
    SchemaInfo to check them against (same exclusion KnownTablesRule already
    applies for CTEs).
    """

    def violation(self, tree: exp.Expression, schema: SchemaInfo) -> str | None:
        cte_names = KnownTablesRule._cte_names(tree)
        alias_to_table: dict[str, str] = {}
        for table in tree.find_all(exp.Table):
            name = table.name
            if not name or name.lower() in cte_names:
                continue
            alias_to_table[(table.alias_or_name or name).lower()] = name
            alias_to_table[name.lower()] = name  # `table.col` with no alias too

        derived = {
            d.alias_or_name.lower() for d in tree.find_all(exp.Subquery) if d.alias_or_name
        }

        for col in tree.find_all(exp.Column):
            qualifier = (col.table or "").lower()
            if not qualifier or qualifier in cte_names or qualifier in derived:
                continue  # unqualified, or aliases a CTE/subquery - not checkable
            table = alias_to_table.get(qualifier)
            if table is None:
                return f"'{qualifier}' in '{qualifier}.{col.name}' is not a table or alias in this query"
            if not schema.has(table, col.name):
                carriers = sorted(t for t in schema.table_names if schema.has(t, col.name))
                hint = f"; it exists on: {', '.join(carriers)}" if carriers else ""
                return f"column '{qualifier}.{col.name}' does not exist on '{table}'{hint}"
        return None


class AllowedFunctionsRule(GuardRule):
    """Functions must be on the explicit allowlist in config."""

    def violation(self, tree: exp.Expression, schema: SchemaInfo) -> str | None:
        for node in iter_nodes(tree):
            names = function_names(node)
            if names and not (names & config.SQL_ALLOWED_FUNCTIONS):
                return f"function '{sorted(names)[0]}' is not allowed"
        return None


class ScopedQueryGuardRule(GuardRule):
    """Per-user backstop: reject SQL touching a scoped table without a
    region/segment predicate fully inside the user's allowed set.

    Bound to one user's allowed set at construction time - never share an
    instance of this rule across users (see orchestrator._sql_path_for).

    This is a backstop, not the primary gate: router/rules.py's
    RegionScopeRule already narrows/refuses the resolved slot before SQL
    generation. This rule catches the case where the SQL-generation model
    ignored that resolved filter, or generated a query that has no way to
    honour it at all - `deployments`/`activities` carry no region/segment
    column of their own, so a query touching only those tables would
    otherwise return a cross-region/segment aggregate with nothing for a
    literal-value check to catch. Requiring SOME region/segment predicate
    whenever a scoped user's query touches any of the four business tables
    closes that gap by construction, at the cost of also rejecting a
    deployments/activities query that a user might reasonably expect to run
    unscoped (e.g. joined to accounts but filtered some other way) - an
    accepted trade-off for a fixed, small team rather than building full
    join-graph/predicate-coverage analysis.
    """

    _SCOPED_TABLES = frozenset({"accounts", "opportunities", "deployments", "activities"})
    _SCOPED_COLUMNS = frozenset({"region", "segment"})

    def __init__(self, allowed: frozenset[str] | None) -> None:
        self.allowed = allowed  # None = unrestricted, this rule is a no-op

    def violation(self, tree: exp.Expression, schema: SchemaInfo) -> str | None:
        if self.allowed is None:
            return None
        tables = {t.name.lower() for t in tree.find_all(exp.Table)}
        if not (tables & self._SCOPED_TABLES):
            return None  # query doesn't touch a scoped business table at all

        literals: set[str] = set()
        for node in tree.find_all(exp.EQ):
            col, lit = node.this, node.expression
            if (
                isinstance(col, exp.Column)
                and col.name.lower() in self._SCOPED_COLUMNS
                and isinstance(lit, exp.Literal)
            ):
                literals.add(lit.this)
        for node in tree.find_all(exp.In):
            col = node.this
            if isinstance(col, exp.Column) and col.name.lower() in self._SCOPED_COLUMNS:
                literals.update(
                    e.this for e in node.expressions if isinstance(e, exp.Literal)
                )

        if not literals:
            return "this query must filter on region/segment within your permitted scope"
        outside = literals - self.allowed
        if outside:
            return (
                "query filters on a region/segment outside your allowed scope: "
                + ", ".join(sorted(outside))
            )
        return None


class SqlGuard:
    """Runs every `GuardRule` in order, then bounds the result set."""

    DEFAULT_RULES: tuple[type[GuardRule], ...] = (
        SelectOnlyRule,
        NoWriteNodeRule,
        KnownTablesRule,
        KnownColumnsRule,
        AllowedFunctionsRule,
    )

    def __init__(
        self, schema: SchemaInfo, rules: list[GuardRule] | None = None, row_limit: int | None = None
    ) -> None:
        self.schema = schema
        self.rules = rules if rules is not None else [cls() for cls in self.DEFAULT_RULES]
        self.row_limit = row_limit or config.SQL_ROW_LIMIT

    def check(self, sql: str) -> GuardResult:
        """Validate and normalise model-generated SQL. Never raises."""
        raw = (sql or "").strip().rstrip(";").strip()
        if not raw:
            return GuardResult(False, sql or "", "empty query")

        # Parse first - this is also what detects stacked statements.
        try:
            statements = [s for s in sqlglot.parse(sql, dialect="sqlite") if s is not None]
        except Exception as exc:
            return GuardResult(False, sql, f"could not parse SQL ({exc})")

        if not statements:
            return GuardResult(False, sql, "no statement found")
        if len(statements) > 1:
            return GuardResult(False, sql, "multiple statements are not allowed")

        tree = statements[0]
        for rule in self.rules:
            reason = rule.violation(tree, self.schema)
            if reason:
                return GuardResult(False, sql, reason)

        # Bound the result set. Injected rather than rejected: an unbounded
        # SELECT is a resource risk, not an intent problem, so we fix it silently
        # and record it in the trace.
        if isinstance(tree, exp.Subquery):
            tree = tree.unnest() if hasattr(tree, "unnest") else tree
        if tree.args.get("limit") is None:
            tree = tree.limit(self.row_limit)

        return GuardResult(True, tree.sql(dialect="sqlite"), "")


def guard(sql: str, schema: SchemaInfo) -> GuardResult:
    """Module-level convenience wrapper around `SqlGuard`."""
    return SqlGuard(schema).check(sql)
