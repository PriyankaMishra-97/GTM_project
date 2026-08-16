"""The safety core: allowlist-by-AST validation of model-generated SQL.

Role in architecture: nothing reaches sqlite3 without passing `SqlGuard.check()`.
This is an ALLOWLIST over a parsed syntax tree, not a regex denylist, because a
denylist loses to trivial evasion (`DR/**/OP`, `dRoP`, unicode, nested
statements) while a parser sees the real structure. Layered with the read-only
connection in sql/execute.py: a guard bypass still cannot write.

The checks are individual `GuardRule` objects run in order, so adding a rule is
adding a class - and each rule is independently testable.

Rejects: parse failures, multiple statements, any root that is not SELECT, any
DDL/DML/PRAGMA/ATTACH node anywhere in the tree, unknown tables, functions
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


class AllowedFunctionsRule(GuardRule):
    """Functions must be on the explicit allowlist in config."""

    def violation(self, tree: exp.Expression, schema: SchemaInfo) -> str | None:
        for node in iter_nodes(tree):
            names = function_names(node)
            if names and not (names & config.SQL_ALLOWED_FUNCTIONS):
                return f"function '{sorted(names)[0]}' is not allowed"
        return None


class SqlGuard:
    """Runs every `GuardRule` in order, then bounds the result set."""

    DEFAULT_RULES: tuple[type[GuardRule], ...] = (
        SelectOnlyRule,
        NoWriteNodeRule,
        KnownTablesRule,
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
