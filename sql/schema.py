"""Two-tier schema knowledge: live introspection + hand-curated business card.

Role in architecture:
  Tier 1 (machine truth) - `introspect()` reads sqlite_master + PRAGMA table_info
  at startup. This is what the guard allowlists against, so a table that does not
  exist in the file can never be referenced, whatever the card says.
  Tier 2 (business truth) - the schema card that goes in the generation prompt.
  Its FACTUAL half (TABLES, ENUM VALUES, JOIN KEYS) is GENERATED from Tier 1 by
  `build_schema_card()`, so what the model is told about columns, enum strings
  and foreign keys is read from the database rather than transcribed by hand.
  Its SEMANTIC half stays curated, because no amount of DDL reveals that "won"
  means stage='Closed Won', that the win-rate denominator is closed deals only,
  or that the tracker's 1-6 stage names are NOT the DB's stage values.

`SchemaCatalog` owns both tiers: it introspects on construction and builds the
card from that same introspection. Generation removes drift by construction for
the factual half; `validate_card()` still guards the hand-written half, whose
prose names real columns and can go stale silently when one is renamed.

Generation is never done at import time - the module-level default is built
lazily by `default_schema_card()` and degrades to `SCHEMA_CARD_FALLBACK` when no
database is reachable, so importing this module (and everything downstream of
it) works on a fresh clone with no assets.

In:  the SQLite file at config.DB_PATH.
Out: `SchemaInfo` (tables -> columns) + the schema card string.
"""

from __future__ import annotations

import re
import sqlite3
import textwrap
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from core import config


class DatabaseMissing(FileNotFoundError):
    """The provided SQLite file is not where the app expects it."""


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    notnull: bool
    pk: bool


@dataclass
class SchemaInfo:
    """Machine-introspected schema. `tables` maps table name -> columns."""

    tables: dict[str, list[Column]] = field(default_factory=dict)

    @property
    def table_names(self) -> set[str]:
        return set(self.tables)

    def has(self, table: str, column: str | None = None) -> bool:
        # SQLite identifiers are case-insensitive; normalise both sides.
        tbl = {t.lower(): cols for t, cols in self.tables.items()}
        if table.lower() not in tbl:
            return False
        if column is None:
            return True
        return column.lower() in {c.name.lower() for c in tbl[table.lower()]}

    def ddl_summary(self) -> str:
        """Compact `table(col type, ...)` listing - used in the router prompt."""
        return "\n".join(
            f"{t}({', '.join(f'{c.name} {c.type}' for c in cols)})"
            for t, cols in sorted(self.tables.items())
        )


def _connect_ro(db_path: Path | None = None) -> sqlite3.Connection:
    """Read-only connection. See sql/execute.py for why this is the ONLY mode."""
    path = Path(db_path or config.DB_PATH)
    if not path.exists():
        raise DatabaseMissing(
            f"SQLite database not found at {path}.\n"
            f"Place the provided `gtm_mock.db` in {config.ASSETS_DIR}/."
        )
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=config.SQL_TIMEOUT_S)


def introspect(db_path: Path | None = None) -> SchemaInfo:
    """Read the live schema. Views are included; internal sqlite_* tables are not."""
    info = SchemaInfo()
    with _connect_ro(db_path) as conn:
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]
        for name in names:
            # PRAGMA is safe here: it is OUR code, not model-generated SQL.
            cols = conn.execute(f'PRAGMA table_info("{name}")').fetchall()
            info.tables[name] = [
                Column(name=c[1], type=c[2] or "", notnull=bool(c[3]), pk=bool(c[5]))
                for c in cols
            ]
    return info


# --------------------------------------------------------------------------
# Business enums, as real Python values (single source of truth for anything
# outside a prompt that needs to validate against them, e.g. core/auth.py's
# per-user region/segment scope).
# --------------------------------------------------------------------------
REGIONS: frozenset[str] = frozenset({"NA", "EMEA", "APAC", "LATAM"})
SEGMENTS: frozenset[str] = frozenset({"Enterprise", "Mid-Market", "SMB"})

# --------------------------------------------------------------------------
# Tier 2, part A: the FACTUAL sections (tables/columns, enum values, join keys)
# are GENERATED from Tier 1 introspection - see build_schema_card(). Only the
# two dicts below are hand-maintained, because neither can be read off DDL:
# a table's business meaning, and the semantic ordering of an enum.
# --------------------------------------------------------------------------

#: Per-table prose. Insertion order is also the DISPLAY order in the card -
#: introspect() returns tables alphabetically, but the curated reading order
#: (account -> deal -> deployment -> line item -> touchpoint) follows the
#: business flow. Live tables missing from this dict are appended at the end
#: with no note, so a swapped database can never hide a table from the model.
TABLE_NOTES: dict[str, str] = {
    "accounts": "One row per customer/prospect account.",
    "opportunities": "One row per deal. region/segment are denormalised onto the deal.",
    "deployments": "One row per account deployment.",
    "opp_products": "Line items on a deal. discount_pct is a fraction (0.15 = 15%).",
    "activities": "Touchpoint log. account_id/opportunity_id may be NULL.",
}

#: Columns whose distinct values are small, closed sets worth spelling out.
ENUM_COLUMNS: frozenset[str] = frozenset({
    "stage", "region", "segment", "product_code", "owner_role", "account_tier",
    "industry", "deployment_status", "deployment_mode", "version",
    "activity_type", "channel", "outcome",
})

#: Rendered as `table.column` rather than a bare column name, because the bare
#: name would be ambiguous or uninformative on its own.
QUALIFIED_ENUMS: frozenset[str] = frozenset({"stage", "version"})

#: Display order for enums whose sequence carries meaning (a funnel, a
#: lifecycle, a size ordering). Alphabetical would destroy that signal, and the
#: model uses it for "later-stage"/"before Commit" style questions. Values found
#: live but absent here are appended alphabetically, so this is a preference,
#: never a filter.
ENUM_ORDER: dict[str, tuple[str, ...]] = {
    "stage": (
        "Prospecting", "Discovery", "Evaluation", "Proposal", "Negotiation",
        "Closed Won", "Closed Lost",
    ),
    "region": ("NA", "EMEA", "APAC", "LATAM"),
    "segment": ("Enterprise", "Mid-Market", "SMB"),
    "deployment_status": ("Not Started", "In Progress", "Live", "Paused", "Churn Risk"),
    "deployment_mode": ("Cloud", "On-Prem", "Hybrid"),
}

#: A column with more distinct values than this is not an enum; spelling it out
#: would bloat the prompt. Guards against a swapped database.
MAX_ENUM_VALUES: int = 30

CARD_HEADER = (
    "DATABASE: {db_name} (SQLite). Synthetic B2B SaaS GTM data. READ ONLY."
)

# --------------------------------------------------------------------------
# Tier 2, part B: hand-written business semantics. An LLM cannot infer any of
# this from DDL, so it stays curated. Split in two because JOIN KEYS sits
# between them in the rendered card and is generated from real foreign keys.
# --------------------------------------------------------------------------
SCHEMA_CARD_SEMANTICS = """\
BUSINESS DEFINITIONS
--------------------
won             : opportunities.stage = 'Closed Won'
lost            : opportunities.stage = 'Closed Lost'
open / in-flight: stage NOT IN ('Closed Won','Closed Lost')
pipeline value  : SUM(amount_usd) over OPEN opportunities
weighted pipeline: SUM(amount_usd * probability) over OPEN opportunities
bookings / closed-won value : SUM(amount_usd) WHERE stage='Closed Won'
win rate        : COUNT(stage='Closed Won') * 1.0 /
                  COUNT(stage IN ('Closed Won','Closed Lost'))   -- closed deals only
ACV / deal size : opportunities.amount_usd (already USD, not per-seat)
seat adoption   : deployments.seats_active * 1.0 / deployments.seats_purchased
net line value  : opp_products.quantity * unit_price_usd * (1 - discount_pct)

COPY THESE PATTERNS EXACTLY (they encode definitions models get wrong)
----------------------------------------------------------------------
-- win rate: denominator is CLOSED deals only, never all rows.
--   COUNT(stage IN (...)) is WRONG - it counts every row, because the boolean
--   is non-NULL. Use SUM(CASE ...) or COUNT(CASE ... THEN 1 END).
SELECT ROUND(
         SUM(CASE WHEN stage = 'Closed Won' THEN 1 ELSE 0 END) * 1.0
         / NULLIF(SUM(CASE WHEN stage IN ('Closed Won','Closed Lost') THEN 1 ELSE 0 END), 0)
       , 4) AS win_rate
FROM opportunities
WHERE close_date BETWEEN '2024-01-01' AND '2024-12-31' AND segment = 'Enterprise';

-- open pipeline value
SELECT ROUND(SUM(amount_usd), 2) AS pipeline_usd
FROM opportunities
WHERE stage NOT IN ('Closed Won','Closed Lost') AND region = 'EMEA';

-- year filter: both forms work; prefer the range (it is index friendly)
WHERE close_date BETWEEN '2024-01-01' AND '2024-12-31'

DATES
-----
All dates are TEXT in 'YYYY-MM-DD' form, so string comparison == date comparison.
Use date(...) / strftime('%Y-%m', col) for bucketing.
opportunities.created_date spans 2023-01-01 .. 2024-12-21
opportunities.close_date   spans 2023-02-23 .. 2025-09-09 (NULL while open)
"Closed in <period>" filters on close_date; "created in <period>" on created_date.
There is no "current date" in this data - never use date('now'); always use the
explicit range the user gave.
"""

SCHEMA_CARD_CAUTION = """\
CAUTION
-------
The Opportunity Tracker Field Guide describes a 6-stage playbook
(1-Qualify .. 6-Closed Won/Handoff). Those names DO NOT exist in this database.
Never invent them in SQL; use the opportunities.stage enum above.

owner_role is a ROLE CATEGORY (AE, SDR, SE, CSM, Partner Manager), not an
individual person. There is no per-salesperson name or ID anywhere in this
schema - a question asking for a specific rep ("which AE...") cannot be
answered per-individual; at most, group by owner_role itself.
"""

#: Used ONLY when no database is reachable, so that importing this module (and
#: everything downstream of it) never requires the DB file to exist - see the
#: module docstring. Deliberately carries no table/column/enum facts: inventing
#: them here would recreate exactly the drift this module now prevents.
SCHEMA_CARD_FALLBACK = (
    "DATABASE: (unavailable - no SQLite file could be opened, so the live "
    "tables, columns, enum values and join keys could not be read).\n\n"
    + SCHEMA_CARD_SEMANTICS
    + "\n"
    + SCHEMA_CARD_CAUTION
)


# --------------------------------------------------------------------------
# Card generation: read the facts once, then render them. The readers touch the
# DB; the renderers are pure functions of what was read, so every line of card
# formatting is unit-testable without a database.
# --------------------------------------------------------------------------
@dataclass
class SchemaFacts:
    """Everything the factual card sections need, read in one pass."""

    row_counts: dict[str, int] = field(default_factory=dict)
    #: table -> from_column -> (referenced_table, referenced_column)
    foreign_keys: dict[str, dict[str, tuple[str, str]]] = field(default_factory=dict)
    #: enum label (bare column, or table.column) -> ordered distinct values
    enums: dict[str, list[str]] = field(default_factory=dict)


def _ordered_tables(info: SchemaInfo) -> list[str]:
    """Curated order first, then any live table the notes don't know about."""
    known = [t for t in TABLE_NOTES if t in info.tables]
    extra = sorted(set(info.tables) - set(known))
    return known + extra


def _sort_enum(column: str, values: list[str]) -> list[str]:
    """Curated order where the sequence means something, else alphabetical.

    Values present live but absent from ENUM_ORDER are appended rather than
    dropped: the curated list is a display preference, never a filter.
    """
    preferred = ENUM_ORDER.get(column, ())
    ranked = [v for v in preferred if v in values]
    rest = sorted(v for v in values if v not in set(preferred))
    return ranked + rest


def read_schema_facts(conn: sqlite3.Connection, info: SchemaInfo) -> SchemaFacts:
    """Single read pass: row counts, foreign keys, enum domains.

    Table and column identifiers come from introspection, never from user or
    model input - the same trust boundary as the PRAGMA in `introspect()`.
    """
    facts = SchemaFacts()
    for table in _ordered_tables(info):
        facts.row_counts[table] = conn.execute(
            f'SELECT COUNT(*) FROM "{table}"'
        ).fetchone()[0]

        fks: dict[str, tuple[str, str]] = {}
        for row in conn.execute(f'PRAGMA foreign_key_list("{table}")'):
            # (id, seq, ref_table, from_col, to_col, on_update, on_delete, match)
            fks[row[3]] = (row[2], row[4])
        if fks:
            facts.foreign_keys[table] = fks

    # Enum domains are unioned across every table carrying the column: region,
    # segment, owner_role and deployment_status each live on two tables, so
    # reading only one could silently omit a legal value.
    domains: dict[str, set[str]] = {}
    for table in _ordered_tables(info):
        for col in info.tables[table]:
            if col.name not in ENUM_COLUMNS:
                continue
            label = f"{table}.{col.name}" if col.name in QUALIFIED_ENUMS else col.name
            rows = conn.execute(
                f'SELECT DISTINCT "{col.name}" FROM "{table}" '
                f'WHERE "{col.name}" IS NOT NULL ORDER BY 1'
            ).fetchall()
            domains.setdefault(label, set()).update(str(r[0]) for r in rows)

    for label, values in domains.items():
        if len(values) > MAX_ENUM_VALUES:
            continue  # too many distinct values to be a useful closed set
        column = label.split(".")[-1]
        facts.enums[label] = _sort_enum(column, sorted(values))
    return facts


def _wrap_items(
    items: list[str],
    first_prefix: str,
    cont_prefix: str,
    width: int = 79,
    suffix: str = "",
) -> list[str]:
    """Pack comma-separated items into lines, never splitting an item.

    textwrap breaks on any whitespace, which would split `col FK->other.col`
    across lines and - worse - split a quoted enum value like 'Churn Risk'.
    Either misrepresents the schema to a model told the strings are literal.
    """
    lines: list[str] = []
    current = first_prefix
    started = False
    for index, item in enumerate(items):
        token = item + ("," if index < len(items) - 1 else suffix)
        if started and len(current) + 1 + len(token) > width:
            lines.append(current)
            current = cont_prefix + token
        else:
            current += token if not started else " " + token
            started = True
    lines.append(current)
    return lines


def render_tables(info: SchemaInfo, facts: SchemaFacts) -> str:
    """`table(col PK, col FK->other.col, ...)` blocks plus note and row count."""
    blocks: list[str] = ["TABLES", "------"]
    for table in _ordered_tables(info):
        fks = facts.foreign_keys.get(table, {})
        parts: list[str] = []
        for col in info.tables[table]:
            if col.pk:
                parts.append(f"{col.name} PK")
            elif col.name in fks:
                ref_table, ref_col = fks[col.name]
                parts.append(f"{col.name} FK->{ref_table}.{ref_col}")
            else:
                parts.append(col.name)

        blocks.extend(
            _wrap_items(
                parts,
                first_prefix=f"{table}(",
                cont_prefix=" " * (len(table) + 1),
                suffix=")",
            )
        )

        note = TABLE_NOTES.get(table, "")
        count = facts.row_counts.get(table)
        detail = " ".join(x for x in (note, f"{count} rows." if count is not None else "") if x)
        if detail:
            blocks.append(f"    {detail}")
        blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"


def render_enums(facts: SchemaFacts) -> str:
    """`column : 'A', 'B', ...` lines, wrapped and aligned like the curated card."""
    header = [
        "ENUM VALUES (exact strings - match them literally, they are case sensitive)",
        "-" * 74,
    ]
    if not facts.enums:
        return "\n".join(header) + "\n(none detected)\n"

    pad = max(len(label) for label in facts.enums) + 1
    lines: list[str] = []
    for label, values in facts.enums.items():
        lines.extend(
            _wrap_items(
                [f"'{v}'" for v in values],
                first_prefix=f"{label.ljust(pad)}: ",
                cont_prefix=" " * (pad + 2),
            )
        )
    return "\n".join(header + lines) + "\n"


def render_join_keys(facts: SchemaFacts) -> str:
    """Derived from real foreign keys, so the join graph can never drift."""
    pairs: list[tuple[str, str]] = []
    for table, fks in facts.foreign_keys.items():
        for from_col, (ref_table, ref_col) in fks.items():
            pairs.append((f"{table}.{from_col}", f"{ref_table}.{ref_col}"))
    if not pairs:
        return "JOIN KEYS\n---------\n(no foreign keys declared)\n"

    pad = max(len(left) for left, _ in pairs) + 1
    lines = [f"{left.ljust(pad)}= {right}" for left, right in pairs]
    return "\n".join(["JOIN KEYS", "---------", *lines]) + "\n"


def build_schema_card(db_path: Path | None = None) -> str:
    """Assemble the full card: generated facts + hand-written semantics.

    Byte-stable for a given database (fixed table order, ORDER BY on every
    distinct read), which matters because everything else in this system is
    pinned for determinism.
    """
    path = Path(db_path or config.DB_PATH)
    info = introspect(path)
    conn = _connect_ro(path)
    try:
        facts = read_schema_facts(conn, info)
    finally:
        conn.close()

    return "\n".join(
        [
            CARD_HEADER.format(db_name=path.name),
            "",
            render_tables(info, facts),
            render_enums(facts),
            SCHEMA_CARD_SEMANTICS,
            render_join_keys(facts),
            SCHEMA_CARD_CAUTION,
        ]
    )


@lru_cache(maxsize=1)
def default_schema_card() -> str:
    """Process-wide default card, built once from the configured database.

    Callers that hold a `SchemaCatalog` should use `catalog.card` instead; this
    exists for the module-level convenience wrappers, which have no catalog.
    Degrades to the fact-free fallback rather than raising, so a missing
    database stays a rendered warning instead of an import-time crash.
    """
    try:
        return build_schema_card()
    except DatabaseMissing:
        return SCHEMA_CARD_FALLBACK


class SchemaCatalog:
    """Owns both schema tiers for one database file.

    Introspects once on construction (the live schema is the guard's allowlist)
    and exposes the card - factual sections generated from that introspection -
    plus a drift check over the hand-written half.
    """

    def __init__(self, db_path: Path | None = None, card: str | None = None) -> None:
        self.db_path = Path(db_path or config.DB_PATH)
        self.info: SchemaInfo = introspect(self.db_path)
        self.card = card if card is not None else build_schema_card(self.db_path)

    @property
    def tables(self) -> dict[str, list[Column]]:
        return self.info.tables

    @property
    def table_names(self) -> set[str]:
        return self.info.table_names

    def has(self, table: str, column: str | None = None) -> bool:
        return self.info.has(table, column)

    def drift_warnings(self) -> list[str]:
        """Where the curated card and the live file disagree."""
        return validate_card(self.info, self.card)

    def refresh(self) -> "SchemaCatalog":
        """Re-introspect (used after the DB file is swapped)."""
        self.info = introspect(self.db_path)
        return self


#: `table.column` mentions inside the hand-written prose (e.g. "won :
#: opportunities.stage = 'Closed Won'"). Anchored on a word boundary so
#: "0.15" or "v3.0" cannot match.
_QUALIFIED_REF_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\b")


def validate_card(info: SchemaInfo, card: str | None = None) -> list[str]:
    """Compare a schema card against live introspection; return drift warnings.

    The card's factual sections are generated from `info` itself, so they can no
    longer drift - the checks below therefore earn their keep on the parts that
    still can: the hand-written business definitions, worked SQL examples and
    date notes, which name real columns (`opportunities.stage`,
    `deployments.seats_active`, ...) and go stale silently when a column is
    renamed. The table-block checks are retained for custom injected cards.
    """
    card = card if card is not None else default_schema_card()
    warnings: list[str] = []
    # Table blocks in the card look like `name(col, col, ...)` at line start.
    for match in re.finditer(r"^(\w+)\(([^)]*)\)", card, flags=re.MULTILINE):
        table, cols_blob = match.group(1), match.group(2)
        if not info.has(table):
            warnings.append(f"schema card references unknown table '{table}'")
            continue
        for raw in cols_blob.split(","):
            col = raw.strip().split()[0] if raw.strip() else ""
            col = col.split("->")[0].strip()
            if col and not info.has(table, col):
                warnings.append(f"schema card references unknown column '{table}.{col}'")
    for table in sorted(info.table_names):
        if not re.search(rf"^{re.escape(table)}\(", card, flags=re.MULTILINE):
            warnings.append(f"table '{table}' exists in the DB but is missing from the card")

    # Prose references: only flag `known_table.unknown_column`, never an unknown
    # prefix - the card is full of non-schema dotted text (file names, versions,
    # "1-Qualify .. 6-Closed Won") that must not be mistaken for a column.
    seen: set[str] = set()
    for match in _QUALIFIED_REF_RE.finditer(card):
        table, col = match.group(1), match.group(2)
        ref = f"{table}.{col}"
        if ref in seen or not info.has(table) or info.has(table, col):
            continue
        seen.add(ref)
        warnings.append(f"schema card prose references unknown column '{ref}'")
    return warnings
