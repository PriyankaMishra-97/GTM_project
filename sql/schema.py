"""Two-tier schema knowledge: live introspection + hand-curated business card.

Role in architecture:
  Tier 1 (machine truth) - `introspect()` reads sqlite_master + PRAGMA table_info
  at startup. This is what the guard allowlists against, so a table that does not
  exist in the file can never be referenced, whatever the card says.
  Tier 2 (business truth) - SCHEMA_CARD, a prose description with the definitions
  an LLM cannot infer from DDL ("won" means stage='Closed Won', the tracker's
  1-6 stage names are NOT the DB's stage values, etc.). This is what goes in the
  generation prompt.

`SchemaCatalog` owns both tiers: it introspects on construction and can report
drift between the card and the live file, so a rotting card is a logged warning
rather than a wrong answer.

In:  the SQLite file at config.DB_PATH.
Out: `SchemaInfo` (tables -> columns) + the schema card string.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
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
# Tier 2: hand-curated schema card (business semantics)
# --------------------------------------------------------------------------
SCHEMA_CARD = """\
DATABASE: gtm_mock.db (SQLite). Synthetic B2B SaaS GTM data. READ ONLY.

TABLES
------
accounts(account_id PK, account_name, segment, region, industry, employee_count,
         annual_revenue_usd, account_tier, created_date, last_activity_date,
         deployment_status, primary_product_code)
    One row per customer/prospect account. 1200 rows.

opportunities(opportunity_id PK, account_id FK->accounts.account_id, opp_name,
              stage, amount_usd, probability, created_date, close_date, region,
              segment, product_code, next_step, last_activity_date, owner_role)
    One row per deal. 1800 rows. region/segment are denormalised onto the deal.

deployments(deployment_id PK, account_id FK->accounts.account_id, deployment_mode,
            deployment_status, start_date, go_live_date, seats_purchased,
            seats_active, version)
    One row per account deployment. 1200 rows.

opp_products(opp_product_id PK, opportunity_id FK->opportunities.opportunity_id,
             product_code, quantity, unit_price_usd, discount_pct)
    Line items on a deal. 2408 rows. discount_pct is a fraction (0.15 = 15%).

activities(activity_id PK, account_id FK, opportunity_id FK, activity_type,
           activity_date, channel, notes, outcome, owner_role)
    Touchpoint log. 5344 rows. account_id/opportunity_id may be NULL.

ENUM VALUES (exact strings - match them literally, they are case sensitive)
--------------------------------------------------------------------------
opportunities.stage : 'Prospecting', 'Discovery', 'Evaluation', 'Proposal',
                      'Negotiation', 'Closed Won', 'Closed Lost'
region              : 'NA', 'EMEA', 'APAC', 'LATAM'
segment             : 'Enterprise', 'Mid-Market', 'SMB'
product_code        : 'XYZ-CORE', 'XYZ-ANALYTICS', 'XYZ-AUTOMATION', 'XYZ-SECURITY'
owner_role          : 'AE', 'SDR', 'SE', 'CSM', 'Partner Manager'
account_tier        : 'Tier 1', 'Tier 2', 'Tier 3'
industry            : 'SaaS', 'FinTech', 'HealthTech', 'Retail', 'Manufacturing',
                      'Logistics', 'EdTech', 'Energy'
deployment_status   : 'Not Started', 'In Progress', 'Live', 'Paused', 'Churn Risk'
deployment_mode     : 'Cloud', 'On-Prem', 'Hybrid'
deployments.version : '1.9', '2.0', '2.1', '2.2', '3.0'
activity_type       : 'Call', 'Email', 'Meeting', 'Demo', 'Workshop', 'QBR', 'Onsite'
channel             : 'Email', 'Phone', 'Zoom', 'Slack', 'In-Person'
outcome             : 'Positive', 'Neutral', 'Needs Follow-up', 'Blocked', 'No Show'

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

JOIN KEYS
---------
opportunities.account_id   = accounts.account_id
deployments.account_id     = accounts.account_id
opp_products.opportunity_id = opportunities.opportunity_id
activities.account_id      = accounts.account_id
activities.opportunity_id  = opportunities.opportunity_id

CAUTION
-------
The Opportunity Tracker Field Guide describes a 6-stage playbook
(1-Qualify .. 6-Closed Won/Handoff). Those names DO NOT exist in this database.
Never invent them in SQL; use the opportunities.stage enum above.
"""


class SchemaCatalog:
    """Owns both schema tiers for one database file.

    Introspects once on construction (the live schema is the guard's allowlist)
    and exposes the curated card plus a drift check against it.
    """

    def __init__(self, db_path: Path | None = None, card: str | None = None) -> None:
        self.db_path = Path(db_path or config.DB_PATH)
        self.card = card if card is not None else SCHEMA_CARD
        self.info: SchemaInfo = introspect(self.db_path)

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


def validate_card(info: SchemaInfo, card: str | None = None) -> list[str]:
    """Compare the curated card against live introspection; return drift warnings.

    Cheap insurance: if someone swaps the DB, the card's claims about tables and
    columns stop being true and every generated query silently degrades. This
    surfaces that at startup instead.
    """
    card = card if card is not None else SCHEMA_CARD
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
    return warnings
