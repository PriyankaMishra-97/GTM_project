"""Guard unit tests - the highest-risk component, so tested without any LLM or DB.

A fake SchemaInfo is used deliberately: the guard's contract is "reject anything
not in the schema you were handed", and hard-coding the fake proves that contract
independently of what the shipped database happens to contain.
"""

from __future__ import annotations

import pytest

from sql.guard import ScopedQueryGuardRule, SqlGuard, guard
from sql.schema import Column, SchemaInfo


def _schema() -> SchemaInfo:
    return SchemaInfo(
        tables={
            "opportunities": [
                Column("opportunity_id", "INTEGER", True, True),
                Column("stage", "TEXT", True, False),
                Column("amount_usd", "REAL", True, False),
                Column("region", "TEXT", True, False),
                Column("close_date", "TEXT", False, False),
                Column("account_id", "INTEGER", True, False),
            ],
            "accounts": [
                Column("account_id", "INTEGER", True, True),
                Column("segment", "TEXT", True, False),
            ],
        }
    )


SCHEMA = _schema()


# --------------------------------------------------------------- rejections --
@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE opportunities",
        "DELETE FROM opportunities WHERE 1=1",
        "UPDATE opportunities SET stage = 'Closed Won'",
        "INSERT INTO opportunities (stage) VALUES ('Closed Won')",
        "ALTER TABLE opportunities ADD COLUMN x TEXT",
        "CREATE TABLE evil (a INT)",
        "PRAGMA table_info(opportunities)",
        "ATTACH DATABASE '/etc/passwd' AS leak",
        "VACUUM",
    ],
)
def test_rejects_write_and_ddl(sql: str) -> None:
    result = guard(sql, SCHEMA)
    assert not result.ok, f"guard let through: {sql}"
    assert result.verdict.startswith("REJECT")


def test_rejects_stacked_statements() -> None:
    result = guard("SELECT 1 AS x; DROP TABLE opportunities", SCHEMA)
    assert not result.ok
    assert "multiple statements" in result.reason


def test_rejects_stacked_statement_hidden_in_comment_style() -> None:
    # A regex denylist keyed on "^SELECT" would pass this; the parser does not.
    result = guard(
        "SELECT stage FROM opportunities LIMIT 1;\n-- harmless\nDELETE FROM accounts",
        SCHEMA,
    )
    assert not result.ok


def test_rejects_unknown_table() -> None:
    result = guard("SELECT * FROM salaries LIMIT 5", SCHEMA)
    assert not result.ok
    assert "unknown table 'salaries'" in result.reason


def test_rejects_unknown_table_in_subquery() -> None:
    result = guard(
        "SELECT stage FROM opportunities "
        "WHERE account_id IN (SELECT account_id FROM secret_table) LIMIT 5",
        SCHEMA,
    )
    assert not result.ok
    assert "secret_table" in result.reason


# ------------------------------------------------------- known columns --
def test_rejects_a_column_aliased_onto_the_wrong_table() -> None:
    """Regression: the model aliasing `segment` (accounts-only) onto the
    opportunities alias - the exact shape of a real generation failure."""
    result = guard(
        "SELECT o.segment, o.amount_usd FROM opportunities AS o "
        "JOIN accounts AS a ON o.account_id = a.account_id",
        SCHEMA,
    )
    assert not result.ok
    assert "o.segment" in result.reason
    assert "opportunities" in result.reason
    assert "accounts" in result.reason  # hint names where it actually lives


def test_rejects_an_alias_that_is_never_bound_to_any_table() -> None:
    """Regression: a repair pass once invented `ae.owner_role` where `ae` was
    never introduced in any FROM/JOIN clause."""
    result = guard(
        "SELECT ae.segment FROM opportunities AS o "
        "JOIN accounts AS a ON o.account_id = a.account_id",
        SCHEMA,
    )
    assert not result.ok
    assert "'ae'" in result.reason


def test_known_columns_rule_accepts_the_correctly_aliased_column() -> None:
    result = guard(
        "SELECT a.segment, o.region FROM opportunities AS o "
        "JOIN accounts AS a ON o.account_id = a.account_id",
        SCHEMA,
    )
    assert result.ok, result.reason


def test_known_columns_rule_ignores_unqualified_columns() -> None:
    """No `o.`/`a.` prefix - ambiguous with multiple tables in scope, so this
    rule leaves resolving it to SQLite rather than risk a false rejection."""
    result = guard(
        "SELECT stage FROM opportunities AS o "
        "JOIN accounts AS a ON o.account_id = a.account_id",
        SCHEMA,
    )
    assert result.ok, result.reason


def test_known_columns_rule_ignores_cte_and_subquery_aliases() -> None:
    result = guard(
        "WITH won AS (SELECT amount_usd AS x FROM opportunities WHERE stage = 'Closed Won') "
        "SELECT won.x FROM won",
        SCHEMA,
    )
    assert result.ok, result.reason

    result2 = guard(
        "SELECT sub.x FROM (SELECT amount_usd AS x FROM opportunities) AS sub",
        SCHEMA,
    )
    assert result2.ok, result2.reason


def test_rejects_disallowed_function() -> None:
    result = guard("SELECT load_extension('evil.so') AS x", SCHEMA)
    assert not result.ok
    assert "load_extension" in result.reason


def test_rejects_readfile_function() -> None:
    result = guard("SELECT readfile('/etc/passwd') AS leak", SCHEMA)
    assert not result.ok


def test_rejects_unparseable_sql() -> None:
    result = guard("SELECT FROM WHERE ((((", SCHEMA)
    assert not result.ok


def test_rejects_empty() -> None:
    assert not guard("", SCHEMA).ok
    assert not guard("   ", SCHEMA).ok


# ------------------------------------------------------------------ passes --
def test_clean_select_passes() -> None:
    result = guard(
        "SELECT stage, COUNT(*) AS deals FROM opportunities "
        "WHERE region = 'EMEA' GROUP BY stage LIMIT 10",
        SCHEMA,
    )
    assert result.ok, result.reason
    assert result.verdict == "PASS"


def test_join_and_allowed_functions_pass() -> None:
    result = guard(
        "SELECT a.segment, ROUND(SUM(o.amount_usd), 2) AS pipeline_usd "
        "FROM opportunities AS o JOIN accounts AS a ON o.account_id = a.account_id "
        "WHERE o.close_date >= '2024-01-01' GROUP BY a.segment",
        SCHEMA,
    )
    assert result.ok, result.reason


@pytest.mark.parametrize(
    "sql",
    [
        # Regression: sqlglot normalises strftime -> TimeToStr, whose internal
        # sql_name() ("TIME_TO_STR") is not a name any user typed. Matching only
        # that name rejected valid SQLite and broke both the SQL and HYBRID paths.
        "SELECT COUNT(*) AS n FROM opportunities WHERE STRFTIME('%Y', close_date) = '2024'",
        "SELECT DATE(close_date) AS d FROM opportunities LIMIT 5",
        "SELECT CAST(amount_usd AS REAL) AS amt FROM opportunities LIMIT 5",
        "SELECT COALESCE(close_date, '-') AS d FROM opportunities LIMIT 5",
        "SELECT JULIANDAY(close_date) - JULIANDAY('2024-01-01') AS age FROM opportunities LIMIT 5",
        # CASE is exp.Case holding exp.If children; exp.If is a Func subclass
        # that renders as syntax, not a call. Checking it as one rejected every
        # win-rate query the model wrote.
        "SELECT CAST(COUNT(CASE WHEN stage = 'Closed Won' THEN 1 END) AS REAL) "
        "/ COUNT(*) AS win_rate FROM opportunities",
        "SELECT region, COUNT(DISTINCT account_id) AS n FROM opportunities "
        "GROUP BY region HAVING COUNT(*) > 10",
        "SELECT opp_name, ROW_NUMBER() OVER (PARTITION BY region ORDER BY amount_usd DESC) "
        "AS rn FROM opportunities",
        "SELECT COUNT(*) FILTER (WHERE stage = 'Closed Won') AS won FROM opportunities",
        "SELECT ROUND(AVG(amount_usd), 2) AS avg_amt FROM opportunities "
        "WHERE close_date BETWEEN '2024-01-01' AND '2024-12-31'",
    ],
)
def test_allowlisted_sqlite_functions_pass(sql: str) -> None:
    result = guard(sql, SCHEMA)
    assert result.ok, result.reason


def test_cte_alias_is_not_treated_as_unknown_table() -> None:
    result = guard(
        "WITH won AS (SELECT amount_usd FROM opportunities WHERE stage = 'Closed Won') "
        "SELECT COUNT(*) AS n FROM won",
        SCHEMA,
    )
    assert result.ok, result.reason


# ------------------------------------------------------ LIMIT normalisation --
def test_limit_is_injected_when_missing() -> None:
    result = guard("SELECT stage FROM opportunities", SCHEMA)
    assert result.ok
    assert "LIMIT 200" in result.sql.upper()


def test_existing_limit_is_preserved() -> None:
    result = guard("SELECT stage FROM opportunities LIMIT 7", SCHEMA)
    assert result.ok
    assert "LIMIT 7" in result.sql.upper()
    assert "LIMIT 200" not in result.sql.upper()


# ------------------------------------------------------ per-user data scope --
def _schema_with_deployments() -> SchemaInfo:
    info = _schema()
    info.tables["deployments"] = [
        Column("deployment_id", "INTEGER", True, True),
        Column("account_id", "INTEGER", True, False),
        Column("seats_active", "INTEGER", True, False),
        Column("seats_purchased", "INTEGER", True, False),
    ]
    return info


SCOPED_SCHEMA = _schema_with_deployments()


def _scoped_guard(allowed: frozenset[str] | None) -> SqlGuard:
    return SqlGuard(
        SCOPED_SCHEMA,
        rules=[*[cls() for cls in SqlGuard.DEFAULT_RULES], ScopedQueryGuardRule(allowed)],
    )


def test_scope_rule_passes_a_query_filtered_to_an_allowed_region() -> None:
    result = _scoped_guard(frozenset({"NA"})).check(
        "SELECT COUNT(*) AS n FROM opportunities WHERE region = 'NA'"
    )
    assert result.ok, result.reason


def test_scope_rule_rejects_a_query_filtered_to_a_disallowed_region() -> None:
    result = _scoped_guard(frozenset({"NA"})).check(
        "SELECT COUNT(*) AS n FROM opportunities WHERE region = 'EMEA'"
    )
    assert not result.ok
    assert "EMEA" in result.reason


def test_scope_rule_rejects_a_partial_list_evasion_via_in() -> None:
    """Naming one allowed region alongside a disallowed one must still reject."""
    result = _scoped_guard(frozenset({"NA"})).check(
        "SELECT COUNT(*) AS n FROM opportunities WHERE region IN ('NA', 'EMEA')"
    )
    assert not result.ok
    assert "EMEA" in result.reason


def test_scope_rule_rejects_a_scoped_table_with_no_region_predicate_at_all() -> None:
    """Closes the deployments/activities gap: those tables carry no region/
    segment column, so a query touching them alone has nothing for a literal
    check to catch - requiring SOME predicate rejects it instead."""
    result = _scoped_guard(frozenset({"NA"})).check(
        "SELECT AVG(seats_active * 1.0 / seats_purchased) AS adoption FROM deployments"
    )
    assert not result.ok
    assert "region/segment" in result.reason


def test_scope_rule_is_a_noop_for_an_unrestricted_user() -> None:
    """allowed=None (unrestricted) must not change behavior at all - regression
    guard so the ACL feature never affects a user with no restrictions."""
    result = _scoped_guard(None).check(
        "SELECT AVG(seats_active) AS n FROM deployments"
    )
    assert result.ok, result.reason
    result2 = _scoped_guard(None).check(
        "SELECT COUNT(*) AS n FROM opportunities WHERE region = 'EMEA'"
    )
    assert result2.ok, result2.reason
