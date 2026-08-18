"""Integration tests against the PROVIDED assets/gtm_mock.db.

The database is static and synthetic, so it is a stable oracle: the expected
values below were observed by querying the shipped file and are hard-coded on
purpose. If they ever fail, either the DB was swapped or the read path changed -
both are things a test should catch.

Skipped wholesale when the DB is absent so a fresh clone without assets does not
error.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from core import config
from sql.execute import connect_readonly, execute
from sql.guard import guard
from sql.schema import introspect, validate_card

pytestmark = pytest.mark.skipif(
    not Path(config.DB_PATH).exists(), reason="provided DB not present"
)


# ------------------------------------------------------------ read-only ----
def test_connection_is_read_only() -> None:
    """The second safety layer: writes fail at the driver, not just at the guard."""
    conn = connect_readonly()
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("UPDATE opportunities SET stage = 'Closed Won'")
    finally:
        conn.close()


def test_ddl_also_fails_on_the_connection() -> None:
    conn = connect_readonly()
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE evil (a INT)")
    finally:
        conn.close()


# ---------------------------------------------------------- introspection --
def test_introspection_finds_the_expected_tables() -> None:
    info = introspect()
    assert info.table_names == {
        "accounts", "opportunities", "deployments", "opp_products", "activities"
    }


def test_introspection_finds_expected_columns() -> None:
    info = introspect()
    assert info.has("opportunities", "amount_usd")
    assert info.has("opportunities", "stage")
    assert info.has("accounts", "segment")
    assert not info.has("opportunities", "salary")


def test_schema_card_matches_the_live_schema() -> None:
    """Guards against curated-card drift, the failure mode called out in the README."""
    assert validate_card(introspect()) == []


# ------------------------------------------------------ known-answer rows --
@pytest.mark.parametrize(
    "table,expected",
    [
        ("accounts", 1200),
        ("opportunities", 1800),
        ("deployments", 1200),
        ("opp_products", 2408),
        ("activities", 5344),
    ],
)
def test_row_counts(table: str, expected: int) -> None:
    result = execute(f"SELECT COUNT(*) AS n FROM {table}")
    assert result.ok, result.error
    assert result.rows[0][0] == expected


def test_known_aggregate_closed_won_emea_2024() -> None:
    result = execute(
        "SELECT COUNT(*) AS won_deals FROM opportunities "
        "WHERE stage = 'Closed Won' AND region = 'EMEA' "
        "AND close_date BETWEEN '2024-01-01' AND '2024-12-31'"
    )
    assert result.ok, result.error
    assert result.rows[0][0] == 28


def test_known_aggregate_stage_distribution() -> None:
    result = execute(
        "SELECT stage, COUNT(*) AS deals FROM opportunities GROUP BY stage ORDER BY stage"
    )
    assert result.ok, result.error
    assert result.rows == [
        ["Closed Lost", 155],
        ["Closed Won", 179],
        ["Discovery", 287],
        ["Evaluation", 326],
        ["Negotiation", 243],
        ["Proposal", 277],
        ["Prospecting", 333],
    ]


def test_known_aggregate_bookings_2024() -> None:
    result = execute(
        "SELECT ROUND(SUM(amount_usd), 2) AS bookings_usd FROM opportunities "
        "WHERE stage = 'Closed Won' AND close_date LIKE '2024%'"
    )
    assert result.ok, result.error
    assert result.rows[0][0] == pytest.approx(4043988.70, abs=0.01)


# ------------------------------------------------- guard + execute together --
def test_guard_approved_query_runs_end_to_end() -> None:
    verdict = guard(
        "SELECT region, COUNT(*) AS deals FROM opportunities GROUP BY region ORDER BY region",
        introspect(),
    )
    assert verdict.ok, verdict.reason
    assert "LIMIT 200" in verdict.sql.upper()  # injected
    result = execute(verdict.sql)
    assert result.ok, result.error
    assert result.rows == [["APAC", 373], ["EMEA", 430], ["LATAM", 174], ["NA", 823]]


def test_guard_blocks_a_write_before_it_reaches_sqlite() -> None:
    verdict = guard("DELETE FROM opportunities", introspect())
    assert not verdict.ok
    assert "DELETE" in verdict.reason.upper()


def test_row_cap_is_enforced() -> None:
    result = execute("SELECT opportunity_id FROM opportunities")
    assert result.ok, result.error
    assert result.row_count <= config.SQL_MAX_ROWS_RENDERED
    assert result.truncated
