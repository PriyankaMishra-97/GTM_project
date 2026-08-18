"""Read-only SQL execution - the system's only code-execution surface.

Role in architecture: second safety layer under sql/guard.py. `QueryExecutor`
opens the connection with SQLite's `mode=ro` URI, which makes writes fail at the
driver level. Even a guard bypass (a write we failed to recognise in the AST)
cannot mutate the file. There is no eval/exec anywhere else in the project.

In:  a guard-approved SELECT.
Out: `QueryResult(columns, rows, error)` - errors are returned, not raised, so the
     repair loop in sql/pipeline.py can feed the SQLite message back to the model.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core import config, safety
from sql.schema import DatabaseMissing


@dataclass
class QueryResult:
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    error: str | None = None
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_markdown(self, max_rows: int | None = None) -> str:
        """Render as a markdown table (used when no narrator LLM call is needed)."""
        if not self.columns:
            return "_(no columns)_"
        limit = max_rows if max_rows is not None else config.SQL_MAX_ROWS_RENDERED
        shown = self.rows[:limit]
        head = "| " + " | ".join(str(c) for c in self.columns) + " |"
        sep = "| " + " | ".join("---" for _ in self.columns) + " |"
        body = [
            "| " + " | ".join(_fmt(v) for v in row) + " |" for row in shown
        ]
        table = "\n".join([head, sep, *body])
        if len(self.rows) > len(shown):
            table += f"\n\n_(showing {len(shown)} of {len(self.rows)} rows)_"
        return table

    def flat_values(self) -> list[Any]:
        """Every scalar in the result - the oracle for the verbatim-number check."""
        return [v for row in self.rows for v in row]


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        # Trim float noise (SQLite AVG returns e.g. 61234.567890123) without
        # changing the value the number-check compares against: verify.py
        # normalises both sides the same way.
        return f"{value:.2f}".rstrip("0").rstrip(".") if abs(value) < 1e12 else str(value)
    return str(value)


class QueryExecutor:
    """Opens read-only connections and runs guard-approved SELECTs."""

    def __init__(
        self,
        db_path: Path | None = None,
        timeout_s: int | None = None,
        max_rows: int | None = None,
    ) -> None:
        self.db_path = Path(db_path or config.DB_PATH)
        self.timeout_s = timeout_s or config.SQL_TIMEOUT_S
        self.max_rows = max_rows or config.SQL_MAX_ROWS_RENDERED

    def connect(self) -> sqlite3.Connection:
        """Open the provided DB read-only. Raises DatabaseMissing with a fix hint."""
        if not self.db_path.exists():
            raise DatabaseMissing(
                f"SQLite database not found at {self.db_path}.\n"
                f"Place the provided `gtm_mock.db` in {config.ASSETS_DIR}/."
            )
        return sqlite3.connect(
            f"file:{self.db_path}?mode=ro", uri=True, timeout=self.timeout_s
        )

    def _install_deadline(self, conn: sqlite3.Connection) -> None:
        """Interrupt anything that outruns the latency budget instead of hanging the UI."""
        state = {"n": 0}
        budget = self.timeout_s * 1000

        def _progress() -> int:
            state["n"] += 1
            return 1 if state["n"] > budget else 0

        conn.set_progress_handler(_progress, 1000)

    def execute(self, sql: str) -> QueryResult:
        """Run a guard-approved SELECT and return rows (never raises on SQL errors)."""
        try:
            conn = self.connect()
        except DatabaseMissing as exc:
            return QueryResult(error=str(exc))

        try:
            self._install_deadline(conn)
            cursor = conn.execute(sql)
            columns = [d[0] for d in (cursor.description or [])]
            raw_rows = cursor.fetchmany(self.max_rows + 1)
            truncated = len(raw_rows) > self.max_rows
            rows = [list(r) for r in raw_rows[: self.max_rows]]
            columns, rows = safety.strip_denied_columns(columns, rows)
            return QueryResult(columns=columns, rows=rows, truncated=truncated)
        except sqlite3.Error as exc:
            return QueryResult(error=f"{type(exc).__name__}: {exc}")
        finally:
            conn.close()


def connect_readonly(db_path: Path | None = None) -> sqlite3.Connection:
    """Module-level convenience wrapper around `QueryExecutor.connect`."""
    return QueryExecutor(db_path).connect()


def execute(sql: str, db_path: Path | None = None) -> QueryResult:
    """Module-level convenience wrapper around `QueryExecutor.execute`."""
    return QueryExecutor(db_path).execute(sql)
