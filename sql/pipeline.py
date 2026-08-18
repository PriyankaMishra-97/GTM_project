"""SQL path: generate -> guard -> execute -> (one repair) -> render.

Role in architecture: `SqlPath` is the SQL branch of the router. It is also
reused verbatim by the hybrid path for its `sql_subquestion`, so guard, repair
and verification behaviour cannot drift between the two. Every step writes into
the shared Trace.

Repair is capped at ONE attempt: a local model that misreads the schema twice
will usually keep misreading it, and each attempt costs seconds against a 10s
budget. On second failure the attempted SQL is preserved in the trace.

In:  question, Trace.
Out: `SqlOutcome(answer, result, sql, ok)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.llm_client import LLMClient, LLMUnavailable, get_client
from core.trace import Trace
from sql.execute import QueryExecutor, QueryResult
from sql.generate import SqlGenerator
from sql.guard import SqlGuard
from sql.narrate import ResultRenderer
from sql.schema import SchemaCatalog, SchemaInfo


@dataclass
class SqlOutcome:
    answer: str
    result: QueryResult
    sql: str | None
    ok: bool


class SqlPath:
    """Owns the full SQL branch. Collaborators are injected, so each is testable alone."""

    def __init__(
        self,
        schema: SchemaInfo | SchemaCatalog,
        client: LLMClient | None = None,
        generator: SqlGenerator | None = None,
        guard: SqlGuard | None = None,
        executor: QueryExecutor | None = None,
        renderer: ResultRenderer | None = None,
    ) -> None:
        self.client = client or get_client()
        self.schema = schema.info if isinstance(schema, SchemaCatalog) else schema
        # A catalog carries the card generated from ITS database, so hand it to
        # the generator; a bare SchemaInfo has none, and the generator falls
        # back to the process default.
        card = schema.card if isinstance(schema, SchemaCatalog) else None
        self.generator = generator or SqlGenerator(self.client, schema_card=card)
        self.guard = guard or SqlGuard(self.schema)
        self.executor = executor or QueryExecutor()
        self.renderer = renderer or ResultRenderer(self.client)

    def run(
        self,
        question: str,
        trace: Trace,
        *,
        slots: dict[str, Any] | None = None,
        render_answer: bool = True,
    ) -> SqlOutcome:
        """Execute the full SQL branch for one question."""
        # --- generate -------------------------------------------------------
        try:
            with trace.stage("sql_generate"):
                sql = self.generator.generate(question, slots, trace=trace)
        except LLMUnavailable as exc:
            trace.error(str(exc))
            return SqlOutcome(f"The model is unavailable: {exc}", QueryResult(), None, False)

        trace.generated_sql = sql

        # --- guard ----------------------------------------------------------
        with trace.stage("sql_guard"):
            verdict = self.guard.check(sql)
        trace.guard_verdict = verdict.verdict
        if not verdict.ok:
            trace.error(f"guard rejected: {verdict.reason}")
            return SqlOutcome(
                "I generated a query that failed the safety guard and did not run it.\n\n"
                f"**Guard verdict:** {verdict.reason}\n\n"
                "Only read-only SELECT statements over the known GTM tables are permitted.",
                QueryResult(error=verdict.reason),
                sql,
                False,
            )

        trace.generated_sql = verdict.sql

        # --- execute (+ one repair) -----------------------------------------
        with trace.stage("sql_execute"):
            result = self.executor.execute(verdict.sql)

        if not result.ok:
            result = self._repair(question, verdict.sql, result, trace, slots)
            if not result.ok:
                trace.error(f"sql failed after repair: {result.error}")
                return SqlOutcome(
                    "The query failed to run even after one repair attempt.\n\n"
                    f"**SQLite error:** `{result.error}`\n\n"
                    f"**Attempted SQL:**\n```sql\n{trace.generated_sql}\n```",
                    result,
                    trace.generated_sql,
                    False,
                )

        trace.rows_returned = result.row_count

        if not render_answer:
            # The hybrid path renders through the composer instead.
            return SqlOutcome("", result, trace.generated_sql, True)

        with trace.stage("sql_render"):
            answer, number_ok = self.renderer.render(
                question, trace.generated_sql or "", result, trace=trace
            )
        trace.number_check_passed = number_ok
        return SqlOutcome(answer, result, trace.generated_sql, True)

    def _repair(
        self,
        question: str,
        failed_sql: str,
        result: QueryResult,
        trace: Trace,
        slots: dict[str, Any] | None,
    ) -> QueryResult:
        """One repair attempt driven by the verbatim SQLite error."""
        trace.repair_attempted = True
        try:
            with trace.stage("sql_repair"):
                repaired = self.generator.repair(
                    question, failed_sql, result.error or "", slots, trace=trace
                )
            reguard = self.guard.check(repaired)
            trace.guard_verdict = f"{trace.guard_verdict} -> repair {reguard.verdict}"
            if reguard.ok:
                trace.generated_sql = reguard.sql
                with trace.stage("sql_execute_retry"):
                    return self.executor.execute(reguard.sql)
        except LLMUnavailable as exc:
            trace.error(str(exc))
        return result


def run_sql_path(
    question: str,
    trace: Trace,
    schema: SchemaInfo | SchemaCatalog,
    *,
    slots: dict[str, Any] | None = None,
    client: LLMClient | None = None,
    render_answer: bool = True,
) -> SqlOutcome:
    """Module-level convenience wrapper around `SqlPath`."""
    return SqlPath(schema, client).run(
        question, trace, slots=slots, render_answer=render_answer
    )
