"""Hybrid path: a FIXED four-step pipeline, deliberately not a ReAct loop.

    1. router already split the question into sql_subquestion + doc_subquestion
    2. run the SQL path on sql_subquestion (guard + repair included)
    3. condition doc retrieval on a one-line summary of the SQL result
    4. compose, then verify every number

Why fixed instead of agentic: a ReAct loop over a local 7B costs 3-6 LLM calls
(blowing the 10s target), is non-deterministic turn to turn, and its extra
freedom buys nothing here - the question decomposition is already done by the
router. A fixed pipeline is inspectable, cheap and reproducible. The cost is
stated honestly in the README's limitations: it cannot iterate SQL <-> docs.

Step 3 matters: retrieving on the bare doc_subquestion loses the result's
vocabulary. Appending "result: Negotiation 128 deals, $9.4M" pulls in the chunks
about the stages/statuses that actually appeared.

In:  question + RouterDecision fields + Trace.
Out: (markdown answer, retrieved Hits).
"""

from __future__ import annotations

from core.llm_client import LLMClient, get_client
from core.trace import Trace
from hybrid.composer import Composer
from rag.answer import RagPath
from rag.retrieve import Hit, Retriever
from sql.execute import QueryResult
from sql.pipeline import SqlPath
from sql.schema import SchemaCatalog, SchemaInfo


class HybridPath:
    """Runs the SQL branch and the RAG branch, then merges them."""

    def __init__(
        self,
        schema: SchemaInfo | SchemaCatalog,
        client: LLMClient | None = None,
        sql_path: SqlPath | None = None,
        retriever: Retriever | None = None,
        composer: Composer | None = None,
    ) -> None:
        self.client = client or get_client()
        self.sql_path = sql_path or SqlPath(schema, self.client)
        self.retriever = retriever or Retriever()
        self.composer = composer or Composer(self.client)

    @staticmethod
    def summarise_result(result: QueryResult, max_items: int = 6) -> str:
        """One-line, generation-free digest of the result set used to steer retrieval."""
        if not result.ok or not result.rows:
            return ""
        cols = ", ".join(result.columns)
        head = "; ".join(
            " ".join(f"{c}={v}" for c, v in zip(result.columns, row))
            for row in result.rows[:max_items]
        )
        return f"columns: {cols}. rows: {head}"

    def run(
        self,
        question: str,
        sql_subquestion: str | None,
        doc_subquestion: str | None,
        trace: Trace,
        *,
        slots: dict | None = None,
    ) -> tuple[str, list[Hit]]:
        sql_q = sql_subquestion or question
        doc_q = doc_subquestion or question

        # --- step 2: SQL ----------------------------------------------------
        outcome = self.sql_path.run(sql_q, trace, slots=slots, render_answer=False)
        if not outcome.ok:
            # SQL failed: degrade to a docs-only answer rather than fabricating
            # the numeric half. The failure is already recorded in the trace.
            text, hits = RagPath(self.retriever, self.client).run(doc_q, trace)
            return (
                "_The quantitative half of this question could not be computed, so "
                "this answer covers the documentation half only._\n\n" + text,
                hits,
            )

        # --- step 3: retrieval conditioned on the SQL result -----------------
        digest = self.summarise_result(outcome.result)
        retrieval_query = f"{doc_q}\n[SQL result] {digest}" if digest else doc_q
        try:
            with trace.stage("hybrid_retrieve"):
                hits = self.retriever.retrieve(
                    retrieval_query, trace=trace, stage_prefix="hybrid_retrieve"
                )
        except FileNotFoundError as exc:
            trace.error(str(exc))
            hits = []
        trace.add_chunks([h.to_trace() for h in hits])

        # --- step 4: compose + verify ---------------------------------------
        with trace.stage("hybrid_compose"):
            text, number_ok = self.composer.compose(
                question, trace.generated_sql or "", outcome.result, hits, trace=trace
            )
        trace.number_check_passed = number_ok
        return text, hits


def run_hybrid_path(
    question: str,
    sql_subquestion: str | None,
    doc_subquestion: str | None,
    trace: Trace,
    schema: SchemaInfo | SchemaCatalog,
    *,
    slots: dict | None = None,
    client: LLMClient | None = None,
) -> tuple[str, list[Hit]]:
    """Module-level convenience wrapper around `HybridPath`."""
    return HybridPath(schema, client).run(
        question, sql_subquestion, doc_subquestion, trace, slots=slots
    )
