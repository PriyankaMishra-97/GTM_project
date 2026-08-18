"""Hybrid composition: merge a SQL result with cited doc chunks.

Role in architecture: the last step of the hybrid pipeline. `Composer.compose()`
makes one ANSWER_MODEL call under the separation-of-sources rules in
hybrid/prompts.py, then runs the verbatim number check. One regeneration on
violation; on a second violation it stops generating and renders the two
verified sources side by side - a slightly clunky answer built from checked
parts beats a fluent one containing a number nobody can trace.

In:  question, SQL result + SQL text, doc Hits.
Out: (markdown answer, number_check_passed).
"""

from __future__ import annotations

from core import safety
from core.llm_client import LLMClient, LLMUnavailable, get_client
from core.trace import Trace
from hybrid import prompts
from hybrid.verify import NumberVerifier
from rag.answer import build_context
from rag.retrieve import Hit
from sql.execute import QueryResult


class Composer:
    """Merges numbers (SQL) and explanation (docs) under strict source separation."""

    def __init__(
        self, client: LLMClient | None = None, verifier: NumberVerifier | None = None
    ) -> None:
        self.client = client or get_client()
        self.verifier = verifier or NumberVerifier()

    @staticmethod
    def fallback(sql: str, result: QueryResult, hits: list[Hit]) -> str:
        """Deterministic, generation-free rendering of both verified sources."""
        cites = "\n".join(
            f"- {h.citation()} {h.section or '-'}: {h.text.splitlines()[-1][:180]}"
            for h in hits[:3]
        )
        return (
            "_The composed answer introduced numbers that are not in the query result, "
            "so it was discarded. Showing the verified sources instead._\n\n"
            "**Finding (from SQL)**\n\n"
            f"{result.to_markdown()}\n\n"
            f"```sql\n{sql}\n```\n\n"
            "**Relevant documentation**\n\n"
            f"{cites if cites else '_No relevant doc chunks retrieved._'}"
        )

    def compose(
        self,
        question: str,
        sql: str,
        result: QueryResult,
        hits: list[Hit],
        trace: Trace | None = None,
    ) -> tuple[str, bool]:
        base_user = prompts.COMPOSER_USER.format(
            question=question,
            sql=sql,
            table=result.to_markdown(max_rows=40),
            context=build_context(hits) or "(no doc chunks retrieved)",
        )
        # The question and SQL are legitimate sources of echoed numbers ("2024",
        # "top 5"); the check treats them as context, not as fabrications.
        ctx = f"{question}\n{sql}"

        try:
            text = safety.filter_response(
                self.client.chat(
                    prompts.COMPOSER_SYSTEM, base_user, trace=trace, stage="hybrid_compose"
                )
            )
        except LLMUnavailable:
            return self.fallback(sql, result, hits), False

        verdict = self.verifier.check(text, result, context=ctx)
        if verdict.ok:
            return text, True

        retry_user = base_user + prompts.COMPOSER_REPAIR.format(
            offending=", ".join(verdict.offending)
        )
        try:
            text2 = safety.filter_response(
                self.client.chat(
                    prompts.COMPOSER_SYSTEM,
                    retry_user,
                    trace=trace,
                    stage="hybrid_compose_retry",
                )
            )
        except LLMUnavailable:
            return self.fallback(sql, result, hits), False

        if self.verifier.check(text2, result, context=ctx).ok:
            return text2, True

        return self.fallback(sql, result, hits), False


def compose(
    question: str,
    sql: str,
    result: QueryResult,
    hits: list[Hit],
    *,
    client: LLMClient | None = None,
) -> tuple[str, bool]:
    """Module-level convenience wrapper around `Composer`."""
    return Composer(client).compose(question, sql, result, hits)
