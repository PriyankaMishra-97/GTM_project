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

import re

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

    _ROW_RE = re.compile(r"^\d+\s*[—-]")  # table data rows: "1 — Qualify | ...", "2 - Discover | ..."
    _SEPARATOR_RE = re.compile(r"^[\s|:-]+$")  # "TABLE" header/dash-rule rows, no real content

    @staticmethod
    def _snippet(text: str, question: str, limit: int = 180) -> str:
        """Pick the content line most relevant to the question for a fallback citation.

        A multi-row table chunk (e.g. the Stage Playbook holds all 6 stages in
        one chunk) makes `splitlines()[-1]` arbitrary - it can show a
        completely unrelated row (e.g. "Closed Won" for a question about
        "Discover"). Score candidate lines by substring overlap with the
        question's words (not exact match, so "discovery" still finds a row
        titled "Discover"), preferring numbered table rows over the header/
        column-title row when both are present.
        """
        lines = [ln for ln in text.splitlines() if ln.strip()]
        lines = lines[1:] or lines  # drop the section heading; it repeats h.section
        rows = [ln for ln in lines if Composer._ROW_RE.match(ln.strip())]
        candidates = rows or [ln for ln in lines if not Composer._SEPARATOR_RE.match(ln)] or lines

        q_words = [w.strip(".,;:—-").lower() for w in question.split() if len(w) > 3]

        def overlap(ln: str) -> int:
            ln_words = [w.strip(".,;:—-").lower() for w in ln.split()]
            return sum(
                1
                for qw in q_words
                for lw in ln_words
                if len(lw) > 3 and (qw in lw or lw in qw)
            )

        best_line = max(candidates, key=overlap, default=text)
        return best_line[:limit]

    @staticmethod
    def fallback(sql: str, result: QueryResult, hits: list[Hit], question: str = "") -> str:
        """Deterministic, generation-free rendering of both verified sources."""
        cites = "\n".join(
            f"- {h.citation()} {h.section or '-'}: {Composer._snippet(h.text, question)}"
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
            return self.fallback(sql, result, hits, question), False

        doc_texts = [h.text for h in hits]
        verdict = self.verifier.check(text, result, context=ctx, doc_texts=doc_texts)
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
            return self.fallback(sql, result, hits, question), False

        if self.verifier.check(text2, result, context=ctx, doc_texts=doc_texts).ok:
            return text2, True

        return self.fallback(sql, result, hits, question), False


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
