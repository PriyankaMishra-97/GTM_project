"""Result rendering: table-only by default, narrator LLM only when it earns its cost.

Role in architecture: `ResultRenderer` decides whether a SQL result needs prose
at all. `needs_narration()` is a deterministic policy, not a model call:

  - small result (<= SQL_NARRATE_ROW_THRESHOLD rows) + factual question
        -> render the markdown table directly, ZERO LLM calls.
  - large result, or an interpretive question ("why", "what's driving", "explain")
        -> one ANSWER_MODEL call, then the verbatim-number check.

This is the single biggest latency win in the system: the most common SQL turn
("how many X in Y") returns in the time of one generation call instead of two.

In:  question + QueryResult.
Out: markdown answer string + whether the number check passed.
"""

from __future__ import annotations

import re

from core import config, safety
from core.llm_client import LLMClient, get_client
from core.trace import Trace
from hybrid.verify import NumberVerifier
from sql import prompts
from sql.execute import QueryResult

# Words that mean "explain it to me" rather than "give me the number".
INTERPRETIVE_MARKERS = (
    "why", "what's driving", "whats driving", "what is driving", "explain",
    "reason", "cause", "root cause", "insight", "interpret", "so what",
    "summarize", "summarise", "trend", "story",
)


def needs_narration(
    question: str, result: QueryResult, threshold: int | None = None
) -> bool:
    """Deterministic policy - no LLM, no client, so it is unit-testable alone."""
    if result.row_count > (threshold or config.SQL_NARRATE_ROW_THRESHOLD):
        return True
    q = question.lower()
    return any(marker in q for marker in INTERPRETIVE_MARKERS)


class ResultRenderer:
    """Turns a QueryResult into the user-facing answer."""

    def __init__(
        self,
        client: LLMClient | None = None,
        verifier: NumberVerifier | None = None,
        narrate_threshold: int | None = None,
    ) -> None:
        self.client = client or get_client()
        self.verifier = verifier or NumberVerifier()
        self.narrate_threshold = narrate_threshold or config.SQL_NARRATE_ROW_THRESHOLD

    def needs_narration(self, question: str, result: QueryResult) -> bool:
        """Deterministic: does this result need an LLM to be understandable?"""
        return needs_narration(question, result, threshold=self.narrate_threshold)

    def narrate(
        self, question: str, sql: str, result: QueryResult, trace: Trace | None = None
    ) -> str:
        """One ANSWER_MODEL call summarising the result set."""
        user = prompts.NARRATE_USER.format(
            question=question,
            sql=sql,
            row_count=result.row_count,
            table=result.to_markdown(max_rows=40),
        )
        return safety.filter_response(
            self.client.chat(
                prompts.NARRATE_SYSTEM, user, trace=trace, stage="sql_render"
            )
        )

    def render(
        self, question: str, sql: str, result: QueryResult, trace: Trace | None = None
    ) -> tuple[str, bool | None]:
        """Produce the user-facing answer. Returns (markdown, number_check_passed).

        number_check_passed is None when no narration happened - there is no
        model text to verify, so the check is not applicable rather than passed.
        """
        table = result.to_markdown()
        if not self.needs_narration(question, result):
            return table, None

        # The question and SQL are legitimate sources of echoed numbers.
        context = f"{question}\n{sql}"
        text = self.narrate(question, sql, result, trace=trace)
        verdict = self.verifier.check(text, result, context=context)
        if verdict.ok:
            return f"{text}\n\n{table}", True

        # One regeneration with the violation named, then fall back to the table.
        retry_user = prompts.NARRATE_USER.format(
            question=question,
            sql=sql,
            row_count=result.row_count,
            table=result.to_markdown(max_rows=40),
        ) + (
            "\n\nYour previous summary contained numbers that are NOT in the result "
            f"set: {', '.join(verdict.offending)}. Rewrite it using only numbers that "
            "appear verbatim above."
        )
        text2 = safety.filter_response(
            self.client.chat(
                prompts.NARRATE_SYSTEM, retry_user, trace=trace, stage="sql_render_retry"
            )
        )
        if self.verifier.check(text2, result, context=context).ok:
            return f"{text2}\n\n{table}", True

        return (
            "_Narration was suppressed because it introduced numbers not present in "
            "the result set. Showing the verified query output instead._\n\n" + table
        ), False


def render(
    question: str, sql: str, result: QueryResult, *, client: LLMClient | None = None
) -> tuple[str, bool | None]:
    """Module-level convenience wrapper around `ResultRenderer`."""
    return ResultRenderer(client).render(question, sql, result)


def strip_markdown_numbers(text: str) -> list[str]:
    """Helper used by tests: every numeric token in a string."""
    return re.findall(r"-?\d[\d,]*\.?\d*%?", text)
