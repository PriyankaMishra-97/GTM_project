"""ASK path: turn missing slots into 1-3 answerable clarification questions.

Role in architecture: the terminal branch when the rule engine decides the
question is not answerable as asked. One ANSWER_MODEL call, schema-constrained.

Failure policy: if the model is unavailable or returns junk, `Clarifier` falls
back to the hand-written question per slot in router/slots.py. The ASK path must
never fail - it IS the failure path.

In:  question + missing slot names + the rule that fired.
Out: markdown clarification message.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ask import prompts
from core.llm_client import LLMClient, LLMJSONError, LLMUnavailable, get_client
from core.trace import Trace
from router.slots import SLOT_QUESTIONS


class Clarifications(BaseModel):
    questions: list[str] = Field(default_factory=list)


class Clarifier:
    """Generates clarification questions, with a deterministic fallback."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or get_client()

    @staticmethod
    def fallback(missing: list[str]) -> list[str]:
        """Deterministic questions - also what the tests assert against."""
        return [SLOT_QUESTIONS[s] for s in missing if s in SLOT_QUESTIONS][:3] or [
            "Could you add a time range and a region or segment so I can compute this accurately?"
        ]

    def clarify(
        self, question: str, missing: list[str], reason: str, trace: Trace
    ) -> str:
        missing_block = "\n".join(
            f"- {slot}: {SLOT_QUESTIONS.get(slot, 'unspecified')}" for slot in missing
        ) or "- the question is too vague to act on"

        questions: list[str]
        try:
            with trace.stage("ask_clarify"):
                result = self.client.chat_json(
                    prompts.CLARIFY_SYSTEM,
                    prompts.CLARIFY_USER.format(
                        question=question,
                        reason=reason or "the question is underspecified",
                        missing=missing_block,
                    ),
                    Clarifications,
                    trace=trace,
                    stage="ask_clarify",
                )
            questions = [q.strip() for q in result.questions if q.strip()][:3]
        except (LLMJSONError, LLMUnavailable) as exc:
            trace.error(f"clarify fallback: {exc}")
            questions = []

        if not questions:
            questions = self.fallback(missing)

        bullets = "\n".join(f"- {q}" for q in questions)
        return (
            "I need one more detail before I can answer this accurately:\n\n"
            f"{bullets}\n\n"
            "_Answering with a guessed filter would give you a confident wrong number, "
            "so I'd rather ask._"
        )


def clarify(
    question: str,
    missing: list[str],
    reason: str,
    trace: Trace,
    *,
    client: LLMClient | None = None,
) -> str:
    """Module-level convenience wrapper around `Clarifier`."""
    return Clarifier(client).clarify(question, missing, reason, trace)
