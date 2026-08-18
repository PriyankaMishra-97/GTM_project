"""Reframe path: decide if a reply to a clarification continues it, or pivots.

Role in architecture: runs ONLY on a turn that follows an ASK. `app.py`'s
slot-merge fix previously assumed every such reply answers the clarification
and blindly concatenated it onto the pending question - wrong when the user
instead asks something unrelated. This is a separate, small, dedicated call
rather than folded into router/llm_router.py's main classification prompt:
that prompt is proven fragile (one new few-shot example there broke 1-2
previously-passing golden-set questions, regardless of placement), so
continuation-vs-pivot detection stays fully isolated from it - zero risk to
routing accuracy, at the cost of one extra ROUTER_MODEL call, paid only on
turns that follow an ASK.

Failure policy: if the model is unavailable or returns junk, fall back to
the ORIGINAL blind-concatenation behaviour. This step can only match or
improve on today's behaviour, never make it worse.

In:  the pending question, its missing slots, and the user's new reply.
Out: `ReframeDecision` (pydantic) - is_new_topic + the effective question.
"""

from __future__ import annotations

from pydantic import BaseModel

from ask import prompts
from ask.text_overlap import content_words, preserves_content
from core.llm_client import LLMClient, LLMJSONError, LLMUnavailable, get_client
from core.trace import Trace
from router.slots import SLOT_QUESTIONS


class ReframeDecision(BaseModel):
    is_new_topic: bool
    effective_question: str


class Reframer:
    """Decides continuation vs. pivot, with a deterministic fallback."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or get_client()

    def reframe(
        self, pending_question: str, missing: list[str], reply: str, trace: Trace
    ) -> ReframeDecision:
        missing_block = "\n".join(
            f"- {slot}: {SLOT_QUESTIONS.get(slot, 'unspecified')}" for slot in missing
        ) or "- the question is too vague to act on"

        fallback = ReframeDecision(
            is_new_topic=False, effective_question=f"{pending_question} {reply}".strip()
        )
        try:
            with trace.stage("ask_reframe"):
                result = self.client.chat_json(
                    prompts.REFRAME_SYSTEM,
                    prompts.REFRAME_USER.format(
                        pending_question=pending_question,
                        missing=missing_block,
                        reply=reply,
                    ),
                    ReframeDecision,
                    model=self.client.router_model,
                    trace=trace,
                    stage="ask_reframe",
                )
        except (LLMJSONError, LLMUnavailable) as exc:
            trace.error(f"reframe fallback: {exc}")
            return fallback

        # Self-contradiction guard: a small model saying is_new_topic=true
        # while ALSO handing back a rewritten (not verbatim) effective_question
        # has, by its own rewrite, just demonstrated it found a connection to
        # the pending question - real case: "2024" against a pending pipeline
        # question came back {"is_new_topic": true, "effective_question":
        # "What is the pipeline looking like for deals closed after 2024?"}.
        # Trusting the boolean over the model's own text would have thrown
        # away a correct merge. Consistent with "when unsure, prefer merge".
        candidate = result
        if result.is_new_topic and result.effective_question.strip() != reply.strip():
            trace.error(
                "reframe: is_new_topic=true but effective_question was rewritten - "
                "treating as a continuation instead"
            )
            candidate = ReframeDecision(is_new_topic=False, effective_question=result.effective_question)

        if candidate.is_new_topic:
            return candidate  # genuine pivot: effective_question is unused downstream anyway

        # Lossy-merge guard (continuation path only, reached via either branch
        # above): a compound pending question ("count X and explain Y") can get
        # paraphrased down to just one half instead of combined, or the reply's
        # new detail can get dropped while the pending question is echoed back
        # unchanged - real cases: "2024 all" against "how many opportunities
        # are in negotiation stage and what is the exit criteria of this
        # stage?" came back "What is the negotiation stage in all regions?"
        # (drops the count); the same input, after tightening the prompt to
        # fix that, instead came back with is_new_topic=true and
        # effective_question equal to the pending question verbatim (drops
        # "2024 all" entirely). Both are caught the same way: the merge must
        # keep most of the pending question's content AND actually incorporate
        # the reply's own content - if either fails, fall back to blind
        # concatenation over a fluent but incomplete or stale merge.
        merged_words = content_words(candidate.effective_question)
        pending_words = content_words(pending_question)
        reply_words = content_words(reply)
        if not preserves_content(pending_words, reply_words, merged_words):
            trace.error(
                "reframe: merged question dropped pending or reply content - "
                "using blind concatenation instead"
            )
            return fallback
        return candidate


def reframe(
    pending_question: str,
    missing: list[str],
    reply: str,
    trace: Trace,
    *,
    client: LLMClient | None = None,
) -> ReframeDecision:
    """Module-level convenience wrapper around `Reframer`."""
    return Reframer(client).reframe(pending_question, missing, reply, trace)
