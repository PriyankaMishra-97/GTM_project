"""Reframe unit tests - no network, no provider.

Mirrors the StubClient pattern in tests/test_pipeline_offline.py, scoped to
just the Reframer's own continuation-vs-pivot decision.
"""

from __future__ import annotations

import json
from typing import Any

from ask.reframe import Reframer
from core.llm_client import LLMClient, LLMUnavailable
from core.trace import Trace


class StubClient(LLMClient):
    """Scripted LLM: returns a fixed JSON decision, or raises on demand."""

    provider = "stub"

    def __init__(self, response: dict[str, Any] | None = None, raises: bool = False) -> None:
        super().__init__(router_model="stub-router", answer_model="stub-answer")
        self.response = response
        self.raises = raises

    def available_models(self) -> list[str]:
        return ["stub-router", "stub-answer"]

    def _complete(
        self,
        model: str,
        system: str,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        if self.raises:
            raise LLMUnavailable("stub down")
        return json.dumps(self.response)


def test_continuation_merges_pending_with_reply() -> None:
    client = StubClient(
        {"is_new_topic": False, "effective_question": "How's pipeline looking in NA in 2024?"}
    )
    decision = Reframer(client).reframe(
        "How's pipeline looking?",
        ["time_range", "segment_or_region"],
        "NA in 2024",
        Trace(question="NA in 2024"),
    )
    assert decision.is_new_topic is False
    assert decision.effective_question == "How's pipeline looking in NA in 2024?"


def test_pivot_leaves_the_new_message_unchanged() -> None:
    client = StubClient(
        {"is_new_topic": True, "effective_question": "What's the pricing for Growth tier?"}
    )
    decision = Reframer(client).reframe(
        "How's pipeline looking?",
        ["time_range", "segment_or_region"],
        "What's the pricing for Growth tier?",
        Trace(question="What's the pricing for Growth tier?"),
    )
    assert decision.is_new_topic is True
    assert decision.effective_question == "What's the pricing for Growth tier?"


def test_llm_failure_falls_back_to_blind_concatenation() -> None:
    """The 'never worse than today' contract: on failure, degrade to exactly
    what app.py did before this feature existed - never raise, never drop
    the user's reply."""
    client = StubClient(raises=True)
    trace = Trace(question="2024")
    decision = Reframer(client).reframe("How's pipeline looking?", ["time_range"], "2024", trace)
    assert decision.is_new_topic is False
    assert decision.effective_question == "How's pipeline looking? 2024"
    assert trace.errors  # the failure is recorded, not silently swallowed


def test_reframe_call_is_recorded_in_the_trace() -> None:
    client = StubClient({"is_new_topic": False, "effective_question": "merged"})
    trace = Trace(question="reply")
    Reframer(client).reframe("pending", [], "reply", trace)
    assert "ask_reframe" in trace.per_stage_latency_ms
    assert any(c["stage"] == "ask_reframe" for c in trace.llm_calls)


def test_self_contradictory_new_topic_with_a_rewrite_is_treated_as_merge() -> None:
    """Regression: the model can say is_new_topic=true while ALSO handing
    back a rewritten (non-verbatim) effective_question - real case: "2024"
    against a pending pipeline question came back is_new_topic=true with
    effective_question="What is the pipeline looking like for deals closed
    after 2024?". The rewrite itself proves a connection was found; trust it
    over the contradictory boolean."""
    client = StubClient(
        {
            "is_new_topic": True,
            "effective_question": "What is the pipeline looking like for deals closed after 2024?",
        }
    )
    decision = Reframer(client).reframe(
        "How's pipeline looking?",
        ["time_range", "segment_or_region"],
        "2024",
        Trace(question="2024"),
    )
    assert decision.is_new_topic is False
    assert decision.effective_question == "What is the pipeline looking like for deals closed after 2024?"


def test_genuine_new_topic_with_reply_unchanged_stays_a_pivot() -> None:
    """The consistency guard must not fire when the model IS consistent -
    is_new_topic=true with effective_question == the reply, verbatim."""
    client = StubClient(
        {"is_new_topic": True, "effective_question": "What's the pricing for Growth tier?"}
    )
    decision = Reframer(client).reframe(
        "How's pipeline looking?",
        ["time_range", "segment_or_region"],
        "What's the pricing for Growth tier?",
        Trace(question="What's the pricing for Growth tier?"),
    )
    assert decision.is_new_topic is True
    assert decision.effective_question == "What's the pricing for Growth tier?"


def test_empty_missing_slots_uses_the_vague_fallback_line() -> None:
    """Covers a router-native ASK with no slots at all (e.g. the 'champion'
    misroute) - missing=[] must not crash the prompt formatting."""
    client = StubClient(
        {"is_new_topic": False, "effective_question": "Who is a champion in 2024?"}
    )
    decision = Reframer(client).reframe("Who is a champion?", [], "In 2024", Trace(question="In 2024"))
    assert decision.effective_question == "Who is a champion in 2024?"


def test_lossy_merge_of_a_compound_question_falls_back_to_concatenation() -> None:
    """Real case: "2024 all" against a two-part pending question came back
    paraphrased down to just the doc half, silently dropping the count."""
    pending = (
        "How many opportunities are in negotiation stage and what is the "
        "exit criteria of this stage?"
    )
    client = StubClient(
        {"is_new_topic": False, "effective_question": "What is the negotiation stage in all regions?"}
    )
    decision = Reframer(client).reframe(
        pending, ["time_range", "segment_or_region"], "2024 all", Trace(question="2024 all")
    )
    assert decision.effective_question == f"{pending} 2024 all"


def test_stale_echo_of_pending_question_falls_back_to_concatenation() -> None:
    """Real case: after tightening the prompt to fix the above, the model
    instead returned is_new_topic=true with effective_question equal to the
    pending question verbatim - the self-contradiction guard correctly flips
    it back to a continuation, but the reply's own content ("2024 all") was
    never incorporated. The lossy-merge guard must still catch this."""
    pending = (
        "How many opportunities are in negotiation stage and what is the "
        "exit criteria of this stage?"
    )
    client = StubClient({"is_new_topic": True, "effective_question": pending})
    decision = Reframer(client).reframe(
        pending, ["time_range", "segment_or_region"], "2024 all", Trace(question="2024 all")
    )
    assert decision.is_new_topic is False
    assert decision.effective_question == f"{pending} 2024 all"


def test_faithful_compound_merge_is_kept_as_is() -> None:
    """A merge that genuinely preserves both halves must NOT be discarded."""
    pending = (
        "How many opportunities are in negotiation stage and what is the "
        "exit criteria of this stage?"
    )
    merged = (
        "How many opportunities are in negotiation stage in 2024 for all "
        "segments/regions, and what is the exit criteria of this stage?"
    )
    client = StubClient({"is_new_topic": False, "effective_question": merged})
    decision = Reframer(client).reframe(
        pending, ["time_range", "segment_or_region"], "2024 all", Trace(question="2024 all")
    )
    assert decision.effective_question == merged
