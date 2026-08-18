"""History reframe unit tests - no network, no provider.

Mirrors the StubClient pattern in tests/test_reframe.py, scoped to
HistoryReframer's own follow-up-vs-self-contained decision. Several cases
here are the exact scenarios dry-run against the real router model before
this feature was implemented - kept as regression tests since they're the
concrete evidence the design works.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ask.history_reframe import HistoryReframer, looks_like_followup
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


class ExplodingClient(LLMClient):
    """Fails the test if the LLM is ever called - for the fast-path gate."""

    provider = "exploding"

    def available_models(self) -> list[str]:
        return []

    def _complete(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError("the LLM should not have been called")


QUALIFY_TURN = {
    "question": "What are the exit criteria for Stage 1 — Qualify?",
    "answer": (
        "The exit criteria for Stage 1 — Qualify are: ICP match confirmed; "
        "problem statement documented; next meeting scheduled with champion. "
        "[Field Guide, p.3]"
    ),
}

COMMIT_TURN = {
    "question": "what about commit",
    "answer": (
        "The Commit stage is defined as follows: all approvals complete; "
        "signature plan confirmed; delivery kickoff scheduled. [Field Guide, p.3]"
    ),
}

SOLUTION_FIT_TURN = {
    "question": "What is the exit criteria for Solution Fit Stage?",
    "answer": (
        "The exit criteria for the Solution Fit stage are: Proposed "
        "architecture validated; demo/use-case proof aligns to success "
        "criteria. [Field Guide, p.3]"
    ),
}

EMEA_TURN = {
    "question": "How many opportunities were Closed Won in EMEA in 2024?",
    "answer": "There were 28 opportunities Closed Won in EMEA in 2024.",
}


# ---------------------------------------------------------------------------
# looks_like_followup() - the latency gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What is the SLA days?", True),
        ("and for APAC?", True),
        ("What about EMEA?", True),
        ("Why is that stage risky?", True),
        ("What is the exit criteria?", True),
        ("What deployment modes does Product XYZ support?", False),
        ("How many deals closed in 2024?", False),
        ("How many opportunities were Closed Won in EMEA in 2024?", False),
        ("What is the exit criteria for Solution Fit Stage?", False),
    ],
)
def test_looks_like_followup(question: str, expected: bool) -> None:
    assert looks_like_followup(question) is expected


# ---------------------------------------------------------------------------
# Fast path - must make ZERO LLM calls
# ---------------------------------------------------------------------------


def test_no_history_skips_the_llm_entirely() -> None:
    decision = HistoryReframer(ExplodingClient()).reframe(
        [], "What deployment modes does Product XYZ support?", Trace(question="x")
    )
    assert decision.is_new_topic is True
    assert decision.effective_question == "What deployment modes does Product XYZ support?"


def test_self_contained_question_skips_the_llm_despite_history() -> None:
    decision = HistoryReframer(ExplodingClient()).reframe(
        [EMEA_TURN], "What deployment modes does Product XYZ support?", Trace(question="x")
    )
    assert decision.is_new_topic is True
    assert decision.effective_question == "What deployment modes does Product XYZ support?"


# ---------------------------------------------------------------------------
# Real dry-run cases
# ---------------------------------------------------------------------------


def test_sla_followup_merges_in_the_stage_from_history() -> None:
    """Real case: the model reliably returns is_new_topic=True even
    alongside a correct rewrite - the self-contradiction guard must flip it."""
    client = StubClient(
        {"is_new_topic": True, "effective_question": "What is the SLA days for Solution Fit Stage?"}
    )
    decision = HistoryReframer(client).reframe(
        [SOLUTION_FIT_TURN], "What is the SLA days?", Trace(question="x")
    )
    assert decision.is_new_topic is False
    assert decision.effective_question == "What is the SLA days for Solution Fit Stage?"


def test_and_for_apac_merges_using_the_prior_sql_question() -> None:
    client = StubClient(
        {
            "is_new_topic": True,
            "effective_question": "How many opportunities were Closed Won in APAC in 2024?",
        }
    )
    decision = HistoryReframer(client).reframe([EMEA_TURN], "and for APAC?", Trace(question="x"))
    assert decision.is_new_topic is False
    assert decision.effective_question == "How many opportunities were Closed Won in APAC in 2024?"


def test_fabricated_entity_is_rejected_and_falls_back_to_no_rewrite() -> None:
    """Real case: history about EMEA opportunity counts, message "Why is
    that stage risky?" (no stage anywhere in that history) - the model
    invented "the Solution Fit stage" from its own prompt's worked example
    rather than admitting it couldn't resolve the reference. Must be caught
    regardless of what gets invented."""
    client = StubClient(
        {"is_new_topic": True, "effective_question": "Why is the Solution Fit stage risky?"}
    )
    decision = HistoryReframer(client).reframe([EMEA_TURN], "Why is that stage risky?", Trace(question="x"))
    assert decision.is_new_topic is True
    assert decision.effective_question == "Why is that stage risky?"


def test_merge_that_drops_the_replys_own_content_falls_back() -> None:
    """The merge must not just re-echo the prior question, ignoring the new
    message's own detail."""
    client = StubClient(
        {"is_new_topic": False, "effective_question": SOLUTION_FIT_TURN["question"]}
    )
    decision = HistoryReframer(client).reframe(
        [SOLUTION_FIT_TURN], "What is the SLA days?", Trace(question="x")
    )
    assert decision.is_new_topic is True
    assert decision.effective_question == "What is the SLA days?"


def test_stale_stage_name_retained_falls_back_to_no_rewrite() -> None:
    """Real case: even with an explicit prompt example showing this exact
    wrong pattern, the model produced "What are the exit criteria for Stage 1
    - Qualify and in Discover?" - keeping "Qualify" instead of substituting
    "Discover". Guards 2/3 don't catch this (nothing dropped or fabricated -
    "Qualify" is legitimately in the source turn), so guard 4 must.

    Uses a reply with content beyond the bare stage name ("what's the SLA for
    Discover") so it bypasses the deterministic _bare_stage_substitution
    shortcut and actually exercises the LLM path + guard 4."""
    client = StubClient(
        {
            "is_new_topic": True,
            "effective_question": "What is the SLA for Stage 1 — Qualify and Discover?",
        }
    )
    decision = HistoryReframer(client).reframe(
        [QUALIFY_TURN], "and what's the SLA for Discover", Trace(question="x")
    )
    assert decision.is_new_topic is True
    assert decision.effective_question == "and what's the SLA for Discover"


def test_stale_stage_checked_across_all_recent_turns_not_just_the_last() -> None:
    """Real case: with 2 turns in context (Qualify, then Commit), a stale
    merge retained the OLDER turn's stage ("Qualify"), not the most recent
    one ("Commit") - the check must not assume the last turn only."""
    client = StubClient(
        {
            "is_new_topic": True,
            "effective_question": "what is the SLA for Stage 1 — Qualify and Discover?",
        }
    )
    decision = HistoryReframer(client).reframe(
        [QUALIFY_TURN, COMMIT_TURN], "and what's the SLA for Discover", Trace(question="x")
    )
    assert decision.is_new_topic is True
    assert decision.effective_question == "and what's the SLA for Discover"


def test_correct_stage_substitution_via_llm_is_kept() -> None:
    """The positive case: guard 4 must not reject a merge that correctly
    substitutes the new stage in place of the old one, when the LLM path is
    exercised (reply has content beyond the bare stage name)."""
    client = StubClient(
        {
            "is_new_topic": True,
            "effective_question": "What is the SLA for the Discover stage?",
        }
    )
    decision = HistoryReframer(client).reframe(
        [QUALIFY_TURN], "and what's the SLA for Discover", Trace(question="x")
    )
    assert decision.is_new_topic is False
    assert decision.effective_question == "What is the SLA for the Discover stage?"


# ---------------------------------------------------------------------------
# Deterministic bare-stage-name substitution - no LLM call at all
# ---------------------------------------------------------------------------


def test_bare_stage_reference_substitutes_deterministically() -> None:
    """Real case: the LLM reliably APPENDS the new stage instead of
    REPLACING it for this exact pattern (see the guard-4 tests above) - a
    bare reference is resolved by substitution alone, no LLM judgment
    needed. Also drops the stale "Stage 1 -" prefix, which belongs to the
    OLD stage: keeping it produced wrong answers in practice (a rewritten
    question anchored on the wrong stage's retrieved content)."""
    decision = HistoryReframer(ExplodingClient()).reframe(
        [QUALIFY_TURN], "and in Discover?", Trace(question="x")
    )
    assert decision.is_new_topic is False
    assert decision.effective_question == "What are the exit criteria for Discover?"


def test_bare_stage_reference_skips_a_bare_template_turn() -> None:
    """With 2 turns in context where the MOST RECENT one is itself just a
    bare reference ("what about commit"), that turn isn't a usable
    template - the substitution must fall back to the older, fully-specified
    turn instead of producing another bare fragment."""
    commit_bare_turn = {"question": "what about commit", "answer": "x"}
    decision = HistoryReframer(ExplodingClient()).reframe(
        [QUALIFY_TURN, commit_bare_turn], "and in Discover?", Trace(question="x")
    )
    assert decision.is_new_topic is False
    assert decision.effective_question == "What are the exit criteria for Discover?"


def test_reply_with_extra_content_does_not_use_the_deterministic_shortcut() -> None:
    """"and what's the SLA for Discover" says more than just the stage name, so
    it must go through the LLM (and guards), not the bare-substitution
    shortcut - confirmed by the stub actually being invoked."""
    client = StubClient(
        {"is_new_topic": True, "effective_question": "What is the SLA for the Discover stage?"}
    )
    decision = HistoryReframer(client).reframe(
        [QUALIFY_TURN], "and what's the SLA for Discover", Trace(question="x")
    )
    assert decision.effective_question == "What is the SLA for the Discover stage?"


def test_genuine_pivot_is_left_unchanged() -> None:
    client = StubClient(
        {"is_new_topic": True, "effective_question": "What's the pricing for Growth tier?"}
    )
    decision = HistoryReframer(client).reframe(
        [EMEA_TURN], "What's the pricing for Growth tier?", Trace(question="x")
    )
    assert decision.is_new_topic is True
    assert decision.effective_question == "What's the pricing for Growth tier?"


def test_llm_failure_falls_back_to_no_rewrite_not_concatenation() -> None:
    """Unlike ask/reframe.py, failure must NOT blind-concatenate - two full
    past Q&A pairs onto a new question is noisy; leaving the question as
    typed can only match today's status quo, never make it worse."""
    client = StubClient(raises=True)
    trace = Trace(question="x")
    decision = HistoryReframer(client).reframe([SOLUTION_FIT_TURN], "What is the SLA days?", trace)
    assert decision.is_new_topic is True
    assert decision.effective_question == "What is the SLA days?"
    assert trace.errors


def test_history_reframe_call_is_recorded_in_the_trace() -> None:
    client = StubClient(
        {"is_new_topic": True, "effective_question": "What is the SLA days for Solution Fit Stage?"}
    )
    trace = Trace(question="x")
    HistoryReframer(client).reframe([SOLUTION_FIT_TURN], "What is the SLA days?", trace)
    assert "ask_history_reframe" in trace.per_stage_latency_ms
    assert any(c["stage"] == "ask_history_reframe" for c in trace.llm_calls)
