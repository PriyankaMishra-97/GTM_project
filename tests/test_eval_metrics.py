"""Eval metric unit tests - no network, no provider.

context_recall/context_precision are pure functions: exact-value assertions.
faithfulness/answer_relevancy need an LLM judge - tested with a StubClient to
prove the SCORING ARITHMETIC is correct given a scripted judge response, not
to test judge quality itself (that can't be asserted deterministically).
"""

from __future__ import annotations

import json
from typing import Any

from core.llm_client import LLMClient, LLMUnavailable
from eval.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevance,
    faithfulness,
)

STAGE_PLAYBOOK = ("Field Guide", "Stage progression playbook")
PRICING = ("Enablement Pack", "Packaging & Pricing Cheat Sheet")
UNRELATED = ("Enablement Pack", "FAQ")


class StubClient(LLMClient):
    """Scripted LLM: returns a fixed JSON response, or raises on demand."""

    provider = "stub"

    def __init__(self, response: dict[str, Any] | None = None, raises: bool = False) -> None:
        super().__init__(router_model="stub-router", answer_model="stub-answer")
        self.response = response
        self.raises = raises

    def available_models(self) -> list[str]:
        return ["stub-router", "stub-answer"]

    def _complete(self, model, system, messages, json_schema=None) -> str:
        if self.raises:
            raise LLMUnavailable("stub down")
        return json.dumps(self.response)


# ------------------------------------------------------------ context recall --
def test_context_recall_is_perfect_when_all_relevant_sections_are_retrieved() -> None:
    assert context_recall([STAGE_PLAYBOOK, PRICING], [STAGE_PLAYBOOK]) == 1.0


def test_context_recall_is_zero_when_relevant_section_never_retrieved() -> None:
    assert context_recall([PRICING, UNRELATED], [STAGE_PLAYBOOK]) == 0.0


def test_context_recall_does_not_care_about_rank() -> None:
    """Recall is rank-blind by design - context_precision is the rank-aware one."""
    retrieved_late = [UNRELATED] * 5 + [STAGE_PLAYBOOK]
    assert context_recall(retrieved_late, [STAGE_PLAYBOOK]) == 1.0


def test_context_recall_is_vacuously_perfect_with_no_labelled_sections() -> None:
    assert context_recall([STAGE_PLAYBOOK], []) == 1.0


# --------------------------------------------------------- context precision --
def test_context_precision_is_perfect_when_relevant_section_ranks_first() -> None:
    assert context_precision([STAGE_PLAYBOOK, PRICING], [STAGE_PLAYBOOK]) == 1.0


def test_context_precision_penalises_a_relevant_section_ranked_late() -> None:
    """Regression proof: this is the metric that would have caught the
    'stages' bug - the right chunk WAS eventually retrieved (recall=1.0) but
    at fused rank 6-7, past the top-5 cutoff in production."""
    retrieved_late = [UNRELATED] * 5 + [STAGE_PLAYBOOK]
    score = context_precision(retrieved_late, [STAGE_PLAYBOOK])
    assert 0 < score < 1.0
    assert score == 1 / 6


def test_context_precision_is_zero_on_a_total_miss() -> None:
    assert context_precision([PRICING, UNRELATED], [STAGE_PLAYBOOK]) == 0.0


def test_context_precision_is_zero_on_empty_retrieval() -> None:
    assert context_precision([], [STAGE_PLAYBOOK]) == 0.0


# -------------------------------------------------------------- context relevance --
def test_context_relevance_scores_fraction_of_chunks_judged_relevant() -> None:
    client = StubClient(
        {
            "judgments": [
                {"index": 0, "relevant": True},
                {"index": 1, "relevant": False},
                {"index": 2, "relevant": True},
            ]
        }
    )
    score = context_relevance("What deployment modes?", ["ctx0", "ctx1", "ctx2"], client)
    assert score == 2 / 3


def test_context_relevance_is_none_with_no_retrieved_chunks() -> None:
    """No chunks retrieved isn't 'perfectly relevant' or 'perfectly irrelevant' -
    it's a different failure (retrieval found nothing) that context_recall
    already reports as 0.0; this metric abstains rather than double-count it."""
    client = StubClient({"judgments": []})
    assert context_relevance("q", [], client) is None


def test_context_relevance_is_none_when_the_judge_returns_no_judgments() -> None:
    client = StubClient({"judgments": []})
    assert context_relevance("q", ["ctx0"], client) is None


def test_context_relevance_is_none_when_the_judge_is_unavailable() -> None:
    client = StubClient(raises=True)
    assert context_relevance("q", ["ctx0"], client) is None


# ---------------------------------------------------------------- faithfulness --
def test_faithfulness_scores_the_fraction_of_supported_claims() -> None:
    client = StubClient(
        {
            "claims": [
                {"text": "Growth costs $12,000/month", "supported": True},
                {"text": "Growth includes on-prem hosting", "supported": False},
            ]
        }
    )
    result = faithfulness("What's in Growth?", "answer text", ["context text"], client)
    assert result.score == 0.5
    assert len(result.claims) == 2


def test_faithfulness_is_none_when_there_are_no_substantive_claims() -> None:
    """An ASK/REFUSE-style answer has nothing to fact-check - 0/0 is undefined,
    not 0.0, so callers can exclude it from an aggregate mean instead of
    dragging the average down for a case that was never wrong."""
    client = StubClient({"claims": []})
    result = faithfulness("q", "I need more detail before I can answer.", [], client)
    assert result.score is None


def test_faithfulness_is_none_when_the_judge_is_unavailable() -> None:
    client = StubClient(raises=True)
    result = faithfulness("q", "a", ["ctx"], client)
    assert result.score is None


# ------------------------------------------------------------ answer relevancy --
def test_answer_relevancy_averages_cosine_similarity_to_hypothetical_questions() -> None:
    vectors = {
        "What deployment modes are offered?": [1.0, 0.0],
        "Which modes does Product XYZ support?": [1.0, 0.0],  # identical -> sim 1.0
        "What is the weather today?": [0.0, 1.0],  # orthogonal -> sim 0.0
    }
    client = StubClient(
        {
            "questions": [
                "Which modes does Product XYZ support?",
                "What is the weather today?",
            ]
        }
    )
    score = answer_relevancy(
        "What deployment modes are offered?",
        "Cloud, On-Prem, and Hybrid.",
        client,
        embed_fn=lambda text: vectors[text],
    )
    assert score == 0.5  # mean(1.0, 0.0)


def test_answer_relevancy_is_none_when_the_judge_returns_no_questions() -> None:
    client = StubClient({"questions": []})
    score = answer_relevancy("q", "a", client, embed_fn=lambda t: [0.0])
    assert score is None


def test_answer_relevancy_is_none_when_the_judge_is_unavailable() -> None:
    client = StubClient(raises=True)
    score = answer_relevancy("q", "a", client, embed_fn=lambda t: [0.0])
    assert score is None
