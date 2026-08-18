"""RAGAS-style metrics, adapted to run on this project's local stack.

Two are pure functions (context recall/precision) - deterministic, no LLM,
unit-testable with exact-value assertions, but both need a pre-labelled
ground-truth section per question (see eval/dataset.py). Two need an LLM
judge (faithfulness, answer relevancy) - `ANSWER_MODEL` (7B), matching the
model-tiering rule already in core/config.py: cheap classification goes to
the 3B router model, anything needing reading-comprehension quality goes to
the 7B answer model.

In:  retrieved/relevant (doc, section) pairs, or question+answer+context text.
Out: a 0.0-1.0 score (context metrics), or None when a score isn't well-defined
     (no claims to judge, no hypothetical questions generated, no chunks
     judged, LLM unavailable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from pydantic import BaseModel, Field

from core.llm_client import LLMClient, LLMJSONError, LLMUnavailable
from eval import prompts

Section = tuple[str, str]  # (doc, section)


# --------------------------------------------------------------------------
# Context recall / precision - deterministic, no LLM.
# --------------------------------------------------------------------------
def context_recall(retrieved: Sequence[Section], relevant: Sequence[Section]) -> float:
    """Fraction of the labelled-relevant sections that retrieval actually found.

    Vacuously 1.0 if no sections were labelled relevant - nothing to recall.
    """
    if not relevant:
        return 1.0
    found = set(retrieved) & set(relevant)
    return len(found) / len(set(relevant))


def context_precision(retrieved: Sequence[Section], relevant: Sequence[Section]) -> float:
    """Rank-aware precision: rewards relevant sections retrieved EARLIER.

    For each rank where `retrieved[rank]` is relevant, precision@rank = (count
    of relevant sections in retrieved[0..rank]) / (rank + 1); the metric is the
    mean of those. A relevant section retrieved at rank 0 contributes 1.0; the
    same section retrieved at rank 6 (past a fixed top-k cutoff, like the
    "stages" bug this session) contributes far less, even if it's eventually
    present - this is the metric that would have caught that regression.
    """
    if not retrieved or not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = 0
    precisions: list[float] = []
    for rank, section in enumerate(retrieved):
        if section in relevant_set:
            hits += 1
            precisions.append(hits / (rank + 1))
    return sum(precisions) / len(precisions) if precisions else 0.0


# --------------------------------------------------------------------------
# Faithfulness - needs an LLM judge.
# --------------------------------------------------------------------------
class Claim(BaseModel):
    text: str
    supported: bool


class FaithfulnessJudgment(BaseModel):
    claims: list[Claim] = Field(default_factory=list)


@dataclass
class FaithfulnessResult:
    score: float | None  # None when there were no substantive claims to judge
    claims: list[Claim] = field(default_factory=list)


def faithfulness(
    question: str, answer: str, contexts: Sequence[str], client: LLMClient
) -> FaithfulnessResult:
    """Fraction of the answer's factual claims that are supported by `contexts`."""
    try:
        judgment = client.chat_json(
            prompts.FAITHFULNESS_SYSTEM,
            prompts.FAITHFULNESS_USER.format(
                question=question, context="\n\n".join(contexts), answer=answer
            ),
            FaithfulnessJudgment,
            model=client.answer_model,
        )
    except (LLMJSONError, LLMUnavailable):
        return FaithfulnessResult(score=None)

    if not judgment.claims:
        return FaithfulnessResult(score=None)
    supported = sum(1 for c in judgment.claims if c.supported)
    return FaithfulnessResult(score=supported / len(judgment.claims), claims=judgment.claims)


# --------------------------------------------------------------------------
# Answer relevancy - LLM generates hypothetical questions, scored by embedding
# similarity against the real question. Reuses rag/index.py's BGE embedder
# (already normalize_embeddings=True, so cosine similarity is a plain dot
# product) - no new embedding model, no new math dependency.
# --------------------------------------------------------------------------
class HypotheticalQuestions(BaseModel):
    questions: list[str] = Field(default_factory=list)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def answer_relevancy(
    question: str,
    answer: str,
    client: LLMClient,
    embed_fn: Callable[[str], list[float]],
) -> float | None:
    """Mean cosine similarity between the real question and N hypothetical
    questions the answer would suit - low when the answer wanders off-topic
    even if it's individually well-supported (faithfulness) and well-formed.
    """
    try:
        result = client.chat_json(
            prompts.RELEVANCY_SYSTEM,
            prompts.RELEVANCY_USER.format(answer=answer),
            HypotheticalQuestions,
            model=client.answer_model,
        )
    except (LLMJSONError, LLMUnavailable):
        return None

    questions = [q.strip() for q in result.questions if q.strip()]
    if not questions:
        return None
    q_vec = embed_fn(question)
    similarities = [_cosine(q_vec, embed_fn(hq)) for hq in questions]
    return sum(similarities) / len(similarities)
