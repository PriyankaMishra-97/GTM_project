"""RRF fusion: known lists -> known order, and byte-stable across runs."""

from __future__ import annotations

from core import config
from rag.retrieve import reciprocal_rank_fusion


def test_known_lists_produce_expected_order() -> None:
    dense = ["a", "b", "c"]
    lexical = ["c", "a", "d"]
    # a: 1/61 + 1/62 = 0.032662...   c: 1/63 + 1/61 = 0.032267...
    # b: 1/62 = 0.016129             d: 1/63 = 0.015873
    fused = reciprocal_rank_fusion([dense, lexical])
    assert [doc for doc, _ in fused] == ["a", "c", "b", "d"]


def test_agreement_beats_a_single_strong_hit() -> None:
    """A doc ranked 2nd in BOTH lists outranks one ranked 1st in only one.

    This is the property that makes RRF worth having: it rewards agreement
    between the dense and lexical views instead of trusting either alone.
    """
    fused = dict(reciprocal_rank_fusion([["solo", "both"], ["other", "both"]]))
    assert fused["both"] > fused["solo"]


def test_ties_break_by_chunk_id() -> None:
    # Identical positions in both lists -> identical scores -> id ordering.
    fused = reciprocal_rank_fusion([["zeta", "alpha"], ["zeta", "alpha"]])
    scores = dict(fused)
    assert scores["zeta"] > scores["alpha"]
    same = reciprocal_rank_fusion([["b", "a"], ["a", "b"]])
    assert [doc for doc, _ in same] == ["a", "b"]  # equal scores, id ascending


def test_deterministic_across_runs() -> None:
    lists = [["x1", "x2", "x3", "x4"], ["x3", "x9", "x1", "x7"]]
    first = reciprocal_rank_fusion(lists)
    for _ in range(5):
        assert reciprocal_rank_fusion(lists) == first


def test_k_matches_configured_value() -> None:
    fused = dict(reciprocal_rank_fusion([["only"]]))
    assert abs(fused["only"] - 1.0 / (config.RRF_K + 1)) < 1e-12


def test_top_k_truncates() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c", "d", "e", "f"]], top_k=3)
    assert len(fused) == 3
