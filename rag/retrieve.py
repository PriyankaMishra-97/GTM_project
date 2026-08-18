"""Hybrid retrieval with Reciprocal Rank Fusion.

Role in architecture: `Retriever.retrieve()` queries both indexes (top-10 each),
fuses to top-5, hands the winners to rag/answer.py. RRF is used instead of score
normalisation because the two indexes produce incomparable scores (cosine
similarity vs BM25 saturation); RRF only needs each list's RANK, so no tuning
constant has to be re-fit when the corpus changes.

    rrf(d) = sum over lists L of  1 / (k + rank_L(d)),  k = 60

k=60 is the value from Cormack et al. (2009). It damps the head of each list so a
single index cannot dominate; smaller k makes rank-1 hits nearly unbeatable.

Determinism: both lists are deterministic, and ties are broken by chunk_id so the
fused order is byte-stable across runs.

In:  query string.
Out: list[Hit] (top-5) with fused scores + the per-index ranks that produced them.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, ContextManager, Sequence

from core import config
from core.trace import RetrievedChunk, Trace
from rag.index import Index


@dataclass
class Hit:
    chunk_id: str
    text: str
    doc: str
    page: int
    section: str
    rrf_score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None

    def citation(self) -> str:
        return f"[{self.doc}, p.{self.page}]"

    def to_trace(self) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=self.chunk_id,
            doc=self.doc,
            page=self.page,
            section=self.section,
            rrf_score=round(self.rrf_score, 6),
            text=self.text,
            dense_rank=self.dense_rank,
            bm25_rank=self.bm25_rank,
        )


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]], k: int = config.RRF_K, top_k: int | None = None
) -> list[tuple[str, float]]:
    """Fuse ranked ID lists. Pure function - unit tested without any index.

    Ties are broken by chunk_id (ascending) so the output is fully deterministic.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered[:top_k] if top_k else ordered


class Retriever:
    """Dense + lexical retrieval fused by RRF, backed by one `Index`."""

    def __init__(self, index: Index | None = None) -> None:
        self.index = index or Index()

    @staticmethod
    def _stage(trace: Trace | None, name: str) -> ContextManager[None]:
        return trace.stage(name) if trace is not None else contextlib.nullcontext()

    def _dense_search(self, query: str, top_k: int) -> tuple[list[str], dict[str, dict[str, Any]]]:
        collection = self.index.load_collection()
        res = collection.query(
            query_embeddings=[self.index.embed_query(query)],
            n_results=top_k,
            include=["documents", "metadatas"],
        )
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        payloads = {
            cid: {"text": doc, **(meta or {})} for cid, doc, meta in zip(ids, docs, metas)
        }
        return list(ids), payloads

    def _bm25_search(self, query: str, top_k: int) -> tuple[list[str], dict[str, dict[str, Any]]]:
        store = self.index.load_bm25()
        scores = store.bm25.get_scores(self.index.tokenize(query))
        # Sort by score desc, then chunk_id asc - deterministic on ties.
        order = sorted(
            range(len(store.chunk_ids)),
            key=lambda i: (-float(scores[i]), store.chunk_ids[i]),
        )[:top_k]
        ids = [store.chunk_ids[i] for i in order]
        return ids, {cid: store.payloads[cid] for cid in ids}

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        trace: Trace | None = None,
        stage_prefix: str = "rag_retrieve",
    ) -> list[Hit]:
        """Dense + lexical retrieval fused by RRF."""
        top_k = top_k or config.RRF_TOP_K
        with self._stage(trace, f"{stage_prefix}_dense"):
            dense_ids, dense_payloads = self._dense_search(query, config.DENSE_TOP_K)
        with self._stage(trace, f"{stage_prefix}_bm25"):
            bm25_ids, bm25_payloads = self._bm25_search(query, config.BM25_TOP_K)

        with self._stage(trace, f"{stage_prefix}_fuse"):
            payloads = {**bm25_payloads, **dense_payloads}
            fused = reciprocal_rank_fusion([dense_ids, bm25_ids], top_k=top_k)

        dense_rank = {cid: i + 1 for i, cid in enumerate(dense_ids)}
        bm25_rank = {cid: i + 1 for i, cid in enumerate(bm25_ids)}

        hits: list[Hit] = []
        for chunk_id, score in fused:
            payload = payloads.get(chunk_id, {})
            hits.append(
                Hit(
                    chunk_id=chunk_id,
                    text=payload.get("text", ""),
                    doc=payload.get("doc", "?"),
                    page=int(payload.get("page", 0)),
                    section=payload.get("section", ""),
                    rrf_score=score,
                    dense_rank=dense_rank.get(chunk_id),
                    bm25_rank=bm25_rank.get(chunk_id),
                )
            )
        return hits


def retrieve(query: str, top_k: int | None = None) -> list[Hit]:
    """Module-level convenience wrapper around `Retriever` (uses a fresh default Index)."""
    return Retriever().retrieve(query, top_k)
