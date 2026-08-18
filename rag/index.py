"""Dual index: dense (ChromaDB + bge-small) and lexical (BM25).

Role in architecture: the storage half of RAG, wrapped in one `Index` class so
`Retriever` (rag/retrieve.py) and `scripts/ingest.py` share exactly one code path
for building/loading it. Two indexes because they fail differently - dense
retrieval finds paraphrases ("can it run offline?" -> "air-gapped"), BM25 finds
exact tokens the embedder blurs (SKU codes like XYZ-ANALYTICS, field names like
deployment_risk_score).

Determinism: pinned embedding model, normalize_embeddings=True (so cosine ==
dot product and scores are comparable across runs), stable chunk IDs, and a
pickled BM25 built from the same chunk order every time.

Idempotent: `build()` is a no-op if storage already exists unless force=True.

In:  list[Chunk] from rag/ingest.py.
Out: ./storage/chroma (vectors) + ./storage/bm25.pkl (lexical).
"""

from __future__ import annotations

import logging
import pickle
import re
import shutil
from dataclasses import dataclass
from typing import Any

from core import config
from rag.ingest import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9_\-]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Hyphens kept so 'XYZ-CORE' stays one token."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Store:
    """Pickled lexical index plus the chunk payloads it ranks."""

    bm25: Any
    chunk_ids: list[str]
    payloads: dict[str, dict[str, Any]]


class Index:
    """Owns both indexes on disk: build, load, embed. One instance per process is fine."""

    def __init__(self) -> None:
        self._embedder = None  # lazy: loading costs ~2s, and the UI reruns often

    # -------------------------------------------------------------- embedding --
    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(config.EMBED_MODEL)
        return self._embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # BGE is asymmetric: documents and queries take different prefixes. Using
        # the wrong one (or none) measurably degrades recall on short queries.
        prefixed = [config.BGE_DOC_PREFIX + t for t in texts]
        return self._get_embedder().encode(
            prefixed, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._get_embedder().encode(
            [config.BGE_QUERY_PREFIX + text], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return tokenize(text)

    # ------------------------------------------------------------------ chroma --
    @staticmethod
    def _chroma_client():
        import chromadb
        from chromadb.config import Settings

        # Chroma 0.5.x ships a posthog telemetry hook that is both unwanted here
        # (no network calls, by requirement) and broken against current posthog,
        # so it spams "Failed to send telemetry event" on every call. Disabled
        # via Settings; the logger is silenced so the failed-hook noise never
        # reaches the console either.
        logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
        logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

        config.ensure_storage()
        return chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )

    @staticmethod
    def exists() -> bool:
        return config.BM25_PATH.exists() and config.CHROMA_DIR.exists()

    def build(self, chunks: list[Chunk], force: bool = False) -> dict[str, int]:
        """Build both indexes. Returns a small stats dict for the CLI/UI."""
        if self.exists() and not force:
            return {"chunks": 0, "skipped": 1}

        config.ensure_storage()
        if force:
            shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)
            config.BM25_PATH.unlink(missing_ok=True)

        # --- dense ---
        client = self._chroma_client()
        try:
            client.delete_collection(config.CHROMA_COLLECTION)
        except Exception:
            pass  # first run: nothing to delete
        collection = client.create_collection(
            name=config.CHROMA_COLLECTION,
            # Cosine on normalised vectors; explicit so a Chroma default change
            # cannot silently alter ranking.
            metadata={"hnsw:space": "cosine"},
        )
        collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=self.embed_documents([c.text for c in chunks]),
            metadatas=[c.metadata() for c in chunks],
        )

        # --- lexical ---
        from rank_bm25 import BM25Okapi

        store = BM25Store(
            bm25=BM25Okapi([tokenize(c.text) for c in chunks]),
            chunk_ids=[c.chunk_id for c in chunks],
            payloads={c.chunk_id: {"text": c.text, **c.metadata()} for c in chunks},
        )
        with config.BM25_PATH.open("wb") as fh:
            pickle.dump(store, fh)

        return {"chunks": len(chunks), "skipped": 0}

    def load_bm25(self) -> BM25Store:
        if not config.BM25_PATH.exists():
            raise FileNotFoundError(
                f"Lexical index missing at {config.BM25_PATH}. Run `python -m scripts.ingest`."
            )
        with config.BM25_PATH.open("rb") as fh:
            return pickle.load(fh)

    def load_collection(self):
        client = self._chroma_client()
        try:
            return client.get_collection(config.CHROMA_COLLECTION)
        except Exception as exc:
            raise FileNotFoundError(
                f"Vector index missing at {config.CHROMA_DIR}. Run `python -m scripts.ingest`."
            ) from exc

    def stats(self) -> dict[str, Any]:
        """Sidebar status. Never raises - a missing index is a normal first-run state."""
        if not self.exists():
            return {"ready": False, "chunks": 0}
        try:
            return {"ready": True, "chunks": self.load_collection().count()}
        except Exception:
            return {"ready": False, "chunks": 0}


# --------------------------------------------------------------------------
# module-level convenience wrappers - a single shared Index for callers (the
# CLI, the UI) that don't need to hold their own instance.
# --------------------------------------------------------------------------
_DEFAULT_INDEX = Index()


def index_exists() -> bool:
    return Index.exists()


def build_index(chunks: list[Chunk], force: bool = False) -> dict[str, int]:
    return _DEFAULT_INDEX.build(chunks, force=force)


def load_bm25() -> BM25Store:
    return _DEFAULT_INDEX.load_bm25()


def load_collection():
    return _DEFAULT_INDEX.load_collection()


def index_stats() -> dict[str, Any]:
    return _DEFAULT_INDEX.stats()


def get_embedder():
    return _DEFAULT_INDEX._get_embedder()


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _DEFAULT_INDEX.embed_documents(texts)


def embed_query(text: str) -> list[float]:
    return _DEFAULT_INDEX.embed_query(text)
