"""Central configuration for GTM Analyst Copilot.

Role in architecture: single source of truth for model names, paths, retrieval
constants and safety knobs. Every other module imports from here so that a
change (e.g. swapping the answer model) is a one-line edit.

In:  environment variables (optional overrides).
Out: module-level constants consumed across core/, router/, rag/, sql/, hybrid/, ask/.
"""

from __future__ import annotations

import os
from pathlib import Path

# Hard requirement, not a preference: this system makes no network calls. Chroma
# reads this env var at import time, so it must be set before chromadb loads.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# PROJECT_ROOT is resolved from this file, not the CWD, so `streamlit run app.py`
# and `python -m tests.run_all` from any directory resolve assets identically.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
ASSETS_DIR: Path = PROJECT_ROOT / "assets"
STORAGE_DIR: Path = PROJECT_ROOT / "storage"

DB_PATH: Path = ASSETS_DIR / "gtm_mock.db"
USERS_PATH: Path = ASSETS_DIR / "users.yaml"  # gitignored; see users.example.yaml
PDF_PATHS: dict[str, Path] = {
    # short name -> file. The short name is what appears inside citations,
    # e.g. "[Enablement Pack, p.3]", so keep it short and human readable.
    "Enablement Pack": ASSETS_DIR / "Product_XYZ_Enablement_Pack.pdf",
    # v2: fixes source-PDF defects in v1 (table cells fused across column
    # boundaries with no separator, at least one mid-word truncation) that no
    # amount of chunker logic could recover - see rag/ingest.py's rows_by_span.
    "Field Guide": ASSETS_DIR / "Opportunity_Tracker_FieldGuide_v2.pdf",
}

CHROMA_DIR: Path = STORAGE_DIR / "chroma"
BM25_PATH: Path = STORAGE_DIR / "bm25.pkl"
TRACES_PATH: Path = STORAGE_DIR / "traces.jsonl"
CHROMA_COLLECTION: str = "gtm_docs"

# --------------------------------------------------------------------------
# LLM tiering
# --------------------------------------------------------------------------
# Two tiers on purpose: routing is a cheap classification that a 3B model does
# well and fast (it gates end-to-end latency on EVERY turn), while answering /
# SQL generation needs the stronger 7B model. Mixing them keeps the p50 latency
# down without giving up answer quality.
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ROUTER_MODEL: str = os.getenv("GTM_ROUTER_MODEL", "llama3.2:3b")
ANSWER_MODEL: str = os.getenv("GTM_ANSWER_MODEL", "qwen2.5:7b")

# Determinism: fixed seed + temperature 0 everywhere. Ollama honours both.
LLM_SEED: int = 42
LLM_TEMPERATURE: float = 0.0
LLM_TIMEOUT_S: int = 180
# num_predict caps runaway generation (a local 7B can otherwise ramble past the
# 10s latency target). Generous enough for a cited multi-paragraph answer.
LLM_MAX_TOKENS: int = 1024
# Ollama defaults num_ctx to the MODEL's own built-in context length when unset
# (131072 for llama3.2:3b, 32768 for qwen2.5:7b per `ollama show`) - wildly
# oversized for this system's actual prompts (schema card + question, or a
# handful of retrieved chunks: a few thousand tokens at most). Allocating a
# 32k-131k KV cache on every call inflates load time and, on a memory-
# constrained machine, creates pressure that can evict the OTHER tier's model
# between calls. 4096 covers every prompt this system sends with headroom.
LLM_NUM_CTX: int = 4096
# Ollama's default keep_alive (5 minutes) unloads an idle model, so any gap
# between chat turns pays full reload cost again. Keeping both tiers resident
# longer avoids that - cheap on RAM now that num_ctx isn't oversized.
LLM_KEEP_ALIVE: str = "30m"

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
# bge-small-en-v1.5 is pinned: embeddings are only comparable against an index
# built by the SAME model, so changing this requires `--force` re-ingest.
EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
# BGE is trained asymmetrically: documents and queries get different prefixes.
# Dropping these costs several points of recall@5 on this corpus.
BGE_DOC_PREFIX: str = "Represent this document for retrieval: "
BGE_QUERY_PREFIX: str = "Represent this query for retrieval: "

CHUNK_TARGET_TOKENS: int = 300  # ~300 tokens per section chunk
TABLE_ROWS_PER_CHUNK: int = 6  # data rows grouped under a repeated header row

DENSE_TOP_K: int = 10
BM25_TOP_K: int = 10
# 5 systematically dropped chunks that ranked well in dense (top 10) but never
# appeared in BM25 at all - the "different stages" investigation: a table
# whose content is proper nouns/numbers ("1 - Qualify") shares few tokens with
# a query that paraphrases ("what are the stages"), so it's invisible to BM25
# regardless of rank, and RRF then favours chunks scored by BOTH lists (even
# at a middling rank) over a chunk scored by only one (even at a strong rank).
# 7 is the smallest bump that captures that case without ~doubling context.
RRF_TOP_K: int = 7
# k=60 is the value from the original RRF paper (Cormack et al. 2009). It damps
# the head of each list so a single index cannot dominate the fusion.
RRF_K: int = 60

# --------------------------------------------------------------------------
# SQL safety
# --------------------------------------------------------------------------
SQL_ROW_LIMIT: int = 200  # injected LIMIT when the model omits one
SQL_TIMEOUT_S: int = 5
SQL_MAX_ROWS_RENDERED: int = 200
# Above this row count we let the narrator LLM summarise; at or below it, and
# for purely factual questions, we render the table with NO LLM call at all.
SQL_NARRATE_ROW_THRESHOLD: int = 20

# Functions the generated SQL may call. Anything else is rejected by the AST
# guard - an allowlist cannot be bypassed by a novel function name the way a
# regex denylist can.
SQL_ALLOWED_FUNCTIONS: frozenset[str] = frozenset(
    {
        # aggregates
        "count", "sum", "avg", "min", "max", "total", "group_concat",
        # numeric / null handling
        "round", "abs", "coalesce", "ifnull", "nullif", "cast",
        # dates (SQLite)
        "date", "datetime", "julianday", "strftime", "time",
        # strings (read-only, harmless)
        "lower", "upper", "substr", "length", "trim", "replace",
        # windowing / ranking used by "top N per group" questions
        "row_number", "rank", "dense_rank", "ntile", "lag", "lead",
    }
)

# --------------------------------------------------------------------------
# Safety / output filtering
# --------------------------------------------------------------------------
# Columns stripped from rendered SQL results. Empty by default (the synthetic DB
# has no PII); populated it becomes the "sensitive field suppression" control.
COLUMN_DENYLIST: frozenset[str] = frozenset()
# Length of verbatim prompt-template substring that counts as a prompt leak.
PROMPT_LEAK_NGRAM_CHARS: int = 60

# --------------------------------------------------------------------------
# Router thresholds
# --------------------------------------------------------------------------
ROUTER_MIN_CONFIDENCE: float = 0.6

REQUIRED_ENV_HINT = (
    "Ollama must be running locally. Start it with `ollama serve`, then:\n"
    f"    ollama pull {ROUTER_MODEL}\n"
    f"    ollama pull {ANSWER_MODEL}"
)


def ensure_storage() -> None:
    """Create ./storage (gitignored) on demand. Called by ingest and by the app."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
