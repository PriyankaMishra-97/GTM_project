"""Per-turn observability record.

Role in architecture: exactly one `Trace` object is created per user turn and
threaded through Router -> (RAG | SQL | HYBRID | ASK) -> renderer. Every stage
writes into it; nothing else logs. At the end of the turn the trace is appended
as one JSON line to ./storage/traces.jsonl and rendered in the UI.

In:  written by every pipeline stage.
Out: dict (UI panel) + one JSONL line (audit trail).
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from core import config


@dataclass
class RetrievedChunk:
    """One fused retrieval hit, as recorded in the trace."""

    chunk_id: str
    doc: str
    page: int
    section: str
    rrf_score: float
    text: str = ""
    dense_rank: int | None = None
    bm25_rank: int | None = None


@dataclass
class Trace:
    """Everything an operator needs to audit one turn."""

    question: str
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    user: str | None = None  # logged-in username, for the audit trail

    # `question` gets overwritten with the effective/routed question whenever
    # either reframe path below fires - `raw_question` is set once (to the
    # same initial value) and never mutated again, so the user's literal
    # input for this turn is never lost from the trace even after a rewrite.
    raw_question: str | None = None

    # --- reframe (only set when this turn followed an ASK) ---
    pending_question: str | None = None  # the carried-over question this turn started with
    is_new_topic: bool | None = None  # Reframer's verdict; None = no pending question existed

    # --- history reframe (only set when this turn did NOT follow an ASK and
    # recent_turns was available) - mutually exclusive with the two fields
    # above; see orchestrator.py's if/elif ---
    history_reframe_applied: bool | None = None

    # --- routing ---
    llm_proposed_route: str | None = None
    rule_override: str | None = None  # rule id that fired, e.g. "R2_VAGUE_TIME"
    final_route: str | None = None
    router_confidence: float | None = None
    router_rationale: str | None = None
    rule_detail: str = ""  # why the overriding rule fired
    refusal_message: str | None = None  # set only on the REFUSE route
    slots: dict[str, Any] = field(default_factory=dict)
    missing_slots: list[str] = field(default_factory=list)
    doc_subquestion: str | None = None
    sql_subquestion: str | None = None

    # --- RAG ---
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)

    # --- SQL ---
    generated_sql: str | None = None
    guard_verdict: str | None = None  # "PASS" or "REJECT: <reason>"
    rows_returned: int | None = None
    repair_attempted: bool = False

    # --- verification ---
    number_check_passed: bool | None = None

    # --- timing ---
    per_stage_latency_ms: dict[str, int] = field(default_factory=dict)
    total_latency_ms: int | None = None

    # --- errors (never swallowed silently) ---
    errors: list[str] = field(default_factory=list)

    # --- LLM calls: full prompt/response + attempts, written by core/llm_client.py ---
    llm_calls: list[dict[str, Any]] = field(default_factory=list)

    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def __post_init__(self) -> None:
        if self.raw_question is None:
            self.raw_question = self.question

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a pipeline stage: `with trace.stage("retrieve"): ...`."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = int((time.perf_counter() - start) * 1000)
            if name in self.per_stage_latency_ms:
                # A flat dict silently overwrites on key reuse - surface it as an
                # error instead of losing the earlier timing without a trace.
                self.error(f"duplicate stage timing key overwritten: {name}")
            self.per_stage_latency_ms[name] = elapsed

    def add_chunks(self, chunks: list[RetrievedChunk]) -> None:
        self.retrieved_chunks = [asdict(c) for c in chunks]

    def add_llm_call(self, record: dict[str, Any]) -> None:
        self.llm_calls.append(record)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def finish(self) -> "Trace":
        self.total_latency_ms = int((time.perf_counter() - self._t0) * 1000)
        return self

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_t0", None)
        return d

    def persist(self) -> None:
        """Append one JSON line. Best-effort: observability must never break a turn."""
        try:
            config.ensure_storage()
            with config.TRACES_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(self.to_dict(), default=str) + "\n")
        except OSError:  # pragma: no cover - disk full / read-only fs
            pass
