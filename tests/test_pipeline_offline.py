"""End-to-end orchestration tests with a stub LLM - no network, no provider.

These exist for two reasons:

  1. Coverage. Router -> rules -> path dispatch -> guard -> execute -> render is
     the code most likely to break in a refactor, and until now it was only
     exercised by the live demo.
  2. Proof that the provider abstraction is real. `StubClient` subclasses
     `LLMClient` and implements ONE method (`_complete`). If the rest of the
     system can run end to end on it, then swapping Ollama for Gemini genuinely
     is a one-class change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import config
from core.auth import UserProfile
from core.llm_client import LLMClient
from orchestrator import GTMCopilot
from sql.schema import SchemaCatalog

pytestmark = pytest.mark.skipif(
    not Path(config.DB_PATH).exists(), reason="provided DB not present"
)

# Unrestricted user - these tests exercise routing/SQL/guard behavior that
# predates the region/segment ACL feature and should be unaffected by it.
UNRESTRICTED_USER = UserProfile(
    username="test", allowed_regions=None, allowed_segments=None
)


class StubClient(LLMClient):
    """Scripted LLM. Records every call so tests can assert on call counts."""

    provider = "stub"

    def __init__(
        self,
        router_json: dict[str, Any],
        sql: str = "",
        prose: str = "",
        reframe_json: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(router_model="stub-router", answer_model="stub-answer")
        self.router_json = router_json
        self.sql = sql
        self.prose = prose
        self.reframe_json = reframe_json
        self.calls: list[str] = []

    def available_models(self) -> list[str]:
        return ["stub-router", "stub-answer"]

    def _complete(
        self,
        model: str,
        system: str,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        self.calls.append(model)
        if json_schema is not None:
            if "ROUTER" in system:
                return json.dumps(self.router_json)
            if "continues a pending clarification" in system:
                assert self.reframe_json is not None, "test forgot to script reframe_json"
                return json.dumps(self.reframe_json)
            # ask/clarify is the only other JSON caller
            return json.dumps({"questions": ["Which time range should I use?"]})
        if "SQLite analyst" in system:
            return self.sql
        return self.prose


def _copilot(client: StubClient) -> GTMCopilot:
    # A real SchemaCatalog against the real DB - only the LLM is stubbed.
    return GTMCopilot(client=client, catalog=SchemaCatalog())


def test_sql_route_runs_end_to_end_and_renders_a_table() -> None:
    client = StubClient(
        router_json={
            "route": "SQL",
            "missing_slots": [],
            "confidence": 0.95,
            "rationale": "count with explicit filters",
            "slots": {"time_range": "2024", "segment_or_region": "EMEA"},
            "doc_subquestion": None,
            "sql_subquestion": None,
        },
        sql=(
            "SELECT COUNT(*) AS won FROM opportunities "
            "WHERE stage = 'Closed Won' AND region = 'EMEA' "
            "AND close_date BETWEEN '2024-01-01' AND '2024-12-31'"
        ),
    )
    answer = _copilot(client).answer("How many were Closed Won in EMEA in 2024?", UNRESTRICTED_USER)

    assert answer.route == "SQL"
    assert answer.trace.guard_verdict == "PASS"
    assert answer.trace.rows_returned == 1
    assert "28" in answer.text  # the known value in the shipped DB
    # Small factual result -> table only, so the narrator must NOT have run.
    assert client.calls == ["stub-router", "stub-answer"]
    assert answer.trace.number_check_passed is None


def test_guard_rejection_is_surfaced_not_executed() -> None:
    client = StubClient(
        router_json={
            "route": "SQL", "missing_slots": [], "confidence": 0.95, "rationale": "x",
            "slots": {"time_range": "2024", "segment_or_region": "EMEA"},
            "doc_subquestion": None, "sql_subquestion": None,
        },
        sql="DROP TABLE opportunities",
    )
    answer = _copilot(client).answer("Counts for EMEA in 2024", UNRESTRICTED_USER)

    assert answer.route == "SQL"
    assert answer.trace.guard_verdict.startswith("REJECT")
    assert answer.trace.rows_returned is None
    assert "safety guard" in answer.text


def test_write_intent_refuses_before_any_answer_model_call() -> None:
    client = StubClient(
        router_json={
            "route": "SQL", "missing_slots": [], "confidence": 0.99, "rationale": "x",
            "slots": {"time_range": "2024", "segment_or_region": "EMEA"},
            "doc_subquestion": None, "sql_subquestion": None,
        },
        sql="SELECT 1 AS x",
    )
    answer = _copilot(client).answer("Delete all Closed Lost opportunities in EMEA in 2024", UNRESTRICTED_USER)

    assert answer.route == "REFUSE"
    assert answer.trace.rule_override == "R0_WRITE_INTENT"
    assert answer.trace.generated_sql is None
    # The refusal is a fixed string: only the router was ever called.
    assert client.calls == ["stub-router"]


def test_ask_route_when_the_rule_engine_overrides() -> None:
    client = StubClient(
        router_json={
            "route": "SQL", "missing_slots": [], "confidence": 0.9, "rationale": "x",
            "slots": {}, "doc_subquestion": None, "sql_subquestion": None,
        }
    )
    answer = _copilot(client).answer("How's pipeline looking recently?", UNRESTRICTED_USER)

    assert answer.route == "ASK"
    assert answer.trace.llm_proposed_route == "SQL"  # proposal vs final is visible
    assert answer.trace.rule_override in ("R1_MISSING_SLOT", "R2_VAGUE_TIME")
    assert "time range" in answer.text.lower()


def test_trace_records_stage_latencies_and_persists() -> None:
    client = StubClient(
        router_json={
            "route": "SQL", "missing_slots": [], "confidence": 0.95, "rationale": "x",
            "slots": {"time_range": "2024", "segment_or_region": "EMEA"},
            "doc_subquestion": None, "sql_subquestion": None,
        },
        sql="SELECT COUNT(*) AS n FROM opportunities",
    )
    answer = _copilot(client).answer("How many opportunities in EMEA in 2024?", UNRESTRICTED_USER)

    trace = answer.trace.to_dict()
    assert {"route", "sql_generate", "sql_guard", "sql_execute"} <= set(
        trace["per_stage_latency_ms"]
    )
    assert trace["total_latency_ms"] is not None
    assert trace["final_route"] == "SQL"
    assert config.TRACES_PATH.exists()


def test_repair_loop_runs_once_on_a_bad_column() -> None:
    """First SQL references a non-existent column; the guard passes it (the column
    check is SQLite's job), execution fails, and exactly one repair is attempted."""

    class RepairingClient(StubClient):
        def _complete(self, model, system, messages, json_schema=None):
            self.calls.append(model)
            if json_schema is not None:
                return json.dumps(self.router_json)
            if "SQLite error" in messages[-1]["content"]:
                return "SELECT COUNT(*) AS n FROM opportunities"
            return "SELECT nonexistent_column AS x FROM opportunities"

    client = RepairingClient(
        router_json={
            "route": "SQL", "missing_slots": [], "confidence": 0.95, "rationale": "x",
            "slots": {"time_range": "2024", "segment_or_region": "EMEA"},
            "doc_subquestion": None, "sql_subquestion": None,
        }
    )
    answer = _copilot(client).answer("How many opportunities in EMEA in 2024?", UNRESTRICTED_USER)

    assert answer.trace.repair_attempted is True
    assert answer.trace.rows_returned == 1
    assert "1800" in answer.text


def test_scoped_user_is_refused_for_an_out_of_scope_region() -> None:
    """End-to-end: proves the user param threading and per-user SqlPath/guard
    wiring (orchestrator._sql_path_for) compose correctly, not just each piece
    in isolation."""
    na_only_user = UserProfile(
        username="alice", allowed_regions=frozenset({"NA"}), allowed_segments=None
    )
    client = StubClient(
        router_json={
            "route": "SQL", "missing_slots": [], "confidence": 0.95, "rationale": "x",
            "slots": {"time_range": "2024", "segment_or_region": "EMEA"},
            "doc_subquestion": None, "sql_subquestion": None,
        },
        sql="SELECT COUNT(*) AS n FROM opportunities WHERE region = 'EMEA'",
    )
    answer = _copilot(client).answer(
        "How many opportunities closed in EMEA in 2024?", na_only_user
    )

    assert answer.route == "REFUSE"
    assert answer.trace.rule_override == "R1B_REGION_SCOPE"
    assert answer.trace.generated_sql is None
    # Refused before any SQL was ever generated - only the router ran.
    assert client.calls == ["stub-router"]


def test_pivot_to_a_new_question_is_not_merged_with_the_stale_pending_text() -> None:
    """Regression: a reply to a clarifying question could itself be a brand new,
    unrelated question - app.py used to blindly concatenate it onto the stale
    pending text instead of asking an LLM to tell the two apart."""

    class PivotClient(StubClient):
        """Turn 1's question is vague (missing slots); turn 2 is a fully
        specified, unrelated question - the router must see EACH one on its
        own merits, never a garbled merge of both."""

        def _complete(self, model, system, messages, json_schema=None):
            self.calls.append(model)
            if json_schema is not None:
                if "ROUTER" in system:
                    user_text = messages[-1]["content"]
                    if "EMEA" in user_text:
                        return json.dumps(
                            {
                                "route": "SQL", "missing_slots": [], "confidence": 0.95,
                                "rationale": "x",
                                "slots": {"time_range": "2024", "segment_or_region": "EMEA"},
                                "doc_subquestion": None, "sql_subquestion": None,
                            }
                        )
                    return json.dumps(self.router_json)
                if "continues a pending clarification" in system:
                    return json.dumps(self.reframe_json)
                return json.dumps({"questions": ["Which time range should I use?"]})
            if "SQLite analyst" in system:
                return self.sql
            return self.prose

    client = PivotClient(
        router_json={
            "route": "SQL", "missing_slots": [], "confidence": 0.9, "rationale": "x",
            "slots": {}, "doc_subquestion": None, "sql_subquestion": None,
        },
        reframe_json={
            "is_new_topic": True,
            "effective_question": "How many opportunities were Closed Won in EMEA in 2024?",
        },
        sql="SELECT COUNT(*) AS won FROM opportunities WHERE stage = 'Closed Won' "
        "AND region = 'EMEA' AND close_date BETWEEN '2024-01-01' AND '2024-12-31'",
    )
    copilot = _copilot(client)

    turn1 = copilot.answer("How's pipeline looking recently?", UNRESTRICTED_USER)
    assert turn1.route == "ASK"

    turn2 = copilot.answer(
        "How many opportunities were Closed Won in EMEA in 2024?",
        UNRESTRICTED_USER,
        pending_clarification=turn1.trace.question,
        pending_missing_slots=turn1.trace.missing_slots,
    )

    assert turn2.trace.is_new_topic is True
    # Exactly the new question - NOT concatenated with turn 1's stale text.
    assert turn2.trace.question == "How many opportunities were Closed Won in EMEA in 2024?"
    assert turn2.route == "SQL"
    assert "28" in turn2.text


def test_continuation_still_merges_correctly_with_the_reframe_step_in_place() -> None:
    """The opposite case, in the same harness: a genuine answer to the
    clarification must still be merged, not treated as a pivot."""
    client = StubClient(
        router_json={
            "route": "SQL", "missing_slots": [], "confidence": 0.9, "rationale": "x",
            "slots": {}, "doc_subquestion": None, "sql_subquestion": None,
        },
        reframe_json={
            "is_new_topic": False,
            "effective_question": "How's pipeline looking? EMEA 2024",
        },
    )
    copilot = _copilot(client)

    turn1 = copilot.answer("How's pipeline looking?", UNRESTRICTED_USER)
    assert turn1.route == "ASK"

    turn2 = copilot.answer(
        "EMEA 2024",
        UNRESTRICTED_USER,
        pending_clarification=turn1.trace.question,
        pending_missing_slots=turn1.trace.missing_slots,
    )
    assert turn2.trace.is_new_topic is False
    assert turn2.trace.question == "How's pipeline looking? EMEA 2024"
