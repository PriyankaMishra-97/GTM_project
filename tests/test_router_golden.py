"""Routing tests, in two layers.

LAYER 1 (always runs, no model): the rule engine is driven with MOCKED router
outputs. Rules are the safety guarantee, so they must be covered even on a
machine with no Ollama.

LAYER 2 (skipped without Ollama): ~20 labelled prompts, 5 per route, asserting
the FINAL route end to end. This is the honest accuracy signal for the local 3B
router; it is skipped rather than mocked because mocking it would test nothing.
"""

from __future__ import annotations

import pytest

from core.auth import UserProfile
from core.llm_client import OllamaClient
from router.llm_router import RouterDecision, route_question
from router.rules import apply_rules

# ---------------------------------------------------------------------------
# Layer 1 - rule engine with mocked LLM proposals
# ---------------------------------------------------------------------------

# Unrestricted user - these Layer 1 tests predate the region/segment ACL
# feature and assert on the ORIGINAL rules (R0-R3); R1B_REGION_SCOPE tests
# below use their own restricted fixtures.
UNRESTRICTED_USER = UserProfile(username="test", allowed_regions=None, allowed_segments=None)


def _decision(route: str, **kw) -> RouterDecision:
    kw.setdefault("confidence", 0.9)
    return RouterDecision(route=route, **kw)


def _apply(decision: RouterDecision, question: str, user: UserProfile = UNRESTRICTED_USER):
    return apply_rules(decision, question, user)


def test_r0_write_intent_refuses_even_when_llm_said_sql() -> None:
    out = _apply(_decision("SQL"), "delete all Closed Lost opportunities")
    assert out.final_route == "REFUSE"
    assert out.rule_id == "R0_WRITE_INTENT"
    assert out.refusal_message


@pytest.mark.parametrize(
    "question",
    [
        "drop table opportunities",
        "update the stage of opportunity 42 to Closed Won",
        "insert a new account for Acme",
        "truncate the activities table",
    ],
)
def test_r0_covers_all_write_verbs(question: str) -> None:
    assert _apply(_decision("SQL"), question).rule_id == "R0_WRITE_INTENT"


def test_r0_does_not_fire_on_innocent_words() -> None:
    # "Updated pricing note" is a real heading in the Enablement Pack.
    out = _apply(_decision("RAG"), "what does the updated pricing note say?")
    assert out.final_route == "RAG"
    assert out.rule_id is None


def test_r1_missing_slot_overrides_sql_to_ask() -> None:
    out = _apply(_decision("SQL"), "how many deals did we win?")
    assert out.final_route == "ASK"
    assert out.rule_id == "R1_MISSING_SLOT"
    assert "time_range" in out.missing_slots


def test_r0c_hybrid_without_sql_subquestion_downgrades_to_rag() -> None:
    """Real case: "what does the field guide require before a deal reaches
    Commit?" got misrouted HYBRID (copying a near-identical few-shot example)
    with doc_subquestion filled but sql_subquestion empty - nothing to
    compute, so it isn't really hybrid. Must route straight to RAG, not ASK
    for filters a doc-only question never needed."""
    out = _apply(
        _decision("HYBRID", doc_subquestion="what gates a deal before Commit", sql_subquestion=None),
        "what does the field guide require before a deal reaches Commit?",
    )
    assert out.final_route == "RAG"
    assert out.rule_id == "R0C_HYBRID_NO_SQL"


def test_r0c_does_not_fire_on_a_real_hybrid_proposal() -> None:
    out = _apply(
        _decision(
            "HYBRID",
            doc_subquestion="what gates a deal before Commit",
            sql_subquestion="win rate for Enterprise in 2024",
            slots={"time_range": "2024", "segment_or_region": "Enterprise"},
        ),
        "What is our win rate for Enterprise in 2024, and what does the field "
        "guide say should gate a deal before Commit?",
    )
    assert out.final_route == "HYBRID"
    assert out.rule_id is None


def test_r1_stage_question_does_not_need_a_stage_definition() -> None:
    """stage_definition is auto-resolved by route (SQL -> database stages,
    doc -> playbook), not asked for - see router/slots.py::required_slots."""
    out = _apply(
        _decision("SQL", slots={"time_range": "2024", "segment_or_region": "EMEA"}),
        "how many deals are in each stage in EMEA in 2024?",
    )
    assert out.final_route == "SQL"
    assert out.rule_id is None


def test_r1_does_not_fire_when_slots_are_complete() -> None:
    out = _apply(
        _decision("SQL", slots={"time_range": "2024", "segment_or_region": "EMEA"}),
        "how many opportunities were Closed Won in EMEA in 2024?",
    )
    assert out.final_route == "SQL"
    assert out.rule_id is None


def test_r1_accepts_slots_implied_by_the_question_text() -> None:
    """The router forgot to fill the slots, but the question states both."""
    out = _apply(
        _decision("SQL", slots={}),
        "total bookings in APAC for Enterprise in 2024",
    )
    assert out.final_route == "SQL"


@pytest.mark.parametrize(
    "question",
    [
        "how's pipeline looking recently?",
        "show me the top deals",
        "what's the best region lately?",
        "how is the funnel trending?",
    ],
)
def test_r2_vague_quantifier_without_time_range(question: str) -> None:
    out = _apply(_decision("SQL"), question)
    assert out.final_route == "ASK"
    assert out.rule_id in ("R1_MISSING_SLOT", "R2_VAGUE_TIME")


def test_r2_does_not_fire_when_time_is_explicit() -> None:
    out = _apply(
        _decision("SQL", slots={"time_range": "2024", "segment_or_region": "NA"}),
        "show me the top 10 deals in NA in 2024",
    )
    assert out.final_route == "SQL"
    assert out.rule_id is None


def test_r2_does_not_hijack_a_well_specified_doc_question() -> None:
    """Regression: "How is X calculated?" is a doc question, not a vague one."""
    out = _apply(_decision("RAG"), "How is the deployment risk score calculated?")
    assert out.final_route == "RAG"
    assert out.rule_id is None


def test_r3_low_confidence_overrides_to_ask() -> None:
    out = _apply(
        _decision("RAG", confidence=0.35), "what does the guide say about artifacts?"
    )
    assert out.final_route == "ASK"
    assert out.rule_id == "R3_LOW_CONFIDENCE"


NA_ONLY_USER = UserProfile(
    username="alice", allowed_regions=frozenset({"NA"}), allowed_segments=None
)


def test_r1b_refuses_a_named_region_outside_the_users_scope() -> None:
    out = _apply(
        _decision("SQL", slots={"time_range": "2024", "segment_or_region": "EMEA"}),
        "how many opportunities closed in EMEA in 2024?",
        user=NA_ONLY_USER,
    )
    assert out.final_route == "REFUSE"
    assert out.rule_id == "R1B_REGION_SCOPE"
    assert "NA" in out.refusal_message


def test_r1b_narrows_all_to_the_users_scope_without_overriding_the_route() -> None:
    decision = _decision(
        "SQL", slots={"time_range": "2024", "segment_or_region": "all"}
    )
    out = _apply(
        decision, "total pipeline value in 2024 across all regions", user=NA_ONLY_USER
    )
    assert out.final_route == "SQL"
    assert out.rule_id is None
    # NA_ONLY_USER restricts region only, so segments stay unrestricted -
    # the narrowed list is NA plus every segment, never EMEA/APAC/LATAM.
    narrowed = decision.slots["segment_or_region"]
    assert "NA" in narrowed
    assert "EMEA" not in narrowed


def test_r1b_does_not_fire_for_an_unrestricted_user() -> None:
    out = _apply(
        _decision("SQL", slots={"time_range": "2024", "segment_or_region": "EMEA"}),
        "how many opportunities closed in EMEA in 2024?",
    )
    assert out.final_route == "SQL"
    assert out.rule_id is None


def test_rule_order_write_intent_beats_missing_slot() -> None:
    out = _apply(_decision("SQL", confidence=0.1), "delete deals from 2024")
    assert out.rule_id == "R0_WRITE_INTENT"


def test_high_confidence_rag_passes_through() -> None:
    out = _apply(
        _decision("RAG", confidence=0.95), "what deployment modes does Product XYZ support?"
    )
    assert out.final_route == "RAG"
    assert out.rule_id is None


def test_hybrid_with_full_slots_passes_through() -> None:
    out = _apply(
        _decision(
            "HYBRID",
            confidence=0.88,
            slots={"time_range": "2024", "segment_or_region": "Enterprise"},
            sql_subquestion="win rate for Enterprise in 2024",
            doc_subquestion="what gates a deal before Commit",
        ),
        "what is our 2024 Enterprise win rate and what gates a deal before Commit?",
    )
    assert out.final_route == "HYBRID"


# ---------------------------------------------------------------------------
# Layer 2 - end-to-end golden set (needs a running Ollama)
# ---------------------------------------------------------------------------
_client = OllamaClient()
_OLLAMA_UP = _client.is_ready()

GOLDEN: list[tuple[str, str]] = [
    # --- RAG ---
    ("What deployment modes does Product XYZ support?", "RAG"),
    ("What is included in the Growth pricing tier?", "RAG"),
    ("What are the exit criteria for the Solution Fit stage in the playbook?", "RAG"),
    ("How is the deployment risk score calculated?", "RAG"),
    ("What does the field guide say about PII handling in notes?", "RAG"),
    # --- SQL ---
    ("How many opportunities were Closed Won in EMEA in 2024?", "SQL"),
    ("What is total pipeline value by region for deals created in 2024, all segments?", "SQL"),
    ("List the 10 largest Enterprise opportunities created in 2024 across all regions.", "SQL"),
    ("What is the average deal size for SMB deals closed in 2023 in NA?", "SQL"),
    ("How many accounts in APAC across all segments have a Live deployment as of 2024?", "SQL"),
    # --- HYBRID ---
    (
        "What is our 2024 win rate for Enterprise, and what does the field guide "
        "require before a deal reaches Commit?",
        "HYBRID",
    ),
    (
        "Show 2024 NA deals in Negotiation using the database stages and explain "
        "the risk scoring rubric that applies to them.",
        "HYBRID",
    ),
    (
        "How many APAC accounts across all segments are On-Prem in 2024, and what "
        "does the enablement pack say about On-Prem limitations?",
        "HYBRID",
    ),
    (
        "Total 2024 pipeline for XYZ-ANALYTICS in all regions, and is ANALYTICS "
        "included in the Starter tier?",
        "HYBRID",
    ),
    (
        "Count Churn Risk deployments in EMEA in 2024 and summarise the recommended "
        "action for that deployment status.",
        "HYBRID",
    ),
    # --- ASK ---
    ("How's pipeline looking recently?", "ASK"),
    ("Show me the top deals", "ASK"),
    ("What's our win rate?", "ASK"),
    ("How many deals are stuck?", "ASK"),
    ("Give me a summary of performance lately", "ASK"),
]


@pytest.mark.skipif(not _OLLAMA_UP, reason="Ollama unreachable or models not pulled")
@pytest.mark.parametrize("question,expected", GOLDEN, ids=[q[:40] for q, _ in GOLDEN])
def test_golden_routes(question: str, expected: str) -> None:
    decision, err = route_question(question, client=_client)
    assert err is None, err
    outcome = _apply(decision, question)
    assert outcome.final_route == expected, (
        f"{question!r}: llm proposed {decision.route} "
        f"(conf {decision.confidence}), rule {outcome.rule_id}, "
        f"final {outcome.final_route}, expected {expected}"
    )
