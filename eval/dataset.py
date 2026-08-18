"""Labelled RAG/HYBRID questions for the evaluation pipeline.

Every `relevant_sections` label was checked against the actual PDF content
this session (either a live retrieval+answer run, or a direct read of the
extracted table text) - not guessed. `reference_answer` isn't consumed by any
of the four metrics in eval/metrics.py (context recall/precision only need
`relevant_sections`; faithfulness/answer_relevancy only need the live
question+answer) - it exists purely so a human reading EVALUATION.md can
sanity-check a low score against what the right answer actually looks like.

For HYBRID cases, `relevant_sections` labels only the DOCUMENTATION half of
the question (see hybrid_doc_subquestion) - consistent with this being a
RAG-quality eval, not a SQL-correctness one.

Growing this list over time is the intended way to make the eval more
useful, the same way tests/test_router_golden.py's GOLDEN list is meant to
grow.
"""

from __future__ import annotations

from dataclasses import dataclass

Section = tuple[str, str]  # (doc, section) - matches eval/metrics.py's Section


@dataclass(frozen=True)
class EvalCase:
    question: str
    route: str  # "RAG" or "HYBRID" - which route this question should take
    relevant_sections: tuple[Section, ...]
    reference_answer: str
    # For HYBRID cases: the sub-question actually routed to RAG internally
    # (HybridPath splits doc_subquestion/sql_subquestion) - None for RAG cases,
    # where the whole question is the doc question.
    hybrid_doc_subquestion: str | None = None


RAG_EVAL_SET: tuple[EvalCase, ...] = (
    EvalCase(
        question="What deployment modes does Product XYZ support?",
        route="RAG",
        relevant_sections=(("Enablement Pack", "Deployment Guide (v3.0)"),),
        reference_answer="Cloud, On-Prem, and Hybrid.",
    ),
    EvalCase(
        question="What is included in the Growth pricing tier and how much does it cost?",
        route="RAG",
        relevant_sections=(("Enablement Pack", "Packaging & Pricing Cheat Sheet"),),
        reference_answer=(
            "$12,000/month; includes XYZ-CORE + XYZ-ANALYTICS; up to 250 seats, "
            "3 workspaces."
        ),
    ),
    EvalCase(
        question="What are the common SKUs for Product XYZ?",
        route="RAG",
        relevant_sections=(("Enablement Pack", "Common SKUs"),),
        reference_answer=(
            "XYZ-CORE (orchestration + basic RAG), XYZ-ANALYTICS (metrics, "
            "dashboards, cohort analysis), XYZ-AUTOMATION (triggered workflows + "
            "connectors), XYZ-SECURITY (audit export, data masking, policy packs)."
        ),
    ),
    EvalCase(
        question="What are the different stages in the Opportunity Tracker's stage "
        "progression playbook?",
        route="RAG",
        relevant_sections=(("Field Guide", "Stage progression playbook"),),
        reference_answer=(
            "1-Qualify, 2-Discover, 3-Solution Fit, 4-Commercial Align, 5-Commit, "
            "6-Closed Won/Handoff."
        ),
    ),
    EvalCase(
        question="What are the exit criteria for the Solution Fit stage?",
        route="RAG",
        relevant_sections=(("Field Guide", "Stage progression playbook"),),
        reference_answer=(
            "Proposed architecture validated; demo/use-case proof aligns to "
            "success criteria."
        ),
    ),
    EvalCase(
        question="What is the recommended action when a deployment status is "
        "Ready - Pilot scheduled?",
        route="RAG",
        relevant_sections=(("Field Guide", "Deployment status taxonomy & risk scoring"),),
        reference_answer="Run pilot; capture results for go/no-go.",
    ),
    EvalCase(
        question="What are the four dimensions of the deployment risk scoring rubric, "
        "and what do the total-score thresholds mean?",
        route="RAG",
        relevant_sections=(("Field Guide", "Deployment status taxonomy & risk scoring"),),
        reference_answer=(
            "Stakeholder risk, technical risk, commercial risk, delivery readiness "
            "(each 0-25); total 0-100, thresholds 0-30 Low, 31-60 Medium, 61-100 High."
        ),
    ),
    EvalCase(
        question="What is the source system and example value for the "
        "expected_close_date field in the tracker's data dictionary?",
        route="RAG",
        relevant_sections=(("Field Guide", "Data dictionary (core fields)"),),
        reference_answer="Source system CRM; example value 2026-09-30.",
    ),
    EvalCase(
        question="Who is a champion, per the tracker's data dictionary?",
        route="RAG",
        relevant_sections=(("Field Guide", "Data dictionary (core fields)"),),
        reference_answer=(
            "Named person who benefits from the solution and drives internal "
            "approval."
        ),
    ),
    EvalCase(
        question="What is deployment_risk_score and what does its value represent?",
        route="RAG",
        relevant_sections=(("Field Guide", "Data dictionary (core fields)"),),
        reference_answer="0-100 risk score from the rubric; higher means riskier.",
    ),
    EvalCase(
        question="What is our 2024 win rate for Enterprise, and what does the field "
        "guide require before a deal reaches Commit?",
        route="HYBRID",
        relevant_sections=(("Field Guide", "Stage progression playbook"),),
        reference_answer=(
            "Doc half: all approvals complete; signature plan confirmed; delivery "
            "kickoff scheduled."
        ),
        hybrid_doc_subquestion="what gates a deal before the Commit stage",
    ),
    EvalCase(
        question="Show 2024 NA deals stuck in Negotiation and explain the risk "
        "scoring rubric that applies to them",
        route="HYBRID",
        relevant_sections=(("Field Guide", "Deployment status taxonomy & risk scoring"),),
        reference_answer=(
            "Doc half: sum of four 0-25 dimensions (stakeholder, technical, "
            "commercial, delivery readiness); thresholds 0-30 Low, 31-60 Medium, "
            "61-100 High."
        ),
        hybrid_doc_subquestion="the 0-100 deployment risk scoring rubric",
    ),
)
