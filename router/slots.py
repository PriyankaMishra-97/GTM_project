"""Slot schemas: what a question must specify before it can be answered.

Role in architecture: turns "is this question answerable?" into a mechanical
check. The router LLM extracts slot values in the same call that picks a route;
this module decides whether what it extracted is sufficient. Anything missing
forces the ASK route (rule R1 in router/rules.py).

Why slots at all: on this dataset the same question means different things
without them - "how's pipeline?" over 2023 vs 2024, or NA vs global, differs by
an order of magnitude. Guessing produces a confident wrong number, which is worse
than one clarifying question.

In:  extracted slot dict + route.
Out: list of missing slot names.
"""

from __future__ import annotations

import re
from typing import Any

# Slot names are stable identifiers shared with the router prompt and the trace.
TIME_RANGE = "time_range"
SEGMENT_OR_REGION = "segment_or_region"
STAGE_DEFINITION = "stage_definition"
PRODUCT_AREA = "product_area"

# Quantitative routes need a scope: a period and a population.
SQL_REQUIRED = (TIME_RANGE, SEGMENT_OR_REGION)

# Doc questions only need a product area when the question is broad enough that
# the two PDFs would give different answers.
AMBIGUOUS_DOC_MARKERS = (
    "how does it work", "what does it do", "tell me about the product",
    "what are the features", "overview", "explain the tool", "what is it",
)

# Vague quantifiers that masquerade as a time range (rule R2).
VAGUE_TIME_WORDS = (
    "recently", "lately", "these days", "nowadays", "currently", "at the moment",
    "right now", "of late", "so far", "trending",
)
# Note: "how is" is deliberately NOT here - "how is the risk score calculated?" is
# a well-specified documentation question. Only the colloquial contraction is vague.
VAGUE_SUPERLATIVES = ("top", "best", "worst", "biggest", "how's", "hows", "looking")

# Anything that looks like an explicit period satisfies TIME_RANGE.
_EXPLICIT_TIME_RE = re.compile(
    r"(20\d{2})"                                   # a year: 2023, 2024
    r"|(q[1-4])"                                   # a quarter: Q3
    r"|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*20\d{2}"
    r"|(last|past|previous|trailing)\s+\d+\s*(day|week|month|quarter|year)"
    r"|(h1|h2|first half|second half)"
    r"|\d{4}-\d{2}(-\d{2})?"                       # ISO date / month
    r"|(since|between|from)\s+\w+",
    re.IGNORECASE,
)

_REGION_WORDS = ("na", "north america", "emea", "apac", "latam", "global", "all regions", "worldwide")
_SEGMENT_WORDS = ("enterprise", "mid-market", "midmarket", "mid market", "smb", "all segments", "every segment")


def is_filled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in ("", "none", "null", "unknown", "unspecified", "n/a")
    return bool(value)


def has_explicit_time(question: str) -> bool:
    """Does the raw question name a period the SQL layer can filter on?"""
    return bool(_EXPLICIT_TIME_RE.search(question))


def has_explicit_population(question: str) -> bool:
    """Does the raw question name a region or segment (including 'all')?"""
    q = question.lower()
    return any(w in q for w in _REGION_WORDS + _SEGMENT_WORDS)


def is_ambiguous_doc_question(question: str) -> bool:
    q = question.lower()
    return any(m in q for m in AMBIGUOUS_DOC_MARKERS)


def required_slots(route: str, question: str) -> tuple[str, ...]:
    """Slots this route+question combination must have filled.

    `stage_definition` is deliberately NOT required here even for a stage
    question: it isn't information only the user has - the SQL side can only
    ever query `opportunities.stage`'s real enum (the playbook's 6-stage names
    don't exist in the database), and the doc side only ever explains the
    playbook (the database enum has no exit criteria of its own). Which one
    applies is determined by the route, not by asking - see
    `sql.generate.SqlGenerator.slot_block`, which forces "database stages" for
    every SQL/HYBRID call regardless of what the router extracted.
    """
    if route in ("SQL", "HYBRID"):
        return SQL_REQUIRED
    if route == "RAG" and is_ambiguous_doc_question(question):
        return (PRODUCT_AREA,)
    return ()


def missing_slots(route: str, question: str, slots: dict[str, Any] | None) -> list[str]:
    """Which required slots are unfilled, considering the raw question as evidence.

    The raw question is checked as a fallback because a small router model
    sometimes routes correctly but forgets to copy an obvious filter into the
    slot dict; penalising the user for the model's sloppiness would mean asking
    a clarifying question they already answered.
    """
    slots = slots or {}
    missing: list[str] = []
    for slot in required_slots(route, question):
        if is_filled(slots.get(slot)):
            continue
        if slot == TIME_RANGE and has_explicit_time(question):
            continue
        if slot == SEGMENT_OR_REGION and has_explicit_population(question):
            continue
        missing.append(slot)
    return missing


SLOT_QUESTIONS = {
    TIME_RANGE: "Which time range should I use? (data covers 2023-01-01 to 2024-12-21 for created deals, close dates run to 2025-09-09)",
    SEGMENT_OR_REGION: "Which region or segment? (regions: NA, EMEA, APAC, LATAM; segments: Enterprise, Mid-Market, SMB - or 'all')",
    PRODUCT_AREA: "Which area do you mean - Product XYZ itself (Enablement Pack) or the Opportunity Tracker (Field Guide)?",
}
