"""Deterministic rule engine that can OVERRIDE the LLM router.

Role in architecture: "retrieval proposes, the rule engine disposes." The LLM is
good at intent and bad at guarantees. These rules are the guarantees.

Each rule is a `RoutingRule` subclass with a stable `rule_id`. `RuleEngine` runs
them in order and the first one that fires wins; the firing rule ID is recorded
in the trace, so any route is explainable after the fact. Adding a guarantee
means adding a class - not editing a growing if/elif chain.

RULES (in order)
  R00_PII_REQUEST  - question asks for personal PII -> REFUSE, never answered.
  R0_WRITE_INTENT  - write/destructive intent -> REFUSE, never SQL.
  R0B_OFF_DOMAIN   - router proposed OFF_TOPIC -> REFUSE, out of scope.
  R0C_HYBRID_NO_SQL- HYBRID proposal with no sql_subquestion -> RAG (it isn't
                     really hybrid; nothing to compute).
  R1_MISSING_SLOT  - SQL/HYBRID with a required slot unfilled -> ASK.
  R1B_REGION_SCOPE - SQL/HYBRID names a region/segment outside the logged-in
                     user's allowed scope -> REFUSE; "all" narrows in place to
                     the user's allowed set instead of overriding the route.
  R2_VAGUE_TIME    - vague quantifier with no explicit time range -> ASK.
  R3_LOW_CONFIDENCE- router confidence below threshold -> ASK.

In:  RouterDecision + question + the logged-in user's UserProfile.
Out: `RoutingOutcome(final_route, rule_id, missing_slots, refusal_message)`.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from core import config
from core.auth import UserProfile
from router.llm_router import RouterDecision
from router import slots as slot_lib

# Write/destructive intent. Matched on word boundaries so "updated pricing note"
# or "insertion" do not trip it.
WRITE_INTENT_RE = re.compile(
    r"\b(delete|drop|truncate|update|insert|upsert|alter|overwrite|wipe|purge|"
    r"remove\s+(the\s+)?(row|record|table|data)|set\s+\w+\s*=|grant|revoke)\b",
    re.IGNORECASE,
)

REFUSAL_MESSAGE = (
    "I can't do that. This assistant has **read-only** access to the GTM database "
    "by design - every query runs through an allowlist guard and a read-only "
    "SQLite connection, so writes, schema changes and deletions are not possible.\n\n"
    "If you want to *see* the rows you were going to change, ask me to list them "
    "and I'll run a SELECT instead."
)

# Personal PII: literal identifier formats plus phrasing that seeks someone's
# private contact/identity details. Deliberately NOT bare "email"/"phone" -
# those are legitimate activity channel values in the schema (channel='Email').
PII_RE = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"                              # SSN format
    r"|\b(?:\d[ -]?){13,16}\b"                            # card-number-shaped digit run
    r"|\b(email\s*(address|id)|"
    r"(phone|mobile|cell)\s*(no\.?|number|#)|"
    r"home (address|phone)|mailing address|contact (number|info|details)|"
    r"personal (email|phone|contact|address|information|details)|"
    r"social security( number)?|\bssn\b|passport(\s+number)?|"
    r"driver'?s licen[sc]e|credit card( number)?|bank account( number)?|"
    r"routing number|date of birth|\bdob\b)\b",
    re.IGNORECASE,
)

PII_REFUSAL_MESSAGE = (
    "I can't share or look up personal identifying information (PII) - things "
    "like SSNs, personal emails/phone numbers, home addresses, or financial "
    "account numbers. This assistant only surfaces aggregated GTM business data "
    "(accounts, opportunities, deployments) and product/process documentation, "
    "not individuals' personal details."
)

OFF_DOMAIN_REFUSAL_MESSAGE = (
    "That's outside what I can help with. This assistant only answers questions "
    "about Product XYZ, the Opportunity Tracker, and the GTM database (pipeline, "
    "bookings, deployments, and the two reference documents). Ask me something "
    "about those and I'll take it from there."
)


@dataclass
class RoutingOutcome:
    final_route: str
    rule_id: str | None = None  # None => the LLM's proposal stood
    missing_slots: list[str] = field(default_factory=list)
    refusal_message: str | None = None
    detail: str = ""


class RoutingRule(ABC):
    """One deterministic post-check on the router's proposal."""

    rule_id: str = "R?"

    @abstractmethod
    def evaluate(
        self, decision: RouterDecision, question: str, user: UserProfile
    ) -> RoutingOutcome | None:
        """Return an overriding outcome, or None to let the next rule run."""


class PiiRequestRule(RoutingRule):
    """R00 - refuse requests for personal PII regardless of the proposed route.

    Runs first: a PII ask dressed up as a SQL or RAG question (e.g. "look up
    John's SSN in the account notes") must never reach a data path.
    """

    rule_id = "R00_PII_REQUEST"

    def evaluate(
        self, decision: RouterDecision, question: str, user: UserProfile
    ) -> RoutingOutcome | None:
        if not PII_RE.search(question):
            return None
        return RoutingOutcome(
            final_route="REFUSE",
            rule_id=self.rule_id,
            refusal_message=PII_REFUSAL_MESSAGE,
            detail="personal PII request detected",
        )


class OffDomainRule(RoutingRule):
    """R0B - the router itself judged the question out of scope.

    Domain-vs-not is a semantic call the LLM already makes when proposing
    OFF_TOPIC; this rule just turns that proposal into the same hard REFUSE
    the other guarantees produce, so it can't be downgraded to ASK later by
    R3_LOW_CONFIDENCE.
    """

    rule_id = "R0B_OFF_DOMAIN"

    def evaluate(
        self, decision: RouterDecision, question: str, user: UserProfile
    ) -> RoutingOutcome | None:
        if decision.route != "OFF_TOPIC":
            return None
        return RoutingOutcome(
            final_route="REFUSE",
            rule_id=self.rule_id,
            refusal_message=OFF_DOMAIN_REFUSAL_MESSAGE,
            detail="router classified question as outside the GTM domain",
        )


class WriteIntentRule(RoutingRule):
    """R0 - refuse write/destructive intent regardless of the proposed route.

    Fires even when the LLM helpfully proposed RAG for "how do I delete an
    opportunity": the refusal is about what the user asked for, not the route.
    """

    rule_id = "R0_WRITE_INTENT"

    def evaluate(
        self, decision: RouterDecision, question: str, user: UserProfile
    ) -> RoutingOutcome | None:
        if not WRITE_INTENT_RE.search(question):
            return None
        return RoutingOutcome(
            final_route="REFUSE",
            rule_id=self.rule_id,
            refusal_message=REFUSAL_MESSAGE,
            detail="write/destructive intent detected",
        )


class HybridWithoutSqlRule(RoutingRule):
    """R0C - a HYBRID proposal with nothing to compute is not really hybrid.

    The router sometimes copies "HYBRID" from a near-identical few-shot
    example (a compound win-rate-and-policy question) onto a question that is
    actually pure documentation: doc_subquestion gets filled but
    sql_subquestion stays empty - even the router's own output shows there is
    no computation to do. Runs before R1_MISSING_SLOT so a bogus HYBRID never
    demands time_range/segment_or_region a purely doc-only question doesn't
    need. Real case: "what does the field guide require before a deal reaches
    Commit?" was misrouted HYBRID and then blocked on an ASK for filters a
    plain RAG lookup never needed.
    """

    rule_id = "R0C_HYBRID_NO_SQL"

    def evaluate(
        self, decision: RouterDecision, question: str, user: UserProfile
    ) -> RoutingOutcome | None:
        if decision.route != "HYBRID" or slot_lib.is_filled(decision.sql_subquestion):
            return None
        return RoutingOutcome(
            final_route="RAG",
            rule_id=self.rule_id,
            detail="HYBRID proposal had no sql_subquestion - downgraded to RAG",
        )


class MissingSlotRule(RoutingRule):
    """R1 - a quantitative route with an unfilled required slot cannot be correct.

    Without a period and a population it can only produce a number that is
    confident, not right.
    """

    rule_id = "R1_MISSING_SLOT"

    def evaluate(
        self, decision: RouterDecision, question: str, user: UserProfile
    ) -> RoutingOutcome | None:
        if decision.route not in ("SQL", "HYBRID"):
            return None
        missing = slot_lib.missing_slots(decision.route, question, decision.slots)
        if not missing:
            return None
        return RoutingOutcome(
            final_route="ASK",
            rule_id=self.rule_id,
            missing_slots=missing,
            detail=f"missing required slots: {', '.join(missing)}",
        )


class RegionScopeRule(RoutingRule):
    """R1B - a filled segment_or_region slot must be within the user's ACL.

    Runs after R1_MISSING_SLOT, which guarantees the slot is filled by the
    time this rule sees it - so "missing" is not a case this rule handles.
    A named value outside the user's allowed set is refused outright (asking
    the user to pick a different value would be confusing - they asked for
    something specific and don't have access to it). "all" is narrowed
    in-place to the user's allowed set rather than treated as an override, so
    routing continues unchanged and sql/generate.py picks up the narrowed
    list with no new plumbing.
    """

    rule_id = "R1B_REGION_SCOPE"

    def evaluate(
        self, decision: RouterDecision, question: str, user: UserProfile
    ) -> RoutingOutcome | None:
        if decision.route not in ("SQL", "HYBRID"):
            return None
        value = decision.slots.get(slot_lib.SEGMENT_OR_REGION)
        if not slot_lib.is_filled(value):
            return None  # R1_MISSING_SLOT already handles this case

        allowed = user.allowed_scope_values()
        if allowed is None:
            return None  # unrestricted user

        if str(value).strip().lower() == "all":
            decision.slots[slot_lib.SEGMENT_OR_REGION] = sorted(allowed)
            return None  # mutate in place; not an override

        if value not in allowed:
            return RoutingOutcome(
                final_route="REFUSE",
                rule_id=self.rule_id,
                refusal_message=(
                    f"You don't have access to {value}. "
                    f"Your allowed scope: {', '.join(sorted(allowed))}."
                ),
                detail=f"user '{user.username}' requested out-of-scope value '{value}'",
            )
        return None


class VagueTimeRule(RoutingRule):
    """R2 - vague quantifier with no explicit period.

    Applies to doc questions too ("what's the best deployment mode recently?" is
    not answerable as asked), but the superlative half is scoped to quantitative
    routes so "How is the risk score calculated?" is left alone.
    """

    rule_id = "R2_VAGUE_TIME"

    @staticmethod
    def _is_vague(question: str, route: str) -> bool:
        q = question.lower()
        if any(w in q for w in slot_lib.VAGUE_TIME_WORDS):
            return True
        if route not in ("SQL", "HYBRID"):
            return False
        return any(
            re.search(rf"\b{re.escape(w)}\b", q) for w in slot_lib.VAGUE_SUPERLATIVES
        )

    def evaluate(
        self, decision: RouterDecision, question: str, user: UserProfile
    ) -> RoutingOutcome | None:
        if not self._is_vague(question, decision.route):
            return None
        if slot_lib.has_explicit_time(question):
            return None
        if slot_lib.is_filled(decision.slots.get(slot_lib.TIME_RANGE)):
            return None

        missing = [slot_lib.TIME_RANGE]
        if decision.route in ("SQL", "HYBRID") and not slot_lib.has_explicit_population(question):
            missing.append(slot_lib.SEGMENT_OR_REGION)
        return RoutingOutcome(
            final_route="ASK",
            rule_id=self.rule_id,
            missing_slots=missing,
            detail="vague quantifier without an explicit time range",
        )


class LowConfidenceRule(RoutingRule):
    """R3 - the model itself is unsure. Cheaper to ask than to be wrong."""

    rule_id = "R3_LOW_CONFIDENCE"

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = threshold if threshold is not None else config.ROUTER_MIN_CONFIDENCE

    def evaluate(
        self, decision: RouterDecision, question: str, user: UserProfile
    ) -> RoutingOutcome | None:
        if decision.route == "ASK" or decision.confidence >= self.threshold:
            return None
        return RoutingOutcome(
            final_route="ASK",
            rule_id=self.rule_id,
            missing_slots=decision.missing_slots
            or slot_lib.missing_slots(decision.route, question, decision.slots),
            detail=f"router confidence {decision.confidence:.2f} < {self.threshold}",
        )


class RuleEngine:
    """Runs rules in order; first match wins."""

    DEFAULT_RULES: tuple[type[RoutingRule], ...] = (
        PiiRequestRule,
        WriteIntentRule,
        OffDomainRule,
        HybridWithoutSqlRule,
        MissingSlotRule,
        RegionScopeRule,
        VagueTimeRule,
        LowConfidenceRule,
    )

    def __init__(self, rules: list[RoutingRule] | None = None) -> None:
        self.rules = rules if rules is not None else [cls() for cls in self.DEFAULT_RULES]

    def apply(self, decision: RouterDecision, question: str, user: UserProfile) -> RoutingOutcome:
        for rule in self.rules:
            outcome = rule.evaluate(decision, question, user)
            if outcome is not None:
                return outcome
        # No rule fired: the LLM's proposal stands.
        return RoutingOutcome(
            final_route=decision.route,
            rule_id=None,
            missing_slots=decision.missing_slots if decision.route == "ASK" else [],
        )


def apply_rules(decision: RouterDecision, question: str, user: UserProfile) -> RoutingOutcome:
    """Module-level convenience wrapper around `RuleEngine`."""
    return RuleEngine().apply(decision, question, user)
