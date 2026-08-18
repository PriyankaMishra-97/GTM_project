"""Turn orchestrator: `GTMCopilot`, the one object the UI talks to.

Role in architecture: owns the collaborators and the Trace lifecycle, and
dispatches each turn to exactly one path. One user turn is:

    question
      -> Router          (LLM proposes route + slots)
      -> RuleEngine      (deterministic override; may force ASK/REFUSE)
      -> one of RagPath | SqlPath | HybridPath | Clarifier | refusal
      -> Answer(text, route, trace)

Dependencies are constructor-injected, so a test (or the Gemini build) can swap
the LLM client, the retriever or the schema without touching this file. No path
may call another path's internals - HybridPath composes SqlPath and RagPath as
objects, so behaviour cannot drift between routes.

In:  question string.
Out: `Answer` (text + route + citations + finished Trace).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ask.clarify import Clarifier
from ask.history_reframe import HistoryReframer
from ask.reframe import Reframer
from core import config
from core.auth import UserProfile
from core.llm_client import LLMClient, get_client
from core.trace import Trace
from hybrid.pipeline import HybridPath
from rag.answer import RagPath
from rag.index import Index
from rag.retrieve import Hit, Retriever
from router.llm_router import Router, RouterDecision
from router.rules import RuleEngine
from sql.guard import ScopedQueryGuardRule, SqlGuard
from sql.pipeline import SqlPath
from sql.schema import SchemaCatalog


@dataclass
class Answer:
    text: str
    route: str
    trace: Trace
    hits: list[Hit] = field(default_factory=list)
    sql: str | None = None

    def persisted(self) -> "Answer":
        """Append the trace to ./storage/traces.jsonl and return self."""
        self.trace.persist()
        return self


class GTMCopilot:
    """The assembled system. Construct once, call `answer()` per turn."""

    def __init__(
        self,
        client: LLMClient | None = None,
        catalog: SchemaCatalog | None = None,
        index: Index | None = None,
        router: Router | None = None,
        rule_engine: RuleEngine | None = None,
    ) -> None:
        self.client = client or get_client()
        self.catalog = catalog or SchemaCatalog()
        self.index = index or Index()
        self.retriever = Retriever(self.index)

        # The router sees the same generated card as the SQL path (compacted
        # to table/column facts), so both reason about one schema description.
        self.router = router or Router(self.client, schema_card=self.catalog.card)
        self.rules = rule_engine or RuleEngine()

        # One instance per path, built once and reused across turns.
        self.rag_path = RagPath(self.retriever, self.client)
        self.clarifier = Clarifier(self.client)
        self.reframer = Reframer(self.client)
        self.history_reframer = HistoryReframer(self.client)

        # SQL/HYBRID paths are per-user: each carries a guard bound to that
        # user's region/segment ACL (ScopedQueryGuardRule), so they cannot be
        # a single shared instance - see _sql_path_for's docstring.
        self._sql_paths: dict[str, SqlPath] = {}
        self._hybrid_paths: dict[str, HybridPath] = {}

        for warning in self.catalog.drift_warnings():
            print(f"[schema-card drift] {warning}")

    def _sql_path_for(self, user: UserProfile) -> SqlPath:
        """One SqlPath per distinct logged-in user, built lazily and cached.

        The guard's ScopedQueryGuardRule is bound to this user's allowed set
        at construction time. A single shared SqlPath/SqlGuard instance would
        leak one user's ACL into a concurrent user's query, since Streamlit
        serves multiple browser sessions from the same process - so this
        cache, keyed by username (a small, fixed team), replaces the single
        shared instance the rest of this class still uses for user-agnostic
        collaborators (rag_path, clarifier).
        """
        if user.username not in self._sql_paths:
            guard = SqlGuard(
                self.catalog.info,
                rules=[
                    *[cls() for cls in SqlGuard.DEFAULT_RULES],
                    ScopedQueryGuardRule(user.allowed_scope_values()),
                ],
            )
            self._sql_paths[user.username] = SqlPath(self.catalog, self.client, guard=guard)
        return self._sql_paths[user.username]

    def _hybrid_path_for(self, user: UserProfile) -> HybridPath:
        """One HybridPath per user, wired to that user's per-user SqlPath."""
        if user.username not in self._hybrid_paths:
            self._hybrid_paths[user.username] = HybridPath(
                self.catalog,
                self.client,
                sql_path=self._sql_path_for(user),
                retriever=self.retriever,
            )
        return self._hybrid_paths[user.username]

    # ------------------------------------------------------------------ turn --
    def answer(
        self,
        question: str,
        user: UserProfile,
        pending_clarification: str | None = None,
        pending_missing_slots: list[str] | None = None,
        recent_turns: list[dict] | None = None,
    ) -> Answer:
        """Run one full turn.

        `pending_clarification`/`pending_missing_slots` are the caller's record
        of the last ASK, if this turn follows one. When set, a reframe step
        decides whether `question` answers that clarification (merge into one
        standalone question) or pivots to something unrelated (route it
        alone) - see ask/reframe.py for why that's a separate call rather than
        part of the main router prompt.

        `recent_turns` is the caller's record of the last (up to 2) turns that
        produced a real SQL/RAG/HYBRID answer, used ONLY when this turn does
        NOT follow an ASK - see ask/history_reframe.py. The two mechanisms are
        mutually exclusive (this if/elif is the guarantee): a turn following
        an ASK never also consults recent_turns, and vice versa.
        """
        trace = Trace(question=question, user=user.username)

        effective_question = question
        if pending_clarification:
            reframe_decision = self.reframer.reframe(
                pending_clarification, pending_missing_slots or [], question, trace
            )
            trace.pending_question = pending_clarification
            trace.is_new_topic = reframe_decision.is_new_topic
            effective_question = (
                question if reframe_decision.is_new_topic else reframe_decision.effective_question
            )
            trace.question = effective_question
        elif recent_turns:
            history_decision = self.history_reframer.reframe(recent_turns, question, trace)
            trace.history_reframe_applied = not history_decision.is_new_topic
            effective_question = (
                question if history_decision.is_new_topic else history_decision.effective_question
            )
            trace.question = effective_question

        decision, outcome_route = self._route(effective_question, trace, user)
        route = outcome_route

        if route == "REFUSE":
            return Answer(
                text=trace.refusal_message or "I can't do that.",
                route="REFUSE",
                trace=trace.finish(),
            ).persisted()

        if route == "ASK":
            text = self.clarifier.clarify(
                effective_question,
                trace.missing_slots or decision.missing_slots,
                trace.rule_detail or decision.rationale,
                trace,
            )
            return Answer(text=text, route="ASK", trace=trace.finish()).persisted()

        if route == "RAG":
            text, hits = self.rag_path.run(effective_question, trace)
            return Answer(
                text=text, route="RAG", trace=trace.finish(), hits=hits
            ).persisted()

        if route == "SQL":
            result = self._sql_path_for(user).run(
                effective_question, trace, slots=decision.slots
            )
            return Answer(
                text=result.answer, route="SQL", trace=trace.finish(), sql=result.sql
            ).persisted()

        if route == "HYBRID":
            text, hits = self._hybrid_path_for(user).run(
                effective_question,
                decision.sql_subquestion,
                decision.doc_subquestion,
                trace,
                slots=decision.slots,
            )
            return Answer(
                text=text,
                route="HYBRID",
                trace=trace.finish(),
                hits=hits,
                sql=trace.generated_sql,
            ).persisted()

        # Unreachable given the Route literal, but a defensive default beats a
        # KeyError in front of a user.
        trace.error(f"unknown route '{route}'")
        return Answer(
            text="I couldn't determine how to answer that. Could you rephrase?",
            route="ASK",
            trace=trace.finish(),
        ).persisted()

    def _route(
        self, question: str, trace: Trace, user: UserProfile
    ) -> tuple[RouterDecision, str]:
        """Stage 1: LLM proposal, then deterministic override. Writes the trace."""
        with trace.stage("route"):
            decision, err = self.router.decide(question, trace=trace)
        if err:
            trace.error(f"router: {err}")

        trace.llm_proposed_route = decision.route
        trace.router_confidence = decision.confidence
        trace.router_rationale = decision.rationale
        trace.doc_subquestion = decision.doc_subquestion
        trace.sql_subquestion = decision.sql_subquestion

        outcome = self.rules.apply(decision, question, user)
        # Read AFTER rules run: RegionScopeRule may narrow decision.slots
        # in place (e.g. "all" -> a concrete allowed list), and the trace
        # should record what was actually used, not the pre-narrowed value.
        trace.slots = dict(decision.slots)
        trace.rule_override = outcome.rule_id
        trace.final_route = outcome.final_route
        trace.missing_slots = outcome.missing_slots
        trace.rule_detail = outcome.detail
        trace.refusal_message = outcome.refusal_message
        return decision, outcome.final_route

    # ---------------------------------------------------------------- status --
    def preflight(self) -> dict[str, Any]:
        """Startup status for the UI sidebar / CLI. Never raises."""
        status: dict[str, Any] = {
            "db_ready": config.DB_PATH.exists(),
            "index": self.index.stats(),
            "schema_warnings": [],
            "tables": [],
        }
        status.update(self.client.status())
        status["provider_ready"] = status.get("ready", False)
        try:
            status["tables"] = sorted(self.catalog.table_names)
            status["schema_warnings"] = self.catalog.drift_warnings()
        except Exception as exc:
            status["error"] = f"{status.get('error', '')} {exc}".strip()
        return status


# --------------------------------------------------------------------------
# module-level convenience wrappers (a single shared copilot for the UI/CLI)
# --------------------------------------------------------------------------
_COPILOT: GTMCopilot | None = None


def get_copilot(client: LLMClient | None = None) -> GTMCopilot:
    global _COPILOT
    if _COPILOT is None:
        _COPILOT = GTMCopilot(client=client)
    return _COPILOT


def answer_question(
    question: str,
    *,
    user: UserProfile,
    client: LLMClient | None = None,
    pending_clarification: str | None = None,
    pending_missing_slots: list[str] | None = None,
    recent_turns: list[dict] | None = None,
) -> Answer:
    return get_copilot(client).answer(
        question, user, pending_clarification, pending_missing_slots, recent_turns
    )


def preflight(client: LLMClient | None = None) -> dict[str, Any]:
    try:
        return get_copilot(client).preflight()
    except Exception as exc:
        # A missing DB must not crash the sidebar - report it instead.
        return {
            "ready": False, "provider_ready": False, "models": [],
            "db_ready": config.DB_PATH.exists(), "index": Index().stats(),
            "schema_warnings": [], "tables": [], "error": str(exc),
        }


def get_schema() -> SchemaCatalog:
    return get_copilot().catalog
