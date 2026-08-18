"""LLM routing: one ROUTER_MODEL call that classifies and extracts slots together.

Role in architecture: stage 1 of every turn. `Router.decide()` produces a
*proposal* only - router/rules.py has the final say ("retrieval proposes, the
rule engine disposes"). Classification and slot extraction happen in ONE call:
a second call would add ~1s to every turn for information the model already has
in context.

Failure policy: if the model cannot produce valid JSON twice, we degrade to ASK.
Asking a clarifying question is always safe; guessing a route is not.

In:  question string.
Out: `RouterDecision` (pydantic) - proposal + slots + confidence + rationale.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from core.llm_client import LLMClient, LLMJSONError, LLMUnavailable, get_client
from core.trace import Trace
from router import prompts
from sql.schema import default_schema_card

Route = Literal["RAG", "SQL", "HYBRID", "ASK", "OFF_TOPIC"]


class RouterDecision(BaseModel):
    """Strict schema handed to the provider's JSON mode."""

    route: Route
    missing_slots: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""
    slots: dict[str, Any] = Field(default_factory=dict)
    doc_subquestion: str | None = None
    sql_subquestion: str | None = None

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @field_validator("slots", mode="before")
    @classmethod
    def _drop_nulls(cls, v: Any) -> dict[str, Any]:
        """Models emit `"time_range": "null"` as often as a real null; normalise."""
        if not isinstance(v, dict):
            return {}
        return {
            k: val
            for k, val in v.items()
            if val is not None and str(val).strip().lower() not in ("null", "none", "", "n/a")
        }


class Router:
    """Wraps the router LLM call. Stateless beyond the client it holds."""

    def __init__(
        self, client: LLMClient | None = None, schema_card: str | None = None
    ) -> None:
        self.client = client or get_client()
        # Injected by the orchestrator from its SchemaCatalog; built lazily
        # otherwise, so importing this module never needs the database.
        self._schema_card = schema_card if schema_card is not None else None

    def _compact_schema_card(self) -> str:
        # The router does not write SQL, so business definitions would be
        # wasted tokens on the small model - table/column names only.
        card = self._schema_card if self._schema_card is not None else default_schema_card()
        return card.split("BUSINESS DEFINITIONS")[0].strip()

    def decide(
        self, question: str, trace: Trace | None = None
    ) -> tuple[RouterDecision, str | None]:
        """Return (decision, error). On failure the decision is a safe ASK fallback."""
        system = prompts.ROUTER_SYSTEM.format(schema_card=self._compact_schema_card())
        user = prompts.ROUTER_USER.format(question=question)
        try:
            decision = self.client.chat_json(
                system,
                user,
                RouterDecision,
                model=self.client.router_model,
                trace=trace,
                stage="route",
            )
            return decision, None
        except (LLMJSONError, LLMUnavailable) as exc:
            return (
                RouterDecision(
                    route="ASK",
                    missing_slots=["time_range", "segment_or_region"],
                    confidence=0.0,
                    rationale="router failed; defaulting to clarification",
                ),
                str(exc),
            )


def route_question(
    question: str, *, client: LLMClient | None = None, trace: Trace | None = None
) -> tuple[RouterDecision, str | None]:
    """Module-level convenience wrapper, kept for callers that don't need a Router instance."""
    return Router(client).decide(question, trace=trace)
