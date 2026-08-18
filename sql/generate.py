"""Text-to-SQL generation (ANSWER_MODEL).

Role in architecture: `SqlGenerator` turns a question (+ resolved slots) into a
candidate SELECT. It produces *candidates only* - correctness of intent is the
model's job, safety is sql/guard.py's job. The two are deliberately separate
classes so the guard can be unit tested without an LLM.

In:  question, schema card, slot values.
Out: raw SQL string (unvalidated).
"""

from __future__ import annotations

import re
from typing import Any

from core.llm_client import LLMClient, get_client
from core.trace import Trace
from router.slots import STAGE_DEFINITION
from sql import prompts
from sql.schema import default_schema_card

_FENCE = re.compile(r"^```(?:sql)?|```$", re.MULTILINE)


class SqlGenerator:
    """Produces candidate SQL. One instance per orchestrator."""

    def __init__(self, client: LLMClient | None = None, schema_card: str | None = None) -> None:
        self.client = client or get_client()
        # Callers holding a SchemaCatalog pass its generated card; the default
        # is built lazily (never at import time - the DB may not exist yet).
        self.schema_card = schema_card if schema_card is not None else default_schema_card()

    @staticmethod
    def clean(raw: str) -> str:
        """Strip markdown fences / stray prose the model may add despite rule 1."""
        text = _FENCE.sub("", raw).strip()
        # Some models prefix "SQL:" or "Query:".
        text = re.sub(r"^\s*(sql|query)\s*:\s*", "", text, flags=re.IGNORECASE)
        return text.strip().rstrip(";").strip()

    @staticmethod
    def _render_value(v: Any) -> str:
        """A list means "any of these" (e.g. a region ACL narrowed from "all") -
        render it as natural language, not Python's list repr."""
        if isinstance(v, (list, tuple, set, frozenset)):
            return "one of: " + ", ".join(str(x) for x in v)
        return str(v)

    @staticmethod
    def slot_block(slots: dict[str, Any] | None) -> str:
        """Render resolved slots as explicit filters so the model cannot forget them.

        `stage_definition` is always forced to "database stages" here, regardless
        of what the router extracted: the Field Guide's 6-stage playbook names
        don't exist in `opportunities.stage` (see the schema card's own caution),
        so telling the SQL model anything else would ask it to invent a column
        value that has no real database equivalent.
        """
        if not slots:
            return ""
        if STAGE_DEFINITION in slots:
            slots = {**slots, STAGE_DEFINITION: "database stages"}
        lines = [
            f"- {k}: {SqlGenerator._render_value(v)}"
            for k, v in slots.items()
            if v not in (None, "", [])
        ]
        if not lines:
            return ""
        return "RESOLVED FILTERS (apply all of them)\n" + "\n".join(lines) + "\n"

    def _user_prompt(self, question: str, slots: dict[str, Any] | None) -> str:
        return prompts.SQL_USER.format(
            schema_card=self.schema_card,
            question=question,
            slot_block=self.slot_block(slots),
        )

    def generate(
        self,
        question: str,
        slots: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> str:
        """First-pass SQL generation."""
        return self.clean(
            self.client.chat(
                prompts.SQL_SYSTEM,
                self._user_prompt(question, slots),
                trace=trace,
                stage="sql_generate",
            )
        )

    def repair(
        self,
        question: str,
        failed_sql: str,
        error: str,
        slots: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> str:
        """Single repair attempt driven by the verbatim SQLite error message."""
        user = (
            self._user_prompt(question, slots)
            + "\n\n"
            + prompts.SQL_REPAIR_USER.format(sql=failed_sql, error=error)
        )
        return self.clean(
            self.client.chat(prompts.SQL_SYSTEM, user, trace=trace, stage="sql_repair")
        )


def generate_sql(
    question: str, slots: dict[str, Any] | None = None, *, client: LLMClient | None = None
) -> str:
    """Module-level convenience wrapper."""
    return SqlGenerator(client).generate(question, slots)


def repair_sql(
    question: str,
    failed_sql: str,
    error: str,
    slots: dict[str, Any] | None = None,
    *,
    client: LLMClient | None = None,
) -> str:
    """Module-level convenience wrapper."""
    return SqlGenerator(client).repair(question, failed_sql, error, slots)
