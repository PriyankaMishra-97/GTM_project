"""History reframe: resolve a follow-up question using the last 1-2 answered
turns, when it is NOT following an ASK.

Role in architecture: mutually exclusive with ask/reframe.py's Reframer.
That one runs ONLY when the turn follows an ASK (`pending_clarification` is
set) and merges a reply against ONE pending question. This one runs when
`pending_clarification` is NOT set, against the last 1-2 turns that produced
a real SQL/RAG/HYBRID answer (ASK/REFUSE turns carry no factual content
worth referencing). orchestrator.py's `if pending_clarification: ... elif
recent_turns: ...` is the mutual-exclusivity guarantee - only one of the two
reframers can ever run on a given turn.

Closes a documented gap (README's "Known limitations": "Conversation history
is also not fed to the router, so follow-ups like 'and for EMEA?' are
treated as fresh questions and will usually route to ASK").

Latency: `looks_like_followup()` gates the LLM call so a self-contained new
question costs nothing extra. Validated against 9 real questions before
implementation - pure word-count was too loose (flagged "What deployment
modes does Product XYZ support?" and "How many deals closed in 2024?" as
follow-ups despite being complete on their own), so this also checks for an
explicit entity (time/region/segment/stage/product), reusing
router/slots.py's existing has_explicit_time/has_explicit_population rather
than reinventing them.

Guards: dry-run against the real router model surfaced two failure modes
that make ALL THREE guards required, not optional hardening:
  1. Self-contradiction: the model reliably returns is_new_topic=True even
     alongside a correct, non-verbatim rewrite - same as ask/reframe.py.
  2. Fabrication: when the recent turns don't actually contain the entity a
     short follow-up implies, the model confidently INVENTS one instead of
     admitting it can't resolve the reference (real case: history about EMEA
     opportunity counts, message "Why is that stage risky?" -> model
     invented "the Solution Fit stage" from its own prompt's worked example;
     swapping the example for a fictional placeholder just produced a
     different fabricated name). ask/text_overlap.no_fabricated_entities
     catches this regardless of what gets invented.

Failure policy: on LLM failure or any guard rejection, fail OPEN to no
rewrite (return the question unchanged) - never blind concatenation. Unlike
ask/reframe.py's single short pending question, concatenating two full past
Q&A pairs onto a new question is noisy garbage; leaving the question as
typed can only match today's status quo (worst case, an ASK), never make it
worse.

In:  up to 2 prior (question, answer) turns + the new message.
Out: `HistoryReframeDecision` (pydantic) - is_new_topic + effective_question.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from ask import prompts
from ask.text_overlap import content_words, no_fabricated_entities
from core.llm_client import LLMClient, LLMJSONError, LLMUnavailable, get_client
from core.trace import Trace
from router.slots import has_explicit_population, has_explicit_time

_CONNECTORS = ("and ", "what about", "same for", "same but", "also", "how about", "what if", "compared to")
_REFERENCE_WORDS = ("it", "that", "those", "this")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
# A "Stage N -" prefix directly before a stage name (e.g. "Stage 1 - Qualify")
# must be dropped, not kept, when substituting a different stage in: the
# number belongs to the OLD stage, so "Stage 1 - Qualify" -> "Discover" must
# not become "Stage 1 - Discover" (stage 1 is Qualify, not Discover) - that
# stale number caused retrieval/the answer model to anchor on the wrong
# stage's content entirely, not just look odd cosmetically.
_STAGE_NUMBER_PREFIX_RE = re.compile(r"stage\s*\d+\s*[—\-:]\s*$", re.IGNORECASE)


def _substitute_stage(template: str, old_text: str, new_text: str) -> str:
    """Replace `old_text` in `template` with `new_text`, also dropping an
    immediately preceding "Stage N -" prefix - it names the OLD stage's
    number, so keeping it would attach the wrong number to the new stage."""
    idx = template.find(old_text)
    if idx == -1:
        return template.replace(old_text, new_text)
    prefix_match = _STAGE_NUMBER_PREFIX_RE.search(template[:idx])
    start = prefix_match.start() if prefix_match else idx
    return template[:start] + new_text + template[idx + len(old_text):]

# Proper nouns that make a question self-contained even when short. Database
# stage enum + Field Guide playbook names + product/doc names.
_STAGE_NAMES = (
    "prospecting", "discovery", "evaluation", "proposal", "negotiation",
    "closed won", "closed lost", "qualify", "discover", "solution fit",
    "commercial align", "commit", "handoff",
)
_PRODUCT_NAMES = (
    "product xyz", "opportunity tracker", "xyz-core", "xyz-analytics",
    "xyz-automation", "xyz-security",
)


def _has_explicit_entity(question: str) -> bool:
    """Does the question already name something concrete enough to stand alone?"""
    q = question.lower()
    return (
        has_explicit_time(question)
        or has_explicit_population(question)
        or any(s in q for s in _STAGE_NAMES)
        or any(p in q for p in _PRODUCT_NAMES)
    )


def _find_stage_span(text: str) -> tuple[str, str] | None:
    """(canonical stage key, as-typed substring) for the first known stage
    name mentioned in text, or None. Order matters: "discovery" is checked
    before "discover" so a text containing "discovery" doesn't get
    mis-detected as the shorter "discover" (a substring of it). The as-typed
    substring preserves the user's original casing, for substitution."""
    t = text.lower()
    for name in _STAGE_NAMES:
        idx = t.find(name)
        if idx != -1:
            return name, text[idx : idx + len(name)]
    return None


def _mentioned_stage(text: str) -> str | None:
    """First known stage name mentioned in text, if any."""
    match = _find_stage_span(text)
    return match[0] if match else None


def _strip_leading_connector(text: str) -> str:
    t = text.strip()
    tl = t.lower()
    for c in _CONNECTORS:
        if tl.startswith(c):
            return t[len(c):].strip()
    return t


def _bare_stage_substitution(recent_turns: list[dict], question: str) -> str | None:
    """Deterministic shortcut for "what about X" / "and in X" / bare "X"
    follow-ups that name a DIFFERENT specific stage than a recent turn.

    Skips the LLM call entirely - the real reason this exists, not just a
    latency win: dry-run showed the model reliably APPENDS the new stage
    name instead of REPLACING the old one ("What are the exit criteria for
    Stage 1 - Qualify and in Discover?"), even with an explicit prompt
    example telling it not to. Guard 4 catches that failure after the fact;
    this avoids it up front for the common case where the reply is nothing
    but the new stage's name - a pure substitution, no LLM judgment needed.

    Only applies when the reply is a BARE reference (no content beyond the
    stage name itself, once a leading connector phrase like "and "/"what
    about" is stripped) - anything more ("what about commit's SLA") still
    needs the LLM to blend the new detail in properly.
    """
    new_match = _find_stage_span(question)
    if new_match is None:
        return None
    new_key, new_text = new_match

    remainder = _strip_leading_connector(question)
    extra = content_words(remainder) - set(new_key.split())
    if extra:
        return None  # the reply says more than just the stage name

    # Most recent turn whose OWN question names a different stage AND is
    # itself a full question (not just another bare reference) - the right
    # template to substitute into.
    for turn in reversed(recent_turns):
        old_match = _find_stage_span(turn["question"])
        if old_match is None or old_match[0] == new_key:
            continue
        old_key, old_text = old_match
        template_remainder = _strip_leading_connector(turn["question"])
        template_extra = content_words(template_remainder) - set(old_key.split())
        if not template_extra:
            continue  # that turn was itself a bare reference - not a template
        return _substitute_stage(turn["question"], old_text, new_text)
    return None


def looks_like_followup(question: str) -> bool:
    """Cheap gate: does this look like it depends on recent context?

    Validated against 9 real questions during planning, including the exact
    case this feature exists for ("What is the SLA days?" after a Solution
    Fit question) and the false-positive case a naive word-count check
    would have flagged ("What deployment modes does Product XYZ support?").
    """
    q = question.strip().lower().rstrip("?")
    if any(q.startswith(c) for c in _CONNECTORS):
        return True
    tokens = set(_TOKEN_RE.findall(q))
    if tokens & set(_REFERENCE_WORDS):
        return True
    words = q.split()
    if len(words) <= 8 and not _has_explicit_entity(question):
        return True
    return False


class HistoryReframeDecision(BaseModel):
    is_new_topic: bool
    effective_question: str


def _render_history(turns: list[dict]) -> str:
    return "\n\n".join(f"Q: {t['question']}\nA: {t['answer']}" for t in turns)


class HistoryReframer:
    """Resolves a follow-up against the last 1-2 answered turns, with three guards."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or get_client()

    def reframe(
        self, recent_turns: list[dict], question: str, trace: Trace
    ) -> HistoryReframeDecision:
        no_rewrite = HistoryReframeDecision(is_new_topic=True, effective_question=question)

        # Fast path: no history, or the question doesn't look like a
        # follow-up at all - skip the LLM call entirely (latency gate).
        if not recent_turns or not looks_like_followup(question):
            return no_rewrite

        # Deterministic shortcut for bare stage-name substitution - see
        # _bare_stage_substitution's docstring for why this exists instead of
        # trusting the LLM for this specific pattern.
        substituted = _bare_stage_substitution(recent_turns, question)
        if substituted is not None:
            return HistoryReframeDecision(is_new_topic=False, effective_question=substituted)

        try:
            with trace.stage("ask_history_reframe"):
                result = self.client.chat_json(
                    prompts.HISTORY_REFRAME_SYSTEM,
                    prompts.HISTORY_REFRAME_USER.format(
                        history=_render_history(recent_turns), question=question
                    ),
                    HistoryReframeDecision,
                    model=self.client.router_model,
                    trace=trace,
                    stage="ask_history_reframe",
                )
        except (LLMJSONError, LLMUnavailable) as exc:
            trace.error(f"history_reframe fallback: {exc}")
            return no_rewrite

        # Guard 1 - self-contradiction: is_new_topic=true alongside a
        # rewritten (non-verbatim) effective_question means the model
        # demonstrably found a connection - trust the rewrite, not the flag.
        candidate = result
        if result.is_new_topic and result.effective_question.strip() != question.strip():
            trace.error(
                "history_reframe: is_new_topic=true but effective_question was "
                "rewritten - treating as a continuation instead"
            )
            candidate = HistoryReframeDecision(
                is_new_topic=False, effective_question=result.effective_question
            )

        if candidate.is_new_topic:
            return candidate  # genuine pivot: nothing to guard, question is unchanged

        source_words = content_words(_render_history(recent_turns))
        reply_words = content_words(question)
        merged_words = content_words(candidate.effective_question)

        # Guard 2 - the reply's own new detail must actually be incorporated,
        # not silently dropped in favor of just re-asking the prior question.
        # Unlike ask/reframe.py's pending_question (the literal target the
        # user needs answered), recent_turns includes a full prior ANSWER
        # too - requiring the merge to overlap most of THAT text as well is
        # too strict and rejects correct rewrites (dry-run: a correct merge
        # naming "Solution Fit" failed a >=60%-of-source-words check because
        # the prior turn's ANSWER contributed 16 more content words about
        # exit criteria that a follow-up about SLA days has no reason to
        # restate). Guard 3 below is the real check for whether the entity a
        # short follow-up references is actually grounded in recent turns.
        if reply_words and not (reply_words & merged_words):
            trace.error(
                "history_reframe: merged question dropped the new message's "
                "own content - leaving the question unchanged instead"
            )
            return no_rewrite

        # Guard 3 - fabrication: must not introduce content absent from BOTH
        # the recent turns and the reply (real case: an invented stage name).
        if not no_fabricated_entities(source_words, reply_words, merged_words):
            trace.error(
                "history_reframe: merged question introduced content not present "
                "in recent turns or the reply - leaving the question unchanged instead"
            )
            return no_rewrite

        # Guard 4 - stage substitution: if the reply itself names a specific
        # stage different from one already mentioned in recent_turns, the
        # merge must actually SUBSTITUTE it, not retain the old stage
        # alongside the new one. Dry-run: even with an explicit prompt
        # example showing this exact wrong pattern, the model produced "What
        # are the exit criteria for Stage 1 - Qualify and in Discover?" -
        # keeping "Qualify" instead of replacing it with "Discover". Checked
        # against ALL recent turns, not just the most recent one: with 2
        # turns in context (Qualify, then Commit), a stale merge retained
        # the OLDER turn's stage ("Qualify"), not the most recent ("Commit") -
        # the model's fabrication isn't reliably tied to the last turn only.
        # Guards 2/3 don't catch this (nothing was dropped or fabricated -
        # the stale stage name is legitimately present in a source turn), so
        # this needs its own check.
        new_stage = _mentioned_stage(question)
        if new_stage:
            old_stages = {
                s for t in recent_turns if (s := _mentioned_stage(t["question"])) is not None
            }
            merged_lower = candidate.effective_question.lower()
            stale = {s for s in old_stages if s != new_stage and s in merged_lower}
            if stale:
                trace.error(
                    "history_reframe: merged question kept an old stage name "
                    f"({sorted(stale)}) instead of substituting the new one - "
                    "leaving the question unchanged instead"
                )
                return no_rewrite

        return candidate


def reframe(
    recent_turns: list[dict],
    question: str,
    trace: Trace,
    *,
    client: LLMClient | None = None,
) -> HistoryReframeDecision:
    """Module-level convenience wrapper around `HistoryReframer`."""
    return HistoryReframer(client).reframe(recent_turns, question, trace)
