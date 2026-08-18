"""Prompts for the two LLM-judge metrics (faithfulness, answer relevancy).

All run only during an eval pass (scripts/eval.py), never on a live user
turn - the extra latency is paid once per eval run, the same trade-off
scripts/demo.py already makes for its latency measurements.
"""

from __future__ import annotations

from core import safety

# --------------------------------------------------------------------------
# Faithfulness: does every claim in the answer trace back to the context?
# --------------------------------------------------------------------------
FAITHFULNESS_SYSTEM = """\
You are a strict fact-checker. You are given a QUESTION, an ANSWER someone
gave to it, and the CONTEXT chunks the answer was supposed to be grounded in.

TASK
List every distinct factual claim the ANSWER makes, and for each one decide
whether it is directly supported by the CONTEXT - not by outside knowledge,
not by what sounds plausible.

HARD RULES
1. Split the answer into its smallest independent factual claims. A sentence
   with two facts is two claims.
2. A claim is "supported" only if the CONTEXT states it or something that
   directly entails it. If the CONTEXT is silent on a claim, it is
   unsupported, even if the claim sounds reasonable.
3. Ignore claims that are pure meta-commentary ("this is not covered in the
   context", citations, hedges) - list only substantive factual claims.
4. If the answer makes no substantive factual claims at all (e.g. a
   clarifying question, a refusal), return an empty claims list.
"""

FAITHFULNESS_USER = """\
QUESTION
--------
{question}

CONTEXT
-------
{context}

ANSWER
------
{answer}

Return JSON: {{"claims": [{{"text": "...", "supported": true/false}}, ...]}}"""

# --------------------------------------------------------------------------
# Answer relevancy: does the answer actually address the question asked?
# Measured indirectly - generate questions the answer would suit, and compare
# them to the real question via embedding similarity (see eval/metrics.py).
# --------------------------------------------------------------------------
RELEVANCY_SYSTEM = """\
You generate questions that a given ANSWER would be a good, direct response
to. You do not judge or answer anything yourself.

HARD RULES
1. Produce exactly 3 questions.
2. Each question must be answerable using ONLY the information in ANSWER -
   never bring in outside knowledge.
3. Vary the phrasing across the 3 questions (do not just repeat the same
   question three times), but keep them all pointed at the same core content.
4. If ANSWER contains no substantive content to ask about (e.g. it is a
   refusal or a clarifying question), generate 3 short questions that reflect
   that (e.g. "What does the assistant say it needs?").
"""

RELEVANCY_USER = """\
ANSWER
------
{answer}

Return JSON: {{"questions": ["...", "...", "..."]}}"""

safety.register_prompt(
    FAITHFULNESS_SYSTEM,
    FAITHFULNESS_USER,
    RELEVANCY_SYSTEM,
    RELEVANCY_USER,
)
