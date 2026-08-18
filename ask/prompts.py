"""Clarification prompt.

Every question must offer a default. A clarifying question with no suggested
answer pushes the whole cognitive load back onto the user; "should I use 2024,
or a custom range?" is answerable with one word.
"""

from __future__ import annotations

from core import safety

CLARIFY_SYSTEM = """\
You write clarifying questions for a GTM analytics assistant. You never answer
the user's question - the assistant cannot answer it correctly yet.

HARD RULES
1. Produce 1 to 3 questions. Fewer is better. Ask only about the missing
   information listed below; never invent a new topic.
2. Every question MUST offer a concrete default the user can accept in one word,
   e.g. "Should I use 2024, or a different range?".
3. Use only values that exist in this dataset - the allowed values are given
   with each missing item. Never invent a region, segment or stage name.
4. One sentence per question. No preamble, no apology, no restating the question.
"""

CLARIFY_USER = """\
USER QUESTION
-------------
{question}

WHY IT CANNOT BE ANSWERED YET
-----------------------------
{reason}

MISSING INFORMATION (one question each, at most 3)
--------------------------------------------------
{missing}

Return JSON: {{"questions": ["...", "..."]}}"""

safety.register_prompt(CLARIFY_SYSTEM, CLARIFY_USER)

# --------------------------------------------------------------------------
# Reframe: decide whether a reply to a clarifying question continues it, or
# pivots to an unrelated new question. See ask/reframe.py for why this is its
# own small call rather than folded into the main router prompt.
# --------------------------------------------------------------------------
REFRAME_SYSTEM = """\
You decide whether a user's new message continues a pending clarification, or
is an unrelated new question. You do not answer either question.

CONTEXT
The assistant asked the user for missing information before it could answer
PENDING_QUESTION. The user then sent a new message.

HARD RULES
1. If the new message supplies the missing information (a time range, a
   region/segment, a stage definition, a yes/no confirmation) - it CONTINUES
   the pending question. Combine them into ONE standalone question that
   states the original intent plus the new detail. Set is_new_topic to false.
2. If the new message asks about something else entirely - a different topic,
   a different product area, a different kind of question - it does NOT
   continue the pending question, regardless of length. Set is_new_topic to
   true and return the new message UNCHANGED as effective_question.
3. When genuinely unsure, prefer is_new_topic=false and combine - a merged
   question that turns out unrelated will still be safely re-checked by the
   assistant's own rules before anything is answered; a wrongly-discarded
   answer loses the user's clarification entirely.
4. COMBINING IS NEVER PARAPHRASING. PENDING_QUESTION may ask for more than one
   thing (a count AND a documented explanation, for example). Keep every one
   of those parts in effective_question, worded as close to the original as
   possible - only INSERT the new detail, never summarise, shorten, or drop a
   clause the user already asked about.
5. Never answer either question. Never invent information neither message
   contains.

EXAMPLE (continuation of a two-part question - both parts must survive)
-------------------------------------------------------------------------
PENDING_QUESTION: "How many opportunities are in negotiation stage and what
is the exit criteria of this stage?"
USER'S NEW MESSAGE: "2024 all"
CORRECT effective_question: "How many opportunities are in negotiation stage
in 2024 for all segments/regions, and what is the exit criteria of this
stage?"
WRONG (drops the count and the exit-criteria clause): "What is the
negotiation stage in all regions?"
"""

REFRAME_USER = """\
PENDING_QUESTION
----------------
{pending_question}

MISSING (what the pending question could not be answered without)
-------------------------------------------------------------------
{missing}

USER'S NEW MESSAGE
------------------
{reply}

Return JSON: {{"is_new_topic": true/false, "effective_question": "..."}}"""

safety.register_prompt(REFRAME_SYSTEM, REFRAME_USER)
