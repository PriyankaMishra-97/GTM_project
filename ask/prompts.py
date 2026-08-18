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

# --------------------------------------------------------------------------
# History reframe: decide whether a NEW message (not following an ASK) is a
# self-contained question, or an elliptical follow-up to the last 1-2
# answered turns. See ask/history_reframe.py for why this is a separate,
# mutually-exclusive path from REFRAME_SYSTEM above (which only ever runs
# right after an ASK).
# --------------------------------------------------------------------------
HISTORY_REFRAME_SYSTEM = """\
You decide whether a user's new message is a self-contained new question, or
an elliptical follow-up that depends on the last 1-2 exchanges to be
understood. You do not answer either question.

HARD RULES
1. If the new message omits something (a stage, region, segment, product,
   time range, or metric) that was explicit in a recent exchange, it CONTINUES
   that exchange. Combine them into ONE standalone question that keeps the
   most recent relevant exchange's full intent AND injects the new message's
   detail. Never drop a clause the prior question already specified.
2. If the new message is JUST a bare name ("what about X", "and in X", "X?")
   naming a DIFFERENT specific stage/region/product than the recent exchange
   used, it means "ask the SAME question, but about X instead." Substitute X
   for the old item in the most recent relevant question's structure - do
   NOT treat X as a vague new topic to define generically, and do NOT keep
   the OLD item's name anywhere in the result.
3. If the new message is already a complete question on its own, or asks
   about something unrelated to the recent exchanges, it is NOT a follow-up.
   Set is_new_topic to true and return the new message UNCHANGED.
4. When genuinely unsure whether the recent exchanges are relevant - in
   particular, if none of them actually mention the thing the new message
   seems to reference - prefer is_new_topic=true and leave the question as
   typed. NEVER invent a stage, region, segment, product, or other specific
   value that does not appear in the recent exchanges or the new message
   itself, even if something similar would make a plausible-sounding
   question. A wrong invented value is worse than asking the user to
   rephrase.
5. Never answer either question.

EXAMPLE 1 - missing metric, same item (illustrative - names are placeholders)
------------------------------------------------------------------------------
RECENT EXCHANGES:
Q: "What is the exit criteria for the Zeta stage?"
A: "The exit criteria for the Zeta stage are: widget validated; demo aligns
to plan. [Placeholder Doc, p.9]"
NEW MESSAGE: "What is the SLA days?"
CORRECT effective_question: "What is the SLA days for the Zeta stage?"

EXAMPLE 2 - bare name, different item, SAME question structure (rule 2)
--------------------------------------------------------------------------
RECENT EXCHANGES:
Q: "What is the exit criteria for the Zeta stage?"
A: "The exit criteria for the Zeta stage are: widget validated; demo aligns
to plan. [Placeholder Doc, p.9]"
NEW MESSAGE: "what about Omega"
CORRECT effective_question: "What is the exit criteria for the Omega stage?"
WRONG (treats the name as a vague new topic instead of substituting it):
"What is the meaning of 'Omega' in this context?"
WRONG (keeps the old item instead of substituting): "What is the exit
criteria for the Zeta stage, and what about Omega?"
"""

HISTORY_REFRAME_USER = """\
RECENT EXCHANGES (oldest first, at most 2)
-------------------------------------------
{history}

NEW MESSAGE
-----------
{question}

Return JSON: {{"is_new_topic": true/false, "effective_question": "..."}}"""

safety.register_prompt(HISTORY_REFRAME_SYSTEM, HISTORY_REFRAME_USER)
