"""Verbatim-number check - the anti-hallucination control for every numeric claim.

Role in architecture: runs on ANY model text that is supposed to be grounded in a
SQL result set (the hybrid composer AND the SQL narrator). Extracts every numeric
token from the generated text and asserts each one is traceable to the result set,
the user's question, or the executed SQL. A number that appears nowhere in those
sources was invented, and inventing a number is the single most damaging failure
mode for a GTM analytics assistant.

NORMALISATION POLICY (documented because it is a judgement call)
  * thousands separators ignored:      "1,200"  == 1200
  * currency symbols ignored:          "$1,200" == 1200
  * trailing % accepted two ways:      "12%"    matches a cell holding 12
                                                or a cell holding 0.12
  * rounding is allowed only DOWNWARD in precision: a cell holding 61234.5678
    licenses "61234.57", "61234.6", "61235", "61,234.57". The reverse (text more
    precise than the cell) fails.
  * numbers echoed from the question or the executed SQL are allowed ("last 90
    days", "top 5", "2024") - they are user-supplied context, not fabricated data.
  * ordinals that start a markdown list line ("1. ", "2) ") are ignored.
  * page numbers inside a citation bracket ("[Field Guide, p.3]") are ignored -
    they identify a source location, not a data claim.

In:  generated text + QueryResult (+ optional question/sql context, + optional
     retrieved doc-chunk text).
Out: `NumberCheck(ok, offending)`.

DOC-CHUNK GROUNDING
  A number is also allowed if it appears verbatim in `doc_texts` (the text of
  the retrieved RAG chunks, e.g. `[h.text for h in hits]`) - a real, correctly
  cited number that only lives in a doc (an SLA day count, a risk-rubric
  threshold) must not be flagged as fabricated just because it isn't in the
  SQL result. This checks the number against the POOL of all retrieved chunks,
  not the specific chunk the answer cites for that claim - the composer
  prompt already hands the model every hit's full text as legitimate source
  material (see `build_context()` in hybrid/composer.py), so this only
  extends the verifier's trust boundary to match what the prompt already
  grants; it does not let the model draw from anything new.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from sql.execute import QueryResult

# $1,234.56  |  1234.56%  |  -42  |  0.15
_NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?")
# "1. " / "2) " at the start of a line = list marker, not a claim.
_LIST_MARKER_RE = re.compile(r"^\s*[-*]?\s*\d+[.)]\s", re.MULTILINE)
# "[Field Guide, p.3]" - a citation built from Hit.citation(), not a numeric
# claim. Hit.citation() itself never puts a space after "p.", but the
# composer LLM sometimes paraphrases citations as "p. 4" - tolerate an
# optional space so that page number isn't treated as an ungrounded claim.
_CITATION_RE = re.compile(r"\[[^\[\]]*?\bp\.\s?\d+[^\[\]]*?\]")


@dataclass
class NumberCheck:
    ok: bool
    offending: list[str] = field(default_factory=list)


def _to_float(token: str) -> tuple[float | None, bool]:
    """Parse a token -> (value, was_percent)."""
    is_pct = token.endswith("%")
    cleaned = token.rstrip("%").replace(",", "").replace("$", "").strip()
    try:
        return float(cleaned), is_pct
    except ValueError:
        return None, is_pct


def _decimals(token: str) -> int:
    body = token.rstrip("%").replace(",", "").replace("$", "")
    return len(body.split(".")[1]) if "." in body else 0


def _allowed_values(
    result: QueryResult, extra_text: str, doc_texts: Sequence[str] = ()
) -> set[float]:
    """Every number the model is licensed to state."""
    values: set[float] = set()
    for cell in result.flat_values():
        if isinstance(cell, bool):
            continue
        if isinstance(cell, (int, float)):
            values.add(float(cell))
        elif isinstance(cell, str):
            # Numbers embedded in text cells (dates like '2024-03-01', ids) count:
            # the model may legitimately quote them.
            for tok in _NUMBER_RE.findall(cell):
                v, _ = _to_float(tok)
                if v is not None:
                    values.add(v)
    # Row count is a legitimate thing to state ("3 regions returned").
    values.add(float(result.row_count))
    for tok in _NUMBER_RE.findall(extra_text or ""):
        v, _ = _to_float(tok)
        if v is not None:
            values.add(v)
    # Numbers verbatim in a retrieved doc chunk are real, cited facts (an SLA
    # day count, a risk-rubric threshold) - not fabrications just because
    # they aren't in the SQL result. See module docstring for the pooled-
    # across-all-chunks tradeoff this accepts.
    for chunk in doc_texts:
        for tok in _NUMBER_RE.findall(chunk):
            v, _ = _to_float(tok)
            if v is not None:
                values.add(v)
    return values


def _matches(token: str, allowed: set[float]) -> bool:
    value, is_pct = _to_float(token)
    if value is None:
        return True  # unparseable -> not a numeric claim
    places = _decimals(token)
    candidates = [value]
    if is_pct:
        # "12%" may come from a cell holding 12 or a cell holding 0.12.
        candidates.append(value / 100.0)
    for cand in candidates:
        for source in allowed:
            if abs(source - cand) < 1e-9:
                return True
            # Downward-precision rounding: does the source round to the token?
            if round(source, places) == round(cand, places):
                return True
            if is_pct and abs(round(source * 100, places) - round(cand, places)) < 1e-9:
                return True
    return False


class NumberVerifier:
    """Checks that every numeric token in generated text is traceable.

    One instance is shared by the SQL narrator and the hybrid composer, so both
    paths enforce exactly the same policy.
    """

    def check(
        self,
        text: str,
        result: QueryResult,
        *,
        context: str = "",
        doc_texts: Sequence[str] = (),
    ) -> NumberCheck:
        """Assert every numeric token in `text` is grounded. Never raises.

        `doc_texts` (e.g. `[h.text for h in hits]`) is an additional, optional
        grounding source: numbers found there are treated as real cited facts,
        same as SQL result cells / row_count / `context`. Defaults to `()` so
        callers with no retrieved chunks (e.g. sql/narrate.py's SQL-only path)
        are unaffected.
        """
        stripped = _LIST_MARKER_RE.sub("\n", _CITATION_RE.sub("", text or ""))
        allowed = _allowed_values(result, context, doc_texts)
        offending = [
            tok for tok in _NUMBER_RE.findall(stripped) if not _matches(tok, allowed)
        ]
        # De-duplicate while preserving order, for a readable retry instruction.
        seen: set[str] = set()
        unique = [t for t in offending if not (t in seen or seen.add(t))]
        return NumberCheck(ok=not unique, offending=unique)


def check_numbers(
    text: str,
    result: QueryResult,
    *,
    context: str = "",
    doc_texts: Sequence[str] = (),
) -> NumberCheck:
    """Module-level convenience wrapper around `NumberVerifier`."""
    return NumberVerifier().check(text, result, context=context, doc_texts=doc_texts)
