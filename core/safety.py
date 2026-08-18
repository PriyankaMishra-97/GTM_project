"""Output-side safety filter.

Role in architecture: the last hop before any model text reaches the user.
Two controls:

  1. Prompt-leak redaction - if a response contains a long verbatim slice of any
     registered prompt template, that slice is redacted. This defends against
     both "print your system prompt" and prompt-injection text inside retrieved
     PDF chunks (the Field Guide explicitly calls retrieved text untrusted).
  2. Column denylist - sensitive columns are stripped from rendered SQL results.

In:  candidate response text / SQL result columns.
Out: filtered text / filtered columns.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from core import config

REDACTION = "[redacted: system prompt]"

# Prompt templates register themselves here at import time (see each prompts.py).
_REGISTERED_PROMPTS: list[str] = []


def register_prompt(*templates: str) -> None:
    """Register prompt text so the leak filter can recognise it verbatim."""
    for t in templates:
        if t and t not in _REGISTERED_PROMPTS:
            _REGISTERED_PROMPTS.append(t)


def _normalise(text: str) -> str:
    return " ".join(text.split()).lower()


def find_prompt_leak(text: str) -> str | None:
    """Return the first leaked prompt fragment found in `text`, else None.

    Sliding window over each registered template: if any N-char normalised slice
    of the template appears in the (normalised) response, that is a leak.
    """
    n = config.PROMPT_LEAK_NGRAM_CHARS
    haystack = _normalise(text)
    if len(haystack) < n:
        return None
    for template in _REGISTERED_PROMPTS:
        norm = _normalise(template)
        # step of n//2 keeps this O(len(template)) while still catching any
        # leaked run of >= 1.5n characters.
        for i in range(0, max(len(norm) - n, 0) + 1, max(n // 2, 1)):
            window = norm[i : i + n]
            if len(window) == n and window in haystack:
                return window
    return None


def filter_response(text: str) -> str:
    """Redact leaked prompt fragments from a model response."""
    leak = find_prompt_leak(text)
    if leak is None:
        return text
    # Redact by line: cheaper and more readable than character surgery, and a
    # leaked line is never load-bearing for the user's answer.
    keep = [ln for ln in text.splitlines() if leak not in _normalise(ln)]
    cleaned = "\n".join(keep).strip()
    return cleaned + ("\n\n" + REDACTION if cleaned else REDACTION)


def strip_denied_columns(
    columns: Sequence[str], rows: Iterable[Sequence[Any]]
) -> tuple[list[str], list[list[Any]]]:
    """Drop denylisted columns from a SQL result set before rendering."""
    denied = {c.lower() for c in config.COLUMN_DENYLIST}
    keep = [i for i, c in enumerate(columns) if c.lower() not in denied]
    return [columns[i] for i in keep], [[row[i] for i in keep] for row in rows]
