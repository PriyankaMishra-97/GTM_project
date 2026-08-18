"""Shared content-overlap guards for the reframe paths (ask/reframe.py,
ask/history_reframe.py).

Role in architecture: both reframers ask a small model to rewrite a question
by merging in prior context. A rewrite can fail in two directions - it can
DROP content the prior context/reply actually specified (lossy merge), or it
can INTRODUCE content that appears nowhere in the source material (a
fabricated entity). Both checks are pure word-overlap functions so they run
without another LLM call.

`_content_words` threshold is `len(w) >= 3`, not `> 3`: a follow-up like
"What is the SLA days?" needs "SLA" recognised as real content, and a `> 3`
threshold silently drops it (real case that motivated this fix - see
ask/history_reframe.py's module docstring).
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "the", "this", "that", "what", "which", "does", "should", "have",
        "with", "many", "about", "into", "from", "were", "will", "would",
        "could", "when", "where",
    }
)


def content_words(text: str) -> set[str]:
    """Words a merge must not silently drop or introduce unsourced."""
    return {
        w for w in _TOKEN_RE.findall(text.lower())
        if len(w) >= 3 and w not in _STOPWORDS
    }


def preserves_content(
    source_words: set[str], reply_words: set[str], merged_words: set[str]
) -> bool:
    """Lossy-merge guard: the merge must keep most of the source's content
    AND actually incorporate the reply's own content."""
    source_ok = not source_words or len(source_words & merged_words) / len(source_words) >= 0.6
    reply_ok = not reply_words or bool(reply_words & merged_words)
    return source_ok and reply_ok


def no_fabricated_entities(source_words: set[str], reply_words: set[str], merged_words: set[str]) -> bool:
    """Fabrication guard: the merge must not introduce a content word absent
    from BOTH the source context and the reply.

    Real case: history about "EMEA opportunities Closed Won", reply "Why is
    that stage risky?" (no stage anywhere in that history) - a small model
    confidently invented an entity from its own prompt's worked example
    ("Why is the Solution Fit stage risky?") rather than admitting it
    couldn't resolve the reference. Swapping the example for a fictional
    placeholder didn't stop the fabrication, it just invented a different
    fake entity - the model reliably fabricates *something* plausible when
    under-context, so this must be caught structurally, not prompted away.
    """
    fabricated = merged_words - source_words - reply_words
    return not fabricated
