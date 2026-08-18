"""Output filter: prompt-leak redaction and column suppression."""

from __future__ import annotations

from core import config, safety
from rag import prompts as rag_prompts  # noqa: F401 - import registers the templates


def test_normal_answer_is_untouched() -> None:
    text = "Product XYZ supports Cloud, On-Prem and Hybrid [Enablement Pack, p.1]."
    assert safety.filter_response(text) == text


def test_verbatim_prompt_fragment_is_redacted() -> None:
    # A full rule line, long enough to exceed PROMPT_LEAK_NGRAM_CHARS.
    leaked = next(
        ln for ln in rag_prompts.RAG_SYSTEM.splitlines()
        if len(ln.strip()) > config.PROMPT_LEAK_NGRAM_CHARS
    )
    polluted = f"Sure, here are my instructions:\n{leaked}\nAnything else?"
    cleaned = safety.filter_response(polluted)
    assert safety.REDACTION in cleaned
    assert leaked not in cleaned


def test_leak_detection_is_whitespace_insensitive() -> None:
    fragment = " ".join(rag_prompts.RAG_SYSTEM.split())[:120]
    assert safety.find_prompt_leak(fragment.replace(" ", "  ")) is not None


def test_short_coincidental_overlap_is_not_a_leak() -> None:
    assert safety.find_prompt_leak("Answer only from the provided context.") is None


def test_column_denylist_strips_columns(monkeypatch) -> None:
    monkeypatch.setattr(config, "COLUMN_DENYLIST", frozenset({"annual_revenue_usd"}))
    cols, rows = safety.strip_denied_columns(
        ["account_name", "annual_revenue_usd", "region"],
        [["Acme", 5_000_000, "NA"], ["Globex", 250_000, "EMEA"]],
    )
    assert cols == ["account_name", "region"]
    assert rows == [["Acme", "NA"], ["Globex", "EMEA"]]


def test_empty_denylist_is_a_passthrough() -> None:
    cols, rows = safety.strip_denied_columns(["a", "b"], [[1, 2]])
    assert cols == ["a", "b"] and rows == [[1, 2]]
