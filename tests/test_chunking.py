"""Chunking invariants, tested against a small synthetic PDF fixture.

The real PDFs are NOT used here on purpose: a test that breaks when someone edits
the Enablement Pack is testing the document, not the chunker.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core import config
from rag.ingest import chunk_pdf, estimate_tokens, make_chunk_id, split_sentences

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic.pdf"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="fixture missing; regenerate with `python -m tests.fixtures.make_fixture`",
)


@pytest.fixture(scope="module")
def chunks():
    return chunk_pdf(FIXTURE, "Fixture")


def test_produces_chunks(chunks) -> None:
    assert len(chunks) >= 2


def test_table_is_split_into_multiple_chunks(chunks) -> None:
    """8 data rows with TABLE_ROWS_PER_CHUNK=6 must produce >1 table chunk."""
    tables = [c for c in chunks if c.kind == "table"]
    assert tables, "no table chunk was produced - the table finder failed"
    assert len(tables) >= 2


def test_every_table_chunk_keeps_its_header(chunks) -> None:
    """The invariant: a data row is never separated from its header row."""
    tables = [c for c in chunks if c.kind == "table"]
    for chunk in tables:
        for header_cell in ("Tier", "Price", "Seats"):
            assert header_cell in chunk.text, (
                f"table chunk {chunk.chunk_id} lost header cell '{header_cell}'"
            )


def test_second_table_group_has_header_and_later_rows(chunks) -> None:
    tables = [c for c in chunks if c.kind == "table"]
    later = [c for c in tables if "Trial" in c.text or "Academic" in c.text]
    assert later, "later table rows were dropped"
    assert all("Tier" in c.text for c in later)


def test_chunk_ids_are_stable_across_runs() -> None:
    first = [c.chunk_id for c in chunk_pdf(FIXTURE, "Fixture")]
    second = [c.chunk_id for c in chunk_pdf(FIXTURE, "Fixture")]
    assert first == second
    assert len(set(first)) == len(first), "chunk ids must be unique"


def test_chunk_id_is_a_pure_function_of_doc_page_offset() -> None:
    assert make_chunk_id("A", 1, 0) == make_chunk_id("A", 1, 0)
    assert make_chunk_id("A", 1, 0) != make_chunk_id("A", 2, 0)
    assert make_chunk_id("A", 1, 0) != make_chunk_id("B", 1, 0)
    assert len(make_chunk_id("A", 1, 0)) == 12


def test_section_chunks_carry_their_heading(chunks) -> None:
    sections = [c for c in chunks if c.kind == "section"]
    assert sections
    assert any("Synthetic Section One" in c.text for c in sections)


def test_section_chunks_respect_the_token_budget(chunks) -> None:
    # One sentence may overshoot the target; two full sentences past it means the
    # splitter is not splitting.
    for chunk in (c for c in chunks if c.kind == "section"):
        assert estimate_tokens(chunk.text) <= config.CHUNK_TARGET_TOKENS * 2


def test_sentences_are_never_split_mid_sentence(chunks) -> None:
    joined = " ".join(c.text for c in chunks if c.kind == "section")
    for sentence in (
        "Deal desk reviews every non standard discount before approval.",
        "Row level security is enforced before any artifact is retrieved.",
    ):
        assert sentence in joined, f"sentence was broken across chunks: {sentence}"


def test_split_sentences_is_pure() -> None:
    assert split_sentences("A. B. C.") == ["A.", "B.", "C."]
    assert split_sentences("") == []
