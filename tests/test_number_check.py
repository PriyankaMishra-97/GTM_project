"""Verbatim-number check.

Policy under test (also documented in hybrid/verify.py):
  * "1,200" == 1200, "$1,200" == 1200
  * "12%" matches a cell holding 12 OR a cell holding 0.12
  * rounding is allowed downward in precision only (a cell of 61234.5678
    licenses "61234.57"; a cell of 61234 does NOT license "61234.57")
  * numbers echoed from the question/SQL are allowed
"""

from __future__ import annotations

from hybrid.verify import check_numbers
from sql.execute import QueryResult


def _result() -> QueryResult:
    return QueryResult(
        columns=["region", "deals", "pipeline_usd", "win_rate"],
        rows=[
            ["EMEA", 128, 9400000.0, 0.42],
            ["NA", 1200, 61234.5678, 0.1234],
        ],
    )


def test_verbatim_numbers_pass() -> None:
    assert check_numbers("EMEA closed 128 deals worth 9400000.", _result()).ok


def test_fabricated_number_fails() -> None:
    verdict = check_numbers("EMEA closed 999 deals.", _result())
    assert not verdict.ok
    assert "999" in verdict.offending


def test_computed_total_is_rejected() -> None:
    """1200 + 128 = 1328 is arithmetic the model is forbidden to do."""
    verdict = check_numbers("Together that is 1328 deals.", _result())
    assert not verdict.ok
    assert "1328" in verdict.offending


def test_thousands_separator_normalised() -> None:
    assert check_numbers("NA had 1,200 deals.", _result()).ok


def test_currency_symbol_normalised() -> None:
    assert check_numbers("Pipeline was $9400000.", _result()).ok


def test_percent_matches_fraction_cell() -> None:
    """0.42 in the cell licenses '42%' - documented percent policy."""
    assert check_numbers("Win rate was 42%.", _result()).ok


def test_percent_matches_percent_cell() -> None:
    result = QueryResult(columns=["pct"], rows=[[42]])
    assert check_numbers("Win rate was 42%.", result).ok


def test_downward_rounding_allowed() -> None:
    assert check_numbers("Pipeline was 61234.57.", _result()).ok
    assert check_numbers("Pipeline was 61235.", _result()).ok


def test_upward_precision_is_rejected() -> None:
    """A cell holding an integer does not license invented decimal places."""
    result = QueryResult(columns=["deals"], rows=[[128]])
    verdict = check_numbers("There were 128.47 deals.", result)
    assert not verdict.ok


def test_numbers_echoed_from_the_question_are_allowed() -> None:
    verdict = check_numbers(
        "In 2024 EMEA closed 128 deals.", _result(), context="deals in 2024"
    )
    assert verdict.ok


def test_row_count_is_allowed() -> None:
    assert check_numbers("2 regions were returned.", _result()).ok


def test_list_markers_are_not_claims() -> None:
    text = "1. EMEA had 128 deals.\n2. NA had 1200 deals."
    assert check_numbers(text, _result()).ok


def test_dates_in_text_cells_are_allowed() -> None:
    result = QueryResult(columns=["close_date"], rows=[["2024-03-01"]])
    assert check_numbers("The deal closed on 2024-03-01.", result).ok


def test_empty_text_passes() -> None:
    assert check_numbers("", _result()).ok


def test_citation_page_number_is_not_a_claim() -> None:
    """[Field Guide, p.3] cites a page; "3" is not a data claim (hybrid/verify.py)."""
    verdict = check_numbers(
        "EMEA closed 128 deals [Field Guide, p.3].", _result()
    )
    assert verdict.ok


def test_citation_page_number_with_space_is_not_a_claim() -> None:
    """The composer LLM sometimes paraphrases citations as "p. 4" (space after
    the period) instead of Hit.citation()'s "p.4" - the page number must still
    not be treated as an ungrounded numeric claim."""
    verdict = check_numbers(
        "EMEA closed 128 deals [Field Guide, p. 4].", _result()
    )
    assert verdict.ok


def test_fabricated_number_still_caught_next_to_a_citation() -> None:
    verdict = check_numbers(
        "EMEA closed 999 deals [Field Guide, p.3].", _result()
    )
    assert not verdict.ok
    assert "999" in verdict.offending
    assert "3" not in verdict.offending


def test_number_grounded_only_in_doc_chunk() -> None:
    """A real, cited number that lives only in a doc chunk is not fabricated."""
    verdict = check_numbers(
        "The resolution SLA is 14 days.",
        _result(),
        doc_texts=["Stage progression playbook: the resolution SLA is 14 days."],
    )
    assert verdict.ok


def test_number_not_in_result_or_doc_chunks_still_fails() -> None:
    """doc_texts must not make the check permissive by default."""
    verdict = check_numbers(
        "The resolution SLA is 14 days.",
        _result(),
        doc_texts=["Stage progression playbook has no SLA figures here."],
    )
    assert not verdict.ok
    assert "14" in verdict.offending


def test_number_across_multiple_doc_chunks() -> None:
    """Every chunk in doc_texts is scanned, not just the first."""
    verdict = check_numbers(
        "The rubric totals 42 points.",
        _result(),
        doc_texts=["Executive summary, no numbers.", "Risk rubric totals 42 points.", "FAQ."],
    )
    assert verdict.ok


def test_doc_texts_default_omitted_no_behavior_change() -> None:
    """Omitting doc_texts entirely preserves today's behavior exactly."""
    verdict = check_numbers("EMEA closed 999 deals.", _result())
    assert not verdict.ok
    assert "999" in verdict.offending


def test_doc_texts_normalises_currency_and_separators() -> None:
    """doc_texts reuses the same normalisation as every other grounding source."""
    verdict = check_numbers(
        "The deal was worth 12500.",
        _result(),
        doc_texts=["Deal value: $12,500"],
    )
    assert verdict.ok
