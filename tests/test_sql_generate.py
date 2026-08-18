"""SqlGenerator.slot_block - resolved-filter rendering for the SQL prompt.

Policy under test: `stage_definition` is always forced to "database stages"
regardless of what the router extracted, because the Field Guide's playbook
stage names don't exist in `opportunities.stage` (router/slots.py::required_slots
no longer asks the user for this - it's determined by route, not user input).
"""

from __future__ import annotations

from sql.generate import SqlGenerator


def test_stage_definition_is_forced_to_database_stages() -> None:
    block = SqlGenerator.slot_block({"stage_definition": "playbook"})
    assert "stage_definition: database stages" in block
    assert "playbook" not in block


def test_stage_definition_forced_even_when_absent_becomes_present_only_if_key_exists() -> None:
    """No stage_definition key at all -> nothing invented."""
    block = SqlGenerator.slot_block({"time_range": "2024"})
    assert "stage_definition" not in block


def test_other_slots_render_unchanged() -> None:
    block = SqlGenerator.slot_block(
        {"time_range": "2024", "stage_definition": "database stages"}
    )
    assert "time_range: 2024" in block
    assert "stage_definition: database stages" in block
