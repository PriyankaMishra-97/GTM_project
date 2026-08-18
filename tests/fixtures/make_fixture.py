"""Regenerate tests/fixtures/synthetic.pdf.

Run once; the resulting PDF is committed so the test suite does not depend on
PDF rendering being byte-identical across PyMuPDF versions.

    python -m tests.fixtures.make_fixture

The fixture deliberately mirrors the real PDFs' structure: a heading, prose long
enough to force a multi-chunk split, and a ruled table with more data rows than
config.TABLE_ROWS_PER_CHUNK so header repetition is exercised.
"""

from __future__ import annotations

from pathlib import Path

import fitz

OUT = Path(__file__).parent / "synthetic.pdf"

HEADING = "Synthetic Section One"
PROSE = (
    "The tracker normalises stage values before rollup. "
    "Each opportunity carries a close confidence score between zero and one hundred. "
    "Deal desk reviews every non standard discount before approval. "
    "Solution engineers validate the proposed architecture during solution fit. "
    "Delivery readiness is scored again after every security review. "
    "Forecast calls use the same definitions as the warehouse semantic model. "
    "Handoff to customer success requires an accepted deployment plan. "
    "Blockers are logged against the mutual action plan rather than the stage. "
    "Risk sub scores are stored so the total can always be explained. "
    "Retention rules purge exported copies on the corporate schedule. "
    "Artifact links are immutable and are never copied into free text fields. "
    "Row level security is enforced before any artifact is retrieved. "
)
HEADING2 = "Synthetic Table Section"
TABLE_HEADER = ["Tier", "Price", "Seats"]
TABLE_ROWS = [
    ["Starter", "4000", "50"],
    ["Growth", "12000", "250"],
    ["Enterprise", "Custom", "Unlimited"],
    ["Pilot", "0", "10"],
    ["Legacy", "3000", "25"],
    ["Partner", "6000", "100"],
    ["Academic", "1500", "40"],
    ["Trial", "0", "5"],
]


def build() -> Path:
    doc = fitz.open()
    page = doc.new_page()

    page.insert_text((60, 70), HEADING, fontsize=16, fontname="helv")
    box = fitz.Rect(60, 90, 540, 300)
    page.insert_textbox(box, PROSE, fontsize=9, fontname="helv")

    page.insert_text((60, 330), HEADING2, fontsize=16, fontname="helv")

    # Ruled table: PyMuPDF's default table finder keys on the lines.
    x0, y0, row_h = 60.0, 350.0, 18.0
    widths = [140.0, 140.0, 140.0]
    rows = [TABLE_HEADER, *TABLE_ROWS]
    for r, cells in enumerate(rows):
        y = y0 + r * row_h
        x = x0
        for c, cell in enumerate(cells):
            rect = fitz.Rect(x, y, x + widths[c], y + row_h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.6)
            page.insert_text((x + 4, y + 12), str(cell), fontsize=9, fontname="helv")
            x += widths[c]

    doc.save(OUT)
    doc.close()
    return OUT


if __name__ == "__main__":
    print(f"wrote {build()}")
