"""PDF parsing and structure-aware chunking.

Role in architecture: the offline half of RAG. `PdfChunker` converts both PDFs
into `Chunk` objects that the index layer embeds. Two chunk kinds, because these
documents carry meaning in two different shapes:

  SECTION chunks - heading + prose, ~300 tokens, never split mid-sentence. The
      heading is prepended to the chunk text so the embedding sees the topic even
      when the body is a bare bullet list.
  TABLE chunks   - the header row is repeated with EVERY group of data rows. A
      data row separated from its header is unusable ("21" means nothing without
      "SLA (days)"), and both PDFs put their highest-value content in tables
      (pricing tiers, stage playbook, risk rubric).

Chunk IDs are sha1(doc + page + char_offset)[:12] - deterministic, so re-ingest
produces identical IDs and traces stay comparable across runs.

In:  PDF paths from config.PDF_PATHS.
Out: list[Chunk] with {doc, page, section, chunk_id} metadata.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import fitz  # PyMuPDF

from core import config


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    page: int
    section: str
    text: str
    kind: str = "section"  # "section" | "table"

    def metadata(self) -> dict[str, Any]:
        return {
            "doc": self.doc,
            "page": self.page,
            "section": self.section,
            "kind": self.kind,
            "chunk_id": self.chunk_id,
        }


@dataclass
class TextLine:
    """One rendered line of the page, with the typography needed to classify it."""

    text: str
    size: float
    bold_ratio: float  # fraction of characters rendered in a bold span
    y: float
    order: int = field(default=0)

    @property
    def bold(self) -> bool:
        # "Mostly bold", not "contains bold": both PDFs use inline bold for
        # emphasis inside bullets ("**XYZ-SECURITY**: Audit export..."), and
        # treating those as headings shatters sections.
        return self.bold_ratio >= 0.8


class AssetMissing(FileNotFoundError):
    """A required PDF is not where the app expects it."""


# --------------------------------------------------------------------------
# pure helpers (module level - they have no state and are unit-tested directly)
# --------------------------------------------------------------------------
def make_chunk_id(doc: str, page: int, char_offset: int) -> str:
    """Stable across runs: same document + page + offset => same id."""
    return hashlib.sha1(f"{doc}|{page}|{char_offset}".encode("utf-8")).hexdigest()[:12]


def estimate_tokens(text: str) -> int:
    """~4 chars/token. Good enough for chunk sizing; avoids a tokenizer dependency."""
    return max(1, len(text) // 4)


_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+|\n")


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries so chunks never cut mid-sentence."""
    parts = [p.strip() for p in _SENTENCE_END.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


_FOOTER_RE = re.compile(r"^(page\s*\d+|.*\|\s*page\s*\d+|.*page\s*\d+)$", re.IGNORECASE)


class PdfChunker:
    """Turns one PDF into section and table chunks.

    Stateless between calls: `chunk(path, doc_name)` owns all per-document state
    (repeated-line furniture, per-page offsets), so one instance can process the
    whole corpus.
    """

    def __init__(
        self,
        target_tokens: int | None = None,
        table_rows_per_chunk: int | None = None,
    ) -> None:
        self.target_tokens = target_tokens or config.CHUNK_TARGET_TOKENS
        self.table_rows_per_chunk = table_rows_per_chunk or config.TABLE_ROWS_PER_CHUNK

    # ------------------------------------------------------------ typography --
    @staticmethod
    def page_lines(page: "fitz.Page", exclude: list[Any]) -> list[TextLine]:
        """Lines of the page with font size/bold, skipping anything in a table bbox."""
        lines: list[TextLine] = []
        data = page.get_text("dict")
        order = 0
        for block in data.get("blocks", []):
            if block.get("type") != 0:  # 0 = text
                continue
            bbox = fitz.Rect(block["bbox"])
            if any(bbox.intersects(t) for t in exclude):
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue
                size = max((s.get("size", 0.0) for s in spans), default=0.0)
                # PyMuPDF font flag bit 4 (value 16) = bold.
                bold_chars = sum(
                    len(s.get("text", "")) for s in spans if int(s.get("flags", 0)) & 16
                )
                total_chars = sum(len(s.get("text", "")) for s in spans) or 1
                lines.append(
                    TextLine(
                        text=text,
                        size=round(size, 1),
                        bold_ratio=bold_chars / total_chars,
                        y=line["bbox"][1],
                        order=order,
                    )
                )
                order += 1
        return lines

    @staticmethod
    def body_size(lines: Iterable[TextLine]) -> float:
        """Modal font size = body text size."""
        counts: dict[float, int] = {}
        for ln in lines:
            counts[ln.size] = counts.get(ln.size, 0) + len(ln.text)
        return max(counts, key=lambda k: counts[k]) if counts else 10.0

    @staticmethod
    def is_heading(line: TextLine, body: float) -> bool:
        # A heading is bigger than body text, or bold-and-short. The length cap
        # stops a bold inline emphasis run from being promoted to a section break.
        if line.size >= body + 1.2:
            return True
        return line.bold and len(line.text) <= 60 and not line.text.endswith(".")

    @staticmethod
    def is_furniture(text: str, repeated: frozenset[str] = frozenset()) -> bool:
        """Running headers/footers add noise to every chunk; drop them.

        Two signals: an explicit "Page N" pattern, and - more reliably - text
        that repeats on most pages. Without the second signal the Enablement
        Pack's left-hand footer ("Product XYZ Enablement Pack", drawn as its own
        text block) becomes a chunk of its own on every page.
        """
        stripped = text.strip()
        if stripped in repeated:
            return True
        return bool(_FOOTER_RE.match(stripped)) and len(stripped) < 60

    def repeated_lines(self, pdf: "fitz.Document", min_pages: int = 2) -> frozenset[str]:
        """Short lines appearing on at least half the pages = running furniture."""
        counts: dict[str, int] = {}
        n_pages = pdf.page_count
        for page in pdf:
            seen = {
                ln.text.strip()
                for ln in self.page_lines(page, exclude=[])
                if 0 < len(ln.text.strip()) <= 60
            }
            for text in seen:
                counts[text] = counts.get(text, 0) + 1
        threshold = max(min_pages, (n_pages + 1) // 2)
        return frozenset(t for t, c in counts.items() if c >= threshold)

    # ----------------------------------------------------------------- tables --
    @staticmethod
    def render_row(cells: list[Any]) -> str:
        return " | ".join((str(c).replace("\n", " ").strip() if c else "") for c in cells)

    @staticmethod
    def rows_by_span(page: "fitz.Page", table: Any) -> list[list[str]] | None:
        """Rebuild table rows from text SPANS instead of characters.

        Why this exists: both provided PDFs have table cells whose text overflows
        into the neighbouring column. PyMuPDF's default cell extraction assigns
        *characters* by position, so two overlapping runs interleave character by
        character and come out as noise ("eDxits mcoeveetirnyg"). Spans are the
        contiguous strings the PDF actually draws, so assigning whole spans to
        the column their left edge falls in keeps each phrase intact.

        Column boundaries come from the HEADER row, not each row's own `.cells`:
        PyMuPDF only records a cell where it detects a grid intersection for
        THAT row, so a row whose text wraps to a second/third line - reporting
        no intersection near columns with no text on that particular line -
        comes back with FEWER cells than the header. Bucketing against a
        shorter, row-specific cell list silently re-numbers every column for
        that row alone: text that visually continues the 5th column can land
        in bucket 0. The header's cell layout is the one stable, complete
        column grid every row should be read against.

        Returns None if row/cell geometry is unavailable, in which case the
        caller falls back to `table.extract()`.
        """
        rows = getattr(table, "rows", None)
        if not rows:
            return None

        header_cells = [c for c in (getattr(table.header, "cells", None) or []) if c is not None]
        if not header_cells:
            header_cells = [c for c in (rows[0].cells or []) if c is not None]
        if not header_cells:
            return None

        spans: list[tuple[Any, str]] = []
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        spans.append((fitz.Rect(span["bbox"]), text))

        out: list[list[str]] = []
        anchor_rect: fitz.Rect | None = None
        for row in rows:
            row_rect = fitz.Rect(row.bbox)
            # A row whose vertical extent falls entirely inside the last
            # emitted (non-duplicate) row's extent is a duplicate detection of
            # the SAME wrapped lines that row's own (taller) bbox already
            # covers - not a new logical row. Geometric, not column-based:
            # column-blank detection misses this when the wrap happens in the
            # key column itself (e.g. a stage name spanning two lines), since
            # then the duplicate row's key column isn't blank either.
            if (
                anchor_rect is not None
                and anchor_rect.y0 - 1 <= row_rect.y0
                and row_rect.y1 <= anchor_rect.y1 + 1
            ):
                continue
            anchor_rect = row_rect
            buckets: list[list[tuple[float, str]]] = [[] for _ in header_cells]
            for rect, text in spans:
                centre_y = (rect.y0 + rect.y1) / 2
                if not (row_rect.y0 - 1 <= centre_y <= row_rect.y1 + 1):
                    continue
                # Assign by the span's LEFT edge: an overflowing span belongs to
                # the column it starts in, not the one it spills into.
                idx = next(
                    (i for i, c in enumerate(header_cells) if c[0] - 1 <= rect.x0 < c[2] + 1),
                    None,
                )
                if idx is None:
                    idx = min(
                        range(len(header_cells)), key=lambda i: abs(header_cells[i][0] - rect.x0)
                    )
                # Sort key is (top, left): when a row's bbox is tall enough to
                # span several wrapped physical lines (see _drop_continuation_
                # rows below for why that happens), sorting by x0 alone
                # interleaves those lines by horizontal position instead of
                # preserving top-to-bottom reading order.
                buckets[idx].append((rect.y0, rect.x0, text))
            out.append([" ".join(t for _, _, t in sorted(b)) for b in buckets])
        return out or None

    @staticmethod
    def _drop_continuation_rows(rows: list[list[str]]) -> list[list[str]]:
        """Drop PyMuPDF "row" detections that are pure duplicates of wrapped text.

        Some PDFs give a table row a bbox TALL enough to span every wrapped
        line of its cells (its key column, e.g. a stage name, is drawn once on
        the first line, but the row's height covers all lines) - `rows_by_span`
        already collects every span in that y-range, so the row's rendered
        text is complete on its own. PyMuPDF ALSO reports narrower "rows" for
        the later wrapped lines individually, each with an empty first (key)
        cell; since their y-range sits INSIDE the tall row's, `rows_by_span`
        collects the same spans again for them - pure duplicates, not new
        content. Left in, one logical row's 3 wrapped lines become 3 output
        rows (1 complete + 2 duplicate fragments), and a fixed
        `table_rows_per_chunk` then splits what should be one chunk into
        several incomplete-looking ones - a real case: an 18-raw-row, 6-stage
        table produced 3 garbled chunks instead of the 1 its 6 real rows
        warranted.

        Detection: a row whose first cell is blank is a continuation
        fragment - true for every table in this corpus, where the first
        column is always the row's key.
        """
        return [row for row in rows if (row[0] or "").strip()] or rows

    def table_chunks(
        self, page: "fitz.Page", doc_name: str, page_no: int, section: str, base_offset: int
    ) -> tuple[list[Chunk], list[Any], int]:
        """Extract tables as header-preserving chunks. Returns (chunks, bboxes, offset)."""
        chunks: list[Chunk] = []
        bboxes: list[Any] = []
        offset = base_offset
        try:
            found = page.find_tables()
        except Exception:  # pragma: no cover - older PyMuPDF without table finder
            return [], [], offset

        for table in getattr(found, "tables", []):
            raw_rows = self.rows_by_span(page, table) or table.extract()
            rows = [r for r in raw_rows if any(c and str(c).strip() for c in r)]
            if len(rows) < 2:
                continue
            bboxes.append(fitz.Rect(table.bbox))
            header = self.render_row(rows[0])
            data = self._drop_continuation_rows(rows[1:])
            step = self.table_rows_per_chunk
            for i in range(0, len(data), step):
                group = data[i : i + step]
                body = "\n".join(self.render_row(r) for r in group)
                # Header repeated with every group - the invariant asserted in
                # tests/test_chunking.py.
                text = f"{section}\nTABLE\n{header}\n{'-' * len(header)}\n{body}"
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(doc_name, page_no, offset),
                        doc=doc_name,
                        page=page_no,
                        section=section,
                        text=text,
                        kind="table",
                    )
                )
                offset += len(text)
        return chunks, bboxes, offset

    # ------------------------------------------------------------ entry points --
    def chunk(self, path: Path, doc_name: str) -> list[Chunk]:
        """Parse one PDF into section + table chunks."""
        if not Path(path).exists():
            raise AssetMissing(
                f"PDF not found at {path}.\nPlace the provided PDFs in {config.ASSETS_DIR}/."
            )
        chunks: list[Chunk] = []

        with fitz.open(path) as pdf:
            furniture = self.repeated_lines(pdf)
            for page_index, page in enumerate(pdf):
                page_no = page_index + 1
                offset = 0

                # Tables first so their text can be excluded from the prose pass.
                # Table `section` needs a heading, so resolve the page's dominant
                # heading up front.
                all_lines = self.page_lines(page, exclude=[])
                body = self.body_size(all_lines)
                page_heading = next(
                    (
                        ln.text
                        for ln in all_lines
                        if self.is_heading(ln, body) and not self.is_furniture(ln.text, furniture)
                    ),
                    f"{doc_name} p.{page_no}",
                )

                tbl_chunks, tbl_boxes, offset = self.table_chunks(
                    page, doc_name, page_no, page_heading, offset
                )
                chunks.extend(tbl_chunks)

                # Prose pass, excluding table regions.
                sections = self._split_sections(
                    self.page_lines(page, exclude=tbl_boxes), body, page_heading, furniture
                )
                for section, body_lines in sections:
                    new_chunks, offset = self._chunk_section(
                        section, body_lines, doc_name, page_no, offset
                    )
                    chunks.extend(new_chunks)

        return chunks

    def _split_sections(
        self,
        lines: list[TextLine],
        body: float,
        page_heading: str,
        furniture: frozenset[str],
    ) -> list[tuple[str, list[str]]]:
        """Group body lines under the heading that precedes them."""
        sections: list[tuple[str, list[str]]] = []
        current = page_heading
        buffer: list[str] = []
        for ln in lines:
            if self.is_furniture(ln.text, furniture):
                continue
            if self.is_heading(ln, body):
                if buffer:
                    sections.append((current, buffer))
                    buffer = []
                current = ln.text
            else:
                buffer.append(ln.text)
        if buffer:
            sections.append((current, buffer))
        return sections

    def _chunk_section(
        self, section: str, body_lines: list[str], doc_name: str, page_no: int, offset: int
    ) -> tuple[list[Chunk], int]:
        """Pack a section's sentences into ~target_tokens chunks, never splitting one."""
        chunks: list[Chunk] = []
        pending: list[str] = []

        def flush() -> None:
            nonlocal offset
            text = f"{section}\n" + " ".join(pending)
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc_name, page_no, offset),
                    doc=doc_name,
                    page=page_no,
                    section=section,
                    text=text,
                )
            )
            offset += len(text)

        for sentence in split_sentences(" ".join(body_lines)):
            trial = pending + [sentence]
            if pending and estimate_tokens(" ".join(trial)) > self.target_tokens:
                flush()
                pending = [sentence]
            else:
                pending = trial
        if pending:
            flush()
        return chunks, offset

    def chunk_corpus(self, pdf_paths: dict[str, Path] | None = None) -> list[Chunk]:
        """Chunk every configured PDF. Deterministic order: config order, then page."""
        paths = pdf_paths or config.PDF_PATHS
        out: list[Chunk] = []
        for doc_name, path in paths.items():
            out.extend(self.chunk(Path(path), doc_name))
        return out


# --------------------------------------------------------------------------
# module-level convenience wrappers
# --------------------------------------------------------------------------
def chunk_pdf(path: Path, doc_name: str) -> list[Chunk]:
    return PdfChunker().chunk(Path(path), doc_name)


def chunk_all(pdf_paths: dict[str, Path] | None = None) -> list[Chunk]:
    return PdfChunker().chunk_corpus(pdf_paths)
