"""CLI: build the RAG indexes.

    python -m scripts.ingest            # no-op if ./storage already has an index
    python -m scripts.ingest --force    # rebuild from scratch

Role in architecture: the only offline step. Idempotent by design so a demo can
be re-run without paying the embedding cost twice.
"""

from __future__ import annotations

import argparse
import sys
import time

from core import config
from rag.index import Index
from rag.ingest import AssetMissing, PdfChunker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the GTM Copilot RAG index.")
    parser.add_argument("--force", action="store_true", help="rebuild even if an index exists")
    args = parser.parse_args(argv)

    index = Index()
    if index.exists() and not args.force:
        stats = index.stats()
        print(f"Index already present ({stats.get('chunks', 0)} chunks). Use --force to rebuild.")
        return 0

    t0 = time.perf_counter()
    try:
        chunks = PdfChunker().chunk_corpus()
    except AssetMissing as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    by_doc: dict[str, int] = {}
    tables = 0
    for c in chunks:
        by_doc[c.doc] = by_doc.get(c.doc, 0) + 1
        tables += c.kind == "table"
    print(f"Parsed {len(chunks)} chunks ({tables} table chunks) from {len(config.PDF_PATHS)} PDFs:")
    for doc, n in by_doc.items():
        print(f"  - {doc}: {n}")

    print(f"Embedding with {config.EMBED_MODEL} ...")
    stats = index.build(chunks, force=args.force)
    print(
        f"Indexed {stats['chunks']} chunks into {config.CHROMA_DIR} "
        f"and {config.BM25_PATH} in {time.perf_counter() - t0:.1f}s."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
