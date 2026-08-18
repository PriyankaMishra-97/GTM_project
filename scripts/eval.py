"""RAGAS-style quality evaluation for the RAG/HYBRID paths.

    python -m scripts.eval               # run the labelled set, print scores
    python -m scripts.eval --md out.md   # also write an EVALUATION.md report

Role: measures what tests/test_router_golden.py does not - whether retrieval
found the right context (context recall/precision), and whether the
generated answer is actually grounded in it (faithfulness) and on-topic
(answer relevancy). See eval/metrics.py for each metric's definition and
eval/dataset.py for the labelled question set. Mirrors scripts/demo.py's
structure.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from core.auth import UserProfile
from eval.dataset import RAG_EVAL_SET, EvalCase
from eval.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from orchestrator import get_copilot

# A fixed, unrestricted identity for eval runs - not a real login, just enough
# to satisfy answer_question's required `user` parameter (added for the
# region/segment ACL feature) without any region/segment scoping getting in
# the way of measuring RAG/HYBRID quality.
EVAL_USER = UserProfile(username="eval", allowed_regions=None, allowed_segments=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", type=Path, help="write a markdown evaluation report")
    args = parser.parse_args(argv)

    copilot = get_copilot()
    try:
        copilot.client.preflight()
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2

    rows: list[dict] = []
    for case in RAG_EVAL_SET:
        answer = copilot.answer(case.question, EVAL_USER)
        trace = answer.trace

        row: dict = {
            "question": case.question,
            "expected_route": case.route,
            "route": answer.route,
            "context_recall": None,
            "context_precision": None,
            "faithfulness": None,
            "answer_relevancy": None,
            "claims": [],
        }

        if answer.route != case.route:
            # Misrouted - nothing to score against a retrieval/answer that
            # wasn't produced by the path this case is meant to test. The
            # mismatch itself is the signal, printed and reported as-is.
            rows.append(row)
            print(f"[MISROUTE] expected {case.route}, got {answer.route}: {case.question[:60]}")
            continue

        retrieved = [(c["doc"], c["section"]) for c in trace.retrieved_chunks]
        contexts = [c["text"] for c in trace.retrieved_chunks]

        row["context_recall"] = context_recall(retrieved, list(case.relevant_sections))
        row["context_precision"] = context_precision(retrieved, list(case.relevant_sections))

        faith = faithfulness(case.question, answer.text, contexts, copilot.client)
        row["faithfulness"] = faith.score
        row["claims"] = faith.claims

        row["answer_relevancy"] = answer_relevancy(
            case.question, answer.text, copilot.client, copilot.index.embed_query
        )

        print(
            f"[{answer.route:6}] recall={_fmt(row['context_recall'])} "
            f"precision={_fmt(row['context_precision'])} "
            f"faithfulness={_fmt(row['faithfulness'])} "
            f"relevancy={_fmt(row['answer_relevancy'])}  {case.question[:50]}"
        )
        rows.append(row)

    print()
    for metric in (
        "context_recall",
        "context_precision",
        "faithfulness",
        "answer_relevancy",
    ):
        values = [r[metric] for r in rows if r[metric] is not None]
        if values:
            print(f"mean {metric}: {statistics.mean(values):.3f}  (n={len(values)})")
        else:
            print(f"mean {metric}: n/a (no scoreable cases)")

    if args.md:
        _write_report(args.md, rows)
        print(f"\nwrote {args.md}")
    return 0


def _fmt(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else " n/a"


def _write_report(path: Path, rows: list[dict]) -> None:
    lines = [
        "# RAG/HYBRID quality evaluation (RAGAS-style, local)",
        "",
        "Native implementation of RAGAS-style metrics - not the `ragas` package, "
        "which hard-depends on LangChain. See `eval/metrics.py` for each "
        "definition. `context_recall`/`context_precision` are deterministic "
        "(no LLM, need a pre-labelled ground-truth section); "
        "`faithfulness`/`answer_relevancy` use an LLM judge "
        "(no labels needed) and are therefore approximate, not exact.",
        "",
        "| Route | Question | Recall | Precision | Faithfulness | Relevancy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        q = r["question"][:60] + ("..." if len(r["question"]) > 60 else "")
        route = r["route"] if r["route"] == r["expected_route"] else f"{r['route']} (expected {r['expected_route']})"
        lines.append(
            f"| {route} | {q} | {_fmt(r['context_recall'])} | "
            f"{_fmt(r['context_precision'])} | "
            f"{_fmt(r['faithfulness'])} | {_fmt(r['answer_relevancy'])} |"
        )

    lines += ["", "## Means", ""]
    for metric in (
        "context_recall",
        "context_precision",
        "faithfulness",
        "answer_relevancy",
    ):
        values = [r[metric] for r in rows if r[metric] is not None]
        if values:
            lines.append(f"- **{metric}**: {statistics.mean(values):.3f} (n={len(values)})")
        else:
            lines.append(f"- **{metric}**: n/a (no scoreable cases)")

    unsupported = [
        (r["question"], c.text)
        for r in rows
        for c in r["claims"]
        if not c.supported
    ]
    if unsupported:
        lines += ["", "## Unsupported claims (faithfulness failures)", ""]
        for question, claim in unsupported:
            lines.append(f"- *{question[:50]}...*: \"{claim}\"")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
