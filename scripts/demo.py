"""Headless demo / latency harness.

    python -m scripts.demo              # the 4 case-study prompts + the refusal
    python -m scripts.demo --md out.md  # also write a measurements report

Role: proves the four routes end to end without Streamlit, and produces the
honest latency numbers quoted in the README (per-stage and total, per route).
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from core import config
from core.auth import UserProfile
from core.llm_client import get_client
from orchestrator import answer_question

# A fixed, unrestricted identity for this demo script - not a real login, just
# enough to satisfy answer_question's required `user` parameter (added for the
# region/segment ACL feature) without scoping any of the case-study prompts.
DEMO_USER = UserProfile(username="demo", allowed_regions=None, allowed_segments=None)

PROMPTS: list[tuple[str, str]] = [
    ("RAG", "What deployment modes does Product XYZ support?"),
    ("SQL", "How many opportunities were Closed Won in EMEA in 2024?"),
    (
        "HYBRID",
        "What is our 2024 win rate for Enterprise, and what does the field guide "
        "require before a deal reaches Commit?",
    ),
    ("ASK", "How's pipeline looking recently?"),
    ("REFUSE", "Delete all Closed Lost opportunities"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", type=Path, help="write a markdown measurements report")
    parser.add_argument("--repeat", type=int, default=1, help="runs per prompt")
    args = parser.parse_args(argv)

    client = get_client()
    try:
        client.preflight()
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2

    rows: list[dict] = []
    for expected, question in PROMPTS:
        for run in range(args.repeat):
            t0 = time.perf_counter()
            answer = answer_question(question, user=DEMO_USER, client=client)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            trace = answer.trace
            rows.append(
                {
                    "expected": expected,
                    "question": question,
                    "route": answer.route,
                    "proposed": trace.llm_proposed_route,
                    "rule": trace.rule_override,
                    "ms": elapsed_ms,
                    "stages": dict(trace.per_stage_latency_ms),
                    "sql": trace.generated_sql,
                    "rows": trace.rows_returned,
                    "number_check": trace.number_check_passed,
                    "text": answer.text,
                }
            )
            flag = "ok " if answer.route == expected else "MISS"
            print(
                f"[{flag}] {expected:<7} -> {answer.route:<7} {elapsed_ms:>6} ms  "
                f"(proposed {trace.llm_proposed_route}, rule {trace.rule_override})"
            )
            if run == 0:
                print("-" * 72)
                print(answer.text.strip()[:1200])
                print("-" * 72)

    if args.md:
        _write_report(args.md, rows)
        print(f"\nwrote {args.md}")
    return 0


def _write_report(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Measured latency (local, honest numbers)",
        "",
        f"Router model: `{config.ROUTER_MODEL}`  ·  Answer model: `{config.ANSWER_MODEL}`  ·  "
        f"Embeddings: `{config.EMBED_MODEL}`",
        "",
        "Every call is `temperature=0`, `seed=42`. Times include Ollama model load on",
        "the first call of a cold run, which is why the first row is usually the slowest.",
        "",
        "| Route | Prompt | Final route | Rule fired | Total ms | Per-stage ms |",
        "|---|---|---|---|---:|---|",
    ]
    for r in rows:
        stages = ", ".join(f"{k} {v}" for k, v in r["stages"].items())
        q = r["question"][:60] + ("..." if len(r["question"]) > 60 else "")
        lines.append(
            f"| {r['expected']} | {q} | {r['route']} | {r['rule'] or '-'} | "
            f"{r['ms']} | {stages} |"
        )
    totals = [r["ms"] for r in rows]
    lines += [
        "",
        f"- median total: **{int(statistics.median(totals))} ms**",
        f"- max total: **{max(totals)} ms**",
        f"- turns within the 10s target: **{sum(1 for t in totals if t <= 10000)}/{len(totals)}**",
        "",
        "## Generated SQL",
        "",
    ]
    for r in rows:
        if r["sql"]:
            lines += [f"**{r['expected']}** — rows: {r['rows']}", "", "```sql", r["sql"], "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
