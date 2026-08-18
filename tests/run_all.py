"""Single test entry point:  python -m tests.run_all

Prints an environment banner first, because roughly half the suite is
conditionally skipped (Ollama-dependent routing, DB-dependent integration) and a
run with 30 silent skips looks identical to a clean run otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _banner() -> None:
    from core import config
    from core.llm_client import OllamaClient
    from rag import index as index_mod

    ollama = OllamaClient().is_ready()
    print("=" * 68)
    print("GTM Analyst Copilot - test environment")
    print("=" * 68)
    print(f"  database    : {'present' if config.DB_PATH.exists() else 'MISSING'}  ({config.DB_PATH})")
    print(f"  PDFs        : " + ", ".join(
        f"{name}={'ok' if p.exists() else 'MISSING'}" for name, p in config.PDF_PATHS.items()
    ))
    print(f"  RAG index   : {'built' if index_mod.index_exists() else 'not built (RAG tests skip)'}")
    print(f"  Ollama      : {'ready' if ollama else 'unreachable (router golden set skips)'}")
    print(f"                router={config.ROUTER_MODEL}  answer={config.ANSWER_MODEL}")
    print("=" * 68)
    print()


def main() -> int:
    _banner()
    return pytest.main([str(ROOT / "tests"), "-q", "--no-header", "-p", "no:cacheprovider"])


if __name__ == "__main__":
    raise SystemExit(main())
