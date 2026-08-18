"""RAG answer generation: retrieve -> ground -> cite.

Role in architecture: the RAG branch's terminal step. `RagPath.run()` is what
the orchestrator calls; it wraps retrieval + one grounded, cited ANSWER_MODEL
call. No re-ranking pass, no self-critique loop - each extra call is ~2-4s
against a 10s budget, and the citation rule plus the safety filter already cover
the failure modes a critique pass would catch.

In:  question + Trace.
Out: (markdown answer with [doc, p.N] citations, retrieved Hits).
"""

from __future__ import annotations

from core import safety
from core.llm_client import LLMClient, LLMUnavailable, get_client
from core.trace import Trace
from rag import prompts
from rag.retrieve import Hit, Retriever

NO_CONTEXT_MESSAGE = (
    "I couldn't find anything relevant in the indexed documents.\n\n"
    "Product capability, deployment modes, packaging/pricing and the FAQ are in the "
    "**Enablement Pack**; tracker fields, the stage playbook, deployment status "
    "taxonomy, risk scoring and compliance guardrails are in the **Field Guide**. "
    "If your question is about one of those, try naming the specific term - "
    "otherwise the answer may not be in either PDF."
)

# Deterministic conflict detection. The Enablement Pack states superseded and
# current positions in the same breath ("Important note (legacy): ... Updated
# note: ..."). Relying on the model to notice rule 4 on its own is unreliable on
# a 7B, so a cheap regex pass finds the pattern and injects a pointed reminder
# naming the offending chunk. Rules dispose; the model just writes it up.
_STALE_MARKERS = ("legacy", "older runbook", "older doc", "previously", "used to")
_CURRENT_MARKERS = (
    "updated note", "updated pricing", "as of v3.0", "v3.0 recommends",
    "v3.0 guide", "v3.0 explicitly", "the v3.0",
)


def detect_conflicts(hits: list[Hit]) -> list[Hit]:
    """Chunks that assert both a superseded and a current position."""
    flagged: list[Hit] = []
    for hit in hits:
        text = hit.text.lower()
        if any(m in text for m in _STALE_MARKERS) and any(m in text for m in _CURRENT_MARKERS):
            flagged.append(hit)
    return flagged


def conflict_notice(flagged: list[Hit]) -> str:
    if not flagged:
        return ""
    refs = "; ".join(f"{h.citation()} ({h.section})" for h in flagged)
    return (
        "\n\nCONFLICT ALERT (rule 4 applies): the following retrieved chunk(s) "
        f"contain BOTH a superseded/legacy statement and an updated statement: {refs}. "
        "If your answer touches that topic you MUST quote both positions with both "
        "citations and flag the conflict explicitly. Do not present only the newer one."
    )


def build_context(hits: list[Hit]) -> str:
    """Render hits into the chunk block the prompt expects.

    Deliberately no "chunk N" ordinal: it sat right next to `page: {page}` in
    the header, and the small model would sometimes cite the chunk's position
    in the list instead of its actual page (chunk 4 happened to be page 2;
    the answer cited "p.4"). Removing the second small integer removes the
    thing it was substituting.
    """
    return "\n".join(
        prompts.CHUNK_TEMPLATE.format(doc=h.doc, page=h.page, section=h.section or "-", text=h.text)
        for h in hits
    )


class RagPath:
    """The RAG branch of the router: retrieve, ground, cite."""

    def __init__(self, retriever: Retriever | None = None, client: LLMClient | None = None) -> None:
        self.retriever = retriever or Retriever()
        self.client = client or get_client()

    def answer_from_hits(
        self, question: str, hits: list[Hit], trace: Trace | None = None
    ) -> str:
        if not hits:
            return NO_CONTEXT_MESSAGE
        user = prompts.RAG_USER.format(context=build_context(hits), question=question)
        user += conflict_notice(detect_conflicts(hits))
        text = self.client.chat(prompts.RAG_SYSTEM, user, trace=trace, stage="rag_answer")
        # Output filter runs on every model response - retrieved chunks are
        # untrusted input, so a prompt-injection attempt inside a PDF is caught.
        return safety.filter_response(text)

    def run(self, question: str, trace: Trace) -> tuple[str, list[Hit]]:
        """Full RAG branch: retrieve -> record in trace -> answer."""
        try:
            with trace.stage("rag_retrieve"):
                hits = self.retriever.retrieve(
                    question, trace=trace, stage_prefix="rag_retrieve"
                )
        except FileNotFoundError as exc:
            trace.error(str(exc))
            return f"The document index is not built yet.\n\n`{exc}`", []

        trace.add_chunks([h.to_trace() for h in hits])

        try:
            with trace.stage("rag_answer"):
                text = self.answer_from_hits(question, hits, trace=trace)
        except LLMUnavailable as exc:
            trace.error(str(exc))
            return f"The local model is unavailable: {exc}", hits

        return text, hits


def run_rag_path(
    question: str, trace: Trace, *, client: LLMClient | None = None
) -> tuple[str, list[Hit]]:
    """Module-level convenience wrapper around `RagPath`."""
    return RagPath(client=client).run(question, trace)
