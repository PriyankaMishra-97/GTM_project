"""Streamlit chat UI.

Role in architecture: It calls `orchestrator.answer_question`
and renders three things per turn: the answer, a route badge, and a collapsed
Trace panel. No pipeline logic lives here - if the UI is deleted, the system is
still fully usable through `orchestrator.answer_question`.

Run: streamlit run app.py
"""

from __future__ import annotations

import json

import streamlit as st

from core import config
from core.auth import UserProfile, UsersFileMissing, UserStore
from orchestrator import answer_question, preflight

st.set_page_config(page_title="GTM Analyst Copilot", page_icon="📊", layout="wide")

# ------------------------------------------------------------------- login --
if "user" not in st.session_state:
    st.title("GTM Analyst Copilot")
    try:
        _store = UserStore()
    except UsersFileMissing as exc:
        st.error(str(exc))
        st.stop()

    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
    if submitted:
        profile = _store.authenticate(username, password)
        if profile is None:
            st.error("Incorrect username or password.")
        else:
            st.session_state.user = profile
            st.rerun()
    st.stop()

user: UserProfile = st.session_state.user

ROUTE_STYLE = {
    "RAG": ("#1f6feb", "📄"),
    "SQL": ("#1a7f37", "🗄️"),
    "HYBRID": ("#8250df", "🔀"),
    "ASK": ("#9a6700", "❓"),
    "REFUSE": ("#cf222e", "🛑"),
}


def route_badge(route: str) -> str:
    colour, icon = ROUTE_STYLE.get(route, ("#57606a", "•"))
    return (
        f'<span style="background:{colour};color:#fff;padding:2px 10px;'
        f'border-radius:12px;font-size:0.78rem;font-weight:600;">{icon} {route}</span>'
    )


def recent_answered_turns(messages: list[dict], limit: int = 2) -> list[dict]:
    """Last `limit` (question, answer) pairs from real SQL/RAG/HYBRID turns.

    ASK/REFUSE turns carry no factual content worth referencing (see
    ask/history_reframe.py's module docstring), so they're skipped here -
    only an answered turn is worth feeding to the history-reframe step.
    """
    turns: list[dict] = []
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg["role"] == "assistant" and msg.get("route") in ("SQL", "RAG", "HYBRID"):
            if i > 0 and messages[i - 1]["role"] == "user":
                turns.append({"question": messages[i - 1]["content"], "answer": msg["content"]})
        if len(turns) >= limit:
            break
    turns.reverse()
    return turns


# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("GTM Analyst Copilot")
    st.caption("Fully local: Ollama + sentence-transformers. No cloud calls.")

    st.write(f"Logged in as **{user.username}**")
    if st.button("Log out", use_container_width=True):
        for key in ("user", "messages", "pending_clarification", "pending_missing_slots"):
            st.session_state.pop(key, None)
        st.rerun()

    status = preflight()

    st.subheader("Models")
    st.write(f"**Router:** `{config.ROUTER_MODEL}`")
    st.write(f"**Answer:** `{config.ANSWER_MODEL}`")
    if status["provider_ready"]:
        st.success("Ollama reachable, both models pulled")
    else:
        st.error(status.get("error", "Ollama unavailable"))
        st.code(f"ollama pull {config.ROUTER_MODEL}\nollama pull {config.ANSWER_MODEL}")

    st.subheader("Indexes")
    idx = status["index"]
    if idx.get("ready"):
        st.success(f"Vector + BM25 index ready ({idx.get('chunks', 0)} chunks)")
    else:
        st.warning("No index yet - run `python -m scripts.ingest`")

    if st.button("Re-ingest PDFs", use_container_width=True):
        with st.spinner("Re-chunking and re-embedding both PDFs..."):
            from rag.index import Index
            from rag.ingest import PdfChunker

            stats = Index().build(PdfChunker().chunk_corpus(), force=True)
        st.success(f"Re-indexed {stats['chunks']} chunks")
        st.rerun()

    st.subheader("Database")
    if status.get("db_ready"):
        st.write("`assets/gtm_mock.db` (read-only)")
        st.write(", ".join(f"`{t}`" for t in status.get("tables", [])))
    else:
        st.error(f"Missing DB. Place gtm_mock.db in {config.ASSETS_DIR}/")
    for warning in status.get("schema_warnings", []):
        st.warning(f"schema card drift: {warning}")

    st.divider()
    st.caption("Demo prompts")
    st.code(
        "What deployment modes does Product XYZ support?\n"
        "How many opportunities were Closed Won in EMEA in 2024?\n"
        "What is our 2024 win rate for Enterprise, and what does the\n"
        "  field guide require before a deal reaches Commit?\n"
        "How's pipeline looking recently?",
        language=None,
    )

# ------------------------------------------------------------------- chat --
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None
if "pending_missing_slots" not in st.session_state:
    st.session_state.pending_missing_slots = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("route"):
            st.markdown(route_badge(msg["route"]), unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander("Trace", expanded=False):
                st.json(msg["trace"], expanded=False)

if prompt := st.chat_input("Ask about Product XYZ, the tracker, or the GTM pipeline..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Routing..."):
            answer = answer_question(
                prompt,
                user=user,
                pending_clarification=st.session_state.pending_clarification,
                pending_missing_slots=st.session_state.pending_missing_slots,
                recent_turns=recent_answered_turns(st.session_state.messages),
            )
        if answer.route == "ASK":
            st.session_state.pending_clarification = answer.trace.question
            st.session_state.pending_missing_slots = answer.trace.missing_slots
        else:
            st.session_state.pending_clarification = None
            st.session_state.pending_missing_slots = None

        st.markdown(route_badge(answer.route), unsafe_allow_html=True)
        st.markdown(answer.text)

        trace = answer.trace.to_dict()
        with st.expander(
            f"Trace - {answer.route}"
            + (f" (rule {answer.trace.rule_override})" if answer.trace.rule_override else "")
            + f" - {answer.trace.total_latency_ms} ms",
            expanded=False,
        ):
            st.json(trace, expanded=False)
            if answer.sql:
                st.caption("Executed SQL")
                st.code(answer.sql, language="sql")
            if answer.hits:
                st.caption("Retrieved chunks")
                for h in answer.hits:
                    st.markdown(
                        f"**{h.citation()}** {h.section} - rrf `{h.rrf_score:.5f}` "
                        f"(dense #{h.dense_rank}, bm25 #{h.bm25_rank})"
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer.text,
            "route": answer.route,
            "trace": json.loads(json.dumps(trace, default=str)),
        }
    )
