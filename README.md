# GTM Analyst Copilot

A local-only "GTM Analyst Copilot" for a B2B SaaS team. It answers questions from two product PDFs (**RAG**), from a synthetic GTM SQLite database (**Text-to-SQL**), from both at once (**Hybrid**), or it asks a clarifying question when the request is underspecified (**Ask**). Everything runs on the laptop — Ollama for the LLMs, a local `sentence-transformers` model for embeddings, ChromaDB and BM25 on disk — so no data and no prompt ever leaves the machine.

The system is a **routed pipeline, not a free-form agent**: one user turn takes exactly one deterministic path, and every stage writes into a single `Trace` object that is appended to `storage/traces.jsonl` and rendered under each answer in the UI.

---

## Architecture

```
                       ┌───────────────────────────────────────┐
   user question ─────▶│  app.py  (Streamlit chat)             │
                       └───────────────┬───────────────────────┘
                                       │
                       ┌───────────────▼───────────────────────┐
                       │  orchestrator.answer_question()       │
                       │  owns the Trace for this turn         │
                       └───────────────┬───────────────────────┘
                                       │
             ┌─────────────────────────▼──────────────────────────┐
             │  ROUTER                                            │
             │  router/llm_router.py   1 call, llama3.2:3b        │
             │    └─ JSON: route, slots, confidence, subquestions │
             │  router/rules.py        deterministic OVERRIDE     │
             │    R0 write intent → REFUSE                        │
             │    R1 missing slot  → ASK                          │
             │    R2 vague time    → ASK                          │
             │    R3 conf < 0.60   → ASK                          │
             └───┬──────────┬───────────┬──────────────┬──────────┘
                 │          │           │              │
       ┌─────────▼──┐ ┌─────▼──────┐ ┌──▼───────────┐ ┌▼─────────────┐
       │  RAG       │ │  SQL       │ │  HYBRID      │ │ ASK / REFUSE │
       ├────────────┤ ├────────────┤ ├──────────────┤ ├──────────────┤
       │ retrieve   │ │ generate   │ │ 1. SQL path  │ │ ask/clarify  │
       │  ├ dense   │ │ guard(AST) │ │ 2. summarise │ │ 1-3 questions│
       │  │ chroma  │ │ execute RO │ │ 3. retrieve  │ │ each with a  │
       │  └ bm25    │ │  └ 1 repair│ │    conditioned│ │ default      │
       │ RRF k=60   │ │ render:    │ │    on result │ │              │
       │ top-7      │ │  table only│ │ 4. compose   │ │ REFUSE is a  │
       │ answer     │ │  or narrate│ │ 5. verify #s │ │ fixed string │
       │ + citations│ │ + verify #s│ │              │ │ (no LLM)     │
       └─────┬──────┘ └─────┬──────┘ └──────┬───────┘ └──────┬───────┘
             │              │               │                │
             └──────────────┴───────┬───────┴────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │ core/safety.py  output filter │
                    │  prompt-leak redaction        │
                    │  column denylist              │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │ answer + route badge + Trace  │
                    │ storage/traces.jsonl (1 line) │
                    └───────────────────────────────┘

  offline:  scripts/ingest.py ──▶ rag/ingest.py (structure-aware chunking)
                              ──▶ rag/index.py  (bge-small → chroma, BM25 → pkl)
```

**Module map**

| Package | Responsibility |
|---|---|
| `core/` | `config.py` (all constants), `llm_client.py` (abstract `LLMClient` + `OllamaClient`), `trace.py`, `safety.py`, `auth.py` (bcrypt login + region/segment ACL) |
| `router/` | `llm_router.py` (LLM proposal), `rules.py` (deterministic override), `slots.py`, `prompts.py` |
| `rag/` | `ingest.py` (PDF → chunks), `index.py` (dense + lexical), `retrieve.py` (RRF), `answer.py`, `prompts.py` |
| `sql/` | `schema.py` (introspection + generated card), `generate.py`, `guard.py`, `execute.py`, `narrate.py`, `pipeline.py`, `prompts.py` |
| `hybrid/` | `pipeline.py` (fixed 4 steps), `composer.py`, `verify.py` (verbatim-number check), `prompts.py` |
| `ask/` | `clarify.py`, `reframe.py` (continuation-vs-pivot after an ASK), `history_reframe.py` (resolves follow-ups like "and for EMEA?" against the last 1-2 answered turns, when NOT following an ASK), `text_overlap.py` (shared merge-safety guards), `prompts.py` |
| `eval/` | `metrics.py` (RAGAS-style context recall/precision, faithfulness, answer relevancy), `dataset.py`, `prompts.py` — offline quality evaluation, no LangChain |
| `orchestrator.py` | wires the above; the only entry point the UI uses |
| `scripts/` | `ingest.py` (build the index), `demo.py` (headless 5-route run + latency report), `eval.py` (runs `eval/` against the real stack → `EVALUATION.md`) |
| `tests/` | `run_all.py` single entry; guard, RRF, number-check, chunking, router, safety, DB integration, reframe, eval metrics, offline end-to-end pipeline |

Companion doc: **`ARCHITECTURE.md`** — a from-scratch build reference: every
class/function signature, every algorithm, config knob, and file-by-file build
order, precise enough to reproduce this repo without seeing the original code.

### Object model

The system is built as collaborating objects, not a bag of functions. Every stage is a class with injected dependencies, so each one is testable in isolation and swappable without editing its callers.

| Class | File | Responsibility |
|---|---|---|
| `LLMClient` (ABC) | `core/llm_client.py` | determinism settings, JSON repair loop, preflight — provider-agnostic |
| `OllamaClient` | `core/llm_client.py` | the one provider-specific method, `_complete()` |
| `Trace` | `core/trace.py` | per-turn observability record; `stage()` context manager times each step |
| `Router` | `router/llm_router.py` | one LLM call → `RouterDecision` (proposal only) |
| `RoutingRule` (ABC) → `PiiRequestRule`, `WriteIntentRule`, `OffDomainRule`, `HybridWithoutSqlRule`, `InvalidPopulationValueRule`, `MissingSlotRule`, `RegionScopeRule`, `VagueTimeRule`, `LowConfidenceRule` | `router/rules.py` | one guarantee each; adding a rule is adding a class |
| `RuleEngine` | `router/rules.py` | runs rules in order, first match wins |
| `PdfChunker` | `rag/ingest.py` | PDF → structure-aware `Chunk` objects |
| `Index` | `rag/index.py` | owns the dense + lexical indexes: build, load, embed |
| `Retriever` | `rag/retrieve.py` | dense + BM25 + RRF fusion → `Hit` objects |
| `RagPath` | `rag/answer.py` | the RAG branch: retrieve → ground → cite |
| `SchemaCatalog` | `sql/schema.py` | both schema tiers + drift detection |
| `GuardRule` (ABC) → `SelectOnlyRule`, `NoWriteNodeRule`, `KnownTablesRule`, `AllowedFunctionsRule` | `sql/guard.py` | one AST check each |
| `SqlGuard` | `sql/guard.py` | runs guard rules, injects `LIMIT` |
| `SqlGenerator` / `QueryExecutor` / `ResultRenderer` | `sql/` | generate candidates / run read-only / render or narrate |
| `SqlPath` | `sql/pipeline.py` | the SQL branch, including the one-shot repair loop |
| `NumberVerifier` | `hybrid/verify.py` | the verbatim-number check, shared by narrator and composer |
| `Composer` | `hybrid/composer.py` | merges numbers + cited docs under source separation |
| `HybridPath` | `hybrid/pipeline.py` | composes `SqlPath` + `Retriever` + `Composer` |
| `Clarifier` | `ask/clarify.py` | ASK branch, with a deterministic fallback |
| `Reframer` | `ask/reframe.py` | after an ASK, decides if the next message answers it (merge) or pivots (route fresh) |
| `HistoryReframer` | `ask/history_reframe.py` | when NOT following an ASK, resolves a follow-up against the last 1-2 answered turns - mutually exclusive with `Reframer` |
| `UserProfile` / `UserStore` | `core/auth.py` | bcrypt login + per-user region/segment ACL, threaded into every turn |
| `GTMCopilot` | `orchestrator.py` | assembles everything; `answer(question, user) -> Answer` |

Two inheritance hierarchies carry real weight — `LLMClient` (swapping the provider is one subclass) and the two rule families (`RoutingRule`, `GuardRule`), where each guarantee is an independently testable class. Elsewhere composition is used instead: `GTMCopilot` *holds* the paths, and `HybridPath` *holds* a `SqlPath`, so behaviour cannot drift between routes.

Each module also exposes a thin module-level function wrapper (`guard()`, `retrieve()`, `answer_question()`) over its class, so scripts and tests can call a single step without assembling the whole object graph.

Companion docs: `DESIGN.md` (2-page write-up), `MEASUREMENTS.md` (generated by `python -m scripts.demo --md MEASUREMENTS.md`).

---

> **Sibling project.** `../gtm_copilot_gemini` is the same system running on the Gemini Flash API instead of Ollama. The only code difference is one `LLMClient` subclass.

## Prerequisites

- **Python 3.11+**
- **Ollama** installed and running (`ollama serve`)
- Two models pulled:

```bash
ollama pull llama3.2:3b
ollama pull qwen2.5:7b
```

`llama3.2:3b` routes (small, fast — it gates latency on every turn); `qwen2.5:7b` answers, writes SQL, composes and clarifies. Both names live in `core/config.py` and can be overridden without editing code:

```bash
export GTM_ROUTER_MODEL=llama3.2:3b
export GTM_ANSWER_MODEL=qwen2.5-coder:7b
```

> **Note on the numbers in this README.** The shipped default is `qwen2.5:7b`. The measured run in `MEASUREMENTS.md` used `qwen2.5-coder:7b` via the env override, because that 4.7 GB model was already present on the build machine and the network there could not pull another one in reasonable time. Routing, guard, SQL and trace behaviour are model-independent; a general-instruct model (`qwen2.5:7b`) follows the RAG prose rules — contradiction flagging in particular — noticeably better than the code-tuned variant.

## Setup

```bash
cd gtm_copilot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The provided assets ship with the repo — confirm they are present:

```
assets/Product_XYZ_Enablement_Pack.pdf
assets/Opportunity_Tracker_FieldGuide_v2.pdf
assets/gtm_mock.db
assets/users.yaml
```

Only `storage/` is gitignored. If an asset is missing at runtime the app fails with the exact path where it should go.

Build the RAG index (idempotent; `--force` rebuilds):

```bash
python -m scripts.ingest
```

### Login & access control

The app sits behind a login form (`core/auth.py`) — there is no anonymous
access. `assets/users.yaml` holds one entry per user: a bcrypt password hash
and an optional per-user **region/segment ACL** enforced on every SQL/HYBRID
query.

```yaml
users:
  priyanka:
    password_hash: "$2b$12$..."   # bcrypt.hashpw(b"password", bcrypt.gensalt())
    allowed_regions: ["NA"]        # or ["all"] / [] for unrestricted
    allowed_segments: ["all"]
```

Generate a hash:

```bash
python -c "import bcrypt; print(bcrypt.hashpw(b'yourpassword', bcrypt.gensalt()).decode())"
```

An unrestricted user must have both `allowed_regions` and `allowed_segments`
set to `["all"]` (or omitted). The ACL is enforced twice — once on the slot
the router extracts (`router/rules.py::RegionScopeRule`), once directly on the
generated SQL's AST (`sql/guard.py::ScopedQueryGuardRule`) as a backstop for
tables with no region/segment column of their own. See `ARCHITECTURE.md` §16.5
for the full two-layer design.

## Run

```bash
streamlit run app.py
```

## Test

```bash
python -m tests.run_all
```

The runner prints an environment banner first (DB present? index built? Ollama reachable?), because roughly a third of the suite is conditionally skipped and a heavily-skipped run otherwise looks identical to a clean one.

## Evaluate RAG/HYBRID quality

`tests/` proves the pipeline is *correct* (routes fire right, the guard
blocks writes, numbers are traceable). It doesn't say whether retrieval found
the *right* context or whether the answer is actually grounded in it — that's
what `eval/` measures, with RAGAS-style metrics implemented natively (no
`ragas` package, since it hard-depends on LangChain):

```bash
python -m scripts.eval               # print scores for the labelled question set
python -m scripts.eval --md EVALUATION.md   # also write a report
```

| Metric | Needs an LLM judge? | What it catches |
|---|---|---|
| Context Recall | No (pure function) | did retrieval find the labelled-relevant chunk at all |
| Context Precision | No (pure function) | did it rank early, or get buried past the top-k cutoff |
| Faithfulness | Yes | does every claim in the answer trace back to retrieved context |
| Answer Relevancy | Yes | does the answer actually address the question asked |

The LLM-judge metrics run on `ANSWER_MODEL` and only during an eval pass —
never on a live user turn. `eval/dataset.py` holds the labelled question set;
grow it the same way `tests/test_router_golden.py`'s golden set is meant to
grow.

---

## Four demo prompts, one per route

| # | Prompt | Expected behaviour |
|---|---|---|
| 1 · **RAG** | *"What deployment modes does Product XYZ support?"* | Routes RAG. Retrieves from the Enablement Pack and **surfaces the planted contradiction**: the legacy note says Cloud-only, the v3.0 note says Cloud/On-Prem/Hybrid. Both positions, both citations, conflict flagged — never silently resolved. |
| 2 · **SQL** | *"How many opportunities were Closed Won in EMEA in 2024?"* | Routes SQL (both required slots present). Generates a SELECT, passes the AST guard, runs read-only, returns **28** as a table with **no narrator LLM call** (≤20 rows, factual question). |
| 3 · **Hybrid** | *"What is our 2024 win rate for Enterprise, and what does the field guide require before a deal reaches Commit?"* | Routes HYBRID. Computes the win rate from SQL, retrieves the Field Guide's stage-5 (Commit) exit criteria conditioned on that result, composes *Finding → What the documentation says → Caveats*, then verifies every number appears verbatim in the result set. |
| 4 · **Ask** | *"How's pipeline looking recently?"* | Rule **R2_VAGUE_TIME** fires ("recently" is not a time range). Returns 1–3 clarifying questions, each with a concrete default. The trace shows the LLM's proposed route *and* the override. |

Bonus — safety: *"Delete all Closed Lost opportunities"* trips **R0_WRITE_INTENT** and is refused before any SQL is generated.

---

## Design decisions

**Routed pipeline over ReAct.** A ReAct loop on a local 7B costs 3–6 LLM calls per turn (blowing the <10s target), is non-deterministic turn to turn, and buys freedom this problem does not need — the router already decomposes the question. A fixed pipeline is inspectable, cheap, reproducible, and every branch is unit-testable in isolation. The cost is stated in *Known limitations*: hybrid cannot iterate SQL ↔ docs.

**"Retrieval proposes, the rule engine disposes."** The LLM router is good at intent and bad at guarantees. `router/rules.py` holds the guarantees: write intent is refused regardless of what the model proposed, a quantitative route with a missing slot becomes ASK, a vague quantifier without a time range becomes ASK, and low confidence becomes ASK. Rules fire in a fixed order, first match wins, and the firing rule ID is written into the trace — so every route is explainable after the fact, not just plausible.

**Slots exist because a confident wrong number is worse than a question.** On this dataset "how's pipeline?" differs by an order of magnitude between 2023 and 2024, and between NA and LATAM. Guessing produces a number the user will act on. `router/slots.py` also re-checks the raw question text, so a router that routes correctly but forgets to copy an obvious filter does not cause a pointless clarification.

**Allowlist-AST guard over regex denylist.** A denylist loses to `DR/**/OP`, case tricks, unicode, and statements nested inside CTEs. `sql/guard.py` parses with `sqlglot` and allows only: one statement, SELECT at the root, tables that exist in the **live introspected** schema, and functions on an explicit allowlist. It injects `LIMIT 200` when absent. Underneath it, `sql/execute.py` opens the DB with `file:...?mode=ro` — so even a guard bypass cannot write. Every rejection case is unit-tested without an LLM or a DB.

**Two-tier schema knowledge.** Live introspection is machine truth — it is what the guard allowlists against, *and* what the prompt's factual sections are built from: `build_schema_card()` generates TABLES, ENUM VALUES and JOIN KEYS from `PRAGMA table_info`, `SELECT DISTINCT` and `PRAGMA foreign_key_list`, so the columns, enum strings and join graph the model is shown are read from the database rather than transcribed by hand. Only business truth stays curated (what "won" means, that `discount_pct` is a fraction, that the Field Guide's 6-stage playbook names **do not exist** in the DB's 7-value stage enum). That makes structural drift impossible for the generated half; `validate_card()` still checks the hand-written half, whose prose names real columns (`deployments.seats_active`, …) and would otherwise go stale silently on a rename.

**RRF hybrid retrieval.** Dense and lexical retrieval fail differently: bge-small finds paraphrases ("can it run offline?" → "air-gapped"), BM25 finds exact tokens the embedder blurs (`XYZ-ANALYTICS`, `deployment_risk_score`). Reciprocal Rank Fusion needs only each list's *rank*, not its score, so no normalisation constant has to be re-fit when the corpus changes. `k=60` is the value from Cormack et al. (2009); it damps each list's head so one index cannot dominate. Ties break on `chunk_id`, making the fused order byte-stable.

**Structure-aware chunking.** Both PDFs put their highest-value content in tables (pricing tiers, the stage playbook with SLA days, the risk rubric). A data row separated from its header is unusable — "21" means nothing without "SLA (days)" — so table chunks repeat the header row with every group of data rows, and that invariant is a test. Section chunks carry their heading into the embedded text and never split mid-sentence.

**The verbatim-number rule.** The single most damaging failure for an analytics assistant is a fluent, well-cited answer containing a number nobody can trace. `hybrid/verify.py` extracts every numeric token from generated text and requires each to be traceable to the SQL result set, the question, or the executed SQL — with a documented normalisation policy (commas, `$`, `%` as either 12 or 0.12, downward-precision rounding only). On violation: one regeneration naming the offending numbers; on a second violation, generation is abandoned and the verified table + cited chunks are rendered instead. Same check runs on the SQL narrator and the hybrid composer.

**Narrate only when it earns its cost.** `sql/narrate.py` renders results with **zero** LLM calls when the result is ≤20 rows and the question is factual. Narration is invoked only for large result sets or interpretive questions ("why", "what's driving"). This is the biggest latency win in the system: the most common SQL turn costs one generation call instead of two.

**Model tiering.** Routing is a cheap classification on every single turn, so it runs on the 3B; answering, SQL generation and composition run on the 7B. Splitting them keeps p50 latency down without giving up answer quality. Both are `temperature=0`, `seed=42`, `top_k=1`.

**Safety by construction.** No `eval`/`exec` anywhere; the only execution surface is guarded SQL on a read-only connection. Retrieved PDF text is treated as untrusted data (the Field Guide's own prompt-injection guardrail), and `core/safety.py` redacts any response containing a long verbatim slice of a registered prompt template. A configurable column denylist (empty by default — the synthetic DB has no PII) strips sensitive fields from rendered results.

---

## Determinism

| Knob | Setting |
|---|---|
| LLM sampling | `temperature=0`, `seed=42`, `top_k=1`, `top_p=1.0` |
| Embeddings | `BAAI/bge-small-en-v1.5` pinned, `normalize_embeddings=True`, fixed BGE asymmetric prefixes |
| Chunk IDs | `sha1(doc + page + char_offset)[:12]` — identical across re-ingests |
| Fusion | RRF is a pure function; ties break on `chunk_id` |
| Retrieval | fixed `top-10 / top-10 → top-7`, cosine space set explicitly on the Chroma collection |
| Dependencies | every version pinned in `requirements.txt` |
| Synthetic data | the provided `gtm_mock.db` is used **as-is**; any future data augmentation would use seed 42, satisfying the synthetic-data seed NFR |

Local LLMs are not bit-deterministic across hardware even at temperature 0, so identical *routes, SQL, retrieved chunk IDs and trace structure* are the guarantee — not identical prose.

---

## Observability

Every turn appends one JSON line to `storage/traces.jsonl` and renders the same object in a collapsible panel:

```jsonc
{
  "turn_id": "a3f19c2b81d0",
  "question": "How's pipeline looking recently?",
  "raw_question": "How's pipeline looking recently?",  // the user's literal input,
                                                        // never overwritten even if
                                                        // either reframe path rewrites `question`
  "llm_proposed_route": "SQL",       // what the model wanted
  "rule_override": "R2_VAGUE_TIME",  // what the rule engine did
  "final_route": "ASK",
  "history_reframe_applied": null,   // set only when NOT following an ASK and
                                      // recent_turns was available - see HistoryReframer
  "slots": {}, "missing_slots": ["time_range", "segment_or_region"],
  "retrieved_chunks": [{"chunk_id": "...", "doc": "...", "page": 3, "rrf_score": 0.0325}],
  "generated_sql": null, "guard_verdict": null, "rows_returned": null,
  "repair_attempted": false, "number_check_passed": null,
  "per_stage_latency_ms": {"route": 812, "ask_clarify": 1940},
  "total_latency_ms": 2761
}
```

---

## Known limitations

- **Local-model latency misses the <10s target on this machine, and I'm not going to round it down.** Two warm runs of `python -m scripts.demo`, same laptop, different background load:

  | Route | Light load | Loaded machine (committed `MEASUREMENTS.md`) |
  |---|---:|---:|
  | REFUSE | 1.6 s | 6.4 s |
  | ASK | 4.1 s | 13.0 s |
  | SQL | 6.7 s | 18.4 s |
  | RAG | 20.3 s | 20.1 s |
  | HYBRID | 24.1 s | 36.6 s |
  | **inside 10 s** | **3/5** | **1/5** |

  The spread is the headline finding: the same code is 3× slower when the machine is busy, because a local 7B competes for the same RAM and GPU as everything else. Routing alone moves from 1.2 s to 6.7 s. Two other costs are visible in the per-stage numbers: the first RAG turn pays ~10 s to load the sentence-transformers model (lazy, once per process), and HYBRID's composer is a single ~16 s generation call.

  The mitigations already in the design — 3B router, table-only rendering for small factual results, the `num_predict` cap, one-retry caps everywhere, and an explicit `num_ctx=4096`/`keep_alive=30m` on every Ollama call (`core/config.py` — the model default of 32k-131k tokens of context was pure waste for prompts a few thousand tokens long, and inflated both load time and memory pressure) — are what buy the best case. Getting the rest of the way needs either more machine or a hosted model: the sibling `gtm_copilot_gemini` build puts every route inside the target because Flash answers in 1-3 s, at the cost of sending prompts off the laptop.
- **Single-shot routing.** One classification per turn, no re-route if the chosen path turns out to be wrong. A question that is 80% docs and 20% numbers gets whichever the router picked.
- **Multi-turn context is bounded and best-effort.** `ask/history_reframe.py`'s `HistoryReframer` resolves follow-ups like "and for EMEA?" or a bare stage reference ("what about commit") against the last 1-2 answered turns — but only when it's confident: a cheap heuristic skips the LLM call entirely for self-contained questions, and guards against self-contradiction, fabricated entities, dropped reply content, and stale-entity retention mean an ambiguous follow-up falls back to being routed as typed (worst case, an ASK) rather than risk a wrong merge. It's mutually exclusive with the ASK-continuation `Reframer` (`ask/reframe.py`) — a turn following an ASK never also consults `HistoryReframer`, and vice versa.
- **Fixed hybrid pipeline.** SQL runs once, then docs run once. It cannot notice that the SQL result makes a *different* document question interesting, or refine the query based on what the docs said. That is the deliberate trade for determinism and a bounded call count.
- **Schema-card drift.** The factual sections are generated from the live DB, so tables, columns, enum strings and join keys cannot drift. The business definitions remain hand-written: `validate_card()` flags prose that names a column which no longer exists, but it cannot catch a *semantic* drift — e.g. if "Closed Won" were renamed to "Won", the generated enum list would update while the card's hand-written definition of win rate silently became wrong.
- **Retrieval is corpus-shaped.** Chunking was tuned for these two short, heavily-tabular PDFs. A 200-page contract corpus would want a re-ranker and larger chunks.
- **No row-level security.** The Field Guide describes ACL checks and row-level security; this prototype has a column denylist only, since the synthetic DB has no per-user ownership model.
- **Number check is conservative, not exhaustive.** It proves every stated number is *traceable*; it cannot prove the number was used in the right sentence.
