# GTM Analyst Copilot — Design Write-up

*Two-page companion to the README. Covers architecture, the decisions that were genuinely contested, and what this design cannot do.*

---

## 1. Problem shape

Three question types share one chat box:

| Type | Example | Ground truth lives in |
|---|---|---|
| Documentary | "What deployment modes are supported?" | two PDFs |
| Quantitative | "How many Closed Won in EMEA in 2024?" | `gtm_mock.db` |
| Both | "2024 Enterprise win rate, and what gates a deal before Commit?" | both |

They fail differently. A wrong doc answer is a wrong sentence someone can check. A wrong *number* is acted on. So the design spends most of its safety budget on the numeric path, and most of its honesty budget on knowing when it does not have enough information to compute anything.

## 2. Architecture in one line

**A routed pipeline: one classification, one deterministic path, one trace.**

```
question → LLM router (3B) → rule engine (deterministic override) → RAG | SQL | HYBRID | ASK | REFUSE → safety filter → answer + trace
```

The router *proposes*; `router/rules.py` *disposes*. Nine rules, fixed order, first match wins, firing rule ID recorded in the trace:

| Rule | Fires when | Result |
|---|---|---|
| `R00_PII_REQUEST` | question asks for personal PII | **REFUSE** |
| `R0B_OFF_DOMAIN` | router itself classified the question as out of scope | **REFUSE** |
| `R0_WRITE_INTENT` | delete/update/insert/drop/… in the question | **REFUSE** (no SQL is ever generated) |
| `R0C_HYBRID_NO_SQL` | HYBRID proposed but `sql_subquestion` is empty - nothing to compute | downgrade to **RAG** |
| `R0D_INVALID_POPULATION_VALUE` | `segment_or_region` filled with something that isn't a real region/segment (a router hallucination, e.g. a stage name) | cleared, so R1 asks for it properly |
| `R1_MISSING_SLOT` | SQL/HYBRID with no time range or no region/segment | **ASK** |
| `R1B_REGION_SCOPE` | SQL/HYBRID names a region/segment outside the logged-in user's ACL | **REFUSE** ("all" narrows in place instead) |
| `R2_VAGUE_TIME` | "recently", "lately", "top", "best", "how's" with no explicit period | **ASK** |
| `R3_LOW_CONFIDENCE` | router confidence < 0.60 | **ASK** |

This split is the core bet: **use the LLM for intent, never for guarantees.** An LLM that is right 90% of the time is a good router and an unacceptable safety control.

## 3. Decisions that were actually contested

**Routed pipeline vs. ReAct agent.** ReAct would let the system iterate SQL ↔ docs. It also costs 3–6 local LLM calls per turn (the target is <10s on a laptop), varies turn to turn, and needs its tool loop mocked to be testable. The router already decomposes hybrid questions into a `sql_subquestion` and a `doc_subquestion`, which is the only decomposition this problem needs. Chose the fixed pipeline; the lost capability is listed as a limitation rather than hidden.

**AST allowlist vs. regex denylist for SQL safety.** A denylist is one obfuscation away from failing (`DR/**/OP`, casing, a `DELETE` nested in a CTE). `sql/guard.py` parses with `sqlglot` and permits only: one statement · `SELECT` at the root · tables present in the **live introspected** schema · functions on an explicit allowlist. `LIMIT 200` is injected when absent. Beneath it, `sql/execute.py` connects with `file:…?mode=ro`, so a guard bug still cannot write. Two independent layers, both tested — the read-only property is asserted by attempting a real `UPDATE` against the real connection.

**Ask vs. assume.** The tempting default is "assume last 12 months and answer". On this dataset that changes the answer by an order of magnitude between 2023 and 2024, and between NA and LATAM. A confident wrong number is the worst output this system can produce, so under-specified quantitative questions route to ASK — with 1–3 questions that each carry a concrete default, so the user replies with one word.

**Hybrid retrieval (RRF) vs. dense-only.** Dense finds paraphrases ("run offline" → "air-gapped"); BM25 finds the exact tokens embeddings blur (`XYZ-ANALYTICS`, `deployment_risk_score`) — and this corpus is dense with such tokens. Reciprocal Rank Fusion (k=60) needs only ranks, so no score-normalisation constant needs re-fitting when the corpus changes, and it is a pure function that unit-tests without an index.

**Verbatim numbers vs. fluency.** Every number in generated text must be traceable to the SQL result set, the question, or the executed SQL (`hybrid/verify.py`, with a documented policy for commas, `$`, `%` and rounding). On violation: one regeneration naming the offending tokens; on a second, generation is abandoned and the verified table plus cited chunks are rendered instead. A slightly clunky answer assembled from checked parts beats a fluent one containing an untraceable figure.

**Table-only rendering.** If a result is ≤20 rows and the question is factual, the answer is the table — **zero** LLM calls after generation. Narration is reserved for large results and interpretive questions. This is the largest latency win available and it costs nothing in quality.

**Surfacing contradictions.** The Enablement Pack deliberately contradicts itself (Cloud-only vs. Cloud/On-Prem/Hybrid; ANALYTICS in Starter vs. not; CRM writeback). The RAG prompt requires both positions with both citations and an explicit conflict flag. Silently picking the newer statement would look better and be less useful — a GTM rep needs to know the deck they are quoting is stale.

## 3a. Code shape

Data is modelled with dataclasses and pydantic (`Trace`, `GuardResult`, `QueryResult`, `Chunk`, `Hit`, `RouterDecision`, `Answer`); behaviour is modelled with classes that take their collaborators in the constructor (`Router`, `RuleEngine`, `Retriever`, `SqlGuard`, `SqlPath`, `HybridPath`, `Composer`, `Reframer`, `HistoryReframer`, `GTMCopilot`).

Inheritance is used exactly where it pays for itself:

- **`LLMClient` (ABC) → `OllamaClient` / `GeminiClient`.** The base class owns determinism settings, the JSON repair loop and preflight; a provider supplies one `_complete()` method. Switching the entire system from a local model to a hosted one is a single subclass — which is why the sibling `gtm_copilot_gemini` project is a near-identical tree.
- **`RoutingRule` (ABC) and `GuardRule` (ABC).** Each safety guarantee is its own class with a stable id, run in order by `RuleEngine` / `SqlGuard`. Adding a guarantee is adding a class, and each one is unit-tested on its own rather than through a growing `if/elif` chain.

Everywhere else the answer is composition, not a class hierarchy: `GTMCopilot` holds the four paths, and `HybridPath` holds a `SqlPath` and a `Retriever` rather than re-implementing them, so SQL behaviour cannot drift between the SQL and Hybrid routes. Thin module-level wrappers (`guard()`, `retrieve()`, `check_numbers()`) sit over the classes so a script or test can exercise one step without building the object graph.

## 4. Observability

One `Trace` per turn, appended to `storage/traces.jsonl` and shown in a collapsible panel: the raw question as typed (`raw_question`, never overwritten) alongside the effective/routed question (`question`, which a reframe path may rewrite), proposed route, rule override + rule ID, final route, whether `HistoryReframer` applied a rewrite, slots, retrieved chunk IDs with RRF scores, generated SQL, guard verdict, rows returned, repair attempted, number-check result, per-stage and total latency. The **proposed vs. final** pair is the important one: it makes disagreements between the model and the rules visible instead of invisible.

## 5. Determinism

`temperature=0`, `seed=42`, `top_k=1` on every call · pinned `bge-small-en-v1.5` with `normalize_embeddings=True` and fixed BGE asymmetric prefixes · chunk IDs = `sha1(doc+page+offset)[:12]` · RRF pure with `chunk_id` tie-break · all versions pinned. Local LLMs are not bit-reproducible across hardware even at temperature 0, so the guarantee is **identical routes, SQL, chunk IDs and trace structure** — not identical prose.

## 6. What this design cannot do

- **Iterate.** Hybrid runs SQL once, then docs once. It cannot refine the query based on what the documents said.
- **Follow up with full confidence.** `ask/history_reframe.py`'s `HistoryReframer` resolves a bounded class of follow-ups - "and for EMEA?", a bare stage reference ("what about commit") - against the last 1-2 answered turns, so these no longer route to a needless ASK. But it's guarded, not omniscient: a heuristic gate skips the LLM call for questions that already look self-contained, and several checks (self-contradiction, fabricated entities, dropped content, stale-entity retention) reject an ambiguous or wrong merge rather than risk one - falling back to routing the question as typed. It also only ever sees SQL/RAG/HYBRID turns (ASK/REFUSE carry no answer worth referencing), and is mutually exclusive with the ASK-continuation `Reframer` in `ask/reframe.py`.
- **Beat its own latency floor.** Local 7B generation dominates every turn that needs it, and it competes with whatever else the laptop is doing: measured warm runs ranged from 3/5 routes inside the 10 s target on an idle machine to 1/5 on a busy one. `MEASUREMENTS.md` has the per-stage breakdown; the numbers are reported rather than rounded down. The Gemini build is the escape hatch.
- **Catch semantic schema drift.** The card's factual sections are generated from live introspection, so structural drift cannot happen; `validate_card()` additionally flags hand-written prose naming a column that no longer exists. Neither catches a changed *meaning* (e.g. if `'Closed Won'` were renamed, the generated enum list would follow but the hand-written win-rate definition would silently be wrong).
- **Enforce row-level security.** The Field Guide describes ACLs and RLS; this prototype has a column denylist only, because the synthetic data has no per-user ownership model.
- **Prove a number is used in the right sentence.** The number check proves traceability, not relevance.
