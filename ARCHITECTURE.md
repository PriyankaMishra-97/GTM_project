# ARCHITECTURE.md — build-from-scratch reference

`README.md` says what the system does and how to run it. `DESIGN.md` says *why*
the contested decisions went the way they did. **This document says how to
build every file, in what order, precisely enough that someone with no access
to this repo could reproduce it.** It also lists every discrepancy found
between a docstring/comment and the actual verified runtime behavior, so a
rebuild can decide deliberately whether to keep the quirk or fix it.

Everything below was verified against the code as it exists today (module
docstrings quoted only where they match reality; every mismatch is called out
explicitly in §17).

---

## 0. How to use this document

Read top to bottom once to understand the shape, then use it as a per-file
checklist while implementing: §5–§14 map 1:1 onto the package tree, in the
order you should build them (each package only depends on packages above it).
§16 pulls the five algorithms worth extra care into one place. §17 is a punch
list of known gaps.

---

## 1. Repository layout

```
gtm_copilot/
├── app.py                     # Streamlit UI — presentation only
├── orchestrator.py             # GTMCopilot: the one class the UI calls
├── requirements.txt            # every version pinned, no LangChain/LlamaIndex
├── README.md  DESIGN.md  ARCHITECTURE.md  MEASUREMENTS.md  EVALUATION.md
│
├── core/                       # cross-cutting: config, LLM transport, trace, safety, auth
│   ├── config.py                  every constant, env-overridable
│   ├── llm_client.py               LLMClient (ABC) + OllamaClient
│   ├── trace.py                    Trace — one per turn, persisted to storage/traces.jsonl
│   ├── safety.py                   prompt-leak redaction, column denylist
│   └── auth.py                     UserProfile, UserStore (bcrypt login + region/segment ACL)
│
├── router/                     # LLM proposes a route, rules decide the guarantee
│   ├── llm_router.py                Router, RouterDecision
│   ├── rules.py                     RoutingRule (ABC) + 9 concrete rules, RuleEngine
│   ├── slots.py                     slot definitions + raw-text fallback checks
│   └── prompts.py                   ROUTER_SYSTEM/ROUTER_USER
│
├── rag/                        # documentary path
│   ├── ingest.py                    PdfChunker — structure-aware chunking
│   ├── index.py                     Index — Chroma (dense) + BM25 (lexical)
│   ├── retrieve.py                  Retriever — RRF fusion
│   ├── answer.py                    RagPath — retrieve → ground → cite → flag conflicts
│   └── prompts.py                   RAG_SYSTEM/RAG_USER/CHUNK_TEMPLATE
│
├── sql/                         # quantitative path
│   ├── schema.py                    SchemaCatalog — two-tier schema knowledge
│   ├── guard.py                     GuardRule (ABC) + 6 rules, SqlGuard (AST allowlist)
│   ├── generate.py                  SqlGenerator
│   ├── execute.py                   QueryExecutor — read-only connection
│   ├── narrate.py                   ResultRenderer — table-only vs. narrated
│   ├── pipeline.py                  SqlPath — generate → guard → execute → (repair×1) → render
│   └── prompts.py                   SQL_SYSTEM/SQL_USER/SQL_REPAIR_USER
│
├── hybrid/                      # both at once
│   ├── pipeline.py                  HybridPath — SQL → conditioned retrieval → compose
│   ├── composer.py                  Composer — Finding/Docs/Caveats + verify + fallback
│   ├── verify.py                    NumberVerifier — shared with sql/narrate.py
│   └── prompts.py                   COMPOSER_SYSTEM/COMPOSER_USER/COMPOSER_REPAIR
│
├── ask/                          # underspecified / follow-up questions
│   ├── clarify.py                    Clarifier — 1-3 questions + deterministic fallback
│   ├── reframe.py                    Reframer — continuation vs. pivot after an ASK
│   ├── history_reframe.py            HistoryReframer — follow-ups against the last 1-2 answered
│   │                                 turns, when NOT following an ASK (mutually exclusive with Reframer)
│   ├── text_overlap.py               shared merge-safety guards (content_words, preserves_content,
│   │                                 no_fabricated_entities) — used by both reframers
│   └── prompts.py                    CLARIFY_*/REFRAME_*/HISTORY_REFRAME_*
│
├── eval/                         # RAGAS-style offline quality evaluation (no LangChain)
│   ├── metrics.py                    context_recall/precision/relevance, faithfulness, answer_relevancy
│   ├── dataset.py                    EvalCase, hand-labelled RAG_EVAL_SET
│   └── prompts.py                    FAITHFULNESS_*/RELEVANCY_*
│
├── scripts/
│   ├── ingest.py                     builds the RAG index (idempotent)
│   ├── demo.py                       headless 5-prompt latency run → MEASUREMENTS.md
│   └── eval.py                       runs eval/dataset.py through the real stack → EVALUATION.md
│
├── assets/                       # everything the app reads, nothing it writes
│   ├── Product_XYZ_Enablement_Pack.pdf
│   ├── Opportunity_Tracker_FieldGuide_v2.pdf
│   ├── gtm_mock.db
│   └── users.yaml                    bcrypt credentials + per-user region/segment ACL
│
├── storage/                      # gitignored, created at runtime
│   ├── chroma/                       Chroma's on-disk collection
│   ├── bm25.pkl                      pickled BM25Okapi + payloads
│   └── traces.jsonl                  one JSON line per turn
│
└── tests/                        # python -m tests.run_all
```

---

## 2. Build order

Each layer only imports layers above it in this list — build top to bottom and
every layer is testable in isolation the moment it exists.

1. **`core/config.py`** — no dependencies. Every other file reads constants
   from here; get this right first.
2. **`core/llm_client.py`** — depends on `config` only. Build `LLMClient` (ABC)
   and `OllamaClient` together; this is the one class every LLM-calling module
   needs.
3. **`core/trace.py`**, **`core/safety.py`** — depend on `config` only.
4. **`sql/schema.py`** — depends on `config`. Needed early because
   `router/llm_router.py` and `router/prompts.py` both need a schema card.
5. **`core/auth.py`** — depends on `config` and `sql/schema.py` (for
   `REGIONS`/`SEGMENTS`).
6. **`router/`** (`slots.py` → `prompts.py` → `llm_router.py` → `rules.py`) —
   depends on `core.llm_client`, `core.config`, `sql.schema` (schema card),
   `core.auth` (rules need `UserProfile` for ACL checks).
7. **`rag/`** (`ingest.py` → `index.py` → `retrieve.py` → `prompts.py` →
   `answer.py`) — depends on `core.config`, `core.llm_client`, `core.safety`.
   Build and run `scripts/ingest.py` here so you have a live index to test
   against before continuing.
8. **`sql/`** (`guard.py` → `generate.py` → `execute.py` → `narrate.py` →
   `prompts.py` → `pipeline.py`) — depends on `sql.schema`, `hybrid.verify`
   (built next, or inline `NumberVerifier` temporarily — `sql/narrate.py` needs
   it too, so in practice build `hybrid/verify.py` **before** `sql/narrate.py`
   despite the package name suggesting otherwise).
9. **`hybrid/`** (`verify.py` → `prompts.py` → `composer.py` → `pipeline.py`)
   — depends on `sql.pipeline.SqlPath`, `rag.retrieve.Retriever`,
   `rag.answer.build_context`.
10. **`ask/`** (`text_overlap.py` → `prompts.py` → `clarify.py` → `reframe.py`
    → `history_reframe.py`) — depends on `router.slots.SLOT_QUESTIONS`,
    `router.slots.has_explicit_time`/`has_explicit_population`,
    `core.llm_client`. Build `text_overlap.py` first — both reframers import
    its guards.
11. **`orchestrator.py`** — depends on everything above. Wires one `Router`,
    one `RuleEngine`, one `RagPath`, one `Clarifier`, one `Reframer`, one
    `HistoryReframer`, and a **per-user** cache of `SqlPath`/`HybridPath`
    (see §9.4 — this is the one piece of shared state that must not be a
    singleton).
12. **`app.py`** — depends only on `orchestrator` and `core.auth`. Pure
    presentation; if you find yourself importing `router`/`rag`/`sql` here,
    something has leaked.
13. **`eval/`** (`prompts.py` → `metrics.py` → `dataset.py`) + `scripts/eval.py`
    — depends on `orchestrator.get_copilot()`, `rag.index.Index.embed_query`,
    `core.llm_client`. Build last; it evaluates everything above.
14. **`tests/`** — write alongside each layer above, not at the end. The
    existing suite's shape (one file per module, plus
    `test_pipeline_offline.py` for cross-layer integration with a
    `StubClient`) is the pattern to copy.

---

## 3. Configuration surface — `core/config.py`

One module, no classes, sets `ANONYMIZED_TELEMETRY=False` at import time
(must precede any `chromadb` import or its telemetry hook spams stderr).

| Constant | Default | Env override | Used by |
|---|---|---|---|
| `PROJECT_ROOT` | resolved from `__file__` | — | everything path-based (never depends on cwd) |
| `ASSETS_DIR` / `STORAGE_DIR` | `assets/` / `storage/` | — | |
| `DB_PATH` | `assets/gtm_mock.db` | — | `sql.schema`, `sql.execute` |
| `USERS_PATH` | `assets/users.yaml` | — | `core.auth.UserStore` |
| `PDF_PATHS` | `{"Enablement Pack": ..., "Field Guide": ...}` (dict, insertion order matters) | — | `rag.ingest` |
| `CHROMA_DIR` / `BM25_PATH` / `TRACES_PATH` | under `storage/` | — | `rag.index`, `core.trace` |
| `CHROMA_COLLECTION` | `"gtm_docs"` | — | `rag.index` |
| `OLLAMA_HOST` | `http://localhost:11434` | `OLLAMA_HOST` | `core.llm_client.OllamaClient` |
| `ROUTER_MODEL` | `llama3.2:3b` | `GTM_ROUTER_MODEL` | routing, reframe (cheap tier) |
| `ANSWER_MODEL` | `qwen2.5:7b` | `GTM_ANSWER_MODEL` | answering, SQL gen, compose, **clarify** (quality tier) |
| `LLM_SEED` / `LLM_TEMPERATURE` | `42` / `0.0` | — | every Ollama call |
| `LLM_TIMEOUT_S` / `LLM_MAX_TOKENS` | `180` / `1024` | — | `_complete` |
| `LLM_NUM_CTX` | `4096` | — | `_complete`'s `options.num_ctx`. Ollama otherwise defaults to the *model's own* context length (131072 for llama3.2:3b, 32768 for qwen2.5:7b per `ollama show`) — wildly oversized for this system's prompts (a few thousand tokens at most), inflating load time and, on a memory-constrained host, creating pressure that evicts the *other* tier's model between calls. |
| `LLM_KEEP_ALIVE` | `"30m"` | — | `_complete`'s top-level `keep_alive` field. Ollama's own default (5 min) unloads an idle model, so any gap between chat turns pays full reload cost again. |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | — | `rag.index` |
| `BGE_DOC_PREFIX` / `BGE_QUERY_PREFIX` | `"Represent this document for retrieval: "` / `"...query..."` | — | asymmetric embedding — **do not swap these** |
| `CHUNK_TARGET_TOKENS` / `TABLE_ROWS_PER_CHUNK` | `300` / `6` | — | `rag.ingest.PdfChunker` |
| `DENSE_TOP_K` / `BM25_TOP_K` | `10` / `10` | — | candidate pool per index before fusion |
| `RRF_TOP_K` | **`7`** | — | final fused hit count (docs/comments elsewhere say "top-5" — that's stale, see §17) |
| `RRF_K` | `60` | — | Cormack et al. (2009) damping constant |
| `SQL_ROW_LIMIT` | `200` | — | injected `LIMIT` when the model's SQL has none |
| `SQL_TIMEOUT_S` | `5` | — | connect timeout + instruction-count deadline proxy |
| `SQL_MAX_ROWS_RENDERED` | `200` | — | markdown table cap |
| `SQL_NARRATE_ROW_THRESHOLD` | `20` | — | `sql/narrate.py`'s skip-LLM heuristic |
| `SQL_ALLOWED_FUNCTIONS` | frozenset of ~24 names | — | `sql/guard.py`'s `AllowedFunctionsRule` |
| `COLUMN_DENYLIST` | `frozenset()` (empty) | — | `core.safety.strip_denied_columns` — wired but inert until populated |
| `PROMPT_LEAK_NGRAM_CHARS` | `60` | — | `core.safety.find_prompt_leak` window size |
| `ROUTER_MIN_CONFIDENCE` | `0.6` | — | `router.rules.LowConfidenceRule` |

**Note**: `OllamaClient._complete` hardcodes `top_k=1, top_p=1.0` directly in
`core/llm_client.py` rather than reading them from `config.py` — if you want
"every determinism knob lives in one file" to be literally true, add
`LLM_TOP_K`/`LLM_TOP_P` constants and reference them there.

`ensure_storage()` — `STORAGE_DIR.mkdir(parents=True, exist_ok=True)`, called
by `scripts/ingest.py` and `Trace.persist()`.

---

## 4. Data assets

### 4.1 `assets/gtm_mock.db` — schema & enums

A synthetic SQLite DB. Introspect it live rather than hand-copying a schema —
that live-introspection habit is the whole point of §7.1. For reference, the
business enums both `core/auth.py` and `sql/schema.py` hardcode as module
constants (the single source of truth for ACL parsing and enum-order
rendering):

```python
REGIONS  = frozenset({"NA", "EMEA", "APAC", "LATAM"})
SEGMENTS = frozenset({"Enterprise", "Mid-Market", "SMB"})
```

The DB's own `stage` enum (7 values) intentionally does **not** match the
Field Guide PDF's 6-stage playbook names — this mismatch is deliberate seed
data, documented in the hand-curated half of the schema card
(`SCHEMA_CARD_CAUTION`, see §7.1) so the SQL-generation LLM is warned rather
than silently guessing a mapping.

Tables carrying `region`/`segment` columns for ACL scoping:
`accounts`, `opportunities`. Two more tables (`deployments`, `activities`)
join to those but carry **no region/segment column of their own** — this gap
is exactly why `sql/guard.py`'s `ScopedQueryGuardRule` exists as a backstop
(see §16.5).

### 4.2 `assets/users.yaml` — login + ACL schema

```yaml
users:
  priyanka:
    password_hash: "$2b$12$....."      # bcrypt hash, e.g. bcrypt.hashpw(pw.encode(), bcrypt.gensalt())
    allowed_regions: ["NA"]             # or ["all"] for unrestricted, or [] / omitted for unrestricted
    allowed_segments: ["all"]
  darshan:
    password_hash: "$2b$12$....."
    allowed_regions: ["all"]
    allowed_segments: ["Enterprise"]
```

Parsing rules (`core/auth.py::_scope_set`): an empty/absent list, or any
element case-insensitively equal to `"all"`, means **unrestricted** on that
dimension (`None` internally). Otherwise it's a `frozenset` of the listed
values, case preserved. `UserProfile.allowed_scope_values()` unions the two
dimensions into one set for `ScopedQueryGuardRule`, materializing whichever
dimension is unrestricted into the *full* enum first — never both-unrestricted
being conflated with "used no ACL at all" (that case returns `None` and skips
the check entirely).

`assets/users.example.yaml` (sanitized, no real hashes) exists as the
template both the module docstring and the runtime error message point at.

To generate a password hash: `python -c "import bcrypt; print(bcrypt.hashpw(b'yourpassword', bcrypt.gensalt()).decode())"`.

### 4.3 The two source PDFs

`Product_XYZ_Enablement_Pack.pdf` (pricing tiers, SKUs, deployment modes) and
`Opportunity_Tracker_FieldGuide_v2.pdf` (stage playbook, risk-scoring rubric,
data dictionary). The Enablement Pack **deliberately contradicts itself**
(legacy "Cloud-only" note vs. v3.0 "Cloud/On-Prem/Hybrid" note) — this is
seeded on purpose to exercise the RAG contradiction-flagging rule (§7.4); do
not "fix" it when reproducing the corpus.

---

## 5. `core/` — cross-cutting layer

### 5.1 `core/llm_client.py`

```python
class LLMUnavailable(RuntimeError): ...   # provider unreachable / model missing
class LLMJSONError(RuntimeError): ...     # model failed schema validation twice

class LLMClient(ABC):
    provider: str = "abstract"
    def __init__(self, router_model=None, answer_model=None): ...
    @abstractmethod
    def _complete(self, model, system, messages, json_schema=None) -> str: ...
    @abstractmethod
    def available_models(self) -> list[str]: ...
    def preflight(self) -> None: ...        # raises LLMUnavailable if router/answer model missing
    def is_ready(self) -> bool: ...          # non-throwing wrapper
    def status(self) -> dict: ...            # never raises — sidebar-safe
    def chat(self, system, user, *, model=None, trace=None, stage=None) -> str: ...
    def chat_json(self, system, user, schema: type[BaseModel], *, model=None, trace=None, stage=None) -> BaseModel: ...

class OllamaClient(LLMClient):
    provider = "ollama"
    def __init__(self, host=None, router_model=None, answer_model=None): ...
    def available_models(self) -> list[str]: ...   # GET /api/tags
    def _complete(self, model, system, messages, json_schema=None) -> str: ...  # POST /api/chat

def extract_json(raw: str) -> str: ...   # salvage parser: strips code fences, finds outermost {...}
def get_client() -> LLMClient: ...        # process-wide OllamaClient singleton
def set_client(client) -> None: ...       # override for tests
```

**`preflight()` model matching**: builds `normalised = {m.split(":")[0] for m in have} | have`
so a request for `"llama3.2:3b"` matches an installed tag of that exact name,
*or* a bare `"llama3.2"` request matches any tag with that prefix. This is
what lets `GTM_ROUTER_MODEL` be overridden with or without a `:tag` suffix.

**`chat_json` — the JSON repair loop, exactly**: model defaults to
`self.answer_model` (callers needing the cheap router tier — `Router.decide`,
`Reframer.reframe`, `HistoryReframer.reframe` — pass `model=self.router_model`
explicitly; `Clarifier` does **not** override it, so clarification wording
always uses the 7B model even though its schema is trivial — a
deliberate-or-not asymmetry worth knowing). Builds `json_schema = schema.model_json_schema()` and sends it as
Ollama's `format` field (constrained decoding, not prompt instruction alone).
Loop over exactly 2 attempts: call `_complete`, try
`schema.model_validate_json(extract_json(raw))`. On `ValidationError |
ValueError | json.JSONDecodeError` and it's not the last attempt, append the
bad output as an assistant turn plus a user turn quoting the validation error
and asking for a fix, then retry once more. One repair retry total. Every
call — success or failure — logs one record to `trace.llm_calls` with stage,
model, system/user text, attempt count, and `ok`.

**`OllamaClient._complete`**: POSTs `{host}/api/chat`,
`stream=False`, top-level `keep_alive: "30m"`, `options={temperature: 0.0,
seed: 42, top_k: 1, top_p: 1.0, num_predict: 1024, num_ctx: 4096}`
(temperature/seed/num_predict/num_ctx/keep_alive from config; top_k/top_p are
literals in this file — see the §3 note), and `format=json_schema` when a
schema was passed. Timeout 180s. Any `requests.RequestException` becomes
`LLMUnavailable` with an "is `ollama serve` running?" hint plus the exact
`ollama pull <model>` command for any missing model.

### 5.2 `core/trace.py`

```python
@dataclass
class RetrievedChunk:
    chunk_id: str; doc: str; page: int; section: str
    rrf_score: float; text: str = ""
    dense_rank: int | None = None; bm25_rank: int | None = None

@dataclass
class Trace:
    question: str                          # overwritten with the reframed text on a follow-up turn
    turn_id: str = <uuid4hex[:12]>
    user: str | None = None
    raw_question: str | None = None        # set once in __post_init__ (defaults to `question`),
                                            # NEVER mutated again - the user's literal input survives
                                            # even after either reframe path rewrites `question`
    pending_question: str | None = None    # set only when this turn followed an ASK
    is_new_topic: bool | None = None       # Reframer's verdict
    history_reframe_applied: bool | None = None  # set only when this turn did NOT follow an ASK
                                            # and recent_turns was available - HistoryReframer's
                                            # verdict; mutually exclusive with the two fields above
    llm_proposed_route: str | None = None
    rule_override: str | None = None       # firing rule id, or None if the LLM's route stood
    final_route: str | None = None
    router_confidence: float | None = None
    router_rationale: str | None = None
    rule_detail: str = ""
    refusal_message: str | None = None
    slots: dict = {}
    missing_slots: list[str] = []
    doc_subquestion: str | None = None
    sql_subquestion: str | None = None
    retrieved_chunks: list[dict] = []
    generated_sql: str | None = None
    guard_verdict: str | None = None       # "PASS" or "REJECT: <reason>"
    rows_returned: int | None = None
    repair_attempted: bool = False
    number_check_passed: bool | None = None
    per_stage_latency_ms: dict[str, int] = {}
    total_latency_ms: int | None = None
    errors: list[str] = []
    llm_calls: list[dict] = []

    def __post_init__(self) -> None: ...   # if raw_question is None, set it to `question`
    def stage(self, name: str) -> ContextManager: ...   # times a block; warns (doesn't crash) on a duplicate name
    def add_chunks(self, chunks: list[RetrievedChunk]) -> None: ...   # full replace
    def add_llm_call(self, record: dict) -> None: ...                 # append-only
    def error(self, message: str) -> None: ...                        # append-only
    def finish(self) -> "Trace": ...      # sets total_latency_ms, returns self (chainable)
    def to_dict(self) -> dict: ...        # asdict() minus the internal _t0 timer field
    def persist(self) -> None: ...        # appends one JSON line to storage/traces.jsonl; catches OSError only
```

`persist()` uses `json.dumps(..., default=str)` so any stray non-serializable
field is stringified rather than crashing the turn — observability must never
be able to break a user-facing answer.

### 5.3 `core/safety.py`

```python
REDACTION = "[redacted: system prompt]"

def register_prompt(*templates: str) -> None: ...   # called at import time by every prompts.py
def find_prompt_leak(text: str) -> str | None: ...
def filter_response(text: str) -> str: ...
def strip_denied_columns(columns: list[str], rows: list[list]) -> tuple[list[str], list[list]]: ...
```

`find_prompt_leak` normalises whitespace + case, then for each registered
template slides an `N`-char (`PROMPT_LEAK_NGRAM_CHARS=60`) window across it
with **step `N//2`** (30 chars) — a sampled check, not exhaustive. This
reliably catches any leaked run of roughly ≥1.5×N chars (~90); a leak shorter
than that could in principle land entirely between two sampled windows. If you
want an exhaustive check, step by 1 instead — the tradeoff is speed on long
templates.

`filter_response` removes **entire lines** whose normalized form contains the
leaked window (line-level, not character-level, for readability), then
appends `REDACTION`.

Every `prompts.py` module in the codebase calls `safety.register_prompt(...)`
on its templates at module import time — do this in every new prompt module
you add, or leaked text from it won't be detected.

### 5.4 `core/auth.py`

```python
class UsersFileMissing(FileNotFoundError): ...

@dataclass(frozen=True)
class UserProfile:
    username: str
    allowed_regions: frozenset[str] | None   # None = unrestricted
    allowed_segments: frozenset[str] | None
    def allowed_scope_values(self) -> frozenset[str] | None: ...

class UserStore:
    def __init__(self, path: Path | None = None): ...   # loads ALL users eagerly, not lazily
    def authenticate(self, username: str, password: str) -> UserProfile | None: ...
```

`authenticate` returns `None` indistinguishably for "unknown username" and
"wrong password" — a deliberate anti-enumeration choice, quoted directly in
the module docstring and worth preserving in a rebuild. A malformed stored
hash raises `ValueError` inside `bcrypt.checkpw`, caught here and treated as a
failed login rather than a crash.

`allowed_scope_values()` — if **both** dimensions are `None`, returns `None`
(skip ACL entirely). Otherwise, whichever dimension is `None` gets
materialized to the *full* live enum (`REGIONS`/`SEGMENTS` from
`sql/schema.py`) before unioning with the other, restricted dimension — this
prevents a region-only-restricted user's segment check from silently becoming
"everything" when it should stay scoped by region alone (the union is over
values, and an unrestricted dimension must contribute its full domain to that
union, not be skipped).

---

## 6. `router/` — propose, then rule-engine-override

### 6.1 `router/llm_router.py`

```python
Route = Literal["RAG", "SQL", "HYBRID", "ASK", "OFF_TOPIC"]   # note: "REFUSE" is never LLM-proposed

class RouterDecision(BaseModel):
    route: Route
    missing_slots: list[str] = []
    confidence: float = 0.0                 # clamped to [0, 1] by a field_validator
    rationale: str = ""
    slots: dict[str, Any] = {}              # a before-validator drops None/"null"/"none"/""/"n/a" entries
    doc_subquestion: str | None = None
    sql_subquestion: str | None = None

class Router:
    def __init__(self, client=None, schema_card=None): ...
    def decide(self, question: str, trace=None) -> tuple[RouterDecision, str | None]: ...
```

`decide()` compacts the schema card via
`card.split("BUSINESS DEFINITIONS")[0].strip()` (the router only needs
table/column facts, not business prose — token savings on the 3B model), then
one `chat_json(..., model=self.client.router_model, stage="route")` call. On
`LLMJSONError | LLMUnavailable`, returns a **fixed** fallback —
`RouterDecision(route="ASK", missing_slots=["time_range","segment_or_region"],
confidence=0.0, rationale="router failed; defaulting to clarification")` —
regardless of what the question actually was.

### 6.2 `router/rules.py`

```python
@dataclass
class RoutingOutcome:
    final_route: str
    rule_id: str | None = None       # None = the LLM's proposal stood, unmodified
    missing_slots: list[str] = []
    refusal_message: str | None = None
    detail: str = ""

class RoutingRule(ABC):
    rule_id: str = "R?"
    @abstractmethod
    def evaluate(self, decision, question, user) -> RoutingOutcome | None: ...

class RuleEngine:
    DEFAULT_RULES = (PiiRequestRule, WriteIntentRule, OffDomainRule,
                      HybridWithoutSqlRule, InvalidPopulationValueRule,
                      MissingSlotRule, RegionScopeRule, VagueTimeRule, LowConfidenceRule)
    def apply(self, decision, question, user) -> RoutingOutcome: ...   # first match wins
```

Nine rules, fixed order, first match wins:

| id | fires when | outcome |
|---|---|---|
| `R00_PII_REQUEST` | regex hit on SSN/card-number shapes or phrases like "email address", "social security", "date of birth" (deliberately excludes bare "email"/"phone" — legitimate schema values like `channel='Email'`) | REFUSE |
| `R0_WRITE_INTENT` | regex hit on `delete/drop/truncate/update/insert/upsert/alter/overwrite/wipe/purge`, `remove the row/record`, `set col=`, `grant/revoke` | REFUSE, fires regardless of the LLM's proposed route |
| `R0B_OFF_DOMAIN` | `decision.route == "OFF_TOPIC"` (trusts the LLM's own classification, no independent check) | REFUSE — runs before `LowConfidenceRule` so a low-confidence OFF_TOPIC can't get downgraded to ASK instead |
| `R0C_HYBRID_NO_SQL` | route is HYBRID but `sql_subquestion` is unfilled — the router sometimes copies "HYBRID" from a near-identical few-shot example onto a question that's actually pure documentation | downgrade to RAG |
| `R0D_INVALID_POPULATION_VALUE` | route is SQL/HYBRID, `segment_or_region` is filled with a scalar (not `"all"`, not already a narrowed list) that isn't a real member of `REGIONS ∪ SEGMENTS` — a router hallucination, e.g. filling it with a stage name copied from an unrelated example | clears the slot in place (so `R1_MISSING_SLOT`, next, asks for it properly instead of `R1B_REGION_SCOPE` refusing over a value that was never a real region) |
| `R1_MISSING_SLOT` | route is SQL/HYBRID and `slots.missing_slots(route, question, decision.slots)` is non-empty | ASK |
| `R1B_REGION_SCOPE` | route is SQL/HYBRID, the `segment_or_region` slot is filled, and either it equals `"all"` (case-insensitive — **mutates `decision.slots` in place** to the user's actual allowed list) or it names a value outside `user.allowed_scope_values()` | narrows in place and passes through (no rule fires), or REFUSE naming the allowed scope |
| `R2_VAGUE_TIME` | a vague-time word (`"recently","lately","currently",...`) anywhere, or (SQL/HYBRID only) a superlative (`"top","best","how's",...`) with no explicit time pattern matched and the `time_range` slot unfilled | ASK for `time_range` (+ `segment_or_region` too, if that's also unfilled and unstated) |
| `R3_LOW_CONFIDENCE` | `decision.confidence < config.ROUTER_MIN_CONFIDENCE (0.6)` and route isn't already ASK | ASK |

If none fire, the LLM's proposed route stands (`rule_id=None`).

**Real failure mode `R0D` exists for**: for the follow-up "and in Discover?",
the router proposed HYBRID with `slots={"segment_or_region": "Discover"}` —
"Discover" is a Field Guide stage name, not a region/segment, but it
superficially resembled the shape of a near-identical few-shot example ("win
rate for Enterprise..."), and the router copied that example's structure with
"Discover" substituted into the wrong slot (`doc_subquestion` was even left
stale, still referencing "Commit" from the example). Left unchecked, this
reached `R1B_REGION_SCOPE` and refused with "You don't have access to
Discover" — confusing, since Discover was never a real region/segment to
begin with.

### 6.3 `router/slots.py`

A slot is a named required filter. Four are defined: `time_range`,
`segment_or_region`, `stage_definition`, `product_area` — but only three are
ever *required*; see below. This module does **not** extract slots (the
router LLM does, into `RouterDecision.slots`) — it *defines* what's required
per route and *re-validates* against the raw question text as a fallback:

```python
def required_slots(route: str, question: str) -> tuple[str, ...]: ...
    # SQL/HYBRID: (time_range, segment_or_region) — always, never conditional
    # RAG: (product_area,) only if is_ambiguous_doc_question, else ()
def missing_slots(route: str, question: str, slots: dict) -> list[str]: ...
    # a required slot counts as present if slots[name] is filled,
    # OR (for time_range) has_explicit_time(question) is true,
    # OR (for segment_or_region) has_explicit_population(question) is true
    # — this is the "raw-text fallback" that stops the system penalizing the
    # user for the LLM forgetting to copy an obvious filter into slots{}.
def has_explicit_time(question: str) -> bool: ...          # reused by ask/history_reframe.py too
def has_explicit_population(question: str) -> bool: ...    # reused by ask/history_reframe.py too
SLOT_QUESTIONS: dict[str, str]   # hand-written clarification text per slot, with real dataset defaults
```

**`stage_definition` is never asked for.** It isn't information only the user
has: the SQL side can only ever query `opportunities.stage`'s real enum (the
Field Guide playbook's 6-stage names don't exist in the database), and the
doc side only ever explains the playbook (the database enum has no exit
criteria of its own). Which one applies is determined by the *route*, not by
asking — `sql/generate.py::SqlGenerator.slot_block` forces `"database
stages"` into the SQL prompt for every SQL/HYBRID call regardless of what the
router extracted. (An earlier version of this module had an
`is_stage_question()`/`STAGE_MARKERS` helper that conditionally added
`stage_definition` to `required_slots()` — removed entirely once the SQL-side
forcing made asking for it redundant, and it was also a source of a
production bug: the router would fill `stage_definition` with the *stage
name itself* — e.g. `"negotiation"` — rather than one of the two real values
(`"database stages"`/`"playbook"`), and `is_filled()`'s shallow truthy check
didn't catch it, silently masking a slot that was never actually resolved.)

### 6.4 `router/prompts.py`

`ROUTER_SYSTEM` interpolates the compacted schema card and hardcodes a
one-paragraph summary of each PDF's contents (including a callout that the
Enablement Pack has legacy/v3.0 contradictions). Ten few-shot examples, two
per route (RAG/SQL/HYBRID/ASK/OFF_TOPIC) — if you copy this file's own inline
comment claiming "eight," fix the count while you're there.
`ROUTER_USER = 'Q: "{question}"\nA:'`.

---

## 7. `rag/` — documentary path

### 7.1 `rag/ingest.py` — structure-aware chunking

```python
@dataclass
class Chunk:
    chunk_id: str; doc: str; page: int; section: str; text: str; kind: str = "section"
    def metadata(self) -> dict: ...

def make_chunk_id(doc: str, page: int, char_offset: int) -> str:
    return sha1(f"{doc}|{page}|{char_offset}".encode()).hexdigest()[:12]

class PdfChunker:
    def __init__(self, target_tokens=None, table_rows_per_chunk=None): ...
    def chunk_corpus(self, pdf_paths=None) -> list[Chunk]: ...
```

**Algorithm, per page**:
1. Find tables first (`page.find_tables()`), record their bounding boxes.
2. Reconstruct each table's rows by **re-bucketing raw text spans** against
   the *header row's* column x-boundaries — not each data row's own reported
   cells, since a wrapped row reports fewer/misaligned cells than its header.
   A span joins whichever column's `[x0, x2]` range contains the span's left
   edge (nearest-column fallback otherwise); spans within a bucket are ordered
   `(y0, x0)` to survive multi-line wraps. Rows whose y-range falls entirely
   inside the previous emitted row's y-range are dropped as duplicate wrapped
   detections; rows with a blank first (key) column are also dropped
   (corpus-specific: column 1 is always the row's key).
3. Group data rows into chunks of `TABLE_ROWS_PER_CHUNK` (6); **prepend the
   header row's text to every group**, not just the first — `"21"` means
   nothing without `"SLA (days)"` next to it.
4. Run prose extraction on the remainder of the page (lines whose bbox doesn't
   intersect a table's bbox): detect headings (`size >= body_size + 1.2` OR
   mostly-bold + short + no trailing period), strip running headers/footers
   (a short line repeated on a majority of pages), start a new "section" at
   each heading, and inside each section greedily pack whole sentences into a
   chunk while `estimate_tokens(joined) <= 300` (never splitting mid-sentence).
   Every chunk's text is prefixed with its section heading.

### 7.2 `rag/index.py`

```python
class Index:
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...  # BGE_DOC_PREFIX + normalize
    def embed_query(self, text: str) -> list[float]: ...                    # BGE_QUERY_PREFIX + normalize
    def build(self, chunks: list[Chunk], force=False) -> dict: ...
    def load_bm25(self) -> BM25Store: ...
    def load_collection(self): ...    # Chroma collection
    def stats(self) -> dict: ...      # never raises
    def exists(self) -> bool: ...
```

**BGE asymmetric prefixes are load-bearing**: a document embedded with the
query prefix (or vice versa) is not comparable via cosine similarity to the
other side — always prefix documents at index time and queries at search
time, never swap them. Both embeddings are `normalize_embeddings=True`, so
cosine similarity reduces to a plain dot product everywhere downstream.

Chroma collection created with `metadata={"hnsw:space": "cosine"}` set
explicitly (a client-library default change must not silently alter ranking).
BM25 side is a self-contained pickle: `BM25Okapi` object + parallel
`chunk_ids` list + a `payloads` dict (`chunk_id -> {"text":..., **metadata}`)
— it never needs Chroma to render a hit, and vice versa.

### 7.3 `rag/retrieve.py` — RRF fusion

```python
def reciprocal_rank_fusion(ranked_lists: list[list[str]], k=60, top_k=None) -> list[tuple[str, float]]:
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
```

Pure function — no index dependency, unit-testable with hand-built lists.
Ties broken by ascending `chunk_id` for byte-stable output. `Retriever`:
dense top-10 + BM25 top-10 (`config.DENSE_TOP_K`/`BM25_TOP_K`) → fused and
truncated to **top-7** (`config.RRF_TOP_K` — several comments elsewhere say
"top-5"; 7 is the actual live value, chosen to avoid dropping a
BM25-invisible-but-dense-relevant table chunk). `k=60` is the Cormack et al.
(2009) damping constant.

### 7.4 `rag/answer.py`

```python
def detect_conflicts(hits: list[Hit]) -> list[Hit]:
    # a hit is flagged iff its lowercased text contains BOTH a stale marker
    # ("legacy","older runbook","previously","used to")
    # AND a current marker ("updated note","v3.0","as of v3.0",...)
    # — purely lexical, no LLM.

class RagPath:
    def answer_from_hits(self, question, hits, trace=None) -> str: ...
    def run(self, question, trace) -> tuple[str, list[Hit]]: ...
```

Zero-hit questions never reach the LLM (`NO_CONTEXT_MESSAGE` returned
immediately, naming which PDF likely has it). Every model response passes
through `safety.filter_response()` — retrieved PDF text is untrusted input and
could itself carry a prompt-injection attempt (`rag/prompts.py` rule 5 tells
the model to treat context as data, never instructions). Exactly one
retrieval + one LLM call per RAG turn; no re-ranker, no self-critique loop.

### 7.5 `rag/prompts.py`

Seven hard rules in `RAG_SYSTEM`. Rule 4 (contradiction-flagging): if two
chunks (or two statements in one chunk) disagree, the model must present both
positions with both citations and an explicit `**Conflict:**` callout — never
silently pick a side. Rule 7 (scope-fidelity, added after a real failure): if
the question asks about ONE specific item (a stage, product, SKU, tier, …)
and a retrieved chunk covers SEVERAL items in a table or list, answer ONLY
the item asked about — never restate the whole table just because it
happened to be in the chunk handed over. `CHUNK_TEMPLATE = "--- doc: {doc} |
page: {page} | section: {section} ---\n{text}\n"` is the exact header format
rule 2 tells the model to parse citations from (deliberately no "chunk N"
ordinal — models sometimes cited their position-in-list instead of the real
page number when one was present).

**Both rules 4 and 7 ship a worked example (`_CONTRADICTION_EXAMPLE`,
`_SCOPE_EXAMPLE`) that is f-string-interpolated into `RAG_SYSTEM` for the
model to see, but explicitly EXCLUDED from what gets passed to
`safety.register_prompt(...)`**
(`RAG_SYSTEM.replace(_CONTRADICTION_EXAMPLE, "").replace(_SCOPE_EXAMPLE, "")`).
Real failure this fixes: "What deployment modes does Product XYZ support?"
produced a textbook-correct contradiction answer whose wording closely
echoed `_CONTRADICTION_EXAMPLE` (as instructed — that's what a correct
contradiction call-out is supposed to look like) — `core/safety.py`'s
prompt-leak filter then matched >60 chars of overlap with the *registered*
system prompt and redacted the entire answer down to `"[redacted: system
prompt]"`. The fix is not "loosen the filter" (that would weaken real-leak
detection); it's "don't register the one part of the prompt the model is
*supposed* to echo." `_SCOPE_EXAMPLE` uses deliberately fictional item names
("Zeta", "Alpha", "Gamma" tiers) rather than real corpus entities, so if a
weak model ever echoes it despite the exclusion, the result is an obviously
wrong placeholder rather than a plausible-looking real answer.

---

## 8. `sql/` — quantitative path

### 8.1 `sql/schema.py` — two-tier schema knowledge

**Tier 1 (machine truth)**: `introspect(db_path) -> SchemaInfo` reads
`sqlite_master` (tables/views, excluding `sqlite_%`) then `PRAGMA
table_info("{table}")` per table. This is what the guard's `KnownTablesRule`/
`KnownColumnsRule` check against directly — nothing in the prompt card can
ever grant access to a table this doesn't know about.

**Tier 2 (prompt truth)**, `build_schema_card(db_path) -> str`, assembled as
`[CARD_HEADER, render_tables(...), render_enums(...), SCHEMA_CARD_SEMANTICS,
render_join_keys(...), SCHEMA_CARD_CAUTION]`:
- `render_tables`/`render_enums`/`render_join_keys` are **generated fresh from
  a live connection every time** (`read_schema_facts`: row counts, `PRAGMA
  foreign_key_list` for join keys, `SELECT DISTINCT` unioned across every
  table carrying a given enum-worthy column name, skipped if the union exceeds
  `MAX_ENUM_VALUES=30`) — structural drift is impossible for this half.
- `SCHEMA_CARD_SEMANTICS`/`SCHEMA_CARD_CAUTION` are **hand-written prose**
  (what "won" means, that `discount_pct` is a fraction not a percent, the
  stage-taxonomy mismatch warning). This half can drift silently on a rename;
  `validate_card(info, card) -> list[str]` catches a hand-written reference to
  a table/column that no longer exists, but cannot catch a *semantic* drift
  (e.g. if `'Closed Won'` were renamed to `'Won'`, the generated enum list
  would follow the rename but the hand-written win-rate prose would not).

`default_schema_card()` (`@lru_cache(maxsize=1)`) degrades to a stub
(`SCHEMA_CARD_FALLBACK`) if the DB is missing, rather than raising — importing
`sql.schema` must be safe on a fresh clone with no DB asset yet.

### 8.2 `sql/guard.py` — AST allowlist

```python
class GuardResult:
    ok: bool; sql: str; reason: str = ""
    @property
    def verdict(self) -> str: ...   # "PASS" or "REJECT: {reason}"

class GuardRule(ABC):
    def violation(self, tree, schema) -> str | None: ...

class SqlGuard:
    DEFAULT_RULES = (SelectOnlyRule, NoWriteNodeRule, KnownTablesRule,
                      KnownColumnsRule, AllowedFunctionsRule)
    def check(self, sql: str) -> GuardResult: ...
```

Rules, run in order, first violation wins:

| Rule | Check |
|---|---|
| `SelectOnlyRule` | root node is `SELECT`/`UNION`/`Subquery` |
| `NoWriteNodeRule` | walks **every** node (not just root) rejecting `Insert/Update/Delete/Drop/Alter/Create/Attach/.../TruncateTable/Reindex` — catches a write nested inside a CTE or subquery |
| `KnownTablesRule` | every non-CTE-alias table reference exists in the live `SchemaInfo`; rejects any `catalog.` or non-`main` `db.` qualifier (blocks cross-database `ATTACH` tricks) |
| `KnownColumnsRule` | every **qualified** column (`table.col`) resolves its qualifier to a real table/alias and exists on it; unqualified columns and CTE/subquery-derived columns are skipped (ambiguous or not real schema columns) |
| `AllowedFunctionsRule` | every function call name (matched against the **rendered SQL text**, `r"\s*([A-Za-z_]\w*)\s*\("`, not sqlglot's internal node taxonomy — so an implicit-cast wrapper node doesn't get mis-flagged as a function call) is in `config.SQL_ALLOWED_FUNCTIONS` |
| `ScopedQueryGuardRule` *(constructed per-user, not in `DEFAULT_RULES`)* | see §16.5 |

`check()`: strips a trailing `;`, parses with `sqlglot.parse(sql,
dialect="sqlite")`; **more than one parsed statement is rejected outright**
(stacked-statement injection, since `sqlglot.parse` splits on top-level `;`
even without one present in the string). LIMIT injection: `if
tree.args.get("limit") is None: tree = tree.limit(row_limit)` — only when no
LIMIT exists at all; an existing (even larger) LIMIT is left untouched. The
**rewritten** tree, re-rendered to SQLite dialect, is what `GuardResult.sql`
carries forward — not the original input string.

### 8.3 `sql/generate.py` / `sql/execute.py` / `sql/narrate.py`

`SqlGenerator.generate(question, slots)` / `.repair(question, failed_sql,
error, slots)` each make exactly **one** LLM call — no self-consistency
sampling. `repair()`'s prompt is the full original prompt plus an appended
block naming the failed SQL and the verbatim SQLite error text.

`SqlGenerator.slot_block(slots)` renders the resolved filters into the
prompt as explicit "RESOLVED FILTERS" text — and **always forces
`stage_definition` to the literal string `"database stages"`** if that key
is present in `slots` at all, regardless of what the router actually
extracted (`{**slots, STAGE_DEFINITION: "database stages"}` before
rendering). The database only ever has the 7-value `opportunities.stage`
enum; the Field Guide's 6-stage playbook names don't exist as data, so
telling the SQL model anything else would ask it to invent a column value
with no real equivalent. This is the mechanism `router/slots.py::required_slots`
(§6.3) relies on to make asking the user for `stage_definition` unnecessary.

`QueryExecutor.execute(sql)` opens
`sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)` — this is
the **second, independent** safety layer under the AST guard: even a guard
bypass that let a write statement through still cannot mutate the file,
because the OS/driver refuses to open it for writing. Installs a
`set_progress_handler` callback (fires every 1000 VM instructions, aborts once
the running count implies `timeout_s * 1000` instructions have passed) as a
crude instruction-count proxy for a wall-clock timeout — not a true timer, so
a query blocked on I/O rather than burning CPU wouldn't be interrupted by
this specific mechanism (SQLite's own lock-contention `timeout=` covers that
separately). Fetches `max_rows + 1` rows to detect truncation with one probe
row, then trims to `max_rows` and runs `safety.strip_denied_columns`. Errors
are **returned** in `QueryResult.error`, never raised — that's what lets
`sql/pipeline.py` feed the message back into a repair prompt.

`needs_narration(question, result, threshold=20)`:
```python
if result.row_count > threshold: return True
return any(marker in question.lower() for marker in INTERPRETIVE_MARKERS)
```
i.e. the two triggers are **OR'd**: either more than 20 rows, or interpretive
language ("why", "what's driving", "trend", ...), forces a narration LLM
call. The skip condition (zero LLM calls, table only) requires *both*
`row_count <= 20` **and** no interpretive marker. On a verbatim-number
violation (§16.4), exactly one retry naming the offending tokens; on a second
failure, narration is abandoned and the verified table is shown alone.

### 8.4 `sql/pipeline.py` — the one-shot repair loop

```python
class SqlPath:
    def run(self, question, trace, *, slots=None, render_answer=True) -> SqlOutcome: ...
```

Exact sequence: generate → guard.check() → (guard reject ⇒ **stop, never
executes**) → execute() → (`not result.ok` ⇒ repair: one more
generate-with-error-context call → guard.check() again on the repaired SQL →
execute if it passes, else give up and surface the *original* failure) →
render (skipped entirely when `render_answer=False`, which is how the hybrid
path avoids a redundant narration call). **Repair fires only on a SQLite
execution error** — never on a guard rejection, and never merely because the
result was an empty-but-successful row set. Capped at exactly one repair
attempt; there is no loop.

---

## 9. `hybrid/` — both paths, composed

### 9.1 The 4-step sequence (`hybrid/pipeline.py`)

(Step 1, "router decomposes the question into `sql_subquestion` +
`doc_subquestion`," already happened in the orchestrator before `HybridPath.run`
is called.)

2. **SQL**: `SqlPath.run(sql_subquestion or question, ..., render_answer=False)`
   — no narration; the composer writes the final text itself. **If SQL
   fails**, this is a full graceful-degradation branch, not an error: fall
   back to a RAG-only answer over `doc_subquestion`, prepend a fixed
   disclaimer that the quantitative half couldn't be computed, and return.
3. **Conditioned retrieval**: build a generation-free digest of the SQL result
   (`summarise_result`: up to 6 rows rendered as `"col=val col=val"`, joined —
   pure string formatting, no LLM), append it to the doc sub-question as
   `f"{doc_q}\n[SQL result] {digest}"`, and retrieve against *that* combined
   query — so a follow-up doc question can be informed by what the numbers
   turned out to be (e.g. "what gates a deal before Commit" retrieves
   differently once the digest mentions a specific stage name).
4. **Compose + verify**: `Composer.compose(question, sql, result, hits)` — note
   the **original full question** is passed here, not the doc sub-question.

### 9.2 `hybrid/composer.py`

```python
class Composer:
    def compose(self, question, sql, result, hits, trace=None) -> tuple[str, bool]: ...
    def fallback(self, sql, result, hits) -> str: ...   # deterministic, no LLM
```

The **Finding / What the documentation says / Caveats** three-part structure
is enforced entirely by the prompt (`COMPOSER_SYSTEM` rule 5 in
`hybrid/prompts.py`) — nothing in `composer.py` parses or validates section
headings; only the numeric content is checked. `compose()` calls the LLM once,
verifies via `NumberVerifier`, retries once with the offending tokens named,
and on a second failure or `LLMUnavailable` falls back to `fallback()` — which
renders its **own**, different two-heading structure (`**Finding (from
SQL)**` / `**Relevant documentation**`, no Caveats section at all). If you
want the fallback to be structurally indistinguishable from the LLM path,
that's a deliberate improvement to make when rebuilding, not something the
current code does.

### 9.3 `hybrid/verify.py` (imported by both `hybrid/composer.py` and
`sql/narrate.py` — build this before `sql/narrate.py` despite the package
order in the file tree)

See §16.4 for the full algorithm — it's identical logic reused in both
places, which is exactly why it lives in `hybrid/` as a shared module rather
than being duplicated.

### 9.4 Per-user path instances (`orchestrator.py`, not `hybrid/` itself)

`SqlPath`/`HybridPath` are **not** process-wide singletons. Each carries a
`ScopedQueryGuardRule` bound to one user's `allowed_scope_values()` at
construction time; a single shared instance would leak one logged-in user's
ACL into a concurrent user's query, since Streamlit serves multiple browser
sessions from one process. `GTMCopilot` keeps `dict[username, SqlPath]` /
`dict[username, HybridPath]` caches, built lazily on first use per user. Every
*other* collaborator (`RagPath`, `Clarifier`, `Reframer`, `HistoryReframer`,
the shared `Index`) is user-agnostic and stays a single shared instance.

---

## 10. `ask/` — underspecified and follow-up questions

### 10.1 `ask/clarify.py`

```python
class Clarifications(BaseModel):
    questions: list[str] = []

class Clarifier:
    @staticmethod
    def fallback(missing: list[str]) -> list[str]: ...   # deterministic, test-covered contract
    def clarify(self, question, missing, reason, trace) -> str: ...
```

`fallback()` looks up each missing slot in `SLOT_QUESTIONS`, takes the first
3; if that's empty, returns one generic fallback question. `clarify()` calls
`chat_json(CLARIFY_SYSTEM, ..., Clarifications, trace=trace,
stage="ask_clarify")` with **no `model=` override** (uses the 7B
`answer_model` by default — see the §5.1 note on this asymmetry). On
`LLMJSONError | LLMUnavailable`, swallows the error and falls through to the
deterministic fallback — the ASK path must never itself fail. Final text is a
fixed wrapper: a lead-in sentence, up to 3 bulleted questions, and a closing
line explaining why a guess wasn't made instead.

### 10.2 `ask/reframe.py` — continuation vs. pivot

```python
class ReframeDecision(BaseModel):
    is_new_topic: bool
    effective_question: str

class Reframer:
    def reframe(self, pending_question, missing, reply, trace) -> ReframeDecision: ...
```

Precomputes a deterministic fallback —
`ReframeDecision(is_new_topic=False, effective_question=f"{pending_question} {reply}")`
(literal string concatenation) — used verbatim if the LLM call fails. On
success, calls `chat_json(..., model=self.client.router_model, ...)` (cheap
tier, same reasoning as the main router). **Self-contradiction guard**: if the
model claims `is_new_topic=True` but also returned an `effective_question`
that differs from `reply` (i.e. it rewrote/merged text while also claiming
"unrelated pivot," which contradicts the prompt's own instruction to return
the reply *unchanged* on a genuine pivot), the code overrides `is_new_topic`
to `False` and treats the model's rewritten text as the merge candidate,
rather than trusting the contradictory boolean.

**Lossy-merge guard** (runs on that candidate, whichever branch produced it):
uses `ask/text_overlap.py::content_words` + `preserves_content` to check the
merge kept most of `pending_question`'s content AND actually incorporated
`reply`'s own new detail — if either check fails, discard the candidate and
use the deterministic concatenation fallback instead of a fluent-but-broken
merge. Two real failures motivated this: (1) a compound pending question
("count X and explain Y") got paraphrased down to just one half, silently
dropping the count; (2) after tightening the prompt to fix that, the model
instead returned `is_new_topic=true` with `effective_question` equal to
`pending_question` *verbatim* — the self-contradiction guard correctly flips
that back to a continuation, but the reply's own new content was never
incorporated, which only the lossy-merge guard catches.

Orchestrator integration: when `pending_clarification` is set,
`GTMCopilot.answer()` calls `reframe()` before routing, sets
`trace.pending_question`/`trace.is_new_topic`, computes `effective_question`
(the reframed merge, or the raw new question if it's a genuine pivot), and
**overwrites `trace.question`** with it (`trace.raw_question`, set once at
`Trace` construction, is unaffected — see §5.2) — routing proceeds on the
effective question, not the user's literal new message.

### 10.3 `ask/text_overlap.py` — shared merge-safety guards

Extracted out of `ask/reframe.py` once `ask/history_reframe.py` (§10.4)
needed the identical logic — a rewrite that drops or invents content is
exactly as dangerous for a general follow-up as it is for an ASK reply.

```python
def content_words(text: str) -> set[str]: ...
    # lowercased [a-z0-9]+ tokens, len >= 3, minus a small stopword list
    # ("the","this","that","what","which","does","should","have","with",
    # "many","about","into","from","were","will","would","could","when","where")
    # threshold is >= 3, NOT > 3: a "> 3" cutoff silently drops real 3-letter
    # domain terms like "SLA" - a follow-up like "What is the SLA days?"
    # needs "SLA" recognised as content the merge must preserve.
def preserves_content(source_words, reply_words, merged_words) -> bool: ...
    # source_ok: len(source_words & merged_words) / len(source_words) >= 0.6
    # (skipped if source_words is empty)
    # reply_ok: bool(reply_words & merged_words) (skipped if reply_words is empty)
    # returns source_ok AND reply_ok
def no_fabricated_entities(source_words, reply_words, merged_words) -> bool: ...
    # returns not (merged_words - source_words - reply_words)
    # i.e. every content word in the merge must be traceable to EITHER the
    # source context OR the reply - anything else was invented.
```

`ask/reframe.py` uses `content_words` + `preserves_content` (its "source" is
the single `pending_question`, which legitimately must survive close to
intact — see §10.2). `ask/history_reframe.py` uses all three, but see §10.4
for why it does **not** reuse `preserves_content`'s `source_ok` half as-is.

### 10.4 `ask/history_reframe.py` — follow-ups without a pending ASK

```python
class HistoryReframeDecision(BaseModel):
    is_new_topic: bool
    effective_question: str

class HistoryReframer:
    def reframe(self, recent_turns: list[dict], question: str, trace) -> HistoryReframeDecision: ...
        # recent_turns: up to 2 {"question": str, "answer": str} dicts, oldest first

def looks_like_followup(question: str) -> bool: ...
def _find_stage_span(text: str) -> tuple[str, str] | None: ...
def _mentioned_stage(text: str) -> str | None: ...
def _strip_leading_connector(text: str) -> str: ...
def _bare_stage_substitution(recent_turns, question) -> str | None: ...
def _substitute_stage(template: str, old_text: str, new_text: str) -> str: ...
```

**Role and mutual exclusivity.** Closes README's documented gap
("conversation history is not fed to the router, so 'and for EMEA?' routes
to ASK"). Runs when `pending_clarification` is **NOT** set (i.e. the turn
does not follow an ASK) — `orchestrator.py`'s `if pending_clarification: ...
elif recent_turns: ...` is the structural guarantee that `Reframer` and
`HistoryReframer` never both run on the same turn. `recent_turns` is
pre-filtered by the caller (`app.py::recent_answered_turns`, §13) to only
`SQL`/`RAG`/`HYBRID` turns — an `ASK`/`REFUSE` turn has no factual answer
worth referencing.

**`looks_like_followup(question)` — the latency gate.** Skips the LLM call
entirely (the common case: a fresh, self-contained question) unless the
question starts with a connector phrase (`"and "`, `"what about"`, `"same
for"`, `"also"`, `"how about"`, `"what if"`, `"compared to"`), contains a
bare reference pronoun (`"it"`, `"that"`, `"those"`, `"this"`), or is short
(≤8 words) **and** lacks an explicit entity — reusing
`router/slots.py::has_explicit_time`/`has_explicit_population`, plus its own
small stage-name/product-name lists, rather than reinventing entity
detection. Pure word-count alone was tried and rejected during
implementation: it flagged fully self-contained short questions ("What
deployment modes does Product XYZ support?", "How many deals closed in
2024?") as follow-ups.

**`reframe()` control flow, in order:**
1. No `recent_turns`, or `not looks_like_followup(question)` → return
   `is_new_topic=True, effective_question=question` unchanged. Zero LLM
   calls.
2. `_bare_stage_substitution(recent_turns, question)` — a **deterministic
   shortcut**, checked before ever calling the LLM (see below).
3. Otherwise, one `chat_json(HISTORY_REFRAME_SYSTEM, ..., model=router_model)`
   call, then four guards.
4. On `LLMUnavailable`/`LLMJSONError`, or if any guard rejects the merge,
   fail OPEN to **no rewrite** (`is_new_topic=True, effective_question`
   unchanged) — deliberately NOT blind concatenation like `ask/reframe.py`'s
   fallback. Concatenating two full past Q&A pairs onto a new question is
   noisy garbage; leaving the question as typed can only match today's
   status quo (worst case, an ASK), never make it worse.

**The four guards (all real production failures, not speculative
hardening):**

| # | Guard | Real failure it catches |
|---|---|---|
| 1 | Self-contradiction (same idea as `ask/reframe.py`, reused inline) | The model reliably returns `is_new_topic=True` even alongside a genuinely correct, non-verbatim rewrite - skipping this guard would silently discard good merges on nearly every real call. |
| 2 | Reply-inclusion only, **not** full `preserves_content` | `preserves_content`'s `source_ok` check (≥60% of *combined Q+A* content) is too strict here: `recent_turns` includes a full prior ANSWER, and a correct merge naming "Solution Fit" failed that check because the prior answer's exit-criteria prose contributed 16 more content words a follow-up about SLA days has no reason to restate. Only checks the reply's own new content actually survived. |
| 3 | Fabrication (`no_fabricated_entities`) | History about "EMEA opportunities Closed Won", message "Why is that stage risky?" (no stage anywhere in that history) → the model invented "the Solution Fit stage," copied from its own prompt's worked example. Swapping the example for a fictional "Zeta stage" placeholder just produced a *different* fabricated name - the model reliably invents *something* plausible under-context, so this must be a structural check, not a prompt-wording fix. |
| 4 | Stale-stage retention, checked against **every** `recent_turns` entry, not just the last | Even with an explicit "don't do this" prompt example, the model produced "What are the exit criteria for Stage 1 - Qualify and in Discover?" - keeping "Qualify" instead of substituting "Discover". With 2 turns in context, a stale merge once retained the *older* turn's stage, not the most recent one - checking only `recent_turns[-1]` would have missed it. |

**`_bare_stage_substitution` — deterministic shortcut, not just a latency
optimization.** Guard 4 exists because the LLM demonstrably cannot reliably
substitute a bare stage reference ("what about commit", "and in Discover?")
even when told not to append instead of replace — so for this one common
pattern, the code does the substitution itself and skips the LLM entirely:
finds the new stage named in `question` (`_find_stage_span`, order-sensitive
— "discovery" checked before "discover" so the longer name isn't
mis-detected as a substring match), confirms the reply is a *bare* reference
(nothing left over, once a leading connector is stripped, besides the stage
name itself — `content_words(remainder) - set(new_key.split())` must be
empty), then walks `recent_turns` most-recent-first for the first turn whose
own question names a **different**, **non-bare** stage (skipping a turn
that's itself just a bare reference — not a usable template) and substitutes.

**`_substitute_stage` also strips a stale "Stage N -" prefix.** A first
version of the substitution left `"Stage 1 - Qualify"` → `"Stage 1 -
Discover"` — cosmetically odd, but a *real* regression, not merely
cosmetic: the stale stage **number** anchored retrieval/the answer model on
the wrong stage's content, producing a wrong answer (Discover's exit
criteria came back as Qualify's, mislabeled). `_STAGE_NUMBER_PREFIX_RE =
r"stage\s*\d+\s*[—\-:]\s*$"` matches only when the prefix sits immediately
before the matched stage name, so the whole `"Stage 1 - "` span is dropped,
not just the name.

---

## 11. `eval/` — RAGAS-style offline evaluation (no LangChain)

Built to mirror `ragas`'s best-known metrics without pulling in its
LangChain dependency. Two are pure functions (no ground-truth-free judge
needed to reason about them); two need an LLM judge on `ANSWER_MODEL`.

```python
# eval/metrics.py
def context_recall(retrieved: list[Section], relevant: list[Section]) -> float: ...
def context_precision(retrieved: list[Section], relevant: list[Section]) -> float: ...
def faithfulness(question, answer, contexts: list[str], client) -> FaithfulnessResult: ...
def answer_relevancy(question, answer, client, embed_fn) -> float | None: ...
```

See §16.6 for exact formulas. `eval/dataset.py` holds hand-labelled
`EvalCase`s (`question`, expected `route`, `relevant_sections: tuple[(doc,
section), ...]`, a `reference_answer` for human sanity-checking only — no
metric consumes it). `scripts/eval.py` runs each case through the real
`GTMCopilot` (no stubs — same philosophy as `scripts/demo.py`), scores all
four metrics, prints per-question and mean scores, and optionally writes
`EVALUATION.md`. A misrouted case (actual route ≠ expected) is reported but
excluded from the metric means — there's nothing meaningful to score against
a retrieval/answer that wasn't produced by the path the case is meant to
test.

**Removed**: a `context_relevance` metric (LLM judges each retrieved chunk
independently for relevance, no ground truth needed) previously lived here
alongside these four.

---

## 12. `orchestrator.py` — `GTMCopilot`

```python
@dataclass
class Answer:
    text: str; route: str; trace: Trace
    hits: list[Hit] = []; sql: str | None = None
    def persisted(self) -> "Answer": ...   # appends trace to storage/traces.jsonl

class GTMCopilot:
    def __init__(self, client=None, catalog=None, index=None, router=None, rule_engine=None): ...
    def answer(self, question: str, user: UserProfile,
               pending_clarification: str | None = None,
               pending_missing_slots: list[str] | None = None,
               recent_turns: list[dict] | None = None) -> Answer: ...
    def preflight(self) -> dict: ...   # never raises — sidebar-safe
```

One turn: build a `Trace` → reframe (`if pending_clarification:` call
`Reframer`; `elif recent_turns:` call `HistoryReframer` instead — see §10.2/
§10.4, mutually exclusive by construction) → `_route()` (one
`Router.decide()` call, then `RuleEngine.apply()`, writing every field of
the trace along the way) → dispatch to exactly one of REFUSE / ASK / RAG / SQL
/ HYBRID based on `final_route` → wrap the result in `Answer` → `.persisted()`.
No path calls another path's internals directly — `HybridPath` *holds* a
`SqlPath` and a `Retriever` as constructor-injected objects, so SQL behavior
cannot silently diverge between the SQL and HYBRID routes.

Module-level singleton: `get_copilot()` / `answer_question(question, *, user,
...)` / `preflight()` / `get_schema()` — thin wrappers so `app.py` and
`scripts/*.py` never need to construct the object graph by hand.

---

## 13. `app.py` — Streamlit UI

Pure presentation; the only pipeline-adjacent import is a lazy
`rag.index.Index` + `rag.ingest.PdfChunker` inside the "Re-ingest PDFs"
sidebar button handler. Everything else goes through
`orchestrator.answer_question` / `orchestrator.preflight`.

**Login**: gated on `"user" not in st.session_state`. Constructs `UserStore()`;
a missing `assets/users.yaml` shows an error and `st.stop()`s the whole run.
On successful `store.authenticate(username, password)`, stores the returned
`UserProfile` in session state and `st.rerun()`s; on failure, shows one
generic "Incorrect username or password" message (never reveals which part
was wrong). `st.stop()` after the login block prevents any of the chat UI
from rendering pre-login.

**Session state**: `user`, `messages` (list of `{"role","content","route"?,
"trace"?}`), `pending_clarification`, `pending_missing_slots`. Both pending
keys are threaded into every `answer_question()` call; after the call, if the
route was ASK they're updated to `answer.trace.question` (the *resolved*
question text, post-reframe) and `answer.trace.missing_slots` — for any other
route, both reset to `None`. This is transient Streamlit session state only;
it is not part of the persisted `messages` chat log and does not survive a
hard refresh.

**`recent_answered_turns(messages, limit=2) -> list[dict]`**: no new session
key — derived fresh from the existing `messages` list on every turn, scanning
backwards for the last `limit` assistant messages whose `route` is
`SQL`/`RAG`/`HYBRID` (an `ASK`/`REFUSE` turn is skipped; it has no answer
worth referencing), pairing each with its preceding user message into
`{"question": ..., "answer": ...}`. Passed as `recent_turns=` on every
`answer_question()` call, unconditionally — `orchestrator.py`'s
`if pending_clarification: ... elif recent_turns: ...` is what enforces
mutual exclusivity with the ASK-continuation path, so `app.py` doesn't need
its own duplicate conditional here.

**Trace panel**: `st.json` on the trace dict (round-tripped through
`json.loads(json.dumps(trace, default=str))` so it survives storage in session
state across reruns without holding live objects); for the current turn, the
collapsed expander label itself shows route, any firing rule, and total
latency before the user even expands it.

---

## 14. `scripts/`

`scripts/ingest.py` — idempotent (`Index.exists()` short-circuits unless
`--force`); prints a per-doc chunk-count breakdown before embedding.
`scripts/demo.py` — five fixed prompts, one per route (including the REFUSE
bonus case), measures per-stage and total latency, optionally writes
`MEASUREMENTS.md`. `scripts/eval.py` — see §11.

Both `demo.py` and `eval.py` construct one fixed, **fully unrestricted**
`UserProfile` (`allowed_regions=None, allowed_segments=None`) — not a real
login, just enough to satisfy `answer_question`'s required `user` keyword
without any ACL narrowing interfering with a latency or quality measurement.

---

## 15. `tests/` — strategy

One test file per module (`test_rrf.py`, `test_sql_guard.py`,
`test_number_check.py`, `test_chunking.py`, `test_safety.py`,
`test_reframe.py`, `test_history_reframe.py`, `test_eval_metrics.py`, ...)
plus two cross-cutting files:

- **`test_pipeline_offline.py`** — full orchestration (router → rules → path
  dispatch → guard → execute → render) against a hand-rolled `StubClient`
  subclassing `LLMClient` and overriding `_complete()` to return scripted
  JSON. No real Ollama, no network — this is how repair loops, refusals, and
  ACL scoping get tested deterministically. **Copy this `StubClient` pattern**
  for any new LLM-judge-style test (`eval/metrics.py`'s tests use exactly
  this shape).
- **`test_router_golden.py`** — two layers. Layer 1 (always runs) drives the
  rule engine with mocked router outputs, covering every rule and their
  precedence order. Layer 2 (`skipif` no reachable Ollama) runs ~20 labelled
  prompts through the *real* local router model and asserts the final route —
  the one genuinely non-deterministic, "honest accuracy" signal in the suite,
  deliberately left unmocked.
- **`test_db_integration.py`** / **`test_chunking.py`** — `skipif` the real DB
  / a generated fixture PDF is absent, so a fresh clone without assets still
  runs the rest of the suite green rather than failing on missing files.

`tests/run_all.py` prints an environment banner (DB present? PDFs present?
index built? Ollama reachable?) **before** invoking pytest, specifically
because roughly a third of the suite is conditionally skipped via the
`skipif` guards above — without the banner, a heavily-skipped run looks
identical in the terminal to a fully green one.

---

## 16. Key algorithms, consolidated

### 16.1 Reciprocal Rank Fusion

```
score(doc) = Σ over each ranked list containing doc of  1 / (k + rank_in_that_list)
```
`k=60`. Needs only each list's *rank*, never its score — no normalization
constant to re-fit when the corpus changes. Pure function; unit-test with
hand-built lists and hand-computed expected scores.

### 16.2 Structure-aware PDF chunking

See §7.1. The two invariants worth testing explicitly: (a) every table-chunk's
text repeats the header row, even for the 2nd/3rd/... group of data rows from
the same table; (b) prose chunking never splits inside a sentence, only at
sentence boundaries once the running token estimate would exceed the target.

### 16.3 AST SQL guard

See §8.2. The two-layer defense is the point: the AST allowlist proves no
forbidden *statement shape* can be sent to SQLite; the read-only connection
string (`file:...?mode=ro`) proves that even if the allowlist had a bug,
SQLite itself refuses to write to the file. Test both independently — the
guard with a fake in-memory `SchemaInfo` (no real DB needed), the read-only
property against the real DB (attempt a real `UPDATE`, assert it raises).

### 16.4 Verbatim number verification (`hybrid/verify.py`)

Every numeric token (`_NUMBER_RE = r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?"`) found in
generated text must trace back to one of: a cell in the SQL result set (via
`QueryResult.flat_values()`, which also scans numeric substrings *inside*
string cells — a date like `2024-03-01` licenses `2024`, `03`, and `01`
separately), the result's own `row_count`, or a numeric substring in the
question text or the executed SQL string (both treated as user-supplied
context, never model fabrication).

Matching policy: exact match within `1e-9`, **or** rounding the *source* value
down to the token's own decimal-place count and comparing — so a source of
`61234.5678` licenses a model writing `"61234.57"` (2 places), but a source of
exactly `61234` does not license `"61234.57"` (more precision than the source
actually has is never licensed; less precision always is). A `%`-suffixed
token is checked against both the literal value and `value/100`, plus a
reverse `source*100` check, so a fractional DB value like `0.42` licenses a
model writing `"42%"`.

On failure: exactly one regeneration attempt naming the offending tokens
verbatim; on a second failure, abandon the generated text and render the
verified table (SQL narrator) or the deterministic fallback layout (hybrid
composer) instead.

### 16.5 ACL scoping — two independent layers

1. `router/rules.py::RegionScopeRule` (R1B) — checks the `segment_or_region`
   *slot value* the router extracted, narrows a literal `"all"` to the user's
   actual allowed list in place, or refuses if it names a disallowed value.
   This only sees what the router bothered to extract into the slot.
2. `sql/guard.py::ScopedQueryGuardRule` — a **backstop** at the SQL-AST level,
   constructed per-user with `allowed_scope_values()`. Walks the parsed query
   for literal equality/`IN`-list values on `region`/`segment` columns on any
   of the four scoped tables (`accounts, opportunities, deployments,
   activities`); rejects if the query touches a scoped table with **no**
   region/segment predicate at all, or if any literal found lies outside the
   allowed set. This exists specifically because `deployments`/`activities`
   carry no region/segment column of their own — a query joining only those
   two tables could otherwise produce a cross-scope aggregate that the
   slot-level check never even looks at.

Both layers must pass; either one can refuse independently.

### 16.6 RAGAS-style eval metrics

```
context_recall(retrieved, relevant)    = |retrieved ∩ relevant| / |relevant|         (1.0 if relevant is empty)
context_precision(retrieved, relevant) = mean over each relevant hit's rank r of (count of relevant hits in retrieved[0..r] / (r+1))
faithfulness(answer, contexts)         = (LLM lists every factual claim in the answer) fraction marked supported by contexts
answer_relevancy(question, answer)     = mean cosine similarity between question and N LLM-generated hypothetical questions the answer would suit
```

`context_recall`/`context_precision` need a pre-labelled `relevant_sections`
set per question (§11's `EvalCase`); the other two need no labels, only an
LLM judge on `ANSWER_MODEL`. With exactly one labelled-relevant section per
question (this repo's current dataset shape), `context_recall` reduces to a
binary hit/miss (equivalent to IR's "Hit Rate") and `context_precision`
reduces to `1/(rank_of_the_hit+1)` (equivalent to Mean Reciprocal Rank) — both
would generalize to their fuller forms (multi-item recall, Average Precision)
the moment a question is labelled with more than one relevant section.

---

## 17. Known quirks / open gaps (from verification against live code)

1. ~~`assets/users.example.yaml` doesn't exist.~~ **Resolved** — the file now
   exists (sanitized structure, no real hashes; see §4.2 for the schema).
2. **`rag/retrieve.py`'s docstrings say "top-5"; the live value is
   `RRF_TOP_K=7`.** Not a functional bug — just update the comments if you
   copy this file, or intentionally set it back to 5 if you want the
   documented behavior instead of the current tuned value.
3. **`hybrid/composer.py`'s deterministic `fallback()` does not reproduce the
   Finding/Docs/Caveats three-heading structure** the LLM path is prompted to
   produce (it uses its own two-heading layout with no Caveats section). Not
   wrong, just inconsistent between the two branches — worth unifying if you
   want callers to be able to rely on a stable answer shape regardless of
   which branch produced it.
4. **`core/llm_client.py`'s `top_k`/`top_p` are literals inside
   `OllamaClient._complete`,** not constants in `core/config.py`, despite
   `config.py` documenting itself as the single source of truth for
   determinism knobs. Cosmetic; move them if you want that claim to be
   literally exhaustive.
5. **`router/prompts.py`'s inline comment claims "eight few-shot examples;"
   the template actually has ten** (two per route × five routes, including
   OFF_TOPIC). Stale comment from before OFF_TOPIC's examples were added.
6. **`ask/clarify.py` uses the 7B `answer_model` by default** (no `model=`
   override on its `chat_json` call), while `router/llm_router.py` and
   `ask/reframe.py` both explicitly pass the 3B `router_model`. If this
   asymmetry wasn't deliberate, decide one way when rebuilding; if it *was*
   deliberate (clarification wording quality matters more than routing
   latency), document it as such rather than leaving it implicit.
7. **No row-level security beyond the region/segment ACL** — `core/safety.py`
   ships a column denylist (`config.COLUMN_DENYLIST`) that is empty by
   default; the synthetic DB has no per-user ownership model to enforce
   further.
