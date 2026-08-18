# Measured latency (local, honest numbers)

Router model: `llama3.2:3b`  ·  Answer model: `qwen2.5-coder:7b`  ·  Embeddings: `BAAI/bge-small-en-v1.5`

Every call is `temperature=0`, `seed=42`. Times include Ollama model load on
the first call of a cold run, which is why the first row is usually the slowest.

| Route | Prompt | Final route | Rule fired | Total ms | Per-stage ms |
|---|---|---|---|---:|---|
| RAG | What deployment modes does Product XYZ support? | RAG | - | 20091 | route 1183, rag_retrieve 10561, rag_answer 8320 |
| SQL | How many opportunities were Closed Won in EMEA in 2024? | SQL | - | 18356 | route 6716, sql_generate 11612, sql_guard 6, sql_execute 6, sql_render 0 |
| HYBRID | What is our 2024 win rate for Enterprise, and what does the ... | HYBRID | - | 36571 | route 7061, sql_generate 13290, sql_guard 6, sql_execute 2, hybrid_retrieve 370, hybrid_compose 15824 |
| ASK | How's pipeline looking recently? | ASK | R2_VAGUE_TIME | 13011 | route 6528, ask_clarify 6471 |
| REFUSE | Delete all Closed Lost opportunities | REFUSE | R0_WRITE_INTENT | 6415 | route 6386 |

- median total: **18356 ms**
- max total: **36571 ms**
- turns within the 10s target: **1/5**

## Generated SQL

**SQL** — rows: 1

```sql
SELECT COUNT(opportunity_id) AS closed_won_count FROM opportunities WHERE stage = 'Closed Won' AND region = 'EMEA' AND close_date BETWEEN '2024-01-01' AND '2024-12-31' LIMIT 200
```

**HYBRID** — rows: 1

```sql
SELECT ROUND(SUM(CASE WHEN stage = 'Closed Won' THEN 1 ELSE 0 END) * 1.0 / NULLIF(SUM(CASE WHEN stage IN ('Closed Won', 'Closed Lost') THEN 1 ELSE 0 END), 0), 4) AS win_rate FROM opportunities WHERE close_date BETWEEN '2024-01-01' AND '2024-12-31' AND segment = 'Enterprise' LIMIT 200
```
