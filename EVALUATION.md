# RAG/HYBRID quality evaluation (RAGAS-style, local)

Native implementation of RAGAS-style metrics - not the `ragas` package, which hard-depends on LangChain. See `eval/metrics.py` for each definition. `context_recall`/`context_precision` are deterministic (no LLM, need a pre-labelled ground-truth section); `faithfulness`/`answer_relevancy` use an LLM judge (no labels needed) and are therefore approximate, not exact.

| Route | Question | Recall | Precision | Faithfulness | Relevancy |
|---|---|---:|---:|---:|---:|
| RAG | What deployment modes does Product XYZ support? | 1.00 | 0.17 | 1.00 | 0.88 |
| RAG | What is included in the Growth pricing tier and how much doe... | 1.00 | 1.00 | 1.00 | 0.91 |
| RAG | What are the common SKUs for Product XYZ? | 1.00 | 1.00 | 1.00 | 0.82 |
| RAG | What are the different stages in the Opportunity Tracker's s... | 1.00 | 0.29 | 1.00 | 0.94 |
| RAG | What are the exit criteria for the Solution Fit stage? | 1.00 | 0.75 | 1.00 | 0.91 |
| RAG | What is the recommended action when a deployment status is R... | 1.00 | 0.75 | 1.00 | 0.91 |
| RAG | What are the four dimensions of the deployment risk scoring ... | 1.00 | 0.70 | 1.00 | 0.83 |
| RAG | What is the source system and example value for the expected... | 1.00 | 0.89 | 1.00 | 0.90 |
| RAG | Who is a champion, per the tracker's data dictionary? | 1.00 | 0.92 | 1.00 | 0.79 |
| RAG | What is deployment_risk_score and what does its value repres... | 1.00 | 0.64 | 1.00 | 0.88 |
| HYBRID | What is our 2024 win rate for Enterprise, and what does the ... | 1.00 | 0.50 | 0.67 | 0.78 |
| HYBRID | Show 2024 NA deals stuck in Negotiation and explain the risk... | 1.00 | 0.42 | 1.00 | 0.81 |

## Means

- **context_recall**: 1.000 (n=12)
- **context_precision**: 0.669 (n=12)
- **faithfulness**: 0.972 (n=12)
- **answer_relevancy**: 0.864 (n=12)

## Unsupported claims (faithfulness failures)

- *What is our 2024 win rate for Enterprise, and what...*: "The win rate for Enterprise deals in 2024 is 0.45."