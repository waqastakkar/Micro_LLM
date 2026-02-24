# Track A — Model shortlist + decision rule (7–8B)

This folder adds a repeatable CLI benchmark for comparing BioMistral (default) against optional 7–8B comparators under the same RAG + prompt + guardrail setup.

## Inputs

- `eval/models.yaml`: model shortlist (`provider: hf`, `type: causal_lm`), BioMistral set as `default: true`.
- `eval/gold.jsonl`: evaluation items across tasks (`ast`, `blood_culture`, `rapid_dx`, etc.).
- `rag/rag_chunks.jsonl`: retrieval corpus chunks (JSONL with `text` and optional `chunk_id`).

## Run

```bash
python -m eval.run_model_eval \
  --models eval/models.yaml \
  --gold eval/gold.jsonl \
  --chunks rag/rag_chunks.jsonl \
  --top_k 6 \
  --out eval/results.jsonl
```

Optional flags:

- `--model_id <hf_model_id>` to run a specific model.
- `--device auto|cpu|cuda` (default `auto`).
- `--seed 7` for deterministic runs.
- `--max_new_tokens 320` to cap completion length.
- `--summary_out eval/summary.json` for summary destination.

## Outputs

- `eval/results.jsonl`: one line per `(model, item)` with output, repeat output, prompt version, and retrieved chunk IDs.
- `eval/summary.json`: per-model metrics + recommendation.

Metrics include:

- `schema_validity` (JSON tasks parse success)
- `citation_coverage` (% of sentences containing `[chunk_id]`)
- `refusal_correctness` (for `must_refuse` items)
- `hallucination_proxy_uncited_claims`
- `consistency_exact_match` and `consistency_similarity`

## Decision meaning

- **retain (RAG-only)**: grounded/safe/consistent enough to deploy without extra tuning.
- **SFT recommended**: main failures are schema/format/workflow adherence.
- **preference tuning (DPO/ORPO) recommended**: failures are mostly unsafe or uncited overconfident behavior.

If a model cannot be loaded, the run continues and records `error` fields for that model instead of crashing.
