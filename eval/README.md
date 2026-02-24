# Track A — Model shortlist + decision rule (7–8B)

This folder provides a repeatable CLI benchmark for comparing BioMistral (default) against optional 7–8B comparators with evidence-first prompting, constrained outputs, citations, and safety checks.

## Inputs

- `eval/models.yaml`: model shortlist (`provider: hf`, `type: causal_lm`), BioMistral set as `default: true`.
- `eval/gold.jsonl`: evaluation items across tasks (`ast`, `blood_culture`, `rapid_dx`, etc.).
- `rag/rag_chunks.jsonl`: retrieval corpus chunks (JSONL with `text` and optional `chunk_id`).
- `eval/system_prompt.txt`: global safety and evidence-only rules prepended to every request.

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
- `--seed 7` for deterministic runs.
- `--max_new_tokens 320` to cap completion length.
- `--summary_out eval/summary.json` for summary destination.

## Outputs

- `eval/results.jsonl`: one line per `(model, item)` with output, retrieved chunk IDs, schema status, citation metrics, and safety flags.
- `eval/summary.json`: per-model metrics + recommendation.

Metrics include:

- `schema_validity_rate`
- `invalid_citation_rate`
- `uncited_claim_proxy`
- `escalation_rate` (on `must_refuse` items)
- `refusal_correctness`

## Decision meaning

- **retain (RAG-only)**: high schema validity + low invalid citations + high refusal correctness.
- **SFT recommended**: main failures are schema/format/workflow adherence.
- **preference tuning (DPO/ORPO) recommended**: failures are mostly uncited claims, refusal errors, or unsafe behavior.

If a model cannot be loaded, the run continues and records `error` fields for that model instead of crashing.

Step 5 safety behavior:

- Automatic safety gates block non-prescribing violations (dose/regimen language).
- Automatic citation gates enforce valid evidence references and minimum citation/evidence count.
- Gates may convert otherwise answer-like output into escalation/safe-failure responses for clinical safety.
