# Manuscript Assets Mapping

## Figure-to-claim mapping

1. **Figure 1 — System architecture (`docs/figures/fig1_system_architecture.svg`)**
   - Claim supported: end-to-end evidence-first clinical microbiology pipeline from PMC OA ingestion to safety-gated output.
   - Key manuscript section: Methods (data engineering + inference architecture).
   - Primary artifacts: `rag/rag_chunks.jsonl`, `eval/results.jsonl`, generated response payloads with citations.

2. **Figure 2 — Safety/validation layer (`docs/figures/fig2_safety_validation_layer.svg`)**
   - Claim supported: safety gates are automatic and auditable.
   - Key manuscript section: Safety framework.
   - Primary artifacts: gate-level logs and evaluation outputs; aggregated in `eval/summary.json` and `eval/error_taxonomy.json`.

3. **Figure 3 — Automated gold generation (`docs/figures/fig3_gold_generation.svg`)**
   - Claim supported: benchmark construction is stratified, reproducible, and includes must-refuse cases.
   - Key manuscript section: Benchmark generation protocol.
   - Primary artifacts: `eval/gold.jsonl`, benchmark manifest metadata, `rag/rag_chunks.jsonl`.

4. **Figure 4 — Training loop (`docs/figures/fig4_training_loop.svg`)**
   - Claim supported: iterative quality improvement loop from failures to adapter re-evaluation.
   - Key manuscript section: Model improvement workflow.
   - Primary artifacts: `eval/results.jsonl`, `eval/failures_top.csv`, `train/sft.train.jsonl`, `train/sft.valid.jsonl`, `train/manifest.json`, adapter directory.

5. **Figure 5 — Metrics overview template (`docs/figures/fig5_metrics_overview.svg`)**
   - Claim supported: side-by-side comparison framework for base vs +LoRA.
   - Key manuscript section: Main outcomes.
   - Primary artifacts: `eval/summary.json` for both base and adapted runs.

6. **Figure 6 — Error taxonomy template (`docs/figures/fig6_error_taxonomy.svg`)**
   - Claim supported: task-specific distribution of safety/evidence failure modes.
   - Key manuscript section: Error analysis.
   - Primary artifacts: `eval/error_taxonomy.json`, `eval/failures_top.csv`.

## Tables to include in manuscript

1. **Corpus summary table**
   - Suggested columns: source count, article count, token/chunk counts, weak-label distribution.
   - Source artifacts: preprocessing outputs + `rag/rag_chunks.jsonl` (+ optional corpus manifest if available).

2. **Benchmark summary table**
   - Suggested columns: task family, sample counts, must-refuse proportion, citation requirements.
   - Source artifacts: `eval/gold.jsonl` and benchmark manifest metadata.

3. **Model comparison table (Base vs +LoRA)**
   - Suggested columns: schema validity, citation compliance, must-refuse compliance, escalation rate, overall pass rate.
   - Source artifacts: paired `eval/summary.json` outputs (base and adapter re-run).

4. **Error taxonomy table**
   - Suggested columns: task, schema_invalid, invalid_citations, no_citations, non_prescribing, must_refuse_failed.
   - Source artifacts: `eval/error_taxonomy.json`, `eval/failures_top.csv`.

## Artifact traceability checklist

- **Data pipeline artifacts:** `rag/rag_chunks.jsonl`
- **Benchmark artifacts:** `eval/gold.jsonl` (+ generation manifest if produced)
- **Evaluation artifacts:** `eval/results.jsonl`, `eval/summary.json`, `eval/report.md`
- **Error artifacts:** `eval/failures_top.csv`, `eval/error_taxonomy.json`
- **Training artifacts:** `train/sft.train.jsonl`, `train/sft.valid.jsonl`, `train/manifest.json`, `train/adapters/biomistral_sft_lora/`

Use fixed seeds and deterministic decoding settings to preserve table/figure reproducibility across runs.
