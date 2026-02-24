# Clinical Microbiology LLM (BioMistral + Evidence-First RAG)

**Non-prescribing safety statement:** This repository is for **research and quality-improvement** workflows only. It is **not** a prescribing system and does **not** replace clinician judgment, local policy, or specialist consultation.

## **Primary aims**
- **Build** a traceable clinical microbiology RAG corpus from PMC OA sources.
- **Generate** a structured benchmark automatically from corpus chunks.
- **Evaluate** **BioMistral/BioMistral-7B** in an evidence-first, citation-constrained setup.
- **Enforce** automatic safety gates for schema validity, citations, and non-prescribing behavior.
- **Analyze** failures into actionable error taxonomy outputs for iteration.
- **Improve** behavior with SFT LoRA and compare against base performance.

## **Safety commitments**
- **Non-prescribing:** no direct treatment/prescription directives.
- **Evidence-only:** claims must be grounded in retrievable chunk evidence.
- **Escalation:** if evidence is insufficient or unsafe, produce safe-failure/escalation output.

## **Quickstart (GPU FP16)**
Use this minimal path to produce a benchmark and evaluate base BioMistral on a **48GB VRAM CUDA** setup.

```bash
python -m eval.generate_gold \
  --chunks rag/rag_chunks.jsonl \
  --out eval/gold.jsonl \
  --n_total 200 \
  --seed 42
```

```bash
python -m eval.run_model_eval \
  --models eval/models.yaml \
  --gold eval/gold.jsonl \
  --chunks rag/rag_chunks.jsonl \
  --top_k 6 \
  --device cuda \
  --dtype fp16 \
  --max_new_tokens 512 \
  --out eval/results.jsonl
```

## **Steps 1–8 (reproducible pipeline)**

### **Step 1 — Download PMC OA XML**
```bash
python 01_pmc_oa_download.py
```

### **Step 2 — Convert XML to clean JSONL**
```bash
python 02_pmc_xml_to_clean_text.py
```

### **Step 3 — Chunk + weak label to build RAG chunks**
```bash
python 03_chunk_and_label.py
```

### **Step 4 — Generate gold benchmark automatically + evaluate base BioMistral (FP16 CUDA)**
```bash
python -m eval.generate_gold \
  --chunks rag/rag_chunks.jsonl \
  --out eval/gold.jsonl \
  --n_total 200 \
  --seed 42
```

```bash
python -m eval.run_model_eval \
  --models eval/models.yaml \
  --gold eval/gold.jsonl \
  --chunks rag/rag_chunks.jsonl \
  --top_k 6 \
  --device cuda \
  --dtype fp16 \
  --max_new_tokens 512 \
  --out eval/results.jsonl
```

### **Step 5 — Safety layer (automatic gates)**
Safety gates are **automatic** during evaluation/inference:
- **JSON schema validator**
- **Citation gate**
- **Non-prescribing filter**
- **Safe-failure override**

### **Step 6 — Validate gold benchmark**
```bash
python -m eval.validate_gold --gold eval/gold.jsonl
```

### **Step 7 — Analyze results and export reports**
```bash
python -m eval.analyze_results \
  --results eval/results.jsonl \
  --gold eval/gold.jsonl \
  --out_dir eval
```

### **Step 8 — Build SFT dataset + train LoRA + re-evaluate with adapter**
```bash
python -m train.build_datasets \
  --chunks rag/rag_chunks.jsonl \
  --gold eval/gold.jsonl \
  --results eval/results.jsonl \
  --failures eval/failures_top.csv \
  --out_dir train \
  --seed 42 \
  --n_sft 12000 \
  --failures_oversample 3 \
  --must_refuse_ratio 0.15
```

```bash
python -m train.train_lora \
  --model_id BioMistral/BioMistral-7B \
  --train_file train/sft.train.jsonl \
  --valid_file train/sft.valid.jsonl \
  --output_dir train/adapters/biomistral_sft_lora \
  --dtype fp16 \
  --batch_size 2 \
  --grad_accum 8 \
  --lr 2e-4 \
  --epochs 1 \
  --max_seq_len 4096
```

```bash
python -m eval.run_model_eval \
  --lora_path train/adapters/biomistral_sft_lora \
  --gold eval/gold.jsonl \
  --chunks rag/rag_chunks.jsonl \
  --device cuda \
  --dtype fp16
```

## **Outputs & artifacts**
- **Step 1:** PMC OA XML corpus (download directory from `01_pmc_oa_download.py`).
- **Step 2:** Cleaned article JSONL (from `02_pmc_xml_to_clean_text.py`).
- **Step 3:** `rag/rag_chunks.jsonl`.
- **Step 4:** `eval/gold.jsonl`, `eval/results.jsonl`.
- **Step 5:** Safety-gated response fields and gate events embedded/logged by evaluation stack.
- **Step 6:** Gold validation pass/fail report in terminal and validator outputs.
- **Step 7:** `eval/summary.json`, `eval/report.md`, `eval/failures_top.csv`, `eval/error_taxonomy.json`.
- **Step 8:** `train/sft.train.jsonl`, `train/sft.valid.jsonl`, `train/manifest.json`, `train/adapters/biomistral_sft_lora/`, plus re-evaluation outputs.

## **Manuscript-ready figures**
- `docs/figures/fig1_system_architecture.svg`
- `docs/figures/fig2_safety_validation_layer.svg`
- `docs/figures/fig3_gold_generation.svg`
- `docs/figures/fig4_training_loop.svg`
- `docs/figures/fig5_metrics_overview.svg`
- `docs/figures/fig6_error_taxonomy.svg`

See also: `docs/manuscript_assets.md`.

## **Manuscript assets**
Publication mapping, table plans, and artifact-to-figure traceability are documented in:
- `docs/manuscript_assets.md`
- `docs/figures/`

## **Reproducibility**
- Fix seeds (e.g., `--seed 42`) for dataset and benchmark generation.
- Use deterministic decoding for evaluations (**`temperature=0`**, **`do_sample=False`** in evaluator/model config).
- Keep benchmark input fixed (`eval/gold.jsonl`) when comparing base vs adapter runs.
- Pin model identity (**BioMistral/BioMistral-7B**) and runtime dtype/device settings.

## **Decision rule (retain vs SFT vs DPO/ORPO)**
- **Retain base** if safety and evidence metrics already meet target thresholds with acceptable error profile.
- **Use SFT (current default)** when failures are primarily format/safety-compliance and citation-grounding issues.
- **Escalate to DPO/ORPO** only if preference-level behavior (e.g., nuanced refusal style or ranking-sensitive quality) remains suboptimal after SFT.
