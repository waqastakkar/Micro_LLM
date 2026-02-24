# Training Datasets + LoRA (BioMistral)

## 1) Build SFT + weak DPO datasets

```bash
python -m train.build_datasets \
  --chunks rag/rag_chunks.jsonl \
  --gold eval/gold.jsonl \
  --results eval/results.jsonl \
  --out_dir train \
  --failures eval/failures_top.csv \
  --failures_oversample 3 \
  --task_mix organism_id=0.20,ast=0.25,blood_culture=0.20,rapid_dx=0.15,reporting=0.15,stewardship=0.05 \
  --must_refuse_ratio 0.15 \
  --seed 42 \
  --n_sft 5000 \
  --n_dpo 2000
```

Outputs:
- `train/sft.train.jsonl`
- `train/sft.valid.jsonl`
- `train/dpo.jsonl`
- `train/manifest.json`

## 2) Train LoRA adapter

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
  --max_seq_len 4096 \
  --grad_ckpt true
```

## 3) Run eval with adapter

`eval/run_model_eval.py` supports optional `--lora_path`.

```bash
python -m eval.run_model_eval \
  --models eval/models.yaml \
  --gold eval/gold.jsonl \
  --chunks rag/rag_chunks.jsonl \
  --out eval/results_lora.jsonl \
  --summary_out eval/summary_lora.json \
  --model_id BioMistral/BioMistral-7B \
  --lora_path train/adapters/biomistral_sft_lora
```

When `--lora_path` is provided, the base model is loaded first and the PEFT adapter is applied before generation.
