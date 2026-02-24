# Training Datasets + LoRA (BioMistral)

## 1) Build SFT + weak DPO datasets

```bash
python -m train.build_datasets \
  --chunks rag/rag_chunks.jsonl \
  --gold eval/gold.jsonl \
  --results eval/results.jsonl \
  --out_dir train \
  --seed 42 \
  --n_sft 5000 \
  --n_dpo 2000
```

Outputs:
- `train/sft.jsonl`
- `train/dpo.jsonl`
- `train/manifest.json`

## 2) Train LoRA adapter

```bash
python -m train.train_lora \
  --model_id BioMistral/BioMistral-7B \
  --train_file train/sft.jsonl \
  --output_dir train/adapters/biomistral_sft_lora \
  --dtype fp16 \
  --batch_size 2 \
  --grad_accum 8 \
  --lr 2e-4 \
  --epochs 1
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
