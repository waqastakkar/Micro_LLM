"""LoRA SFT training for BioMistral on generated SFT data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sft(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
    return rows


def render_messages(messages: list[dict], tokenizer: AutoTokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    text_parts = []
    for m in messages:
        text_parts.append(f"{m.get('role', 'user').upper()}: {m.get('content', '')}")
    return "\n\n".join(text_parts) + "\n"


def tokenize_rows(rows: list[dict], tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    rendered = [{"text": render_messages(r["messages"], tokenizer)} for r in rows]
    ds = Dataset.from_list(rendered)

    def _tok(batch: dict) -> dict:
        tokenized = tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return ds.map(_tok, batched=True, remove_columns=["text"])


def parse_dtype(dtype: str) -> torch.dtype:
    if dtype.lower() == "fp16":
        return torch.float16
    if dtype.lower() == "bf16":
        return torch.bfloat16
    raise ValueError("dtype must be one of: fp16, bf16")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BioMistral LoRA adapter on SFT dataset.")
    parser.add_argument("--model_id", type=str, default="BioMistral/BioMistral-7B")
    parser.add_argument("--train_file", type=Path, default=Path("train/sft.jsonl"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    torch_dtype = parse_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    rows = load_sft(args.train_file)
    if not rows:
        raise RuntimeError(f"No rows found in {args.train_file}")
    train_ds = tokenize_rows(rows, tokenizer, args.max_length)

    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=args.dtype.lower() == "fp16",
        bf16=args.dtype.lower() == "bf16",
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(model=model, args=train_args, train_dataset=train_ds, tokenizer=tokenizer)
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
