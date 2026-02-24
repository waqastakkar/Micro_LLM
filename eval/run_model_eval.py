"""CLI runner for Track A model shortlist + decision benchmarking."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from eval.decision import generate_decisions
from eval.prompt_templates import PROMPT_VERSION, build_prompt
from eval.scoring import summarize_results

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # handled gracefully at runtime
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


class BM25Retriever:
    """Minimal lexical BM25 retriever over JSONL chunks."""

    def __init__(self, chunks: List[Dict], text_key: str = "text"):
        self.chunks = chunks
        self.text_key = text_key
        self.doc_tokens: List[List[str]] = []
        self.doc_freq: Counter = Counter()
        self.doc_len: List[int] = []

        for chunk in chunks:
            text = str(chunk.get(text_key, ""))
            toks = self._tokenize(text)
            self.doc_tokens.append(toks)
            self.doc_len.append(len(toks))
            for token in set(toks):
                self.doc_freq[token] += 1

        self.N = len(chunks)
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.k1 = 1.5
        self.b = 0.75

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t.lower() for t in TOKEN_PATTERN.findall(text)]

    def retrieve(self, query: str, top_k: int = 6) -> List[Dict]:
        q_toks = self._tokenize(query)
        if not q_toks or not self.chunks:
            return []

        q_counts = Counter(q_toks)
        scores = []
        for idx, tokens in enumerate(self.doc_tokens):
            tf = Counter(tokens)
            score = 0.0
            dl = self.doc_len[idx] or 1
            for term, qf in q_counts.items():
                df = self.doc_freq.get(term, 0)
                if not df:
                    continue
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
                term_tf = tf.get(term, 0)
                numer = term_tf * (self.k1 + 1)
                denom = term_tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score += idf * ((numer / denom) if denom else 0.0) * qf
            if score > 0:
                scores.append((score, idx))

        scores.sort(reverse=True)
        return [self.chunks[i] for _, i in scores[:top_k]]


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_models(path: Path) -> List[Dict]:
    try:
        import yaml  # lazy import so --help works without PyYAML
    except Exception as exc:
        raise SystemExit("PyYAML is required to read --models YAML. Install with: pip install pyyaml") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data.get("models", [])


def resolve_models(all_models: List[Dict], model_id: str | None) -> List[Dict]:
    if model_id:
        selected = [m for m in all_models if m.get("id") == model_id]
        return selected
    defaults = [m for m in all_models if m.get("default")]
    return defaults or all_models[:1]


def get_device(name: str) -> str:
    if name in {"cpu", "cuda"}:
        return name
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def format_evidence(chunks: List[Dict]) -> Tuple[str, List[str]]:
    lines = []
    ids = []
    for i, chunk in enumerate(chunks):
        chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or f"chunk_{i}")
        ids.append(chunk_id)
        lines.append(f"[{chunk_id}] {chunk.get('text', '')}")
    return "\n".join(lines), ids


def generate(model, tokenizer, prompt: Dict[str, str], max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        raw = f"System: {prompt['system']}\n\nUser: {prompt['user']}\nAssistant:"
        tokenized = tokenizer(raw, return_tensors="pt").input_ids

    tokenized = tokenized.to(model.device)
    output = model.generate(
        tokenized,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output[0][tokenized.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Track A model shortlist evaluation.")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary_out", type=Path, default=Path("eval/summary.json"))
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=320)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    random.seed(args.seed)
    if torch is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    all_models = load_models(args.models)
    chosen_models = resolve_models(all_models, args.model_id)
    if not chosen_models:
        raise SystemExit("No models selected. Check --model_id and models.yaml.")

    gold = load_jsonl(args.gold)
    chunks = load_jsonl(args.chunks)
    retriever = BM25Retriever(chunks)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    device = get_device(args.device)

    for model_cfg in chosen_models:
        model_id = model_cfg.get("id")
        print(f"\n=== Evaluating {model_id} on device={device} ===")

        if AutoTokenizer is None or AutoModelForCausalLM is None:
            print(f"[WARN] transformers/torch not available. Skipping {model_id}.")
            for item in gold:
                results.append(
                    {
                        "model_id": model_id,
                        "item_id": item["id"],
                        "task": item.get("task"),
                        "expected_type": item.get("expected_type"),
                        "must_refuse": bool(item.get("must_refuse", False)),
                        "prompt_version": PROMPT_VERSION,
                        "retrieved_chunk_ids": [],
                        "output": "",
                        "output_repeat": "",
                        "error": "transformers_or_torch_unavailable",
                    }
                )
            continue

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            model.to(device)
            model.eval()
        except Exception as exc:
            print(f"[WARN] Failed to load {model_id}: {exc}")
            for item in gold:
                results.append(
                    {
                        "model_id": model_id,
                        "item_id": item["id"],
                        "task": item.get("task"),
                        "expected_type": item.get("expected_type"),
                        "must_refuse": bool(item.get("must_refuse", False)),
                        "prompt_version": PROMPT_VERSION,
                        "retrieved_chunk_ids": [],
                        "output": "",
                        "output_repeat": "",
                        "error": f"load_failure: {exc}",
                    }
                )
            continue

        for item in gold:
            retrieved = retriever.retrieve(item.get("question", ""), top_k=args.top_k)
            evidence_text, chunk_ids = format_evidence(retrieved)
            prompt = build_prompt(item.get("task", ""), item.get("question", ""), evidence_text)

            try:
                output = generate(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
                output_repeat = generate(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
                error = None
            except Exception as exc:
                output = ""
                output_repeat = ""
                error = f"generation_failure: {exc}"

            results.append(
                {
                    "model_id": model_id,
                    "item_id": item["id"],
                    "task": item.get("task"),
                    "expected_type": item.get("expected_type"),
                    "must_refuse": bool(item.get("must_refuse", False)),
                    "prompt_version": PROMPT_VERSION,
                    "retrieved_chunk_ids": chunk_ids,
                    "output": output,
                    "output_repeat": output_repeat,
                    "error": error,
                }
            )

    with args.out.open("w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = summarize_results(results)
    decisions = generate_decisions(summary)
    payload = {"summary": summary, "decisions": decisions}
    args.summary_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    for model_id, decision in decisions.items():
        print(f"- {model_id}: {decision['recommendation']} | {decision['justification']}")


if __name__ == "__main__":
    main()
