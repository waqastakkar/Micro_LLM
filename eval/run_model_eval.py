"""CLI runner for Track A model shortlist + decision benchmarking."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from eval.decision import generate_decisions
from eval.prompt_templates import PROMPT_VERSION, build_prompt
from eval.repair import repair_to_valid_json
from eval.schemas import SCHEMAS
from eval.scoring import summarize_results
from eval.validators import parse_json_strict, safe_failure, validate_citations, validate_schema

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # handled gracefully at runtime
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


class BM25Retriever:
    def __init__(self, chunks: list[dict], text_key: str = "text"):
        self.chunks = chunks
        self.text_key = text_key
        self.doc_tokens: list[list[str]] = []
        self.doc_freq: Counter = Counter()
        self.doc_len: list[int] = []

        for chunk in chunks:
            toks = self._tokenize(str(chunk.get(text_key, "")))
            self.doc_tokens.append(toks)
            self.doc_len.append(len(toks))
            for token in set(toks):
                self.doc_freq[token] += 1

        self.N = len(chunks)
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.k1 = 1.5
        self.b = 0.75

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t.lower() for t in TOKEN_PATTERN.findall(text)]

    def retrieve(self, query: str, top_k: int = 6) -> list[dict]:
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


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_models(path: Path) -> list[dict]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")).get("models", [])


def resolve_models(all_models: list[dict], model_id: str | None) -> list[dict]:
    if model_id:
        return [m for m in all_models if m.get("id") == model_id]
    defaults = [m for m in all_models if m.get("default")]
    return defaults or all_models[:1]


def format_evidence(chunks: list[dict]) -> tuple[str, list[str]]:
    lines, ids = [], []
    for i, chunk in enumerate(chunks):
        chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or f"chunk_{i}")
        ids.append(chunk_id)
        lines.append(f"[{chunk_id}] {chunk.get('text', '')}")
    return "\n".join(lines), ids


def _extract_escalation_and_safety(obj: Any) -> tuple[bool | None, bool]:
    if isinstance(obj, dict):
        escalation = obj.get("escalation_flag") if isinstance(obj.get("escalation_flag"), bool) else None
        safety_notes = obj.get("safety_notes")
        return escalation, isinstance(safety_notes, list) and len(safety_notes) > 0
    return None, False


def generate(model, tokenizer, prompt: dict[str, str], max_new_tokens: int, seed: int) -> str:
    messages = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}]

    if hasattr(tokenizer, "apply_chat_template"):
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    else:
        raw = f"System: {prompt['system']}\n\nUser: {prompt['user']}\nAssistant:"
        inputs = tokenizer(raw, return_tensors="pt").input_ids

    inputs = inputs.to(model.device)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    with torch.inference_mode():
        output = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs.shape[-1] :]
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
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    if torch is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    models = resolve_models(load_models(args.models), args.model_id)
    gold = load_jsonl(args.gold)
    chunks = load_jsonl(args.chunks)
    retriever = BM25Retriever(chunks)

    results: list[dict] = []
    for model_cfg in models:
        model_id = model_cfg["id"]

        if AutoTokenizer is None or AutoModelForCausalLM is None:
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
                        "schema_ok": False,
                        "schema_error": "transformers_or_torch_unavailable",
                        "citation_count": 0,
                        "invalid_citations": [],
                        "has_any_citations": False,
                        "escalation_flag": None,
                        "safety_notes_present": False,
                        "error": "transformers_or_torch_unavailable",
                    }
                )
            continue

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", low_cpu_mem_usage=True)
        model.eval()

        for item in gold:
            task = (item.get("task") or "").strip().lower()
            schema = SCHEMAS.get(task)
            expected_type = (item.get("expected_type") or "text").lower()

            retrieved = retriever.retrieve(item.get("question", ""), top_k=args.top_k)
            evidence_text, chunk_ids = format_evidence(retrieved)
            prompt = build_prompt(task, item.get("question", ""), evidence_text)

            output = generate(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, seed=args.seed)
            record_obj: Any = output
            schema_ok = expected_type != "json"
            schema_error = None

            if expected_type == "json" and schema is not None:
                ok_json, parsed_or_err = parse_json_strict(output)
                if ok_json:
                    ok_schema, err = validate_schema(schema, parsed_or_err)
                    if ok_schema:
                        record_obj = parsed_or_err
                        schema_ok = True
                    else:
                        repaired = repair_to_valid_json(
                            task,
                            schema,
                            output,
                            lambda rp: generate(model, tokenizer, {"system": prompt["system"], "user": rp}, args.max_new_tokens, args.seed),
                            retrieved,
                        )
                        ok_schema, err = validate_schema(schema, repaired)
                        record_obj = repaired
                        schema_ok = ok_schema
                        schema_error = None if ok_schema else err
                else:
                    repaired = repair_to_valid_json(
                        task,
                        schema,
                        output,
                        lambda rp: generate(model, tokenizer, {"system": prompt["system"], "user": rp}, args.max_new_tokens, args.seed),
                        retrieved,
                    )
                    ok_schema, err = validate_schema(schema, repaired)
                    record_obj = repaired if ok_schema else safe_failure(task, str(parsed_or_err), chunk_ids)
                    schema_ok = ok_schema
                    schema_error = None if ok_schema else err

            citations = validate_citations(record_obj, set(chunk_ids))
            escalation_flag, safety_notes_present = _extract_escalation_and_safety(record_obj)

            results.append(
                {
                    "model_id": model_id,
                    "item_id": item["id"],
                    "task": task,
                    "expected_type": expected_type,
                    "must_refuse": bool(item.get("must_refuse", False)),
                    "prompt_version": PROMPT_VERSION,
                    "retrieved_chunk_ids": chunk_ids,
                    "output": json.dumps(record_obj, ensure_ascii=False) if isinstance(record_obj, dict) else str(record_obj),
                    "schema_ok": bool(schema_ok),
                    "schema_error": schema_error,
                    "citation_count": citations["citation_count"],
                    "invalid_citations": citations["invalid_citations"],
                    "has_any_citations": citations["has_any_citations"],
                    "escalation_flag": escalation_flag,
                    "safety_notes_present": safety_notes_present,
                    "error": None,
                }
            )

    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n", encoding="utf-8")
    summary = summarize_results(results)
    decisions = generate_decisions(summary)
    args.summary_out.write_text(json.dumps({"summary": summary, "decisions": decisions}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
