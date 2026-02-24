"""Build SFT + weak DPO datasets from evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from eval.prompt_templates import build_prompt
from eval.validators import enforce_citation_gate_json, enforce_citation_gate_text, parse_json_strict, safe_failure

GENERATOR_VERSION = "train-builder-v1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def _chunk_id(chunk: dict[str, Any], fallback: int) -> str:
    return str(chunk.get("chunk_id") or chunk.get("id") or f"chunk_{fallback:05d}")


def load_chunks(path: Path) -> dict[str, dict[str, Any]]:
    return {_chunk_id(chunk, i): chunk for i, chunk in enumerate(load_jsonl(path))}


def build_evidence_bundle(source_chunk_ids: list[str], chunks_by_id: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    used_ids: list[str] = []
    lines: list[str] = ["EVIDENCE:"]
    for chunk_id in source_chunk_ids or []:
        chunk = chunks_by_id.get(chunk_id)
        if not chunk:
            continue
        used_ids.append(chunk_id)
        lines.append(f"[{chunk_id}] {chunk.get('text', '').strip()}")
    return "\n".join(lines), used_ids


def _safe_text_target(reason: str, chunk_ids: list[str]) -> str:
    cits = " ".join(f"[{cid}]" for cid in chunk_ids[:2])
    return (
        f"INSUFFICIENT EVIDENCE {cits}.\n"
        "I cannot provide prescribing instructions; escalate to a clinician/ID stewardship review.\n"
        "Follow-up questions:\n"
        "1) Can you provide additional microbiology context and timestamps?\n"
        "2) What host factors/allergies/renal-hepatic constraints are relevant?\n"
        "3) Can you share complete susceptibility and source/site details?\n"
        f"Safety reason: {reason}"
    )


def _ensure_safe_target(item: dict[str, Any], candidate: str | None, allowed_ids: list[str]) -> str:
    expected_type = (item.get("expected_type") or "text").lower()
    task = (item.get("task") or "").strip().lower()
    id_set = set(allowed_ids)

    if expected_type == "json":
        if candidate:
            ok, parsed = parse_json_strict(candidate)
            if ok and isinstance(parsed, dict) and enforce_citation_gate_json(parsed, id_set)["pass"]:
                return json.dumps(parsed, ensure_ascii=False)
        return json.dumps(safe_failure(task, "dataset_builder_safe_target", allowed_ids), ensure_ascii=False)

    text = (candidate or "").strip()
    if text and enforce_citation_gate_text(text, id_set)["pass"]:
        return text
    return _safe_text_target("citation_or_quality_guard", allowed_ids)


def _extract_output(result: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _candidate_results(results: list[dict[str, Any]], item_id: str) -> list[dict[str, Any]]:
    rows = [r for r in results if str(r.get("item_id", "")) == item_id]
    bio = [r for r in rows if r.get("model_id") == "BioMistral/BioMistral-7B"]
    return bio or rows


def load_failures(path: Path | None) -> dict[str, int]:
    if path is None or not path.exists():
        return {}
    scores: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item_id = (row.get("item_id") or "").strip()
            if not item_id:
                continue
            try:
                scores[item_id] = max(scores.get(item_id, 0), int(row.get("severity", "0")))
            except ValueError:
                scores[item_id] = scores.get(item_id, 0)
    return scores


def build_datasets(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    chunks_by_id = load_chunks(args.chunks)
    gold = load_jsonl(args.gold)
    results = load_jsonl(args.results)
    failures = load_failures(args.failures)

    sft_rows: list[dict[str, Any]] = []
    dpo_rows: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()

    for item in gold:
        item_id = str(item.get("id"))
        evidence_bundle, allowed_ids = build_evidence_bundle(item.get("source_chunk_ids", []), chunks_by_id)
        prompt = build_prompt(item.get("task", ""), item.get("question", ""), evidence_bundle)

        candidates = _candidate_results(results, item_id)
        chosen_result = candidates[0] if candidates else {}

        final_output = _extract_output(chosen_result, ("final_output", "validated_output", "repaired_output", "output"))
        chosen_text = _ensure_safe_target(item, final_output, allowed_ids)

        sft_rows.append(
            {
                "messages": [
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                    {"role": "assistant", "content": chosen_text},
                ]
            }
        )
        task_counts[str(item.get("task", "unknown"))] += 1

        rejected = _extract_output(chosen_result, ("raw_output", "output_pre_repair", "output_repeat", "output"))
        if not rejected:
            rejected = "INSUFFICIENT EVIDENCE"
        if rejected.strip() != chosen_text.strip():
            dpo_rows.append(
                {
                    "item_id": item_id,
                    "task": item.get("task"),
                    "prompt": f"{prompt['system']}\n\n{prompt['user']}",
                    "chosen": chosen_text,
                    "rejected": rejected,
                    "failure_severity": failures.get(item_id, 0),
                }
            )

    random.shuffle(sft_rows)
    random.shuffle(dpo_rows)
    sft_rows = sft_rows[: args.n_sft]
    dpo_rows = sorted(dpo_rows, key=lambda r: r.get("failure_severity", 0), reverse=True)[: args.n_dpo]
    for row in dpo_rows:
        row.pop("failure_severity", None)

    dump_jsonl(args.out_dir / "sft.jsonl", sft_rows)
    dump_jsonl(args.out_dir / "dpo.jsonl", dpo_rows)

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "seed": args.seed,
        "min_citations_policy": {"text": 2, "json_evidence_refs": 2},
        "inputs": {
            "chunks": str(args.chunks),
            "gold": str(args.gold),
            "results": str(args.results),
            "failures_top": str(args.failures) if args.failures else None,
        },
        "counts": {"gold_items": len(gold), "sft": len(sft_rows), "dpo": len(dpo_rows), "tasks": dict(task_counts)},
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT and weak DPO training datasets.")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("train"))
    parser.add_argument("--failures", type=Path, default=Path("eval/failures_top.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_sft", type=int, default=5000)
    parser.add_argument("--n_dpo", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    build_datasets(parse_args())
