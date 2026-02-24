"""Generate an automated silver/gold-v0 benchmark from RAG chunks."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from eval.templates_gold import GENERATOR_VERSION, MUST_REFUSE_TEMPLATES, TASK_TEMPLATE_CONFIG

TASK_ORDER = ["organism_id", "ast", "blood_culture", "rapid_dx", "reporting", "stewardship"]
DEFAULT_DIST = {
    "organism_id": 0.20,
    "ast": 0.25,
    "blood_culture": 0.20,
    "rapid_dx": 0.15,
    "reporting": 0.15,
    "stewardship": 0.05,
}

RAPID_DX_TERMS = re.compile(r"\b(pcr|panel|gene|mec[a-z]?|ctx-m|kpc|ndm|oxa-48|van[a-z]|rapid)\b", re.IGNORECASE)
STEWARDSHIP_TERMS = re.compile(r"\b(steward|guideline|de-?escalat|audit|review)\b", re.IGNORECASE)


def _norm_label(chunk: dict[str, Any]) -> str:
    label = str(chunk.get("weak_label") or "").strip().lower()
    if label:
        return label
    text = str(chunk.get("text") or "").lower()
    if any(k in text for k in ["maldi", "gram stain", "morpholog", "identif"]):
        return "identification"
    if any(k in text for k in ["susceptib", "mic", "breakpoint", "ast"]):
        return "ast"
    if "blood culture" in text or "bacteremia" in text:
        return "blood_culture"
    if RAPID_DX_TERMS.search(text):
        return "rapid_dx"
    if any(k in text for k in ["report", "critical value", "corrected"]):
        return "reporting"
    if STEWARDSHIP_TERMS.search(text):
        return "stewardship"
    return "unknown"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_pools(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    pools = {k: [] for k in TASK_ORDER}
    for chunk in chunks:
        label = _norm_label(chunk)
        text = str(chunk.get("text") or "")
        if label in {"identification", "gram_stain", "morphology", "mald", "organism_id"}:
            pools["organism_id"].append(chunk)
        if label == "ast":
            pools["ast"].append(chunk)
        if label == "blood_culture":
            pools["blood_culture"].append(chunk)
        if label in {"resistance", "rapid_dx"} or RAPID_DX_TERMS.search(text):
            pools["rapid_dx"].append(chunk)
        if label == "reporting":
            pools["reporting"].append(chunk)
        if label == "stewardship" or STEWARDSHIP_TERMS.search(text):
            pools["stewardship"].append(chunk)
    return pools


def allocate_counts(n_non_refuse: int, dist: dict[str, float]) -> dict[str, int]:
    raw = {k: n_non_refuse * dist[k] for k in TASK_ORDER}
    counts = {k: int(math.floor(v)) for k, v in raw.items()}
    remainder = n_non_refuse - sum(counts.values())
    ranked = sorted(TASK_ORDER, key=lambda k: (raw[k] - counts[k]), reverse=True)
    for i in range(remainder):
        counts[ranked[i % len(ranked)]] += 1
    return counts


def _safe_chunk_id(chunk: dict[str, Any], idx: int) -> str:
    return str(chunk.get("chunk_id") or f"chunk_auto_{idx:05d}")


def _build_neighbors(anchor: dict[str, Any], by_pmcid: dict[str, list[dict[str, Any]]], rng: random.Random) -> list[str]:
    anchor_id = str(anchor.get("chunk_id"))
    pmcid = anchor.get("pmcid")
    if not pmcid or pmcid not in by_pmcid:
        return [anchor_id]
    sibs = [c for c in by_pmcid[pmcid] if str(c.get("chunk_id")) != anchor_id]
    rng.shuffle(sibs)
    chosen = [anchor_id] + [str(c.get("chunk_id")) for c in sibs[:2]]
    return chosen


def make_item(task: str, chunk: dict[str, Any], idx: int, rng: random.Random, by_pmcid: dict[str, list[dict[str, Any]]], seed: int, min_citations_default: int, ast_min_citations: int) -> dict[str, Any]:
    cfg = TASK_TEMPLATE_CONFIG[task]
    question = rng.choice(cfg["templates"])
    notes = rng.choice(cfg["notes"])
    min_citations = ast_min_citations if task == "ast" else max(min_citations_default, int(cfg["min_citations"]))
    return {
        "id": f"auto_{idx:04d}",
        "task": task,
        "question": question,
        "expected_type": cfg["expected_type"],
        "must_refuse": False,
        "min_citations": min_citations,
        "notes": notes,
        "source_chunk_ids": _build_neighbors(chunk, by_pmcid, rng),
        "seed": seed,
        "generator_version": GENERATOR_VERSION,
    }


def make_must_refuse(idx: int, rng: random.Random, seed: int) -> dict[str, Any]:
    task = rng.choice(TASK_ORDER)
    return {
        "id": f"auto_{idx:04d}",
        "task": task,
        "question": rng.choice(MUST_REFUSE_TEMPLATES) + " Use only the evidence; cite chunk_ids.",
        "expected_type": TASK_TEMPLATE_CONFIG[task]["expected_type"],
        "must_refuse": True,
        "min_citations": 0,
        "notes": "synthetic refusal test: unsafe request or missing critical context",
        "source_chunk_ids": [],
        "seed": seed,
        "generator_version": GENERATOR_VERSION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eval/gold.jsonl from rag chunks.")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n_total", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_citations_default", type=int, default=2)
    parser.add_argument("--ast_min_citations", type=int, default=3)
    parser.add_argument("--must_refuse_ratio", type=float, default=0.15)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    chunks = load_jsonl(args.chunks)
    for i, chunk in enumerate(chunks):
        chunk.setdefault("chunk_id", _safe_chunk_id(chunk, i))

    pools = build_pools(chunks)
    by_pmcid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        if c.get("pmcid"):
            by_pmcid[str(c["pmcid"])].append(c)

    n_refuse = max(1, int(math.ceil(args.n_total * args.must_refuse_ratio)))
    n_non_refuse = max(0, args.n_total - n_refuse)
    targets = allocate_counts(n_non_refuse, DEFAULT_DIST)

    items: list[dict[str, Any]] = []
    idx = 1
    for task in TASK_ORDER:
        pool = pools.get(task, [])
        if not pool:
            # deterministic fallback: reuse full chunk pool
            pool = chunks
        for _ in range(targets[task]):
            chosen = rng.choice(pool)
            items.append(
                make_item(
                    task,
                    chosen,
                    idx,
                    rng,
                    by_pmcid,
                    args.seed,
                    args.min_citations_default,
                    args.ast_min_citations,
                )
            )
            idx += 1

    for _ in range(n_refuse):
        items.append(make_must_refuse(idx, rng, args.seed))
        idx += 1

    rng.shuffle(items)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n", encoding="utf-8")

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "seed": args.seed,
        "n_total": len(items),
        "must_refuse_ratio_target": args.must_refuse_ratio,
        "must_refuse_count": sum(1 for x in items if x["must_refuse"]),
        "must_refuse_ratio_actual": sum(1 for x in items if x["must_refuse"]) / max(1, len(items)),
        "distribution_target": DEFAULT_DIST,
        "distribution_actual": dict(Counter(x["task"] for x in items)),
        "pool_sizes": {k: len(v) for k, v in pools.items()},
        "sampling": {
            "min_citations_default": args.min_citations_default,
            "ast_min_citations": args.ast_min_citations,
        },
        "label_distribution_in_chunks": dict(Counter(_norm_label(c) for c in chunks)),
    }
    manifest_path = args.out.parent / "gold_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
