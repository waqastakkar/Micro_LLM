"""Validate generated eval/gold.jsonl format and coverage constraints."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

TASKS = {"organism_id", "ast", "blood_culture", "rapid_dx", "reporting", "stewardship"}
REQUIRED = {
    "id",
    "task",
    "question",
    "expected_type",
    "must_refuse",
    "min_citations",
    "notes",
    "source_chunk_ids",
    "seed",
    "generator_version",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate auto-generated gold benchmark")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--must_refuse_min_ratio", type=float, default=0.10)
    args = parser.parse_args()

    rows = load_jsonl(args.gold)
    assert rows, "gold file is empty"

    ids = [r.get("id") for r in rows]
    assert len(ids) == len(set(ids)), "ids must be unique"

    task_counts = Counter()
    refuse_count = 0
    for row in rows:
        missing = REQUIRED - set(row.keys())
        assert not missing, f"missing required fields for {row.get('id')}: {sorted(missing)}"
        assert row["task"] in TASKS, f"invalid task {row['task']}"
        assert row["expected_type"] in {"json", "text"}, f"invalid expected_type {row['expected_type']}"
        assert isinstance(row["source_chunk_ids"], list), "source_chunk_ids must be list"
        if not row["must_refuse"]:
            assert len(row["source_chunk_ids"]) >= 1, f"non-refusal item must include source_chunk_ids: {row['id']}"
            assert row["min_citations"] >= 1, f"non-refusal item must require citations: {row['id']}"
        task_counts[row["task"]] += 1
        refuse_count += int(bool(row["must_refuse"]))

    for task in sorted(TASKS):
        assert task_counts[task] > 0, f"missing task coverage: {task}"

    refuse_ratio = refuse_count / len(rows)
    assert refuse_ratio >= args.must_refuse_min_ratio, (
        f"must_refuse ratio {refuse_ratio:.3f} below required minimum {args.must_refuse_min_ratio:.3f}"
    )

    print(
        json.dumps(
            {
                "ok": True,
                "n_items": len(rows),
                "task_counts": dict(task_counts),
                "must_refuse_count": refuse_count,
                "must_refuse_ratio": refuse_ratio,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
