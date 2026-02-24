"""Generate markdown evaluation report from analysis artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from eval.decision import generate_decisions


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return "\n".join(out)


def load_failures(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_report(summary: dict[str, Any], failures: list[dict[str, str]]) -> str:
    lines: list[str] = ["# Evaluation Report", "", "## Overview", "", f"Models analyzed: **{len(summary)}**."]

    overall_headers = [
        "model_id",
        "schema_validity_rate",
        "invalid_citation_rate",
        "has_any_citations_rate",
        "citation_gate_trigger_rate",
        "non_prescribing_gate_rate",
        "must_refuse_compliance_rate",
        "escalation_rate",
    ]
    overall_rows = []
    decision_input = {}
    for model_id, payload in summary.items():
        overall = payload.get("overall", {})
        overall_rows.append([model_id] + [_fmt(overall.get(k)) for k in overall_headers[1:]])
        decision_input[model_id] = {
            "schema_validity_rate": overall.get("schema_validity_rate"),
            "invalid_citation_rate": overall.get("invalid_citation_rate"),
            "uncited_claim_proxy": (1 - overall["has_any_citations_rate"]) if overall.get("has_any_citations_rate") is not None else 1.0,
            "refusal_correctness": overall.get("must_refuse_compliance_rate"),
        }

    lines += ["", "## Per-model overall metrics", "", _table(overall_headers, overall_rows)]

    lines += ["", "## Per-task breakdown", ""]
    task_headers = [
        "task",
        "schema_validity_rate",
        "invalid_citation_rate",
        "has_any_citations_rate",
        "must_refuse_compliance_rate",
        "citation_gate_trigger_rate",
        "non_prescribing_gate_rate",
    ]
    for model_id, payload in summary.items():
        lines.append(f"### {model_id}")
        rows = []
        for task, task_metrics in payload.get("by_task", {}).items():
            rows.append([task] + [_fmt(task_metrics.get(k)) for k in task_headers[1:]])
        lines += ["", _table(task_headers, rows), ""]

    lines += ["## Safety summary", ""]
    safety_headers = ["model_id", "must_refuse_compliance_rate", "non_prescribing_gate_rate", "citation_gate_trigger_rate"]
    safety_rows = []
    for model_id, payload in summary.items():
        overall = payload.get("overall", {})
        safety_rows.append(
            [
                model_id,
                _fmt(overall.get("must_refuse_compliance_rate")),
                _fmt(overall.get("non_prescribing_gate_rate")),
                _fmt(overall.get("citation_gate_trigger_rate")),
            ]
        )
    lines += [_table(safety_headers, safety_rows), ""]

    lines += ["## Top 10 failure examples", ""]
    top_10 = failures[:10]
    if top_10:
        fail_headers = ["model_id", "item_id", "task", "severity", "safety_gate_type", "question", "answer"]
        fail_rows = [
            [
                r.get("model_id", ""),
                r.get("item_id", ""),
                r.get("task", ""),
                r.get("severity", ""),
                r.get("safety_gate_type", ""),
                r.get("excerpt_question", ""),
                r.get("excerpt_answer", ""),
            ]
            for r in top_10
        ]
        lines.append(_table(fail_headers, fail_rows))
    else:
        lines.append("No failures detected.")

    decisions = generate_decisions(decision_input)
    lines += ["", "## Recommendations", ""]
    for model_id, dec in decisions.items():
        lines.append(f"- **{model_id}**: **{dec['recommendation']}** — {dec['justification']}")

    biomistral = next((m for m in decisions if "biomistral" in m.lower()), None)
    if biomistral:
        lines += [
            "",
            "### BioMistral recommendation",
            f"**{decisions[biomistral]['recommendation']}** for `{biomistral}` based on current summary metrics.",
        ]

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate markdown eval report.")
    parser.add_argument("--summary", type=Path, default=Path("eval/summary.json"))
    parser.add_argument("--failures", type=Path, default=Path("eval/failures_top.csv"))
    parser.add_argument("--out", type=Path, default=Path("eval/report.md"))
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    failures = load_failures(args.failures)
    report = build_report(summary, failures)
    args.out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
