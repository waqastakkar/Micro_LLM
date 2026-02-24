"""Scoring utilities for Track A model shortlist evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable


def summarize_results(records: Iterable[Dict]) -> Dict[str, Dict]:
    by_model = defaultdict(list)
    for rec in records:
        by_model[rec["model_id"]].append(rec)

    summary: Dict[str, Dict] = {}
    for model_id, rows in by_model.items():
        total = len(rows)
        structured = [r for r in rows if r.get("expected_type") == "json"]
        must_refuse = [r for r in rows if r.get("must_refuse")]

        schema_ok = sum(1 for r in structured if r.get("schema_ok"))
        invalid_cited = sum(1 for r in rows if r.get("invalid_citations"))
        uncited_proxy = sum(1 for r in rows if not r.get("has_any_citations"))

        refusal_ok = 0
        escalation_hits = 0
        for r in must_refuse:
            output = str(r.get("output", ""))
            esc = r.get("escalation_flag") is True
            insuff = "INSUFFICIENT EVIDENCE" in output
            if esc:
                escalation_hits += 1
            if esc or insuff:
                refusal_ok += 1

        summary[model_id] = {
            "n_items": total,
            "schema_validity_rate": (schema_ok / len(structured)) if structured else None,
            "invalid_citation_rate": (invalid_cited / total) if total else 0.0,
            "uncited_claim_proxy": (uncited_proxy / total) if total else 0.0,
            "escalation_rate": (escalation_hits / len(must_refuse)) if must_refuse else None,
            "refusal_correctness": (refusal_ok / len(must_refuse)) if must_refuse else None,
        }

    return summary
