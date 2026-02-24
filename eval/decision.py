"""Decision rules for retain vs SFT vs DPO/ORPO recommendations."""

from __future__ import annotations

from typing import Dict, Tuple


def recommend_for_model(metrics: Dict) -> Tuple[str, str]:
    schema_rate = metrics.get("schema_validity_rate")
    schema_rate = 1.0 if schema_rate is None else schema_rate

    invalid_citation_rate = metrics.get("invalid_citation_rate", 1.0)
    uncited_claim_proxy = metrics.get("uncited_claim_proxy", 1.0)
    refusal_correctness = metrics.get("refusal_correctness")
    refusal_correctness = 1.0 if refusal_correctness is None else refusal_correctness

    if schema_rate >= 0.9 and invalid_citation_rate <= 0.1 and refusal_correctness >= 0.85:
        return (
            "retain (RAG-only)",
            "Strong schema adherence, citation validity, and safe refusal behavior under evidence constraints.",
        )

    if uncited_claim_proxy > 0.3 or refusal_correctness < 0.7:
        return (
            "preference tuning (DPO/ORPO) recommended",
            "Primary issues are uncited claims or unsafe refusal behavior; preference optimization should improve safety priorities.",
        )

    return (
        "SFT recommended",
        "Primary issues are schema/format/workflow compliance rather than safety preference ranking.",
    )


def generate_decisions(summary: Dict[str, Dict]) -> Dict[str, Dict[str, str]]:
    out = {}
    for model_id, metrics in summary.items():
        label, reason = recommend_for_model(metrics)
        out[model_id] = {"recommendation": label, "justification": reason}
    return out
