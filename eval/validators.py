"""Validation helpers for evidence-linked constrained outputs."""

from __future__ import annotations

import json
import re
from typing import Any

from eval.schemas import SCHEMAS, STRUCTURED_REPORT_SCHEMA

try:
    import jsonschema
except Exception:  # pragma: no cover
    jsonschema = None

CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")


def extract_citations(text: str) -> set[str]:
    return set(CITATION_PATTERN.findall(text or ""))


def _collect_strings(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, list):
        for item in value:
            out.extend(_collect_strings(item))
    elif isinstance(value, dict):
        for item in value.values():
            out.extend(_collect_strings(item))
    return out


def validate_citations(text_or_obj: Any, allowed_chunk_ids: set[str]) -> dict[str, Any]:
    strings = _collect_strings(text_or_obj)
    cited: set[str] = set()
    for s in strings:
        cited.update(extract_citations(s))

    invalid = sorted(cid for cid in cited if cid not in allowed_chunk_ids)
    return {
        "cited_ids": sorted(cited),
        "invalid_citations": invalid,
        "citation_count": len(cited),
        "has_any_citations": bool(cited),
    }


def parse_json_strict(text: str) -> tuple[bool, Any]:
    raw = (text or "").strip()
    if not raw:
        return False, "empty_output"

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[idx:])
            if isinstance(obj, dict):
                return True, obj
        except json.JSONDecodeError:
            continue

    return False, "no_valid_json_object_found"


def _minimal_validate(schema: dict[str, Any], obj: dict[str, Any]) -> tuple[bool, str | None]:
    required = schema.get("required", [])
    missing = [k for k in required if k not in obj]
    if missing:
        return False, f"missing_required_fields: {missing}"

    if schema.get("additionalProperties") is False:
        allowed = set(schema.get("properties", {}).keys())
        extras = [k for k in obj.keys() if k not in allowed]
        if extras:
            return False, f"unexpected_fields: {extras}"

    return True, None


def validate_schema(schema: dict, obj: dict) -> tuple[bool, str | None]:
    if jsonschema is not None:
        try:
            jsonschema.validate(instance=obj, schema=schema)
            return True, None
        except Exception as exc:
            return False, str(exc)

    return _minimal_validate(schema, obj)


def safe_failure(task: str, reason: str, allowed_chunk_ids: list[str]) -> dict[str, Any]:
    safe_note = (
        "No autonomous prescribing. Escalate to clinician and antimicrobial stewardship for decisions."
    )
    evidence = allowed_chunk_ids[:3]

    if task == "ast":
        return {
            "task": "ast",
            "organism": "unknown",
            "isolate_site": None,
            "antibiotic_results": [],
            "overall_summary": "INSUFFICIENT EVIDENCE",
            "confidence": 0.0,
            "escalation_flag": True,
            "follow_up_questions": [
                "Please provide organism identification, full AST panel, and source/site context.",
            ],
            "safety_notes": [safe_note, f"safe_failure_reason: {reason}"],
        }

    if task == "blood_culture":
        return {
            "task": "blood_culture",
            "likely_contaminant": "uncertain",
            "rationale": "INSUFFICIENT EVIDENCE",
            "organism": None,
            "supporting_evidence": evidence,
            "recommended_next_steps": [
                "Correlate with repeat cultures and clinical status.",
                "Escalate to treating team for management decisions.",
            ],
            "confidence": 0.0,
            "escalation_flag": True,
            "follow_up_questions": ["Please provide bottle/set count, timing, symptoms, and source details."],
            "safety_notes": [safe_note, f"safe_failure_reason: {reason}"],
        }

    if task == "reporting":
        return {
            "task": "reporting",
            "report_type": "routine",
            "draft_report": "INSUFFICIENT EVIDENCE",
            "critical_value_message_template": None,
            "supporting_evidence": evidence,
            "escalation_flag": True,
            "safety_notes": [safe_note, f"safe_failure_reason: {reason}"],
        }

    schema = SCHEMAS.get(task, STRUCTURED_REPORT_SCHEMA)
    fallback = {k: None for k in schema.get("required", [])}
    fallback["task"] = task
    fallback["escalation_flag"] = True
    fallback["safety_notes"] = [safe_note, f"safe_failure_reason: {reason}"]
    return fallback
