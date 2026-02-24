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

PRESCRIBING_PATTERNS = [
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|µg|units)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:q\d+h|q\d+hr|q\d+ hours|bid|tid|qid|od|once daily|twice daily|three times daily|every \d+ hours)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:iv|intravenous|po|oral|im|subcutaneous)\b", re.IGNORECASE),
    re.compile(r"\b(?:start|give|administer|treat with|prescribe|initiate)\b", re.IGNORECASE),
    re.compile(r"\bfor\s+\d+\s*(?:days|day|weeks|week)\b", re.IGNORECASE),
    re.compile(r"\b(?:loading dose|maintenance dose)\b", re.IGNORECASE),
]


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


def detect_prescribing_language(text: str) -> dict[str, Any]:
    if not text:
        return {"flagged": False, "matches": []}

    matches: list[str] = []
    seen: set[str] = set()
    for pattern in PRESCRIBING_PATTERNS:
        for hit in pattern.finditer(text):
            snippet = hit.group(0).strip()
            key = snippet.lower()
            if snippet and key not in seen:
                seen.add(key)
                matches.append(snippet)

    return {"flagged": bool(matches), "matches": matches}


def enforce_citation_gate_text(
    text: str, allowed_chunk_ids: set[str], min_citations: int = 2
) -> dict[str, Any]:
    metrics = validate_citations(text, allowed_chunk_ids)
    if metrics["invalid_citations"]:
        return {
            "pass": False,
            "reason": f"invalid_citations_present: {metrics['invalid_citations']}",
            "metrics": metrics,
        }
    if metrics["citation_count"] < min_citations:
        return {
            "pass": False,
            "reason": f"insufficient_citations: {metrics['citation_count']} < {min_citations}",
            "metrics": metrics,
        }
    return {"pass": True, "reason": None, "metrics": metrics}


def _extract_evidence_ids(evidence_value: Any) -> list[str]:
    ids: list[str] = []
    if isinstance(evidence_value, list):
        for entry in evidence_value:
            if isinstance(entry, str):
                ids.append(entry)
            elif isinstance(entry, dict):
                chunk_id = entry.get("chunk_id") or entry.get("id")
                if isinstance(chunk_id, str):
                    ids.append(chunk_id)
    return ids


def enforce_citation_gate_json(
    obj: dict, allowed_chunk_ids: set[str], min_evidence_refs: int = 2
) -> dict[str, Any]:
    evidence_fields = [field for field in ("evidence", "supporting_evidence") if field in obj]
    if not evidence_fields:
        return {
            "pass": False,
            "reason": "missing_evidence_fields: expected one of ['evidence', 'supporting_evidence']",
            "metrics": {"evidence_fields_present": [], "evidence_ref_count": 0, "invalid_chunk_ids": []},
        }

    refs: list[str] = []
    for field in evidence_fields:
        refs.extend(_extract_evidence_ids(obj.get(field)))
    invalid = sorted(ref for ref in set(refs) if ref not in allowed_chunk_ids)
    metrics = {
        "evidence_fields_present": evidence_fields,
        "evidence_ref_count": len(refs),
        "invalid_chunk_ids": invalid,
    }
    if invalid:
        return {"pass": False, "reason": f"invalid_evidence_chunk_ids: {invalid}", "metrics": metrics}
    if len(refs) < min_evidence_refs:
        return {
            "pass": False,
            "reason": f"insufficient_evidence_refs: {len(refs)} < {min_evidence_refs}",
            "metrics": metrics,
        }
    return {"pass": True, "reason": None, "metrics": metrics}


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


def make_escalation_override(task: str, reason: str, allowed_chunk_ids: list[str]) -> dict[str, Any]:
    response = safe_failure(task, reason, allowed_chunk_ids)
    schema = SCHEMAS.get(task, STRUCTURED_REPORT_SCHEMA)
    allowed_props = set(schema.get("properties", {}).keys())
    follow_ups = [
        "What are the key patient factors (age, renal/hepatic function, allergies, pregnancy/immunosuppression)?",
        "Can you provide specimen source, timing, and relevant clinical signs/symptoms?",
        "Please share complete organism identification/AST or diagnostic panel details with timestamps.",
        "Should this be escalated now to ID/stewardship for clinician-directed management?",
    ]
    response["escalation_flag"] = True
    notes = response.get("safety_notes")
    if not isinstance(notes, list):
        notes = []
    notes.append(f"escalation_override_reason: {reason}")
    response["safety_notes"] = notes
    if "follow_up_questions" in allowed_props:
        response["follow_up_questions"] = follow_ups[:4]
    return response
