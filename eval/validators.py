"""Validation and repair helpers for structured model outputs."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Iterable, Tuple

try:
    import jsonschema
except Exception:  # pragma: no cover
    jsonschema = None

CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")


def _extract_json_block(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def validate_json(schema: Dict[str, Any], text: str) -> Tuple[bool, Any]:
    """Validate JSON text against a schema and return (ok, obj|error)."""
    candidate = _extract_json_block(text)
    try:
        obj = json.loads(candidate)
    except Exception as exc:
        return False, f"json_parse_error: {exc}"

    if jsonschema is None:
        return True, obj

    try:
        jsonschema.validate(instance=obj, schema=schema)
    except Exception as exc:
        return False, f"schema_validation_error: {exc}"
    return True, obj


def repair_json(schema: Dict[str, Any], raw_text: str, model_call_fn: Callable[[str], str]) -> Dict[str, Any]:
    """Attempt one schema-aware repair call and return valid object or safe-failure object."""
    repair_prompt = (
        "Return JSON ONLY that satisfies this schema. No markdown, no extra keys.\n"
        f"SCHEMA: {json.dumps(schema, ensure_ascii=False)}\n"
        f"RAW_OUTPUT: {raw_text}\n"
    )
    repaired = model_call_fn(repair_prompt)
    ok, obj_or_error = validate_json(schema, repaired)
    if ok:
        return obj_or_error

    return {
        "safe_failure": True,
        "error": str(obj_or_error),
        "message": "INSUFFICIENT EVIDENCE",
        "needs_escalation": True,
    }


def _collect_citations(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        found.extend(CITATION_PATTERN.findall(value))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_citations(item))
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(_collect_citations(item))
    return found


def validate_citations(obj_or_text: Any, allowed_chunk_ids: Iterable[str]) -> Dict[str, Any]:
    """Compute citation coverage and out-of-bundle citation stats."""
    allowed = set(allowed_chunk_ids)
    citations = _collect_citations(obj_or_text)
    unique = sorted(set(citations))
    invalid = sorted(c for c in unique if c not in allowed)

    return {
        "citation_count": len(citations),
        "unique_citation_count": len(unique),
        "unique_citations": unique,
        "invalid_citation_count": len(invalid),
        "invalid_citations": invalid,
        "all_citations_allowed": len(invalid) == 0,
    }
