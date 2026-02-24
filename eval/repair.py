"""Single-pass repair utilities for constrained JSON outputs."""

from __future__ import annotations

import json
from typing import Any, Callable

from eval.validators import parse_json_strict, safe_failure, validate_schema


def _schema_outline(schema: dict[str, Any]) -> str:
    props = schema.get("properties", {})
    required = schema.get("required", [])
    lines = [f"required={required}"]
    for key in required:
        p = props.get(key, {})
        p_type = p.get("type") or p.get("const") or p.get("enum")
        lines.append(f"- {key}: {p_type}")
    return "\n".join(lines)


def repair_to_valid_json(
    task: str,
    schema: dict,
    raw_text: str,
    model_generate_fn: Callable[[str], str],
    evidence_bundle: list[dict[str, Any]],
) -> dict[str, Any]:
    allowed_chunk_ids = [str(c.get("chunk_id") or c.get("id") or "") for c in evidence_bundle]
    allowed_chunk_ids = [c for c in allowed_chunk_ids if c]

    repair_prompt = (
        "Convert the following into VALID JSON ONLY matching this schema. "
        "Do not add any extra keys.\n\n"
        f"TASK: {task}\n"
        f"SCHEMA SUMMARY:\n{_schema_outline(schema)}\n\n"
        f"ALLOWED CHUNK IDS: {allowed_chunk_ids}\n"
        "If information is missing, use INSUFFICIENT EVIDENCE and escalation_flag=true.\n"
        "RAW MODEL OUTPUT:\n"
        f"{raw_text}\n"
    )

    repaired_text = model_generate_fn(repair_prompt)
    ok_json, obj_or_err = parse_json_strict(repaired_text)
    if not ok_json:
        return safe_failure(task, f"repair_parse_failed: {obj_or_err}", allowed_chunk_ids)

    ok_schema, schema_err = validate_schema(schema, obj_or_err)
    if not ok_schema:
        return safe_failure(task, f"repair_schema_failed: {schema_err}", allowed_chunk_ids)

    return obj_or_err
