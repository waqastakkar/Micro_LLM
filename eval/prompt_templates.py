"""Prompt templates used by Track A model benchmarking."""

from __future__ import annotations

import json
from typing import Dict

from eval.schemas import (
    AST_INTERPRETATION_SCHEMA,
    BLOOD_CULTURE_ASSESSMENT_SCHEMA,
    STRUCTURED_REPORT_DRAFT_SCHEMA,
)

PROMPT_VERSION = "v1.1"

SYSTEM_RULES = (
    "Follow the system prompt guardrails exactly.\n"
    "For structured tasks, output JSON ONLY and satisfy the requested schema."
)

GENERAL_TEMPLATE = (
    "Task: Answer the microbiology question using only supplied evidence.\n"
    "Return concise bullet points with citations for each key claim.\n"
    "Question:\n{question}\n\n"
    "EVIDENCE:\n{evidence}\n\n"
    "Output format:\n"
    "- Finding: ... [chunk_id]\n"
    "- Limitation / escalation if needed: ... [chunk_id]"
)

AST_JSON_TEMPLATE = (
    "Task: Interpret antimicrobial susceptibility testing (AST).\n"
    "Output JSON ONLY. Do not include markdown or prose outside JSON.\n"
    "Question:\n{question}\n\n"
    "EVIDENCE:\n{evidence}\n\n"
    "Schema summary:\n{schema_summary}\n"
)

BLOOD_CULTURE_JSON_TEMPLATE = (
    "Task: Assess whether blood culture findings are contaminant vs true pathogen.\n"
    "Output JSON ONLY. Do not include markdown or prose outside JSON.\n"
    "Question:\n{question}\n\n"
    "EVIDENCE:\n{evidence}\n\n"
    "Schema summary:\n{schema_summary}\n"
)

REPORT_JSON_TEMPLATE = (
    "Task: Draft a structured microbiology report comment.\n"
    "Output JSON ONLY. Do not include markdown or prose outside JSON.\n"
    "Question:\n{question}\n\n"
    "EVIDENCE:\n{evidence}\n\n"
    "Schema summary:\n{schema_summary}\n"
)


def select_template(task: str) -> str:
    """Select prompt template by task label."""
    normalized = (task or "").strip().lower()
    if normalized == "ast":
        return AST_JSON_TEMPLATE
    if normalized == "blood_culture":
        return BLOOD_CULTURE_JSON_TEMPLATE
    if normalized == "reporting":
        return REPORT_JSON_TEMPLATE
    return GENERAL_TEMPLATE


def build_prompt(task: str, question: str, evidence: str) -> Dict[str, str]:
    """Build chat-style prompt payload."""
    normalized = (task or "").strip().lower()
    template = select_template(normalized)
    schema_summary = ""
    if normalized == "ast":
        schema_summary = json.dumps(AST_INTERPRETATION_SCHEMA, ensure_ascii=False)
    elif normalized == "blood_culture":
        schema_summary = json.dumps(BLOOD_CULTURE_ASSESSMENT_SCHEMA, ensure_ascii=False)
    elif normalized == "reporting":
        schema_summary = json.dumps(STRUCTURED_REPORT_DRAFT_SCHEMA, ensure_ascii=False)

    return {
        "system": SYSTEM_RULES,
        "user": template.format(
            question=question.strip(),
            evidence=evidence.strip(),
            schema_summary=schema_summary,
        ),
    }
