"""Prompt templates used by Track A model benchmarking."""

from __future__ import annotations

from typing import Dict

PROMPT_VERSION = "v1.0"

SYSTEM_RULES = (
    "You are a clinical microbiology assistant operating under strict safety guardrails.\n"
    "Rules:\n"
    "1) Evidence-first: only use the provided evidence chunks.\n"
    "2) Every key claim must cite supporting chunk_id(s) in square brackets like [chunk_12].\n"
    "3) If evidence is missing or ambiguous, say so and escalate to a clinician/lab lead.\n"
    "4) Never autonomously prescribe treatment or dosing."
)

GENERAL_TEMPLATE = (
    "Task: Answer the microbiology question using only supplied evidence.\n"
    "Return concise bullet points with citations for each key claim.\n"
    "Question:\n{question}\n\n"
    "Evidence:\n{evidence}\n\n"
    "Output format:\n"
    "- Finding: ... [chunk_id]\n"
    "- Limitation / escalation if needed: ... [chunk_id]"
)

AST_JSON_TEMPLATE = (
    "Task: Interpret antimicrobial susceptibility testing (AST).\n"
    "Return valid JSON only, no markdown.\n"
    "Question:\n{question}\n\n"
    "Evidence:\n{evidence}\n\n"
    "JSON schema:\n"
    "{\n"
    "  \"organism\": \"string\",\n"
    "  \"antibiotics\": [{\"name\": \"string\", \"interpretation\": \"S|I|R|unknown\", \"evidence\": [\"chunk_id\"]}],\n"
    "  \"overall_comment\": \"string with citations\",\n"
    "  \"needs_escalation\": true\n"
    "}"
)

BLOOD_CULTURE_TEMPLATE = (
    "Task: Reason whether blood culture finding suggests contaminant vs true pathogen.\n"
    "Only rely on evidence chunks. Cite every major rationale with [chunk_id].\n"
    "Question:\n{question}\n\n"
    "Evidence:\n{evidence}\n\n"
    "Return sections:\n"
    "1) Classification: contaminant|true_pathogen|indeterminate [chunk_id]\n"
    "2) Rationale: 2-4 bullet points with citations\n"
    "3) Safety note: request clinician/lab escalation when data is incomplete"
)


def select_template(task: str) -> str:
    """Select prompt template by task label."""
    normalized = (task or "").strip().lower()
    if normalized == "ast":
        return AST_JSON_TEMPLATE
    if normalized == "blood_culture":
        return BLOOD_CULTURE_TEMPLATE
    return GENERAL_TEMPLATE


def build_prompt(task: str, question: str, evidence: str) -> Dict[str, str]:
    """Build chat-style prompt payload."""
    template = select_template(task)
    return {
        "system": SYSTEM_RULES,
        "user": template.format(question=question.strip(), evidence=evidence.strip()),
    }
