"""Prompt templates used by Track A model benchmarking."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

PROMPT_VERSION = "v2.0"
SYSTEM_PROMPT_PATH = Path("eval/system_prompt.txt")

TASK_SCHEMA_OUTLINE = {
    "ast": (
        '{"task":"ast","organism":"...","isolate_site":null|"...",'
        '"antibiotic_results":[{"drug":"...","mic_or_zone":null|"...",'
        '"unit":"mg/L|ug/mL|mm|null","interpretation":"S|I|R|SDD|NA",'
        '"breakpoint_source":"CLSI|EUCAST|LAB_DEFINED|UNKNOWN",'
        '"breakpoint_notes":"...","intrinsic_resistance_alert":true|false,'
        '"cascade_reporting_note":null|"...","evidence":["chunk_id"]}],'
        '"overall_summary":"...","confidence":0.0,"escalation_flag":true|false,'
        '"follow_up_questions":["..."],"safety_notes":["..."]}'
    ),
    "blood_culture": (
        '{"task":"blood_culture","likely_contaminant":true|false|"uncertain",'
        '"rationale":"...","organism":null|"...","supporting_evidence":["chunk_id"],'
        '"recommended_next_steps":["non-prescribing"],"confidence":0.0,'
        '"escalation_flag":true|false,"follow_up_questions":["..."],"safety_notes":["..."]}'
    ),
    "reporting": (
        '{"task":"reporting","report_type":"routine|critical_value|sterile_site|blood_culture",'
        '"draft_report":"...","critical_value_message_template":null|"...",'
        '"supporting_evidence":["chunk_id"],"escalation_flag":true|false,"safety_notes":["..."]}'
    ),
}


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def _structured_template(task: str, question: str, evidence: str) -> str:
    return (
        f"Task: {task}\n"
        "Use ONLY the EVIDENCE section.\n"
        "Output JSON only. No markdown. No commentary.\n"
        "Never prescribe specific antibiotic regimen/dose/duration. "
        "If evidence is insufficient/discordant, set escalation_flag=true and ask follow-up questions.\n"
        f"Schema outline:\n{TASK_SCHEMA_OUTLINE[task]}\n\n"
        f"Question:\n{question}\n\n"
        f"EVIDENCE:\n{evidence}"
    )


def _general_template(question: str, evidence: str) -> str:
    return (
        "Answer using ONLY the EVIDENCE section.\n"
        "Every clinical claim must include citations like [chunk_id].\n"
        "If evidence is insufficient, say INSUFFICIENT EVIDENCE and escalate.\n"
        "Never prescribe specific regimen/dose/duration.\n\n"
        f"Question:\n{question}\n\n"
        f"EVIDENCE:\n{evidence}"
    )


def build_prompt(task: str, question: str, evidence: str) -> Dict[str, str]:
    normalized = (task or "").strip().lower()
    system = load_system_prompt()
    if normalized in TASK_SCHEMA_OUTLINE:
        user = _structured_template(normalized, question.strip(), evidence.strip())
    else:
        user = _general_template(question.strip(), evidence.strip())
    return {"system": system, "user": user}
