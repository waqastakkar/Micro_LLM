"""JSON schema definitions for constrained clinical microbiology tasks."""

from __future__ import annotations

AST_INTERPRETATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "task",
        "organism",
        "isolate_site",
        "antibiotic_results",
        "overall_summary",
        "confidence",
        "escalation_flag",
        "follow_up_questions",
        "safety_notes",
    ],
    "properties": {
        "task": {"const": "ast"},
        "organism": {"type": "string"},
        "isolate_site": {"type": ["string", "null"]},
        "antibiotic_results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "drug",
                    "mic_or_zone",
                    "unit",
                    "interpretation",
                    "breakpoint_source",
                    "breakpoint_notes",
                    "intrinsic_resistance_alert",
                    "cascade_reporting_note",
                    "evidence",
                ],
                "properties": {
                    "drug": {"type": "string"},
                    "mic_or_zone": {"type": ["string", "null"]},
                    "unit": {"type": ["string", "null"], "enum": ["mg/L", "ug/mL", "mm", None]},
                    "interpretation": {"type": "string", "enum": ["S", "I", "R", "SDD", "NA"]},
                    "breakpoint_source": {
                        "type": "string",
                        "enum": ["CLSI", "EUCAST", "LAB_DEFINED", "UNKNOWN"],
                    },
                    "breakpoint_notes": {"type": "string"},
                    "intrinsic_resistance_alert": {"type": "boolean"},
                    "cascade_reporting_note": {"type": ["string", "null"]},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "overall_summary": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "escalation_flag": {"type": "boolean"},
        "follow_up_questions": {"type": "array", "items": {"type": "string"}},
        "safety_notes": {"type": "array", "items": {"type": "string"}},
    },
}

BLOOD_CULTURE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "task",
        "likely_contaminant",
        "rationale",
        "organism",
        "supporting_evidence",
        "recommended_next_steps",
        "confidence",
        "escalation_flag",
        "follow_up_questions",
        "safety_notes",
    ],
    "properties": {
        "task": {"const": "blood_culture"},
        "likely_contaminant": {"anyOf": [{"type": "boolean"}, {"const": "uncertain"}]},
        "rationale": {"type": "string"},
        "organism": {"type": ["string", "null"]},
        "supporting_evidence": {"type": "array", "items": {"type": "string"}},
        "recommended_next_steps": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "escalation_flag": {"type": "boolean"},
        "follow_up_questions": {"type": "array", "items": {"type": "string"}},
        "safety_notes": {"type": "array", "items": {"type": "string"}},
    },
}

STRUCTURED_REPORT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "task",
        "report_type",
        "draft_report",
        "critical_value_message_template",
        "supporting_evidence",
        "escalation_flag",
        "safety_notes",
    ],
    "properties": {
        "task": {"const": "reporting"},
        "report_type": {
            "type": "string",
            "enum": ["routine", "critical_value", "sterile_site", "blood_culture"],
        },
        "draft_report": {"type": "string"},
        "critical_value_message_template": {"type": ["string", "null"]},
        "supporting_evidence": {"type": "array", "items": {"type": "string"}},
        "escalation_flag": {"type": "boolean"},
        "safety_notes": {"type": "array", "items": {"type": "string"}},
    },
}

SCHEMAS = {
    "ast": AST_INTERPRETATION_SCHEMA,
    "blood_culture": BLOOD_CULTURE_SCHEMA,
    "reporting": STRUCTURED_REPORT_SCHEMA,
}
