"""JSON schemas for structured evaluation tasks."""

from __future__ import annotations

AST_INTERPRETATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "organism",
        "antibiotics",
        "overall_interpretation",
        "safety_flags",
        "needs_escalation",
    ],
    "properties": {
        "organism": {"type": "string", "minLength": 1},
        "antibiotics": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "interpretation", "evidence"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "interpretation": {
                        "type": "string",
                        "enum": ["S", "I", "R", "SDD", "unknown"],
                    },
                    "evidence": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
        "overall_interpretation": {"type": "string", "minLength": 1},
        "safety_flags": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "none",
                    "insufficient_evidence",
                    "discordant_evidence",
                    "critical_result",
                    "requires_id_consult",
                ],
            },
            "minItems": 1,
        },
        "needs_escalation": {"type": "boolean"},
    },
}

BLOOD_CULTURE_ASSESSMENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["classification", "rationale", "confidence", "safety_flags", "needs_escalation"],
    "properties": {
        "classification": {
            "type": "string",
            "enum": ["contaminant", "true_pathogen", "indeterminate", "insufficient_evidence"],
        },
        "rationale": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "confidence": {"type": "string", "enum": ["low", "moderate", "high"]},
        "safety_flags": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "none",
                    "insufficient_evidence",
                    "discordant_evidence",
                    "possible_sepsis",
                    "requires_urgent_review",
                ],
            },
            "minItems": 1,
        },
        "needs_escalation": {"type": "boolean"},
    },
}

STRUCTURED_REPORT_DRAFT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["report_type", "summary", "key_findings", "safety_flags", "needs_escalation"],
    "properties": {
        "report_type": {"type": "string", "enum": ["preliminary", "final", "corrected"]},
        "summary": {"type": "string", "minLength": 1},
        "key_findings": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "recommended_follow_up_fields": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "safety_flags": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "none",
                    "insufficient_evidence",
                    "discordant_evidence",
                    "critical_result",
                    "requires_correction_notice",
                ],
            },
            "minItems": 1,
        },
        "needs_escalation": {"type": "boolean"},
    },
}

SCHEMAS = {
    "ast_interpretation": AST_INTERPRETATION_SCHEMA,
    "blood_culture_assessment": BLOOD_CULTURE_ASSESSMENT_SCHEMA,
    "structured_report_draft": STRUCTURED_REPORT_DRAFT_SCHEMA,
}
