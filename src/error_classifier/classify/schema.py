from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints


class Category(StrEnum):
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    VALIDATION_ERROR = "validation_error"
    NETWORK_ERROR = "network_error"
    DATABASE_ERROR = "database_error"
    DEPENDENCY_ERROR = "dependency_error"
    CONFIGURATION_ERROR = "configuration_error"
    RESOURCE_ERROR = "resource_error"
    PERMISSION_ERROR = "permission_error"
    UNKNOWN = "unknown"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


SUMMARY_MAX_CHARS = 300
ACTION_MAX_CHARS = 400
SCHEMA_NAME = "classification"

Summary = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=SUMMARY_MAX_CHARS)
]
Action = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=ACTION_MAX_CHARS)
]


class ClassificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    category: Category
    severity: Severity
    summary: Summary
    suggested_action: Action


def response_schema() -> dict[str, object]:
    # Written by hand: pydantic's schema uses $defs/$ref, which llama-server's converter mishandles.
    return {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": [c.value for c in Category]},
            "severity": {"type": "string", "enum": [s.value for s in Severity]},
            "summary": {"type": "string"},
            "suggested_action": {"type": "string"},
        },
        "required": ["category", "severity", "summary", "suggested_action"],
        "additionalProperties": False,
    }
