import json
from typing import cast

import pytest
from pydantic import ValidationError

from error_classifier.classify.schema import (
    Category,
    ClassificationResult,
    Severity,
    response_schema,
)

EXPECTED_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": [
                "syntax_error",
                "runtime_error",
                "validation_error",
                "network_error",
                "database_error",
                "dependency_error",
                "configuration_error",
                "resource_error",
                "permission_error",
                "unknown",
            ],
        },
        "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "summary": {"type": "string"},
        "suggested_action": {"type": "string"},
    },
    "required": ["category", "severity", "summary", "suggested_action"],
    "additionalProperties": False,
}

VALID: dict[str, str] = {
    "category": "runtime_error",
    "severity": "medium",
    "summary": "Something broke.",
    "suggested_action": "Check the logs.",
}


def _with(**changes: str) -> str:
    return json.dumps({**VALID, **changes})


def test_schema_literal_shape() -> None:
    schema = response_schema()
    assert schema == EXPECTED_SCHEMA
    dumped = json.dumps(schema)
    assert "$ref" not in dumped
    assert "$defs" not in dumped


def test_schema_returns_fresh_dict() -> None:
    first = response_schema()
    first["type"] = "changed"
    assert response_schema()["type"] == "object"


def test_schema_agrees_with_model() -> None:
    schema = response_schema()
    properties = cast(dict[str, dict[str, object]], schema["properties"])
    fields = set(ClassificationResult.model_fields)
    assert set(properties) == fields
    assert set(cast(list[str], schema["required"])) == fields
    assert properties["category"]["enum"] == [c.value for c in Category]
    assert properties["severity"]["enum"] == [s.value for s in Severity]


def test_valid_result() -> None:
    raw = _with(summary="  Boom.  ", suggested_action="\tRetry.\n")
    result = ClassificationResult.model_validate_json(raw)
    assert result.category is Category.RUNTIME_ERROR
    assert result.severity is Severity.MEDIUM
    assert result.summary == "Boom."
    assert result.suggested_action == "Retry."


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        _with(category="bogus"),
        _with(severity="catastrophic"),
        _with(summary=""),
        _with(summary="   "),
        _with(summary="x" * 301),
        _with(suggested_action="x" * 401),
        _with(extra="field"),
        json.dumps({k: v for k, v in VALID.items() if k != "summary"}),
    ],
    ids=[
        "not_json",
        "bad_category",
        "bad_severity",
        "empty_summary",
        "blank_summary",
        "long_summary",
        "long_action",
        "extra_field",
        "missing_field",
    ],
)
def test_invalid_results(raw: str) -> None:
    with pytest.raises(ValidationError):
        ClassificationResult.model_validate_json(raw)


def test_limits_are_inclusive() -> None:
    raw = _with(summary="x" * 300, suggested_action="y" * 400)
    assert len(ClassificationResult.model_validate_json(raw).summary) == 300
