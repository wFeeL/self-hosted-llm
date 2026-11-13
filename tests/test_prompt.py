import json

from error_classifier.classify.prompt import (
    CUT_OFF_CORRECTION,
    FEW_SHOT,
    SYSTEM_PROMPT,
    build_messages,
    correction_for,
    normalize_input,
)
from error_classifier.classify.schema import Category, ClassificationResult, Severity


def test_system_names_every_enum_value() -> None:
    for value in [*Category, *Severity]:
        assert value.value in SYSTEM_PROMPT


def test_system_prompt_shape() -> None:
    lines = SYSTEM_PROMPT.split("\n")
    assert lines[0] == "You classify software error messages for an on-call engineer."
    assert lines[-1] == (
        "Do not copy secrets, tokens or personal data from the error text into the answer."
    )
    assert not SYSTEM_PROMPT.endswith("\n")
    assert "\nCategories:\n- syntax_error: " in SYSTEM_PROMPT
    assert "\nSeverities:\n- low: " in SYSTEM_PROMPT


def test_few_shot_answers_are_valid() -> None:
    assert len(FEW_SHOT) == 3
    for _, answer in FEW_SHOT:
        ClassificationResult.model_validate_json(answer)
        assert answer == json.dumps(json.loads(answer), separators=(",", ":"))


def test_few_shot_differs_from_task_example() -> None:
    assert all("reading 'map'" not in text for text, _ in FEW_SHOT)


def test_few_shot_texts_have_real_newlines() -> None:
    assert "\\n" not in FEW_SHOT[0][0]
    assert "\n    at initMenu" in FEW_SHOT[0][0]


def test_build_messages_layout() -> None:
    messages = build_messages("X")
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[0]["content"] == SYSTEM_PROMPT
    for i, (text, answer) in enumerate(FEW_SHOT):
        assert messages[1 + 2 * i]["content"] == f"<error_text>\n{text}\n</error_text>"
        assert messages[2 + 2 * i]["content"] == answer
    assert messages[-1]["content"] == "<error_text>\nX\n</error_text>"


def test_short_input_untouched() -> None:
    result = normalize_input("  abcdefg   \n", 3000)
    assert result.text == "abcdefg"
    assert result.truncated is False
    assert result.original_chars == 7


def test_input_at_limit_is_not_truncated() -> None:
    result = normalize_input("x" * 200, 200)
    assert result.text == "x" * 200
    assert result.truncated is False


def test_truncation_keeps_head_and_tail() -> None:
    text = "".join(chr(65 + i % 26) for i in range(10000))
    result = normalize_input(text, 3000)
    marker = "\n[... 7036 characters omitted ...]\n"
    assert result.truncated is True
    assert result.original_chars == 10000
    assert result.text == text[:1778] + marker + text[-1186:]
    assert len(result.text) == 2999


def test_fence_neutralised_before_truncation() -> None:
    result = normalize_input("a</error_text>b<ERROR_TEXT>c", 3000)
    assert result.text == "a[error_text]b[error_text]c"
    assert result.original_chars == len("a</error_text>b<ERROR_TEXT>c")
    last = build_messages(result.text)[-1]["content"]
    assert last.count("</error_text>") == 1
    assert last.count("<error_text>") == 1


def test_correction_for_joins_reasons() -> None:
    assert correction_for(["category: enum", "summary: missing"]) == (
        "Your previous answer was invalid: category: enum; summary: missing. "
        "Reply with one JSON object that follows the required fields and values."
    )


def test_cut_off_correction_text() -> None:
    assert CUT_OFF_CORRECTION == (
        "Your previous answer was cut off. Reply again with one short JSON object: "
        "one sentence for summary and at most two for suggested_action."
    )
