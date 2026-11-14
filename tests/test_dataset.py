import dataclasses
import json
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from error_classifier.classify.prompt import FEW_SHOT
from error_classifier.classify.schema import Category, Severity
from error_classifier.config import Settings
from error_classifier.evaluation.dataset import (
    DEFAULT_DATASET,
    DatasetError,
    EvalItem,
    check_dataset,
    few_shot_texts,
    find_duplicates,
    find_leaks,
    load_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / DEFAULT_DATASET
MAX_CHARS: int = Settings.model_fields["max_input_chars"].default
HEARTBEAT = "\n".join(
    f"2026-03-02 10:1{i}:00 INFO worker heartbeat ok, queue depth {i * 3}" for i in range(5)
)


def _row(id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": id,
        "text": "KeyError: 'user_id' in handler",
        "category": "runtime_error",
        "severity": "medium",
        "stack": "python",
        "alt_categories": [],
        "secret": None,
        "lang": "en",
        "note": "A missing key in a handler.",
    }
    return row | overrides


def _write(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "set.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _item(id: str, text: str) -> EvalItem:
    return EvalItem(
        id=id,
        text=text,
        category=Category.RUNTIME_ERROR,
        severity=Severity.MEDIUM,
        stack="python",
        alt_categories=(),
        secret=None,
        lang="en",
        note="n",
    )


def _leaks(text: str) -> list[str]:
    return find_leaks([_item("t-001", text)], few_shot_texts())


def _real() -> list[EvalItem]:
    return load_dataset(REAL)


def _check(items: list[EvalItem], limit: int = MAX_CHARS) -> list[str]:
    return check_dataset(items, few_shot_texts(), limit)


def test_real_dataset_passes() -> None:
    assert _check(_real()) == []


def test_real_dataset_size_and_ids() -> None:
    items = _real()
    assert len(items) == 60
    assert len({i.id for i in items}) == 60


def test_copy_of_few_shot_with_other_numbers_is_a_leak() -> None:
    original = FEW_SHOT[0][0]
    text = original.replace("menu.js:14:9", "menu.js:27:31").replace("main.js:3:1", "main.js:8:5")
    assert text != original
    assert _leaks(text) == ["t-001: copies few-shot example 1"]


def test_few_shot_with_other_ip_and_date_is_a_leak() -> None:
    original = FEW_SHOT[1][0]
    text = original.replace("10.1.2.3", "192.168.44.77").replace(
        "2026/03/02 10:14:07", "2027/11/09 22:01:55"
    )
    assert text != original
    assert _leaks(text) == ["t-001: copies few-shot example 2"]


def test_few_shot_with_crlf_and_extra_frame_is_a_leak() -> None:
    text = FEW_SHOT[0][0].replace("\n", "\r\n") + "\r\n    at boot (app.js:88:2)"
    assert _leaks(text) == ["t-001: copies few-shot example 1"]


def test_few_shot_inside_longer_text_is_a_leak() -> None:
    text = "deploy log\n" + FEW_SHOT[2][0] + "\nretrying"
    assert _leaks(text) == ["t-001: copies few-shot example 3"]


def test_edited_few_shot_inside_a_long_log_is_a_leak() -> None:
    edited = FEW_SHOT[0][0].replace("addEventListener", "removeEventListener")
    assert _leaks(f"{HEARTBEAT}\n{edited}\n{HEARTBEAT}") == ["t-001: copies few-shot example 1"]


def test_same_error_type_inside_a_long_log_is_not_a_leak() -> None:
    error = "OSError: [Errno 28] No space left on device: '/data/cache/blob_77'"
    assert _leaks(f"{HEARTBEAT}\n{error}\n{HEARTBEAT}") == []


@pytest.mark.parametrize(
    "text",
    [
        "TypeError: Cannot read properties of undefined (reading 'map')\n"
        "    at renderProducts (catalog.js:42:18)",
        "Uncaught TypeError: Cannot read properties of null (reading 'value')\n"
        "    at submitForm (form.js:31:7)",
        "OSError: [Errno 28] No space left on device: '/tmp/x'",
    ],
)
def test_same_error_type_is_not_a_leak(text: str) -> None:
    assert _leaks(text) == []


def test_short_common_line_is_not_a_leak() -> None:
    assert _leaks("    at main.js:3:1") == []


def test_long_repetitive_text_is_fast_and_not_a_leak() -> None:
    words = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta"]
    trace = "\n".join(
        f'  File "/app/{words[i % 8]}.py", line {i}, in {words[i % 8]}' for i in range(40)
    )
    items = [_item("t-001", "x" * 3001), _item("t-002", trace)]
    started = time.perf_counter()
    leaks = find_leaks(items, few_shot_texts())
    duplicates = find_duplicates(items)
    elapsed = time.perf_counter() - started
    assert leaks == []
    assert duplicates == []
    assert elapsed < 2


def test_near_duplicates_inside_the_set() -> None:
    first = "retry 3 after 250 ms for job 4411 on host build-7, queue depth 18, status 503"
    second = "retry 5 after 400 ms for job 9021 on host build-2, queue depth 41, status 502"
    items = [_item("t-001", first), _item("t-002", second)]
    assert find_duplicates(items) == ["t-001 and t-002 are near duplicates"]


def test_unknown_category_names_the_line(tmp_path: Path) -> None:
    path = _write(tmp_path, [_row("a-001"), _row("a-002"), _row("a-003", category="nonsense")])
    with pytest.raises(DatasetError, match=r"line 3: category: not a category"):
        load_dataset(path)


def test_severity_error_says_not_a_severity(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match=r"line 1: severity: not a severity"):
        load_dataset(_write(tmp_path, [_row("a-001", severity="fatal")]))


def test_extra_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match=r"line 1: extra"):
        load_dataset(_write(tmp_path, [_row("a-001", extra=1)]))


def test_missing_key_is_rejected(tmp_path: Path) -> None:
    row = _row("a-001")
    del row["note"]
    with pytest.raises(DatasetError, match=r"line 1: note"):
        load_dataset(_write(tmp_path, [row]))


def test_duplicate_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match=r"line 2: id"):
        load_dataset(_write(tmp_path, [_row("a-001"), _row("a-001")]))


@pytest.mark.parametrize(
    ("category", "alt"),
    [
        ("runtime_error", ["runtime_error"]),
        ("runtime_error", ["unknown"]),
        ("unknown", ["network_error"]),
        ("runtime_error", ["network_error", "database_error", "resource_error"]),
        ("runtime_error", ["network_error", "network_error"]),
        ("runtime_error", ["nonsense"]),
        ("runtime_error", "network_error"),
    ],
)
def test_alt_rules(tmp_path: Path, category: str, alt: object) -> None:
    path = _write(tmp_path, [_row("a-001", category=category, alt_categories=alt)])
    with pytest.raises(DatasetError, match=r"line 1: alt_categories"):
        load_dataset(path)


def test_valid_alt_is_loaded(tmp_path: Path) -> None:
    path = _write(tmp_path, [_row("a-001", alt_categories=["network_error"])])
    assert load_dataset(path)[0].alt_categories == (Category.NETWORK_ERROR,)


def test_short_secret_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match=r"line 1: secret"):
        load_dataset(_write(tmp_path, [_row("a-001", secret="abc")]))


def test_blank_line_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text(
        json.dumps(_row("a-001")) + "\n\n" + json.dumps(_row("a-002")) + "\n", encoding="utf-8"
    )
    with pytest.raises(DatasetError, match=r"line 2: line"):
        load_dataset(path)


@pytest.mark.parametrize("raw", ["not json", "[1, 2]"])
def test_non_object_line_is_rejected(tmp_path: Path, raw: str) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text(raw + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=r"line 1: line"):
        load_dataset(path)


def test_error_text_does_not_contain_the_value(tmp_path: Path) -> None:
    with pytest.raises(DatasetError) as info:
        load_dataset(_write(tmp_path, [_row("a-001", stack="ruby-secret-value")]))
    assert "ruby-secret-value" not in str(info.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "JS-1"),
        ("stack", "ruby"),
        ("lang", "eng"),
        ("note", ""),
        ("text", "   "),
        ("text", 5),
        ("secret", 5),
    ],
)
def test_field_value_rules(tmp_path: Path, field: str, value: object) -> None:
    path = _write(tmp_path, [_row("a-001") | {field: value}])
    with pytest.raises(DatasetError, match=rf"line 1: {field}"):
        load_dataset(path)


def test_secret_missing_from_text() -> None:
    items = _real()
    index = next(i for i, item in enumerate(items) if item.secret)
    items[index] = dataclasses.replace(items[index], text="nothing to see here")
    assert f"{items[index].id}: secret is not in the text" in _check(items)


def test_too_few_items() -> None:
    assert "expected 60 items, got 50" in _check(_real()[:50])


def test_too_long_text() -> None:
    items = _real()
    items[0] = dataclasses.replace(items[0], text="x" * 3001)
    assert any("text is longer than 3000" in p for p in _check(items, 3000))


def _go_to_other(i: EvalItem) -> EvalItem:
    return dataclasses.replace(i, stack="other" if i.stack == "go" else i.stack)


def _critical_to_high(i: EvalItem) -> EvalItem:
    return dataclasses.replace(
        i, severity=Severity.HIGH if i.severity is Severity.CRITICAL else i.severity
    )


def _web_to_other(i: EvalItem) -> EvalItem:
    return dataclasses.replace(i, stack="other" if i.stack in ("browser", "node") else i.stack)


def _all_english(i: EvalItem) -> EvalItem:
    return dataclasses.replace(i, lang="en")


def _syntax_to_runtime(i: EvalItem) -> EvalItem:
    if i.category is not Category.SYNTAX_ERROR:
        return i
    return dataclasses.replace(i, category=Category.RUNTIME_ERROR, alt_categories=())


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (_go_to_other, "stack go has 0 items, need at least 4"),
        (_critical_to_high, "severity critical has 0 items, need at least 6"),
        (_web_to_other, "browser and node are 0 of 60 items, need at least 30%"),
        (_all_english, "0 items are not in English, need at least 3"),
        (_syntax_to_runtime, "category syntax_error has 0 items, need at least 4"),
    ],
)
def test_composition_rules(change: Callable[[EvalItem], EvalItem], expected: str) -> None:
    assert expected in _check([change(i) for i in _real()])


def test_single_secret_is_too_few() -> None:
    items = _real()
    first = next(i.id for i in items if i.secret)
    items = [i if i.id == first else dataclasses.replace(i, secret=None) for i in items]
    assert "1 items have a secret, need at least 6" in _check(items)
