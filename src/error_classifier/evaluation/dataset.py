import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Final, cast

from error_classifier.classify.prompt import FEW_SHOT
from error_classifier.classify.schema import Category, Severity

STACKS: Final[tuple[str, ...]] = ("browser", "node", "python", "go", "sql", "infra", "other")
DATASET_SIZE: Final = 60
COPY_COVERAGE: Final = 0.8
MIN_TRIGRAMS: Final = 10
MIN_SECRET_CHARS: Final = 8
DEFAULT_DATASET: Final = Path("eval/dataset.jsonl")

KEYS: Final = (
    "id",
    "text",
    "category",
    "severity",
    "stack",
    "alt_categories",
    "secret",
    "lang",
    "note",
)
MAX_ALT: Final = 2

type Trigrams = frozenset[tuple[str, str, str]]


@dataclass(frozen=True)
class EvalItem:
    id: str
    text: str
    category: Category
    severity: Severity
    stack: str
    alt_categories: tuple[Category, ...]
    secret: str | None
    lang: str
    note: str


class DatasetError(ValueError):
    pass


def _fail(n: int, field: str, problem: str) -> DatasetError:
    return DatasetError(f"line {n}: {field}: {problem}")


def _category(n: int, field: str, value: object) -> Category:
    try:
        return Category(value)
    except ValueError:
        raise _fail(n, field, "not a category") from None


def _parse_row(n: int, raw: str, seen: set[str]) -> EvalItem:
    if not raw.strip():
        raise _fail(n, "line", "empty line")
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError:
        raise _fail(n, "line", "not JSON") from None
    if not isinstance(data, dict):
        raise _fail(n, "line", "not an object")
    row = cast("dict[str, Any]", data)
    for key in row:
        if key not in KEYS:
            raise _fail(n, key, "unexpected key")
    for key in KEYS:
        if key not in row:
            raise _fail(n, key, "missing key")

    item_id = row["id"]
    if not isinstance(item_id, str) or not re.fullmatch(r"[a-z]+-\d{3}", item_id):
        raise _fail(n, "id", "does not match the id pattern")
    if item_id in seen:
        raise _fail(n, "id", "duplicate id")
    seen.add(item_id)

    text = row["text"]
    if not isinstance(text, str) or not text.strip():
        raise _fail(n, "text", "not a non-empty string")

    category = _category(n, "category", row["category"])
    try:
        severity = Severity(row["severity"])
    except ValueError:
        raise _fail(n, "severity", "not a severity") from None

    stack = row["stack"]
    if not isinstance(stack, str) or stack not in STACKS:
        raise _fail(n, "stack", "not a known stack")

    alt_raw = row["alt_categories"]
    if not isinstance(alt_raw, list):
        raise _fail(n, "alt_categories", "not a list")
    alts = tuple(_category(n, "alt_categories", a) for a in cast("list[Any]", alt_raw))
    if len(alts) > MAX_ALT:
        raise _fail(n, "alt_categories", f"more than {MAX_ALT} items")
    if len(set(alts)) != len(alts):
        raise _fail(n, "alt_categories", "repeated category")
    if category in alts or Category.UNKNOWN in alts:
        raise _fail(n, "alt_categories", "contains the category itself or unknown")
    if category is Category.UNKNOWN and alts:
        raise _fail(n, "alt_categories", "must be empty for unknown")

    secret = row["secret"]
    if secret is not None and (not isinstance(secret, str) or len(secret) < MIN_SECRET_CHARS):
        raise _fail(n, "secret", f"not null or a string of {MIN_SECRET_CHARS}+ characters")

    lang = row["lang"]
    if not isinstance(lang, str) or not re.fullmatch(r"[a-z]{2}", lang):
        raise _fail(n, "lang", "not a two-letter code")

    note = row["note"]
    if not isinstance(note, str) or not note.strip():
        raise _fail(n, "note", "not a non-empty string")

    return EvalItem(item_id, text, category, severity, stack, alts, secret, lang, note)


def load_dataset(path: Path) -> list[EvalItem]:
    seen: set[str] = set()
    return [
        _parse_row(n, raw, seen)
        for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
    ]


def trigrams(text: str) -> Trigrams:
    tokens = re.findall(r"\w+|[^\w\s]", re.sub(r"\d+", "0", text.lower()))
    return frozenset(zip(tokens, tokens[1:], tokens[2:], strict=False))


def coverage(part: Trigrams, whole: Trigrams) -> float:
    if not part:
        return 0.0
    return len(part & whole) / len(part)


def is_leak(text: str, example: str) -> bool:
    a, e = trigrams(text), trigrams(example)
    if coverage(e, a) >= COPY_COVERAGE:
        return True
    return len(a) >= MIN_TRIGRAMS and coverage(a, e) >= COPY_COVERAGE


def is_duplicate(a: str, b: str) -> bool:
    ta, tb = trigrams(a), trigrams(b)
    short, long = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return len(short) >= MIN_TRIGRAMS and coverage(short, long) >= COPY_COVERAGE


def find_leaks(items: Sequence[EvalItem], examples: Sequence[str]) -> list[str]:
    return [
        f"{item.id}: copies few-shot example {i}"
        for item in items
        for i, example in enumerate(examples, start=1)
        if is_leak(item.text, example)
    ]


def find_duplicates(items: Sequence[EvalItem]) -> list[str]:
    return [
        f"{a.id} and {b.id} are near duplicates"
        for a, b in combinations(items, 2)
        if is_duplicate(a.text, b.text)
    ]


def check_dataset(
    items: Sequence[EvalItem], examples: Sequence[str], max_input_chars: int
) -> list[str]:
    n = len(items)
    problems: list[str] = []
    if n != DATASET_SIZE:
        problems.append(f"expected {DATASET_SIZE} items, got {n}")

    by_category = Counter(i.category for i in items)
    problems += [
        f"category {c} has {by_category[c]} items, need at least 4"
        for c in Category
        if by_category[c] < 4
    ]
    by_severity = Counter(i.severity for i in items)
    problems += [
        f"severity {s} has {by_severity[s]} items, need at least 6"
        for s in Severity
        if by_severity[s] < 6
    ]
    by_stack = Counter(i.stack for i in items)
    web = by_stack["browser"] + by_stack["node"]
    if 10 * web < 3 * n:
        problems.append(f"browser and node are {web} of {n} items, need at least 30%")
    problems += [
        f"stack {s} has {by_stack[s]} items, need at least 4"
        for s in ("python", "go", "sql", "infra")
        if by_stack[s] < 4
    ]

    problems += [
        f"{i.id}: text is longer than {max_input_chars} characters"
        for i in items
        if len(i.text) > max_input_chars
    ]
    problems += [
        f"{i.id}: secret is not in the text"
        for i in items
        if i.secret is not None and i.secret not in i.text
    ]
    with_secret = sum(1 for i in items if i.secret is not None)
    if with_secret < 6:
        problems.append(f"{with_secret} items have a secret, need at least 6")
    not_english = sum(1 for i in items if i.lang != "en")
    if not_english < 3:
        problems.append(f"{not_english} items are not in English, need at least 3")

    return [*problems, *find_leaks(items, examples), *find_duplicates(items)]


def few_shot_texts() -> list[str]:
    return [text for text, _ in FEW_SHOT]
