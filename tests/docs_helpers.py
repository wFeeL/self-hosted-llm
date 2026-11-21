import re
from pathlib import Path

from error_classifier.benchmark.report import read_bench
from error_classifier.evaluation.dataset import DEFAULT_DATASET, load_dataset
from error_classifier.evaluation.results import ResultDoc, model_run, read_result
from error_classifier.recommend import choose_per_tier

ROOT = Path(__file__).resolve().parents[1]


def env_example() -> dict[str, str]:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    return dict(re.findall(r"^([A-Z_]+)=(.*)$", text, re.MULTILINE))


def readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def marked_block(text: str, name: str) -> str:
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    assert text.count(start) == 1, start
    assert text.count(end) == 1, end
    return text.split(start)[1].split(end)[0]


def table_rows(block: str) -> list[list[str]]:
    """Cells of every table row after the header and the separator, tables read in order."""
    rows: list[list[str]] = []
    lines = [line.strip() for line in block.splitlines() if line.strip().startswith("|")]
    for line in lines[2:]:
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def tier_choice(tier_gib: int) -> tuple[str, ResultDoc]:
    """The model `choose_per_tier` gives for the row, and its evaluation run."""
    items = load_dataset(ROOT / DEFAULT_DATASET)
    docs = [read_result(path) for path in sorted((ROOT / "eval" / "results").glob("*.json"))]
    benches = [read_bench(path) for path in sorted((ROOT / "bench" / "results").glob("*.jsonl"))]
    choices = choose_per_tier([model_run(doc, items) for doc in docs], benches)
    model = next(c.model for c in choices if c.tier_gib == tier_gib)
    return str(model), next(doc for doc in docs if doc.meta.model == model)


def report_tables(report: str, model: str) -> list[dict[str, str]]:
    """One column -> cell dict per markdown table of a bench report, for the row of `model`."""
    tables: list[dict[str, str]] = []
    group: list[list[str]] = []
    for line in [*report.splitlines(), ""]:
        if line.startswith("|"):
            group.append([cell.strip() for cell in line.strip("|").split("|")])
            continue
        if group:
            header, _, *rows = group
            row = next((r for r in rows if r[0] == model), None)
            assert row is not None, (model, header)
            tables.append(dict(zip(header, row, strict=True)))
            group = []
    return tables


def report_cell(tables: list[dict[str, str]], column: str, *, anchor: str | None = None) -> str:
    """The cell of the first table that has the column, or of the first one with `anchor`."""
    wanted = anchor or column
    table = next((t for t in tables if wanted in t), None)
    assert table is not None, wanted
    return table[column]
