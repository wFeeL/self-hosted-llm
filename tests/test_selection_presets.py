import re
from pathlib import Path

import pytest

from error_classifier.selection.presets import Candidate, PresetError, load_presets, parse_size

A = "[qwen3-4b]\nhf-repo = org/qwen\nhf-file = q.Q4_K_M.gguf\n"
B = "[granite-1b]\nhf-repo = ibm/granite\nhf-file = g.gguf\nthreads = 4\n"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "presets.ini"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_presets_reads_sections_in_order(tmp_path: Path) -> None:
    got = load_presets(_write(tmp_path, B + "\n" + A))
    assert got == [
        Candidate("granite-1b", "ibm/granite", "g.gguf", False),
        Candidate("qwen3-4b", "org/qwen", "q.Q4_K_M.gguf", False),
    ]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ('chat-template-kwargs = {"enable_thinking": false}\n', True),
        ("", False),
        ('chat-template-kwargs = {"enable_thinking": true}\n', False),
    ],
)
def test_load_presets_thinking_flag(tmp_path: Path, line: str, expected: bool) -> None:
    (candidate,) = load_presets(_write(tmp_path, A + line))
    assert candidate.disable_thinking is expected


def test_load_presets_skips_star_section(tmp_path: Path) -> None:
    text = "[*]\nctx-size = 8192\n\n" + A
    assert [c.alias for c in load_presets(_write(tmp_path, text))] == ["qwen3-4b"]


@pytest.mark.parametrize(
    ("text", "needle"),
    [
        (None, "presets"),
        ("", "no model"),
        ("[*]\nthreads = 1\n", "no model"),
        ("[m]\nhf-repo = org/x\n", "[m]"),
        ("[m]\nhf-file = x.gguf\n", "[m]"),
        ("[m]\nhf-repo = org/x\nhf-file = x.bin\n", "[m]"),
        ("[m]\nhf-repo = nosep\nhf-file = x.gguf\n", "[m]"),
        ("[m]\nhf-repo = org/x\nhf-file = x.gguf\nchat-template-kwargs = {bad\n", "[m]"),
        ("[m]\nhf-repo = org/x\nhf-file = x.gguf\nchat-template-kwargs = [1]\n", "[m]"),
        ("[bad alias]\nhf-repo = org/x\nhf-file = x.gguf\n", "bad alias"),
        (A + A, "qwen3-4b"),
        ("hf-repo = org/x\n", "presets"),
    ],
)
def test_load_presets_rejects(tmp_path: Path, text: str | None, needle: str) -> None:
    path = tmp_path / "missing.ini" if text is None else _write(tmp_path, text)
    with pytest.raises(PresetError, match=re.escape(needle)):
        load_presets(path)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("6656m", 6656 * 2**20),
        ("6g", 6 * 2**30),
        ("6G", 6 * 2**30),
        ("512m", 512 * 2**20),
        ("1048576", 1048576),
        ("2k", 2048),
        ("5b", 5),
    ],
)
def test_parse_size(text: str, expected: int) -> None:
    assert parse_size(text) == expected


@pytest.mark.parametrize("text", ["", "6x", "-1m", "m", "1.5g", " 6g"])
def test_parse_size_rejects(text: str) -> None:
    with pytest.raises(ValueError, match="size"):
        parse_size(text)
