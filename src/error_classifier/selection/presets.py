import configparser
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

ALIAS: Final = re.compile(r"[a-z0-9][a-z0-9._-]*")
REPO: Final = re.compile(r"[^/\s]+/[^/\s]+")
SIZE: Final = re.compile(r"(\d+)([bkmg]?)", re.IGNORECASE)
UNITS: Final = {"": 1, "b": 1, "k": 2**10, "m": 2**20, "g": 2**30}


@dataclass(frozen=True)
class Candidate:
    alias: str
    hf_repo: str
    hf_file: str
    disable_thinking: bool


class PresetError(ValueError):
    pass


def _candidate(name: str, section: configparser.SectionProxy) -> Candidate:
    if ALIAS.fullmatch(name) is None:
        raise PresetError(f"[{name}]: the alias must match [a-z0-9][a-z0-9._-]*")
    repo = section.get("hf-repo", "").strip()
    if REPO.fullmatch(repo) is None:
        raise PresetError(f"[{name}]: hf-repo must look like org/name")
    file = section.get("hf-file", "").strip()
    if not file.endswith(".gguf"):
        raise PresetError(f"[{name}]: hf-file must name a .gguf file")
    disable_thinking = False
    raw = section.get("chat-template-kwargs")
    if raw is not None:
        try:
            kwargs: object = json.loads(raw)
        except ValueError as exc:
            raise PresetError(f"[{name}]: chat-template-kwargs is not valid JSON") from exc
        if not isinstance(kwargs, dict):
            raise PresetError(f"[{name}]: chat-template-kwargs must be a JSON object")
        disable_thinking = kwargs.get("enable_thinking") is False  # pyright: ignore[reportUnknownMemberType]
    return Candidate(name, repo, file, disable_thinking)


def load_presets(path: Path) -> list[Candidate]:
    # No interpolation: llama-server values may contain "%".
    parser = configparser.ConfigParser(interpolation=None)
    try:
        text = path.read_text(encoding="utf-8")
        parser.read_string(text)
    except OSError as exc:
        raise PresetError(f"presets file {path} cannot be read: {exc.strerror}") from exc
    except configparser.Error as exc:
        raise PresetError(f"presets file {path} is malformed: {exc.message}") from exc
    candidates = [_candidate(name, parser[name]) for name in parser.sections() if name != "*"]
    if not candidates:
        raise PresetError(f"presets file {path} has no model section")
    return candidates


def parse_size(text: str) -> int:
    match = SIZE.fullmatch(text)
    if match is None:
        raise ValueError(f"size {text!r} must be a number with an optional b, k, m or g suffix")
    return int(match[1]) * UNITS[match[2].lower()]
