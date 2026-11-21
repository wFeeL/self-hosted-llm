import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

KNOWN_FLAGS: Final = ("asimd", "avx2", "avx512f")
_MODEL_KEYS: Final = frozenset({"model name", "Model"})
_FLAG_KEYS: Final = frozenset({"flags", "Features"})


@dataclass(frozen=True)
class HostInfo:
    cpus: int
    cpu_model: str | None
    cpu_flags: tuple[str, ...]  # only of KNOWN_FLAGS, sorted
    mem_total: int | None  # bytes
    mem_available: int | None
    disk_free: int | None  # bytes free on models_dir


@dataclass(frozen=True)
class ChildProcess:
    pid: int
    argv: tuple[str, ...]


@dataclass(frozen=True)
class ProcessMemory:
    rss_anon: int | None  # bytes
    vm_hwm: int | None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _fields(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            pairs.append((key.strip(), value.strip()))
    return pairs


def _kb(fields: dict[str, str], key: str) -> int | None:
    parts = fields.get(key, "").split()
    if len(parts) == 2 and parts[1] == "kB" and parts[0].isdecimal():
        return int(parts[0]) * 1024
    return None


def _disk_free(path: Path) -> int | None:
    try:
        stat = os.statvfs(path)
    except OSError:
        return None
    return stat.f_bavail * stat.f_frsize


def read_host(proc: Path, models_dir: Path, cpus: Callable[[], int]) -> HostInfo:
    cpu_model: str | None = None
    flags: set[str] = set()
    cpuinfo = _read_text(proc / "cpuinfo")
    for key, value in _fields(cpuinfo or ""):
        if key in _MODEL_KEYS and cpu_model is None:
            cpu_model = value
        elif key in _FLAG_KEYS:
            flags.update(value.split())
    meminfo = dict(_fields(_read_text(proc / "meminfo") or ""))
    return HostInfo(
        cpus=cpus(),
        cpu_model=cpu_model,
        cpu_flags=tuple(sorted(flags & set(KNOWN_FLAGS))),
        mem_total=_kb(meminfo, "MemTotal"),
        mem_available=_kb(meminfo, "MemAvailable"),
        disk_free=_disk_free(models_dir),
    )


def _argv(directory: Path) -> tuple[str, ...] | None:
    try:
        raw = (directory / "cmdline").read_bytes()
    except OSError:
        return None
    parts = raw.decode("utf-8", errors="replace").split("\0")
    if parts and parts[-1] == "":
        parts.pop()
    return tuple(parts)


def child_arg(child: ChildProcess, flag: str) -> str | None:
    argv = child.argv
    for i, part in enumerate(argv[:-1]):
        if part == flag:
            return argv[i + 1]
    return None


def find_child(alias: str, proc: Path) -> ChildProcess | None:
    try:
        entries = sorted(proc.iterdir())
    except OSError:
        return None
    found: list[ChildProcess] = []
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        argv = _argv(entry)
        if argv is None:
            continue
        child = ChildProcess(int(entry.name), argv)
        if child_arg(child, "--alias") == alias:
            found.append(child)
    return found[0] if len(found) == 1 else None


def read_memory(pid: int, proc: Path) -> ProcessMemory | None:
    text = _read_text(proc / str(pid) / "status")
    if text is None:
        return None
    fields = dict(_fields(text))
    return ProcessMemory(rss_anon=_kb(fields, "RssAnon"), vm_hwm=_kb(fields, "VmHWM"))


def model_file_size(models_dir: Path, hf_repo: str, hf_file: str) -> int | None:
    org, _, name = hf_repo.partition("/")
    for path in sorted(models_dir.glob(f"models--{org}--{name}/**/{hf_file}")):
        try:
            return path.stat().st_size
        except OSError:
            continue
    return None
