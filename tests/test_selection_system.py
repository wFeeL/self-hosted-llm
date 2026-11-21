import os
from pathlib import Path

from error_classifier.selection.system import (
    ChildProcess,
    ProcessMemory,
    child_arg,
    find_child,
    model_file_size,
    read_host,
    read_memory,
)

X86_CPUINFO = """processor\t: 0
vendor_id\t: AuthenticAMD
model\t\t: 49
model name\t: AMD EPYC 7282 16-Core Processor
flags\t\t: fpu sse4_2 avx2 avx512f popcnt

processor\t: 1
model name\t: Other
flags\t\t: fpu
"""
ARM_CPUINFO = """processor\t: 0
Features\t: fp asimd evtstrm
Model\t\t: Raspberry Pi 5
"""
MEMINFO = "MemTotal:       15728640 kB\nMemFree:  100 kB\nMemAvailable:    8388608 kB\n"


def _cpus() -> int:
    return 6


def _proc_file(proc: Path, name: str, text: str) -> None:
    proc.mkdir(parents=True, exist_ok=True)
    (proc / name).write_text(text, encoding="utf-8")


def _cmdline(proc: Path, pid: int, *argv: str) -> None:
    directory = proc / str(pid)
    directory.mkdir(parents=True)
    (directory / "cmdline").write_bytes(("\0".join(argv) + "\0").encode())


def test_read_host_x86(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    _proc_file(proc, "cpuinfo", X86_CPUINFO)
    _proc_file(proc, "meminfo", MEMINFO)
    models = tmp_path / "models"
    models.mkdir()
    host = read_host(proc, models, _cpus)
    assert host.cpus == 6
    assert host.cpu_model == "AMD EPYC 7282 16-Core Processor"
    assert host.cpu_flags == ("avx2", "avx512f")
    assert host.mem_total == 15728640 * 1024
    assert host.mem_available == 8388608 * 1024
    stat = os.statvfs(models)
    assert host.disk_free == stat.f_bavail * stat.f_frsize


def test_read_host_arm(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    _proc_file(proc, "cpuinfo", ARM_CPUINFO)
    host = read_host(proc, tmp_path, _cpus)
    assert host.cpu_model == "Raspberry Pi 5"
    assert host.cpu_flags == ("asimd",)


def test_read_host_missing_files(tmp_path: Path) -> None:
    host = read_host(tmp_path / "proc", tmp_path / "models", _cpus)
    assert host.cpus == 6
    assert host.cpu_model is None
    assert host.cpu_flags == ()
    assert host.mem_total is None
    assert host.mem_available is None
    assert host.disk_free is None


def test_find_child_exact_alias(tmp_path: Path) -> None:
    _cmdline(tmp_path, 10, "llama-server", "--alias", "qwen3-4b-instruct", "--hf-file", "a.gguf")
    _cmdline(tmp_path, 11, "llama-server", "--alias", "qwen3-4b", "--hf-file", "b.gguf")
    _cmdline(tmp_path, 12, "llama-server", "--models-preset", "/select/presets.ini")
    child = find_child("qwen3-4b", tmp_path)
    assert child is not None
    assert child.pid == 11
    assert child_arg(child, "--hf-file") == "b.gguf"
    assert find_child("qwen3", tmp_path) is None


def test_find_child_two_matches_is_none(tmp_path: Path) -> None:
    _cmdline(tmp_path, 10, "llama-server", "--alias", "m")
    _cmdline(tmp_path, 11, "llama-server", "--alias", "m")
    assert find_child("m", tmp_path) is None


def test_find_child_skips_unreadable(tmp_path: Path) -> None:
    (tmp_path / "20").mkdir()
    (tmp_path / "self").mkdir()
    _cmdline(tmp_path, 21, "llama-server", "--alias", "m")
    child = find_child("m", tmp_path)
    assert child is not None
    assert child.pid == 21


def test_child_arg() -> None:
    child = ChildProcess(1, ("srv", "--alias", "m", "--ctx-size"))
    assert child_arg(child, "--alias") == "m"
    assert child_arg(child, "--ctx-size") is None
    assert child_arg(child, "--missing") is None


def test_read_memory_parses_kb(tmp_path: Path) -> None:
    directory = tmp_path / "7"
    directory.mkdir()
    (directory / "status").write_text(
        "Name:\tllama\nVmHWM:\t  2048 kB\nRssAnon:\t 1024 kB\nRssFile:\t 5 kB\n", encoding="utf-8"
    )
    assert read_memory(7, tmp_path) == ProcessMemory(rss_anon=1024 * 1024, vm_hwm=2048 * 1024)


def test_read_memory_missing(tmp_path: Path) -> None:
    assert read_memory(7, tmp_path) is None


def test_model_file_size_follows_symlink(tmp_path: Path) -> None:
    repo = tmp_path / "models--org--name"
    blob = repo / "blobs" / "abc123"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"x" * 1234)
    link = repo / "snapshots" / "x" / "m.gguf"
    link.parent.mkdir(parents=True)
    link.symlink_to(blob)
    assert model_file_size(tmp_path, "org/name", "m.gguf") == 1234


def test_model_file_size_missing(tmp_path: Path) -> None:
    assert model_file_size(tmp_path, "org/name", "m.gguf") is None
