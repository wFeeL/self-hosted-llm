import re
from math import isfinite
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

Phase = Literal["warmup", "series", "long", "sequential", "parallel", "queue"]

_QUANT: Final = re.compile(r"(Q\d_K_[SML]|Q\d_\d|F16|BF16)")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RequestRecord(_Model):
    type: Literal["request"] = "request"
    phase: Phase
    index: int  # order inside the phase, from 0
    client: int  # 0, or 0/1 in the parallel phase
    item_id: str  # "long" for the long input
    started_s: float  # seconds from the run start
    status: int | None  # None without an HTTP answer
    # "ok", API error code, "http_<code>", "client_timeout", "transport_error", "runtime_busy"
    outcome: str
    wall_ms: int
    prompt_tokens: int | None  # processed + cached
    cached_tokens: int | None
    completion_tokens: int | None


class CgroupReading(_Model):
    container: Literal["llm", "api"]
    cpu_usage_usec: int | None
    memory_current: int | None
    memory_anon: int | None  # "anon" of memory.stat


class Reading(_Model):
    type: Literal["reading"] = "reading"
    phase: Phase
    edge: Literal["start", "end"]
    at_s: float
    metrics: dict[str, float]  # llamacpp:* counters from /metrics, empty if unavailable
    cgroups: list[CgroupReading]


def parse_int(text: str) -> int | None:
    try:
        return int(text.strip())
    except ValueError:
        return None


def parse_metrics(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2 or parts[0].startswith("#"):
            continue
        try:
            value = float(parts[1])
        except ValueError:
            continue
        # nan and inf parse as floats but are not counters.
        if isfinite(value):
            values[parts[0]] = value
    return values


def parse_memory_stat(text: str) -> dict[str, int]:
    stat: dict[str, int] = {}
    for line in text.splitlines():
        key, _, value = line.partition(" ")
        number = parse_int(value)
        if key and number is not None:
            stat[key] = number
    return stat


def parse_cpu_stat(text: str) -> int | None:
    return parse_memory_stat(text).get("usage_usec")


def quant_of(file_name: str) -> str | None:
    match = _QUANT.search(file_name)
    return match.group(1) if match else None
