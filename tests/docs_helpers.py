import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def env_example() -> dict[str, str]:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    return dict(re.findall(r"^([A-Z_]+)=(.*)$", text, re.MULTILINE))
