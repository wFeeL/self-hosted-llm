import os

# Read at import, before the autouse clean_env fixture runs.
_KEY = os.environ.get("API_KEY", "")
HEADERS: dict[str, str] = {"Authorization": f"Bearer {_KEY}"} if _KEY else {}
