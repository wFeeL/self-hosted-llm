# llm-error-classifier

A service that classifies error messages by category and severity with a small language model that runs locally on CPU. The classification endpoint is not built yet: this version provides the container stack and the health and readiness endpoints.

## Quick start

```
cp .env.example .env
docker compose up -d
until curl -fsS http://127.0.0.1:8000/ready; do sleep 5; done
```

- The first start downloads about 2.1 GB of model weights into the `models` volume; watch it with `docker compose logs -f llm`.
- Every `docker compose` command needs `.env` to be present.
- If the network is slow and `llm` becomes unhealthy before the download ends, run `docker compose up -d` again after the download, or raise `LLM_START_PERIOD` in `.env`.
- `/ready` answers 503 until the runtime has loaded the model, `/health` always answers 200.
- The API has no authentication and listens on `127.0.0.1` by default; on a VPS keep `API_BIND` as is and use a reverse proxy.
- The stack has been checked with `podman compose`, not with Docker yet.
- Live tests (`uv run pytest -m live`) take `MODEL_ALIAS` and `API_URL` from the shell environment, not from `.env`.

Decisions and module layout are in [docs/architecture.md](docs/architecture.md). The full README comes later.
