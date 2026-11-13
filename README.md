# llm-error-classifier

A service that classifies error messages by category and severity with a small language model that runs locally on CPU. `POST /classify` takes the text of an error and returns its category, severity, a short summary and a suggested first action.

## Quick start

```
cp .env.example .env
docker compose up -d
until curl -fsS http://127.0.0.1:8000/ready; do sleep 5; done
```

- The first start downloads about 2.1 GB of model weights into the `models` volume; watch it with `docker compose logs -f llm`.
- If the loop keeps waiting, check `docker compose ps` and `docker compose logs llm`; an unhealthy `llm` keeps `api` from starting.
- Every `docker compose` command needs `.env` to be present.
- If the network is slow and `llm` becomes unhealthy before the download ends, run `docker compose up -d` again after the download, or raise `LLM_START_PERIOD` in `.env`.
- `/ready` answers 503 until the runtime has loaded the model, `/health` always answers 200.
- The API has no authentication and listens on `127.0.0.1` by default; on a VPS keep `API_BIND` as is and use a reverse proxy.
- The stack has been checked with `podman compose`, not with Docker yet.
- Live tests (`uv run pytest -m live`) take `MODEL_ALIAS` and `API_URL` from the shell environment, not from `.env`.

## Classify an error

```
curl -sS -X POST http://127.0.0.1:8000/classify -H 'content-type: application/json' -d '{"message": "TypeError: Cannot read properties of undefined (reading '"'"'map'"'"')"}'
```

Response from the stack described above:

```
{"category":"runtime_error","severity":"medium","summary":"The code tried to call map on a value that is undefined, causing a TypeError.","suggested_action":"Verify that the variable before map is properly initialized and not null or undefined before calling the method.","model":"granite-4.1-3b-q4km","latency_ms":6780,"truncated":false}
```

- An answer takes several seconds on CPU: this one took 6.8 s, and long inputs take up to about a minute.
- `summary` and `suggested_action` are always in English, even for an error text in another language. The example in the task statement is in Russian; this is a deliberate difference.
- The service talks to the model only through the OpenAI-compatible API of llama-server (`/v1/chat/completions`). The mapping of runtime errors (context overflow, loading state) and the thinking switch are specific to llama-server.
- Categories, severity levels and error codes are listed in [docs/architecture.md](docs/architecture.md).

Decisions and module layout are in [docs/architecture.md](docs/architecture.md). The full README comes later.
