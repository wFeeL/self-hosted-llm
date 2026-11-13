# Architecture

The service classifies error messages with a small language model that runs
locally on CPU. Two containers make up the stack: `api` (FastAPI) and `llm`
(llama-server from llama.cpp). Rows marked `(planned)` describe decisions for
parts that are not built yet.

## Services

```
client --> api :8000 --> llm :8080
           published     internal network only
           on API_BIND   (debug: compose.debug.yaml publishes 127.0.0.1:8080)
           :API_PORT
                         volume "models" mounted at /models in llm
```

- `api` starts after `llm` reports healthy and answers `/health` (always) and
  `/ready` (only when the runtime lists the configured model).
- `llm` downloads the model file on first start into the named volume `models`
  and reuses it afterwards.
- Configuration lives in `.env` (see `.env.example`); compose passes each
  service only its own variables.

## Decisions

| # | Decision | Alternative rejected | Price |
|---|---|---|---|
| 1 | Two compose services, `api` (FastAPI) and `llm` (llama-server), on one internal network | One container with llama-cpp-python | A network call and a second healthcheck, but changing the model or runtime does not touch the API code |
| 2 | (planned) The API talks to the runtime only through the OpenAI-compatible `/v1/chat/completions` | The native llama-server API | No native slot timings, but the runtime can be replaced without changing the client |
| 3 | (planned) The runtime enforces the JSON schema through `response_format` (json_schema); the model does not see the schema, fields and enums are described in the prompt; the API validates the answer again with a pydantic model. See [server-common.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-common.cpp) | Prompt and parsing only | The schema stays flat because of converter limits ([grammars/README.md](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)); invalid JSON becomes rare, the check of enums and meaning remains |
| 4 | (planned) One retry on an invalid model answer, with the error text, then 502 `model_output_invalid` | A placeholder `unknown`, unlimited retries | The second request doubles latency in the rare case |
| 5 | (planned) A semaphore in the API equal to the number of runtime slots; a request waits at most `QUEUE_TIMEOUT_S`, then gets 429 with `Retry-After` | Relying on the llama-server queue | The llama-server queue has no limit and no deadline and never returns 429 ([server-queue.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-queue.cpp)) |
| 6 | (planned) A timeout `REQUEST_TIMEOUT_S` on every runtime call, then 504 and cancellation of the HTTP call | No timeout | That the runtime stops generating after a cancelled call is assumed and still has to be checked |
| 7 | The model is set in `.env`: repository and GGUF file for `llm`, one alias for both services | Model in code or in the image | Changing the model means editing `.env` and restarting with `docker compose up -d llm` |
| 8 | (planned) Answers are in English | Russian, as in the example of the task statement | The difference from the example is explained in the README |
| 9 | (planned) Quality and model comparison come from one script over a labelled set, which also reports latency | Checking a few examples by hand | The set is written and labelled by hand |
| 10 | (planned) Benchmarks follow a fixed protocol and the raw results are stored in the repository next to the table | Rough numbers | CPU time for the runs, and the machine is shared with another project |
| 11 | `llm` image is `ghcr.io/ggml-org/llama.cpp:server-b12352@sha256:<digest>`: the tag of the latest master build at the time, pinned by digest | The floating `server` tag; a tag without a digest (GHCR tags can be overwritten, assumed) | Updates are manual; the tag format may change after v1.0.0 ([image tags](https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp/versions?filters%5Bversion_type%5D=tagged)) |
| 12 | llama-server downloads the model itself from `LLAMA_ARG_HF_REPO` and `LLAMA_ARG_HF_FILE` into the named volume `models` (`LLAMA_CACHE=/models`; see [hf-cache.cpp](https://github.com/ggml-org/llama.cpp/blob/b12360/common/hf-cache.cpp)) | An init container or a script with `huggingface-cli` | The first start depends on Hugging Face; the file revision is not pinned, so a swapped file is caught only by checking its size; the volume uses the Hugging Face hub layout with symlinks |
| 13 | Default model in `.env.example` is `ibm-granite/granite-4.1-3b-GGUF`, file `granite-4.1-3b-Q4_K_M.gguf` ([model card](https://huggingface.co/ibm-granite/granite-4.1-3b)) | Qwen3-4B-Instruct-2507 (third-party GGUF) | The Granite GGUF repository has no LICENSE file, only the license in the model metadata; if the final model changes, `.env.example` and the quick start change |
| 14 | One `MODEL_ALIAS` in `.env`: `LLAMA_ARG_ALIAS` for `llm` and `LLM_MODEL` for `api` | A generic alias such as `default` | The `model` field in answers changes with the model, which shows which model answered |
| 15 | `/ready` calls `GET {LLM_BASE_URL}/models` and succeeds only on 200 with a JSON `data` list that contains the alias in `data[].id`; otherwise 503 `llm_unavailable`. It relies on llama-server answering 503 on API paths while the model loads ([server-http.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-http.cpp)) | The native `/health` of llama-server plus `/v1/models` | A runtime that lists models before loading would give a false ready; the check is repeated when the llama-server tag changes; the client only needs to know `/v1` |
| 16 | The `llm` port is not published; `compose.debug.yaml` adds `127.0.0.1:8080:8080` (`-f compose.yaml -f compose.debug.yaml`). The `api` port is `${API_BIND:-127.0.0.1}:${API_PORT:-8000}:8000`; uvicorn always listens on 8000 inside the container | A compose profile (it applies to a whole service, not to one port); publishing on 0.0.0.0 | A second compose file; on a VPS `API_BIND=0.0.0.0` or a proxy is needed, and there is no authentication |
| 17 | The `llm` healthcheck is written out in compose: `curl -f http://localhost:8080/health`, interval 10s, timeout 5s, retries 3, `start_period` 600s | Only `start_period` on top of the image healthcheck (podman 5.7 does not see the healthcheck of that image) | It duplicates the image line; if download and load take longer than 600 s plus 30 s, `llm` becomes unhealthy and `up` does not wait for `api`, so `docker compose up -d` is repeated after the download |
| 18 | `api` waits for `llm` with `depends_on: condition: service_healthy`. Its healthcheck calls `/health` from Python (`urllib.request`, interval 10s, timeout 3s, retries 3, `start_period` 10s) | `curl` (not in the slim image); `/ready` as the healthcheck | `api` is healthy while the runtime is down; honesty lives in `/ready`. Support of the condition in podman-compose 1.5 was assumed; if `api` starts early, `/ready` still answers 503 until `llm` is ready |
| 19 | Explicit llama-server settings in `.env.example`: `LLAMA_ARG_N_PARALLEL=2`, `LLAMA_ARG_CTX_SIZE=8192` (4096 per slot), `LLAMA_ARG_THREADS=6`, `LLAMA_ARG_MMPROJ_AUTO=off`, `LLAMA_ARG_ENDPOINT_METRICS=1` (groundwork for metrics, the port is not published). `--jinja` is on by default and not set ([arg.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp)) | Server defaults (4 slots with a shared KV cache, threads by host cores, 8 here) | The values are not measured yet; 6 threads leave 2 vCPUs for the API and other work on the machine |
| 20 | `api` image is `python:3.14-slim` pinned by digest, uv from `ghcr.io/astral-sh/uv:0.12.24` pinned by digest, `uv sync --frozen --no-dev` in two layers (dependencies, then code), a non-root user, one uvicorn process | One layer with pip; a floating tag without a digest | The build depends on GHCR for uv; digests are raised by hand |
| 21 | Limits come from `.env`: `llm` 6 CPUs and 6g, `api` 1 CPU and 512m; `restart: unless-stopped` on both | No limits; limits as numbers in compose | The values are assumed (2.1 GB model plus KV cache) and not measured; start after a reboot is not checked yet |
| 22 | Async tests use the pytest plugin of anyio, which FastAPI and httpx already depend on | pytest-asyncio | The marker is `anyio` instead of `asyncio` |
| 23 | `.gitignore` is the first commit on `main` (`.env`, `.venv/`, `__pycache__/`, `*.gguf`, tool caches) | A branch from an empty `main` | One housekeeping commit |
| 24 | HTTP tests of the app use `httpx.AsyncClient` with `httpx.ASGITransport` and run the app lifespan by hand | Starlette `TestClient` | Starlette 1.10 types `TestClient` through the `httpx2` package, which would be another dependency, and pyright strict fails without it; the cost is a small helper instead of the standard client |

## Modules

Package `error_classifier`. `Status` shows what exists after the skeleton.

| Module | Responsibility | Status |
|---|---|---|
| `config.py` | Settings from the environment with pydantic-settings, checked at startup | present |
| `errors.py` | Service error taxonomy with code, HTTP status and text | partial (base class and `llm_unavailable` only) |
| `llm/client.py` | httpx AsyncClient, OpenAI chat completions, timeouts, usage from the answer | partial (readiness probe only) |
| `classify/schema.py` | Pydantic model of the answer, enums, JSON schema for `response_format` | planned |
| `classify/prompt.py` | System prompt, few-shot examples, message assembly, fencing of the input | planned |
| `classify/service.py` | Pipeline: normalisation, request, validation, retry, latency | planned |
| `concurrency.py` | Semaphore that waits at most `QUEUE_TIMEOUT_S` | planned |
| `api/app.py`, `api/routes.py` | FastAPI app, error handlers, `/classify`, `/health`, `/ready` | partial (`/health` and `/ready` only) |
| `scripts/evaluate.py` | Runs the set, accuracy per field, share of valid answers, p50 and p95, markdown report | planned |
| `scripts/bench.py` | Benchmark protocol: cold start, series, memory, CPU | planned |
| `data/eval_set.jsonl` | Evaluation set | planned |
| `docs/architecture.md` | Decisions (this file) | present |
| `docs/evaluation.md`, `docs/benchmarks.md` | Evaluation and benchmark results with raw files | planned |
