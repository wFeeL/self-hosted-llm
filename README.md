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

## Benchmarks

`scripts/bench.py run` measures the whole service: a client on the host calls `POST /classify` through the published API port and reads CPU and memory from the cgroups of both containers at the start and end of each phase. One run is a cold start of the stack with the model already in the volume, 3 warm-up requests, 20 requests one at a time, one long input (3000 characters of base64), 50 requests one at a time (default model only), two clients sending 10 requests each, and a burst of `MAX_CONCURRENCY + MAX_QUEUE + 1` requests to check the queue. Inputs are 20 rows of `eval/dataset.jsonl`. Raw files are in `bench/results/`, and `scripts/bench.py report bench/results/*.jsonl` prints the tables below; a test checks that they match.

The runs used `podman compose` on a shared host: 8 vCPU AMD EPYC Rome (AVX2), 15 GB RAM, no GPU, with another project's stack of 11 containers running. Each file records the load average before the run.

<!-- bench-table:start -->
Engine: podman, podman version 5.7.0; 8 vCPU x86_64; llm threads 6, context 8192, slots 2, batch 512.

| Model | Quant | GGUF | Load at start | Cold start to /health, s | Cold start to /ready, s | p50, s | p95, s | Max, s | n ok | Non-ok | Generation, tok/s | Prompt, tok/s | CPU llm, % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| granite-4.1-3b-q4km | Q4_K_M | 1.96 GiB | 1.46 | 5.1 | 13.0 | 7.2 | 8.8 | 9.3 | 20 | 0 | 10.1 | 41.6 | 561.7 |
| qwen3-4b-instruct-2507-q4km | Q4_K_M | 2.33 GiB | 0.27 | 4.6 | 13.5 | 8.1 | 9.3 | 9.7 | 20 | 0 | 9.6 | 36.9 | 580.6 |
| qwen3.5-2b-q4km | Q4_K_M | 1.30 GiB | 1.28 | 2.5 | 13.0 | 5.8 | 10.8 | 11.2 | 20 | 0 | 12.0 | 87.0 | 532.3 |

| Model | Long input, s | Long input outcome | Two clients p95, s | Two clients non-ok | Queue probe: immediate 429 | Queue probe: 429 after wait | Queue probe: other non-ok |
|---|---|---|---|---|---|---|---|
| granite-4.1-3b-q4km | 54.2 | ok | 15.7 | 0 | 1 | 0 | 0 |
| qwen3-4b-instruct-2507-q4km | 70.9 | ok | 19.1 | 0 | 1 | 1 | 0 |
| qwen3.5-2b-q4km | 24.6 | ok | 15.0 | 0 | 1 | 0 | 0 |

| Model | p50 first 10, s | p95 first 10, s | p50 last 10, s | p95 last 10, s | Memory before | Memory after | n ok | Non-ok |
|---|---|---|---|---|---|---|---|---|
| granite-4.1-3b-q4km | - | - | - | - | - | - | - | - |
| qwen3-4b-instruct-2507-q4km | - | - | - | - | - | - | - | - |
| qwen3.5-2b-q4km | 6.1 | 7.6 | 6.8 | 12.5 | 0.85 GiB | 0.94 GiB | 50 | 0 |

| Model | llm peak | llm file | api peak | api file | GGUF | RAM floor | RAM, conservative | RAM, worst | llm image | api image |
|---|---|---|---|---|---|---|---|---|---|---|
| granite-4.1-3b-q4km | 2.45 GiB | 0.03 GiB | 0.06 GiB | - | 1.96 GiB | 2.34 GiB | 4.29 GiB | 4.40 GiB | 0.78 GiB | 0.19 GiB |
| qwen3-4b-instruct-2507-q4km | 5.77 GiB | 2.28 GiB | 0.09 GiB | - | 2.33 GiB | 3.06 GiB | 5.39 GiB | 8.09 GiB | 0.78 GiB | 0.19 GiB |
| qwen3.5-2b-q4km | 2.34 GiB | 1.32 GiB | 0.10 GiB | - | 1.30 GiB | 1.01 GiB | 2.31 GiB | 3.64 GiB | 0.78 GiB | 0.19 GiB |

| Model | llm need | api peak | Disk |
|---|---|---|---|
| granite-4.1-3b-q4km | 4.29 GiB | 0.06 GiB | 2.92 GiB |
| qwen3-4b-instruct-2507-q4km | 5.77 GiB | 0.09 GiB | 3.29 GiB |
| qwen3.5-2b-q4km | 2.34 GiB | 0.10 GiB | 2.27 GiB |
<!-- bench-table:end -->

How to read the tables:

- Latency is wall time of one `POST /classify` from the client, including the queue, both attempts and the API. Generation and prompt speeds come from the llama-server counters for the 20-request series.
- `CPU llm, %` is the CPU time of the `llm` container over the 20-request series divided by wall time, where 100% is one core. With `LLM_CPUS=6` and 6 threads the runtime used 5.3 to 5.8 cores in that series and about 5.2 in the other phases.
- `RAM floor` is the largest anonymous memory of `llm` (the process, its buffers and the KV cache). `RAM, conservative` adds the whole GGUF file, as if every page of the file stayed in the page cache; it can count the weights twice when the runtime also keeps them in anonymous memory, which is likely for Granite, whose anonymous memory (2.34 GiB) is larger than its file (1.96 GiB). The measured peak of the container can still be higher, because the page cache also holds other files. `RAM, worst` adds the file to the peak. `llm need` in the last table is the larger of the conservative value and the measured peak, and the requirements below use it.
- The queue probe sends more requests than the service accepts; at least one is refused at once with 429, as configured.

### Resource requirements

For the default model (Qwen3.5-2B, Q4_K_M):

| Resource | Requirement | Source |
|---|---|---|
| CPU | x86_64 with AVX2; measured with 6 threads on 8 vCPU, smaller machines were not measured | `CPU llm, %` above |
| RAM | 2.34 GiB for `llm` and 0.10 GiB for `api` (`llm need` and `api peak`) | last table |
| Disk | 1.0 GiB for the two images and 1.3 GiB for the model file; every other model adds its GGUF to the `models` volume | memory table |

Qwen3-4B needs 5.77 GiB for `llm` and Granite 4.1 3B 4.29 GiB by the same measure. With the default `LLM_MEM_LIMIT=6g`, Qwen3-4B peaked at 5.77 GiB, so it has about 0.2 GiB to spare; give it a larger limit on a machine that allows one. Speed on machines with fewer cores was not measured.

### Limitations of the measurements

- All numbers come from podman on a shared host, not from Docker on a dedicated one. Every run had the 11 containers of the other project up; the load average at the start of each run is in the first table. A first run of the default model had single requests several times slower than the median and was repeated; the repeat started at a load average of 1.28 and still has single requests up to 21.6 s in the 50-request series, so the tail of the latency is partly the host.
- Inputs are short error messages (up to a few hundred characters). One long input per model is measured, not a series of them.
- Concurrency stops at two clients, which is the number of runtime slots.
- Only the default model ran the 50-request series; the other two skip it to keep each run under 15 minutes.
- The cold start begins with the model file already in the volume and probably in the page cache of the host, so loading from a cold disk takes longer. The time to `/ready` includes the healthcheck interval of `llm` (10 s): `api` starts only after `llm` is reported healthy, so this column moves in steps of the interval. The first start also downloads the file, and its time depends on the network.

Decisions and module layout are in [docs/architecture.md](docs/architecture.md). The full README comes later.
