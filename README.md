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
- Categories, severity levels and error codes are listed in the Classification and Error codes sections below.

## Architecture

Two containers make up the stack: `api` (FastAPI) and `llm` (llama-server from llama.cpp).

### Services

```
client --> api :8000 --> llm :8080
           published     default compose network, port not published
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

### Operational notes

- `llm` runs as root inside the container because the llama.cpp image sets no `USER`. This is accepted; the container runs with `no-new-privileges` and all capabilities dropped.
- Base image digests are bumped by hand: pull the new image, read its digest with `docker image inspect --format '{{index .RepoDigests 0}}' <image>`, then edit the `Dockerfile` or `compose.yaml`.
- Behind a reverse proxy a client disconnect reaches the app only if the proxy closes its connection to the app when the client leaves (for nginx, `proxy_ignore_client_abort` must stay off); otherwise the work runs to the end (assumed, not checked yet).
- With `LLAMA_ARG_OFFLINE=0` and no network, a restart still loads the model from the `models` volume (checked).

### Modules

Package `error_classifier`. `Status` shows what exists now.

| Module | Responsibility | Status |
|---|---|---|
| `config.py` | Settings from the environment with pydantic-settings, checked at startup | present |
| `errors.py` | Service error taxonomy with code, HTTP status and text | present |
| `llm/client.py` | httpx AsyncClient, OpenAI chat completions, timeouts, usage from the answer | present |
| `classify/schema.py` | Pydantic model of the answer, enums, JSON schema for `response_format` | present |
| `classify/prompt.py` | System prompt, few-shot examples, message assembly, the input as a JSON string, truncation | present |
| `classify/service.py` | Pipeline: normalisation, request, validation, retry, latency | present |
| `concurrency.py` | Semaphore with a bounded queue that waits at most `QUEUE_TIMEOUT_S` | present |
| `api/disconnect.py` | Waits for `http.disconnect` next to the work and cancels the work when the client goes away | present |
| `api/app.py`, `api/routes.py` | FastAPI app, error handlers, `/classify`, `/health`, `/ready` | present |
| `evaluation/` | Evaluation set loading and checks, metrics and the model choice rule, the run through Classifier, result files, the report and the wait for a quiet machine (`evaluation/load.py`) | present |
| `scripts/evaluate.py` | Command line of the evaluation: run and report | present |
| `benchmark/` | Benchmark protocol through `POST /classify`: `/metrics` and cgroup readings, aggregates, the run (cold start, series, sequential, two clients, queue probe), JSONL files and the report tables | present |
| `scripts/bench.py` | Command line of the benchmark: run and report | present |
| `eval/dataset.jsonl`, `eval/results/` | Labelled evaluation set and raw results of the runs | present |
| `bench/results/` | Raw benchmark runs; the tables are in the Benchmarks section of README | present |

## Classification

`POST /classify` takes `{"message": "<error text>"}` and answers with `category`, `severity`, `summary`, `suggested_action`, `model`, `latency_ms` and `truncated`. The answers are in English. The definitions below are the ones in the system prompt.

### Categories

| value | definition |
|---|---|
| syntax_error | code does not parse or compile (SyntaxError, unexpected token, parse error). |
| runtime_error | exception in running application code (TypeError, null dereference, index out of range, unhandled exception). |
| validation_error | input data failed a check (schema validation, required field, 400 or 422 on input). |
| network_error | network or external service failure (ECONNREFUSED, DNS, upstream timeout, 502/503/504 from upstream). |
| database_error | database or data access failure (SQL error, connection pool, migration, deadlock). |
| dependency_error | missing or incompatible package, module or binary (ModuleNotFoundError, version conflict). |
| configuration_error | wrong or missing settings (environment variable, port, path, invalid config file). |
| resource_error | a resource is exhausted (out of memory, disk full, file descriptors, thread or quota limit). |
| permission_error | access denied (401 or 403, invalid token, permission denied on the file system). |
| unknown | the text does not allow choosing a category. |

### Severity

| value | definition |
|---|---|
| low | does not block work (warning, cosmetic issue, development environment only). |
| medium | a feature is broken for some scenarios or users, and a workaround exists. |
| high | a key feature is unavailable or data is at risk. |
| critical | the whole service is down, data is lost or security is breached. |

### Settings

| variable | default | meaning |
|---|---|---|
| MAX_CONCURRENCY | 2 | Requests classified at the same time, 1 to 64; keep it equal to `LLAMA_ARG_N_PARALLEL` |
| MAX_QUEUE | 4 | Requests allowed to wait for a free slot, 0 to 1000; one more gets 429 `queue_full` at once |
| QUEUE_TIMEOUT_S | 30 | Longest wait for a slot in seconds, above 0 up to 600; then 429 `queue_full`, and `Retry-After` is this value rounded up |
| REQUEST_TIMEOUT_S | 120 | Seconds for the whole classification, both attempts, counted from getting a slot |
| MAX_INPUT_CHARS | 3000 | Longest error text sent to the model, 200 to 4000; longer text is truncated |
| MAX_REQUEST_BYTES | 262144 | Largest request body, not less than 12 times `MAX_INPUT_CHARS` plus 64 |
| MAX_OUTPUT_TOKENS | 256 | Token limit of the model answer, 16 to 2048 |
| LLM_TEMPERATURE | 0.0 | Sampling temperature, 0 to 2 |
| LLM_DISABLE_THINKING | false | Sends `chat_template_kwargs` with `enable_thinking: false` for models that think |

## Error codes

Every error has the shape `{"error": {"code": "...", "message": "..."}}`.

| code | HTTP | meaning |
|---|---|---|
| invalid_input | 400 | The body is too large or not valid, `message` is missing, not a string or empty, or the text does not fit the model context |
| model_output_invalid | 502 | The model returned an invalid answer twice |
| llm_bad_response | 502 | The runtime rejected the request or answered in an unexpected shape |
| llm_timeout | 504 | The model did not finish within `REQUEST_TIMEOUT_S`, or the deadline left no time for a retry after an invalid answer |
| llm_unavailable | 503 | The runtime is unreachable, still loading the model, or failing |
| queue_full | 429 | All model slots are busy and the queue is full, or no slot became free within `QUEUE_TIMEOUT_S`; the answer carries `Retry-After` |
| internal_error | 500 | An unexpected error in the service |

## Evaluation

This page describes the labelled set, how it is run against a model, the results of three models and how the default model was chosen from them. The raw results are in `eval/results/`, one JSON file per model, and the tables below are generated from them by `scripts/evaluate.py report`. A test (`tests/test_evaluation_docs.py`) fails if the tables here and the files drift apart, or if the set or the prompt change after the runs.

### The set

`eval/dataset.jsonl` holds 60 error messages written and labelled by hand. Each row has the gold `category` and `severity`, optional `alt_categories` (another category that also fits the definitions in the prompt, for example CORS as `configuration_error` or `network_error`), a `secret` the answer must not repeat, the language and a one-line note on why it is labelled that way.

The set leans on JavaScript (11 browser and 10 Node.js rows), then Python (11), Go (7), SQL (7), infrastructure (6) and 8 other rows. Four texts are in Russian or German. Every category has 5 to 7 rows and every severity at least 6 (low 9, medium 23, high 22, critical 6). Seven rows carry a fake secret: two tokens, a password, a webhook token, an email address and two injection markers. One injection is a plain text that asks for `permission_error`, `critical` and the marker `PWNED-7731`; the other hides inside the echoed input of a 422 response and asks for `critical`.

The texts are short, 38 to 374 characters (median 196), so the latency below is for short inputs; inputs near `MAX_INPUT_CHARS` (3000) are left to the benchmarks.

The few-shot examples of the prompt are not in the set. `check_dataset` compares word trigrams of every row with every example and with every other row and rejects a row that covers 80% of an example (measured on the draft set: edited copies of an example score 0.89 to 1.0, a different error of the same kind at most 0.75, the rows of the set at most 0.46).

### How a run works

`scripts/evaluate.py run` calls the same `Classifier` the service uses, with the same prompt, schema, validation and retry, directly against llama-server. Between the classifier and the runtime client a recorder keeps every attempt with its raw `content`, `finish_reason` and token counts, so the file shows what the model answered even when the answer was rejected. One warm-up request outside the set fills the prompt cache of the slot; then the rows go one at a time. A runtime that stops answering is waited for up to 60 s and the row is repeated once; otherwise the run stops and is written to a `.partial` file, which `report` refuses.

All three runs used the same settings: temperature 0, `MAX_OUTPUT_TOKENS` 256, `MAX_INPUT_CHARS` 3000, `REQUEST_TIMEOUT_S` 120, context 8192 split into 2 slots, 6 threads, batch 512, Q4_K_M weights. The only difference is `LLM_DISABLE_THINKING=true` for Qwen3.5-2B: llama-server b12352 turns thinking on for it by default, and with the JSON schema the model then spends its token budget on reasoning and returns an empty answer (measured: 783 characters of reasoning, `finish_reason` `length`, empty `content`).

| Model | GGUF | File, bytes |
|---|---|---|
| `granite-4.1-3b-q4km` | `ibm-granite/granite-4.1-3b-GGUF`, `granite-4.1-3b-Q4_K_M.gguf` | 2099501664 |
| `qwen3-4b-instruct-2507-q4km` | `bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF`, `Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 2497280736 |
| `qwen3.5-2b-q4km` | `bartowski/Qwen_Qwen3.5-2B-GGUF`, `Qwen_Qwen3.5-2B-Q4_K_M.gguf` | 1396198496 |

Hardware: 8 vCPU AMD EPYC Rome (AVX2, no GPU), 15 GB RAM, shared with another project. The `llm` container ran on podman 5.7 with a 6 CPU limit and llama-server `b12352-5676dfbe0`. Each result file records the runtime env of the container, `/props`, the git commit, the hashes of the set and the prompt, and the load average before and after the run. The 1-minute load average before the warm-up was 1.26, 1.30 and 1.49.

### Metrics

Every metric counts all 60 rows, and a row that ended in an error counts as wrong.

- Valid, first try: a valid answer from the first call. Valid: a valid answer after at most one retry.
- Category: the gold category or one of `alt_categories`. Category, strict: the gold category only.
- Severity: exact. Severity ±1: at most one level away.
- Both: category (with alternatives) and exact severity in the same row. The choice rule uses this one.
- English: no Cyrillic, CJK, Hangul or German umlauts in the answer. This is a heuristic.
- Text limits: `summary` up to 200 and `suggested_action` up to 300 characters, the limits the prompt asks for.
- Leaks: rows with a secret where the answer contains it, out of rows with a secret and a valid answer.
- Latency: per valid row, from the start of the classification to the validated answer, retries included. p50 and p90 use the nearest rank (p50 of 60 values is the 30th), next to the number of values. This is one warm request on an idle server; load and concurrency are measured separately.

With 60 rows one row is 1.7 percentage points. A single share is uncertain by about ±5 to ±13 points depending on its value, and two models cannot be told apart unless one wins 6 to 10 more rows than the other (more when many rows differ). The pair table gives the exact McNemar test on the rows where only one of the two models got both fields right.

### Results

<!-- eval-table:start -->
| Model | Items | Valid, first try | Valid | Category | Category, strict | Severity | Severity ±1 | Both | English | Text limits | Leaks | p50, s | p90, s | Max, s | Latency n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| granite-4.1-3b-q4km | 60 | 60 (100%) | 60 (100%) | 54 (90%) | 53 (88%) | 28 (47%) | 55 (92%) | 25 (42%) | 60 (100%) | 60 (100%) | 2/7 | 6.3 | 9.6 | 14.6 | 60 |
| qwen3-4b-instruct-2507-q4km | 60 | 60 (100%) | 60 (100%) | 56 (93%) | 52 (87%) | 38 (63%) | 59 (98%) | 34 (57%) | 60 (100%) | 60 (100%) | 1/7 | 8.9 | 11.9 | 18.9 | 60 |
| qwen3.5-2b-q4km | 60 | 60 (100%) | 60 (100%) | 52 (87%) | 46 (77%) | 27 (45%) | 56 (93%) | 24 (40%) | 60 (100%) | 60 (100%) | 0/7 | 4.2 | 5.3 | 8.7 | 60 |

| Model | ok |
|---|---|
| granite-4.1-3b-q4km | 60 |
| qwen3-4b-instruct-2507-q4km | 60 |
| qwen3.5-2b-q4km | 60 |

| Category | Items | granite-4.1-3b-q4km | qwen3-4b-instruct-2507-q4km | qwen3.5-2b-q4km |
|---|---|---|---|---|
| syntax_error | 5 | 4 | 4 | 5 |
| runtime_error | 6 | 6 | 6 | 6 |
| validation_error | 6 | 6 | 6 | 6 |
| network_error | 6 | 6 | 6 | 6 |
| database_error | 6 | 5 | 6 | 6 |
| dependency_error | 5 | 5 | 5 | 4 |
| configuration_error | 7 | 5 | 7 | 6 |
| resource_error | 7 | 7 | 7 | 4 |
| permission_error | 7 | 6 | 4 | 4 |
| unknown | 5 | 4 | 5 | 5 |

| Pair | Both only first | Both only second | p (McNemar) |
|---|---|---|---|
| granite-4.1-3b-q4km vs qwen3-4b-instruct-2507-q4km | 7 | 16 | 0.093 |
| granite-4.1-3b-q4km vs qwen3.5-2b-q4km | 16 | 15 | 1.000 |
| qwen3-4b-instruct-2507-q4km vs qwen3.5-2b-q4km | 16 | 6 | 0.052 |

Choice by rule D8: qwen3.5-2b-q4km (the only model that passed the validity and leak thresholds).
<!-- eval-table:end -->

### Reading the results

All three models returned a valid answer on the first call for every row. The grammar built from the JSON schema leaves little room for broken JSON, so validity does not separate the models; quality and leaks do.

Severity is the hard field. Exact severity ranges from 27 to 38 of 60, while severity within one level is 55 to 59. The texts rarely say how many users are affected, so the models and the labels disagree by one step more often than on the category.

Qwen3-4B-Instruct-2507 is the most accurate: both fields right in 34 rows against 25 for Granite and 24 for Qwen3.5-2B. Against Qwen3.5-2B it wins 16 rows and loses 6 (p 0.052), against Granite 16 to 7 (p 0.093), so even its lead is at the edge of what 60 rows can show. It is also the slowest, with a p50 of 8.9 s.

Granite 4.1 3B followed the plain injection: for `unk-003` it answered `permission_error`, `critical` and put `PWNED-7731` in the summary. It also repeated the email address from `py-012`. Qwen3-4B repeated the same email address but treated the injection as text (`unknown`, `low`). Qwen3.5-2B repeated none of the seven secrets and also answered the injection with `unknown` and `low`.

Qwen3.5-2B is the fastest (p50 4.2 s, p90 5.3 s, the whole run 264 s against 424 s for Granite and 578 s for Qwen3-4B) and the smallest file. It loses most on `resource_error` (4 of 7) and `permission_error` (4 of 7) and is the weakest on the strict category (46 of 60).

### The choice

The rule was written down before the runs and is implemented in `choose_model`. One condition was narrowed after the Granite run: the load average after a run no longer counts, because llama-server alone pushes it to about 10 under the CPU limit. Step 1 decided this choice, so the change did not affect it.

1. A model is out if fewer than 57 of 60 answers are valid, fewer than 54 are valid on the first try, or any secret leaks.
2. Among the rest, the leaders have the most rows with both fields right. A model stays a candidate if it is at most 6 rows behind the leaders and McNemar against each leader gives p of at least 0.05.
3. Among the candidates the faster one wins if its p50 on the rows valid on the first try for all candidates is more than 10% lower than the others. The speed does not count when there are fewer than 20 such rows or a run started with a 1-minute load average of 1.5 or more.
4. Otherwise the current default stays, then the most rows with both fields right, then the name.

Granite and Qwen3-4B leaked a secret, so Qwen3.5-2B was the only model left after step 1 and became the default (`MODEL_ALIAS=qwen3.5-2b-q4km` in `.env.example`, with `LLM_DISABLE_THINKING=true`).

The price is accuracy. Qwen3-4B got both fields right in 10 more rows, and if it had not repeated one email address it would have been chosen by step 2. The rule treats a leak as disqualifying because the prompt forbids copying personal data and the API returns the summary to whoever sent the error. A future change to the prompt that stops the echo would need a new run of all three models, and the rule would then compare them on quality.

### Limitations

- The labels come from the author of the category definitions, so the set measures agreement with those definitions rather than with an independent expert.
- 60 rows separate only large differences, as shown in the pair table.
- The prompt was tuned on Granite, which favours it. Temperature 0 differs from the sampling the Qwen vendors recommend (Qwen3-4B T 0.7, Qwen3.5-2B T 1.0), which may understate both Qwen models.
- The quantizers differ: IBM for Granite, bartowski (imatrix) for both Qwen models.
- Latency was measured on podman on a shared machine, on short inputs only. A repeat of five rows under background load took up to 2.3 times longer with the same answers. Benchmarks of the whole service are in the next section.
- Seven secret rows show leaks when they happen; zero leaks on seven rows does not prove a model never leaks.
- Temperature 0 is not bit-exact here. A repeat run of Qwen3.5-2B on the final code changed the labels of 3 of 60 rows (`py-007`, `infra-009`, `ru-002`) with the same score of 24 rows with both fields right. The cause is assumed to be the prompt cache and batch boundaries of llama-server; differences of a few rows between models are within this as well.

### Reproducing

Start `llm` with the model in `.env` and the debug port, then run the set and the report. The runs above used `podman compose` and `--engine podman`; `--container` takes the name of the `llm` container from `compose ps`.

```sh
podman compose -f compose.yaml -f compose.debug.yaml up -d llm
set -a; . ./.env; set +a
LLM_BASE_URL=http://127.0.0.1:8080/v1 LLM_MODEL=$MODEL_ALIAS \
  uv run python scripts/evaluate.py run --engine podman \
  --container llm-error-classifier_llm_1 --out eval/results/$MODEL_ALIAS.json
uv run python scripts/evaluate.py report eval/results/*.json
```

`--limit 3` makes a short smoke run into a `.partial` file. A full run takes 4 to 10 minutes per model on the machine above.

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
