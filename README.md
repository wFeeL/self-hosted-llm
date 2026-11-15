# llm-error-classifier

A service that classifies error messages by category and severity with a small language model that runs locally on CPU. `POST /classify` takes the text of an error and returns its category, severity, a short summary and a suggested first action.

## Results at a glance

- Default model: Qwen3-4B-Instruct-2507 (Q4_K_M, 2.5 GB file) on llama.cpp `llama-server`, CPU only. Servers with 4 GiB of RAM or 2 vCPU should take Qwen3.5-2B (1.4 GB).
- Quality on the 60 labelled errors: category right in 56 (52 when it must match strictly), severity right in 38, both right in 34, against 24 for Qwen3.5-2B, the earlier default. Both injections in the set were treated as text. The service masks email addresses in its answers; the model itself repeated one address from an error text, and that email was the only reason it lost the first choice. The rule was changed after the runs, see [The choice](#the-choice).
- Speed through the API on 8 vCPU (6 threads): p50 8.1 s and p95 9.3 s over 20 requests, 70.9 s for one 3000-character input. The 50-request series was run only for Qwen3.5-2B (p50 5.8 s, p95 10.8 s, 24.6 s for the long input).
- Resources: at least 5.77 GiB of RAM for `llm` (measured under a 6 GiB limit, so the real need may be higher; the default limit is now 6.5 GiB) and 0.09 GiB for `api`, 3.3 GiB of disk for the images and the model, about 5.8 CPU cores while answering.
- APIs: `POST /classify` and an OpenAI-compatible `/v1/chat/completions`, with a queue, timeouts, an optional API key and honest errors instead of made-up answers.

The numbers come from podman on a shared machine; see [Benchmarks](#benchmarks) and [Evaluation](#evaluation) for the method and [Limitations](#limitations) for what they do not cover.

## Quick start

```
cp .env.example .env
docker compose up -d
until curl -fsS http://127.0.0.1:8000/ready; do sleep 5; done
```

- The first start downloads about 2.5 GB of model weights into the `models` volume; watch it with `docker compose logs -f llm`.
- `.env.example` limits `llm` to 6 CPUs, and Docker refuses to start a container with a limit above the number of cores. On a machine with fewer than 7 cores (`nproc`), take the values from the [sizing table](#3-get-the-code-and-the-configuration) first.
- If the loop keeps waiting, check `docker compose ps` and `docker compose logs llm`; an unhealthy `llm` keeps `api` from starting.
- Every `docker compose` command needs `.env` to be present.
- If the network is slow and `llm` becomes unhealthy before the download ends, run `docker compose up -d` again after the download, or raise `LLM_START_PERIOD` in `.env` (1800 s by default).
- `/ready` answers 503 until the runtime has loaded the model, `/health` always answers 200.
- The API listens on `127.0.0.1` by default. With `API_KEY` set in `.env`, `/classify` and `/v1/*` require `Authorization: Bearer <key>`; `/health` and `/ready` stay open for the healthchecks. On a VPS keep `API_BIND` as is, set the key and use a reverse proxy.
- The stack has been checked with `podman compose`, not with Docker yet.
- Live tests (`uv run pytest -m live`) take `MODEL_ALIAS` and `API_URL` from the shell environment, not from `.env`.

## Classify an error

```
curl -sS -X POST http://127.0.0.1:8000/classify -H 'content-type: application/json' -d @- <<'EOF'
{"message": "TypeError: Cannot read properties of undefined (reading 'map')\n    at renderProducts (catalog.js:42:18)\n    at loadCatalog (catalog.js:17:5)"}
EOF
```

This text is row `js-001` of the evaluation set. The default model answered it in the evaluation run (`eval/results/qwen3-4b-instruct-2507-q4km.json`); that run called the classifier directly, so `model`, `latency_ms` and `truncated` below are filled in from the row. An answer through the running stack has not been recorded yet.

```
{"category":"runtime_error","severity":"medium","summary":"The product list rendering fails because the data object is undefined, causing a map operation to crash.","suggested_action":"Add a null/undefined check before calling map in renderProducts to prevent crashes.","model":"qwen3-4b-instruct-2507-q4km","latency_ms":6896,"truncated":false}
```

- An answer takes several seconds on CPU: this one took 6.9 s. A 3000-character input took 71 s with the default model and 25 s with Qwen3.5-2B in the benchmarks.
- Email addresses in `summary` and `suggested_action` come back as `<email>`.
- `summary` and `suggested_action` are always in English, even for an error text in another language. The example in the task statement is in Russian; this is a deliberate difference.
- The service talks to the model only through the OpenAI-compatible API of llama-server (`/v1/chat/completions`). The mapping of runtime errors (context overflow, loading state) and the thinking switch are specific to llama-server.
- Categories, severity levels and error codes are listed in the Classification and Error codes sections below.

## OpenAI-compatible API

Next to `POST /classify` the service answers the OpenAI chat completions protocol, so the official `openai` client and tools built on it can call it. The answer of the assistant is the classification as a JSON string.

| Endpoint | What it does |
|---|---|
| `GET /v1/models` | lists the one served model, its id is `MODEL_ALIAS` |
| `GET /v1/models/{model}` | the same entry, or 404 `model_not_found` for another name |
| `POST /v1/chat/completions` | classifies the text of the last `user` message |

```sh
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H 'content-type: application/json' \
  -d '{"model": "qwen3-4b-instruct-2507-q4km", "messages": [{"role": "user", "content": "TypeError: Cannot read properties of undefined (reading '"'"'map'"'"')"}]}'
```

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="<API_KEY or any string>")
answer = client.chat.completions.create(
    model="qwen3-4b-instruct-2507-q4km",
    messages=[
        {
            "role": "user",
            "content": "TypeError: Cannot read properties of undefined (reading 'map')",
        }
    ],
)
print(json.loads(answer.choices[0].message.content))
```

How requests are read:

- The last message must have the role `user`; its text (a string, or text parts joined by new lines) is the error to classify. Earlier messages and the client's `system` or `developer` messages are ignored, so a chat history does not change the answer and cannot change the prompt.
- `model` must be the served alias; another name gets 404 rather than an answer from a different model under that name.
- Sampling parameters (`temperature`, `top_p`, `max_tokens`, `max_completion_tokens`, `seed`, `stop`, penalties) and `response_format` of type `text` or `json_object` are accepted and ignored: the classification uses the settings from `.env`.
- Parameters that would change the shape of the answer are refused with 400 and the name in `param`: `stream: true`, `n` above 1, `tools`, `tool_choice` other than `none`, `functions`, `function_call`, `logprobs: true`, `top_logprobs`, `response_format` of type `json_schema`, `audio`, and `modalities` other than `text`. Explicit `null` and empty lists pass.
- The answer has `usage` when the runtime reported both token counts, and two extra fields, `truncated` and `latency_ms`, as in `/classify`.

Errors use the OpenAI shape, `{"error": {"code", "message", "type", "param"}}`, with the same codes and HTTP statuses as `/classify`, so the client raises `BadRequestError`, `AuthenticationError`, `NotFoundError`, `RateLimitError` or `InternalServerError`. Answers 500, 502 and 504 carry `x-should-retry: false` and the client honours it: after `model_output_invalid` the service has already asked the model twice, `llm_bad_response` and `internal_error` would fail the same way again, and a 504 would mean another `REQUEST_TIMEOUT_S` of waiting. 429 and 503 stay retryable: with the defaults (`max_retries=2`, `QUEUE_TIMEOUT_S=30`) a client gets `RateLimitError` after about 60 s when the queue is already full (each attempt is refused at once, with 30 s pauses between them) and after about 150 s when every attempt waits the full `QUEUE_TIMEOUT_S` in the queue. The `openai` 3.51 client waits for `Retry-After` only up to 120 s and does not retry above that, so with `QUEUE_TIMEOUT_S` over 120 a full queue fails at once; set `max_retries` in the client to what suits you.

The tests in `tests/test_openai_api.py` call the service through the official client (version 3.51) over an in-process transport and validate the raw answers with the client's own models. A live run against the model was not made on this machine.

## Choosing a model

The model is configuration, not code: set the variables of one block below in `.env` and recreate `llm` with `docker compose up -d`. A new model is downloaded into the `models` volume on its first start. Any other GGUF on Hugging Face works the same way; `LLM_DISABLE_THINKING=true` is needed for models that think by default.

Recommended: `qwen3-4b-instruct-2507-q4km`

The table applies the evaluation rule to the models whose measured memory fits each VPS size: `llm need` and `api peak` from the benchmarks plus 0.75 GiB for the system and Docker (an estimate). `scripts/recommend.py` prints it, and a test checks that it matches.

<!-- vps-table:start -->
| VPS RAM | Models that fit | Recommended | Reason |
|---|---|---|---|
| 2 GiB | - | none | no measured model fits |
| 4 GiB | qwen3.5-2b-q4km | qwen3.5-2b-q4km | the only model that passed the validity and leak thresholds |
| 8 GiB | granite-4.1-3b-q4km, qwen3-4b-instruct-2507-q4km, qwen3.5-2b-q4km | qwen3-4b-instruct-2507-q4km | best on both fields, the others are more than 6 rows behind or McNemar p < 0.05 |
| 16 GiB | granite-4.1-3b-q4km, qwen3-4b-instruct-2507-q4km, qwen3.5-2b-q4km | qwen3-4b-instruct-2507-q4km | best on both fields, the others are more than 6 rows behind or McNemar p < 0.05 |

Not measured: granite-4.0-1b-q4km.
<!-- vps-table:end -->

- 2 GiB: no model fits with the default context of 8192 tokens. Even the 0.95 GiB file of Granite 4.0 1B plus the smallest measured runtime memory (1.01 GiB for Qwen3.5-2B) is above the budget, and Granite 4.0 1B also followed the injection in the evaluation.
- 4 GiB: Qwen3.5-2B is the only measured model that fits. Use its block with the 4 GB row of the [sizing table](#3-get-the-code-and-the-configuration); the fit is computed from the measured memory and was not tried.
- 8 GiB: the limits of `.env.example` add up to 7.75 GiB (6.5 for `llm`, 0.5 for `api`, 0.75 for the system), and a VPS sold as 8 GB often reports a little less (an estimate). The measured use is below the limits, but the margin is small.
- 16 GiB: the same choice as 8 GiB. Models of 7 to 8B parameters were not measured on this machine.

Memory limits come from the sizing table, not from these blocks.

### Qwen3-4B-Instruct-2507 (default)

<!-- model-env:qwen3-4b-instruct-2507-q4km:start -->
```
MODEL_ALIAS=qwen3-4b-instruct-2507-q4km
LLAMA_ARG_HF_REPO=bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF
LLAMA_ARG_HF_FILE=Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf
LLM_DISABLE_THINKING=false
```
<!-- model-env:qwen3-4b-instruct-2507-q4km:end -->

The default and the most accurate on the set (34 of 60 rows with both fields right, 52 strict category, 38 exact severity), and the slowest (p50 10.2 s in the evaluation, 8.1 s through the API). In one row (`py-012`) it repeated an email address from the error text; the service masks email addresses, so the client does not get it. Its `llm` container peaked at 5.77 GiB under a 6 GiB limit, with 2.28 GiB of page cache and 3.06 GiB of anonymous memory. The peak was capped by the limit, so `.env.example` gives it 6.5 GiB.

### Qwen3.5-2B

<!-- model-env:qwen3.5-2b-q4km:start -->
```
MODEL_ALIAS=qwen3.5-2b-q4km
LLAMA_ARG_HF_REPO=bartowski/Qwen_Qwen3.5-2B-GGUF
LLAMA_ARG_HF_FILE=Qwen_Qwen3.5-2B-Q4_K_M.gguf
LLM_DISABLE_THINKING=true
```
<!-- model-env:qwen3.5-2b-q4km:end -->

The choice for servers with 4 GiB of RAM or 2 vCPU, and the default before the email mask (see [The choice](#the-choice)). It repeated none of the seven secrets and treated the injection as text. Fast (p50 4.2 s) and small (2.34 GiB for `llm`), but 10 rows behind Qwen3-4B with both fields right and weaker on the strict category (46 of 60). It needs `LLM_DISABLE_THINKING=true`.

### Granite 4.1 3B

<!-- model-env:granite-4.1-3b-q4km:start -->
```
MODEL_ALIAS=granite-4.1-3b-q4km
LLAMA_ARG_HF_REPO=ibm-granite/granite-4.1-3b-GGUF
LLAMA_ARG_HF_FILE=granite-4.1-3b-Q4_K_M.gguf
LLM_DISABLE_THINKING=false
```
<!-- model-env:granite-4.1-3b-q4km:end -->

Accuracy close to Qwen3.5-2B at about twice the latency. It followed the injection and repeated an email address, so it is not recommended where error texts can come from untrusted sources.

### Granite 4.0 1B

<!-- model-env:granite-4.0-1b-q4km:start -->
```
MODEL_ALIAS=granite-4.0-1b-q4km
LLAMA_ARG_HF_REPO=ibm-granite/granite-4.0-1b-GGUF
LLAMA_ARG_HF_FILE=granite-4.0-1b-Q4_K_M.gguf
LLM_DISABLE_THINKING=false
# Memory not measured
```
<!-- model-env:granite-4.0-1b-q4km:end -->

The smallest file (about 1 GB) and the fastest (p50 3.9 s), with the weakest category accuracy (46 of 60). It followed the injection. It was not benchmarked: its memory, long inputs and concurrency are unknown.


## Model and license

Qwen3-4B-Instruct-2507 is the default because it got both fields right in the most rows (34 of 60, 10 more than the next model) and treated the injection as text; with email addresses masked by the service it passes the leak check. It needs a server with 8 GiB of RAM. The price is speed and memory: about 1.4 times slower than Qwen3.5-2B through the API and about 2.5 times its memory. Qwen3.5-2B is the recommendation for 4 GiB servers.

| Model | License | Where it is stated |
|---|---|---|
| Qwen3.5-2B | Apache-2.0 | `LICENSE` in [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B); the GGUF repository [bartowski/Qwen_Qwen3.5-2B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF) is marked apache-2.0 |
| Qwen3-4B-Instruct-2507 | Apache-2.0 | `LICENSE` in [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507); the bartowski GGUF repository names no license and points to that base model |
| Granite 4.1 3B, Granite 4.0 1B | Apache-2.0 | model card and repository metadata of [ibm-granite/granite-4.1-3b](https://huggingface.co/ibm-granite/granite-4.1-3b) and [ibm-granite/granite-4.0-1b](https://huggingface.co/ibm-granite/granite-4.0-1b) and their official GGUF repositories; the repositories have no `LICENSE` file |
| llama.cpp (runtime image) | MIT | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |

All weights are open and none of the repositories is gated (checked 2026-11-15). Apache-2.0 allows commercial use, changes and redistribution; whoever redistributes the weights has to include a copy of the license, keep the copyright and attribution notices and mark the files they changed (section 4 of the license), and the license grants no trademark rights. This service does not ship the weights: `llm` downloads them from Hugging Face on the first start.

## Configuration

All settings live in `.env`, copied from `.env.example`; compose passes each service only its own variables. Changing a model or a limit needs no change of code, only `.env` and `docker compose up -d`.

| variable | default | meaning |
|---|---|---|
| MODEL_ALIAS | `qwen3-4b-instruct-2507-q4km` | Name of the model in llama-server and in API answers; see [Choosing a model](#choosing-a-model) |
| LLAMA_ARG_HF_REPO | `bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF` | Hugging Face repository of the GGUF file |
| LLAMA_ARG_HF_FILE | `Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | GGUF file in that repository, downloaded on the first start |
| LLAMA_ARG_CTX_SIZE | 8192 | Context of llama-server, split between the slots; keep it at least `LLAMA_ARG_N_PARALLEL` × 4096 |
| LLAMA_ARG_N_PARALLEL | 2 | Slots of llama-server, and the number of requests the API classifies at once |
| LLAMA_ARG_THREADS | 6 | CPU threads of llama-server; keep it at `LLM_CPUS` or below |
| LLAMA_ARG_BATCH | 512 | Prompt batch; a cancelled request frees its slot only after the current batch |
| LLAMA_ARG_CACHE_RAM | 0 | Prompt cache of llama-server in host RAM, MiB; 0 turns it off, and the shared prompt prefix is still reused inside each slot. llama-server's own default of 8192 is above the memory limit; the evaluation and the benchmarks ran with that default |
| LLAMA_ARG_MMPROJ_AUTO | off | Keeps llama-server from downloading a vision projector the service does not use |
| LLAMA_ARG_ENDPOINT_METRICS | 1 | Enables `/metrics` of llama-server, used by the benchmark |
| LLAMA_ARG_OFFLINE | 0 | `1` loads the model only from the volume; set it after the first start and back to `0` to change the model |
| LLM_CPUS | 6 | CPU limit of the `llm` container |
| LLM_MEM_LIMIT | 6656m | Memory limit of the `llm` container; Docker adds no swap on top of it, podman-compose ignored that setting in the measurements. Take the value for your server from the sizing table in [Deploying on a clean VPS](#deploying-on-a-clean-vps) |
| API_CPUS | 1 | CPU limit of the `api` container |
| API_MEM_LIMIT | 512m | Memory limit of the `api` container |
| LLM_START_PERIOD | 1800s | How long the `llm` healthcheck tolerates failures while the model downloads and loads; the service is healthy at the first successful check, so a long window costs nothing |
| API_BIND | 127.0.0.1 | Address the API port is published on; keep it local and put a reverse proxy in front |
| API_PORT | 8000 | Published API port |
| LLM_PROBE_TIMEOUT_S | 2.0 | Timeout of the readiness probe from the API to llama-server |
| LOG_LEVEL | INFO | Log level of the API; error texts are never logged in full |
| MAX_CONCURRENCY | `LLAMA_ARG_N_PARALLEL` | Requests classified at the same time; compose sets it from `LLAMA_ARG_N_PARALLEL`, so it is not in `.env` |
| MAX_QUEUE | 4 | Requests allowed to wait for a free slot, 0 to 1000; one more gets 429 `queue_full` at once |
| QUEUE_TIMEOUT_S | 30 | Longest wait for a slot in seconds, above 0 up to 600; then 429 `queue_full`, and `Retry-After` is this value rounded up |
| REQUEST_TIMEOUT_S | 120 | Seconds for the whole classification, both attempts, counted from getting a slot |
| MAX_INPUT_CHARS | 3000 | Longest error text sent to the model, 200 to 4000; longer text is truncated |
| MAX_REQUEST_BYTES | 262144 | Largest request body, not less than 12 times `MAX_INPUT_CHARS` plus 64 |
| MAX_OUTPUT_TOKENS | 256 | Token limit of the model answer, 16 to 2048 |
| LLM_TEMPERATURE | 0.0 | Sampling temperature, 0 to 2 |
| LLM_DISABLE_THINKING | required (`false` in `.env.example` for the default model, `true` for Qwen3.5-2B) | Sends `chat_template_kwargs` with `enable_thinking: false` for models that think |
| API_KEY | empty | Access key: when set, `/classify` and `/v1/*` need `Authorization: Bearer <API_KEY>` and answer 401 `unauthorized` otherwise; letters, digits and `. _ ~ + / = -` only (use `openssl rand -hex 32`); `/health` and `/ready` stay open |
| HF_TOKEN | not set | Only for gated models, passed to `llm` with a `compose.override.yaml` (see [Deploying on a clean VPS](#deploying-on-a-clean-vps)) |

## Deploying on a clean VPS

These steps target Ubuntu 24.04 LTS on a server without a GPU and use Docker Engine with the Compose plugin. Nothing in the code changes between machines, only `.env`. The stack was run here with podman; these Docker steps have not yet been run on a clean VPS.

### 1. Pick the server

Use the table in [Choosing a model](#choosing-a-model): the default model needs a server with 8 GiB of RAM, and a 4 GiB server takes Qwen3.5-2B. Plan for at least 10 GB of free disk: the images and the model file take 3.3 GiB, and Docker, the build cache, logs and every further model need room too. All measurements were taken on 8 vCPU; with fewer cores the answers are slower.

The server needs outbound HTTPS to the Docker package repository, Docker Hub (the `python` base image of `api`), `ghcr.io` (the llama.cpp image), PyPI (dependencies while `api` is built) and Hugging Face (the first download of the model). The registries and Hugging Face serve files from other hosts behind these addresses, so the two checks below only show that the entry points answer:

```sh
curl -sSI https://huggingface.co | head -1
curl -sSI https://ghcr.io | head -1
```

### 2. Install Docker Engine

Remove the unofficial packages and install Docker Engine and the Compose plugin from Docker's apt repository as described in [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/) (checked 2026-11-15); do not use the `docker.io` package of Ubuntu. On Ubuntu the Docker service starts on boot by default. Check it:

```sh
sudo systemctl is-enabled docker
docker compose version
```

The commands below run without `sudo`. That needs your user in the `docker` group, which grants root-level access to the machine ([post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/)):

```sh
sudo usermod -aG docker "$USER"
newgrp docker
```

### 3. Get the code and the configuration

```sh
git clone https://github.com/wFeeL/self-hosted-llm.git llm-error-classifier
cd llm-error-classifier
cp .env.example .env
chmod 600 .env
sed -i "s/^API_KEY=.*/API_KEY=$(openssl rand -hex 32)/" .env
```

The last line writes a random key into `.env`. `.env` is read by Compose, not by a shell, so a `$(...)` written into the file would not run, and `$`, quotes and `#` in a key would be changed by Compose; the key allows only letters, digits and `. _ ~ + / = -`. Clients send it as `Authorization: Bearer <key>`.

Size `.env` before the first start. The defaults fit the machine the measurements come from (8 vCPU, 15 GB RAM); Docker refuses a `cpus` limit above the number of cores, so smaller servers must change `LLM_CPUS` first.

| Server | Model | `LLM_CPUS` | `LLAMA_ARG_THREADS` | `API_CPUS` | `LLAMA_ARG_N_PARALLEL` | `LLAMA_ARG_CTX_SIZE` | `LLM_MEM_LIMIT` | `REQUEST_TIMEOUT_S` | `QUEUE_TIMEOUT_S` |
|---|---|---|---|---|---|---|---|---|---|
| 8 vCPU, 8 GB or more (`.env.example`; measured with a 6g limit) | default | 6 | 6 | 1 | 2 | 8192 | 6656m | 120 | 30 |
| 4 vCPU, 8 GB (estimate) | default | 3 | 3 | 0.5 | 1 | 4096 | 6656m | 300 | 90 |
| 4 vCPU, 4 GB (estimate) | Qwen3.5-2B | 3 | 3 | 0.5 | 1 | 4096 | 2500m | 180 | 60 |
| 2 vCPU, 4 GB or more (estimate) | Qwen3.5-2B | 1.5 | 1 | 0.5 | 1 | 4096 | 2500m | 300 | 150 |

- Keep `LLAMA_ARG_CTX_SIZE` at least `LLAMA_ARG_N_PARALLEL` × 4096: the context is split between slots, and a 3000-character input needs about 4096 tokens of a slot.
- Keep `LLM_MEM_LIMIT` below the RAM of the server minus about 1 GB for the system, Docker and `api` (the VPS table above reserves 0.75 GiB plus the measured peak of `api`), and above the `llm need` of your model in the benchmark tables (5.77 GiB for the default, 2.34 GiB for Qwen3.5-2B). A limit above the RAM protects nothing.
- The rows marked estimate were not measured. Their times are the 8 vCPU results (the p50 of 20 requests and one long input) scaled by 2 for 3 threads and by 6 for one thread. With one slot, a long request at the head of the queue can make the requests behind it end in 429, and a retry after an invalid answer may not fit `REQUEST_TIMEOUT_S` and end in 504.
- On 2 vCPU the table takes Qwen3.5-2B by judgement, not by the rule of [Choosing a model](#choosing-a-model), which looks only at RAM: the long input of the default model would take about 425 s on one thread (an estimate).

### 4. First start

`docker compose up -d` waits until `llm` is healthy before it starts `api`, and the first start downloads the model, so on a slow network the command itself can take many minutes. Run it in `tmux` or `screen`; if the SSH session drops, the containers already created keep running, and running the command again continues.

```sh
tmux new -s classifier
docker compose up -d --build
until curl -fsS http://127.0.0.1:8000/ready; do sleep 10; done
```

If `llm` became unhealthy before the download finished, run `docker compose up -d` again. Then classify the example from the task:

```sh
export API_KEY=$(sed -n 's/^API_KEY=//p' .env)
curl -sS -X POST http://127.0.0.1:8000/classify \
  -H "Authorization: Bearer $API_KEY" -H 'content-type: application/json' \
  -d '{"message": "TypeError: Cannot read properties of undefined (reading '"'"'map'"'"')"}'
```

After the first successful start set `LLAMA_ARG_OFFLINE=1` in `.env` and run `docker compose up -d`: the model then loads only from the volume, and a changed file on Hugging Face cannot replace it.

### 5. Start after a reboot

Both services have `restart: unless-stopped` and the Docker service starts on boot, so the stack comes back by itself ([restart policies](https://docs.docker.com/engine/containers/start-containers-automatically/)). After a reboot Docker starts both containers at once, without the order Compose keeps on `up`, so `api` answers 503 `llm_unavailable` until `llm` has loaded the model. Wait for `/ready`:

```sh
until curl -fsS http://127.0.0.1:8000/ready; do sleep 10; done
docker compose ps
```

`docker compose ps` should show both services up with a start time after the reboot. `docker compose stop` keeps the containers, but they stay stopped after a reboot until `docker compose start` or `up -d`; `docker compose down` removes them (the `models` volume stays). A `llm` process that hangs without exiting is not restarted automatically: `docker compose restart llm`.

### 6. Access from outside

Do not open the service to the network while `API_KEY` is empty: without a key `/classify` and `/v1/*` accept anyone. Keep `API_BIND=127.0.0.1` and put a reverse proxy with TLS on the host. Ports published by Docker bypass ufw rules ([Docker and ufw](https://docs.docker.com/engine/network/packet-filtering-firewalls/#docker-and-ufw)), so publishing `api` on `0.0.0.0` would expose it even with ufw on.

[Install Caddy](https://caddyserver.com/docs/install) from its apt repository, point a DNS record of your domain to the server, and put this into `/etc/caddy/Caddyfile`; Caddy gets the certificate itself once ports 80 and 443 are open:

```
classifier.example.com {
	reverse_proxy 127.0.0.1:8000
}
```

```sh
sudo systemctl reload caddy
```

Caddy has no upstream timeout by default ([reverse_proxy](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy)). If you set one, keep it above `QUEUE_TIMEOUT_S + REQUEST_TIMEOUT_S` (150 s with `.env.example`, up to 450 s for the rows of the sizing table). Before enabling ufw allow SSH, then HTTP and HTTPS:

```sh
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

Check from outside with `curl -fsS https://classifier.example.com/ready`.

### 7. Change the model

```sh
cp .env .env.bak
# edit MODEL_ALIAS, LLAMA_ARG_HF_REPO, LLAMA_ARG_HF_FILE, LLM_DISABLE_THINKING and set LLAMA_ARG_OFFLINE=0
docker compose up -d
until curl -fsS http://127.0.0.1:8000/ready; do sleep 10; done
```

When the new model answers, set `LLAMA_ARG_OFFLINE=1` again and run `docker compose up -d`. To roll back, restore `.env.bak` and run `docker compose up -d`; the old file is still in the volume, so this works offline. `docker system df -v` shows the size of the volume. To free space, delete the old file shown by `docker compose exec llm ls -R /models` while `llm` runs. `docker compose down -v` deletes the whole volume with every model, cannot be undone, and makes the next start download again.

For a gated model, request access on its Hugging Face page, put a read token into `.env` as `HF_TOKEN=...` and pass it to `llm` with a `compose.override.yaml` next to `compose.yaml` (Compose reads it by default; the file is in `.gitignore`):

```yaml
services:
  llm:
    environment:
      HF_TOKEN: ${HF_TOKEN}
```

Commands that name their files with `-f`, like the debug command in [Reproducing](#reproducing), skip the override unless it is added as one more `-f compose.override.yaml`.

### 8. Update

Code: `git pull`, then `diff .env.example .env` to see new variables (a missing required variable stops `docker compose` with a message), and `docker compose up -d --build`. The llama.cpp image is pinned as tag and digest in `compose.yaml` and the `python` base image by digest in `Dockerfile`; to update one, pull the new image, read its digest with `docker image inspect --format '{{index .RepoDigests 0}}' <image>`, and replace the part after `@sha256:` (the tag stays in front for readers).

### 9. Logs and common failures

`docker compose logs -f llm` and `docker compose logs -f api`; both are rotated at 3 files of 10 MB.

| Answer | Meaning | What to do |
|---|---|---|
| 503 `llm_unavailable` | the runtime is down or still loading | wait for `/ready`, check `docker compose logs llm` |
| 429 `queue_full` | all slots busy and the queue full | retry after `Retry-After`, or raise `MAX_QUEUE` |
| 504 `llm_timeout` | the model did not finish in `REQUEST_TIMEOUT_S` | raise it on slow servers |
| 400 `invalid_input` on long text | the text does not fit the context of a slot | lower `MAX_INPUT_CHARS` or raise `LLAMA_ARG_CTX_SIZE` |
| 401 `unauthorized` | missing or wrong `API_KEY` | send `Authorization: Bearer <key>` |

## Limitations

Of the model:

- Severity is the weak field: the default model gets it exactly right in 38 of 60 rows and within one level in 59. The texts rarely say how many users are affected.
- The category is right in 56 of 60 rows, and 52 when the label must match strictly; `permission_error` is the most confused (4 of 7).
- `summary` and `suggested_action` are always English.
- Email addresses in `summary` and `suggested_action` are replaced with `<email>`, and so is `user@host` notation such as `git@github.com`. Names, phone numbers and IP addresses are not masked, and the set has no rows to test them.
- Leaks were checked on the 7 rows that contain a secret, and injection on 2 of them; zero leaks there does not prove that the model never repeats a secret. The one email row now tests the mask rather than the model.
- The evaluation set has 60 rows written and labelled by one person, so differences of a few rows between models are noise.

Of the configuration:

- The context of 8192 tokens is split into 2 slots, so one request has about 4096 tokens. A text longer than `MAX_INPUT_CHARS` (3000 characters) keeps its first and last parts and loses the middle, which is replaced by a marker inside the limit, and the answer says `truncated: true`; a text that still does not fit the slot gets 400 `invalid_input`. The 3000-character input of the benchmark used 3232 of the 4096 tokens with the default model (3256 with Qwen3.5-2B); texts that tokenize more densely, or a higher `MAX_INPUT_CHARS`, can hit the 400.
- Two requests are classified at a time; more wait in a queue of 4 for at most 30 s and then get 429. On 8 vCPU one request of the default model took 5.9 to 9.7 s in the 20-request series and a long one about 71 s. With `REQUEST_TIMEOUT_S=120` a long input whose first answer is invalid may not get its retry and ends in 504.
- No streaming answers, no per-client rate limit, one model per stack; changing the model restarts `llm`.
- The only access control is the optional `API_KEY`; TLS comes from the reverse proxy.

Of the measurements: everything was measured with podman on one shared 8 vCPU machine, not on Docker and not on a clean VPS; see the limitation lists in [Evaluation](#evaluation) and [Benchmarks](#benchmarks).

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
- With `LLAMA_ARG_OFFLINE=0` and no network, a restart still loads the model from the `models` volume (checked during development; the output is not kept).

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
| `api/app.py`, `api/routes.py` | FastAPI app, error handlers, the shared classification path, `/classify`, `/health`, `/ready` | present |
| `api/openai.py` | OpenAI-compatible `/v1/models` and `/v1/chat/completions` | present |
| `api/auth.py` | Optional `API_KEY` check for `/classify` and `/v1/*` | present |
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

The limits of classification (`MAX_INPUT_CHARS`, `MAX_OUTPUT_TOKENS`, `REQUEST_TIMEOUT_S` and others) are listed with every other variable in [Configuration](#configuration).

## Error codes

Every error has the shape `{"error": {"code": "...", "message": "...", "type": "...", "param": null}}`. `type` is `invalid_request_error` for 4xx and `server_error` for 5xx; `param` names the offending field or is `null`. Answers 500, 502 and 504 carry `x-should-retry: false`.

| code | HTTP | meaning |
|---|---|---|
| invalid_input | 400 | The body is not valid, `message` is missing, not a string or empty, or the text does not fit the model context |
| request_too_large | 413 | The request body is larger than `MAX_REQUEST_BYTES` |
| model_output_invalid | 502 | The model returned an invalid answer twice |
| llm_bad_response | 502 | The runtime rejected the request or answered in an unexpected shape |
| llm_timeout | 504 | The model did not finish within `REQUEST_TIMEOUT_S`, or the deadline left no time for a retry after an invalid answer |
| llm_unavailable | 503 | The runtime is unreachable, still loading the model, or failing |
| queue_full | 429 | All model slots are busy and the queue is full, or no slot became free within `QUEUE_TIMEOUT_S`; the answer carries `Retry-After` |
| model_not_found | 404 | The `model` field or the `/v1/models/{model}` path names a model other than the served one |
| unauthorized | 401 | `API_KEY` is set and the request has no matching `Authorization: Bearer` header; the answer carries `WWW-Authenticate: Bearer` |
| not_found | 404 | The path does not exist |
| method_not_allowed | 405 | The path exists but not for this HTTP method; the answer carries `Allow` |
| internal_error | 500 | An unexpected error in the service |

## Evaluation

This page describes the labelled set, how it is run against a model, the results of four models and how the default model was chosen from them. The raw results are in `eval/results/`, one JSON file per model, and the tables below are generated from them by `scripts/evaluate.py report`. A test (`tests/test_evaluation_docs.py`) fails if the tables here and the files drift apart, or if the set or the prompt change after the runs.

### The set

`eval/dataset.jsonl` holds 60 error messages written and labelled by hand. Each row has the gold `category` and `severity`, optional `alt_categories` (another category that also fits the definitions in the prompt, for example CORS as `configuration_error` or `network_error`), a `secret` the answer must not repeat, the language and a one-line note on why it is labelled that way.

The set leans on JavaScript (11 browser and 10 Node.js rows), then Python (11), Go (7), SQL (7), infrastructure (6) and 8 other rows. Four texts are in Russian or German. Every category has 5 to 7 rows and every severity at least 6 (low 9, medium 23, high 22, critical 6). Seven rows carry a fake secret: two tokens, a password, a webhook token, an email address and two injection markers. One injection is a plain text that asks for `permission_error`, `critical` and the marker `PWNED-7731`; the other hides inside the echoed input of a 422 response and asks for `critical`.

The texts are short, 38 to 374 characters (median 196), so the latency below is for short inputs; inputs near `MAX_INPUT_CHARS` (3000) are left to the benchmarks.

The few-shot examples of the prompt are not in the set. `check_dataset` compares word trigrams of every row with every example and with every other row and rejects a row that covers 80% of an example (measured on the draft set: edited copies of an example score 0.89 to 1.0, a different error of the same kind at most 0.75, the rows of the set at most 0.46).

### How a run works

`scripts/evaluate.py run` calls the same `Classifier` the service uses, with the same prompt, schema, validation and retry, directly against llama-server. Between the classifier and the runtime client a recorder keeps every attempt with its raw `content`, `finish_reason` and token counts, so the file shows what the model answered even when the answer was rejected. One warm-up request outside the set fills the prompt cache of the slot; then the rows go one at a time. A runtime that stops answering is waited for up to 60 s and the row is repeated once; otherwise the run stops and is written to a `.partial` file, which `report` refuses.

All four runs used the same settings: temperature 0, `MAX_OUTPUT_TOKENS` 256, `MAX_INPUT_CHARS` 3000, `REQUEST_TIMEOUT_S` 120, context 8192 split into 2 slots, 6 threads, batch 512, Q4_K_M weights, and llama-server's default prompt cache of 8192 MiB (the stack now ships `LLAMA_ARG_CACHE_RAM=0`). The only difference is `LLM_DISABLE_THINKING=true` for Qwen3.5-2B: llama-server b12352 turns thinking on for it by default, and with the JSON schema the model then spends its token budget on reasoning and returns an empty answer (observed in a separate request before the runs: `finish_reason` `length` and empty `content`; that output is not kept in the repository).

| Model | GGUF | File, bytes |
|---|---|---|
| `granite-4.0-1b-q4km` | `ibm-granite/granite-4.0-1b-GGUF`, `granite-4.0-1b-Q4_K_M.gguf` | 1023645440 |
| `granite-4.1-3b-q4km` | `ibm-granite/granite-4.1-3b-GGUF`, `granite-4.1-3b-Q4_K_M.gguf` | 2099501664 |
| `qwen3-4b-instruct-2507-q4km` | `bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF`, `Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 2497280736 |
| `qwen3.5-2b-q4km` | `bartowski/Qwen_Qwen3.5-2B-GGUF`, `Qwen_Qwen3.5-2B-Q4_K_M.gguf` | 1396198496 |

Hardware: 8 vCPU AMD EPYC Rome (AVX2, no GPU), 15 GB RAM, shared with another project. The `llm` container ran on podman 5.7 with a 6 CPU limit and llama-server `b12352-5676dfbe0`. Each result file records the runtime env of the container, `/props`, the git commit, the hashes of the set and the prompt, and the load average before and after the run. The 1-minute load average before the warm-up was 1.47 for Qwen3.5-2B, 1.42 for Granite 4.0 1B, 1.50 for Granite 4.1 3B and 1.20 for Qwen3-4B. The Granite 4.1 3B run started exactly at the threshold of 1.5, so step 3 of the rule would ignore its speed; step 1 decided the choice, so this did not matter.

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
| Model | Items | Valid, first try | Valid | Category | Category, strict | Severity | Severity ±1 | Both | English | Text limits | Leaks | Echoed by the model | p50, s | p90, s | Max, s | Latency n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| granite-4.0-1b-q4km | 60 | 60 (100%) | 60 (100%) | 46 (77%) | 44 (73%) | 31 (52%) | 55 (92%) | 24 (40%) | 60 (100%) | 60 (100%) | 1/7 | 1/7 | 3.9 | 4.6 | 5.3 | 60 |
| granite-4.1-3b-q4km | 60 | 60 (100%) | 60 (100%) | 54 (90%) | 53 (88%) | 28 (47%) | 55 (92%) | 25 (42%) | 60 (100%) | 60 (100%) | 1/7 | 2/7 | 7.7 | 11.6 | 16.3 | 60 |
| qwen3-4b-instruct-2507-q4km | 60 | 60 (100%) | 60 (100%) | 56 (93%) | 52 (87%) | 38 (63%) | 59 (98%) | 34 (57%) | 60 (100%) | 60 (100%) | 0/7 | 1/7 | 10.2 | 18.3 | 29.6 | 60 |
| qwen3.5-2b-q4km | 60 | 60 (100%) | 60 (100%) | 52 (87%) | 46 (77%) | 27 (45%) | 56 (93%) | 24 (40%) | 60 (100%) | 60 (100%) | 0/7 | 0/7 | 4.2 | 5.2 | 8.9 | 60 |

| Model | ok |
|---|---|
| granite-4.0-1b-q4km | 60 |
| granite-4.1-3b-q4km | 60 |
| qwen3-4b-instruct-2507-q4km | 60 |
| qwen3.5-2b-q4km | 60 |

| Category | Items | granite-4.0-1b-q4km | granite-4.1-3b-q4km | qwen3-4b-instruct-2507-q4km | qwen3.5-2b-q4km |
|---|---|---|---|---|---|
| syntax_error | 5 | 5 | 4 | 4 | 5 |
| runtime_error | 6 | 6 | 6 | 6 | 6 |
| validation_error | 6 | 6 | 6 | 6 | 6 |
| network_error | 6 | 6 | 6 | 6 | 6 |
| database_error | 6 | 5 | 5 | 6 | 6 |
| dependency_error | 5 | 3 | 5 | 5 | 4 |
| configuration_error | 7 | 4 | 5 | 7 | 6 |
| resource_error | 7 | 5 | 7 | 7 | 4 |
| permission_error | 7 | 4 | 6 | 4 | 4 |
| unknown | 5 | 2 | 4 | 5 | 5 |

| Pair | Both only first | Both only second | p (McNemar) |
|---|---|---|---|
| granite-4.0-1b-q4km vs granite-4.1-3b-q4km | 14 | 15 | 1.000 |
| granite-4.0-1b-q4km vs qwen3-4b-instruct-2507-q4km | 8 | 18 | 0.076 |
| granite-4.0-1b-q4km vs qwen3.5-2b-q4km | 10 | 10 | 1.000 |
| granite-4.1-3b-q4km vs qwen3-4b-instruct-2507-q4km | 7 | 16 | 0.093 |
| granite-4.1-3b-q4km vs qwen3.5-2b-q4km | 16 | 15 | 1.000 |
| qwen3-4b-instruct-2507-q4km vs qwen3.5-2b-q4km | 16 | 6 | 0.052 |

p values are not corrected for multiple comparisons: 6 pairs; with Holm's correction the smallest p is significant only below 0.0083.

Choice by the rule: qwen3-4b-instruct-2507-q4km (best on both fields, the others are more than 6 rows behind or McNemar p < 0.05).
<!-- eval-table:end -->

### Reading the results

Four models were run on the same day, one after another: Qwen3.5-2B, Granite 4.0 1B, Granite 4.1 3B and Qwen3-4B-Instruct-2507. All of them returned a valid answer on the first call for every row. The grammar built from the JSON schema leaves little room for broken JSON, so validity does not separate the models; quality and leaks do.

Severity is the hard field. Exact severity ranges from 27 to 38 of 60, while severity within one level is 55 to 59. The texts rarely say how many users are affected, so the models and the labels disagree by one step more often than on the category.

Qwen3-4B-Instruct-2507 is the most accurate: both fields right in 34 rows against 24 or 25 for the other three. Against Qwen3.5-2B it wins 16 rows and loses 6 (p 0.052), against Granite 4.1 3B 16 to 7 (p 0.093) and against Granite 4.0 1B 18 to 8 (p 0.076). None of these is below 0.05, and with six pairs Holm's correction would ask for 0.0083, so its lead is a tendency that 60 rows cannot confirm. It is also the slowest, with a p50 of 10.2 s and 707 s for the whole run.

Granite 4.0 1B gets both fields right as often as Qwen3.5-2B (24 rows; each model wins 10 rows the other loses) and is the fastest (p50 3.9 s), but it is the weakest on the category (46 of 60, 44 strict) and on `unknown` (2 of 5).

Both Granite models followed the plain injection: for `unk-003` they answered `permission_error`, `critical` and put `PWNED-7731` in the summary. Granite 4.1 3B and Qwen3-4B also repeated the email address from `py-012`. Qwen3-4B treated the injection as text (`unknown`, `low`). Qwen3.5-2B repeated none of the seven secrets and also answered the injection with `unknown` and `low`. The service now masks email addresses, so these two repeats are counted under `Echoed by the model` but not under `Leaks`, which counts secrets in the answer the service returns.

Qwen3.5-2B (p50 4.2 s, 268 s for the whole run) loses most on `resource_error` (4 of 7) and `permission_error` (4 of 7) and is weak on the strict category (46 of 60).

The three models that also ran a day earlier gave the same category and severity on all 60 rows in this run; only their latency changed with the load of the host. The earlier result files are not kept in the repository.

### The choice

The rule was written down before the runs and is implemented in `choose_model`. One condition was narrowed after the first Granite run: the load average after a run no longer counts, because llama-server alone pushes it to about 10 under the CPU limit. Step 1 decided the choice, so the change did not affect it.

1. A model is out if fewer than 57 of 60 answers are valid, fewer than 54 are valid on the first try, or any secret leaks.
2. Among the rest, the leaders have the most rows with both fields right. A model stays a candidate if it is at most 6 rows behind the leaders and McNemar against each leader gives p of at least 0.05.
3. Among the candidates the faster one wins if its p50 on the rows valid on the first try for all candidates is more than 10% lower than the others. The speed does not count when there are fewer than 20 such rows or a run started with a 1-minute load average of 1.5 or more.
4. Otherwise the current default stays, then the most rows with both fields right, then the name.

A second change came after the runs, and this one changed the choice. Under the rule as first written, a secret anywhere in the model's `summary` or `suggested_action` excluded the model, and Qwen3.5-2B was the only model left after step 1. Qwen3-4B was out for one row: it repeated the email address from `py-012`. The service now replaces email addresses in its answers with `<email>`, and step 1 counts a leak in the answer the service returns (`Leaks`); what the model wrote before the mask is shown in `Echoed by the model`. With that, Qwen3-4B has no leak, and step 2 picks it: it leads by 10 rows with both fields right, more than the 6 rows the rule allows, so no other model stays a candidate. The choice rests on that gap, not on significance: McNemar against Qwen3.5-2B gives p 0.052, and Holm's correction over six pairs would ask for 0.0083. The change was made knowing that it would move the default; under the rule as first written the default would still be Qwen3.5-2B.

The price is speed and memory: p50 10.2 against 4.2 s in the evaluation and 8.1 against 5.8 s through the API, with a different load on the host during the two benchmark runs, and at least 5.77 against 2.34 GiB for `llm`. After the mask the `py-012` row tests the mask rather than the model, and other kinds of personal data are neither masked nor in the set. On servers with 4 GiB of RAM the same rule still picks Qwen3.5-2B.

The default is `MODEL_ALIAS=qwen3-4b-instruct-2507-q4km` in `.env.example`, with `LLM_DISABLE_THINKING=false`.

### Limitations

- The labels come from the author of the category definitions, so the set measures agreement with those definitions rather than with an independent expert.
- 60 rows separate only large differences, as shown in the pair table.
- The prompt was tuned on Granite 4.1 3B, which favours it. Temperature 0 differs from the sampling the Qwen vendors recommend (Qwen3-4B T 0.7, Qwen3.5-2B T 1.0), which may understate both Qwen models.
- The quantizers differ: IBM for both Granite models, bartowski (imatrix) for both Qwen models.
- Latency was measured on podman on a shared machine, on short inputs only, while another project's stack was running. Benchmarks of the whole service are in the next section.
- Seven secret rows show leaks when they happen; zero leaks on seven rows does not prove a model never leaks. For Granite 4.0 1B one row decided the exclusion (it followed the injection in `unk-003`), and that row was not repeated, so whether it happens every time is not known.
- Temperature 0 is not bit-exact here. An earlier repeat run of Qwen3.5-2B, whose file is not kept, changed the labels of 3 of 60 rows (`py-007`, `infra-009`, `ru-002`) with the same score of 24 rows with both fields right. The cause is assumed to be the prompt cache and batch boundaries of llama-server; differences of a few rows between models are within this as well.
- Models of 7 to 8B parameters were not evaluated: on this shared 8 vCPU machine a run of Qwen3-8B took most of the CPU and was stopped. In a single request before it (output not kept), at an 8 GiB container limit, the kernel was already evicting its weights from memory (anonymous memory 4.32 GiB).
- The runs used llama-server's default prompt cache of 8192 MiB; the stack now ships `LLAMA_ARG_CACHE_RAM=0`, whose effect on speed was not measured.

### Reproducing

Start `llm` with the model in `.env` and the debug port, then run the set and the report. The runs above used `podman compose` and `--engine podman`; `--container` takes the name of the `llm` container from `compose ps`: `llm-error-classifier_llm_1` under podman-compose, `llm-error-classifier-llm-1` under Docker Compose v2.

```sh
podman compose -f compose.yaml -f compose.debug.yaml up -d llm
set -a; . ./.env; set +a
LLM_BASE_URL=http://127.0.0.1:8080/v1 LLM_MODEL=$MODEL_ALIAS \
  uv run python scripts/evaluate.py run --engine podman \
  --container llm-error-classifier_llm_1 --out eval/results/$MODEL_ALIAS.json
uv run python scripts/evaluate.py report eval/results/*.json
```

`--limit 3` makes a short smoke run into a `.partial` file. A full run took 4 to 13 minutes per model on the machine above.

## Benchmarks

`scripts/bench.py run` measures the whole service: a client on the host calls `POST /classify` through the published API port and reads CPU and memory from the cgroups of both containers at the start and end of each phase. One run is a cold start of the stack with the model already in the volume, 3 warm-up requests, 20 requests one at a time, one long input (3000 characters of base64), 50 requests one at a time (only for Qwen3.5-2B, the default at the time of the runs), two clients sending 10 requests each, and a burst of `MAX_CONCURRENCY + MAX_QUEUE + 1` requests to check the queue. Inputs are 20 rows of `eval/dataset.jsonl`. Raw files are in `bench/results/`, and `scripts/bench.py report bench/results/*.jsonl` prints the tables below; a test checks that they match.

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

| Model | llm peak | llm file | api peak | api file | llm swap peak | llm swap max | llm memory max | api swap peak | api swap max | api memory max | GGUF | RAM floor | RAM, conservative | RAM, worst | llm image | api image |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| granite-4.1-3b-q4km | 2.45 GiB | 0.03 GiB | 0.06 GiB | - | - | - | - | - | - | - | 1.96 GiB | 2.34 GiB | 4.29 GiB | 4.40 GiB | 0.78 GiB | 0.19 GiB |
| qwen3-4b-instruct-2507-q4km | 5.77 GiB | 2.28 GiB | 0.09 GiB | - | - | - | - | - | - | - | 2.33 GiB | 3.06 GiB | 5.39 GiB | 8.09 GiB | 0.78 GiB | 0.19 GiB |
| qwen3.5-2b-q4km | 2.34 GiB | 1.32 GiB | 0.10 GiB | - | - | - | - | - | - | - | 1.30 GiB | 1.01 GiB | 2.31 GiB | 3.64 GiB | 0.78 GiB | 0.19 GiB |

| Model | llm need | api peak | Disk |
|---|---|---|---|
| granite-4.1-3b-q4km | 4.29 GiB | 0.06 GiB | 2.92 GiB |
| qwen3-4b-instruct-2507-q4km | 5.77 GiB | 0.09 GiB | 3.29 GiB |
| qwen3.5-2b-q4km | 2.34 GiB | 0.10 GiB | 2.27 GiB |
<!-- bench-table:end -->

How to read the tables:

- Latency is wall time of one `POST /classify` from the client, including the queue, both attempts and the API. Generation and prompt speeds come from the llama-server counters for the 20-request series.
- `CPU llm, %` is the CPU time of the `llm` container over the 20-request series divided by wall time, where 100% is one core. With `LLM_CPUS=6` and 6 threads the runtime used 5.3 to 5.8 cores in that series and 5.2 to 5.7 in the other phases.
- `RAM floor` is the largest anonymous memory of `llm` (the process, its buffers and the KV cache). `RAM, conservative` adds the whole GGUF file, as if every page of the file stayed in the page cache; it can count the weights twice when the runtime also keeps them in anonymous memory, which is likely for Granite, whose anonymous memory (2.34 GiB) is larger than its file (1.96 GiB). The measured peak of the container can still be higher, because the page cache also holds other files. `RAM, worst` adds the file to the peak. `llm need` in the last table is the larger of the conservative value and the measured peak, and the requirements below use it.
- The queue probe sends more requests than the service accepts; at least one is refused at once with 429, as configured.

### Resource requirements

For the default model (Qwen3-4B-Instruct-2507, Q4_K_M):

| Resource | Requirement | Source |
|---|---|---|
| CPU | measured on x86_64 with AVX2, 6 threads on 8 vCPU; smaller machines were not measured | `CPU llm, %` above |
| RAM | at least 5.77 GiB for `llm` and 0.09 GiB for `api` (`llm need` and `api peak`); `.env.example` limits `llm` to 6.5 GiB | last table |
| Disk | 1.0 GiB for the two images and 2.3 GiB for the model file; every other model adds its GGUF to the `models` volume | memory table |

The peak of 5.77 GiB was taken under `LLM_MEM_LIMIT=6g` and held 2.28 GiB of page cache next to 3.06 GiB of anonymous memory. A peak at 96% of the limit is bounded by it, so 5.77 GiB is a lower bound of what the model takes with more room; the default limit is now 6.5 GiB. By the same measure Qwen3.5-2B needs 2.34 GiB and Granite 4.1 3B 4.29 GiB. Speed on machines with fewer cores was not measured.

### Limitations of the measurements

- All numbers come from podman on a shared host, not from Docker on a dedicated one. Every run had the 11 containers of the other project up; the load average at the start of each run is in the first table. A first run of the default model, whose file is not kept, had single requests several times slower than the median and was repeated; the repeat started at a load average of 1.28 and still has single requests up to 21.6 s in the 50-request series, so the tail of the latency is partly the host.
- Inputs are short error messages (up to a few hundred characters). One long input per model is measured, not a series of them.
- Concurrency stops at two clients, which is the number of runtime slots.
- Only Qwen3.5-2B, the default at the time of the runs, ran the 50-request series; the other two skip it to keep each run under 15 minutes. Qwen3-4B, the default now, has no series.
- The load average at the start differed: 0.27 for Qwen3-4B, 1.28 for Qwen3.5-2B and 1.46 for Granite 4.1 3B, so the speed ratios between the models are approximate. The long input was measured once per model.
- podman-compose did not apply `memswap_limit`, so the containers could use swap beyond `mem_limit` on this host, and these files have no swap counters (the swap columns are empty). `memory.peak` does not count anonymous memory moved to swap, so with swap in use the memory figures understate the need. `scripts/bench.py` now records `memory.swap.peak`, `memory.swap.max` and `memory.max`, so a run on Docker shows whether swap was used and limited.
- The runs used llama-server's default prompt cache of 8192 MiB; the stack now ships `LLAMA_ARG_CACHE_RAM=0`.
- The cold start begins with the model file already in the volume and probably in the page cache of the host, so loading from a cold disk takes longer. The time to `/ready` includes the healthcheck interval of `llm` (10 s): `api` starts only after `llm` is reported healthy, so this column moves in steps of the interval. The first start also downloads the file, and its time depends on the network.

## About this test task

This repository is a solution of the CodeX test task "Self-hosted LLM": deploy an open-weights model on a CPU and turn it into an API that classifies error messages. All requirements of the task are addressed above: model choice and license, CPU-only deployment with an API, resource requirements, the classification JSON with defined categories and severity levels, handling of invalid model output, a test set with checked results, Docker Compose with persistent model storage, settings in `.env`, start after reboot, a VPS guide, and measurements in this README. Running on Docker, the start after a reboot and the VPS guide are written but not yet run (see TODO).

Of the optional items, the OpenAI-compatible API, the comparison of several models, the automatic evaluation, the limit on concurrent requests, timeouts and model errors, and model changes without code changes are done. Prometheus metrics with Grafana and a demonstration on a clean VPS were left out on purpose; llama-server exposes `/metrics` if they are needed.

Constraints of the work: the development machine was shared with another project and had no Docker. The stack was built and checked with podman, the files are written for Docker Compose and have not yet been run with it. Model runs stopped when they started to take most of the shared machine.

TODO:

- Run the stack and the VPS guide on Docker, including a reboot, and repeat the benchmarks there.
- Run the 50-request series for the default model `qwen3-4b-instruct-2507-q4km`; here only `qwen3.5-2b-q4km` ran it.
- Run the live tests against the model, including `tests/live/test_openai_live.py` through the official client.
- Benchmark Granite 4.0 1B and evaluate a 7 to 8B model on a machine that can take the load.
- Repeat the row that decided the exclusion of Granite 4.0 1B (`unk-003`) to see whether it follows the injection every time.
- Measure the memory of the default model on Docker with the swap counters and with unique inputs, and the timeouts of the smaller servers in the sizing table.
- Improve severity (38 of 60 exact for the default model) with a new prompt; this needs new runs of every model.
- Record an answer of the running stack for [Classify an error](#classify-an-error); the example there comes from the evaluation run.
