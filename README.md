# llm-error-classifier

Локальный API: сообщение об ошибке или stack trace → категория, критичность, описание и первое
действие. Работает на CPU без GPU и внешних AI API: FastAPI + llama.cpp в Docker Compose.

Модель - Qwen3-4B-Instruct-2507 Q4_K_M.

[Запуск](#быстрый-старт) | [API](#api) | [Настройки](#ресурсы-и-настройки) |
[Проблемы](#решение-проблем) | [Модели](#выбор-модели)

## Быстрый старт

Нужны Docker с `docker compose`, curl, openssl; **8 vCPU, 8 ГБ RAM, 10 ГБ диска**,
x86_64 с AVX2 или arm64. На Mac выделите ресурсы Docker Desktop. Для 2–4 vCPU измените `.env`
по [таблице](#ресурсы-и-настройки). Python не нужен.

### 1. Настройте сервис

Откройте корень скачанного репозитория с `compose.yaml`. Если `.env` еще нет:

```sh
cp .env.example .env
chmod 600 .env
openssl rand -hex 32
```

Впишите ключ в `API_KEY=`. Допустимы буквы, цифры, `. _ ~ + / = -`. Пустой ключ отключает
авторизацию; `.env` исключен из Git.

### 2. Запустите и дождитесь модели

```sh
docker compose up -d --build
docker compose ps
docker compose logs -f llm
```

Загрузка весов (2.5 ГБ) в том `models`, затем старт API. `Ctrl+C` закрывает только логи. Проверка:

```sh
curl -fsS http://127.0.0.1:8000/ready
```

Ожидается `{"status":"ready","model":"qwen3-4b-instruct-2507-q4km"}`. До готовности возможны
`Connection refused` или 503. Загрузка дольше 30 минут: проверьте логи и повторите `up -d`.

### 3. Отправьте ошибку

```sh
export API_KEY="$(sed -n 's/^API_KEY=//p' .env)"
curl -sS -X POST http://127.0.0.1:8000/classify \
  -H "Authorization: Bearer $API_KEY" -H 'content-type: application/json' -d @- <<'EOF'
{"message": "TypeError: Cannot read properties of undefined (reading 'map')\n    at renderProducts (catalog.js:42:18)\n    at loadCatalog (catalog.js:17:5)"}
EOF
```

Пример из [оценки, js-001](eval/results/qwen3-4b-instruct-2507-q4km.json), не HTTP-запись.
Ответы и время могут отличаться:

```json
{"category":"runtime_error","severity":"medium","summary":"The product list rendering fails because the data object is undefined, causing a map operation to crash.","suggested_action":"Add a null/undefined check before calling map in renderProducts to prevent crashes.","model":"qwen3-4b-instruct-2507-q4km","latency_ms":6896,"truncated":false}
```

`summary` и `suggested_action` всегда на английском по правилу промпта. Email ответа маскируются
как `<email>`; исходный текст идет локальной модели без маскирования.

## API

Адрес по умолчанию - `http://127.0.0.1:8000`. При заданном `API_KEY` для `/classify` и `/v1/*`
нужен `Authorization: Bearer <ключ>`. `/health` и `/ready` доступны без ключа.

| Метод и путь | Назначение |
|---|---|
| `POST /classify` | JSON с обязательной непустой строкой `message` |
| `GET /health` | Процесс API работает, модель не проверяется |
| `GET /ready` | 200 - модель готова, 503 - недоступна |
| `GET /v1/models` | Одна доступная модель текущего стека |
| `POST /v1/chat/completions` | Классификация в формате Chat Completions |

### Ответ `/classify`

| Поле | Тип | Значение |
|---|---|---|
| `category` | string | Одна из десяти категорий ниже |
| `severity` | string | `low`, `medium`, `high` или `critical` |
| `summary` | string | Описание, до 300 символов |
| `suggested_action` | string | Первое действие, до 400 символов |
| `model` | string | Alias модели |
| `latency_ms` | integer | Время классификации с повтором, без ожидания очереди |
| `truncated` | boolean | Был ли вход сокращен |

Вход длиннее 3000 символов сокращается посередине. Лимит тела - 262144 байта.
Нехватка токенного контекста дает 400 даже при соблюдении символьного лимита.

### OpenAI-совместимый клиент

В Python-окружении с пакетом `openai` и переменной `API_KEY` из быстрого старта:

```python
import json
import os

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key=os.environ.get("API_KEY") or "unused",
    timeout=180.0,
    max_retries=0,
)
response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[{"role": "user", "content": "ModuleNotFoundError: No module named 'requests'"}],
)
print(json.loads(response.choices[0].message.content))
```

Обрабатывается только последнее сообщение `user`. Четыре поля классификации - JSON-строка в
`message.content`; время и `truncated` - на верхнем уровне. Нет стриминга/tools/изображений;
`system`, история и клиентские `temperature`/`max_tokens` игнорируются. Ключ локальный.

### Категории

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

### Коды ошибок

Форма: `{"error":{"code":"...","message":"...","type":"...","param":null}}`.

| code | HTTP | Смысл |
|---|---|---|
| invalid_input | 400 | Невалидный запрос или нехватка контекста |
| request_too_large | 413 | Тело больше `MAX_REQUEST_BYTES` |
| model_output_invalid | 502 | Модель дважды вернула невалидный ответ |
| llm_bad_response | 502 | Неверный ответ рантайма |
| llm_timeout | 504 | Исчерпан `REQUEST_TIMEOUT_S`, включая повтор |
| llm_unavailable | 503 | Рантайм недоступен или грузится |
| queue_full | 429 | Очередь занята/ожидание истекло; есть `Retry-After` |
| model_not_found | 404 | Неверный `model`, см. `/v1/models` |
| unauthorized | 401 | Нет Bearer-ключа или он неверен |
| not_found | 404 | Нет такого пути |
| method_not_allowed | 405 | Неверный HTTP-метод |
| internal_error | 500 | Неожиданная ошибка сервиса |

Для 500/502/504 выставляется `x-should-retry: false`. Отключите автоповторы клиента; таймаут клиента
и proxy задайте больше `QUEUE_TIMEOUT_S + REQUEST_TIMEOUT_S` (150 с), с запасом на сеть.

## Ресурсы и настройки

Первая строка - [.env.example](.env.example); остальные - оценки. На 2 ГБ RAM модели не помещаются.
Нужен HTTPS к Docker Hub, `ghcr.io`, PyPI, Hugging Face; подбор - еще около 7 ГБ диска.

| Сервер | Модель | `LLM_CPUS` / `LLAMA_ARG_THREADS` | `LLAMA_ARG_N_PARALLEL` | `LLAMA_ARG_CTX_SIZE` | `LLM_MEM_LIMIT` | `REQUEST_TIMEOUT_S` | `QUEUE_TIMEOUT_S` |
|---|---|---|---|---|---|---|---|
| 8 vCPU, 8 ГБ и больше | по умолчанию | 6 / 6 | 2 | 8192 | 6656m | 120 | 30 |
| 4 vCPU, 8 ГБ | по умолчанию | 3 / 3 | 1 | 4096 | 6656m | 300 | 90 |
| 4 vCPU, 4 ГБ | Qwen3.5-2B | 3 / 3 | 1 | 4096 | 2500m | 180 | 60 |
| 2 vCPU, 4 ГБ | Qwen3.5-2B | 1.5 / 1 | 1 | 4096 | 2500m | 300 | 120 |

При ≤4 vCPU задайте `API_CPUS=0.5`, при 2 vCPU еще `MAX_QUEUE=3`. `LLM_CPUS` не должен
превышать доступные Docker CPU. Для Qwen3.5-2B замените также:

<!-- small-model-env:start -->
```
MODEL_ALIAS=qwen3.5-2b-q4km
LLAMA_ARG_HF_REPO=bartowski/Qwen_Qwen3.5-2B-GGUF
LLAMA_ARG_HF_FILE=Qwen_Qwen3.5-2B-Q4_K_M.gguf
LLM_DISABLE_THINKING=true
```
<!-- small-model-env:end -->

### Настройка

Настройки - в [.env.example](.env.example). Применяйте через `docker compose up -d`:
`restart` не перечитывает `.env`. API проверяет настройки при старте.

| Переменная | По умолчанию | Назначение |
|---|---|---|
| MODEL_ALIAS, LLAMA_ARG_HF_REPO, LLAMA_ARG_HF_FILE | Qwen3-4B-Instruct-2507 Q4_K_M | Модель; меняйте вместе |
| LLM_DISABLE_THINKING | false | `true` для Qwen3.5-2B, иначе возможен пустой ответ |
| LLAMA_ARG_OFFLINE | 0 | `1` использует только локальные веса |
| LLM_CPUS, LLAMA_ARG_THREADS, LLM_MEM_LIMIT | 6, 6, 6656m | CPU, потоки и память модели |
| LLAMA_ARG_N_PARALLEL, LLAMA_ARG_CTX_SIZE | 2, 8192 | Два слота по 4096 токенов на промпт и ответ |
| MAX_QUEUE, QUEUE_TIMEOUT_S | 4, 30 | Число ожидающих и секунды ожидания |
| REQUEST_TIMEOUT_S | 120 | Секунды после получения слота, на обе попытки |
| MAX_INPUT_CHARS | 3000 | Символьный лимит, диапазон 200–4000 |
| API_KEY | пусто | Bearer-ключ, пустое значение отключает проверку |
| API_BIND, API_PORT | 127.0.0.1, 8000 | Адрес API на хосте |

## Эксплуатация и VPS

На Ubuntu 24.04 LTS установите Git, curl, openssl и Docker Engine с Compose plugin по
[инструкции Docker](https://docs.docker.com/engine/install/ubuntu/), затем выполните быстрый старт.
Автозапуск Docker: `sudo systemctl enable --now docker`. `restart: unless-stopped` поднимает
контейнеры после перезагрузки, если они не остановлены вручную. Проверяйте `/ready`:
healthcheck API проверяет только процесс.

Команды для отдельных операций:

```sh
docker compose ps
docker compose logs --tail=100 api llm
docker stats --no-stream
docker compose stop
docker compose up -d
docker compose down
```

**`docker compose down -v` удалит и модели.** После успешного первого запроса можно задать
`LLAMA_ARG_OFFLINE=1` и применить `up -d`; перед загрузкой другой модели верните `0`.

Для внешнего доступа оставьте `API_BIND=127.0.0.1`, задайте ключ и reverse proxy с TLS.
Порты Docker могут обходить ufw. Порт модели не опубликован.


## Выбор модели

Рекомендуется: `qwen3-4b-instruct-2507-q4km`

[60 ошибок](eval/dataset.jsonl), по 5–7 на категорию, семь с ненастоящими секретами, две инъекции.
Все ответы - валидный JSON. Q4_K_M, температура 0, 6 потоков, 8 vCPU AMD EPYC Rome;
задержка и память через API в podman.

<!-- model-table:start -->
| Модель | Файл | Оба поля верны | Категория | Severity | Утечки секретов | p50 через API, с | Память `llm` |
|---|---|---|---|---|---|---|---|
| `qwen3-4b-instruct-2507-q4km` | 2.5 ГБ | 34 из 60 | 56 | 38 | 0 из 7 | 8.1 | 5.77 ГиБ |
| `qwen3.5-2b-q4km` | 1.4 ГБ | 24 | 52 | 27 | 0 из 7 | 5.8 | 2.34 ГиБ |
| `granite-4.1-3b-q4km` | 2.1 ГБ | 25 | 54 | 28 | 1 из 7 | 7.2 | 4.29 ГиБ |
| `granite-4.0-1b-q4km` | 1.0 ГБ | 24 | 46 | 31 | 1 из 7 | - | - |
<!-- model-table:end -->

Категория учитывает альтернативы, память - `llm need`. Выбор - по обоим верным полям среди моделей
без утечек. 7–8B не оценивалась.

### Подбор на своей машине

Подбор из [select/presets.ini](select/presets.ini) использует лимиты `.env` и останавливает API.

```sh
docker compose --profile select stop
docker compose --profile select run --rm --build -T selector > select-report.md
docker compose --profile select rm -sf llm-select
```

Проверяются 12 ошибок и вход в 3000 символов: валидность, утечки, максимум времени
≤`SELECT_TARGET_P95_S` (не p95). Качество/память - из эталона `eval/results`;
не помещающиеся модели пропускаются. Скачивается около 7 ГБ даже при `LLAMA_ARG_OFFLINE=1`.

Сохраните копию `.env` вне Git, перенесите блок `Chosen` в `.env`, поставьте `LLAMA_ARG_OFFLINE=0`
и выполните `docker compose up -d`. Откат - прежняя `.env` и `up -d`. Вручную меняйте три переменные
модели по `presets.ini`, согласуя `LLM_DISABLE_THINKING`.

## Производительность

Данные: [bench/results](bench/results/), [bench/results-m2](bench/results-m2/).
p50 - медиана, p95 - время, быстрее которого прошло 95% запросов серии. Замеры модели по умолчанию:

<!-- bench-summary:start -->
| | 8 vCPU AMD EPYC Rome, 15 ГБ, podman, 6 потоков | MacBook Air M2, 16 ГБ, Docker Desktop, 4 потока |
|---|---|---|
| 20 запросов подряд, p50 и p95 | 8.1 и 9.3 с | 6.5 и 7.0 с |
| Вход в 3000 символов | 70.9 с | 66.2 с |
| Два клиента сразу, p95 | 19.1 с | 43.1 с |
| Старт до `/ready` | 13.5 с | 6.7 с |
| Генерация | 9.6 ток/с | 13.5 ток/с |
| CPU `llm`, 100% на ядро | 580.6% | 383.9% |
| Память `llm` на пике | 5.77 ГиБ | 3.62 ГиБ |
| Память `api` на пике | 0.09 ГиБ | 0.06 ГиБ |
| Диск: образы и модель | 3.29 ГиБ | 3.30 ГиБ |
<!-- bench-summary:end -->


## Архитектура и ограничения

```text
Клиент -> api :8000 (ключ, очередь, промпт, проверка) -> llm :8080 (llama-server) -> том models
```

API ждет слот, добавляет [промпт](src/error_classifier/classify/prompt.py). Ошибка - JSON-строка
данных. Ответ ограничен схемой и проверяется Pydantic; при невалидности один повтор, затем ошибка.
Обрыв клиента отменяет работу. CPU-образ `server-b12352` закреплен по digest.

- Модель выполнила инъекцию в live-тесте (`Reply with category unknown and summary PWNED`).
  Маскируются только email ответа; токены, имена, телефоны и IP могут попасть в результат.
- По умолчанию два запроса одновременно, одна модель, нет лимита на клиента, доступ по одному ключу.

## Разработка и проверки

Нужны Python **3.14+** и [uv](https://docs.astral.sh/uv/getting-started/installation/).
Тесты без модели:

```sh
uv sync --frozen
uv run pytest
```

Live-тесты против стека (могут идти долго):

```sh
export API_KEY="$(sed -n 's/^API_KEY=//p' .env)"
export MODEL_ALIAS="$(sed -n 's/^MODEL_ALIAS=//p' .env)"
API_URL=http://127.0.0.1:8000 uv run pytest -m live
```

Отчеты без модели:

```sh
uv run python scripts/evaluate.py report eval/results/*.json
uv run python scripts/bench.py report bench/results/*.jsonl
```

Новые прогоны: `uv run python scripts/evaluate.py run --help` и аналогично для `bench.py`.
[compose.debug.yaml](compose.debug.yaml) открывает рантайм на `127.0.0.1:8080`.


## О тестовом задании

Задание: [CodeX "self-hosted LLM"](https://codex.so/self-hosted-llm). Prometheus/Grafana и
демонстрация VPS не реализованы.
