import json
import os
import time
from collections.abc import Callable

import openai
import pytest
from openai.types.chat import ChatCompletion

from error_classifier.classify.schema import Category, ClassificationResult

# Read at import, before the autouse clean_env fixture runs.
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "qwen3.5-2b-q4km")
API_KEY = os.environ.get("API_KEY") or "unused"

pytestmark = pytest.mark.live

TASK_EXAMPLE = "TypeError: Cannot read properties of undefined (reading 'map')"


def _client() -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{API_URL}/v1", api_key=API_KEY, timeout=300, max_retries=0)


def _report(case: str, started: float, **fields: object) -> None:
    # One line per case for the run log; read it with pytest -s.
    seconds = round(time.perf_counter() - started, 1)
    print(json.dumps({"case": case, "seconds": seconds, **fields}))


def _call[T](case: str, action: Callable[[], T]) -> T:
    started = time.perf_counter()
    try:
        result = action()
    except openai.APIError as exc:
        _report(case, started, error=f"{type(exc).__name__}: {exc.message}")
        raise
    return result


def test_task_example_through_the_sdk() -> None:
    started = time.perf_counter()
    completion: ChatCompletion = _call(
        "task-example",
        lambda: _client().chat.completions.create(
            model=MODEL_ALIAS, messages=[{"role": "user", "content": TASK_EXAMPLE}]
        ),
    )
    content = completion.choices[0].message.content
    _report("task-example", started, content=content)
    assert content is not None
    answer = ClassificationResult.model_validate_json(content)
    assert answer.category in set(Category)
    assert set(json.loads(content)) == {"category", "severity", "summary", "suggested_action"}
    assert completion.model == MODEL_ALIAS
    assert completion.usage is not None
    assert completion.usage.total_tokens == (
        completion.usage.prompt_tokens + completion.usage.completion_tokens
    )


def test_foreign_model_is_not_found() -> None:
    with pytest.raises(openai.NotFoundError) as caught:
        _call(
            "foreign-model",
            lambda: _client().chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": TASK_EXAMPLE}]
            ),
        )
    assert caught.value.code == "model_not_found"
    assert caught.value.param == "model"


def test_stream_is_rejected() -> None:
    with pytest.raises(openai.BadRequestError) as caught:
        _call(
            "stream",
            lambda: _client().chat.completions.create(
                model=MODEL_ALIAS,
                messages=[{"role": "user", "content": TASK_EXAMPLE}],
                stream=True,
            ),
        )
    assert caught.value.param == "stream"
