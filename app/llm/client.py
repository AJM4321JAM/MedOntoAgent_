import json
import time
from collections.abc import Callable
from typing import TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from app.config.settings import Settings, get_settings
from app.llm.prompts import (
    LLM_TEST_SYSTEM_PROMPT,
    LLM_TEST_USER_PROMPT,
)
from app.llm.structured_output import LLMTestResponse


class LLMConfigurationError(RuntimeError):
    """
    Raised when required LLM configuration is missing.
    """


class LLMServiceError(RuntimeError):
    """
    Raised when the external LLM service cannot safely
    complete a request.
    """


T = TypeVar("T")


def create_openrouter_client(
    settings: Settings | None = None,
) -> OpenAI:
    """
    Create an OpenAI-compatible client pointed at OpenRouter.

    SDK retries are deliberately disabled. MedOntoAgent owns
    its retry policy explicitly.
    """
    settings = settings or get_settings()

    if settings.openrouter_api_key is None:
        raise LLMConfigurationError(
            "OPENROUTER_API_KEY is not configured."
        )

    return OpenAI(
        api_key=(
            settings
            .openrouter_api_key
            .get_secret_value()
        ),
        base_url=settings.openrouter_base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=0,
    )


def _extract_json_object(
    content: str,
) -> dict:
    """
    Extract a JSON object from a model response.

    Free models may wrap JSON in Markdown code fences even
    when instructed to return JSON only.
    """
    cleaned = content.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip() == "```"
        ):
            lines = lines[:-1]

        cleaned = "\n".join(
            lines
        ).strip()

        if cleaned.lower().startswith(
            "json"
        ):
            cleaned = cleaned[4:].strip()

    try:
        data = json.loads(
            cleaned
        )

    except json.JSONDecodeError as exc:
        raise LLMServiceError(
            (
                "The model returned an invalid JSON response. "
                f"Raw response: {content!r}"
            )
        ) from exc

    if not isinstance(
        data,
        dict,
    ):
        raise LLMServiceError(
            (
                "The model response was valid JSON but "
                "was not a JSON object."
            )
        )

    return data


def _call_with_one_retry(
    operation: Callable[[], T],
    *,
    operation_name: str,
) -> T:
    """
    Execute an external LLM operation with at most one retry.

    Retry only transient provider conditions:
    - HTTP 429 rate limiting
    - timeout
    - connection failure

    The OpenAI SDK itself performs zero retries.
    """
    total_attempts = 2

    for attempt in range(
        1,
        total_attempts + 1,
    ):
        try:
            return operation()

        except RateLimitError as exc:
            if attempt == total_attempts:
                raise LLMServiceError(
                    (
                        f"{operation_name} failed because the "
                        "free model/provider remained rate-limited "
                        "after one retry."
                    )
                ) from exc

            time.sleep(2)

        except APITimeoutError as exc:
            if attempt == total_attempts:
                raise LLMServiceError(
                    (
                        f"{operation_name} timed out after "
                        "one retry."
                    )
                ) from exc

            time.sleep(1)

        except APIConnectionError as exc:
            if attempt == total_attempts:
                raise LLMServiceError(
                    (
                        f"{operation_name} could not connect "
                        "to OpenRouter after one retry."
                    )
                ) from exc

            time.sleep(1)

        except APIStatusError as exc:
            raise LLMServiceError(
                (
                    "OpenRouter returned HTTP "
                    f"{exc.status_code} during "
                    f"{operation_name}."
                )
            ) from exc

    raise LLMServiceError(
        f"{operation_name} failed unexpectedly."
    )


def run_llm_test(
    settings: Settings | None = None,
) -> LLMTestResponse:
    """
    Make one structured connectivity test to the configured
    OpenRouter model.

    Because openrouter/free can route among free models, one
    bounded second attempt is allowed if the first response is
    malformed or does not match the expected schema.
    """
    settings = settings or get_settings()

    client = create_openrouter_client(
        settings
    )

    total_attempts = 2
    last_error: Exception | None = None

    for attempt in range(
        1,
        total_attempts + 1,
    ):
        try:
            completion = _call_with_one_retry(
                lambda: (
                    client
                    .chat
                    .completions
                    .create(
                        model=(
                            settings.openrouter_model
                        ),
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    LLM_TEST_SYSTEM_PROMPT
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    LLM_TEST_USER_PROMPT
                                ),
                            },
                        ],
                        max_tokens=min(
                            settings.llm_max_output_tokens,
                            500,
                        ),
                        response_format={
                            "type": "json_object",
                        },
                    )
                ),
                operation_name=(
                    "LLM connectivity test"
                ),
            )

            if not completion.choices:
                raise LLMServiceError(
                    "OpenRouter returned no choices."
                )

            content = (
                completion
                .choices[0]
                .message
                .content
            )

            if not content:
                raise LLMServiceError(
                    (
                        "OpenRouter returned an empty "
                        "model response."
                    )
                )

            raw_data = _extract_json_object(
                content
            )

            return LLMTestResponse.model_validate(
                raw_data
            )

        except (
            LLMServiceError,
            ValueError,
        ) as exc:
            last_error = exc

            if attempt == total_attempts:
                break

    raise LLMServiceError(
        (
            "The free routed model did not produce a valid "
            "LLMTestResponse after one bounded retry. "
            f"Last error: {last_error}"
        )
    ) from last_error