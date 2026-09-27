import json

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
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
    Raised when the external LLM service cannot complete a request.
    """


def create_openrouter_client(
    settings: Settings | None = None,
) -> OpenAI:
    """
    Create an OpenAI-compatible client pointed at OpenRouter.
    """
    settings = settings or get_settings()

    if settings.openrouter_api_key is None:
        raise LLMConfigurationError(
            "OPENROUTER_API_KEY is not configured."
        )

    return OpenAI(
        api_key=settings.openrouter_api_key.get_secret_value(),
        base_url=settings.openrouter_base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=0,
    )


def _extract_json_object(
    content: str,
) -> dict:
    """
    Extract a JSON object from a model response.

    Free models may wrap JSON in Markdown code fences even when
    instructed to return JSON only.
    """
    cleaned = content.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        data = json.loads(cleaned)

    except json.JSONDecodeError as exc:
        raise LLMServiceError(
            "The model returned an invalid JSON response. "
            f"Raw response: {content!r}"
        ) from exc

    if not isinstance(data, dict):
        raise LLMServiceError(
            "The model response was valid JSON but was not a JSON object."
        )

    return data


def run_llm_test(
    settings: Settings | None = None,
) -> LLMTestResponse:
    """
    Make one structured test request to the configured OpenRouter model.

    For free OpenRouter models we request JSON output and then validate
    the returned data locally with Pydantic.
    """
    settings = settings or get_settings()
    client = create_openrouter_client(settings)

    try:
        completion = client.chat.completions.create(
            model=settings.openrouter_model,
            messages=[
                {
                    "role": "system",
                    "content": LLM_TEST_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": LLM_TEST_USER_PROMPT,
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

    except APITimeoutError as exc:
        raise LLMServiceError(
            "The LLM request timed out."
        ) from exc

    except APIConnectionError as exc:
        raise LLMServiceError(
            "Could not connect to OpenRouter."
        ) from exc

    except APIStatusError as exc:
        raise LLMServiceError(
            f"OpenRouter returned HTTP {exc.status_code}."
        ) from exc

    content = completion.choices[0].message.content

    if not content:
        raise LLMServiceError(
            "OpenRouter returned an empty model response."
        )

    raw_data = _extract_json_object(content)

    try:
        return LLMTestResponse.model_validate(raw_data)

    except ValueError as exc:
        raise LLMServiceError(
            "The model returned JSON, but it did not match "
            "the expected LLMTestResponse schema. "
            f"Raw response: {content!r}"
        ) from exc