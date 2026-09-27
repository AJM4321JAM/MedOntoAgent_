from unittest.mock import MagicMock, patch

from app.config.settings import Settings
from app.llm.client import (
    LLMConfigurationError,
    run_llm_test,
)


def test_missing_api_key_is_rejected() -> None:
    settings = Settings(
        _env_file=None,
        openrouter_api_key=None,
    )

    try:
        run_llm_test(settings)

    except LLMConfigurationError as exc:
        assert "OPENROUTER_API_KEY" in str(exc)

    else:
        raise AssertionError(
            "Expected missing API key to raise LLMConfigurationError."
        )


@patch("app.llm.client.OpenAI")
def test_structured_llm_response_is_valid(
    mock_openai: MagicMock,
) -> None:
    mock_message = MagicMock()
    mock_message.content = (
        '{"answer": "RAG combines retrieval with generation.", '
        '"status": "ok"}'
    )

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion

    mock_openai.return_value = mock_client

    settings = Settings(
        _env_file=None,
        openrouter_api_key="test-key",
    )

    result = run_llm_test(settings)

    assert result.status == "ok"
    assert result.answer == "RAG combines retrieval with generation."

    mock_client.chat.completions.create.assert_called_once()