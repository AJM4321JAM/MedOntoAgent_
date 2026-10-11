from unittest.mock import patch

import pytest

from app.agent.execution import (
    execute_medonto,
)
from app.llm.baseline import (
    BaselineRAGError,
)
from app.llm.medonto_generation import (
    MedOntoGenerationError,
)
from app.models.responses import (
    BaselineQueryResponse,
    CitationReference,
)


def baseline_response() -> BaselineQueryResponse:
    return BaselineQueryResponse(
        query="Synthetic question?",
        answer="Synthetic baseline answer.",
        citations=[
            CitationReference(
                chunk_id="CHUNK-1",
            )
        ],
        citation_document_ids=[
            "DOC-1",
        ],
        retrieved_chunk_ids=[
            "CHUNK-1",
        ],
        retrieved_document_ids=[
            "DOC-1",
        ],
        retrieved_contexts=[
            "Synthetic context.",
        ],
        status="ANSWERED",
    )


def test_medonto_success_does_not_use_fallback() -> None:
    state = {
        "orchestration_status": "VERIFIED",
    }

    with (
        patch(
            "app.agent.execution."
            "medonto_graph.invoke",
            return_value=state,
        ),
        patch(
            "app.agent.execution."
            "run_baseline_rag",
        ) as mock_baseline,
    ):
        result = execute_medonto(
            "Synthetic question?"
        )

    assert (
        result.execution_mode
        == "MEDONTO"
    )

    assert result.medonto_state == state

    assert (
        result.baseline_response
        is None
    )

    assert result.fallback_reason is None

    mock_baseline.assert_not_called()


def test_known_medonto_failure_uses_baseline() -> None:
    response = baseline_response()

    with (
        patch(
            "app.agent.execution."
            "medonto_graph.invoke",
            side_effect=(
                MedOntoGenerationError(
                    "Synthetic operational failure."
                )
            ),
        ),
        patch(
            "app.agent.execution."
            "run_baseline_rag",
            return_value=response,
        ) as mock_baseline,
    ):
        result = execute_medonto(
            "Synthetic question?"
        )

    assert (
        result.execution_mode
        == "BASELINE_FALLBACK"
    )

    assert result.medonto_state is None

    assert (
        result.baseline_response
        is response
    )

    assert (
        result.fallback_reason
        == "MedOntoGenerationError"
    )

    mock_baseline.assert_called_once_with(
        "Synthetic question?",
        settings=None,
    )


def test_unexpected_programming_error_is_not_hidden() -> None:
    with (
        patch(
            "app.agent.execution."
            "medonto_graph.invoke",
            side_effect=AttributeError(
                "Synthetic programming defect."
            ),
        ),
        patch(
            "app.agent.execution."
            "run_baseline_rag",
        ) as mock_baseline,
    ):
        with pytest.raises(
            AttributeError,
            match="Synthetic programming defect",
        ):
            execute_medonto(
                "Synthetic question?"
            )

    mock_baseline.assert_not_called()



def test_failed_baseline_fallback_is_auditable() -> None:
    with (
        patch(
            "app.agent.execution."
            "medonto_graph.invoke",
            side_effect=(
                MedOntoGenerationError(
                    "Synthetic MedOnto failure."
                )
            ),
        ),
        patch(
            "app.agent.execution."
            "run_baseline_rag",
            side_effect=(
                BaselineRAGError(
                    "Synthetic baseline failure."
                )
            ),
        ),
    ):
        result = execute_medonto(
            "Synthetic question?"
        )

    assert (
        result.execution_mode
        == "BASELINE_FALLBACK"
    )

    assert (
        result.baseline_response
        is None
    )

    assert (
        result.fallback_reason
        == "MedOntoGenerationError"
    )

    assert (
        result.fallback_error
        == "BaselineRAGError"
    )



def test_baseline_routed_state_runs_baseline_without_failure_reason() -> None:
    response = baseline_response()

    with (
        patch(
            "app.agent.execution."
            "medonto_graph.invoke",
            return_value={
                "orchestration_status": (
                    "BASELINE_ROUTED"
                ),
            },
        ),
        patch(
            "app.agent.execution."
            "run_baseline_rag",
            return_value=response,
        ) as mock_baseline,
    ):
        result = execute_medonto(
            "Synthetic question?"
        )

    assert (
        result.execution_mode
        == "BASELINE_ROUTED"
    )

    assert result.medonto_state is None

    assert (
        result.baseline_response
        is response
    )

    assert result.fallback_reason is None

    assert result.fallback_error is None

    mock_baseline.assert_called_once_with(
        "Synthetic question?",
        settings=None,
    )



def test_successful_fallback_preserves_medonto_error_category() -> None:
    from unittest.mock import MagicMock, patch

    from app.agent.execution import execute_medonto
    from app.llm.medonto_generation import MedOntoGenerationError

    with (
        patch(
            "app.agent.execution.medonto_graph.invoke",
            side_effect=MedOntoGenerationError(
                "Synthetic generation failure"
            ),
        ),
        patch(
            "app.agent.execution.run_baseline_rag",
            return_value=MagicMock(),
        ),
    ):
        result = execute_medonto("Synthetic question?")

    assert result.execution_mode == "BASELINE_FALLBACK"
    assert result.medonto_error_category == "GENERATION_ERROR"
    assert result.error_category is None


def test_routed_baseline_preserves_rate_limit_category() -> None:
    from unittest.mock import patch, MagicMock

    from openai import RateLimitError

    from app.agent.execution import execute_medonto
    from app.llm.baseline import BaselineRAGError
    from app.llm.client import LLMServiceError

    response = MagicMock()
    response.status_code = 429
    response.headers = {}

    provider_error = RateLimitError(
        "Synthetic provider failure",
        response=response,
        body=None,
    )

    service_error = LLMServiceError("Synthetic service error")
    service_error.__cause__ = provider_error

    baseline_error = BaselineRAGError("Synthetic baseline error")
    baseline_error.__cause__ = service_error

    with (
        patch(
            "app.agent.execution.medonto_graph.invoke",
            return_value={
                "orchestration_status": "BASELINE_ROUTED"
            },
        ),
        patch(
            "app.agent.execution.run_baseline_rag",
            side_effect=baseline_error,
        ),
    ):
        result = execute_medonto("Synthetic question?")

    assert result.execution_mode == "BASELINE_ROUTED"
    assert result.fallback_error == "BaselineRAGError"
    assert result.error_category == "RATE_LIMIT"
    assert result.medonto_error_category is None
