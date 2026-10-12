"""Tests for safe, type-based error classification."""

from app.llm.baseline import BaselineRAGError
from app.llm.client import LLMServiceError
from app.llm.error_classification import classify_error
from app.llm.medonto_generation import MedOntoGenerationError
from app.retrieval.retriever import RetrievalError


def test_nested_provider_rate_limit() -> None:
    # An exception chain is constructed without making an API call.
    from unittest.mock import MagicMock
    from openai import RateLimitError

    response = MagicMock()
    response.status_code = 429
    response.headers = {}

    provider_error = RateLimitError(
        "Synthetic rate limit",
        response=response,
        body=None,
    )

    service_error = LLMServiceError("Service unavailable")
    service_error.__cause__ = provider_error

    baseline_error = BaselineRAGError("Generation failed")
    baseline_error.__cause__ = service_error

    assert classify_error(baseline_error) == "RATE_LIMIT"


def test_generation_error_without_provider_cause() -> None:
    error = MedOntoGenerationError("Synthetic failure")

    assert classify_error(error) == "GENERATION_ERROR"


def test_retrieval_error() -> None:
    error = RetrievalError("Synthetic retrieval failure")

    assert classify_error(error) == "RETRIEVAL_ERROR"


def test_unexpected_programming_error() -> None:
    error = AttributeError("Synthetic coding error")

    assert classify_error(error) == "OTHER_ERROR"



def _make_evaluation_record(**overrides):
    from app.models.evaluation import EvaluationResultRecord

    fields = {
        "question_id": "SYNTHETIC-001",
        "category": "ONTOLOGY_CONSTRAINT",
        "question": "Synthetic clinical question?",
        "expected_behavior": "ANSWER",
        "system": "MEDONTO",
        "execution_mode": "BASELINE_FALLBACK",
        "system_status": "ERROR",
        "behavior_match": False,
        "latency_ms": 10.0,
        "fallback_reason": "MedOntoGenerationError",
        "fallback_error": "BaselineRAGError",
    }

    fields.update(overrides)
    return EvaluationResultRecord(**fields)


def test_evaluation_record_error_categories_round_trip() -> None:
    from app.models.evaluation import EvaluationResultRecord

    record = _make_evaluation_record(
        error_category="RATE_LIMIT",
        medonto_error_category="GENERATION_ERROR",
    )

    serialized = record.model_dump_json()
    restored = EvaluationResultRecord.model_validate_json(
        serialized
    )

    assert restored.error_category == "RATE_LIMIT"
    assert restored.medonto_error_category == "GENERATION_ERROR"
    assert restored.fallback_error == "BaselineRAGError"


def test_legacy_evaluation_record_remains_compatible() -> None:
    from app.models.evaluation import EvaluationResultRecord

    record = _make_evaluation_record()
    old_data = record.model_dump(
        exclude={
            "error_category",
            "medonto_error_category",
            "execution_mode",
        }
    )

    restored = EvaluationResultRecord.model_validate(
        old_data
    )

    assert restored.error_category is None
    assert restored.medonto_error_category is None
    assert restored.execution_mode == "MEDONTO"


def test_error_categories_do_not_include_exception_details() -> None:
    record = _make_evaluation_record(
        error_category="RATE_LIMIT",
        medonto_error_category="GENERATION_ERROR",
    )

    serialized = record.model_dump_json()

    assert "RATE_LIMIT" in serialized
    assert "GENERATION_ERROR" in serialized
    assert "secret-provider-error-body" not in serialized



import pytest


@pytest.mark.parametrize(
    "code",
    [
        "GENERATION_CONSTRAINT_MISMATCH",
        "GENERATION_CITATION_INVALID",
        "GENERATION_STRUCTURE_INVALID",
        "GENERATION_SCHEMA_INVALID",
        "GENERATION_EMPTY_RESPONSE",
    ],
)
def test_structured_generation_codes_survive_retry_wrapping(
    code: str,
) -> None:
    original = MedOntoGenerationError(
        "Synthetic failure",
        code=code,
    )

    final = MedOntoGenerationError(
        "Generation failed after retry"
    )
    final.__cause__ = original

    assert classify_error(final) == code
