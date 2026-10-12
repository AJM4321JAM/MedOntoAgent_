from unittest.mock import (
    MagicMock,
    patch,
)

import pytest

from app.config.settings import (
    Settings,
)
from app.models.documents import (
    SourceAuthority,
)
from app.models.responses import (
    MedOntoClaim,
    MedOntoDraftOutput,
)
from app.models.retrieval import (
    RetrievedChunk,
    RetrievalResult,
)
from app.validation.citation_validator import (
    CitationJudgeOutput,
    CitationValidationError,
    validate_citations,
)


def create_settings() -> Settings:
    return Settings(
        _env_file=None,
        gemini_api_key="test-key",
        gemini_model="test-model",
    )


def create_retrieval() -> RetrievalResult:
    chunk = RetrievedChunk(
        chunk_id="CHUNK-1",
        document_id="DOC-1",
        title="Synthetic Test Source",
        section_title="Treatment",
        page_number=1,
        text=(
            "Treatment X is contraindicated "
            "in Group Y."
        ),
        score=0.9,
        version="1.0",
        publisher="MedOntoAgent",
        clinical_domain="Testing",
        document_type="Synthetic Test",
        source_authority=(
            SourceAuthority.OTHER
        ),
        source_url=None,
    )

    return RetrievalResult(
        query="test",
        top_k=5,
        retrieved_chunks=[
            chunk
        ],
        result_count=1,
    )


def create_draft(
    *,
    chunk_id: str = "CHUNK-1",
) -> MedOntoDraftOutput:
    return MedOntoDraftOutput(
        answer=(
            "Treatment X is contraindicated "
            "in Group Y."
        ),
        claims=[
            MedOntoClaim(
                claim_text=(
                    "Treatment X is contraindicated "
                    "in Group Y."
                ),
                citation_chunk_ids=[
                    chunk_id
                ],
            )
        ],
        considered_constraints=[],
        status="DRAFT",
    )


@patch(
    "app.validation.citation_validator."
    "_judge_claim"
)
def test_supported_claim_returns_supported(
    mock_judge: MagicMock,
) -> None:
    mock_judge.return_value = (
        CitationJudgeOutput(
            verdict="SUPPORTED",
            reason=(
                "The cited evidence directly "
                "supports the claim."
            ),
        )
    )

    result = validate_citations(
        draft=create_draft(),
        retrieval=create_retrieval(),
        settings=create_settings(),
    )

    assert (
        result.all_claims_supported
        is True
    )

    assert len(
        result.items
    ) == 1

    assert (
        result.items[0].verdict
        == "SUPPORTED"
    )


@patch(
    "app.validation.citation_validator."
    "_judge_claim"
)
def test_unsupported_claim_is_detected(
    mock_judge: MagicMock,
) -> None:
    mock_judge.return_value = (
        CitationJudgeOutput(
            verdict="UNSUPPORTED",
            reason=(
                "The cited evidence does not "
                "support the claim."
            ),
        )
    )

    result = validate_citations(
        draft=create_draft(),
        retrieval=create_retrieval(),
        settings=create_settings(),
    )

    assert (
        result.all_claims_supported
        is False
    )

    assert (
        result.items[0].verdict
        == "UNSUPPORTED"
    )


def test_validator_rejects_nonretrieved_chunk_id() -> None:
    with pytest.raises(
        CitationValidationError,
        match="not retrieved",
    ):
        validate_citations(
            draft=create_draft(
                chunk_id="FAKE-CHUNK"
            ),
            retrieval=create_retrieval(),
            settings=create_settings(),
        )


@patch(
    "app.validation.citation_validator."
    "_judge_claims_batch"
)
def test_each_claim_is_validated_independently(
    mock_judge: MagicMock,
) -> None:
    draft = MedOntoDraftOutput(
        answer="Synthetic answer.",
        claims=[
            MedOntoClaim(
                claim_text="Claim one.",
                citation_chunk_ids=[
                    "CHUNK-1"
                ],
            ),
            MedOntoClaim(
                claim_text="Claim two.",
                citation_chunk_ids=[
                    "CHUNK-1"
                ],
            ),
        ],
        considered_constraints=[],
        status="DRAFT",
    )

    mock_judge.return_value = [
        CitationJudgeOutput(
            verdict="SUPPORTED",
            reason="Supported.",
        ),
        CitationJudgeOutput(
            verdict="UNSUPPORTED",
            reason="Not supported.",
        ),
    ]

    result = validate_citations(
        draft=draft,
        retrieval=create_retrieval(),
        settings=create_settings(),
    )

    assert (
        mock_judge.call_count
        == 1
    )

    assert [
        item.verdict
        for item in result.items
    ] == [
        "SUPPORTED",
        "UNSUPPORTED",
    ]

    assert (
        result.all_claims_supported
        is False
    )

def test_batch_output_requires_complete_unique_indices() -> None:
    from app.validation.citation_validator import CitationBatchOutput
    from pydantic import ValidationError
    from unittest.mock import MagicMock, patch
    from app.validation.citation_validator import _judge_claims_batch

    draft = MedOntoDraftOutput(
        answer="Synthetic answer.",
        claims=[
            MedOntoClaim(claim_text="First claim", citation_chunk_ids=["CHUNK-1"]),
            MedOntoClaim(claim_text="Second claim", citation_chunk_ids=["CHUNK-1"]),
        ],
        considered_constraints=[], status="DRAFT",
    )
    assert CitationBatchOutput.model_validate({"judgments": []}).judgments == []

    for bad_indices in ([0], [0, 0], [0, 2]):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = __import__("json").dumps({
            "judgments": [
                {"claim_index": i, "verdict": "SUPPORTED", "reason": "Synthetic"}
                for i in bad_indices
            ]
        })
        with (
            patch("app.validation.citation_validator.create_llm_client"),
            patch("app.validation.citation_validator._call_with_one_retry", return_value=response) as call,
        ):
            with pytest.raises(CitationValidationError, match="complete judgments"):
                _judge_claims_batch(
                    draft=draft, retrieval=create_retrieval(), settings=create_settings(),
                )
            assert call.call_count == 2


def test_batch_output_orders_judgments_by_claim_index() -> None:
    from unittest.mock import MagicMock, patch
    from app.validation.citation_validator import _judge_claims_batch
    import json

    draft = MedOntoDraftOutput(
        answer="Synthetic answer.",
        claims=[
            MedOntoClaim(claim_text="First", citation_chunk_ids=["CHUNK-1"]),
            MedOntoClaim(claim_text="Second", citation_chunk_ids=["CHUNK-1"]),
        ],
        considered_constraints=[], status="DRAFT",
    )
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps({"judgments": [
        {"claim_index": 1, "verdict": "UNSUPPORTED", "reason": "Second fails"},
        {"claim_index": 0, "verdict": "SUPPORTED", "reason": "First passes"},
    ]})
    with (
        patch("app.validation.citation_validator.create_llm_client"),
        patch("app.validation.citation_validator._call_with_one_retry", return_value=response) as call,
    ):
        judgments = _judge_claims_batch(
            draft=draft, retrieval=create_retrieval(), settings=create_settings(),
        )
    assert call.call_count == 1
    assert [item.verdict for item in judgments] == ["SUPPORTED", "UNSUPPORTED"]


def test_batch_malformed_json_retries_once_then_fails_closed() -> None:
    from unittest.mock import MagicMock, patch
    from app.validation.citation_validator import _judge_claims_batch

    draft = MedOntoDraftOutput(
        answer="Synthetic answer",
        claims=[
            MedOntoClaim(claim_text="First", citation_chunk_ids=["CHUNK-1"]),
            MedOntoClaim(claim_text="Second", citation_chunk_ids=["CHUNK-1"]),
        ],
        considered_constraints=[], status="DRAFT",
    )
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = '{invalid json'
    with (
        patch("app.validation.citation_validator.create_llm_client"),
        patch(
            "app.validation.citation_validator._call_with_one_retry",
            return_value=response,
        ) as call,
    ):
        with pytest.raises(CitationValidationError, match="complete judgments"):
            _judge_claims_batch(
                draft=draft, retrieval=create_retrieval(), settings=create_settings(),
            )
        assert call.call_count == 2


@pytest.mark.parametrize("invalid_verdict,invalid_reason", [
    ("MAYBE", "Synthetic reason"),
    ("SUPPORTED", ""),
])
def test_batch_invalid_judgment_fails_closed(
    invalid_verdict: str, invalid_reason: str,
) -> None:
    import json
    from unittest.mock import MagicMock, patch
    from app.validation.citation_validator import _judge_claims_batch

    draft = MedOntoDraftOutput(
        answer="Synthetic answer",
        claims=[
            MedOntoClaim(claim_text="First", citation_chunk_ids=["CHUNK-1"]),
            MedOntoClaim(claim_text="Second", citation_chunk_ids=["CHUNK-1"]),
        ],
        considered_constraints=[], status="DRAFT",
    )
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps({"judgments": [
        {"claim_index": 0, "verdict": "SUPPORTED", "reason": "Good"},
        {"claim_index": 1, "verdict": invalid_verdict, "reason": invalid_reason},
    ]})
    with (
        patch("app.validation.citation_validator.create_llm_client"),
        patch(
            "app.validation.citation_validator._call_with_one_retry",
            return_value=response,
        ) as call,
    ):
        with pytest.raises(CitationValidationError, match="complete judgments"):
            _judge_claims_batch(
                draft=draft, retrieval=create_retrieval(), settings=create_settings(),
            )
        assert call.call_count == 2


def test_batch_provider_failure_does_not_trigger_output_retry() -> None:
    from unittest.mock import patch
    from app.llm.client import LLMServiceError
    from app.validation.citation_validator import _judge_claims_batch

    draft = MedOntoDraftOutput(
        answer="Synthetic answer",
        claims=[
            MedOntoClaim(claim_text="First", citation_chunk_ids=["CHUNK-1"]),
            MedOntoClaim(claim_text="Second", citation_chunk_ids=["CHUNK-1"]),
        ],
        considered_constraints=[], status="DRAFT",
    )
    with (
        patch("app.validation.citation_validator.create_llm_client"),
        patch(
            "app.validation.citation_validator._call_with_one_retry",
            side_effect=LLMServiceError("Synthetic provider outage"),
        ) as call,
    ):
        with pytest.raises(LLMServiceError, match="Synthetic provider outage"):
            _judge_claims_batch(
                draft=draft, retrieval=create_retrieval(), settings=create_settings(),
            )
        assert call.call_count == 1
