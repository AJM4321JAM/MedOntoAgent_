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
    "_judge_claim"
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

    mock_judge.side_effect = [
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
        == 2
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