from unittest.mock import (
    MagicMock,
    patch,
)

import pytest

from app.config.settings import Settings
from app.llm.baseline import (
    BaselineRAGError,
    format_evidence,
    generate_baseline_answer,
    run_baseline_rag,
)
from app.models.documents import (
    SourceAuthority,
)
from app.models.retrieval import (
    RetrievedChunk,
    RetrievalResult,
)


def create_test_settings() -> Settings:
    return Settings(
        _env_file=None,
        gemini_api_key="test-key",
        gemini_model="test-model",
        embedding_model="test-embedding-model",
        embedding_dimensions=4,
        embedding_batch_size=2,
        embedding_device="cpu",
        retrieval_top_k=5,
    )


def create_chunk(
    chunk_id: str = (
        "DOC-test-CHUNK-0000"
    ),
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="DOC-test",
        title="Synthetic Clinical Guideline",
        section_title="Treatment",
        page_number=2,
        text=(
            "Hypertension is described in this "
            "synthetic test evidence."
        ),
        score=0.82,
        version="1.0",
        publisher="MedOntoAgent",
        source_authority=(
            SourceAuthority.OTHER
        ),
        source_url=None,
    )


def create_retrieval_result(
    chunks: list[RetrievedChunk],
) -> RetrievalResult:
    return RetrievalResult(
        query="What does the evidence say?",
        top_k=5,
        retrieved_chunks=chunks,
        result_count=len(chunks),
    )


def create_completion(
    content: str,
) -> MagicMock:
    completion = MagicMock()

    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    completion.choices = [
        choice
    ]

    return completion


def test_format_evidence_contains_source_id() -> None:
    chunk = create_chunk()

    result = format_evidence(
        [
            chunk
        ]
    )

    assert (
        "SOURCE_ID: DOC-test-CHUNK-0000"
        in result
    )

    assert (
        "Synthetic Clinical Guideline"
        in result
    )

    assert (
        chunk.text
        in result
    )


@patch(
    "app.llm.baseline.create_llm_client"
)
def test_no_evidence_abstains_without_llm_call(
    mock_create_client: MagicMock,
) -> None:
    settings = create_test_settings()

    retrieval = (
        create_retrieval_result(
            []
        )
    )

    result = generate_baseline_answer(
        query="Unknown question",
        retrieval=retrieval,
        settings=settings,
    )

    assert (
        result.status
        == "ABSTAINED"
    )

    assert result.citations == []

    mock_create_client.assert_not_called()


@patch(
    "app.llm.baseline.create_llm_client"
)
def test_answered_output_with_valid_citation(
    mock_create_client: MagicMock,
) -> None:
    settings = create_test_settings()

    chunk = create_chunk()

    retrieval = (
        create_retrieval_result(
            [
                chunk
            ]
        )
    )

    client = MagicMock()

    client.chat.completions.create.return_value = (
        create_completion(
            """
            {
              "answer": "The supplied evidence describes hypertension.",
              "citations": [
                {
                  "chunk_id": "DOC-test-CHUNK-0000"
                }
              ],
              "status": "ANSWERED"
            }
            """
        )
    )

    mock_create_client.return_value = (
        client
    )

    result = generate_baseline_answer(
        query=(
            "What does the evidence "
            "say about hypertension?"
        ),
        retrieval=retrieval,
        settings=settings,
    )

    assert (
        result.status
        == "ANSWERED"
    )

    assert len(
        result.citations
    ) == 1

    assert (
        result.citations[0].chunk_id
        == "DOC-test-CHUNK-0000"
    )


@patch(
    "app.llm.baseline.create_llm_client"
)
def test_invented_citation_is_rejected(
    mock_create_client: MagicMock,
) -> None:
    settings = create_test_settings()

    chunk = create_chunk()

    retrieval = (
        create_retrieval_result(
            [
                chunk
            ]
        )
    )

    client = MagicMock()

    client.chat.completions.create.return_value = (
        create_completion(
            """
            {
              "answer": "Unsupported answer.",
              "citations": [
                {
                  "chunk_id": "FAKE-CHUNK"
                }
              ],
              "status": "ANSWERED"
            }
            """
        )
    )

    mock_create_client.return_value = (
        client
    )

    with pytest.raises(
        BaselineRAGError,
        match="not retrieved",
    ):
        generate_baseline_answer(
            query="Clinical question",
            retrieval=retrieval,
            settings=settings,
        )


@patch(
    "app.llm.baseline.create_llm_client"
)
def test_answered_response_requires_citation(
    mock_create_client: MagicMock,
) -> None:
    settings = create_test_settings()

    chunk = create_chunk()

    retrieval = (
        create_retrieval_result(
            [
                chunk
            ]
        )
    )

    client = MagicMock()

    client.chat.completions.create.return_value = (
        create_completion(
            """
            {
              "answer": "Answer without citation.",
              "citations": [],
              "status": "ANSWERED"
            }
            """
        )
    )

    mock_create_client.return_value = (
        client
    )

    with pytest.raises(
        BaselineRAGError,
        match="requires at least one",
    ):
        generate_baseline_answer(
            query="Clinical question",
            retrieval=retrieval,
            settings=settings,
        )


@patch(
    "app.llm.baseline.create_llm_client"
)
def test_abstained_response_rejects_citations(
    mock_create_client: MagicMock,
) -> None:
    settings = create_test_settings()

    chunk = create_chunk()

    retrieval = (
        create_retrieval_result(
            [
                chunk
            ]
        )
    )

    client = MagicMock()

    client.chat.completions.create.return_value = (
        create_completion(
            """
            {
              "answer": "Evidence insufficient.",
              "citations": [
                {
                  "chunk_id": "DOC-test-CHUNK-0000"
                }
              ],
              "status": "ABSTAINED"
            }
            """
        )
    )

    mock_create_client.return_value = (
        client
    )

    with pytest.raises(
        BaselineRAGError,
        match="must not contain citations",
    ):
        generate_baseline_answer(
            query="Clinical question",
            retrieval=retrieval,
            settings=settings,
        )


@patch(
    "app.llm.baseline.generate_baseline_answer"
)
@patch(
    "app.llm.baseline.retrieve_baseline"
)
def test_complete_baseline_orchestration(
    mock_retrieve: MagicMock,
    mock_generate: MagicMock,
) -> None:
    settings = create_test_settings()

    chunk = create_chunk()

    mock_retrieve.return_value = (
        create_retrieval_result(
            [
                chunk
            ]
        )
    )

    from app.models.responses import (
        BaselineLLMOutput,
        CitationReference,
    )

    mock_generate.return_value = (
        BaselineLLMOutput(
            answer=(
                "Synthetic grounded answer."
            ),
            citations=[
                CitationReference(
                    chunk_id=(
                        "DOC-test-CHUNK-0000"
                    )
                )
            ],
            status="ANSWERED",
        )
    )

    result = run_baseline_rag(
        "What does the evidence say?",
        settings=settings,
    )

    assert (
        result.status
        == "ANSWERED"
    )

    assert (
        result.answer
        == "Synthetic grounded answer."
    )

    assert (
        result.retrieved_chunk_ids
        == [
            "DOC-test-CHUNK-0000"
        ]
    )

    assert (
        result.retrieved_document_ids
        == [
            "DOC-test"
        ]
    )

    assert (
        result.retrieved_contexts
        == [
            chunk.text
        ]
    )

    assert (
        result.citation_document_ids
        == [
            "DOC-test"
        ]
    )

    assert (
        result.citations[0].chunk_id
        == "DOC-test-CHUNK-0000"
    )
