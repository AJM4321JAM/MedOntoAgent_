from unittest.mock import (
    MagicMock,
    patch,
)

import pytest

from app.config.settings import Settings
from app.retrieval.retriever import (
    RetrievalError,
    retrieve_baseline,
)


def create_test_settings() -> Settings:
    """
    Create isolated test configuration.

    The vector dimension is deliberately reduced to 4 so
    tests remain small and do not load the real embedding model.
    """

    return Settings(
        _env_file=None,
        openrouter_api_key=None,
        qdrant_url=(
            "https://example.qdrant.io"
        ),
        qdrant_api_key=(
            "test-qdrant-key"
        ),
        qdrant_collection=(
            "test_collection"
        ),
        embedding_model=(
            "test-model"
        ),
        embedding_dimensions=4,
        embedding_batch_size=2,
        embedding_device="cpu",
        retrieval_top_k=5,
    )


def create_scored_point(
    *,
    chunk_id: str,
    score: float,
) -> MagicMock:
    """
    Build a fake Qdrant scored point.
    """

    point = MagicMock()

    point.score = score

    point.payload = {
        "chunk_id": chunk_id,
        "document_id": "DOC-TEST-001",
        "title": "Synthetic Test Guideline",
        "section_title": "Treatment",
        "page_number": 2,
        "text": (
            "Synthetic evidence used only "
            "for retrieval testing."
        ),
        "version": "1.0",
        "publisher": "MedOntoAgent",
        "source_authority": "OTHER",
        "source_url": (
            "https://example.com/test"
        ),
        "approval_status": "APPROVED",
    }

    return point


def test_empty_query_is_rejected() -> None:
    settings = (
        create_test_settings()
    )

    client = MagicMock()

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        retrieve_baseline(
            "   ",
            settings=settings,
            client=client,
        )

    client.query_points.assert_not_called()


@patch(
    "app.retrieval.retriever.embed_text"
)
def test_query_is_embedded(
    mock_embed_text: MagicMock,
) -> None:
    settings = (
        create_test_settings()
    )

    mock_embed_text.return_value = [
        0.1,
        0.2,
        0.3,
        0.4,
    ]

    client = MagicMock()

    response = MagicMock()
    response.points = []

    client.query_points.return_value = (
        response
    )

    retrieve_baseline(
        "hypertension treatment",
        settings=settings,
        client=client,
    )

    mock_embed_text.assert_called_once_with(
        "hypertension treatment",
        settings,
    )


@patch(
    "app.retrieval.retriever.embed_text"
)
def test_top_k_is_passed_to_qdrant(
    mock_embed_text: MagicMock,
) -> None:
    settings = (
        create_test_settings()
    )

    mock_embed_text.return_value = [
        0.1,
        0.2,
        0.3,
        0.4,
    ]

    client = MagicMock()

    response = MagicMock()
    response.points = []

    client.query_points.return_value = (
        response
    )

    retrieve_baseline(
        "clinical query",
        settings=settings,
        client=client,
    )

    call = (
        client
        .query_points
        .call_args
    )

    assert (
        call.kwargs["limit"]
        == 5
    )


@patch(
    "app.retrieval.retriever.embed_text"
)
def test_stored_vectors_are_not_returned(
    mock_embed_text: MagicMock,
) -> None:
    settings = (
        create_test_settings()
    )

    mock_embed_text.return_value = [
        0.1,
        0.2,
        0.3,
        0.4,
    ]

    client = MagicMock()

    response = MagicMock()
    response.points = []

    client.query_points.return_value = (
        response
    )

    retrieve_baseline(
        "clinical query",
        settings=settings,
        client=client,
    )

    call = (
        client
        .query_points
        .call_args
    )

    assert (
        call.kwargs[
            "with_vectors"
        ]
        is False
    )

    assert (
        call.kwargs[
            "with_payload"
        ]
        is True
    )


@patch(
    "app.retrieval.retriever.embed_text"
)
def test_qdrant_result_is_converted(
    mock_embed_text: MagicMock,
) -> None:
    settings = (
        create_test_settings()
    )

    mock_embed_text.return_value = [
        0.1,
        0.2,
        0.3,
        0.4,
    ]

    point = create_scored_point(
        chunk_id=(
            "DOC-TEST-001-CHUNK-0000"
        ),
        score=0.91,
    )

    client = MagicMock()

    response = MagicMock()
    response.points = [
        point
    ]

    client.query_points.return_value = (
        response
    )

    result = retrieve_baseline(
        "hypertension treatment",
        settings=settings,
        client=client,
    )

    assert result.result_count == 1

    assert len(
        result.retrieved_chunks
    ) == 1

    chunk = (
        result.retrieved_chunks[0]
    )

    assert (
        chunk.chunk_id
        == "DOC-TEST-001-CHUNK-0000"
    )

    assert chunk.score == pytest.approx(
        0.91
    )

    assert (
        chunk.document_id
        == "DOC-TEST-001"
    )

    assert (
        chunk.title
        == "Synthetic Test Guideline"
    )


@patch(
    "app.retrieval.retriever.embed_text"
)
def test_baseline_applies_approved_filter(
    mock_embed_text: MagicMock,
) -> None:
    settings = (
        create_test_settings()
    )

    mock_embed_text.return_value = [
        0.1,
        0.2,
        0.3,
        0.4,
    ]

    client = MagicMock()

    response = MagicMock()
    response.points = []

    client.query_points.return_value = (
        response
    )

    retrieve_baseline(
        "clinical query",
        settings=settings,
        client=client,
    )

    call = (
        client
        .query_points
        .call_args
    )

    query_filter = (
        call.kwargs[
            "query_filter"
        ]
    )

    condition = (
        query_filter.must[0]
    )

    assert (
        condition.key
        == "approval_status"
    )

    assert (
        condition.match.value
        == "APPROVED"
    )


@patch(
    "app.retrieval.retriever.embed_text"
)
def test_qdrant_failure_is_wrapped(
    mock_embed_text: MagicMock,
) -> None:
    settings = (
        create_test_settings()
    )

    mock_embed_text.return_value = [
        0.1,
        0.2,
        0.3,
        0.4,
    ]

    client = MagicMock()

    client.query_points.side_effect = (
        RuntimeError(
            "Qdrant unavailable"
        )
    )

    with pytest.raises(
        RetrievalError,
        match="Qdrant baseline retrieval failed",
    ):
        retrieve_baseline(
            "clinical query",
            settings=settings,
            client=client,
        )


@patch(
    "app.retrieval.retriever.embed_text"
)
def test_invalid_payload_is_rejected(
    mock_embed_text: MagicMock,
) -> None:
    settings = (
        create_test_settings()
    )

    mock_embed_text.return_value = [
        0.1,
        0.2,
        0.3,
        0.4,
    ]

    point = MagicMock()

    point.score = 0.8

    point.payload = {
        "chunk_id": "chunk-1",

        # document_id deliberately missing

        "title": "Test",
        "text": "Evidence",
        "version": "1.0",
        "publisher": "Test",
        "source_authority": "OTHER",
    }

    client = MagicMock()

    response = MagicMock()

    response.points = [
        point
    ]

    client.query_points.return_value = (
        response
    )

    with pytest.raises(
        RetrievalError,
        match="document_id",
    ):
        retrieve_baseline(
            "clinical query",
            settings=settings,
            client=client,
        )


@patch(
    "app.retrieval.retriever.embed_text"
)
def test_embedding_dimension_mismatch_is_rejected(
    mock_embed_text: MagicMock,
) -> None:
    settings = (
        create_test_settings()
    )

    mock_embed_text.return_value = [
        0.1,
        0.2,
        0.3,
    ]

    client = MagicMock()

    with pytest.raises(
        RetrievalError,
        match="dimension mismatch",
    ):
        retrieve_baseline(
            "clinical query",
            settings=settings,
            client=client,
        )

    client.query_points.assert_not_called()