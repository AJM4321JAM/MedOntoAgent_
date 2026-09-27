from unittest.mock import MagicMock

import pytest

from app.config.settings import Settings
from app.retrieval.qdrant_client import (
    QdrantConfigurationError,
    QdrantStorageError,
    chunk_point_id,
    chunk_to_payload,
    create_qdrant_client,
    ensure_collection,
    retrieve_point_by_chunk_id,
    upsert_chunks,
)


def create_test_settings(
    *,
    dimensions: int = 4,
) -> Settings:
    """
    Small isolated configuration for tests.
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
        embedding_model="test-model",
        embedding_dimensions=dimensions,
        embedding_batch_size=2,
        embedding_device="cpu",
    )


def create_test_chunk() -> MagicMock:
    """
    Fake DocumentChunk-like object.

    Using a mock keeps this test focused on the Qdrant
    adapter rather than duplicating the full document model.
    """
    chunk = MagicMock()

    chunk.chunk_id = (
        "DOC-123456789abc-CHUNK-0000"
    )

    chunk.model_dump.return_value = {
        "chunk_id": (
            "DOC-123456789abc-CHUNK-0000"
        ),
        "document_id": (
            "DOC-123456789abc"
        ),
        "chunk_index": 0,
        "title": "Synthetic Test Document",
        "section_title": "Treatment",
        "page_number": 1,
        "document_type": (
            "Synthetic Test Document"
        ),
        "clinical_domain": (
            "Software Testing"
        ),
        "publisher": "MedOntoAgent",
        "source_authority": "OTHER",
        "version": "1.0",
        "approval_status": "APPROVED",
        "publication_date": None,
        "effective_date": None,
        "source_url": None,
        "text": (
            "Synthetic evidence used only "
            "for vector-storage testing."
        ),
        "word_count": 8,
    }

    return chunk


def test_missing_qdrant_url_is_rejected() -> None:
    settings = Settings(
        _env_file=None,
        qdrant_url=None,
        qdrant_api_key="test-key",
    )

    with pytest.raises(
        QdrantConfigurationError,
        match="QDRANT_URL",
    ):
        create_qdrant_client(
            settings
        )


def test_missing_qdrant_api_key_is_rejected() -> None:
    settings = Settings(
        _env_file=None,
        qdrant_url=(
            "https://example.qdrant.io"
        ),
        qdrant_api_key=None,
    )

    with pytest.raises(
        QdrantConfigurationError,
        match="QDRANT_API_KEY",
    ):
        create_qdrant_client(
            settings
        )


def test_chunk_point_id_is_deterministic() -> None:
    first = chunk_point_id(
        "DOC-123-CHUNK-0000"
    )

    second = chunk_point_id(
        "DOC-123-CHUNK-0000"
    )

    assert first == second


def test_chunk_to_payload_preserves_metadata() -> None:
    chunk = create_test_chunk()

    payload = chunk_to_payload(
        chunk
    )

    assert (
        payload["chunk_id"]
        == "DOC-123456789abc-CHUNK-0000"
    )

    assert (
        payload["document_id"]
        == "DOC-123456789abc"
    )

    assert (
        payload["approval_status"]
        == "APPROVED"
    )

    assert (
        payload["clinical_domain"]
        == "Software Testing"
    )


def test_chunk_vector_count_mismatch_is_rejected() -> None:
    settings = create_test_settings()

    chunk = create_test_chunk()

    client = MagicMock()

    with pytest.raises(
        QdrantStorageError,
        match="count mismatch",
    ):
        upsert_chunks(
            chunks=[
                chunk
            ],
            vectors=[],
            settings=settings,
            client=client,
        )

    client.upsert.assert_not_called()


def test_vector_dimension_mismatch_is_rejected() -> None:
    settings = (
        create_test_settings(
            dimensions=4
        )
    )

    chunk = create_test_chunk()

    client = MagicMock()

    with pytest.raises(
        QdrantStorageError,
        match="dimension mismatch",
    ):
        upsert_chunks(
            chunks=[
                chunk
            ],
            vectors=[
                [
                    0.1,
                    0.2,
                    0.3,
                ]
            ],
            settings=settings,
            client=client,
        )

    client.upsert.assert_not_called()


def test_valid_chunk_is_upserted() -> None:
    settings = (
        create_test_settings(
            dimensions=4
        )
    )

    chunk = create_test_chunk()

    client = MagicMock()

    count = upsert_chunks(
        chunks=[
            chunk
        ],
        vectors=[
            [
                0.1,
                0.2,
                0.3,
                0.4,
            ]
        ],
        settings=settings,
        client=client,
    )

    assert count == 1

    client.upsert.assert_called_once()

    call = (
        client
        .upsert
        .call_args
    )

    assert (
        call.kwargs[
            "collection_name"
        ]
        == "test_collection"
    )

    assert (
        call.kwargs["wait"]
        is True
    )


def test_existing_collection_is_not_recreated() -> None:
    settings = (
        create_test_settings(
            dimensions=4
        )
    )

    client = MagicMock()

    client.collection_exists.return_value = (
        True
    )

    vector_config = MagicMock()

    vector_config.size = 4

    from qdrant_client.http import models

    vector_config.distance = (
        models.Distance.COSINE
    )

    info = MagicMock()

    info.config.params.vectors = (
        vector_config
    )

    info.payload_schema = {
        "chunk_id": {},
        "document_id": {},
        "approval_status": {},
        "clinical_domain": {},
        "document_type": {},
        "source_authority": {},
        "version": {},
    }

    client.get_collection.return_value = (
        info
    )

    ensure_collection(
        settings=settings,
        client=client,
    )

    client.create_collection.assert_not_called()


def test_retrieve_point_by_chunk_id_returns_none_when_missing() -> None:
    settings = create_test_settings()

    client = MagicMock()

    client.scroll.return_value = (
        [],
        None,
    )

    result = retrieve_point_by_chunk_id(
        chunk_id="DOC-123-CHUNK-0000",
        settings=settings,
        client=client,
    )

    assert result is None

def test_chunk_id_is_part_of_payload_indexes() -> None:
    from app.retrieval.qdrant_client import (
        PAYLOAD_INDEX_FIELDS,
    )

    assert "chunk_id" in PAYLOAD_INDEX_FIELDS