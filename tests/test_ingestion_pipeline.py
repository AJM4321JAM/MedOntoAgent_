from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.config.settings import Settings
from app.ingestion.pipeline import (
    DocumentIngestionError,
    ingest_document,
)
from app.ingestion.registry import DocumentRegistry
from app.models.documents import (
    ApprovalStatus,
    IndexStatus,
    IngestionStatus,
    SourceAuthority,
)


def create_source(
    tmp_path: Path,
) -> Path:
    """
    Create one deterministic synthetic source file.
    """
    path = (
        tmp_path
        / "test_document.txt"
    )

    path.write_text(
        (
            "Example Antihypertensive Class "
            "is discussed during Pregnancy."
        ),
        encoding="utf-8",
    )

    return path


def create_settings() -> Settings:
    """
    Isolated settings for ingestion unit tests.

    External services are mocked, so no real credentials
    are required.
    """
    return Settings(
        _env_file=None,
        qdrant_url=None,
        qdrant_api_key=None,
        openrouter_api_key=None,
    )


def test_pending_document_is_registered_but_not_indexed(
    tmp_path: Path,
) -> None:
    source = create_source(
        tmp_path
    )

    registry_path = (
        tmp_path
        / "registry.json"
    )

    with (
        patch(
            "app.ingestion.pipeline.load_document"
        ) as mock_load,
        patch(
            "app.ingestion.pipeline.embed_texts"
        ) as mock_embed,
        patch(
            "app.ingestion.pipeline.ensure_collection"
        ) as mock_ensure,
        patch(
            "app.ingestion.pipeline.upsert_chunks"
        ) as mock_upsert,
    ):
        result = ingest_document(
            source,
            title="Pending Test",
            publisher="MedOntoAgent",
            document_type=(
                "Synthetic Test Document"
            ),
            clinical_domain=(
                "Synthetic Evaluation"
            ),
            source_authority=(
                SourceAuthority.OTHER
            ),
            version="1.0",
            approval_status=(
                ApprovalStatus.PENDING
            ),
            registry_path=(
                registry_path
            ),
            settings=(
                create_settings()
            ),
        )

    assert (
        result.status
        == IngestionStatus.NOT_APPROVED
    )

    assert result.chunk_count == 0
    assert result.vector_count == 0

    registry = DocumentRegistry(
        registry_path
    )

    documents = (
        registry.list_documents()
    )

    assert len(documents) == 1

    metadata = documents[0]

    assert (
        metadata.approval_status
        == ApprovalStatus.PENDING
    )

    assert (
        metadata.index_status
        == IndexStatus.NOT_INDEXED
    )

    assert metadata.indexed_at is None
    assert metadata.index_error is None

    mock_load.assert_not_called()
    mock_embed.assert_not_called()
    mock_ensure.assert_not_called()
    mock_upsert.assert_not_called()


def test_approved_document_is_processed_and_marked_indexed(
    tmp_path: Path,
) -> None:
    source = create_source(
        tmp_path
    )

    registry_path = (
        tmp_path
        / "registry.json"
    )

    loaded = SimpleNamespace(
        filename=source.name,
    )

    cleaned = SimpleNamespace(
        filename=source.name,
    )

    chunk = SimpleNamespace(
        text=(
            "Example Antihypertensive Class "
            "is discussed during Pregnancy."
        ),
    )

    tagged_chunk = SimpleNamespace(
        text=chunk.text,
        ontology_concepts=[
            "ExampleAntihypertensiveClass",
            "Pregnancy",
        ],
    )

    with (
        patch(
            "app.ingestion.pipeline.load_document",
            return_value=loaded,
        ) as mock_load,
        patch(
            "app.ingestion.pipeline.clean_document",
            return_value=cleaned,
        ) as mock_clean,
        patch(
            "app.ingestion.pipeline.chunk_document",
            return_value=[
                chunk
            ],
        ) as mock_chunk,
        patch(
            "app.ingestion.pipeline.load_ontology",
            return_value=SimpleNamespace(),
        ) as mock_ontology,
        patch(
            "app.ingestion.pipeline.tag_chunks_with_ontology",
            return_value=[
                tagged_chunk
            ],
        ) as mock_tag,
        patch(
            "app.ingestion.pipeline.embed_texts",
            return_value=[
                [0.1] * 384
            ],
        ) as mock_embed,
        patch(
            "app.ingestion.pipeline.ensure_collection"
        ) as mock_ensure,
        patch(
            "app.ingestion.pipeline.upsert_chunks",
            return_value=1,
        ) as mock_upsert,
    ):
        result = ingest_document(
            source,
            title="Approved Test",
            publisher="MedOntoAgent",
            document_type=(
                "Synthetic Test Document"
            ),
            clinical_domain=(
                "Synthetic Evaluation"
            ),
            source_authority=(
                SourceAuthority.OTHER
            ),
            version="1.0",
            approval_status=(
                ApprovalStatus.APPROVED
            ),
            registry_path=(
                registry_path
            ),
            settings=(
                create_settings()
            ),
        )

    assert (
        result.status
        == IngestionStatus.INGESTED
    )

    assert result.chunk_count == 1
    assert result.vector_count == 1

    assert (
        result.ontology_tagged_chunk_count
        == 1
    )

    registry = DocumentRegistry(
        registry_path
    )

    metadata = (
        registry.list_documents()[0]
    )

    assert (
        metadata.approval_status
        == ApprovalStatus.APPROVED
    )

    assert (
        metadata.index_status
        == IndexStatus.INDEXED
    )

    assert metadata.indexed_at is not None
    assert metadata.index_error is None

    mock_load.assert_called_once()
    mock_clean.assert_called_once()
    mock_chunk.assert_called_once()
    mock_ontology.assert_called_once()
    mock_tag.assert_called_once()
    mock_embed.assert_called_once()
    mock_ensure.assert_called_once()
    mock_upsert.assert_called_once()


def test_already_indexed_file_returns_duplicate_without_reindexing(
    tmp_path: Path,
) -> None:
    source = create_source(
        tmp_path
    )

    registry_path = (
        tmp_path
        / "registry.json"
    )

    tagged_chunk = SimpleNamespace(
        text=(
            "Example Antihypertensive Class "
            "is discussed during Pregnancy."
        ),
        ontology_concepts=[
            "ExampleAntihypertensiveClass",
            "Pregnancy",
        ],
    )

    with (
        patch(
            "app.ingestion.pipeline.load_document",
            return_value=SimpleNamespace(),
        ),
        patch(
            "app.ingestion.pipeline.clean_document",
            return_value=SimpleNamespace(),
        ),
        patch(
            "app.ingestion.pipeline.chunk_document",
            return_value=[
                SimpleNamespace(
                    text=tagged_chunk.text
                )
            ],
        ),
        patch(
            "app.ingestion.pipeline.load_ontology",
            return_value=SimpleNamespace(),
        ),
        patch(
            "app.ingestion.pipeline.tag_chunks_with_ontology",
            return_value=[
                tagged_chunk
            ],
        ),
        patch(
            "app.ingestion.pipeline.embed_texts",
            return_value=[
                [0.1] * 384
            ],
        ),
        patch(
            "app.ingestion.pipeline.ensure_collection"
        ),
        patch(
            "app.ingestion.pipeline.upsert_chunks",
            return_value=1,
        ),
    ):
        first = ingest_document(
            source,
            title="Approved Test",
            publisher="MedOntoAgent",
            document_type=(
                "Synthetic Test Document"
            ),
            clinical_domain=(
                "Synthetic Evaluation"
            ),
            source_authority=(
                SourceAuthority.OTHER
            ),
            version="1.0",
            approval_status=(
                ApprovalStatus.APPROVED
            ),
            registry_path=(
                registry_path
            ),
            settings=(
                create_settings()
            ),
        )

    assert (
        first.status
        == IngestionStatus.INGESTED
    )

    with (
        patch(
            "app.ingestion.pipeline.load_document"
        ) as mock_load,
        patch(
            "app.ingestion.pipeline.embed_texts"
        ) as mock_embed,
        patch(
            "app.ingestion.pipeline.ensure_collection"
        ) as mock_ensure,
        patch(
            "app.ingestion.pipeline.upsert_chunks"
        ) as mock_upsert,
    ):
        second = ingest_document(
            source,
            title="Approved Test",
            publisher="MedOntoAgent",
            document_type=(
                "Synthetic Test Document"
            ),
            clinical_domain=(
                "Synthetic Evaluation"
            ),
            source_authority=(
                SourceAuthority.OTHER
            ),
            version="1.0",
            approval_status=(
                ApprovalStatus.APPROVED
            ),
            registry_path=(
                registry_path
            ),
            settings=(
                create_settings()
            ),
        )

    assert (
        second.status
        == IngestionStatus.DUPLICATE
    )

    assert second.chunk_count == 0
    assert second.vector_count == 0

    assert (
        second.ontology_tagged_chunk_count
        == 0
    )

    mock_load.assert_not_called()
    mock_embed.assert_not_called()
    mock_ensure.assert_not_called()
    mock_upsert.assert_not_called()


def test_failed_indexing_is_recorded_and_retry_is_allowed(
    tmp_path: Path,
) -> None:
    source = create_source(
        tmp_path
    )

    registry_path = (
        tmp_path
        / "registry.json"
    )

    tagged_chunk = SimpleNamespace(
        text=(
            "Example Antihypertensive Class "
            "is discussed during Pregnancy."
        ),
        ontology_concepts=[
            "ExampleAntihypertensiveClass",
            "Pregnancy",
        ],
    )

    common_patches = (
        patch(
            "app.ingestion.pipeline.load_document",
            return_value=SimpleNamespace(),
        ),
        patch(
            "app.ingestion.pipeline.clean_document",
            return_value=SimpleNamespace(),
        ),
        patch(
            "app.ingestion.pipeline.chunk_document",
            return_value=[
                SimpleNamespace(
                    text=tagged_chunk.text
                )
            ],
        ),
        patch(
            "app.ingestion.pipeline.load_ontology",
            return_value=SimpleNamespace(),
        ),
        patch(
            "app.ingestion.pipeline.tag_chunks_with_ontology",
            return_value=[
                tagged_chunk
            ],
        ),
        patch(
            "app.ingestion.pipeline.embed_texts",
            return_value=[
                [0.1] * 384
            ],
        ),
        patch(
            "app.ingestion.pipeline.ensure_collection"
        ),
    )

    with (
        common_patches[0],
        common_patches[1],
        common_patches[2],
        common_patches[3],
        common_patches[4],
        common_patches[5],
        common_patches[6],
        patch(
            "app.ingestion.pipeline.upsert_chunks",
            side_effect=RuntimeError(
                "simulated storage failure"
            ),
        ),
    ):
        with pytest.raises(
            DocumentIngestionError,
            match=(
                "Document processing or "
                "indexing failed"
            ),
        ):
            ingest_document(
                source,
                title="Retry Test",
                publisher="MedOntoAgent",
                document_type=(
                    "Synthetic Test Document"
                ),
                clinical_domain=(
                    "Synthetic Evaluation"
                ),
                source_authority=(
                    SourceAuthority.OTHER
                ),
                version="1.0",
                approval_status=(
                    ApprovalStatus.APPROVED
                ),
                registry_path=(
                    registry_path
                ),
                settings=(
                    create_settings()
                ),
            )

    registry = DocumentRegistry(
        registry_path
    )

    failed_metadata = (
        registry.list_documents()[0]
    )

    assert (
        failed_metadata.index_status
        == IndexStatus.FAILED
    )

    assert (
        failed_metadata.index_error
        == (
            "Document processing or "
            "indexing failed."
        )
    )

    with (
        patch(
            "app.ingestion.pipeline.load_document",
            return_value=SimpleNamespace(),
        ),
        patch(
            "app.ingestion.pipeline.clean_document",
            return_value=SimpleNamespace(),
        ),
        patch(
            "app.ingestion.pipeline.chunk_document",
            return_value=[
                SimpleNamespace(
                    text=tagged_chunk.text
                )
            ],
        ),
        patch(
            "app.ingestion.pipeline.load_ontology",
            return_value=SimpleNamespace(),
        ),
        patch(
            "app.ingestion.pipeline.tag_chunks_with_ontology",
            return_value=[
                tagged_chunk
            ],
        ),
        patch(
            "app.ingestion.pipeline.embed_texts",
            return_value=[
                [0.1] * 384
            ],
        ),
        patch(
            "app.ingestion.pipeline.ensure_collection"
        ),
        patch(
            "app.ingestion.pipeline.upsert_chunks",
            return_value=1,
        ) as mock_upsert,
    ):
        retry_result = ingest_document(
            source,
            title="Retry Test",
            publisher="MedOntoAgent",
            document_type=(
                "Synthetic Test Document"
            ),
            clinical_domain=(
                "Synthetic Evaluation"
            ),
            source_authority=(
                SourceAuthority.OTHER
            ),
            version="1.0",
            approval_status=(
                ApprovalStatus.APPROVED
            ),
            registry_path=(
                registry_path
            ),
            settings=(
                create_settings()
            ),
        )

    assert (
        retry_result.status
        == IngestionStatus.INGESTED
    )

    mock_upsert.assert_called_once()

    indexed_metadata = (
        DocumentRegistry(
            registry_path
        )
        .list_documents()[0]
    )

    assert (
        indexed_metadata.index_status
        == IndexStatus.INDEXED
    )

    assert (
        indexed_metadata.index_error
        is None
    )