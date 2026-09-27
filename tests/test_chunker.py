from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.ingestion.chunker import (
    ChunkingError,
    chunk_document,
)
from app.models.documents import (
    ApprovalStatus,
    DocumentMetadata,
    DocumentPage,
    LoadedDocument,
    SourceAuthority,
)


def create_metadata(
    approval_status: ApprovalStatus = ApprovalStatus.APPROVED,
) -> DocumentMetadata:
    return DocumentMetadata(
        document_id="DOC-123456789abc",
        filename="guideline.txt",
        title="Test Clinical Guideline",
        publisher="Test Medical Society",
        document_type="Clinical Practice Guideline",
        clinical_domain="Hypertension",
        source_authority=SourceAuthority.GUIDELINE,
        version="2026",
        approval_status=approval_status,
        checksum_sha256="a" * 64,
        file_size_bytes=100,
        registered_at=datetime.now(timezone.utc),
    )


def create_loaded_document(
    text: str,
) -> LoadedDocument:
    return LoadedDocument(
        filename="guideline.txt",
        file_path=Path("guideline.txt"),
        file_type="txt",
        pages=[
            DocumentPage(
                page_number=None,
                text=text,
            )
        ],
        full_text=text,
        character_count=len(text),
    )


def test_short_document_creates_one_chunk() -> None:
    document = create_loaded_document(
        "Clinical treatment recommendation."
    )

    chunks = chunk_document(
        document,
        create_metadata(),
        max_chunk_words=100,
        overlap_words=10,
    )

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].document_id == "DOC-123456789abc"
    assert chunks[0].word_count == 3


def test_chunk_ids_are_stable_and_ordered() -> None:
    text = " ".join(
        f"word{i}"
        for i in range(250)
    )

    document = create_loaded_document(text)

    chunks = chunk_document(
        document,
        create_metadata(),
        max_chunk_words=100,
        overlap_words=20,
    )

    assert len(chunks) == 3

    assert (
        chunks[0].chunk_id
        == "DOC-123456789abc-CHUNK-0000"
    )

    assert (
        chunks[1].chunk_id
        == "DOC-123456789abc-CHUNK-0001"
    )


def test_chunks_have_expected_overlap() -> None:
    words = [
        f"word{i}"
        for i in range(180)
    ]

    document = create_loaded_document(
        " ".join(words)
    )

    chunks = chunk_document(
        document,
        create_metadata(),
        max_chunk_words=100,
        overlap_words=20,
    )

    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()

    assert first_words[-20:] == second_words[:20]


def test_numbered_heading_is_preserved_as_section() -> None:
    text = (
        "1. Diagnosis\n"
        "Diagnosis information appears here.\n"
        "2. Treatment\n"
        "Treatment information appears here."
    )

    document = create_loaded_document(text)

    chunks = chunk_document(
        document,
        create_metadata(),
        max_chunk_words=100,
        overlap_words=10,
    )

    section_titles = [
        chunk.section_title
        for chunk in chunks
    ]

    assert "1. Diagnosis" in section_titles
    assert "2. Treatment" in section_titles


def test_uppercase_heading_is_detected() -> None:
    text = (
        "CONTRAINDICATIONS\n"
        "This section contains contraindication evidence."
    )

    document = create_loaded_document(text)

    chunks = chunk_document(
        document,
        create_metadata(),
        max_chunk_words=100,
        overlap_words=10,
    )

    assert (
        chunks[0].section_title
        == "CONTRAINDICATIONS"
    )


def test_pending_document_cannot_enter_active_chunks() -> None:
    document = create_loaded_document(
        "Clinical recommendation."
    )

    metadata = create_metadata(
        ApprovalStatus.PENDING
    )

    with pytest.raises(
        ChunkingError,
        match="Only APPROVED documents",
    ):
        chunk_document(
            document,
            metadata,
        )


def test_overlap_cannot_equal_chunk_size() -> None:
    document = create_loaded_document(
        "Some clinical information."
    )

    with pytest.raises(
        ChunkingError,
        match="overlap_words must be smaller",
    ):
        chunk_document(
            document,
            create_metadata(),
            max_chunk_words=100,
            overlap_words=100,
        )