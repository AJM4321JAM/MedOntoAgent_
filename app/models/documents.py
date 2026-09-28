from datetime import (
    date,
    datetime,
)
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
)


SupportedFileType = Literal[
    "pdf",
    "docx",
    "txt",
]


class ApprovalStatus(StrEnum):
    """
    Governance state of a clinical source document.
    """

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    SUPERSEDED = "SUPERSEDED"
    REJECTED = "REJECTED"


class IndexStatus(StrEnum):
    """
    Vector-index lifecycle state of a registered document.
    """

    NOT_INDEXED = "NOT_INDEXED"
    INDEXING = "INDEXING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"


class SourceAuthority(StrEnum):
    """
    Broad authority/type classification of the source.
    """

    GUIDELINE = "GUIDELINE"
    REGULATORY = "REGULATORY"
    PROTOCOL = "PROTOCOL"
    REVIEW = "REVIEW"
    RESEARCH = "RESEARCH"
    POLICY = "POLICY"
    OTHER = "OTHER"


class IngestionStatus(StrEnum):
    """
    Public result of one controlled ingestion attempt.
    """

    INGESTED = "INGESTED"
    DUPLICATE = "DUPLICATE"
    NOT_APPROVED = "NOT_APPROVED"


class DocumentPage(BaseModel):
    """
    One extracted unit from a source document.

    For PDFs this normally represents one physical page.
    For DOCX and TXT files, page_number may be None.
    """

    page_number: int | None = Field(
        default=None,
        ge=1,
    )

    text: str = Field(
        min_length=1,
    )


class LoadedDocument(BaseModel):
    """
    Standard internal representation of an extracted source document.
    """

    filename: str
    file_path: Path
    file_type: SupportedFileType

    pages: list[
        DocumentPage
    ] = Field(
        min_length=1,
    )

    full_text: str = Field(
        min_length=1,
    )

    character_count: int = Field(
        ge=1,
    )


class DocumentMetadata(BaseModel):
    """
    Governance and provenance metadata associated with one
    clinical source document.
    """

    document_id: str = Field(
        min_length=5,
    )

    filename: str = Field(
        min_length=1,
    )

    title: str = Field(
        min_length=1,
    )

    publisher: str = Field(
        min_length=1,
    )

    document_type: str = Field(
        min_length=1,
    )

    clinical_domain: str = Field(
        min_length=1,
    )

    source_authority: SourceAuthority

    version: str = Field(
        min_length=1,
    )

    publication_date: date | None = None
    effective_date: date | None = None

    approval_status: ApprovalStatus = (
        ApprovalStatus.PENDING
    )

    source_url: str | None = None

    checksum_sha256: str = Field(
        min_length=64,
        max_length=64,
    )

    file_size_bytes: int = Field(
        ge=1,
    )

    supersedes_document_id: (
        str | None
    ) = None

    superseded_by_document_id: (
        str | None
    ) = None

    registered_at: datetime

    index_status: IndexStatus = (
        IndexStatus.NOT_INDEXED
    )

    indexed_at: datetime | None = None

    index_error: str | None = None


class DocumentChunk(BaseModel):
    """
    One retrieval-ready text chunk derived from an approved
    clinical source document.
    """

    chunk_id: str = Field(
        min_length=5,
    )

    document_id: str = Field(
        min_length=5,
    )

    chunk_index: int = Field(
        ge=0,
    )

    title: str = Field(
        min_length=1,
    )

    section_title: str | None = None

    page_number: int | None = Field(
        default=None,
        ge=1,
    )

    document_type: str = Field(
        min_length=1,
    )

    clinical_domain: str = Field(
        min_length=1,
    )

    publisher: str = Field(
        min_length=1,
    )

    source_authority: SourceAuthority

    version: str = Field(
        min_length=1,
    )

    approval_status: ApprovalStatus

    publication_date: date | None = None
    effective_date: date | None = None

    source_url: str | None = None

    text: str = Field(
        min_length=1,
    )

    word_count: int = Field(
        ge=1,
    )

    checksum_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )


class IngestionResult(BaseModel):
    """
    Final result of one controlled document-ingestion attempt.
    """

    document_id: str = Field(
        min_length=5,
    )

    filename: str = Field(
        min_length=1,
    )

    status: IngestionStatus

    checksum_sha256: str = Field(
        min_length=64,
        max_length=64,
    )

    chunk_count: int = Field(
        ge=0,
    )

    vector_count: int = Field(
        ge=0,
    )

    ontology_tagged_chunk_count: int = Field(
        ge=0,
    )

    message: str = Field(
        min_length=1,
    )