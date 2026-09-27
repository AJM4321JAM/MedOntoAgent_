from pydantic import BaseModel, Field

from app.models.documents import SourceAuthority


class RetrievedChunk(BaseModel):
    """
    One evidence chunk returned by vector retrieval.

    This keeps the rest of MedOntoAgent independent of
    Qdrant-specific response objects.
    """

    chunk_id: str = Field(
        min_length=1,
    )

    document_id: str = Field(
        min_length=1,
    )

    title: str = Field(
        min_length=1,
    )

    section_title: str | None = None

    page_number: int | None = Field(
        default=None,
        ge=1,
    )

    text: str = Field(
        min_length=1,
    )

    score: float

    version: str = Field(
        min_length=1,
    )

    publisher: str = Field(
        min_length=1,
    )

    source_authority: SourceAuthority

    source_url: str | None = None


class RetrievalResult(BaseModel):
    """
    Complete result of one retrieval operation.
    """

    query: str = Field(
        min_length=1,
    )

    top_k: int = Field(
        ge=1,
    )

    retrieved_chunks: list[RetrievedChunk]

    result_count: int = Field(
        ge=0,
    )