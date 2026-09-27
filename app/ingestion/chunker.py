import re
from dataclasses import dataclass

from app.models.documents import (
    ApprovalStatus,
    DocumentChunk,
    DocumentMetadata,
    LoadedDocument,
)


class ChunkingError(RuntimeError):
    """
    Raised when a document cannot be safely chunked.
    """


@dataclass
class TextSection:
    """
    Internal representation of a detected section.
    """

    title: str | None
    text: str
    page_number: int | None


def chunk_document(
    document: LoadedDocument,
    metadata: DocumentMetadata,
    *,
    max_chunk_words: int = 600,
    overlap_words: int = 80,
) -> list[DocumentChunk]:
    """
    Split a cleaned document into traceable, section-aware chunks.
    """
    if metadata.approval_status != ApprovalStatus.APPROVED:
        raise ChunkingError(
            "Only APPROVED documents may be chunked "
            "for the active corpus."
        )

    if max_chunk_words < 1:
        raise ChunkingError(
            "max_chunk_words must be greater than zero."
        )

    if overlap_words < 0:
        raise ChunkingError(
            "overlap_words cannot be negative."
        )

    if overlap_words >= max_chunk_words:
        raise ChunkingError(
            "overlap_words must be smaller than max_chunk_words."
        )

    sections = _extract_sections(document)

    chunks: list[DocumentChunk] = []

    for section in sections:
        section_chunks = _split_text_with_overlap(
            section.text,
            max_chunk_words=max_chunk_words,
            overlap_words=overlap_words,
        )

        for chunk_text in section_chunks:
            chunk_index = len(chunks)

            chunk_id = (
                f"{metadata.document_id}"
                f"-CHUNK-{chunk_index:04d}"
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=metadata.document_id,
                    chunk_index=chunk_index,
                    title=metadata.title,
                    section_title=section.title,
                    page_number=section.page_number,
                    document_type=metadata.document_type,
                    clinical_domain=metadata.clinical_domain,
                    publisher=metadata.publisher,
                    source_authority=metadata.source_authority,
                    version=metadata.version,
                    approval_status=metadata.approval_status,
                    publication_date=metadata.publication_date,
                    effective_date=metadata.effective_date,
                    source_url=metadata.source_url,
                    text=chunk_text,
                    word_count=len(chunk_text.split()),
                )
            )

    if not chunks:
        raise ChunkingError(
            f"No chunks were created from document "
            f"{metadata.document_id}."
        )

    return chunks


def _extract_sections(
    document: LoadedDocument,
) -> list[TextSection]:
    """
    Detect likely section headings while preserving page provenance.

    Heading detection is intentionally conservative.
    """
    sections: list[TextSection] = []

    current_title: str | None = None
    current_lines: list[str] = []
    current_page: int | None = None

    def flush_current_section() -> None:
        nonlocal current_lines

        text = "\n".join(
            current_lines
        ).strip()

        if text:
            sections.append(
                TextSection(
                    title=current_title,
                    text=text,
                    page_number=current_page,
                )
            )

        current_lines = []

    for page in document.pages:
        lines = [
            line.strip()
            for line in page.text.splitlines()
            if line.strip()
        ]

        for line in lines:
            if _looks_like_heading(line):
                flush_current_section()

                current_title = line
                current_page = page.page_number
                continue

            if not current_lines:
                current_page = page.page_number

            current_lines.append(line)

    flush_current_section()

    if not sections and document.full_text.strip():
        sections.append(
            TextSection(
                title=None,
                text=document.full_text.strip(),
                page_number=(
                    document.pages[0].page_number
                    if document.pages
                    else None
                ),
            )
        )

    return sections


def _looks_like_heading(
    line: str,
) -> bool:
    """
    Conservative heuristic for likely headings.

    It recognizes short all-uppercase headings and common numbered
    section-heading patterns.
    """
    stripped = line.strip()

    if not stripped:
        return False

    if len(stripped) > 120:
        return False

    words = stripped.split()

    if len(words) > 14:
        return False

    # Example:
    # "3. Treatment"
    # "4.2 Pregnancy"
    numbered_heading = re.match(
        r"^\d+(?:\.\d+)*\.?\s+\S+",
        stripped,
    )

    if numbered_heading:
        return True

    # Example:
    # "TREATMENT IN PREGNANCY"
    letters = [
        char
        for char in stripped
        if char.isalpha()
    ]

    if letters and stripped.upper() == stripped:
        return True

    return False


def _split_text_with_overlap(
    text: str,
    *,
    max_chunk_words: int,
    overlap_words: int,
) -> list[str]:
    """
    Split text into word-based windows with bounded overlap.
    """
    words = text.split()

    if not words:
        return []

    if len(words) <= max_chunk_words:
        return [
            " ".join(words)
        ]

    chunks: list[str] = []

    start = 0

    while start < len(words):
        end = min(
            start + max_chunk_words,
            len(words),
        )

        chunk_words = words[start:end]

        chunk_text = " ".join(
            chunk_words
        ).strip()

        if chunk_text:
            chunks.append(chunk_text)

        if end >= len(words):
            break

        start = end - overlap_words

    return chunks