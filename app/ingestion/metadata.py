import hashlib
from datetime import date, datetime, timezone
from pathlib import Path

from app.models.documents import (
    ApprovalStatus,
    DocumentMetadata,
    SourceAuthority,
)


def calculate_sha256(
    file_path: str | Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Calculate the SHA-256 checksum of a file.

    The file is read in chunks so large source documents do not need
    to be loaded entirely into memory.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot calculate checksum. File does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Cannot calculate checksum. Path is not a file: {path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        while True:
            block = file_handle.read(chunk_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def build_document_id(
    checksum_sha256: str,
) -> str:
    """
    Create a stable MedOntoAgent document ID from the file checksum.
    """
    if len(checksum_sha256) != 64:
        raise ValueError(
            "A valid SHA-256 checksum must contain 64 hexadecimal characters."
        )

    return f"DOC-{checksum_sha256[:12]}"


def create_document_metadata(
    *,
    file_path: str | Path,
    title: str,
    publisher: str,
    document_type: str,
    clinical_domain: str,
    source_authority: SourceAuthority,
    version: str,
    approval_status: ApprovalStatus = ApprovalStatus.PENDING,
    publication_date: date | None = None,
    effective_date: date | None = None,
    source_url: str | None = None,
    supersedes_document_id: str | None = None,
) -> DocumentMetadata:
    """
    Build validated governance metadata for a source document.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Source document does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Source path is not a file: {path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"Source document is empty: {path.name}"
        )

    checksum = calculate_sha256(path)
    document_id = build_document_id(checksum)

    return DocumentMetadata(
        document_id=document_id,
        filename=path.name,
        title=title.strip(),
        publisher=publisher.strip(),
        document_type=document_type.strip(),
        clinical_domain=clinical_domain.strip(),
        source_authority=source_authority,
        version=version.strip(),
        publication_date=publication_date,
        effective_date=effective_date,
        approval_status=approval_status,
        source_url=source_url,
        checksum_sha256=checksum,
        file_size_bytes=path.stat().st_size,
        supersedes_document_id=supersedes_document_id,
        superseded_by_document_id=None,
        registered_at=datetime.now(timezone.utc),
    )