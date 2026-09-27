from datetime import date
from pathlib import Path

from app.ingestion.metadata import (
    build_document_id,
    calculate_sha256,
    create_document_metadata,
)
from app.models.documents import (
    ApprovalStatus,
    SourceAuthority,
)


def test_checksum_is_deterministic(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "guideline.txt"

    file_path.write_text(
        "Stable clinical source content.",
        encoding="utf-8",
    )

    first = calculate_sha256(file_path)
    second = calculate_sha256(file_path)

    assert first == second
    assert len(first) == 64


def test_different_files_have_different_checksums(
    tmp_path: Path,
) -> None:
    first_file = tmp_path / "one.txt"
    second_file = tmp_path / "two.txt"

    first_file.write_text(
        "Version one",
        encoding="utf-8",
    )

    second_file.write_text(
        "Version two",
        encoding="utf-8",
    )

    assert (
        calculate_sha256(first_file)
        != calculate_sha256(second_file)
    )


def test_document_id_is_derived_from_checksum() -> None:
    checksum = "a" * 64

    document_id = build_document_id(checksum)

    assert document_id == "DOC-aaaaaaaaaaaa"


def test_create_document_metadata(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "guideline.txt"

    file_path.write_text(
        "Example guideline content.",
        encoding="utf-8",
    )

    metadata = create_document_metadata(
        file_path=file_path,
        title="Example Hypertension Guideline",
        publisher="Example Medical Society",
        document_type="Clinical Practice Guideline",
        clinical_domain="Hypertension",
        source_authority=SourceAuthority.GUIDELINE,
        version="2026",
        approval_status=ApprovalStatus.APPROVED,
        publication_date=date(2026, 1, 1),
        effective_date=date(2026, 2, 1),
        source_url="https://example.org/guideline",
    )

    assert metadata.document_id.startswith("DOC-")
    assert metadata.filename == "guideline.txt"
    assert metadata.title == (
        "Example Hypertension Guideline"
    )
    assert (
        metadata.source_authority
        == SourceAuthority.GUIDELINE
    )
    assert (
        metadata.approval_status
        == ApprovalStatus.APPROVED
    )
    assert len(metadata.checksum_sha256) == 64
    assert metadata.file_size_bytes > 0