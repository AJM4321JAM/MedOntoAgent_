from pathlib import Path

import pytest

from app.ingestion.metadata import (
    create_document_metadata,
)
from app.ingestion.registry import (
    DocumentRegistry,
    DuplicateDocumentError,
)
from app.models.documents import (
    ApprovalStatus,
    SourceAuthority,
)


def create_test_metadata(
    file_path: Path,
    *,
    version: str,
    approval_status: ApprovalStatus,
):
    return create_document_metadata(
        file_path=file_path,
        title=f"Test Guideline {version}",
        publisher="Test Medical Society",
        document_type="Clinical Practice Guideline",
        clinical_domain="Hypertension",
        source_authority=SourceAuthority.GUIDELINE,
        version=version,
        approval_status=approval_status,
        source_url="https://example.org",
    )


def test_register_document(
    tmp_path: Path,
) -> None:
    source = tmp_path / "guideline.txt"

    source.write_text(
        "Clinical source version one.",
        encoding="utf-8",
    )

    registry = DocumentRegistry(
        tmp_path / "registry.json"
    )

    metadata = create_test_metadata(
        source,
        version="1",
        approval_status=ApprovalStatus.PENDING,
    )

    registry.register_document(metadata)

    documents = registry.list_documents()

    assert len(documents) == 1
    assert (
        documents[0].document_id
        == metadata.document_id
    )


def test_exact_duplicate_is_rejected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "guideline.txt"

    source.write_text(
        "Same exact source.",
        encoding="utf-8",
    )

    registry = DocumentRegistry(
        tmp_path / "registry.json"
    )

    metadata = create_test_metadata(
        source,
        version="1",
        approval_status=ApprovalStatus.PENDING,
    )

    registry.register_document(metadata)

    with pytest.raises(
        DuplicateDocumentError,
        match="identical document",
    ):
        registry.register_document(metadata)


def test_update_approval_status(
    tmp_path: Path,
) -> None:
    source = tmp_path / "guideline.txt"

    source.write_text(
        "Clinical source.",
        encoding="utf-8",
    )

    registry = DocumentRegistry(
        tmp_path / "registry.json"
    )

    metadata = create_test_metadata(
        source,
        version="1",
        approval_status=ApprovalStatus.PENDING,
    )

    registry.register_document(metadata)

    updated = registry.update_approval_status(
        metadata.document_id,
        ApprovalStatus.APPROVED,
    )

    assert (
        updated.approval_status
        == ApprovalStatus.APPROVED
    )


def test_only_approved_documents_are_active(
    tmp_path: Path,
) -> None:
    pending_file = tmp_path / "pending.txt"
    approved_file = tmp_path / "approved.txt"

    pending_file.write_text(
        "Pending content.",
        encoding="utf-8",
    )

    approved_file.write_text(
        "Approved content.",
        encoding="utf-8",
    )

    registry = DocumentRegistry(
        tmp_path / "registry.json"
    )

    pending = create_test_metadata(
        pending_file,
        version="1",
        approval_status=ApprovalStatus.PENDING,
    )

    approved = create_test_metadata(
        approved_file,
        version="2",
        approval_status=ApprovalStatus.APPROVED,
    )

    registry.register_document(pending)
    registry.register_document(approved)

    active = registry.get_active_approved_documents()

    assert len(active) == 1
    assert (
        active[0].document_id
        == approved.document_id
    )


def test_new_version_can_supersede_old_version(
    tmp_path: Path,
) -> None:
    old_file = tmp_path / "old.txt"
    new_file = tmp_path / "new.txt"

    old_file.write_text(
        "Old guideline recommendation.",
        encoding="utf-8",
    )

    new_file.write_text(
        "Updated guideline recommendation.",
        encoding="utf-8",
    )

    registry = DocumentRegistry(
        tmp_path / "registry.json"
    )

    old_metadata = create_test_metadata(
        old_file,
        version="2024",
        approval_status=ApprovalStatus.APPROVED,
    )

    new_metadata = create_test_metadata(
        new_file,
        version="2026",
        approval_status=ApprovalStatus.APPROVED,
    )

    registry.register_document(old_metadata)
    registry.register_document(new_metadata)

    old_updated, new_updated = (
        registry.mark_superseded(
            old_document_id=old_metadata.document_id,
            new_document_id=new_metadata.document_id,
        )
    )

    assert (
        old_updated.approval_status
        == ApprovalStatus.SUPERSEDED
    )

    assert (
        old_updated.superseded_by_document_id
        == new_metadata.document_id
    )

    assert (
        new_updated.supersedes_document_id
        == old_metadata.document_id
    )

    active = registry.get_active_approved_documents()

    assert len(active) == 1
    assert (
        active[0].document_id
        == new_metadata.document_id
    )