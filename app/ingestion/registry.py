import json
from datetime import (
    datetime,
)
from pathlib import Path

from app.models.documents import (
    ApprovalStatus,
    DocumentMetadata,
    IndexStatus,
)


class DocumentRegistryError(
    RuntimeError
):
    """
    Raised when document-registry operations cannot be completed.
    """


class DuplicateDocumentError(
    DocumentRegistryError
):
    """
    Raised when an identical source document is already registered.
    """


class DocumentNotFoundError(
    DocumentRegistryError
):
    """
    Raised when a requested document ID does not exist.
    """


class DocumentRegistry:
    """
    Lightweight JSON-backed document registry used by the
    MedOntoAgent prototype.
    """

    def __init__(
        self,
        registry_path: str | Path,
    ) -> None:
        self.registry_path = Path(
            registry_path
        )

    def list_documents(
        self,
    ) -> list[DocumentMetadata]:
        """
        Return all registered documents.
        """
        return (
            self._load_documents()
        )

    def get_document(
        self,
        document_id: str,
    ) -> DocumentMetadata:
        """
        Return one document by ID.
        """
        for document in (
            self._load_documents()
        ):
            if (
                document.document_id
                == document_id
            ):
                return document

        raise DocumentNotFoundError(
            (
                "Document not found: "
                f"{document_id}"
            )
        )

    def find_by_checksum(
        self,
        checksum_sha256: str,
    ) -> DocumentMetadata | None:
        """
        Find an existing document with the exact checksum.
        """
        for document in (
            self._load_documents()
        ):
            if (
                document.checksum_sha256
                == checksum_sha256
            ):
                return document

        return None

    def register_document(
        self,
        metadata: DocumentMetadata,
    ) -> DocumentMetadata:
        """
        Register a new source document.

        Exact byte-for-byte duplicates are rejected.
        """
        documents = (
            self._load_documents()
        )

        for existing in documents:
            if (
                existing.checksum_sha256
                == metadata.checksum_sha256
            ):
                raise DuplicateDocumentError(
                    (
                        "An identical document is "
                        "already registered as "
                        f"{existing.document_id}."
                    )
                )

            if (
                existing.document_id
                == metadata.document_id
            ):
                raise DuplicateDocumentError(
                    (
                        "Document ID already exists: "
                        f"{metadata.document_id}"
                    )
                )

        documents.append(
            metadata
        )

        self._save_documents(
            documents
        )

        return metadata

    def update_approval_status(
        self,
        document_id: str,
        approval_status: ApprovalStatus,
    ) -> DocumentMetadata:
        """
        Change the approval state of an existing document.
        """
        documents = (
            self._load_documents()
        )

        for index, document in enumerate(
            documents
        ):
            if (
                document.document_id
                != document_id
            ):
                continue

            updated = (
                document.model_copy(
                    update={
                        "approval_status": (
                            approval_status
                        ),
                    }
                )
            )

            documents[
                index
            ] = updated

            self._save_documents(
                documents
            )

            return updated

        raise DocumentNotFoundError(
            (
                "Document not found: "
                f"{document_id}"
            )
        )

    def update_index_status(
        self,
        document_id: str,
        *,
        index_status: IndexStatus,
        indexed_at: (
            datetime | None
        ) = None,
        index_error: (
            str | None
        ) = None,
    ) -> DocumentMetadata:
        """
        Update vector-index lifecycle state for one
        registered document.
        """
        documents = (
            self._load_documents()
        )

        for index, document in enumerate(
            documents
        ):
            if (
                document.document_id
                != document_id
            ):
                continue

            updated = (
                document.model_copy(
                    update={
                        "index_status": (
                            index_status
                        ),
                        "indexed_at": (
                            indexed_at
                        ),
                        "index_error": (
                            index_error
                        ),
                    }
                )
            )

            documents[
                index
            ] = updated

            self._save_documents(
                documents
            )

            return updated

        raise DocumentNotFoundError(
            (
                "Document not found: "
                f"{document_id}"
            )
        )

    def mark_superseded(
        self,
        *,
        old_document_id: str,
        new_document_id: str,
    ) -> tuple[
        DocumentMetadata,
        DocumentMetadata,
    ]:
        """
        Link a newer document to the document it supersedes.

        The old document becomes SUPERSEDED.
        The newer document remains in its existing approval state.
        """
        if (
            old_document_id
            == new_document_id
        ):
            raise DocumentRegistryError(
                (
                    "A document cannot "
                    "supersede itself."
                )
            )

        documents = (
            self._load_documents()
        )

        old_index: (
            int | None
        ) = None

        new_index: (
            int | None
        ) = None

        for index, document in enumerate(
            documents
        ):
            if (
                document.document_id
                == old_document_id
            ):
                old_index = index

            if (
                document.document_id
                == new_document_id
            ):
                new_index = index

        if old_index is None:
            raise DocumentNotFoundError(
                (
                    "Old document not found: "
                    f"{old_document_id}"
                )
            )

        if new_index is None:
            raise DocumentNotFoundError(
                (
                    "New document not found: "
                    f"{new_document_id}"
                )
            )

        old_document = documents[
            old_index
        ]

        new_document = documents[
            new_index
        ]

        updated_old = (
            old_document.model_copy(
                update={
                    "approval_status": (
                        ApprovalStatus
                        .SUPERSEDED
                    ),
                    "superseded_by_document_id": (
                        new_document_id
                    ),
                }
            )
        )

        updated_new = (
            new_document.model_copy(
                update={
                    "supersedes_document_id": (
                        old_document_id
                    ),
                }
            )
        )

        documents[
            old_index
        ] = updated_old

        documents[
            new_index
        ] = updated_new

        self._save_documents(
            documents
        )

        return (
            updated_old,
            updated_new,
        )

    def get_active_approved_documents(
        self,
    ) -> list[DocumentMetadata]:
        """
        Return only documents eligible for the active
        knowledge corpus.
        """
        return [
            document
            for document
            in self._load_documents()
            if (
                document.approval_status
                == ApprovalStatus.APPROVED
            )
        ]

    def _load_documents(
        self,
    ) -> list[DocumentMetadata]:
        """
        Load and validate registry content from disk.
        """
        if not (
            self.registry_path.exists()
        ):
            return []

        try:
            raw_data = json.loads(
                self.registry_path
                .read_text(
                    encoding="utf-8",
                )
            )

        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise DocumentRegistryError(
                (
                    "Could not read document registry: "
                    f"{self.registry_path}"
                )
            ) from exc

        if not isinstance(
            raw_data,
            list,
        ):
            raise DocumentRegistryError(
                (
                    "Document registry must "
                    "contain a JSON list."
                )
            )

        try:
            return [
                DocumentMetadata
                .model_validate(
                    item
                )
                for item
                in raw_data
            ]

        except ValueError as exc:
            raise DocumentRegistryError(
                (
                    "Document registry contains "
                    "invalid metadata."
                )
            ) from exc

    def _save_documents(
        self,
        documents: list[
            DocumentMetadata
        ],
    ) -> None:
        """
        Save the complete validated registry atomically enough
        for prototype/local use.
        """
        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        serialized = [
            document.model_dump(
                mode="json",
            )
            for document
            in documents
        ]

        temporary_path = (
            self.registry_path
            .with_suffix(
                self.registry_path.suffix
                + ".tmp"
            )
        )

        temporary_path.write_text(
            json.dumps(
                serialized,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(
            self.registry_path
        )