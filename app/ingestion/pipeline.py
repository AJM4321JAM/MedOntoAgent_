from datetime import (
    date,
    datetime,
    timezone,
)
from pathlib import Path

from app.config.settings import (
    Settings,
    get_settings,
)
from app.ingestion.chunker import (
    chunk_document,
)
from app.ingestion.cleaner import (
    clean_document,
)
from app.ingestion.loader import (
    load_document,
)
from app.ingestion.metadata import (
    calculate_sha256,
    create_document_metadata,
)
from app.ingestion.registry import (
    DEFAULT_REGISTRY_PATH,
    DocumentRegistry,
)
from app.models.documents import (
    ApprovalStatus,
    IndexStatus,
    IngestionResult,
    IngestionStatus,
    SourceAuthority,
)
from app.ontology.ontology_loader import (
    load_ontology,
)
from app.ontology.tagger import (
    tag_chunks_with_ontology,
)
from app.retrieval.embeddings import (
    embed_texts,
)
from app.retrieval.qdrant_client import (
    ensure_collection,
    upsert_chunks,
)


class DocumentIngestionError(
    RuntimeError
):
    """
    Raised when processing or indexing a registered document
    cannot be completed safely.
    """


def ingest_document(
    file_path: str | Path,
    *,
    title: str,
    publisher: str,
    document_type: str,
    clinical_domain: str,
    source_authority: SourceAuthority,
    version: str,
    approval_status: ApprovalStatus = (
        ApprovalStatus.PENDING
    ),
    publication_date: date | None = None,
    effective_date: date | None = None,
    source_url: str | None = None,
    supersedes_document_id: str | None = None,
    registry_path: str | Path = (
        DEFAULT_REGISTRY_PATH
    ),
    settings: Settings | None = None,
) -> IngestionResult:
    """
    Run one governed source document through the complete
    MedOntoAgent ingestion pipeline.

    Registration, approval, and indexing are deliberately
    represented as separate states.
    """
    settings = (
        settings
        or get_settings()
    )

    path = Path(
        file_path
    )

    try:
        checksum = (
            calculate_sha256(
                path
            )
        )

    except Exception as exc:
        raise DocumentIngestionError(
            (
                "Source document could not "
                "be validated."
            )
        ) from exc

    registry = DocumentRegistry(
        registry_path
    )

    try:
        existing = (
            registry.find_by_checksum(
                checksum
            )
        )

    except Exception as exc:
        raise DocumentIngestionError(
            (
                "Document registry could not "
                "be inspected."
            )
        ) from exc

    if (
        existing is not None
        and existing.index_status
        == IndexStatus.INDEXED
    ):
        return IngestionResult(
            document_id=(
                existing.document_id
            ),
            filename=(
                existing.filename
            ),
            status=(
                IngestionStatus
                .DUPLICATE
            ),
            checksum_sha256=(
                checksum
            ),
            chunk_count=0,
            vector_count=0,
            ontology_tagged_chunk_count=0,
            message=(
                "An identical source file is "
                "already registered and indexed."
            ),
        )

    if existing is None:
        try:
            metadata = (
                create_document_metadata(
                    file_path=path,
                    title=title,
                    publisher=publisher,
                    document_type=(
                        document_type
                    ),
                    clinical_domain=(
                        clinical_domain
                    ),
                    source_authority=(
                        source_authority
                    ),
                    version=version,
                    publication_date=(
                        publication_date
                    ),
                    effective_date=(
                        effective_date
                    ),
                    approval_status=(
                        approval_status
                    ),
                    source_url=(
                        source_url
                    ),
                    supersedes_document_id=(
                        supersedes_document_id
                    ),
                )
            )

            registry.register_document(
                metadata
            )

        except Exception as exc:
            raise DocumentIngestionError(
                (
                    "Document could not "
                    "be registered."
                )
            ) from exc

    else:
        # Registry governance is authoritative on retry.
        metadata = existing

    if (
        metadata.approval_status
        != ApprovalStatus.APPROVED
    ):
        return IngestionResult(
            document_id=(
                metadata.document_id
            ),
            filename=(
                metadata.filename
            ),
            status=(
                IngestionStatus
                .NOT_APPROVED
            ),
            checksum_sha256=(
                metadata.checksum_sha256
            ),
            chunk_count=0,
            vector_count=0,
            ontology_tagged_chunk_count=0,
            message=(
                "Document is registered but is "
                "not APPROVED for indexing."
            ),
        )

    try:
        registry.update_index_status(
            metadata.document_id,
            index_status=(
                IndexStatus.INDEXING
            ),
            indexed_at=None,
            index_error=None,
        )

    except Exception as exc:
        raise DocumentIngestionError(
            (
                "Document index state could "
                "not be updated."
            )
        ) from exc

    try:
        loaded = load_document(
            path
        )

        cleaned = clean_document(
            loaded
        )

        chunks = chunk_document(
            cleaned,
            metadata,
            max_chunk_words=(
                settings.max_chunk_words
            ),
            overlap_words=(
                settings
                .chunk_overlap_words
            ),
        )

        ontology_graph = (
            load_ontology()
        )

        tagged_chunks = (
            tag_chunks_with_ontology(
                chunks=chunks,
                graph=(
                    ontology_graph
                ),
            )
        )

        vectors = embed_texts(
            [
                chunk.text
                for chunk
                in tagged_chunks
            ],
            settings=settings,
        )

        ensure_collection(
            settings=settings
        )

        stored_count = (
            upsert_chunks(
                chunks=(
                    tagged_chunks
                ),
                vectors=vectors,
                settings=settings,
            )
        )

        if (
            stored_count
            != len(
                tagged_chunks
            )
        ):
            raise DocumentIngestionError(
                (
                    "Vector storage count did not "
                    "match the processed chunk count."
                )
            )

        registry.update_index_status(
            metadata.document_id,
            index_status=(
                IndexStatus.INDEXED
            ),
            indexed_at=(
                datetime.now(
                    timezone.utc
                )
            ),
            index_error=None,
        )

        ontology_tagged_count = sum(
            1
            for chunk
            in tagged_chunks
            if chunk.ontology_concepts
        )

        return IngestionResult(
            document_id=(
                metadata.document_id
            ),
            filename=(
                metadata.filename
            ),
            status=(
                IngestionStatus
                .INGESTED
            ),
            checksum_sha256=(
                metadata.checksum_sha256
            ),
            chunk_count=len(
                tagged_chunks
            ),
            vector_count=(
                stored_count
            ),
            ontology_tagged_chunk_count=(
                ontology_tagged_count
            ),
            message=(
                "Approved document was successfully "
                "processed and indexed."
            ),
        )

    except Exception as exc:
        safe_error = (
            "Document processing or indexing failed."
        )

        try:
            registry.update_index_status(
                metadata.document_id,
                index_status=(
                    IndexStatus.FAILED
                ),
                indexed_at=None,
                index_error=(
                    safe_error
                ),
            )

        except Exception:
            pass

        if isinstance(
            exc,
            DocumentIngestionError,
        ):
            raise

        raise DocumentIngestionError(
            safe_error
        ) from exc