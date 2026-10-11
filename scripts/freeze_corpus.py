import hashlib
import json
from collections import Counter
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path

from rdflib import RDF

from app.config.settings import (
    get_settings,
)
from app.ingestion.registry import (
    DEFAULT_REGISTRY_PATH,
    DocumentRegistry,
)
from app.models.documents import (
    ApprovalStatus,
    IndexStatus,
)
from app.models.evaluation import (
    CorpusFreezeManifest,
    FrozenCorpusDocument,
)
from app.ontology.ontology_loader import (
    load_ontology,
)
from app.ontology.ontology_queries import (
    MED,
)
from app.retrieval.qdrant_client import (
    create_qdrant_client,
)


CORPUS_VERSION = "clinical-corpus-v1"

CORPUS_DOCUMENT_IDS = [
    "DOC-e943ba2bdc31",
    "DOC-21a51be8081f",
    "DOC-dad82839ba53",
    "DOC-d21df49f3ed9",
]

ONTOLOGY_PATH = Path(
    "ontology/medical_ontology.ttl"
)

OUTPUT_PATH = Path(
    "evals/corpus_manifest_v1.json"
)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def qdrant_document_counts(
    *,
    client,
    collection_name: str,
) -> Counter[str]:
    counts: Counter[str] = Counter()

    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}

            document_id = str(
                payload.get("document_id")
                or ""
            )

            counts[document_id] += 1

        if next_offset is None:
            break

        offset = next_offset

    return counts


def main() -> None:
    settings = get_settings()

    registry = DocumentRegistry(
        DEFAULT_REGISTRY_PATH
    )

    client = create_qdrant_client(
        settings
    )

    counts = qdrant_document_counts(
        client=client,
        collection_name=(
            settings.qdrant_collection
        ),
    )

    actual_ids = set(counts)
    expected_ids = set(
        CORPUS_DOCUMENT_IDS
    )

    if actual_ids != expected_ids:
        raise RuntimeError(
            (
                "Clinical collection document IDs "
                "do not exactly match frozen corpus. "
                f"Expected {sorted(expected_ids)}, "
                f"got {sorted(actual_ids)}."
            )
        )

    frozen_documents = []

    for document_id in (
        CORPUS_DOCUMENT_IDS
    ):
        document = registry.get_document(
            document_id
        )

        if document is None:
            raise RuntimeError(
                (
                    "Frozen document missing from registry: "
                    f"{document_id}"
                )
            )

        if (
            document.approval_status
            != ApprovalStatus.APPROVED
        ):
            raise RuntimeError(
                (
                    "Frozen document is not APPROVED: "
                    f"{document_id}"
                )
            )

        if (
            document.superseded_by_document_id
            is not None
        ):
            raise RuntimeError(
                (
                    "Frozen document has been superseded: "
                    f"{document_id}"
                )
            )

        if (
            document.index_status
            != IndexStatus.INDEXED
        ):
            raise RuntimeError(
                (
                    "Frozen document is not INDEXED: "
                    f"{document_id}"
                )
            )

        if not document.checksum_sha256:
            raise RuntimeError(
                (
                    "Frozen document has no checksum: "
                    f"{document_id}"
                )
            )

        chunk_count = counts[
            document_id
        ]

        if chunk_count < 1:
            raise RuntimeError(
                (
                    "Frozen document has no Qdrant chunks: "
                    f"{document_id}"
                )
            )

        frozen_documents.append(
            FrozenCorpusDocument(
                document_id=(
                    document.document_id
                ),
                title=document.title,
                publisher=document.publisher,
                version=document.version,
                checksum_sha256=(
                    document.checksum_sha256
                ),
                approval_status=(
                    document.approval_status.value
                ),
                superseded_by_document_id=(
                    document
                    .superseded_by_document_id
                ),
                chunk_count=chunk_count,
            )
        )

    graph = load_ontology()

    ontology_assertions = list(
        graph.subjects(
            RDF.type,
            MED.OntologyAssertion,
        )
    )

    active_assertions = [
        assertion
        for assertion in ontology_assertions
        if str(
            graph.value(
                assertion,
                MED.assertionStatus,
            )
            or ""
        )
        == "ACTIVE"
    ]

    manifest = CorpusFreezeManifest(
        corpus_version=CORPUS_VERSION,
        created_at=datetime.now(
            timezone.utc
        ),
        qdrant_collection=(
            settings.qdrant_collection
        ),
        qdrant_point_count=sum(
            counts.values()
        ),
        embedding_model=(
            settings.embedding_model
        ),
        embedding_dimensions=(
            settings.embedding_dimensions
        ),
        max_chunk_words=(
            settings.max_chunk_words
        ),
        chunk_overlap_words=(
            settings.chunk_overlap_words
        ),
        retrieval_top_k=(
            settings.retrieval_top_k
        ),
        openrouter_model=(
            settings.openrouter_model
        ),
        max_agent_retries=(
            settings.max_agent_retries
        ),
        ontology_path=(
            ONTOLOGY_PATH.as_posix()
        ),
        ontology_checksum_sha256=(
            sha256_file(
                ONTOLOGY_PATH
            )
        ),
        ontology_assertion_count=len(
            ontology_assertions
        ),
        ontology_active_assertion_count=len(
            active_assertions
        ),
        documents=frozen_documents,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            manifest.model_dump(
                mode="json"
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "Frozen corpus manifest:",
        OUTPUT_PATH,
    )

    print(
        "CORPUS_VERSION:",
        manifest.corpus_version,
    )

    print(
        "DOCUMENTS:",
        len(manifest.documents),
    )

    print(
        "QDRANT_POINTS:",
        manifest.qdrant_point_count,
    )

    print(
        "ONTOLOGY_ASSERTIONS:",
        manifest.ontology_assertion_count,
    )

    print(
        "ACTIVE_ASSERTIONS:",
        manifest.ontology_active_assertion_count,
    )


if __name__ == "__main__":
    main()
