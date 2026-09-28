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
    create_document_metadata,
)
from app.models.documents import (
    ApprovalStatus,
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


DOCUMENTS = [
    (
        (
            "data/sample_documents/"
            "synthetic_general_treatment.txt"
        ),
        "Synthetic General Treatment Document",
    ),
    (
        (
            "data/sample_documents/"
            "synthetic_pregnancy_constraint.txt"
        ),
        "Synthetic Pregnancy Constraint Document",
    ),
]


graph = load_ontology()

ensure_collection()


for file_path, title in DOCUMENTS:
    loaded = load_document(
        file_path
    )

    cleaned = clean_document(
        loaded
    )

    metadata = create_document_metadata(
        file_path=file_path,
        title=title,
        publisher="MedOntoAgent",
        document_type=(
            "Synthetic Test Document"
        ),
        clinical_domain=(
            "Synthetic Evaluation"
        ),
        source_authority=(
            SourceAuthority.OTHER
        ),
        version="1.0",
        approval_status=(
            ApprovalStatus.APPROVED
        ),
    )

    chunks = chunk_document(
        cleaned,
        metadata,
    )

    tagged_chunks = (
        tag_chunks_with_ontology(
            chunks=chunks,
            graph=graph,
        )
    )

    vectors = embed_texts(
        [
            chunk.text
            for chunk
            in tagged_chunks
        ]
    )

    stored = upsert_chunks(
        chunks=tagged_chunks,
        vectors=vectors,
    )

    print(
        f"{title} stored: {stored}"
    )

    for chunk in tagged_chunks:
        print(
            " ",
            chunk.chunk_id,
        )

        print(
            " ",
            chunk.ontology_concepts,
        )