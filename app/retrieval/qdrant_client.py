from collections.abc import Sequence
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config.settings import (
    Settings,
    get_settings,
)
from app.models.documents import (
    DocumentChunk,
)


class QdrantConfigurationError(RuntimeError):
    """
    Raised when required Qdrant configuration is missing.
    """


class QdrantStorageError(RuntimeError):
    """
    Raised when a Qdrant storage operation fails.
    """


PAYLOAD_INDEX_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "document_id",
    "approval_status",
    "clinical_domain",
    "document_type",
    "source_authority",
    "version",
    "ontology_concepts",
)


def create_qdrant_client(
    settings: Settings | None = None,
) -> QdrantClient:
    """
    Create a client connected to Qdrant Cloud.
    """
    settings = settings or get_settings()

    if not settings.qdrant_url:
        raise QdrantConfigurationError(
            "QDRANT_URL is not configured."
        )

    if settings.qdrant_api_key is None:
        raise QdrantConfigurationError(
            "QDRANT_API_KEY is not configured."
        )

    return QdrantClient(
        url=settings.qdrant_url,
        api_key=(
            settings
            .qdrant_api_key
            .get_secret_value()
        ),
        timeout=30,
    )


def ensure_collection(
    settings: Settings | None = None,
    client: QdrantClient | None = None,
) -> None:
    """
    Create the collection if absent.

    Existing collections are validated and receive any
    missing payload indexes.

    Existing collections are never deleted automatically.
    """
    settings = settings or get_settings()

    client = (
        client
        or create_qdrant_client(settings)
    )

    collection_name = (
        settings.qdrant_collection
    )

    try:
        exists = client.collection_exists(
            collection_name
        )

    except Exception as exc:
        raise QdrantStorageError(
            "Failed to check Qdrant collection."
        ) from exc

    if exists:
        _validate_collection(
            client=client,
            collection_name=collection_name,
            expected_dimensions=(
                settings.embedding_dimensions
            ),
        )

        _ensure_payload_indexes(
            client=client,
            collection_name=collection_name,
        )

        return

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=settings.embedding_dimensions,
                distance=models.Distance.COSINE,
            ),
        )

        _ensure_payload_indexes(
            client=client,
            collection_name=collection_name,
        )

    except Exception as exc:
        raise QdrantStorageError(
            "Failed to create Qdrant collection."
        ) from exc


def _validate_collection(
    *,
    client: QdrantClient,
    collection_name: str,
    expected_dimensions: int,
) -> None:
    """
    Validate existing collection vector configuration.
    """
    try:
        info = client.get_collection(
            collection_name
        )

    except Exception as exc:
        raise QdrantStorageError(
            "Failed to inspect Qdrant collection."
        ) from exc

    vector_config = (
        info.config.params.vectors
    )

    if isinstance(
        vector_config,
        dict,
    ):
        raise QdrantStorageError(
            (
                "Expected an unnamed dense vector "
                "configuration but found named vectors."
            )
        )

    if (
        vector_config.size
        != expected_dimensions
    ):
        raise QdrantStorageError(
            (
                "Qdrant collection dimension mismatch. "
                f"Expected {expected_dimensions}, "
                f"found {vector_config.size}."
            )
        )

    if (
        vector_config.distance
        != models.Distance.COSINE
    ):
        raise QdrantStorageError(
            (
                "Qdrant collection distance mismatch. "
                "Expected COSINE."
            )
        )


def _ensure_payload_indexes(
    *,
    client: QdrantClient,
    collection_name: str,
) -> None:
    """
    Add any missing keyword indexes.

    Existing indexes are left untouched.
    """
    try:
        info = client.get_collection(
            collection_name
        )

        existing_fields = set(
            (
                info.payload_schema
                or {}
            ).keys()
        )

        for field_name in (
            PAYLOAD_INDEX_FIELDS
        ):
            if (
                field_name
                in existing_fields
            ):
                continue

            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=(
                    models.PayloadSchemaType.KEYWORD
                ),
                wait=True,
            )

    except Exception as exc:
        raise QdrantStorageError(
            "Failed to create Qdrant payload indexes."
        ) from exc


def chunk_point_id(
    chunk_id: str,
) -> str:
    """
    Generate deterministic Qdrant UUID.
    """
    if not chunk_id.strip():
        raise ValueError(
            "chunk_id cannot be empty."
        )

    return str(
        uuid5(
            NAMESPACE_URL,
            chunk_id,
        )
    )


def chunk_to_payload(
    chunk: DocumentChunk,
) -> dict[str, Any]:
    """
    Convert a chunk into JSON-compatible Qdrant payload.

    OntologyTaggedChunk is a DocumentChunk subclass, so its
    ontology fields are automatically included by model_dump().
    """
    payload = chunk.model_dump(
        mode="json"
    )

    required_fields = (
        "chunk_id",
        "document_id",
        "text",
        "approval_status",
    )

    for field_name in required_fields:
        value = payload.get(
            field_name
        )

        if value is None:
            raise QdrantStorageError(
                (
                    "DocumentChunk payload is missing "
                    f"required field: {field_name}"
                )
            )

        if (
            isinstance(value, str)
            and not value.strip()
        ):
            raise QdrantStorageError(
                (
                    "DocumentChunk payload contains "
                    f"blank field: {field_name}"
                )
            )

    return payload


def upsert_chunks(
    *,
    chunks: Sequence[DocumentChunk],
    vectors: Sequence[Sequence[float]],
    settings: Settings | None = None,
    client: QdrantClient | None = None,
) -> int:
    """
    Store chunk vectors and metadata in Qdrant.
    """
    settings = settings or get_settings()

    client = (
        client
        or create_qdrant_client(settings)
    )

    if len(chunks) != len(vectors):
        raise QdrantStorageError(
            (
                "Chunk/vector count mismatch. "
                f"Received {len(chunks)} chunks and "
                f"{len(vectors)} vectors."
            )
        )

    if not chunks:
        return 0

    points: list[
        models.PointStruct
    ] = []

    for chunk, vector in zip(
        chunks,
        vectors,
        strict=True,
    ):
        vector_list = [
            float(value)
            for value in vector
        ]

        if (
            len(vector_list)
            != settings.embedding_dimensions
        ):
            raise QdrantStorageError(
                (
                    "Vector dimension mismatch. "
                    f"Expected "
                    f"{settings.embedding_dimensions}, "
                    f"received {len(vector_list)}."
                )
            )

        points.append(
            models.PointStruct(
                id=chunk_point_id(
                    chunk.chunk_id
                ),
                vector=vector_list,
                payload=chunk_to_payload(
                    chunk
                ),
            )
        )

    try:
        client.upsert(
            collection_name=(
                settings.qdrant_collection
            ),
            points=points,
            wait=True,
        )

    except Exception as exc:
        raise QdrantStorageError(
            "Failed to upsert chunks into Qdrant."
        ) from exc

    return len(points)


def retrieve_point_by_chunk_id(
    *,
    chunk_id: str,
    settings: Settings | None = None,
    client: QdrantClient | None = None,
):
    """
    Retrieve a point by human-readable chunk_id.
    """
    settings = settings or get_settings()

    client = (
        client
        or create_qdrant_client(settings)
    )

    if not chunk_id.strip():
        raise ValueError(
            "chunk_id cannot be empty."
        )

    query_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="chunk_id",
                match=models.MatchValue(
                    value=chunk_id,
                ),
            )
        ]
    )

    try:
        points, _ = client.scroll(
            collection_name=(
                settings.qdrant_collection
            ),
            scroll_filter=query_filter,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

    except Exception as exc:
        raise QdrantStorageError(
            "Failed to retrieve chunk from Qdrant."
        ) from exc

    if not points:
        return None

    return points[0]