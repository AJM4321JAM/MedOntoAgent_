from typing import Any

from qdrant_client import QdrantClient

from app.config.settings import Settings, get_settings
from app.models.retrieval import (
    RetrievedChunk,
    RetrievalResult,
)
from app.retrieval.embeddings import embed_text
from app.retrieval.filters import (
    approved_document_filter,
)
from app.retrieval.qdrant_client import (
    create_qdrant_client,
)


class RetrievalError(RuntimeError):
    """
    Raised when vector retrieval cannot be completed safely.
    """


def _require_payload_value(
    payload: dict[str, Any],
    key: str,
) -> Any:
    """
    Return one required Qdrant payload field.

    Missing or blank required values are treated as invalid
    retrieval data rather than silently ignored.
    """

    value = payload.get(
        key
    )

    if value is None:
        raise RetrievalError(
            f"Retrieved point is missing required payload field: {key}"
        )

    if (
        isinstance(value, str)
        and not value.strip()
    ):
        raise RetrievalError(
            f"Retrieved point has blank required payload field: {key}"
        )

    return value


def _convert_scored_point(
    point: Any,
) -> RetrievedChunk:
    """
    Convert one Qdrant scored point into the application
    retrieval model.
    """

    payload = point.payload

    if not isinstance(
        payload,
        dict,
    ):
        raise RetrievalError(
            "Retrieved Qdrant point has no valid payload."
        )

    score = point.score

    if score is None:
        raise RetrievalError(
            "Retrieved Qdrant point has no similarity score."
        )

    try:
        return RetrievedChunk(
            chunk_id=str(
                _require_payload_value(
                    payload,
                    "chunk_id",
                )
            ),
            document_id=str(
                _require_payload_value(
                    payload,
                    "document_id",
                )
            ),
            title=str(
                _require_payload_value(
                    payload,
                    "title",
                )
            ),
            section_title=payload.get(
                "section_title"
            ),
            page_number=payload.get(
                "page_number"
            ),
            text=str(
                _require_payload_value(
                    payload,
                    "text",
                )
            ),
            score=float(
                score
            ),
            version=str(
                _require_payload_value(
                    payload,
                    "version",
                )
            ),
            publisher=str(
                _require_payload_value(
                    payload,
                    "publisher",
                )
            ),
            source_authority=(
                _require_payload_value(
                    payload,
                    "source_authority",
                )
            ),
            source_url=payload.get(
                "source_url"
            ),
        )

    except RetrievalError:
        raise

    except Exception as exc:
        raise RetrievalError(
            "Retrieved Qdrant payload could not be converted "
            "into RetrievedChunk."
        ) from exc


def retrieve_baseline(
    query: str,
    *,
    settings: Settings | None = None,
    client: QdrantClient | None = None,
) -> RetrievalResult:
    """
    Retrieve the most semantically similar APPROVED chunks.

    This is the conventional vector-RAG baseline.

    It deliberately does not use:
    - ontology constraints
    - query rewriting
    - reranking
    - symbolic reasoning
    """

    settings = (
        settings
        or get_settings()
    )

    normalized_query = (
        query.strip()
    )

    if not normalized_query:
        raise ValueError(
            "Retrieval query cannot be empty."
        )

    qdrant_client = (
        client
        or create_qdrant_client(
            settings
        )
    )

    try:
        query_vector = embed_text(
            normalized_query,
            settings,
        )

    except Exception as exc:
        raise RetrievalError(
            "Failed to generate query embedding."
        ) from exc

    if (
        len(query_vector)
        != settings.embedding_dimensions
    ):
        raise RetrievalError(
            (
                "Query embedding dimension mismatch. "
                f"Expected {settings.embedding_dimensions}, "
                f"received {len(query_vector)}."
            )
        )

    try:
        response = (
            qdrant_client
            .query_points(
                collection_name=(
                    settings.qdrant_collection
                ),
                query=query_vector,
                query_filter=(
                    approved_document_filter()
                ),
                limit=(
                    settings.retrieval_top_k
                ),
                with_payload=True,
                with_vectors=False,
            )
        )

    except Exception as exc:
        raise RetrievalError(
            "Qdrant baseline retrieval failed."
        ) from exc

    retrieved_chunks: list[
        RetrievedChunk
    ] = []

    try:
        for point in response.points:
            retrieved_chunks.append(
                _convert_scored_point(
                    point
                )
            )

    except RetrievalError:
        raise

    except Exception as exc:
        raise RetrievalError(
            "Failed to process Qdrant retrieval results."
        ) from exc

    return RetrievalResult(
        query=normalized_query,
        top_k=(
            settings.retrieval_top_k
        ),
        retrieved_chunks=(
            retrieved_chunks
        ),
        result_count=len(
            retrieved_chunks
        ),
    )