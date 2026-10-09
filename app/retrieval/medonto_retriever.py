from qdrant_client import QdrantClient

from app.config.settings import (
    Settings,
    get_settings,
)
from app.models.medonto import (
    MedOntoRetrievalResult,
)
from app.models.ontology import (
    OntologyContext,
)
from app.models.constraints import (
    OntologyConstraintSet,
)
from app.models.retrieval import (
    RetrievedChunk,
    RetrievalResult,
)
from app.ontology.concept_mapper import (
    build_ontology_context,
)
from app.ontology.constraints import (
    build_constraints,
)
from app.ontology.ontology_loader import (
    load_ontology,
)
from app.retrieval.embeddings import (
    embed_text,
)
from app.retrieval.filters import (
    approved_document_filter,
    ontology_concept_filter,
)
from app.retrieval.qdrant_client import (
    create_qdrant_client,
)
from app.retrieval.retriever import (
    RetrievalError,
)
from app.observability.tracing import (
    observed_span,
)


def _convert_points(
    points,
) -> list[RetrievedChunk]:
    """
    Convert Qdrant scored points into the existing
    RetrievedChunk model.
    """
    results: list[
        RetrievedChunk
    ] = []

    for point in points:
        payload = dict(
            point.payload
            or {}
        )

        payload["score"] = float(
            point.score
        )

        results.append(
            RetrievedChunk.model_validate(
                payload
            )
        )

    return results


def _constraint_concept_names(
    constraint_set: OntologyConstraintSet,
) -> list[str]:
    """
    Return only concepts participating in explicit ontology
    constraints.

    Merely mapped concepts do not become strict filters.
    """
    names: list[str] = []

    for constraint in (
        constraint_set.constraints
    ):
        names.extend(
            [
                constraint.subject_name,
                constraint.object_name,
            ]
        )

    return list(
        dict.fromkeys(
            names
        )
    )


def _query_qdrant(
    *,
    client: QdrantClient,
    settings: Settings,
    vector: list[float],
    query_filter,
):
    """
    Execute one vector search.
    """
    response = client.query_points(
        collection_name=(
            settings.qdrant_collection
        ),
        query=vector,
        query_filter=query_filter,
        limit=settings.retrieval_top_k,
        with_payload=True,
        with_vectors=False,
    )

    return response.points


def retrieve_with_ontology_state(
    *,
    query: str,
    ontology_context: OntologyContext,
    constraint_set: OntologyConstraintSet,
    settings: Settings | None = None,
    client: QdrantClient | None = None,
) -> MedOntoRetrievalResult:
    """
    Perform ontology-aware retrieval when concept mapping and
    constraint construction have already happened.

    This is the retrieval entry point used by LangGraph.
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
            "MedOnto retrieval query cannot be empty."
        )

    retrieval_query = (
        constraint_set
        .retrieval_hints
        .expanded_query
    )

    try:
        vector = embed_text(
            retrieval_query,
            settings=settings,
        )

        qdrant = (
            client
            or create_qdrant_client(
                settings
            )
        )

        filter_concepts = (
            _constraint_concept_names(
                constraint_set
            )
        )

        used_ontology_filter = bool(
            filter_concepts
        )

        retrieval_relaxed = False

        if used_ontology_filter:
            with observed_span(
                "qdrant.query",
                attributes={
                    "medonto.retrieval.top_k": (
                        settings.retrieval_top_k
                    ),
                    "medonto.retrieval.mode": (
                        "ontology"
                    ),
                    "medonto.retrieval.ontology_filter": True,
                },
            ):
                points = _query_qdrant(
                    client=qdrant,
                    settings=settings,
                    vector=vector,
                    query_filter=(
                        ontology_concept_filter(
                            filter_concepts
                        )
                    ),
                )

            if not points:
                retrieval_relaxed = True

                with observed_span(
                    "qdrant.query",
                    attributes={
                        "medonto.retrieval.top_k": (
                            settings.retrieval_top_k
                        ),
                        "medonto.retrieval.mode": (
                            "ontology_relaxed"
                        ),
                        "medonto.retrieval.ontology_filter": False,
                    },
                ):
                    points = _query_qdrant(
                        client=qdrant,
                        settings=settings,
                        vector=vector,
                        query_filter=(
                            approved_document_filter()
                        ),
                    )

        else:
            with observed_span(
                "qdrant.query",
                attributes={
                    "medonto.retrieval.top_k": (
                        settings.retrieval_top_k
                    ),
                    "medonto.retrieval.mode": (
                        "ontology"
                    ),
                    "medonto.retrieval.ontology_filter": False,
                },
            ):
                points = _query_qdrant(
                    client=qdrant,
                    settings=settings,
                    vector=vector,
                    query_filter=(
                        approved_document_filter()
                    ),
                )

    except Exception as exc:
        if isinstance(
            exc,
            RetrievalError,
        ):
            raise

        raise RetrievalError(
            "Ontology-aware retrieval failed."
        ) from exc

    retrieved_chunks = (
        _convert_points(
            points
        )
    )

    retrieval = RetrievalResult(
        query=retrieval_query,
        top_k=settings.retrieval_top_k,
        retrieved_chunks=(
            retrieved_chunks
        ),
        result_count=len(
            retrieved_chunks
        ),
    )

    return MedOntoRetrievalResult(
        query=normalized_query,
        ontology_context=(
            ontology_context
        ),
        constraint_set=(
            constraint_set
        ),
        retrieval=retrieval,
        used_ontology_filter=(
            used_ontology_filter
        ),
        retrieval_relaxed=(
            retrieval_relaxed
        ),
    )


def retrieve_medonto(
    query: str,
    *,
    settings: Settings | None = None,
    client: QdrantClient | None = None,
) -> MedOntoRetrievalResult:
    """
    Convenience wrapper for the complete standalone
    ontology-aware retrieval path.
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
            "MedOnto retrieval query cannot be empty."
        )

    graph = load_ontology()

    ontology_context = (
        build_ontology_context(
            normalized_query,
            graph=graph,
            settings=settings,
        )
    )

    constraint_set = (
        build_constraints(
            query=normalized_query,
            ontology_context=(
                ontology_context
            ),
            graph=graph,
        )
    )

    return retrieve_with_ontology_state(
        query=normalized_query,
        ontology_context=(
            ontology_context
        ),
        constraint_set=(
            constraint_set
        ),
        settings=settings,
        client=client,
    )