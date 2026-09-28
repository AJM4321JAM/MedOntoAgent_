from rdflib import Graph, URIRef

from app.models.constraints import (
    ConstraintType,
    OntologyConstraint,
    OntologyConstraintSet,
    RetrievalHints,
)
from app.models.ontology import (
    OntologyContext,
)
from app.ontology.ontology_queries import (
    find_relationships_between_concepts,
    get_provenance,
    ontology_concept_name,
    ontology_predicate_name,
)


RELATIONSHIP_TYPES: dict[
    str,
    ConstraintType,
] = {
    "contraindicatedIn": (
        "CONTRAINDICATION"
    ),
    "notRecommendedFor": (
        "NOT_RECOMMENDED"
    ),
    "recommendedFor": (
        "RECOMMENDATION"
    ),
    "treatedBy": (
        "TREATMENT_RELATION"
    ),
    "requiresInvestigation": (
        "INVESTIGATION_REQUIREMENT"
    ),
}


def classify_relationship(
    predicate_name: str,
) -> ConstraintType:
    """
    Convert a controlled ontology relationship into the
    application's semantic constraint category.
    """
    return RELATIONSHIP_TYPES.get(
        predicate_name,
        "OTHER",
    )


def _relationship_term(
    *,
    subject_label: str,
    predicate_label: str,
    object_label: str,
) -> str:
    """
    Produce human-readable ontology context suitable for
    retrieval query expansion.
    """
    return (
        f"{subject_label} "
        f"{predicate_label} "
        f"{object_label}"
    )


def _deduplicate_preserving_order(
    values: list[str],
) -> list[str]:
    """
    Remove duplicate strings while preserving order.
    """
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        if value in seen:
            continue

        result.append(
            value
        )

        seen.add(
            value
        )

    return result


def build_constraints(
    *,
    query: str,
    ontology_context: OntologyContext,
    graph: Graph,
) -> OntologyConstraintSet:
    """
    Discover explicit relationships among concepts mapped
    from the current query and translate them into structured
    constraints plus retrieval hints.

    No relationship is inferred or invented.
    """
    normalized_query = query.strip()

    if not normalized_query:
        raise ValueError(
            "Constraint query cannot be empty."
        )

    concept_uris = [
        concept.uri
        for concept
        in ontology_context.mapped_concepts
    ]

    relations = (
        find_relationships_between_concepts(
            graph,
            concept_uris=concept_uris,
        )
    )

    constraints: list[
        OntologyConstraint
    ] = []

    relationship_terms: list[
        str
    ] = []

    for relation in relations:
        predicate_name = (
            ontology_predicate_name(
                URIRef(
                    relation.predicate_uri
                )
            )
        )

        subject_name = (
            ontology_concept_name(
                URIRef(
                    relation.subject_uri
                )
            )
        )

        object_name = (
            ontology_concept_name(
                URIRef(
                    relation.object_uri
                )
            )
        )

        provenance = get_provenance(
            graph,
            subject_name=subject_name,
        )

        constraint = (
            OntologyConstraint(
                constraint_type=(
                    classify_relationship(
                        predicate_name
                    )
                ),
                subject_name=subject_name,
                subject_label=(
                    relation.subject_label
                ),
                subject_uri=(
                    relation.subject_uri
                ),
                predicate_name=(
                    predicate_name
                ),
                predicate_label=(
                    relation.predicate_label
                ),
                predicate_uri=(
                    relation.predicate_uri
                ),
                object_name=object_name,
                object_label=(
                    relation.object_label
                ),
                object_uri=(
                    relation.object_uri
                ),
                source_document_id=(
                    provenance[
                        "source_document_id"
                    ]
                ),
                source_section=(
                    provenance[
                        "source_section"
                    ]
                ),
                source_version=(
                    provenance[
                        "source_version"
                    ]
                ),
                assertion_status=(
                    provenance[
                        "assertion_status"
                    ]
                ),
            )
        )

        constraints.append(
            constraint
        )

        relationship_terms.append(
            _relationship_term(
                subject_label=(
                    relation.subject_label
                ),
                predicate_label=(
                    relation.predicate_label
                ),
                object_label=(
                    relation.object_label
                ),
            )
        )

    concept_names = (
        _deduplicate_preserving_order(
            [
                concept.concept_name
                for concept
                in ontology_context.mapped_concepts
            ]
        )
    )

    concept_labels = (
        _deduplicate_preserving_order(
            [
                concept.label
                for concept
                in ontology_context.mapped_concepts
            ]
        )
    )

    relationship_terms = (
        _deduplicate_preserving_order(
            relationship_terms
        )
    )

    if relationship_terms:
        expanded_query = (
            normalized_query
            + "\n\nOntology relationship context:\n"
            + "\n".join(
                relationship_terms
            )
        )

    else:
        expanded_query = (
            normalized_query
        )

    retrieval_hints = RetrievalHints(
        concept_names=concept_names,
        concept_labels=concept_labels,
        relationship_terms=(
            relationship_terms
        ),
        expanded_query=expanded_query,
    )

    return OntologyConstraintSet(
        query=normalized_query,
        constraints=constraints,
        retrieval_hints=(
            retrieval_hints
        ),
        constraint_count=len(
            constraints
        ),
    )