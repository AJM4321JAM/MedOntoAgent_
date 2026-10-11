from dataclasses import dataclass

from rdflib import (
    Graph,
    Literal,
    Namespace,
    RDF,
    RDFS,
    URIRef,
)


MED = Namespace(
    "https://medontoagent.example.org/ontology/"
)


@dataclass(
    frozen=True
)
class OntologyRelation:
    """
    One resolved ontology object-property relationship.
    """

    subject_uri: str
    subject_label: str

    predicate_uri: str
    predicate_label: str

    object_uri: str
    object_label: str


def _concept_uri(
    concept_name: str,
) -> URIRef:
    """
    Convert a controlled concept name into its ontology URI.
    """
    normalized = concept_name.strip()

    if not normalized:
        raise ValueError(
            "Ontology concept name cannot be empty."
        )

    return MED[
        normalized
    ]


def _predicate_uri(
    predicate_name: str,
) -> URIRef:
    """
    Convert a controlled predicate name into its ontology URI.
    """
    normalized = predicate_name.strip()

    if not normalized:
        raise ValueError(
            "Ontology predicate name cannot be empty."
        )

    return MED[
        normalized
    ]


def _label_for(
    graph: Graph,
    uri: URIRef,
) -> str:
    """
    Return rdfs:label when present.

    Fall back to the URI local identifier.
    """
    label = graph.value(
        uri,
        RDFS.label,
    )

    if isinstance(
        label,
        Literal,
    ):
        return str(
            label
        )

    return ontology_concept_name(
        uri
    )


def concept_exists(
    graph: Graph,
    concept_name: str,
) -> bool:
    """
    Return True when the named ontology resource exists.
    """
    concept_uri = _concept_uri(
        concept_name
    )

    return any(
        True
        for _ in graph.triples(
            (
                concept_uri,
                None,
                None,
            )
        )
    )


def relationship_exists(
    graph: Graph,
    *,
    subject_name: str,
    predicate_name: str,
    object_name: str,
) -> bool:
    """
    Return True when the exact RDF relationship exists.
    """
    subject_uri = _concept_uri(
        subject_name
    )

    predicate_uri = _predicate_uri(
        predicate_name
    )

    object_uri = _concept_uri(
        object_name
    )

    return (
        subject_uri,
        predicate_uri,
        object_uri,
    ) in graph


def get_relationships_for_subject(
    graph: Graph,
    *,
    subject_name: str,
) -> list[OntologyRelation]:
    """
    Return explicit URI-to-URI relationships for one subject.

    RDF type declarations, labels, and literal provenance
    properties are excluded.
    """
    subject_uri = _concept_uri(
        subject_name
    )

    if not concept_exists(
        graph,
        subject_name,
    ):
        return []

    relationships: list[
        OntologyRelation
    ] = []

    for predicate, object_value in (
        graph.predicate_objects(
            subject_uri
        )
    ):
        if predicate in {
            RDF.type,
            RDFS.label,
        }:
            continue

        if not isinstance(
            object_value,
            URIRef,
        ):
            continue

        relationships.append(
            OntologyRelation(
                subject_uri=str(
                    subject_uri
                ),
                subject_label=_label_for(
                    graph,
                    subject_uri,
                ),
                predicate_uri=str(
                    predicate
                ),
                predicate_label=_label_for(
                    graph,
                    predicate,
                ),
                object_uri=str(
                    object_value
                ),
                object_label=_label_for(
                    graph,
                    object_value,
                ),
            )
        )

    return relationships


def get_provenance(
    graph: Graph,
    *,
    subject_name: str,
) -> dict[str, str | None]:
    """
    Return current Phase-11/13 subject-level provenance.

    This will later be upgraded to assertion-level provenance.
    """
    subject_uri = _concept_uri(
        subject_name
    )

    def get_literal(
        predicate: URIRef,
    ) -> str | None:
        value = graph.value(
            subject_uri,
            predicate,
        )

        if value is None:
            return None

        return str(
            value
        )

    return {
        "source_document_id": get_literal(
            MED.sourceDocumentId
        ),
        "source_section": get_literal(
            MED.sourceSection
        ),
        "source_version": get_literal(
            MED.sourceVersion
        ),
        "assertion_status": get_literal(
            MED.assertionStatus
        ),
    }


def _literal_text(
    value,
) -> str | None:
    """
    Convert one RDF literal-like value to text.
    """
    if value is None:
        return None

    return str(
        value
    )


def _literal_int(
    value,
) -> int | None:
    """
    Convert one RDF literal-like value to integer.

    Invalid or missing values fail closed to None.
    """
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_assertion_provenance(
    graph: Graph,
    subject_uri: URIRef,
    predicate_uri: URIRef,
    object_uri: URIRef,
) -> dict[str, str | None]:
    """
    Return provenance for one exact ontology assertion.

    Provenance is considered usable only when exactly one
    OntologyAssertion describes the requested triple.

    Missing or ambiguous provenance fails closed.
    """
    matches = []

    for assertion in graph.subjects(
        RDF.type,
        MED.OntologyAssertion,
    ):
        if (
            (
                assertion,
                MED.assertionSubject,
                subject_uri,
            )
            not in graph
        ):
            continue

        if (
            (
                assertion,
                MED.assertionPredicate,
                predicate_uri,
            )
            not in graph
        ):
            continue

        if (
            (
                assertion,
                MED.assertionObject,
                object_uri,
            )
            not in graph
        ):
            continue

        matches.append(
            assertion
        )

    if len(matches) != 1:
        return {
            "source_document_id": None,
            "source_section": None,
            "source_version": None,
            "assertion_status": None,
            "recommendation_priority": None,
            "recommendation_condition": None,
            "applicability_concepts": [],
        }

    assertion = matches[0]

    return {
        "source_document_id": _literal_text(
            graph.value(
                assertion,
                MED.sourceDocumentId,
            )
        ),
        "source_section": _literal_text(
            graph.value(
                assertion,
                MED.sourceSection,
            )
        ),
        "source_version": _literal_text(
            graph.value(
                assertion,
                MED.sourceVersion,
            )
        ),
        "assertion_status": _literal_text(
            graph.value(
                assertion,
                MED.assertionStatus,
            )
        ),
        "recommendation_priority": _literal_int(
            graph.value(
                assertion,
                MED.recommendationPriority,
            )
        ),
        "recommendation_condition": _literal_text(
            graph.value(
                assertion,
                MED.recommendationCondition,
            )
        ),
        "applicability_concepts": [
            ontology_concept_name(
                concept
            )
            for concept in graph.objects(
                assertion,
                MED.applicabilityConcept,
            )
            if isinstance(
                concept,
                URIRef,
            )
        ],
    }


def ontology_predicate_name(
    predicate_uri: URIRef,
) -> str:
    """
    Return the local MedOntoAgent identifier for a predicate.
    """
    value = str(
        predicate_uri
    )

    namespace = str(
        MED
    )

    if value.startswith(
        namespace
    ):
        return value.removeprefix(
            namespace
        )

    if "#" in value:
        return value.rsplit(
            "#",
            1,
        )[-1]

    if "/" in value:
        return value.rsplit(
            "/",
            1,
        )[-1]

    return value


def ontology_concept_name(
    concept_uri: URIRef,
) -> str:
    """
    Return the local identifier for an ontology concept.
    """
    value = str(
        concept_uri
    )

    namespace = str(
        MED
    )

    if value.startswith(
        namespace
    ):
        return value.removeprefix(
            namespace
        )

    if "#" in value:
        return value.rsplit(
            "#",
            1,
        )[-1]

    if "/" in value:
        return value.rsplit(
            "/",
            1,
        )[-1]

    return value


def find_relationships_between_concepts(
    graph: Graph,
    *,
    concept_uris: list[str],
) -> list[OntologyRelation]:
    """
    Find explicit RDF relationships whose subject and object
    are both among the ontology concepts mapped from the query.

    No inference is performed.
    """
    selected: set[
        URIRef
    ] = {
        URIRef(
            uri
        )
        for uri in concept_uris
        if uri.strip()
    }

    if len(
        selected
    ) < 2:
        return []

    relationships: list[
        OntologyRelation
    ] = []

    seen: set[
        tuple[
            str,
            str,
            str,
        ]
    ] = set()

    for subject_uri in selected:
        for (
            predicate,
            object_value,
        ) in graph.predicate_objects(
            subject_uri
        ):
            if not isinstance(
                predicate,
                URIRef,
            ):
                continue

            if not isinstance(
                object_value,
                URIRef,
            ):
                continue

            if object_value not in selected:
                continue

            if predicate in {
                RDF.type,
                RDFS.label,
            }:
                continue

            key = (
                str(subject_uri),
                str(predicate),
                str(object_value),
            )

            if key in seen:
                continue

            relationships.append(
                OntologyRelation(
                    subject_uri=str(
                        subject_uri
                    ),
                    subject_label=_label_for(
                        graph,
                        subject_uri,
                    ),
                    predicate_uri=str(
                        predicate
                    ),
                    predicate_label=_label_for(
                        graph,
                        predicate,
                    ),
                    object_uri=str(
                        object_value
                    ),
                    object_label=_label_for(
                        graph,
                        object_value,
                    ),
                )
            )

            seen.add(
                key
            )

    return relationships