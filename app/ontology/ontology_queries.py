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
    Convert a controlled MedOntoAgent concept name into
    its ontology URI.
    """
    normalized = (
        concept_name.strip()
    )

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
    Convert a controlled relationship name into
    its ontology URI.
    """
    normalized = (
        predicate_name.strip()
    )

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
    Return an RDF label when one exists.

    Otherwise return the final URI component.
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

    text = str(
        uri
    )

    if "/" in text:
        return text.rsplit(
            "/",
            1,
        )[-1]

    return text


def concept_exists(
    graph: Graph,
    concept_name: str,
) -> bool:
    """
    Return True when a named MedOntoAgent ontology
    resource exists in the graph.
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
    Return True when the exact controlled relationship
    exists in the ontology.
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
    Return ontology object relationships for one subject.

    RDF type declarations, labels, and datatype provenance
    fields are intentionally excluded.
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
    Return the current subject-level provenance metadata.

    This is sufficient for the Phase-11 implementation test.

    Later, clinically meaningful ontology assertions will
    move to assertion-level provenance so separate relations
    can reference separate evidence sources.
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