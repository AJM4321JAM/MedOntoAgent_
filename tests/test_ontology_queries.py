from app.ontology.ontology_loader import (
    load_ontology,
)
from app.ontology.ontology_queries import (
    MED,
    concept_exists,
    find_relationships_between_concepts,
    get_assertion_provenance,
    get_provenance,
    get_relationships_for_subject,
    relationship_exists,
)


EXAMPLE_URI = (
    "https://medontoagent.example.org/"
    "ontology/ExampleAntihypertensiveClass"
)

PREGNANCY_URI = (
    "https://medontoagent.example.org/"
    "ontology/Pregnancy"
)

HYPERTENSION_URI = (
    "https://medontoagent.example.org/"
    "ontology/Hypertension"
)


def test_known_concept_exists() -> None:
    graph = load_ontology()

    assert concept_exists(
        graph,
        "Pregnancy",
    )


def test_unknown_concept_does_not_exist() -> None:
    graph = load_ontology()

    assert not concept_exists(
        graph,
        "ImaginaryClinicalConcept",
    )


def test_synthetic_contraindication_exists() -> None:
    graph = load_ontology()

    assert relationship_exists(
        graph,
        subject_name=(
            "ExampleAntihypertensiveClass"
        ),
        predicate_name=(
            "contraindicatedIn"
        ),
        object_name="Pregnancy",
    )


def test_nonexistent_relationship_returns_false() -> None:
    graph = load_ontology()

    assert not relationship_exists(
        graph,
        subject_name=(
            "ExampleAntihypertensiveClass"
        ),
        predicate_name=(
            "recommendedFor"
        ),
        object_name="Pregnancy",
    )


def test_relationships_return_labels() -> None:
    graph = load_ontology()

    relationships = (
        get_relationships_for_subject(
            graph,
            subject_name=(
                "ExampleAntihypertensiveClass"
            ),
        )
    )

    assert len(
        relationships
    ) == 1

    relation = (
        relationships[0]
    )

    assert (
        relation.subject_label
        == "Example Antihypertensive Class"
    )

    assert (
        relation.predicate_label
        == "contraindicated in"
    )

    assert (
        relation.object_label
        == "Pregnancy"
    )


def test_unknown_subject_returns_no_relationships() -> None:
    graph = load_ontology()

    relationships = (
        get_relationships_for_subject(
            graph,
            subject_name=(
                "ImaginaryClinicalConcept"
            ),
        )
    )

    assert relationships == []


def test_synthetic_assertion_has_exact_provenance() -> None:
    graph = load_ontology()

    provenance = get_assertion_provenance(
        graph,
        MED.ExampleAntihypertensiveClass,
        MED.contraindicatedIn,
        MED.Pregnancy,
    )

    assert (
        provenance[
            "source_document_id"
        ]
        == "SYNTHETIC-ONTOLOGY-TEST"
    )

    assert (
        provenance[
            "source_section"
        ]
        == "Implementation Test"
    )

    assert (
        provenance[
            "source_version"
        ]
        == "1.0"
    )

    assert (
        provenance[
            "assertion_status"
        ]
        == "SYNTHETIC_TEST_ONLY"
    )


def test_ambiguous_assertion_provenance_fails_closed() -> None:
    from rdflib import Literal, RDF

    graph = load_ontology()

    duplicate = MED.DuplicateTestAssertion

    graph.add(
        (
            duplicate,
            RDF.type,
            MED.OntologyAssertion,
        )
    )
    graph.add(
        (
            duplicate,
            MED.assertionSubject,
            MED.ExampleAntihypertensiveClass,
        )
    )
    graph.add(
        (
            duplicate,
            MED.assertionPredicate,
            MED.contraindicatedIn,
        )
    )
    graph.add(
        (
            duplicate,
            MED.assertionObject,
            MED.Pregnancy,
        )
    )
    graph.add(
        (
            duplicate,
            MED.sourceDocumentId,
            Literal("DOC-DUPLICATE"),
        )
    )
    graph.add(
        (
            duplicate,
            MED.assertionStatus,
            Literal("ACTIVE"),
        )
    )

    provenance = get_assertion_provenance(
        graph,
        MED.ExampleAntihypertensiveClass,
        MED.contraindicatedIn,
        MED.Pregnancy,
    )

    assert provenance == {
        "source_document_id": None,
        "source_section": None,
        "source_version": None,
        "assertion_status": None,
    }


def test_known_condition_exists() -> None:
    graph = load_ontology()

    assert concept_exists(
        graph,
        "Hypertension",
    )


def test_relationship_discovery_between_selected_concepts() -> None:
    graph = load_ontology()

    relationships = (
        find_relationships_between_concepts(
            graph,
            concept_uris=[
                EXAMPLE_URI,
                PREGNANCY_URI,
            ],
        )
    )

    assert len(
        relationships
    ) == 1

    relation = (
        relationships[0]
    )

    assert (
        relation.subject_label
        == "Example Antihypertensive Class"
    )

    assert (
        relation.predicate_label
        == "contraindicated in"
    )

    assert (
        relation.object_label
        == "Pregnancy"
    )


def test_relationship_discovery_does_not_pull_unrelated_data() -> None:
    graph = load_ontology()

    relationships = (
        find_relationships_between_concepts(
            graph,
            concept_uris=[
                HYPERTENSION_URI,
                PREGNANCY_URI,
            ],
        )
    )

    assert relationships == []


def test_single_concept_cannot_form_relationship_pair() -> None:
    graph = load_ontology()

    relationships = (
        find_relationships_between_concepts(
            graph,
            concept_uris=[
                PREGNANCY_URI,
            ],
        )
    )

    assert relationships == []