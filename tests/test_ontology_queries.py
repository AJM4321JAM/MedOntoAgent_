from app.ontology.ontology_loader import (
    load_ontology,
)
from app.ontology.ontology_queries import (
    concept_exists,
    get_provenance,
    get_relationships_for_subject,
    relationship_exists,
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


def test_synthetic_provenance_is_available() -> None:
    graph = load_ontology()

    provenance = get_provenance(
        graph,
        subject_name=(
            "ExampleAntihypertensiveClass"
        ),
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


def test_known_condition_exists() -> None:
    graph = load_ontology()

    assert concept_exists(
        graph,
        "Hypertension",
    )