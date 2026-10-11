from rdflib import (
    Literal,
)

from app.models.ontology import (
    MappedOntologyConcept,
    OntologyContext,
)
from app.ontology.constraints import (
    build_constraints,
    classify_relationship,
    is_constraint_retrieval_eligible,
)
from app.ontology.ontology_loader import (
    load_ontology,
)
from app.ontology.ontology_queries import (
    MED,
)


def create_constraint_context() -> OntologyContext:
    return OntologyContext(
        query=(
            "Can Example Antihypertensive Class "
            "be used during pregnancy?"
        ),
        extracted_concepts=[
            "Example Antihypertensive Class",
            "pregnancy",
        ],
        mapped_concepts=[
            MappedOntologyConcept(
                original_text=(
                    "Example Antihypertensive Class"
                ),
                concept_name=(
                    "ExampleAntihypertensiveClass"
                ),
                label=(
                    "Example Antihypertensive Class"
                ),
                uri=(
                    "https://medontoagent.example.org/"
                    "ontology/"
                    "ExampleAntihypertensiveClass"
                ),
            ),
            MappedOntologyConcept(
                original_text="pregnancy",
                concept_name="Pregnancy",
                label="Pregnancy",
                uri=(
                    "https://medontoagent.example.org/"
                    "ontology/Pregnancy"
                ),
            ),
        ],
        unknown_concepts=[],
        mapped_count=2,
        unknown_count=0,
    )


def test_classify_contraindication() -> None:
    assert (
        classify_relationship(
            "contraindicatedIn"
        )
        == "CONTRAINDICATION"
    )


def test_classify_recommendation() -> None:
    assert (
        classify_relationship(
            "recommendedFor"
        )
        == "RECOMMENDATION"
    )


def test_classify_not_recommended() -> None:
    assert (
        classify_relationship(
            "notRecommendedFor"
        )
        == "NOT_RECOMMENDED"
    )


def test_unknown_relationship_becomes_other() -> None:
    assert (
        classify_relationship(
            "someFutureRelationship"
        )
        == "OTHER"
    )


def test_build_constraints_discovers_relationship() -> None:
    graph = load_ontology()

    context = (
        create_constraint_context()
    )

    result = build_constraints(
        query=context.query,
        ontology_context=context,
        graph=graph,
    )

    assert (
        result.constraint_count
        == 1
    )

    assert len(
        result.constraints
    ) == 1

    constraint = (
        result.constraints[0]
    )

    assert (
        constraint.constraint_type
        == "CONTRAINDICATION"
    )

    assert (
        constraint.subject_name
        == "ExampleAntihypertensiveClass"
    )

    assert (
        constraint.predicate_name
        == "contraindicatedIn"
    )

    assert (
        constraint.object_name
        == "Pregnancy"
    )


def test_constraint_retains_provenance() -> None:
    graph = load_ontology()

    context = (
        create_constraint_context()
    )

    result = build_constraints(
        query=context.query,
        ontology_context=context,
        graph=graph,
    )

    constraint = (
        result.constraints[0]
    )

    assert (
        constraint.source_document_id
        == "SYNTHETIC-ONTOLOGY-TEST"
    )

    assert (
        constraint.source_section
        == "Implementation Test"
    )

    assert (
        constraint.source_version
        == "1.0"
    )

    assert (
        constraint.assertion_status
        == "SYNTHETIC_TEST_ONLY"
    )


def test_synthetic_relationship_does_not_expand_query() -> None:
    graph = load_ontology()

    context = (
        create_constraint_context()
    )

    result = build_constraints(
        query=context.query,
        ontology_context=context,
        graph=graph,
    )

    assert (
        result.constraint_count
        == 1
    )

    assert (
        result.retrieval_hints.relationship_terms
        == []
    )

    assert (
        result.retrieval_hints.expanded_query
        == context.query
    )


def test_original_query_is_preserved() -> None:
    graph = load_ontology()

    context = (
        create_constraint_context()
    )

    result = build_constraints(
        query=context.query,
        ontology_context=context,
        graph=graph,
    )

    assert (
        result.retrieval_hints.expanded_query
        .startswith(
            context.query
        )
    )


def test_no_relation_produces_no_constraint() -> None:
    graph = load_ontology()

    context = OntologyContext(
        query=(
            "What about hypertension "
            "during pregnancy?"
        ),
        extracted_concepts=[
            "hypertension",
            "pregnancy",
        ],
        mapped_concepts=[
            MappedOntologyConcept(
                original_text="hypertension",
                concept_name="Hypertension",
                label="Hypertension",
                uri=(
                    "https://medontoagent.example.org/"
                    "ontology/Hypertension"
                ),
            ),
            MappedOntologyConcept(
                original_text="pregnancy",
                concept_name="Pregnancy",
                label="Pregnancy",
                uri=(
                    "https://medontoagent.example.org/"
                    "ontology/Pregnancy"
                ),
            ),
        ],
        unknown_concepts=[],
        mapped_count=2,
        unknown_count=0,
    )

    result = build_constraints(
        query=context.query,
        ontology_context=context,
        graph=graph,
    )

    assert result.constraints == []

    assert (
        result.constraint_count
        == 0
    )

    assert (
        result.retrieval_hints.expanded_query
        == context.query
    )

def test_synthetic_constraint_is_not_retrieval_eligible() -> None:
    graph = load_ontology()

    context = create_constraint_context()

    result = build_constraints(
        query=context.query,
        ontology_context=context,
        graph=graph,
    )

    matching = [
        constraint
        for constraint
        in result.constraints
        if (
            constraint.predicate_name
            == "contraindicatedIn"
        )
    ]

    assert matching

    assert (
        matching[0].assertion_status
        == "SYNTHETIC_TEST_ONLY"
    )

    assert (
        matching[0].retrieval_eligible
        is False
    )


def test_active_provenanced_constraint_is_eligible() -> None:
    graph = load_ontology()

    assertion = (
        MED.AssertionExampleContraindication
    )

    graph.remove(
        (
            assertion,
            MED.assertionStatus,
            None,
        )
    )

    graph.add(
        (
            assertion,
            MED.assertionStatus,
            Literal("ACTIVE"),
        )
    )

    context = create_constraint_context()

    result = build_constraints(
        query=context.query,
        ontology_context=context,
        graph=graph,
    )

    matching = [
        constraint
        for constraint
        in result.constraints
        if (
            constraint.predicate_name
            == "contraindicatedIn"
        )
    ]

    assert matching

    assert (
        matching[0].assertion_status
        == "ACTIVE"
    )

    assert (
        matching[0].retrieval_eligible
        is True
    )


def test_unknown_predicate_is_not_control_eligible() -> None:
    assert (
        is_constraint_retrieval_eligible(
            predicate_name=(
                "someArbitraryRelation"
            ),
            source_document_id=(
                "DOC-TEST"
            ),
            assertion_status="ACTIVE",
        )
        is False
    )


def test_missing_source_document_id_is_not_control_eligible() -> None:
    assert (
        is_constraint_retrieval_eligible(
            predicate_name=(
                "contraindicatedIn"
            ),
            source_document_id=None,
            assertion_status="ACTIVE",
        )
        is False
    )
