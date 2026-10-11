from app.agent.nodes import (
    constraint_generation_node,
)
from app.ontology.concept_mapper import (
    build_ontology_context,
)


def test_agent_constraint_generation_uses_registry() -> None:
    query = (
        "Can ACE inhibitors be used during pregnancy?"
    )

    ontology_context = build_ontology_context(
        query=query
    )

    result = constraint_generation_node(
        {
            "query": query,
            "ontology_context": ontology_context,
        }
    )

    constraint_set = result[
        "constraint_set"
    ]

    matching = [
        constraint
        for constraint
        in constraint_set.constraints
        if (
            constraint.subject_name
            == "ACEInhibitor"
            and constraint.predicate_name
            == "notRecommendedFor"
            and constraint.object_name
            == "Pregnancy"
        )
    ]

    assert len(matching) == 1

    constraint = matching[0]

    assert (
        constraint.source_document_id
        == "DOC-e943ba2bdc31"
    )

    assert (
        constraint.assertion_status
        == "ACTIVE"
    )

    assert (
        constraint.control_eligible
        is True
    )
