from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.agent.nodes import (
    abstain_node,
    concept_mapping_node,
    constraint_generation_node,
    increment_retry_node,
    initialize_node,
    mock_generation_node,
    mock_validated_node,
    mock_validation_node,
    retrieval_node,
)
from app.agent.routing import (
    route_after_retrieval,
    route_after_validation,
)
from app.agent.state import (
    MedOntoState,
)


def build_medonto_graph():
    """
    Build and compile the bounded Phase-15 workflow.
    """
    builder = StateGraph(
        MedOntoState
    )

    builder.add_node(
        "initialize",
        initialize_node,
    )

    builder.add_node(
        "map_concepts",
        concept_mapping_node,
    )

    builder.add_node(
        "build_constraints",
        constraint_generation_node,
    )

    builder.add_node(
        "retrieve",
        retrieval_node,
    )

    builder.add_node(
        "generate",
        mock_generation_node,
    )

    builder.add_node(
        "validate",
        mock_validation_node,
    )

    builder.add_node(
        "retry",
        increment_retry_node,
    )

    builder.add_node(
        "validated",
        mock_validated_node,
    )

    builder.add_node(
        "abstain",
        abstain_node,
    )

    builder.add_edge(
        START,
        "initialize",
    )

    builder.add_edge(
        "initialize",
        "map_concepts",
    )

    builder.add_edge(
        "map_concepts",
        "build_constraints",
    )

    builder.add_edge(
        "build_constraints",
        "retrieve",
    )

    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "generate": "generate",
            "abstain": "abstain",
        },
    )

    builder.add_edge(
        "generate",
        "validate",
    )

    builder.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "validated": "validated",
            "retry": "retry",
            "abstain": "abstain",
        },
    )

    builder.add_edge(
        "retry",
        "generate",
    )

    builder.add_edge(
        "validated",
        END,
    )

    builder.add_edge(
        "abstain",
        END,
    )

    return builder.compile()


medonto_graph = (
    build_medonto_graph()
)