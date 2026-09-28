from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.agent.graph import (
    build_medonto_graph,
)
from app.agent.nodes import (
    abstain_node,
    increment_retry_node,
    mock_generation_node,
    mock_validated_node,
    mock_validation_node,
)
from app.agent.routing import (
    route_after_retrieval,
    route_after_validation,
)
from app.agent.state import (
    MedOntoState,
)
from app.models.documents import (
    SourceAuthority,
)
from app.models.retrieval import (
    RetrievedChunk,
    RetrievalResult,
)


def empty_retrieval() -> RetrievalResult:
    return RetrievalResult(
        query="test",
        top_k=5,
        retrieved_chunks=[],
        result_count=0,
    )


def evidence_retrieval() -> RetrievalResult:
    chunk = RetrievedChunk(
        chunk_id=(
            "DOC-test-CHUNK-0000"
        ),
        document_id="DOC-test",
        title="Synthetic Test",
        section_title="Test",
        page_number=None,
        text=(
            "Synthetic evidence "
            "for graph testing."
        ),
        score=0.9,
        version="1",
        publisher="MedOntoAgent",
        clinical_domain="Testing",
        document_type="Synthetic Test",
        source_authority=(
            SourceAuthority.OTHER
        ),
        source_url=None,
    )

    return RetrievalResult(
        query="test",
        top_k=5,
        retrieved_chunks=[
            chunk
        ],
        result_count=1,
    )


def test_graph_compiles() -> None:
    graph = build_medonto_graph()

    assert graph is not None


def test_route_after_retrieval_abstains_without_evidence() -> None:
    state: MedOntoState = {
        "retrieval_result": (
            empty_retrieval()
        ),
    }

    assert (
        route_after_retrieval(
            state
        )
        == "abstain"
    )


def test_route_after_retrieval_generates_with_evidence() -> None:
    state: MedOntoState = {
        "retrieval_result": (
            evidence_retrieval()
        ),
    }

    assert (
        route_after_retrieval(
            state
        )
        == "generate"
    )


def test_validation_success_routes_to_validated() -> None:
    state: MedOntoState = {
        "validation_passed": True,
        "retry_count": 0,
        "max_retries": 1,
    }

    assert (
        route_after_validation(
            state
        )
        == "validated"
    )


def test_validation_failure_routes_to_retry_when_available() -> None:
    state: MedOntoState = {
        "validation_passed": False,
        "retry_count": 0,
        "max_retries": 1,
    }

    assert (
        route_after_validation(
            state
        )
        == "retry"
    )


def test_validation_failure_abstains_when_retry_exhausted() -> None:
    state: MedOntoState = {
        "validation_passed": False,
        "retry_count": 1,
        "max_retries": 1,
    }

    assert (
        route_after_validation(
            state
        )
        == "abstain"
    )


def build_deterministic_test_graph(
    retrieval_result: RetrievalResult,
):
    """
    Test graph containing only deterministic Phase-15
    generation, validation and terminal routing.
    """

    def inject_retrieval(
        state: MedOntoState,
    ) -> dict:
        return {
            "retrieval_result": (
                retrieval_result
            ),
        }

    builder = StateGraph(
        MedOntoState
    )

    builder.add_node(
        "retrieve",
        inject_retrieval,
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


def test_graph_abstains_when_retrieval_is_empty() -> None:
    graph = (
        build_deterministic_test_graph(
            empty_retrieval()
        )
    )

    result = graph.invoke(
        {
            "query": "test question",
            "retry_count": 0,
            "max_retries": 1,
            "orchestration_status": (
                "RUNNING"
            ),
        }
    )

    assert (
        result[
            "orchestration_status"
        ]
        == "ABSTAINED"
    )


def test_graph_reaches_mock_validated_with_evidence() -> None:
    graph = (
        build_deterministic_test_graph(
            evidence_retrieval()
        )
    )

    result = graph.invoke(
        {
            "query": "test question",
            "retry_count": 0,
            "max_retries": 1,
            "orchestration_status": (
                "RUNNING"
            ),
        }
    )

    assert (
        result[
            "orchestration_status"
        ]
        == "MOCK_VALIDATED"
    )

    assert (
        result[
            "validation_passed"
        ]
        is True
    )

    assert (
        "PHASE 15 MOCK DRAFT"
        in result[
            "draft_answer"
        ]
    )


def test_graph_retry_is_bounded() -> None:
    """
    Force validation failure and prove exactly one retry
    can occur.
    """

    generation_count = {
        "value": 0,
    }

    def inject_evidence(
        state: MedOntoState,
    ) -> dict:
        return {
            "retrieval_result": (
                evidence_retrieval()
            ),
        }

    def counting_generation(
        state: MedOntoState,
    ) -> dict:
        generation_count[
            "value"
        ] += 1

        return {
            "draft_answer": (
                "PHASE 15 MOCK DRAFT"
            ),
        }

    def always_fail_validation(
        state: MedOntoState,
    ) -> dict:
        return {
            "validation_passed": False,
            "validation_reason": (
                "Forced validation failure."
            ),
        }

    builder = StateGraph(
        MedOntoState
    )

    builder.add_node(
        "retrieve",
        inject_evidence,
    )

    builder.add_node(
        "generate",
        counting_generation,
    )

    builder.add_node(
        "validate",
        always_fail_validation,
    )

    builder.add_node(
        "retry",
        increment_retry_node,
    )

    builder.add_node(
        "abstain",
        abstain_node,
    )

    builder.add_edge(
        START,
        "retrieve",
    )

    builder.add_edge(
        "retrieve",
        "generate",
    )

    builder.add_edge(
        "generate",
        "validate",
    )

    builder.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "validated": "abstain",
            "retry": "retry",
            "abstain": "abstain",
        },
    )

    builder.add_edge(
        "retry",
        "generate",
    )

    builder.add_edge(
        "abstain",
        END,
    )

    graph = builder.compile()

    result = graph.invoke(
        {
            "query": "test question",
            "retry_count": 0,
            "max_retries": 1,
            "orchestration_status": (
                "RUNNING"
            ),
        }
    )

    assert (
        generation_count[
            "value"
        ]
        == 2
    )

    assert (
        result[
            "retry_count"
        ]
        == 1
    )

    assert (
        result[
            "orchestration_status"
        ]
        == "ABSTAINED"
    )