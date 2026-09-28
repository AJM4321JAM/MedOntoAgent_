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
    mock_validated_node,
    mock_validation_node,
)
from app.agent.routing import (
    route_after_generation,
    route_after_retrieval,
    route_after_validation,
)
from app.agent.state import (
    MedOntoState,
)
from app.models.documents import (
    SourceAuthority,
)
from app.models.responses import (
    MedOntoClaim,
    MedOntoConstraintReference,
    MedOntoDraftOutput,
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


def deterministic_draft() -> MedOntoDraftOutput:
    return MedOntoDraftOutput(
        answer=(
            "Synthetic grounded answer."
        ),
        claims=[
            MedOntoClaim(
                claim_text=(
                    "Synthetic evidence "
                    "supports this claim."
                ),
                citation_chunk_ids=[
                    "DOC-test-CHUNK-0000"
                ],
            )
        ],
        considered_constraints=[],
        status="DRAFT",
    )


def deterministic_generation_node(
    state: MedOntoState,
) -> dict:
    """
    Test-only deterministic replacement for the real
    Phase-16 OpenRouter generation node.
    """
    return {
        "draft_output": (
            deterministic_draft()
        ),
        "generation_abstained": False,
    }


def deterministic_abstention_node(
    state: MedOntoState,
) -> dict:
    """
    Simulate explicit model abstention without calling an LLM.
    """
    draft = MedOntoDraftOutput(
        answer=(
            "The approved retrieved evidence is "
            "insufficient to answer the question."
        ),
        claims=[],
        considered_constraints=[],
        status="ABSTAINED",
    )

    return {
        "draft_output": draft,
        "generation_abstained": True,
    }


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


def test_route_after_generation_validates_draft() -> None:
    state: MedOntoState = {
        "generation_abstained": False,
    }

    assert (
        route_after_generation(
            state
        )
        == "validate"
    )


def test_route_after_generation_abstains_when_model_abstains() -> None:
    state: MedOntoState = {
        "generation_abstained": True,
    }

    assert (
        route_after_generation(
            state
        )
        == "abstain"
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
    Deterministic Phase-16 graph.

    The production graph uses real OpenRouter generation.
    This test graph replaces only generation so unit tests
    never call external services.
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
        deterministic_generation_node,
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

    builder.add_conditional_edges(
        "generate",
        route_after_generation,
        {
            "validate": "validate",
            "abstain": "abstain",
        },
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


def test_graph_reaches_mock_validated_with_draft() -> None:
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
        result[
            "draft_output"
        ]
        .status
        == "DRAFT"
    )


def test_graph_routes_model_abstention_directly_to_abstain() -> None:
    """
    Explicit generation abstention must not enter validation
    or retry.
    """

    def inject_retrieval(
        state: MedOntoState,
    ) -> dict:
        return {
            "retrieval_result": (
                evidence_retrieval()
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
        deterministic_abstention_node,
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

    builder.add_conditional_edges(
        "generate",
        route_after_generation,
        {
            "validate": "abstain",
            "abstain": "abstain",
        },
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
        result[
            "orchestration_status"
        ]
        == "ABSTAINED"
    )

    assert (
        result[
            "generation_abstained"
        ]
        is True
    )

    assert (
        result[
            "retry_count"
        ]
        == 0
    )


def test_graph_retry_is_bounded() -> None:
    """
    Force validation failure and prove exactly one
    regeneration occurs.
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
            "draft_output": (
                deterministic_draft()
            ),
            "generation_abstained": False,
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

    assert (
        result[
            "validation_reason"
        ]
        == "Forced validation failure."
    )