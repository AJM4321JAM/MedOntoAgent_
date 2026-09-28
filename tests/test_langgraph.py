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
    verified_node,
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
    MedOntoDraftOutput,
)
from app.models.retrieval import (
    RetrievedChunk,
    RetrievalResult,
)
from app.models.validation import (
    CitationValidationItem,
    CitationValidationResult,
    OntologyValidationItem,
    OntologyValidationResult,
    ResponseValidationResult,
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
    Test-only generator.

    Production Phase 17 uses the real grounded generation node.
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
    Simulate explicit model abstention without external calls.
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


def supported_citation_result() -> (
    CitationValidationResult
):
    return CitationValidationResult(
        items=[
            CitationValidationItem(
                claim_index=0,
                claim_text=(
                    "Synthetic evidence "
                    "supports this claim."
                ),
                citation_chunk_ids=[
                    "DOC-test-CHUNK-0000"
                ],
                verdict="SUPPORTED",
                reason=(
                    "The cited evidence supports "
                    "the generated claim."
                ),
            )
        ],
        all_claims_supported=True,
    )


def compliant_ontology_result() -> (
    OntologyValidationResult
):
    return OntologyValidationResult(
        items=[],
        has_conflict=False,
    )


def deterministic_validation_node(
    state: MedOntoState,
) -> dict:
    """
    Deterministic successful Phase-17 validation.

    No external validator model is called.
    """
    citation = (
        supported_citation_result()
    )

    ontology = (
        compliant_ontology_result()
    )

    response = (
        ResponseValidationResult(
            status="VERIFIED",
            citation_validation=citation,
            ontology_validation=ontology,
            feedback=None,
        )
    )

    return {
        "citation_validation": citation,
        "ontology_validation": ontology,
        "response_validation": response,
        "validation_passed": True,
        "validation_reason": None,
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


def test_verified_response_routes_to_verified() -> None:
    citation = (
        supported_citation_result()
    )

    ontology = (
        compliant_ontology_result()
    )

    response = ResponseValidationResult(
        status="VERIFIED",
        citation_validation=citation,
        ontology_validation=ontology,
        feedback=None,
    )

    state: MedOntoState = {
        "response_validation": (
            response
        ),
        "retry_count": 0,
        "max_retries": 1,
    }

    assert (
        route_after_validation(
            state
        )
        == "verified"
    )


def test_retry_response_routes_to_retry() -> None:
    citation = CitationValidationResult(
        items=[
            CitationValidationItem(
                claim_index=0,
                claim_text=(
                    "Unsupported claim."
                ),
                citation_chunk_ids=[
                    "DOC-test-CHUNK-0000"
                ],
                verdict="UNSUPPORTED",
                reason="Not supported.",
            )
        ],
        all_claims_supported=False,
    )

    ontology = (
        compliant_ontology_result()
    )

    response = ResponseValidationResult(
        status="RETRY",
        citation_validation=citation,
        ontology_validation=ontology,
        feedback=(
            "Claim 1 is unsupported."
        ),
    )

    state: MedOntoState = {
        "response_validation": (
            response
        ),
        "retry_count": 0,
        "max_retries": 1,
    }

    assert (
        route_after_validation(
            state
        )
        == "retry"
    )


def test_abstained_response_routes_to_abstain() -> None:
    citation = CitationValidationResult(
        items=[],
        all_claims_supported=False,
    )

    ontology = (
        compliant_ontology_result()
    )

    response = ResponseValidationResult(
        status="ABSTAINED",
        citation_validation=citation,
        ontology_validation=ontology,
        feedback=(
            "Validation failed after "
            "the retry budget was exhausted."
        ),
    )

    state: MedOntoState = {
        "response_validation": (
            response
        ),
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
    Deterministic Phase-17 graph.

    Real external generation and semantic validators are replaced
    with deterministic test nodes.
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
        deterministic_validation_node,
    )

    builder.add_node(
        "retry",
        increment_retry_node,
    )

    builder.add_node(
        "verified",
        verified_node,
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
            "verified": "verified",
            "retry": "retry",
            "abstain": "abstain",
        },
    )

    builder.add_edge(
        "retry",
        "generate",
    )

    builder.add_edge(
        "verified",
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


def test_graph_reaches_verified_with_validated_draft() -> None:
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
        == "VERIFIED"
    )

    assert (
        result[
            "response_validation"
        ]
        .status
        == "VERIFIED"
    )

    assert (
        result[
            "validation_passed"
        ]
        is True
    )


def test_graph_model_abstention_bypasses_validation_and_retry() -> None:
    """
    Explicit generation abstention goes directly to ABSTAINED.
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


def test_graph_retry_is_bounded_and_second_failure_abstains() -> None:
    """
    First validation failure returns RETRY.

    Second validation failure returns ABSTAINED.

    Therefore total generation attempts are exactly two.
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

    def failing_validation(
        state: MedOntoState,
    ) -> dict:
        citation = (
            CitationValidationResult(
                items=[
                    CitationValidationItem(
                        claim_index=0,
                        claim_text=(
                            "Unsupported claim."
                        ),
                        citation_chunk_ids=[
                            "DOC-test-CHUNK-0000"
                        ],
                        verdict=(
                            "UNSUPPORTED"
                        ),
                        reason=(
                            "Forced test failure."
                        ),
                    )
                ],
                all_claims_supported=False,
            )
        )

        ontology = (
            OntologyValidationResult(
                items=[],
                has_conflict=False,
            )
        )

        retry_count = state.get(
            "retry_count",
            0,
        )

        max_retries = state.get(
            "max_retries",
            1,
        )

        if retry_count < max_retries:
            status = "RETRY"
        else:
            status = "ABSTAINED"

        response = (
            ResponseValidationResult(
                status=status,
                citation_validation=(
                    citation
                ),
                ontology_validation=(
                    ontology
                ),
                feedback=(
                    "Claim 1 is unsupported: "
                    "Forced test failure."
                ),
            )
        )

        return {
            "citation_validation": (
                citation
            ),
            "ontology_validation": (
                ontology
            ),
            "response_validation": (
                response
            ),
            "validation_passed": False,
            "validation_reason": (
                response.feedback
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
        failing_validation,
    )

    builder.add_node(
        "retry",
        increment_retry_node,
    )

    builder.add_node(
        "verified",
        verified_node,
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
            "verified": "verified",
            "retry": "retry",
            "abstain": "abstain",
        },
    )

    builder.add_edge(
        "retry",
        "generate",
    )

    builder.add_edge(
        "verified",
        END,
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
            "response_validation"
        ]
        .status
        == "ABSTAINED"
    )

    assert (
        "Forced test failure"
        in result[
            "validation_reason"
        ]
    )