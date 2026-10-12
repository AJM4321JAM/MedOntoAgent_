from types import (
    SimpleNamespace,
)
from unittest.mock import (
    patch,
)

from fastapi.testclient import (
    TestClient,
)

from app.agent.execution import (
    MedOntoExecution,
)
from app.main import app
from app.models.documents import (
    SourceAuthority,
)
from app.models.responses import (
    BaselineQueryResponse,
    CitationReference,
    MedOntoClaim,
    MedOntoDraftOutput,
)
from app.models.retrieval import (
    RetrievedChunk,
    RetrievalResult,
)


client = TestClient(
    app
)


def create_retrieval() -> (
    RetrievalResult
):
    chunk = RetrievedChunk(
        chunk_id="CHUNK-1",
        document_id="DOC-1",
        title=(
            "Synthetic Test Source"
        ),
        section_title="Constraint",
        page_number=4,
        text=(
            "Treatment X is contraindicated "
            "in Group Y."
        ),
        score=0.95,
        version="1.0",
        publisher="MedOntoAgent",
        clinical_domain="Testing",
        document_type=(
            "Synthetic Test"
        ),
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


def create_constraint_set():
    constraint = SimpleNamespace(
        constraint_type=(
            "CONTRAINDICATION"
        ),
        subject_name="TreatmentX",
        predicate_name=(
            "contraindicatedIn"
        ),
        object_name="GroupY",
        control_eligible=True,
    )

    return SimpleNamespace(
        constraints=[
            constraint
        ]
    )


def create_verified_state() -> dict:
    draft = MedOntoDraftOutput(
        answer=(
            "Treatment X is contraindicated "
            "in Group Y."
        ),
        claims=[
            MedOntoClaim(
                claim_text=(
                    "Treatment X is contraindicated "
                    "in Group Y."
                ),
                citation_chunk_ids=[
                    "CHUNK-1"
                ],
            )
        ],
        considered_constraints=[],
        status="DRAFT",
    )

    return {
        "orchestration_status": (
            "VERIFIED"
        ),
        "draft_output": draft,
        "retrieval_result": (
            create_retrieval()
        ),
        "constraint_set": (
            create_constraint_set()
        ),
        "used_ontology_filter": True,
        "retrieval_relaxed": False,
    }


def create_abstained_state() -> dict:
    return {
        "orchestration_status": (
            "ABSTAINED"
        ),
        "abstention_reason": (
            "No sufficient retrieved evidence "
            "was available."
        ),
        "retrieval_result": (
            RetrievalResult(
                query="test",
                top_k=5,
                retrieved_chunks=[],
                result_count=0,
            )
        ),
        "constraint_set": (
            SimpleNamespace(
                constraints=[]
            )
        ),
        "used_ontology_filter": False,
        "retrieval_relaxed": False,
    }


@patch(
    "app.api.routes_query."
    "execute_medonto"
)
def test_verified_query_returns_safe_public_response(
    mock_invoke,
) -> None:
    mock_invoke.return_value = (
        MedOntoExecution(
            execution_mode="MEDONTO",
            medonto_state=(
                create_verified_state()
            ),
        )
    )

    response = client.post(
        "/query",
        json={
            "query": (
                "Can Treatment X be used "
                "in Group Y?"
            )
        },
    )

    assert (
        response.status_code
        == 200
    )

    body = response.json()

    assert (
        body["status"]
        == "VERIFIED"
    )

    assert (
        body["execution_mode"]
        == "MEDONTO"
    )

    assert (
        body["answer"]
        == (
            "Treatment X is contraindicated "
            "in Group Y."
        )
    )

    assert len(
        body["citations"]
    ) == 1

    citation = (
        body["citations"][0]
    )

    assert (
        citation["chunk_id"]
        == "CHUNK-1"
    )

    assert (
        citation["document_id"]
        == "DOC-1"
    )

    assert (
        citation["title"]
        == "Synthetic Test Source"
    )

    assert (
        citation["section_title"]
        == "Constraint"
    )

    assert (
        citation["page_number"]
        == 4
    )

    assert (
        citation["source_url"]
        is None
    )

    assert (
        body[
            "ontology_constraints"
        ][0][
            "constraint_type"
        ]
        == "CONTRAINDICATION"
    )

    assert (
        body[
            "ontology_constraints"
        ][0][
            "subject_name"
        ]
        == "TreatmentX"
    )

    assert (
        body[
            "ontology_constraints"
        ][0][
            "predicate_name"
        ]
        == "contraindicatedIn"
    )

    assert (
        body[
            "ontology_constraints"
        ][0][
            "object_name"
        ]
        == "GroupY"
    )

    assert body["retrieval"] == {
        "result_count": 1,
        "used_ontology_filter": True,
        "retrieval_relaxed": False,
    }

    assert set(
        body.keys()
    ) == {
        "query",
        "answer",
        "status",
        "execution_mode",
        "citations",
        "ontology_constraints",
        "retrieval",
    }


@patch(
    "app.api.routes_query."
    "execute_medonto"
)
def test_abstained_query_returns_no_citations(
    mock_invoke,
) -> None:
    mock_invoke.return_value = (
        MedOntoExecution(
            execution_mode="MEDONTO",
            medonto_state=(
                create_abstained_state()
            ),
        )
    )

    response = client.post(
        "/query",
        json={
            "query": (
                "What does the corpus say "
                "about something unrelated?"
            )
        },
    )

    assert (
        response.status_code
        == 200
    )

    body = response.json()

    assert (
        body["status"]
        == "ABSTAINED"
    )

    assert (
        body["answer"]
        == (
            "No sufficient retrieved evidence "
            "was available."
        )
    )

    assert (
        body["citations"]
        == []
    )

    assert (
        body[
            "ontology_constraints"
        ]
        == []
    )

    assert body["retrieval"] == {
        "result_count": 0,
        "used_ontology_filter": False,
        "retrieval_relaxed": False,
    }


@patch(
    "app.api.routes_query."
    "execute_medonto"
)
def test_invalid_request_is_rejected_before_graph(
    mock_invoke,
) -> None:
    response = client.post(
        "/query",
        json={
            "query": "x"
        },
    )

    assert (
        response.status_code
        == 422
    )

    mock_invoke.assert_not_called()


@patch(
    "app.api.routes_query."
    "execute_medonto"
)
def test_internal_exception_is_not_exposed(
    mock_invoke,
) -> None:
    mock_invoke.side_effect = RuntimeError(
        (
            "SECRET provider error at "
            "C:\\private\\path with API key"
        )
    )

    response = client.post(
        "/query",
        json={
            "query": (
                "Can Treatment X be used "
                "in Group Y?"
            )
        },
    )

    assert (
        response.status_code
        == 500
    )

    body = response.json()

    assert body == {
        "detail": (
            "MedOntoAgent could not "
            "complete the request."
        )
    }

    serialized = str(
        body
    )

    assert (
        "SECRET"
        not in serialized
    )

    assert (
        "private"
        not in serialized
    )

    assert (
        "API key"
        not in serialized
    )


@patch(
    "app.api.routes_query."
    "execute_medonto"
)
def test_operational_failure_returns_baseline_fallback(
    mock_execute,
) -> None:
    retrieval = create_retrieval()
    chunk = retrieval.retrieved_chunks[0]

    mock_execute.return_value = (
        MedOntoExecution(
            execution_mode=(
                "BASELINE_FALLBACK"
            ),
            baseline_response=(
                BaselineQueryResponse(
                    query=(
                        "Can Treatment X be used "
                        "in Group Y?"
                    ),
                    answer=(
                        "Baseline supported answer."
                    ),
                    citations=[
                        CitationReference(
                            chunk_id=chunk.chunk_id,
                        )
                    ],
                    citation_document_ids=[
                        chunk.document_id,
                    ],
                    retrieved_chunk_ids=[
                        chunk.chunk_id,
                    ],
                    retrieved_document_ids=[
                        chunk.document_id,
                    ],
                    retrieved_contexts=[
                        chunk.text,
                    ],
                    retrieved_chunks=[
                        chunk,
                    ],
                    status="ANSWERED",
                )
            ),
            fallback_reason=(
                "MedOntoGenerationError"
            ),
        )
    )

    response = client.post(
        "/query",
        json={
            "query": (
                "Can Treatment X be used "
                "in Group Y?"
            )
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ANSWERED"

    assert (
        body["execution_mode"]
        == "BASELINE_FALLBACK"
    )

    assert (
        body["answer"]
        == "Baseline supported answer."
    )

    assert len(body["citations"]) == 1

    assert (
        body["citations"][0]["chunk_id"]
        == "CHUNK-1"
    )

    assert (
        body["citations"][0]["document_id"]
        == "DOC-1"
    )

    assert (
        body["ontology_constraints"]
        == []
    )

    assert body["retrieval"] == {
        "result_count": 1,
        "used_ontology_filter": False,
        "retrieval_relaxed": False,
    }



@patch(
    "app.api.routes_query."
    "execute_medonto"
)
def test_nonontology_query_returns_baseline_routed(
    mock_execute,
) -> None:
    retrieval = create_retrieval()
    chunk = retrieval.retrieved_chunks[0]

    mock_execute.return_value = (
        MedOntoExecution(
            execution_mode=(
                "BASELINE_ROUTED"
            ),
            baseline_response=(
                BaselineQueryResponse(
                    query=(
                        "Synthetic evidence question?"
                    ),
                    answer=(
                        "Baseline routed answer."
                    ),
                    citations=[
                        CitationReference(
                            chunk_id=chunk.chunk_id,
                        )
                    ],
                    citation_document_ids=[
                        chunk.document_id,
                    ],
                    retrieved_chunk_ids=[
                        chunk.chunk_id,
                    ],
                    retrieved_document_ids=[
                        chunk.document_id,
                    ],
                    retrieved_contexts=[
                        chunk.text,
                    ],
                    retrieved_chunks=[
                        chunk,
                    ],
                    status="ANSWERED",
                )
            ),
        )
    )

    response = client.post(
        "/query",
        json={
            "query": (
                "Synthetic evidence question?"
            )
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["execution_mode"]
        == "BASELINE_ROUTED"
    )

    assert body["status"] == "ANSWERED"

    assert (
        body["ontology_constraints"]
        == []
    )


# Provider-budget integration tests require no external LLM calls.

def test_query_budget_exhaustion_returns_safe_503() -> None:
    from app.llm.request_budget import get_active_budget
    from app.config.settings import get_settings

    def invoke(_query):
        budget = get_active_budget()
        assert budget is not None
        budget.before_request()
        budget.before_request()
        raise AssertionError("Cap did not block second provider attempt")

    with patch("app.api.routes_query.execute_medonto", side_effect=invoke), \
         patch("app.api.routes_query.get_settings") as mock_settings:
        mock_settings.return_value = SimpleNamespace(max_provider_requests_per_query=1)
        response = client.post("/query", json={"query": "Synthetic safety question?"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": "MedOntoAgent temporarily cannot complete the request."
    }
    assert get_active_budget() is None


def test_separate_query_requests_receive_fresh_budgets() -> None:
    from app.llm.request_budget import get_active_budget

    seen = []

    def invoke(_query):
        budget = get_active_budget()
        assert budget is not None
        seen.append(budget)
        budget.before_request()
        return MedOntoExecution(
            execution_mode="MEDONTO",
            medonto_state=create_verified_state(),
        )

    with patch("app.api.routes_query.execute_medonto", side_effect=invoke), \
         patch("app.api.routes_query.get_settings") as mock_settings:
        mock_settings.return_value = SimpleNamespace(max_provider_requests_per_query=1)
        first = client.post("/query", json={"query": "Synthetic safety question?"})
        second = client.post("/query", json={"query": "Synthetic safety question?"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(seen) == 2 and seen[0] is not seen[1]
    assert [budget.attempts for budget in seen] == [1, 1]


def test_baseline_endpoint_budget_exhaustion_returns_safe_503() -> None:
    from app.llm.request_budget import get_active_budget

    def run(_query):
        budget = get_active_budget()
        assert budget is not None
        budget.before_request()
        budget.before_request()
        raise AssertionError("Baseline cap did not block second request")

    with patch("app.api.routes_baseline.run_baseline_rag", side_effect=run), \
         patch("app.api.routes_baseline.get_settings") as mock_settings:
        mock_settings.return_value = SimpleNamespace(max_provider_requests_per_query=1)
        response = client.post(
            "/query-baseline", json={"query": "Synthetic safety question?"}
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "MedOntoAgent temporarily cannot complete the request."
    }
