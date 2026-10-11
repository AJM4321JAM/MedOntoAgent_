from types import (
    SimpleNamespace,
)
from unittest.mock import (
    patch,
)

from fastapi.testclient import (
    TestClient,
)

from app.main import app
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
    "medonto_graph.invoke"
)
def test_verified_query_returns_safe_public_response(
    mock_invoke,
) -> None:
    mock_invoke.return_value = (
        create_verified_state()
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
    "medonto_graph.invoke"
)
def test_abstained_query_returns_no_citations(
    mock_invoke,
) -> None:
    mock_invoke.return_value = (
        create_abstained_state()
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
    "medonto_graph.invoke"
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
    "medonto_graph.invoke"
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