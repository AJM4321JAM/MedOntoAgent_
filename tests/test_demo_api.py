"""Offline integration tests for the protected demo API."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent.execution import MedOntoExecution
from app.agent.audited_execution import AuditedExecution
from app.agent.execution_audit import ExecutionEvent
from app.api import routes_demo
from app.models.responses import MedOntoQueryResponse, RetrievalSummary
from app.security import auth


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        auth, "require_admin_secret", lambda: "local-test-secret"
    )
    monkeypatch.setattr(
        routes_demo.query_routes,
        "get_settings",
        lambda: SimpleNamespace(max_provider_requests_per_query=8),
    )
    monkeypatch.setattr(
        routes_demo.query_routes,
        "activate_request_budget",
        lambda budget: nullcontext(),
    )
    monkeypatch.setattr(
        routes_demo.query_routes,
        "observed_span",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        routes_demo.query_routes,
        "record_request",
        lambda **kwargs: None,
    )

    app = FastAPI()
    app.include_router(routes_demo.router)
    return TestClient(app)


def public_response(query, mode="MEDONTO"):
    return MedOntoQueryResponse(
        query=query,
        answer="Synthetic demonstration response",
        status="ABSTAINED",
        execution_mode=mode,
        citations=[],
        ontology_constraints=[],
        retrieval=RetrievalSummary(
            result_count=0,
            used_ontology_filter=False,
            retrieval_relaxed=False,
        ),
    )


def audited_result(mode="MEDONTO"):
    return AuditedExecution(
        execution=MedOntoExecution(
            execution_mode=mode,
            medonto_state={
                "orchestration_status": "ABSTAINED",
                "retry_count": 0,
            } if mode == "MEDONTO" else None,
            baseline_response=(
                None if mode == "MEDONTO" else object()
            ),
        ),
        events=(
            ExecutionEvent(
                sequence=1,
                node="initialize",
                elapsed_since_previous_update_ms=2.5,
            ),
        ),
    )


def test_missing_admin_key_rejected(client):
    response = client.post(
        "/demo/query",
        json={"query": "Synthetic question"},
    )
    assert response.status_code == 401


def test_invalid_admin_key_rejected(client):
    response = client.post(
        "/demo/query",
        headers={"X-MedOnto-Admin-Key": "incorrect"},
        json={"query": "Synthetic question"},
    )
    assert response.status_code == 401


def test_authorized_medonto_query(client, monkeypatch):
    calls = []

    def execute(query):
        calls.append(query)
        return audited_result()

    monkeypatch.setattr(
        routes_demo, "execute_medonto_with_audit", execute
    )
    monkeypatch.setattr(
        routes_demo.query_routes,
        "_build_public_response",
        lambda *, query, state: public_response(query),
    )

    response = client.post(
        "/demo/query",
        headers={"X-MedOnto-Admin-Key": "local-test-secret"},
        json={"query": "Synthetic question"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["response"]["execution_mode"] == "MEDONTO"
    assert body["execution_trace"][0]["node"] == "initialize"
    assert calls == ["Synthetic question"]
    assert "medonto_state" not in body
    assert "baseline_response" not in body


def test_authorized_baseline_route(client, monkeypatch):
    monkeypatch.setattr(
        routes_demo,
        "execute_medonto_with_audit",
        lambda query: audited_result("BASELINE_ROUTED"),
    )
    monkeypatch.setattr(
        routes_demo.query_routes,
        "_build_fallback_public_response",
        lambda *, query, baseline_response, execution_mode:
            public_response(query, execution_mode),
    )

    response = client.post(
        "/demo/query",
        headers={"X-MedOnto-Admin-Key": "local-test-secret"},
        json={"query": "Synthetic question"},
    )

    assert response.status_code == 200, response.text
    assert (
        response.json()["response"]["execution_mode"]
        == "BASELINE_ROUTED"
    )


def test_unexpected_failure_has_generic_response(client, monkeypatch):
    def fail(query):
        raise TypeError("private internal implementation detail")

    monkeypatch.setattr(
        routes_demo, "execute_medonto_with_audit", fail
    )

    response = client.post(
        "/demo/query",
        headers={"X-MedOnto-Admin-Key": "local-test-secret"},
        json={"query": "Synthetic question"},
    )

    assert response.status_code == 500
    assert "private internal" not in response.text
    assert "MedOntoAgent could not complete" in response.text
