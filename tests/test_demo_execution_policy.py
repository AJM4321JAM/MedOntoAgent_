
"""Offline tests for isolated demo execution settings."""

from typing import TypedDict

from fastapi import HTTPException
from langgraph.graph import StateGraph

from app.config.settings import (
    activate_demo_settings,
    get_settings,
)
from app.llm.request_budget import get_active_budget


def test_demo_settings_activate_and_restore():
    original = get_settings()

    with activate_demo_settings():
        scoped = get_settings()

        assert scoped.llm_timeout_seconds == 20.0
        assert scoped.max_provider_requests_per_query == 8
        assert scoped is not original

    restored = get_settings()

    assert (
        restored.llm_timeout_seconds
        == original.llm_timeout_seconds
    )
    assert (
        restored.max_provider_requests_per_query
        == original.max_provider_requests_per_query
    )


def test_nested_demo_context_restores_outer_scope():
    original = get_settings()

    with activate_demo_settings():
        outer = get_settings()

        with activate_demo_settings():
            assert get_settings().llm_timeout_seconds == 20.0

        assert get_settings() is outer

    assert get_settings() is not outer
    assert (
        get_settings().llm_timeout_seconds
        == original.llm_timeout_seconds
    )


def test_compiled_langgraph_receives_demo_settings():
    class ProbeState(TypedDict):
        timeout: float
        budget: int

    def probe_node(state: ProbeState) -> dict:
        settings = get_settings()
        return {
            "timeout": settings.llm_timeout_seconds,
            "budget": settings.max_provider_requests_per_query,
        }

    builder = StateGraph(ProbeState)
    builder.add_node("probe", probe_node)
    builder.set_entry_point("probe")
    builder.set_finish_point("probe")

    graph = builder.compile()

    with activate_demo_settings():
        result = graph.invoke({
            "timeout": 0.0,
            "budget": 0,
        })

    assert result["timeout"] == 20.0
    assert result["budget"] == 8


def test_demo_query_enforces_scoped_budget(monkeypatch):
    from app.api import routes_demo

    observed = {}

    def synthetic_execution(query):
        settings = get_settings()
        budget = get_active_budget()

        observed["timeout"] = settings.llm_timeout_seconds
        observed["attempts"] = (
            budget.max_attempts if budget else None
        )

        # Stop before any real retrieval or provider call.
        raise RuntimeError("synthetic offline failure")

    monkeypatch.setattr(
        routes_demo,
        "execute_medonto_with_audit",
        synthetic_execution,
    )

    monkeypatch.setattr(
        routes_demo.query_routes,
        "record_request",
        lambda **kwargs: None,
    )

    from app.models.requests import QueryRequest

    try:
        routes_demo.query_demo(
            QueryRequest(query="Synthetic test question")
        )
    except HTTPException as exc:
        assert exc.status_code == 500
    else:
        raise AssertionError(
            "Expected the synthetic failure to be reported."
        )

    assert observed["timeout"] == 20.0
    assert observed["attempts"] == 8

    # The ContextVar and budget must both be reset.
    assert get_active_budget() is None
