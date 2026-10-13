"""Offline tests for LangGraph execution auditing."""

import pytest
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

from app.agent.execution_audit import (
    execute_graph_with_audit,
    GraphAuditInterrupted,
)


class AuditState(TypedDict, total=False):
    query: str
    retry_count: int
    orchestration_status: str


def build_graph(*, terminal="VERIFIED"):
    builder = StateGraph(AuditState)

    def initialize(state):
        return {"retry_count": 0}

    def finish(state):
        return {"orchestration_status": terminal}

    builder.add_node("initialize", initialize)
    builder.add_node("verified", finish)
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "verified")
    builder.add_edge("verified", END)
    return builder.compile()


def test_records_actual_langgraph_nodes():
    result = execute_graph_with_audit(
        build_graph(), "Synthetic test question"
    )

    assert [event.node for event in result.events] == [
        "initialize", "verified"
    ]
    assert result.state["orchestration_status"] == "VERIFIED"
    assert result.state["retry_count"] == 0


def test_events_do_not_contain_query():
    result = execute_graph_with_audit(
        build_graph(), "Synthetic private question"
    )
    assert "Synthetic private question" not in repr(result.events)


def test_blank_query_rejected():
    with pytest.raises(ValueError):
        execute_graph_with_audit(build_graph(), "   ")


def test_unrecognized_terminal_status_fails():
    with pytest.raises(RuntimeError, match="terminal state"):
        execute_graph_with_audit(
            build_graph(terminal="RUNNING"),
            "Synthetic test question",
        )


def test_event_timing_nonnegative():
    result = execute_graph_with_audit(
        build_graph(), "Synthetic test question"
    )
    assert all(
        event.elapsed_since_previous_update_ms >= 0
        for event in result.events
    )



class InterruptedGraph:
    def stream(self, input, *, stream_mode):
        assert stream_mode == ["updates", "values"]

        yield ("updates", {
            "initialize": {"retry_count": 0}
        })

        yield ("values", {
            "query": input["query"],
            "retry_count": 0,
        })

        yield ("updates", {
            "map_concepts": {"mapped": True}
        })

        raise ValueError("private-provider-error-with-sensitive-text")


def test_interrupted_graph_preserves_completed_steps():
    graph = InterruptedGraph()

    with pytest.raises(GraphAuditInterrupted) as caught:
        execute_graph_with_audit(
            graph,
            "Synthetic confidential clinical question",
        )

    assert [event.node for event in caught.value.events] == [
        "initialize",
        "map_concepts",
    ]


def test_interrupted_audit_does_not_expose_private_text():
    with pytest.raises(GraphAuditInterrupted) as caught:
        execute_graph_with_audit(
            InterruptedGraph(),
            "Synthetic confidential clinical question",
        )

    public_message = str(caught.value)
    audit_events = repr(caught.value.events)

    assert "private-provider-error" not in public_message
    assert "confidential" not in public_message
    assert "confidential" not in audit_events
    assert "private-provider-error" not in audit_events


def test_interrupted_graph_preserves_original_error_type():
    with pytest.raises(GraphAuditInterrupted) as caught:
        execute_graph_with_audit(
            InterruptedGraph(),
            "Synthetic confidential clinical question",
        )

    assert isinstance(caught.value.__cause__, ValueError)
