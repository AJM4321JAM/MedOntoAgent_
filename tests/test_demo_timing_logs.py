
"""Offline, content-free timing-log regression tests."""

import logging

from typing import TypedDict

from langgraph.graph import StateGraph

from app.agent.execution_audit import (
    execute_graph_with_audit,
)
from app.agent import audited_execution


SENSITIVE_SENTINEL = "SENSITIVE_TEST_CONTENT_NOT_FOR_LOGS"


class ProbeState(TypedDict):
    query: str
    orchestration_status: str
    answer: str


def make_probe_graph():
    def first_node(state):
        return {
            "answer": SENSITIVE_SENTINEL,
        }

    def finish_node(state):
        return {
            "orchestration_status": "VERIFIED",
        }

    builder = StateGraph(ProbeState)
    builder.add_node("initialize", first_node)
    builder.add_node("verified", finish_node)
    builder.set_entry_point("initialize")
    builder.add_edge("initialize", "verified")
    builder.set_finish_point("verified")

    return builder.compile()


def test_audit_logs_completed_stage_without_content(caplog):
    caplog.set_level(
        logging.INFO,
        logger="app.agent.execution_audit",
    )

    result = execute_graph_with_audit(
        make_probe_graph(),
        SENSITIVE_SENTINEL,
    )

    messages = caplog.text

    assert len(result.events) == 2
    assert "node=initialize" in messages
    assert "node=verified" in messages
    assert "elapsed_ms=" in messages
    assert SENSITIVE_SENTINEL not in messages


def test_baseline_logs_duration_without_content(
    monkeypatch,
    caplog,
):
    caplog.set_level(
        logging.INFO,
        logger="app.agent.audited_execution",
    )

    def fake_baseline(query, *, settings=None):
        return object()

    monkeypatch.setattr(
        audited_execution,
        "run_baseline_rag",
        fake_baseline,
    )

    result = audited_execution._run_baseline(
        SENSITIVE_SENTINEL,
        settings=None,
        mode="BASELINE_ROUTED",
        events=(),
    )

    assert result.execution.baseline_response is not None
    assert "MedOnto baseline started:" in caplog.text
    assert "MedOnto baseline complete:" in caplog.text
    assert "outcome=success" in caplog.text
    assert "elapsed_ms=" in caplog.text
    assert SENSITIVE_SENTINEL not in caplog.text
