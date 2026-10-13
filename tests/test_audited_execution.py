"""Offline tests for the audited execution controller."""

import pytest

from app.agent import audited_execution as module
from app.llm.client import LLMServiceError
from app.retrieval.retriever import RetrievalError


class FakeGraph:
    def __init__(self, status=None, error=None):
        self.status = status
        self.error = error

    def stream(self, input, *, stream_mode):
        assert stream_mode == ["updates", "values"]

        yield ("values", {"query": input["query"]})
        yield ("updates", {"initialize": {"retry_count": 0}})
        yield ("values", {
            "query": input["query"],
            "retry_count": 0,
        })

        if self.error is not None:
            raise self.error

        node = (
            "baseline_routed"
            if self.status == "BASELINE_ROUTED"
            else "verified"
        )

        yield ("updates", {
            node: {"orchestration_status": self.status}
        })
        yield ("values", {
            "query": input["query"],
            "retry_count": 0,
            "orchestration_status": self.status,
        })


def test_verified_execution_does_not_call_baseline(monkeypatch):
    monkeypatch.setattr(
        module, "medonto_graph", FakeGraph("VERIFIED")
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Baseline must not execute.")

    monkeypatch.setattr(module, "run_baseline_rag", forbidden)

    result = module.execute_medonto_with_audit(
        "Synthetic test question"
    )

    assert result.execution.execution_mode == "MEDONTO"
    assert result.execution.medonto_state["orchestration_status"] == "VERIFIED"
    assert [e.node for e in result.events] == [
        "initialize", "verified"
    ]


def test_deliberate_baseline_routing(monkeypatch):
    sentinel = object()

    monkeypatch.setattr(
        module, "medonto_graph", FakeGraph("BASELINE_ROUTED")
    )
    monkeypatch.setattr(
        module,
        "run_baseline_rag",
        lambda query, settings=None: sentinel,
    )

    result = module.execute_medonto_with_audit(
        "Synthetic test question"
    )

    assert result.execution.execution_mode == "BASELINE_ROUTED"
    assert result.execution.baseline_response is sentinel
    assert result.events[-1].node == "baseline_routed"


def test_known_graph_failure_triggers_fallback(monkeypatch):
    sentinel = object()

    monkeypatch.setattr(
        module,
        "medonto_graph",
        FakeGraph(error=RetrievalError("private detail")),
    )
    monkeypatch.setattr(
        module,
        "run_baseline_rag",
        lambda query, settings=None: sentinel,
    )

    result = module.execute_medonto_with_audit(
        "Synthetic test question"
    )

    assert result.execution.execution_mode == "BASELINE_FALLBACK"
    assert result.execution.baseline_response is sentinel
    assert result.execution.fallback_reason == "RetrievalError"
    assert [e.node for e in result.events] == ["initialize"]


def test_unexpected_graph_failure_propagates(monkeypatch):
    monkeypatch.setattr(
        module,
        "medonto_graph",
        FakeGraph(error=TypeError("programming defect")),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected errors must not use baseline.")

    monkeypatch.setattr(module, "run_baseline_rag", forbidden)

    with pytest.raises(TypeError, match="programming defect"):
        module.execute_medonto_with_audit(
            "Synthetic test question"
        )


def test_baseline_failure_is_recorded(monkeypatch):
    monkeypatch.setattr(
        module, "medonto_graph", FakeGraph("BASELINE_ROUTED")
    )

    def failed_baseline(query, settings=None):
        raise RetrievalError("private retrieval detail")

    monkeypatch.setattr(
        module, "run_baseline_rag", failed_baseline
    )

    result = module.execute_medonto_with_audit(
        "Synthetic test question"
    )

    assert result.execution.execution_mode == "BASELINE_ROUTED"
    assert result.execution.baseline_response is None
    assert result.execution.fallback_error == "RetrievalError"
