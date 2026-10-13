"""Audited MedOnto execution with existing fallback semantics."""

import logging
from time import perf_counter
from dataclasses import dataclass

from app.agent.execution import (
    FALLBACK_EXCEPTIONS,
    MedOntoExecution,
)
from app.agent.execution_audit import (
    ExecutionEvent,
    GraphAuditInterrupted,
    execute_graph_with_audit,
)
from app.agent.graph import medonto_graph
from app.config.settings import Settings
from app.llm.baseline import BaselineRAGError, run_baseline_rag
from app.llm.client import LLMServiceError
from app.llm.error_classification import classify_error
from app.retrieval.retriever import RetrievalError


logger = logging.getLogger(__name__)

BASELINE_ERRORS = (
    BaselineRAGError,
    RetrievalError,
    LLMServiceError,
)


@dataclass(frozen=True)
class AuditedExecution:
    execution: MedOntoExecution
    events: tuple[ExecutionEvent, ...]


def execute_medonto_with_audit(
    query: str,
    *,
    settings: Settings | None = None,
) -> AuditedExecution:
    """Execute one audited graph and retain normal fallback policy.

    Audit events contain no query, answer, document or model text.
    The clinical response is retained internally in MedOntoExecution.
    """
    events: tuple[ExecutionEvent, ...] = ()

    try:
        result = execute_graph_with_audit(
            medonto_graph,
            query,
        )
        events = result.events
        state = result.state

    except GraphAuditInterrupted as interrupted:
        events = interrupted.events
        original = interrupted.__cause__

        if not isinstance(original, FALLBACK_EXCEPTIONS):
            if original is not None:
                raise original
            raise

        return _run_baseline(
            query,
            settings=settings,
            mode="BASELINE_FALLBACK",
            events=events,
            fallback_reason=type(original).__name__,
            medonto_error_category=classify_error(original),
        )

    status = state.get("orchestration_status")

    if status == "BASELINE_ROUTED":
        return _run_baseline(
            query,
            settings=settings,
            mode="BASELINE_ROUTED",
            events=events,
        )

    return AuditedExecution(
        execution=MedOntoExecution(
            execution_mode="MEDONTO",
            medonto_state=state,
        ),
        events=events,
    )


def _run_baseline(
    query: str,
    *,
    settings: Settings | None,
    mode: str,
    events: tuple[ExecutionEvent, ...],
    fallback_reason: str | None = None,
    medonto_error_category: str | None = None,
) -> AuditedExecution:
    baseline_start = perf_counter()
    logger.info(
        "MedOnto baseline started: mode=%s",
        mode,
    )

    try:
        response = run_baseline_rag(
            query,
            settings=settings,
        )
    except BASELINE_ERRORS as exc:
        execution = MedOntoExecution(
            execution_mode=mode,
            baseline_response=None,
            fallback_reason=fallback_reason,
            fallback_error=type(exc).__name__,
            error_category=classify_error(exc),
            medonto_error_category=medonto_error_category,
        )
    else:
        execution = MedOntoExecution(
            execution_mode=mode,
            baseline_response=response,
            fallback_reason=fallback_reason,
            medonto_error_category=medonto_error_category,
        )

    logger.info(
        "MedOnto baseline complete: "
        "mode=%s outcome=%s elapsed_ms=%.2f",
        mode,
        "success" if execution.baseline_response is not None
        else "dependency_failure",
        (perf_counter() - baseline_start) * 1000,
    )

    return AuditedExecution(
        execution=execution,
        events=events,
    )
