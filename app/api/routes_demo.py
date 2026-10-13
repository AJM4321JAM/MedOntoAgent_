"""Authenticated endpoint for the MedOntoAgent decision audit."""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from app.agent.audited_execution import execute_medonto_with_audit
from app.api import routes_query as query_routes
from app.models.requests import QueryRequest
from app.models.responses import MedOntoQueryResponse
from app.security.auth import require_admin


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/demo",
    tags=["demo"],
    dependencies=[Depends(require_admin)],
)


class PublicExecutionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int
    node: str
    elapsed_since_previous_update_ms: float


class DemoQueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: MedOntoQueryResponse
    execution_trace: list[PublicExecutionEvent]


@router.post("/query", response_model=DemoQueryResponse)
def query_demo(request: QueryRequest) -> DemoQueryResponse:
    """Execute one clinical query with a bounded, safe node timeline."""
    start = time.perf_counter()

    try:
        with query_routes.observed_span(
            "medonto.demo.graph",
            attributes={"medonto.query.length": len(request.query)},
        ):
            with query_routes.activate_request_budget(
                query_routes.RequestBudget(
                    max_attempts=(
                        query_routes.get_settings()
                        .max_provider_requests_per_query
                    )
                )
            ):
                audited = execute_medonto_with_audit(request.query)

        execution = audited.execution

        if execution.execution_mode == "MEDONTO":
            state = execution.medonto_state
            if state is None:
                raise ValueError("Missing MedOnto graph state.")

            response = query_routes._build_public_response(
                query=request.query,
                state=state,
            )
            retries = int(state.get("retry_count", 0))

        else:
            baseline = execution.baseline_response
            if baseline is None:
                raise ValueError("Baseline execution did not complete.")

            response = query_routes._build_fallback_public_response(
                query=request.query,
                baseline_response=baseline,
                execution_mode=execution.execution_mode,
            )
            retries = 0

        result = DemoQueryResponse(
            response=response,
            execution_trace=[
                PublicExecutionEvent(
                    sequence=event.sequence,
                    node=event.node,
                    elapsed_since_previous_update_ms=(
                        event.elapsed_since_previous_update_ms
                    ),
                )
                for event in audited.events
            ],
        )

        query_routes.record_request(
            status=response.status,
            latency_ms=(time.perf_counter() - start) * 1000,
            retry_count=retries,
        )
        return result

    except query_routes.EvaluationBudgetExhausted as exc:
        query_routes.record_request(
            status="ERROR",
            latency_ms=(time.perf_counter() - start) * 1000,
            retry_count=0,
        )
        logger.warning("Demo provider request budget exhausted")
        raise HTTPException(
            status_code=503,
            detail="MedOntoAgent temporarily cannot complete the request.",
        ) from exc

    except Exception as exc:
        query_routes.record_request(
            status="ERROR",
            latency_ms=(time.perf_counter() - start) * 1000,
            retry_count=0,
        )
        logger.error("Demo query failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail="MedOntoAgent could not complete the request.",
        ) from exc
