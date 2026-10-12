from typing import Literal

import logging
import time

from fastapi import (
    APIRouter,
    HTTPException,
    status,
)

from app.config.settings import get_settings
from app.llm.request_budget import (
    EvaluationBudgetExhausted,
    RequestBudget,
    activate_request_budget,
)

from app.agent.execution import (
    execute_medonto,
)
from app.models.requests import (
    QueryRequest,
)
from app.models.responses import (
    BaselineQueryResponse,
    MedOntoQueryResponse,
    PublicCitation,
    PublicOntologyConstraint,
    RetrievalSummary,
)
from app.observability.metrics import (
    record_request,
)
from app.observability.tracing import (
    observed_span,
)


logger = logging.getLogger(
    __name__
)


router = APIRouter(
    prefix="/query",
    tags=[
        "MedOntoAgent"
    ],
)


class MedOntoAPIError(
    RuntimeError
):
    """
    Raised when internal graph state cannot be converted into
    the safe public API contract.
    """


def _enum_value(
    value,
) -> str:
    """
    Return a stable string for either an Enum value or a string.
    """
    enum_value = getattr(
        value,
        "value",
        None,
    )

    if enum_value is not None:
        return str(
            enum_value
        )

    return str(
        value
    )


def _build_retrieval_summary(
    state: dict,
) -> RetrievalSummary:
    retrieval = state.get(
        "retrieval_result"
    )

    result_count = (
        retrieval.result_count
        if retrieval is not None
        else 0
    )

    return RetrievalSummary(
        result_count=result_count,
        used_ontology_filter=bool(
            state.get(
                "used_ontology_filter",
                False,
            )
        ),
        retrieval_relaxed=bool(
            state.get(
                "retrieval_relaxed",
                False,
            )
        ),
    )


def _build_public_constraints(
    state: dict,
) -> list[
    PublicOntologyConstraint
]:
    """
    Rebuild public constraint metadata from the application-owned
    constraint set, never from model-generated prose.
    """
    constraint_set = state.get(
        "constraint_set"
    )

    if constraint_set is None:
        return []

    public_constraints: list[
        PublicOntologyConstraint
    ] = []

    for constraint in (
        constraint
        for constraint
        in constraint_set.constraints
        if constraint.control_eligible
    ):
        public_constraints.append(
            PublicOntologyConstraint(
                constraint_type=(
                    _enum_value(
                        constraint.constraint_type
                    )
                ),
                subject_name=(
                    constraint.subject_name
                ),
                predicate_name=(
                    constraint.predicate_name
                ),
                object_name=(
                    constraint.object_name
                ),
            )
        )

    return public_constraints


def _build_verified_citations(
    state: dict,
) -> list[
    PublicCitation
]:
    """
    Reconstruct citation metadata from actual retrieved chunks.

    The draft may identify chunk IDs, but it is never trusted
    to provide public source metadata.
    """
    draft = state.get(
        "draft_output"
    )

    retrieval = state.get(
        "retrieval_result"
    )

    if draft is None:
        raise MedOntoAPIError(
            (
                "VERIFIED state does not contain "
                "a structured draft."
            )
        )

    if retrieval is None:
        raise MedOntoAPIError(
            (
                "VERIFIED state does not contain "
                "retrieval results."
            )
        )

    chunk_lookup = {
        chunk.chunk_id: chunk
        for chunk
        in retrieval.retrieved_chunks
    }

    cited_chunk_ids: list[str] = []
    seen_chunk_ids: set[str] = set()

    for claim in draft.claims:
        for chunk_id in (
            claim.citation_chunk_ids
        ):
            if chunk_id in seen_chunk_ids:
                continue

            cited_chunk_ids.append(
                chunk_id
            )

            seen_chunk_ids.add(
                chunk_id
            )

    citations: list[
        PublicCitation
    ] = []

    for chunk_id in cited_chunk_ids:
        chunk = chunk_lookup.get(
            chunk_id
        )

        if chunk is None:
            raise MedOntoAPIError(
                (
                    "VERIFIED response referenced "
                    "a chunk absent from retrieval."
                )
            )

        source_url = (
            str(
                chunk.source_url
            )
            if chunk.source_url is not None
            else None
        )

        citations.append(
            PublicCitation(
                chunk_id=(
                    chunk.chunk_id
                ),
                document_id=(
                    chunk.document_id
                ),
                title=(
                    chunk.title
                ),
                section_title=(
                    chunk.section_title
                ),
                page_number=(
                    chunk.page_number
                ),
                source_url=(
                    source_url
                ),
            )
        )

    if not citations:
        raise MedOntoAPIError(
            (
                "VERIFIED response contains "
                "no public citations."
            )
        )

    return citations


def _build_fallback_public_response(
    *,
    query: str,
    baseline_response: BaselineQueryResponse,
    execution_mode: Literal[
        "BASELINE_ROUTED",
        "BASELINE_FALLBACK",
    ],
) -> MedOntoQueryResponse:
    """
    Convert a conventional baseline fallback into the public
    MedOnto response contract without claiming ontology verification.
    """
    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk
        in baseline_response.retrieved_chunks
    }

    citations: list[
        PublicCitation
    ] = []

    if baseline_response.status == "ANSWERED":
        for citation_reference in (
            baseline_response.citations
        ):
            chunk = chunks_by_id.get(
                citation_reference.chunk_id
            )

            if chunk is None:
                raise MedOntoAPIError(
                    (
                        "Baseline fallback referenced "
                        "a chunk absent from retrieval."
                    )
                )

            citations.append(
                PublicCitation(
                    chunk_id=chunk.chunk_id,
                    document_id=(
                        chunk.document_id
                    ),
                    title=chunk.title,
                    section_title=(
                        chunk.section_title
                    ),
                    page_number=(
                        chunk.page_number
                    ),
                    source_url=(
                        chunk.source_url
                    ),
                )
            )

    return MedOntoQueryResponse(
        query=query,
        answer=baseline_response.answer,
        status=baseline_response.status,
        execution_mode=(
            execution_mode
        ),
        citations=citations,
        ontology_constraints=[],
        retrieval=RetrievalSummary(
            result_count=len(
                baseline_response
                .retrieved_chunks
            ),
            used_ontology_filter=False,
            retrieval_relaxed=False,
        ),
    )


def _build_public_response(
    *,
    query: str,
    state: dict,
) -> MedOntoQueryResponse:
    """
    Convert internal LangGraph state into the stable public API
    response without exposing internal reasoning machinery.
    """
    orchestration_status = (
        state.get(
            "orchestration_status"
        )
    )

    retrieval_summary = (
        _build_retrieval_summary(
            state
        )
    )

    if (
        orchestration_status
        == "VERIFIED"
    ):
        draft = state.get(
            "draft_output"
        )

        if draft is None:
            raise MedOntoAPIError(
                (
                    "VERIFIED state does not contain "
                    "a generated draft."
                )
            )

        if draft.status != "DRAFT":
            raise MedOntoAPIError(
                (
                    "VERIFIED graph state contains "
                    "an invalid draft status."
                )
            )

        return MedOntoQueryResponse(
            query=query,
            answer=draft.answer,
            status="VERIFIED",
            execution_mode="MEDONTO",
            citations=(
                _build_verified_citations(
                    state
                )
            ),
            ontology_constraints=(
                _build_public_constraints(
                    state
                )
            ),
            retrieval=(
                retrieval_summary
            ),
        )

    if (
        orchestration_status
        == "ABSTAINED"
    ):
        abstention_reason = (
            state.get(
                "abstention_reason"
            )
            or (
                "MedOntoAgent could not produce "
                "a sufficiently supported answer."
            )
        )

        return MedOntoQueryResponse(
            query=query,
            answer=(
                abstention_reason
            ),
            status="ABSTAINED",
            execution_mode="MEDONTO",
            citations=[],
            ontology_constraints=(
                _build_public_constraints(
                    state
                )
            ),
            retrieval=(
                retrieval_summary
            ),
        )

    raise MedOntoAPIError(
        (
            "Graph completed without a valid "
            "public terminal status."
        )
    )


@router.post(
    "",
    response_model=(
        MedOntoQueryResponse
    ),
)
def query_medontoagent(
    request: QueryRequest,
) -> MedOntoQueryResponse:
    """
    Execute the validated MedOntoAgent workflow.

    Unexpected internal failures are deliberately converted into
    a generic HTTP error so implementation details, provider
    errors, filesystem paths, credentials, and stack traces are
    not returned to API consumers.
    """
    start_time = time.perf_counter()

    try:
        with observed_span(
            "medonto.graph",
            attributes={
                "medonto.query.length": len(
                    request.query
                ),
            },
        ) as span:
            with activate_request_budget(
                RequestBudget(
                    max_attempts=get_settings().max_provider_requests_per_query
                )
            ):
                execution = execute_medonto(request.query)

            if (
                execution.execution_mode
                == "MEDONTO"
            ):
                state = (
                    execution.medonto_state
                )

                if state is None:
                    raise MedOntoAPIError(
                        (
                            "MEDONTO execution did "
                            "not return graph state."
                        )
                    )

                orchestration_status = (
                    state.get(
                        "orchestration_status",
                        "ERROR",
                    )
                )

                retry_count = int(
                    state.get(
                        "retry_count",
                        0,
                    )
                )

                response = (
                    _build_public_response(
                        query=request.query,
                        state=state,
                    )
                )

            else:
                baseline_response = (
                    execution
                    .baseline_response
                )

                if baseline_response is None:
                    raise MedOntoAPIError(
                        (
                            "Fallback execution did "
                            "not return a baseline "
                            "response."
                        )
                    )

                orchestration_status = (
                    baseline_response.status
                )

                retry_count = 0

                response = (
                    _build_fallback_public_response(
                        query=request.query,
                        baseline_response=(
                            baseline_response
                        ),
                        execution_mode=(
                            execution.execution_mode
                        ),
                    )
                )

            span.set_attribute(
                "medonto.response.status",
                orchestration_status,
            )

            span.set_attribute(
                "medonto.execution_mode",
                execution.execution_mode,
            )

            span.set_attribute(
                "medonto.retry_count",
                retry_count,
            )

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        record_request(
            status=orchestration_status,
            latency_ms=latency_ms,
            retry_count=retry_count,
        )

        return response

    except EvaluationBudgetExhausted as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        record_request(status="ERROR", latency_ms=latency_ms, retry_count=0)
        logger.warning("MedOntoAgent provider request budget exhausted")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MedOntoAgent temporarily cannot complete the request.",
        ) from exc

    except Exception as exc:
        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        record_request(
            status="ERROR",
            latency_ms=latency_ms,
            retry_count=0,
        )

        logger.error(
            "MedOntoAgent request failed: %s",
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "MedOntoAgent could not "
                "complete the request."
            ),
        ) from exc
