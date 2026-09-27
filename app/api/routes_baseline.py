from fastapi import (
    APIRouter,
    HTTPException,
    status,
)

from app.llm.baseline import (
    BaselineRAGError,
    run_baseline_rag,
)
from app.llm.client import (
    LLMConfigurationError,
    LLMServiceError,
)
from app.models.requests import (
    QueryRequest,
)
from app.models.responses import (
    BaselineQueryResponse,
)
from app.retrieval.retriever import (
    RetrievalError,
)


router = APIRouter(
    prefix="/query-baseline",
    tags=["Baseline RAG"],
)


@router.post(
    "",
    response_model=BaselineQueryResponse,
)
def query_baseline(
    request: QueryRequest,
) -> BaselineQueryResponse:
    """
    Execute the conventional vector-RAG baseline.
    """
    try:
        return run_baseline_rag(
            request.query
        )

    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=str(exc),
        ) from exc

    except RetrievalError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_502_BAD_GATEWAY
            ),
            detail=str(exc),
        ) from exc

    except LLMServiceError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_502_BAD_GATEWAY
            ),
            detail=str(exc),
        ) from exc

    except BaselineRAGError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_502_BAD_GATEWAY
            ),
            detail=str(exc),
        ) from exc