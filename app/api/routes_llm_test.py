from fastapi import APIRouter, HTTPException, status

from app.llm.client import (
    LLMConfigurationError,
    LLMServiceError,
    run_llm_test,
)
from app.llm.structured_output import LLMTestResponse


router = APIRouter(
    prefix="/llm-test",
    tags=["Development"],
)


@router.post(
    "",
    response_model=LLMTestResponse,
)
def test_llm_connection() -> LLMTestResponse:
    """
    Temporary development endpoint used to verify Gemini connectivity.

    This endpoint will not be part of the final clinical query workflow.
    """
    try:
        return run_llm_test()

    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    except LLMServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc