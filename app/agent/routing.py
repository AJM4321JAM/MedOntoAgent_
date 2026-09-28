from typing import Literal

from app.agent.state import (
    MedOntoState,
)


def route_after_retrieval(
    state: MedOntoState,
) -> Literal[
    "generate",
    "abstain",
]:
    """
    Generate only when retrieved evidence exists.
    """
    retrieval = state.get(
        "retrieval_result"
    )

    if (
        retrieval is None
        or retrieval.result_count == 0
    ):
        return "abstain"

    return "generate"


def route_after_validation(
    state: MedOntoState,
) -> Literal[
    "validated",
    "retry",
    "abstain",
]:
    """
    Finish when mock validation passes.

    Otherwise retry only while budget remains.
    """
    if state.get(
        "validation_passed",
        False,
    ):
        return "validated"

    retry_count = state.get(
        "retry_count",
        0,
    )

    max_retries = state.get(
        "max_retries",
        1,
    )

    if retry_count < max_retries:
        return "retry"

    return "abstain"