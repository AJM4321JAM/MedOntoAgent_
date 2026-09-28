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
    Continue only when evidence exists.
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


def route_after_generation(
    state: MedOntoState,
) -> Literal[
    "validate",
    "abstain",
]:
    """
    Explicit model abstention terminates safely.

    We do not pressure the model to answer by retrying.
    """
    if state.get(
        "generation_abstained",
        False,
    ):
        return "abstain"

    return "validate"


def route_after_validation(
    state: MedOntoState,
) -> Literal[
    "validated",
    "retry",
    "abstain",
]:
    """
    Finish, retry within budget, or abstain.
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