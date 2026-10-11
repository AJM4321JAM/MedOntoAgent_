from typing import Literal

from app.agent.state import (
    MedOntoState,
)


def route_after_constraints(
    state: MedOntoState,
) -> Literal[
    "retrieve",
    "baseline",
]:
    """
    Use the ontology path only when at least one governed,
    control-eligible ontology constraint applies to the query.

    Otherwise route deliberately to conventional baseline RAG.
    """
    constraint_set = state.get(
        "constraint_set"
    )

    if constraint_set is None:
        return "baseline"

    if any(
        constraint.control_eligible
        for constraint
        in constraint_set.constraints
    ):
        return "retrieve"

    return "baseline"


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
    Explicit model abstention terminates immediately.
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
    "verified",
    "retry",
    "abstain",
]:
    """
    Route from deterministic response-validation policy.
    """
    result = state.get(
        "response_validation"
    )

    if result is None:
        retry_count = state.get(
            "retry_count",
            0,
        )

        max_retries = state.get(
            "max_retries",
            0,
        )

        if retry_count < max_retries:
            return "retry"

        return "abstain"

    if result.status == "VERIFIED":
        return "verified"

    if result.status == "RETRY":
        return "retry"

    return "abstain"