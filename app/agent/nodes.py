from app.agent.state import (
    MedOntoState,
)
from app.config.settings import (
    get_settings,
)
from app.llm.medonto_generation import (
    generate_medonto_draft,
)
from app.ontology.concept_mapper import (
    build_ontology_context,
)
from app.ontology.constraints import (
    build_constraints,
)
from app.ontology.ontology_loader import (
    load_ontology,
)
from app.retrieval.medonto_retriever import (
    retrieve_with_ontology_state,
)
from app.validation.citation_validator import (
    validate_citations,
)
from app.validation.ontology_validator import (
    validate_ontology_compliance,
)
from app.validation.response_validator import (
    validate_response,
)


DEFAULT_MAX_AGENT_RETRIES = 1


def initialize_node(
    state: MedOntoState,
) -> dict:
    """
    Initialize bounded workflow-control fields.
    """
    settings = get_settings()

    max_retries = getattr(
        settings,
        "max_agent_retries",
        DEFAULT_MAX_AGENT_RETRIES,
    )

    return {
        "retry_count": 0,
        "max_retries": int(
            max_retries
        ),
        "orchestration_status": (
            "RUNNING"
        ),
        "validation_passed": False,
        "validation_reason": None,
        "abstention_reason": None,
        "generation_abstained": False,
        "citation_validation": None,
        "ontology_validation": None,
        "response_validation": None,
    }


def concept_mapping_node(
    state: MedOntoState,
) -> dict:
    """
    Build the ontology context.
    """
    graph = load_ontology()

    context = build_ontology_context(
        state["query"],
        graph=graph,
    )

    return {
        "ontology_context": context,
    }


def constraint_generation_node(
    state: MedOntoState,
) -> dict:
    """
    Build explicit ontology constraints.
    """
    graph = load_ontology()

    constraint_set = build_constraints(
        query=state["query"],
        ontology_context=(
            state[
                "ontology_context"
            ]
        ),
        graph=graph,
    )

    return {
        "constraint_set": (
            constraint_set
        ),
    }


def retrieval_node(
    state: MedOntoState,
) -> dict:
    """
    Execute ontology-aware retrieval.
    """
    result = (
        retrieve_with_ontology_state(
            query=state["query"],
            ontology_context=(
                state[
                    "ontology_context"
                ]
            ),
            constraint_set=(
                state[
                    "constraint_set"
                ]
            ),
        )
    )

    return {
        "retrieval_result": (
            result.retrieval
        ),
        "used_ontology_filter": (
            result.used_ontology_filter
        ),
        "retrieval_relaxed": (
            result.retrieval_relaxed
        ),
    }


def grounded_generation_node(
    state: MedOntoState,
) -> dict:
    """
    Generate a real grounded structured draft.
    """
    draft = generate_medonto_draft(
        query=state["query"],
        retrieval=(
            state[
                "retrieval_result"
            ]
        ),
        constraint_set=(
            state[
                "constraint_set"
            ]
        ),
        validation_feedback=(
            state.get(
                "validation_reason"
            )
        ),
    )

    return {
        "draft_output": draft,
        "generation_abstained": (
            draft.status
            == "ABSTAINED"
        ),
    }


def validation_node(
    state: MedOntoState,
) -> dict:
    """
    Run independent citation and ontology validation and then
    combine their outputs deterministically.
    """
    draft = state.get(
        "draft_output"
    )

    retrieval = state.get(
        "retrieval_result"
    )

    constraint_set = state.get(
        "constraint_set"
    )

    if draft is None:
        return {
            "validation_passed": False,
            "validation_reason": (
                "Structured draft output is missing."
            ),
            "response_validation": None,
        }

    if retrieval is None:
        return {
            "validation_passed": False,
            "validation_reason": (
                "Retrieval result is missing."
            ),
            "response_validation": None,
        }

    if constraint_set is None:
        return {
            "validation_passed": False,
            "validation_reason": (
                "Ontology constraint set is missing."
            ),
            "response_validation": None,
        }

    citation_validation = (
        validate_citations(
            draft=draft,
            retrieval=retrieval,
        )
    )

    ontology_validation = (
        validate_ontology_compliance(
            draft=draft,
            constraint_set=(
                constraint_set
            ),
        )
    )

    response_validation = (
        validate_response(
            citation_validation=(
                citation_validation
            ),
            ontology_validation=(
                ontology_validation
            ),
            retry_count=(
                state.get(
                    "retry_count",
                    0,
                )
            ),
            max_retries=(
                state.get(
                    "max_retries",
                    1,
                )
            ),
        )
    )

    return {
        "citation_validation": (
            citation_validation
        ),
        "ontology_validation": (
            ontology_validation
        ),
        "response_validation": (
            response_validation
        ),
        "validation_passed": (
            response_validation.status
            == "VERIFIED"
        ),
        "validation_reason": (
            response_validation.feedback
        ),
    }


def increment_retry_node(
    state: MedOntoState,
) -> dict:
    """
    Consume one retry while preserving validation feedback.

    Old validation objects are cleared because they refer to
    the previous draft.
    """
    return {
        "retry_count": (
            state.get(
                "retry_count",
                0,
            )
            + 1
        ),
        "validation_passed": False,
        "generation_abstained": False,
        "citation_validation": None,
        "ontology_validation": None,
        "response_validation": None,
    }


def verified_node(
    state: MedOntoState,
) -> dict:
    """
    Terminal successful validation state.
    """
    return {
        "orchestration_status": (
            "VERIFIED"
        ),
    }


def abstain_node(
    state: MedOntoState,
) -> dict:
    """
    Safely terminate without fabricating an answer.
    """
    retrieval = state.get(
        "retrieval_result"
    )

    draft = state.get(
        "draft_output"
    )

    response_validation = state.get(
        "response_validation"
    )

    if (
        retrieval is None
        or retrieval.result_count == 0
    ):
        reason = (
            "No sufficient retrieved evidence "
            "was available."
        )

    elif (
        state.get(
            "generation_abstained",
            False,
        )
        and draft is not None
    ):
        reason = draft.answer

    elif (
        response_validation is not None
        and response_validation.feedback
    ):
        reason = (
            response_validation.feedback
        )

    else:
        reason = (
            "Validation did not pass within "
            "the configured retry budget."
        )

    return {
        "orchestration_status": (
            "ABSTAINED"
        ),
        "abstention_reason": reason,
    }