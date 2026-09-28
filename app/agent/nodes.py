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
    }


def concept_mapping_node(
    state: MedOntoState,
) -> dict:
    """
    Build the Phase-12 OntologyContext.
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
    Build explicit Phase-13 ontology constraints.
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
    Execute Phase-14 ontology-aware retrieval.
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
    Execute real evidence-grounded generation.
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


def mock_validation_node(
    state: MedOntoState,
) -> dict:
    """
    Temporary Phase-16 structural validation.

    The generation module has already checked citation IDs and
    constraint identities.

    This node does NOT perform semantic claim entailment.
    """
    draft = state.get(
        "draft_output"
    )

    retrieval = state.get(
        "retrieval_result"
    )

    if draft is None:
        return {
            "validation_passed": False,
            "validation_reason": (
                "No structured draft was generated."
            ),
        }

    if (
        retrieval is None
        or retrieval.result_count == 0
    ):
        return {
            "validation_passed": False,
            "validation_reason": (
                "No retrieved evidence is available."
            ),
        }

    if draft.status != "DRAFT":
        return {
            "validation_passed": False,
            "validation_reason": (
                "Generated output is not a DRAFT."
            ),
        }

    if not draft.claims:
        return {
            "validation_passed": False,
            "validation_reason": (
                "DRAFT contains no claims."
            ),
        }

    return {
        "validation_passed": True,
        "validation_reason": (
            "Phase-16 structural checks passed. "
            "Semantic validation is not yet implemented."
        ),
    }


def increment_retry_node(
    state: MedOntoState,
) -> dict:
    """
    Consume one graph-level semantic retry.

    validation_reason is intentionally preserved so the next
    generation attempt receives corrective feedback.
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
    }


def mock_validated_node(
    state: MedOntoState,
) -> dict:
    """
    Temporary success status until Phase-17 validators exist.
    """
    return {
        "orchestration_status": (
            "MOCK_VALIDATED"
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