from app.agent.state import (
    MedOntoState,
)
from app.config.settings import (
    get_settings,
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
    Initialize workflow-control state.

    If max_agent_retries is present in Settings, use it.
    Otherwise Phase 15 safely defaults to one regeneration.
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
    }


def concept_mapping_node(
    state: MedOntoState,
) -> dict:
    """
    Build Phase-12 OntologyContext.
    """
    query = (
        state["query"]
        .strip()
    )

    graph = load_ontology()

    context = (
        build_ontology_context(
            query,
            graph=graph,
        )
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
    query = state["query"]

    ontology_context = (
        state["ontology_context"]
    )

    graph = load_ontology()

    constraint_set = (
        build_constraints(
            query=query,
            ontology_context=(
                ontology_context
            ),
            graph=graph,
        )
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
    Perform Phase-14 ontology-aware retrieval using the
    ontology state already present in the graph.
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


def mock_generation_node(
    state: MedOntoState,
) -> dict:
    """
    Deterministic Phase-15 draft generation.

    No LLM call is made here.
    """
    retrieval = (
        state["retrieval_result"]
    )

    if (
        retrieval.result_count
        == 0
    ):
        return {
            "draft_answer": "",
        }

    first_chunk = (
        retrieval
        .retrieved_chunks[0]
    )

    draft = (
        "PHASE 15 MOCK DRAFT\n"
        f"Evidence chunk: "
        f"{first_chunk.chunk_id}\n"
        f"Evidence: "
        f"{first_chunk.text}"
    )

    return {
        "draft_answer": draft,
    }


def mock_validation_node(
    state: MedOntoState,
) -> dict:
    """
    Deterministically validate workflow mechanics only.

    This is not clinical verification.
    """
    retrieval = state.get(
        "retrieval_result"
    )

    draft = (
        state.get(
            "draft_answer",
            "",
        )
        .strip()
    )

    if (
        retrieval is None
        or retrieval.result_count == 0
    ):
        return {
            "validation_passed": False,
            "validation_reason": (
                "No retrieved evidence "
                "is available."
            ),
        }

    if not draft:
        return {
            "validation_passed": False,
            "validation_reason": (
                "No draft answer "
                "was generated."
            ),
        }

    if not draft.startswith(
        "PHASE 15 MOCK DRAFT"
    ):
        return {
            "validation_passed": False,
            "validation_reason": (
                "Draft is not the expected "
                "Phase-15 mock format."
            ),
        }

    return {
        "validation_passed": True,
        "validation_reason": (
            "Phase-15 orchestration "
            "requirements passed."
        ),
    }


def increment_retry_node(
    state: MedOntoState,
) -> dict:
    """
    Consume exactly one graph-level retry.
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
        "validation_reason": None,
    }


def mock_validated_node(
    state: MedOntoState,
) -> dict:
    """
    Temporary successful terminal status.

    MOCK_VALIDATED means only that orchestration mechanics
    passed.
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
    Terminate safely without producing a fabricated answer.
    """
    retrieval = state.get(
        "retrieval_result"
    )

    if (
        retrieval is None
        or retrieval.result_count == 0
    ):
        reason = (
            "No sufficient retrieved evidence "
            "was available."
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