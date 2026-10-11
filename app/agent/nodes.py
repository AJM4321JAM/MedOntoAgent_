from app.agent.state import (
    MedOntoState,
)
from app.config.settings import (
    get_settings,
)
from app.llm.medonto_generation import (
    generate_medonto_draft,
)
from app.observability.metrics import (
    record_retrieval,
)
from app.observability.tracing import (
    observed_span,
)
from app.ingestion.registry import (
    DEFAULT_REGISTRY_PATH,
    DocumentRegistry,
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
    with observed_span(
        "agent.initialize"
    ):
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
    with observed_span(
        "agent.map_concepts",
        attributes={
            "medonto.query.length": len(
                state["query"]
            ),
        },
    ):
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
    with observed_span(
        "agent.build_constraints"
    ) as span:
        graph = load_ontology()

        registry = DocumentRegistry(
            DEFAULT_REGISTRY_PATH
        )

        constraint_set = build_constraints(
            query=state["query"],
            ontology_context=(
                state[
                    "ontology_context"
                ]
            ),
            graph=graph,
            registry=registry,
        )

        span.set_attribute(
            "medonto.ontology.constraint_count",
            len(
                constraint_set.constraints
            ),
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
    with observed_span(
        "agent.retrieve"
    ) as span:
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

        span.set_attribute(
            "medonto.retrieval.result_count",
            result.retrieval.result_count,
        )

        span.set_attribute(
            "medonto.retrieval.ontology_filter",
            result.used_ontology_filter,
        )

        span.set_attribute(
            "medonto.retrieval.relaxed",
            result.retrieval_relaxed,
        )

        record_retrieval(
            result_count=(
                result.retrieval.result_count
            ),
            ontology_filter=(
                result.used_ontology_filter
            ),
            relaxed=(
                result.retrieval_relaxed
            ),
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
    with observed_span(
        "agent.generate",
        attributes={
            "medonto.retry_count": (
                state.get(
                    "retry_count",
                    0,
                )
            ),
        },
    ) as span:
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

        span.set_attribute(
            "medonto.generation.status",
            draft.status,
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
    with observed_span(
        "agent.validate"
    ) as span:
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
            span.set_attribute(
                "medonto.validation.status",
                "MISSING_DRAFT",
            )

            return {
                "validation_passed": False,
                "validation_reason": (
                    "Structured draft output is missing."
                ),
                "response_validation": None,
            }

        if retrieval is None:
            span.set_attribute(
                "medonto.validation.status",
                "MISSING_RETRIEVAL",
            )

            return {
                "validation_passed": False,
                "validation_reason": (
                    "Retrieval result is missing."
                ),
                "response_validation": None,
            }

        if constraint_set is None:
            span.set_attribute(
                "medonto.validation.status",
                "MISSING_CONSTRAINTS",
            )

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

        span.set_attribute(
            "medonto.validation.status",
            response_validation.status,
        )

        span.set_attribute(
            "medonto.validation.passed",
            (
                response_validation.status
                == "VERIFIED"
            ),
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
    next_count = (
        state.get(
            "retry_count",
            0,
        )
        + 1
    )

    with observed_span(
        "agent.retry",
        attributes={
            "medonto.retry_count": (
                next_count
            ),
        },
    ):
        return {
            "retry_count": (
                next_count
            ),
            "validation_passed": False,
            "generation_abstained": False,
            "citation_validation": None,
            "ontology_validation": None,
            "response_validation": None,
        }


def baseline_routed_node(
    state: MedOntoState,
) -> dict:
    """
    Terminal graph state indicating that no governed ontology
    constraint applies and conventional baseline RAG should run.
    """
    with observed_span(
        "agent.baseline_routed"
    ):
        return {
            "orchestration_status": (
                "BASELINE_ROUTED"
            ),
        }


def verified_node(
    state: MedOntoState,
) -> dict:
    """
    Terminal successful validation state.
    """
    with observed_span(
        "agent.verified"
    ):
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