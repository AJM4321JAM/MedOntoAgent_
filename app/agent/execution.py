from dataclasses import dataclass
from typing import Any, Literal

from app.agent.graph import medonto_graph
from app.config.settings import Settings
from app.llm.baseline import (
    BaselineRAGError,
    run_baseline_rag,
)
from app.llm.client import LLMServiceError
from app.llm.medonto_generation import (
    MedOntoGenerationError,
)
from app.models.responses import (
    BaselineQueryResponse,
)
from app.ontology.concept_mapper import (
    ConceptMappingError,
)
from app.ontology.ontology_loader import (
    OntologyLoadError,
)
from app.retrieval.retriever import (
    RetrievalError,
)
from app.validation.citation_validator import (
    CitationValidationError,
)
from app.validation.ontology_validator import (
    OntologyValidationError,
)


ExecutionMode = Literal[
    "MEDONTO",
    "BASELINE_FALLBACK",
]


FALLBACK_EXCEPTIONS = (
    ConceptMappingError,
    OntologyLoadError,
    CitationValidationError,
    OntologyValidationError,
    RetrievalError,
    MedOntoGenerationError,
    LLMServiceError,
)


@dataclass(frozen=True)
class MedOntoExecution:
    """
    Result of executing the MedOnto workflow with a bounded,
    explicitly auditable baseline fallback.

    Exactly one of medonto_state or baseline_response is populated.
    """

    execution_mode: ExecutionMode

    medonto_state: dict[str, Any] | None = None

    baseline_response: (
        BaselineQueryResponse | None
    ) = None

    fallback_reason: str | None = None

    fallback_error: str | None = None


def execute_medonto(
    query: str,
    *,
    settings: Settings | None = None,
) -> MedOntoExecution:
    """
    Run MedOnto normally.

    Known MedOnto operational failures degrade to the unchanged
    conventional baseline RAG pipeline.

    Unexpected programming/invariant failures are deliberately not
    swallowed and must continue to surface as errors.
    """
    try:
        state = medonto_graph.invoke(
            {
                "query": query,
            }
        )

    except FALLBACK_EXCEPTIONS as exc:
        fallback_reason = (
            type(exc).__name__
        )

        try:
            baseline_response = run_baseline_rag(
                query,
                settings=settings,
            )

        except (
            BaselineRAGError,
            RetrievalError,
            LLMServiceError,
        ) as fallback_exc:
            return MedOntoExecution(
                execution_mode=(
                    "BASELINE_FALLBACK"
                ),
                baseline_response=None,
                fallback_reason=(
                    fallback_reason
                ),
                fallback_error=(
                    type(fallback_exc).__name__
                ),
            )

        return MedOntoExecution(
            execution_mode=(
                "BASELINE_FALLBACK"
            ),
            baseline_response=(
                baseline_response
            ),
            fallback_reason=(
                fallback_reason
            ),
            fallback_error=None,
        )

    return MedOntoExecution(
        execution_mode="MEDONTO",
        medonto_state=state,
    )
