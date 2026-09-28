from typing import Literal, TypedDict

from app.models.constraints import (
    OntologyConstraintSet,
)
from app.models.ontology import (
    OntologyContext,
)
from app.models.retrieval import (
    RetrievalResult,
)


OrchestrationStatus = Literal[
    "RUNNING",
    "MOCK_VALIDATED",
    "ABSTAINED",
    "ERROR",
]


class MedOntoState(TypedDict, total=False):
    """
    Shared state carried through the bounded MedOntoAgent graph.

    Phase 15 uses deterministic mock generation and validation.
    """

    # Original request
    query: str

    # Ontology processing
    ontology_context: OntologyContext
    constraint_set: OntologyConstraintSet

    # Retrieval
    retrieval_result: RetrievalResult
    used_ontology_filter: bool
    retrieval_relaxed: bool

    # Temporary Phase-15 generation state
    draft_answer: str

    # Temporary Phase-15 validation state
    validation_passed: bool
    validation_reason: str | None

    # Bounded workflow control
    retry_count: int
    max_retries: int

    orchestration_status: OrchestrationStatus
    abstention_reason: str | None