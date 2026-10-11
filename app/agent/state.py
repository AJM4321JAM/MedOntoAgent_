from typing import (
    Literal,
    TypedDict,
)

from app.models.constraints import (
    OntologyConstraintSet,
)
from app.models.ontology import (
    OntologyContext,
)
from app.models.responses import (
    MedOntoDraftOutput,
)
from app.models.retrieval import (
    RetrievalResult,
)
from app.models.validation import (
    CitationValidationResult,
    OntologyValidationResult,
    ResponseValidationResult,
)


OrchestrationStatus = Literal[
    "RUNNING",
    "VERIFIED",
    "ABSTAINED",
    "BASELINE_ROUTED",
    "ERROR",
]


class MedOntoState(
    TypedDict,
    total=False,
):
    """
    Shared state carried through the validated MedOntoAgent graph.
    """

    query: str

    ontology_context: OntologyContext
    constraint_set: OntologyConstraintSet

    retrieval_result: RetrievalResult
    used_ontology_filter: bool
    retrieval_relaxed: bool

    draft_output: MedOntoDraftOutput
    generation_abstained: bool

    citation_validation: (
        CitationValidationResult
        | None
    )

    ontology_validation: (
        OntologyValidationResult
        | None
    )

    response_validation: (
        ResponseValidationResult
        | None
    )

    validation_passed: bool
    validation_reason: str | None

    retry_count: int
    max_retries: int

    orchestration_status: (
        OrchestrationStatus
    )

    abstention_reason: str | None