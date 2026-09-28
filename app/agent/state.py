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


OrchestrationStatus = Literal[
    "RUNNING",
    "MOCK_VALIDATED",
    "ABSTAINED",
    "ERROR",
]


class MedOntoState(
    TypedDict,
    total=False,
):
    """
    Shared state carried through MedOntoAgent.

    Phase 16 uses real grounded generation but still uses
    temporary structural/mock validation.
    """

    query: str

    ontology_context: OntologyContext
    constraint_set: OntologyConstraintSet

    retrieval_result: RetrievalResult
    used_ontology_filter: bool
    retrieval_relaxed: bool

    draft_output: MedOntoDraftOutput
    generation_abstained: bool

    validation_passed: bool
    validation_reason: str | None

    retry_count: int
    max_retries: int

    orchestration_status: (
        OrchestrationStatus
    )

    abstention_reason: str | None