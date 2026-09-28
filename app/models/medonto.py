from pydantic import BaseModel, Field

from app.models.constraints import (
    OntologyConstraintSet,
)
from app.models.ontology import (
    OntologyContext,
)
from app.models.retrieval import (
    RetrievalResult,
)


class MedOntoRetrievalResult(BaseModel):
    """
    Complete result of ontology-aware retrieval.
    """

    query: str = Field(
        min_length=1,
    )

    ontology_context: OntologyContext

    constraint_set: OntologyConstraintSet

    retrieval: RetrievalResult

    used_ontology_filter: bool

    retrieval_relaxed: bool