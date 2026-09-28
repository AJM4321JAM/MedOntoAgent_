from typing import Literal

from pydantic import BaseModel, Field


ConstraintType = Literal[
    "CONTRAINDICATION",
    "NOT_RECOMMENDED",
    "RECOMMENDATION",
    "TREATMENT_RELATION",
    "INVESTIGATION_REQUIREMENT",
    "OTHER",
]


class OntologyConstraint(BaseModel):
    """
    One explicit RDF relationship relevant to the
    current user query.
    """

    constraint_type: ConstraintType

    subject_name: str = Field(
        min_length=1,
    )

    subject_label: str = Field(
        min_length=1,
    )

    subject_uri: str = Field(
        min_length=1,
    )

    predicate_name: str = Field(
        min_length=1,
    )

    predicate_label: str = Field(
        min_length=1,
    )

    predicate_uri: str = Field(
        min_length=1,
    )

    object_name: str = Field(
        min_length=1,
    )

    object_label: str = Field(
        min_length=1,
    )

    object_uri: str = Field(
        min_length=1,
    )

    source_document_id: str | None = None

    source_section: str | None = None

    source_version: str | None = None

    assertion_status: str | None = None


class RetrievalHints(BaseModel):
    """
    Ontology-derived information that may later help
    retrieval.

    These are not yet Qdrant payload filters.
    """

    concept_names: list[str] = Field(
        default_factory=list,
    )

    concept_labels: list[str] = Field(
        default_factory=list,
    )

    relationship_terms: list[str] = Field(
        default_factory=list,
    )

    expanded_query: str = Field(
        min_length=1,
    )


class OntologyConstraintSet(BaseModel):
    """
    Complete ontology-constraint result for one query.
    """

    query: str = Field(
        min_length=1,
    )

    constraints: list[
        OntologyConstraint
    ] = Field(
        default_factory=list,
    )

    retrieval_hints: RetrievalHints

    constraint_count: int = Field(
        ge=0,
    )