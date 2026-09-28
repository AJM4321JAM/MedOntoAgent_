from typing import Literal

from pydantic import BaseModel, Field


class CitationValidationItem(BaseModel):
    """
    Independent semantic judgment for one generated claim.
    """

    claim_index: int = Field(
        ge=0,
    )

    claim_text: str = Field(
        min_length=1,
    )

    citation_chunk_ids: list[str] = Field(
        min_length=1,
    )

    verdict: Literal[
        "SUPPORTED",
        "UNSUPPORTED",
    ]

    reason: str = Field(
        min_length=1,
    )


class CitationValidationResult(BaseModel):
    """
    Complete claim-level citation validation result.
    """

    items: list[
        CitationValidationItem
    ]

    all_claims_supported: bool


class OntologyValidationItem(BaseModel):
    """
    Semantic compliance judgment for one explicit
    ontology constraint.
    """

    constraint_index: int = Field(
        ge=0,
    )

    constraint_type: str = Field(
        min_length=1,
    )

    subject_name: str = Field(
        min_length=1,
    )

    predicate_name: str = Field(
        min_length=1,
    )

    object_name: str = Field(
        min_length=1,
    )

    verdict: Literal[
        "COMPLIES",
        "CONFLICTS",
        "NOT_APPLICABLE",
    ]

    reason: str = Field(
        min_length=1,
    )


class OntologyValidationResult(BaseModel):
    """
    Complete ontology-compliance validation result.
    """

    items: list[
        OntologyValidationItem
    ]

    has_conflict: bool


class ResponseValidationResult(BaseModel):
    """
    Deterministic final validation-policy result.
    """

    status: Literal[
        "VERIFIED",
        "RETRY",
        "ABSTAINED",
    ]

    citation_validation: (
        CitationValidationResult
    )

    ontology_validation: (
        OntologyValidationResult
    )

    feedback: str | None = None