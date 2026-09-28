from typing import Literal

from pydantic import BaseModel, Field

from app.models.constraints import (
    ConstraintType,
)


class CitationReference(BaseModel):
    """
    Reference to one retrieved evidence chunk used
    in an answer.
    """

    chunk_id: str = Field(
        min_length=1,
    )


class BaselineLLMOutput(BaseModel):
    """
    Structured output required directly from the baseline LLM.
    """

    answer: str = Field(
        min_length=1,
    )

    citations: list[
        CitationReference
    ]

    status: Literal[
        "ANSWERED",
        "ABSTAINED",
    ]


class BaselineQueryResponse(BaseModel):
    """
    Public API response returned by the conventional
    baseline RAG system.
    """

    query: str = Field(
        min_length=1,
    )

    answer: str = Field(
        min_length=1,
    )

    citations: list[
        CitationReference
    ]

    retrieved_chunk_ids: list[str]

    status: Literal[
        "ANSWERED",
        "ABSTAINED",
    ]


class MedOntoClaim(BaseModel):
    """
    One factual claim in a MedOntoAgent draft together
    with the retrieved chunks claimed to support it.
    """

    claim_text: str = Field(
        min_length=1,
    )

    citation_chunk_ids: list[str] = Field(
        min_length=1,
    )


class MedOntoConstraintReference(BaseModel):
    """
    One ontology constraint reported back by generation.

    Python later verifies that this exactly corresponds to
    an application-supplied ontology constraint.
    """

    constraint_type: ConstraintType

    subject_name: str = Field(
        min_length=1,
    )

    predicate_name: str = Field(
        min_length=1,
    )

    object_name: str = Field(
        min_length=1,
    )


class MedOntoDraftOutput(BaseModel):
    """
    Structured candidate answer produced by MedOntoAgent.

    DRAFT is not equivalent to VERIFIED.
    """

    answer: str = Field(
        min_length=1,
    )

    claims: list[
        MedOntoClaim
    ] = Field(
        default_factory=list,
    )

    considered_constraints: list[
        MedOntoConstraintReference
    ] = Field(
        default_factory=list,
    )

    status: Literal[
        "DRAFT",
        "ABSTAINED",
    ]