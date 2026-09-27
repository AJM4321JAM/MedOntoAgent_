from typing import Literal

from pydantic import BaseModel, Field


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
    Structured output required directly from the LLM.
    """

    answer: str = Field(
        min_length=1,
    )

    citations: list[CitationReference]

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

    citations: list[CitationReference]

    retrieved_chunk_ids: list[str]

    status: Literal[
        "ANSWERED",
        "ABSTAINED",
    ]