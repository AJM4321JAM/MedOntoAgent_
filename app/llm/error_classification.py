"""Safe classification of application and provider failures."""

from typing import Literal

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from app.llm.client import LLMServiceError
from app.llm.baseline import BaselineRAGError
from app.llm.medonto_generation import MedOntoGenerationError
from app.retrieval.retriever import RetrievalError
from app.validation.citation_validator import CitationValidationError
from app.validation.ontology_validator import OntologyValidationError


ErrorCategory = Literal[
    "RATE_LIMIT",
    "TIMEOUT",
    "CONNECTION",
    "PROVIDER_HTTP_ERROR",
    "LLM_SERVICE_ERROR",
    "GENERATION_ERROR",
    "CITATION_VALIDATION",
    "ONTOLOGY_VALIDATION",
    "RETRIEVAL_ERROR",
    "OTHER_ERROR",
]


def classify_error(exc: BaseException) -> ErrorCategory:
    """Classify known failures without reading exception messages."""
    visited: set[int] = set()
    current: BaseException | None = exc
    chain: list[BaseException] = []

    while current is not None and id(current) not in visited:
        visited.add(id(current))
        chain.append(current)
        current = current.__cause__

    for error_type, category in (
        (RateLimitError, "RATE_LIMIT"),
        (APITimeoutError, "TIMEOUT"),
        (APIConnectionError, "CONNECTION"),
        (APIStatusError, "PROVIDER_HTTP_ERROR"),
    ):
        if any(isinstance(item, error_type) for item in chain):
            return category

    if any(isinstance(item, CitationValidationError) for item in chain):
        return "CITATION_VALIDATION"

    if any(isinstance(item, OntologyValidationError) for item in chain):
        return "ONTOLOGY_VALIDATION"

    if any(isinstance(item, RetrievalError) for item in chain):
        return "RETRIEVAL_ERROR"

    if any(isinstance(item, LLMServiceError) for item in chain):
        return "LLM_SERVICE_ERROR"

    if any(
        isinstance(item, (BaselineRAGError, MedOntoGenerationError))
        for item in chain
    ):
        return "GENERATION_ERROR"

    return "OTHER_ERROR"
