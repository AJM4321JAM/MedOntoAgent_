from pydantic import (
    BaseModel,
    Field,
    ValidationError,
)

from app.config.settings import (
    Settings,
    get_settings,
)
from app.llm.client import (
    LLMServiceError,
    _call_with_one_retry,
    _extract_json_object,
    create_llm_client,
)
from app.llm.prompts import (
    CITATION_VALIDATION_SYSTEM_PROMPT,
)
from app.models.responses import (
    MedOntoDraftOutput,
)
from app.models.retrieval import (
    RetrievalResult,
)
from app.models.validation import (
    CitationValidationItem,
    CitationValidationResult,
)
from app.observability.tracing import (
    observed_span,
)


class CitationValidationError(
    RuntimeError
):
    """
    Raised when citation validation cannot be completed safely.
    """


class CitationJudgeOutput(BaseModel):
    verdict: str = Field(
        pattern=(
            "^(SUPPORTED|UNSUPPORTED)$"
        ),
    )

    reason: str = Field(
        min_length=1,
    )


def _evidence_for_claim(
    *,
    citation_chunk_ids: list[str],
    retrieval: RetrievalResult,
) -> str:
    """
    Give the validator only the chunks cited by this claim.
    """
    chunk_lookup = {
        chunk.chunk_id: chunk
        for chunk
        in retrieval.retrieved_chunks
    }

    sections: list[str] = []

    for chunk_id in citation_chunk_ids:
        chunk = chunk_lookup.get(
            chunk_id
        )

        if chunk is None:
            raise CitationValidationError(
                (
                    "Citation validation received "
                    "a chunk ID that was not retrieved: "
                    f"{chunk_id}"
                )
            )

        sections.append(
            "\n".join(
                [
                    f"SOURCE_ID: {chunk.chunk_id}",
                    f"TITLE: {chunk.title}",
                    "TEXT:",
                    chunk.text,
                ]
            )
        )

    return "\n\n---\n\n".join(
        sections
    )


def _judge_claim(
    *,
    claim_text: str,
    evidence: str,
    settings: Settings,
) -> CitationJudgeOutput:
    client = create_llm_client(
        settings
    )

    user_prompt = f"""
CLAIM:
{claim_text}

CITED RETRIEVED EVIDENCE:
{evidence}

Return only the required JSON object.
""".strip()

    total_attempts = 2
    last_error: Exception | None = None

    for attempt in range(
        1,
        total_attempts + 1,
    ):
        try:
            completion = (
                _call_with_one_retry(
                    lambda: (
                        client
                        .chat
                        .completions
                        .create(
                            model=(
                                settings
                                .gemini_model
                            ),
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        CITATION_VALIDATION_SYSTEM_PROMPT
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        user_prompt
                                    ),
                                },
                            ],
                            max_tokens=min(
                                settings
                                .llm_max_output_tokens,
                                500,
                            ),
                            response_format={
                                "type": "json_object",
                            },
                        )
                    ),
                    operation_name=(
                        "citation validation"
                    ),
                )
            )

            if not completion.choices:
                raise CitationValidationError(
                    (
                        "Citation validator returned "
                        "no choices."
                    )
                )

            content = (
                completion
                .choices[0]
                .message
                .content
            )

            if not content:
                raise CitationValidationError(
                    (
                        "Citation validator returned "
                        "an empty response."
                    )
                )

            raw_data = (
                _extract_json_object(
                    content
                )
            )

            try:
                return (
                    CitationJudgeOutput
                    .model_validate(
                        raw_data
                    )
                )

            except ValidationError as exc:
                raise CitationValidationError(
                    (
                        "Citation validator output "
                        "did not match the required schema."
                    )
                ) from exc

        except (
            LLMServiceError,
            CitationValidationError,
        ) as exc:
            last_error = exc

            if attempt == total_attempts:
                break

    raise CitationValidationError(
        (
            "Citation validation did not produce "
            "a valid judgment after one bounded retry. "
            f"Last error: {last_error}"
        )
    ) from last_error


def validate_citations(
    *,
    draft: MedOntoDraftOutput,
    retrieval: RetrievalResult,
    settings: Settings | None = None,
) -> CitationValidationResult:
    """
    Validate each generated claim independently against only
    the retrieved chunks cited by that claim.
    """
    settings = (
        settings
        or get_settings()
    )

    if draft.status != "DRAFT":
        return CitationValidationResult(
            items=[],
            all_claims_supported=True,
        )

    items: list[
        CitationValidationItem
    ] = []

    for index, claim in enumerate(
        draft.claims
    ):
        evidence = _evidence_for_claim(
            citation_chunk_ids=(
                claim.citation_chunk_ids
            ),
            retrieval=retrieval,
        )

        with observed_span(
            "llm.citation_validation",
            attributes={
                "medonto.model": (
                    settings.gemini_model
                ),
                "medonto.operation": (
                    "citation_validation"
                ),
                "medonto.claim_index": index,
            },
        ):
            judgment = _judge_claim(
                claim_text=(
                    claim.claim_text
                ),
                evidence=evidence,
                settings=settings,
            )

        items.append(
            CitationValidationItem(
                claim_index=index,
                claim_text=(
                    claim.claim_text
                ),
                citation_chunk_ids=(
                    claim.citation_chunk_ids
                ),
                verdict=judgment.verdict,
                reason=judgment.reason,
            )
        )

    return CitationValidationResult(
        items=items,
        all_claims_supported=all(
            item.verdict
            == "SUPPORTED"
            for item in items
        ),
    )