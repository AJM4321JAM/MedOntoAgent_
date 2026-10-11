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
    create_openrouter_client,
)
from app.llm.prompts import (
    ONTOLOGY_VALIDATION_SYSTEM_PROMPT,
)
from app.models.constraints import (
    OntologyConstraintSet,
)
from app.models.responses import (
    MedOntoDraftOutput,
)
from app.models.validation import (
    OntologyValidationItem,
    OntologyValidationResult,
)
from app.observability.tracing import (
    observed_span,
)


class OntologyValidationError(
    RuntimeError
):
    """
    Raised when ontology validation cannot be completed safely.
    """


class OntologyJudgeOutput(BaseModel):
    verdict: str = Field(
        pattern=(
            "^(COMPLIES|CONFLICTS|NOT_APPLICABLE)$"
        ),
    )

    reason: str = Field(
        min_length=1,
    )


def _format_generated_response(
    draft: MedOntoDraftOutput,
) -> str:
    claim_lines = [
        f"- {claim.claim_text}"
        for claim
        in draft.claims
    ]

    claims = (
        "\n".join(
            claim_lines
        )
        if claim_lines
        else "None."
    )

    return (
        f"ANSWER:\n{draft.answer}\n\n"
        f"CLAIMS:\n{claims}"
    )


def _judge_constraint(
    *,
    generated_response: str,
    constraint_type: str,
    subject_name: str,
    predicate_name: str,
    object_name: str,
    settings: Settings,
) -> OntologyJudgeOutput:
    client = create_openrouter_client(
        settings
    )

    user_prompt = f"""
GENERATED RESPONSE:
{generated_response}

ONTOLOGY CONSTRAINT:
constraint_type: {constraint_type}
subject_name: {subject_name}
predicate_name: {predicate_name}
object_name: {object_name}

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
                                .openrouter_model
                            ),
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        ONTOLOGY_VALIDATION_SYSTEM_PROMPT
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
                        "ontology validation"
                    ),
                )
            )

            if not completion.choices:
                raise OntologyValidationError(
                    (
                        "Ontology validator returned "
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
                raise OntologyValidationError(
                    (
                        "Ontology validator returned "
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
                    OntologyJudgeOutput
                    .model_validate(
                        raw_data
                    )
                )

            except ValidationError as exc:
                raise OntologyValidationError(
                    (
                        "Ontology validator output "
                        "did not match the required schema."
                    )
                ) from exc

        except (
            LLMServiceError,
            OntologyValidationError,
        ) as exc:
            last_error = exc

            if attempt == total_attempts:
                break

    raise OntologyValidationError(
        (
            "Ontology validation did not produce "
            "a valid judgment after one bounded retry. "
            f"Last error: {last_error}"
        )
    ) from last_error


def validate_ontology_compliance(
    *,
    draft: MedOntoDraftOutput,
    constraint_set: OntologyConstraintSet,
    settings: Settings | None = None,
) -> OntologyValidationResult:
    """
    Independently check the generated response against every
    explicit ontology constraint.
    """
    settings = (
        settings
        or get_settings()
    )

    if draft.status != "DRAFT":
        return OntologyValidationResult(
            items=[],
            has_conflict=False,
        )

    eligible = [
        constraint
        for constraint
        in constraint_set.constraints
        if constraint.retrieval_eligible
    ]

    if not eligible:
        return OntologyValidationResult(
            items=[],
            has_conflict=False,
        )

    generated_response = (
        _format_generated_response(
            draft
        )
    )

    items: list[
        OntologyValidationItem
    ] = []

    for index, constraint in enumerate(
        eligible
    ):
        with observed_span(
            "llm.ontology_validation",
            attributes={
                "medonto.model": (
                    settings.openrouter_model
                ),
                "medonto.operation": (
                    "ontology_validation"
                ),
                "medonto.constraint_index": index,
            },
        ):
            judgment = _judge_constraint(
                generated_response=(
                    generated_response
                ),
                constraint_type=(
                    constraint.constraint_type
                ),
                subject_name=(
                    constraint.subject_name
                ),
                predicate_name=(
                    constraint.predicate_name
                ),
                object_name=(
                    constraint.object_name
                ),
                settings=settings,
            )

        items.append(
            OntologyValidationItem(
                constraint_index=index,
                constraint_type=(
                    constraint.constraint_type
                ),
                subject_name=(
                    constraint.subject_name
                ),
                predicate_name=(
                    constraint.predicate_name
                ),
                object_name=(
                    constraint.object_name
                ),
                verdict=judgment.verdict,
                reason=judgment.reason,
            )
        )

    return OntologyValidationResult(
        items=items,
        has_conflict=any(
            item.verdict
            == "CONFLICTS"
            for item in items
        ),
    )