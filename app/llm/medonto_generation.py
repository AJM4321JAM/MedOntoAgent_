from pydantic import ValidationError

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
    MEDONTO_GENERATION_SYSTEM_PROMPT,
)
from app.models.constraints import (
    OntologyConstraintSet,
)
from app.models.responses import (
    MedOntoConstraintReference,
    MedOntoDraftOutput,
)
from app.models.retrieval import (
    RetrievalResult,
)
from app.observability.tracing import (
    observed_span,
)


class MedOntoGenerationError(RuntimeError):
    """
    Raised when grounded generation cannot produce a safe,
    structurally valid draft.
    """


def format_medonto_evidence(
    retrieval: RetrievalResult,
) -> str:
    """
    Format only the evidence actually returned by retrieval.
    """
    sections: list[str] = []

    for chunk in (
        retrieval.retrieved_chunks
    ):
        sections.append(
            "\n".join(
                [
                    "<UNTRUSTED_RETRIEVED_EVIDENCE>",
                    (
                        f"SOURCE_ID: "
                        f"{chunk.chunk_id}"
                    ),
                    (
                        f"TITLE: "
                        f"{chunk.title}"
                    ),
                    "TEXT:",
                    chunk.text,
                    "</UNTRUSTED_RETRIEVED_EVIDENCE>",
                ]
            )
        )

    return "\n\n---\n\n".join(
        sections
    )


def get_control_constraints(
    constraint_set: OntologyConstraintSet,
):
    """
    Return only ontology constraints permitted to control
    generation behavior.
    """
    return [
        constraint
        for constraint
        in constraint_set.constraints
        if constraint.control_eligible
    ]


def format_ontology_constraints(
    constraint_set: OntologyConstraintSet,
) -> str:
    """
    Format application-generated RDF constraints for the model.
    """
    control_constraints = (
        get_control_constraints(
            constraint_set
        )
    )

    if not control_constraints:
        return (
            "No control-eligible ontology constraints "
            "were identified for this query."
        )

    sections: list[str] = []

    for constraint in (
        control_constraints
    ):
        sections.append(
            "\n".join(
                [
                    (
                        "constraint_type: "
                        f"{constraint.constraint_type}"
                    ),
                    (
                        "subject_name: "
                        f"{constraint.subject_name}"
                    ),
                    (
                        "predicate_name: "
                        f"{constraint.predicate_name}"
                    ),
                    (
                        "object_name: "
                        f"{constraint.object_name}"
                    ),
                    (
                        "recommendation_priority: "
                        f"{constraint.recommendation_priority}"
                    ),
                    (
                        "recommendation_condition: "
                        f"{constraint.recommendation_condition}"
                    ),
                ]
            )
        )

    return "\n\n".join(
        sections
    )


def _constraint_reference_key(
    reference: MedOntoConstraintReference,
) -> tuple[str, str, str, str]:
    return (
        reference.constraint_type,
        reference.subject_name,
        reference.predicate_name,
        reference.object_name,
    )


def _expected_constraint_keys(
    constraint_set: OntologyConstraintSet,
) -> set[
    tuple[
        str,
        str,
        str,
        str,
    ]
]:
    return {
        (
            constraint.constraint_type,
            constraint.subject_name,
            constraint.predicate_name,
            constraint.object_name,
        )
        for constraint
        in get_control_constraints(
            constraint_set
        )
    }


def validate_draft_references(
    *,
    draft: MedOntoDraftOutput,
    retrieval: RetrievalResult,
    constraint_set: OntologyConstraintSet,
) -> None:
    """
    Perform deterministic Phase-16 identity checks.

    This proves citation and ontology references are legitimate
    identifiers supplied by the application.

    It does NOT yet prove semantic entailment.
    """
    allowed_chunk_ids = {
        chunk.chunk_id
        for chunk
        in retrieval.retrieved_chunks
    }

    expected_constraints = (
        _expected_constraint_keys(
            constraint_set
        )
    )

    returned_constraint_keys = [
        _constraint_reference_key(
            reference
        )
        for reference
        in draft.considered_constraints
    ]

    returned_constraint_set = set(
        returned_constraint_keys
    )

    if (
        len(returned_constraint_keys)
        != len(returned_constraint_set)
    ):
        raise MedOntoGenerationError(
            (
                "Generation returned duplicate "
                "ontology constraint references."
            )
        )

    if (
        returned_constraint_set
        != expected_constraints
    ):
        raise MedOntoGenerationError(
            (
                "Generation did not reproduce exactly "
                "the ontology constraints supplied "
                "by the application."
            )
        )

    if draft.status == "ABSTAINED":
        if draft.claims:
            raise MedOntoGenerationError(
                (
                    "An ABSTAINED draft must not "
                    "contain factual claims."
                )
            )

        return

    if not draft.claims:
        raise MedOntoGenerationError(
            (
                "A DRAFT response must contain "
                "at least one factual claim."
            )
        )

    for claim in draft.claims:
        if not claim.citation_chunk_ids:
            raise MedOntoGenerationError(
                (
                    "Every DRAFT claim must contain "
                    "at least one citation."
                )
            )

        for chunk_id in (
            claim.citation_chunk_ids
        ):
            if (
                chunk_id
                not in allowed_chunk_ids
            ):
                raise MedOntoGenerationError(
                    (
                        "Generation cited a chunk "
                        "that was not retrieved: "
                        f"{chunk_id}"
                    )
                )


def _build_generation_user_prompt(
    *,
    query: str,
    retrieval: RetrievalResult,
    constraint_set: OntologyConstraintSet,
    validation_feedback: str | None,
) -> str:
    """
    Build the complete grounded generation request.
    """
    evidence = format_medonto_evidence(
        retrieval
    )

    constraints = (
        format_ontology_constraints(
            constraint_set
        )
    )

    feedback = (
        validation_feedback.strip()
        if validation_feedback
        else "None."
    )

    return f"""
QUESTION:
{query}

ONTOLOGY CONSTRAINTS:
{constraints}

APPROVED RETRIEVED EVIDENCE:
{evidence}

FEEDBACK FROM THE PREVIOUS VALIDATION ATTEMPT:
{feedback}

Return only the required JSON object.
""".strip()


def _deterministic_abstention(
    *,
    constraint_set: OntologyConstraintSet,
) -> MedOntoDraftOutput:
    """
    Abstain without calling the LLM when no evidence exists.
    """
    references = [
        MedOntoConstraintReference(
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
        )
        for constraint
        in get_control_constraints(
            constraint_set
        )
    ]

    return MedOntoDraftOutput(
        answer=(
            "The approved retrieved evidence is "
            "insufficient to answer the question."
        ),
        claims=[],
        considered_constraints=references,
        status="ABSTAINED",
    )


def generate_medonto_draft(
    *,
    query: str,
    retrieval: RetrievalResult,
    constraint_set: OntologyConstraintSet,
    validation_feedback: str | None = None,
    settings: Settings | None = None,
) -> MedOntoDraftOutput:
    """
    Generate one grounded MedOntoAgent draft.

    A single bounded regeneration is allowed when the free
    routed model returns malformed or structurally invalid
    output.

    Graph-level semantic retry remains separate.
    """
    settings = (
        settings
        or get_settings()
    )

    normalized_query = (
        query.strip()
    )

    if not normalized_query:
        raise ValueError(
            "Generation query cannot be empty."
        )

    if retrieval.result_count == 0:
        return _deterministic_abstention(
            constraint_set=(
                constraint_set
            )
        )

    client = create_llm_client(
        settings
    )

    user_prompt = (
        _build_generation_user_prompt(
            query=normalized_query,
            retrieval=retrieval,
            constraint_set=(
                constraint_set
            ),
            validation_feedback=(
                validation_feedback
            ),
        )
    )

    total_attempts = 2
    last_error: Exception | None = None

    for attempt in range(
        1,
        total_attempts + 1,
    ):
        try:
            with observed_span(
                "llm.medonto_generation",
                attributes={
                    "medonto.model": (
                        settings.gemini_model
                    ),
                    "medonto.operation": (
                        "generation"
                    ),
                },
            ):
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
                                            MEDONTO_GENERATION_SYSTEM_PROMPT
                                        ),
                                    },
                                    {
                                        "role": "user",
                                        "content": (
                                            user_prompt
                                        ),
                                    },
                                ],
                                max_tokens=(
                                    settings
                                    .llm_max_output_tokens
                                ),
                                response_format={
                                    "type": "json_object",
                                },
                            )
                        ),
                        operation_name=(
                            "MedOnto grounded generation"
                        ),
                    )
                )

            if not completion.choices:
                raise MedOntoGenerationError(
                    "Generation returned no choices."
                )

            content = (
                completion
                .choices[0]
                .message
                .content
            )

            if not content:
                raise MedOntoGenerationError(
                    (
                        "Generation returned an "
                        "empty response."
                    )
                )

            raw_data = (
                _extract_json_object(
                    content
                )
            )

            try:
                draft = (
                    MedOntoDraftOutput
                    .model_validate(
                        raw_data
                    )
                )

            except ValidationError as exc:
                raise MedOntoGenerationError(
                    (
                        "Generation JSON did not match "
                        "MedOntoDraftOutput."
                    )
                ) from exc

            validate_draft_references(
                draft=draft,
                retrieval=retrieval,
                constraint_set=(
                    constraint_set
                ),
            )

            return draft

        except (
            LLMServiceError,
            MedOntoGenerationError,
        ) as exc:
            last_error = exc

            if attempt == total_attempts:
                break

    raise MedOntoGenerationError(
        (
            "MedOnto generation did not produce a valid "
            "grounded draft after one bounded retry. "
            f"Last error: {last_error}"
        )
    ) from last_error