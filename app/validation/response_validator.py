from app.models.validation import (
    CitationValidationResult,
    OntologyValidationResult,
    ResponseValidationResult,
)


def validate_response(
    *,
    citation_validation: CitationValidationResult,
    ontology_validation: OntologyValidationResult,
    retry_count: int,
    max_retries: int,
) -> ResponseValidationResult:
    """
    Deterministically combine semantic validator results.

    VERIFIED:
        every claim is supported and there are no ontology
        conflicts.

    RETRY:
        validation failed but retry budget remains.

    ABSTAINED:
        validation failed and retry budget is exhausted.
    """
    problems: list[str] = []

    for item in (
        citation_validation.items
    ):
        if (
            item.verdict
            == "UNSUPPORTED"
        ):
            problems.append(
                (
                    f"Claim {item.claim_index + 1} "
                    f"is unsupported: "
                    f"{item.reason}"
                )
            )

    for item in (
        ontology_validation.items
    ):
        if (
            item.verdict
            == "CONFLICTS"
        ):
            problems.append(
                (
                    "Ontology conflict for "
                    f"{item.subject_name} "
                    f"{item.predicate_name} "
                    f"{item.object_name}: "
                    f"{item.reason}"
                )
            )

    if not problems:
        return ResponseValidationResult(
            status="VERIFIED",
            citation_validation=(
                citation_validation
            ),
            ontology_validation=(
                ontology_validation
            ),
            feedback=None,
        )

    feedback = "\n".join(
        problems
    )

    if retry_count < max_retries:
        status = "RETRY"
    else:
        status = "ABSTAINED"

    return ResponseValidationResult(
        status=status,
        citation_validation=(
            citation_validation
        ),
        ontology_validation=(
            ontology_validation
        ),
        feedback=feedback,
    )