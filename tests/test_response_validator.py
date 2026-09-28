from app.models.validation import (
    CitationValidationItem,
    CitationValidationResult,
    OntologyValidationItem,
    OntologyValidationResult,
)
from app.validation.response_validator import (
    validate_response,
)


def supported_citations() -> (
    CitationValidationResult
):
    return CitationValidationResult(
        items=[
            CitationValidationItem(
                claim_index=0,
                claim_text=(
                    "Supported claim."
                ),
                citation_chunk_ids=[
                    "CHUNK-1"
                ],
                verdict="SUPPORTED",
                reason=(
                    "The evidence supports "
                    "the claim."
                ),
            )
        ],
        all_claims_supported=True,
    )


def unsupported_citations() -> (
    CitationValidationResult
):
    return CitationValidationResult(
        items=[
            CitationValidationItem(
                claim_index=0,
                claim_text=(
                    "Unsupported claim."
                ),
                citation_chunk_ids=[
                    "CHUNK-1"
                ],
                verdict="UNSUPPORTED",
                reason=(
                    "The evidence does not "
                    "support the claim."
                ),
            )
        ],
        all_claims_supported=False,
    )


def compliant_ontology() -> (
    OntologyValidationResult
):
    return OntologyValidationResult(
        items=[
            OntologyValidationItem(
                constraint_index=0,
                constraint_type=(
                    "CONTRAINDICATION"
                ),
                subject_name="TreatmentX",
                predicate_name=(
                    "contraindicatedIn"
                ),
                object_name="GroupY",
                verdict="COMPLIES",
                reason=(
                    "The response is consistent "
                    "with the ontology constraint."
                ),
            )
        ],
        has_conflict=False,
    )


def conflicting_ontology() -> (
    OntologyValidationResult
):
    return OntologyValidationResult(
        items=[
            OntologyValidationItem(
                constraint_index=0,
                constraint_type=(
                    "CONTRAINDICATION"
                ),
                subject_name="TreatmentX",
                predicate_name=(
                    "contraindicatedIn"
                ),
                object_name="GroupY",
                verdict="CONFLICTS",
                reason=(
                    "The response reverses "
                    "the constraint."
                ),
            )
        ],
        has_conflict=True,
    )


def test_supported_and_compliant_is_verified() -> None:
    result = validate_response(
        citation_validation=(
            supported_citations()
        ),
        ontology_validation=(
            compliant_ontology()
        ),
        retry_count=0,
        max_retries=1,
    )

    assert (
        result.status
        == "VERIFIED"
    )

    assert result.feedback is None


def test_unsupported_claim_retries_when_budget_remains() -> None:
    result = validate_response(
        citation_validation=(
            unsupported_citations()
        ),
        ontology_validation=(
            compliant_ontology()
        ),
        retry_count=0,
        max_retries=1,
    )

    assert (
        result.status
        == "RETRY"
    )

    assert result.feedback is not None

    assert (
        "unsupported"
        in result.feedback.lower()
    )


def test_ontology_conflict_retries_when_budget_remains() -> None:
    result = validate_response(
        citation_validation=(
            supported_citations()
        ),
        ontology_validation=(
            conflicting_ontology()
        ),
        retry_count=0,
        max_retries=1,
    )

    assert (
        result.status
        == "RETRY"
    )

    assert result.feedback is not None

    assert (
        "ontology conflict"
        in result.feedback.lower()
    )


def test_failure_abstains_when_retry_budget_exhausted() -> None:
    result = validate_response(
        citation_validation=(
            unsupported_citations()
        ),
        ontology_validation=(
            conflicting_ontology()
        ),
        retry_count=1,
        max_retries=1,
    )

    assert (
        result.status
        == "ABSTAINED"
    )

    assert result.feedback is not None


def test_not_applicable_ontology_can_still_verify() -> None:
    ontology = OntologyValidationResult(
        items=[
            OntologyValidationItem(
                constraint_index=0,
                constraint_type=(
                    "CONTRAINDICATION"
                ),
                subject_name="TreatmentX",
                predicate_name=(
                    "contraindicatedIn"
                ),
                object_name="GroupY",
                verdict=(
                    "NOT_APPLICABLE"
                ),
                reason=(
                    "No relevant assertion "
                    "was made."
                ),
            )
        ],
        has_conflict=False,
    )

    result = validate_response(
        citation_validation=(
            supported_citations()
        ),
        ontology_validation=ontology,
        retry_count=0,
        max_retries=1,
    )

    assert (
        result.status
        == "VERIFIED"
    )