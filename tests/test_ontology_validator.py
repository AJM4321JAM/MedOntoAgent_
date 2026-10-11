from unittest.mock import (
    MagicMock,
    patch,
)

from app.config.settings import (
    Settings,
)
from app.models.constraints import (
    OntologyConstraint,
    OntologyConstraintSet,
    RetrievalHints,
)
from app.models.responses import (
    MedOntoClaim,
    MedOntoDraftOutput,
)
from app.validation.ontology_validator import (
    OntologyJudgeOutput,
    validate_ontology_compliance,
)


def create_settings() -> Settings:
    return Settings(
        _env_file=None,
        gemini_api_key="test-key",
        gemini_model="test-model",
    )


def create_constraints(
    *,
    include_constraint: bool = True,
) -> OntologyConstraintSet:
    constraints = []

    if include_constraint:
        constraints = [
            OntologyConstraint(
                constraint_type=(
                    "CONTRAINDICATION"
                ),
                subject_name="TreatmentX",
                subject_label="Treatment X",
                subject_uri=(
                    "https://example.org/"
                    "TreatmentX"
                ),
                predicate_name=(
                    "contraindicatedIn"
                ),
                predicate_label=(
                    "contraindicated in"
                ),
                predicate_uri=(
                    "https://example.org/"
                    "contraindicatedIn"
                ),
                object_name="GroupY",
                object_label="Group Y",
                object_uri=(
                    "https://example.org/"
                    "GroupY"
                ),
                source_document_id=(
                    "DOC-ACTIVE-TEST"
                ),
                source_section="Test",
                source_version="1.0",
                assertion_status=(
                    "ACTIVE"
                ),
                control_eligible=True,
            )
        ]

    return OntologyConstraintSet(
        query="test",
        constraints=constraints,
        retrieval_hints=RetrievalHints(
            concept_names=[],
            concept_labels=[],
            relationship_terms=[],
            expanded_query="test",
        ),
        constraint_count=len(
            constraints
        ),
    )


def create_draft() -> MedOntoDraftOutput:
    return MedOntoDraftOutput(
        answer=(
            "Treatment X is contraindicated "
            "in Group Y."
        ),
        claims=[
            MedOntoClaim(
                claim_text=(
                    "Treatment X is contraindicated "
                    "in Group Y."
                ),
                citation_chunk_ids=[
                    "CHUNK-1"
                ],
            )
        ],
        considered_constraints=[],
        status="DRAFT",
    )


@patch(
    "app.validation.ontology_validator."
    "_judge_constraint"
)
def test_compliant_answer_returns_no_conflict(
    mock_judge: MagicMock,
) -> None:
    mock_judge.return_value = (
        OntologyJudgeOutput(
            verdict="COMPLIES",
            reason=(
                "The answer is consistent "
                "with the constraint."
            ),
        )
    )

    result = (
        validate_ontology_compliance(
            draft=create_draft(),
            constraint_set=(
                create_constraints()
            ),
            settings=create_settings(),
        )
    )

    assert (
        result.has_conflict
        is False
    )

    assert (
        result.items[0].verdict
        == "COMPLIES"
    )


@patch(
    "app.validation.ontology_validator."
    "_judge_constraint"
)
def test_conflicting_answer_is_detected(
    mock_judge: MagicMock,
) -> None:
    mock_judge.return_value = (
        OntologyJudgeOutput(
            verdict="CONFLICTS",
            reason=(
                "The answer reverses the "
                "ontology constraint."
            ),
        )
    )

    result = (
        validate_ontology_compliance(
            draft=create_draft(),
            constraint_set=(
                create_constraints()
            ),
            settings=create_settings(),
        )
    )

    assert (
        result.has_conflict
        is True
    )

    assert (
        result.items[0].verdict
        == "CONFLICTS"
    )


@patch(
    "app.validation.ontology_validator."
    "_judge_constraint"
)
def test_not_applicable_does_not_create_conflict(
    mock_judge: MagicMock,
) -> None:
    mock_judge.return_value = (
        OntologyJudgeOutput(
            verdict="NOT_APPLICABLE",
            reason=(
                "The response makes no relevant "
                "statement about the relationship."
            ),
        )
    )

    result = (
        validate_ontology_compliance(
            draft=create_draft(),
            constraint_set=(
                create_constraints()
            ),
            settings=create_settings(),
        )
    )

    assert (
        result.has_conflict
        is False
    )

    assert (
        result.items[0].verdict
        == "NOT_APPLICABLE"
    )


@patch(
    "app.validation.ontology_validator."
    "_judge_constraint"
)
def test_no_constraints_requires_no_llm_validation(
    mock_judge: MagicMock,
) -> None:
    result = (
        validate_ontology_compliance(
            draft=create_draft(),
            constraint_set=(
                create_constraints(
                    include_constraint=False
                )
            ),
            settings=create_settings(),
        )
    )

    assert result.items == []

    assert (
        result.has_conflict
        is False
    )

    mock_judge.assert_not_called()

def test_validator_receives_recommendation_qualifiers() -> None:
    from unittest.mock import MagicMock, patch

    constraint_set = create_constraints()

    constraint_set.constraints[0] = (
        constraint_set.constraints[0].model_copy(
            update={
                "recommendation_priority": 2,
                "recommendation_condition": (
                    "If the primary option is not suitable"
                ),
            }
        )
    )

    with patch(
        "app.validation.ontology_validator."
        "_judge_constraint"
    ) as mock_judge:
        mock_judge.return_value = (
            OntologyJudgeOutput(
                verdict="COMPLIES",
                reason="Synthetic test.",
            )
        )

        validate_ontology_compliance(
            draft=create_draft(),
            constraint_set=constraint_set,
            settings=create_settings(),
        )

        mock_judge.assert_called_once()

        kwargs = (
            mock_judge.call_args.kwargs
        )

        assert (
            kwargs[
                "recommendation_priority"
            ]
            == 2
        )

        assert (
            kwargs[
                "recommendation_condition"
            ]
            == "If the primary option is not suitable"
        )
