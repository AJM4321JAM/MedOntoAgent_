from unittest.mock import (
    MagicMock,
    patch,
)

import pytest

from app.config.settings import (
    Settings,
)
from app.llm.medonto_generation import (
    MedOntoGenerationError,
    format_medonto_evidence,
    format_ontology_constraints,
    generate_medonto_draft,
    validate_draft_references,
)
from app.models.constraints import (
    OntologyConstraint,
    OntologyConstraintSet,
    RetrievalHints,
)
from app.models.documents import (
    SourceAuthority,
)
from app.models.responses import (
    MedOntoClaim,
    MedOntoConstraintReference,
    MedOntoDraftOutput,
)
from app.models.retrieval import (
    RetrievedChunk,
    RetrievalResult,
)


def create_settings() -> Settings:
    return Settings(
        _env_file=None,
        openrouter_api_key="test-key",
        openrouter_model="test-model",
    )


def create_retrieval(
    *,
    with_evidence: bool = True,
) -> RetrievalResult:
    if not with_evidence:
        return RetrievalResult(
            query="test",
            top_k=5,
            retrieved_chunks=[],
            result_count=0,
        )

    chunk = RetrievedChunk(
        chunk_id="DOC-test-CHUNK-0000",
        document_id="DOC-test",
        title="Synthetic Test",
        section_title="Test",
        page_number=None,
        text=(
            "Example Antihypertensive Class is "
            "contraindicated in Pregnancy."
        ),
        score=0.9,
        version="1",
        publisher="MedOntoAgent",
        clinical_domain="Testing",
        document_type="Synthetic Test",
        source_authority=(
            SourceAuthority.OTHER
        ),
        source_url=None,
    )

    return RetrievalResult(
        query="test",
        top_k=5,
        retrieved_chunks=[
            chunk
        ],
        result_count=1,
    )


def create_constraint_set() -> OntologyConstraintSet:
    constraint = OntologyConstraint(
        constraint_type="CONTRAINDICATION",
        subject_name=(
            "ExampleAntihypertensiveClass"
        ),
        subject_label=(
            "Example Antihypertensive Class"
        ),
        subject_uri=(
            "https://medontoagent.example.org/"
            "ontology/"
            "ExampleAntihypertensiveClass"
        ),
        predicate_name=(
            "contraindicatedIn"
        ),
        predicate_label=(
            "contraindicated in"
        ),
        predicate_uri=(
            "https://medontoagent.example.org/"
            "ontology/contraindicatedIn"
        ),
        object_name="Pregnancy",
        object_label="Pregnancy",
        object_uri=(
            "https://medontoagent.example.org/"
            "ontology/Pregnancy"
        ),
        source_document_id=(
            "SYNTHETIC-ONTOLOGY-TEST"
        ),
        source_section=(
            "Implementation Test"
        ),
        source_version="1.0",
        assertion_status=(
            "SYNTHETIC_TEST_ONLY"
        ),
    )

    return OntologyConstraintSet(
        query="test",
        constraints=[
            constraint
        ],
        retrieval_hints=RetrievalHints(
            concept_names=[
                "ExampleAntihypertensiveClass",
                "Pregnancy",
            ],
            concept_labels=[
                "Example Antihypertensive Class",
                "Pregnancy",
            ],
            relationship_terms=[
                (
                    "Example Antihypertensive Class "
                    "contraindicated in Pregnancy"
                )
            ],
            expanded_query="test",
        ),
        constraint_count=1,
    )


def test_evidence_formatter_preserves_source_id() -> None:
    result = format_medonto_evidence(
        create_retrieval()
    )

    assert (
        "SOURCE_ID: DOC-test-CHUNK-0000"
        in result
    )


def test_constraint_formatter_contains_relationship() -> None:
    result = (
        format_ontology_constraints(
            create_constraint_set()
        )
    )

    assert (
        "contraindicatedIn"
        in result
    )


def test_valid_draft_references_pass() -> None:
    draft = MedOntoDraftOutput(
        answer="Synthetic grounded answer.",
        claims=[
            MedOntoClaim(
                claim_text=(
                    "The synthetic evidence reports "
                    "a contraindication."
                ),
                citation_chunk_ids=[
                    "DOC-test-CHUNK-0000"
                ],
            )
        ],
        considered_constraints=[
            MedOntoConstraintReference(
                constraint_type=(
                    "CONTRAINDICATION"
                ),
                subject_name=(
                    "ExampleAntihypertensiveClass"
                ),
                predicate_name=(
                    "contraindicatedIn"
                ),
                object_name="Pregnancy",
            )
        ],
        status="DRAFT",
    )

    validate_draft_references(
        draft=draft,
        retrieval=create_retrieval(),
        constraint_set=(
            create_constraint_set()
        ),
    )


def test_invented_citation_is_rejected() -> None:
    draft = MedOntoDraftOutput(
        answer="Synthetic answer.",
        claims=[
            MedOntoClaim(
                claim_text="Synthetic claim.",
                citation_chunk_ids=[
                    "FAKE-CHUNK"
                ],
            )
        ],
        considered_constraints=[
            MedOntoConstraintReference(
                constraint_type=(
                    "CONTRAINDICATION"
                ),
                subject_name=(
                    "ExampleAntihypertensiveClass"
                ),
                predicate_name=(
                    "contraindicatedIn"
                ),
                object_name="Pregnancy",
            )
        ],
        status="DRAFT",
    )

    with pytest.raises(
        MedOntoGenerationError,
        match="not retrieved",
    ):
        validate_draft_references(
            draft=draft,
            retrieval=create_retrieval(),
            constraint_set=(
                create_constraint_set()
            ),
        )


def test_invented_constraint_is_rejected() -> None:
    draft = MedOntoDraftOutput(
        answer="Synthetic answer.",
        claims=[
            MedOntoClaim(
                claim_text="Synthetic claim.",
                citation_chunk_ids=[
                    "DOC-test-CHUNK-0000"
                ],
            )
        ],
        considered_constraints=[
            MedOntoConstraintReference(
                constraint_type=(
                    "RECOMMENDATION"
                ),
                subject_name=(
                    "ExampleAntihypertensiveClass"
                ),
                predicate_name=(
                    "recommendedFor"
                ),
                object_name="Pregnancy",
            )
        ],
        status="DRAFT",
    )

    with pytest.raises(
        MedOntoGenerationError,
        match="exactly",
    ):
        validate_draft_references(
            draft=draft,
            retrieval=create_retrieval(),
            constraint_set=(
                create_constraint_set()
            ),
        )


def test_omitted_constraint_is_rejected() -> None:
    draft = MedOntoDraftOutput(
        answer="Synthetic answer.",
        claims=[
            MedOntoClaim(
                claim_text="Synthetic claim.",
                citation_chunk_ids=[
                    "DOC-test-CHUNK-0000"
                ],
            )
        ],
        considered_constraints=[],
        status="DRAFT",
    )

    with pytest.raises(
        MedOntoGenerationError,
        match="exactly",
    ):
        validate_draft_references(
            draft=draft,
            retrieval=create_retrieval(),
            constraint_set=(
                create_constraint_set()
            ),
        )


@patch(
    "app.llm.medonto_generation."
    "create_openrouter_client"
)
def test_zero_evidence_abstains_without_llm(
    mock_create_client: MagicMock,
) -> None:
    result = generate_medonto_draft(
        query="test",
        retrieval=create_retrieval(
            with_evidence=False
        ),
        constraint_set=(
            create_constraint_set()
        ),
        settings=create_settings(),
    )

    assert (
        result.status
        == "ABSTAINED"
    )

    assert result.claims == []

    mock_create_client.assert_not_called()