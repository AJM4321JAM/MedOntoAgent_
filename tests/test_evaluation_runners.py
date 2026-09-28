import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.config.settings import (
    Settings,
)
from app.models.evaluation import (
    EvaluationDataset,
)
from app.models.responses import (
    BaselineQueryResponse,
    CitationReference,
    MedOntoClaim,
    MedOntoDraftOutput,
)
from evals.baseline_runner import (
    run_baseline_dataset,
)
from evals.common import (
    EvaluationConfigurationError,
    load_evaluation_dataset,
    save_run_artifact,
)
from evals.medonto_runner import (
    run_medonto_dataset,
)


def settings() -> Settings:
    return Settings(
        _env_file=None,
        openrouter_api_key="test-key",
        openrouter_model="test-model",
        qdrant_url=None,
        qdrant_api_key=None,
    )


def one_question_dataset() -> EvaluationDataset:
    return EvaluationDataset.model_validate(
        {
            "dataset_name": "Runner Test",
            "dataset_version": "1.0",
            "description": (
                "Synthetic runner unit test."
            ),
            "split": "DEVELOPMENT",
            "corpus_version": (
                "synthetic-v1"
            ),
            "frozen": False,
            "questions": [
                {
                    "question_id": "TEST-001",
                    "split": "DEVELOPMENT",
                    "category": (
                        "ONTOLOGY_CONSTRAINT"
                    ),
                    "question": (
                        "Can Example "
                        "Antihypertensive Class "
                        "be used during pregnancy?"
                    ),
                    "expected_behavior": (
                        "ANSWER"
                    ),
                    "reference_answer": (
                        "Synthetic answer."
                    ),
                }
            ],
        }
    )


def test_development_dataset_is_runnable() -> None:
    dataset = (
        load_evaluation_dataset(
            "evals/"
            "development_dataset.json"
        )
    )

    assert (
        dataset.split
        == "DEVELOPMENT"
    )


def test_unfinished_final_dataset_is_blocked() -> None:
    with pytest.raises(
        EvaluationConfigurationError,
        match=(
            "Final evaluation dataset "
            "is not ready"
        ),
    ):
        load_evaluation_dataset(
            "evals/golden_dataset.json"
        )


def test_baseline_runner_records_standard_result() -> None:
    response = BaselineQueryResponse(
        query=(
            "Can Example "
            "Antihypertensive Class "
            "be used during pregnancy?"
        ),
        answer=(
            "Synthetic grounded answer."
        ),
        citations=[
            CitationReference(
                chunk_id=(
                    "DOC-TEST-CHUNK-0000"
                )
            )
        ],
        citation_document_ids=[
            "DOC-TEST"
        ],
        retrieved_chunk_ids=[
            "DOC-TEST-CHUNK-0000",
            "DOC-OTHER-CHUNK-0000",
        ],
        retrieved_document_ids=[
            "DOC-TEST",
            "DOC-OTHER",
        ],
        retrieved_contexts=[
            "Synthetic retrieved context.",
            "Second synthetic retrieved context.",
        ],
        status="ANSWERED",
    )

    with patch(
        "evals.baseline_runner."
        "run_baseline_rag",
        return_value=response,
    ):
        artifact = (
            run_baseline_dataset(
                one_question_dataset(),
                settings=settings(),
            )
        )

    assert (
        artifact.system
        == "BASELINE"
    )

    assert len(
        artifact.results
    ) == 1

    result = (
        artifact.results[0]
    )

    assert (
        result.system_status
        == "ANSWERED"
    )

    assert (
        result.behavior_match
        is True
    )

    assert (
        result.citation_chunk_ids
        == [
            "DOC-TEST-CHUNK-0000"
        ]
    )

    assert (
        result.citation_document_ids
        == [
            "DOC-TEST"
        ]
    )

    assert (
        result.retrieved_chunk_ids
        == [
            "DOC-TEST-CHUNK-0000",
            "DOC-OTHER-CHUNK-0000",
        ]
    )

    assert (
        result.retrieved_document_ids
        == [
            "DOC-TEST",
            "DOC-OTHER",
        ]
    )

    assert (
        result.retrieved_contexts
        == [
            "Synthetic retrieved context.",
            "Second synthetic retrieved context.",
        ]
    )

    assert (
        result.used_ontology_filter
        is None
    )

    assert (
        result.retry_count
        is None
    )


def test_medonto_runner_records_graph_metadata() -> None:
    draft = MedOntoDraftOutput(
        answer=(
            "Synthetic verified answer."
        ),
        claims=[
            MedOntoClaim(
                claim_text=(
                    "Synthetic claim."
                ),
                citation_chunk_ids=[
                    "DOC-TEST-CHUNK-0000"
                ],
            )
        ],
        considered_constraints=[],
        status="DRAFT",
    )

    constraint = SimpleNamespace(
        subject_name=(
            "ExampleAntihypertensiveClass"
        ),
        predicate_name=(
            "contraindicatedIn"
        ),
        object_name="Pregnancy",
    )

    constraint_set = SimpleNamespace(
        constraints=[
            constraint
        ]
    )

    retrieval_result = (
        SimpleNamespace(
            retrieved_chunks=[
                SimpleNamespace(
                    chunk_id=(
                        "DOC-TEST-CHUNK-0000"
                    ),
                    document_id=(
                        "DOC-TEST"
                    ),
                    text=(
                        "Synthetic retrieved context."
                    ),
                )
            ]
        )
    )

    state = {
        "draft_output": draft,
        "constraint_set": (
            constraint_set
        ),
        "retrieval_result": (
            retrieval_result
        ),
        "used_ontology_filter": True,
        "retrieval_relaxed": False,
        "retry_count": 0,
        "orchestration_status": (
            "VERIFIED"
        ),
        "abstention_reason": None,
    }

    with patch(
        "evals.medonto_runner."
        "medonto_graph.invoke",
        return_value=state,
    ):
        artifact = (
            run_medonto_dataset(
                one_question_dataset(),
                settings=settings(),
            )
        )

    result = (
        artifact.results[0]
    )

    assert (
        result.system_status
        == "VERIFIED"
    )

    assert (
        result.behavior_match
        is True
    )

    assert (
        result.citation_chunk_ids
        == [
            "DOC-TEST-CHUNK-0000"
        ]
    )

    assert (
        result.citation_document_ids
        == [
            "DOC-TEST"
        ]
    )

    assert (
        result.retrieved_chunk_ids
        == [
            "DOC-TEST-CHUNK-0000"
        ]
    )

    assert (
        result.retrieved_document_ids
        == [
            "DOC-TEST"
        ]
    )

    assert (
        result.retrieved_contexts
        == [
            "Synthetic retrieved context."
        ]
    )

    assert (
        result.ontology_constraints
        == [
            (
                "ExampleAntihypertensiveClass "
                "contraindicatedIn Pregnancy"
            )
        ]
    )

    assert (
        result.used_ontology_filter
        is True
    )

    assert (
        result.retrieval_relaxed
        is False
    )

    assert (
        result.retry_count
        == 0
    )


def test_runner_error_is_safe_and_does_not_store_private_detail() -> None:
    private_detail = (
        "super-secret-provider-detail"
    )

    with patch(
        "evals.baseline_runner."
        "run_baseline_rag",
        side_effect=RuntimeError(
            private_detail
        ),
    ):
        artifact = (
            run_baseline_dataset(
                one_question_dataset(),
                settings=settings(),
            )
        )

    result = (
        artifact.results[0]
    )

    assert (
        result.system_status
        == "ERROR"
    )

    assert (
        result.behavior_match
        is False
    )

    assert (
        result.retrieved_contexts
        == []
    )

    assert (
        result.error_message
        == (
            "Execution failed: "
            "RuntimeError"
        )
    )

    assert (
        private_detail
        not in result.error_message
    )


def test_run_artifact_can_be_persisted(
    tmp_path: Path,
) -> None:
    response = BaselineQueryResponse(
        query=(
            "Can Example "
            "Antihypertensive Class "
            "be used during pregnancy?"
        ),
        answer="Synthetic answer.",
        citations=[
            CitationReference(
                chunk_id=(
                    "DOC-TEST-CHUNK-0000"
                )
            )
        ],
        citation_document_ids=[
            "DOC-TEST"
        ],
        retrieved_chunk_ids=[
            "DOC-TEST-CHUNK-0000"
        ],
        retrieved_document_ids=[
            "DOC-TEST"
        ],
        retrieved_contexts=[
            "Synthetic retrieved context."
        ],
        status="ANSWERED",
    )

    with patch(
        "evals.baseline_runner."
        "run_baseline_rag",
        return_value=response,
    ):
        artifact = (
            run_baseline_dataset(
                one_question_dataset(),
                settings=settings(),
            )
        )

    output = (
        tmp_path
        / "artifact.json"
    )

    saved = save_run_artifact(
        artifact,
        output,
    )

    assert saved == output
    assert output.exists()

    data = json.loads(
        output.read_text(
            encoding="utf-8"
        )
    )

    assert (
        data["system"]
        == "BASELINE"
    )

    assert (
        data["dataset_version"]
        == "1.0"
    )

    assert len(
        data["results"]
    ) == 1

    assert (
        data["results"][0]
        ["citation_document_ids"]
        == ["DOC-TEST"]
    )

    assert (
        data["results"][0]
        ["retrieved_document_ids"]
        == ["DOC-TEST"]
    )

    assert (
        data["results"][0]
        ["retrieved_contexts"]
        == [
            "Synthetic retrieved context."
        ]
    )

    assert (
        "openrouter_api_key"
        not in data["configuration"]
    )

    assert (
        "qdrant_api_key"
        not in data["configuration"]
    )
