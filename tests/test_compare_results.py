from datetime import (
    datetime,
    timezone,
)

import pytest

from app.models.evaluation import (
    EvaluationDataset,
    EvaluationResultRecord,
    EvaluationRunArtifact,
)
from evals.compare_results import (
    EvaluationComparisonError,
    calculate_metrics,
    create_comparison,
)


def dataset() -> EvaluationDataset:
    return EvaluationDataset.model_validate(
        {
            "dataset_name": (
                "Comparison Test"
            ),
            "dataset_version": "1.0",
            "description": (
                "Synthetic deterministic "
                "comparison test."
            ),
            "split": "DEVELOPMENT",
            "corpus_version": "test-v1",
            "frozen": False,
            "questions": [
                {
                    "question_id": "CMP-001",
                    "split": "DEVELOPMENT",
                    "category": (
                        "ONTOLOGY_CONSTRAINT"
                    ),
                    "question": (
                        "Synthetic constraint "
                        "question?"
                    ),
                    "expected_behavior": (
                        "ANSWER"
                    ),
                    "reference_answer": (
                        "Synthetic answer."
                    ),
                    "expected_document_ids": [
                        "DOC-1"
                    ],
                    "expected_chunk_ids": [
                        "CHUNK-1"
                    ],
                    "expected_ontology_relationships": [
                        (
                            "ExampleClass "
                            "contraindicatedIn "
                            "Pregnancy"
                        )
                    ],
                },
                {
                    "question_id": "CMP-002",
                    "split": "DEVELOPMENT",
                    "category": (
                        "NO_EVIDENCE"
                    ),
                    "question": (
                        "Synthetic no-evidence "
                        "question?"
                    ),
                    "expected_behavior": (
                        "ABSTAIN"
                    ),
                    "reference_answer": None,
                    "expected_document_ids": [],
                    "expected_chunk_ids": [],
                    "expected_ontology_relationships": [],
                },
            ],
        }
    )


def configuration() -> dict:
    return {
        "generation_model": (
            "test-model"
        ),
        "embedding_model": (
            "test-embedding"
        ),
        "embedding_dimensions": 384,
        "embedding_device": "cpu",
        "retrieval_top_k": 5,
        "max_agent_retries": 1,
        "max_chunk_words": 600,
        "chunk_overlap_words": 80,
    }


def baseline_artifact() -> EvaluationRunArtifact:
    return EvaluationRunArtifact(
        run_id="baseline-test",
        created_at=datetime.now(
            timezone.utc
        ),
        system="BASELINE",
        dataset_name=(
            "Comparison Test"
        ),
        dataset_version="1.0",
        corpus_version="test-v1",
        configuration=(
            configuration()
        ),
        results=[
            EvaluationResultRecord(
                question_id="CMP-001",
                category=(
                    "ONTOLOGY_CONSTRAINT"
                ),
                question=(
                    "Synthetic constraint "
                    "question?"
                ),
                expected_behavior="ANSWER",
                system="BASELINE",
                system_status="ANSWERED",
                behavior_match=True,
                answer="Synthetic answer.",
                citation_chunk_ids=[
                    "CHUNK-1"
                ],
                citation_document_ids=[
                    "DOC-1"
                ],
                retrieved_chunk_ids=[
                    "CHUNK-1",
                    "CHUNK-2",
                ],
                retrieved_document_ids=[
                    "DOC-1",
                    "DOC-2",
                ],
                ontology_constraints=[],
                used_ontology_filter=None,
                retrieval_relaxed=None,
                retry_count=None,
                latency_ms=100.0,
                error_message=None,
            ),
            EvaluationResultRecord(
                question_id="CMP-002",
                category="NO_EVIDENCE",
                question=(
                    "Synthetic no-evidence "
                    "question?"
                ),
                expected_behavior="ABSTAIN",
                system="BASELINE",
                system_status="ABSTAINED",
                behavior_match=True,
                answer=(
                    "Insufficient evidence."
                ),
                citation_chunk_ids=[],
                citation_document_ids=[],
                retrieved_chunk_ids=[
                    "CHUNK-2"
                ],
                retrieved_document_ids=[
                    "DOC-2"
                ],
                ontology_constraints=[],
                used_ontology_filter=None,
                retrieval_relaxed=None,
                retry_count=None,
                latency_ms=200.0,
                error_message=None,
            ),
        ],
    )


def medonto_artifact() -> EvaluationRunArtifact:
    return EvaluationRunArtifact(
        run_id="medonto-test",
        created_at=datetime.now(
            timezone.utc
        ),
        system="MEDONTO",
        dataset_name=(
            "Comparison Test"
        ),
        dataset_version="1.0",
        corpus_version="test-v1",
        configuration=(
            configuration()
        ),
        results=[
            EvaluationResultRecord(
                question_id="CMP-001",
                category=(
                    "ONTOLOGY_CONSTRAINT"
                ),
                question=(
                    "Synthetic constraint "
                    "question?"
                ),
                expected_behavior="ANSWER",
                system="MEDONTO",
                system_status="VERIFIED",
                behavior_match=True,
                answer="Synthetic answer.",
                citation_chunk_ids=[
                    "CHUNK-1"
                ],
                citation_document_ids=[
                    "DOC-1"
                ],
                retrieved_chunk_ids=[
                    "CHUNK-1"
                ],
                retrieved_document_ids=[
                    "DOC-1"
                ],
                ontology_constraints=[
                    (
                        "ExampleClass "
                        "contraindicatedIn "
                        "Pregnancy"
                    )
                ],
                used_ontology_filter=True,
                retrieval_relaxed=False,
                retry_count=0,
                latency_ms=300.0,
                error_message=None,
            ),
            EvaluationResultRecord(
                question_id="CMP-002",
                category="NO_EVIDENCE",
                question=(
                    "Synthetic no-evidence "
                    "question?"
                ),
                expected_behavior="ABSTAIN",
                system="MEDONTO",
                system_status="ABSTAINED",
                behavior_match=True,
                answer=(
                    "Insufficient evidence."
                ),
                citation_chunk_ids=[],
                citation_document_ids=[],
                retrieved_chunk_ids=[
                    "CHUNK-2"
                ],
                retrieved_document_ids=[
                    "DOC-2"
                ],
                ontology_constraints=[],
                used_ontology_filter=False,
                retrieval_relaxed=False,
                retry_count=0,
                latency_ms=400.0,
                error_message=None,
            ),
        ],
    )


def test_calculate_metrics() -> None:
    data = dataset()

    baseline_metrics = (
        calculate_metrics(
            dataset=data,
            artifact=(
                baseline_artifact()
            ),
        )
    )

    medonto_metrics = (
        calculate_metrics(
            dataset=data,
            artifact=(
                medonto_artifact()
            ),
        )
    )

    assert (
        baseline_metrics
        .behavior_match_rate
        == 1.0
    )

    assert (
        baseline_metrics
        .error_rate
        == 0.0
    )

    assert (
        baseline_metrics
        .abstention_rate
        == 0.5
    )

    assert (
        baseline_metrics
        .expected_document_retrieval_hit_rate
        == 1.0
    )

    assert (
        baseline_metrics
        .expected_chunk_retrieval_hit_rate
        == 1.0
    )

    assert (
        baseline_metrics
        .citation_document_precision
        == 1.0
    )

    assert (
        baseline_metrics
        .citation_document_recall
        == 1.0
    )

    assert (
        baseline_metrics
        .citation_chunk_precision
        == 1.0
    )

    assert (
        baseline_metrics
        .citation_chunk_recall
        == 1.0
    )

    assert (
        baseline_metrics
        .ontology_relationship_recall
        == 0.0
    )

    assert (
        baseline_metrics
        .ontology_exact_match_rate
        == 0.0
    )

    assert (
        baseline_metrics.retry_rate
        is None
    )

    assert (
        baseline_metrics
        .retrieval_relaxation_rate
        is None
    )

    assert (
        baseline_metrics
        .latency.mean_ms
        == 150.0
    )

    assert (
        baseline_metrics
        .latency.median_ms
        == 150.0
    )

    assert (
        baseline_metrics
        .latency.p95_ms
        == 200.0
    )

    assert (
        medonto_metrics
        .ontology_relationship_recall
        == 1.0
    )

    assert (
        medonto_metrics
        .ontology_exact_match_rate
        == 1.0
    )

    assert (
        medonto_metrics.retry_rate
        == 0.0
    )

    assert (
        medonto_metrics
        .retrieval_relaxation_rate
        == 0.0
    )


def test_missing_annotations_produce_none() -> None:
    data = EvaluationDataset.model_validate(
        {
            "dataset_name": "No Annotation",
            "dataset_version": "1.0",
            "description": (
                "Synthetic no-annotation test."
            ),
            "split": "DEVELOPMENT",
            "corpus_version": "test-v1",
            "frozen": False,
            "questions": [
                {
                    "question_id": "NA-001",
                    "split": "DEVELOPMENT",
                    "category": (
                        "STRAIGHTFORWARD_EVIDENCE"
                    ),
                    "question": (
                        "Synthetic question?"
                    ),
                    "expected_behavior": "ANSWER",
                    "reference_answer": (
                        "Synthetic answer."
                    ),
                }
            ],
        }
    )

    artifact = EvaluationRunArtifact(
        run_id="baseline-na",
        created_at=datetime.now(
            timezone.utc
        ),
        system="BASELINE",
        dataset_name="No Annotation",
        dataset_version="1.0",
        corpus_version="test-v1",
        configuration=(
            configuration()
        ),
        results=[
            EvaluationResultRecord(
                question_id="NA-001",
                category=(
                    "STRAIGHTFORWARD_EVIDENCE"
                ),
                question=(
                    "Synthetic question?"
                ),
                expected_behavior="ANSWER",
                system="BASELINE",
                system_status="ANSWERED",
                behavior_match=True,
                answer="Synthetic answer.",
                citation_chunk_ids=[],
                citation_document_ids=[],
                retrieved_chunk_ids=[],
                retrieved_document_ids=[],
                ontology_constraints=[],
                used_ontology_filter=None,
                retrieval_relaxed=None,
                retry_count=None,
                latency_ms=10.0,
                error_message=None,
            )
        ],
    )

    metrics = calculate_metrics(
        dataset=data,
        artifact=artifact,
    )

    assert (
        metrics
        .expected_document_retrieval_hit_rate
        is None
    )

    assert (
        metrics
        .expected_chunk_retrieval_hit_rate
        is None
    )

    assert (
        metrics.citation_document_precision
        is None
    )

    assert (
        metrics.citation_document_recall
        is None
    )

    assert (
        metrics.citation_chunk_precision
        is None
    )

    assert (
        metrics.citation_chunk_recall
        is None
    )

    assert (
        metrics.ontology_relationship_recall
        is None
    )

    assert (
        metrics.ontology_exact_match_rate
        is None
    )


def test_create_comparison() -> None:
    comparison = (
        create_comparison(
            dataset=dataset(),
            baseline=(
                baseline_artifact()
            ),
            medonto=(
                medonto_artifact()
            ),
        )
    )

    assert (
        comparison.dataset_name
        == "Comparison Test"
    )

    assert (
        comparison.baseline_run_id
        == "baseline-test"
    )

    assert (
        comparison.medonto_run_id
        == "medonto-test"
    )

    assert len(
        comparison.questions
    ) == 2

    first = (
        comparison.questions[0]
    )

    assert (
        first
        .baseline_retrieved_expected_document
        is True
    )

    assert (
        first
        .medonto_retrieved_expected_document
        is True
    )

    assert (
        first
        .baseline_expected_ontology_relationships_found
        is False
    )

    assert (
        first
        .medonto_expected_ontology_relationships_found
        is True
    )


def test_configuration_mismatch_is_rejected() -> None:
    baseline = (
        baseline_artifact()
    )

    medonto = (
        medonto_artifact()
    )

    medonto.configuration[
        "embedding_model"
    ] = "different-model"

    with pytest.raises(
        EvaluationComparisonError,
        match=(
            "Shared evaluation configuration"
        ),
    ):
        create_comparison(
            dataset=dataset(),
            baseline=baseline,
            medonto=medonto,
        )


def test_dataset_version_mismatch_is_rejected() -> None:
    baseline = (
        baseline_artifact()
    )

    baseline.dataset_version = (
        "different-version"
    )

    with pytest.raises(
        EvaluationComparisonError,
        match=(
            "Dataset version mismatch"
        ),
    ):
        create_comparison(
            dataset=dataset(),
            baseline=baseline,
            medonto=(
                medonto_artifact()
            ),
        )