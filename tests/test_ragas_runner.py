from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.config.settings import Settings
from app.models.evaluation import (
    EvaluationDataset,
    EvaluationResultRecord,
    EvaluationRunArtifact,
    RagasQuestionScore,
)
from evals.ragas_runner import (
    RagasEvaluationError,
    _build_summary,
    evaluate_run_artifact,
    save_ragas_artifact,
    validate_run_against_dataset,
)


class FakeMetric:
    def __init__(
        self,
        value: float,
    ) -> None:
        self.value = value
        self.calls: list[dict] = []

    async def ascore(
        self,
        **kwargs,
    ):
        self.calls.append(
            kwargs
        )

        return SimpleNamespace(
            value=self.value
        )


def settings() -> Settings:
    return Settings(
        _env_file=None,
        gemini_api_key="test-key",
        gemini_model="test-model",
        ragas_evaluator_model="openrouter/free",
        embedding_model=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        embedding_dimensions=384,
        embedding_batch_size=2,
        embedding_device="cpu",
        qdrant_url=None,
        qdrant_api_key=None,
    )


def dataset() -> EvaluationDataset:
    return EvaluationDataset.model_validate(
        {
            "dataset_name": "Ragas Test Dataset",
            "dataset_version": "1.0",
            "description": (
                "Synthetic semantic evaluation test."
            ),
            "split": "DEVELOPMENT",
            "corpus_version": "synthetic-v1",
            "frozen": False,
            "questions": [
                {
                    "question_id": "TEST-001",
                    "split": "DEVELOPMENT",
                    "category": (
                        "STRAIGHTFORWARD_EVIDENCE"
                    ),
                    "question": (
                        "What does the evidence say?"
                    ),
                    "expected_behavior": "ANSWER",
                    "reference_answer": (
                        "The synthetic evidence "
                        "supports the answer."
                    ),
                },
                {
                    "question_id": "TEST-002",
                    "split": "DEVELOPMENT",
                    "category": "NO_EVIDENCE",
                    "question": (
                        "What unsupported treatment "
                        "should be used?"
                    ),
                    "expected_behavior": "ABSTAIN",
                    "reference_answer": None,
                },
            ],
        }
    )


def run_artifact() -> EvaluationRunArtifact:
    return EvaluationRunArtifact(
        run_id="baseline-test-run",
        created_at=datetime.now(
            timezone.utc
        ),
        system="BASELINE",
        dataset_name="Ragas Test Dataset",
        dataset_version="1.0",
        corpus_version="synthetic-v1",
        configuration={
            "generation_model": "test-model",
            "embedding_model": (
                "test-embedding-model"
            ),
            "embedding_dimensions": 384,
            "retrieval_top_k": 5,
        },
        results=[
            EvaluationResultRecord(
                question_id="TEST-001",
                category=(
                    "STRAIGHTFORWARD_EVIDENCE"
                ),
                question=(
                    "What does the evidence say?"
                ),
                expected_behavior="ANSWER",
                system="BASELINE",
                system_status="ANSWERED",
                behavior_match=True,
                answer=(
                    "The synthetic evidence "
                    "supports the answer."
                ),
                citation_chunk_ids=[
                    "CHUNK-001"
                ],
                citation_document_ids=[
                    "DOC-001"
                ],
                retrieved_chunk_ids=[
                    "CHUNK-001",
                    "CHUNK-002",
                ],
                retrieved_document_ids=[
                    "DOC-001",
                    "DOC-002",
                ],
                retrieved_contexts=[
                    "First ranked context.",
                    "Second ranked context.",
                ],
                ontology_constraints=[],
                used_ontology_filter=None,
                retrieval_relaxed=None,
                retry_count=None,
                latency_ms=10.0,
                error_message=None,
            ),
            EvaluationResultRecord(
                question_id="TEST-002",
                category="NO_EVIDENCE",
                question=(
                    "What unsupported treatment "
                    "should be used?"
                ),
                expected_behavior="ABSTAIN",
                system="BASELINE",
                system_status="ABSTAINED",
                behavior_match=True,
                answer=(
                    "The approved retrieved evidence "
                    "is insufficient."
                ),
                citation_chunk_ids=[],
                citation_document_ids=[],
                retrieved_chunk_ids=[],
                retrieved_document_ids=[],
                retrieved_contexts=[],
                ontology_constraints=[],
                used_ontology_filter=None,
                retrieval_relaxed=None,
                retry_count=None,
                latency_ms=5.0,
                error_message=None,
            ),
        ],
    )


def test_validate_matching_run_and_dataset() -> None:
    validate_run_against_dataset(
        run=run_artifact(),
        dataset=dataset(),
    )


def test_dataset_version_mismatch_is_rejected() -> None:
    data = dataset()
    data.dataset_version = "2.0"

    with pytest.raises(
        RagasEvaluationError,
        match="Dataset version",
    ):
        validate_run_against_dataset(
            run=run_artifact(),
            dataset=data,
        )


def test_semantic_scores_preserve_context_order() -> None:
    faithfulness = FakeMetric(0.91)
    context_precision = FakeMetric(0.82)
    answer_relevancy = FakeMetric(0.73)

    with patch(
        "evals.ragas_runner."
        "_create_ragas_components",
        return_value=(
            faithfulness,
            context_precision,
            answer_relevancy,
        ),
    ):
        artifact = evaluate_run_artifact(
            run=run_artifact(),
            dataset=dataset(),
            settings=settings(),
        )

    answered = artifact.scores[0]
    abstained = artifact.scores[1]

    assert (
        answered.faithfulness
        == pytest.approx(0.91)
    )

    assert (
        answered.context_precision
        == pytest.approx(0.82)
    )

    assert (
        answered.answer_relevancy
        == pytest.approx(0.73)
    )

    assert (
        context_precision.calls[0]
        ["retrieved_contexts"]
        == [
            "First ranked context.",
            "Second ranked context.",
        ]
    )

    assert (
        faithfulness.calls[0]
        ["retrieved_contexts"]
        == [
            "First ranked context.",
            "Second ranked context.",
        ]
    )

    assert abstained.faithfulness is None
    assert abstained.context_precision is None
    assert abstained.answer_relevancy is None

    assert len(faithfulness.calls) == 1
    assert len(context_precision.calls) == 1
    assert len(answer_relevancy.calls) == 1

    assert (
        artifact.summary.total_records
        == 2
    )

    assert (
        artifact.summary.scored_records
        == 1
    )


def test_metric_failure_is_safely_recorded() -> None:
    class FailingMetric:
        async def ascore(
            self,
            **kwargs,
        ):
            raise RuntimeError(
                "private provider detail"
            )

    good_metric = FakeMetric(0.5)

    with patch(
        "evals.ragas_runner."
        "_create_ragas_components",
        return_value=(
            FailingMetric(),
            good_metric,
            good_metric,
        ),
    ):
        artifact = evaluate_run_artifact(
            run=run_artifact(),
            dataset=dataset(),
            settings=settings(),
        )

    score = artifact.scores[0]

    assert score.faithfulness is None

    assert (
        score.context_precision
        == pytest.approx(0.5)
    )

    assert (
        score.answer_relevancy
        == pytest.approx(0.5)
    )

    assert (
        score.metric_errors
        == [
            (
                "faithfulness failed: "
                "RuntimeError"
            )
        ]
    )

    assert (
        "private provider detail"
        not in " ".join(
            score.metric_errors
        )
    )


def test_ragas_artifact_can_be_saved(
    tmp_path: Path,
) -> None:
    metric = FakeMetric(0.8)

    with patch(
        "evals.ragas_runner."
        "_create_ragas_components",
        return_value=(
            metric,
            metric,
            metric,
        ),
    ):
        artifact = evaluate_run_artifact(
            run=run_artifact(),
            dataset=dataset(),
            settings=settings(),
        )

    output = (
        tmp_path
        / "ragas.json"
    )

    saved = save_ragas_artifact(
        artifact,
        output,
    )

    assert saved == output
    assert output.exists()

    saved_text = output.read_text(
        encoding="utf-8"
    )

    assert (
        '"source_run_id": "baseline-test-run"'
        in saved_text
    )

    assert (
        '"system": "BASELINE"'
        in saved_text
    )


def test_summary_ignores_none_values() -> None:
    summary = _build_summary(
        [
            RagasQuestionScore(
                question_id="Q1",
                system="BASELINE",
                faithfulness=0.8,
                context_precision=0.6,
                answer_relevancy=0.9,
            ),
            RagasQuestionScore(
                question_id="Q2",
                system="BASELINE",
                faithfulness=None,
                context_precision=None,
                answer_relevancy=None,
            ),
        ]
    )

    assert (
        summary.total_records
        == 2
    )

    assert (
        summary.scored_records
        == 1
    )

    assert (
        summary.faithfulness_mean
        == pytest.approx(0.8)
    )

    assert (
        summary.context_precision_mean
        == pytest.approx(0.6)
    )

    assert (
        summary.answer_relevancy_mean
        == pytest.approx(0.9)
    )
