import argparse
import asyncio
import json
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import Any

from app.config.settings import (
    Settings,
    get_settings,
)
from app.models.evaluation import (
    EvaluationDataset,
    EvaluationRunArtifact,
    RagasEvaluationArtifact,
    RagasMetricSummary,
    RagasQuestionScore,
)
from evals.common import (
    load_evaluation_dataset,
)


DEFAULT_DATASET_PATH = Path(
    "evals/development_dataset.json"
)

RESULTS_DIRECTORY = Path(
    "evals/results"
)


class RagasEvaluationError(RuntimeError):
    """
    Raised when semantic evaluation cannot be configured
    or when source artifacts are incompatible.
    """


def load_run_artifact(
    path: str | Path,
) -> EvaluationRunArtifact:
    """
    Load one immutable baseline or MedOnto execution artifact.
    """
    artifact_path = Path(
        path
    )

    if not artifact_path.exists():
        raise RagasEvaluationError(
            (
                "Evaluation run artifact does not exist: "
                f"{artifact_path}"
            )
        )

    try:
        return (
            EvaluationRunArtifact
            .model_validate_json(
                artifact_path.read_text(
                    encoding="utf-8"
                )
            )
        )

    except Exception as exc:
        raise RagasEvaluationError(
            (
                "Evaluation run artifact could not "
                "be validated."
            )
        ) from exc


def validate_run_against_dataset(
    *,
    run: EvaluationRunArtifact,
    dataset: EvaluationDataset,
) -> None:
    """
    Ensure semantic evaluation uses the exact dataset metadata
    associated with the original system execution.
    """
    if (
        run.dataset_name
        != dataset.dataset_name
    ):
        raise RagasEvaluationError(
            "Dataset name does not match source run."
        )

    if (
        run.dataset_version
        != dataset.dataset_version
    ):
        raise RagasEvaluationError(
            "Dataset version does not match source run."
        )

    if (
        run.corpus_version
        != dataset.corpus_version
    ):
        raise RagasEvaluationError(
            "Corpus version does not match source run."
        )

    dataset_question_ids = {
        question.question_id
        for question in dataset.questions
    }

    run_question_ids = {
        result.question_id
        for result in run.results
    }

    if (
        run_question_ids
        != dataset_question_ids
    ):
        raise RagasEvaluationError(
            (
                "Source run question IDs do not "
                "match the evaluation dataset."
            )
        )


def _create_ragas_components(
    settings: Settings,
):
    """
    Create Ragas judge and metric objects.

    Imports are intentionally local so the normal MedOntoAgent
    environment does not require Ragas to be installed.
    """
    try:
        from openai import (
            AsyncOpenAI,
        )
        from ragas.embeddings import (
            HuggingFaceEmbeddings,
        )
        from ragas.llms import (
            llm_factory,
        )
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecisionWithReference,
            Faithfulness,
        )

    except ImportError as exc:
        raise RagasEvaluationError(
            (
                "Ragas dependencies are not installed. "
                "Run this evaluator from .ragas-venv."
            )
        ) from exc

    if settings.gemini_api_key is None:
        raise RagasEvaluationError(
            (
                "GEMINI_API_KEY is required "
                "for Ragas semantic evaluation."
            )
        )

    api_key = (
        settings
        .gemini_api_key
        .get_secret_value()
    )

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=(
            settings.gemini_base_url
        ),
        timeout=(
            settings.llm_timeout_seconds
        ),
        max_retries=0,
    )

    ragas_llm = llm_factory(
        model=(
            settings.ragas_evaluator_model
        ),
        provider="openai",
        client=client,
    )

    embeddings = HuggingFaceEmbeddings(
        model=(
            settings.embedding_model
        ),
        use_api=False,
        device=(
            settings.embedding_device
        ),
        normalize_embeddings=True,
        batch_size=(
            settings.embedding_batch_size
        ),
    )

    faithfulness = Faithfulness(
        llm=ragas_llm
    )

    context_precision = (
        ContextPrecisionWithReference(
            llm=ragas_llm
        )
    )

    answer_relevancy = AnswerRelevancy(
        llm=ragas_llm,
        embeddings=embeddings,
    )

    return (
        faithfulness,
        context_precision,
        answer_relevancy,
    )


def _metric_value(
    result: Any,
) -> float:
    """
    Extract a numeric score from a Ragas MetricResult.

    The helper is deliberately tolerant of minor result-wrapper
    differences while remaining strict about the final value.
    """
    if isinstance(
        result,
        (int, float),
    ):
        value = float(
            result
        )

    elif hasattr(
        result,
        "value",
    ):
        value = float(
            result.value
        )

    elif hasattr(
        result,
        "score",
    ):
        value = float(
            result.score
        )

    elif isinstance(
        result,
        dict,
    ):
        if "value" in result:
            value = float(
                result["value"]
            )

        elif "score" in result:
            value = float(
                result["score"]
            )

        else:
            raise RagasEvaluationError(
                (
                    "Ragas metric result contained "
                    "no numeric score."
                )
            )

    else:
        raise RagasEvaluationError(
            (
                "Ragas metric returned an "
                "unsupported result type."
            )
        )

    if not 0.0 <= value <= 1.0:
        raise RagasEvaluationError(
            (
                "Ragas metric returned a score "
                "outside the expected 0..1 range."
            )
        )

    return value


def _safe_metric_error(
    metric_name: str,
    exc: Exception,
) -> str:
    """
    Record only a safe error classification.

    Provider response bodies, prompts, retrieved clinical text,
    and credentials are deliberately excluded.
    """
    return (
        f"{metric_name} failed: "
        f"{type(exc).__name__}"
    )


async def _metric_ascore_with_one_retry(
    metric,
    **kwargs,
):
    """
    Run one Ragas metric with one bounded retry for
    transient Instructor/provider failures.
    """
    total_attempts = 2

    for attempt in range(
        1,
        total_attempts + 1,
    ):
        try:
            return await metric.ascore(
                **kwargs
            )

        except Exception as exc:
            retryable = (
                type(exc).__name__
                == "InstructorRetryException"
            )

            if (
                not retryable
                or attempt == total_attempts
            ):
                raise

            await asyncio.sleep(3)

    raise RuntimeError(
        "Ragas metric failed unexpectedly."
    )


async def _score_one_question(
    *,
    result_record,
    reference_answer: str | None,
    faithfulness_metric,
    context_precision_metric,
    answer_relevancy_metric,
) -> RagasQuestionScore:
    """
    Score one eligible answered result.

    ABSTAINED and ERROR records are deliberately not semantically
    scored because their behavior is evaluated deterministically.
    """
    if result_record.system_status not in {
        "ANSWERED",
        "VERIFIED",
    }:
        return RagasQuestionScore(
            question_id=(
                result_record.question_id
            ),
            system=(
                result_record.system
            ),
            faithfulness=None,
            context_precision=None,
            answer_relevancy=None,
            metric_errors=[],
        )

    if not result_record.answer:
        return RagasQuestionScore(
            question_id=(
                result_record.question_id
            ),
            system=(
                result_record.system
            ),
            faithfulness=None,
            context_precision=None,
            answer_relevancy=None,
            metric_errors=[
                (
                    "Eligible result had no "
                    "answer text."
                )
            ],
        )

    faithfulness_score: (
        float | None
    ) = None

    context_precision_score: (
        float | None
    ) = None

    answer_relevancy_score: (
        float | None
    ) = None

    metric_errors: list[str] = []

    if result_record.retrieved_contexts:
        try:
            metric_result = (
                await _metric_ascore_with_one_retry(
                    faithfulness_metric,
                    user_input=(
                        result_record.question
                    ),
                    response=(
                        result_record.answer
                    ),
                    retrieved_contexts=(
                        result_record
                        .retrieved_contexts
                    ),
                )
            )

            faithfulness_score = (
                _metric_value(
                    metric_result
                )
            )

        except Exception as exc:
            metric_errors.append(
                _safe_metric_error(
                    "faithfulness",
                    exc,
                )
            )

    else:
        metric_errors.append(
            (
                "faithfulness not applicable: "
                "no retrieved contexts"
            )
        )

    if (
        reference_answer
        and result_record.retrieved_contexts
    ):
        try:
            metric_result = (
                await _metric_ascore_with_one_retry(
                    context_precision_metric,
                    user_input=(
                        result_record.question
                    ),
                    reference=(
                        reference_answer
                    ),
                    retrieved_contexts=(
                        result_record
                        .retrieved_contexts
                    ),
                )
            )

            context_precision_score = (
                _metric_value(
                    metric_result
                )
            )

        except Exception as exc:
            metric_errors.append(
                _safe_metric_error(
                    "context_precision",
                    exc,
                )
            )

    elif reference_answer is None:
        metric_errors.append(
            (
                "context_precision not applicable: "
                "no reference answer"
            )
        )

    else:
        metric_errors.append(
            (
                "context_precision not applicable: "
                "no retrieved contexts"
            )
        )

    try:
        metric_result = (
            await _metric_ascore_with_one_retry(
                answer_relevancy_metric,
                user_input=(
                    result_record.question
                ),
                response=(
                    result_record.answer
                ),
            )
        )

        answer_relevancy_score = (
            _metric_value(
                metric_result
            )
        )

    except Exception as exc:
        metric_errors.append(
            _safe_metric_error(
                "answer_relevancy",
                exc,
            )
        )

    return RagasQuestionScore(
        question_id=(
            result_record.question_id
        ),
        system=(
            result_record.system
        ),
        faithfulness=(
            faithfulness_score
        ),
        context_precision=(
            context_precision_score
        ),
        answer_relevancy=(
            answer_relevancy_score
        ),
        metric_errors=(
            metric_errors
        ),
    )


def _mean_or_none(
    values: list[float],
) -> float | None:
    """
    Calculate an arithmetic mean when at least one score exists.
    """
    if not values:
        return None

    return (
        sum(values)
        / len(values)
    )


def _build_summary(
    scores: list[
        RagasQuestionScore
    ],
) -> RagasMetricSummary:
    """
    Aggregate available Ragas scores without treating N/A
    values as zero.
    """
    faithfulness_values = [
        score.faithfulness
        for score in scores
        if score.faithfulness is not None
    ]

    context_precision_values = [
        score.context_precision
        for score in scores
        if score.context_precision
        is not None
    ]

    answer_relevancy_values = [
        score.answer_relevancy
        for score in scores
        if score.answer_relevancy
        is not None
    ]

    scored_records = sum(
        1
        for score in scores
        if any(
            value is not None
            for value in (
                score.faithfulness,
                score.context_precision,
                score.answer_relevancy,
            )
        )
    )

    return RagasMetricSummary(
        total_records=len(
            scores
        ),
        scored_records=(
            scored_records
        ),
        faithfulness_mean=(
            _mean_or_none(
                faithfulness_values
            )
        ),
        context_precision_mean=(
            _mean_or_none(
                context_precision_values
            )
        ),
        answer_relevancy_mean=(
            _mean_or_none(
                answer_relevancy_values
            )
        ),
    )


async def evaluate_run_artifact_async(
    *,
    run: EvaluationRunArtifact,
    dataset: EvaluationDataset,
    settings: Settings | None = None,
) -> RagasEvaluationArtifact:
    """
    Perform offline semantic evaluation of one immutable
    execution artifact.
    """
    settings = (
        settings
        or get_settings()
    )

    validate_run_against_dataset(
        run=run,
        dataset=dataset,
    )

    questions_by_id = {
        question.question_id: question
        for question in dataset.questions
    }

    (
        faithfulness_metric,
        context_precision_metric,
        answer_relevancy_metric,
    ) = _create_ragas_components(
        settings
    )

    scores: list[
        RagasQuestionScore
    ] = []

    total_records = len(
        run.results
    )

    for index, result_record in enumerate(
        run.results,
        start=1,
    ):
        print(
            (
                f"[RAGAS {index}/"
                f"{total_records}] "
                f"{result_record.question_id}"
            )
        )

        question = (
            questions_by_id[
                result_record.question_id
            ]
        )

        score = await _score_one_question(
            result_record=(
                result_record
            ),
            reference_answer=(
                question.reference_answer
            ),
            faithfulness_metric=(
                faithfulness_metric
            ),
            context_precision_metric=(
                context_precision_metric
            ),
            answer_relevancy_metric=(
                answer_relevancy_metric
            ),
        )

        scores.append(
            score
        )

    return RagasEvaluationArtifact(
        created_at=datetime.now(
            timezone.utc
        ),
        source_run_id=(
            run.run_id
        ),
        system=(
            run.system
        ),
        dataset_name=(
            run.dataset_name
        ),
        dataset_version=(
            run.dataset_version
        ),
        corpus_version=(
            run.corpus_version
        ),
        evaluator_model=(
            settings.ragas_evaluator_model
        ),
        embedding_model=(
            settings.embedding_model
        ),
        scores=scores,
        summary=(
            _build_summary(
                scores
            )
        ),
    )


def evaluate_run_artifact(
    *,
    run: EvaluationRunArtifact,
    dataset: EvaluationDataset,
    settings: Settings | None = None,
) -> RagasEvaluationArtifact:
    """
    Synchronous entry point used by the CLI and unit tests.
    """
    return asyncio.run(
        evaluate_run_artifact_async(
            run=run,
            dataset=dataset,
            settings=settings,
        )
    )


def save_ragas_artifact(
    artifact: RagasEvaluationArtifact,
    path: str | Path,
) -> Path:
    """
    Persist semantic scores separately from the source
    execution artifact.
    """
    output_path = Path(
        path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            artifact.model_dump(
                mode="json"
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_path


def default_ragas_result_path(
    run: EvaluationRunArtifact,
) -> Path:
    """
    Create a deterministic output filename derived from the
    immutable source run ID.
    """
    return (
        RESULTS_DIRECTORY
        / f"{run.run_id}-ragas.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline Ragas semantic evaluation "
            "over one saved MedOntoAgent execution artifact."
        )
    )

    parser.add_argument(
        "--run",
        required=True,
        help=(
            "Path to a saved baseline or MedOnto "
            "EvaluationRunArtifact JSON file."
        ),
    )

    parser.add_argument(
        "--dataset",
        default=str(
            DEFAULT_DATASET_PATH
        ),
        help=(
            "Path to the exact evaluation dataset "
            "used by the source run."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output JSON path."
        ),
    )

    args = parser.parse_args()

    run = load_run_artifact(
        args.run
    )

    dataset = (
        load_evaluation_dataset(
            args.dataset
        )
    )

    settings = get_settings()

    artifact = (
        evaluate_run_artifact(
            run=run,
            dataset=dataset,
            settings=settings,
        )
    )

    output_path = (
        Path(args.output)
        if args.output
        else default_ragas_result_path(
            run
        )
    )

    saved_path = (
        save_ragas_artifact(
            artifact,
            output_path,
        )
    )

    print()
    print(
        "Ragas semantic evaluation complete."
    )
    print(
        f"Source run: {run.run_id}"
    )
    print(
        f"System: {run.system}"
    )
    print(
        f"Scored records: "
        f"{artifact.summary.scored_records}/"
        f"{artifact.summary.total_records}"
    )
    print(
        f"Result: {saved_path}"
    )


if __name__ == "__main__":
    main()
