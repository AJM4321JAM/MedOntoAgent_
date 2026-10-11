import argparse
import json
import math
import statistics
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path

from pydantic import ValidationError

from app.models.evaluation import (
    DeterministicMetricSummary,
    EvaluationComparisonArtifact,
    EvaluationDataset,
    EvaluationResultRecord,
    EvaluationRunArtifact,
    LatencySummary,
    QuestionComparison,
)
from evals.common import (
    load_evaluation_dataset,
)


class EvaluationComparisonError(
    RuntimeError
):
    """
    Raised when evaluation artifacts cannot be compared safely.
    """


SHARED_CONFIGURATION_KEYS = (
    "generation_model",
    "embedding_model",
    "embedding_dimensions",
    "retrieval_top_k",
)


def load_run_artifact(
    path: str | Path,
) -> EvaluationRunArtifact:
    """
    Load and validate one evaluation run artifact.
    """
    artifact_path = Path(
        path
    )

    if not artifact_path.exists():
        raise EvaluationComparisonError(
            (
                "Evaluation run artifact "
                f"does not exist: {artifact_path}"
            )
        )

    try:
        raw_data = json.loads(
            artifact_path.read_text(
                encoding="utf-8",
            )
        )

        return (
            EvaluationRunArtifact
            .model_validate(
                raw_data
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
        ValidationError,
    ) as exc:
        raise EvaluationComparisonError(
            (
                "Evaluation run artifact "
                "could not be loaded or validated."
            )
        ) from exc


def _result_lookup(
    artifact: EvaluationRunArtifact,
) -> dict[
    str,
    EvaluationResultRecord,
]:
    """
    Build a unique question-ID lookup for one run.
    """
    lookup: dict[
        str,
        EvaluationResultRecord,
    ] = {}

    for result in artifact.results:
        if result.question_id in lookup:
            raise EvaluationComparisonError(
                (
                    "Run artifact contains "
                    "duplicate question IDs."
                )
            )

        lookup[
            result.question_id
        ] = result

    return lookup


def _validate_artifact_against_dataset(
    *,
    dataset: EvaluationDataset,
    artifact: EvaluationRunArtifact,
    expected_system: str,
) -> None:
    """
    Ensure one artifact belongs to the supplied dataset.
    """
    if artifact.system != expected_system:
        raise EvaluationComparisonError(
            (
                f"Expected {expected_system} artifact "
                f"but received {artifact.system}."
            )
        )

    if (
        artifact.dataset_name
        != dataset.dataset_name
    ):
        raise EvaluationComparisonError(
            "Dataset name mismatch."
        )

    if (
        artifact.dataset_version
        != dataset.dataset_version
    ):
        raise EvaluationComparisonError(
            "Dataset version mismatch."
        )

    if (
        artifact.corpus_version
        != dataset.corpus_version
    ):
        raise EvaluationComparisonError(
            "Corpus version mismatch."
        )

    expected_ids = {
        question.question_id
        for question
        in dataset.questions
    }

    actual_ids = set(
        _result_lookup(
            artifact
        )
    )

    if actual_ids != expected_ids:
        raise EvaluationComparisonError(
            (
                "Run artifact question IDs "
                "do not exactly match the dataset."
            )
        )


def _validate_shared_configuration(
    baseline: EvaluationRunArtifact,
    medonto: EvaluationRunArtifact,
) -> None:
    """
    Ensure shared experimental configuration is identical.

    MedOnto-only settings such as max_agent_retries are
    intentionally not required to match.
    """
    mismatches: list[str] = []

    for key in SHARED_CONFIGURATION_KEYS:
        baseline_value = (
            baseline.configuration.get(
                key
            )
        )

        medonto_value = (
            medonto.configuration.get(
                key
            )
        )

        if (
            baseline_value
            != medonto_value
        ):
            mismatches.append(
                (
                    f"{key}: "
                    f"baseline={baseline_value!r}, "
                    f"medonto={medonto_value!r}"
                )
            )

    if mismatches:
        detail = "\n".join(
            f"- {item}"
            for item
            in mismatches
        )

        raise EvaluationComparisonError(
            (
                "Shared evaluation configuration "
                "does not match:\n"
                f"{detail}"
            )
        )


def validate_comparable_runs(
    *,
    dataset: EvaluationDataset,
    baseline: EvaluationRunArtifact,
    medonto: EvaluationRunArtifact,
) -> None:
    """
    Validate that two runs form a controlled comparison.
    """
    _validate_artifact_against_dataset(
        dataset=dataset,
        artifact=baseline,
        expected_system="BASELINE",
    )

    _validate_artifact_against_dataset(
        dataset=dataset,
        artifact=medonto,
        expected_system="MEDONTO",
    )

    if (
        baseline.dataset_name
        != medonto.dataset_name
        or baseline.dataset_version
        != medonto.dataset_version
        or baseline.corpus_version
        != medonto.corpus_version
    ):
        raise EvaluationComparisonError(
            (
                "Baseline and MedOnto artifacts "
                "do not describe the same experiment."
            )
        )

    _validate_shared_configuration(
        baseline,
        medonto,
    )


def _mean_or_none(
    values: list[float],
) -> float | None:
    """
    Return arithmetic mean, or None when no metric is defined.
    """
    if not values:
        return None

    return statistics.fmean(
        values
    )


def _binary_rate(
    values: list[bool],
) -> float | None:
    """
    Calculate a rate from a list of boolean observations.
    """
    if not values:
        return None

    return (
        sum(
            1
            for value
            in values
            if value
        )
        / len(values)
    )


def _retrieval_hit(
    *,
    expected: list[str],
    actual: list[str],
) -> bool | None:
    """
    Return whether retrieval found at least one expected ID.

    No expected annotation means the metric is undefined.
    """
    if not expected:
        return None

    return bool(
        set(expected)
        & set(actual)
    )


def _precision_recall(
    *,
    expected: list[str],
    actual: list[str],
) -> tuple[
    float | None,
    float | None,
]:
    """
    Calculate precision and recall against annotated evidence.

    Missing golden annotation produces N/A.

    When golden annotation exists but the system cites nothing,
    both precision and recall are recorded as zero.
    """
    if not expected:
        return (
            None,
            None,
        )

    expected_set = set(
        expected
    )

    actual_set = set(
        actual
    )

    intersection_count = len(
        expected_set
        & actual_set
    )

    if actual_set:
        precision = (
            intersection_count
            / len(actual_set)
        )
    else:
        precision = 0.0

    recall = (
        intersection_count
        / len(expected_set)
    )

    return (
        precision,
        recall,
    )


def _ontology_metrics(
    *,
    expected: list[str],
    actual: list[str],
) -> tuple[
    float | None,
    bool | None,
]:
    """
    Calculate ontology relationship recall and exact match.
    """
    if not expected:
        return (
            None,
            None,
        )

    expected_set = set(
        expected
    )

    actual_set = set(
        actual
    )

    recall = (
        len(
            expected_set
            & actual_set
        )
        / len(expected_set)
    )

    exact_match = (
        actual_set
        == expected_set
    )

    return (
        recall,
        exact_match,
    )


def _nearest_rank_p95(
    values: list[float],
) -> float | None:
    """
    Calculate deterministic nearest-rank p95.
    """
    if not values:
        return None

    ordered = sorted(
        values
    )

    rank = math.ceil(
        0.95
        * len(ordered)
    )

    index = max(
        0,
        rank - 1,
    )

    return ordered[
        index
    ]


def _latency_summary(
    results: list[
        EvaluationResultRecord
    ],
) -> LatencySummary:
    """
    Calculate latency statistics for one complete run.
    """
    values = [
        result.latency_ms
        for result
        in results
    ]

    if not values:
        return LatencySummary(
            count=0,
            mean_ms=None,
            median_ms=None,
            p95_ms=None,
        )

    return LatencySummary(
        count=len(values),
        mean_ms=(
            statistics.fmean(
                values
            )
        ),
        median_ms=(
            statistics.median(
                values
            )
        ),
        p95_ms=(
            _nearest_rank_p95(
                values
            )
        ),
    )


def calculate_metrics(
    *,
    dataset: EvaluationDataset,
    artifact: EvaluationRunArtifact,
) -> DeterministicMetricSummary:
    """
    Calculate deterministic benchmark metrics for one system.
    """
    results_by_id = (
        _result_lookup(
            artifact
        )
    )

    total_questions = len(
        dataset.questions
    )

    behavior_values: list[
        bool
    ] = []

    error_values: list[
        bool
    ] = []

    abstention_values: list[
        bool
    ] = []

    document_hits: list[
        bool
    ] = []

    chunk_hits: list[
        bool
    ] = []

    citation_document_precisions: list[
        float
    ] = []

    citation_document_recalls: list[
        float
    ] = []

    citation_chunk_precisions: list[
        float
    ] = []

    citation_chunk_recalls: list[
        float
    ] = []

    ontology_recalls: list[
        float
    ] = []

    ontology_exact_matches: list[
        bool
    ] = []

    retry_observations: list[
        bool
    ] = []

    relaxation_observations: list[
        bool
    ] = []

    ordered_results: list[
        EvaluationResultRecord
    ] = []

    for question in dataset.questions:
        result = results_by_id.get(
            question.question_id
        )

        if result is None:
            raise EvaluationComparisonError(
                (
                    "Missing result for "
                    f"{question.question_id}."
                )
            )

        ordered_results.append(
            result
        )

        behavior_values.append(
            result.behavior_match
        )

        error_values.append(
            result.system_status
            == "ERROR"
        )

        abstention_values.append(
            result.system_status
            == "ABSTAINED"
        )

        document_hit = (
            _retrieval_hit(
                expected=(
                    question
                    .expected_document_ids
                ),
                actual=(
                    result
                    .retrieved_document_ids
                ),
            )
        )

        if document_hit is not None:
            document_hits.append(
                document_hit
            )

        chunk_hit = (
            _retrieval_hit(
                expected=(
                    question
                    .expected_chunk_ids
                ),
                actual=(
                    result
                    .retrieved_chunk_ids
                ),
            )
        )

        if chunk_hit is not None:
            chunk_hits.append(
                chunk_hit
            )

        (
            document_precision,
            document_recall,
        ) = _precision_recall(
            expected=(
                question
                .expected_document_ids
            ),
            actual=(
                result
                .citation_document_ids
            ),
        )

        if (
            document_precision
            is not None
        ):
            citation_document_precisions.append(
                document_precision
            )

        if (
            document_recall
            is not None
        ):
            citation_document_recalls.append(
                document_recall
            )

        (
            chunk_precision,
            chunk_recall,
        ) = _precision_recall(
            expected=(
                question
                .expected_chunk_ids
            ),
            actual=(
                result
                .citation_chunk_ids
            ),
        )

        if (
            chunk_precision
            is not None
        ):
            citation_chunk_precisions.append(
                chunk_precision
            )

        if (
            chunk_recall
            is not None
        ):
            citation_chunk_recalls.append(
                chunk_recall
            )

        (
            ontology_recall,
            ontology_exact,
        ) = _ontology_metrics(
            expected=(
                question
                .expected_ontology_relationships
            ),
            actual=(
                result
                .ontology_constraints
            ),
        )

        if (
            ontology_recall
            is not None
        ):
            ontology_recalls.append(
                ontology_recall
            )

        if (
            ontology_exact
            is not None
        ):
            ontology_exact_matches.append(
                ontology_exact
            )

        if (
            result.retry_count
            is not None
        ):
            retry_observations.append(
                result.retry_count
                > 0
            )

        if (
            result.retrieval_relaxed
            is not None
        ):
            relaxation_observations.append(
                result.retrieval_relaxed
            )

    return DeterministicMetricSummary(
        total_questions=(
            total_questions
        ),
        behavior_match_rate=(
            _binary_rate(
                behavior_values
            )
        ),
        error_rate=(
            _binary_rate(
                error_values
            )
        ),
        abstention_rate=(
            _binary_rate(
                abstention_values
            )
        ),
        expected_document_retrieval_hit_rate=(
            _binary_rate(
                document_hits
            )
        ),
        expected_chunk_retrieval_hit_rate=(
            _binary_rate(
                chunk_hits
            )
        ),
        citation_document_precision=(
            _mean_or_none(
                citation_document_precisions
            )
        ),
        citation_document_recall=(
            _mean_or_none(
                citation_document_recalls
            )
        ),
        citation_chunk_precision=(
            _mean_or_none(
                citation_chunk_precisions
            )
        ),
        citation_chunk_recall=(
            _mean_or_none(
                citation_chunk_recalls
            )
        ),
        ontology_relationship_recall=(
            _mean_or_none(
                ontology_recalls
            )
        ),
        ontology_exact_match_rate=(
            _binary_rate(
                ontology_exact_matches
            )
        ),
        retry_rate=(
            _binary_rate(
                retry_observations
            )
        ),
        retrieval_relaxation_rate=(
            _binary_rate(
                relaxation_observations
            )
        ),
        latency=(
            _latency_summary(
                ordered_results
            )
        ),
    )


def _question_comparison(
    *,
    question,
    baseline_result: EvaluationResultRecord,
    medonto_result: EvaluationResultRecord,
) -> QuestionComparison:
    """
    Build one side-by-side question record.
    """
    baseline_document_hit = (
        _retrieval_hit(
            expected=(
                question
                .expected_document_ids
            ),
            actual=(
                baseline_result
                .retrieved_document_ids
            ),
        )
    )

    medonto_document_hit = (
        _retrieval_hit(
            expected=(
                question
                .expected_document_ids
            ),
            actual=(
                medonto_result
                .retrieved_document_ids
            ),
        )
    )

    baseline_chunk_hit = (
        _retrieval_hit(
            expected=(
                question
                .expected_chunk_ids
            ),
            actual=(
                baseline_result
                .retrieved_chunk_ids
            ),
        )
    )

    medonto_chunk_hit = (
        _retrieval_hit(
            expected=(
                question
                .expected_chunk_ids
            ),
            actual=(
                medonto_result
                .retrieved_chunk_ids
            ),
        )
    )

    expected_relationships = (
        question
        .expected_ontology_relationships
    )

    if expected_relationships:
        expected_set = set(
            expected_relationships
        )

        baseline_ontology_found = (
            expected_set.issubset(
                set(
                    baseline_result
                    .ontology_constraints
                )
            )
        )

        medonto_ontology_found = (
            expected_set.issubset(
                set(
                    medonto_result
                    .ontology_constraints
                )
            )
        )

    else:
        baseline_ontology_found = None
        medonto_ontology_found = None

    return QuestionComparison(
        question_id=(
            question.question_id
        ),
        category=(
            question.category
        ),
        expected_behavior=(
            question.expected_behavior
        ),
        baseline_status=(
            baseline_result
            .system_status
        ),
        medonto_status=(
            medonto_result
            .system_status
        ),
        baseline_behavior_match=(
            baseline_result
            .behavior_match
        ),
        medonto_behavior_match=(
            medonto_result
            .behavior_match
        ),
        baseline_retrieved_expected_document=(
            baseline_document_hit
        ),
        medonto_retrieved_expected_document=(
            medonto_document_hit
        ),
        baseline_retrieved_expected_chunk=(
            baseline_chunk_hit
        ),
        medonto_retrieved_expected_chunk=(
            medonto_chunk_hit
        ),
        baseline_expected_ontology_relationships_found=(
            baseline_ontology_found
        ),
        medonto_expected_ontology_relationships_found=(
            medonto_ontology_found
        ),
        baseline_latency_ms=(
            baseline_result
            .latency_ms
        ),
        medonto_latency_ms=(
            medonto_result
            .latency_ms
        ),
    )


def create_comparison(
    *,
    dataset: EvaluationDataset,
    baseline: EvaluationRunArtifact,
    medonto: EvaluationRunArtifact,
) -> EvaluationComparisonArtifact:
    """
    Create one deterministic Baseline-vs-MedOnto comparison.
    """
    validate_comparable_runs(
        dataset=dataset,
        baseline=baseline,
        medonto=medonto,
    )

    baseline_lookup = (
        _result_lookup(
            baseline
        )
    )

    medonto_lookup = (
        _result_lookup(
            medonto
        )
    )

    question_comparisons = [
        _question_comparison(
            question=question,
            baseline_result=(
                baseline_lookup[
                    question.question_id
                ]
            ),
            medonto_result=(
                medonto_lookup[
                    question.question_id
                ]
            ),
        )
        for question
        in dataset.questions
    ]

    return EvaluationComparisonArtifact(
        created_at=datetime.now(
            timezone.utc
        ),
        dataset_name=(
            dataset.dataset_name
        ),
        dataset_version=(
            dataset.dataset_version
        ),
        corpus_version=(
            dataset.corpus_version
        ),
        baseline_run_id=(
            baseline.run_id
        ),
        medonto_run_id=(
            medonto.run_id
        ),
        baseline_metrics=(
            calculate_metrics(
                dataset=dataset,
                artifact=baseline,
            )
        ),
        medonto_metrics=(
            calculate_metrics(
                dataset=dataset,
                artifact=medonto,
            )
        ),
        questions=(
            question_comparisons
        ),
    )


def save_comparison_artifact(
    artifact: EvaluationComparisonArtifact,
    path: str | Path,
) -> Path:
    """
    Persist comparison using temporary-file replacement.
    """
    output_path = Path(
        path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        output_path.with_suffix(
            output_path.suffix
            + ".tmp"
        )
    )

    temporary_path.write_text(
        json.dumps(
            artifact.model_dump(
                mode="json"
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(
        output_path
    )

    return output_path


def _format_metric(
    value: float | None,
) -> str:
    """
    Format one console metric.
    """
    if value is None:
        return "N/A"

    return f"{value:.4f}"


def _print_metric_row(
    name: str,
    baseline_value: float | None,
    medonto_value: float | None,
) -> None:
    print(
        f"{name:<40}"
        f"{_format_metric(baseline_value):>12}"
        f"{_format_metric(medonto_value):>12}"
    )


def print_comparison(
    artifact: EvaluationComparisonArtifact,
) -> None:
    """
    Print a compact deterministic comparison table.
    """
    baseline = (
        artifact.baseline_metrics
    )

    medonto = (
        artifact.medonto_metrics
    )

    print()
    print(
        "DETERMINISTIC COMPARISON"
    )

    print(
        (
            f"Dataset: "
            f"{artifact.dataset_name} "
            f"{artifact.dataset_version}"
        )
    )

    print()

    print(
        f"{'Metric':<40}"
        f"{'Baseline':>12}"
        f"{'MedOnto':>12}"
    )

    print(
        "-" * 64
    )

    _print_metric_row(
        "Behavior match rate",
        baseline.behavior_match_rate,
        medonto.behavior_match_rate,
    )

    _print_metric_row(
        "Error rate",
        baseline.error_rate,
        medonto.error_rate,
    )

    _print_metric_row(
        "Abstention rate",
        baseline.abstention_rate,
        medonto.abstention_rate,
    )

    _print_metric_row(
        "Expected document retrieval hit",
        (
            baseline
            .expected_document_retrieval_hit_rate
        ),
        (
            medonto
            .expected_document_retrieval_hit_rate
        ),
    )

    _print_metric_row(
        "Expected chunk retrieval hit",
        (
            baseline
            .expected_chunk_retrieval_hit_rate
        ),
        (
            medonto
            .expected_chunk_retrieval_hit_rate
        ),
    )

    _print_metric_row(
        "Citation document precision",
        baseline.citation_document_precision,
        medonto.citation_document_precision,
    )

    _print_metric_row(
        "Citation document recall",
        baseline.citation_document_recall,
        medonto.citation_document_recall,
    )

    _print_metric_row(
        "Citation chunk precision",
        baseline.citation_chunk_precision,
        medonto.citation_chunk_precision,
    )

    _print_metric_row(
        "Citation chunk recall",
        baseline.citation_chunk_recall,
        medonto.citation_chunk_recall,
    )

    _print_metric_row(
        "Ontology relationship recall",
        (
            baseline
            .ontology_relationship_recall
        ),
        (
            medonto
            .ontology_relationship_recall
        ),
    )

    _print_metric_row(
        "Ontology exact match",
        baseline.ontology_exact_match_rate,
        medonto.ontology_exact_match_rate,
    )

    _print_metric_row(
        "Retry rate",
        baseline.retry_rate,
        medonto.retry_rate,
    )

    _print_metric_row(
        "Retrieval relaxation rate",
        (
            baseline
            .retrieval_relaxation_rate
        ),
        (
            medonto
            .retrieval_relaxation_rate
        ),
    )

    _print_metric_row(
        "Mean latency ms",
        baseline.latency.mean_ms,
        medonto.latency.mean_ms,
    )

    _print_metric_row(
        "Median latency ms",
        baseline.latency.median_ms,
        medonto.latency.median_ms,
    )

    _print_metric_row(
        "P95 latency ms",
        baseline.latency.p95_ms,
        medonto.latency.p95_ms,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one baseline and one "
            "MedOntoAgent evaluation run."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--baseline",
        required=True,
    )

    parser.add_argument(
        "--medonto",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    dataset = (
        load_evaluation_dataset(
            args.dataset
        )
    )

    baseline = (
        load_run_artifact(
            args.baseline
        )
    )

    medonto = (
        load_run_artifact(
            args.medonto
        )
    )

    comparison = (
        create_comparison(
            dataset=dataset,
            baseline=baseline,
            medonto=medonto,
        )
    )

    save_comparison_artifact(
        comparison,
        args.output,
    )

    print_comparison(
        comparison
    )

    print()
    print(
        f"Result: {args.output}"
    )


if __name__ == "__main__":
    main()