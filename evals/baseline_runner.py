import argparse
import time
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path

from app.config.settings import (
    Settings,
    get_settings,
)
from app.llm.baseline import (
    run_baseline_rag,
)
from app.models.evaluation import (
    EvaluationDataset,
    EvaluationResultRecord,
    EvaluationRunArtifact,
)
from evals.common import (
    create_run_id,
    default_result_path,
    evaluation_configuration,
    expected_behavior_matches,
    load_evaluation_dataset,
    save_run_artifact,
)


DEFAULT_DATASET_PATH = Path(
    "evals/development_dataset.json"
)


def run_baseline_dataset(
    dataset: EvaluationDataset,
    *,
    settings: Settings | None = None,
) -> EvaluationRunArtifact:
    """
    Execute conventional baseline RAG over every dataset question.

    One question failure is recorded and does not terminate
    the remaining evaluation run.
    """
    settings = (
        settings
        or get_settings()
    )

    run_id = create_run_id(
        "baseline"
    )

    results: list[
        EvaluationResultRecord
    ] = []

    total_questions = len(
        dataset.questions
    )

    for index, question in enumerate(
        dataset.questions,
        start=1,
    ):
        print(
            (
                f"[BASELINE {index}/"
                f"{total_questions}] "
                f"{question.question_id}"
            )
        )

        started = (
            time.perf_counter()
        )

        try:
            response = (
                run_baseline_rag(
                    question.question,
                    settings=settings,
                )
            )

            latency_ms = (
                (
                    time.perf_counter()
                    - started
                )
                * 1000.0
            )

            citation_chunk_ids = [
                citation.chunk_id
                for citation
                in response.citations
            ]

            result = (
                EvaluationResultRecord(
                    question_id=(
                        question.question_id
                    ),
                    category=(
                        question.category
                    ),
                    question=(
                        question.question
                    ),
                    expected_behavior=(
                        question.expected_behavior
                    ),
                    system="BASELINE",
                    system_status=(
                        response.status
                    ),
                    behavior_match=(
                        expected_behavior_matches(
                            expected_behavior=(
                                question
                                .expected_behavior
                            ),
                            system_status=(
                                response.status
                            ),
                        )
                    ),
                    answer=(
                        response.answer
                    ),
                    citation_chunk_ids=(
                        citation_chunk_ids
                    ),
                    retrieved_chunk_ids=(
                        response
                        .retrieved_chunk_ids
                    ),
                    ontology_constraints=[],
                    used_ontology_filter=None,
                    retrieval_relaxed=None,
                    retry_count=None,
                    latency_ms=(
                        latency_ms
                    ),
                    error_message=None,
                )
            )

        except Exception as exc:
            latency_ms = (
                (
                    time.perf_counter()
                    - started
                )
                * 1000.0
            )

            result = (
                EvaluationResultRecord(
                    question_id=(
                        question.question_id
                    ),
                    category=(
                        question.category
                    ),
                    question=(
                        question.question
                    ),
                    expected_behavior=(
                        question.expected_behavior
                    ),
                    system="BASELINE",
                    system_status="ERROR",
                    behavior_match=False,
                    answer=None,
                    citation_chunk_ids=[],
                    retrieved_chunk_ids=[],
                    ontology_constraints=[],
                    used_ontology_filter=None,
                    retrieval_relaxed=None,
                    retry_count=None,
                    latency_ms=(
                        latency_ms
                    ),
                    error_message=(
                        "Execution failed: "
                        f"{type(exc).__name__}"
                    ),
                )
            )

        results.append(
            result
        )

    return EvaluationRunArtifact(
        run_id=run_id,
        created_at=datetime.now(
            timezone.utc
        ),
        system="BASELINE",
        dataset_name=(
            dataset.dataset_name
        ),
        dataset_version=(
            dataset.dataset_version
        ),
        corpus_version=(
            dataset.corpus_version
        ),
        configuration=(
            evaluation_configuration(
                settings
            )
        ),
        results=results,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the MedOntoAgent conventional "
            "baseline evaluation."
        )
    )

    parser.add_argument(
        "--dataset",
        default=str(
            DEFAULT_DATASET_PATH
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    dataset = (
        load_evaluation_dataset(
            args.dataset
        )
    )

    settings = get_settings()

    artifact = (
        run_baseline_dataset(
            dataset,
            settings=settings,
        )
    )

    output_path = (
        Path(args.output)
        if args.output
        else default_result_path(
            artifact.run_id
        )
    )

    saved_path = (
        save_run_artifact(
            artifact,
            output_path,
        )
    )

    print()
    print(
        "Baseline evaluation complete."
    )
    print(
        f"Run ID: {artifact.run_id}"
    )
    print(
        f"Result: {saved_path}"
    )


if __name__ == "__main__":
    main()