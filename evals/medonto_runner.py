import argparse
import time
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path

from app.agent.graph import (
    medonto_graph,
)
from app.config.settings import (
    Settings,
    get_settings,
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


def _constraint_strings(
    constraint_set,
) -> list[str]:
    """
    Convert explicit ontology constraints into stable
    human-readable relationship strings.

    The helper deliberately reads application-owned constraint
    objects rather than model-generated prose.
    """
    if constraint_set is None:
        return []

    constraints = getattr(
        constraint_set,
        "constraints",
        None,
    )

    if constraints is None:
        if isinstance(
            constraint_set,
            (list, tuple),
        ):
            constraints = (
                constraint_set
            )
        else:
            return []

    results: list[str] = []

    for constraint in constraints:
        subject = getattr(
            constraint,
            "subject_name",
            None,
        )

        predicate = getattr(
            constraint,
            "predicate_name",
            None,
        )

        object_name = getattr(
            constraint,
            "object_name",
            None,
        )

        if (
            subject
            and predicate
            and object_name
        ):
            relationship = (
                f"{subject} "
                f"{predicate} "
                f"{object_name}"
            )

        else:
            relationship = str(
                constraint
            )

        if (
            relationship
            not in results
        ):
            results.append(
                relationship
            )

    return results


def _citation_chunk_ids(
    draft_output,
) -> list[str]:
    """
    Extract unique citation IDs from MedOnto draft claims.
    """
    if draft_output is None:
        return []

    results: list[str] = []

    for claim in getattr(
        draft_output,
        "claims",
        [],
    ):
        for chunk_id in getattr(
            claim,
            "citation_chunk_ids",
            [],
        ):
            if chunk_id not in results:
                results.append(
                    chunk_id
                )

    return results


def _retrieved_chunk_ids(
    retrieval_result,
) -> list[str]:
    """
    Extract retrieval IDs from the graph retrieval result.
    """
    if retrieval_result is None:
        return []

    return [
        chunk.chunk_id
        for chunk
        in getattr(
            retrieval_result,
            "retrieved_chunks",
            [],
        )
    ]


def run_medonto_dataset(
    dataset: EvaluationDataset,
    *,
    settings: Settings | None = None,
) -> EvaluationRunArtifact:
    """
    Execute MedOntoAgent over every dataset question.

    One question failure is recorded and does not terminate
    the remaining evaluation run.
    """
    settings = (
        settings
        or get_settings()
    )

    run_id = create_run_id(
        "medonto"
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
                f"[MEDONTO {index}/"
                f"{total_questions}] "
                f"{question.question_id}"
            )
        )

        started = (
            time.perf_counter()
        )

        try:
            state = (
                medonto_graph.invoke(
                    {
                        "query": (
                            question.question
                        )
                    }
                )
            )

            latency_ms = (
                (
                    time.perf_counter()
                    - started
                )
                * 1000.0
            )

            orchestration_status = (
                state.get(
                    "orchestration_status",
                    "ERROR",
                )
            )

            if orchestration_status not in {
                "VERIFIED",
                "ABSTAINED",
            }:
                system_status = "ERROR"
            else:
                system_status = (
                    orchestration_status
                )

            draft_output = state.get(
                "draft_output"
            )

            answer: str | None = None

            if draft_output is not None:
                answer = getattr(
                    draft_output,
                    "answer",
                    None,
                )

            if (
                answer is None
                and system_status
                == "ABSTAINED"
            ):
                answer = (
                    state.get(
                        "abstention_reason"
                    )
                    or (
                        "The approved retrieved "
                        "evidence is insufficient "
                        "to answer the question."
                    )
                )

            error_message = None

            if system_status == "ERROR":
                error_message = (
                    "Execution ended without "
                    "a valid terminal state."
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
                    system="MEDONTO",
                    system_status=(
                        system_status
                    ),
                    behavior_match=(
                        expected_behavior_matches(
                            expected_behavior=(
                                question
                                .expected_behavior
                            ),
                            system_status=(
                                system_status
                            ),
                        )
                    ),
                    answer=answer,
                    citation_chunk_ids=(
                        _citation_chunk_ids(
                            draft_output
                        )
                    ),
                    retrieved_chunk_ids=(
                        _retrieved_chunk_ids(
                            state.get(
                                "retrieval_result"
                            )
                        )
                    ),
                    ontology_constraints=(
                        _constraint_strings(
                            state.get(
                                "constraint_set"
                            )
                        )
                    ),
                    used_ontology_filter=(
                        state.get(
                            "used_ontology_filter"
                        )
                    ),
                    retrieval_relaxed=(
                        state.get(
                            "retrieval_relaxed"
                        )
                    ),
                    retry_count=(
                        state.get(
                            "retry_count"
                        )
                    ),
                    latency_ms=(
                        latency_ms
                    ),
                    error_message=(
                        error_message
                    ),
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
                    system="MEDONTO",
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
        system="MEDONTO",
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
            "Run the MedOntoAgent ontology-aware "
            "evaluation."
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
        run_medonto_dataset(
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
        "MedOntoAgent evaluation complete."
    )
    print(
        f"Run ID: {artifact.run_id}"
    )
    print(
        f"Result: {saved_path}"
    )


if __name__ == "__main__":
    main()