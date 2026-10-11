import argparse
import time
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path

from app.agent.execution import (
    execute_medonto,
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
    Convert application-owned ontology constraints into stable
    human-readable relationship strings.
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
            constraints = constraint_set
        else:
            return []

    results: list[str] = []

    for constraint in constraints:
        if not getattr(
            constraint,
            "control_eligible",
            False,
        ):
            continue

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

        if relationship not in results:
            results.append(
                relationship
            )

    return results


def _citation_chunk_ids(
    draft_output,
    *,
    verified: bool,
) -> list[str]:
    """
    Extract unique final citation IDs from verified claims.
    """
    if (
        draft_output is None
        or not verified
    ):
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
    Extract retrieved chunk IDs in retrieval order.
    """
    if retrieval_result is None:
        return []

    return [
        chunk.chunk_id
        for chunk in getattr(
            retrieval_result,
            "retrieved_chunks",
            [],
        )
    ]


def _retrieved_document_ids(
    retrieval_result,
) -> list[str]:
    """
    Extract unique source document IDs directly from retrieval.
    """
    if retrieval_result is None:
        return []

    return list(
        dict.fromkeys(
            chunk.document_id
            for chunk
            in getattr(
                retrieval_result,
                "retrieved_chunks",
                [],
            )
        )
    )


def _retrieved_contexts(
    retrieval_result,
) -> list[str]:
    """
    Preserve exact retrieved context text in retrieval order.
    """
    if retrieval_result is None:
        return []

    return [
        chunk.text
        for chunk in getattr(
            retrieval_result,
            "retrieved_chunks",
            [],
        )
    ]


def _citation_document_ids(
    *,
    draft_output,
    retrieval_result,
    verified: bool,
) -> list[str]:
    """
    Resolve verified citation chunk IDs to actual document IDs.
    """
    if (
        draft_output is None
        or retrieval_result is None
        or not verified
    ):
        return []

    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk
        in getattr(
            retrieval_result,
            "retrieved_chunks",
            [],
        )
    }

    cited_chunk_ids = (
        _citation_chunk_ids(
            draft_output,
            verified=True,
        )
    )

    return list(
        dict.fromkeys(
            chunks_by_id[chunk_id]
            .document_id
            for chunk_id
            in cited_chunk_ids
            if chunk_id in chunks_by_id
        )
    )


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
            execution = execute_medonto(
                question.question,
                settings=settings,
            )

            latency_ms = (
                (
                    time.perf_counter()
                    - started
                )
                * 1000.0
            )

            if execution.execution_mode in {
                "BASELINE_ROUTED",
                "BASELINE_FALLBACK",
            }:
                baseline_response = (
                    execution.baseline_response
                )

                if baseline_response is None:
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
                            execution_mode=(
                                execution.execution_mode
                            ),
                            system_status="ERROR",
                            behavior_match=False,
                            answer=None,
                            citation_chunk_ids=[],
                            citation_document_ids=[],
                            retrieved_chunk_ids=[],
                            retrieved_document_ids=[],
                            retrieved_contexts=[],
                            ontology_constraints=[],
                            used_ontology_filter=False,
                            retrieval_relaxed=False,
                            retry_count=None,
                            latency_ms=(
                                latency_ms
                            ),
                            error_message=(
                                (
                                    "Fallback baseline failed: "
                                    if execution.execution_mode
                                    == "BASELINE_FALLBACK"
                                    else "Routed baseline failed: "
                                )
                                + str(
                                    execution.fallback_error
                                )
                            ),
                            fallback_reason=(
                                execution.fallback_reason
                            ),
                            fallback_error=(
                                execution.fallback_error
                            ),
                        )
                    )

                    results.append(
                        result
                    )

                    continue

                system_status = (
                    baseline_response.status
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
                        execution_mode=(
                            execution.execution_mode
                        ),
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
                        answer=(
                            baseline_response.answer
                        ),
                        citation_chunk_ids=[
                            citation.chunk_id
                            for citation
                            in (
                                baseline_response
                                .citations
                            )
                        ],
                        citation_document_ids=(
                            baseline_response
                            .citation_document_ids
                        ),
                        retrieved_chunk_ids=(
                            baseline_response
                            .retrieved_chunk_ids
                        ),
                        retrieved_document_ids=(
                            baseline_response
                            .retrieved_document_ids
                        ),
                        retrieved_contexts=(
                            baseline_response
                            .retrieved_contexts
                        ),
                        ontology_constraints=[],
                        used_ontology_filter=False,
                        retrieval_relaxed=False,
                        retry_count=None,
                        latency_ms=(
                            latency_ms
                        ),
                        error_message=None,
                        fallback_reason=(
                            execution
                            .fallback_reason
                        ),
                        fallback_error=None,
                    )
                )

                results.append(
                    result
                )

                continue

            state = execution.medonto_state

            if state is None:
                raise RuntimeError(
                    (
                        "MEDONTO execution did "
                        "not return graph state."
                    )
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

            verified = (
                system_status
                == "VERIFIED"
            )

            draft_output = state.get(
                "draft_output"
            )

            retrieval_result = state.get(
                "retrieval_result"
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
                    execution_mode="MEDONTO",
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
                            draft_output,
                            verified=verified,
                        )
                    ),
                    citation_document_ids=(
                        _citation_document_ids(
                            draft_output=(
                                draft_output
                            ),
                            retrieval_result=(
                                retrieval_result
                            ),
                            verified=verified,
                        )
                    ),
                    retrieved_chunk_ids=(
                        _retrieved_chunk_ids(
                            retrieval_result
                        )
                    ),
                    retrieved_document_ids=(
                        _retrieved_document_ids(
                            retrieval_result
                        )
                    ),
                    retrieved_contexts=(
                        _retrieved_contexts(
                            retrieval_result
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
                    fallback_reason=None,
                    fallback_error=None,
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
                    execution_mode="MEDONTO",
                    system_status="ERROR",
                    behavior_match=False,
                    answer=None,
                    citation_chunk_ids=[],
                    citation_document_ids=[],
                    retrieved_chunk_ids=[],
                    retrieved_document_ids=[],
                    retrieved_contexts=[],
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
                    fallback_reason=None,
                    fallback_error=None,
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
