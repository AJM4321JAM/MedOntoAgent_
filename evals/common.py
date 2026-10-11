import json
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path

from pydantic import ValidationError

from app.config.settings import (
    Settings,
)
from app.models.evaluation import (
    EvaluationDataset,
    EvaluationRunArtifact,
    ExpectedBehavior,
    validate_final_dataset_ready,
)


class EvaluationConfigurationError(
    RuntimeError
):
    """
    Raised when an evaluation dataset is not safe or ready to run.
    """


def load_evaluation_dataset(
    path: str | Path,
) -> EvaluationDataset:
    """
    Load and validate one evaluation dataset from JSON.

    Development datasets may run unfrozen.

    Final datasets must pass the formal readiness check.
    """
    dataset_path = Path(
        path
    )

    if not dataset_path.exists():
        raise EvaluationConfigurationError(
            (
                "Evaluation dataset does not exist: "
                f"{dataset_path}"
            )
        )

    try:
        raw_data = json.loads(
            dataset_path.read_text(
                encoding="utf-8",
            )
        )

        dataset = (
            EvaluationDataset.model_validate(
                raw_data
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
        ValidationError,
    ) as exc:
        raise EvaluationConfigurationError(
            (
                "Evaluation dataset could not "
                "be loaded or validated."
            )
        ) from exc

    if dataset.split == "FINAL":
        problems = (
            validate_final_dataset_ready(
                dataset
            )
        )

        if problems:
            details = "\n".join(
                f"- {problem}"
                for problem
                in problems
            )

            raise EvaluationConfigurationError(
                (
                    "Final evaluation dataset "
                    "is not ready:\n"
                    f"{details}"
                )
            )

    return dataset


def expected_behavior_matches(
    *,
    expected_behavior: ExpectedBehavior,
    system_status: str,
) -> bool:
    """
    Compare terminal behavior only.

    This is not an answer-quality or clinical-correctness metric.
    """
    if expected_behavior == "ANSWER":
        return system_status in {
            "ANSWERED",
            "VERIFIED",
        }

    return (
        system_status
        == "ABSTAINED"
    )


def create_run_id(
    system: str,
) -> str:
    """
    Create a filesystem-safe UTC run identifier.
    """
    timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
    )

    return (
        f"{system.lower()}-"
        f"{timestamp}"
    )


def default_result_path(
    run_id: str,
) -> Path:
    """
    Return the default local path for a generated run artifact.
    """
    return (
        Path("evals")
        / "results"
        / f"{run_id}.json"
    )


def evaluation_configuration(
    settings: Settings,
) -> dict[
    str,
    str | int | float | bool | None,
]:
    """
    Build a reproducible configuration snapshot without secrets.
    """
    return {
        "generation_model": (
            settings.gemini_model
        ),
        "embedding_model": (
            settings.embedding_model
        ),
        "embedding_dimensions": (
            settings.embedding_dimensions
        ),
        "embedding_device": (
            settings.embedding_device
        ),
        "retrieval_top_k": (
            settings.retrieval_top_k
        ),
        "max_agent_retries": (
            settings.max_agent_retries
        ),
        "max_chunk_words": (
            settings.max_chunk_words
        ),
        "chunk_overlap_words": (
            settings.chunk_overlap_words
        ),
    }


def save_run_artifact(
    artifact: EvaluationRunArtifact,
    path: str | Path,
) -> Path:
    """
    Persist one artifact using temporary-file replacement.

    This reduces the chance of leaving a partially written JSON
    result that appears complete.
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

    serialized = json.dumps(
        artifact.model_dump(
            mode="json",
        ),
        indent=2,
        ensure_ascii=False,
    )

    temporary_path.write_text(
        serialized + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(
        output_path
    )

    return output_path