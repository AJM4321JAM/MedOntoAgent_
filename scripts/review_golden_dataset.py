import json
from collections import Counter
from pathlib import Path

from app.models.evaluation import (
    EvaluationDataset,
    validate_final_dataset_ready,
)


EXPECTED_DISTRIBUTION = {
    "STRAIGHTFORWARD_EVIDENCE": 7,
    "ONTOLOGY_CONSTRAINT": 6,
    "PATIENT_GROUP": 4,
    "GUIDELINE_VERSION": 3,
    "NO_EVIDENCE": 3,
    "DIFFICULT_AMBIGUOUS": 2,
}


def main() -> None:
    path = Path(
        "evals/golden_dataset.json"
    )

    raw = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    dataset = (
        EvaluationDataset
        .model_validate(
            raw
        )
    )

    problems = (
        validate_final_dataset_ready(
            dataset
        )
    )

    counts = Counter(
        question.category
        for question in dataset.questions
    )

    if len(dataset.questions) != 25:
        problems.append(
            (
                "Final golden dataset must contain "
                "exactly 25 questions."
            )
        )

    if counts != Counter(
        EXPECTED_DISTRIBUTION
    ):
        problems.append(
            (
                "Category distribution does not match "
                "the required 7/6/4/3/3/2 split. "
                f"Observed: {dict(counts)}"
            )
        )

    for question in dataset.questions:
        if (
            question.expected_behavior
            == "ANSWER"
            and not question.reference_answer
        ):
            problems.append(
                (
                    f"{question.question_id} "
                    "has no reference answer."
                )
            )

        if (
            question.expected_behavior
            == "ANSWER"
            and not question.expected_document_ids
        ):
            problems.append(
                (
                    f"{question.question_id} "
                    "has no expected document IDs."
                )
            )

        if (
            question.expected_behavior
            == "ANSWER"
            and not question.expected_chunk_ids
        ):
            problems.append(
                (
                    f"{question.question_id} "
                    "has no expected chunk IDs."
                )
            )

        if (
            question.category
            == "ONTOLOGY_CONSTRAINT"
            and not (
                question
                .expected_ontology_relationships
            )
        ):
            problems.append(
                (
                    f"{question.question_id} "
                    "has no expected ontology relationship."
                )
            )

        if (
            question.expected_behavior
            == "ABSTAIN"
            and (
                question.expected_document_ids
                or question.expected_chunk_ids
            )
        ):
            problems.append(
                (
                    f"{question.question_id} "
                    "is ABSTAIN but has expected evidence IDs."
                )
            )

    if problems:
        print(
            "Golden dataset readiness check failed."
        )
        print()

        for problem in problems:
            print(
                "-",
                problem,
            )

        raise SystemExit(1)

    print(
        "Golden dataset readiness check passed."
    )

    print(
        "DATASET VERSION:",
        dataset.dataset_version,
    )

    print(
        "CORPUS VERSION:",
        dataset.corpus_version,
    )

    print(
        "QUESTIONS:",
        len(dataset.questions),
    )

    print(
        "DISTRIBUTION:",
        dict(counts),
    )


if __name__ == "__main__":
    main()
