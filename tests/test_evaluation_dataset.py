import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.models.evaluation import (
    EvaluationDataset,
    validate_final_dataset_ready,
)


def load_dataset(
    path: str | Path,
) -> EvaluationDataset:
    data = json.loads(
        Path(path).read_text(
            encoding="utf-8",
        )
    )

    return EvaluationDataset.model_validate(
        data
    )


def test_synthetic_dataset_loads() -> None:
    dataset = load_dataset(
        "tests/fixtures/"
        "synthetic_eval_dataset.json"
    )

    assert (
        dataset.dataset_name
        == "Synthetic Test Dataset"
    )

    assert (
        dataset.split
        == "DEVELOPMENT"
    )

    assert len(
        dataset.questions
    ) == 2


def test_development_dataset_loads() -> None:
    dataset = load_dataset(
        "evals/development_dataset.json"
    )

    assert (
        dataset.split
        == "DEVELOPMENT"
    )

    assert len(
        dataset.questions
    ) == 3


def test_golden_template_has_exactly_25_questions() -> None:
    dataset = load_dataset(
        "evals/golden_dataset.json"
    )

    assert len(
        dataset.questions
    ) == 25


def test_golden_template_category_distribution() -> None:
    dataset = load_dataset(
        "evals/golden_dataset.json"
    )

    counts = Counter(
        question.category
        for question
        in dataset.questions
    )

    assert counts == {
        "STRAIGHTFORWARD_EVIDENCE": 7,
        "ONTOLOGY_CONSTRAINT": 6,
        "PATIENT_GROUP": 4,
        "GUIDELINE_VERSION": 3,
        "NO_EVIDENCE": 3,
        "DIFFICULT_AMBIGUOUS": 2,
    }


def test_question_ids_must_be_unique() -> None:
    with pytest.raises(
        ValidationError
    ):
        EvaluationDataset.model_validate(
            {
                "dataset_name": "Invalid",
                "dataset_version": "1",
                "description": "Invalid test.",
                "split": "DEVELOPMENT",
                "corpus_version": "test",
                "frozen": False,
                "questions": [
                    {
                        "question_id": "DUP-1",
                        "split": "DEVELOPMENT",
                        "category": (
                            "STRAIGHTFORWARD_EVIDENCE"
                        ),
                        "question": "Question one?",
                        "expected_behavior": "ANSWER",
                        "reference_answer": "Answer one.",
                    },
                    {
                        "question_id": "DUP-1",
                        "split": "DEVELOPMENT",
                        "category": (
                            "STRAIGHTFORWARD_EVIDENCE"
                        ),
                        "question": "Question two?",
                        "expected_behavior": "ANSWER",
                        "reference_answer": "Answer two.",
                    },
                ],
            }
        )


def test_no_evidence_question_must_abstain() -> None:
    with pytest.raises(
        ValidationError
    ):
        EvaluationDataset.model_validate(
            {
                "dataset_name": "Invalid",
                "dataset_version": "1",
                "description": "Invalid test.",
                "split": "DEVELOPMENT",
                "corpus_version": "test",
                "frozen": False,
                "questions": [
                    {
                        "question_id": "BAD-1",
                        "split": "DEVELOPMENT",
                        "category": "NO_EVIDENCE",
                        "question": "Unknown question?",
                        "expected_behavior": "ANSWER",
                        "reference_answer": "Bad answer.",
                    }
                ],
            }
        )


def test_abstention_question_cannot_have_reference_answer() -> None:
    with pytest.raises(
        ValidationError
    ):
        EvaluationDataset.model_validate(
            {
                "dataset_name": "Invalid",
                "dataset_version": "1",
                "description": "Invalid test.",
                "split": "DEVELOPMENT",
                "corpus_version": "test",
                "frozen": False,
                "questions": [
                    {
                        "question_id": "BAD-1",
                        "split": "DEVELOPMENT",
                        "category": "NO_EVIDENCE",
                        "question": "Unknown question?",
                        "expected_behavior": "ABSTAIN",
                        "reference_answer": (
                            "This should not exist."
                        ),
                    }
                ],
            }
        )


def test_golden_dataset_is_frozen() -> None:
    dataset = load_dataset(
        "evals/golden_dataset.json"
    )

    assert dataset.frozen is True

    assert (
        dataset.dataset_version
        == "1.0"
    )

    assert (
        dataset.corpus_version
        == "clinical-corpus-v1"
    )

    assert len(
        dataset.questions
    ) == 25


def test_golden_dataset_is_ready_for_final_evaluation() -> None:
    dataset = load_dataset(
        "evals/golden_dataset.json"
    )

    problems = (
        validate_final_dataset_ready(
            dataset
        )
    )

    assert problems == []


def test_dataset_question_split_must_match_dataset_split() -> None:
    with pytest.raises(
        ValidationError
    ):
        EvaluationDataset.model_validate(
            {
                "dataset_name": "Invalid",
                "dataset_version": "1",
                "description": "Invalid test.",
                "split": "FINAL",
                "corpus_version": "test",
                "frozen": False,
                "questions": [
                    {
                        "question_id": "BAD-SPLIT-1",
                        "split": "DEVELOPMENT",
                        "category": (
                            "STRAIGHTFORWARD_EVIDENCE"
                        ),
                        "question": "Question?",
                        "expected_behavior": "ANSWER",
                        "reference_answer": "Answer.",
                    }
                ],
            }
        )