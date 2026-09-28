from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    model_validator,
)


EvaluationCategory = Literal[
    "STRAIGHTFORWARD_EVIDENCE",
    "ONTOLOGY_CONSTRAINT",
    "PATIENT_GROUP",
    "GUIDELINE_VERSION",
    "NO_EVIDENCE",
    "DIFFICULT_AMBIGUOUS",
]


ExpectedBehavior = Literal[
    "ANSWER",
    "ABSTAIN",
]


DatasetSplit = Literal[
    "DEVELOPMENT",
    "FINAL",
]


class GoldenQuestion(BaseModel):
    """
    One versioned evaluation question.
    """

    question_id: str = Field(
        min_length=1,
    )

    split: DatasetSplit

    category: EvaluationCategory

    question: str = Field(
        min_length=1,
    )

    expected_behavior: ExpectedBehavior

    reference_answer: str | None = None

    expected_document_ids: list[str] = Field(
        default_factory=list,
    )

    expected_chunk_ids: list[str] = Field(
        default_factory=list,
    )

    expected_ontology_concepts: list[str] = Field(
        default_factory=list,
    )

    expected_ontology_relationships: list[str] = Field(
        default_factory=list,
    )

    notes: str | None = None

    @model_validator(
        mode="after"
    )
    def validate_expected_behavior(
        self,
    ) -> "GoldenQuestion":
        if (
            self.category
            == "NO_EVIDENCE"
            and self.expected_behavior
            != "ABSTAIN"
        ):
            raise ValueError(
                (
                    "NO_EVIDENCE questions must "
                    "expect ABSTAIN."
                )
            )

        if (
            self.expected_behavior
            == "ABSTAIN"
            and self.reference_answer
            is not None
        ):
            raise ValueError(
                (
                    "ABSTAIN questions cannot "
                    "have a reference answer."
                )
            )

        return self


class EvaluationDataset(BaseModel):
    """
    Versioned collection of evaluation questions.
    """

    dataset_name: str = Field(
        min_length=1,
    )

    dataset_version: str = Field(
        min_length=1,
    )

    description: str = Field(
        min_length=1,
    )

    split: DatasetSplit

    corpus_version: str = Field(
        min_length=1,
    )

    frozen: bool

    questions: list[
        GoldenQuestion
    ] = Field(
        min_length=1,
    )

    @model_validator(
        mode="after"
    )
    def validate_dataset(
        self,
    ) -> "EvaluationDataset":
        question_ids = [
            question.question_id
            for question
            in self.questions
        ]

        if (
            len(question_ids)
            != len(set(question_ids))
        ):
            raise ValueError(
                (
                    "Evaluation question IDs "
                    "must be unique."
                )
            )

        mismatched = [
            question.question_id
            for question
            in self.questions
            if (
                question.split
                != self.split
            )
        ]

        if mismatched:
            raise ValueError(
                (
                    "Every question split must "
                    "match the dataset split."
                )
            )

        return self


def validate_final_dataset_ready(
    dataset: EvaluationDataset,
) -> list[str]:
    """
    Return readiness problems that prevent formal final evaluation.
    """
    problems: list[str] = []

    if dataset.split != "FINAL":
        problems.append(
            "Dataset is not the FINAL split."
        )

    if not dataset.frozen:
        problems.append(
            "Final dataset is not frozen."
        )

    if (
        dataset.corpus_version
        == "TO_BE_FROZEN"
    ):
        problems.append(
            (
                "Corpus version has not "
                "been frozen."
            )
        )

    for question in dataset.questions:
        if (
            "TO_BE_COMPLETED"
            in question.question
        ):
            problems.append(
                (
                    f"{question.question_id} "
                    "still contains placeholder text."
                )
            )

        if (
            question.expected_behavior
            == "ANSWER"
            and question.reference_answer
            is None
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
            and not (
                question.expected_document_ids
                or question.expected_chunk_ids
            )
        ):
            problems.append(
                (
                    f"{question.question_id} "
                    "has no expected evidence identifiers."
                )
            )

    return problems