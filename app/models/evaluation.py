from datetime import datetime
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


EvaluationSystem = Literal[
    "BASELINE",
    "MEDONTO",
]


EvaluationSystemStatus = Literal[
    "ANSWERED",
    "VERIFIED",
    "ABSTAINED",
    "ERROR",
]


ExecutionMode = Literal[
    "BASELINE",
    "MEDONTO",
    "BASELINE_ROUTED",
    "BASELINE_FALLBACK",
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

    @model_validator(mode="after")
    def validate_expected_behavior(
        self,
    ) -> "GoldenQuestion":
        if (
            self.category == "NO_EVIDENCE"
            and self.expected_behavior != "ABSTAIN"
        ):
            raise ValueError(
                "NO_EVIDENCE questions must expect ABSTAIN."
            )

        if (
            self.expected_behavior == "ABSTAIN"
            and self.reference_answer is not None
        ):
            raise ValueError(
                "ABSTAIN questions cannot have a reference answer."
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

    questions: list[GoldenQuestion] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_dataset(
        self,
    ) -> "EvaluationDataset":
        question_ids = [
            question.question_id
            for question in self.questions
        ]

        if len(question_ids) != len(
            set(question_ids)
        ):
            raise ValueError(
                "Evaluation question IDs must be unique."
            )

        mismatched = [
            question.question_id
            for question in self.questions
            if question.split != self.split
        ]

        if mismatched:
            raise ValueError(
                "Every question split must match the dataset split."
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

    if dataset.corpus_version == "TO_BE_FROZEN":
        problems.append(
            "Corpus version has not been frozen."
        )

    for question in dataset.questions:
        if "TO_BE_COMPLETED" in question.question:
            problems.append(
                (
                    f"{question.question_id} "
                    "still contains placeholder text."
                )
            )

        if (
            question.expected_behavior == "ANSWER"
            and question.reference_answer is None
        ):
            problems.append(
                (
                    f"{question.question_id} "
                    "has no reference answer."
                )
            )

        if (
            question.expected_behavior == "ANSWER"
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


class EvaluationResultRecord(BaseModel):
    """
    One standardized result produced by either evaluation system.

    retrieved_contexts stores the exact ranked evidence text
    presented to the system during that run. This makes later
    offline semantic evaluation independent of future changes
    to the vector database.
    """

    question_id: str = Field(
        min_length=1,
    )

    category: EvaluationCategory

    question: str = Field(
        min_length=3,
    )

    expected_behavior: ExpectedBehavior

    system: EvaluationSystem

    execution_mode: ExecutionMode | None = None

    system_status: EvaluationSystemStatus

    behavior_match: bool

    answer: str | None = None

    citation_chunk_ids: list[str] = Field(
        default_factory=list,
    )

    citation_document_ids: list[str] = Field(
        default_factory=list,
    )

    retrieved_chunk_ids: list[str] = Field(
        default_factory=list,
    )

    retrieved_document_ids: list[str] = Field(
        default_factory=list,
    )

    retrieved_contexts: list[str] = Field(
        default_factory=list,
    )

    ontology_constraints: list[str] = Field(
        default_factory=list,
    )

    used_ontology_filter: bool | None = None

    retrieval_relaxed: bool | None = None

    retry_count: int | None = Field(
        default=None,
        ge=0,
    )

    latency_ms: float = Field(
        ge=0,
    )

    error_message: str | None = None

    fallback_reason: str | None = None

    fallback_error: str | None = None

    @model_validator(mode="after")
    def infer_execution_mode(
        self,
    ) -> "EvaluationResultRecord":
        """
        Preserve compatibility with older artifacts while making
        the actual execution path explicit for new evaluations.
        """
        if self.execution_mode is None:
            self.execution_mode = self.system

        return self


class EvaluationRunArtifact(BaseModel):
    """
    Versioned result artifact for one complete dataset execution.
    """

    run_id: str = Field(
        min_length=1,
    )

    created_at: datetime

    system: EvaluationSystem

    dataset_name: str = Field(
        min_length=1,
    )

    dataset_version: str = Field(
        min_length=1,
    )

    corpus_version: str = Field(
        min_length=1,
    )

    configuration: dict[
        str,
        str | int | float | bool | None,
    ]

    results: list[EvaluationResultRecord] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_result_systems(
        self,
    ) -> "EvaluationRunArtifact":
        if any(
            result.system != self.system
            for result in self.results
        ):
            raise ValueError(
                "Every result record must use the artifact system."
            )

        return self


class LatencySummary(BaseModel):
    """
    Latency statistics for one evaluated system.
    """

    count: int = Field(
        ge=0,
    )

    mean_ms: float | None = Field(
        default=None,
        ge=0,
    )

    median_ms: float | None = Field(
        default=None,
        ge=0,
    )

    p95_ms: float | None = Field(
        default=None,
        ge=0,
    )


class DeterministicMetricSummary(BaseModel):
    """
    Deterministic metrics calculated from one evaluation run.
    """

    total_questions: int = Field(
        ge=0,
    )

    behavior_match_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    error_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    abstention_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    expected_document_retrieval_hit_rate: (
        float | None
    ) = Field(
        default=None,
        ge=0,
        le=1,
    )

    expected_chunk_retrieval_hit_rate: (
        float | None
    ) = Field(
        default=None,
        ge=0,
        le=1,
    )

    citation_document_precision: (
        float | None
    ) = Field(
        default=None,
        ge=0,
        le=1,
    )

    citation_document_recall: (
        float | None
    ) = Field(
        default=None,
        ge=0,
        le=1,
    )

    citation_chunk_precision: (
        float | None
    ) = Field(
        default=None,
        ge=0,
        le=1,
    )

    citation_chunk_recall: (
        float | None
    ) = Field(
        default=None,
        ge=0,
        le=1,
    )

    ontology_relationship_recall: (
        float | None
    ) = Field(
        default=None,
        ge=0,
        le=1,
    )

    ontology_exact_match_rate: (
        float | None
    ) = Field(
        default=None,
        ge=0,
        le=1,
    )

    retry_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    retrieval_relaxation_rate: (
        float | None
    ) = Field(
        default=None,
        ge=0,
        le=1,
    )

    latency: LatencySummary


class QuestionComparison(BaseModel):
    """
    Side-by-side deterministic result for one question.
    """

    question_id: str = Field(
        min_length=1,
    )

    category: EvaluationCategory

    expected_behavior: ExpectedBehavior

    baseline_status: EvaluationSystemStatus

    medonto_status: EvaluationSystemStatus

    baseline_behavior_match: bool

    medonto_behavior_match: bool

    baseline_retrieved_expected_document: (
        bool | None
    ) = None

    medonto_retrieved_expected_document: (
        bool | None
    ) = None

    baseline_retrieved_expected_chunk: (
        bool | None
    ) = None

    medonto_retrieved_expected_chunk: (
        bool | None
    ) = None

    baseline_expected_ontology_relationships_found: (
        bool | None
    ) = None

    medonto_expected_ontology_relationships_found: (
        bool | None
    ) = None

    baseline_latency_ms: float = Field(
        ge=0,
    )

    medonto_latency_ms: float = Field(
        ge=0,
    )


class EvaluationComparisonArtifact(BaseModel):
    """
    Deterministic comparison of one baseline and one MedOnto run.
    """

    created_at: datetime

    dataset_name: str = Field(
        min_length=1,
    )

    dataset_version: str = Field(
        min_length=1,
    )

    corpus_version: str = Field(
        min_length=1,
    )

    baseline_run_id: str = Field(
        min_length=1,
    )

    medonto_run_id: str = Field(
        min_length=1,
    )

    baseline_metrics: DeterministicMetricSummary

    medonto_metrics: DeterministicMetricSummary

    questions: list[QuestionComparison] = Field(
        min_length=1,
    )


class RagasQuestionScore(BaseModel):
    """
    Semantic Ragas scores for one evaluated question.

    A None value means that the metric was not applicable
    or could not be scored.
    """

    question_id: str = Field(
        min_length=1,
    )

    system: EvaluationSystem

    faithfulness: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    context_precision: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    answer_relevancy: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    metric_errors: list[str] = Field(
        default_factory=list,
    )


class RagasMetricSummary(BaseModel):
    """
    Aggregate semantic metrics for one system-run artifact.
    """

    total_records: int = Field(
        ge=0,
    )

    scored_records: int = Field(
        ge=0,
    )

    faithfulness_mean: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    context_precision_mean: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    answer_relevancy_mean: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )


class RagasEvaluationArtifact(BaseModel):
    """
    Offline semantic-evaluation artifact produced from one
    immutable system execution artifact.

    Ragas scores are deliberately kept separate from the original
    system execution artifact.
    """

    created_at: datetime

    source_run_id: str = Field(
        min_length=1,
    )

    system: EvaluationSystem

    dataset_name: str = Field(
        min_length=1,
    )

    dataset_version: str = Field(
        min_length=1,
    )

    corpus_version: str = Field(
        min_length=1,
    )

    evaluator_model: str = Field(
        min_length=1,
    )

    embedding_model: str = Field(
        min_length=1,
    )

    scores: list[RagasQuestionScore] = Field(
        min_length=1,
    )

    summary: RagasMetricSummary


class FrozenCorpusDocument(BaseModel):
    """
    One immutable source-document entry in a frozen
    evaluation corpus.
    """

    document_id: str = Field(
        min_length=1,
    )

    title: str = Field(
        min_length=1,
    )

    publisher: str = Field(
        min_length=1,
    )

    version: str = Field(
        min_length=1,
    )

    checksum_sha256: str = Field(
        min_length=64,
        max_length=64,
    )

    approval_status: str = Field(
        min_length=1,
    )

    superseded_by_document_id: str | None = None

    chunk_count: int = Field(
        ge=1,
    )


class CorpusFreezeManifest(BaseModel):
    """
    Reproducibility manifest for one frozen clinical corpus.
    """

    corpus_version: str = Field(
        min_length=1,
    )

    created_at: datetime

    qdrant_collection: str = Field(
        min_length=1,
    )

    qdrant_point_count: int = Field(
        ge=1,
    )

    embedding_model: str = Field(
        min_length=1,
    )

    embedding_dimensions: int = Field(
        ge=1,
    )

    max_chunk_words: int = Field(
        ge=1,
    )

    chunk_overlap_words: int = Field(
        ge=0,
    )

    retrieval_top_k: int = Field(
        ge=1,
    )

    openrouter_model: str = Field(
        min_length=1,
    )

    max_agent_retries: int = Field(
        ge=0,
    )

    ontology_path: str = Field(
        min_length=1,
    )

    ontology_checksum_sha256: str = Field(
        min_length=64,
        max_length=64,
    )

    ontology_assertion_count: int = Field(
        ge=0,
    )

    ontology_active_assertion_count: int = Field(
        ge=0,
    )

    documents: list[
        FrozenCorpusDocument
    ] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_documents(
        self,
    ) -> "CorpusFreezeManifest":
        document_ids = [
            document.document_id
            for document in self.documents
        ]

        if len(document_ids) != len(
            set(document_ids)
        ):
            raise ValueError(
                "Frozen corpus document IDs must be unique."
            )

        return self
