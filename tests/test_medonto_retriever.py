from unittest.mock import (
    MagicMock,
    patch,
)

from app.config.settings import (
    Settings,
)
from app.models.constraints import (
    OntologyConstraint,
    OntologyConstraintSet,
    RetrievalHints,
)
from app.models.ontology import (
    MappedOntologyConcept,
    OntologyContext,
)
from app.retrieval.medonto_retriever import (
    _constraint_concept_names,
    retrieve_medonto,
)


def create_settings() -> Settings:
    return Settings(
        _env_file=None,
        openrouter_api_key="test-key",
        openrouter_model="test-model",
        qdrant_url=(
            "https://example.qdrant.io"
        ),
        qdrant_api_key="test-key",
        qdrant_collection="test_collection",
        embedding_model="test-model",
        embedding_dimensions=4,
        embedding_batch_size=2,
        embedding_device="cpu",
        retrieval_top_k=5,
    )


def create_context() -> OntologyContext:
    return OntologyContext(
        query="test question",
        extracted_concepts=[
            "Example Antihypertensive Class",
            "pregnancy",
        ],
        mapped_concepts=[
            MappedOntologyConcept(
                original_text=(
                    "Example Antihypertensive Class"
                ),
                concept_name=(
                    "ExampleAntihypertensiveClass"
                ),
                label=(
                    "Example Antihypertensive Class"
                ),
                uri=(
                    "https://medontoagent.example.org/"
                    "ontology/"
                    "ExampleAntihypertensiveClass"
                ),
            ),
            MappedOntologyConcept(
                original_text="pregnancy",
                concept_name="Pregnancy",
                label="Pregnancy",
                uri=(
                    "https://medontoagent.example.org/"
                    "ontology/Pregnancy"
                ),
            ),
        ],
        unknown_concepts=[],
        mapped_count=2,
        unknown_count=0,
    )


def create_constraint_set() -> OntologyConstraintSet:
    return OntologyConstraintSet(
        query="test question",
        constraints=[
            OntologyConstraint(
                constraint_type=(
                    "CONTRAINDICATION"
                ),
                subject_name=(
                    "ExampleAntihypertensiveClass"
                ),
                subject_label=(
                    "Example Antihypertensive Class"
                ),
                subject_uri=(
                    "https://medontoagent.example.org/"
                    "ontology/"
                    "ExampleAntihypertensiveClass"
                ),
                predicate_name=(
                    "contraindicatedIn"
                ),
                predicate_label=(
                    "contraindicated in"
                ),
                predicate_uri=(
                    "https://medontoagent.example.org/"
                    "ontology/contraindicatedIn"
                ),
                object_name="Pregnancy",
                object_label="Pregnancy",
                object_uri=(
                    "https://medontoagent.example.org/"
                    "ontology/Pregnancy"
                ),
                source_document_id=(
                    "DOC-ACTIVE-TEST"
                ),
                source_section=(
                    "Implementation Test"
                ),
                source_version="1.0",
                assertion_status=(
                    "ACTIVE"
                ),
                control_eligible=True,
            )
        ],
        retrieval_hints=RetrievalHints(
            concept_names=[
                "ExampleAntihypertensiveClass",
                "Pregnancy",
            ],
            concept_labels=[
                "Example Antihypertensive Class",
                "Pregnancy",
            ],
            relationship_terms=[
                (
                    "Example Antihypertensive Class "
                    "contraindicated in Pregnancy"
                )
            ],
            expanded_query=(
                "test question\n\n"
                "Ontology relationship context:\n"
                "Example Antihypertensive Class "
                "contraindicated in Pregnancy"
            ),
        ),
        constraint_count=1,
    )


def create_point(
    *,
    title: str = (
        "Synthetic Pregnancy Constraint Document"
    ),
) -> MagicMock:
    point = MagicMock()

    point.score = 0.75

    point.payload = {
        "chunk_id": (
            "DOC-test-CHUNK-0000"
        ),
        "document_id": "DOC-test",
        "chunk_index": 0,
        "title": title,
        "section_title": None,
        "page_number": None,
        "document_type": (
            "Synthetic Test Document"
        ),
        "clinical_domain": (
            "Synthetic Evaluation"
        ),
        "publisher": "MedOntoAgent",
        "source_authority": "OTHER",
        "version": "1.0",
        "approval_status": "APPROVED",
        "publication_date": None,
        "effective_date": None,
        "source_url": None,
        "text": (
            "Synthetic ontology-aware evidence."
        ),
        "word_count": 4,
    }

    return point


@patch(
    "app.retrieval.medonto_retriever."
    "_query_qdrant"
)
@patch(
    "app.retrieval.medonto_retriever."
    "create_qdrant_client"
)
@patch(
    "app.retrieval.medonto_retriever."
    "embed_text"
)
@patch(
    "app.retrieval.medonto_retriever."
    "build_constraints"
)
@patch(
    "app.retrieval.medonto_retriever."
    "build_ontology_context"
)
def test_constraint_uses_expanded_query_and_strict_filter(
    mock_context: MagicMock,
    mock_constraints: MagicMock,
    mock_embed: MagicMock,
    mock_client: MagicMock,
    mock_query: MagicMock,
) -> None:
    settings = create_settings()

    context = create_context()

    constraint_set = (
        create_constraint_set()
    )

    mock_context.return_value = (
        context
    )

    mock_constraints.return_value = (
        constraint_set
    )

    mock_embed.return_value = [
        0.1,
        0.2,
        0.3,
        0.4,
    ]

    mock_client.return_value = (
        MagicMock()
    )

    mock_query.return_value = [
        create_point()
    ]

    result = retrieve_medonto(
        "test question",
        settings=settings,
    )

    mock_embed.assert_called_once()

    call = (
        mock_embed.call_args
    )

    assert (
        call.args[0]
        == constraint_set
        .retrieval_hints
        .expanded_query
    )

    assert (
        result.used_ontology_filter
        is True
    )

    assert (
        result.retrieval_relaxed
        is False
    )

    assert (
        result.constraint_set
        .constraint_count
        == 1
    )

    assert (
        len(
            result.retrieval
            .retrieved_chunks
        )
        == 1
    )


@patch(
    "app.retrieval.medonto_retriever."
    "_query_qdrant"
)
@patch(
    "app.retrieval.medonto_retriever."
    "create_qdrant_client"
)
@patch(
    "app.retrieval.medonto_retriever."
    "embed_text"
)
@patch(
    "app.retrieval.medonto_retriever."
    "build_constraints"
)
@patch(
    "app.retrieval.medonto_retriever."
    "build_ontology_context"
)
def test_zero_strict_results_relaxes_once(
    mock_context: MagicMock,
    mock_constraints: MagicMock,
    mock_embed: MagicMock,
    mock_client: MagicMock,
    mock_query: MagicMock,
) -> None:
    settings = create_settings()

    mock_context.return_value = (
        create_context()
    )

    mock_constraints.return_value = (
        create_constraint_set()
    )

    mock_embed.return_value = [
        0.1,
        0.2,
        0.3,
        0.4,
    ]

    mock_client.return_value = (
        MagicMock()
    )

    mock_query.side_effect = [
        [],
        [
            create_point()
        ],
    ]

    result = retrieve_medonto(
        "test question",
        settings=settings,
    )

    assert (
        mock_query.call_count
        == 2
    )

    assert (
        result.used_ontology_filter
        is True
    )

    assert (
        result.retrieval_relaxed
        is True
    )

    assert (
        len(
            result.retrieval
            .retrieved_chunks
        )
        == 1
    )


@patch(
    "app.retrieval.medonto_retriever."
    "_query_qdrant"
)
@patch(
    "app.retrieval.medonto_retriever."
    "create_qdrant_client"
)
@patch(
    "app.retrieval.medonto_retriever."
    "embed_text"
)
@patch(
    "app.retrieval.medonto_retriever."
    "build_constraints"
)
@patch(
    "app.retrieval.medonto_retriever."
    "build_ontology_context"
)
def test_no_constraint_uses_approved_search_only(
    mock_context: MagicMock,
    mock_constraints: MagicMock,
    mock_embed: MagicMock,
    mock_client: MagicMock,
    mock_query: MagicMock,
) -> None:
    settings = create_settings()

    context = create_context()

    constraint_set = (
        OntologyConstraintSet(
            query="test question",
            constraints=[],
            retrieval_hints=(
                RetrievalHints(
                    concept_names=[
                        "Pregnancy",
                    ],
                    concept_labels=[
                        "Pregnancy",
                    ],
                    relationship_terms=[],
                    expanded_query=(
                        "test question"
                    ),
                )
            ),
            constraint_count=0,
        )
    )

    mock_context.return_value = (
        context
    )

    mock_constraints.return_value = (
        constraint_set
    )

    mock_embed.return_value = [
        0.1,
        0.2,
        0.3,
        0.4,
    ]

    mock_client.return_value = (
        MagicMock()
    )

    mock_query.return_value = [
        create_point()
    ]

    result = retrieve_medonto(
        "test question",
        settings=settings,
    )

    assert (
        mock_query.call_count
        == 1
    )

    assert (
        result.used_ontology_filter
        is False
    )

    assert (
        result.retrieval_relaxed
        is False
    )

    assert (
        result.constraint_set
        .constraint_count
        == 0
    )


@patch(
    "app.retrieval.medonto_retriever."
    "_query_qdrant"
)
@patch(
    "app.retrieval.medonto_retriever."
    "create_qdrant_client"
)
@patch(
    "app.retrieval.medonto_retriever."
    "embed_text"
)
@patch(
    "app.retrieval.medonto_retriever."
    "build_constraints"
)
@patch(
    "app.retrieval.medonto_retriever."
    "build_ontology_context"
)
def test_original_query_is_preserved_in_result(
    mock_context: MagicMock,
    mock_constraints: MagicMock,
    mock_embed: MagicMock,
    mock_client: MagicMock,
    mock_query: MagicMock,
) -> None:
    settings = create_settings()

    mock_context.return_value = (
        create_context()
    )

    mock_constraints.return_value = (
        create_constraint_set()
    )

    mock_embed.return_value = [
        0.1,
        0.2,
        0.3,
        0.4,
    ]

    mock_client.return_value = (
        MagicMock()
    )

    mock_query.return_value = [
        create_point()
    ]

    result = retrieve_medonto(
        "test question",
        settings=settings,
    )

    assert (
        result.query
        == "test question"
    )

    assert (
        result.retrieval.query
        != ""
    )

def test_ineligible_constraint_does_not_create_filter() -> None:
    constraint = OntologyConstraint(
        constraint_type="CONTRAINDICATION",
        subject_name=(
            "ExampleAntihypertensiveClass"
        ),
        subject_label=(
            "Example Antihypertensive Class"
        ),
        subject_uri=(
            "https://medontoagent.example.org/"
            "ontology/"
            "ExampleAntihypertensiveClass"
        ),
        predicate_name="contraindicatedIn",
        predicate_label="contraindicated in",
        predicate_uri=(
            "https://medontoagent.example.org/"
            "ontology/contraindicatedIn"
        ),
        object_name="Pregnancy",
        object_label="Pregnancy",
        object_uri=(
            "https://medontoagent.example.org/"
            "ontology/Pregnancy"
        ),
        source_document_id=(
            "SYNTHETIC-ONTOLOGY-TEST"
        ),
        assertion_status=(
            "SYNTHETIC_TEST_ONLY"
        ),
        control_eligible=False,
    )

    constraint_set = OntologyConstraintSet(
        query="test",
        constraints=[
            constraint
        ],
        retrieval_hints=RetrievalHints(
            concept_names=[],
            concept_labels=[],
            relationship_terms=[],
            expanded_query="test",
        ),
        constraint_count=1,
    )

    assert (
        _constraint_concept_names(
            constraint_set
        )
        == []
    )


@patch(
    "app.retrieval.medonto_retriever."
    "_query_qdrant"
)
@patch(
    "app.retrieval.medonto_retriever."
    "create_qdrant_client"
)
@patch(
    "app.retrieval.medonto_retriever."
    "embed_text"
)
@patch(
    "app.retrieval.medonto_retriever."
    "build_constraints"
)
@patch(
    "app.retrieval.medonto_retriever."
    "build_ontology_context"
)
def test_strict_retrieval_scopes_to_constraint_source(
    mock_context: MagicMock,
    mock_constraints: MagicMock,
    mock_embed: MagicMock,
    mock_client: MagicMock,
    mock_query: MagicMock,
) -> None:
    context = create_context()
    constraint_set = create_constraint_set()

    mock_context.return_value = context
    mock_constraints.return_value = constraint_set
    mock_embed.return_value = [0.1, 0.2, 0.3, 0.4]
    mock_client.return_value = MagicMock()
    mock_query.return_value = [
        create_point()
    ]

    retrieve_medonto(
        "test question",
        settings=create_settings(),
    )

    query_filter = (
        mock_query.call_args.kwargs[
            "query_filter"
        ]
    )

    conditions = (
        query_filter.must
        or []
    )

    document_conditions = [
        condition
        for condition in conditions
        if getattr(
            condition,
            "key",
            None,
        )
        == "document_id"
    ]

    assert len(document_conditions) == 1

    assert (
        document_conditions[0]
        .match
        .value
        == "DOC-ACTIVE-TEST"
    )


@patch(
    "app.retrieval.medonto_retriever."
    "_query_qdrant"
)
@patch(
    "app.retrieval.medonto_retriever."
    "create_qdrant_client"
)
@patch(
    "app.retrieval.medonto_retriever."
    "embed_text"
)
@patch(
    "app.retrieval.medonto_retriever."
    "build_constraints"
)
@patch(
    "app.retrieval.medonto_retriever."
    "build_ontology_context"
)
def test_relaxed_retrieval_keeps_constraint_source_scope(
    mock_context: MagicMock,
    mock_constraints: MagicMock,
    mock_embed: MagicMock,
    mock_client: MagicMock,
    mock_query: MagicMock,
) -> None:
    context = create_context()
    constraint_set = create_constraint_set()

    mock_context.return_value = context
    mock_constraints.return_value = constraint_set
    mock_embed.return_value = [0.1, 0.2, 0.3, 0.4]
    mock_client.return_value = MagicMock()

    mock_query.side_effect = [
        [],
        [create_point()],
    ]

    result = retrieve_medonto(
        "test question",
        settings=create_settings(),
    )

    assert (
        result.retrieval_relaxed
        is True
    )

    assert (
        mock_query.call_count
        == 2
    )

    relaxed_filter = (
        mock_query.call_args_list[1]
        .kwargs[
            "query_filter"
        ]
    )

    conditions = (
        relaxed_filter.must
        or []
    )

    document_conditions = [
        condition
        for condition in conditions
        if getattr(
            condition,
            "key",
            None,
        )
        == "document_id"
    ]

    assert len(document_conditions) == 1

    assert (
        document_conditions[0]
        .match
        .value
        == "DOC-ACTIVE-TEST"
    )
