from unittest.mock import (
    MagicMock,
    patch,
)

from app.config.settings import (
    Settings,
)
from app.models.ontology import (
    ExtractedConcepts,
)
from app.ontology.concept_mapper import (
    ConceptMappingError,
    build_ontology_context,
    build_ontology_lexicon,
    extract_candidate_concepts,
    find_known_concepts_in_query,
    map_candidate_concepts,
    normalize_concept_text,
)
from app.ontology.ontology_loader import (
    load_ontology,
)


def create_test_settings() -> Settings:
    return Settings(
        _env_file=None,
        openrouter_api_key="test-key",
        openrouter_model="test-model",
    )


def test_normalize_concept_text() -> None:
    assert (
        normalize_concept_text(
            "  Example-Antihypertensive_Class! "
        )
        == "example antihypertensive class"
    )


def test_lexicon_contains_canonical_label() -> None:
    graph = load_ontology()

    lexicon = build_ontology_lexicon(
        graph
    )

    assert (
        "pregnancy"
        in lexicon
    )


def test_lexicon_contains_alias() -> None:
    graph = load_ontology()

    lexicon = build_ontology_lexicon(
        graph
    )

    assert (
        "pregnant patients"
        in lexicon
    )


def test_canonical_label_maps() -> None:
    graph = load_ontology()

    mapped, unknown = (
        map_candidate_concepts(
            candidates=[
                "Pregnancy"
            ],
            graph=graph,
        )
    )

    assert unknown == []

    assert len(
        mapped
    ) == 1

    assert (
        mapped[0].concept_name
        == "Pregnancy"
    )


def test_alias_maps_to_canonical_concept() -> None:
    graph = load_ontology()

    mapped, unknown = (
        map_candidate_concepts(
            candidates=[
                "pregnant patients"
            ],
            graph=graph,
        )
    )

    assert unknown == []

    assert len(
        mapped
    ) == 1

    assert (
        mapped[0].concept_name
        == "Pregnancy"
    )

    assert (
        mapped[0].label
        == "Pregnancy"
    )


def test_unknown_concept_remains_unknown() -> None:
    graph = load_ontology()

    mapped, unknown = (
        map_candidate_concepts(
            candidates=[
                "Imaginary Medicine"
            ],
            graph=graph,
        )
    )

    assert mapped == []

    assert unknown == [
        "Imaginary Medicine"
    ]


def test_direct_query_scan_finds_known_concepts() -> None:
    graph = load_ontology()

    matches = (
        find_known_concepts_in_query(
            query=(
                "Can example antihypertensive "
                "be used in pregnant patients?"
            ),
            graph=graph,
        )
    )

    names = {
        match.concept_name
        for match in matches
    }

    assert (
        "ExampleAntihypertensiveClass"
        in names
    )

    assert (
        "Pregnancy"
        in names
    )


@patch(
    "app.ontology.concept_mapper."
    "create_openrouter_client"
)
@patch(
    "app.ontology.concept_mapper."
    "_call_with_one_retry"
)
def test_llm_candidate_extraction_is_structured(
    mock_retry: MagicMock,
    mock_create_client: MagicMock,
) -> None:
    settings = create_test_settings()

    message = MagicMock()

    message.content = (
        '{"concepts":['
        '"Example Antihypertensive Class",'
        '"pregnancy"'
        ']}'
    )

    choice = MagicMock()
    choice.message = message

    completion = MagicMock()

    completion.choices = [
        choice
    ]

    mock_retry.return_value = (
        completion
    )

    mock_create_client.return_value = (
        MagicMock()
    )

    result = extract_candidate_concepts(
        (
            "Can Example Antihypertensive Class "
            "be used during pregnancy?"
        ),
        settings=settings,
    )

    assert isinstance(
        result,
        ExtractedConcepts,
    )

    assert (
        "Example Antihypertensive Class"
        in result.concepts
    )

    assert (
        "pregnancy"
        in result.concepts
    )


@patch(
    "app.ontology.concept_mapper."
    "extract_candidate_concepts"
)
def test_build_context_merges_without_duplicates(
    mock_extract: MagicMock,
) -> None:
    graph = load_ontology()

    mock_extract.return_value = (
        ExtractedConcepts(
            concepts=[
                "Pregnancy",
                "Example Antihypertensive Class",
            ]
        )
    )

    context = build_ontology_context(
        (
            "Can Example Antihypertensive Class "
            "be used during pregnancy?"
        ),
        graph=graph,
    )

    names = [
        concept.concept_name
        for concept
        in context.mapped_concepts
    ]

    assert (
        names.count(
            "Pregnancy"
        )
        == 1
    )

    assert (
        names.count(
            "ExampleAntihypertensiveClass"
        )
        == 1
    )

    assert (
        context.mapped_count
        == 2
    )


@patch(
    "app.ontology.concept_mapper."
    "extract_candidate_concepts"
)
def test_build_context_retains_unknowns(
    mock_extract: MagicMock,
) -> None:
    graph = load_ontology()

    mock_extract.return_value = (
        ExtractedConcepts(
            concepts=[
                "Imaginary Medicine",
            ]
        )
    )

    context = build_ontology_context(
        "Is Imaginary Medicine useful?",
        graph=graph,
    )

    assert (
        context.unknown_concepts
        == [
            "Imaginary Medicine"
        ]
    )

    assert (
        context.unknown_count
        == 1
    )


@patch(
    "app.ontology.concept_mapper."
    "extract_candidate_concepts"
)
def test_build_context_falls_back_when_llm_extraction_fails(
    mock_extract: MagicMock,
) -> None:
    """
    A transient or policy refusal from the free routed model
    must not destroy ontology context when deterministic ontology
    matching can identify the concepts directly.
    """
    graph = load_ontology()

    mock_extract.side_effect = (
        ConceptMappingError(
            (
                "Concept extraction did not produce "
                "valid structured output."
            )
        )
    )

    context = build_ontology_context(
        (
            "Can Example Antihypertensive Class "
            "be used during pregnancy?"
        ),
        graph=graph,
    )

    names = {
        concept.concept_name
        for concept
        in context.mapped_concepts
    }

    assert (
        context.extracted_concepts
        == []
    )

    assert (
        "ExampleAntihypertensiveClass"
        in names
    )

    assert (
        "Pregnancy"
        in names
    )

    assert (
        context.mapped_count
        == 2
    )

    assert (
        context.unknown_concepts
        == []
    )

    assert (
        context.unknown_count
        == 0
    )


@patch(
    "app.ontology.concept_mapper."
    "extract_candidate_concepts"
)
def test_build_context_fallback_does_not_invent_unknown_concepts(
    mock_extract: MagicMock,
) -> None:
    """
    When extraction fails and the query contains no ontology term,
    the fallback must return no fabricated concept.
    """
    graph = load_ontology()

    mock_extract.side_effect = (
        ConceptMappingError(
            "Synthetic extraction failure."
        )
    )

    context = build_ontology_context(
        (
            "What does this completely unrelated "
            "phrase mean?"
        ),
        graph=graph,
    )

    assert (
        context.extracted_concepts
        == []
    )

    assert (
        context.mapped_concepts
        == []
    )

    assert (
        context.unknown_concepts
        == []
    )

    assert (
        context.mapped_count
        == 0
    )

    assert (
        context.unknown_count
        == 0
    )
def test_lexicon_contains_clinical_instances_only() -> None:
    graph = load_ontology()

    lexicon = build_ontology_lexicon(
        graph
    )

    uris = {
        str(uri)
        for uri in lexicon.values()
    }

    assert any(
        uri.endswith("/Pregnancy")
        for uri in uris
    )

    assert any(
        uri.endswith(
            "/ExampleAntihypertensiveClass"
        )
        for uri in uris
    )

    assert not any(
        uri.endswith("/ClinicalConcept")
        for uri in uris
    )

    assert not any(
        uri.endswith("/contraindicatedIn")
        for uri in uris
    )

    assert not any(
        uri.endswith("/sourceDocumentId")
        for uri in uris
    )

    assert not any(
        uri.endswith("/OntologyAssertion")
        for uri in uris
    )

    assert not any(
        uri.endswith("/assertionSubject")
        for uri in uris
    )
