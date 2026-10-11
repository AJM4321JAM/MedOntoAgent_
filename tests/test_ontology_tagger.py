from app.models.documents import (
    ApprovalStatus,
    DocumentChunk,
    SourceAuthority,
)
from app.models.ontology_indexing import (
    OntologyTaggedChunk,
)
from app.ontology.ontology_loader import (
    load_ontology,
)
from app.ontology.tagger import (
    tag_chunk_with_ontology,
    tag_chunks_with_ontology,
)


def create_chunk(
    text: str,
    *,
    chunk_index: int = 0,
    section_title: str | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=(
            f"DOC-test-CHUNK-{chunk_index:04d}"
        ),
        document_id="DOC-test",
        chunk_index=chunk_index,
        title="Synthetic Test",
        section_title=section_title,
        page_number=None,
        document_type=(
            "Synthetic Test Document"
        ),
        clinical_domain="Testing",
        publisher="MedOntoAgent",
        source_authority=(
            SourceAuthority.OTHER
        ),
        version="1.0",
        approval_status=(
            ApprovalStatus.APPROVED
        ),
        publication_date=None,
        effective_date=None,
        source_url=None,
        text=text,
        word_count=len(
            text.split()
        ),
    )


def test_tag_chunk_returns_tagged_chunk() -> None:
    graph = load_ontology()

    chunk = create_chunk(
        (
            "Example Antihypertensive Class "
            "is discussed in Pregnancy."
        )
    )

    tagged = (
        tag_chunk_with_ontology(
            chunk=chunk,
            graph=graph,
        )
    )

    assert isinstance(
        tagged,
        OntologyTaggedChunk,
    )


def test_tagger_detects_both_known_concepts() -> None:
    graph = load_ontology()

    chunk = create_chunk(
        (
            "Example Antihypertensive Class "
            "is contraindicated in Pregnancy."
        )
    )

    tagged = (
        tag_chunk_with_ontology(
            chunk=chunk,
            graph=graph,
        )
    )

    assert (
        "ExampleAntihypertensiveClass"
        in tagged.ontology_concepts
    )

    assert (
        "Pregnancy"
        in tagged.ontology_concepts
    )


def test_general_chunk_does_not_gain_pregnancy_tag() -> None:
    graph = load_ontology()

    chunk = create_chunk(
        (
            "Example Antihypertensive Class "
            "is discussed as a general treatment concept."
        )
    )

    tagged = (
        tag_chunk_with_ontology(
            chunk=chunk,
            graph=graph,
        )
    )

    assert (
        tagged.ontology_concepts
        == [
            "ExampleAntihypertensiveClass"
        ]
    )


def test_alias_is_used_for_chunk_tagging() -> None:
    graph = load_ontology()

    chunk = create_chunk(
        (
            "This synthetic statement discusses "
            "pregnant patients."
        )
    )

    tagged = (
        tag_chunk_with_ontology(
            chunk=chunk,
            graph=graph,
        )
    )

    assert (
        tagged.ontology_concepts
        == [
            "Pregnancy"
        ]
    )

    assert (
        tagged.ontology_labels
        == [
            "Pregnancy"
        ]
    )


def test_unknown_text_gets_no_ontology_tags() -> None:
    graph = load_ontology()

    chunk = create_chunk(
        "Imaginary Medicine Z is discussed here."
    )

    tagged = (
        tag_chunk_with_ontology(
            chunk=chunk,
            graph=graph,
        )
    )

    assert (
        tagged.ontology_concepts
        == []
    )

    assert (
        tagged.ontology_labels
        == []
    )


def test_tagging_does_not_change_chunk_text() -> None:
    graph = load_ontology()

    text = (
        "Example Antihypertensive Class "
        "is discussed during Pregnancy."
    )

    chunk = create_chunk(
        text
    )

    tagged = (
        tag_chunk_with_ontology(
            chunk=chunk,
            graph=graph,
        )
    )

    assert (
        tagged.text
        == text
    )

    assert (
        tagged.chunk_id
        == chunk.chunk_id
    )


def test_multiple_chunks_are_tagged() -> None:
    graph = load_ontology()

    chunks = [
        create_chunk(
            (
                "Example Antihypertensive Class "
                "is discussed generally."
            ),
            chunk_index=0,
        ),
        create_chunk(
            (
                "Pregnancy is discussed in this "
                "synthetic test."
            ),
            chunk_index=1,
        ),
    ]

    tagged = (
        tag_chunks_with_ontology(
            chunks=chunks,
            graph=graph,
        )
    )

    assert len(
        tagged
    ) == 2

    assert (
        tagged[0].ontology_concepts
        == [
            "ExampleAntihypertensiveClass"
        ]
    )

    assert (
        tagged[1].ontology_concepts
        == [
            "Pregnancy"
        ]
    )

def test_section_title_contributes_ontology_tags() -> None:
    graph = load_ontology()

    chunk = create_chunk(
        "become pregnant and offer alternatives.",
        section_title=(
            "Stop antihypertensive treatment in women "
            "taking ACE inhibitors or ARBs"
        ),
    )

    tagged = tag_chunk_with_ontology(
        chunk=chunk,
        graph=graph,
    )

    assert (
        "ACEInhibitor"
        in tagged.ontology_concepts
    )

    assert (
        "AngiotensinReceptorBlocker"
        in tagged.ontology_concepts
    )

    assert (
        "Pregnancy"
        in tagged.ontology_concepts
    )
