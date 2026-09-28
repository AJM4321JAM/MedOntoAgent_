from rdflib import Graph

from app.models.documents import (
    DocumentChunk,
)
from app.models.ontology_indexing import (
    OntologyTaggedChunk,
)
from app.ontology.concept_mapper import (
    find_known_concepts_in_query,
)


def tag_chunk_with_ontology(
    *,
    chunk: DocumentChunk,
    graph: Graph,
) -> OntologyTaggedChunk:
    """
    Deterministically enrich one document chunk with ontology
    concepts explicitly appearing in the chunk text.

    No LLM is used.
    """
    matches = find_known_concepts_in_query(
        query=chunk.text,
        graph=graph,
    )

    concept_names = list(
        dict.fromkeys(
            match.concept_name
            for match in matches
        )
    )

    concept_labels = list(
        dict.fromkeys(
            match.label
            for match in matches
        )
    )

    return OntologyTaggedChunk(
        **chunk.model_dump(),
        ontology_concepts=concept_names,
        ontology_labels=concept_labels,
    )


def tag_chunks_with_ontology(
    *,
    chunks: list[DocumentChunk],
    graph: Graph,
) -> list[OntologyTaggedChunk]:
    """
    Deterministically tag a collection of chunks.
    """
    return [
        tag_chunk_with_ontology(
            chunk=chunk,
            graph=graph,
        )
        for chunk in chunks
    ]