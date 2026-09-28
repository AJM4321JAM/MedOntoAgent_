from pydantic import Field

from app.models.documents import DocumentChunk


class OntologyTaggedChunk(DocumentChunk):
    """
    A normal DocumentChunk enriched with deterministic
    ontology metadata.

    The original chunk ID, text, document metadata, and
    embedding input remain unchanged.
    """

    ontology_concepts: list[str] = Field(
        default_factory=list,
    )

    ontology_labels: list[str] = Field(
        default_factory=list,
    )