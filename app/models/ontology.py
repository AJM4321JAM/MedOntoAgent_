from pydantic import BaseModel, Field


class ExtractedConcepts(BaseModel):
    """
    Candidate clinical phrases identified from natural language.

    These values are not trusted ontology mappings.
    """

    concepts: list[str] = Field(
        default_factory=list,
    )


class MappedOntologyConcept(BaseModel):
    """
    One candidate phrase resolved deterministically to a
    controlled ontology resource.
    """

    original_text: str = Field(
        min_length=1,
    )

    concept_name: str = Field(
        min_length=1,
    )

    label: str = Field(
        min_length=1,
    )

    uri: str = Field(
        min_length=1,
    )


class OntologyContext(BaseModel):
    """
    Structured bridge between a natural-language query and
    the MedOntoAgent ontology.
    """

    query: str = Field(
        min_length=1,
    )

    extracted_concepts: list[str] = Field(
        default_factory=list,
    )

    mapped_concepts: list[
        MappedOntologyConcept
    ] = Field(
        default_factory=list,
    )

    unknown_concepts: list[str] = Field(
        default_factory=list,
    )

    mapped_count: int = Field(
        ge=0,
    )

    unknown_count: int = Field(
        ge=0,
    )