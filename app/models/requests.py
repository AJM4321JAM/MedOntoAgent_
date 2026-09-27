from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Request body for a MedOntoAgent knowledge query.
    """

    query: str = Field(
        min_length=3,
        max_length=2000,
        description=(
            "Clinical knowledge question to retrieve "
            "evidence for."
        ),
    )