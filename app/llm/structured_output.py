from typing import Literal

from pydantic import BaseModel, Field


class LLMTestResponse(BaseModel):
    """
    Temporary structured response used to verify the LLM integration.
    """

    answer: str = Field(
        min_length=1,
        description="A concise answer to the test question.",
    )

    status: Literal["ok"] = Field(
        description="Indicates that the model produced a valid test response.",
    )