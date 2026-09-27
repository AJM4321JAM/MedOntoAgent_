from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


SupportedFileType = Literal["pdf", "docx", "txt"]


class DocumentPage(BaseModel):
    """
    One extracted unit from a source document.

    For PDFs this normally represents one physical page.
    For DOCX and TXT files, page_number may be None because reliable
    page boundaries are not encoded in the same way.
    """

    page_number: int | None = Field(
        default=None,
        ge=1,
    )

    text: str = Field(
        min_length=1,
    )


class LoadedDocument(BaseModel):
    """
    Standard internal representation of an extracted source document.
    """

    filename: str
    file_path: Path
    file_type: SupportedFileType

    pages: list[DocumentPage] = Field(
        min_length=1,
    )

    full_text: str = Field(
        min_length=1,
    )

    character_count: int = Field(
        ge=1,
    )