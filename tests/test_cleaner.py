from pathlib import Path

from app.ingestion.cleaner import (
    clean_document,
    clean_text,
)
from app.models.documents import (
    DocumentPage,
    LoadedDocument,
)


def test_clean_text_normalizes_spacing() -> None:
    raw_text = (
        "Treatment   recommendation.\r\n"
        "\r\n"
        "\r\n"
        "Second paragraph.\tValue"
    )

    cleaned = clean_text(raw_text)

    assert "Treatment recommendation." in cleaned
    assert "\n\n\n" not in cleaned
    assert "\t" not in cleaned


def test_clean_document_preserves_page_number() -> None:
    document = LoadedDocument(
        filename="sample.pdf",
        file_path=Path("sample.pdf"),
        file_type="pdf",
        pages=[
            DocumentPage(
                page_number=4,
                text="  Clinical   recommendation.  ",
            )
        ],
        full_text="Clinical recommendation.",
        character_count=24,
    )

    cleaned = clean_document(document)

    assert len(cleaned.pages) == 1
    assert cleaned.pages[0].page_number == 4
    assert (
        cleaned.pages[0].text
        == "Clinical recommendation."
    )