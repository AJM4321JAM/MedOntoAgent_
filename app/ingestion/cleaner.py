import re

from app.models.documents import (
    DocumentPage,
    LoadedDocument,
)


def clean_document(
    document: LoadedDocument,
) -> LoadedDocument:
    """
    Conservatively clean extracted document text without
    intentionally changing its clinical meaning.
    """
    cleaned_pages: list[DocumentPage] = []

    for page in document.pages:
        cleaned_text = clean_text(page.text)

        if cleaned_text:
            cleaned_pages.append(
                DocumentPage(
                    page_number=page.page_number,
                    text=cleaned_text,
                )
            )

    if not cleaned_pages:
        raise ValueError(
            f"Document contains no usable text after cleaning: "
            f"{document.filename}"
        )

    full_text = "\n\n".join(
        page.text
        for page in cleaned_pages
    ).strip()

    return document.model_copy(
        update={
            "pages": cleaned_pages,
            "full_text": full_text,
            "character_count": len(full_text),
        }
    )


def clean_text(text: str) -> str:
    """
    Apply conservative formatting cleanup.

    The function intentionally avoids rewriting clinical wording.
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Replace tabs with normal spaces.
    text = text.replace("\t", " ")

    # Remove trailing spaces from lines.
    lines = [
        line.rstrip()
        for line in text.split("\n")
    ]

    text = "\n".join(lines)

    # Collapse repeated spaces inside lines.
    text = re.sub(
        r"[ ]{2,}",
        " ",
        text,
    )

    # Reduce excessive blank lines while preserving paragraph breaks.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()