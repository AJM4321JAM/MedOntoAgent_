from collections import Counter

from app.config.settings import get_settings
from app.retrieval.qdrant_client import (
    create_qdrant_client,
)


ALLOWED_DOCUMENT_IDS = {
    "DOC-e943ba2bdc31",
    "DOC-21a51be8081f",
    "DOC-dad82839ba53",
    "DOC-d21df49f3ed9",
}


def main() -> None:
    settings = get_settings()
    client = create_qdrant_client(
        settings
    )

    counts: Counter[str] = Counter()

    offset = None
    total_points = 0

    while True:
        points, next_offset = client.scroll(
            collection_name=(
                settings.qdrant_collection
            ),
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}

            document_id = str(
                payload.get(
                    "document_id"
                )
                or ""
            )

            counts[
                document_id
            ] += 1

            total_points += 1

        if next_offset is None:
            break

        offset = next_offset

    actual_document_ids = set(
        counts
    )

    unexpected = (
        actual_document_ids
        - ALLOWED_DOCUMENT_IDS
    )

    missing = (
        ALLOWED_DOCUMENT_IDS
        - actual_document_ids
    )

    print(
        "COLLECTION:",
        settings.qdrant_collection,
    )

    print(
        "TOTAL POINTS:",
        total_points,
    )

    print()

    for document_id, count in sorted(
        counts.items()
    ):
        print(
            document_id,
            "|",
            count,
        )

    if unexpected:
        raise RuntimeError(
            (
                "Clinical collection contains "
                "unexpected document IDs: "
                f"{sorted(unexpected)}"
            )
        )

    if missing:
        raise RuntimeError(
            (
                "Clinical collection is missing "
                "required frozen documents: "
                f"{sorted(missing)}"
            )
        )

    if total_points != 1535:
        raise RuntimeError(
            (
                "Clinical collection point count "
                "does not match the reviewed "
                "Phase 28B corpus. "
                f"Expected 1535, got "
                f"{total_points}."
            )
        )

    print()
    print(
        "Clinical collection audit passed."
    )


if __name__ == "__main__":
    main()
