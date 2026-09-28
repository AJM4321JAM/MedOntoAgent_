import argparse
from datetime import (
    date,
)

from app.ingestion.pipeline import (
    ingest_document,
)
from app.models.documents import (
    ApprovalStatus,
    SourceAuthority,
)


def _parse_optional_date(
    value: str | None,
) -> date | None:
    if value is None:
        return None

    return date.fromisoformat(
        value
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest one governed source document "
            "into MedOntoAgent."
        )
    )

    parser.add_argument(
        "file_path",
    )

    parser.add_argument(
        "--title",
        required=True,
    )

    parser.add_argument(
        "--publisher",
        required=True,
    )

    parser.add_argument(
        "--document-type",
        required=True,
    )

    parser.add_argument(
        "--clinical-domain",
        required=True,
    )

    parser.add_argument(
        "--source-authority",
        required=True,
        choices=[
            value.value
            for value
            in SourceAuthority
        ],
    )

    parser.add_argument(
        "--version",
        required=True,
    )

    parser.add_argument(
        "--approval-status",
        default=(
            ApprovalStatus
            .PENDING
            .value
        ),
        choices=[
            value.value
            for value
            in ApprovalStatus
        ],
    )

    parser.add_argument(
        "--publication-date",
        default=None,
        help="ISO date: YYYY-MM-DD",
    )

    parser.add_argument(
        "--effective-date",
        default=None,
        help="ISO date: YYYY-MM-DD",
    )

    parser.add_argument(
        "--source-url",
        default=None,
    )

    parser.add_argument(
        "--supersedes-document-id",
        default=None,
    )

    parser.add_argument(
        "--registry-path",
        default=(
            "data/processed/"
            "document_registry.json"
        ),
    )

    args = parser.parse_args()

    result = ingest_document(
        args.file_path,
        title=args.title,
        publisher=args.publisher,
        document_type=(
            args.document_type
        ),
        clinical_domain=(
            args.clinical_domain
        ),
        source_authority=(
            SourceAuthority(
                args.source_authority
            )
        ),
        version=args.version,
        approval_status=(
            ApprovalStatus(
                args.approval_status
            )
        ),
        publication_date=(
            _parse_optional_date(
                args.publication_date
            )
        ),
        effective_date=(
            _parse_optional_date(
                args.effective_date
            )
        ),
        source_url=(
            args.source_url
        ),
        supersedes_document_id=(
            args.supersedes_document_id
        ),
        registry_path=(
            args.registry_path
        ),
    )

    print(
        result.model_dump_json(
            indent=2
        )
    )


if __name__ == "__main__":
    main()