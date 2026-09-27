from pathlib import Path

import pytest

from app.ontology.ontology_loader import (
    OntologyLoadError,
    load_ontology,
)


def test_load_real_project_ontology() -> None:
    graph = load_ontology()

    assert len(
        graph
    ) > 0


def test_missing_ontology_is_rejected(
    tmp_path: Path,
) -> None:
    missing = (
        tmp_path
        / "missing.ttl"
    )

    with pytest.raises(
        OntologyLoadError,
        match="does not exist",
    ):
        load_ontology(
            missing
        )


def test_empty_ontology_is_rejected(
    tmp_path: Path,
) -> None:
    empty = (
        tmp_path
        / "empty.ttl"
    )

    empty.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        OntologyLoadError,
        match="empty",
    ):
        load_ontology(
            empty
        )


def test_invalid_turtle_is_rejected(
    tmp_path: Path,
) -> None:
    invalid = (
        tmp_path
        / "invalid.ttl"
    )

    invalid.write_text(
        """
        this is not valid turtle {
        """,
        encoding="utf-8",
    )

    with pytest.raises(
        OntologyLoadError,
        match="Failed to parse ontology",
    ):
        load_ontology(
            invalid
        )