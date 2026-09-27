from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.config.settings import Settings
from app.retrieval.embeddings import (
    EmbeddingServiceError,
    embed_text,
    embed_texts,
)


def create_test_settings(
    *,
    dimensions: int = 4,
    batch_size: int = 2,
) -> Settings:
    return Settings(
        _env_file=None,
        openrouter_api_key=None,
        embedding_model="test-model",
        embedding_dimensions=dimensions,
        embedding_batch_size=batch_size,
        embedding_device="cpu",
    )


def test_empty_text_is_rejected() -> None:
    settings = create_test_settings()

    with pytest.raises(
        ValueError,
        match="empty text",
    ):
        embed_text(
            "   ",
            settings,
        )


def test_empty_text_collection_is_rejected() -> None:
    settings = create_test_settings()

    with pytest.raises(
        ValueError,
        match="empty text collection",
    ):
        embed_texts(
            [],
            settings,
        )


@patch(
    "app.retrieval.embeddings._load_model"
)
def test_single_embedding_is_returned(
    mock_load_model: MagicMock,
) -> None:
    settings = create_test_settings(
        dimensions=4,
    )

    mock_model = MagicMock()

    mock_model.encode.return_value = (
        np.array(
            [
                0.1,
                0.2,
                0.3,
                0.4,
            ],
            dtype=float,
        )
    )

    mock_load_model.return_value = (
        mock_model
    )

    result = embed_text(
        "Hypertension guideline",
        settings,
    )

    assert len(result) == 4

    assert result == pytest.approx(
        [
            0.1,
            0.2,
            0.3,
            0.4,
        ]
    )


@patch(
    "app.retrieval.embeddings._load_model"
)
def test_embedding_dimension_mismatch_is_rejected(
    mock_load_model: MagicMock,
) -> None:
    settings = create_test_settings(
        dimensions=4,
    )

    mock_model = MagicMock()

    mock_model.encode.return_value = (
        np.array(
            [
                0.1,
                0.2,
                0.3,
            ],
            dtype=float,
        )
    )

    mock_load_model.return_value = (
        mock_model
    )

    with pytest.raises(
        EmbeddingServiceError,
        match="dimension mismatch",
    ):
        embed_text(
            "Hypertension",
            settings,
        )


@patch(
    "app.retrieval.embeddings._load_model"
)
def test_batch_embeddings_preserve_order(
    mock_load_model: MagicMock,
) -> None:
    settings = create_test_settings(
        dimensions=4,
        batch_size=2,
    )

    mock_model = MagicMock()

    mock_model.encode.return_value = (
        np.array(
            [
                [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                ],
                [
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                ],
            ],
            dtype=float,
        )
    )

    mock_load_model.return_value = (
        mock_model
    )

    results = embed_texts(
        [
            "first text",
            "second text",
        ],
        settings,
    )

    assert len(results) == 2

    assert results[0] == pytest.approx(
        [
            0.1,
            0.2,
            0.3,
            0.4,
        ]
    )

    assert results[1] == pytest.approx(
        [
            0.5,
            0.6,
            0.7,
            0.8,
        ]
    )


@patch(
    "app.retrieval.embeddings._load_model"
)
def test_batch_contains_no_blank_text(
    mock_load_model: MagicMock,
) -> None:
    settings = create_test_settings()

    with pytest.raises(
        ValueError,
        match="empty text",
    ):
        embed_texts(
            [
                "valid",
                "   ",
            ],
            settings,
        )

    mock_load_model.assert_not_called()


@patch(
    "app.retrieval.embeddings._load_model"
)
def test_local_model_failure_is_wrapped(
    mock_load_model: MagicMock,
) -> None:
    settings = create_test_settings()

    mock_model = MagicMock()

    mock_model.encode.side_effect = (
        RuntimeError(
            "model failure"
        )
    )

    mock_load_model.return_value = (
        mock_model
    )

    with pytest.raises(
        EmbeddingServiceError,
        match="generation failed",
    ):
        embed_text(
            "Hypertension",
            settings,
        )