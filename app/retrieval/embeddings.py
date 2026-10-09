from collections.abc import Sequence
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config.settings import Settings, get_settings
from app.observability.tracing import (
    observed_span,
)


class EmbeddingConfigurationError(RuntimeError):
    """
    Raised when embedding configuration is invalid.
    """


class EmbeddingServiceError(RuntimeError):
    """
    Raised when local embedding generation fails.
    """


@lru_cache(maxsize=4)
def _load_model(
    model_name: str,
    device: str,
) -> SentenceTransformer:
    """
    Load and cache one Sentence Transformer model.

    The first call may download the model from Hugging Face.
    Later calls reuse the cached local model.
    """
    try:
        return SentenceTransformer(
            model_name,
            device=device,
        )

    except Exception as exc:
        raise EmbeddingServiceError(
            (
                "Failed to load local embedding model "
                f"'{model_name}'."
            )
        ) from exc


def _validate_text(
    text: str,
) -> str:
    """
    Validate and normalize one text input.
    """
    cleaned = text.strip()

    if not cleaned:
        raise ValueError(
            "Cannot embed empty text."
        )

    return cleaned


def _validate_vector_dimension(
    vector: list[float],
    expected_dimensions: int,
) -> None:
    """
    Ensure generated vectors match configured dimensions.
    """
    actual_dimensions = len(vector)

    if actual_dimensions != expected_dimensions:
        raise EmbeddingServiceError(
            (
                "Embedding dimension mismatch. "
                f"Expected {expected_dimensions}, "
                f"received {actual_dimensions}."
            )
        )


def embed_text(
    text: str,
    settings: Settings | None = None,
) -> list[float]:
    """
    Embed one text string using the local embedding model.
    """
    settings = settings or get_settings()

    cleaned = _validate_text(
        text
    )

    model = _load_model(
        settings.embedding_model,
        settings.embedding_device,
    )

    try:
        with observed_span(
            "embedding.generate",
            attributes={
                "medonto.embedding.model": (
                    settings.embedding_model
                ),
                "medonto.embedding.batch_size": 1,
                "medonto.embedding.dimensions": (
                    settings.embedding_dimensions
                ),
            },
        ):
            vector = model.encode(
                cleaned,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

    except Exception as exc:
        raise EmbeddingServiceError(
            "Local embedding generation failed."
        ) from exc

    result = [
        float(value)
        for value in vector.tolist()
    ]

    _validate_vector_dimension(
        result,
        settings.embedding_dimensions,
    )

    return result


def embed_texts(
    texts: Sequence[str],
    settings: Settings | None = None,
) -> list[list[float]]:
    """
    Embed multiple texts using local batching.
    """
    settings = settings or get_settings()

    if not texts:
        raise ValueError(
            "Cannot embed an empty text collection."
        )

    cleaned_texts = [
        _validate_text(text)
        for text in texts
    ]

    model = _load_model(
        settings.embedding_model,
        settings.embedding_device,
    )

    try:
        with observed_span(
            "embedding.generate",
            attributes={
                "medonto.embedding.model": (
                    settings.embedding_model
                ),
                "medonto.embedding.batch_size": (
                    len(cleaned_texts)
                ),
                "medonto.embedding.dimensions": (
                    settings.embedding_dimensions
                ),
            },
        ):
            vectors = model.encode(
                cleaned_texts,
                batch_size=settings.embedding_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

    except Exception as exc:
        raise EmbeddingServiceError(
            "Local batch embedding generation failed."
        ) from exc

    results: list[list[float]] = []

    for vector in vectors:
        result = [
            float(value)
            for value in vector.tolist()
        ]

        _validate_vector_dimension(
            result,
            settings.embedding_dimensions,
        )

        results.append(
            result
        )

    if len(results) != len(cleaned_texts):
        raise EmbeddingServiceError(
            (
                "Embedding output count does not "
                "match input count."
            )
        )

    return results