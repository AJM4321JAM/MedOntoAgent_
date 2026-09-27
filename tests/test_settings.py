import pytest
from pydantic import ValidationError

from app.config.settings import Settings


def test_default_settings_are_valid() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_name == "MedOntoAgent API"
    assert settings.app_version == "0.1.0"
    assert settings.app_env == "development"
    assert settings.aws_region == "ap-south-1"
    assert settings.qdrant_collection == "medontoagent_clinical_documents"
    assert settings.retrieval_top_k == 5
    assert settings.max_chunk_words == 600
    assert settings.chunk_overlap_words == 80
    assert settings.max_agent_retries == 1
    assert settings.log_level == "INFO"

    assert (
        settings.embedding_model
        == "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert settings.embedding_dimensions == 384
    assert settings.embedding_batch_size == 32
    assert settings.embedding_device == "cpu"


def test_top_k_cannot_be_zero() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            retrieval_top_k=0,
        )


def test_retry_limit_cannot_be_negative() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            max_agent_retries=-1,
        )


def test_invalid_environment_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            app_env="invalid",
        )


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            max_chunk_words=100,
            chunk_overlap_words=100,
        )


def test_embedding_dimensions_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            embedding_dimensions=0,
        )
