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
    assert settings.max_agent_retries == 1
    assert settings.log_level == "INFO"


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