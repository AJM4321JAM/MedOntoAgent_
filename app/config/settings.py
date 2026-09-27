from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central runtime configuration for MedOntoAgent.
    """

    app_name: str = "MedOntoAgent API"
    app_version: str = "0.1.0"

    app_env: Literal[
        "development",
        "test",
        "staging",
        "production",
    ] = "development"

    aws_region: str = "ap-south-1"

    qdrant_collection: str = "medontoagent_clinical_documents"

    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
    )

    max_agent_retries: int = Field(
        default=1,
        ge=0,
        le=5,
    )

    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    openrouter_api_key: SecretStr | None = None

    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Free OpenRouter routing for development.
    openrouter_model: str = "openrouter/free"

    # Prevent accidental very large output requests.
    llm_max_output_tokens: int = Field(
        default=1000,
        ge=1,
        le=4096,
    )

    llm_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        le=300,
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Return one cached configuration instance.
    """
    return Settings()