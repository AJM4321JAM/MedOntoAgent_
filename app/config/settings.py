from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central runtime configuration for MedOntoAgent.

    Values can come from:
    1. environment variables
    2. the local .env file during development
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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    The cache ensures that configuration is loaded once and reused
    throughout the application.
    """
    return Settings()