from functools import lru_cache
from typing import Literal

from pydantic import (
    Field,
    SecretStr,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    """Central runtime configuration for MedOntoAgent."""

    app_name: str = "MedOntoAgent API"
    app_version: str = "0.1.0"

    app_env: Literal[
        "development",
        "test",
        "staging",
        "production",
    ] = "development"

    aws_region: str = "ap-south-1"

    qdrant_collection: str = (
        "medontoagent_clinical_documents"
    )

    qdrant_test_collection: str = (
        "medontoagent_test_documents"
    )

    qdrant_url: str | None = None

    qdrant_api_key: SecretStr | None = None

    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
    )

    max_chunk_words: int = Field(
        default=600,
        ge=100,
        le=2000,
    )

    chunk_overlap_words: int = Field(
        default=80,
        ge=0,
        le=500,
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

    admin_api_key: SecretStr | None = None

    enable_dev_routes: bool = False

    max_request_bytes: int = Field(
        default=64 * 1024,
        ge=1024,
        le=10 * 1024 * 1024,
    )

    openrouter_api_key: SecretStr | None = None

    openrouter_base_url: str = (
        "https://openrouter.ai/api/v1"
    )

    # Free OpenRouter routing for development.
    openrouter_model: str = "openrouter/free"

    # Offline semantic-evaluation judge.
    # Kept separately configurable from the application model.
    ragas_evaluator_model: str = "openrouter/free"

    otel_enabled: bool = False

    otel_service_name: str = (
        "medontoagent"
    )

    otel_service_version: str = (
        "0.1.0"
    )

    otel_exporter_otlp_endpoint: str = (
        "http://localhost:4318"
    )

    otel_export_interval_ms: int = Field(
        default=10000,
        ge=1000,
    )

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

    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embedding_dimensions: int = Field(
        default=384,
        ge=1,
        le=65535,
    )

    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
    )

    embedding_device: str = "cpu"

    @model_validator(mode="after")
    def validate_chunking_settings(
        self,
    ) -> "Settings":
        if (
            self.chunk_overlap_words
            >= self.max_chunk_words
        ):
            raise ValueError(
                "CHUNK_OVERLAP_WORDS must be "
                "smaller than MAX_CHUNK_WORDS."
            )

        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return one cached configuration instance."""
    return Settings()
