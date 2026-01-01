"""Application configuration with environment variable support."""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application configuration.

    Loads from environment variables and .env file.
    """

    # LLM Configuration
    llm_provider: str = Field(
        default="openai",
        description="LLM provider: openai, gemini, groq"
    )
    llm_model: Optional[str] = Field(
        default=None,
        description="Specific model (uses provider default if None)"
    )

    # Embedding Configuration
    embedding_provider: str = Field(
        default="sentence_transformers",
        description="Embedding provider: sentence_transformers, openai, cohere, google"
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description="Specific embedding model"
    )

    # API Keys
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    cohere_api_key: Optional[str] = Field(default=None, alias="COHERE_API_KEY")

    # ChromaDB Configuration
    chroma_persist_dir: str = Field(default="./data/chroma_db")

    # Retrieval Configuration
    retrieval_min_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    retrieval_max_k: int = Field(default=10, ge=1, le=50)
    retrieval_drop_off_ratio: float = Field(default=0.7, ge=0.0, le=1.0)

    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    debug: bool = Field(default=True)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get application settings (cached singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Backwards compatibility
settings = get_settings()
