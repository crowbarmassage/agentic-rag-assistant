"""LLM Provider factory for runtime provider selection."""

from enum import Enum
from typing import Optional

from .base import BaseLLMProvider
from .llm_openai import OpenAIProvider
from .llm_gemini import GeminiProvider
from .llm_groq import GroqProvider


class LLMProviderType(str, Enum):
    """Available LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    GROQ = "groq"


class LLMProviderFactory:
    """Factory for creating LLM provider instances."""

    _providers = {
        LLMProviderType.OPENAI: OpenAIProvider,
        LLMProviderType.GEMINI: GeminiProvider,
        LLMProviderType.GROQ: GroqProvider,
    }

    _default_models = {
        LLMProviderType.OPENAI: "gpt-4o-mini",
        LLMProviderType.GEMINI: "gemini-1.5-flash",
        LLMProviderType.GROQ: "llama-3.3-70b-versatile",
    }

    @classmethod
    def create(
        cls,
        provider: LLMProviderType | str,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> BaseLLMProvider:
        """
        Create an LLM provider instance.

        Args:
            provider: Provider type (enum or string)
            model: Model identifier (uses provider default if None)
            api_key: API key (uses environment variable if None)

        Returns:
            Configured LLM provider instance
        """
        # Convert string to enum if needed
        if isinstance(provider, str):
            provider = LLMProviderType(provider.lower())

        if provider not in cls._providers:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(LLMProviderType)}")

        provider_class = cls._providers[provider]
        kwargs = {}
        if model:
            kwargs["model"] = model
        if api_key:
            kwargs["api_key"] = api_key

        return provider_class(**kwargs)

    @classmethod
    def get_default_model(cls, provider: LLMProviderType | str) -> str:
        """Return default model for a provider."""
        if isinstance(provider, str):
            provider = LLMProviderType(provider.lower())
        return cls._default_models.get(provider, "unknown")

    @classmethod
    def list_providers(cls) -> list[str]:
        """List available provider names."""
        return [p.value for p in LLMProviderType]
