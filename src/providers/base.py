"""Abstract base classes for providers."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Type
from pydantic import BaseModel


class LLMResponse(BaseModel):
    """Standardized LLM response wrapper."""

    content: str
    model: str
    usage: dict[str, int]  # prompt_tokens, completion_tokens, total_tokens
    raw_response: Optional[Any] = None

    model_config = {"arbitrary_types_allowed": True}


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def __init__(self, model: str, **kwargs):
        """Initialize with model identifier."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """Generate text completion."""
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0
    ) -> BaseModel:
        """Generate structured output conforming to Pydantic model."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider identifier string."""
        pass


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def __init__(self, model: str, **kwargs):
        pass

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        pass

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings (batch)."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass
