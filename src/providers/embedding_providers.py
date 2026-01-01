"""Embedding provider implementations and factory."""

from enum import Enum
from typing import Optional

from .base import BaseEmbeddingProvider


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Local Sentence Transformers provider (free, no API key)."""

    SUPPORTED_MODELS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-MiniLM-L6-v2": 384,
    }
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model: Optional[str] = None, **kwargs):
        from sentence_transformers import SentenceTransformer

        self.model_name = model or self.DEFAULT_MODEL
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {self.model_name} not supported")

        self._dimension = self.SUPPORTED_MODELS[self.model_name]
        self.model = SentenceTransformer(self.model_name)

    def embed_text(self, text: str) -> list[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "sentence_transformers"


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embeddings provider."""

    SUPPORTED_MODELS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        from openai import OpenAI

        self.model_name = model or self.DEFAULT_MODEL
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {self.model_name} not supported")

        self._dimension = self.SUPPORTED_MODELS[self.model_name]
        self.client = OpenAI(api_key=api_key)

    def embed_text(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_name, input=text)
        return response.data[0].embedding

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "openai"


class CohereEmbeddingProvider(BaseEmbeddingProvider):
    """Cohere embeddings provider."""

    SUPPORTED_MODELS = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
    }
    DEFAULT_MODEL = "embed-english-v3.0"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        import cohere

        self.model_name = model or self.DEFAULT_MODEL
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {self.model_name} not supported")

        self._dimension = self.SUPPORTED_MODELS[self.model_name]
        self.client = cohere.Client(api_key=api_key)

    def embed_text(self, text: str) -> list[float]:
        response = self.client.embed(
            texts=[text],
            model=self.model_name,
            input_type="search_query"
        )
        return response.embeddings[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embed(
            texts=texts,
            model=self.model_name,
            input_type="search_document"
        )
        return response.embeddings

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "cohere"


class GoogleEmbeddingProvider(BaseEmbeddingProvider):
    """Google embeddings provider."""

    SUPPORTED_MODELS = {
        "text-embedding-004": 768,
        "embedding-001": 768,
    }
    DEFAULT_MODEL = "text-embedding-004"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        import google.generativeai as genai

        self.model_name = model or self.DEFAULT_MODEL
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {self.model_name} not supported")

        self._dimension = self.SUPPORTED_MODELS[self.model_name]
        if api_key:
            genai.configure(api_key=api_key)
        self.genai = genai

    def embed_text(self, text: str) -> list[float]:
        result = self.genai.embed_content(
            model=f"models/{self.model_name}",
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "google"


# ==================== Factory ====================

class EmbeddingProviderType(str, Enum):
    """Available embedding providers."""
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    COHERE = "cohere"
    GOOGLE = "google"


class EmbeddingProviderFactory:
    """Factory for creating embedding provider instances."""

    _providers = {
        EmbeddingProviderType.SENTENCE_TRANSFORMERS: SentenceTransformerProvider,
        EmbeddingProviderType.OPENAI: OpenAIEmbeddingProvider,
        EmbeddingProviderType.COHERE: CohereEmbeddingProvider,
        EmbeddingProviderType.GOOGLE: GoogleEmbeddingProvider,
    }

    @classmethod
    def create(
        cls,
        provider: EmbeddingProviderType | str,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> BaseEmbeddingProvider:
        """Create an embedding provider instance."""
        if isinstance(provider, str):
            provider = EmbeddingProviderType(provider.lower())

        if provider not in cls._providers:
            raise ValueError(f"Unknown provider: {provider}")

        provider_class = cls._providers[provider]
        kwargs = {}
        if model:
            kwargs["model"] = model
        if api_key:
            kwargs["api_key"] = api_key

        return provider_class(**kwargs)

    @classmethod
    def list_providers(cls) -> list[str]:
        """List available provider names."""
        return [p.value for p in EmbeddingProviderType]
