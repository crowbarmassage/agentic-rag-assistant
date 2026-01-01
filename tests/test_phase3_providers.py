"""Phase 3 Tests: Provider Abstraction Layer."""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestBaseClasses:
    """Test base provider abstract classes."""

    def test_base_llm_provider_import(self):
        """Verify base LLM provider class can be imported."""
        from src.providers.base import BaseLLMProvider
        assert BaseLLMProvider is not None

    def test_base_embedding_provider_import(self):
        """Verify base embedding provider class can be imported."""
        from src.providers.base import BaseEmbeddingProvider
        assert BaseEmbeddingProvider is not None

    def test_llm_response_model(self):
        """Verify LLMResponse model."""
        from src.providers.base import LLMResponse

        response = LLMResponse(
            content="Hello, world!",
            model="gpt-4o-mini",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        )

        assert response.content == "Hello, world!"
        assert response.model == "gpt-4o-mini"
        assert response.usage["total_tokens"] == 15


class TestLLMProviderFactory:
    """Test LLM provider factory."""

    def test_factory_import(self):
        """Verify factory can be imported."""
        from src.providers import LLMProviderFactory, LLMProviderType
        assert LLMProviderFactory is not None
        assert LLMProviderType is not None

    def test_factory_list_providers(self):
        """Verify factory lists all providers."""
        from src.providers import LLMProviderFactory

        providers = LLMProviderFactory.list_providers()
        assert "openai" in providers
        assert "gemini" in providers
        assert "groq" in providers
        assert len(providers) == 3

    def test_factory_get_default_model(self):
        """Verify factory returns default models."""
        from src.providers import LLMProviderFactory

        assert LLMProviderFactory.get_default_model("openai") == "gpt-4o-mini"
        assert LLMProviderFactory.get_default_model("gemini") == "gemini-1.5-flash"
        assert LLMProviderFactory.get_default_model("groq") == "llama-3.3-70b-versatile"

    def test_factory_invalid_provider(self):
        """Verify factory raises error for invalid provider."""
        from src.providers import LLMProviderFactory

        with pytest.raises(ValueError):
            LLMProviderFactory.create("invalid_provider")


class TestOpenAIProvider:
    """Test OpenAI LLM provider."""

    def test_openai_provider_import(self):
        """Verify OpenAI provider can be imported."""
        from src.providers import OpenAIProvider
        assert OpenAIProvider is not None

    def test_openai_supported_models(self):
        """Verify OpenAI supported models list."""
        from src.providers import OpenAIProvider

        assert "gpt-4o-mini" in OpenAIProvider.SUPPORTED_MODELS
        assert "gpt-4o" in OpenAIProvider.SUPPORTED_MODELS
        assert OpenAIProvider.DEFAULT_MODEL == "gpt-4o-mini"

    def test_openai_invalid_model(self):
        """Verify error for unsupported model."""
        from src.providers import OpenAIProvider

        with pytest.raises(ValueError, match="not supported"):
            OpenAIProvider(model="invalid-model", api_key="test")

    @patch('src.providers.llm_openai.OpenAI')
    def test_openai_provider_creation(self, mock_openai):
        """Verify OpenAI provider can be created with mock."""
        from src.providers import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")
        assert provider.provider_name == "openai"
        assert provider.model == "gpt-4o-mini"


class TestGeminiProvider:
    """Test Gemini LLM provider."""

    def test_gemini_provider_import(self):
        """Verify Gemini provider can be imported."""
        from src.providers import GeminiProvider
        assert GeminiProvider is not None

    def test_gemini_supported_models(self):
        """Verify Gemini supported models list."""
        from src.providers import GeminiProvider

        assert "gemini-1.5-flash" in GeminiProvider.SUPPORTED_MODELS
        assert "gemini-1.5-pro" in GeminiProvider.SUPPORTED_MODELS
        assert GeminiProvider.DEFAULT_MODEL == "gemini-1.5-flash"

    def test_gemini_invalid_model(self):
        """Verify error for unsupported model."""
        from src.providers import GeminiProvider

        with pytest.raises(ValueError, match="not supported"):
            GeminiProvider(model="invalid-model", api_key="test")


class TestGroqProvider:
    """Test Groq LLM provider."""

    def test_groq_provider_import(self):
        """Verify Groq provider can be imported."""
        from src.providers import GroqProvider
        assert GroqProvider is not None

    def test_groq_supported_models(self):
        """Verify Groq supported models list."""
        from src.providers import GroqProvider

        assert "llama-3.3-70b-versatile" in GroqProvider.SUPPORTED_MODELS
        assert GroqProvider.DEFAULT_MODEL == "llama-3.3-70b-versatile"

    def test_groq_invalid_model(self):
        """Verify error for unsupported model."""
        from src.providers import GroqProvider

        with pytest.raises(ValueError, match="not supported"):
            GroqProvider(model="invalid-model", api_key="test")


class TestEmbeddingProviderFactory:
    """Test embedding provider factory."""

    def test_embedding_factory_import(self):
        """Verify embedding factory can be imported."""
        from src.providers import EmbeddingProviderFactory, EmbeddingProviderType
        assert EmbeddingProviderFactory is not None
        assert EmbeddingProviderType is not None

    def test_embedding_factory_list_providers(self):
        """Verify factory lists all embedding providers."""
        from src.providers import EmbeddingProviderFactory

        providers = EmbeddingProviderFactory.list_providers()
        assert "sentence_transformers" in providers
        assert "openai" in providers
        assert "cohere" in providers
        assert "google" in providers

    def test_embedding_factory_invalid_provider(self):
        """Verify factory raises error for invalid provider."""
        from src.providers import EmbeddingProviderFactory

        with pytest.raises(ValueError):
            EmbeddingProviderFactory.create("invalid_provider")


class TestSentenceTransformerProvider:
    """Test Sentence Transformers embedding provider."""

    def test_sentence_transformer_import(self):
        """Verify SentenceTransformer provider can be imported."""
        from src.providers import SentenceTransformerProvider
        assert SentenceTransformerProvider is not None

    def test_sentence_transformer_supported_models(self):
        """Verify supported models and dimensions."""
        from src.providers import SentenceTransformerProvider

        assert "all-MiniLM-L6-v2" in SentenceTransformerProvider.SUPPORTED_MODELS
        assert SentenceTransformerProvider.SUPPORTED_MODELS["all-MiniLM-L6-v2"] == 384
        assert SentenceTransformerProvider.DEFAULT_MODEL == "all-MiniLM-L6-v2"

    @pytest.mark.slow
    def test_sentence_transformer_creation(self):
        """Verify SentenceTransformer provider creation (requires model download)."""
        from src.providers import SentenceTransformerProvider

        provider = SentenceTransformerProvider()
        assert provider.provider_name == "sentence_transformers"
        assert provider.dimension == 384

    @pytest.mark.slow
    def test_sentence_transformer_embed_text(self):
        """Verify single text embedding."""
        from src.providers import SentenceTransformerProvider

        provider = SentenceTransformerProvider()
        embedding = provider.embed_text("Hello, world!")

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.slow
    def test_sentence_transformer_embed_texts_batch(self):
        """Verify batch text embedding."""
        from src.providers import SentenceTransformerProvider

        provider = SentenceTransformerProvider()
        texts = ["Hello", "World", "Test"]
        embeddings = provider.embed_texts(texts)

        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)


class TestOpenAIEmbeddingProvider:
    """Test OpenAI embedding provider."""

    def test_openai_embedding_import(self):
        """Verify OpenAI embedding provider can be imported."""
        from src.providers import OpenAIEmbeddingProvider
        assert OpenAIEmbeddingProvider is not None

    def test_openai_embedding_supported_models(self):
        """Verify supported models and dimensions."""
        from src.providers import OpenAIEmbeddingProvider

        assert "text-embedding-3-small" in OpenAIEmbeddingProvider.SUPPORTED_MODELS
        assert OpenAIEmbeddingProvider.SUPPORTED_MODELS["text-embedding-3-small"] == 1536


class TestCohereEmbeddingProvider:
    """Test Cohere embedding provider."""

    def test_cohere_embedding_import(self):
        """Verify Cohere embedding provider can be imported."""
        from src.providers import CohereEmbeddingProvider
        assert CohereEmbeddingProvider is not None

    def test_cohere_embedding_supported_models(self):
        """Verify supported models and dimensions."""
        from src.providers import CohereEmbeddingProvider

        assert "embed-english-v3.0" in CohereEmbeddingProvider.SUPPORTED_MODELS
        assert CohereEmbeddingProvider.SUPPORTED_MODELS["embed-english-v3.0"] == 1024


class TestGoogleEmbeddingProvider:
    """Test Google embedding provider."""

    def test_google_embedding_import(self):
        """Verify Google embedding provider can be imported."""
        from src.providers import GoogleEmbeddingProvider
        assert GoogleEmbeddingProvider is not None

    def test_google_embedding_supported_models(self):
        """Verify supported models and dimensions."""
        from src.providers import GoogleEmbeddingProvider

        assert "text-embedding-004" in GoogleEmbeddingProvider.SUPPORTED_MODELS
        assert GoogleEmbeddingProvider.SUPPORTED_MODELS["text-embedding-004"] == 768
