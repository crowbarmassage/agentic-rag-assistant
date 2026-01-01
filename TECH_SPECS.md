# TECH_SPECS.md — ShopUNow Agentic AI Assistant

## Document Metadata
- **Project**: Agentic AI Assistant for ShopUNow (Retail Company)
- **Version**: 1.0
- **Last Updated**: 2025-01-01
- **Framework**: Pure Python (LangGraph fallback if complexity warrants)

---

## Table of Contents
1. [System Architecture Overview](#1-system-architecture-overview)
2. [Provider Abstraction Layer](#2-provider-abstraction-layer)
3. [Data Models (Pydantic)](#3-data-models-pydantic)
4. [ChromaDB Schema & Metadata Strategy](#4-chromadb-schema--metadata-strategy)
5. [Classification Pipeline](#5-classification-pipeline)
6. [RAG Pipeline](#6-rag-pipeline)
7. [Routing Logic](#7-routing-logic)
8. [Orchestration Layer](#8-orchestration-layer)
9. [FastAPI Specifications](#9-fastapi-specifications)
10. [Configuration Management](#10-configuration-management)
11. [Error Handling Strategy](#11-error-handling-strategy)
12. [Testing Strategy](#12-testing-strategy)

---

## 1. System Architecture Overview

### 1.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER QUERY                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLASSIFICATION PIPELINE                              │
│  ┌─────────────────────┐    ┌─────────────────────┐                         │
│  │  Sentiment Detector │───▶│ Department Classifier│                         │
│  │     (LLM-based)     │    │  (LLM + Structured)  │                         │
│  └─────────────────────┘    └─────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ROUTER                                          │
│                                                                              │
│   [Negative Sentiment]              [Positive/Neutral + Valid Department]   │
│          OR                                        │                         │
│   [Unknown Department]                             │                         │
│          │                                         │                         │
│          ▼                                         ▼                         │
│   ┌──────────────┐                    ┌───────────────────────┐             │
│   │    HUMAN     │                    │     RAG PIPELINE      │             │
│   │  ESCALATION  │                    │  (Filtered Retrieval) │             │
│   └──────────────┘                    └───────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RESPONSE GENERATION                                  │
│              (Answer + Source Attribution + Confidence)                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Inventory

| Component | Technology | Purpose |
|-----------|------------|---------|
| LLM Provider | OpenAI / Gemini / Groq (abstracted) | Classification, response generation |
| Embedding Provider | Sentence Transformers / OpenAI / Cohere / Google (abstracted) | Document & query embedding |
| Vector Store | ChromaDB | Persistent storage with metadata filtering |
| API Layer | FastAPI | RESTful interface |
| Data Validation | Pydantic v2 | Schema enforcement, serialization |
| Configuration | Pydantic Settings + .env | Environment-based config |

### 1.3 Directory Structure

```
shopunow_assistant/
├── src/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application entry
│   ├── config.py                   # Configuration management
│   ├── models/
│   │   ├── __init__.py
│   │   ├── data_models.py          # Core Pydantic models
│   │   ├── api_models.py           # Request/Response models
│   │   └── enums.py                # Enumerations
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract base classes
│   │   ├── llm_provider.py         # LLM abstraction
│   │   └── embedding_provider.py   # Embedding abstraction
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── classification.py       # Sentiment + Department classification
│   │   ├── retrieval.py            # RAG retrieval logic
│   │   └── generation.py           # Response generation
│   ├── routing/
│   │   ├── __init__.py
│   │   └── router.py               # Query routing logic
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── chroma_client.py        # ChromaDB wrapper
│   │   └── ingestion.py            # Data ingestion utilities
│   └── utils/
│       ├── __init__.py
│       ├── prompts.py              # Prompt templates
│       └── logging_config.py       # Logging setup
├── data/
│   ├── raw/                        # Generated QA JSON files
│   │   ├── hr_faqs.json
│   │   ├── it_faqs.json
│   │   ├── billing_faqs.json
│   │   └── shipping_faqs.json
│   └── chroma_db/                  # Persistent ChromaDB storage
├── tests/
│   ├── __init__.py
│   ├── test_classification.py
│   ├── test_retrieval.py
│   ├── test_routing.py
│   └── test_api.py
├── notebooks/
│   └── ShopUNow_Capstone.ipynb     # Final deliverable notebook
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 2. Provider Abstraction Layer

### 2.1 Design Philosophy

The provider abstraction enables runtime switching between LLM and embedding providers without code changes. This is achieved through:
- Abstract base classes defining contracts
- Factory pattern for provider instantiation
- Configuration-driven provider selection

### 2.2 LLM Provider Abstraction

```python
# src/providers/base.py

from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel

class LLMResponse(BaseModel):
    """Standardized LLM response wrapper."""
    content: str
    model: str
    usage: dict[str, int]  # tokens: prompt, completion, total
    raw_response: Optional[Any] = None

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def __init__(self, model: str, **kwargs):
        """Initialize with model identifier and provider-specific kwargs."""
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
        response_model: type[BaseModel],
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
```

### 2.3 LLM Provider Implementations

```python
# src/providers/llm_provider.py

import json
from typing import Optional, Type
from openai import OpenAI
import google.generativeai as genai
from groq import Groq
from pydantic import BaseModel

from .base import BaseLLMProvider, LLMResponse

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider implementation."""
    
    SUPPORTED_MODELS = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo"
    ]
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model} not in supported models: {self.SUPPORTED_MODELS}")
        self.model = model
        self.client = OpenAI(api_key=api_key)  # Uses OPENAI_API_KEY env var if None
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            raw_response=response
        )
    
    def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0
    ) -> BaseModel:
        """Use OpenAI's JSON mode with schema enforcement."""
        messages = []
        
        # Build schema instruction
        schema_instruction = f"""
You must respond with valid JSON that conforms to this schema:
{json.dumps(response_model.model_json_schema(), indent=2)}

Respond ONLY with the JSON object, no additional text.
"""
        
        effective_system = (system_prompt or "") + "\n\n" + schema_instruction
        messages.append({"role": "system", "content": effective_system.strip()})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        
        raw_content = response.choices[0].message.content
        return response_model.model_validate_json(raw_content)
    
    @property
    def provider_name(self) -> str:
        return "openai"


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider implementation."""
    
    SUPPORTED_MODELS = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-pro"
    ]
    
    def __init__(self, model: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model} not in supported models: {self.SUPPORTED_MODELS}")
        self.model_name = model
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024
    ) -> LLMResponse:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        
        response = self.model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        return LLMResponse(
            content=response.text,
            model=self.model_name,
            usage={
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count
            },
            raw_response=response
        )
    
    def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0
    ) -> BaseModel:
        schema_instruction = f"""
Respond with valid JSON conforming to this schema:
{json.dumps(response_model.model_json_schema(), indent=2)}

Output ONLY the JSON object.
"""
        full_prompt = f"{system_prompt or ''}\n\n{schema_instruction}\n\n{prompt}"
        
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json"
        )
        
        response = self.model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        return response_model.model_validate_json(response.text)
    
    @property
    def provider_name(self) -> str:
        return "gemini"


class GroqProvider(BaseLLMProvider):
    """Groq API provider implementation."""
    
    SUPPORTED_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768"
    ]
    
    def __init__(self, model: str = "llama-3.3-70b-versatile", api_key: Optional[str] = None):
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model} not in supported models: {self.SUPPORTED_MODELS}")
        self.model = model
        self.client = Groq(api_key=api_key)  # Uses GROQ_API_KEY env var if None
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            raw_response=response
        )
    
    def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0
    ) -> BaseModel:
        schema_instruction = f"""
Respond with valid JSON conforming to this schema:
{json.dumps(response_model.model_json_schema(), indent=2)}

Output ONLY the JSON object, no markdown formatting.
"""
        messages = []
        effective_system = (system_prompt or "") + "\n\n" + schema_instruction
        messages.append({"role": "system", "content": effective_system.strip()})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        
        raw_content = response.choices[0].message.content
        return response_model.model_validate_json(raw_content)
    
    @property
    def provider_name(self) -> str:
        return "groq"
```

### 2.4 LLM Provider Factory

```python
# src/providers/llm_provider.py (continued)

from enum import Enum
from typing import Optional

class LLMProviderType(str, Enum):
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
    
    @classmethod
    def create(
        cls,
        provider: LLMProviderType,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> BaseLLMProvider:
        """
        Create an LLM provider instance.
        
        Args:
            provider: Provider type enum
            model: Model identifier (uses provider default if None)
            api_key: API key (uses environment variable if None)
            
        Returns:
            Configured LLM provider instance
        """
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
    def get_default_model(cls, provider: LLMProviderType) -> str:
        """Return default model for a provider."""
        defaults = {
            LLMProviderType.OPENAI: "gpt-4o-mini",
            LLMProviderType.GEMINI: "gemini-1.5-flash",
            LLMProviderType.GROQ: "llama-3.3-70b-versatile",
        }
        return defaults[provider]
```

### 2.5 Embedding Provider Abstraction

```python
# src/providers/embedding_provider.py

from abc import ABC, abstractmethod
from typing import Optional
from enum import Enum
import numpy as np

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


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Local Sentence Transformers provider (free, no API key)."""
    
    SUPPORTED_MODELS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-MiniLM-L6-v2": 384,
    }
    
    def __init__(self, model: str = "all-MiniLM-L6-v2", **kwargs):
        from sentence_transformers import SentenceTransformer
        
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model} not in supported models")
        self.model_name = model
        self._dimension = self.SUPPORTED_MODELS[model]
        self.model = SentenceTransformer(model)
    
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
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        from openai import OpenAI
        
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model} not in supported models")
        self.model_name = model
        self._dimension = self.SUPPORTED_MODELS[model]
        self.client = OpenAI(api_key=api_key)
    
    def embed_text(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
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
    
    def __init__(self, model: str = "embed-english-v3.0", api_key: Optional[str] = None):
        import cohere
        
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model} not in supported models")
        self.model_name = model
        self._dimension = self.SUPPORTED_MODELS[model]
        self.client = cohere.Client(api_key=api_key)  # Uses COHERE_API_KEY env var
    
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
    
    def __init__(self, model: str = "text-embedding-004", api_key: Optional[str] = None):
        import google.generativeai as genai
        
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model} not in supported models")
        self.model_name = model
        self._dimension = self.SUPPORTED_MODELS[model]
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
        # Google doesn't have native batch, so we iterate
        return [self.embed_text(text) for text in texts]
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def provider_name(self) -> str:
        return "google"


# Embedding Provider Factory
class EmbeddingProviderType(str, Enum):
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
        provider: EmbeddingProviderType,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> BaseEmbeddingProvider:
        if provider not in cls._providers:
            raise ValueError(f"Unknown provider: {provider}")
        
        provider_class = cls._providers[provider]
        kwargs = {}
        if model:
            kwargs["model"] = model
        if api_key:
            kwargs["api_key"] = api_key
            
        return provider_class(**kwargs)
```

### 2.6 Embedding Provider Comparison

| Provider | Model | Dimensions | Similarity Score Range | Recommended Threshold |
|----------|-------|------------|----------------------|----------------------|
| **Sentence Transformers** | all-MiniLM-L6-v2 | 384 | 0.3 - 0.7 | 0.3 |
| **Sentence Transformers** | all-mpnet-base-v2 | 768 | 0.4 - 0.8 | 0.4 |
| **OpenAI** | text-embedding-3-small | 1536 | 0.5 - 0.9 | 0.5 |
| **OpenAI** | text-embedding-3-large | 3072 | 0.6 - 0.95 | 0.6 |
| **Cohere** | embed-english-v3.0 | 1024 | 0.4 - 0.85 | 0.45 |
| **Google** | text-embedding-004 | 768 | 0.4 - 0.8 | 0.4 |

**Key Observations:**

1. **Dimension vs. Score Range**: Higher-dimensional embeddings (OpenAI) typically produce higher similarity scores for relevant matches and greater separation between relevant/irrelevant results.

2. **MiniLM Trade-offs**:
   - ✅ Free, local, fast, no API calls
   - ✅ Works well for FAQ-style retrieval
   - ⚠️ Lower similarity scores (0.3-0.5 for good matches)
   - ⚠️ Less semantic distinction for complex queries

3. **When to Switch Providers**:
   - Use **Sentence Transformers** for: Cost-sensitive deployments, privacy requirements, offline usage
   - Use **OpenAI embeddings** for: Higher retrieval precision, complex semantic queries, production at scale

4. **Threshold Adjustment Rule**: When switching embedding providers, adjust your threshold proportionally to the score range. A query scoring 0.45 with MiniLM might score 0.75 with OpenAI embeddings.

---

## 3. Data Models (Pydantic)

### 3.1 Enumerations

```python
# src/models/enums.py

from enum import Enum

class Department(str, Enum):
    """ShopUNow department identifiers."""
    HR = "hr"
    IT_SUPPORT = "it_support"
    BILLING = "billing"
    SHIPPING = "shipping"
    UNKNOWN = "unknown"
    
    @classmethod
    def internal_departments(cls) -> list["Department"]:
        """Departments serving internal employees."""
        return [cls.HR, cls.IT_SUPPORT]
    
    @classmethod
    def external_departments(cls) -> list["Department"]:
        """Departments serving external customers."""
        return [cls.BILLING, cls.SHIPPING]
    
    @classmethod
    def valid_departments(cls) -> list["Department"]:
        """All valid departments (excludes UNKNOWN)."""
        return [cls.HR, cls.IT_SUPPORT, cls.BILLING, cls.SHIPPING]


class UserType(str, Enum):
    """User classification."""
    INTERNAL_EMPLOYEE = "internal_employee"
    EXTERNAL_CUSTOMER = "external_customer"
    UNKNOWN = "unknown"


class Sentiment(str, Enum):
    """Query sentiment classification."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class RouteDecision(str, Enum):
    """Routing decision outcomes."""
    RAG_PIPELINE = "rag_pipeline"
    HUMAN_ESCALATION = "human_escalation"
```

### 3.2 Core Data Models

```python
# src/models/data_models.py

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from .enums import Department, UserType, Sentiment, RouteDecision


# ============== QA Data Models ==============

class QAPair(BaseModel):
    """Single question-answer pair for knowledge base."""
    id: str = Field(..., description="Unique identifier for QA pair")
    question: str = Field(..., min_length=10, description="The FAQ question")
    answer: str = Field(..., min_length=20, description="The FAQ answer")
    department: Department = Field(..., description="Owning department")
    user_type: UserType = Field(..., description="Target user type")
    keywords: list[str] = Field(default_factory=list, description="Search keywords")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator('keywords', mode='before')
    @classmethod
    def lowercase_keywords(cls, v):
        if isinstance(v, list):
            return [kw.lower().strip() for kw in v]
        return v


class DepartmentFAQs(BaseModel):
    """Collection of FAQs for a single department."""
    department: Department
    user_type: UserType
    description: str = Field(..., description="Department description")
    qa_pairs: list[QAPair] = Field(..., min_length=10, max_length=25)
    
    @property
    def count(self) -> int:
        return len(self.qa_pairs)


# ============== Classification Models ==============

class ClassificationResult(BaseModel):
    """Result of query classification pipeline."""
    sentiment: Sentiment = Field(..., description="Detected sentiment")
    department: Department = Field(..., description="Classified department")
    user_type: UserType = Field(..., description="Inferred user type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    reasoning: str = Field(..., description="LLM's reasoning for classification")


class SentimentAnalysis(BaseModel):
    """Structured output for sentiment detection."""
    sentiment: Sentiment
    confidence: float = Field(..., ge=0.0, le=1.0)
    indicators: list[str] = Field(
        default_factory=list, 
        description="Key phrases/words that influenced sentiment detection"
    )


class DepartmentClassification(BaseModel):
    """Structured output for department classification."""
    department: Department
    user_type: UserType
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Why this department was selected")


# ============== Retrieval Models ==============

class RetrievedDocument(BaseModel):
    """Single document retrieved from vector store."""
    id: str
    content: str = Field(..., description="QA pair as concatenated text")
    question: str
    answer: str
    department: Department
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Result of RAG retrieval operation."""
    query: str
    department_filter: Department
    documents: list[RetrievedDocument]
    retrieval_count: int
    threshold_used: float
    
    @property
    def has_results(self) -> bool:
        return len(self.documents) > 0
    
    @property
    def top_document(self) -> Optional[RetrievedDocument]:
        return self.documents[0] if self.documents else None


# ============== Response Models ==============

class SourceAttribution(BaseModel):
    """Source attribution for response transparency."""
    document_id: str
    question_matched: str
    relevance_score: float


class GeneratedResponse(BaseModel):
    """Final generated response to user query."""
    answer: str = Field(..., description="Generated answer text")
    sources: list[SourceAttribution] = Field(
        default_factory=list,
        description="Sources used to generate answer"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    department: Department
    is_escalated: bool = Field(default=False)
    escalation_reason: Optional[str] = None


# ============== Routing Models ==============

class RoutingDecision(BaseModel):
    """Complete routing decision with all context."""
    route: RouteDecision
    classification: ClassificationResult
    reasoning: str
    
    @property
    def should_escalate(self) -> bool:
        return self.route == RouteDecision.HUMAN_ESCALATION
```

### 3.3 API Request/Response Models

```python
# src/models/api_models.py

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field
from .enums import Department, Sentiment, RouteDecision
from .data_models import SourceAttribution


# ============== Request Models ==============

class QueryRequest(BaseModel):
    """Incoming user query request."""
    query: str = Field(..., min_length=3, max_length=1000, description="User's question")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "How do I apply for paid time off?",
                    "user_id": "emp_12345"
                },
                {
                    "query": "Where is my order #98765?",
                    "user_id": "cust_67890"
                }
            ]
        }
    }


class HealthCheckRequest(BaseModel):
    """Health check request with optional deep check flag."""
    deep_check: bool = Field(default=False, description="If true, verify all dependencies")


# ============== Response Models ==============

class QueryResponse(BaseModel):
    """Response to user query."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    department: Department = Field(..., description="Department that handled query")
    sentiment: Sentiment = Field(..., description="Detected sentiment")
    sources: list[SourceAttribution] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    was_escalated: bool = Field(default=False)
    escalation_message: Optional[str] = None
    processing_time_ms: float = Field(..., description="Total processing time")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "How do I apply for paid time off?",
                    "answer": "You can apply for PTO through the HR portal...",
                    "department": "hr",
                    "sentiment": "neutral",
                    "sources": [{"document_id": "hr_001", "question_matched": "...", "relevance_score": 0.92}],
                    "confidence": 0.89,
                    "was_escalated": False,
                    "processing_time_ms": 1250.5,
                    "timestamp": "2025-01-01T12:00:00Z"
                }
            ]
        }
    }


class EscalationResponse(BaseModel):
    """Response for escalated queries."""
    query: str
    message: str = Field(
        default="Your query has been escalated to a human support agent. "
                "A representative will contact you shortly."
    )
    reason: str
    ticket_id: Optional[str] = None
    estimated_response_time: str = Field(default="24-48 hours")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Overall status: healthy, degraded, unhealthy")
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: dict[str, str] = Field(
        default_factory=dict,
        description="Status of individual components"
    )


class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

---

## 4. ChromaDB Schema & Metadata Strategy

### 4.1 Collection Design

**Single Collection Approach** with rich metadata for filtered retrieval.

```python
# src/vectorstore/chroma_client.py

import chromadb
from chromadb.config import Settings
from typing import Optional
from pathlib import Path

from src.models.enums import Department, UserType
from src.models.data_models import QAPair, RetrievedDocument
from src.providers.embedding_provider import BaseEmbeddingProvider


class ChromaDBClient:
    """ChromaDB wrapper with metadata filtering support."""
    
    COLLECTION_NAME = "shopunow_faqs"
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        embedding_provider: BaseEmbeddingProvider = None
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.embedding_provider = embedding_provider
        self._collection = None
    
    @property
    def collection(self):
        """Lazy initialization of collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"description": "ShopUNow FAQ knowledge base"}
            )
        return self._collection
    
    def add_qa_pair(self, qa_pair: QAPair) -> None:
        """Add single QA pair to collection."""
        # Concatenate Q&A for embedding
        document_text = f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer}"
        
        # Generate embedding
        embedding = self.embedding_provider.embed_text(document_text)
        
        # Prepare metadata
        metadata = {
            "department": qa_pair.department.value,
            "user_type": qa_pair.user_type.value,
            "question": qa_pair.question,
            "answer": qa_pair.answer,
            "keywords": ",".join(qa_pair.keywords),
            "created_at": qa_pair.created_at.isoformat()
        }
        
        self.collection.add(
            ids=[qa_pair.id],
            embeddings=[embedding],
            documents=[document_text],
            metadatas=[metadata]
        )
    
    def add_qa_pairs_batch(self, qa_pairs: list[QAPair]) -> None:
        """Batch add multiple QA pairs."""
        if not qa_pairs:
            return
        
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        for qa in qa_pairs:
            doc_text = f"Question: {qa.question}\nAnswer: {qa.answer}"
            documents.append(doc_text)
            ids.append(qa.id)
            metadatas.append({
                "department": qa.department.value,
                "user_type": qa.user_type.value,
                "question": qa.question,
                "answer": qa.answer,
                "keywords": ",".join(qa.keywords),
                "created_at": qa.created_at.isoformat()
            })
        
        # Batch embed
        embeddings = self.embedding_provider.embed_texts(documents)
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def query(
        self,
        query_text: str,
        department: Optional[Department] = None,
        user_type: Optional[UserType] = None,
        n_results: int = 10,
        score_threshold: Optional[float] = None
    ) -> list[RetrievedDocument]:
        """
        Query the collection with optional filtering.
        
        Args:
            query_text: User's query
            department: Filter by department
            user_type: Filter by user type
            n_results: Maximum results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of RetrievedDocument objects
        """
        # Build where clause for filtering
        where_clause = None
        if department or user_type:
            conditions = []
            if department and department != Department.UNKNOWN:
                conditions.append({"department": {"$eq": department.value}})
            if user_type and user_type != UserType.UNKNOWN:
                conditions.append({"user_type": {"$eq": user_type.value}})
            
            if len(conditions) == 1:
                where_clause = conditions[0]
            elif len(conditions) > 1:
                where_clause = {"$and": conditions}
        
        # Generate query embedding
        query_embedding = self.embedding_provider.embed_text(query_text)
        
        # Execute query
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert to RetrievedDocument objects
        retrieved_docs = []
        
        if results and results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                # ChromaDB returns distances, convert to similarity
                # For cosine distance: similarity = 1 - distance
                distance = results['distances'][0][i]
                similarity = 1 - distance  # Assumes cosine distance
                
                # Apply threshold filter
                if score_threshold and similarity < score_threshold:
                    continue
                
                metadata = results['metadatas'][0][i]
                
                retrieved_docs.append(RetrievedDocument(
                    id=doc_id,
                    content=results['documents'][0][i],
                    question=metadata.get('question', ''),
                    answer=metadata.get('answer', ''),
                    department=Department(metadata.get('department', 'unknown')),
                    similarity_score=similarity,
                    metadata=metadata
                ))
        
        return retrieved_docs
    
    def get_document_count(self, department: Optional[Department] = None) -> int:
        """Get count of documents, optionally filtered by department."""
        if department:
            results = self.collection.get(
                where={"department": {"$eq": department.value}},
                include=[]
            )
            return len(results['ids'])
        return self.collection.count()
    
    def reset_collection(self) -> None:
        """Delete and recreate collection."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self._collection = None
        _ = self.collection  # Recreate
```

### 4.2 Metadata Schema

| Field | Type | Description | Filterable |
|-------|------|-------------|------------|
| `department` | string | Department enum value (hr, it_support, billing, shipping) | Yes |
| `user_type` | string | User type enum value (internal_employee, external_customer) | Yes |
| `question` | string | Original question text | No |
| `answer` | string | Original answer text | No |
| `keywords` | string | Comma-separated keywords | No (future: full-text) |
| `created_at` | string | ISO timestamp | No |

### 4.3 Embedding Strategy

The document text embedded is a concatenation of question and answer:

```
Question: {question_text}
Answer: {answer_text}
```

This approach captures both the query intent (question) and the knowledge content (answer) in a single embedding, improving retrieval for both question-matching and answer-content similarity.

---

## 5. Classification Pipeline

### 5.1 Sentiment Detection

```python
# src/pipelines/classification.py

from src.providers.base import BaseLLMProvider
from src.models.data_models import SentimentAnalysis, DepartmentClassification, ClassificationResult
from src.models.enums import Department, UserType, Sentiment

SENTIMENT_SYSTEM_PROMPT = """You are a sentiment analysis expert for ShopUNow, a retail company.
Analyze the customer/employee query and determine the sentiment.

Guidelines:
- POSITIVE: Grateful, satisfied, complimentary, or happy tone
- NEUTRAL: Informational questions, standard requests, no emotional indicators
- NEGATIVE: Frustrated, angry, complaining, threatening, or dissatisfied tone

Be sensitive to subtle cues:
- Excessive punctuation (!!!, ???) often indicates frustration
- ALL CAPS often indicates shouting/anger
- Words like "terrible", "awful", "ridiculous", "unacceptable" indicate negative sentiment
- Words like "thanks", "appreciate", "great" indicate positive sentiment
"""

DEPARTMENT_SYSTEM_PROMPT = """You are a query router for ShopUNow, a retail company.
Classify the user query into the appropriate department.

Available Departments:

INTERNAL EMPLOYEE DEPARTMENTS (for company employees):
1. HR (hr): Employee lifecycle - leave requests, payroll questions, benefits, performance reviews, policies, onboarding
2. IT_SUPPORT (it_support): Technical issues - hardware, software, system access, password resets, VPN, email issues

EXTERNAL CUSTOMER DEPARTMENTS (for customers):
3. BILLING (billing): Payment issues - invoices, refunds, payment methods, overcharges, subscription billing
4. SHIPPING (shipping): Delivery issues - order tracking, delivery delays, damaged goods, returns, pickup scheduling

UNKNOWN (unknown): Use only if the query doesn't fit ANY department above.

Classification Rules:
1. Look for keywords indicating department: "my order", "tracking" → shipping; "password", "login" → IT
2. Consider context: "my paycheck" → HR; "my payment" → billing
3. If query mentions both internal (employee) and external (customer) topics, prefer the dominant theme
4. Only use UNKNOWN for completely irrelevant queries (e.g., "What's the weather?")
"""


class ClassificationPipeline:
    """Pipeline for sentiment and department classification."""
    
    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm = llm_provider
    
    def detect_sentiment(self, query: str) -> SentimentAnalysis:
        """Detect sentiment of user query."""
        prompt = f"""Analyze the sentiment of this query:

Query: "{query}"

Classify the sentiment and explain your reasoning."""
        
        return self.llm.generate_structured(
            prompt=prompt,
            response_model=SentimentAnalysis,
            system_prompt=SENTIMENT_SYSTEM_PROMPT,
            temperature=0.0
        )
    
    def classify_department(self, query: str) -> DepartmentClassification:
        """Classify query into appropriate department."""
        prompt = f"""Classify this query into the appropriate department:

Query: "{query}"

Determine the department and user type based on the query content."""
        
        return self.llm.generate_structured(
            prompt=prompt,
            response_model=DepartmentClassification,
            system_prompt=DEPARTMENT_SYSTEM_PROMPT,
            temperature=0.0
        )
    
    def classify(self, query: str) -> ClassificationResult:
        """Full classification pipeline: sentiment + department."""
        sentiment_result = self.detect_sentiment(query)
        department_result = self.classify_department(query)
        
        # Combined confidence is the minimum of both
        combined_confidence = min(
            sentiment_result.confidence,
            department_result.confidence
        )
        
        return ClassificationResult(
            sentiment=sentiment_result.sentiment,
            department=department_result.department,
            user_type=department_result.user_type,
            confidence=combined_confidence,
            reasoning=f"Sentiment: {sentiment_result.indicators}. Department: {department_result.reasoning}"
        )
```

### 5.2 Classification Decision Matrix

| Sentiment | Department | Route |
|-----------|------------|-------|
| Positive | Valid (HR/IT/Billing/Shipping) | RAG Pipeline |
| Neutral | Valid (HR/IT/Billing/Shipping) | RAG Pipeline |
| Negative | Any | Human Escalation |
| Any | Unknown | Human Escalation |

---

## 6. RAG Pipeline

### 6.1 Dynamic K Retrieval Logic

```python
# src/pipelines/retrieval.py

from typing import Optional
from src.vectorstore.chroma_client import ChromaDBClient
from src.models.data_models import RetrievalResult, RetrievedDocument
from src.models.enums import Department, UserType


class DynamicKRetriever:
    """
    Retriever with dynamic k selection based on relevance scores.
    
    Strategy:
    1. Retrieve initial pool (max_k candidates)
    2. Filter by minimum threshold
    3. Apply relevance drop-off detection
    4. Return documents that meet quality criteria
    """
    
    def __init__(
        self,
        chroma_client: ChromaDBClient,
        min_threshold: float = 0.5,
        max_k: int = 10,
        drop_off_ratio: float = 0.7
    ):
        """
        Args:
            chroma_client: ChromaDB client instance
            min_threshold: Minimum similarity score to include
            max_k: Maximum documents to retrieve initially
            drop_off_ratio: If doc[i+1] score < doc[i] score * ratio, stop including
        """
        self.chroma = chroma_client
        self.min_threshold = min_threshold
        self.max_k = max_k
        self.drop_off_ratio = drop_off_ratio
    
    def retrieve(
        self,
        query: str,
        department: Department,
        user_type: Optional[UserType] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant documents with dynamic k selection.
        
        Args:
            query: User query text
            department: Department to filter by
            user_type: Optional user type filter
            
        Returns:
            RetrievalResult with dynamically selected documents
        """
        # Step 1: Get initial candidate pool
        candidates = self.chroma.query(
            query_text=query,
            department=department,
            user_type=user_type,
            n_results=self.max_k,
            score_threshold=None  # We'll filter manually for dynamic k
        )
        
        # Step 2: Apply dynamic k selection
        selected_docs = self._apply_dynamic_k(candidates)
        
        return RetrievalResult(
            query=query,
            department_filter=department,
            documents=selected_docs,
            retrieval_count=len(selected_docs),
            threshold_used=self.min_threshold
        )
    
    def _apply_dynamic_k(
        self,
        candidates: list[RetrievedDocument]
    ) -> list[RetrievedDocument]:
        """
        Apply dynamic k selection algorithm.
        
        Algorithm:
        1. Filter out documents below min_threshold
        2. For remaining docs (sorted by score desc):
           - Always include the first doc if above threshold
           - Include subsequent docs if:
             a) Above min_threshold AND
             b) Score >= previous_score * drop_off_ratio
        """
        if not candidates:
            return []
        
        # Ensure sorted by similarity score descending
        sorted_candidates = sorted(
            candidates,
            key=lambda d: d.similarity_score,
            reverse=True
        )
        
        selected = []
        previous_score = None
        
        for doc in sorted_candidates:
            # Check minimum threshold
            if doc.similarity_score < self.min_threshold:
                break  # Since sorted, no more docs will qualify
            
            # Check drop-off ratio (skip for first doc)
            if previous_score is not None:
                if doc.similarity_score < previous_score * self.drop_off_ratio:
                    break  # Significant quality drop-off detected
            
            selected.append(doc)
            previous_score = doc.similarity_score
        
        return selected


class RetrievalPipeline:
    """Complete retrieval pipeline with fallback handling."""
    
    def __init__(
        self,
        retriever: DynamicKRetriever,
        fallback_message: str = "I couldn't find specific information about that topic."
    ):
        self.retriever = retriever
        self.fallback_message = fallback_message
    
    def retrieve(
        self,
        query: str,
        department: Department,
        user_type: Optional[UserType] = None
    ) -> RetrievalResult:
        """Execute retrieval with logging and fallback handling."""
        result = self.retriever.retrieve(
            query=query,
            department=department,
            user_type=user_type
        )
        
        if not result.has_results:
            # Log for monitoring
            print(f"[RETRIEVAL] No results for query: {query[:50]}... in dept: {department}")
        
        return result
```

### 6.2 Dynamic K Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_threshold` | 0.3 | Minimum similarity score (0-1) to consider |
| `max_k` | 10 | Maximum initial candidates to fetch |
| `drop_off_ratio` | 0.7 | Stop if next score < previous * ratio |

**Example Selection:**
```
Candidates: [0.92, 0.88, 0.85, 0.60, 0.45]
min_threshold=0.5, drop_off_ratio=0.7

0.92 → Include (above threshold, first doc)
0.88 → Include (0.88 >= 0.92 * 0.7 = 0.644)
0.85 → Include (0.85 >= 0.88 * 0.7 = 0.616)
0.60 → STOP (0.60 < 0.85 * 0.7 = 0.595) ← Actually include! 0.60 >= 0.595
0.45 → STOP (below threshold)

Result: [0.92, 0.88, 0.85, 0.60] → 4 documents
```

---

## 7. Routing Logic

### 7.1 Router Implementation

```python
# src/routing/router.py

from src.models.data_models import ClassificationResult, RoutingDecision
from src.models.enums import Department, Sentiment, RouteDecision


class QueryRouter:
    """
    Binary routing logic for query handling.
    
    Routes to HUMAN_ESCALATION if:
    - Sentiment is NEGATIVE
    - Department is UNKNOWN
    
    Routes to RAG_PIPELINE otherwise.
    """
    
    def __init__(self):
        self.valid_departments = Department.valid_departments()
    
    def route(self, classification: ClassificationResult) -> RoutingDecision:
        """
        Determine routing based on classification.
        
        Args:
            classification: Result from classification pipeline
            
        Returns:
            RoutingDecision with route and reasoning
        """
        # Rule 1: Negative sentiment → Escalate
        if classification.sentiment == Sentiment.NEGATIVE:
            return RoutingDecision(
                route=RouteDecision.HUMAN_ESCALATION,
                classification=classification,
                reasoning=f"Query has negative sentiment. Escalating to human agent for better handling."
            )
        
        # Rule 2: Unknown/invalid department → Escalate
        if classification.department not in self.valid_departments:
            return RoutingDecision(
                route=RouteDecision.HUMAN_ESCALATION,
                classification=classification,
                reasoning=f"Query does not match any known department ({classification.department}). Routing to human agent."
            )
        
        # Default: RAG pipeline
        return RoutingDecision(
            route=RouteDecision.RAG_PIPELINE,
            classification=classification,
            reasoning=f"Query classified as {classification.sentiment.value} sentiment for {classification.department.value} department. Processing via RAG."
        )
    
    def should_escalate(self, classification: ClassificationResult) -> bool:
        """Quick check for escalation need."""
        return (
            classification.sentiment == Sentiment.NEGATIVE or
            classification.department not in self.valid_departments
        )
```

### 7.2 Routing Flowchart

```
                    ┌─────────────────┐
                    │  User Query     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Classification │
                    │    Pipeline     │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │ Sentiment =     │           │ Department =    │
    │ NEGATIVE?       │           │ UNKNOWN?        │
    └────────┬────────┘           └────────┬────────┘
             │                             │
        YES  │  NO                    YES  │  NO
             │   │                         │   │
             ▼   │                         ▼   │
    ┌────────────┴──┐             ┌────────┴───┐
    │    HUMAN      │             │            │
    │  ESCALATION   │◄────────────┤            │
    └───────────────┘             │            │
                                  │            │
                                  │            ▼
                                  │   ┌─────────────────┐
                                  │   │  RAG PIPELINE   │
                                  │   │  (Filtered by   │
                                  │   │   Department)   │
                                  │   └─────────────────┘
                                  │
                                  └────────────────────────
```

---

## 8. Orchestration Layer

### 8.1 Main Orchestrator

```python
# src/orchestrator.py

import time
from typing import Optional

from src.config import Settings
from src.models.api_models import QueryRequest, QueryResponse, EscalationResponse
from src.models.data_models import (
    ClassificationResult, 
    RoutingDecision, 
    RetrievalResult,
    GeneratedResponse,
    SourceAttribution
)
from src.models.enums import RouteDecision, Department
from src.providers.llm_provider import LLMProviderFactory, LLMProviderType
from src.providers.embedding_provider import EmbeddingProviderFactory, EmbeddingProviderType
from src.pipelines.classification import ClassificationPipeline
from src.pipelines.retrieval import RetrievalPipeline, DynamicKRetriever
from src.routing.router import QueryRouter
from src.vectorstore.chroma_client import ChromaDBClient


RESPONSE_GENERATION_PROMPT = """You are a helpful customer service assistant for ShopUNow.
Based on the retrieved information, provide a clear and helpful answer to the user's question.

Guidelines:
- Be concise but thorough
- Use a friendly, professional tone
- If the retrieved information doesn't fully answer the question, acknowledge this
- Don't make up information not present in the sources
- For employees: be supportive and solution-oriented
- For customers: be empathetic and action-oriented
"""


class ShopUNowOrchestrator:
    """
    Main orchestrator coordinating all pipeline components.
    
    Flow:
    1. Receive query
    2. Classify (sentiment + department)
    3. Route (RAG vs Human Escalation)
    4. If RAG: Retrieve → Generate Response
    5. If Escalation: Return escalation message
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        
        # Initialize providers
        self.llm = LLMProviderFactory.create(
            provider=LLMProviderType(self.settings.llm_provider),
            model=self.settings.llm_model
        )
        
        self.embedding = EmbeddingProviderFactory.create(
            provider=EmbeddingProviderType(self.settings.embedding_provider),
            model=self.settings.embedding_model
        )
        
        # Initialize ChromaDB
        self.chroma = ChromaDBClient(
            persist_directory=self.settings.chroma_persist_dir,
            embedding_provider=self.embedding
        )
        
        # Initialize pipelines
        self.classifier = ClassificationPipeline(self.llm)
        self.retriever = DynamicKRetriever(
            chroma_client=self.chroma,
            min_threshold=self.settings.retrieval_min_threshold,
            max_k=self.settings.retrieval_max_k,
            drop_off_ratio=self.settings.retrieval_drop_off_ratio
        )
        self.retrieval_pipeline = RetrievalPipeline(self.retriever)
        self.router = QueryRouter()
    
    def process_query(self, request: QueryRequest) -> QueryResponse | EscalationResponse:
        """
        Process user query through the complete pipeline.
        
        Args:
            request: Incoming query request
            
        Returns:
            QueryResponse for successful RAG processing
            EscalationResponse for escalated queries
        """
        start_time = time.time()
        
        # Step 1: Classification
        classification = self.classifier.classify(request.query)
        
        # Step 2: Routing
        routing_decision = self.router.route(classification)
        
        # Step 3: Handle based on route
        if routing_decision.should_escalate:
            return self._handle_escalation(request, classification, routing_decision)
        
        # Step 4: RAG Pipeline
        retrieval_result = self.retrieval_pipeline.retrieve(
            query=request.query,
            department=classification.department,
            user_type=classification.user_type
        )
        
        # Step 5: Generate Response
        response = self._generate_response(
            query=request.query,
            classification=classification,
            retrieval=retrieval_result
        )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return QueryResponse(
            query=request.query,
            answer=response.answer,
            department=classification.department,
            sentiment=classification.sentiment,
            sources=response.sources,
            confidence=response.confidence,
            was_escalated=False,
            processing_time_ms=processing_time
        )
    
    def _handle_escalation(
        self,
        request: QueryRequest,
        classification: ClassificationResult,
        routing: RoutingDecision
    ) -> EscalationResponse:
        """Handle escalation to human support."""
        return EscalationResponse(
            query=request.query,
            reason=routing.reasoning,
            ticket_id=f"ESC-{int(time.time())}"  # Simple ticket ID
        )
    
    def _generate_response(
        self,
        query: str,
        classification: ClassificationResult,
        retrieval: RetrievalResult
    ) -> GeneratedResponse:
        """Generate final response using retrieved context."""
        
        if not retrieval.has_results:
            return GeneratedResponse(
                answer="I couldn't find specific information to answer your question. "
                       "Please try rephrasing or contact support for assistance.",
                sources=[],
                confidence=0.3,
                department=classification.department
            )
        
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieval.documents, 1):
            context_parts.append(
                f"[Source {i}]\n"
                f"Q: {doc.question}\n"
                f"A: {doc.answer}\n"
                f"Relevance: {doc.similarity_score:.2f}"
            )
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""User Query: {query}

Retrieved Information:
{context}

Based on the above information, provide a helpful response to the user's query.
If the information doesn't fully address the query, acknowledge what you can answer and what may need further clarification."""
        
        llm_response = self.llm.generate(
            prompt=prompt,
            system_prompt=RESPONSE_GENERATION_PROMPT,
            temperature=0.3,
            max_tokens=512
        )
        
        # Build source attributions
        sources = [
            SourceAttribution(
                document_id=doc.id,
                question_matched=doc.question,
                relevance_score=doc.similarity_score
            )
            for doc in retrieval.documents
        ]
        
        # Calculate confidence based on retrieval quality
        avg_score = sum(d.similarity_score for d in retrieval.documents) / len(retrieval.documents)
        confidence = min(avg_score, classification.confidence)
        
        return GeneratedResponse(
            answer=llm_response.content,
            sources=sources,
            confidence=confidence,
            department=classification.department
        )
```

---

## 9. FastAPI Specifications

### 9.1 Application Structure

```python
# src/main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import time

from src.config import Settings
from src.models.api_models import (
    QueryRequest,
    QueryResponse,
    EscalationResponse,
    HealthCheckRequest,
    HealthCheckResponse,
    ErrorResponse
)
from src.orchestrator import ShopUNowOrchestrator


# Global orchestrator instance
orchestrator: ShopUNowOrchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    global orchestrator
    
    # Startup
    settings = Settings()
    orchestrator = ShopUNowOrchestrator(settings)
    print(f"[STARTUP] ShopUNow Assistant initialized")
    print(f"[STARTUP] LLM: {settings.llm_provider}/{settings.llm_model}")
    print(f"[STARTUP] Embeddings: {settings.embedding_provider}/{settings.embedding_model}")
    
    yield
    
    # Shutdown
    print("[SHUTDOWN] Cleaning up...")


app = FastAPI(
    title="ShopUNow AI Assistant",
    description="Intelligent AI Assistant for ShopUNow retail company",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/query",
    response_model=QueryResponse | EscalationResponse,
    responses={
        200: {"description": "Successful response"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["Assistant"]
)
async def process_query(request: QueryRequest):
    """
    Process a user query through the AI assistant.
    
    The assistant will:
    1. Analyze sentiment and classify the query
    2. Route to appropriate department or human escalation
    3. Retrieve relevant information and generate a response
    """
    try:
        response = orchestrator.process_query(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["System"]
)
async def health_check(deep: bool = False):
    """
    Health check endpoint.
    
    Args:
        deep: If true, verify all component connections
    """
    components = {"api": "healthy"}
    
    if deep:
        try:
            # Check ChromaDB
            doc_count = orchestrator.chroma.get_document_count()
            components["chromadb"] = f"healthy ({doc_count} docs)"
        except Exception as e:
            components["chromadb"] = f"unhealthy: {str(e)}"
        
        try:
            # Check LLM
            test_response = orchestrator.llm.generate(
                prompt="Say 'OK'",
                max_tokens=5
            )
            components["llm"] = f"healthy ({orchestrator.llm.provider_name})"
        except Exception as e:
            components["llm"] = f"unhealthy: {str(e)}"
    
    overall_status = "healthy" if all(
        "healthy" in v for v in components.values()
    ) else "degraded"
    
    return HealthCheckResponse(
        status=overall_status,
        version="1.0.0",
        components=components
    )


@app.get(
    "/departments",
    tags=["Information"]
)
async def list_departments():
    """List all available departments and their descriptions."""
    from src.models.enums import Department
    
    department_info = {
        "hr": {
            "name": "Human Resources",
            "user_type": "internal_employee",
            "description": "Employee lifecycle: leave, payroll, benefits, performance, policies"
        },
        "it_support": {
            "name": "IT Support",
            "user_type": "internal_employee", 
            "description": "Technical issues: hardware, software, system access, passwords"
        },
        "billing": {
            "name": "Billing & Payments",
            "user_type": "external_customer",
            "description": "Payment issues: invoices, refunds, payment methods, overcharges"
        },
        "shipping": {
            "name": "Shipping & Delivery",
            "user_type": "external_customer",
            "description": "Delivery issues: order tracking, delays, damaged goods, returns"
        }
    }
    
    return {"departments": department_info}
```

### 9.2 API Endpoint Summary

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/query` | Process user query | None (future: API key) |
| GET | `/health` | Health check | None |
| GET | `/health?deep=true` | Deep health check | None |
| GET | `/departments` | List departments | None |
| GET | `/docs` | Swagger UI (auto) | None |
| GET | `/redoc` | ReDoc (auto) | None |

### 9.3 Example Request/Response

**Request:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I reset my password?"}'
```

**Response (RAG):**
```json
{
  "query": "How do I reset my password?",
  "answer": "To reset your password, go to the IT portal at it.shopunow.com and click 'Forgot Password'. You'll receive a reset link via email. If you need immediate assistance, contact the IT helpdesk at ext. 5555.",
  "department": "it_support",
  "sentiment": "neutral",
  "sources": [
    {
      "document_id": "it_003",
      "question_matched": "What is the password reset process?",
      "relevance_score": 0.91
    }
  ],
  "confidence": 0.87,
  "was_escalated": false,
  "processing_time_ms": 1423.5,
  "timestamp": "2025-01-01T12:30:45.123Z"
}
```

**Response (Escalation):**
```json
{
  "query": "This is ridiculous! I've been waiting for my refund for 3 weeks!!!",
  "message": "Your query has been escalated to a human support agent. A representative will contact you shortly.",
  "reason": "Query has negative sentiment. Escalating to human agent for better handling.",
  "ticket_id": "ESC-1735737045",
  "estimated_response_time": "24-48 hours",
  "timestamp": "2025-01-01T12:30:45.123Z"
}
```

---

## 10. Configuration Management

### 10.1 Settings Class

```python
# src/config.py

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application configuration with environment variable support."""
    
    # LLM Configuration
    llm_provider: str = Field(default="openai", description="LLM provider: openai, gemini, groq")
    llm_model: Optional[str] = Field(default=None, description="Specific model (uses provider default if None)")
    
    # Embedding Configuration
    embedding_provider: str = Field(default="sentence_transformers", description="Embedding provider")
    embedding_model: Optional[str] = Field(default=None, description="Specific embedding model")
    
    # API Keys (loaded from environment)
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
    debug: bool = Field(default=False)
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }
```

### 10.2 Environment File Template

```bash
# .env.example

# === LLM Configuration ===
LLM_PROVIDER=openai          # Options: openai, gemini, groq
LLM_MODEL=gpt-4o-mini        # Leave empty for provider default

# === Embedding Configuration ===
EMBEDDING_PROVIDER=sentence_transformers  # Options: sentence_transformers, openai, cohere, google
EMBEDDING_MODEL=all-MiniLM-L6-v2          # Leave empty for provider default

# === API Keys ===
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
GROQ_API_KEY=gsk_...
COHERE_API_KEY=...

# === ChromaDB ===
CHROMA_PERSIST_DIR=./data/chroma_db

# === Retrieval Settings ===
RETRIEVAL_MIN_THRESHOLD=0.3
RETRIEVAL_MAX_K=10
RETRIEVAL_DROP_OFF_RATIO=0.7

# === API Settings ===
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false
```

---

## 11. Error Handling Strategy

### 11.1 Exception Hierarchy

```python
# src/exceptions.py

class ShopUNowException(Exception):
    """Base exception for ShopUNow application."""
    pass


class ClassificationError(ShopUNowException):
    """Error during query classification."""
    pass


class RetrievalError(ShopUNowException):
    """Error during document retrieval."""
    pass


class GenerationError(ShopUNowException):
    """Error during response generation."""
    pass


class ProviderError(ShopUNowException):
    """Error from LLM or embedding provider."""
    pass


class ConfigurationError(ShopUNowException):
    """Configuration or initialization error."""
    pass
```

### 11.2 Error Handling Patterns

```python
# In orchestrator methods:

def process_query(self, request: QueryRequest) -> QueryResponse | EscalationResponse:
    try:
        classification = self.classifier.classify(request.query)
    except Exception as e:
        # Log error
        logger.error(f"Classification failed: {e}")
        # Fallback: escalate to human
        return EscalationResponse(
            query=request.query,
            reason="Unable to process query automatically. Routing to human support.",
            ticket_id=f"ERR-{int(time.time())}"
        )
    
    # ... rest of pipeline
```

### 11.3 Graceful Degradation

| Failure Point | Fallback Behavior |
|---------------|-------------------|
| Classification fails | Escalate to human |
| Retrieval returns empty | Return "no information found" message |
| LLM generation fails | Return retrieved FAQ answer directly |
| ChromaDB unavailable | Return service unavailable error |

---

## 12. Testing Strategy

### 12.1 Unit Test Structure

```python
# tests/test_classification.py

import pytest
from src.pipelines.classification import ClassificationPipeline
from src.models.enums import Sentiment, Department, UserType
from unittest.mock import Mock, MagicMock


class TestSentimentDetection:
    """Test sentiment detection accuracy."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM provider."""
        mock = MagicMock()
        return mock
    
    @pytest.fixture
    def pipeline(self, mock_llm):
        return ClassificationPipeline(mock_llm)
    
    def test_negative_sentiment_detection(self, pipeline, mock_llm):
        """Test detection of negative sentiment."""
        mock_llm.generate_structured.return_value = SentimentAnalysis(
            sentiment=Sentiment.NEGATIVE,
            confidence=0.95,
            indicators=["frustrated", "unacceptable"]
        )
        
        result = pipeline.detect_sentiment(
            "This is unacceptable! I'm extremely frustrated!"
        )
        
        assert result.sentiment == Sentiment.NEGATIVE
        assert result.confidence > 0.8
    
    def test_neutral_sentiment_detection(self, pipeline, mock_llm):
        """Test detection of neutral sentiment."""
        mock_llm.generate_structured.return_value = SentimentAnalysis(
            sentiment=Sentiment.NEUTRAL,
            confidence=0.9,
            indicators=[]
        )
        
        result = pipeline.detect_sentiment(
            "How do I reset my password?"
        )
        
        assert result.sentiment == Sentiment.NEUTRAL


class TestDepartmentClassification:
    """Test department classification accuracy."""
    
    @pytest.mark.parametrize("query,expected_dept,expected_user_type", [
        ("How do I apply for PTO?", Department.HR, UserType.INTERNAL_EMPLOYEE),
        ("I can't login to my computer", Department.IT_SUPPORT, UserType.INTERNAL_EMPLOYEE),
        ("Where is my order?", Department.SHIPPING, UserType.EXTERNAL_CUSTOMER),
        ("I need a refund", Department.BILLING, UserType.EXTERNAL_CUSTOMER),
    ])
    def test_department_routing(self, query, expected_dept, expected_user_type):
        """Test correct department identification."""
        # Integration test with real LLM would go here
        pass
```

### 12.2 Integration Test Structure

```python
# tests/test_integration.py

import pytest
from src.orchestrator import ShopUNowOrchestrator
from src.models.api_models import QueryRequest, QueryResponse, EscalationResponse
from src.config import Settings


class TestEndToEndFlow:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with test configuration."""
        settings = Settings(
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            embedding_provider="sentence_transformers",
            chroma_persist_dir="./data/test_chroma_db"
        )
        return ShopUNowOrchestrator(settings)
    
    def test_hr_query_returns_response(self, orchestrator):
        """Test HR query returns valid response."""
        request = QueryRequest(query="How do I apply for annual leave?")
        response = orchestrator.process_query(request)
        
        assert isinstance(response, QueryResponse)
        assert response.department.value == "hr"
        assert not response.was_escalated
        assert len(response.answer) > 0
    
    def test_negative_sentiment_escalates(self, orchestrator):
        """Test negative sentiment triggers escalation."""
        request = QueryRequest(
            query="This is terrible! Nobody has helped me for weeks!"
        )
        response = orchestrator.process_query(request)
        
        assert isinstance(response, EscalationResponse)
        assert "negative sentiment" in response.reason.lower()
```

### 12.3 Test Data Fixtures

```python
# tests/fixtures.py

TEST_QUERIES = {
    "hr_neutral": [
        "How do I apply for PTO?",
        "When is payroll processed?",
        "What are the company holidays?",
    ],
    "it_neutral": [
        "How do I reset my password?",
        "I need VPN access",
        "My email isn't working",
    ],
    "billing_neutral": [
        "I need an invoice copy",
        "How can I update my payment method?",
        "What payment options do you accept?",
    ],
    "shipping_neutral": [
        "Where is my order?",
        "How do I return an item?",
        "When will my package arrive?",
    ],
    "negative_sentiment": [
        "This is ridiculous! Nobody can help me!",
        "I've been waiting forever and this is unacceptable!!!",
        "Your service is terrible, I want to speak to a manager",
    ],
    "unknown_department": [
        "What's the weather like today?",
        "Can you tell me a joke?",
        "What's the capital of France?",
    ]
}
```

---

## Appendix A: Dependencies

```txt
# requirements.txt

# Core
pydantic>=2.0.0
pydantic-settings>=2.0.0

# LLM Providers
openai>=1.0.0
google-generativeai>=0.3.0
groq>=0.4.0

# Embeddings
sentence-transformers>=2.2.0
cohere>=4.0.0

# Vector Store
chromadb>=0.4.0

# API
fastapi>=0.100.0
uvicorn>=0.23.0

# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
httpx>=0.24.0  # For async test client

# Utilities
python-dotenv>=1.0.0
```

---

## Appendix B: Quick Reference

### Provider Selection Matrix

| Use Case | LLM Provider | Embedding Provider |
|----------|--------------|-------------------|
| Cost-optimized | OpenAI gpt-4o-mini | Sentence Transformers |
| Quality-optimized | OpenAI gpt-4o | OpenAI text-embedding-3-large |
| Speed-optimized | Groq llama-3.3-70b | Sentence Transformers |
| Privacy-focused | Groq (no data retention) | Sentence Transformers (local) |

### Threshold Tuning Guide

| Scenario | min_threshold | drop_off_ratio |
|----------|---------------|----------------|
| High precision (few but accurate) | 0.7 | 0.8 |
| Balanced | 0.5 | 0.7 |
| **Default (MiniLM optimized)** | **0.3** | **0.7** |
| High recall (more results) | 0.2 | 0.6 |

> **Note:** The default threshold of 0.3 is optimized for Sentence Transformers (all-MiniLM-L6-v2).
> MiniLM embeddings typically produce similarity scores in the 0.3-0.6 range for relevant matches.
> If using OpenAI embeddings (higher dimensional), consider increasing to 0.5-0.6.

---

*End of TECH_SPECS.md*
