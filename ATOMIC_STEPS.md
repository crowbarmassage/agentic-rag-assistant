# ATOMIC_STEPS.md — ShopUNow Capstone Implementation Roadmap

## Document Metadata
- **Project**: Agentic AI Assistant for ShopUNow (Retail Company)
- **Total Steps**: 38 steps across 10 phases
- **Estimated Duration**: 15-20 hours
- **Prerequisites**: Python 3.10+, API keys (OpenAI minimum), basic familiarity with Pydantic/FastAPI

---

## Phase Overview

| Phase | Name | Steps | Focus |
|-------|------|-------|-------|
| 1 | Project Scaffolding | 1-4 | Directory structure, dependencies, configuration |
| 2 | Enums & Core Models | 5-8 | Pydantic models, enumerations |
| 3 | Provider Abstraction | 9-14 | LLM and embedding provider layers |
| 4 | Synthetic Data Generation | 15-18 | Generate 60 QA pairs across 4 departments |
| 5 | Vector Store Setup | 19-22 | ChromaDB client, ingestion pipeline |
| 6 | Classification Pipeline | 23-26 | Sentiment detection, department classification |
| 7 | Retrieval Pipeline | 27-30 | Dynamic K retriever, retrieval pipeline |
| 8 | Routing & Orchestration | 31-34 | Router, orchestrator integration |
| 9 | FastAPI Integration | 35-37 | API endpoints, health checks |
| 10 | Testing & Notebook Assembly | 38 | Unit tests, final notebook |

---

## Phase 1: Project Scaffolding (Steps 1-4)

### Step 1: Create Directory Structure
**Action**: Create the complete project directory structure.

```bash
mkdir -p shopunow_assistant/{src/{models,providers,pipelines,routing,vectorstore,utils},data/{raw,chroma_db},tests,notebooks}
touch shopunow_assistant/src/__init__.py
touch shopunow_assistant/src/models/__init__.py
touch shopunow_assistant/src/providers/__init__.py
touch shopunow_assistant/src/pipelines/__init__.py
touch shopunow_assistant/src/routing/__init__.py
touch shopunow_assistant/src/vectorstore/__init__.py
touch shopunow_assistant/src/utils/__init__.py
touch shopunow_assistant/tests/__init__.py
```

**Validation Checkpoint**:
```bash
# Run from project root
find shopunow_assistant -type d | head -20
# Should show all directories created
```

---

### Step 2: Create requirements.txt
**Action**: Create dependency file with all required packages.

**File**: `shopunow_assistant/requirements.txt`

```txt
# Core
pydantic>=2.5.0
pydantic-settings>=2.1.0

# LLM Providers
openai>=1.12.0
google-generativeai>=0.4.0
groq>=0.4.2

# Embeddings
sentence-transformers>=2.3.0
cohere>=4.47

# Vector Store
chromadb>=0.4.22

# API
fastapi>=0.109.0
uvicorn[standard]>=0.27.0

# Testing
pytest>=8.0.0
pytest-asyncio>=0.23.0
httpx>=0.26.0

# Utilities
python-dotenv>=1.0.0
```

**Validation Checkpoint**:
```bash
cd shopunow_assistant
pip install -r requirements.txt
python -c "import pydantic; import chromadb; import fastapi; print('Dependencies OK')"
```

---

### Step 3: Create Environment Configuration
**Action**: Create `.env.example` template and actual `.env` file.

**File**: `shopunow_assistant/.env.example`

```bash
# LLM Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# Embedding Configuration
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2

# API Keys
OPENAI_API_KEY=sk-your-key-here
GOOGLE_API_KEY=
GROQ_API_KEY=
COHERE_API_KEY=

# ChromaDB
CHROMA_PERSIST_DIR=./data/chroma_db

# Retrieval Settings
RETRIEVAL_MIN_THRESHOLD=0.5
RETRIEVAL_MAX_K=10
RETRIEVAL_DROP_OFF_RATIO=0.7

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true
```

**Action**: Copy to `.env` and fill in your actual API keys.

```bash
cp .env.example .env
# Edit .env with your actual OPENAI_API_KEY
```

**Validation Checkpoint**:
```bash
cat .env | grep -v "^#" | grep -v "^$" | wc -l
# Should show ~12 configuration lines
```

---

### Step 4: Create Configuration Module
**Action**: Create Pydantic Settings class for configuration management.

**File**: `shopunow_assistant/src/config.py`

```python
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
    retrieval_min_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
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


def get_settings() -> Settings:
    """Factory function to get settings instance."""
    return Settings()
```

**Validation Checkpoint**:
```python
# Run from shopunow_assistant directory
from src.config import get_settings

settings = get_settings()
print(f"LLM Provider: {settings.llm_provider}")
print(f"OpenAI Key Set: {settings.openai_api_key is not None}")
print(f"ChromaDB Dir: {settings.chroma_persist_dir}")
# Should print configuration values
```

---

## Phase 2: Enums & Core Models (Steps 5-8)

### Step 5: Create Enumerations
**Action**: Define all enum types used throughout the application.

**File**: `shopunow_assistant/src/models/enums.py`

```python
"""Enumeration types for ShopUNow Assistant."""

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
    
    @classmethod
    def get_description(cls, dept: "Department") -> str:
        """Get human-readable description for department."""
        descriptions = {
            cls.HR: "Human Resources - employee lifecycle, leave, payroll, benefits, policies",
            cls.IT_SUPPORT: "IT Support - technical issues, hardware, software, system access",
            cls.BILLING: "Billing & Payments - invoices, refunds, payment methods, overcharges",
            cls.SHIPPING: "Shipping & Delivery - order tracking, delays, returns, damaged goods",
            cls.UNKNOWN: "Unknown department"
        }
        return descriptions.get(dept, "Unknown")


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

**Validation Checkpoint**:
```python
from src.models.enums import Department, Sentiment, UserType

print(f"Valid departments: {[d.value for d in Department.valid_departments()]}")
print(f"Internal depts: {[d.value for d in Department.internal_departments()]}")
print(f"HR description: {Department.get_description(Department.HR)}")
# Should list departments correctly
```

---

### Step 6: Create QA Data Models
**Action**: Define Pydantic models for FAQ data structures.

**File**: `shopunow_assistant/src/models/data_models.py`

```python
"""Core data models for ShopUNow Assistant."""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

from .enums import Department, UserType, Sentiment, RouteDecision


# ==================== QA Data Models ====================

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
            return [kw.lower().strip() for kw in v if kw]
        return v
    
    def to_document_text(self) -> str:
        """Convert to document text for embedding."""
        return f"Question: {self.question}\nAnswer: {self.answer}"


class DepartmentFAQs(BaseModel):
    """Collection of FAQs for a single department."""
    
    department: Department
    user_type: UserType
    description: str = Field(..., description="Department description")
    qa_pairs: list[QAPair] = Field(..., min_length=1)
    
    @property
    def count(self) -> int:
        return len(self.qa_pairs)


# ==================== Classification Models ====================

class SentimentAnalysis(BaseModel):
    """Structured output for sentiment detection."""
    
    sentiment: Sentiment
    confidence: float = Field(..., ge=0.0, le=1.0)
    indicators: list[str] = Field(
        default_factory=list,
        description="Key phrases that influenced sentiment detection"
    )


class DepartmentClassification(BaseModel):
    """Structured output for department classification."""
    
    department: Department
    user_type: UserType
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Why this department was selected")


class ClassificationResult(BaseModel):
    """Combined result of query classification pipeline."""
    
    sentiment: Sentiment
    department: Department
    user_type: UserType
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str


# ==================== Retrieval Models ====================

class RetrievedDocument(BaseModel):
    """Single document retrieved from vector store."""
    
    id: str
    content: str = Field(..., description="Full document text")
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


# ==================== Response Models ====================

class SourceAttribution(BaseModel):
    """Source attribution for response transparency."""
    
    document_id: str
    question_matched: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class GeneratedResponse(BaseModel):
    """Final generated response to user query."""
    
    answer: str
    sources: list[SourceAttribution] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    department: Department
    is_escalated: bool = Field(default=False)
    escalation_reason: Optional[str] = None


# ==================== Routing Models ====================

class RoutingDecision(BaseModel):
    """Complete routing decision with context."""
    
    route: RouteDecision
    classification: ClassificationResult
    reasoning: str
    
    @property
    def should_escalate(self) -> bool:
        return self.route == RouteDecision.HUMAN_ESCALATION
```

**Validation Checkpoint**:
```python
from src.models.data_models import QAPair, ClassificationResult
from src.models.enums import Department, UserType, Sentiment

# Test QAPair creation
qa = QAPair(
    id="hr_001",
    question="How do I apply for paid time off?",
    answer="You can apply for PTO through the HR portal under 'Time Off Requests'.",
    department=Department.HR,
    user_type=UserType.INTERNAL_EMPLOYEE,
    keywords=["PTO", "Leave", "Time Off"]
)
print(f"QA ID: {qa.id}")
print(f"Document text: {qa.to_document_text()[:50]}...")
print(f"Keywords (lowercased): {qa.keywords}")
# Keywords should be lowercase
```

---

### Step 7: Create API Request/Response Models
**Action**: Define FastAPI request and response schemas.

**File**: `shopunow_assistant/src/models/api_models.py`

```python
"""API request and response models."""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

from .enums import Department, Sentiment
from .data_models import SourceAttribution


# ==================== Request Models ====================

class QueryRequest(BaseModel):
    """Incoming user query request."""
    
    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="User's question"
    )
    user_id: Optional[str] = Field(
        None,
        description="Optional user identifier"
    )
    session_id: Optional[str] = Field(
        None,
        description="Optional session identifier"
    )
    
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


# ==================== Response Models ====================

class QueryResponse(BaseModel):
    """Successful response to user query."""
    
    query: str
    answer: str
    department: Department
    sentiment: Sentiment
    sources: list[SourceAttribution] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    was_escalated: bool = Field(default=False)
    escalation_message: Optional[str] = None
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


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
    components: dict[str, str] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Standardized error response."""
    
    error: str
    message: str
    details: Optional[dict] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

**Validation Checkpoint**:
```python
from src.models.api_models import QueryRequest, QueryResponse, EscalationResponse

# Test request validation
req = QueryRequest(query="How do I reset my password?")
print(f"Valid request: {req.query}")

# Test response creation
from src.models.enums import Department, Sentiment
resp = QueryResponse(
    query="test",
    answer="Test answer",
    department=Department.IT_SUPPORT,
    sentiment=Sentiment.NEUTRAL,
    confidence=0.85,
    processing_time_ms=1234.5
)
print(f"Response department: {resp.department.value}")
```

---

### Step 8: Update Models __init__.py
**Action**: Export all models from the models package.

**File**: `shopunow_assistant/src/models/__init__.py`

```python
"""Data models package."""

from .enums import Department, UserType, Sentiment, RouteDecision
from .data_models import (
    QAPair,
    DepartmentFAQs,
    SentimentAnalysis,
    DepartmentClassification,
    ClassificationResult,
    RetrievedDocument,
    RetrievalResult,
    SourceAttribution,
    GeneratedResponse,
    RoutingDecision,
)
from .api_models import (
    QueryRequest,
    QueryResponse,
    EscalationResponse,
    HealthCheckResponse,
    ErrorResponse,
)

__all__ = [
    # Enums
    "Department",
    "UserType", 
    "Sentiment",
    "RouteDecision",
    # Data models
    "QAPair",
    "DepartmentFAQs",
    "SentimentAnalysis",
    "DepartmentClassification",
    "ClassificationResult",
    "RetrievedDocument",
    "RetrievalResult",
    "SourceAttribution",
    "GeneratedResponse",
    "RoutingDecision",
    # API models
    "QueryRequest",
    "QueryResponse",
    "EscalationResponse",
    "HealthCheckResponse",
    "ErrorResponse",
]
```

**Validation Checkpoint**:
```python
from src.models import Department, QAPair, QueryRequest, Sentiment
print(f"All models imported successfully")
print(f"Department.HR = {Department.HR.value}")
```

---

## Phase 3: Provider Abstraction (Steps 9-14)

### Step 9: Create Base Provider Interfaces
**Action**: Define abstract base classes for LLM and embedding providers.

**File**: `shopunow_assistant/src/providers/base.py`

```python
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
```

**Validation Checkpoint**:
```python
from src.providers.base import BaseLLMProvider, BaseEmbeddingProvider, LLMResponse
print("Base classes defined successfully")
# These are abstract, so we can't instantiate them directly
```

---

### Step 10: Implement OpenAI LLM Provider
**Action**: Create OpenAI provider implementation.

**File**: `shopunow_assistant/src/providers/llm_openai.py`

```python
"""OpenAI LLM provider implementation."""

import json
from typing import Optional, Type
from openai import OpenAI
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
    DEFAULT_MODEL = "gpt-4o-mini"
    
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or self.DEFAULT_MODEL
        if self.model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {self.model} not supported. Choose from: {self.SUPPORTED_MODELS}")
        self.client = OpenAI(api_key=api_key)  # Uses OPENAI_API_KEY env var if None
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """Generate text completion."""
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
        """Generate structured output using JSON mode."""
        messages = []
        
        # Build schema instruction
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        schema_instruction = f"""
You must respond with valid JSON that conforms to this schema:
{schema_json}

Respond ONLY with the JSON object, no additional text or markdown formatting.
"""
        
        effective_system = ((system_prompt or "") + "\n\n" + schema_instruction).strip()
        messages.append({"role": "system", "content": effective_system})
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
```

**Validation Checkpoint**:
```python
from src.providers.llm_openai import OpenAIProvider

# This requires OPENAI_API_KEY to be set
provider = OpenAIProvider()
response = provider.generate("Say 'Hello World'", max_tokens=10)
print(f"Response: {response.content}")
print(f"Model: {response.model}")
print(f"Tokens: {response.usage}")
```

---

### Step 11: Implement Gemini LLM Provider
**Action**: Create Google Gemini provider implementation.

**File**: `shopunow_assistant/src/providers/llm_gemini.py`

```python
"""Google Gemini LLM provider implementation."""

import json
from typing import Optional, Type
import google.generativeai as genai
from pydantic import BaseModel

from .base import BaseLLMProvider, LLMResponse


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider implementation."""
    
    SUPPORTED_MODELS = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-pro"
    ]
    DEFAULT_MODEL = "gemini-1.5-flash"
    
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model_name = model or self.DEFAULT_MODEL
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {self.model_name} not supported. Choose from: {self.SUPPORTED_MODELS}")
        
        if api_key:
            genai.configure(api_key=api_key)
        # Otherwise uses GOOGLE_API_KEY env var
        
        self.model = genai.GenerativeModel(self.model_name)
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """Generate text completion."""
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
        
        # Handle usage metadata safely
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
            }
        
        return LLMResponse(
            content=response.text,
            model=self.model_name,
            usage=usage,
            raw_response=response
        )
    
    def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0
    ) -> BaseModel:
        """Generate structured output."""
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        schema_instruction = f"""
Respond with valid JSON conforming to this schema:
{schema_json}

Output ONLY the JSON object, no markdown or additional text.
"""
        full_prompt = f"{system_prompt or ''}\n\n{schema_instruction}\n\n{prompt}".strip()
        
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
```

**Validation Checkpoint** (requires GOOGLE_API_KEY):
```python
from src.providers.llm_gemini import GeminiProvider

provider = GeminiProvider()
response = provider.generate("Say 'Hello World'", max_tokens=10)
print(f"Response: {response.content}")
```

---

### Step 12: Implement Groq LLM Provider
**Action**: Create Groq provider implementation.

**File**: `shopunow_assistant/src/providers/llm_groq.py`

```python
"""Groq LLM provider implementation."""

import json
from typing import Optional, Type
from groq import Groq
from pydantic import BaseModel

from .base import BaseLLMProvider, LLMResponse


class GroqProvider(BaseLLMProvider):
    """Groq API provider implementation."""
    
    SUPPORTED_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768"
    ]
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or self.DEFAULT_MODEL
        if self.model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {self.model} not supported. Choose from: {self.SUPPORTED_MODELS}")
        self.client = Groq(api_key=api_key)  # Uses GROQ_API_KEY env var if None
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """Generate text completion."""
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
        """Generate structured output using JSON mode."""
        messages = []
        
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        schema_instruction = f"""
Respond with valid JSON conforming to this schema:
{schema_json}

Output ONLY the JSON object, no markdown formatting.
"""
        
        effective_system = ((system_prompt or "") + "\n\n" + schema_instruction).strip()
        messages.append({"role": "system", "content": effective_system})
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

**Validation Checkpoint** (requires GROQ_API_KEY):
```python
from src.providers.llm_groq import GroqProvider

provider = GroqProvider()
response = provider.generate("Say 'Hello World'", max_tokens=10)
print(f"Response: {response.content}")
```

---

### Step 13: Create LLM Provider Factory
**Action**: Create factory class for LLM provider instantiation.

**File**: `shopunow_assistant/src/providers/llm_factory.py`

```python
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
```

**Validation Checkpoint**:
```python
from src.providers.llm_factory import LLMProviderFactory, LLMProviderType

# List available providers
print(f"Available providers: {LLMProviderFactory.list_providers()}")

# Create provider using factory
llm = LLMProviderFactory.create("openai")
print(f"Created {llm.provider_name} provider")

# Test generate
response = llm.generate("What is 2+2?", max_tokens=20)
print(f"Response: {response.content}")
```

---

### Step 14: Implement Embedding Providers
**Action**: Create all embedding provider implementations and factory.

**File**: `shopunow_assistant/src/providers/embedding_providers.py`

```python
"""Embedding provider implementations and factory."""

from abc import ABC
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
```

**Validation Checkpoint**:
```python
from src.providers.embedding_providers import EmbeddingProviderFactory

# Create sentence transformers (free, local)
embedder = EmbeddingProviderFactory.create("sentence_transformers")
print(f"Provider: {embedder.provider_name}, Dimension: {embedder.dimension}")

# Test embedding
embedding = embedder.embed_text("Hello world")
print(f"Embedding length: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
```

---

## Phase 4: Synthetic Data Generation (Steps 15-18)

### Step 15: Create Data Generation Prompts
**Action**: Create prompt templates for generating synthetic FAQ data.

**File**: `shopunow_assistant/src/utils/data_generation_prompts.py`

```python
"""Prompts for generating synthetic FAQ data."""

DEPARTMENT_CONTEXTS = {
    "hr": {
        "name": "Human Resources",
        "user_type": "internal_employee",
        "description": """Human Resources department for ShopUNow, a retail company selling clothing, DIY products, books, and toys.
        
HR handles employee lifecycle matters including:
- Paid time off (PTO) and leave requests
- Payroll questions and salary matters
- Benefits enrollment and inquiries
- Performance reviews and feedback
- Company policies and handbook
- Onboarding for new employees
- Internal job transfers
- Employee wellness programs
- Training and development
- Workplace conduct and HR policies"""
    },
    "it_support": {
        "name": "IT Support / Tech Support",
        "user_type": "internal_employee", 
        "description": """IT Support department for ShopUNow, a retail company.
        
IT Support manages technical issues for employees including:
- Password resets and account access
- VPN setup and remote access
- Hardware issues (laptops, monitors, peripherals)
- Software installation and licensing
- Email and calendar problems
- Network and connectivity issues
- Security and phishing awareness
- System outages and maintenance
- New employee tech setup
- Printer and scanner issues"""
    },
    "billing": {
        "name": "Billing & Payments",
        "user_type": "external_customer",
        "description": """Billing & Payments department for ShopUNow, a retail company.
        
Billing handles customer payment matters including:
- Invoice requests and copies
- Refund processing and status
- Payment method updates
- Overcharge disputes
- Subscription and recurring billing
- Promotional code issues
- Gift card balance and redemption
- Payment failed/declined issues
- Tax and receipt questions
- Store credit inquiries"""
    },
    "shipping": {
        "name": "Shipping & Delivery",
        "user_type": "external_customer",
        "description": """Shipping & Delivery department for ShopUNow, a retail company.
        
Shipping handles order fulfillment queries including:
- Order tracking and status
- Delivery time estimates
- Delayed or missing packages
- Damaged goods reporting
- Return initiation and labels
- Address changes before shipping
- International shipping questions
- Pickup location options
- Delivery scheduling
- Package theft/lost claims"""
    }
}

def get_faq_generation_prompt(department_key: str, num_pairs: int = 15) -> str:
    """Generate prompt for creating FAQ pairs for a department."""
    
    context = DEPARTMENT_CONTEXTS.get(department_key)
    if not context:
        raise ValueError(f"Unknown department: {department_key}")
    
    user_type_desc = "internal employees of the company" if context["user_type"] == "internal_employee" else "external customers"
    
    return f"""Act as an expert in running the {context['name']} department for ShopUNow, a retail company selling clothing, DIY products, books, and toys.

{context['description']}

Create a list of {num_pairs} Questions and Answers which could be the most frequently asked questions for the {context['name']} department, typically asked by {user_type_desc}.

Requirements:
1. Questions should be realistic and diverse, covering different aspects of the department
2. Answers should be helpful, specific, and actionable (50-150 words each)
3. Include a mix of simple and complex questions
4. Answers should reference realistic systems, portals, or processes (e.g., "HR Portal", "IT Helpdesk ext. 5555")
5. Make answers sound professional but friendly

Return the response as a JSON array with this exact structure:
[
  {{
    "question": "The question text here",
    "answer": "The detailed answer here",
    "keywords": ["keyword1", "keyword2", "keyword3"]
  }},
  ...
]

Generate exactly {num_pairs} QA pairs. Output ONLY the JSON array, no other text."""
```

**Validation Checkpoint**:
```python
from src.utils.data_generation_prompts import get_faq_generation_prompt, DEPARTMENT_CONTEXTS

print(f"Available departments: {list(DEPARTMENT_CONTEXTS.keys())}")
prompt = get_faq_generation_prompt("hr", 15)
print(f"Prompt length: {len(prompt)} chars")
print(f"First 200 chars:\n{prompt[:200]}")
```

---

### Step 16: Create Data Generation Script
**Action**: Create script to generate FAQ data using LLM.

**File**: `shopunow_assistant/src/utils/generate_faq_data.py`

```python
"""Script to generate synthetic FAQ data for all departments."""

import json
from pathlib import Path
from typing import Optional
from datetime import datetime
from pydantic import BaseModel

from src.config import get_settings
from src.providers.llm_factory import LLMProviderFactory
from src.models import QAPair, DepartmentFAQs, Department, UserType
from src.utils.data_generation_prompts import get_faq_generation_prompt, DEPARTMENT_CONTEXTS


class GeneratedQA(BaseModel):
    """Pydantic model for LLM-generated QA pair."""
    question: str
    answer: str
    keywords: list[str]


class GeneratedQAList(BaseModel):
    """Wrapper for list of generated QAs."""
    items: list[GeneratedQA]


def generate_department_faqs(
    department_key: str,
    llm_provider,
    num_pairs: int = 15,
    output_dir: Optional[Path] = None
) -> DepartmentFAQs:
    """
    Generate FAQ data for a single department.
    
    Args:
        department_key: Department identifier (hr, it_support, billing, shipping)
        llm_provider: Configured LLM provider instance
        num_pairs: Number of QA pairs to generate
        output_dir: Directory to save JSON output
        
    Returns:
        DepartmentFAQs object with generated data
    """
    context = DEPARTMENT_CONTEXTS[department_key]
    prompt = get_faq_generation_prompt(department_key, num_pairs)
    
    print(f"Generating {num_pairs} FAQs for {context['name']}...")
    
    # Generate using LLM
    response = llm_provider.generate(
        prompt=prompt,
        temperature=0.7,  # Some creativity for diverse questions
        max_tokens=4000
    )
    
    # Parse JSON response
    try:
        raw_data = json.loads(response.content)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Raw content: {response.content[:500]}")
        raise
    
    # Convert to QAPair objects
    department = Department(department_key)
    user_type = UserType(context["user_type"])
    
    qa_pairs = []
    for i, item in enumerate(raw_data):
        qa = QAPair(
            id=f"{department_key}_{i+1:03d}",
            question=item["question"],
            answer=item["answer"],
            department=department,
            user_type=user_type,
            keywords=item.get("keywords", [])
        )
        qa_pairs.append(qa)
    
    # Create DepartmentFAQs
    dept_faqs = DepartmentFAQs(
        department=department,
        user_type=user_type,
        description=context["description"],
        qa_pairs=qa_pairs
    )
    
    print(f"  Generated {dept_faqs.count} QA pairs")
    
    # Save to file if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{department_key}_faqs.json"
        
        # Convert to serializable format
        output_data = {
            "department": department.value,
            "user_type": user_type.value,
            "description": context["description"],
            "generated_at": datetime.utcnow().isoformat(),
            "count": dept_faqs.count,
            "qa_pairs": [qa.model_dump(mode='json') for qa in qa_pairs]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"  Saved to {output_file}")
    
    return dept_faqs


def generate_all_faqs(
    output_dir: str = "./data/raw",
    llm_provider_type: str = "openai",
    llm_model: Optional[str] = None,
    num_pairs_per_dept: int = 15
) -> dict[str, DepartmentFAQs]:
    """
    Generate FAQ data for all departments.
    
    Args:
        output_dir: Directory to save JSON files
        llm_provider_type: LLM provider to use
        llm_model: Specific model (uses default if None)
        num_pairs_per_dept: Number of QA pairs per department
        
    Returns:
        Dictionary mapping department keys to DepartmentFAQs
    """
    # Initialize LLM
    llm = LLMProviderFactory.create(llm_provider_type, model=llm_model)
    print(f"Using LLM: {llm.provider_name}")
    
    output_path = Path(output_dir)
    results = {}
    
    for dept_key in DEPARTMENT_CONTEXTS.keys():
        try:
            dept_faqs = generate_department_faqs(
                department_key=dept_key,
                llm_provider=llm,
                num_pairs=num_pairs_per_dept,
                output_dir=output_path
            )
            results[dept_key] = dept_faqs
        except Exception as e:
            print(f"Error generating FAQs for {dept_key}: {e}")
            raise
    
    # Print summary
    total = sum(faqs.count for faqs in results.values())
    print(f"\n=== Generation Complete ===")
    print(f"Total QA pairs: {total}")
    for dept, faqs in results.items():
        print(f"  {dept}: {faqs.count} pairs")
    
    return results


if __name__ == "__main__":
    # Run generation
    generate_all_faqs()
```

**Validation Checkpoint**:
```python
# Run from shopunow_assistant directory
from src.utils.generate_faq_data import generate_all_faqs

# This will generate all FAQs (costs ~$0.10-0.20 with gpt-4o-mini)
results = generate_all_faqs(output_dir="./data/raw", num_pairs_per_dept=15)
print(f"Generated data for {len(results)} departments")
```

---

### Step 17: Create Data Loading Utilities
**Action**: Create utilities to load generated FAQ data.

**File**: `shopunow_assistant/src/utils/data_loader.py`

```python
"""Utilities for loading FAQ data from JSON files."""

import json
from pathlib import Path
from typing import Optional

from src.models import QAPair, DepartmentFAQs, Department, UserType


def load_department_faqs(file_path: str | Path) -> DepartmentFAQs:
    """
    Load FAQ data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        DepartmentFAQs object
    """
    file_path = Path(file_path)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert QA pairs
    qa_pairs = []
    for qa_data in data["qa_pairs"]:
        qa = QAPair(
            id=qa_data["id"],
            question=qa_data["question"],
            answer=qa_data["answer"],
            department=Department(qa_data["department"]),
            user_type=UserType(qa_data["user_type"]),
            keywords=qa_data.get("keywords", [])
        )
        qa_pairs.append(qa)
    
    return DepartmentFAQs(
        department=Department(data["department"]),
        user_type=UserType(data["user_type"]),
        description=data["description"],
        qa_pairs=qa_pairs
    )


def load_all_faqs(data_dir: str = "./data/raw") -> dict[str, DepartmentFAQs]:
    """
    Load all FAQ data from a directory.
    
    Args:
        data_dir: Directory containing JSON files
        
    Returns:
        Dictionary mapping department keys to DepartmentFAQs
    """
    data_path = Path(data_dir)
    results = {}
    
    for json_file in data_path.glob("*_faqs.json"):
        dept_key = json_file.stem.replace("_faqs", "")
        results[dept_key] = load_department_faqs(json_file)
        print(f"Loaded {results[dept_key].count} FAQs for {dept_key}")
    
    return results


def get_all_qa_pairs(data_dir: str = "./data/raw") -> list[QAPair]:
    """
    Load all QA pairs as a flat list.
    
    Args:
        data_dir: Directory containing JSON files
        
    Returns:
        List of all QAPair objects
    """
    all_faqs = load_all_faqs(data_dir)
    all_pairs = []
    
    for dept_faqs in all_faqs.values():
        all_pairs.extend(dept_faqs.qa_pairs)
    
    return all_pairs
```

**Validation Checkpoint**:
```python
from src.utils.data_loader import load_all_faqs, get_all_qa_pairs

# Load all FAQs
all_faqs = load_all_faqs("./data/raw")
print(f"Loaded {len(all_faqs)} departments")

# Get flat list
all_pairs = get_all_qa_pairs("./data/raw")
print(f"Total QA pairs: {len(all_pairs)}")

# Show sample
if all_pairs:
    sample = all_pairs[0]
    print(f"\nSample QA:")
    print(f"  ID: {sample.id}")
    print(f"  Q: {sample.question[:50]}...")
    print(f"  Dept: {sample.department.value}")
```

---

### Step 18: Update Utils __init__.py
**Action**: Export utilities from utils package.

**File**: `shopunow_assistant/src/utils/__init__.py`

```python
"""Utilities package."""

from .data_loader import load_department_faqs, load_all_faqs, get_all_qa_pairs
from .data_generation_prompts import get_faq_generation_prompt, DEPARTMENT_CONTEXTS

__all__ = [
    "load_department_faqs",
    "load_all_faqs", 
    "get_all_qa_pairs",
    "get_faq_generation_prompt",
    "DEPARTMENT_CONTEXTS",
]
```

**Validation Checkpoint**:
```python
from src.utils import load_all_faqs, DEPARTMENT_CONTEXTS
print(f"Utils imported successfully")
print(f"Available depts: {list(DEPARTMENT_CONTEXTS.keys())}")
```

---

## Phase 5: Vector Store Setup (Steps 19-22)

### Step 19: Create ChromaDB Client
**Action**: Implement ChromaDB wrapper with metadata filtering.

**File**: `shopunow_assistant/src/vectorstore/chroma_client.py`

```python
"""ChromaDB client wrapper with metadata filtering support."""

import chromadb
from chromadb.config import Settings
from typing import Optional
from pathlib import Path

from src.models import Department, UserType, QAPair, RetrievedDocument
from src.providers.base import BaseEmbeddingProvider


class ChromaDBClient:
    """ChromaDB wrapper with metadata filtering support."""
    
    COLLECTION_NAME = "shopunow_faqs"
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        embedding_provider: Optional[BaseEmbeddingProvider] = None
    ):
        """
        Initialize ChromaDB client.
        
        Args:
            persist_directory: Directory for persistent storage
            embedding_provider: Embedding provider instance
        """
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
    
    def set_embedding_provider(self, provider: BaseEmbeddingProvider):
        """Set or update embedding provider."""
        self.embedding_provider = provider
    
    def add_qa_pair(self, qa_pair: QAPair) -> None:
        """Add single QA pair to collection."""
        if not self.embedding_provider:
            raise ValueError("Embedding provider not set")
        
        document_text = qa_pair.to_document_text()
        embedding = self.embedding_provider.embed_text(document_text)
        
        metadata = {
            "department": qa_pair.department.value,
            "user_type": qa_pair.user_type.value,
            "question": qa_pair.question,
            "answer": qa_pair.answer,
            "keywords": ",".join(qa_pair.keywords),
        }
        
        self.collection.add(
            ids=[qa_pair.id],
            embeddings=[embedding],
            documents=[document_text],
            metadatas=[metadata]
        )
    
    def add_qa_pairs_batch(self, qa_pairs: list[QAPair], batch_size: int = 50) -> None:
        """
        Batch add multiple QA pairs.
        
        Args:
            qa_pairs: List of QAPair objects
            batch_size: Number of items per batch for embedding
        """
        if not qa_pairs:
            return
        
        if not self.embedding_provider:
            raise ValueError("Embedding provider not set")
        
        # Process in batches
        for i in range(0, len(qa_pairs), batch_size):
            batch = qa_pairs[i:i + batch_size]
            
            documents = []
            metadatas = []
            ids = []
            
            for qa in batch:
                documents.append(qa.to_document_text())
                ids.append(qa.id)
                metadatas.append({
                    "department": qa.department.value,
                    "user_type": qa.user_type.value,
                    "question": qa.question,
                    "answer": qa.answer,
                    "keywords": ",".join(qa.keywords),
                })
            
            # Batch embed
            embeddings = self.embedding_provider.embed_texts(documents)
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            print(f"  Added batch {i // batch_size + 1}: {len(batch)} documents")
    
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
        if not self.embedding_provider:
            raise ValueError("Embedding provider not set")
        
        # Build where clause
        where_clause = self._build_where_clause(department, user_type)
        
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
        return self._parse_results(results, score_threshold)
    
    def _build_where_clause(
        self,
        department: Optional[Department],
        user_type: Optional[UserType]
    ) -> Optional[dict]:
        """Build ChromaDB where clause for filtering."""
        conditions = []
        
        if department and department != Department.UNKNOWN:
            conditions.append({"department": {"$eq": department.value}})
        if user_type and user_type != UserType.UNKNOWN:
            conditions.append({"user_type": {"$eq": user_type.value}})
        
        if len(conditions) == 0:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}
    
    def _parse_results(
        self,
        results: dict,
        score_threshold: Optional[float]
    ) -> list[RetrievedDocument]:
        """Parse ChromaDB results into RetrievedDocument objects."""
        retrieved_docs = []
        
        if not results or not results['ids'] or not results['ids'][0]:
            return retrieved_docs
        
        for i, doc_id in enumerate(results['ids'][0]):
            # Convert distance to similarity (cosine distance)
            distance = results['distances'][0][i]
            similarity = 1 - distance
            
            # Apply threshold
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
        if department and department != Department.UNKNOWN:
            results = self.collection.get(
                where={"department": {"$eq": department.value}},
                include=[]
            )
            return len(results['ids'])
        return self.collection.count()
    
    def reset_collection(self) -> None:
        """Delete and recreate collection."""
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        self._collection = None
        _ = self.collection  # Recreate
        print(f"Collection '{self.COLLECTION_NAME}' reset")
```

**Validation Checkpoint**:
```python
from src.vectorstore.chroma_client import ChromaDBClient
from src.providers.embedding_providers import EmbeddingProviderFactory

# Create client and embedding provider
embedder = EmbeddingProviderFactory.create("sentence_transformers")
chroma = ChromaDBClient(persist_directory="./data/chroma_db", embedding_provider=embedder)

print(f"Collection count: {chroma.get_document_count()}")
```

---

### Step 20: Create Ingestion Pipeline
**Action**: Create pipeline to ingest FAQ data into ChromaDB.

**File**: `shopunow_assistant/src/vectorstore/ingestion.py`

```python
"""Data ingestion pipeline for loading FAQs into ChromaDB."""

from pathlib import Path
from typing import Optional

from src.models import QAPair, Department
from src.utils import load_all_faqs, get_all_qa_pairs
from src.providers.base import BaseEmbeddingProvider
from src.providers.embedding_providers import EmbeddingProviderFactory, EmbeddingProviderType
from .chroma_client import ChromaDBClient


def ingest_faqs(
    data_dir: str = "./data/raw",
    chroma_dir: str = "./data/chroma_db",
    embedding_provider: Optional[BaseEmbeddingProvider] = None,
    embedding_provider_type: str = "sentence_transformers",
    embedding_model: Optional[str] = None,
    reset_collection: bool = False
) -> ChromaDBClient:
    """
    Ingest all FAQ data into ChromaDB.
    
    Args:
        data_dir: Directory containing FAQ JSON files
        chroma_dir: ChromaDB persistence directory
        embedding_provider: Pre-configured embedding provider (optional)
        embedding_provider_type: Embedding provider type if no provider given
        embedding_model: Embedding model if no provider given
        reset_collection: Whether to reset collection before ingestion
        
    Returns:
        Configured ChromaDBClient instance
    """
    # Setup embedding provider
    if embedding_provider is None:
        embedding_provider = EmbeddingProviderFactory.create(
            provider=embedding_provider_type,
            model=embedding_model
        )
    
    print(f"Using embedding provider: {embedding_provider.provider_name}")
    print(f"Embedding dimension: {embedding_provider.dimension}")
    
    # Setup ChromaDB client
    chroma = ChromaDBClient(
        persist_directory=chroma_dir,
        embedding_provider=embedding_provider
    )
    
    # Reset if requested
    if reset_collection:
        print("Resetting collection...")
        chroma.reset_collection()
    
    # Load all QA pairs
    all_pairs = get_all_qa_pairs(data_dir)
    print(f"Loaded {len(all_pairs)} QA pairs from {data_dir}")
    
    # Check for existing documents
    existing_count = chroma.get_document_count()
    if existing_count > 0:
        print(f"Collection already has {existing_count} documents")
        if not reset_collection:
            print("Skipping ingestion. Use reset_collection=True to re-ingest.")
            return chroma
    
    # Ingest
    print("Ingesting documents...")
    chroma.add_qa_pairs_batch(all_pairs, batch_size=20)
    
    # Verify
    final_count = chroma.get_document_count()
    print(f"\n=== Ingestion Complete ===")
    print(f"Total documents: {final_count}")
    
    for dept in Department.valid_departments():
        dept_count = chroma.get_document_count(dept)
        print(f"  {dept.value}: {dept_count} documents")
    
    return chroma


def verify_ingestion(
    chroma_dir: str = "./data/chroma_db",
    embedding_provider: Optional[BaseEmbeddingProvider] = None
) -> bool:
    """
    Verify that ingestion was successful.
    
    Args:
        chroma_dir: ChromaDB persistence directory
        embedding_provider: Embedding provider for test queries
        
    Returns:
        True if verification passes
    """
    if embedding_provider is None:
        embedding_provider = EmbeddingProviderFactory.create("sentence_transformers")
    
    chroma = ChromaDBClient(
        persist_directory=chroma_dir,
        embedding_provider=embedding_provider
    )
    
    # Check counts
    total = chroma.get_document_count()
    print(f"Total documents: {total}")
    
    if total == 0:
        print("VERIFICATION FAILED: No documents in collection")
        return False
    
    # Test query for each department
    test_queries = {
        Department.HR: "How do I apply for time off?",
        Department.IT_SUPPORT: "I forgot my password",
        Department.BILLING: "I need a refund",
        Department.SHIPPING: "Where is my order?",
    }
    
    print("\nTest queries:")
    for dept, query in test_queries.items():
        results = chroma.query(
            query_text=query,
            department=dept,
            n_results=1
        )
        
        if results:
            print(f"  [{dept.value}] '{query[:30]}...' → Score: {results[0].similarity_score:.3f}")
        else:
            print(f"  [{dept.value}] '{query[:30]}...' → NO RESULTS")
    
    return True


if __name__ == "__main__":
    # Run ingestion
    ingest_faqs(reset_collection=True)
    verify_ingestion()
```

**Validation Checkpoint**:
```python
from src.vectorstore.ingestion import ingest_faqs, verify_ingestion

# Run ingestion (requires FAQ data to be generated first)
chroma = ingest_faqs(reset_collection=True)

# Verify
verify_ingestion()
```

---

### Step 21: Update VectorStore __init__.py
**Action**: Export vectorstore classes.

**File**: `shopunow_assistant/src/vectorstore/__init__.py`

```python
"""Vector store package."""

from .chroma_client import ChromaDBClient
from .ingestion import ingest_faqs, verify_ingestion

__all__ = [
    "ChromaDBClient",
    "ingest_faqs",
    "verify_ingestion",
]
```

---

### Step 22: Test Vector Store End-to-End
**Action**: Run complete ingestion and query test.

**Test Script** (run interactively):

```python
# Complete vector store test
from src.vectorstore import ingest_faqs, verify_ingestion, ChromaDBClient
from src.providers.embedding_providers import EmbeddingProviderFactory
from src.models import Department

# Step 1: Ingest
print("=== INGESTION ===")
chroma = ingest_faqs(reset_collection=True)

# Step 2: Verify
print("\n=== VERIFICATION ===")
verify_ingestion()

# Step 3: Manual query test
print("\n=== MANUAL QUERY TEST ===")
embedder = EmbeddingProviderFactory.create("sentence_transformers")
client = ChromaDBClient("./data/chroma_db", embedder)

query = "How do I check my PTO balance?"
results = client.query(query, department=Department.HR, n_results=3)

print(f"Query: {query}")
print(f"Results ({len(results)}):")
for r in results:
    print(f"  [{r.similarity_score:.3f}] {r.question[:60]}...")
```

**Validation Checkpoint**: All queries should return relevant results with similarity scores > 0.5

---

## Phase 6: Classification Pipeline (Steps 23-26)

### Step 23: Create Prompt Templates
**Action**: Define prompts for sentiment and department classification.

**File**: `shopunow_assistant/src/utils/prompts.py`

```python
"""Prompt templates for classification and generation."""

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
- Simple questions without emotional language are NEUTRAL"""


DEPARTMENT_SYSTEM_PROMPT = """You are a query router for ShopUNow, a retail company.
Classify the user query into the appropriate department.

Available Departments:

INTERNAL EMPLOYEE DEPARTMENTS (for company employees):
1. HR (hr): Employee lifecycle - leave requests, PTO, payroll questions, benefits, performance reviews, policies, onboarding
2. IT_SUPPORT (it_support): Technical issues - hardware, software, system access, password resets, VPN, email issues

EXTERNAL CUSTOMER DEPARTMENTS (for customers shopping at ShopUNow):
3. BILLING (billing): Payment issues - invoices, refunds, payment methods, overcharges, subscription billing, gift cards
4. SHIPPING (shipping): Delivery issues - order tracking, delivery delays, damaged goods, returns, pickup scheduling

UNKNOWN (unknown): Use ONLY if the query doesn't fit ANY department above (e.g., "What's the weather?", "Tell me a joke")

Classification Rules:
1. Look for keywords: "my order", "tracking", "delivery" → shipping; "password", "login", "VPN" → IT; "refund", "payment", "invoice" → billing; "PTO", "leave", "payroll" → HR
2. Consider context: "my paycheck" → HR; "my payment" → billing
3. Employee-specific language ("I work here", "as an employee") → internal departments
4. Customer-specific language ("I ordered", "my purchase") → external departments
5. Default to the most likely department based on query content"""


RESPONSE_GENERATION_PROMPT = """You are a helpful customer service assistant for ShopUNow retail company.
Based on the retrieved information, provide a clear and helpful answer to the user's question.

Guidelines:
- Be concise but thorough
- Use a friendly, professional tone
- If the retrieved information partially answers the question, provide what you can and acknowledge limitations
- Don't make up information not present in the sources
- For employees: be supportive and solution-oriented
- For customers: be empathetic and action-oriented
- Reference specific systems/portals mentioned in the sources when relevant"""
```

---

### Step 24: Create Classification Pipeline
**Action**: Implement sentiment and department classification.

**File**: `shopunow_assistant/src/pipelines/classification.py`

```python
"""Classification pipeline for sentiment and department detection."""

from src.providers.base import BaseLLMProvider
from src.models import (
    SentimentAnalysis,
    DepartmentClassification,
    ClassificationResult,
    Sentiment,
    Department,
    UserType
)
from src.utils.prompts import SENTIMENT_SYSTEM_PROMPT, DEPARTMENT_SYSTEM_PROMPT


class ClassificationPipeline:
    """Pipeline for sentiment and department classification."""
    
    def __init__(self, llm_provider: BaseLLMProvider):
        """
        Initialize classification pipeline.
        
        Args:
            llm_provider: Configured LLM provider instance
        """
        self.llm = llm_provider
    
    def detect_sentiment(self, query: str) -> SentimentAnalysis:
        """
        Detect sentiment of user query.
        
        Args:
            query: User's query text
            
        Returns:
            SentimentAnalysis with sentiment, confidence, and indicators
        """
        prompt = f"""Analyze the sentiment of this query:

Query: "{query}"

Determine if the sentiment is positive, neutral, or negative.
Identify key words or phrases that influenced your decision."""
        
        return self.llm.generate_structured(
            prompt=prompt,
            response_model=SentimentAnalysis,
            system_prompt=SENTIMENT_SYSTEM_PROMPT,
            temperature=0.0
        )
    
    def classify_department(self, query: str) -> DepartmentClassification:
        """
        Classify query into appropriate department.
        
        Args:
            query: User's query text
            
        Returns:
            DepartmentClassification with department, user_type, confidence, reasoning
        """
        prompt = f"""Classify this query into the appropriate department:

Query: "{query}"

Determine:
1. Which department should handle this query
2. Whether the user is an internal employee or external customer
3. Your confidence level (0.0 to 1.0)
4. Brief reasoning for your classification"""
        
        return self.llm.generate_structured(
            prompt=prompt,
            response_model=DepartmentClassification,
            system_prompt=DEPARTMENT_SYSTEM_PROMPT,
            temperature=0.0
        )
    
    def classify(self, query: str) -> ClassificationResult:
        """
        Full classification pipeline: sentiment + department.
        
        Args:
            query: User's query text
            
        Returns:
            ClassificationResult with all classification data
        """
        # Run both classifications
        sentiment_result = self.detect_sentiment(query)
        department_result = self.classify_department(query)
        
        # Combined confidence is the minimum
        combined_confidence = min(
            sentiment_result.confidence,
            department_result.confidence
        )
        
        # Build combined reasoning
        reasoning = (
            f"Sentiment ({sentiment_result.sentiment.value}): "
            f"{', '.join(sentiment_result.indicators) if sentiment_result.indicators else 'standard query'}. "
            f"Department: {department_result.reasoning}"
        )
        
        return ClassificationResult(
            sentiment=sentiment_result.sentiment,
            department=department_result.department,
            user_type=department_result.user_type,
            confidence=combined_confidence,
            reasoning=reasoning
        )
```

**Validation Checkpoint**:
```python
from src.pipelines.classification import ClassificationPipeline
from src.providers.llm_factory import LLMProviderFactory

llm = LLMProviderFactory.create("openai")
classifier = ClassificationPipeline(llm)

# Test queries
test_queries = [
    "How do I apply for PTO?",
    "Where is my order #12345?",
    "This is ridiculous! I've been waiting for weeks!!!",
    "I need to reset my password",
]

for query in test_queries:
    result = classifier.classify(query)
    print(f"Query: {query[:40]}...")
    print(f"  Sentiment: {result.sentiment.value}")
    print(f"  Department: {result.department.value}")
    print(f"  Confidence: {result.confidence:.2f}")
    print()
```

---

### Step 25: Update Pipelines __init__.py
**Action**: Export pipeline classes.

**File**: `shopunow_assistant/src/pipelines/__init__.py`

```python
"""Pipelines package."""

from .classification import ClassificationPipeline

__all__ = [
    "ClassificationPipeline",
]
```

---

### Step 26: Test Classification End-to-End
**Action**: Comprehensive classification testing.

**Test Script**:
```python
from src.pipelines.classification import ClassificationPipeline
from src.providers.llm_factory import LLMProviderFactory
from src.models import Sentiment, Department

llm = LLMProviderFactory.create("openai")
classifier = ClassificationPipeline(llm)

# Test cases with expected results
test_cases = [
    # (query, expected_sentiment, expected_department)
    ("How do I check my PTO balance?", Sentiment.NEUTRAL, Department.HR),
    ("I can't login to my email", Sentiment.NEUTRAL, Department.IT_SUPPORT),
    ("I need a copy of my invoice", Sentiment.NEUTRAL, Department.BILLING),
    ("When will my package arrive?", Sentiment.NEUTRAL, Department.SHIPPING),
    ("Thanks so much for your help!", Sentiment.POSITIVE, None),  # Any dept ok
    ("This is unacceptable! Fix this NOW!", Sentiment.NEGATIVE, None),  # Any dept
]

print("=== Classification Test Suite ===\n")
passed = 0
for query, exp_sentiment, exp_dept in test_cases:
    result = classifier.classify(query)
    
    sentiment_ok = result.sentiment == exp_sentiment
    dept_ok = exp_dept is None or result.department == exp_dept
    
    status = "✓" if (sentiment_ok and dept_ok) else "✗"
    passed += 1 if (sentiment_ok and dept_ok) else 0
    
    print(f"{status} Query: {query[:50]}...")
    print(f"  Sentiment: {result.sentiment.value} (expected: {exp_sentiment.value}) {'✓' if sentiment_ok else '✗'}")
    if exp_dept:
        print(f"  Department: {result.department.value} (expected: {exp_dept.value}) {'✓' if dept_ok else '✗'}")
    print()

print(f"Passed: {passed}/{len(test_cases)}")
```

---

## Phase 7: Retrieval Pipeline (Steps 27-30)

### Step 27: Create Dynamic K Retriever
**Action**: Implement dynamic k selection algorithm.

**File**: `shopunow_assistant/src/pipelines/retrieval.py`

```python
"""Retrieval pipeline with dynamic k selection."""

from typing import Optional

from src.vectorstore.chroma_client import ChromaDBClient
from src.models import (
    Department,
    UserType,
    RetrievalResult,
    RetrievedDocument
)


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
        Initialize retriever.
        
        Args:
            chroma_client: ChromaDB client instance
            min_threshold: Minimum similarity score to include (0-1)
            max_k: Maximum documents to retrieve initially
            drop_off_ratio: Stop if next score < previous * ratio
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
        # Get initial candidate pool
        candidates = self.chroma.query(
            query_text=query,
            department=department,
            user_type=user_type,
            n_results=self.max_k,
            score_threshold=None  # Filter manually
        )
        
        # Apply dynamic k selection
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
        1. Sort by score descending (should already be sorted)
        2. Always include first doc if above threshold
        3. Include subsequent docs if:
           a) Above min_threshold AND
           b) Score >= previous_score * drop_off_ratio
        """
        if not candidates:
            return []
        
        # Ensure sorted
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
                break
            
            # Check drop-off ratio (skip for first doc)
            if previous_score is not None:
                if doc.similarity_score < previous_score * self.drop_off_ratio:
                    break
            
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
        """
        Initialize retrieval pipeline.
        
        Args:
            retriever: DynamicKRetriever instance
            fallback_message: Message when no results found
        """
        self.retriever = retriever
        self.fallback_message = fallback_message
    
    def retrieve(
        self,
        query: str,
        department: Department,
        user_type: Optional[UserType] = None
    ) -> RetrievalResult:
        """
        Execute retrieval with logging.
        
        Args:
            query: User query
            department: Department filter
            user_type: Optional user type filter
            
        Returns:
            RetrievalResult
        """
        result = self.retriever.retrieve(
            query=query,
            department=department,
            user_type=user_type
        )
        
        if not result.has_results:
            print(f"[RETRIEVAL] No results for: {query[:50]}... in {department.value}")
        else:
            print(f"[RETRIEVAL] Found {result.retrieval_count} docs for: {query[:30]}...")
        
        return result
```

---

### Step 28: Update Pipelines __init__.py
**Action**: Add retrieval exports.

**File**: `shopunow_assistant/src/pipelines/__init__.py` (update)

```python
"""Pipelines package."""

from .classification import ClassificationPipeline
from .retrieval import DynamicKRetriever, RetrievalPipeline

__all__ = [
    "ClassificationPipeline",
    "DynamicKRetriever",
    "RetrievalPipeline",
]
```

---

### Step 29: Create Response Generation
**Action**: Implement response generation with source attribution.

**File**: `shopunow_assistant/src/pipelines/generation.py`

```python
"""Response generation pipeline."""

from typing import Optional

from src.providers.base import BaseLLMProvider
from src.models import (
    RetrievalResult,
    GeneratedResponse,
    SourceAttribution,
    Department,
    ClassificationResult
)
from src.utils.prompts import RESPONSE_GENERATION_PROMPT


class ResponseGenerator:
    """Generate responses from retrieved context."""
    
    def __init__(self, llm_provider: BaseLLMProvider):
        """
        Initialize response generator.
        
        Args:
            llm_provider: Configured LLM provider
        """
        self.llm = llm_provider
    
    def generate(
        self,
        query: str,
        retrieval: RetrievalResult,
        classification: Optional[ClassificationResult] = None
    ) -> GeneratedResponse:
        """
        Generate response from retrieved context.
        
        Args:
            query: Original user query
            retrieval: Retrieval results
            classification: Optional classification for confidence
            
        Returns:
            GeneratedResponse with answer and sources
        """
        # Handle no results
        if not retrieval.has_results:
            return GeneratedResponse(
                answer="I couldn't find specific information to answer your question. "
                       "Please try rephrasing or contact support for assistance.",
                sources=[],
                confidence=0.3,
                department=retrieval.department_filter
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
        
        # Generate response
        prompt = f"""User Query: {query}

Retrieved Information:
{context}

Based on the above information, provide a helpful response to the user's query.
If the information doesn't fully address the query, acknowledge what you can answer and what may need further clarification.
Be concise and friendly."""
        
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
        
        # Calculate confidence
        avg_score = sum(d.similarity_score for d in retrieval.documents) / len(retrieval.documents)
        confidence = avg_score
        if classification:
            confidence = min(avg_score, classification.confidence)
        
        return GeneratedResponse(
            answer=llm_response.content,
            sources=sources,
            confidence=confidence,
            department=retrieval.department_filter
        )
```

---

### Step 30: Update Pipelines and Test Retrieval
**Action**: Final pipeline exports and testing.

**File**: `shopunow_assistant/src/pipelines/__init__.py` (final)

```python
"""Pipelines package."""

from .classification import ClassificationPipeline
from .retrieval import DynamicKRetriever, RetrievalPipeline
from .generation import ResponseGenerator

__all__ = [
    "ClassificationPipeline",
    "DynamicKRetriever",
    "RetrievalPipeline",
    "ResponseGenerator",
]
```

**Validation Checkpoint**:
```python
from src.vectorstore import ChromaDBClient
from src.providers.embedding_providers import EmbeddingProviderFactory
from src.providers.llm_factory import LLMProviderFactory
from src.pipelines import DynamicKRetriever, RetrievalPipeline, ResponseGenerator
from src.models import Department

# Setup
embedder = EmbeddingProviderFactory.create("sentence_transformers")
chroma = ChromaDBClient("./data/chroma_db", embedder)
llm = LLMProviderFactory.create("openai")

retriever = DynamicKRetriever(chroma, min_threshold=0.5)
pipeline = RetrievalPipeline(retriever)
generator = ResponseGenerator(llm)

# Test
query = "How do I check my PTO balance?"
result = pipeline.retrieve(query, Department.HR)

print(f"Query: {query}")
print(f"Retrieved {result.retrieval_count} documents:")
for doc in result.documents:
    print(f"  [{doc.similarity_score:.3f}] {doc.question[:50]}...")

# Generate response
response = generator.generate(query, result)
print(f"\nGenerated Answer:\n{response.answer}")
print(f"\nSources: {len(response.sources)}")
print(f"Confidence: {response.confidence:.2f}")
```

---

## Phase 8: Routing & Orchestration (Steps 31-34)

### Step 31: Create Router
**Action**: Implement query routing logic.

**File**: `shopunow_assistant/src/routing/router.py`

```python
"""Query routing logic."""

from src.models import (
    ClassificationResult,
    RoutingDecision,
    RouteDecision,
    Department,
    Sentiment
)


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
                reasoning="Query has negative sentiment. Escalating to human agent for better handling."
            )
        
        # Rule 2: Unknown department → Escalate
        if classification.department not in self.valid_departments:
            return RoutingDecision(
                route=RouteDecision.HUMAN_ESCALATION,
                classification=classification,
                reasoning=f"Query does not match any known department ({classification.department.value}). Routing to human agent."
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

---

### Step 32: Update Routing __init__.py
**Action**: Export router.

**File**: `shopunow_assistant/src/routing/__init__.py`

```python
"""Routing package."""

from .router import QueryRouter

__all__ = ["QueryRouter"]
```

---

### Step 33: Create Orchestrator
**Action**: Implement main orchestration layer.

**File**: `shopunow_assistant/src/orchestrator.py`

```python
"""Main orchestrator coordinating all pipeline components."""

import time
from typing import Optional, Union

from src.config import Settings, get_settings
from src.models import (
    QueryRequest,
    QueryResponse,
    EscalationResponse,
    ClassificationResult,
    RouteDecision
)
from src.providers.llm_factory import LLMProviderFactory, LLMProviderType
from src.providers.embedding_providers import EmbeddingProviderFactory, EmbeddingProviderType
from src.pipelines import (
    ClassificationPipeline,
    DynamicKRetriever,
    RetrievalPipeline,
    ResponseGenerator
)
from src.routing import QueryRouter
from src.vectorstore import ChromaDBClient


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
        """
        Initialize orchestrator with all components.
        
        Args:
            settings: Application settings (uses defaults if None)
        """
        self.settings = settings or get_settings()
        
        # Initialize LLM provider
        self.llm = LLMProviderFactory.create(
            provider=self.settings.llm_provider,
            model=self.settings.llm_model
        )
        print(f"[INIT] LLM: {self.llm.provider_name}")
        
        # Initialize embedding provider
        self.embedding = EmbeddingProviderFactory.create(
            provider=self.settings.embedding_provider,
            model=self.settings.embedding_model
        )
        print(f"[INIT] Embeddings: {self.embedding.provider_name} (dim={self.embedding.dimension})")
        
        # Initialize ChromaDB
        self.chroma = ChromaDBClient(
            persist_directory=self.settings.chroma_persist_dir,
            embedding_provider=self.embedding
        )
        doc_count = self.chroma.get_document_count()
        print(f"[INIT] ChromaDB: {doc_count} documents")
        
        # Initialize pipelines
        self.classifier = ClassificationPipeline(self.llm)
        
        self.retriever = DynamicKRetriever(
            chroma_client=self.chroma,
            min_threshold=self.settings.retrieval_min_threshold,
            max_k=self.settings.retrieval_max_k,
            drop_off_ratio=self.settings.retrieval_drop_off_ratio
        )
        self.retrieval_pipeline = RetrievalPipeline(self.retriever)
        
        self.generator = ResponseGenerator(self.llm)
        self.router = QueryRouter()
        
        print("[INIT] Orchestrator ready")
    
    def process_query(
        self,
        request: QueryRequest
    ) -> Union[QueryResponse, EscalationResponse]:
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
        print(f"[PROCESS] Classifying: {request.query[:50]}...")
        classification = self.classifier.classify(request.query)
        print(f"[PROCESS] Sentiment: {classification.sentiment.value}, Dept: {classification.department.value}")
        
        # Step 2: Routing
        routing_decision = self.router.route(classification)
        print(f"[PROCESS] Route: {routing_decision.route.value}")
        
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
        response = self.generator.generate(
            query=request.query,
            retrieval=retrieval_result,
            classification=classification
        )
        
        processing_time = (time.time() - start_time) * 1000
        
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
            ticket_id=f"ESC-{int(time.time())}"
        )
```

---

### Step 34: Test Orchestrator
**Action**: End-to-end orchestrator testing.

**Validation Checkpoint**:
```python
from src.orchestrator import ShopUNowOrchestrator
from src.models import QueryRequest

# Initialize (loads all components)
orchestrator = ShopUNowOrchestrator()

# Test queries
test_queries = [
    "How do I apply for PTO?",
    "Where is my order #12345?",
    "I need to reset my password",
    "This is terrible service! I want a refund NOW!",
    "What's the weather like today?",
]

print("\n=== ORCHESTRATOR TEST ===\n")
for query in test_queries:
    request = QueryRequest(query=query)
    response = orchestrator.process_query(request)
    
    print(f"Query: {query}")
    if hasattr(response, 'was_escalated') and not response.was_escalated:
        print(f"  Department: {response.department.value}")
        print(f"  Answer: {response.answer[:100]}...")
        print(f"  Confidence: {response.confidence:.2f}")
        print(f"  Time: {response.processing_time_ms:.0f}ms")
    else:
        print(f"  ESCALATED: {response.reason}")
    print()
```

---

## Phase 9: FastAPI Integration (Steps 35-37)

### Step 35: Create FastAPI Application
**Action**: Implement FastAPI endpoints.

**File**: `shopunow_assistant/src/main.py`

```python
"""FastAPI application for ShopUNow Assistant."""

from contextlib import asynccontextmanager
from typing import Union
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.models import (
    QueryRequest,
    QueryResponse,
    EscalationResponse,
    HealthCheckResponse,
    ErrorResponse,
    Department
)
from src.orchestrator import ShopUNowOrchestrator


# Global orchestrator instance
orchestrator: ShopUNowOrchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    global orchestrator
    
    # Startup
    print("="*50)
    print("ShopUNow AI Assistant Starting...")
    print("="*50)
    
    settings = get_settings()
    orchestrator = ShopUNowOrchestrator(settings)
    
    print("="*50)
    print("Ready to accept requests")
    print("="*50)
    
    yield
    
    # Shutdown
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="ShopUNow AI Assistant",
    description="Intelligent AI Assistant for ShopUNow retail company. "
                "Handles queries for HR, IT Support, Billing, and Shipping departments.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/query",
    response_model=Union[QueryResponse, EscalationResponse],
    responses={
        200: {"description": "Successful response"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["Assistant"],
    summary="Process a user query"
)
async def process_query(request: QueryRequest):
    """
    Process a user query through the AI assistant.
    
    The assistant will:
    1. Analyze sentiment and classify the query into a department
    2. Route to appropriate handler (RAG or human escalation)
    3. Retrieve relevant information and generate a response
    
    Queries with negative sentiment or unknown department are escalated to human support.
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
    tags=["System"],
    summary="Health check endpoint"
)
async def health_check(deep: bool = False):
    """
    Health check endpoint.
    
    Args:
        deep: If true, verify all component connections
    """
    components = {"api": "healthy"}
    
    if deep and orchestrator:
        try:
            doc_count = orchestrator.chroma.get_document_count()
            components["chromadb"] = f"healthy ({doc_count} docs)"
        except Exception as e:
            components["chromadb"] = f"unhealthy: {str(e)}"
        
        try:
            test_response = orchestrator.llm.generate("Say OK", max_tokens=5)
            components["llm"] = f"healthy ({orchestrator.llm.provider_name})"
        except Exception as e:
            components["llm"] = f"unhealthy: {str(e)}"
        
        components["embeddings"] = f"healthy ({orchestrator.embedding.provider_name})"
    
    overall_status = "healthy" if all("healthy" in str(v) for v in components.values()) else "degraded"
    
    return HealthCheckResponse(
        status=overall_status,
        version="1.0.0",
        components=components
    )


@app.get(
    "/departments",
    tags=["Information"],
    summary="List available departments"
)
async def list_departments():
    """List all available departments and their descriptions."""
    return {
        "departments": {
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
    }


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ShopUNow AI Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# Entry point for running with uvicorn
if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
```

---

### Step 36: Create Run Script
**Action**: Create convenience script to run the API.

**File**: `shopunow_assistant/run.py`

```python
"""Run script for ShopUNow Assistant API."""

import uvicorn
from src.config import get_settings


def main():
    settings = get_settings()
    
    print(f"Starting ShopUNow Assistant API")
    print(f"  Host: {settings.api_host}")
    print(f"  Port: {settings.api_port}")
    print(f"  Debug: {settings.debug}")
    print(f"  Docs: http://{settings.api_host}:{settings.api_port}/docs")
    
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )


if __name__ == "__main__":
    main()
```

**Validation Checkpoint**:
```bash
# Run from shopunow_assistant directory
python run.py

# In another terminal, test the API:
curl http://localhost:8000/health
curl http://localhost:8000/departments
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I apply for PTO?"}'
```

---

### Step 37: Create API Test Script
**Action**: Create comprehensive API test script.

**File**: `shopunow_assistant/test_api.py`

```python
"""API test script for ShopUNow Assistant."""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("=== Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    # Deep health check
    response = requests.get(f"{BASE_URL}/health?deep=true")
    print("\nDeep Health Check:")
    print(json.dumps(response.json(), indent=2))


def test_departments():
    """Test departments endpoint."""
    print("\n=== Departments ===")
    response = requests.get(f"{BASE_URL}/departments")
    print(json.dumps(response.json(), indent=2))


def test_queries():
    """Test query endpoint with various queries."""
    print("\n=== Query Tests ===")
    
    test_cases = [
        ("How do I apply for PTO?", "HR query"),
        ("I can't login to my email", "IT query"),
        ("Where is my order #12345?", "Shipping query"),
        ("I need a refund for my purchase", "Billing query"),
        ("This is terrible! Fix it NOW!!!", "Negative sentiment - should escalate"),
        ("What's the weather today?", "Unknown department - should escalate"),
    ]
    
    for query, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Query: {query}")
        
        response = requests.post(
            f"{BASE_URL}/query",
            json={"query": query}
        )
        
        result = response.json()
        
        if "was_escalated" in result:
            if result["was_escalated"]:
                print(f"Result: ESCALATED - {result.get('escalation_message', 'N/A')}")
            else:
                print(f"Department: {result['department']}")
                print(f"Sentiment: {result['sentiment']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"Answer: {result['answer'][:100]}...")
                print(f"Sources: {len(result['sources'])}")
                print(f"Time: {result['processing_time_ms']:.0f}ms")
        else:
            # Escalation response
            print(f"Result: ESCALATED")
            print(f"Reason: {result.get('reason', 'N/A')}")
            print(f"Ticket: {result.get('ticket_id', 'N/A')}")


def main():
    print("ShopUNow Assistant API Test Suite")
    print("="*50)
    print(f"Target: {BASE_URL}")
    print()
    
    try:
        test_health()
        test_departments()
        test_queries()
        print("\n" + "="*50)
        print("All tests completed!")
    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Could not connect to {BASE_URL}")
        print("Make sure the API is running: python run.py")


if __name__ == "__main__":
    main()
```

---

## Phase 10: Testing & Notebook Assembly (Step 38)

### Step 38: Create Final Jupyter Notebook
**Action**: Create comprehensive Jupyter notebook for submission.

**File**: `shopunow_assistant/notebooks/ShopUNow_Capstone.ipynb`

The notebook should include:

1. **Introduction & Setup**
   - Project overview
   - Dependencies installation
   - Configuration

2. **Data Generation**
   - Generate synthetic FAQ data
   - Inspect generated data

3. **Vector Store**
   - Ingest data into ChromaDB
   - Verify ingestion

4. **Classification Pipeline**
   - Test sentiment detection
   - Test department classification

5. **Retrieval Pipeline**
   - Test dynamic k retrieval
   - Test response generation

6. **Full Pipeline**
   - End-to-end testing
   - Multiple query examples

7. **API Demo**
   - Start API (instructions)
   - Test endpoints

8. **Conclusion**
   - Summary
   - Future improvements

---

## Summary Checklist

### Phase Completion Checklist

| Phase | Steps | Status |
|-------|-------|--------|
| 1. Project Scaffolding | 1-4 | ☐ |
| 2. Enums & Core Models | 5-8 | ☐ |
| 3. Provider Abstraction | 9-14 | ☐ |
| 4. Synthetic Data Generation | 15-18 | ☐ |
| 5. Vector Store Setup | 19-22 | ☐ |
| 6. Classification Pipeline | 23-26 | ☐ |
| 7. Retrieval Pipeline | 27-30 | ☐ |
| 8. Routing & Orchestration | 31-34 | ☐ |
| 9. FastAPI Integration | 35-37 | ☐ |
| 10. Testing & Notebook | 38 | ☐ |

### Key Validation Points

- [ ] Step 4: Config loads from .env
- [ ] Step 10: OpenAI LLM generates response
- [ ] Step 14: Sentence Transformers embeddings work
- [ ] Step 17: FAQ data loads correctly
- [ ] Step 22: ChromaDB queries return results
- [ ] Step 26: Classification detects sentiment/department
- [ ] Step 30: Retrieval returns relevant documents
- [ ] Step 34: Orchestrator processes queries end-to-end
- [ ] Step 37: API responds to HTTP requests

---

*End of ATOMIC_STEPS.md*
