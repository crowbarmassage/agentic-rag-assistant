"""Core data models for ShopUNow Assistant."""

from typing import Optional
from datetime import datetime, timezone
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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

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
