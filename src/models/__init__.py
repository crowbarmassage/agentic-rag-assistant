"""Data models and enumerations for ShopUNow Assistant."""

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
    # Data Models
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
    # API Models
    "QueryRequest",
    "QueryResponse",
    "EscalationResponse",
    "HealthCheckResponse",
    "ErrorResponse",
]
