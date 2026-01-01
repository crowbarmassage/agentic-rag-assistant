"""API request and response models."""

from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from .enums import Department, Sentiment
from .data_models import SourceAttribution


# ==================== Request Models ====================

class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="User query to process"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Optional user identifier for tracking"
    )
    session_id: Optional[str] = Field(
        default=None,
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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Overall status: healthy, degraded, unhealthy")
    version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    components: dict[str, str] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Standardized error response."""

    error: str
    message: str
    details: Optional[dict] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
