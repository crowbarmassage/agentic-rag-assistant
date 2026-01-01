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
