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
