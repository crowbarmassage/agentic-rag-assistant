"""Phase 6 Tests: Classification Pipeline."""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestPrompts:
    """Test prompt templates."""

    def test_prompts_import(self):
        """Verify prompts can be imported."""
        from src.utils.prompts import (
            SENTIMENT_SYSTEM_PROMPT,
            DEPARTMENT_SYSTEM_PROMPT,
            RESPONSE_GENERATION_PROMPT
        )
        assert SENTIMENT_SYSTEM_PROMPT is not None
        assert DEPARTMENT_SYSTEM_PROMPT is not None
        assert RESPONSE_GENERATION_PROMPT is not None

    def test_prompts_from_utils(self):
        """Verify prompts exported from utils package."""
        from src.utils import (
            SENTIMENT_SYSTEM_PROMPT,
            DEPARTMENT_SYSTEM_PROMPT,
            RESPONSE_GENERATION_PROMPT
        )
        assert "sentiment" in SENTIMENT_SYSTEM_PROMPT.lower()
        assert "department" in DEPARTMENT_SYSTEM_PROMPT.lower()
        assert "assistant" in RESPONSE_GENERATION_PROMPT.lower()

    def test_sentiment_prompt_content(self):
        """Verify sentiment prompt contains key guidance."""
        from src.utils import SENTIMENT_SYSTEM_PROMPT

        assert "POSITIVE" in SENTIMENT_SYSTEM_PROMPT
        assert "NEUTRAL" in SENTIMENT_SYSTEM_PROMPT
        assert "NEGATIVE" in SENTIMENT_SYSTEM_PROMPT
        assert "ShopUNow" in SENTIMENT_SYSTEM_PROMPT

    def test_department_prompt_content(self):
        """Verify department prompt contains all departments."""
        from src.utils import DEPARTMENT_SYSTEM_PROMPT

        assert "HR" in DEPARTMENT_SYSTEM_PROMPT
        assert "IT_SUPPORT" in DEPARTMENT_SYSTEM_PROMPT or "it_support" in DEPARTMENT_SYSTEM_PROMPT
        assert "BILLING" in DEPARTMENT_SYSTEM_PROMPT or "billing" in DEPARTMENT_SYSTEM_PROMPT
        assert "SHIPPING" in DEPARTMENT_SYSTEM_PROMPT or "shipping" in DEPARTMENT_SYSTEM_PROMPT
        assert "UNKNOWN" in DEPARTMENT_SYSTEM_PROMPT or "unknown" in DEPARTMENT_SYSTEM_PROMPT


class TestClassificationPipelineImport:
    """Test classification pipeline imports."""

    def test_pipeline_import(self):
        """Verify ClassificationPipeline can be imported."""
        from src.pipelines import ClassificationPipeline
        assert ClassificationPipeline is not None

    def test_pipeline_from_module(self):
        """Verify import from module directly."""
        from src.pipelines.classification import ClassificationPipeline
        assert ClassificationPipeline is not None


class TestClassificationPipelineInit:
    """Test ClassificationPipeline initialization."""

    def test_pipeline_creation(self):
        """Verify pipeline can be created with mock provider."""
        from src.pipelines import ClassificationPipeline

        mock_llm = Mock()
        pipeline = ClassificationPipeline(llm_provider=mock_llm)

        assert pipeline is not None
        assert pipeline.llm == mock_llm

    def test_pipeline_has_methods(self):
        """Verify pipeline has required methods."""
        from src.pipelines import ClassificationPipeline

        mock_llm = Mock()
        pipeline = ClassificationPipeline(llm_provider=mock_llm)

        assert hasattr(pipeline, 'detect_sentiment')
        assert hasattr(pipeline, 'classify_department')
        assert hasattr(pipeline, 'classify')
        assert callable(pipeline.detect_sentiment)
        assert callable(pipeline.classify_department)
        assert callable(pipeline.classify)


class TestClassificationPipelineMocked:
    """Test classification pipeline with mocked LLM."""

    def test_detect_sentiment_calls_llm(self):
        """Verify detect_sentiment calls LLM correctly."""
        from src.pipelines import ClassificationPipeline
        from src.models import SentimentAnalysis, Sentiment

        mock_llm = Mock()
        mock_llm.generate_structured.return_value = SentimentAnalysis(
            sentiment=Sentiment.NEUTRAL,
            confidence=0.9,
            indicators=[]
        )

        pipeline = ClassificationPipeline(llm_provider=mock_llm)
        result = pipeline.detect_sentiment("How do I check my PTO?")

        mock_llm.generate_structured.assert_called_once()
        call_kwargs = mock_llm.generate_structured.call_args[1]
        assert "PTO" in call_kwargs['prompt']
        assert call_kwargs['response_model'] == SentimentAnalysis

    def test_classify_department_calls_llm(self):
        """Verify classify_department calls LLM correctly."""
        from src.pipelines import ClassificationPipeline
        from src.models import DepartmentClassification, Department, UserType

        mock_llm = Mock()
        mock_llm.generate_structured.return_value = DepartmentClassification(
            department=Department.HR,
            user_type=UserType.INTERNAL_EMPLOYEE,
            confidence=0.95,
            reasoning="PTO is an HR topic"
        )

        pipeline = ClassificationPipeline(llm_provider=mock_llm)
        result = pipeline.classify_department("How do I apply for PTO?")

        mock_llm.generate_structured.assert_called_once()
        call_kwargs = mock_llm.generate_structured.call_args[1]
        assert "PTO" in call_kwargs['prompt']
        assert call_kwargs['response_model'] == DepartmentClassification

    def test_classify_combines_results(self):
        """Verify classify runs both pipelines and combines results."""
        from src.pipelines import ClassificationPipeline
        from src.models import (
            SentimentAnalysis, DepartmentClassification, ClassificationResult,
            Sentiment, Department, UserType
        )

        mock_llm = Mock()
        mock_llm.generate_structured.side_effect = [
            SentimentAnalysis(
                sentiment=Sentiment.NEUTRAL,
                confidence=0.9,
                indicators=["standard question"]
            ),
            DepartmentClassification(
                department=Department.HR,
                user_type=UserType.INTERNAL_EMPLOYEE,
                confidence=0.95,
                reasoning="PTO is an HR topic"
            )
        ]

        pipeline = ClassificationPipeline(llm_provider=mock_llm)
        result = pipeline.classify("How do I apply for PTO?")

        assert mock_llm.generate_structured.call_count == 2
        assert isinstance(result, ClassificationResult)
        assert result.sentiment == Sentiment.NEUTRAL
        assert result.department == Department.HR
        assert result.user_type == UserType.INTERNAL_EMPLOYEE
        # Confidence should be min of both
        assert result.confidence == 0.9

    def test_classify_confidence_is_minimum(self):
        """Verify combined confidence is minimum of both."""
        from src.pipelines import ClassificationPipeline
        from src.models import (
            SentimentAnalysis, DepartmentClassification,
            Sentiment, Department, UserType
        )

        mock_llm = Mock()
        mock_llm.generate_structured.side_effect = [
            SentimentAnalysis(
                sentiment=Sentiment.NEGATIVE,
                confidence=0.75,  # Lower
                indicators=["frustration"]
            ),
            DepartmentClassification(
                department=Department.BILLING,
                user_type=UserType.EXTERNAL_CUSTOMER,
                confidence=0.95,  # Higher
                reasoning="Payment issue"
            )
        ]

        pipeline = ClassificationPipeline(llm_provider=mock_llm)
        result = pipeline.classify("Why was I overcharged?!")

        assert result.confidence == 0.75  # Should be the minimum

    def test_classify_reasoning_includes_both(self):
        """Verify combined reasoning includes sentiment and department."""
        from src.pipelines import ClassificationPipeline
        from src.models import (
            SentimentAnalysis, DepartmentClassification,
            Sentiment, Department, UserType
        )

        mock_llm = Mock()
        mock_llm.generate_structured.side_effect = [
            SentimentAnalysis(
                sentiment=Sentiment.POSITIVE,
                confidence=0.9,
                indicators=["thanks", "appreciate"]
            ),
            DepartmentClassification(
                department=Department.IT_SUPPORT,
                user_type=UserType.INTERNAL_EMPLOYEE,
                confidence=0.88,
                reasoning="Password reset is IT support"
            )
        ]

        pipeline = ClassificationPipeline(llm_provider=mock_llm)
        result = pipeline.classify("Thanks for helping me reset my password!")

        assert "positive" in result.reasoning.lower()
        assert "password" in result.reasoning.lower() or "IT" in result.reasoning


class TestSentimentAnalysisModel:
    """Test SentimentAnalysis model structure."""

    def test_sentiment_analysis_creation(self):
        """Verify SentimentAnalysis model can be created."""
        from src.models import SentimentAnalysis, Sentiment

        analysis = SentimentAnalysis(
            sentiment=Sentiment.NEGATIVE,
            confidence=0.85,
            indicators=["frustrated", "!!!", "unacceptable"]
        )

        assert analysis.sentiment == Sentiment.NEGATIVE
        assert analysis.confidence == 0.85
        assert len(analysis.indicators) == 3

    def test_sentiment_analysis_empty_indicators(self):
        """Verify SentimentAnalysis works with empty indicators."""
        from src.models import SentimentAnalysis, Sentiment

        analysis = SentimentAnalysis(
            sentiment=Sentiment.NEUTRAL,
            confidence=0.95,
            indicators=[]
        )

        assert analysis.indicators == []


class TestDepartmentClassificationModel:
    """Test DepartmentClassification model structure."""

    def test_department_classification_creation(self):
        """Verify DepartmentClassification model can be created."""
        from src.models import DepartmentClassification, Department, UserType

        classification = DepartmentClassification(
            department=Department.SHIPPING,
            user_type=UserType.EXTERNAL_CUSTOMER,
            confidence=0.92,
            reasoning="Order tracking is a shipping concern"
        )

        assert classification.department == Department.SHIPPING
        assert classification.user_type == UserType.EXTERNAL_CUSTOMER
        assert classification.confidence == 0.92

    def test_department_classification_all_departments(self):
        """Verify classification works for all departments."""
        from src.models import DepartmentClassification, Department, UserType

        for dept in Department:
            classification = DepartmentClassification(
                department=dept,
                user_type=UserType.UNKNOWN,
                confidence=0.5,
                reasoning=f"Test for {dept.value}"
            )
            assert classification.department == dept
