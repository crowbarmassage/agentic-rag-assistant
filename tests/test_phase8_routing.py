"""Phase 8 Tests: Routing & Orchestration."""

import pytest
from unittest.mock import Mock, MagicMock, patch


class TestQueryRouterImport:
    """Test QueryRouter imports."""

    def test_router_import(self):
        """Verify QueryRouter can be imported."""
        from src.routing import QueryRouter
        assert QueryRouter is not None

    def test_router_from_module(self):
        """Verify import from module directly."""
        from src.routing.router import QueryRouter
        assert QueryRouter is not None


class TestQueryRouterInit:
    """Test QueryRouter initialization."""

    def test_router_creation(self):
        """Verify router can be created."""
        from src.routing import QueryRouter

        router = QueryRouter()
        assert router is not None
        assert len(router.valid_departments) == 4  # HR, IT, Billing, Shipping


class TestQueryRouterRouting:
    """Test QueryRouter routing logic."""

    def test_route_negative_sentiment_escalates(self):
        """Verify negative sentiment triggers escalation."""
        from src.routing import QueryRouter
        from src.models import (
            ClassificationResult, RoutingDecision, RouteDecision,
            Sentiment, Department, UserType
        )

        router = QueryRouter()

        classification = ClassificationResult(
            sentiment=Sentiment.NEGATIVE,
            department=Department.BILLING,
            user_type=UserType.EXTERNAL_CUSTOMER,
            confidence=0.9,
            reasoning="Angry customer"
        )

        decision = router.route(classification)

        assert decision.route == RouteDecision.HUMAN_ESCALATION
        assert "negative sentiment" in decision.reasoning.lower()

    def test_route_unknown_department_escalates(self):
        """Verify unknown department triggers escalation."""
        from src.routing import QueryRouter
        from src.models import (
            ClassificationResult, RoutingDecision, RouteDecision,
            Sentiment, Department, UserType
        )

        router = QueryRouter()

        classification = ClassificationResult(
            sentiment=Sentiment.NEUTRAL,
            department=Department.UNKNOWN,
            user_type=UserType.UNKNOWN,
            confidence=0.5,
            reasoning="Could not classify"
        )

        decision = router.route(classification)

        assert decision.route == RouteDecision.HUMAN_ESCALATION
        assert "unknown" in decision.reasoning.lower()

    def test_route_neutral_valid_dept_to_rag(self):
        """Verify neutral sentiment with valid dept goes to RAG."""
        from src.routing import QueryRouter
        from src.models import (
            ClassificationResult, RoutingDecision, RouteDecision,
            Sentiment, Department, UserType
        )

        router = QueryRouter()

        classification = ClassificationResult(
            sentiment=Sentiment.NEUTRAL,
            department=Department.HR,
            user_type=UserType.INTERNAL_EMPLOYEE,
            confidence=0.95,
            reasoning="PTO question"
        )

        decision = router.route(classification)

        assert decision.route == RouteDecision.RAG_PIPELINE
        assert "rag" in decision.reasoning.lower()

    def test_route_positive_valid_dept_to_rag(self):
        """Verify positive sentiment with valid dept goes to RAG."""
        from src.routing import QueryRouter
        from src.models import (
            ClassificationResult, RouteDecision,
            Sentiment, Department, UserType
        )

        router = QueryRouter()

        classification = ClassificationResult(
            sentiment=Sentiment.POSITIVE,
            department=Department.SHIPPING,
            user_type=UserType.EXTERNAL_CUSTOMER,
            confidence=0.9,
            reasoning="Happy customer with question"
        )

        decision = router.route(classification)

        assert decision.route == RouteDecision.RAG_PIPELINE

    def test_should_escalate_helper(self):
        """Verify should_escalate helper method."""
        from src.routing import QueryRouter
        from src.models import ClassificationResult, Sentiment, Department, UserType

        router = QueryRouter()

        # Should escalate - negative
        negative = ClassificationResult(
            sentiment=Sentiment.NEGATIVE,
            department=Department.HR,
            user_type=UserType.INTERNAL_EMPLOYEE,
            confidence=0.9,
            reasoning="test"
        )
        assert router.should_escalate(negative) is True

        # Should escalate - unknown dept
        unknown = ClassificationResult(
            sentiment=Sentiment.NEUTRAL,
            department=Department.UNKNOWN,
            user_type=UserType.UNKNOWN,
            confidence=0.5,
            reasoning="test"
        )
        assert router.should_escalate(unknown) is True

        # Should NOT escalate
        normal = ClassificationResult(
            sentiment=Sentiment.NEUTRAL,
            department=Department.IT_SUPPORT,
            user_type=UserType.INTERNAL_EMPLOYEE,
            confidence=0.9,
            reasoning="test"
        )
        assert router.should_escalate(normal) is False


class TestRoutingDecisionModel:
    """Test RoutingDecision model properties."""

    def test_routing_decision_should_escalate(self):
        """Verify should_escalate property."""
        from src.models import (
            RoutingDecision, ClassificationResult, RouteDecision,
            Sentiment, Department, UserType
        )

        classification = ClassificationResult(
            sentiment=Sentiment.NEGATIVE,
            department=Department.BILLING,
            user_type=UserType.EXTERNAL_CUSTOMER,
            confidence=0.9,
            reasoning="test"
        )

        escalation = RoutingDecision(
            route=RouteDecision.HUMAN_ESCALATION,
            classification=classification,
            reasoning="Escalating"
        )
        assert escalation.should_escalate is True

        rag = RoutingDecision(
            route=RouteDecision.RAG_PIPELINE,
            classification=classification,
            reasoning="Using RAG"
        )
        assert rag.should_escalate is False


class TestOrchestratorImport:
    """Test Orchestrator imports."""

    def test_orchestrator_import(self):
        """Verify ShopUNowOrchestrator can be imported."""
        from src.orchestrator import ShopUNowOrchestrator
        assert ShopUNowOrchestrator is not None


class TestOrchestratorMocked:
    """Test Orchestrator with mocked dependencies."""

    @patch('src.orchestrator.LLMProviderFactory')
    @patch('src.orchestrator.EmbeddingProviderFactory')
    @patch('src.orchestrator.ChromaDBClient')
    def test_orchestrator_creation(
        self,
        mock_chroma_cls,
        mock_embed_factory,
        mock_llm_factory
    ):
        """Verify orchestrator can be created with mocks."""
        from src.orchestrator import ShopUNowOrchestrator
        from src.config import Settings

        # Setup mocks
        mock_llm = Mock()
        mock_llm.provider_name = "mock_llm"
        mock_llm_factory.create.return_value = mock_llm

        mock_embed = Mock()
        mock_embed.provider_name = "mock_embed"
        mock_embed.dimension = 384
        mock_embed_factory.create.return_value = mock_embed

        mock_chroma = Mock()
        mock_chroma.get_document_count.return_value = 100
        mock_chroma_cls.return_value = mock_chroma

        settings = Settings(
            llm_provider="openai",
            embedding_provider="sentence_transformers"
        )

        orchestrator = ShopUNowOrchestrator(settings=settings)

        assert orchestrator is not None
        assert orchestrator.llm == mock_llm
        assert orchestrator.embedding == mock_embed
        assert orchestrator.chroma == mock_chroma

    @patch('src.orchestrator.LLMProviderFactory')
    @patch('src.orchestrator.EmbeddingProviderFactory')
    @patch('src.orchestrator.ChromaDBClient')
    def test_orchestrator_process_escalation(
        self,
        mock_chroma_cls,
        mock_embed_factory,
        mock_llm_factory
    ):
        """Verify orchestrator handles escalation correctly."""
        from src.orchestrator import ShopUNowOrchestrator
        from src.config import Settings
        from src.models import (
            QueryRequest, EscalationResponse,
            SentimentAnalysis, DepartmentClassification,
            Sentiment, Department, UserType
        )

        # Setup mocks
        mock_llm = Mock()
        mock_llm.provider_name = "mock"
        mock_llm.generate_structured.side_effect = [
            SentimentAnalysis(
                sentiment=Sentiment.NEGATIVE,
                confidence=0.9,
                indicators=["angry"]
            ),
            DepartmentClassification(
                department=Department.BILLING,
                user_type=UserType.EXTERNAL_CUSTOMER,
                confidence=0.9,
                reasoning="billing issue"
            )
        ]
        mock_llm_factory.create.return_value = mock_llm

        mock_embed = Mock()
        mock_embed.provider_name = "mock"
        mock_embed.dimension = 384
        mock_embed_factory.create.return_value = mock_embed

        mock_chroma = Mock()
        mock_chroma.get_document_count.return_value = 0
        mock_chroma_cls.return_value = mock_chroma

        settings = Settings()
        orchestrator = ShopUNowOrchestrator(settings=settings)

        request = QueryRequest(query="This is terrible! I want my money back!")
        response = orchestrator.process_query(request)

        assert isinstance(response, EscalationResponse)
        assert "negative" in response.reason.lower()

    @patch('src.orchestrator.LLMProviderFactory')
    @patch('src.orchestrator.EmbeddingProviderFactory')
    @patch('src.orchestrator.ChromaDBClient')
    def test_orchestrator_process_rag(
        self,
        mock_chroma_cls,
        mock_embed_factory,
        mock_llm_factory
    ):
        """Verify orchestrator handles RAG queries correctly."""
        from src.orchestrator import ShopUNowOrchestrator
        from src.config import Settings
        from src.models import (
            QueryRequest, QueryResponse,
            SentimentAnalysis, DepartmentClassification,
            Sentiment, Department, UserType,
            RetrievedDocument
        )
        from src.providers.base import LLMResponse

        # Setup mocks
        mock_llm = Mock()
        mock_llm.provider_name = "mock"
        mock_llm.generate_structured.side_effect = [
            SentimentAnalysis(
                sentiment=Sentiment.NEUTRAL,
                confidence=0.95,
                indicators=[]
            ),
            DepartmentClassification(
                department=Department.HR,
                user_type=UserType.INTERNAL_EMPLOYEE,
                confidence=0.9,
                reasoning="PTO question"
            )
        ]
        mock_llm.generate.return_value = LLMResponse(
            content="You can check your PTO in the HR Portal.",
            model="test",
            usage={}
        )
        mock_llm_factory.create.return_value = mock_llm

        mock_embed = Mock()
        mock_embed.provider_name = "mock"
        mock_embed.dimension = 384
        mock_embed_factory.create.return_value = mock_embed

        mock_chroma = Mock()
        mock_chroma.get_document_count.return_value = 50
        mock_chroma.query.return_value = [
            RetrievedDocument(
                id="hr_001",
                content="Q: How to check PTO?\nA: Use HR Portal",
                question="How to check PTO?",
                answer="Use HR Portal",
                department=Department.HR,
                similarity_score=0.9
            )
        ]
        mock_chroma_cls.return_value = mock_chroma

        settings = Settings()
        orchestrator = ShopUNowOrchestrator(settings=settings)

        request = QueryRequest(query="How do I check my PTO balance?")
        response = orchestrator.process_query(request)

        assert isinstance(response, QueryResponse)
        assert response.was_escalated is False
        assert response.department == Department.HR
        assert "PTO" in response.answer or "HR" in response.answer


class TestConfigGetSettings:
    """Test config get_settings function."""

    def test_get_settings_returns_settings(self):
        """Verify get_settings returns Settings instance."""
        from src.config import get_settings, Settings

        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_cached(self):
        """Verify get_settings returns same instance."""
        from src.config import get_settings

        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
