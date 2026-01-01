"""Phase 2 Tests: Enums and Core Models."""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError


class TestEnums:
    """Test enumeration types."""

    def test_department_enum_values(self):
        """Verify Department enum values."""
        from src.models.enums import Department

        assert Department.HR.value == "hr"
        assert Department.IT_SUPPORT.value == "it_support"
        assert Department.BILLING.value == "billing"
        assert Department.SHIPPING.value == "shipping"
        assert Department.UNKNOWN.value == "unknown"

    def test_department_internal_departments(self):
        """Verify internal departments list."""
        from src.models.enums import Department

        internal = Department.internal_departments()
        assert len(internal) == 2
        assert Department.HR in internal
        assert Department.IT_SUPPORT in internal

    def test_department_external_departments(self):
        """Verify external departments list."""
        from src.models.enums import Department

        external = Department.external_departments()
        assert len(external) == 2
        assert Department.BILLING in external
        assert Department.SHIPPING in external

    def test_department_valid_departments(self):
        """Verify valid departments excludes UNKNOWN."""
        from src.models.enums import Department

        valid = Department.valid_departments()
        assert len(valid) == 4
        assert Department.UNKNOWN not in valid

    def test_department_get_description(self):
        """Verify department descriptions."""
        from src.models.enums import Department

        hr_desc = Department.get_description(Department.HR)
        assert "Human Resources" in hr_desc
        assert "payroll" in hr_desc.lower() or "leave" in hr_desc.lower()

    def test_user_type_enum(self):
        """Verify UserType enum values."""
        from src.models.enums import UserType

        assert UserType.INTERNAL_EMPLOYEE.value == "internal_employee"
        assert UserType.EXTERNAL_CUSTOMER.value == "external_customer"
        assert UserType.UNKNOWN.value == "unknown"

    def test_sentiment_enum(self):
        """Verify Sentiment enum values."""
        from src.models.enums import Sentiment

        assert Sentiment.POSITIVE.value == "positive"
        assert Sentiment.NEUTRAL.value == "neutral"
        assert Sentiment.NEGATIVE.value == "negative"

    def test_route_decision_enum(self):
        """Verify RouteDecision enum values."""
        from src.models.enums import RouteDecision

        assert RouteDecision.RAG_PIPELINE.value == "rag_pipeline"
        assert RouteDecision.HUMAN_ESCALATION.value == "human_escalation"


class TestQAPairModel:
    """Test QAPair data model."""

    def test_qa_pair_creation(self, sample_qa_data):
        """Verify QAPair can be created from valid data."""
        from src.models import QAPair, Department, UserType

        qa = QAPair(
            id=sample_qa_data["id"],
            question=sample_qa_data["question"],
            answer=sample_qa_data["answer"],
            department=Department(sample_qa_data["department"]),
            user_type=UserType(sample_qa_data["user_type"]),
            keywords=sample_qa_data["keywords"]
        )

        assert qa.id == "hr_001"
        assert qa.department == Department.HR
        assert qa.user_type == UserType.INTERNAL_EMPLOYEE
        assert len(qa.keywords) == 3

    def test_qa_pair_keywords_lowercase(self):
        """Verify keywords are converted to lowercase."""
        from src.models import QAPair, Department, UserType

        qa = QAPair(
            id="test_001",
            question="Test question with sufficient length",
            answer="Test answer with sufficient length for validation",
            department=Department.HR,
            user_type=UserType.INTERNAL_EMPLOYEE,
            keywords=["PTO", "Time Off", "LEAVE"]
        )

        assert all(kw.islower() for kw in qa.keywords)
        assert "pto" in qa.keywords
        assert "time off" in qa.keywords

    def test_qa_pair_to_document_text(self):
        """Verify to_document_text method."""
        from src.models import QAPair, Department, UserType

        qa = QAPair(
            id="test_001",
            question="How do I reset my password?",
            answer="Contact IT Support at ext. 5555.",
            department=Department.IT_SUPPORT,
            user_type=UserType.INTERNAL_EMPLOYEE
        )

        doc_text = qa.to_document_text()
        assert "Question:" in doc_text
        assert "Answer:" in doc_text
        assert "reset my password" in doc_text
        assert "IT Support" in doc_text

    def test_qa_pair_validation_min_length(self):
        """Verify minimum length validation."""
        from src.models import QAPair, Department, UserType

        with pytest.raises(ValidationError):
            QAPair(
                id="test",
                question="Short",  # Too short
                answer="Also short",  # Too short
                department=Department.HR,
                user_type=UserType.INTERNAL_EMPLOYEE
            )


class TestClassificationModels:
    """Test classification-related models."""

    def test_classification_result(self):
        """Verify ClassificationResult model."""
        from src.models import ClassificationResult, Department, UserType, Sentiment

        result = ClassificationResult(
            sentiment=Sentiment.NEUTRAL,
            department=Department.HR,
            user_type=UserType.INTERNAL_EMPLOYEE,
            confidence=0.85,
            reasoning="Query mentions PTO and leave, typical HR topics"
        )

        assert result.sentiment == Sentiment.NEUTRAL
        assert result.department == Department.HR
        assert result.confidence == 0.85

    def test_classification_confidence_bounds(self):
        """Verify confidence must be between 0 and 1."""
        from src.models import ClassificationResult, Department, UserType, Sentiment

        with pytest.raises(ValidationError):
            ClassificationResult(
                sentiment=Sentiment.NEUTRAL,
                department=Department.HR,
                user_type=UserType.INTERNAL_EMPLOYEE,
                confidence=1.5,  # Invalid: > 1.0
                reasoning="Test"
            )


class TestRetrievalModels:
    """Test retrieval-related models."""

    def test_retrieved_document(self):
        """Verify RetrievedDocument model."""
        from src.models import RetrievedDocument, Department

        doc = RetrievedDocument(
            id="hr_001",
            content="Question: How to apply for PTO?\nAnswer: Use HR Portal.",
            question="How to apply for PTO?",
            answer="Use HR Portal.",
            department=Department.HR,
            similarity_score=0.92
        )

        assert doc.id == "hr_001"
        assert doc.similarity_score == 0.92
        assert doc.department == Department.HR

    def test_retrieval_result(self):
        """Verify RetrievalResult model."""
        from src.models import RetrievalResult, RetrievedDocument, Department

        doc = RetrievedDocument(
            id="hr_001",
            content="Test content",
            question="Test?",
            answer="Test answer with sufficient length.",
            department=Department.HR,
            similarity_score=0.85
        )

        result = RetrievalResult(
            query="How do I apply for PTO?",
            department_filter=Department.HR,
            documents=[doc],
            retrieval_count=1,
            threshold_used=0.5
        )

        assert result.has_results is True
        assert result.top_document is not None
        assert result.top_document.id == "hr_001"


class TestAPIModels:
    """Test API request/response models."""

    def test_query_request(self):
        """Verify QueryRequest model."""
        from src.models import QueryRequest

        request = QueryRequest(
            query="How do I check my PTO balance?",
            user_id="emp_12345"
        )

        assert request.query == "How do I check my PTO balance?"
        assert request.user_id == "emp_12345"

    def test_query_request_validation(self):
        """Verify QueryRequest minimum length validation."""
        from src.models import QueryRequest

        with pytest.raises(ValidationError):
            QueryRequest(query="Hi")  # Too short

    def test_query_response(self):
        """Verify QueryResponse model."""
        from src.models import QueryResponse, Department, Sentiment

        response = QueryResponse(
            query="How do I check my PTO balance?",
            answer="Log into the HR Portal to view your PTO balance.",
            department=Department.HR,
            sentiment=Sentiment.NEUTRAL,
            confidence=0.9,
            processing_time_ms=245.5
        )

        assert response.department == Department.HR
        assert response.was_escalated is False
        assert response.timestamp is not None

    def test_escalation_response(self):
        """Verify EscalationResponse model."""
        from src.models import EscalationResponse

        response = EscalationResponse(
            query="I want to speak to a manager!",
            reason="Negative sentiment detected"
        )

        assert "escalated" in response.message.lower()
        assert response.reason == "Negative sentiment detected"


class TestRoutingModels:
    """Test routing-related models."""

    def test_routing_decision(self):
        """Verify RoutingDecision model."""
        from src.models import RoutingDecision, ClassificationResult, RouteDecision
        from src.models import Department, UserType, Sentiment

        classification = ClassificationResult(
            sentiment=Sentiment.NEGATIVE,
            department=Department.BILLING,
            user_type=UserType.EXTERNAL_CUSTOMER,
            confidence=0.9,
            reasoning="Angry customer"
        )

        decision = RoutingDecision(
            route=RouteDecision.HUMAN_ESCALATION,
            classification=classification,
            reasoning="Negative sentiment requires human intervention"
        )

        assert decision.should_escalate is True
        assert decision.route == RouteDecision.HUMAN_ESCALATION
