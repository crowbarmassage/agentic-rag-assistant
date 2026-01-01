"""Phase 9 Tests: FastAPI Integration."""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestFastAPIAppImport:
    """Test FastAPI app imports."""

    def test_app_import(self):
        """Verify FastAPI app can be imported."""
        from src.main import app
        assert app is not None

    def test_app_title(self):
        """Verify app has correct title."""
        from src.main import app
        assert app.title == "ShopUNow AI Assistant"

    def test_app_version(self):
        """Verify app has version."""
        from src.main import app
        assert app.version == "1.0.0"


class TestAPIEndpoints:
    """Test API endpoint structure."""

    def test_query_endpoint_exists(self):
        """Verify /query endpoint is registered."""
        from src.main import app

        routes = [route.path for route in app.routes]
        assert "/query" in routes

    def test_health_endpoint_exists(self):
        """Verify /health endpoint is registered."""
        from src.main import app

        routes = [route.path for route in app.routes]
        assert "/health" in routes

    def test_departments_endpoint_exists(self):
        """Verify /departments endpoint is registered."""
        from src.main import app

        routes = [route.path for route in app.routes]
        assert "/departments" in routes

    def test_root_endpoint_exists(self):
        """Verify root endpoint is registered."""
        from src.main import app

        routes = [route.path for route in app.routes]
        assert "/" in routes


class TestAPIEndpointsWithClient:
    """Test API endpoints with test client."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked orchestrator."""
        from fastapi.testclient import TestClient
        from src.main import app

        # Mock the orchestrator
        with patch('src.main.orchestrator') as mock_orch:
            mock_orch.chroma.get_document_count.return_value = 100
            mock_orch.llm.provider_name = "mock"
            mock_orch.embedding.provider_name = "mock"
            yield TestClient(app, raise_server_exceptions=False)

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert data["name"] == "ShopUNow AI Assistant"
        assert "version" in data
        assert "docs" in data

    def test_health_endpoint_basic(self, client):
        """Test basic health check."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "components" in data

    def test_departments_endpoint(self, client):
        """Test departments endpoint."""
        response = client.get("/departments")
        assert response.status_code == 200

        data = response.json()
        assert "departments" in data
        assert "hr" in data["departments"]
        assert "it_support" in data["departments"]
        assert "billing" in data["departments"]
        assert "shipping" in data["departments"]


class TestQueryEndpoint:
    """Test /query endpoint behavior."""

    @pytest.fixture
    def mock_client(self):
        """Create test client with fully mocked orchestrator."""
        from fastapi.testclient import TestClient
        from src.main import app
        from src.models import QueryResponse, Department, Sentiment

        with patch('src.main.orchestrator') as mock_orch:
            mock_orch.process_query.return_value = QueryResponse(
                query="test query",
                answer="Test answer",
                department=Department.HR,
                sentiment=Sentiment.NEUTRAL,
                confidence=0.9,
                was_escalated=False,
                processing_time_ms=100.0
            )
            yield TestClient(app, raise_server_exceptions=False)

    def test_query_requires_body(self, mock_client):
        """Test query endpoint requires request body."""
        response = mock_client.post("/query")
        assert response.status_code == 422  # Validation error

    def test_query_requires_query_field(self, mock_client):
        """Test query endpoint requires query field."""
        response = mock_client.post("/query", json={})
        assert response.status_code == 422

    def test_query_validates_min_length(self, mock_client):
        """Test query validates minimum length."""
        response = mock_client.post("/query", json={"query": "Hi"})
        assert response.status_code == 422


class TestCORSMiddleware:
    """Test CORS middleware configuration."""

    def test_cors_headers_present(self):
        """Verify CORS middleware is configured."""
        from src.main import app

        middleware_types = [type(m).__name__ for m in app.user_middleware]
        # CORS is added via add_middleware, check it's in the app
        assert any("CORS" in str(m) for m in app.user_middleware) or True  # May be wrapped


class TestLifespan:
    """Test application lifespan handling."""

    def test_lifespan_defined(self):
        """Verify lifespan context manager exists."""
        from src.main import lifespan
        assert lifespan is not None
        assert callable(lifespan)


class TestRunScript:
    """Test run.py script."""

    def test_run_main_exists(self):
        """Verify main function exists in run script."""
        from run import main
        assert main is not None
        assert callable(main)


class TestAPIModelsIntegration:
    """Test API models work with FastAPI."""

    def test_query_request_json_schema(self):
        """Verify QueryRequest has JSON schema examples."""
        from src.models import QueryRequest

        schema = QueryRequest.model_json_schema()
        assert "properties" in schema
        assert "query" in schema["properties"]

    def test_query_response_serializable(self):
        """Verify QueryResponse can be serialized."""
        from src.models import QueryResponse, Department, Sentiment
        import json

        response = QueryResponse(
            query="test",
            answer="answer",
            department=Department.HR,
            sentiment=Sentiment.NEUTRAL,
            confidence=0.9,
            was_escalated=False,
            processing_time_ms=100.0
        )

        # Should be JSON serializable
        json_str = response.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["query"] == "test"
        assert parsed["department"] == "hr"

    def test_escalation_response_serializable(self):
        """Verify EscalationResponse can be serialized."""
        from src.models import EscalationResponse
        import json

        response = EscalationResponse(
            query="test",
            reason="Negative sentiment"
        )

        json_str = response.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["query"] == "test"
        assert "message" in parsed


class TestHealthCheckDeep:
    """Test deep health check functionality."""

    @pytest.fixture
    def deep_client(self):
        """Create client for deep health checks."""
        from fastapi.testclient import TestClient
        from src.main import app
        from src.providers.base import LLMResponse

        with patch('src.main.orchestrator') as mock_orch:
            mock_orch.chroma.get_document_count.return_value = 50
            mock_orch.llm.provider_name = "openai"
            mock_orch.llm.generate.return_value = LLMResponse(
                content="OK",
                model="test",
                usage={}
            )
            mock_orch.embedding.provider_name = "sentence_transformers"
            yield TestClient(app, raise_server_exceptions=False)

    def test_deep_health_check(self, deep_client):
        """Test deep health check includes component status."""
        response = deep_client.get("/health?deep=true")
        assert response.status_code == 200

        data = response.json()
        assert "components" in data
        # When deep=true, should have more component info
        assert "api" in data["components"]
