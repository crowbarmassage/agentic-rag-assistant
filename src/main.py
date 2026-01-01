"""FastAPI application for ShopUNow Assistant."""

from dotenv import load_dotenv
load_dotenv()  # Load .env before any other imports

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
    print("=" * 50)
    print("ShopUNow AI Assistant Starting...")
    print("=" * 50)

    settings = get_settings()
    orchestrator = ShopUNowOrchestrator(settings)

    print("=" * 50)
    print("Ready to accept requests")
    print("=" * 50)

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
