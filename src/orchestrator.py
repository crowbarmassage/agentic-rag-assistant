"""Main orchestrator coordinating all pipeline components."""

import time
from typing import Optional, Union

from src.config import Settings, get_settings
from src.models import (
    QueryRequest,
    QueryResponse,
    EscalationResponse,
    ClassificationResult,
    RoutingDecision,
    RouteDecision
)
from src.providers import LLMProviderFactory
from src.providers import EmbeddingProviderFactory
from src.pipelines import (
    ClassificationPipeline,
    DynamicKRetriever,
    RetrievalPipeline,
    ResponseGenerator
)
from src.routing import QueryRouter
from src.vectorstore import ChromaDBClient


class ShopUNowOrchestrator:
    """
    Main orchestrator coordinating all pipeline components.

    Flow:
    1. Receive query
    2. Classify (sentiment + department)
    3. Route (RAG vs Human Escalation)
    4. If RAG: Retrieve → Generate Response
    5. If Escalation: Return escalation message
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize orchestrator with all components.

        Args:
            settings: Application settings (uses defaults if None)
        """
        self.settings = settings or get_settings()

        # Initialize LLM provider
        self.llm = LLMProviderFactory.create(
            provider=self.settings.llm_provider,
            model=self.settings.llm_model
        )
        print(f"[INIT] LLM: {self.llm.provider_name}")

        # Initialize embedding provider
        self.embedding = EmbeddingProviderFactory.create(
            provider=self.settings.embedding_provider,
            model=self.settings.embedding_model
        )
        print(f"[INIT] Embeddings: {self.embedding.provider_name} (dim={self.embedding.dimension})")

        # Initialize ChromaDB
        self.chroma = ChromaDBClient(
            persist_directory=self.settings.chroma_persist_dir,
            embedding_provider=self.embedding
        )
        doc_count = self.chroma.get_document_count()
        print(f"[INIT] ChromaDB: {doc_count} documents")

        # Initialize pipelines
        self.classifier = ClassificationPipeline(self.llm)

        self.retriever = DynamicKRetriever(
            chroma_client=self.chroma,
            min_threshold=self.settings.retrieval_min_threshold,
            max_k=self.settings.retrieval_max_k,
            drop_off_ratio=self.settings.retrieval_drop_off_ratio
        )
        self.retrieval_pipeline = RetrievalPipeline(self.retriever)

        self.generator = ResponseGenerator(self.llm)
        self.router = QueryRouter()

        print("[INIT] Orchestrator ready")

    def process_query(
        self,
        request: QueryRequest
    ) -> Union[QueryResponse, EscalationResponse]:
        """
        Process user query through the complete pipeline.

        Args:
            request: Incoming query request

        Returns:
            QueryResponse for successful RAG processing
            EscalationResponse for escalated queries
        """
        start_time = time.time()

        # Step 1: Classification
        print(f"[PROCESS] Classifying: {request.query[:50]}...")
        classification = self.classifier.classify(request.query)
        print(f"[PROCESS] Sentiment: {classification.sentiment.value}, Dept: {classification.department.value}")

        # Step 2: Routing
        routing_decision = self.router.route(classification)
        print(f"[PROCESS] Route: {routing_decision.route.value}")

        # Step 3: Handle based on route
        if routing_decision.should_escalate:
            return self._handle_escalation(request, classification, routing_decision)

        # Step 4: RAG Pipeline
        retrieval_result = self.retrieval_pipeline.retrieve(
            query=request.query,
            department=classification.department,
            user_type=classification.user_type
        )

        # Step 5: Generate Response
        response = self.generator.generate(
            query=request.query,
            retrieval=retrieval_result,
            classification=classification
        )

        processing_time = (time.time() - start_time) * 1000

        return QueryResponse(
            query=request.query,
            answer=response.answer,
            department=classification.department,
            sentiment=classification.sentiment,
            sources=response.sources,
            confidence=response.confidence,
            was_escalated=False,
            processing_time_ms=processing_time
        )

    def _handle_escalation(
        self,
        request: QueryRequest,
        classification: ClassificationResult,
        routing: RoutingDecision
    ) -> EscalationResponse:
        """Handle escalation to human support."""
        return EscalationResponse(
            query=request.query,
            reason=routing.reasoning,
            ticket_id=f"ESC-{int(time.time())}"
        )
