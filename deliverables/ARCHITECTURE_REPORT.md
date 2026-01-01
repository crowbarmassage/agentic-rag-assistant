# ShopUNow AI Assistant - Architecture Report

> **Agentic RAG System for Retail Customer & Employee Support**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [LLM Usage & Reasoning](#3-llm-usage--reasoning)
4. [RAG Pipeline](#4-rag-pipeline)
5. [Routing Logic](#5-routing-logic)
6. [Code Walkthrough](#6-code-walkthrough)
7. [Configuration & Deployment](#7-configuration--deployment)
8. [Key Design Decisions](#8-key-design-decisions)

---

## 1. Executive Summary

### Project Goal

Build an intelligent AI assistant for ShopUNow, a retail company, that handles queries from:
- **Internal Employees**: HR questions, IT support requests
- **External Customers**: Billing inquiries, shipping/delivery issues

### Key Features Implemented

| Feature | Implementation |
|---------|---------------|
| 4 Departments | HR, IT Support, Billing, Shipping |
| 60 QA Pairs | 15 per department |
| Sentiment Analysis | LLM-based detection (positive/neutral/negative) |
| Department Classification | LLM-based routing with structured output |
| RAG Pipeline | ChromaDB + Dynamic K retrieval + LLM generation |
| Human Escalation | Automatic for negative sentiment or unknown queries |
| **Stretch Goal** | FastAPI REST API deployment |

### Technology Stack

| Component | Technology |
|-----------|------------|
| LLM | OpenAI GPT-4o-mini (swappable: Gemini, Groq) |
| Embeddings | Sentence Transformers all-MiniLM-L6-v2 (swappable: OpenAI, Cohere) |
| Vector Store | ChromaDB with persistent storage |
| API Framework | FastAPI with uvicorn |
| Data Validation | Pydantic v2 |

---

## 2. Architecture Overview

### High-Level Flow

```
                         USER QUERY
                              |
                              v
              +-------------------------------+
              |     CLASSIFICATION PIPELINE   |
              |  +----------+  +------------+ |
              |  | Sentiment|->| Department | |
              |  | Detector |  | Classifier | |
              |  +----------+  +------------+ |
              +-------------------------------+
                              |
                              v
              +-------------------------------+
              |           ROUTER              |
              |                               |
              |  Negative Sentiment?  ------> ESCALATE
              |  Unknown Department?  ------> ESCALATE
              |  Otherwise            ------> RAG
              +-------------------------------+
                     |                |
                     v                v
              +------------+   +--------------+
              |    RAG     |   |  ESCALATION  |
              | (Retrieve  |   |   RESPONSE   |
              |  + Generate)|   |              |
              +------------+   +--------------+
                     |
                     v
              +-------------------------------+
              |   RESPONSE + SOURCE CITATIONS |
              +-------------------------------+
```

### Component Diagram

```
src/
├── orchestrator.py        # Main coordinator - ties everything together
├── config.py              # Environment-based configuration
├── models/
│   ├── enums.py           # Department, Sentiment, UserType, RouteDecision
│   ├── data_models.py     # QAPair, ClassificationResult, RetrievalResult
│   └── api_models.py      # QueryRequest, QueryResponse, EscalationResponse
├── providers/
│   ├── base.py            # Abstract base classes
│   ├── llm_*.py           # OpenAI, Gemini, Groq implementations
│   ├── embedding_providers.py  # All embedding implementations
│   └── llm_factory.py     # Factory pattern for providers
├── pipelines/
│   ├── classification.py  # Sentiment + Department classification
│   ├── retrieval.py       # Dynamic K retrieval logic
│   └── generation.py      # Response generation
├── routing/
│   └── router.py          # Escalation decision logic
├── vectorstore/
│   ├── chroma_client.py   # ChromaDB wrapper
│   └── ingestion.py       # Data loading utilities
└── main.py                # FastAPI application
```

---

## 3. LLM Usage & Reasoning

### 3.1 Sentiment Detection

The system uses structured output from the LLM to detect sentiment:

```python
# Classification result structure
class SentimentAnalysis(BaseModel):
    sentiment: Sentiment  # POSITIVE, NEUTRAL, NEGATIVE
    confidence: float     # 0.0 - 1.0
    indicators: list[str] # Key phrases that influenced detection
```

**System Prompt Strategy:**
```
You are a sentiment analysis expert for ShopUNow.
Analyze the customer/employee query and determine the sentiment.

Guidelines:
- POSITIVE: Grateful, satisfied, complimentary
- NEUTRAL: Informational questions, standard requests
- NEGATIVE: Frustrated, angry, complaining

Be sensitive to subtle cues:
- Excessive punctuation (!!!) indicates frustration
- ALL CAPS indicates shouting
- Words like "terrible", "awful" indicate negative sentiment
```

**Why LLM-based?**
- Understands context and nuance better than rule-based systems
- Handles edge cases like sarcasm or mixed sentiment
- Provides confidence scores for downstream decision-making

### 3.2 Department Classification

```python
class DepartmentClassification(BaseModel):
    department: Department  # HR, IT_SUPPORT, BILLING, SHIPPING, UNKNOWN
    user_type: UserType     # INTERNAL_EMPLOYEE, EXTERNAL_CUSTOMER
    confidence: float
    reasoning: str          # Explanation for the classification
```

**System Prompt Strategy:**
```
You are a query router for ShopUNow.
Classify the user query into the appropriate department.

INTERNAL EMPLOYEE DEPARTMENTS:
1. HR: Leave requests, payroll, benefits, performance reviews
2. IT_SUPPORT: Hardware, software, password resets, VPN

EXTERNAL CUSTOMER DEPARTMENTS:
3. BILLING: Invoices, refunds, payment methods
4. SHIPPING: Order tracking, returns, delivery issues

UNKNOWN: Only if query doesn't fit ANY department
```

**Why Structured Output?**
- Guarantees valid enum values (no hallucinated departments)
- Forces the LLM to provide reasoning
- Enables programmatic downstream processing

### 3.3 Response Generation

After retrieval, the LLM generates a final response:

```python
prompt = f"""User Query: {query}

Retrieved Information:
[Source 1] Q: {doc1.question} A: {doc1.answer}
[Source 2] Q: {doc2.question} A: {doc2.answer}

Based on the above, provide a helpful response."""
```

**System Prompt Strategy:**
```
You are a helpful customer service assistant for ShopUNow.
Based on the retrieved information, provide a clear answer.

Guidelines:
- Be concise but thorough
- Use a friendly, professional tone
- Don't make up information not in the sources
- Acknowledge if information doesn't fully answer the question
```

---

## 4. RAG Pipeline

### 4.1 Vector Store Design

**Single Collection Strategy:**
- All 60 FAQ documents stored in one ChromaDB collection
- Rich metadata enables filtered retrieval by department

**Document Format:**
```
ID: hr_001
Content: "Question: How do I apply for PTO?\nAnswer: Visit the HR Portal..."
Metadata:
  - department: "hr"
  - user_type: "internal_employee"
  - question: "How do I apply for PTO?"
  - answer: "Visit the HR Portal..."
  - keywords: "pto,vacation,leave,time off"
```

**Why Single Collection?**
- Simpler to manage than 4 separate collections
- Metadata filtering is efficient in ChromaDB
- Allows cross-department retrieval if needed in future

### 4.2 Embedding Strategy

**Model:** Sentence Transformers `all-MiniLM-L6-v2`
- Dimension: 384
- Free, local, no API calls
- Fast inference

**Document Text:**
```
"Question: {question_text}\nAnswer: {answer_text}"
```

**Why Include Both Q&A?**
- Question captures user intent patterns
- Answer captures knowledge content
- Combined embedding improves both question-matching and answer similarity

### 4.3 Dynamic K Retrieval

Instead of fixed K documents, we adaptively select based on relevance:

```python
class DynamicKRetriever:
    def __init__(
        self,
        min_threshold=0.3,   # Minimum similarity to include
        max_k=10,            # Maximum candidates to fetch
        drop_off_ratio=0.7   # Stop if score drops by 30%+
    ):
        ...
```

**Algorithm:**
1. Fetch `max_k` candidates from ChromaDB
2. Filter by `min_threshold` (discard low-quality matches)
3. Apply drop-off detection: if `score[i+1] < score[i] * 0.7`, stop

**Example:**
```
Candidates: [0.65, 0.58, 0.52, 0.35, 0.20]
min_threshold=0.3, drop_off_ratio=0.7

0.65 -> Include (above threshold)
0.58 -> Include (0.58 >= 0.65 * 0.7 = 0.455)
0.52 -> Include (0.52 >= 0.58 * 0.7 = 0.406)
0.35 -> STOP (0.35 < 0.52 * 0.7 = 0.364)

Result: 3 documents returned
```

**Why Dynamic K?**
- Avoids including irrelevant documents when few matches exist
- Stops before quality degrades significantly
- Adapts to query difficulty (easy queries get more results)

### 4.4 Threshold Tuning

| Embedding Model | Recommended Threshold |
|-----------------|----------------------|
| all-MiniLM-L6-v2 | 0.3 |
| all-mpnet-base-v2 | 0.4 |
| OpenAI text-embedding-3-small | 0.5 |

**Why 0.3 for MiniLM?**
- MiniLM produces lower similarity scores (0.3-0.6 for relevant matches)
- Higher thresholds (0.5+) would reject valid results
- Empirically tested on FAQ retrieval

---

## 5. Routing Logic

### 5.1 Decision Rules

```python
class QueryRouter:
    def route(self, classification: ClassificationResult) -> RoutingDecision:
        # Rule 1: Negative sentiment -> ESCALATE
        if classification.sentiment == Sentiment.NEGATIVE:
            return RoutingDecision(
                route=RouteDecision.HUMAN_ESCALATION,
                reasoning="Negative sentiment detected"
            )

        # Rule 2: Unknown department -> ESCALATE
        if classification.department == Department.UNKNOWN:
            return RoutingDecision(
                route=RouteDecision.HUMAN_ESCALATION,
                reasoning="Query doesn't match any department"
            )

        # Default: RAG pipeline
        return RoutingDecision(
            route=RouteDecision.RAG_PIPELINE,
            reasoning="Processing via RAG"
        )
```

### 5.2 Routing Matrix

| Sentiment | Department | Route | Rationale |
|-----------|------------|-------|-----------|
| Positive | Valid | RAG | Happy user with question |
| Neutral | Valid | RAG | Standard inquiry |
| Negative | Valid | **ESCALATE** | Frustrated user needs human touch |
| Any | Unknown | **ESCALATE** | Can't route to department |

### 5.3 Escalation Response

```python
class EscalationResponse(BaseModel):
    query: str
    message: str = "Your query has been escalated to human support..."
    reason: str           # Why we escalated
    ticket_id: str        # For tracking (e.g., "ESC-1735737045")
    estimated_response_time: str = "24-48 hours"
```

**Why Not Try RAG for Negative Sentiment?**
- Frustrated users need empathy, not just information
- Automated responses can feel dismissive
- Human escalation improves customer satisfaction

---

## 6. Code Walkthrough

### 6.1 Main Entry Point: `ShopUNowOrchestrator`

```python
# src/orchestrator.py

class ShopUNowOrchestrator:
    def __init__(self, settings: Settings):
        # Initialize LLM provider (OpenAI, Gemini, or Groq)
        self.llm = LLMProviderFactory.create(
            provider=settings.llm_provider,
            model=settings.llm_model
        )

        # Initialize embedding provider
        self.embedding = EmbeddingProviderFactory.create(
            provider=settings.embedding_provider
        )

        # Initialize ChromaDB
        self.chroma = ChromaDBClient(
            persist_directory=settings.chroma_persist_dir,
            embedding_provider=self.embedding
        )

        # Initialize pipelines
        self.classifier = ClassificationPipeline(self.llm)
        self.retriever = DynamicKRetriever(self.chroma)
        self.router = QueryRouter()

    def process_query(self, request: QueryRequest):
        # Step 1: Classify sentiment + department
        classification = self.classifier.classify(request.query)

        # Step 2: Route decision
        routing = self.router.route(classification)

        # Step 3: Handle based on route
        if routing.should_escalate:
            return EscalationResponse(
                query=request.query,
                reason=routing.reasoning,
                ticket_id=f"ESC-{int(time.time())}"
            )

        # Step 4: RAG retrieval
        retrieval = self.retriever.retrieve(
            query=request.query,
            department=classification.department
        )

        # Step 5: Generate response
        response = self._generate_response(
            query=request.query,
            classification=classification,
            retrieval=retrieval
        )

        return QueryResponse(
            query=request.query,
            answer=response.answer,
            department=classification.department,
            sources=response.sources,
            confidence=response.confidence
        )
```

### 6.2 Classification Pipeline

```python
# src/pipelines/classification.py

class ClassificationPipeline:
    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm = llm_provider

    def classify(self, query: str) -> ClassificationResult:
        # Two-step classification
        sentiment = self.detect_sentiment(query)
        department = self.classify_department(query)

        return ClassificationResult(
            sentiment=sentiment.sentiment,
            department=department.department,
            user_type=department.user_type,
            confidence=min(sentiment.confidence, department.confidence),
            reasoning=f"Sentiment: {sentiment.indicators}. Dept: {department.reasoning}"
        )

    def detect_sentiment(self, query: str) -> SentimentAnalysis:
        return self.llm.generate_structured(
            prompt=f'Analyze sentiment: "{query}"',
            response_model=SentimentAnalysis,
            system_prompt=SENTIMENT_SYSTEM_PROMPT
        )

    def classify_department(self, query: str) -> DepartmentClassification:
        return self.llm.generate_structured(
            prompt=f'Classify department: "{query}"',
            response_model=DepartmentClassification,
            system_prompt=DEPARTMENT_SYSTEM_PROMPT
        )
```

### 6.3 ChromaDB Client

```python
# src/vectorstore/chroma_client.py

class ChromaDBClient:
    def query(
        self,
        query_text: str,
        department: Department = None,
        n_results: int = 10,
        score_threshold: float = None
    ) -> list[RetrievedDocument]:
        # Build metadata filter
        where_clause = None
        if department and department != Department.UNKNOWN:
            where_clause = {"department": {"$eq": department.value}}

        # Embed query
        query_embedding = self.embedding_provider.embed_text(query_text)

        # Execute filtered search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )

        # Convert distances to similarity scores
        # ChromaDB returns L2 distance; we use: similarity = 1 - distance
        return [
            RetrievedDocument(
                id=results['ids'][0][i],
                similarity_score=1 - results['distances'][0][i],
                question=results['metadatas'][0][i]['question'],
                answer=results['metadatas'][0][i]['answer']
            )
            for i in range(len(results['ids'][0]))
            if score_threshold is None or (1 - results['distances'][0][i]) >= score_threshold
        ]
```

### 6.4 FastAPI Application

```python
# src/main.py

from fastapi import FastAPI
from src.orchestrator import ShopUNowOrchestrator

app = FastAPI(title="ShopUNow AI Assistant")
orchestrator: ShopUNowOrchestrator = None

@app.post("/query")
async def process_query(request: QueryRequest):
    response = orchestrator.process_query(request)
    return response

@app.get("/health")
async def health_check(deep: bool = False):
    components = {"api": "healthy"}
    if deep:
        components["chromadb"] = f"healthy ({orchestrator.chroma.get_document_count()} docs)"
        components["llm"] = f"healthy ({orchestrator.llm.provider_name})"
    return {"status": "healthy", "components": components}
```

---

## 7. Configuration & Deployment

### 7.1 Environment Variables

```bash
# .env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2
OPENAI_API_KEY=sk-...

CHROMA_PERSIST_DIR=./data/chroma_db
RETRIEVAL_MIN_THRESHOLD=0.3
RETRIEVAL_MAX_K=10
RETRIEVAL_DROP_OFF_RATIO=0.7

API_HOST=0.0.0.0
API_PORT=8000
```

### 7.2 Running the API

```bash
# Install dependencies
pip install -r requirements.txt

# Generate FAQ data
python datagen/generate_faqs_standalone.py

# Ingest into ChromaDB
python -c "from src.vectorstore import ingest_faqs; ingest_faqs(data_dir='./data/raw', reset_collection=True)"

# Start the API
python run.py
```

### 7.3 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Process user query |
| `/health` | GET | Basic health check |
| `/health?deep=true` | GET | Deep health check |
| `/departments` | GET | List departments |
| `/docs` | GET | Swagger UI |

---

## 8. Key Design Decisions

### 8.1 Provider Abstraction

**Decision:** Abstract LLM and embedding providers behind interfaces

**Rationale:**
- Swap providers without code changes (OpenAI -> Gemini)
- Test with cheaper/faster models, deploy with production models
- Future-proof for new providers

### 8.2 Single Collection with Metadata

**Decision:** One ChromaDB collection with department metadata filtering

**Rationale:**
- Simpler than managing 4 collections
- Efficient metadata filtering in ChromaDB
- Enables cross-department search if needed

### 8.3 Two-Step Classification

**Decision:** Separate sentiment and department classification

**Rationale:**
- Each task has distinct prompts/requirements
- Easier to debug and tune independently
- Can short-circuit on negative sentiment before classifying department

### 8.4 Dynamic K vs Fixed K

**Decision:** Adaptive document selection based on relevance

**Rationale:**
- Fixed K includes irrelevant docs when few matches exist
- Dynamic K improves response quality
- Drop-off detection prevents quality degradation

### 8.5 Escalation-First for Negative Sentiment

**Decision:** Always escalate negative sentiment, even if RAG could answer

**Rationale:**
- Customer satisfaction priority
- Frustrated users need human empathy
- Reduces risk of automated response making things worse

---

## Appendix: Quick Reference

### File Locations

| What | Where |
|------|-------|
| Main orchestrator | `src/orchestrator.py` |
| Classification | `src/pipelines/classification.py` |
| Retrieval | `src/pipelines/retrieval.py` |
| Routing | `src/routing/router.py` |
| ChromaDB wrapper | `src/vectorstore/chroma_client.py` |
| FastAPI app | `src/main.py` |
| FAQ data | `data/raw/*.json` |
| Vector store | `data/chroma_db/` |

### Test Commands

```bash
# Run all tests
pytest

# Test specific phase
pytest tests/test_phase8_routing.py -v

# Run with coverage
pytest --cov=src --cov-report=html
```

---

*End of Architecture Report*
