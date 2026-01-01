# FUTURE_FEATURES.md — ShopUNow Post-MVP Enhancement Roadmap

## Document Metadata
- **Project**: Agentic AI Assistant for ShopUNow (Retail Company)
- **Version**: 1.0
- **Last Updated**: 2025-01-01
- **Status**: Planning Document for Future Iterations

---

## Table of Contents
1. [Stretch Goals from Assignment](#1-stretch-goals-from-assignment)
2. [Conversational Memory System](#2-conversational-memory-system)
3. [Interactive Human Escalation](#3-interactive-human-escalation)
4. [Extended Routing & Departments](#4-extended-routing--departments)
5. [Production Deployment](#5-production-deployment)
6. [Advanced RAG Enhancements](#6-advanced-rag-enhancements)
7. [Analytics & Monitoring](#7-analytics--monitoring)
8. [Multi-Modal Capabilities](#8-multi-modal-capabilities)
9. [Security & Compliance](#9-security--compliance)
10. [Integration Ecosystem](#10-integration-ecosystem)

---

## 1. Stretch Goals from Assignment

The capstone assignment defines four stretch goals. This section maps each to detailed implementation plans.

### 1.1 Stretch Goal Matrix

| Goal | Description | Complexity | Priority |
|------|-------------|------------|----------|
| Advanced Option 1 | Multi-user conversational memory | High | ⭐⭐⭐ |
| Advanced Option 2 | Interactive human escalation form | Medium | ⭐⭐ |
| Advanced Option 3 | Extended departments & routes | Low-Medium | ⭐ |
| Advanced Option 4 | API deployment (FastAPI) | Medium | ✅ Done in MVP |

### 1.2 Recommended Implementation Order

```
Phase 1 (MVP):        API Deployment ✅
Phase 2 (Week 1):     Extended Departments (easiest win)
Phase 3 (Week 2):     Interactive Escalation Form
Phase 4 (Week 3-4):   Conversational Memory System
```

---

## 2. Conversational Memory System

**Assignment Stretch Goal**: Advanced Option 1 — Multi-user conversational Agentic system with memory management.

### 2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        MEMORY LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Session    │  │   User       │  │   Entity     │           │
│  │   Memory     │  │   Memory     │  │   Memory     │           │
│  │  (Short-term)│  │  (Long-term) │  │  (Extracted) │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│         │                 │                 │                    │
│         └─────────────────┼─────────────────┘                    │
│                           │                                      │
│                    ┌──────▼──────┐                               │
│                    │   Memory    │                               │
│                    │   Manager   │                               │
│                    └─────────────┘                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Memory Types

#### Session Memory (Short-term)
- **Scope**: Current conversation session
- **Storage**: In-memory (Redis for production)
- **TTL**: 30 minutes of inactivity
- **Content**: Recent messages, current context, active ticket IDs

```python
class SessionMemory(BaseModel):
    session_id: str
    user_id: str
    messages: list[Message]  # Last N messages
    current_department: Optional[Department]
    active_ticket_id: Optional[str]
    created_at: datetime
    last_activity: datetime
    
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime
    metadata: dict  # department, sentiment, sources used
```

#### User Memory (Long-term)
- **Scope**: Persistent across sessions
- **Storage**: PostgreSQL or MongoDB
- **Content**: Preferences, past issues, interaction history summary

```python
class UserMemory(BaseModel):
    user_id: str
    user_type: UserType  # employee or customer
    name: Optional[str]
    email: Optional[str]
    preferences: UserPreferences
    issue_history: list[IssueSummary]
    first_interaction: datetime
    total_interactions: int
    
class UserPreferences(BaseModel):
    preferred_contact_method: str
    language: str = "en"
    timezone: str = "UTC"
    
class IssueSummary(BaseModel):
    ticket_id: str
    department: Department
    summary: str
    resolved: bool
    created_at: datetime
```

#### Entity Memory (Extracted)
- **Scope**: Extracted entities from conversations
- **Storage**: Linked to user memory
- **Content**: Order numbers, employee IDs, product names mentioned

```python
class EntityMemory(BaseModel):
    user_id: str
    entities: dict[str, list[ExtractedEntity]]
    
class ExtractedEntity(BaseModel):
    entity_type: str  # order_id, product_name, date, etc.
    value: str
    confidence: float
    extracted_from_session: str
    timestamp: datetime
```

### 2.3 Memory Manager Implementation

```python
class MemoryManager:
    """Manages all memory types for the assistant."""
    
    def __init__(
        self,
        session_store: SessionStore,  # Redis or in-memory
        user_store: UserStore,        # PostgreSQL/MongoDB
        max_session_messages: int = 10,
        session_ttl_minutes: int = 30
    ):
        self.session_store = session_store
        self.user_store = user_store
        self.max_messages = max_session_messages
        self.session_ttl = session_ttl_minutes
    
    async def get_context(
        self,
        user_id: str,
        session_id: str
    ) -> ConversationContext:
        """Get full context for a conversation."""
        session = await self.session_store.get(session_id)
        user = await self.user_store.get(user_id)
        
        return ConversationContext(
            session=session,
            user=user,
            recent_messages=session.messages[-self.max_messages:] if session else [],
            relevant_history=self._get_relevant_history(user, session)
        )
    
    async def update_session(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        metadata: dict
    ):
        """Update session with new exchange."""
        session = await self.session_store.get(session_id)
        
        session.messages.append(Message(
            role="user",
            content=user_message,
            timestamp=datetime.utcnow(),
            metadata={}
        ))
        session.messages.append(Message(
            role="assistant", 
            content=assistant_response,
            timestamp=datetime.utcnow(),
            metadata=metadata
        ))
        
        # Trim to max messages
        if len(session.messages) > self.max_messages * 2:
            session.messages = session.messages[-self.max_messages * 2:]
        
        session.last_activity = datetime.utcnow()
        await self.session_store.set(session_id, session)
    
    def _get_relevant_history(
        self,
        user: UserMemory,
        session: SessionMemory
    ) -> list[IssueSummary]:
        """Get history relevant to current conversation."""
        if not user or not session:
            return []
        
        # Filter to same department if known
        if session.current_department:
            return [
                issue for issue in user.issue_history
                if issue.department == session.current_department
            ][-3:]  # Last 3 relevant issues
        
        return user.issue_history[-5:]  # Last 5 any department
```

### 2.4 Context-Aware Prompt Enhancement

```python
def build_context_prompt(context: ConversationContext) -> str:
    """Build context section for LLM prompt."""
    parts = []
    
    # User info
    if context.user:
        parts.append(f"User: {context.user.name or 'Unknown'}")
        parts.append(f"User Type: {context.user.user_type.value}")
        parts.append(f"Previous Interactions: {context.user.total_interactions}")
    
    # Relevant history
    if context.relevant_history:
        parts.append("\nRelevant Past Issues:")
        for issue in context.relevant_history:
            status = "✓ Resolved" if issue.resolved else "○ Open"
            parts.append(f"  - [{status}] {issue.summary}")
    
    # Recent conversation
    if context.recent_messages:
        parts.append("\nRecent Conversation:")
        for msg in context.recent_messages[-4:]:  # Last 2 exchanges
            role = "User" if msg.role == "user" else "Assistant"
            parts.append(f"  {role}: {msg.content[:100]}...")
    
    return "\n".join(parts)
```

### 2.5 Implementation Steps

| Step | Task | Effort |
|------|------|--------|
| 2.5.1 | Define Pydantic models for all memory types | 2 hrs |
| 2.5.2 | Implement in-memory session store (dict-based) | 2 hrs |
| 2.5.3 | Implement MemoryManager class | 3 hrs |
| 2.5.4 | Update Orchestrator to use MemoryManager | 2 hrs |
| 2.5.5 | Add context injection to prompts | 2 hrs |
| 2.5.6 | Add session/user ID to API endpoints | 1 hr |
| 2.5.7 | Test multi-turn conversations | 2 hrs |
| 2.5.8 | (Optional) Add Redis for production session store | 3 hrs |
| 2.5.9 | (Optional) Add PostgreSQL for user persistence | 4 hrs |

**Total Effort**: 14-21 hours

---

## 3. Interactive Human Escalation

**Assignment Stretch Goal**: Advanced Option 2 — Form-based user input with optional email/WhatsApp integration.

### 3.1 Escalation Flow

```
┌─────────────────┐
│  Query Routed   │
│  to Escalation  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Display Form   │
│  (Name, Email,  │
│   Phone, Notes) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Validate &     │
│  Create Ticket  │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│ Email  │ │WhatsApp│
│Notif.  │ │ Notif. │
└────────┘ └────────┘
         │
         ▼
┌─────────────────┐
│  Confirmation   │
│  to User        │
└─────────────────┘
```

### 3.2 Data Models

```python
class EscalationFormData(BaseModel):
    """Data collected from escalation form."""
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    phone: Optional[str] = Field(None, pattern=r'^\+?[\d\s-]{10,}$')
    preferred_contact: Literal["email", "phone", "whatsapp"] = "email"
    issue_summary: str = Field(..., min_length=10, max_length=1000)
    urgency: Literal["low", "medium", "high"] = "medium"
    
class EscalationTicket(BaseModel):
    """Created escalation ticket."""
    ticket_id: str
    status: Literal["created", "assigned", "in_progress", "resolved"]
    form_data: EscalationFormData
    original_query: str
    classification: ClassificationResult
    created_at: datetime
    assigned_to: Optional[str] = None
    resolution: Optional[str] = None

class EscalationConfirmation(BaseModel):
    """Response after form submission."""
    ticket_id: str
    message: str
    estimated_response_time: str
    next_steps: list[str]
    confirmation_sent_via: list[str]  # ["email", "whatsapp"]
```

### 3.3 API Endpoints

```python
@app.post("/escalation/form", tags=["Escalation"])
async def submit_escalation_form(
    form_data: EscalationFormData,
    session_id: Optional[str] = None
) -> EscalationConfirmation:
    """
    Submit escalation form and create support ticket.
    
    Optionally sends confirmation via email/WhatsApp.
    """
    # Create ticket
    ticket = escalation_service.create_ticket(form_data, session_id)
    
    # Send notifications
    confirmations_sent = []
    
    if form_data.email:
        await email_service.send_confirmation(form_data.email, ticket)
        confirmations_sent.append("email")
    
    if form_data.preferred_contact == "whatsapp" and form_data.phone:
        await whatsapp_service.send_confirmation(form_data.phone, ticket)
        confirmations_sent.append("whatsapp")
    
    return EscalationConfirmation(
        ticket_id=ticket.ticket_id,
        message=f"Your request has been received. Ticket #{ticket.ticket_id}",
        estimated_response_time=get_response_time(form_data.urgency),
        next_steps=[
            "A support representative will review your request",
            f"You will be contacted via {form_data.preferred_contact}",
            "Check your email for ticket confirmation"
        ],
        confirmation_sent_via=confirmations_sent
    )


@app.get("/escalation/{ticket_id}", tags=["Escalation"])
async def get_ticket_status(ticket_id: str) -> EscalationTicket:
    """Get status of an escalation ticket."""
    return escalation_service.get_ticket(ticket_id)
```

### 3.4 Email Integration (SendGrid/SMTP)

```python
class EmailService:
    """Email notification service."""
    
    def __init__(self, smtp_config: SMTPConfig):
        self.config = smtp_config
    
    async def send_confirmation(
        self,
        to_email: str,
        ticket: EscalationTicket
    ):
        """Send escalation confirmation email."""
        subject = f"ShopUNow Support Ticket #{ticket.ticket_id}"
        
        body = f"""
        Dear {ticket.form_data.name},
        
        Thank you for contacting ShopUNow support.
        
        Your request has been received and assigned ticket number: {ticket.ticket_id}
        
        Issue Summary:
        {ticket.form_data.issue_summary}
        
        Priority: {ticket.form_data.urgency.upper()}
        Estimated Response Time: {get_response_time(ticket.form_data.urgency)}
        
        A support representative will contact you via {ticket.form_data.preferred_contact}.
        
        Best regards,
        ShopUNow Support Team
        """
        
        await self._send_email(to_email, subject, body)
```

### 3.5 WhatsApp Integration (Twilio)

```python
class WhatsAppService:
    """WhatsApp notification via Twilio."""
    
    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        from twilio.rest import Client
        self.client = Client(account_sid, auth_token)
        self.from_number = from_number  # whatsapp:+14155238886
    
    async def send_confirmation(
        self,
        to_phone: str,
        ticket: EscalationTicket
    ):
        """Send WhatsApp confirmation message."""
        message = f"""
🛍️ *ShopUNow Support*

Your ticket #{ticket.ticket_id} has been created.

📋 *Summary:* {ticket.form_data.issue_summary[:100]}...
⏰ *Response Time:* {get_response_time(ticket.form_data.urgency)}

We'll contact you soon!
        """
        
        self.client.messages.create(
            body=message,
            from_=f"whatsapp:{self.from_number}",
            to=f"whatsapp:{to_phone}"
        )
```

### 3.6 Implementation Steps

| Step | Task | Effort |
|------|------|--------|
| 3.6.1 | Define form and ticket Pydantic models | 1 hr |
| 3.6.2 | Create EscalationService class | 2 hrs |
| 3.6.3 | Add form submission API endpoint | 1 hr |
| 3.6.4 | Add ticket status endpoint | 1 hr |
| 3.6.5 | Implement email service (SMTP/SendGrid) | 2 hrs |
| 3.6.6 | (Optional) Implement WhatsApp via Twilio | 2 hrs |
| 3.6.7 | Create simple frontend form (HTML/React) | 3 hrs |
| 3.6.8 | Test end-to-end flow | 2 hrs |

**Total Effort**: 10-14 hours

---

## 4. Extended Routing & Departments

**Assignment Stretch Goal**: Advanced Option 3 — More departments and custom workflows.

### 4.1 Additional Department Suggestions

| Department | User Type | Description |
|------------|-----------|-------------|
| **Product Support** | External | Product questions, compatibility, usage guides |
| **Returns & Exchanges** | External | Return policy, exchange process, store credit |
| **Loyalty Program** | External | Points balance, rewards, membership tiers |
| **Finance** | Internal | Budget requests, expense reports, reimbursements |
| **Legal** | Internal | Contract questions, compliance, NDA requests |
| **Security** | Internal | Access requests, incident reporting, badge issues |

### 4.2 Custom Workflow: Returns & Exchanges

```python
class ReturnWorkflow:
    """Specialized workflow for returns processing."""
    
    def __init__(self, llm: BaseLLMProvider, order_api: OrderAPIClient):
        self.llm = llm
        self.order_api = order_api
    
    async def process(self, query: str, user_id: str) -> ReturnResponse:
        # Step 1: Extract order information
        order_info = await self._extract_order_info(query)
        
        # Step 2: Validate order exists and is returnable
        if order_info.order_id:
            order = await self.order_api.get_order(order_info.order_id)
            eligibility = self._check_return_eligibility(order)
        else:
            eligibility = None
        
        # Step 3: Generate response based on eligibility
        if eligibility and eligibility.is_eligible:
            return await self._generate_return_instructions(order, eligibility)
        elif eligibility and not eligibility.is_eligible:
            return await self._generate_ineligibility_response(order, eligibility)
        else:
            return await self._request_order_information()
    
    async def _extract_order_info(self, query: str) -> OrderInfo:
        """Use LLM to extract order number from query."""
        prompt = f"""Extract order information from this query:
        
Query: {query}

Return JSON with:
- order_id: string or null
- product_mentioned: string or null
- reason_for_return: string or null
"""
        return self.llm.generate_structured(prompt, OrderInfo)
```

### 4.3 Department Configuration System

```python
# config/departments.yaml
departments:
  hr:
    name: "Human Resources"
    user_type: "internal_employee"
    enabled: true
    workflow: "standard_rag"
    vector_collection: "hr_faqs"
    escalation_email: "hr@shopunow.com"
    
  returns:
    name: "Returns & Exchanges"
    user_type: "external_customer"
    enabled: true
    workflow: "returns_workflow"  # Custom workflow
    vector_collection: "returns_faqs"
    requires_order_lookup: true
    escalation_email: "returns@shopunow.com"
    
  loyalty:
    name: "Loyalty Program"
    user_type: "external_customer"
    enabled: true
    workflow: "standard_rag"
    vector_collection: "loyalty_faqs"
    requires_user_lookup: true  # Look up points balance
    escalation_email: "loyalty@shopunow.com"
```

```python
class DepartmentRegistry:
    """Dynamic department configuration."""
    
    def __init__(self, config_path: str = "config/departments.yaml"):
        self.config = self._load_config(config_path)
        self.departments = self._build_departments()
    
    def get_department(self, key: str) -> DepartmentConfig:
        return self.departments.get(key)
    
    def get_enabled_departments(self) -> list[DepartmentConfig]:
        return [d for d in self.departments.values() if d.enabled]
    
    def get_workflow(self, department_key: str) -> BaseWorkflow:
        """Get workflow instance for department."""
        config = self.get_department(department_key)
        
        workflow_map = {
            "standard_rag": StandardRAGWorkflow,
            "returns_workflow": ReturnWorkflow,
            "loyalty_workflow": LoyaltyWorkflow,
        }
        
        workflow_class = workflow_map.get(config.workflow, StandardRAGWorkflow)
        return workflow_class(config)
```

### 4.4 Implementation Steps

| Step | Task | Effort |
|------|------|--------|
| 4.4.1 | Create department configuration schema | 1 hr |
| 4.4.2 | Implement DepartmentRegistry class | 2 hrs |
| 4.4.3 | Update router to use dynamic departments | 1 hr |
| 4.4.4 | Generate FAQ data for new departments | 1 hr |
| 4.4.5 | Update ChromaDB with new department metadata | 1 hr |
| 4.4.6 | (Optional) Implement custom workflow base class | 2 hrs |
| 4.4.7 | (Optional) Implement Returns workflow | 3 hrs |
| 4.4.8 | Test new departments end-to-end | 2 hrs |

**Total Effort**: 8-13 hours

---

## 5. Production Deployment

### 5.1 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLOUD PROVIDER                           │
│                    (AWS / GCP / Azure / Railway)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐           │
│  │   Load     │     │   API      │     │   API      │           │
│  │  Balancer  │────▶│  Server 1  │     │  Server 2  │           │
│  └────────────┘     └────────────┘     └────────────┘           │
│                            │                 │                   │
│                            └────────┬────────┘                   │
│                                     │                            │
│              ┌──────────────────────┼──────────────────────┐    │
│              │                      │                      │    │
│              ▼                      ▼                      ▼    │
│       ┌────────────┐         ┌────────────┐         ┌──────────┐│
│       │   Redis    │         │  ChromaDB  │         │ Postgres ││
│       │  (Session) │         │  (Vectors) │         │  (Users) ││
│       └────────────┘         └────────────┘         └──────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Containerization (Docker)

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download sentence-transformers model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application
COPY src/ ./src/
COPY data/ ./data/

# Expose port
EXPOSE 8000

# Run with gunicorn for production
CMD ["gunicorn", "src.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - CHROMA_PERSIST_DIR=/app/data/chroma_db
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    volumes:
      - chroma_data:/app/data/chroma_db

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  chroma_data:
  redis_data:
```

### 5.3 Platform-Specific Deployment

#### Railway (Recommended for Simplicity)

```toml
# railway.toml
[build]
builder = "dockerfile"

[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "on_failure"
```

#### Render

```yaml
# render.yaml
services:
  - type: web
    name: shopunow-assistant
    env: docker
    plan: starter
    healthCheckPath: /health
    envVars:
      - key: OPENAI_API_KEY
        sync: false
```

#### AWS (ECS/Fargate)

```yaml
# cloudformation-snippet.yaml
Resources:
  TaskDefinition:
    Type: AWS::ECS::TaskDefinition
    Properties:
      Family: shopunow-assistant
      Cpu: 512
      Memory: 1024
      NetworkMode: awsvpc
      ContainerDefinitions:
        - Name: api
          Image: !Sub ${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com/shopunow:latest
          PortMappings:
            - ContainerPort: 8000
          Environment:
            - Name: CHROMA_PERSIST_DIR
              Value: /data/chroma_db
          Secrets:
            - Name: OPENAI_API_KEY
              ValueFrom: !Ref OpenAIKeySecret
```

### 5.4 Environment Configuration

```python
# src/config.py (production additions)
class Settings(BaseSettings):
    # ... existing fields ...
    
    # Production settings
    environment: Literal["development", "staging", "production"] = "development"
    
    # Redis (for session management)
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")
    
    # Database (for user persistence)  
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")
    
    # Monitoring
    sentry_dsn: Optional[str] = Field(default=None, alias="SENTRY_DSN")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100)
    rate_limit_window_seconds: int = Field(default=60)
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"
```

---

## 6. Advanced RAG Enhancements

### 6.1 Hybrid Search (BM25 + Semantic)

```python
class HybridRetriever:
    """Combine BM25 keyword search with semantic search."""
    
    def __init__(
        self,
        chroma_client: ChromaDBClient,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7
    ):
        self.chroma = chroma_client
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.bm25_index = None  # Built from documents
    
    def retrieve(
        self,
        query: str,
        department: Department,
        n_results: int = 10
    ) -> list[RetrievedDocument]:
        # Get semantic results
        semantic_results = self.chroma.query(
            query_text=query,
            department=department,
            n_results=n_results * 2
        )
        
        # Get BM25 results
        bm25_results = self._bm25_search(query, department, n_results * 2)
        
        # Combine with reciprocal rank fusion
        combined = self._reciprocal_rank_fusion(
            semantic_results,
            bm25_results,
            k=60  # RRF parameter
        )
        
        return combined[:n_results]
```

### 6.2 Query Expansion

```python
class QueryExpander:
    """Expand queries for better retrieval."""
    
    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm
    
    def expand(self, query: str) -> list[str]:
        """Generate query variations."""
        prompt = f"""Generate 3 alternative phrasings for this query that might help find relevant information:

Original: {query}

Return as JSON array of strings."""
        
        response = self.llm.generate_structured(prompt, QueryVariations)
        return [query] + response.variations
```

### 6.3 Re-Ranking with Cross-Encoder

```python
class CrossEncoderReranker:
    """Re-rank results using cross-encoder model."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int = 5
    ) -> list[RetrievedDocument]:
        # Create query-document pairs
        pairs = [(query, doc.content) for doc in documents]
        
        # Score with cross-encoder
        scores = self.model.predict(pairs)
        
        # Sort by cross-encoder score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Update similarity scores and return top_k
        results = []
        for doc, score in scored_docs[:top_k]:
            doc.similarity_score = float(score)
            results.append(doc)
        
        return results
```

### 6.4 Contextual Compression

```python
class ContextualCompressor:
    """Compress retrieved documents to relevant portions."""
    
    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm
    
    def compress(
        self,
        query: str,
        documents: list[RetrievedDocument]
    ) -> list[CompressedDocument]:
        """Extract only relevant portions of documents."""
        compressed = []
        
        for doc in documents:
            prompt = f"""Given this query and document, extract only the sentences that are relevant to answering the query.

Query: {query}

Document:
{doc.content}

Return only the relevant sentences, or "NOT_RELEVANT" if nothing is relevant."""
            
            response = self.llm.generate(prompt, max_tokens=200)
            
            if response.content.strip() != "NOT_RELEVANT":
                compressed.append(CompressedDocument(
                    original_id=doc.id,
                    compressed_content=response.content,
                    original_score=doc.similarity_score
                ))
        
        return compressed
```

---

## 7. Analytics & Monitoring

### 7.1 Metrics to Track

| Category | Metric | Description |
|----------|--------|-------------|
| **Usage** | queries_total | Total queries processed |
| | queries_by_department | Queries per department |
| | escalations_total | Total escalations |
| **Performance** | response_time_ms | End-to-end latency |
| | llm_latency_ms | LLM API call latency |
| | retrieval_latency_ms | ChromaDB query latency |
| **Quality** | retrieval_score_avg | Average retrieval similarity |
| | confidence_avg | Average response confidence |
| | empty_retrieval_rate | % queries with no results |
| **Errors** | error_rate | % of failed requests |
| | classification_failures | Classification errors |

### 7.2 Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Counters
QUERIES_TOTAL = Counter(
    'shopunow_queries_total',
    'Total queries processed',
    ['department', 'sentiment', 'route']
)

ESCALATIONS_TOTAL = Counter(
    'shopunow_escalations_total',
    'Total escalations',
    ['reason']
)

# Histograms
RESPONSE_TIME = Histogram(
    'shopunow_response_time_seconds',
    'Response time in seconds',
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

RETRIEVAL_SCORE = Histogram(
    'shopunow_retrieval_score',
    'Top retrieval similarity score',
    buckets=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Gauges
ACTIVE_SESSIONS = Gauge(
    'shopunow_active_sessions',
    'Number of active sessions'
)
```

### 7.3 Logging Structure

```python
import structlog

logger = structlog.get_logger()

# In orchestrator
logger.info(
    "query_processed",
    query_length=len(request.query),
    department=classification.department.value,
    sentiment=classification.sentiment.value,
    route=routing_decision.route.value,
    retrieval_count=retrieval_result.retrieval_count,
    top_score=retrieval_result.top_document.similarity_score if retrieval_result.has_results else 0,
    confidence=response.confidence,
    processing_time_ms=processing_time,
    session_id=request.session_id,
    user_id=request.user_id
)
```

### 7.4 Dashboard Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    ShopUNow Assistant Dashboard                  │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  Queries/Hour   │  Avg Response   │     Escalation Rate         │
│     1,234       │     1.2s        │         8.5%                │
├─────────────────┴─────────────────┴─────────────────────────────┤
│                                                                  │
│  [Query Volume Chart - 24hr]                                    │
│  ████████████████████████████████████                          │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  Department Distribution          │  Sentiment Distribution     │
│  ┌──────────────────────────────┐ │  ┌────────────────────────┐│
│  │ HR         ████████ 32%     │ │  │ Positive  ████ 15%    ││
│  │ IT         ██████ 24%       │ │  │ Neutral   ██████████ 77%││
│  │ Billing    ████████ 28%     │ │  │ Negative  ██ 8%       ││
│  │ Shipping   ████ 16%         │ │  └────────────────────────┘│
│  └──────────────────────────────┘ │                            │
├─────────────────────────────────────────────────────────────────┤
│  Recent Escalations                                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ ESC-12345 │ Billing │ Negative │ "This is ridiculous..."   ││
│  │ ESC-12344 │ Unknown │ Neutral  │ "What's the weather..."   ││
│  │ ESC-12343 │ Shipping│ Negative │ "Where is my order!!!"    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Multi-Modal Capabilities

### 8.1 Image-Based Queries

Allow customers to upload images of damaged products or shipping labels.

```python
class ImageProcessor:
    """Process image uploads for queries."""
    
    def __init__(self, vision_model: str = "gpt-4o"):
        self.client = OpenAI()
        self.model = vision_model
    
    async def analyze_image(
        self,
        image_base64: str,
        query: str
    ) -> ImageAnalysis:
        """Analyze uploaded image in context of query."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Customer query: {query}\n\nAnalyze this image and describe what you see that's relevant to the customer's issue."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        return ImageAnalysis(
            description=response.choices[0].message.content,
            detected_issues=self._extract_issues(response),
            suggested_department=self._suggest_department(response)
        )
```

### 8.2 Voice Input Support

```python
class VoiceProcessor:
    """Process voice input via Whisper."""
    
    def __init__(self):
        self.client = OpenAI()
    
    async def transcribe(self, audio_file: bytes) -> str:
        """Transcribe audio to text."""
        response = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        return response
```

---

## 9. Security & Compliance

### 9.1 Security Checklist

| Area | Measure | Priority |
|------|---------|----------|
| **Authentication** | API key authentication | High |
| | JWT tokens for users | High |
| | Rate limiting | High |
| **Data Protection** | PII detection & masking | High |
| | Encryption at rest | Medium |
| | Encryption in transit (HTTPS) | High |
| **Input Validation** | Query length limits | High |
| | Content filtering | Medium |
| | Injection prevention | High |
| **Audit** | Request logging | High |
| | PII access logging | Medium |
| **Compliance** | GDPR data deletion | Medium |
| | Data retention policies | Medium |

### 9.2 PII Detection

```python
class PIIDetector:
    """Detect and mask personally identifiable information."""
    
    PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    }
    
    def detect(self, text: str) -> list[PIIMatch]:
        """Detect PII in text."""
        matches = []
        for pii_type, pattern in self.PATTERNS.items():
            for match in re.finditer(pattern, text):
                matches.append(PIIMatch(
                    type=pii_type,
                    start=match.start(),
                    end=match.end(),
                    value=match.group()
                ))
        return matches
    
    def mask(self, text: str) -> str:
        """Mask detected PII."""
        for pii_type, pattern in self.PATTERNS.items():
            mask_char = '*'
            text = re.sub(pattern, lambda m: mask_char * len(m.group()), text)
        return text
```

### 9.3 Rate Limiting

```python
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/query")
@limiter.limit("100/minute")
async def process_query(request: Request, query_request: QueryRequest):
    ...
```

---

## 10. Integration Ecosystem

### 10.1 Potential Integrations

| Integration | Purpose | Complexity |
|-------------|---------|------------|
| **Slack** | Internal employee queries via Slack bot | Medium |
| **Microsoft Teams** | Enterprise chat integration | Medium |
| **Zendesk** | Ticket creation for escalations | Medium |
| **Salesforce** | CRM integration for customer context | High |
| **Shopify** | Order lookup for shipping queries | Medium |
| **Twilio** | SMS/WhatsApp notifications | Low |
| **SendGrid** | Email notifications | Low |

### 10.2 Slack Bot Integration

```python
from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler

slack_app = App(token=os.environ["SLACK_BOT_TOKEN"])
handler = SlackRequestHandler(slack_app)

@slack_app.message("")
async def handle_message(message, say):
    """Handle incoming Slack messages."""
    query = message["text"]
    user_id = message["user"]
    
    # Process through ShopUNow assistant
    request = QueryRequest(query=query, user_id=f"slack_{user_id}")
    response = orchestrator.process_query(request)
    
    if isinstance(response, EscalationResponse):
        await say(f"🔔 *Escalated*\n{response.message}\nTicket: {response.ticket_id}")
    else:
        sources_text = "\n".join([f"• {s.question_matched}" for s in response.sources[:2]])
        await say(f"*Answer:*\n{response.answer}\n\n*Sources:*\n{sources_text}")

@app.post("/slack/events")
async def slack_events(request: Request):
    return await handler.handle(request)
```

### 10.3 Webhook Support

```python
class WebhookService:
    """Send events to external webhooks."""
    
    def __init__(self, webhook_urls: dict[str, str]):
        self.webhooks = webhook_urls
    
    async def send_event(
        self,
        event_type: str,
        payload: dict
    ):
        """Send event to registered webhooks."""
        if event_type not in self.webhooks:
            return
        
        async with httpx.AsyncClient() as client:
            await client.post(
                self.webhooks[event_type],
                json={
                    "event": event_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    "payload": payload
                },
                headers={"X-Webhook-Secret": self.secret}
            )

# Usage
await webhook_service.send_event("escalation_created", {
    "ticket_id": ticket.ticket_id,
    "department": ticket.classification.department.value,
    "user_email": ticket.form_data.email
})
```

---

## Prioritization Matrix

### Effort vs Impact

```
                    HIGH IMPACT
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    │  • Conversational  │  • Extended Depts  │
    │    Memory          │  • Hybrid Search   │
    │  • Production      │  • Re-ranking      │
    │    Deployment      │                    │
    │                    │                    │
HIGH├────────────────────┼────────────────────┤LOW
EFFORT                   │                    EFFORT
    │                    │                    │
    │  • Multi-modal     │  • Interactive     │
    │  • Slack/Teams     │    Escalation Form │
    │  • Full Analytics  │  • Basic Metrics   │
    │                    │  • Email Notifs    │
    │                    │                    │
    └────────────────────┼────────────────────┘
                         │
                    LOW IMPACT
```

### Recommended Roadmap

| Phase | Timeline | Features |
|-------|----------|----------|
| **MVP+** | Week 1 | Extended departments, basic metrics |
| **v1.1** | Week 2-3 | Interactive escalation, email notifications |
| **v1.2** | Week 4-5 | Conversational memory (session only) |
| **v2.0** | Month 2 | Production deployment, full monitoring |
| **v2.1** | Month 3 | Advanced RAG (hybrid search, re-ranking) |
| **v3.0** | Month 4+ | Multi-modal, integrations |

---

*End of FUTURE_FEATURES.md*
