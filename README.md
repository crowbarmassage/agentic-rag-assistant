# ShopUNow AI Assistant

> Intelligent Agentic AI Assistant for Retail Customer & Employee Support

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Overview

ShopUNow AI Assistant is an **Agentic RAG (Retrieval-Augmented Generation) system** built for a fictional retail company. It intelligently routes and answers queries from both **internal employees** and **external customers** across multiple departments.

This project demonstrates key concepts in modern AI agent development:
- **LLM-powered classification** (sentiment analysis + department routing)
- **Vector-based retrieval** with metadata filtering
- **Dynamic context selection** based on relevance scores
- **Configurable provider abstraction** (swap LLMs and embedding models at runtime)
- **Production-ready API** with FastAPI

### 🎯 Business Scenario

ShopUNow is a retail company selling clothing, DIY products, books, and toys. The AI Assistant handles queries for:

| Department | User Type | Examples |
|------------|-----------|----------|
| **Human Resources** | Internal Employee | PTO requests, payroll, benefits |
| **IT Support** | Internal Employee | Password resets, VPN, hardware issues |
| **Billing & Payments** | External Customer | Refunds, invoices, payment methods |
| **Shipping & Delivery** | External Customer | Order tracking, returns, damaged goods |

---

## ✨ Features

### Core Capabilities

- **🧠 Intelligent Query Classification**
  - Sentiment detection (positive/neutral/negative)
  - Automatic department routing
  - User type inference (employee vs customer)

- **🔍 Smart Retrieval (RAG)**
  - ChromaDB vector store with metadata filtering
  - Dynamic K selection based on relevance scores
  - Source attribution for transparency

- **🔀 Adaptive Routing**
  - Positive/neutral queries → RAG pipeline
  - Negative sentiment → Human escalation
  - Unknown department → Human escalation

- **🔌 Provider Abstraction**
  - LLM: OpenAI, Google Gemini, Groq
  - Embeddings: Sentence Transformers, OpenAI, Cohere, Google
  - Swap providers via configuration

- **🚀 Production-Ready API**
  - FastAPI with automatic OpenAPI docs
  - Health checks and monitoring endpoints
  - CORS support

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CLASSIFICATION PIPELINE                        │
│         ┌─────────────────┐    ┌─────────────────┐              │
│         │    Sentiment    │───▶│   Department    │              │
│         │    Detection    │    │  Classification │              │
│         └─────────────────┘    └─────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                          ROUTER                                  │
│                                                                  │
│    [Negative / Unknown]                    [Positive / Neutral]  │
│           │                                        │             │
│           ▼                                        ▼             │
│    ┌─────────────┐                    ┌─────────────────────┐   │
│    │   HUMAN     │                    │    RAG PIPELINE     │   │
│    │ ESCALATION  │                    │  (Dynamic K + Gen)  │   │
│    └─────────────┘                    └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              RESPONSE (Answer + Sources + Confidence)            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
shopunow_assistant/
├── src/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application
│   ├── config.py                   # Configuration management
│   ├── orchestrator.py             # Main pipeline orchestrator
│   ├── models/
│   │   ├── __init__.py
│   │   ├── enums.py                # Department, Sentiment, etc.
│   │   ├── data_models.py          # QAPair, ClassificationResult, etc.
│   │   └── api_models.py           # Request/Response schemas
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract base classes
│   │   ├── llm_openai.py           # OpenAI implementation
│   │   ├── llm_gemini.py           # Gemini implementation
│   │   ├── llm_groq.py             # Groq implementation
│   │   ├── llm_factory.py          # LLM provider factory
│   │   └── embedding_providers.py  # All embedding providers
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── classification.py       # Sentiment + Department
│   │   ├── retrieval.py            # Dynamic K retriever
│   │   └── generation.py           # Response generation
│   ├── routing/
│   │   ├── __init__.py
│   │   └── router.py               # Query routing logic
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── chroma_client.py        # ChromaDB wrapper
│   │   └── ingestion.py            # Data ingestion utilities
│   └── utils/
│       ├── __init__.py
│       ├── prompts.py              # Prompt templates
│       └── data_loader.py          # FAQ data loading utilities
├── datagen/
│   ├── generate_faqs_standalone.py # Standalone data generation script
│   ├── generate_faq_data.py        # Data generation module
│   ├── data_generation_prompts.py  # Generation prompts
│   └── requirements_datagen.txt    # Data generation dependencies
├── data/
│   ├── raw/                        # Generated FAQ JSON files
│   │   ├── hr_faqs.json
│   │   ├── it_support_faqs.json
│   │   ├── billing_faqs.json
│   │   └── shipping_faqs.json
│   └── chroma_db/                  # Vector database storage
├── tests/
│   ├── __init__.py
│   ├── test_classification.py
│   ├── test_retrieval.py
│   └── test_api.py
├── notebooks/
│   └── ShopUNow_Capstone.ipynb     # Jupyter notebook deliverable
├── docs/
│   ├── TECH_SPECS.md               # Technical specifications
│   ├── ATOMIC_STEPS.md             # Implementation roadmap
│   └── FUTURE_FEATURES.md          # Enhancement roadmap
├── generate_faqs_standalone.py     # Standalone data generation
├── requirements.txt
├── requirements_datagen.txt
├── run.py                          # API runner script
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- OpenAI API key (or Gemini/Groq for alternatives)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/shopunow-assistant.git
   cd shopunow-assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Generate FAQ data** (if not already present)
   ```bash
   python datagen/generate_faqs_standalone.py
   ```

6. **Ingest data into vector store**
   ```bash
   python -c "
   from src.vectorstore import ingest_faqs
   ingest_faqs(data_dir='./data/raw', chroma_dir='./data/chroma_db', reset_collection=True)
   "
   ```

7. **Run the API**
   ```bash
   python run.py
   ```

   You should see:
   ```
   [INIT] LLM: openai
   [INIT] Embeddings: sentence_transformers (dim=384)
   [INIT] ChromaDB: 59 documents
   [INIT] Orchestrator ready
   ```

8. **Test it out**
   ```bash
   # Health check
   curl http://localhost:8000/health

   # Query the assistant
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "How do I apply for PTO?"}'
   ```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# LLM Configuration
LLM_PROVIDER=openai                    # openai, gemini, groq
LLM_MODEL=gpt-4o-mini                  # Optional: specific model

# Embedding Configuration
EMBEDDING_PROVIDER=sentence_transformers  # sentence_transformers, openai, cohere, google
EMBEDDING_MODEL=all-MiniLM-L6-v2          # Optional: specific model

# API Keys (set the ones you need)
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
GROQ_API_KEY=gsk_...
COHERE_API_KEY=...

# Vector Store
CHROMA_PERSIST_DIR=./data/chroma_db

# Retrieval Settings
RETRIEVAL_MIN_THRESHOLD=0.3            # Minimum similarity score (0.3 works well with MiniLM)
RETRIEVAL_MAX_K=10                     # Max documents to retrieve
RETRIEVAL_DROP_OFF_RATIO=0.7           # Dynamic K drop-off threshold

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true
```

### Provider Options

| Provider | LLM Models | Embedding Models |
|----------|------------|------------------|
| **OpenAI** | gpt-4o-mini, gpt-4o, gpt-4-turbo | text-embedding-3-small, text-embedding-3-large |
| **Gemini** | gemini-1.5-flash, gemini-1.5-pro | text-embedding-004 |
| **Groq** | llama-3.3-70b-versatile, mixtral-8x7b | - |
| **Sentence Transformers** | - | all-MiniLM-L6-v2, all-mpnet-base-v2 |
| **Cohere** | - | embed-english-v3.0, embed-multilingual-v3.0 |

---

## 📡 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/query` | Process a user query |
| `GET` | `/health` | Basic health check |
| `GET` | `/health?deep=true` | Deep health check (all components) |
| `GET` | `/departments` | List available departments |
| `GET` | `/docs` | Swagger UI documentation |
| `GET` | `/redoc` | ReDoc documentation |

### Query Request

```bash
POST /query
Content-Type: application/json

{
  "query": "How do I reset my password?",
  "user_id": "emp_12345",      # Optional
  "session_id": "sess_abc123"  # Optional
}
```

### Query Response (RAG)

```json
{
  "query": "How do I reset my password?",
  "answer": "To reset your password, visit the IT portal at it.shopunow.com and click 'Forgot Password'. You'll receive a reset link via email within 5 minutes. If you need immediate assistance, contact the IT Helpdesk at ext. 5555.",
  "department": "it_support",
  "sentiment": "neutral",
  "sources": [
    {
      "document_id": "it_support_003",
      "question_matched": "What is the password reset process?",
      "relevance_score": 0.91
    }
  ],
  "confidence": 0.87,
  "was_escalated": false,
  "processing_time_ms": 1423.5,
  "timestamp": "2025-01-01T12:30:45.123Z"
}
```

### Escalation Response

```json
{
  "query": "This is ridiculous! I've been waiting for weeks!",
  "message": "Your query has been escalated to a human support agent. A representative will contact you shortly.",
  "reason": "Query has negative sentiment. Escalating to human agent for better handling.",
  "ticket_id": "ESC-1735737045",
  "estimated_response_time": "24-48 hours",
  "timestamp": "2025-01-01T12:30:45.123Z"
}
```

---

## 🧪 Testing

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test file
pytest tests/test_classification.py -v
```

### Manual API Testing

```bash
# Start the API
python run.py

# Run test script
python test_api.py
```

### Test Queries

| Query | Expected Behavior |
|-------|-------------------|
| "How do I apply for PTO?" | HR department, neutral sentiment, RAG response |
| "I can't login to my email" | IT Support, neutral sentiment, RAG response |
| "Where is my order #12345?" | Shipping, neutral sentiment, RAG response |
| "I need a refund" | Billing, neutral sentiment, RAG response |
| "This is terrible service!!!" | Any dept, negative sentiment, **Escalation** |
| "What's the weather today?" | Unknown dept, **Escalation** |

---

## 📊 Data Generation

Generate synthetic FAQ data for the knowledge base:

```bash
# Generate all departments (~60 QA pairs total)
python datagen/generate_faqs_standalone.py

# Data is saved to ./data/raw/ by default
```

After generating data, ingest it into ChromaDB:

```bash
python -c "
from src.vectorstore import ingest_faqs
ingest_faqs(data_dir='./data/raw', chroma_dir='./data/chroma_db', reset_collection=True)
"
```

### Output Format

```json
{
  "department": "hr",
  "department_name": "Human Resources",
  "user_type": "internal_employee",
  "count": 15,
  "qa_pairs": [
    {
      "id": "hr_001",
      "question": "How do I apply for paid time off (PTO)?",
      "answer": "You can apply for PTO through the HR Portal...",
      "department": "hr",
      "user_type": "internal_employee",
      "keywords": ["PTO", "time off", "leave", "vacation"]
    }
  ]
}
```

---

## 🔧 Development

### Adding a New LLM Provider

1. Create provider class in `src/providers/`:
   ```python
   class NewProvider(BaseLLMProvider):
       def generate(self, prompt, **kwargs) -> LLMResponse:
           ...
       def generate_structured(self, prompt, response_model, **kwargs):
           ...
   ```

2. Register in `src/providers/llm_factory.py`:
   ```python
   class LLMProviderType(str, Enum):
       NEW_PROVIDER = "new_provider"
   
   LLMProviderFactory._providers[LLMProviderType.NEW_PROVIDER] = NewProvider
   ```

### Adding a New Department

1. Add to `src/models/enums.py`:
   ```python
   class Department(str, Enum):
       NEW_DEPT = "new_dept"
   ```

2. Add context in `src/utils/data_generation_prompts.py`

3. Generate FAQ data for the new department

4. Re-ingest data into ChromaDB

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [TECH_SPECS.md](docs/TECH_SPECS.md) | Detailed technical architecture and specifications |
| [ATOMIC_STEPS.md](docs/ATOMIC_STEPS.md) | Step-by-step implementation guide (38 steps) |
| [FUTURE_FEATURES.md](docs/FUTURE_FEATURES.md) | Post-MVP enhancement roadmap |

---

## 🗺️ Roadmap

### MVP (Current)
- [x] LLM-based classification (sentiment + department)
- [x] ChromaDB vector store with metadata filtering
- [x] Dynamic K retrieval
- [x] Provider abstraction (LLM + Embeddings)
- [x] FastAPI with REST endpoints
- [x] Synthetic data generation

### Future Enhancements
- [ ] Conversational memory (session + user persistence)
- [ ] Interactive escalation forms with email/WhatsApp
- [ ] Extended departments with custom workflows
- [ ] Hybrid search (BM25 + semantic)
- [ ] Cross-encoder re-ranking
- [ ] Analytics dashboard
- [ ] Multi-modal support (images, voice)

See [FUTURE_FEATURES.md](docs/FUTURE_FEATURES.md) for detailed roadmap.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Analytics Vidhya** - GenAI Pinnacle Program & Agentic AI Pioneer Program
- **OpenAI** - GPT models and embeddings
- **ChromaDB** - Vector database
- **FastAPI** - Web framework
- **Sentence Transformers** - Local embedding models

---

## 📞 Contact

**Project Author**: Mohsin  
**Program**: Analytics Vidhya GenAI Pinnacle Program

---

<p align="center">
  <i>Built with ❤️ for the Agentic AI Pioneer Capstone Project</i>
</p>
