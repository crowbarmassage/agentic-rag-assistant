"""Pytest configuration and shared fixtures."""

import os
import sys
from pathlib import Path
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_PROVIDER", "sentence_transformers")


@pytest.fixture
def sample_qa_data():
    """Sample QA data for testing."""
    return {
        "id": "hr_001",
        "question": "How do I apply for paid time off (PTO)?",
        "answer": "To request PTO, please log into the HR Portal at hr.shopunow.com and navigate to the Time Off section.",
        "department": "hr",
        "user_type": "internal_employee",
        "keywords": ["PTO", "time off", "leave"]
    }


@pytest.fixture
def sample_qa_pairs():
    """Multiple sample QA pairs for batch testing."""
    return [
        {
            "id": "hr_001",
            "question": "How do I apply for PTO?",
            "answer": "Log into the HR Portal at hr.shopunow.com to request time off.",
            "department": "hr",
            "user_type": "internal_employee",
            "keywords": ["PTO", "time off"]
        },
        {
            "id": "it_001",
            "question": "How do I reset my password?",
            "answer": "Contact IT Support at ext. 5555 or visit helpdesk.shopunow.com.",
            "department": "it_support",
            "user_type": "internal_employee",
            "keywords": ["password", "reset"]
        },
        {
            "id": "billing_001",
            "question": "How do I get a refund?",
            "answer": "Submit a refund request through my.shopunow.com within 30 days.",
            "department": "billing",
            "user_type": "external_customer",
            "keywords": ["refund", "return"]
        },
        {
            "id": "shipping_001",
            "question": "Where is my order?",
            "answer": "Track your order at track.shopunow.com with your order number.",
            "department": "shipping",
            "user_type": "external_customer",
            "keywords": ["order", "tracking"]
        },
    ]


@pytest.fixture
def temp_chroma_dir(tmp_path):
    """Temporary directory for ChromaDB tests."""
    chroma_dir = tmp_path / "chroma_test"
    chroma_dir.mkdir()
    return str(chroma_dir)


@pytest.fixture
def data_dir():
    """Path to generated FAQ data."""
    return Path(__file__).parent.parent / "data" / "raw"
