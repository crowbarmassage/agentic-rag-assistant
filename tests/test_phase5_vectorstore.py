"""Phase 5 Tests: Vector Store Setup."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch


class TestDataLoader:
    """Test data loading utilities."""

    def test_data_loader_import(self):
        """Verify data loader functions can be imported."""
        from src.utils import load_department_faqs, load_all_faqs, get_all_qa_pairs
        assert load_department_faqs is not None
        assert load_all_faqs is not None
        assert get_all_qa_pairs is not None

    def test_load_all_faqs(self, data_dir):
        """Verify loading all FAQs from directory."""
        from src.utils import load_all_faqs

        if not data_dir.exists():
            pytest.skip("FAQ data not generated yet")

        all_faqs = load_all_faqs(str(data_dir))

        assert len(all_faqs) > 0
        assert "hr" in all_faqs or "it_support" in all_faqs

    def test_get_all_qa_pairs(self, data_dir):
        """Verify getting flat list of QA pairs."""
        from src.utils import get_all_qa_pairs

        if not data_dir.exists():
            pytest.skip("FAQ data not generated yet")

        qa_pairs = get_all_qa_pairs(str(data_dir))

        assert len(qa_pairs) > 0
        assert hasattr(qa_pairs[0], 'question')
        assert hasattr(qa_pairs[0], 'answer')
        assert hasattr(qa_pairs[0], 'department')

    def test_load_department_faqs_structure(self, data_dir):
        """Verify loaded FAQ structure."""
        from src.utils import load_department_faqs
        from src.models import DepartmentFAQs, Department

        if not data_dir.exists():
            pytest.skip("FAQ data not generated yet")

        # Find a JSON file
        json_files = list(data_dir.glob("*_faqs.json"))
        if not json_files:
            pytest.skip("No FAQ files found")

        dept_faqs = load_department_faqs(json_files[0])

        assert isinstance(dept_faqs, DepartmentFAQs)
        assert isinstance(dept_faqs.department, Department)
        assert dept_faqs.count > 0
        assert len(dept_faqs.qa_pairs) > 0


class TestChromaDBClient:
    """Test ChromaDB client wrapper."""

    def test_chromadb_client_import(self):
        """Verify ChromaDB client can be imported."""
        from src.vectorstore import ChromaDBClient
        assert ChromaDBClient is not None

    def test_chromadb_client_creation(self, temp_chroma_dir):
        """Verify ChromaDB client can be created."""
        from src.vectorstore import ChromaDBClient

        client = ChromaDBClient(persist_directory=temp_chroma_dir)

        assert client is not None
        assert client.persist_directory == Path(temp_chroma_dir)

    def test_chromadb_collection_name(self, temp_chroma_dir):
        """Verify collection name constant."""
        from src.vectorstore import ChromaDBClient

        assert ChromaDBClient.COLLECTION_NAME == "shopunow_faqs"

    def test_chromadb_collection_lazy_init(self, temp_chroma_dir):
        """Verify collection is lazily initialized."""
        from src.vectorstore import ChromaDBClient

        client = ChromaDBClient(persist_directory=temp_chroma_dir)

        # Collection should be created on first access
        collection = client.collection
        assert collection is not None
        assert collection.name == "shopunow_faqs"

    def test_chromadb_get_document_count_empty(self, temp_chroma_dir):
        """Verify document count for empty collection."""
        from src.vectorstore import ChromaDBClient

        client = ChromaDBClient(persist_directory=temp_chroma_dir)

        count = client.get_document_count()
        assert count == 0

    def test_chromadb_reset_collection(self, temp_chroma_dir):
        """Verify collection reset."""
        from src.vectorstore import ChromaDBClient

        client = ChromaDBClient(persist_directory=temp_chroma_dir)

        # Access collection to create it
        _ = client.collection

        # Reset
        client.reset_collection()

        # Should be empty again
        assert client.get_document_count() == 0

    @pytest.mark.slow
    def test_chromadb_add_qa_pair(self, temp_chroma_dir, sample_qa_data):
        """Verify adding single QA pair."""
        from src.vectorstore import ChromaDBClient
        from src.providers import EmbeddingProviderFactory
        from src.models import QAPair, Department, UserType

        embedder = EmbeddingProviderFactory.create("sentence_transformers")
        client = ChromaDBClient(
            persist_directory=temp_chroma_dir,
            embedding_provider=embedder
        )

        qa = QAPair(
            id=sample_qa_data["id"],
            question=sample_qa_data["question"],
            answer=sample_qa_data["answer"],
            department=Department(sample_qa_data["department"]),
            user_type=UserType(sample_qa_data["user_type"]),
            keywords=sample_qa_data["keywords"]
        )

        client.add_qa_pair(qa)

        assert client.get_document_count() == 1

    @pytest.mark.slow
    def test_chromadb_add_batch(self, temp_chroma_dir, sample_qa_pairs):
        """Verify batch adding QA pairs."""
        from src.vectorstore import ChromaDBClient
        from src.providers import EmbeddingProviderFactory
        from src.models import QAPair, Department, UserType

        embedder = EmbeddingProviderFactory.create("sentence_transformers")
        client = ChromaDBClient(
            persist_directory=temp_chroma_dir,
            embedding_provider=embedder
        )

        qa_list = [
            QAPair(
                id=qa["id"],
                question=qa["question"],
                answer=qa["answer"],
                department=Department(qa["department"]),
                user_type=UserType(qa["user_type"]),
                keywords=qa["keywords"]
            )
            for qa in sample_qa_pairs
        ]

        client.add_qa_pairs_batch(qa_list, batch_size=2)

        assert client.get_document_count() == len(sample_qa_pairs)

    @pytest.mark.slow
    def test_chromadb_query(self, temp_chroma_dir, sample_qa_pairs):
        """Verify querying documents."""
        from src.vectorstore import ChromaDBClient
        from src.providers import EmbeddingProviderFactory
        from src.models import QAPair, Department, UserType

        embedder = EmbeddingProviderFactory.create("sentence_transformers")
        client = ChromaDBClient(
            persist_directory=temp_chroma_dir,
            embedding_provider=embedder
        )

        # Add test data
        qa_list = [
            QAPair(
                id=qa["id"],
                question=qa["question"],
                answer=qa["answer"],
                department=Department(qa["department"]),
                user_type=UserType(qa["user_type"]),
                keywords=qa["keywords"]
            )
            for qa in sample_qa_pairs
        ]
        client.add_qa_pairs_batch(qa_list)

        # Query
        results = client.query(
            query_text="How do I reset my password?",
            n_results=2
        )

        assert len(results) > 0
        assert results[0].similarity_score > 0

    @pytest.mark.slow
    def test_chromadb_query_with_department_filter(self, temp_chroma_dir, sample_qa_pairs):
        """Verify querying with department filter."""
        from src.vectorstore import ChromaDBClient
        from src.providers import EmbeddingProviderFactory
        from src.models import QAPair, Department, UserType

        embedder = EmbeddingProviderFactory.create("sentence_transformers")
        client = ChromaDBClient(
            persist_directory=temp_chroma_dir,
            embedding_provider=embedder
        )

        # Add test data
        qa_list = [
            QAPair(
                id=qa["id"],
                question=qa["question"],
                answer=qa["answer"],
                department=Department(qa["department"]),
                user_type=UserType(qa["user_type"]),
                keywords=qa["keywords"]
            )
            for qa in sample_qa_pairs
        ]
        client.add_qa_pairs_batch(qa_list)

        # Query with filter
        results = client.query(
            query_text="I need help",
            department=Department.HR,
            n_results=5
        )

        # All results should be from HR
        for doc in results:
            assert doc.department == Department.HR

    def test_chromadb_no_embedding_provider_error(self, temp_chroma_dir):
        """Verify error when no embedding provider set."""
        from src.vectorstore import ChromaDBClient
        from src.models import QAPair, Department, UserType

        client = ChromaDBClient(persist_directory=temp_chroma_dir)
        # No embedding provider set

        qa = QAPair(
            id="test",
            question="Test question with sufficient length",
            answer="Test answer with sufficient length for validation",
            department=Department.HR,
            user_type=UserType.INTERNAL_EMPLOYEE
        )

        with pytest.raises(ValueError, match="Embedding provider not set"):
            client.add_qa_pair(qa)


class TestIngestionPipeline:
    """Test data ingestion pipeline."""

    def test_ingestion_import(self):
        """Verify ingestion functions can be imported."""
        from src.vectorstore import ingest_faqs, verify_ingestion
        assert ingest_faqs is not None
        assert verify_ingestion is not None

    @pytest.mark.slow
    def test_ingest_faqs(self, data_dir, temp_chroma_dir):
        """Verify FAQ ingestion."""
        from src.vectorstore import ingest_faqs

        if not data_dir.exists():
            pytest.skip("FAQ data not generated yet")

        chroma = ingest_faqs(
            data_dir=str(data_dir),
            chroma_dir=temp_chroma_dir,
            reset_collection=True
        )

        assert chroma.get_document_count() > 0

    @pytest.mark.slow
    def test_verify_ingestion(self, data_dir, temp_chroma_dir):
        """Verify ingestion verification."""
        from src.vectorstore import ingest_faqs, verify_ingestion

        if not data_dir.exists():
            pytest.skip("FAQ data not generated yet")

        # First ingest
        ingest_faqs(
            data_dir=str(data_dir),
            chroma_dir=temp_chroma_dir,
            reset_collection=True
        )

        # Then verify
        result = verify_ingestion(chroma_dir=temp_chroma_dir)
        assert result is True
