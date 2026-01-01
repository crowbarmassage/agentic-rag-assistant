"""Phase 7 Tests: Retrieval Pipeline."""

import pytest
from unittest.mock import Mock, MagicMock


class TestDynamicKRetrieverImport:
    """Test DynamicKRetriever imports."""

    def test_retriever_import(self):
        """Verify DynamicKRetriever can be imported."""
        from src.pipelines import DynamicKRetriever
        assert DynamicKRetriever is not None

    def test_retrieval_pipeline_import(self):
        """Verify RetrievalPipeline can be imported."""
        from src.pipelines import RetrievalPipeline
        assert RetrievalPipeline is not None


class TestDynamicKRetrieverInit:
    """Test DynamicKRetriever initialization."""

    def test_retriever_creation(self):
        """Verify retriever can be created with mock client."""
        from src.pipelines import DynamicKRetriever

        mock_chroma = Mock()
        retriever = DynamicKRetriever(chroma_client=mock_chroma)

        assert retriever is not None
        assert retriever.chroma == mock_chroma
        assert retriever.min_threshold == 0.5  # default
        assert retriever.max_k == 10  # default
        assert retriever.drop_off_ratio == 0.7  # default

    def test_retriever_custom_params(self):
        """Verify retriever with custom parameters."""
        from src.pipelines import DynamicKRetriever

        mock_chroma = Mock()
        retriever = DynamicKRetriever(
            chroma_client=mock_chroma,
            min_threshold=0.6,
            max_k=5,
            drop_off_ratio=0.8
        )

        assert retriever.min_threshold == 0.6
        assert retriever.max_k == 5
        assert retriever.drop_off_ratio == 0.8


class TestDynamicKAlgorithm:
    """Test dynamic k selection algorithm."""

    def test_empty_candidates(self):
        """Verify empty candidates returns empty list."""
        from src.pipelines.retrieval import DynamicKRetriever

        mock_chroma = Mock()
        retriever = DynamicKRetriever(chroma_client=mock_chroma)

        result = retriever._apply_dynamic_k([])
        assert result == []

    def test_single_doc_above_threshold(self):
        """Verify single doc above threshold is included."""
        from src.pipelines.retrieval import DynamicKRetriever
        from src.models import RetrievedDocument, Department

        mock_chroma = Mock()
        retriever = DynamicKRetriever(chroma_client=mock_chroma, min_threshold=0.5)

        docs = [
            RetrievedDocument(
                id="doc1",
                content="Test content",
                question="Test question?",
                answer="Test answer",
                department=Department.HR,
                similarity_score=0.8
            )
        ]

        result = retriever._apply_dynamic_k(docs)
        assert len(result) == 1
        assert result[0].id == "doc1"

    def test_single_doc_below_threshold(self):
        """Verify single doc below threshold is excluded."""
        from src.pipelines.retrieval import DynamicKRetriever
        from src.models import RetrievedDocument, Department

        mock_chroma = Mock()
        retriever = DynamicKRetriever(chroma_client=mock_chroma, min_threshold=0.5)

        docs = [
            RetrievedDocument(
                id="doc1",
                content="Test content",
                question="Test question?",
                answer="Test answer",
                department=Department.HR,
                similarity_score=0.4  # Below threshold
            )
        ]

        result = retriever._apply_dynamic_k(docs)
        assert len(result) == 0

    def test_drop_off_filtering(self):
        """Verify drop-off ratio filters documents."""
        from src.pipelines.retrieval import DynamicKRetriever
        from src.models import RetrievedDocument, Department

        mock_chroma = Mock()
        retriever = DynamicKRetriever(
            chroma_client=mock_chroma,
            min_threshold=0.5,
            drop_off_ratio=0.8
        )

        docs = [
            RetrievedDocument(
                id="doc1", content="c", question="q?", answer="a",
                department=Department.HR, similarity_score=0.9
            ),
            RetrievedDocument(
                id="doc2", content="c", question="q?", answer="a",
                department=Department.HR, similarity_score=0.85  # 0.85/0.9 = 0.94 > 0.8, include
            ),
            RetrievedDocument(
                id="doc3", content="c", question="q?", answer="a",
                department=Department.HR, similarity_score=0.6  # 0.6/0.85 = 0.71 < 0.8, exclude
            )
        ]

        result = retriever._apply_dynamic_k(docs)
        assert len(result) == 2
        assert result[0].id == "doc1"
        assert result[1].id == "doc2"

    def test_sorts_by_score(self):
        """Verify documents are sorted by score descending."""
        from src.pipelines.retrieval import DynamicKRetriever
        from src.models import RetrievedDocument, Department

        mock_chroma = Mock()
        retriever = DynamicKRetriever(chroma_client=mock_chroma, min_threshold=0.5)

        # Intentionally unsorted
        docs = [
            RetrievedDocument(
                id="doc2", content="c", question="q?", answer="a",
                department=Department.HR, similarity_score=0.7
            ),
            RetrievedDocument(
                id="doc1", content="c", question="q?", answer="a",
                department=Department.HR, similarity_score=0.9
            ),
        ]

        result = retriever._apply_dynamic_k(docs)
        assert result[0].id == "doc1"  # Higher score first


class TestDynamicKRetrieverRetrieve:
    """Test DynamicKRetriever.retrieve method."""

    def test_retrieve_calls_chroma(self):
        """Verify retrieve calls ChromaDB with correct params."""
        from src.pipelines import DynamicKRetriever
        from src.models import Department, UserType

        mock_chroma = Mock()
        mock_chroma.query.return_value = []

        retriever = DynamicKRetriever(chroma_client=mock_chroma, max_k=5)
        result = retriever.retrieve(
            query="test query",
            department=Department.HR,
            user_type=UserType.INTERNAL_EMPLOYEE
        )

        mock_chroma.query.assert_called_once()
        call_kwargs = mock_chroma.query.call_args[1]
        assert call_kwargs['query_text'] == "test query"
        assert call_kwargs['department'] == Department.HR
        assert call_kwargs['user_type'] == UserType.INTERNAL_EMPLOYEE
        assert call_kwargs['n_results'] == 5

    def test_retrieve_returns_retrieval_result(self):
        """Verify retrieve returns RetrievalResult."""
        from src.pipelines import DynamicKRetriever
        from src.models import Department, RetrievalResult

        mock_chroma = Mock()
        mock_chroma.query.return_value = []

        retriever = DynamicKRetriever(chroma_client=mock_chroma)
        result = retriever.retrieve(query="test", department=Department.HR)

        assert isinstance(result, RetrievalResult)
        assert result.query == "test"
        assert result.department_filter == Department.HR
        assert result.retrieval_count == 0


class TestRetrievalPipeline:
    """Test RetrievalPipeline wrapper."""

    def test_pipeline_creation(self):
        """Verify pipeline can be created."""
        from src.pipelines import DynamicKRetriever, RetrievalPipeline

        mock_chroma = Mock()
        retriever = DynamicKRetriever(chroma_client=mock_chroma)
        pipeline = RetrievalPipeline(retriever=retriever)

        assert pipeline is not None
        assert pipeline.retriever == retriever

    def test_pipeline_delegates_to_retriever(self):
        """Verify pipeline delegates to retriever."""
        from src.pipelines import DynamicKRetriever, RetrievalPipeline
        from src.models import Department, RetrievalResult

        mock_chroma = Mock()
        mock_chroma.query.return_value = []

        retriever = DynamicKRetriever(chroma_client=mock_chroma)
        pipeline = RetrievalPipeline(retriever=retriever)

        result = pipeline.retrieve(query="test", department=Department.HR)

        assert isinstance(result, RetrievalResult)
        mock_chroma.query.assert_called_once()


class TestResponseGeneratorImport:
    """Test ResponseGenerator imports."""

    def test_generator_import(self):
        """Verify ResponseGenerator can be imported."""
        from src.pipelines import ResponseGenerator
        assert ResponseGenerator is not None


class TestResponseGeneratorInit:
    """Test ResponseGenerator initialization."""

    def test_generator_creation(self):
        """Verify generator can be created with mock provider."""
        from src.pipelines import ResponseGenerator

        mock_llm = Mock()
        generator = ResponseGenerator(llm_provider=mock_llm)

        assert generator is not None
        assert generator.llm == mock_llm


class TestResponseGeneratorGenerate:
    """Test ResponseGenerator.generate method."""

    def test_generate_no_results(self):
        """Verify generate handles no retrieval results."""
        from src.pipelines import ResponseGenerator
        from src.models import RetrievalResult, Department, GeneratedResponse

        mock_llm = Mock()
        generator = ResponseGenerator(llm_provider=mock_llm)

        retrieval = RetrievalResult(
            query="test query",
            department_filter=Department.HR,
            documents=[],
            retrieval_count=0,
            threshold_used=0.5
        )

        result = generator.generate(query="test", retrieval=retrieval)

        assert isinstance(result, GeneratedResponse)
        assert "couldn't find" in result.answer.lower()
        assert result.confidence == 0.3
        mock_llm.generate.assert_not_called()  # No LLM call when no results

    def test_generate_with_results(self):
        """Verify generate creates response from retrieved docs."""
        from src.pipelines import ResponseGenerator
        from src.models import (
            RetrievalResult, RetrievedDocument, Department,
            GeneratedResponse
        )
        from src.providers.base import LLMResponse

        mock_llm = Mock()
        mock_llm.generate.return_value = LLMResponse(
            content="Here's how to check PTO: Use the HR Portal.",
            model="test-model",
            usage={}
        )

        generator = ResponseGenerator(llm_provider=mock_llm)

        docs = [
            RetrievedDocument(
                id="hr_001",
                content="Q: How to check PTO?\nA: Use HR Portal",
                question="How to check PTO?",
                answer="Use HR Portal",
                department=Department.HR,
                similarity_score=0.9
            )
        ]

        retrieval = RetrievalResult(
            query="How do I check PTO?",
            department_filter=Department.HR,
            documents=docs,
            retrieval_count=1,
            threshold_used=0.5
        )

        result = generator.generate(query="How do I check PTO?", retrieval=retrieval)

        assert isinstance(result, GeneratedResponse)
        assert "PTO" in result.answer or "HR" in result.answer
        assert len(result.sources) == 1
        assert result.sources[0].document_id == "hr_001"
        mock_llm.generate.assert_called_once()

    def test_generate_confidence_from_scores(self):
        """Verify confidence is calculated from similarity scores."""
        from src.pipelines import ResponseGenerator
        from src.models import RetrievalResult, RetrievedDocument, Department
        from src.providers.base import LLMResponse

        mock_llm = Mock()
        mock_llm.generate.return_value = LLMResponse(
            content="Answer here",
            model="test",
            usage={}
        )

        generator = ResponseGenerator(llm_provider=mock_llm)

        docs = [
            RetrievedDocument(
                id="1", content="c", question="q?", answer="a",
                department=Department.HR, similarity_score=0.8
            ),
            RetrievedDocument(
                id="2", content="c", question="q?", answer="a",
                department=Department.HR, similarity_score=0.6
            )
        ]

        retrieval = RetrievalResult(
            query="test",
            department_filter=Department.HR,
            documents=docs,
            retrieval_count=2,
            threshold_used=0.5
        )

        result = generator.generate(query="test", retrieval=retrieval)

        # Average of 0.8 and 0.6 = 0.7
        assert result.confidence == 0.7

    def test_generate_context_includes_sources(self):
        """Verify LLM prompt includes source information."""
        from src.pipelines import ResponseGenerator
        from src.models import RetrievalResult, RetrievedDocument, Department
        from src.providers.base import LLMResponse

        mock_llm = Mock()
        mock_llm.generate.return_value = LLMResponse(
            content="Answer",
            model="test",
            usage={}
        )

        generator = ResponseGenerator(llm_provider=mock_llm)

        docs = [
            RetrievedDocument(
                id="1",
                content="c",
                question="What is PTO policy?",
                answer="Employees get 20 days PTO",
                department=Department.HR,
                similarity_score=0.9
            )
        ]

        retrieval = RetrievalResult(
            query="PTO info",
            department_filter=Department.HR,
            documents=docs,
            retrieval_count=1,
            threshold_used=0.5
        )

        generator.generate(query="PTO info", retrieval=retrieval)

        call_kwargs = mock_llm.generate.call_args[1]
        prompt = call_kwargs['prompt']

        assert "PTO info" in prompt  # Query included
        assert "What is PTO policy?" in prompt  # Source question
        assert "20 days" in prompt  # Source answer
