"""Retrieval pipeline with dynamic k selection."""

from typing import Optional

from src.vectorstore.chroma_client import ChromaDBClient
from src.models import (
    Department,
    UserType,
    RetrievalResult,
    RetrievedDocument
)


class DynamicKRetriever:
    """
    Retriever with dynamic k selection based on relevance scores.

    Strategy:
    1. Retrieve initial pool (max_k candidates)
    2. Filter by minimum threshold
    3. Apply relevance drop-off detection
    4. Return documents that meet quality criteria
    """

    def __init__(
        self,
        chroma_client: ChromaDBClient,
        min_threshold: float = 0.5,
        max_k: int = 10,
        drop_off_ratio: float = 0.7
    ):
        """
        Initialize retriever.

        Args:
            chroma_client: ChromaDB client instance
            min_threshold: Minimum similarity score to include (0-1)
            max_k: Maximum documents to retrieve initially
            drop_off_ratio: Stop if next score < previous * ratio
        """
        self.chroma = chroma_client
        self.min_threshold = min_threshold
        self.max_k = max_k
        self.drop_off_ratio = drop_off_ratio

    def retrieve(
        self,
        query: str,
        department: Department,
        user_type: Optional[UserType] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant documents with dynamic k selection.

        Args:
            query: User query text
            department: Department to filter by
            user_type: Optional user type filter

        Returns:
            RetrievalResult with dynamically selected documents
        """
        # Get initial candidate pool
        candidates = self.chroma.query(
            query_text=query,
            department=department,
            user_type=user_type,
            n_results=self.max_k,
            score_threshold=None  # Filter manually
        )

        # Apply dynamic k selection
        selected_docs = self._apply_dynamic_k(candidates)

        return RetrievalResult(
            query=query,
            department_filter=department,
            documents=selected_docs,
            retrieval_count=len(selected_docs),
            threshold_used=self.min_threshold
        )

    def _apply_dynamic_k(
        self,
        candidates: list[RetrievedDocument]
    ) -> list[RetrievedDocument]:
        """
        Apply dynamic k selection algorithm.

        Algorithm:
        1. Sort by score descending (should already be sorted)
        2. Always include first doc if above threshold
        3. Include subsequent docs if:
           a) Above min_threshold AND
           b) Score >= previous_score * drop_off_ratio
        """
        if not candidates:
            return []

        # Ensure sorted
        sorted_candidates = sorted(
            candidates,
            key=lambda d: d.similarity_score,
            reverse=True
        )

        selected = []
        previous_score = None

        for doc in sorted_candidates:
            # Check minimum threshold
            if doc.similarity_score < self.min_threshold:
                break

            # Check drop-off ratio (skip for first doc)
            if previous_score is not None:
                if doc.similarity_score < previous_score * self.drop_off_ratio:
                    break

            selected.append(doc)
            previous_score = doc.similarity_score

        return selected


class RetrievalPipeline:
    """Complete retrieval pipeline with fallback handling."""

    def __init__(
        self,
        retriever: DynamicKRetriever,
        fallback_message: str = "I couldn't find specific information about that topic."
    ):
        """
        Initialize retrieval pipeline.

        Args:
            retriever: DynamicKRetriever instance
            fallback_message: Message when no results found
        """
        self.retriever = retriever
        self.fallback_message = fallback_message

    def retrieve(
        self,
        query: str,
        department: Department,
        user_type: Optional[UserType] = None
    ) -> RetrievalResult:
        """
        Execute retrieval with logging.

        Args:
            query: User query
            department: Department filter
            user_type: Optional user type filter

        Returns:
            RetrievalResult
        """
        result = self.retriever.retrieve(
            query=query,
            department=department,
            user_type=user_type
        )

        if not result.has_results:
            print(f"[RETRIEVAL] No results for: {query[:50]}... in {department.value}")
        else:
            print(f"[RETRIEVAL] Found {result.retrieval_count} docs for: {query[:30]}...")

        return result
