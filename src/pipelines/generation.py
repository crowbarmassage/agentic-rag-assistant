"""Response generation pipeline."""

from typing import Optional

from src.providers.base import BaseLLMProvider
from src.models import (
    RetrievalResult,
    GeneratedResponse,
    SourceAttribution,
    Department,
    ClassificationResult
)
from src.utils.prompts import RESPONSE_GENERATION_PROMPT


class ResponseGenerator:
    """Generate responses from retrieved context."""

    def __init__(self, llm_provider: BaseLLMProvider):
        """
        Initialize response generator.

        Args:
            llm_provider: Configured LLM provider
        """
        self.llm = llm_provider

    def generate(
        self,
        query: str,
        retrieval: RetrievalResult,
        classification: Optional[ClassificationResult] = None
    ) -> GeneratedResponse:
        """
        Generate response from retrieved context.

        Args:
            query: Original user query
            retrieval: Retrieval results
            classification: Optional classification for confidence

        Returns:
            GeneratedResponse with answer and sources
        """
        # Handle no results
        if not retrieval.has_results:
            return GeneratedResponse(
                answer="I couldn't find specific information to answer your question. "
                       "Please try rephrasing or contact support for assistance.",
                sources=[],
                confidence=0.3,
                department=retrieval.department_filter
            )

        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieval.documents, 1):
            context_parts.append(
                f"[Source {i}]\n"
                f"Q: {doc.question}\n"
                f"A: {doc.answer}\n"
                f"Relevance: {doc.similarity_score:.2f}"
            )

        context = "\n\n".join(context_parts)

        # Generate response
        prompt = f"""User Query: {query}

Retrieved Information:
{context}

Based on the above information, provide a helpful response to the user's query.
If the information doesn't fully address the query, acknowledge what you can answer and what may need further clarification.
Be concise and friendly."""

        llm_response = self.llm.generate(
            prompt=prompt,
            system_prompt=RESPONSE_GENERATION_PROMPT,
            temperature=0.3,
            max_tokens=512
        )

        # Build source attributions
        sources = [
            SourceAttribution(
                document_id=doc.id,
                question_matched=doc.question,
                relevance_score=doc.similarity_score
            )
            for doc in retrieval.documents
        ]

        # Calculate confidence
        avg_score = sum(d.similarity_score for d in retrieval.documents) / len(retrieval.documents)
        confidence = avg_score
        if classification:
            confidence = min(avg_score, classification.confidence)

        return GeneratedResponse(
            answer=llm_response.content,
            sources=sources,
            confidence=confidence,
            department=retrieval.department_filter
        )
