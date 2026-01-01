"""ChromaDB client wrapper with metadata filtering support."""

import chromadb
from chromadb.config import Settings
from typing import Optional
from pathlib import Path

from src.models import Department, UserType, QAPair, RetrievedDocument
from src.providers.base import BaseEmbeddingProvider


class ChromaDBClient:
    """ChromaDB wrapper with metadata filtering support."""

    COLLECTION_NAME = "shopunow_faqs"

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        embedding_provider: Optional[BaseEmbeddingProvider] = None
    ):
        """
        Initialize ChromaDB client.

        Args:
            persist_directory: Directory for persistent storage
            embedding_provider: Embedding provider instance
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        self.embedding_provider = embedding_provider
        self._collection = None

    @property
    def collection(self):
        """Lazy initialization of collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"description": "ShopUNow FAQ knowledge base"}
            )
        return self._collection

    def set_embedding_provider(self, provider: BaseEmbeddingProvider):
        """Set or update embedding provider."""
        self.embedding_provider = provider

    def add_qa_pair(self, qa_pair: QAPair) -> None:
        """Add single QA pair to collection."""
        if not self.embedding_provider:
            raise ValueError("Embedding provider not set")

        document_text = qa_pair.to_document_text()
        embedding = self.embedding_provider.embed_text(document_text)

        metadata = {
            "department": qa_pair.department.value,
            "user_type": qa_pair.user_type.value,
            "question": qa_pair.question,
            "answer": qa_pair.answer,
            "keywords": ",".join(qa_pair.keywords),
        }

        self.collection.add(
            ids=[qa_pair.id],
            embeddings=[embedding],
            documents=[document_text],
            metadatas=[metadata]
        )

    def add_qa_pairs_batch(self, qa_pairs: list[QAPair], batch_size: int = 50) -> None:
        """
        Batch add multiple QA pairs.

        Args:
            qa_pairs: List of QAPair objects
            batch_size: Number of items per batch for embedding
        """
        if not qa_pairs:
            return

        if not self.embedding_provider:
            raise ValueError("Embedding provider not set")

        # Process in batches
        for i in range(0, len(qa_pairs), batch_size):
            batch = qa_pairs[i:i + batch_size]

            documents = []
            metadatas = []
            ids = []

            for qa in batch:
                documents.append(qa.to_document_text())
                ids.append(qa.id)
                metadatas.append({
                    "department": qa.department.value,
                    "user_type": qa.user_type.value,
                    "question": qa.question,
                    "answer": qa.answer,
                    "keywords": ",".join(qa.keywords),
                })

            # Batch embed
            embeddings = self.embedding_provider.embed_texts(documents)

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

            print(f"  Added batch {i // batch_size + 1}: {len(batch)} documents")

    def query(
        self,
        query_text: str,
        department: Optional[Department] = None,
        user_type: Optional[UserType] = None,
        n_results: int = 10,
        score_threshold: Optional[float] = None
    ) -> list[RetrievedDocument]:
        """
        Query the collection with optional filtering.

        Args:
            query_text: User's query
            department: Filter by department
            user_type: Filter by user type
            n_results: Maximum results to return
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of RetrievedDocument objects
        """
        if not self.embedding_provider:
            raise ValueError("Embedding provider not set")

        # Build where clause
        where_clause = self._build_where_clause(department, user_type)

        # Generate query embedding
        query_embedding = self.embedding_provider.embed_text(query_text)

        # Execute query
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )

        # Convert to RetrievedDocument objects
        return self._parse_results(results, score_threshold)

    def _build_where_clause(
        self,
        department: Optional[Department],
        user_type: Optional[UserType]
    ) -> Optional[dict]:
        """Build ChromaDB where clause for filtering."""
        conditions = []

        if department and department != Department.UNKNOWN:
            conditions.append({"department": {"$eq": department.value}})
        if user_type and user_type != UserType.UNKNOWN:
            conditions.append({"user_type": {"$eq": user_type.value}})

        if len(conditions) == 0:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    def _parse_results(
        self,
        results: dict,
        score_threshold: Optional[float]
    ) -> list[RetrievedDocument]:
        """Parse ChromaDB results into RetrievedDocument objects."""
        retrieved_docs = []

        if not results or not results['ids'] or not results['ids'][0]:
            return retrieved_docs

        for i, doc_id in enumerate(results['ids'][0]):
            # Convert distance to similarity (cosine distance)
            distance = results['distances'][0][i]
            similarity = 1 - distance

            # Apply threshold
            if score_threshold and similarity < score_threshold:
                continue

            metadata = results['metadatas'][0][i]

            retrieved_docs.append(RetrievedDocument(
                id=doc_id,
                content=results['documents'][0][i],
                question=metadata.get('question', ''),
                answer=metadata.get('answer', ''),
                department=Department(metadata.get('department', 'unknown')),
                similarity_score=similarity,
                metadata=metadata
            ))

        return retrieved_docs

    def get_document_count(self, department: Optional[Department] = None) -> int:
        """Get count of documents, optionally filtered by department."""
        if department and department != Department.UNKNOWN:
            results = self.collection.get(
                where={"department": {"$eq": department.value}},
                include=[]
            )
            return len(results['ids'])
        return self.collection.count()

    def reset_collection(self) -> None:
        """Delete and recreate collection."""
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        self._collection = None
        _ = self.collection  # Recreate
        print(f"Collection '{self.COLLECTION_NAME}' reset")
