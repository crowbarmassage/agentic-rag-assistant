"""Data ingestion pipeline for loading FAQs into ChromaDB."""

from typing import Optional

from src.models import Department
from src.utils.data_loader import get_all_qa_pairs
from src.providers.base import BaseEmbeddingProvider
from src.providers.embedding_providers import EmbeddingProviderFactory
from .chroma_client import ChromaDBClient


def ingest_faqs(
    data_dir: str = "./data/raw",
    chroma_dir: str = "./data/chroma_db",
    embedding_provider: Optional[BaseEmbeddingProvider] = None,
    embedding_provider_type: str = "sentence_transformers",
    embedding_model: Optional[str] = None,
    reset_collection: bool = False
) -> ChromaDBClient:
    """
    Ingest all FAQ data into ChromaDB.

    Args:
        data_dir: Directory containing FAQ JSON files
        chroma_dir: ChromaDB persistence directory
        embedding_provider: Pre-configured embedding provider (optional)
        embedding_provider_type: Embedding provider type if no provider given
        embedding_model: Embedding model if no provider given
        reset_collection: Whether to reset collection before ingestion

    Returns:
        Configured ChromaDBClient instance
    """
    # Setup embedding provider
    if embedding_provider is None:
        embedding_provider = EmbeddingProviderFactory.create(
            provider=embedding_provider_type,
            model=embedding_model
        )

    print(f"Using embedding provider: {embedding_provider.provider_name}")
    print(f"Embedding dimension: {embedding_provider.dimension}")

    # Setup ChromaDB client
    chroma = ChromaDBClient(
        persist_directory=chroma_dir,
        embedding_provider=embedding_provider
    )

    # Reset if requested
    if reset_collection:
        print("Resetting collection...")
        chroma.reset_collection()

    # Load all QA pairs
    all_pairs = get_all_qa_pairs(data_dir)
    print(f"Loaded {len(all_pairs)} QA pairs from {data_dir}")

    # Check for existing documents
    existing_count = chroma.get_document_count()
    if existing_count > 0:
        print(f"Collection already has {existing_count} documents")
        if not reset_collection:
            print("Skipping ingestion. Use reset_collection=True to re-ingest.")
            return chroma

    # Ingest
    print("Ingesting documents...")
    chroma.add_qa_pairs_batch(all_pairs, batch_size=20)

    # Verify
    final_count = chroma.get_document_count()
    print(f"\n=== Ingestion Complete ===")
    print(f"Total documents: {final_count}")

    for dept in Department.valid_departments():
        dept_count = chroma.get_document_count(dept)
        print(f"  {dept.value}: {dept_count} documents")

    return chroma


def verify_ingestion(
    chroma_dir: str = "./data/chroma_db",
    embedding_provider: Optional[BaseEmbeddingProvider] = None
) -> bool:
    """
    Verify that ingestion was successful.

    Args:
        chroma_dir: ChromaDB persistence directory
        embedding_provider: Embedding provider for test queries

    Returns:
        True if verification passes
    """
    if embedding_provider is None:
        embedding_provider = EmbeddingProviderFactory.create("sentence_transformers")

    chroma = ChromaDBClient(
        persist_directory=chroma_dir,
        embedding_provider=embedding_provider
    )

    # Check counts
    total = chroma.get_document_count()
    print(f"Total documents: {total}")

    if total == 0:
        print("VERIFICATION FAILED: No documents in collection")
        return False

    # Test query for each department
    test_queries = {
        Department.HR: "How do I apply for time off?",
        Department.IT_SUPPORT: "I forgot my password",
        Department.BILLING: "I need a refund",
        Department.SHIPPING: "Where is my order?",
    }

    print("\nTest queries:")
    for dept, query in test_queries.items():
        results = chroma.query(
            query_text=query,
            department=dept,
            n_results=1
        )

        if results:
            print(f"  [{dept.value}] '{query[:30]}...' -> Score: {results[0].similarity_score:.3f}")
        else:
            print(f"  [{dept.value}] '{query[:30]}...' -> NO RESULTS")

    return True


if __name__ == "__main__":
    # Run ingestion
    ingest_faqs(reset_collection=True)
    verify_ingestion()
