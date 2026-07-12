from typing import List, Tuple, Optional
import logging

from langchain_core.documents import Document

from ..cache.cache import CacheManager
from ..data.chroma_indexer import ChromaIndexer


logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    Vector-based document retrieval using ChromaDB with BGE-M3 embeddings.

    Features:
        - Persistent ChromaDB storage (load once, reuse across restarts)
        - On-demand embedding model loading/unloading for memory efficiency
        - Similarity search with distance-to-score conversion
        - Configurable top-k retrieval
        - Metadata-aware document results
        - Cache integration for compatibility with existing cache system

    Attributes:
        config (dict): Configuration settings
        chroma_indexer (ChromaIndexer): ChromaDB indexing/retrieval engine
        cache_manager (CacheManager): Cache management for fallback
        texts (List[str]): Stored document texts (for index mapping)
        sources (List[str]): Document source references
    """

    def __init__(self, config: dict):
        """
        Initialize VectorRetriever with ChromaDB.

        Args:
            config (dict): Configuration containing:
                - vector_store.*: ChromaDB settings
                - model.*: Model settings
                - processing.*: Processing settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize ChromaDB indexer
        self.chroma_indexer = ChromaIndexer(config)

        # Keep cache manager for backward compatibility
        self.cache_manager = CacheManager(config["paths"]["cache_dir"])

        # Compat: store texts/sources for index mapping
        self.texts = []
        self.sources = []

    def create_vectorstore(self, chunks: List[Document]) -> None:
        """
        Index documents into ChromaDB.

        Args:
            chunks (List[Document]): Document chunks to index

        Note:
            Embeds chunks using BGE-M3, stores in ChromaDB,
            then unloads embedding model to free memory.
        """
        self.logger.info(f"Indexing {len(chunks)} chunks into ChromaDB...")

        # Store for index mapping
        self.texts = [chunk.page_content for chunk in chunks]
        self.sources = [chunk.metadata.get("source", "unknown") for chunk in chunks]

        # Index into ChromaDB
        count = self.chroma_indexer.index_documents(chunks)

        self.logger.info(f"ChromaDB indexing complete. {count} chunks stored.")

    def get_retrieved_docs_indexes(self, retrieved_docs):
        """
        Map retrieved documents back to their original indexes.

        Args:
            retrieved_docs (List[Document]): Retrieved documents

        Returns:
            List[int]: Original indexes of retrieved documents
        """
        indexes = []
        for doc in retrieved_docs:
            for i, orig_text in enumerate(self.texts):
                if doc.page_content == orig_text:
                    indexes.append(i)
                    break
        return indexes

    def retrieve(
        self, query: str, top_k: int = 5
    ) -> Tuple[List[Document], List[float]]:
        """
        Retrieve most relevant documents for the given query.

        Uses ChromaDB with on-demand embedding model loading/unloading.

        Args:
            query (str): Search query
            top_k (int): Number of documents to retrieve (default: 5)

        Returns:
            Tuple[List[Document], List[float]]: Retrieved documents and
                their similarity scores (0-1, higher is better)
        """
        # Retrieve from ChromaDB
        docs = self.chroma_indexer.retrieve(query, top_k=top_k)

        if not docs:
            return [], []

        # Extract scores and create output tuple
        retrieved_docs = []
        scores = []

        for doc in docs:
            score = doc.metadata.get("score", 0.0)

            # Map to original source if available
            source = doc.metadata.get("source", "unknown")

            retrieved_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata={"source": source, "score": score},
                )
            )
            scores.append(score)

        return retrieved_docs, scores
