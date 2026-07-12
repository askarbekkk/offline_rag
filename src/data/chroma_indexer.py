"""
ChromaDB-based document indexing and retrieval system with memory management.

This module provides persistent vector storage using ChromaDB with BAAI/bge-m3
embeddings. It implements load/unload patterns to minimize memory usage on
resource-constrained systems (8GB RAM CPU-only).

Features:
    - Persistent ChromaDB storage on disk
    - Load/unload embedding model on demand (memory discipline)
    - Similarity search with configurable top-k
    - Metadata filtering support
    - Cache-aware indexing (skip if already indexed)
"""

import logging
from typing import List, Optional, Dict
from pathlib import Path
import gc

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import torch


logger = logging.getLogger(__name__)


class ChromaIndexer:
    """
    Persistent document indexer using ChromaDB with on-demand model loading.

    Design for 8GB RAM:
    - Embedding model (~2.2GB) is loaded only during indexing or query embedding
    - Immediately unloaded after use to free memory for LLM
    - ChromaDB itself is lightweight and stays loaded

    Attributes:
        persist_dir (str): Directory for ChromaDB persistent storage
        collection_name (str): Name of the ChromaDB collection
        batch_size (int): Batch size for embedding generation
        model_name (str): HuggingFace embedding model name
        _model: Currently loaded embedding model (None if unloaded)
        _client: ChromaDB client instance
        _collection: ChromaDB collection instance
    """

    def __init__(self, config: Dict):
        """
        Initialize ChromaIndexer.

        Args:
            config (Dict): Configuration with:
                - vector_store.persist_directory: ChromaDB storage path
                - vector_store.collection_name: Collection name
                - model.embedding_model_hf: Embedding model name
                - processing.batch_size_embeddings: Batch size
        """
        vector_config = config.get("vector_store", {})
        self.persist_dir = vector_config.get("persist_directory", "./cache/chroma_db")
        self.collection_name = vector_config.get("collection_name", "rag_documents")
        self.top_k = vector_config.get("top_k", 15)
        self.similarity_threshold = vector_config.get("similarity_threshold", 0.3)
        self.model_name = config["model"]["embedding_model_hf"]
        self.batch_size = config["processing"]["batch_size_embeddings"]
        self.device = config["model"].get("device", "cpu")

        self._model = None
        self._client = None
        self._collection = None

        self.logger = logging.getLogger(__name__)

        # Ensure persist directory exists
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self._init_client()

    def _init_client(self) -> None:
        """Initialize ChromaDB client with persistent storage."""
        try:
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False,
                ),
            )
            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
            count = self._collection.count()
            self.logger.info(
                f"ChromaDB initialized at {self.persist_dir} with {count} documents"
            )
        except Exception as e:
            self.logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise

    def load_embedding_model(self) -> None:
        """
        Load the embedding model into memory.

        Call this before embedding operations, then call unload_embedding_model()
        immediately after to free ~2.2GB RAM for the LLM.
        """
        if self._model is not None:
            return  # Already loaded

        self.logger.info(f"Loading embedding model: {self.model_name}")
        model_kwargs = {"device": self.device, "trust_remote_code": True}
        encode_kwargs = {"normalize_embeddings": True, "batch_size": self.batch_size}

        self._model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            show_progress=True,
        )
        self.logger.info("Embedding model loaded successfully")

    def unload_embedding_model(self) -> None:
        """
        Unload the embedding model to free RAM.

        Must be called after indexing or query embedding is complete.
        Forces garbage collection to release GPU/CPU memory.
        """
        if self._model is None:
            return

        self.logger.info("Unloading embedding model to free memory")

        # Delete model reference
        del self._model
        self._model = None

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("Embedding model unloaded, memory freed")

    def index_documents(self, chunks: List[Document]) -> int:
        """
        Index document chunks into ChromaDB.

        Loads embedding model, generates embeddings, stores in ChromaDB,
        then unloads model.

        Args:
            chunks (List[Document]): Document chunks to index

        Returns:
            int: Number of documents indexed
        """
        if not chunks:
            self.logger.warning("No chunks to index")
            return 0

        # Generate deterministic IDs up front (cheap) so we can skip chunks
        # that are already persisted in ChromaDB from a previous run, instead
        # of re-embedding the whole corpus (which is expensive) every time
        # this is called.
        import hashlib

        all_texts = [chunk.page_content for chunk in chunks]
        all_metadatas = []
        all_ids = []
        for i, chunk in enumerate(chunks):
            metadata = dict(chunk.metadata)
            if "source" not in metadata:
                metadata["source"] = "unknown"
            all_metadatas.append(metadata)
            doc_id = hashlib.md5(
                f"{chunk.page_content[:100]}_{i}".encode()
            ).hexdigest()
            all_ids.append(doc_id)

        existing_ids = set(self._collection.get(ids=all_ids, include=[])["ids"])
        new_indices = [i for i, doc_id in enumerate(all_ids) if doc_id not in existing_ids]

        if not new_indices:
            self.logger.info(
                f"All {len(chunks)} chunks already indexed in ChromaDB. Skipping."
            )
            return 0

        texts = [all_texts[i] for i in new_indices]
        metadatas = [all_metadatas[i] for i in new_indices]
        ids = [all_ids[i] for i in new_indices]

        self.logger.info(
            f"Indexing {len(texts)} new document chunks "
            f"({len(chunks) - len(texts)} already indexed, skipped)"
        )

        try:
            # Load model only for embedding
            self.load_embedding_model()

            # Generate embeddings (model is loaded)
            embeddings = self._model.embed_documents(texts)

            # Add to ChromaDB in batches
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                end = min(i + batch_size, len(texts))
                self._collection.add(
                    embeddings=embeddings[i:end],
                    documents=texts[i:end],
                    metadatas=metadatas[i:end],
                    ids=ids[i:end],
                )
                self.logger.debug(
                    f"Indexed batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}"
                )

            count = self._collection.count()
            self.logger.info(
                f"Successfully indexed {len(texts)} chunks. Total in DB: {count}"
            )

            return len(texts)

        except Exception as e:
            self.logger.error(f"Error indexing documents: {str(e)}")
            raise
        finally:
            # Always unload model to free memory
            self.unload_embedding_model()

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Loads embedding model, embeds query, searches ChromaDB, unloads model.

        Args:
            query (str): Search query
            top_k (Optional[int]): Number of results to return. Defaults to config value.

        Returns:
            List[Document]: Retrieved documents with scores in metadata
        """
        if top_k is None:
            top_k = self.top_k

        try:
            # Load model only for query embedding
            self.load_embedding_model()

            # Embed the query
            query_embedding = self._model.embed_query(query)

            # Search ChromaDB
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            # Convert to Document objects
            documents = []
            if results["documents"] and results["documents"][0]:
                for i, (text, metadata, distance) in enumerate(
                    zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                    )
                ):
                    # Convert distance to similarity score (1 - distance for cosine)
                    score = 1.0 - distance

                    doc = Document(
                        page_content=text,
                        metadata={**metadata, "score": score, "distance": distance},
                    )
                    documents.append(doc)

            return documents

        except Exception as e:
            self.logger.error(f"Error retrieving documents: {str(e)}")
            return []
        finally:
            # Always unload model to free memory
            self.unload_embedding_model()

    def retrieve_with_scores(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve documents with detailed score information.

        Similar to retrieve() but returns more detail including raw distances.

        Args:
            query (str): Search query
            top_k (Optional[int]): Number of results

        Returns:
            List[Dict]: Each dict has 'document', 'score', 'distance'
        """
        docs = self.retrieve(query, top_k)
        results = []
        for doc in docs:
            results.append(
                {
                    "document": doc,
                    "score": doc.metadata.get("score", 0.0),
                    "distance": doc.metadata.get("distance", 0.0),
                    "source": doc.metadata.get("source", "unknown"),
                }
            )
        return results

    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the indexed collection.

        Returns:
            Dict: Collection statistics
        """
        try:
            count = self._collection.count()
            return {
                "count": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_dir,
            }
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {str(e)}")
            return {"count": 0}

    def delete_collection(self) -> None:
        """Delete the entire collection (use with caution)."""
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
            self.logger.info(
                f"Collection '{self.collection_name}' deleted and recreated"
            )
        except Exception as e:
            self.logger.error(f"Error deleting collection: {str(e)}")
