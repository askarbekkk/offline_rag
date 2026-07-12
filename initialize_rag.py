from __future__ import annotations

import os

# Thread-count env vars must be set before anything that transitively pulls in
# torch/numpy/BLAS (HybridRetriever -> chromadb -> torch, below) initializes
# its native runtime: these are read once at first init, so setting them
# later (as config/init_config.yaml's OMP_NUM_THREADS was previously only
# applied inside src/data/loaders/file_loader.py) has no effect on the actual
# embedding/reranking hot path.
try:
    import yaml

    with open("config/init_config.yaml") as _f:
        _omp_threads = yaml.safe_load(_f).get("processing", {}).get("OMP_NUM_THREADS")
except Exception:
    _omp_threads = None
_num_threads = int(_omp_threads) if _omp_threads else max(1, os.cpu_count() or 4)
os.environ.setdefault("OMP_NUM_THREADS", str(_num_threads))

import torch

torch.set_num_threads(_num_threads)

import logging
from typing import Dict, List
from dataclasses import dataclass

# HybridRetriever (ChromaDB) must be imported/constructed before pandas,
# PyMuPDF, tree-sitter, and other native-extension-heavy libraries pulled in
# by DataIngestion/DataPreprocessor: on Windows, whichever of these loads its
# native components into the process first "wins", and importing them ahead
# of chromadb causes a segfault the first time chromadb touches its sqlite
# backend. `from __future__ import annotations` above lets the `pd.DataFrame`
# type hints below stay lazy so pandas can be imported later, after chromadb.
from src.retrieval.hybrid_retriever import HybridRetriever
from src.models.reranker import Reranker
from src.response.response_generator import ResponseGenerator
import pandas as pd
from src.data.data_ingestion import DataIngestion
from src.data.data_preprocessing import DataPreprocessor
from src.utils.helpers import setup_logging, load_config


@dataclass
class RAGComponents:
    retriever: HybridRetriever
    reranker: Reranker
    response_generator: ResponseGenerator
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_results: pd.DataFrame
    test_results: pd.DataFrame
    original_chunks: List
    init_config: Dict
    process_config: Dict


class RAGInitializer:
    def __init__(self, init_config_path: str, process_config_path: str):
        """Initialize the RAG system components and data.

        Args:
            init_config_path (str): Path to initialization configuration file
            process_config_path (str): Path to processing configuration file
        """
        self.init_config = load_config(init_config_path)
        self.process_config = load_config(process_config_path)
        setup_logging(self.init_config)

    def initialize(self) -> RAGComponents:
        """Initialize all components and prepare data.

        Returns:
            RAGComponents: Dataclass containing all initialized components and data
        """
        try:
            # Initialize components
            logging.info("Initializing components...")

            # Initialize retrieval components with both configs
            combined_config = {
                **self.init_config,
                "retrieval": self.process_config["retrieval"],
            }

            # Ensure vector_store config is passed through
            if "vector_store" not in combined_config:
                combined_config["vector_store"] = self.init_config.get(
                    "vector_store",
                    {
                        "type": "chroma",
                        "persist_directory": "./cache/chroma_db",
                        "collection_name": "rag_documents",
                        "top_k": 15,
                        "similarity_threshold": 0.3,
                    },
                )

            # HybridRetriever (ChromaDB) must be constructed before DataIngestion/
            # DataPreprocessor: constructing it first forces chromadb's native
            # sqlite/hnswlib bindings to initialize before the document-loader
            # stack (PyMuPDF, tree-sitter, etc.) loads its own native libraries,
            # which otherwise segfaults on Windows due to a native library conflict.
            retriever = HybridRetriever(combined_config)
            reranker = Reranker(combined_config)
            response_generator = ResponseGenerator(combined_config)

            data_ingestion = DataIngestion(self.init_config)
            data_preprocessor = DataPreprocessor(self.init_config)

            # Load and prepare data
            logging.info("Loading and preprocessing data...")
            train_df, test_df = data_ingestion.load_data()
            docs = data_ingestion.load_documents()
            train_results, test_results = data_ingestion.load_existing_results()

            # Process documents
            logging.info("Processing documents...")
            original_chunks = data_preprocessor.process_documents(docs)

            return RAGComponents(
                retriever=retriever,
                reranker=reranker,
                response_generator=response_generator,
                train_df=train_df,
                test_df=test_df,
                train_results=train_results,
                test_results=test_results,
                original_chunks=original_chunks,
                init_config=self.init_config,
                process_config=self.process_config,
            )

        except Exception as e:
            logging.error(f"Error in RAG initialization: {str(e)}")
            raise


if __name__ == "__main__":
    # This can be used for testing the initialization separately
    initializer = RAGInitializer(
        "config/init_config.yaml", "config/process_config.yaml"
    )
    components = initializer.initialize()
    logging.info("Initialization completed successfully")
