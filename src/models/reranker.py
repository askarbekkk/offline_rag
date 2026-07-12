import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Optional

from langchain_core.documents import Document
import logging
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class RerankResult:
    """
    Data class to hold reranking results with scores and rankings.

    Attributes:
        document (Document): The LangChain document being ranked
        score (float): Relevance score assigned by the reranker
        original_rank (int): Original position in the input list
    """

    document: Document
    score: float
    original_rank: int


class Reranker:
    """
    Neural document reranker using transformer models for improved search relevance.

    This class implements a document reranking system using pretrained transformer
    models from Hugging Face. It processes query-document pairs to assign relevance
    scores and reorder documents based on their semantic similarity to the query.

    Features:
        - Batch processing for efficiency
        - GPU acceleration support
        - Configurable model selection
        - Optional top-k filtering
        - Explanation generation

    Attributes:
        config (Dict): Configuration settings
        device (str): Computing device (cuda/cpu)
        model_name (str): HuggingFace model identifier
        batch_size (int): Processing batch size
        model: Transformer model instance
        tokenizer: Associated tokenizer
    """

    def __init__(self, config: Dict):
        """
        Initialize the Reranker with specified configuration.

        Args:
            config (Dict): Configuration dictionary containing:
                - model:
                    - device: Computing device (cuda/cpu)
                    - rerank_model: Model identifier
                - processing:
                    - batch_size_reranking: Batch size for processing

        Raises:
            Exception: If model initialization fails

        Note:
            - Model is NOT loaded on init (lazy loading for memory efficiency)
            - Call load_model() before reranking, unload_model() after
            - Sets up logging
        """
        self.config = config
        self.device = config["model"].get("device_rerank", "cpu")
        self.model_name = config["model"]["rerank_model"]
        self.batch_size = config["processing"]["batch_size_reranking"]

        self.logger = logging.getLogger(__name__)

        # Lazy loading - model starts as None
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        """
        Load the reranker model into memory.

        Call this before reranking operations, then call unload_model()
        immediately after to free ~1GB RAM for the LLM.
        """
        if self.model is not None:
            return  # Already loaded

        try:
            self.logger.info(f"Loading reranker model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model.to(self.device)
            self.model.eval()

            self.logger.info(f"Reranker model loaded: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Error loading reranker: {str(e)}")
            raise

    def unload_model(self) -> None:
        """
        Unload the reranker model to free RAM.

        Must be called after reranking is complete.
        Forces garbage collection to release memory.
        """
        if self.model is None:
            return

        self.logger.info("Unloading reranker model to free memory")

        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None

        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("Reranker model unloaded, memory freed")

    def rerank(
        self, query: str, documents: List[Document], top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank documents based on their relevance to the query.

        Processes query-document pairs through the transformer model to
        generate relevance scores and reorder documents accordingly.

        Args:
            query (str): Search query text
            documents (List[Document]): List of documents to rerank
            top_k (Optional[int]): Limit results to top k documents

        Returns:
            List[RerankResult]: Reranked documents with scores and rankings

        Note:
            - Loads model on demand, unloads after processing
            - Processes documents in batches for efficiency
            - Preserves original ranking information
            - Shows progress bar during processing
            - Returns all documents if top_k is None
        """
        if not documents:
            return []

        self.logger.info(f"Reranking {len(documents)} documents for query: {query}")

        try:
            # Load model only for reranking
            self.load_model()

            # Prepare query-document pairs
            pairs = [[query, doc.page_content] for doc in documents]

            # Process in batches with progress tracking
            all_scores = []
            for i in tqdm(
                range(0, len(pairs), self.batch_size), desc="Reranking documents"
            ):
                batch_pairs = pairs[i : i + self.batch_size]
                batch_scores = self._process_batch(batch_pairs)
                all_scores.extend(batch_scores)

            # Create and sort results
            rerank_results = [
                RerankResult(document=doc, score=float(score), original_rank=idx)
                for idx, (score, doc) in enumerate(zip(all_scores, documents))
            ]

            rerank_results.sort(key=lambda x: x.score, reverse=True)

            return rerank_results[:top_k] if top_k else rerank_results

        finally:
            # Always unload model to free memory
            self.unload_model()

    def _process_batch(self, batch_pairs: List[List[str]]) -> List[float]:
        """
        Process a batch of query-document pairs through the model.

        Args:
            batch_pairs (List[List[str]]): List of [query, document] pairs

        Returns:
            List[float]: Relevance scores for each pair

        Note:
            - Uses torch.no_grad() for inference
            - Handles tokenization and device placement
            - Truncates inputs to max_length=512
            - Returns scores as Python list
        """
        with torch.no_grad():
            # Tokenize and move to device
            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)

            # Get model predictions
            outputs = self.model(**inputs, return_dict=True)
            scores = outputs.logits.view(
                -1,
            ).float()

            return scores.cpu().tolist()

    def rerank_with_explanations(
        self, query: str, documents: List[Document], top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Rerank documents and provide explanations for the rankings.

        Extended version of rerank() that includes additional information
        explaining why documents were ranked in their positions.

        Args:
            query (str): Search query text
            documents (List[Document]): List of documents to rerank
            top_k (Optional[int]): Limit results to top k documents

        Returns:
            List[Dict]: Reranked documents with detailed explanations including:
                - document: Original document
                - score: Relevance score
                - original_rank: Initial position
                - explanation: Dict containing:
                    - relevance_score: Formatted score
                    - rank_change: Position change
                    - content_length: Document length

        Note:
            - Builds on basic reranking
            - Adds interpretability information
            - Tracks ranking changes
        """
        rerank_results = self.rerank(query, documents, top_k)

        explained_results = []
        for result in rerank_results:
            explanation = {
                "document": result.document,
                "score": result.score,
                "original_rank": result.original_rank,
                "explanation": {
                    "relevance_score": f"{result.score:.4f}",
                    "rank_change": result.original_rank - len(explained_results),
                    "content_length": len(result.document.page_content),
                },
            }
            explained_results.append(explanation)

        return explained_results
