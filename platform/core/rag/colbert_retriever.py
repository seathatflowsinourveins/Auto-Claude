"""
ColBERT/RAGatouille Retriever Integration

Implements late-interaction retrieval using ColBERT via the RAGatouille library.
ColBERT encodes passages into token-level embeddings and uses MaxSim operators
for contextual matching, providing superior retrieval quality over single-vector
representations.

Features:
- Token-level late interaction for precise semantic matching
- Index-based retrieval for sub-millisecond search
- Reranking mode for use with existing retrievers
- Automatic fallback to dense retrieval when ColBERT unavailable
- Integration with existing RRF fusion pipeline

References:
- RAGatouille: https://github.com/AnswerDotAI/RAGatouille
- ColBERTv2 Paper: "Effective and Efficient Retrieval via Lightweight Late Interaction"

Version: V1.0.0 (2026-02-04)
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS AND PROTOCOLS
# =============================================================================

class RetrieverProtocol(Protocol):
    """Protocol for retrievers compatible with RAGPipeline."""

    @property
    def name(self) -> str:
        """Retriever name for identification."""
        ...

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Retrieve documents for a query."""
        ...


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers (fallback dense retrieval)."""

    def encode(self, texts: List[str]) -> Any:
        """Encode texts to embeddings."""
        ...


@dataclass
class ColBERTConfig:
    """Configuration for ColBERT retriever.

    Attributes:
        index_path: Path to .ragatouille index directory (default: .ragatouille/)
        model_name: ColBERT model name (default: colbert-ir/colbertv2.0)
        index_name: Name of the index to load/create
        n_probe: Number of probes for approximate search (higher = more accurate, slower)
        doc_maxlen: Maximum document length in tokens
        query_maxlen: Maximum query length in tokens
        use_gpu: Whether to use GPU for inference
        fallback_enabled: Enable fallback to dense retrieval when ColBERT unavailable
    """
    index_path: str = ".ragatouille"
    model_name: str = "colbert-ir/colbertv2.0"
    index_name: str = "default"
    n_probe: int = 10
    doc_maxlen: int = 300
    query_maxlen: int = 32
    use_gpu: bool = False
    fallback_enabled: bool = True


@dataclass
class ColBERTDocument:
    """A document retrieved via ColBERT with late-interaction scores."""
    id: str
    content: str
    score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_scores: Optional[List[float]] = None  # Per-token MaxSim scores


@dataclass
class ColBERTResult:
    """Result from ColBERT retrieval operation."""
    query: str
    documents: List[ColBERTDocument]
    latency_ms: float
    method: str  # "colbert" or "fallback"
    index_name: Optional[str] = None


# =============================================================================
# COLBERT RETRIEVER
# =============================================================================

class ColBERTRetriever:
    """
    ColBERT/RAGatouille-based retriever implementing late interaction.

    ColBERT (Contextualized Late Interaction over BERT) represents queries
    and documents as bags of token-level embeddings, enabling fine-grained
    matching via MaxSim operations. This provides superior semantic matching
    compared to single-vector dense retrieval.

    Usage:
        # From existing index
        retriever = ColBERTRetriever.from_index(".ragatouille/colbert/indexes/my_index")
        results = await retriever.retrieve("What is RAG?", top_k=5)

        # Create new index
        retriever = ColBERTRetriever()
        await retriever.index_documents(documents, index_name="my_index")
        results = await retriever.retrieve("What is RAG?", top_k=5)

        # Reranking mode (no index needed)
        retriever = ColBERTRetriever()
        reranked = await retriever.rerank(query, documents, top_k=5)

    Integration with RAGPipeline:
        pipeline = RAGPipeline(
            llm=my_llm,
            retrievers=[colbert_retriever, exa_retriever, tavily_retriever],
        )
        result = await pipeline.run("Complex question")
    """

    # Class-level availability check
    _ragatouille_available: Optional[bool] = None

    def __init__(
        self,
        config: Optional[ColBERTConfig] = None,
        fallback_embedder: Optional[EmbeddingProvider] = None,
    ):
        """Initialize ColBERT retriever.

        Args:
            config: ColBERT configuration
            fallback_embedder: Optional embedding provider for fallback dense retrieval
        """
        self.config = config or ColBERTConfig()
        self.fallback_embedder = fallback_embedder

        self._model = None
        self._index_loaded = False
        self._load_attempted = False

        # Lazy-loaded fallback components
        self._fallback_index: Dict[str, List[float]] = {}
        self._fallback_documents: Dict[str, str] = {}

    @classmethod
    def from_index(
        cls,
        index_path: str,
        config: Optional[ColBERTConfig] = None,
        **kwargs
    ) -> "ColBERTRetriever":
        """Create retriever from an existing ColBERT index.

        Args:
            index_path: Path to the .ragatouille index directory
            config: Optional configuration overrides
            **kwargs: Additional arguments for initialization

        Returns:
            ColBERTRetriever instance with loaded index
        """
        cfg = config or ColBERTConfig()
        cfg.index_path = str(Path(index_path).parent.parent.parent)  # Navigate up from index
        cfg.index_name = Path(index_path).name

        retriever = cls(config=cfg, **kwargs)
        retriever._load_from_index(index_path)
        return retriever

    @property
    def name(self) -> str:
        """Retriever name for pipeline identification."""
        return "colbert"

    @property
    def is_available(self) -> bool:
        """Check if RAGatouille/ColBERT is available."""
        if ColBERTRetriever._ragatouille_available is None:
            ColBERTRetriever._ragatouille_available = self._check_availability()
        return ColBERTRetriever._ragatouille_available

    def _check_availability(self) -> bool:
        """Check if RAGatouille is installed and functional."""
        try:
            from ragatouille import RAGPretrainedModel
            return True
        except ImportError:
            logger.warning(
                "RAGatouille not installed. Install with: pip install ragatouille. "
                "Falling back to dense retrieval."
            )
            return False
        except Exception as e:
            logger.warning(f"RAGatouille check failed: {e}")
            return False

    def _load_model(self) -> bool:
        """Lazy load the RAGatouille model."""
        if self._load_attempted:
            return self._model is not None

        self._load_attempted = True

        if not self.is_available:
            return False

        try:
            from ragatouille import RAGPretrainedModel

            self._model = RAGPretrainedModel.from_pretrained(
                self.config.model_name,
                verbose=0
            )
            logger.info(f"Loaded ColBERT model: {self.config.model_name}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load ColBERT model: {e}")
            return False

    def _load_from_index(self, index_path: str) -> bool:
        """Load model from an existing index.

        Args:
            index_path: Path to the index directory

        Returns:
            True if successfully loaded
        """
        if not self.is_available:
            return False

        try:
            from ragatouille import RAGPretrainedModel

            self._model = RAGPretrainedModel.from_index(index_path)
            self._index_loaded = True
            self._load_attempted = True
            logger.info(f"Loaded ColBERT index from: {index_path}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load ColBERT index: {e}")
            return False

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Retrieve documents using ColBERT late interaction.

        This method implements the RetrieverProtocol for integration
        with RAGPipeline. Uses token-level MaxSim scoring for retrieval.

        Args:
            query: Search query
            top_k: Number of documents to return
            **kwargs: Additional arguments (index_name, n_probe)

        Returns:
            List of document dictionaries with content, score, and metadata
        """
        import time
        start_time = time.time()

        # Try ColBERT first
        if self._index_loaded and self._model is not None:
            try:
                results = await self._colbert_search(query, top_k, **kwargs)
                latency = (time.time() - start_time) * 1000

                return [
                    {
                        "content": doc.content,
                        "score": doc.score,
                        "metadata": {
                            **doc.metadata,
                            "retriever": "colbert",
                            "rank": doc.rank,
                            "latency_ms": latency,
                        }
                    }
                    for doc in results
                ]

            except Exception as e:
                logger.warning(f"ColBERT search failed, using fallback: {e}")

        # Fallback to dense retrieval
        if self.config.fallback_enabled:
            return await self._fallback_retrieve(query, top_k)

        return []

    async def _colbert_search(
        self,
        query: str,
        top_k: int,
        **kwargs
    ) -> List[ColBERTDocument]:
        """Execute ColBERT search on loaded index.

        Args:
            query: Search query
            top_k: Number of results
            **kwargs: Additional ColBERT parameters

        Returns:
            List of ColBERTDocument results
        """
        if self._model is None:
            raise RuntimeError("ColBERT model not loaded")

        # Run search in executor to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._model.search(
                query,
                k=top_k,
            )
        )

        documents = []
        for rank, result in enumerate(results, start=1):
            # RAGatouille returns dicts with content, score, etc.
            doc = ColBERTDocument(
                id=str(result.get("document_id", result.get("doc_id", rank))),
                content=result.get("content", result.get("text", "")),
                score=float(result.get("score", result.get("relevance_score", 0.0))),
                rank=rank,
                metadata={
                    "passage_id": result.get("passage_id"),
                    "document_metadata": result.get("document_metadata", {}),
                },
            )
            documents.append(doc)

        return documents

    async def rerank(
        self,
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        top_k: int = 10,
    ) -> List[ColBERTDocument]:
        """Rerank documents using ColBERT without building an index.

        This mode uses ColBERT as a reranker over results from other retrievers,
        enabling late-interaction scoring without maintaining a separate index.

        Args:
            query: Search query
            documents: List of documents (strings or dicts with 'content' key)
            top_k: Number of top documents to return

        Returns:
            Reranked list of ColBERTDocument
        """
        if not self._load_model():
            logger.warning("ColBERT unavailable for reranking")
            # Return documents in original order with default scores
            return self._passthrough_rerank(documents, top_k)

        # Normalize documents to strings
        doc_texts = []
        doc_metadata = []
        for doc in documents:
            if isinstance(doc, str):
                doc_texts.append(doc)
                doc_metadata.append({})
            elif isinstance(doc, dict):
                doc_texts.append(doc.get("content", str(doc)))
                doc_metadata.append(doc.get("metadata", {}))
            else:
                doc_texts.append(str(doc))
                doc_metadata.append({})

        try:
            # Run reranking in executor
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._model.rerank(
                    query=query,
                    documents=doc_texts,
                    k=top_k,
                )
            )

            reranked = []
            for rank, result in enumerate(results, start=1):
                # Find original metadata
                orig_idx = doc_texts.index(result.get("content", result.get("text", "")))
                metadata = doc_metadata[orig_idx] if orig_idx < len(doc_metadata) else {}

                doc = ColBERTDocument(
                    id=str(result.get("document_id", orig_idx)),
                    content=result.get("content", result.get("text", "")),
                    score=float(result.get("score", result.get("relevance_score", 0.0))),
                    rank=rank,
                    metadata={**metadata, "reranked": True},
                )
                reranked.append(doc)

            return reranked

        except Exception as e:
            logger.warning(f"ColBERT reranking failed: {e}")
            return self._passthrough_rerank(documents, top_k)

    def _passthrough_rerank(
        self,
        documents: List[Union[str, Dict[str, Any]]],
        top_k: int
    ) -> List[ColBERTDocument]:
        """Passthrough when reranking unavailable - preserves order."""
        results = []
        for rank, doc in enumerate(documents[:top_k], start=1):
            if isinstance(doc, str):
                content = doc
                metadata = {}
            elif isinstance(doc, dict):
                content = doc.get("content", str(doc))
                metadata = doc.get("metadata", {})
            else:
                content = str(doc)
                metadata = {}

            results.append(ColBERTDocument(
                id=str(rank),
                content=content,
                score=1.0 / rank,  # Reciprocal rank as fallback score
                rank=rank,
                metadata={**metadata, "passthrough": True},
            ))
        return results

    async def index_documents(
        self,
        documents: List[Union[str, Dict[str, Any]]],
        index_name: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        document_metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Build a ColBERT index from documents.

        Creates a new ColBERT index that can be used for retrieval.
        The index is persisted to disk in the .ragatouille directory.

        Args:
            documents: List of documents (strings or dicts with 'content')
            index_name: Name for the index (default: config.index_name)
            document_ids: Optional list of document IDs
            document_metadatas: Optional list of metadata dicts

        Returns:
            True if indexing succeeded
        """
        if not self._load_model():
            logger.warning("ColBERT unavailable, cannot create index")
            return False

        index_name = index_name or self.config.index_name

        # Normalize documents
        doc_texts = []
        for doc in documents:
            if isinstance(doc, str):
                doc_texts.append(doc)
            elif isinstance(doc, dict):
                doc_texts.append(doc.get("content", str(doc)))
            else:
                doc_texts.append(str(doc))

        try:
            loop = asyncio.get_event_loop()

            index_path = await loop.run_in_executor(
                None,
                lambda: self._model.index(
                    collection=doc_texts,
                    index_name=index_name,
                    document_ids=document_ids,
                    document_metadatas=document_metadatas,
                    max_document_length=self.config.doc_maxlen,
                    split_documents=True,
                )
            )

            self._index_loaded = True
            logger.info(f"Created ColBERT index: {index_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create ColBERT index: {e}")
            return False

    async def add_to_index(
        self,
        documents: List[Union[str, Dict[str, Any]]],
        document_ids: Optional[List[str]] = None,
        document_metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Add documents to an existing index.

        Args:
            documents: Documents to add
            document_ids: Optional document IDs
            document_metadatas: Optional metadata dicts

        Returns:
            True if addition succeeded
        """
        if not self._index_loaded or self._model is None:
            logger.warning("No index loaded, cannot add documents")
            return False

        doc_texts = []
        for doc in documents:
            if isinstance(doc, str):
                doc_texts.append(doc)
            elif isinstance(doc, dict):
                doc_texts.append(doc.get("content", str(doc)))
            else:
                doc_texts.append(str(doc))

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._model.add_to_index(
                    new_collection=doc_texts,
                    new_document_ids=document_ids,
                    new_document_metadatas=document_metadatas,
                )
            )
            logger.info(f"Added {len(doc_texts)} documents to index")
            return True

        except Exception as e:
            logger.error(f"Failed to add to index: {e}")
            return False

    async def _fallback_retrieve(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Fallback dense retrieval when ColBERT is unavailable.

        Uses the fallback_embedder if provided, otherwise returns empty.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of document dictionaries
        """
        if not self.fallback_embedder or not self._fallback_index:
            logger.debug("No fallback retrieval available")
            return []

        try:
            # Embed query
            query_embedding = self.fallback_embedder.encode([query])[0]

            # Compute similarities
            scores = []
            for doc_id, doc_embedding in self._fallback_index.items():
                # Cosine similarity
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                scores.append((doc_id, similarity))

            # Sort by score
            scores.sort(key=lambda x: x[1], reverse=True)

            # Return top-k
            results = []
            for rank, (doc_id, score) in enumerate(scores[:top_k], start=1):
                results.append({
                    "content": self._fallback_documents.get(doc_id, ""),
                    "score": score,
                    "metadata": {
                        "retriever": "fallback_dense",
                        "rank": rank,
                    }
                })

            return results

        except Exception as e:
            logger.warning(f"Fallback retrieval failed: {e}")
            return []

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def set_fallback_documents(
        self,
        documents: Dict[str, str],
        embeddings: Dict[str, List[float]]
    ) -> None:
        """Set documents and embeddings for fallback retrieval.

        Args:
            documents: Dict mapping doc_id to content
            embeddings: Dict mapping doc_id to embedding vector
        """
        self._fallback_documents = documents
        self._fallback_index = embeddings

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get retriever diagnostics and status."""
        return {
            "name": self.name,
            "ragatouille_available": self.is_available,
            "model_loaded": self._model is not None,
            "index_loaded": self._index_loaded,
            "model_name": self.config.model_name,
            "index_name": self.config.index_name,
            "fallback_enabled": self.config.fallback_enabled,
            "fallback_documents_count": len(self._fallback_documents),
        }


# =============================================================================
# COLBERT RERANKER (STANDALONE)
# =============================================================================

class ColBERTReranker:
    """
    Standalone ColBERT reranker for use with existing retrievers.

    Uses ColBERT's late interaction mechanism to rerank results from
    any retriever without maintaining a separate index.

    Usage:
        reranker = ColBERTReranker()

        # Rerank results from another retriever
        results = await other_retriever.retrieve(query, top_k=50)
        reranked = await reranker.rerank(query, results, top_k=10)

    Integration with SemanticReranker:
        The ColBERTReranker can be used alongside or instead of the
        cross-encoder SemanticReranker for improved precision.
    """

    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        """Initialize ColBERT reranker.

        Args:
            model_name: ColBERT model to use
        """
        self.model_name = model_name
        self._retriever = ColBERTRetriever(
            config=ColBERTConfig(model_name=model_name, fallback_enabled=False)
        )

    @property
    def is_available(self) -> bool:
        """Check if ColBERT reranking is available."""
        return self._retriever.is_available

    async def rerank(
        self,
        query: str,
        documents: List[Any],
        top_k: int = 10,
    ) -> List[Any]:
        """Rerank documents using ColBERT late interaction.

        Args:
            query: Search query
            documents: List of documents (various formats supported)
            top_k: Number of documents to return

        Returns:
            Reranked documents in same format as input
        """
        results = await self._retriever.rerank(query, documents, top_k)
        return results


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_colbert_retriever(
    index_path: Optional[str] = None,
    model_name: str = "colbert-ir/colbertv2.0",
    fallback_enabled: bool = True,
    **kwargs
) -> ColBERTRetriever:
    """Factory function to create a ColBERT retriever.

    Args:
        index_path: Optional path to existing index
        model_name: ColBERT model name
        fallback_enabled: Enable dense retrieval fallback
        **kwargs: Additional config parameters

    Returns:
        Configured ColBERTRetriever instance
    """
    config = ColBERTConfig(
        model_name=model_name,
        fallback_enabled=fallback_enabled,
        **kwargs
    )

    if index_path:
        return ColBERTRetriever.from_index(index_path, config=config)

    return ColBERTRetriever(config=config)


def create_colbert_reranker(
    model_name: str = "colbert-ir/colbertv2.0"
) -> ColBERTReranker:
    """Factory function to create a ColBERT reranker.

    Args:
        model_name: ColBERT model name

    Returns:
        ColBERTReranker instance
    """
    return ColBERTReranker(model_name=model_name)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "ColBERTConfig",
    # Data types
    "ColBERTDocument",
    "ColBERTResult",
    # Main retriever
    "ColBERTRetriever",
    # Reranker
    "ColBERTReranker",
    # Factory functions
    "create_colbert_retriever",
    "create_colbert_reranker",
]
