"""
DSPy-Voyage Retriever Integration (V1.0)

Integrates Voyage AI's advanced search capabilities with DSPy's retriever system.

Key Features:
1. VoyageRetriever - Custom DSPy retriever using Voyage embeddings
2. VoyageEmbedder - DSPy Embedder wrapper for Voyage models
3. HybridRetriever - Combines vector + BM25 for optimal retrieval
4. RAGProposalFn - Custom instruction proposer using retrieved context

Based on Official DSPy Research:
- dspy.Embedder(callable) for custom embedding functions
- dspy.configure(rm=retriever) for retriever injection
- RAG-enhanced instruction proposer pattern

Repository: https://github.com/stanfordnlp/dspy
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import structlog

# Check DSPy availability
DSPY_AVAILABLE = False
dspy = None

try:
    import dspy as _dspy
    dspy = _dspy
    DSPY_AVAILABLE = True
except ImportError:
    pass

# Import voyage infrastructure
VOYAGE_AVAILABLE = False
try:
    from core.orchestration.embedding_layer import (
        EmbeddingLayer,
        EmbeddingConfig,
        EmbeddingModel,
        InputType,
        EmbeddingResult,
        create_embedding_layer,
    )
    VOYAGE_AVAILABLE = True
except ImportError:
    pass

# Register adapter status
from . import register_adapter
register_adapter("dspy_voyage", DSPY_AVAILABLE and VOYAGE_AVAILABLE, "1.0.0")

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class RetrievedPassage:
    """A passage retrieved from the corpus."""
    text: str
    score: float
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text


@dataclass
class RetrieverConfig:
    """Configuration for Voyage retriever."""
    model: str = "voyage-4-large"
    top_k: int = 5
    use_mmr: bool = False
    lambda_mult: float = 0.7
    use_hybrid: bool = True
    hybrid_alpha: float = 0.5
    min_score: float = 0.0
    cache_enabled: bool = True


# =============================================================================
# Voyage Embedder for DSPy
# =============================================================================

class VoyageEmbedder:
    """
    DSPy-compatible embedder using Voyage AI.

    This class wraps the Voyage EmbeddingLayer to work with DSPy's
    dspy.Embedder(callable) pattern.

    Usage:
        embedder = VoyageEmbedder()
        await embedder.initialize()

        # Use with DSPy
        dspy_embedder = dspy.Embedder(embedder.embed_sync)
    """

    def __init__(
        self,
        model: str = EmbeddingModel.VOYAGE_4_LARGE.value if VOYAGE_AVAILABLE else "voyage-4-large",
        embedding_layer: Optional["EmbeddingLayer"] = None,
    ):
        self.model = model
        self._embedding_layer = embedding_layer
        self._initialized = False

    async def initialize(self) -> "VoyageEmbedder":
        """Initialize the embedding layer."""
        if self._initialized:
            return self

        if self._embedding_layer is None:
            self._embedding_layer = create_embedding_layer(
                model=self.model,
                cache_enabled=True,
            )
            await self._embedding_layer.initialize()

        self._initialized = True
        return self

    async def embed(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        """
        Embed texts using Voyage AI (async).

        Args:
            texts: List of texts to embed
            input_type: "document" or "query"

        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            raise RuntimeError("Embedder not initialized. Call await initialize() first.")

        it = InputType.DOCUMENT if input_type == "document" else InputType.QUERY
        result = await self._embedding_layer.embed(texts, input_type=it)
        return result.embeddings

    def embed_sync(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Synchronous embedding for DSPy compatibility.

        DSPy's Embedder expects a sync callable. This runs the async
        embed in an event loop.

        Args:
            texts: Text or list of texts to embed

        Returns:
            List of embedding vectors
        """
        if isinstance(texts, str):
            texts = [texts]

        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, use nest_asyncio or run in executor
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self.embed(texts))
        except RuntimeError:
            # No running loop, create one
            return asyncio.run(self.embed(texts))

    def __call__(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Make embedder callable for DSPy compatibility."""
        return self.embed_sync(texts)


# =============================================================================
# Voyage Retriever for DSPy
# =============================================================================

class VoyageRetriever:
    """
    DSPy-compatible retriever using Voyage AI with hybrid search.

    Supports:
    - Pure vector search using Voyage embeddings
    - Hybrid search (vector + BM25)
    - MMR for diversity
    - Metadata filtering

    Usage:
        retriever = VoyageRetriever(corpus=["doc1", "doc2", ...])
        await retriever.initialize()

        # Use with DSPy
        dspy.configure(rm=retriever)

        # Direct usage
        results = await retriever.retrieve("query", top_k=5)
    """

    def __init__(
        self,
        corpus: Optional[List[str]] = None,
        corpus_metadata: Optional[List[Dict[str, Any]]] = None,
        config: Optional[RetrieverConfig] = None,
        embedding_layer: Optional["EmbeddingLayer"] = None,
    ):
        self.corpus = corpus or []
        self.corpus_metadata = corpus_metadata or [{} for _ in self.corpus]
        self.config = config or RetrieverConfig()

        self._embedding_layer = embedding_layer
        self._doc_embeddings: Optional[List[List[float]]] = None
        self._initialized = False

    async def initialize(self) -> "VoyageRetriever":
        """Initialize retriever with corpus embeddings."""
        if self._initialized:
            return self

        # Create embedding layer
        if self._embedding_layer is None:
            self._embedding_layer = create_embedding_layer(
                model=self.config.model,
                cache_enabled=self.config.cache_enabled,
            )
            await self._embedding_layer.initialize()

        # Pre-compute corpus embeddings
        if self.corpus:
            result = await self._embedding_layer.embed(
                texts=self.corpus,
                input_type=InputType.DOCUMENT,
            )
            self._doc_embeddings = result.embeddings
            logger.info(
                "corpus_embeddings_computed",
                count=len(self.corpus),
                dimension=len(self._doc_embeddings[0]) if self._doc_embeddings else 0,
            )

        self._initialized = True
        return self

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ):
        """Add documents to corpus (requires re-initialization)."""
        self.corpus.extend(documents)
        self.corpus_metadata.extend(metadata or [{} for _ in documents])
        self._initialized = False  # Force re-embedding

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[RetrievedPassage]:
        """
        Retrieve relevant passages for a query.

        Args:
            query: Search query
            top_k: Number of results (overrides config)
            filter_fn: Optional function to filter by metadata

        Returns:
            List of RetrievedPassage objects
        """
        if not self._initialized:
            await self.initialize()

        if not self.corpus or not self._doc_embeddings:
            return []

        k = top_k or self.config.top_k

        # Choose search method
        if self.config.use_hybrid:
            results = await self._hybrid_search(query, k)
        elif self.config.use_mmr:
            results = await self._mmr_search(query, k)
        else:
            results = await self._vector_search(query, k)

        # Apply metadata filter
        if filter_fn:
            results = [r for r in results if filter_fn(r.metadata)]

        # Apply score threshold
        results = [r for r in results if r.score >= self.config.min_score]

        return results[:k]

    async def _vector_search(self, query: str, top_k: int) -> List[RetrievedPassage]:
        """Pure vector search."""
        results = await self._embedding_layer.semantic_search(
            query=query,
            documents=self.corpus,
            doc_embeddings=self._doc_embeddings,
            top_k=top_k,
        )

        return [
            RetrievedPassage(
                text=self.corpus[idx],
                score=score,
                index=idx,
                metadata=self.corpus_metadata[idx],
            )
            for idx, score, _ in results
        ]

    async def _hybrid_search(self, query: str, top_k: int) -> List[RetrievedPassage]:
        """Hybrid vector + BM25 search."""
        results = await self._embedding_layer.hybrid_search(
            query=query,
            documents=self.corpus,
            doc_embeddings=self._doc_embeddings,
            top_k=top_k,
            alpha=self.config.hybrid_alpha,
        )

        return [
            RetrievedPassage(
                text=self.corpus[idx],
                score=score,
                index=idx,
                metadata=self.corpus_metadata[idx],
            )
            for idx, score, _ in results
        ]

    async def _mmr_search(self, query: str, top_k: int) -> List[RetrievedPassage]:
        """Maximal Marginal Relevance search for diversity."""
        results = await self._embedding_layer.semantic_search_mmr(
            query=query,
            documents=self.corpus,
            doc_embeddings=self._doc_embeddings,
            top_k=top_k,
            lambda_mult=self.config.lambda_mult,
            fetch_k=min(len(self.corpus), top_k * 3),
        )

        return [
            RetrievedPassage(
                text=self.corpus[idx],
                score=score,
                index=idx,
                metadata=self.corpus_metadata[idx],
            )
            for idx, score, _ in results
        ]

    def __call__(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[str]:
        """
        DSPy-compatible retrieval (sync).

        Returns list of strings for DSPy compatibility.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of document strings
        """
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
            passages = loop.run_until_complete(self.retrieve(query, top_k=k))
        except RuntimeError:
            passages = asyncio.run(self.retrieve(query, top_k=k))

        return [p.text for p in passages]


# =============================================================================
# DSPy RAG Module with Voyage
# =============================================================================

class VoyageRAGModule:
    """
    RAG module using Voyage retriever with DSPy.

    Provides a complete RAG pipeline:
    1. Retrieve relevant context using Voyage
    2. Format context for prompt
    3. Generate answer using DSPy module

    Usage:
        rag = VoyageRAGModule(corpus=["doc1", "doc2"])
        await rag.initialize()

        answer = await rag.query("What is X?")
    """

    def __init__(
        self,
        corpus: List[str],
        corpus_metadata: Optional[List[Dict[str, Any]]] = None,
        retriever_config: Optional[RetrieverConfig] = None,
        answer_signature: str = "context, question -> answer",
    ):
        self.retriever = VoyageRetriever(
            corpus=corpus,
            corpus_metadata=corpus_metadata,
            config=retriever_config,
        )
        self.answer_signature = answer_signature
        self._dspy_module = None
        self._initialized = False

    async def initialize(self) -> "VoyageRAGModule":
        """Initialize the RAG module."""
        if self._initialized:
            return self

        await self.retriever.initialize()

        if DSPY_AVAILABLE:
            self._dspy_module = dspy.ChainOfThought(self.answer_signature)

        self._initialized = True
        return self

    async def query(
        self,
        question: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.

        Args:
            question: The question to answer
            top_k: Number of context passages to retrieve

        Returns:
            Dictionary with answer, context, and metadata
        """
        if not self._initialized:
            await self.initialize()

        # Retrieve context
        passages = await self.retriever.retrieve(question, top_k=top_k)
        context = "\n\n".join([p.text for p in passages])

        # Generate answer
        if self._dspy_module:
            result = self._dspy_module(context=context, question=question)
            answer = result.answer
        else:
            answer = f"[DSPy not available] Context: {context[:500]}..."

        return {
            "answer": answer,
            "context": context,
            "passages": passages,
            "num_retrieved": len(passages),
        }


# =============================================================================
# DSPy Configuration Helper
# =============================================================================

async def configure_dspy_with_voyage(
    corpus: Optional[List[str]] = None,
    model: str = "voyage-4-large",
    use_hybrid: bool = True,
    lm_model: str = "ollama_chat/llama3.2",
    lm_provider: str = "auto",
) -> Dict[str, Any]:
    """
    Configure DSPy to use Voyage retriever and embedder with flexible LLM support.

    Args:
        corpus: Optional corpus for retrieval
        model: Voyage embedding model
        use_hybrid: Whether to use hybrid search
        lm_model: Language model for DSPy (see provider patterns below)
        lm_provider: LLM provider - "auto", "ollama", "groq", "together", "claude"

    LLM Model Patterns:
        - Ollama (FREE, local): "ollama_chat/llama3.2", "ollama_chat/deepseek-coder-v2"
        - Groq (FREE tier): "groq/llama-3.3-70b-versatile", "groq/mixtral-8x7b-32768"
        - Together (credits): "together_ai/meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
        - Claude (paid): "claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"

    Returns:
        Configuration dictionary
    """
    if not DSPY_AVAILABLE:
        raise ImportError("DSPy not available. Install with: pip install dspy")

    if not VOYAGE_AVAILABLE:
        raise ImportError("Voyage not available. Install with: pip install voyageai")

    config = {}

    # Create and configure embedder
    embedder = VoyageEmbedder(model=model)
    await embedder.initialize()
    config["embedder"] = embedder

    # Create DSPy embedder
    dspy_embedder = dspy.Embedder(embedder)
    config["dspy_embedder"] = dspy_embedder

    # Create retriever if corpus provided
    if corpus:
        retriever = VoyageRetriever(
            corpus=corpus,
            config=RetrieverConfig(
                model=model,
                use_hybrid=use_hybrid,
            ),
        )
        await retriever.initialize()
        config["retriever"] = retriever

        # Configure DSPy with retriever
        dspy.configure(rm=retriever)
        logger.info("dspy_configured_with_voyage_retriever")

    # Auto-detect provider from model string
    if lm_provider == "auto":
        if lm_model.startswith("ollama"):
            lm_provider = "ollama"
        elif lm_model.startswith("groq/"):
            lm_provider = "groq"
        elif lm_model.startswith("together"):
            lm_provider = "together"
        elif "claude" in lm_model.lower():
            lm_provider = "claude"
        else:
            lm_provider = "ollama"  # Default to free local

    # Configure LM based on provider
    if lm_provider == "claude":
        lm = dspy.Claude(model=lm_model)
        logger.info("dspy_configured_with_claude", model=lm_model)
    elif lm_provider == "ollama":
        # Ollama uses litellm format: ollama_chat/model_name
        # V45 FIX: Environment-configurable Ollama URL (was hardcoded)
        import os
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        lm = dspy.LM(model=lm_model, api_base=ollama_url)
        logger.info("dspy_configured_with_ollama", model=lm_model, base_url=ollama_url)
    elif lm_provider == "groq":
        # Groq uses litellm format: groq/model_name
        lm = dspy.LM(model=lm_model)
        logger.info("dspy_configured_with_groq", model=lm_model)
    elif lm_provider == "together":
        # Together uses litellm format: together_ai/model_name
        lm = dspy.LM(model=lm_model)
        logger.info("dspy_configured_with_together", model=lm_model)
    else:
        lm = dspy.LM(model=lm_model)
        logger.info("dspy_configured_with_generic_lm", model=lm_model)

    dspy.configure(lm=lm)
    config["lm"] = lm
    config["provider"] = lm_provider

    return config


# =============================================================================
# Convenience Exports
# =============================================================================

__all__ = [
    "VoyageEmbedder",
    "VoyageRetriever",
    "VoyageRAGModule",
    "RetrieverConfig",
    "RetrievedPassage",
    "configure_dspy_with_voyage",
    "DSPY_AVAILABLE",
    "VOYAGE_AVAILABLE",
]
