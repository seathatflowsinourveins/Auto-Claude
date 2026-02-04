"""
HyDE: Hypothetical Document Embeddings

This module implements HyDE (Hypothetical Document Embeddings), a retrieval
method that generates hypothetical documents from queries to improve recall
for abstract or complex queries.

Key Features:
- Generate hypothetical documents from queries using LLM
- Embed hypothetical documents for retrieval
- Average multiple hypothetical embeddings for robustness
- Improved recall for abstract and complex queries

Architecture:
    Query -> LLM generates hypothetical answer -> Embed hypothetical document
                                                       |
                                                       v
                                             Vector Search -> Actual Documents

Key Insight: Bridges query-document distribution gap by converting
             questions into document format for document-to-document matching.

Reference: https://arxiv.org/abs/2212.10496

Integration:
    from core.rag.hyde import HyDERetriever, HyDEConfig

    hyde = HyDERetriever(llm=my_llm, embedder=my_embedder, vector_store=my_store)
    results = await hyde.retrieve("What are the benefits of microservices?")
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

# Optional numpy import with fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# PROTOCOLS AND TYPES
# =============================================================================

class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        ...


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def encode(self, texts: Union[str, List[str]]) -> Any:
        """Encode text(s) to embedding(s). Returns numpy array or list."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        ...


class VectorStoreProvider(Protocol):
    """Protocol for vector store providers."""

    async def search(
        self,
        embedding: Any,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search vector store with embedding. Returns list of dicts with 'content' key."""
        ...


class HypotheticalDocumentType(str, Enum):
    """Types of hypothetical documents to generate."""
    ANSWER = "answer"              # Direct answer to the question
    DOCUMENT = "document"          # Document that would contain the answer
    PASSAGE = "passage"            # Passage from a document
    EXPLANATION = "explanation"    # Detailed explanation


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class HyDEConfig:
    """Configuration for HyDE retriever.

    Attributes:
        n_hypothetical: Number of hypothetical documents to generate (default: 5)
        hypothesis_type: Type of hypothetical to generate (default: ANSWER)
        max_tokens: Maximum tokens for hypothetical generation (default: 256)
        temperature: Temperature for generation diversity (default: 0.7)
        embedding_strategy: How to combine embeddings - 'average', 'first', 'all' (default: 'average')
        enable_caching: Whether to cache hypothetical embeddings (default: True)
        top_k: Number of documents to retrieve (default: 5)
        include_original_query: Also search with original query embedding (default: False)
        domain_hint: Optional domain hint for better hypotheticals (default: None)
    """
    n_hypothetical: int = 5
    hypothesis_type: HypotheticalDocumentType = HypotheticalDocumentType.ANSWER
    max_tokens: int = 256
    temperature: float = 0.7
    embedding_strategy: str = "average"  # 'average', 'first', 'all'
    enable_caching: bool = True
    top_k: int = 5
    include_original_query: bool = False
    domain_hint: Optional[str] = None


@dataclass
class HypotheticalDocument:
    """A generated hypothetical document."""
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HyDEResult:
    """Result from HyDE retrieval.

    Attributes:
        documents: Retrieved documents
        hypotheticals: Generated hypothetical documents
        query_embedding: Combined query embedding used for search
        retrieval_count: Number of documents retrieved
    """
    documents: List[Dict[str, Any]]
    hypotheticals: List[HypotheticalDocument]
    query_embedding: Optional[List[float]] = None
    retrieval_count: int = 0


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

class HyDEPrompts:
    """Prompt templates for HyDE hypothetical generation."""

    # Basic answer generation
    GENERATE_ANSWER = """Write a detailed paragraph answering this question:

Question: {query}

{domain_context}
Write a comprehensive answer as if you were an expert in this field. Be specific and informative.

Answer:"""

    # Document generation
    GENERATE_DOCUMENT = """Generate a passage from a document that would answer this question:

Question: {query}

{domain_context}
Write the passage as it might appear in a technical document, textbook, or encyclopedia entry.
Include relevant details, definitions, and examples.

Document passage:"""

    # Passage generation
    GENERATE_PASSAGE = """Write a relevant passage from a knowledge base that addresses this query:

Query: {query}

{domain_context}
Write the passage as if it were extracted from a comprehensive reference source.
Focus on factual information and clear explanations.

Passage:"""

    # Explanation generation
    GENERATE_EXPLANATION = """Provide a detailed explanation for this question:

Question: {query}

{domain_context}
Explain the concept thoroughly, covering background, key points, and practical implications.

Explanation:"""

    # Domain context template
    DOMAIN_CONTEXT = """Context: This is about {domain}. Use terminology and concepts appropriate for this field.

"""

    @classmethod
    def get_template(cls, doc_type: HypotheticalDocumentType) -> str:
        """Get the appropriate template for a document type."""
        templates = {
            HypotheticalDocumentType.ANSWER: cls.GENERATE_ANSWER,
            HypotheticalDocumentType.DOCUMENT: cls.GENERATE_DOCUMENT,
            HypotheticalDocumentType.PASSAGE: cls.GENERATE_PASSAGE,
            HypotheticalDocumentType.EXPLANATION: cls.GENERATE_EXPLANATION,
        }
        return templates.get(doc_type, cls.GENERATE_ANSWER)


# =============================================================================
# EMBEDDING CACHE
# =============================================================================

class EmbeddingCache:
    """Cache for hypothetical document embeddings."""

    def __init__(self, max_size: int = 1000):
        """Initialize cache.

        Args:
            max_size: Maximum number of cached embeddings
        """
        self.max_size = max_size
        self._cache: Dict[str, List[float]] = {}
        self._access_order: List[str] = []

    def get(self, key: str) -> Optional[List[float]]:
        """Get cached embedding."""
        if key in self._cache:
            # Move to end (most recent)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, key: str, embedding: List[float]) -> None:
        """Cache an embedding."""
        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size and self._access_order:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)

        self._cache[key] = embedding
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()


# =============================================================================
# HYDE RETRIEVER IMPLEMENTATION
# =============================================================================

class HyDERetriever:
    """
    Hypothetical Document Embeddings (HyDE) Retriever.

    Improves retrieval by generating hypothetical documents that would answer
    the query, then using those embeddings to search for similar actual documents.

    Key benefits:
    - Better recall for abstract queries
    - Bridges query-document distribution gap
    - Works well when LLM has domain knowledge

    Example:
        >>> from core.rag.hyde import HyDERetriever, HyDEConfig
        >>>
        >>> config = HyDEConfig(n_hypothetical=5, hypothesis_type=HypotheticalDocumentType.ANSWER)
        >>> hyde = HyDERetriever(
        ...     llm=my_llm,
        ...     embedder=my_embedder,
        ...     vector_store=my_store,
        ...     config=config
        ... )
        >>>
        >>> result = await hyde.retrieve("What are the benefits of microservices?")
        >>> for doc in result.documents:
        ...     print(doc["content"][:100])
    """

    def __init__(
        self,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        vector_store: VectorStoreProvider,
        config: Optional[HyDEConfig] = None,
    ):
        """Initialize HyDE retriever.

        Args:
            llm: LLM provider for hypothetical generation
            embedder: Embedding provider
            vector_store: Vector store for retrieval
            config: Configuration options
        """
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store
        self.config = config or HyDEConfig()
        self.prompts = HyDEPrompts()

        # Initialize cache if enabled
        self._cache = EmbeddingCache() if self.config.enable_caching else None

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> HyDEResult:
        """Retrieve documents using HyDE.

        Args:
            query: The search query
            top_k: Number of documents to retrieve (overrides config)
            **kwargs: Additional arguments passed to vector store

        Returns:
            HyDEResult with retrieved documents and metadata
        """
        top_k = top_k or self.config.top_k

        # Step 1: Generate hypothetical documents
        hypotheticals = await self._generate_hypotheticals(query)

        if not hypotheticals:
            logger.warning("No hypotheticals generated, falling back to direct query")
            # Fall back to direct query embedding
            query_embedding = self._embed_text(query)
            documents = await self.vector_store.search(
                query_embedding,
                top_k=top_k,
                **kwargs
            )
            return HyDEResult(
                documents=documents,
                hypotheticals=[],
                query_embedding=query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding,
                retrieval_count=len(documents)
            )

        # Step 2: Embed hypothetical documents
        hypotheticals = self._embed_hypotheticals(hypotheticals)

        # Step 3: Combine embeddings according to strategy
        combined_embedding = self._combine_embeddings(hypotheticals, query)

        # Step 4: Search vector store
        documents = await self.vector_store.search(
            combined_embedding,
            top_k=top_k,
            **kwargs
        )

        # Step 5: Optionally include original query results
        if self.config.include_original_query:
            query_embedding = self._embed_text(query)
            query_results = await self.vector_store.search(
                query_embedding,
                top_k=top_k,
                **kwargs
            )
            # Merge and deduplicate
            documents = self._merge_results(documents, query_results)

        return HyDEResult(
            documents=documents,
            hypotheticals=hypotheticals,
            query_embedding=combined_embedding.tolist() if hasattr(combined_embedding, 'tolist') else list(combined_embedding),
            retrieval_count=len(documents)
        )

    async def _generate_hypotheticals(self, query: str) -> List[HypotheticalDocument]:
        """Generate hypothetical documents for a query."""
        # Check cache first
        if self._cache is not None:
            cache_key = f"{query}:{self.config.hypothesis_type.value}"
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Using cached hypotheticals for: {query[:50]}...")
                # Reconstruct hypotheticals from cache (simplified)
                return [HypotheticalDocument(content="[cached]", embedding=cached)]

        # Prepare prompt
        template = self.prompts.get_template(self.config.hypothesis_type)

        domain_context = ""
        if self.config.domain_hint:
            domain_context = self.prompts.DOMAIN_CONTEXT.format(domain=self.config.domain_hint)

        prompt = template.format(query=query, domain_context=domain_context)

        # Generate multiple hypotheticals
        hypotheticals: List[HypotheticalDocument] = []

        # Generate in parallel for efficiency
        tasks = [
            self._generate_single_hypothetical(prompt, i)
            for i in range(self.config.n_hypothetical)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, HypotheticalDocument):
                hypotheticals.append(result)
            elif isinstance(result, Exception):
                logger.warning(f"Hypothetical generation failed: {result}")

        return hypotheticals

    async def _generate_single_hypothetical(
        self,
        prompt: str,
        index: int
    ) -> HypotheticalDocument:
        """Generate a single hypothetical document."""
        # Vary temperature slightly for diversity
        temp = self.config.temperature + (index * 0.05)
        temp = min(temp, 1.0)

        try:
            content = await self.llm.generate(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=temp
            )

            return HypotheticalDocument(
                content=content.strip(),
                metadata={"index": index, "temperature": temp}
            )
        except Exception as e:
            logger.error(f"Failed to generate hypothetical {index}: {e}")
            raise

    def _embed_hypotheticals(
        self,
        hypotheticals: List[HypotheticalDocument]
    ) -> List[HypotheticalDocument]:
        """Embed all hypothetical documents."""
        contents = [h.content for h in hypotheticals]

        try:
            embeddings = self.embedder.encode(contents)

            # Convert to list format
            if NUMPY_AVAILABLE and hasattr(embeddings, 'tolist'):
                embeddings_list = embeddings.tolist()
            elif isinstance(embeddings, list):
                embeddings_list = embeddings
            else:
                embeddings_list = [list(e) for e in embeddings]

            # Assign embeddings to hypotheticals
            for hypo, emb in zip(hypotheticals, embeddings_list):
                hypo.embedding = emb

        except Exception as e:
            logger.error(f"Failed to embed hypotheticals: {e}")

        return hypotheticals

    def _embed_text(self, text: str) -> Any:
        """Embed a single text."""
        embedding = self.embedder.encode(text)
        if NUMPY_AVAILABLE and not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        return embedding

    def _combine_embeddings(
        self,
        hypotheticals: List[HypotheticalDocument],
        query: str
    ) -> Any:
        """Combine hypothetical embeddings according to strategy."""
        valid_embeddings = [
            h.embedding for h in hypotheticals
            if h.embedding is not None
        ]

        if not valid_embeddings:
            # Fall back to query embedding
            logger.warning("No valid hypothetical embeddings, using query embedding")
            return self._embed_text(query)

        if NUMPY_AVAILABLE:
            embeddings_array = np.array(valid_embeddings)
        else:
            embeddings_array = valid_embeddings

        if self.config.embedding_strategy == "average":
            if NUMPY_AVAILABLE:
                combined = np.mean(embeddings_array, axis=0)
            else:
                # Pure Python average
                n = len(valid_embeddings)
                dim = len(valid_embeddings[0])
                combined = [
                    sum(emb[i] for emb in valid_embeddings) / n
                    for i in range(dim)
                ]

        elif self.config.embedding_strategy == "first":
            combined = valid_embeddings[0]
            if NUMPY_AVAILABLE and not isinstance(combined, np.ndarray):
                combined = np.array(combined)

        elif self.config.embedding_strategy == "all":
            # For 'all' strategy, return average but caller could handle differently
            if NUMPY_AVAILABLE:
                combined = np.mean(embeddings_array, axis=0)
            else:
                n = len(valid_embeddings)
                dim = len(valid_embeddings[0])
                combined = [
                    sum(emb[i] for emb in valid_embeddings) / n
                    for i in range(dim)
                ]

        else:
            # Default to average
            if NUMPY_AVAILABLE:
                combined = np.mean(embeddings_array, axis=0)
            else:
                n = len(valid_embeddings)
                dim = len(valid_embeddings[0])
                combined = [
                    sum(emb[i] for emb in valid_embeddings) / n
                    for i in range(dim)
                ]

        # Cache the combined embedding
        if self._cache is not None:
            cache_key = f"{query}:{self.config.hypothesis_type.value}"
            combined_list = combined.tolist() if hasattr(combined, 'tolist') else list(combined)
            self._cache.put(cache_key, combined_list)

        return combined

    def _merge_results(
        self,
        hyde_results: List[Dict[str, Any]],
        query_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge and deduplicate results from HyDE and direct query."""
        seen_contents: set = set()
        merged: List[Dict[str, Any]] = []

        # Add HyDE results first (higher priority)
        for doc in hyde_results:
            content = doc.get("content", str(doc))
            content_key = content[:200]  # Use first 200 chars as key
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                merged.append(doc)

        # Add query results
        for doc in query_results:
            content = doc.get("content", str(doc))
            content_key = content[:200]
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                merged.append(doc)

        return merged

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self._cache is not None:
            self._cache.clear()


# =============================================================================
# INTEGRATION WITH SEMANTIC RERANKER
# =============================================================================

class HyDEWithReranker:
    """
    HyDE retriever integrated with SemanticReranker.

    Combines HyDE's improved recall with semantic reranking for better precision.

    Example:
        >>> from core.rag.hyde import HyDEWithReranker
        >>> from core.rag.reranker import SemanticReranker
        >>>
        >>> reranker = SemanticReranker()
        >>> hyde = HyDEWithReranker(
        ...     llm=my_llm,
        ...     embedder=my_embedder,
        ...     vector_store=my_store,
        ...     reranker=reranker
        ... )
        >>>
        >>> result = await hyde.retrieve("What is RAG?", top_k=10)
    """

    def __init__(
        self,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        vector_store: VectorStoreProvider,
        reranker: Any,  # SemanticReranker
        config: Optional[HyDEConfig] = None,
    ):
        """Initialize HyDE with reranker.

        Args:
            llm: LLM provider
            embedder: Embedding provider
            vector_store: Vector store
            reranker: SemanticReranker instance
            config: Configuration
        """
        self.reranker = reranker
        self.hyde = HyDERetriever(
            llm=llm,
            embedder=embedder,
            vector_store=vector_store,
            config=config
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_k: Optional[int] = None,
        **kwargs
    ) -> HyDEResult:
        """Retrieve and rerank documents.

        Args:
            query: Search query
            top_k: Final number of documents to return
            rerank_top_k: Number to retrieve before reranking (default: top_k * 2)
            **kwargs: Additional arguments

        Returns:
            HyDEResult with reranked documents
        """
        rerank_top_k = rerank_top_k or top_k * 2

        # Get more documents for reranking
        hyde_result = await self.hyde.retrieve(query, top_k=rerank_top_k, **kwargs)

        if not hyde_result.documents:
            return hyde_result

        # Convert to Document objects for reranker
        try:
            from .reranker import Document

            documents = [
                Document(
                    id=str(i),
                    content=doc.get("content", str(doc)),
                    metadata=doc.get("metadata", {})
                )
                for i, doc in enumerate(hyde_result.documents)
            ]

            # Rerank
            reranked = await self.reranker.rerank(query, documents, top_k=top_k)

            # Convert back to dict format
            reranked_docs = [
                {
                    "content": sd.document.content,
                    "metadata": sd.document.metadata,
                    "score": sd.score
                }
                for sd in reranked
            ]

            return HyDEResult(
                documents=reranked_docs,
                hypotheticals=hyde_result.hypotheticals,
                query_embedding=hyde_result.query_embedding,
                retrieval_count=len(reranked_docs)
            )

        except ImportError:
            # Fallback if reranker types not available
            hyde_result.documents = hyde_result.documents[:top_k]
            return hyde_result


# =============================================================================
# MULTI-HYDE: MULTIPLE HYPOTHESIS TYPES
# =============================================================================

class MultiHyDE:
    """
    Multi-HyDE generates multiple types of hypothetical documents
    for improved recall across different query types.

    Example:
        >>> multi_hyde = MultiHyDE(llm=my_llm, embedder=my_embedder, vector_store=my_store)
        >>> result = await multi_hyde.retrieve("What is machine learning?")
    """

    def __init__(
        self,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        vector_store: VectorStoreProvider,
        hypothesis_types: Optional[List[HypotheticalDocumentType]] = None,
        n_per_type: int = 2,
        config: Optional[HyDEConfig] = None,
    ):
        """Initialize Multi-HyDE.

        Args:
            llm: LLM provider
            embedder: Embedding provider
            vector_store: Vector store
            hypothesis_types: Types of hypotheticals to generate
            n_per_type: Number of hypotheticals per type
            config: Base configuration
        """
        self.hypothesis_types = hypothesis_types or [
            HypotheticalDocumentType.ANSWER,
            HypotheticalDocumentType.DOCUMENT,
            HypotheticalDocumentType.EXPLANATION,
        ]

        # Create HyDE retriever for each type
        self.retrievers: Dict[HypotheticalDocumentType, HyDERetriever] = {}

        for hypo_type in self.hypothesis_types:
            type_config = HyDEConfig(
                n_hypothetical=n_per_type,
                hypothesis_type=hypo_type,
                max_tokens=config.max_tokens if config else 256,
                temperature=config.temperature if config else 0.7,
                embedding_strategy="average",
                enable_caching=config.enable_caching if config else True,
                top_k=config.top_k if config else 5,
            )

            self.retrievers[hypo_type] = HyDERetriever(
                llm=llm,
                embedder=embedder,
                vector_store=vector_store,
                config=type_config
            )

        self.config = config or HyDEConfig()

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> HyDEResult:
        """Retrieve using multiple hypothesis types.

        Args:
            query: Search query
            top_k: Number of documents to return
            **kwargs: Additional arguments

        Returns:
            Combined HyDEResult
        """
        top_k = top_k or self.config.top_k

        # Run all retrievers in parallel
        tasks = [
            retriever.retrieve(query, top_k=top_k, **kwargs)
            for retriever in self.retrievers.values()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        all_documents: List[Dict[str, Any]] = []
        all_hypotheticals: List[HypotheticalDocument] = []

        for result in results:
            if isinstance(result, HyDEResult):
                all_documents.extend(result.documents)
                all_hypotheticals.extend(result.hypotheticals)
            elif isinstance(result, Exception):
                logger.warning(f"Multi-HyDE retrieval failed: {result}")

        # Deduplicate and rank
        deduplicated = self._deduplicate_by_content(all_documents)

        # Take top_k
        final_docs = deduplicated[:top_k]

        return HyDEResult(
            documents=final_docs,
            hypotheticals=all_hypotheticals,
            retrieval_count=len(final_docs)
        )

    def _deduplicate_by_content(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate documents by content similarity."""
        seen: set = set()
        unique: List[Dict[str, Any]] = []

        for doc in documents:
            content = doc.get("content", str(doc))
            content_key = content[:200]

            if content_key not in seen:
                seen.add(content_key)
                unique.append(doc)

        return unique


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    "HyDERetriever",
    "HyDEWithReranker",
    "MultiHyDE",
    # Configuration
    "HyDEConfig",
    # Result types
    "HyDEResult",
    "HypotheticalDocument",
    # Enums
    "HypotheticalDocumentType",
    # Cache
    "EmbeddingCache",
    # Protocols
    "LLMProvider",
    "EmbeddingProvider",
    "VectorStoreProvider",
    # Prompts
    "HyDEPrompts",
]
