"""
Semantic Reranker for Improved RAG Retrieval

This module provides a production-ready semantic reranker that improves RAG
retrieval quality through:
1. Cross-encoder reranking using sentence-transformers
2. Reciprocal Rank Fusion (RRF) for combining multiple retrieval methods
3. MMR-style diversity-aware reranking to reduce redundancy
4. Query-aware caching layer for repeated queries

Architecture:
- Primary: Cross-encoder (ms-marco-MiniLM-L-6-v2) for precise relevance scoring
- Fallback: TF-IDF based scoring when models unavailable
- Fusion: RRF for combining dense/sparse/keyword results
- Diversity: MMR algorithm with configurable lambda

Integration:
- Works with platform.core.memory.backends.sqlite.SQLiteTierBackend
- Compatible with MemoryEntry from core.memory.backends.base
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
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
# TYPE DEFINITIONS
# =============================================================================

@dataclass
class Document:
    """A document for reranking."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Document):
            return False
        return self.id == other.id


@dataclass
class ScoredDocument:
    """A document with an associated relevance score."""
    document: Document
    score: float
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "ScoredDocument") -> bool:
        return self.score < other.score


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    def encode(self, texts: List[str]) -> Any:
        """Encode texts to embeddings. Returns numpy array or list."""
        ...


# =============================================================================
# CACHE IMPLEMENTATION
# =============================================================================

class RerankerCache:
    """LRU cache with TTL for reranking results."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """Initialize cache.

        Args:
            max_size: Maximum number of cached entries
            ttl_seconds: Time-to-live in seconds (default 5 minutes)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[List[ScoredDocument], float]] = {}
        self._access_order: List[str] = []
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str, doc_ids: List[str], top_k: int) -> str:
        """Generate cache key from query and document IDs."""
        doc_hash = hashlib.md5("|".join(sorted(doc_ids)).encode()).hexdigest()[:12]
        query_hash = hashlib.md5(query.encode()).hexdigest()[:12]
        return f"{query_hash}:{doc_hash}:{top_k}"

    def get(
        self,
        query: str,
        doc_ids: List[str],
        top_k: int
    ) -> Optional[List[ScoredDocument]]:
        """Get cached results if available and not expired."""
        key = self._make_key(query, doc_ids, top_k)

        if key not in self._cache:
            self._misses += 1
            return None

        results, timestamp = self._cache[key]
        if time.time() - timestamp > self.ttl_seconds:
            # Expired
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            self._misses += 1
            return None

        # Move to end of access order (LRU)
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        self._hits += 1
        return results

    def put(
        self,
        query: str,
        doc_ids: List[str],
        top_k: int,
        results: List[ScoredDocument]
    ) -> None:
        """Cache reranking results."""
        key = self._make_key(query, doc_ids, top_k)

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            self._cache.pop(oldest_key, None)

        self._cache[key] = (results, time.time())
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "ttl_seconds": self.ttl_seconds,
        }


# =============================================================================
# FALLBACK SCORER (TF-IDF based)
# =============================================================================

class TFIDFScorer:
    """Simple TF-IDF based scorer as fallback when ML models unavailable."""

    def __init__(self):
        self._idf_cache: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase and split on non-alphanumeric."""
        import re
        return re.findall(r'\b\w+\b', text.lower())

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency."""
        tf: Dict[str, float] = defaultdict(float)
        for token in tokens:
            tf[token] += 1.0
        # Normalize by document length
        total = len(tokens)
        if total > 0:
            for token in tf:
                tf[token] /= total
        return dict(tf)

    def _compute_idf(self, documents: List[str]) -> Dict[str, float]:
        """Compute inverse document frequency."""
        import math

        doc_freq: Dict[str, int] = defaultdict(int)
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                doc_freq[token] += 1

        n_docs = len(documents)
        idf: Dict[str, float] = {}
        for token, df in doc_freq.items():
            idf[token] = math.log((n_docs + 1) / (df + 1)) + 1

        return idf

    def score(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """Score documents against query using TF-IDF."""
        if not documents:
            return []

        # Compute IDF over document corpus
        doc_texts = [d.content for d in documents]
        idf = self._compute_idf(doc_texts + [query])

        query_tokens = self._tokenize(query)
        query_tf = self._compute_tf(query_tokens)

        results: List[Tuple[Document, float]] = []

        for doc in documents:
            doc_tokens = self._tokenize(doc.content)
            doc_tf = self._compute_tf(doc_tokens)

            # Compute TF-IDF similarity
            score = 0.0
            for token in query_tokens:
                if token in doc_tf:
                    score += query_tf[token] * doc_tf[token] * idf.get(token, 1.0) ** 2

            # Normalize by query length
            if query_tokens:
                score /= len(query_tokens)

            results.append((doc, score))

        return results


# =============================================================================
# CROSS-ENCODER RERANKER
# =============================================================================

class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers."""

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """Initialize cross-encoder.

        Args:
            model_name: Model to use (defaults to ms-marco-MiniLM-L-6-v2)
            device: Device to run on ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name or self.MODEL_NAME
        self.device = device
        self._model = None
        self._available = False
        self._load_attempted = False

    def _load_model(self) -> bool:
        """Lazy load the cross-encoder model."""
        if self._load_attempted:
            return self._available

        self._load_attempted = True

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                max_length=512,
                device=self.device,
            )
            self._available = True
            logger.info(f"Loaded cross-encoder model: {self.model_name}")
            return True

        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            return False
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder model: {e}")
            return False

    @property
    def is_available(self) -> bool:
        """Check if cross-encoder is available."""
        return self._load_model()

    def score(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """Score documents against query using cross-encoder.

        Args:
            query: Search query
            documents: Documents to score

        Returns:
            List of (document, score) tuples
        """
        if not self._load_model() or not documents:
            return []

        # Create query-document pairs
        pairs = [[query, doc.content] for doc in documents]

        # Get scores from cross-encoder
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Normalize scores to 0-1 range using sigmoid
        def sigmoid(x: float) -> float:
            import math
            return 1 / (1 + math.exp(-x))

        results = []
        for doc, score in zip(documents, scores):
            normalized_score = sigmoid(float(score))
            results.append((doc, normalized_score))

        return results

    def score_batch(
        self,
        query: str,
        documents: List[Document],
        batch_size: int = 32
    ) -> List[Tuple[Document, float]]:
        """Score documents in batches for memory efficiency.

        Args:
            query: Search query
            documents: Documents to score
            batch_size: Batch size for processing

        Returns:
            List of (document, score) tuples
        """
        if not self._load_model() or not documents:
            return []

        all_results = []

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_results = self.score(query, batch_docs)
            all_results.extend(batch_results)

        return all_results


# =============================================================================
# SEMANTIC RERANKER (MAIN CLASS)
# =============================================================================

class SemanticReranker:
    """
    Production-ready semantic reranker for improved RAG retrieval.

    Features:
    - Cross-encoder reranking using sentence-transformers (ms-marco-MiniLM-L-6-v2)
    - Reciprocal Rank Fusion (RRF) for combining multiple retrieval methods
    - MMR-style diversity-aware reranking to reduce redundancy
    - Query-aware caching layer for repeated queries
    - Graceful fallback to TF-IDF when ML models unavailable

    Usage:
        reranker = SemanticReranker()

        # Basic reranking
        results = await reranker.rerank(query="What is X?", documents=docs, top_k=10)

        # RRF fusion from multiple sources
        fused = await reranker.rrf_fusion([dense_results, sparse_results, bm25_results])

        # Diversity reranking
        diverse = await reranker.diversity_rerank(results, lambda_diversity=0.3)

    Integration with SQLite backend:
        from core.memory.backends.sqlite import SQLiteTierBackend

        backend = SQLiteTierBackend()
        memory_results = await backend.search("query", limit=50)

        # Convert to Documents
        docs = [Document(id=m.id, content=m.content, metadata=m.metadata)
                for m in memory_results]

        # Rerank
        reranked = await reranker.rerank("query", docs, top_k=10)
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        cache_size: int = 1000,
        cache_ttl: int = 300,
        enable_cache: bool = True,
    ):
        """Initialize the semantic reranker.

        Args:
            model: Cross-encoder model name
            device: Device for inference ('cpu', 'cuda', etc.)
            cache_size: Maximum cache entries
            cache_ttl: Cache TTL in seconds
            enable_cache: Whether to enable caching
        """
        self.model_name = model
        self.device = device

        # Initialize cross-encoder
        self._cross_encoder = CrossEncoderReranker(model_name=model, device=device)

        # Fallback scorer
        self._tfidf_scorer = TFIDFScorer()

        # Cache
        self._cache_enabled = enable_cache
        self._cache = RerankerCache(max_size=cache_size, ttl_seconds=cache_ttl)

        # Embedding provider for diversity (lazy loaded)
        self._embedding_model = None
        self._embedding_available = False
        self._embedding_load_attempted = False

    def _load_embedding_model(self) -> bool:
        """Lazy load embedding model for diversity computation."""
        if self._embedding_load_attempted:
            return self._embedding_available

        self._embedding_load_attempted = True

        try:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                device=self.device,
            )
            self._embedding_available = True
            logger.info("Loaded embedding model for diversity computation")
            return True

        except ImportError:
            logger.debug("sentence-transformers not available for embeddings")
            return False
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            return False

    @property
    def is_available(self) -> bool:
        """Check if the ML-based reranker is available."""
        return self._cross_encoder.is_available

    async def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 10,
        use_cache: bool = True,
    ) -> List[ScoredDocument]:
        """Rerank documents using cross-encoder.

        Uses cross-encoder for precise relevance scoring. Falls back to
        TF-IDF scoring if the model is unavailable.

        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of top documents to return
            use_cache: Whether to use caching

        Returns:
            List of ScoredDocument sorted by relevance (highest first)
        """
        if not documents:
            return []

        # Check cache
        if use_cache and self._cache_enabled:
            doc_ids = [d.id for d in documents]
            cached = self._cache.get(query, doc_ids, top_k)
            if cached is not None:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached

        # Score documents
        if self._cross_encoder.is_available:
            scored_pairs = self._cross_encoder.score_batch(query, documents)
        else:
            logger.info("Cross-encoder unavailable, using TF-IDF fallback")
            scored_pairs = self._tfidf_scorer.score(query, documents)

        # Sort by score descending
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        # Convert to ScoredDocument
        results = []
        for rank, (doc, score) in enumerate(scored_pairs[:top_k], start=1):
            results.append(ScoredDocument(
                document=doc,
                score=score,
                rank=rank,
                metadata={"reranker": "cross-encoder" if self._cross_encoder.is_available else "tfidf"},
            ))

        # Cache results
        if use_cache and self._cache_enabled:
            doc_ids = [d.id for d in documents]
            self._cache.put(query, doc_ids, top_k, results)

        return results

    async def rrf_fusion(
        self,
        result_lists: List[List[Document]],
        k: int = 60,
        weights: Optional[List[float]] = None,
    ) -> List[ScoredDocument]:
        """Combine multiple ranked lists using Reciprocal Rank Fusion.

        RRF formula: score(d) = sum_r (weight_r / (k + rank_r(d)))

        Where:
        - r is each result list
        - k is a constant (typically 60) that dampens the impact of high ranks
        - rank_r(d) is the rank of document d in list r (1-indexed)
        - weight_r is the weight for list r (default 1.0)

        Args:
            result_lists: Multiple ranked lists of documents
            k: RRF constant (higher = more smoothing between ranks)
            weights: Optional weights for each result list

        Returns:
            Fused list of ScoredDocument sorted by RRF score
        """
        if not result_lists:
            return []

        # Default to equal weights
        if weights is None:
            weights = [1.0] * len(result_lists)
        elif len(weights) != len(result_lists):
            raise ValueError("Number of weights must match number of result lists")

        # Calculate RRF scores
        doc_scores: Dict[str, Tuple[Document, float]] = {}

        for list_idx, doc_list in enumerate(result_lists):
            weight = weights[list_idx]

            for rank, doc in enumerate(doc_list, start=1):
                rrf_score = weight / (k + rank)

                if doc.id in doc_scores:
                    existing_doc, existing_score = doc_scores[doc.id]
                    doc_scores[doc.id] = (existing_doc, existing_score + rrf_score)
                else:
                    doc_scores[doc.id] = (doc, rrf_score)

        # Sort by combined score
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )

        # Convert to ScoredDocument
        results = []
        for rank, (doc, score) in enumerate(sorted_docs, start=1):
            results.append(ScoredDocument(
                document=doc,
                score=score,
                rank=rank,
                metadata={
                    "fusion_method": "rrf",
                    "k": k,
                    "num_lists": len(result_lists),
                },
            ))

        return results

    async def diversity_rerank(
        self,
        documents: List[ScoredDocument],
        lambda_diversity: float = 0.3,
        top_k: Optional[int] = None,
    ) -> List[ScoredDocument]:
        """MMR-style diversity reranking to reduce redundancy.

        Maximal Marginal Relevance (MMR) balances relevance and diversity:
        MMR = lambda * relevance(d) - (1 - lambda) * max_selected(similarity(d, s))

        Where:
        - lambda (lambda_diversity) controls relevance vs diversity trade-off
        - Higher lambda = more relevance, less diversity
        - Lower lambda = more diversity, less relevance

        Args:
            documents: Documents with relevance scores
            lambda_diversity: Balance factor (0=max diversity, 1=max relevance)
            top_k: Number of documents to return (default: all)

        Returns:
            Diversity-reranked list of ScoredDocument
        """
        if not documents:
            return []

        if top_k is None:
            top_k = len(documents)

        # Get embeddings for diversity computation
        embeddings = await self._get_embeddings([d.document for d in documents])

        if embeddings is None:
            # Fallback: use Jaccard similarity on tokens
            logger.info("Using token-based diversity (embeddings unavailable)")
            return await self._diversity_rerank_jaccard(documents, lambda_diversity, top_k)

        # MMR selection
        selected: List[ScoredDocument] = []
        remaining = list(documents)

        # Normalize relevance scores to 0-1
        if remaining:
            max_score = max(d.score for d in remaining)
            min_score = min(d.score for d in remaining)
            score_range = max_score - min_score if max_score > min_score else 1.0

        while len(selected) < top_k and remaining:
            best_idx = -1
            best_mmr = float("-inf")

            for idx, doc in enumerate(remaining):
                # Normalized relevance
                relevance = (doc.score - min_score) / score_range if score_range > 0 else doc.score

                # Max similarity to selected documents
                max_sim = 0.0
                if selected:
                    doc_idx = documents.index(doc)
                    for sel in selected:
                        sel_idx = documents.index(sel)
                        sim = self._cosine_similarity(
                            embeddings[doc_idx],
                            embeddings[sel_idx]
                        )
                        max_sim = max(max_sim, sim)

                # MMR score
                mmr = lambda_diversity * relevance - (1 - lambda_diversity) * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            if best_idx >= 0:
                selected_doc = remaining.pop(best_idx)
                # Update rank
                new_doc = ScoredDocument(
                    document=selected_doc.document,
                    score=selected_doc.score,
                    rank=len(selected) + 1,
                    metadata={
                        **selected_doc.metadata,
                        "mmr_rank": len(selected) + 1,
                        "lambda_diversity": lambda_diversity,
                    },
                )
                selected.append(new_doc)
            else:
                break

        return selected

    async def _diversity_rerank_jaccard(
        self,
        documents: List[ScoredDocument],
        lambda_diversity: float,
        top_k: int,
    ) -> List[ScoredDocument]:
        """Fallback diversity reranking using Jaccard token similarity."""
        import re

        def tokenize(text: str) -> set:
            return set(re.findall(r'\b\w+\b', text.lower()))

        def jaccard_similarity(s1: set, s2: set) -> float:
            if not s1 or not s2:
                return 0.0
            intersection = len(s1 & s2)
            union = len(s1 | s2)
            return intersection / union if union > 0 else 0.0

        # Pre-compute token sets
        token_sets = [tokenize(d.document.content) for d in documents]

        selected: List[ScoredDocument] = []
        selected_indices: List[int] = []
        remaining = list(range(len(documents)))

        # Normalize scores
        if documents:
            max_score = max(d.score for d in documents)
            min_score = min(d.score for d in documents)
            score_range = max_score - min_score if max_score > min_score else 1.0

        while len(selected) < top_k and remaining:
            best_idx = -1
            best_mmr = float("-inf")

            for idx in remaining:
                doc = documents[idx]
                relevance = (doc.score - min_score) / score_range if score_range > 0 else doc.score

                # Max Jaccard similarity to selected
                max_sim = 0.0
                if selected_indices:
                    for sel_idx in selected_indices:
                        sim = jaccard_similarity(token_sets[idx], token_sets[sel_idx])
                        max_sim = max(max_sim, sim)

                mmr = lambda_diversity * relevance - (1 - lambda_diversity) * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            if best_idx >= 0:
                remaining.remove(best_idx)
                selected_indices.append(best_idx)
                original_doc = documents[best_idx]
                new_doc = ScoredDocument(
                    document=original_doc.document,
                    score=original_doc.score,
                    rank=len(selected) + 1,
                    metadata={
                        **original_doc.metadata,
                        "mmr_rank": len(selected) + 1,
                        "diversity_method": "jaccard",
                    },
                )
                selected.append(new_doc)
            else:
                break

        return selected

    async def _get_embeddings(
        self,
        documents: List[Document]
    ) -> Optional[Any]:
        """Get embeddings for documents.

        Returns numpy array if available, or list of lists as fallback.
        """
        # First check if documents already have embeddings
        all_have_embeddings = all(d.embedding is not None for d in documents)
        if all_have_embeddings:
            embeddings = [d.embedding for d in documents]
            if NUMPY_AVAILABLE:
                return np.array(embeddings)
            return embeddings

        # Try to load embedding model
        if not self._load_embedding_model():
            return None

        # Encode documents
        texts = [d.content for d in documents]
        embeddings = self._embedding_model.encode(texts, show_progress_bar=False)
        return embeddings

    def _cosine_similarity(self, a: Any, b: Any) -> float:
        """Compute cosine similarity between two vectors.

        Works with numpy arrays or plain Python lists.
        """
        if NUMPY_AVAILABLE and hasattr(a, 'dot'):
            # Use numpy for efficiency
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
        else:
            # Pure Python fallback
            if len(a) != len(b):
                return 0.0
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    async def rerank_with_fusion_and_diversity(
        self,
        query: str,
        result_lists: List[List[Document]],
        top_k: int = 10,
        rrf_k: int = 60,
        lambda_diversity: float = 0.3,
        rrf_weights: Optional[List[float]] = None,
    ) -> List[ScoredDocument]:
        """Full pipeline: RRF fusion -> Cross-encoder rerank -> Diversity.

        This combines all three reranking stages:
        1. RRF fusion of multiple result lists
        2. Cross-encoder reranking for precise relevance
        3. MMR diversity to reduce redundancy

        Args:
            query: Search query
            result_lists: Multiple ranked lists from different retrievers
            top_k: Final number of results
            rrf_k: RRF constant
            lambda_diversity: MMR diversity factor
            rrf_weights: Optional weights for RRF

        Returns:
            Final reranked list with diversity
        """
        # Step 1: RRF fusion
        fused = await self.rrf_fusion(result_lists, k=rrf_k, weights=rrf_weights)

        if not fused:
            return []

        # Step 2: Cross-encoder reranking on top candidates
        # Take 2x top_k for diversity selection
        rerank_count = min(len(fused), top_k * 2)
        docs_to_rerank = [sd.document for sd in fused[:rerank_count]]

        reranked = await self.rerank(query, docs_to_rerank, top_k=rerank_count)

        # Step 3: Diversity reranking
        diverse = await self.diversity_rerank(reranked, lambda_diversity, top_k)

        return diverse

    def clear_cache(self) -> None:
        """Clear the reranking cache."""
        self._cache.clear()

    @property
    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.stats

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get reranker diagnostics."""
        return {
            "cross_encoder_available": self._cross_encoder.is_available,
            "cross_encoder_model": self.model_name,
            "embedding_model_available": self._embedding_available,
            "cache_enabled": self._cache_enabled,
            "cache_stats": self.cache_stats,
            "device": self.device,
        }


# =============================================================================
# INTEGRATION WITH MEMORY BACKEND
# =============================================================================

class MemoryReranker:
    """
    Reranker integration with platform.core.memory.backends.sqlite.

    Provides convenient methods to rerank MemoryEntry results from
    the SQLite backend or other memory backends.

    Usage:
        from core.memory.backends.sqlite import SQLiteTierBackend
        from core.rag.reranker import MemoryReranker

        backend = SQLiteTierBackend()
        reranker = MemoryReranker()

        # Search and rerank
        results = await reranker.search_and_rerank(
            backend=backend,
            query="authentication patterns",
            initial_limit=50,
            final_top_k=10,
        )
    """

    def __init__(
        self,
        semantic_reranker: Optional[SemanticReranker] = None,
        **reranker_kwargs
    ):
        """Initialize memory reranker.

        Args:
            semantic_reranker: Optional pre-configured SemanticReranker
            **reranker_kwargs: Arguments for SemanticReranker if not provided
        """
        self.reranker = semantic_reranker or SemanticReranker(**reranker_kwargs)

    def _memory_to_document(self, entry: Any) -> Document:
        """Convert a MemoryEntry to Document."""
        # Handle both dict-like and object-like entries
        if hasattr(entry, "id"):
            return Document(
                id=entry.id,
                content=entry.content,
                metadata=entry.metadata if hasattr(entry, "metadata") else {},
                embedding=entry.embedding if hasattr(entry, "embedding") else None,
            )
        elif isinstance(entry, dict):
            return Document(
                id=entry.get("id", ""),
                content=entry.get("content", ""),
                metadata=entry.get("metadata", {}),
                embedding=entry.get("embedding"),
            )
        else:
            raise ValueError(f"Cannot convert {type(entry)} to Document")

    async def rerank_memory_results(
        self,
        query: str,
        entries: List[Any],
        top_k: int = 10,
    ) -> List[ScoredDocument]:
        """Rerank MemoryEntry results.

        Args:
            query: Search query
            entries: List of MemoryEntry objects
            top_k: Number of results to return

        Returns:
            Reranked ScoredDocument list
        """
        documents = [self._memory_to_document(e) for e in entries]
        return await self.reranker.rerank(query, documents, top_k)

    async def search_and_rerank(
        self,
        backend: Any,
        query: str,
        initial_limit: int = 50,
        final_top_k: int = 10,
        apply_diversity: bool = True,
        lambda_diversity: float = 0.3,
    ) -> List[ScoredDocument]:
        """Search memory backend and rerank results.

        Args:
            backend: Memory backend with async search method
            query: Search query
            initial_limit: Number of initial results to fetch
            final_top_k: Final number of results after reranking
            apply_diversity: Whether to apply diversity reranking
            lambda_diversity: Diversity factor for MMR

        Returns:
            Reranked and optionally diversity-filtered results
        """
        # Search backend
        entries = await backend.search(query, limit=initial_limit)

        if not entries:
            return []

        # Convert and rerank
        documents = [self._memory_to_document(e) for e in entries]
        reranked = await self.reranker.rerank(query, documents, top_k=final_top_k * 2)

        # Apply diversity if requested
        if apply_diversity and reranked:
            return await self.reranker.diversity_rerank(
                reranked,
                lambda_diversity=lambda_diversity,
                top_k=final_top_k
            )

        return reranked[:final_top_k]

    async def hybrid_search_and_rerank(
        self,
        backends: List[Any],
        query: str,
        initial_limit: int = 30,
        final_top_k: int = 10,
        rrf_k: int = 60,
        lambda_diversity: float = 0.3,
    ) -> List[ScoredDocument]:
        """Search multiple backends and combine with RRF + reranking.

        Args:
            backends: List of memory backends
            query: Search query
            initial_limit: Results per backend
            final_top_k: Final number of results
            rrf_k: RRF constant
            lambda_diversity: Diversity factor

        Returns:
            Combined and reranked results
        """
        # Search all backends in parallel
        search_tasks = [
            backend.search(query, limit=initial_limit)
            for backend in backends
        ]
        results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Convert to document lists
        result_lists: List[List[Document]] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Backend search failed: {result}")
                continue
            if result:
                docs = [self._memory_to_document(e) for e in result]
                result_lists.append(docs)

        if not result_lists:
            return []

        # Full pipeline
        return await self.reranker.rerank_with_fusion_and_diversity(
            query=query,
            result_lists=result_lists,
            top_k=final_top_k,
            rrf_k=rrf_k,
            lambda_diversity=lambda_diversity,
        )


# =============================================================================
# MULTI-QUERY EXPANSION
# =============================================================================

class MultiQueryConfig:
    """Configuration for multi-query expansion.

    Attributes:
        n_queries: Number of query variations to generate (default: 3)
        max_tokens: Maximum tokens for query generation (default: 100)
        temperature: Temperature for generation diversity (default: 0.7)
        include_original: Whether to include original query (default: True)
        fusion_method: Method to combine results - 'rrf' or 'dedupe' (default: 'rrf')
        rrf_k: RRF constant when using RRF fusion (default: 60)
    """

    def __init__(
        self,
        n_queries: int = 3,
        max_tokens: int = 100,
        temperature: float = 0.7,
        include_original: bool = True,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
    ):
        self.n_queries = n_queries
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.include_original = include_original
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k


class LLMProviderProtocol(Protocol):
    """Protocol for LLM providers used in multi-query expansion."""

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        ...


class RetrieverProtocol(Protocol):
    """Protocol for retriever providers."""

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Document]:
        """Retrieve documents for query."""
        ...


class MultiQueryExpander:
    """
    Multi-Query Expansion for improved retrieval recall.

    Generates multiple query variations to capture different aspects of
    the user's intent, then combines results using RRF fusion.

    Benefits:
    - Improved recall by capturing query variations
    - Better handling of ambiguous queries
    - Parallel retrieval for efficiency

    Example:
        >>> from core.rag.reranker import MultiQueryExpander, MultiQueryConfig
        >>>
        >>> config = MultiQueryConfig(n_queries=3, fusion_method='rrf')
        >>> expander = MultiQueryExpander(llm=my_llm, config=config)
        >>>
        >>> queries = await expander.expand_query("What is RAG?")
        >>> # ['What is RAG?', 'How does retrieval augmented generation work?', ...]
    """

    EXPANSION_PROMPT = """Generate {n_queries} different ways to ask this question.
Each variation should capture a different aspect or phrasing of the query.

Original question: {query}

Requirements:
1. Each variation should be semantically related but worded differently
2. Include different keywords that might match relevant documents
3. Keep variations concise and search-friendly

Variations (one per line):"""

    def __init__(
        self,
        llm: LLMProviderProtocol,
        config: Optional[MultiQueryConfig] = None,
    ):
        """Initialize multi-query expander.

        Args:
            llm: LLM provider for query generation
            config: Configuration options
        """
        self.llm = llm
        self.config = config or MultiQueryConfig()

    async def expand_query(self, query: str) -> List[str]:
        """Generate query variations.

        Args:
            query: Original query

        Returns:
            List of query variations including original
        """
        prompt = self.EXPANSION_PROMPT.format(
            n_queries=self.config.n_queries,
            query=query
        )

        try:
            response = await self.llm.generate(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )

            # Parse variations from response
            variations = self._parse_variations(response, query)

            # Include original if configured
            queries = [query] if self.config.include_original else []
            queries.extend(variations)

            return queries

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}, using original query")
            return [query]

    def _parse_variations(self, response: str, original: str) -> List[str]:
        """Parse query variations from LLM response."""
        variations = []

        # Split by newlines and filter
        lines = response.strip().split('\n')

        for line in lines:
            # Clean the line
            cleaned = line.strip()

            # Remove common prefixes
            for prefix in ['- ', '* ', '1. ', '2. ', '3. ', '4. ', '5. ']:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):]

            # Remove quotes
            cleaned = cleaned.strip('"\'')

            # Skip if empty or too similar to original
            if cleaned and cleaned.lower() != original.lower():
                variations.append(cleaned)

        # Limit to configured number
        return variations[:self.config.n_queries]


class MultiQueryRetriever:
    """
    Multi-Query Retriever combining query expansion with retrieval.

    Generates multiple query variations, retrieves for each in parallel,
    and combines results using RRF fusion.

    Example:
        >>> from core.rag.reranker import MultiQueryRetriever, MultiQueryConfig
        >>>
        >>> config = MultiQueryConfig(n_queries=3)
        >>> retriever = MultiQueryRetriever(
        ...     llm=my_llm,
        ...     base_retriever=my_retriever,
        ...     reranker=my_reranker,
        ...     config=config
        ... )
        >>>
        >>> results = await retriever.retrieve("What is RAG?", top_k=10)
    """

    def __init__(
        self,
        llm: LLMProviderProtocol,
        base_retriever: RetrieverProtocol,
        reranker: Optional["SemanticReranker"] = None,
        config: Optional[MultiQueryConfig] = None,
    ):
        """Initialize multi-query retriever.

        Args:
            llm: LLM provider for query expansion
            base_retriever: Base retriever for document retrieval
            reranker: Optional SemanticReranker for result combination
            config: Configuration options
        """
        self.llm = llm
        self.base_retriever = base_retriever
        self.reranker = reranker or SemanticReranker()
        self.config = config or MultiQueryConfig()
        self.expander = MultiQueryExpander(llm=llm, config=config)

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[ScoredDocument]:
        """Retrieve documents using multi-query expansion.

        Args:
            query: Original query
            top_k: Number of documents to return
            **kwargs: Additional arguments passed to retriever

        Returns:
            List of ScoredDocument sorted by combined relevance
        """
        # Step 1: Expand query
        queries = await self.expander.expand_query(query)
        logger.debug(f"Expanded to {len(queries)} queries")

        # Step 2: Retrieve for each query in parallel
        retrieval_tasks = [
            self._retrieve_for_query(q, top_k, **kwargs)
            for q in queries
        ]
        results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

        # Step 3: Collect valid results
        result_lists: List[List[Document]] = []
        for result in results:
            if isinstance(result, list):
                result_lists.append(result)
            elif isinstance(result, Exception):
                logger.warning(f"Retrieval failed: {result}")

        if not result_lists:
            return []

        # Step 4: Combine results
        if self.config.fusion_method == "rrf":
            # Use RRF fusion
            combined = await self.reranker.rrf_fusion(
                result_lists,
                k=self.config.rrf_k
            )
            return combined[:top_k]
        else:
            # Simple deduplication
            return self._deduplicate_results(result_lists, top_k)

    async def _retrieve_for_query(
        self,
        query: str,
        top_k: int,
        **kwargs
    ) -> List[Document]:
        """Retrieve documents for a single query."""
        try:
            docs = await self.base_retriever.retrieve(query, top_k=top_k, **kwargs)

            # Convert to Document if needed
            if docs and not isinstance(docs[0], Document):
                docs = [
                    Document(
                        id=str(i),
                        content=d.get("content", str(d)) if isinstance(d, dict) else str(d),
                        metadata=d.get("metadata", {}) if isinstance(d, dict) else {}
                    )
                    for i, d in enumerate(docs)
                ]

            return docs
        except Exception as e:
            logger.warning(f"Retrieval for query '{query[:50]}...' failed: {e}")
            return []

    def _deduplicate_results(
        self,
        result_lists: List[List[Document]],
        top_k: int
    ) -> List[ScoredDocument]:
        """Deduplicate and rank results from multiple queries."""
        seen_ids: set = set()
        results: List[ScoredDocument] = []
        rank = 1

        # Interleave results from different queries
        max_len = max(len(r) for r in result_lists) if result_lists else 0

        for i in range(max_len):
            for result_list in result_lists:
                if i < len(result_list):
                    doc = result_list[i]
                    if doc.id not in seen_ids:
                        seen_ids.add(doc.id)
                        results.append(ScoredDocument(
                            document=doc,
                            score=1.0 / rank,
                            rank=rank,
                            metadata={"fusion_method": "interleave"}
                        ))
                        rank += 1

                        if len(results) >= top_k:
                            return results

        return results


class MultiQueryReranker:
    """
    Multi-Query Reranker: Expands queries, retrieves, and reranks.

    Full pipeline combining multi-query expansion with semantic reranking
    for optimal retrieval quality.

    Example:
        >>> from core.rag.reranker import MultiQueryReranker
        >>>
        >>> mq_reranker = MultiQueryReranker(
        ...     llm=my_llm,
        ...     base_retriever=my_retriever
        ... )
        >>> results = await mq_reranker.retrieve_and_rerank(
        ...     "What is RAG?",
        ...     top_k=10
        ... )
    """

    def __init__(
        self,
        llm: LLMProviderProtocol,
        base_retriever: RetrieverProtocol,
        reranker: Optional["SemanticReranker"] = None,
        config: Optional[MultiQueryConfig] = None,
    ):
        """Initialize multi-query reranker.

        Args:
            llm: LLM provider
            base_retriever: Base retriever
            reranker: Optional SemanticReranker
            config: Configuration
        """
        self.llm = llm
        self.base_retriever = base_retriever
        self.reranker = reranker or SemanticReranker()
        self.config = config or MultiQueryConfig()
        self.multi_query_retriever = MultiQueryRetriever(
            llm=llm,
            base_retriever=base_retriever,
            reranker=self.reranker,
            config=config
        )

    async def retrieve_and_rerank(
        self,
        query: str,
        top_k: int = 10,
        apply_diversity: bool = True,
        lambda_diversity: float = 0.3,
        **kwargs
    ) -> List[ScoredDocument]:
        """Retrieve with multi-query and rerank results.

        Args:
            query: Original query
            top_k: Number of documents to return
            apply_diversity: Whether to apply diversity reranking
            lambda_diversity: Diversity factor for MMR
            **kwargs: Additional arguments

        Returns:
            List of ScoredDocument
        """
        # Get more candidates for reranking
        candidates = await self.multi_query_retriever.retrieve(
            query,
            top_k=top_k * 2,
            **kwargs
        )

        if not candidates:
            return []

        # Extract documents
        documents = [c.document for c in candidates]

        # Rerank with cross-encoder
        reranked = await self.reranker.rerank(
            query,
            documents,
            top_k=top_k * 2 if apply_diversity else top_k
        )

        # Apply diversity if requested
        if apply_diversity and reranked:
            return await self.reranker.diversity_rerank(
                reranked,
                lambda_diversity=lambda_diversity,
                top_k=top_k
            )

        return reranked[:top_k]


# =============================================================================
# FACTORY AND CONVENIENCE FUNCTIONS
# =============================================================================

def create_reranker(
    model: str = SemanticReranker.DEFAULT_MODEL,
    device: Optional[str] = None,
    enable_cache: bool = True,
    cache_size: int = 1000,
    cache_ttl: int = 300,
) -> SemanticReranker:
    """Create a configured SemanticReranker.

    Args:
        model: Cross-encoder model name
        device: Inference device
        enable_cache: Enable query caching
        cache_size: Cache size
        cache_ttl: Cache TTL in seconds

    Returns:
        Configured SemanticReranker instance
    """
    return SemanticReranker(
        model=model,
        device=device,
        enable_cache=enable_cache,
        cache_size=cache_size,
        cache_ttl=cache_ttl,
    )


def create_memory_reranker(
    semantic_reranker: Optional[SemanticReranker] = None,
    **kwargs
) -> MemoryReranker:
    """Create a MemoryReranker for memory backend integration.

    Args:
        semantic_reranker: Optional pre-configured reranker
        **kwargs: Arguments for SemanticReranker

    Returns:
        Configured MemoryReranker instance
    """
    return MemoryReranker(semantic_reranker=semantic_reranker, **kwargs)


__all__ = [
    # Types
    "Document",
    "ScoredDocument",
    "EmbeddingProvider",
    # Cache
    "RerankerCache",
    # Scorers
    "TFIDFScorer",
    "CrossEncoderReranker",
    # Main reranker
    "SemanticReranker",
    # Memory integration
    "MemoryReranker",
    # Multi-Query Expansion
    "MultiQueryConfig",
    "MultiQueryExpander",
    "MultiQueryRetriever",
    "MultiQueryReranker",
    # Factory functions
    "create_reranker",
    "create_memory_reranker",
]
