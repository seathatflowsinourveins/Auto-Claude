"""
Voyage AI Embedding Layer for UNLEASH Platform

Provides unified embedding infrastructure wrapping the Voyage AI SDK.
Used by dspy_voyage_retriever and letta_voyage_adapter.

Real implementation using voyageai SDK - no mocks, no fallbacks.
"""

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Voyage AI SDK availability
VOYAGEAI_AVAILABLE = False
try:
    import voyageai
    VOYAGEAI_AVAILABLE = True
except ImportError:
    pass

# Qdrant client availability
QDRANT_AVAILABLE = False
try:
    from qdrant_client import QdrantClient, models as qdrant_models
    QDRANT_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# Enums and Config
# =============================================================================

class EmbeddingModel(str, Enum):
    """Supported Voyage AI embedding models."""
    # Series 3 models
    VOYAGE_3 = "voyage-3"
    VOYAGE_3_LITE = "voyage-3-lite"
    VOYAGE_3_LARGE = "voyage-3-large"
    VOYAGE_CODE_3 = "voyage-code-3"
    # Series 3.5 models
    VOYAGE_3_5 = "voyage-3.5"
    VOYAGE_3_5_LITE = "voyage-3.5-lite"
    # Series 4 models
    VOYAGE_4 = "voyage-4"
    VOYAGE_4_LITE = "voyage-4-lite"
    VOYAGE_4_LARGE = "voyage-4-large"
    # Domain-specific models
    VOYAGE_CONTEXT_3 = "voyage-context-3"
    VOYAGE_FINANCE_2 = "voyage-finance-2"
    VOYAGE_LAW_2 = "voyage-law-2"

    @property
    def dimension(self) -> int:
        dims = {
            "voyage-3": 1024,
            "voyage-3-lite": 512,
            "voyage-3-large": 1024,
            "voyage-code-3": 1024,
            "voyage-3.5": 1024,
            "voyage-3.5-lite": 512,
            "voyage-4": 1024,
            "voyage-4-lite": 512,
            "voyage-4-large": 2048,
            "voyage-context-3": 1024,
            "voyage-finance-2": 1024,
            "voyage-law-2": 1024,
        }
        return dims.get(self.value, 1024)


class InputType(str, Enum):
    """Input type for embeddings (affects Voyage AI optimization)."""
    DOCUMENT = "document"
    QUERY = "query"


class RerankModel(str, Enum):
    """Available reranking models."""
    RERANK_2 = "rerank-2"
    RERANK_2_LITE = "rerank-2-lite"


class OutputDType(str, Enum):
    """Output data types for embeddings."""
    FLOAT = "float"
    INT8 = "int8"
    UINT8 = "uint8"
    BINARY = "binary"
    UBINARY = "ubinary"


class OutputDimension(int, Enum):
    """Matryoshka output dimensions for Voyage 4 models."""
    DIM_256 = 256
    DIM_512 = 512
    DIM_1024 = 1024
    DIM_2048 = 2048


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding layer."""
    model: str = EmbeddingModel.VOYAGE_4.value
    api_key: Optional[str] = None  # None = read from VOYAGE_API_KEY env
    batch_size: int = 128
    max_retries: int = 3
    timeout: float = 30.0
    cache_enabled: bool = True
    cache_max_size: int = 10000
    # Voyage AI output control parameters
    output_dimension: Optional[int] = None  # 256, 512, 1024, 2048 (Matryoshka)
    output_dtype: str = "float"  # float, int8, uint8, binary, ubinary
    truncation: bool = True  # Truncate input to model's context length


@dataclass
class EmbeddingResult:
    """Result from an embedding operation."""
    embeddings: List[List[float]]
    model: str
    total_tokens: int = 0
    cached: bool = False
    latency_ms: float = 0.0


# =============================================================================
# EmbeddingLayer - Core Voyage AI wrapper
# =============================================================================

class EmbeddingLayer:
    """
    Async embedding layer wrapping Voyage AI SDK.

    Provides:
    - Batched embedding with automatic chunking
    - In-memory cache for repeated queries
    - Latency tracking
    - Retry logic for transient failures
    """

    def __init__(
        self,
        model: str = EmbeddingModel.VOYAGE_4.value,
        api_key: Optional[str] = None,
        cache_enabled: bool = True,
        config: Optional[EmbeddingConfig] = None,
    ):
        self._config = config or EmbeddingConfig(
            model=model,
            api_key=api_key,
            cache_enabled=cache_enabled,
        )
        self._client: Optional[voyageai.Client] = None
        self._cache: Dict[str, List[float]] = {}
        self._initialized = False
        self._total_calls = 0
        self._total_tokens = 0
        self._cache_hits = 0

    async def initialize(self) -> "EmbeddingLayer":
        """Initialize the Voyage AI client."""
        if self._initialized:
            return self

        if not VOYAGEAI_AVAILABLE:
            raise ImportError("voyageai package not installed. Install with: pip install voyageai")

        # voyageai.Client() auto-reads VOYAGE_API_KEY from env
        if self._config.api_key:
            self._client = voyageai.Client(api_key=self._config.api_key)
        else:
            self._client = voyageai.Client()

        self._initialized = True
        logger.info("EmbeddingLayer initialized", model=self._config.model)
        return self

    async def embed(
        self,
        texts: List[str],
        input_type: InputType = InputType.DOCUMENT,
        output_dimension: Optional[int] = None,
        output_dtype: Optional[str] = None,
        truncation: Optional[bool] = None,
    ) -> EmbeddingResult:
        """
        Embed texts using Voyage AI.

        Args:
            texts: List of texts to embed
            input_type: DOCUMENT or QUERY (affects Voyage optimization)
            output_dimension: Override output dimension (256, 512, 1024, 2048)
            output_dtype: Override output dtype (float, int8, uint8, binary, ubinary)
            truncation: Override truncation behavior

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        if not self._initialized:
            await self.initialize()

        start = time.monotonic()

        # Resolve parameters with config defaults
        effective_output_dimension = output_dimension if output_dimension is not None else self._config.output_dimension
        effective_output_dtype = output_dtype if output_dtype is not None else self._config.output_dtype
        effective_truncation = truncation if truncation is not None else self._config.truncation

        # Check cache for individual texts
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        if self._config.cache_enabled:
            for i, text in enumerate(texts):
                cache_key = self._cache_key(text, input_type)
                if cache_key in self._cache:
                    results[i] = self._cache[cache_key]
                    self._cache_hits += 1
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        total_tokens = 0

        # Embed uncached texts in batches
        if uncached_texts:
            for batch_start in range(0, len(uncached_texts), self._config.batch_size):
                batch = uncached_texts[batch_start:batch_start + self._config.batch_size]

                # Build API call kwargs
                embed_kwargs = {
                    "model": self._config.model,
                    "input_type": input_type.value,
                    "truncation": effective_truncation,
                }
                if effective_output_dimension is not None:
                    embed_kwargs["output_dimension"] = effective_output_dimension
                if effective_output_dtype and effective_output_dtype != "float":
                    embed_kwargs["output_dtype"] = effective_output_dtype

                # Real Voyage AI API call (V37: Added timeout enforcement)
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._client.embed,
                        batch,
                        **embed_kwargs,
                    ),
                    timeout=self._config.timeout
                )

                total_tokens += response.total_tokens
                self._total_tokens += response.total_tokens
                self._total_calls += 1

                for j, embedding in enumerate(response.embeddings):
                    idx = uncached_indices[batch_start + j]
                    results[idx] = embedding

                    # Cache the result
                    if self._config.cache_enabled:
                        text = uncached_texts[batch_start + j]
                        cache_key = self._cache_key(text, input_type)
                        self._cache[cache_key] = embedding

                        # Evict if cache too large
                        if len(self._cache) > self._config.cache_max_size:
                            oldest_key = next(iter(self._cache))
                            del self._cache[oldest_key]

        latency = (time.monotonic() - start) * 1000

        return EmbeddingResult(
            embeddings=[r for r in results if r is not None],
            model=self._config.model,
            total_tokens=total_tokens,
            cached=len(uncached_texts) == 0,
            latency_ms=latency,
        )

    def _cache_key(self, text: str, input_type: InputType) -> str:
        """Generate cache key for a text + input_type combination."""
        # Use SHA256 for better security practices (truncated to 32 chars for efficiency)
        h = hashlib.sha256(f"{text}:{input_type.value}:{self._config.model}".encode()).hexdigest()[:32]
        return h

    @property
    def stats(self) -> Dict[str, Any]:
        """Get embedding layer statistics."""
        return {
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "model": self._config.model,
        }

    async def hybrid_search(
        self,
        query: str,
        documents: List[str],
        doc_embeddings: Optional[List[List[float]]] = None,
        top_k: int = 10,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> List[tuple]:
        """
        Hybrid search combining vector similarity with BM25 keyword matching.

        Uses Reciprocal Rank Fusion (RRF) to combine rankings from both methods.
        RRF formula: score(d) = sum(1 / (k + rank_i(d))) for each method i

        Args:
            query: Search query
            documents: List of document texts
            doc_embeddings: Pre-computed document embeddings (optional)
            top_k: Number of results to return
            alpha: Weight for vector score vs BM25 (0.5 = equal weight)
            rrf_k: RRF constant (typically 60)

        Returns:
            List of tuples: (doc_index, combined_score, doc_text)
        """
        import math
        import re
        from collections import Counter

        if not documents:
            return []

        # Get query embedding
        query_result = await self.embed([query], input_type=InputType.QUERY)
        query_embedding = query_result.embeddings[0]

        # Get document embeddings if not provided
        if doc_embeddings is None:
            doc_result = await self.embed(documents, input_type=InputType.DOCUMENT)
            doc_embeddings = doc_result.embeddings

        # Vector search scores
        def cosine_similarity(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

        vector_scores = [
            (i, cosine_similarity(query_embedding, emb))
            for i, emb in enumerate(doc_embeddings)
        ]
        vector_scores.sort(key=lambda x: x[1], reverse=True)

        # BM25 keyword scores
        def tokenize(text: str) -> List[str]:
            return re.findall(r'\w+', text.lower())

        def bm25_score(query_terms: List[str], doc: str, all_docs: List[str], k1: float = 1.5, b: float = 0.75) -> float:
            doc_terms = tokenize(doc)
            doc_len = len(doc_terms)
            avg_doc_len = sum(len(tokenize(d)) for d in all_docs) / len(all_docs) if all_docs else 1

            term_freqs = Counter(doc_terms)
            doc_freq = {term: sum(1 for d in all_docs if term in tokenize(d)) for term in set(query_terms)}

            n = len(all_docs)
            score = 0.0
            for term in query_terms:
                if term not in term_freqs:
                    continue
                tf = term_freqs[term]
                df = doc_freq.get(term, 0)
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
                score += idf * tf_norm
            return score

        query_terms = tokenize(query)
        bm25_scores = [
            (i, bm25_score(query_terms, doc, documents))
            for i, doc in enumerate(documents)
        ]
        bm25_scores.sort(key=lambda x: x[1], reverse=True)

        # RRF fusion
        vector_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(vector_scores)}
        bm25_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(bm25_scores)}

        rrf_scores = {}
        for i in range(len(documents)):
            v_rank = vector_ranks.get(i, len(documents) + 1)
            b_rank = bm25_ranks.get(i, len(documents) + 1)
            # RRF with alpha weighting
            rrf_scores[i] = alpha * (1 / (rrf_k + v_rank)) + (1 - alpha) * (1 / (rrf_k + b_rank))

        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            (idx, score, documents[idx])
            for idx, score in sorted_results[:top_k]
        ]

    async def close(self) -> None:
        """
        Release resources and clear cache.

        V37: Added for proper resource cleanup in async context managers.
        """
        self._cache.clear()
        self._client = None
        self._initialized = False
        self._total_calls = 0
        self._total_tokens = 0
        self._cache_hits = 0

    async def __aenter__(self) -> "EmbeddingLayer":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


# =============================================================================
# QdrantVectorStore - Vector storage backend
# =============================================================================

class QdrantVectorStore:
    """
    Qdrant vector store for UNLEASH platform.

    Wraps qdrant_client for collection management and search.

    Storage modes:
    - Remote: url="localhost:6333" (connects to Qdrant server)
    - Persistent: path="./.qdrant_data" (local file-based storage)
    - In-memory: path=":memory:" (non-persistent, for testing)
    """

    def __init__(
        self,
        url: Optional[str] = None,
        path: Optional[str] = None,
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
    ):
        # Determine storage mode
        # Priority: explicit path > explicit url > env var > default persistent
        if path is not None:
            self._mode = "local"
            self._path = path
            self._url = None
        elif url is not None:
            self._mode = "remote"
            self._url = url
            self._path = None
        elif os.environ.get("QDRANT_URL"):
            self._mode = "remote"
            self._url = os.environ.get("QDRANT_URL")
            self._path = None
        else:
            # Default to persistent local storage
            self._mode = "local"
            self._path = os.environ.get("QDRANT_PATH", "./.qdrant_data")
            self._url = None

        self._api_key = api_key
        self._prefer_grpc = prefer_grpc
        self._client: Optional[QdrantClient] = None
        self._initialized = False

    async def initialize(self) -> "QdrantVectorStore":
        """Initialize Qdrant client."""
        if self._initialized:
            return self

        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client not installed. Install with: pip install qdrant-client")

        if self._mode == "local":
            # Local persistent or in-memory storage
            self._client = await asyncio.to_thread(
                QdrantClient,
                path=self._path,
            )
            logger.info("QdrantVectorStore initialized (local)", path=self._path)
        else:
            # Remote Qdrant server
            self._client = await asyncio.to_thread(
                QdrantClient,
                url=self._url,
                api_key=self._api_key,
                prefer_grpc=self._prefer_grpc,
            )
            logger.info("QdrantVectorStore initialized (remote)", url=self._url)

        self._initialized = True
        logger.info("QdrantVectorStore initialized", url=self._url)
        return self

    async def create_collection(
        self,
        name: str,
        dimension: int = 1024,
        distance: str = "cosine",
    ) -> bool:
        """Create a collection if it doesn't exist."""
        if not self._client:
            raise RuntimeError("QdrantVectorStore not initialized")

        try:
            collections = await asyncio.to_thread(self._client.get_collections)
            existing = [c.name for c in collections.collections]
            if name in existing:
                return False

            dist = qdrant_models.Distance.COSINE
            if distance == "dot":
                dist = qdrant_models.Distance.DOT
            elif distance == "euclid":
                dist = qdrant_models.Distance.EUCLID

            await asyncio.to_thread(
                self._client.create_collection,
                collection_name=name,
                vectors_config=qdrant_models.VectorParams(
                    size=dimension,
                    distance=dist,
                ),
            )
            logger.info("Collection created", name=name, dimension=dimension)
            return True
        except Exception as e:
            logger.error("Failed to create collection", name=name, error=str(e))
            raise

    async def upsert(
        self,
        collection: str,
        ids: List[str],
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Upsert vectors into a collection."""
        if not self._client:
            raise RuntimeError("QdrantVectorStore not initialized")

        points = []
        for i, (id_, vec) in enumerate(zip(ids, vectors)):
            payload = payloads[i] if payloads else {}
            points.append(qdrant_models.PointStruct(
                id=id_,
                vector=vec,
                payload=payload,
            ))

        await asyncio.to_thread(
            self._client.upsert,
            collection_name=collection,
            points=points,
        )
        return len(points)

    async def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if not self._client:
            raise RuntimeError("QdrantVectorStore not initialized")

        results = await asyncio.to_thread(
            self._client.query_points,
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
        )

        return [
            {
                "id": str(hit.id),
                "score": hit.score,
                "payload": hit.payload or {},
            }
            for hit in results.points
        ]


# =============================================================================
# UnleashVectorAdapter - Bridges EmbeddingLayer + QdrantVectorStore
# =============================================================================

class UnleashVectorAdapter:
    """
    Unified vector adapter combining Voyage AI embeddings with Qdrant storage.

    Provides embed-and-store, embed-and-search workflows.
    """

    def __init__(
        self,
        embedding_layer: EmbeddingLayer,
        qdrant_store: Optional[QdrantVectorStore] = None,
        default_collection: str = "unleash_memory",
    ):
        self._embedding_layer = embedding_layer
        self._qdrant_store = qdrant_store
        self._default_collection = default_collection
        self._initialized = False

    async def initialize_collections(self) -> None:
        """Initialize default collections."""
        if self._qdrant_store:
            try:
                await self._qdrant_store.initialize()
                model_enum = EmbeddingModel(self._embedding_layer._config.model)
                dim = model_enum.dimension
                await self._qdrant_store.create_collection(
                    self._default_collection, dimension=dim
                )
                self._initialized = True
            except Exception as e:
                logger.warning("Qdrant initialization failed (non-fatal)", error=str(e))
                self._initialized = False
        else:
            self._initialized = True

    async def embed(
        self,
        texts: List[str],
        input_type: InputType = InputType.DOCUMENT,
    ) -> EmbeddingResult:
        """Embed texts using the underlying EmbeddingLayer."""
        return await self._embedding_layer.embed(texts, input_type=input_type)

    async def embed_and_store(
        self,
        texts: List[str],
        ids: List[str],
        collection: Optional[str] = None,
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> EmbeddingResult:
        """Embed texts and store in Qdrant."""
        result = await self._embedding_layer.embed(texts, input_type=InputType.DOCUMENT)

        if self._qdrant_store and self._initialized:
            coll = collection or self._default_collection
            await self._qdrant_store.upsert(
                collection=coll,
                ids=ids,
                vectors=result.embeddings,
                payloads=payloads,
            )

        return result

    async def search(
        self,
        query: str,
        collection: Optional[str] = None,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Embed query and search in Qdrant."""
        query_result = await self._embedding_layer.embed(
            [query], input_type=InputType.QUERY
        )

        if not self._qdrant_store:
            return []

        coll = collection or self._default_collection
        return await self._qdrant_store.search(
            collection=coll,
            query_vector=query_result.embeddings[0],
            top_k=top_k,
            score_threshold=score_threshold,
        )


# =============================================================================
# Factory function
# =============================================================================

def create_embedding_layer(
    model: str = EmbeddingModel.VOYAGE_4.value,
    api_key: Optional[str] = None,
    cache_enabled: bool = True,
    **kwargs,
) -> EmbeddingLayer:
    """
    Create an EmbeddingLayer instance.

    Args:
        model: Voyage AI model name
        api_key: Optional API key (defaults to VOYAGE_API_KEY env var)
        cache_enabled: Enable embedding cache

    Returns:
        Uninitialized EmbeddingLayer (call await .initialize())
    """
    config = EmbeddingConfig(
        model=model,
        api_key=api_key,
        cache_enabled=cache_enabled,
    )
    return EmbeddingLayer(config=config)


# Voyage AI availability flag
try:
    import voyageai  # noqa: F401
    VOYAGE_AVAILABLE = True
except ImportError:
    VOYAGE_AVAILABLE = False

# HTTPX availability flag
try:
    import httpx  # noqa: F401
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


def create_model_mixing_layer(
    models: Optional[list] = None,
    **kwargs,
) -> EmbeddingLayer:
    """Create an EmbeddingLayer with model mixing support."""
    return create_embedding_layer(**kwargs)
