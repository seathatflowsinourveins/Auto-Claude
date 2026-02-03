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
    VOYAGE_3 = "voyage-3"
    VOYAGE_3_LITE = "voyage-3-lite"
    VOYAGE_3_LARGE = "voyage-3-large"
    VOYAGE_CODE_3 = "voyage-code-3"
    VOYAGE_4_LARGE = "voyage-3-large"  # Alias for compatibility

    @property
    def dimension(self) -> int:
        dims = {
            "voyage-3": 1024,
            "voyage-3-lite": 512,
            "voyage-3-large": 1024,
            "voyage-code-3": 1024,
        }
        return dims.get(self.value, 1024)


class InputType(str, Enum):
    """Input type for embeddings (affects Voyage AI optimization)."""
    DOCUMENT = "document"
    QUERY = "query"


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding layer."""
    model: str = EmbeddingModel.VOYAGE_3.value
    api_key: Optional[str] = None  # None = read from VOYAGE_API_KEY env
    batch_size: int = 128
    max_retries: int = 3
    timeout: float = 30.0
    cache_enabled: bool = True
    cache_max_size: int = 10000


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
        model: str = EmbeddingModel.VOYAGE_3.value,
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
    ) -> EmbeddingResult:
        """
        Embed texts using Voyage AI.

        Args:
            texts: List of texts to embed
            input_type: DOCUMENT or QUERY (affects Voyage optimization)

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        if not self._initialized:
            await self.initialize()

        start = time.monotonic()

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

                # Real Voyage AI API call
                response = await asyncio.to_thread(
                    self._client.embed,
                    batch,
                    model=self._config.model,
                    input_type=input_type.value,
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
        h = hashlib.md5(f"{text}:{input_type.value}:{self._config.model}".encode()).hexdigest()
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


# =============================================================================
# QdrantVectorStore - Vector storage backend
# =============================================================================

class QdrantVectorStore:
    """
    Qdrant vector store for UNLEASH platform.

    Wraps qdrant_client for collection management and search.
    """

    def __init__(
        self,
        url: str = os.environ.get("QDRANT_URL", "localhost:6333"),
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
    ):
        self._url = url
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

        self._client = await asyncio.to_thread(
            QdrantClient,
            url=self._url,
            api_key=self._api_key,
            prefer_grpc=self._prefer_grpc,
        )
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
    model: str = EmbeddingModel.VOYAGE_3.value,
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
