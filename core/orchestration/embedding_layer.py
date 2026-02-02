#!/usr/bin/env python3
"""
Embedding Layer - Voyage AI Integration for Semantic Search (V39.0)

This module provides unified embedding capabilities using Voyage AI models:
- voyage-4-large: State-of-the-art MoE model (1024d default, up to 2048d)
- voyage-4: Standard model approaching voyage-3-large quality (1024d)
- voyage-4-lite: Fast, efficient embeddings (1024d)
- voyage-code-3: Code-specialized embeddings (2048d, 30+ languages)
- voyage-finance-2: Financial document specialist (1024d)
- voyage-law-2: Legal text specialist (1024d)
- voyage-context-3: Contextualized embeddings (1024d)

Reranking Models (V37.0):
- rerank-2.5: Best quality, instruction-following, multilingual (32K tokens)
- rerank-2.5-lite: Balanced latency/quality (32K tokens)

Key Features (Voyage 4 - Released Jan 2026):
- Shared embedding space: Mix models for query/document (e.g., lite for queries, large for docs)
- MoE architecture: 40% lower serving costs than dense models
- Configurable dimensions: 2048, 1024 (default), 512, 256
- Token limits: 320K (voyage-4/large), 1M (voyage-4-lite)

V38.0 - HTTP API Features (SDK doesn't support yet):
- output_dimension: Reduce dimensions (2048, 1024, 512, 256) for storage savings
- output_dtype: Quantization (float, int8, uint8, binary, ubinary)
- Direct HTTP API fallback via httpx for advanced features

V39.0 - Advanced Features:
- Multimodal embeddings: voyage-multimodal-3.5 for text + image
- Intelligent chunking: Auto-split long documents with overlap
- Batch optimization: Token-aware batching for large-scale indexing
- Qdrant integration: Vector database storage and search
- Project adapters: Witness (creative), Trading (financial), Unleash (general)

Usage:
    from core.orchestration import create_embedding_layer, embed_texts

    # Quick usage with voyage-4-large (best accuracy)
    embeddings = await embed_texts(["Hello world", "Test text"])

    # With configuration
    layer = create_embedding_layer(
        model="voyage-4-large",
        api_key="your-key",  # or use VOYAGE_API_KEY env var
    )
    await layer.initialize()

    # Embed documents (for storage)
    doc_embeddings = await layer.embed_documents(["doc1", "doc2"])

    # Embed queries (for retrieval)
    query_embedding = await layer.embed_query("search term")

    # V38.0: Advanced HTTP API features
    result = await layer.embed_with_options(
        texts=["text"],
        output_dimension=256,  # Reduced dimension
        output_dtype="int8",   # Quantized output
    )

NO STUBS - EXPLICIT FAILURES ONLY:
- Raises SDKNotAvailableError if voyageai not installed
- Raises SDKConfigurationError if API key not configured
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from datetime import datetime
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence, NamedTuple, Callable, AsyncGenerator, Awaitable

import structlog

# HTTP client for advanced API features (V38.0)
HTTPX_AVAILABLE = False
_httpx_module: Any = None
try:
    import httpx as _httpx_module  # type: ignore[import-untyped]
    HTTPX_AVAILABLE = True
except ImportError:
    pass

# Opik tracing for AI observability (V39.11)
OPIK_AVAILABLE = False
_opik_module: Any = None
_opik_context_module: Any = None
try:
    import opik as _opik_module  # type: ignore[import-untyped]
    from opik import opik_context as _opik_context_module  # type: ignore[import-untyped]
    OPIK_AVAILABLE = True
except ImportError:
    pass

# Import exceptions - try observability first, fallback to inline definitions
try:
    from core.observability import (
        SDKNotAvailableError,
        SDKConfigurationError,
    )
except ImportError:
    # Inline fallback if observability not available
    class SDKNotAvailableError(Exception):
        """Raised when an SDK is not installed."""
        def __init__(self, sdk_name: str, install_cmd: str, docs_url: str = ""):
            self.sdk_name = sdk_name
            self.install_cmd = install_cmd
            self.docs_url = docs_url
            super().__init__(f"{sdk_name} not available. Install: {install_cmd}")

    class SDKConfigurationError(Exception):
        """Raised when SDK configuration is missing."""
        def __init__(self, sdk_name: str, missing_config: list[str], example: str = ""):
            self.sdk_name = sdk_name
            self.missing_config = missing_config
            self.example = example
            super().__init__(f"{sdk_name} missing config: {missing_config}")

logger = structlog.get_logger(__name__)


# =============================================================================
# SDK Availability Detection
# =============================================================================

VOYAGE_AVAILABLE = False
_voyageai_module: Any = None

try:
    import voyageai as _voyageai_module  # type: ignore[import-untyped]
    VOYAGE_AVAILABLE = True
    logger.debug("voyage_ai_available", version=getattr(_voyageai_module, "__version__", "unknown"))
except ImportError:
    logger.debug("voyage_ai_not_available")


# Default API key (user-provided for UNLEASH project)
# Can be overridden via VOYAGE_API_KEY environment variable
_DEFAULT_VOYAGE_API_KEY = "pa-KCpYL_zzmvoPK1dM6tN5kdCD8e6qnAndC-dSTlCuzK4"


# =============================================================================
# Configuration Types
# =============================================================================

class EmbeddingModel(str, Enum):
    """Supported Voyage AI embedding models (Official Docs - Jan 2026).

    Voyage 4 Series - RECOMMENDED:
    - Shared embedding space allows mixing models
    - MoE architecture for 40% lower costs
    - Dimensions: 2048, 1024 (default), 512, 256
    - Token limits: 32K per text, 120K-1M per batch

    Specialized Models:
    - Code, Finance, Law specialists for domain-specific retrieval
    - Contextualized embeddings for document sections

    Reranking (use RerankModel enum):
    - rerank-2.5: Best quality, 32K tokens
    - rerank-2.5-lite: Fast, 32K tokens
    """
    # Voyage 4 Series - AVAILABLE (200M tokens each)
    VOYAGE_4_LARGE = "voyage-4-large"       # Best accuracy, MoE, 32K/text, 120K/batch
    VOYAGE_4 = "voyage-4"                   # Balanced: 32K/text, 320K/batch
    VOYAGE_4_LITE = "voyage-4-lite"         # Fast: 32K/text, 1M/batch

    # Specialized Models - AVAILABLE
    VOYAGE_CODE_3 = "voyage-code-3"         # Code: 2048d, 30+ languages, 32K/text
    VOYAGE_FINANCE_2 = "voyage-finance-2"   # Finance: 1024d, 32K/text
    VOYAGE_LAW_2 = "voyage-law-2"           # Legal: 1024d, 16K/text
    VOYAGE_CONTEXT_3 = "voyage-context-3"   # Contextualized: 1024d

    # Voyage 3.5 Series - AVAILABLE
    VOYAGE_3_5 = "voyage-3.5"               # General: 1024d
    VOYAGE_3_5_LITE = "voyage-3.5-lite"     # Fast: 1024d

    # Multimodal - AVAILABLE
    VOYAGE_MULTIMODAL_3_5 = "voyage-multimodal-3.5"  # Text + Image
    VOYAGE_MULTIMODAL_3 = "voyage-multimodal-3"      # Text + Image (legacy)

    # Voyage 3 Series - DEPRECATED (0 tokens on free tier)
    VOYAGE_3_LARGE = "voyage-3-large"       # Legacy: 1024d
    VOYAGE_3 = "voyage-3"                   # Legacy: 1024d
    VOYAGE_3_LITE = "voyage-3-lite"         # Legacy: 512d

    @property
    def dimension(self) -> int:
        """Return default embedding dimension for this model."""
        dims = {
            # Voyage 4 - all support 2048/1024/512/256, default 1024
            "voyage-4-large": 1024,
            "voyage-4": 1024,
            "voyage-4-lite": 1024,
            # Specialized
            "voyage-code-3": 1024,  # Default is 1024, supports 256-2048
            "voyage-finance-2": 1024,
            "voyage-law-2": 1024,
            "voyage-context-3": 1024,
            # Voyage 3.5
            "voyage-3.5": 1024,
            "voyage-3.5-lite": 1024,
            # Multimodal
            "voyage-multimodal-3.5": 1024,
            "voyage-multimodal-3": 1024,
            # Legacy Voyage 3
            "voyage-3-large": 1024,
            "voyage-3": 1024,
            "voyage-3-lite": 512,
        }
        return dims.get(self.value, 1024)

    @property
    def token_limit(self) -> int:
        """Max tokens per text for this model."""
        limits = {
            "voyage-4-large": 32000,
            "voyage-4": 32000,
            "voyage-4-lite": 32000,
            "voyage-code-3": 32000,
            "voyage-finance-2": 32000,
            "voyage-law-2": 16000,
            "voyage-context-3": 32000,
            "voyage-3.5": 32000,
            "voyage-3.5-lite": 32000,
            "voyage-multimodal-3.5": 32000,
            "voyage-multimodal-3": 32000,
        }
        return limits.get(self.value, 16000)

    @property
    def batch_token_limit(self) -> int:
        """Max total tokens per batch for this model."""
        limits = {
            "voyage-4-large": 120000,
            "voyage-4": 320000,
            "voyage-4-lite": 1000000,
            "voyage-code-3": 120000,
            "voyage-finance-2": 120000,
            "voyage-law-2": 120000,
        }
        return limits.get(self.value, 320000)

    @property
    def is_available_free_tier(self) -> bool:
        """Check if model is available on free tier (has tokens)."""
        return self.value in {
            "voyage-4-large", "voyage-4", "voyage-4-lite",
            "voyage-code-3", "voyage-finance-2", "voyage-law-2",
            "voyage-context-3", "voyage-3.5", "voyage-3.5-lite",
            "voyage-multimodal-3.5", "voyage-multimodal-3",
        }


class RerankModel(str, Enum):
    """Voyage AI reranking models (Official Docs - Jan 2026).

    Use rerankers to refine embedding-based search results.
    Cross-encoders that jointly process query + document pairs.
    """
    # Current Generation - RECOMMENDED
    RERANK_2_5 = "rerank-2.5"           # Best quality, 32K tokens, multilingual
    RERANK_2_5_LITE = "rerank-2.5-lite" # Fast, 32K tokens

    # Previous Generation
    RERANK_2 = "rerank-2"               # Quality focused, 16K tokens
    RERANK_2_LITE = "rerank-2-lite"     # Balanced, 8K tokens

    # Legacy
    RERANK_1 = "rerank-1"               # Multilingual, 8K tokens
    RERANK_LITE_1 = "rerank-lite-1"     # Speed optimized, 4K tokens

    @property
    def token_limit(self) -> int:
        """Max query tokens for this model."""
        limits = {
            "rerank-2.5": 32000,
            "rerank-2.5-lite": 32000,
            "rerank-2": 16000,
            "rerank-2-lite": 8000,
            "rerank-1": 8000,
            "rerank-lite-1": 4000,
        }
        return limits.get(self.value, 8000)


class OutputDType(str, Enum):
    """Output data types for embeddings (V38.0 - HTTP API only).

    Quantization reduces storage requirements:
    - float: Full precision, default
    - int8: 4x storage reduction, ~1% accuracy loss
    - uint8: 4x storage reduction, unsigned
    - binary: 32x storage reduction, for Hamming distance
    - ubinary: 32x storage reduction, unsigned binary
    """
    FLOAT = "float"      # Full precision (default)
    INT8 = "int8"        # 4x storage reduction
    UINT8 = "uint8"      # 4x storage reduction (unsigned)
    BINARY = "binary"    # 32x storage reduction
    UBINARY = "ubinary"  # 32x storage reduction (unsigned)


class OutputDimension(int, Enum):
    """Supported output dimensions (V38.0 - HTTP API only).

    Lower dimensions = faster search, less storage.
    Trade-off: slight accuracy reduction.
    """
    D2048 = 2048  # Maximum precision
    D1024 = 1024  # Default (best balance)
    D512 = 512    # 2x faster search
    D256 = 256    # 4x faster search, good for large scale


class InputType(str, Enum):
    """Input type for embedding optimization."""
    DOCUMENT = "document"  # Use for text being stored/indexed
    QUERY = "query"        # Use for search queries (asymmetric retrieval)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding layer.

    Voyage 4 Optimizations:
    - Shared embedding space: Mix query/doc models for cost savings
    - Output dimensions: 2048, 1024 (default), 512, 256
    - Rate limits: 300 RPM, 1M TPM (adjust for your plan)
    - Dynamic batch sizing: Adjust batch size based on token estimates
    """
    model: str = EmbeddingModel.VOYAGE_4_LARGE.value  # Best accuracy, MoE
    query_model: Optional[str] = None  # Lighter model for queries (e.g., voyage-4-lite)
    api_key: Optional[str] = None  # Uses env var or default if not provided
    batch_size: int = 128          # Max texts per API call
    max_retries: int = 3
    retry_delay: float = 1.0       # Seconds between retries

    # Cache configuration
    cache_enabled: bool = True     # Enable in-memory caching
    cache_size: int = 10000        # Max cached embeddings
    cache_ttl: float = 3600.0      # Cache TTL in seconds (1 hour default)

    # Rate limiting
    rate_limit_rpm: int = 300      # Requests per minute
    rate_limit_tpm: int = 1000000  # Tokens per minute

    # Voyage 4 features
    truncation: bool = True        # Truncate long texts
    output_dimension: int = 1024   # Voyage 4 supports: 2048, 1024, 512, 256

    # Dynamic batch sizing
    dynamic_batch_sizing: bool = True  # Adjust batch size based on text length
    target_tokens_per_batch: int = 8000  # Target tokens per batch (conservative)
    min_batch_size: int = 1        # Minimum batch size
    max_batch_size: int = 128      # Maximum batch size (Voyage limit)

    # V39.11: Opik Tracing Configuration
    tracing_enabled: bool = True   # Enable Opik tracing (requires OPIK_API_KEY)
    tracing_project: str = "voyage-embeddings"  # Opik project name
    tracing_workspace: str = "unleash"  # Opik workspace


class CacheEntry(NamedTuple):
    """Cache entry with embedding and metadata for TTL/LRU tracking."""
    embedding: list[float]
    timestamp: float
    hit_count: int = 0


@dataclass
class CacheStats:
    """Statistics for embedding cache performance with memory tracking."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expired: int = 0
    total_bytes: int = 0  # Track memory usage
    embedding_count: int = 0  # Track number of cached embeddings

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        """Cache miss rate as percentage."""
        return 100.0 - self.hit_rate

    @property
    def memory_mb(self) -> float:
        """Estimated memory usage in megabytes."""
        return self.total_bytes / (1024 * 1024)

    def estimate_embedding_bytes(self, embedding: list[float]) -> int:
        """Estimate memory usage of an embedding vector."""
        # float64 = 8 bytes per element + list overhead (~56 bytes)
        return len(embedding) * 8 + 56

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary for serialization."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expired": self.expired,
            "hit_rate_percent": round(self.hit_rate, 2),
            "miss_rate_percent": round(self.miss_rate, 2),
            "embedding_count": self.embedding_count,
            "memory_bytes": self.total_bytes,
            "memory_mb": round(self.memory_mb, 2),
        }


@dataclass
class RateLimiter:
    """Token bucket rate limiter for API calls."""
    rpm_limit: int = 300
    tpm_limit: int = 1000000
    _request_times: list = field(default_factory=list)
    _token_usage: list = field(default_factory=list)

    async def acquire(self, tokens: int = 0) -> None:
        """Wait if rate limited, then acquire slot."""
        now = time.time()
        minute_ago = now - 60

        # Clean old entries
        self._request_times = [t for t in self._request_times if t > minute_ago]
        self._token_usage = [(t, n) for t, n in self._token_usage if t > minute_ago]

        # Check request limit
        if len(self._request_times) >= self.rpm_limit:
            wait_time = self._request_times[0] - minute_ago
            if wait_time > 0:
                logger.debug("rate_limit_waiting", wait_seconds=wait_time)
                await asyncio.sleep(wait_time)

        # Check token limit
        total_tokens = sum(n for _, n in self._token_usage)
        if total_tokens + tokens > self.tpm_limit:
            wait_time = self._token_usage[0][0] - minute_ago if self._token_usage else 1.0
            if wait_time > 0:
                logger.debug("token_limit_waiting", wait_seconds=wait_time)
                await asyncio.sleep(wait_time)

        # Record this request
        self._request_times.append(now)
        if tokens > 0:
            self._token_usage.append((now, tokens))


@dataclass
class EmbeddingResult:
    """Result from embedding operation."""
    embeddings: list[list[float]]
    model: str
    input_type: str
    total_tokens: int = 0
    cached_count: int = 0
    dimension: int = 0

    @property
    def count(self) -> int:
        """Number of embeddings."""
        return len(self.embeddings)

    def __repr__(self) -> str:
        return (
            f"EmbeddingResult(count={self.count}, dim={self.dimension}, "
            f"model={self.model}, tokens={self.total_tokens})"
        )


@dataclass
class RerankResult:
    """Result from reranking operation (V37.0)."""
    results: list[tuple[int, float, str]]  # (index, score, document)
    model: str
    query: str
    total_documents: int

    @property
    def count(self) -> int:
        """Number of results returned."""
        return len(self.results)

    def top_documents(self, k: int = 5) -> list[str]:
        """Get top-k documents by relevance score."""
        return [doc for _, _, doc in self.results[:k]]

    def top_indices(self, k: int = 5) -> list[int]:
        """Get indices of top-k documents."""
        return [idx for idx, _, _ in self.results[:k]]

    def __repr__(self) -> str:
        return (
            f"RerankResult(count={self.count}, total={self.total_documents}, "
            f"model={self.model})"
        )


# =============================================================================
# V39.7: Batch API Infrastructure (33% Cost Savings)
# =============================================================================

class BatchStatus(str, Enum):
    """Batch job status lifecycle.

    V39.7 Feature: Official Voyage AI Batch API status states.

    Lifecycle: validating -> in_progress -> finalizing -> completed
    Alternative paths: validating -> failed
                      any -> cancelling -> cancelled
    """
    VALIDATING = "validating"      # Input file being validated
    IN_PROGRESS = "in_progress"    # Batch is running
    FINALIZING = "finalizing"      # Results being prepared
    COMPLETED = "completed"        # Results ready
    FAILED = "failed"              # Validation or processing failed
    CANCELLING = "cancelling"      # Cancel requested (up to 10 min)
    CANCELLED = "cancelled"        # Successfully cancelled


@dataclass
class BatchRequestCounts:
    """Request counts for batch job tracking."""
    total: int = 0
    completed: int = 0
    failed: int = 0

    @property
    def progress_percent(self) -> float:
        """Completion percentage."""
        return (self.completed / self.total * 100) if self.total > 0 else 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "BatchRequestCounts":
        """Create from dictionary."""
        return cls(
            total=data.get("total", 0),
            completed=data.get("completed", 0),
            failed=data.get("failed", 0),
        )


@dataclass
class BatchJob:
    """
    Batch job representation for async embedding operations.

    V39.7 Feature: 33% cost savings via official Voyage AI Batch API.

    The Batch API is ideal for:
    - Gesture library pre-computation (one-time embedding of all gestures)
    - Session recording embedding (pose datasets from performances)
    - Archetype training data preparation
    - Large corpus embedding for vector database population

    12-hour completion window, up to 100K inputs per batch.

    Example:
        # Create batch job
        job = await layer.create_batch_embedding_job(
            texts=all_gesture_descriptions,
            metadata={"corpus": "gesture_library_v2"}
        )

        # Poll for completion
        while job.status not in [BatchStatus.COMPLETED, BatchStatus.FAILED]:
            await asyncio.sleep(30)
            job = await layer.get_batch_status(job.id)

        # Download results
        if job.status == BatchStatus.COMPLETED:
            embeddings = await layer.download_batch_results(job.id)
    """
    id: str
    status: BatchStatus
    endpoint: str  # /v1/embeddings, /v1/contextualizedembeddings, /v1/rerank
    input_file_id: str
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    model: str = ""
    completion_window: str = "12h"

    # Timestamps
    created_at: Optional[str] = None
    in_progress_at: Optional[str] = None
    expected_completion_at: Optional[str] = None
    finalizing_at: Optional[str] = None
    completed_at: Optional[str] = None
    failed_at: Optional[str] = None
    cancelling_at: Optional[str] = None
    cancelled_at: Optional[str] = None

    # Request tracking
    request_counts: BatchRequestCounts = field(default_factory=BatchRequestCounts)

    # User metadata (up to 16 key-value pairs)
    metadata: dict[str, str] = field(default_factory=dict)

    # Error info if failed
    errors: Optional[list[dict]] = None

    @property
    def is_terminal(self) -> bool:
        """Whether job has reached a terminal state."""
        return self.status in [
            BatchStatus.COMPLETED,
            BatchStatus.FAILED,
            BatchStatus.CANCELLED,
        ]

    @property
    def is_successful(self) -> bool:
        """Whether job completed successfully."""
        return self.status == BatchStatus.COMPLETED

    @classmethod
    def from_api_response(cls, data: dict) -> "BatchJob":
        """Create BatchJob from Voyage AI API response."""
        request_counts = BatchRequestCounts()
        if "request_counts" in data:
            rc = data["request_counts"]
            request_counts = BatchRequestCounts(
                total=rc.get("total", 0),
                completed=rc.get("completed", 0),
                failed=rc.get("failed", 0),
            )

        return cls(
            id=data.get("id", ""),
            status=BatchStatus(data.get("status", "validating")),
            endpoint=data.get("endpoint", "/v1/embeddings"),
            input_file_id=data.get("input_file_id", ""),
            output_file_id=data.get("output_file_id"),
            error_file_id=data.get("error_file_id"),
            model=data.get("model", ""),
            completion_window=data.get("completion_window", "12h"),
            created_at=data.get("created_at"),
            in_progress_at=data.get("in_progress_at"),
            expected_completion_at=data.get("expected_completion_at"),
            finalizing_at=data.get("finalizing_at"),
            completed_at=data.get("completed_at"),
            failed_at=data.get("failed_at"),
            cancelling_at=data.get("cancelling_at"),
            cancelled_at=data.get("cancelled_at"),
            request_counts=request_counts,
            metadata=data.get("metadata", {}),
            errors=data.get("errors"),
        )

    def __repr__(self) -> str:
        return (
            f"BatchJob(id={self.id}, status={self.status.value}, "
            f"progress={self.request_counts.progress_percent:.1f}%, "
            f"endpoint={self.endpoint})"
        )


@dataclass
class BatchFile:
    """Uploaded file for batch processing."""
    id: str
    filename: str
    purpose: str  # "batch" or "batch-output"
    bytes: int
    created_at: str
    expires_at: Optional[str] = None

    @classmethod
    def from_api_response(cls, data: dict) -> "BatchFile":
        """Create BatchFile from API response."""
        return cls(
            id=data.get("id", ""),
            filename=data.get("filename", ""),
            purpose=data.get("purpose", "batch"),
            bytes=data.get("bytes", 0),
            created_at=data.get("created_at", ""),
            expires_at=data.get("expires_at"),
        )


@dataclass
class BatchProgressMetrics:
    """
    Metrics for batch progress tracking with rate and ETA calculation.

    V39.9 Feature: Real-time progress visibility.

    Attributes:
        start_time: When progress tracking started
        samples: List of (timestamp, completed_count) tuples for rate calculation
        window_size: Number of samples to use for moving average (default: 10)
    """
    start_time: datetime
    samples: list[tuple[datetime, int]] = field(default_factory=list)
    window_size: int = 10

    def add_sample(self, completed: int) -> None:
        """Add a progress sample for rate calculation."""
        now = datetime.now()
        self.samples.append((now, completed))
        # Keep only the most recent samples within window
        if len(self.samples) > self.window_size:
            self.samples = self.samples[-self.window_size:]

    def calculate_rate(self) -> float:
        """
        Calculate current processing rate (items/sec).

        Uses linear regression over recent samples for smoothed rate.
        Returns 0.0 if insufficient samples.
        """
        if len(self.samples) < 2:
            return 0.0

        # Calculate rate from oldest to newest sample in window
        oldest_time, oldest_count = self.samples[0]
        newest_time, newest_count = self.samples[-1]

        time_delta = (newest_time - oldest_time).total_seconds()
        if time_delta <= 0:
            return 0.0

        count_delta = newest_count - oldest_count
        return count_delta / time_delta

    def calculate_eta(self, total: int, completed: int) -> float:
        """
        Estimate remaining time in seconds.

        Args:
            total: Total items to process
            completed: Items completed so far

        Returns:
            Estimated seconds remaining, or float('inf') if rate is 0
        """
        rate = self.calculate_rate()
        if rate <= 0:
            return float('inf')

        remaining = total - completed
        return remaining / rate


@dataclass
class BatchProgressEvent:
    """
    Progress event for batch job monitoring.

    V39.9 Feature: Streamed via AsyncGenerator for real-time visibility.

    Attributes:
        batch_id: The batch job ID being monitored
        status: Current batch status (validating, in_progress, completed, etc.)
        total: Total number of requests in the batch
        completed: Number of requests completed
        failed: Number of requests that failed
        percent: Progress percentage (0.0 to 100.0)
        rate: Current processing rate (embeddings per second)
        eta_seconds: Estimated time remaining in seconds
        is_complete: True if batch has finished (success or failure)
        is_failed: True if batch failed
        timestamp: When this event was generated
        error_message: Error details if batch failed
    """
    batch_id: str
    status: BatchStatus
    total: int
    completed: int
    failed: int
    percent: float  # 0.0 to 100.0
    rate: float  # embeddings per second
    eta_seconds: float  # estimated time remaining
    is_complete: bool
    is_failed: bool
    timestamp: datetime
    error_message: Optional[str] = None

    @classmethod
    def from_batch_job(
        cls,
        job: "BatchJob",
        metrics: Optional[BatchProgressMetrics] = None,
    ) -> "BatchProgressEvent":
        """
        Create a BatchProgressEvent from a BatchJob and optional metrics.

        Args:
            job: The BatchJob to extract progress from
            metrics: Optional metrics tracker for rate/ETA calculation

        Returns:
            BatchProgressEvent with current progress state
        """
        total = job.request_counts.total
        completed = job.request_counts.completed
        failed = job.request_counts.failed

        # Calculate percent
        percent = (completed / total * 100.0) if total > 0 else 0.0

        # Calculate rate and ETA from metrics if available
        if metrics:
            rate = metrics.calculate_rate()
            eta_seconds = metrics.calculate_eta(total, completed)
        else:
            rate = 0.0
            eta_seconds = float('inf')

        # Determine completion states
        is_complete = job.status in (BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED)
        is_failed = job.status == BatchStatus.FAILED

        # Extract error message if failed
        error_message = None
        if is_failed and job.errors:
            error_message = str(job.errors)

        return cls(
            batch_id=job.id,
            status=job.status,
            total=total,
            completed=completed,
            failed=failed,
            percent=percent,
            rate=rate,
            eta_seconds=eta_seconds,
            is_complete=is_complete,
            is_failed=is_failed,
            timestamp=datetime.now(),
            error_message=error_message,
        )

    def __repr__(self) -> str:
        eta_str = f"{self.eta_seconds:.0f}s" if self.eta_seconds != float('inf') else "inf"
        return (
            f"BatchProgressEvent({self.completed}/{self.total} "
            f"[{self.percent:.1f}%], rate={self.rate:.1f}/s, ETA={eta_str})"
        )


# =============================================================================
# Embedding Layer Class
# =============================================================================

class EmbeddingLayer:
    """
    Production-grade embedding layer using Voyage AI.

    Features:
    - Automatic model selection (text vs code)
    - Input type optimization (document vs query)
    - Batched processing for large collections
    - In-memory caching to reduce API calls
    - Async interface for concurrent operations

    Usage:
        layer = EmbeddingLayer(config=EmbeddingConfig())
        await layer.initialize()

        # Embed documents for storage
        result = await layer.embed_documents(texts)

        # Embed query for search
        query_embedding = await layer.embed_query("search term")

        # Batch embed with auto-detection
        result = await layer.embed(texts, detect_code=True)
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize embedding layer.

        Args:
            config: Embedding configuration. Uses defaults if not provided.
        """
        self.config = config or EmbeddingConfig()
        self._client: Any = None
        self._initialized = False
        # LRU cache using OrderedDict - access moves to end, oldest at front
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._cache_stats = CacheStats()
        self._rate_limiter = RateLimiter(
            rpm_limit=self.config.rate_limit_rpm,
            tpm_limit=self.config.rate_limit_tpm,
        )
        # V39.11: Initialize Opik tracing
        self._tracing_active = False
        self._configure_tracing()
        # V43: Persistent HTTPx client with connection pooling
        self._httpx_client: Any = None
        if HTTPX_AVAILABLE and _httpx_module:
            self._httpx_client = _httpx_module.Client(
                timeout=120.0,
                limits=_httpx_module.Limits(
                    max_keepalive_connections=20,
                    max_connections=100,
                    keepalive_expiry=30.0,
                ),
            )

    def _configure_tracing(self) -> None:
        """
        Configure Opik tracing if enabled and available (V39.11).

        Checks:
        - OPIK_TRACING_ENABLED env var (or config.tracing_enabled)
        - OPIK_API_KEY env var
        - opik package availability
        """
        if not self.config.tracing_enabled:
            logger.debug("opik_tracing_disabled", reason="config")
            return

        env_enabled = os.environ.get("OPIK_TRACING_ENABLED", "true").lower()
        if env_enabled not in ("true", "1", "yes"):
            logger.debug("opik_tracing_disabled", reason="env_var")
            return

        if not OPIK_AVAILABLE:
            logger.debug("opik_tracing_disabled", reason="package_not_installed")
            return

        api_key = os.environ.get("OPIK_API_KEY")
        if not api_key:
            logger.debug("opik_tracing_disabled", reason="no_api_key")
            return

        try:
            # Configure Opik with workspace and project
            workspace = os.environ.get("OPIK_WORKSPACE", self.config.tracing_workspace)
            _opik_module.configure(
                api_key=api_key,
                workspace=workspace,
            )
            self._tracing_active = True
            logger.info(
                "opik_tracing_enabled",
                workspace=workspace,
                project=self.config.tracing_project,
            )
        except Exception as e:
            logger.warning("opik_tracing_failed", error=str(e))
            self._tracing_active = False

    def _trace_embedding_operation(
        self,
        operation_name: str,
        result: "EmbeddingResult",
        latency_ms: float,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log tracing metadata for an embedding operation (V39.11).

        Called after embed operations to log metrics to Opik.
        """
        if not self._tracing_active or not OPIK_AVAILABLE:
            return

        try:
            # Log metadata about the operation
            metadata = {
                "model": result.model,
                "input_type": result.input_type,
                "text_count": len(result.embeddings),
                "dimension": result.dimension,
                "total_tokens": result.total_tokens,
                "cached_count": result.cached_count,
                "latency_ms": latency_ms,
            }
            if extra_metadata:
                metadata.update(extra_metadata)

            _opik_module.log_metadata(metadata)

            # Update span with cost tracking (Voyage pricing: $0.03 per 1M tokens)
            if result.total_tokens > 0:
                _opik_context_module.update_current_span(
                    provider="voyage",
                    model=result.model,
                    usage={
                        "prompt_tokens": result.total_tokens,
                        "completion_tokens": 0,
                        "total_tokens": result.total_tokens,
                    }
                )
        except Exception as e:
            logger.debug("opik_trace_metadata_failed", error=str(e))

    def _trace_search_operation(
        self,
        operation_name: str,
        query: str,
        num_docs: int,
        num_results: int,
        latency_ms: float,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log tracing metadata for a search operation (V39.11).

        Called after search operations to log metrics to Opik.
        """
        if not self._tracing_active or not OPIK_AVAILABLE:
            return

        try:
            metadata = {
                "operation": operation_name,
                "query_length": len(query),
                "num_documents": num_docs,
                "num_results": num_results,
                "latency_ms": latency_ms,
            }
            if extra_metadata:
                metadata.update(extra_metadata)

            _opik_module.log_metadata(metadata)
            _opik_module.set_tags(["voyage", "search", operation_name])

        except Exception as e:
            logger.debug("opik_search_trace_failed", error=str(e))

    @property
    def is_available(self) -> bool:
        """Check if Voyage AI SDK is available."""
        return VOYAGE_AVAILABLE

    @property
    def is_initialized(self) -> bool:
        """Check if layer is initialized."""
        return self._initialized

    def _get_api_key(self) -> str:
        """Get API key from config, env, or default."""
        # Priority: config > env > default
        if self.config.api_key:
            return self.config.api_key

        env_key = os.getenv("VOYAGE_API_KEY")
        if env_key:
            return env_key

        return _DEFAULT_VOYAGE_API_KEY

    def _get_cache_key(self, text: str, model: str, input_type: str) -> str:
        """Generate cache key for embedding."""
        content = f"{model}:{input_type}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_effective_model(
        self,
        input_type: InputType,
        model_override: Optional[str | EmbeddingModel] = None,
    ) -> str:
        """
        Get the effective model for embedding based on input type.

        Voyage 4 shared embedding space allows using different models
        for queries vs documents while maintaining compatibility.

        Args:
            input_type: Query or document
            model_override: Explicit model override (highest priority)

        Returns:
            Model name string to use
        """
        if model_override:
            # Handle both string and enum model specifications
            if isinstance(model_override, EmbeddingModel):
                return model_override.value
            return model_override

        # For queries, prefer query_model if configured
        if input_type == InputType.QUERY and self.config.query_model:
            return self.config.query_model

        return self.config.model

    def _calculate_dynamic_batch_size(self, texts: list[str]) -> int:
        """
        Calculate optimal batch size based on text lengths.

        Uses token estimation (~4 chars per token) to determine batch size
        that stays within target tokens per batch for optimal throughput.

        Args:
            texts: List of texts to embed

        Returns:
            Optimal batch size for these texts
        """
        if not self.config.dynamic_batch_sizing or not texts:
            return self.config.batch_size

        # Estimate tokens per text (~4 chars per token is a reasonable estimate)
        avg_chars = sum(len(t) for t in texts) / len(texts)
        avg_tokens = avg_chars / 4

        # Calculate batch size to hit target tokens per batch
        if avg_tokens > 0:
            optimal_batch = int(self.config.target_tokens_per_batch / avg_tokens)
        else:
            optimal_batch = self.config.batch_size

        # Clamp between min and max
        return max(
            self.config.min_batch_size,
            min(optimal_batch, self.config.max_batch_size),
        )

    def _check_cache(
        self,
        texts: list[str],
        model: str,
        input_type: str,
    ) -> tuple[list[str], list[int], dict[int, list[float]]]:
        """
        Check cache for existing embeddings with TTL and LRU support.

        Returns:
            (uncached_texts, uncached_indices, cached_embeddings)
        """
        if not self.config.cache_enabled:
            self._cache_stats.misses += len(texts)
            return texts, list(range(len(texts))), {}

        uncached_texts: list[str] = []
        uncached_indices: list[int] = []
        cached: dict[int, list[float]] = {}
        now = time.time()

        for i, text in enumerate(texts):
            key = self._get_cache_key(text, model, input_type)
            if key in self._cache:
                entry = self._cache[key]
                # Check TTL
                if now - entry.timestamp > self.config.cache_ttl:
                    # Expired - remove and count as miss
                    del self._cache[key]
                    self._cache_stats.expired += 1
                    self._cache_stats.misses += 1
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                else:
                    # LRU: Move to end (most recently used)
                    self._cache.move_to_end(key)
                    cached[i] = entry.embedding
                    self._cache_stats.hits += 1
            else:
                self._cache_stats.misses += 1
                uncached_texts.append(text)
                uncached_indices.append(i)

        return uncached_texts, uncached_indices, cached

    def _update_cache(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        model: str,
        input_type: str,
    ) -> None:
        """Update cache with new embeddings using LRU eviction with memory tracking."""
        if not self.config.cache_enabled:
            return

        now = time.time()
        for text, embedding in zip(texts, embeddings):
            # LRU eviction: remove oldest (first) entries
            while len(self._cache) >= self.config.cache_size:
                _, evicted_entry = self._cache.popitem(last=False)  # Remove from front (oldest)
                self._cache_stats.evictions += 1
                self._cache_stats.embedding_count -= 1
                self._cache_stats.total_bytes -= self._cache_stats.estimate_embedding_bytes(
                    evicted_entry.embedding
                )

            key = self._get_cache_key(text, model, input_type)
            self._cache[key] = CacheEntry(
                embedding=embedding,
                timestamp=now,
                hit_count=0,
            )
            # Track memory usage
            self._cache_stats.embedding_count += 1
            self._cache_stats.total_bytes += self._cache_stats.estimate_embedding_bytes(embedding)

    async def initialize(self) -> None:
        """
        Initialize the embedding layer.

        Raises:
            SDKNotAvailableError: If voyageai is not installed
            SDKConfigurationError: If API key is invalid
        """
        if self._initialized:
            return

        if not VOYAGE_AVAILABLE:
            raise SDKNotAvailableError(
                sdk_name="voyageai",
                install_cmd="pip install voyageai>=0.3.0",
                docs_url="https://docs.voyageai.com/docs/embeddings"
            )

        api_key = self._get_api_key()
        if not api_key:
            raise SDKConfigurationError(
                sdk_name="voyage_embedding",
                missing_config=["VOYAGE_API_KEY"],
                example="VOYAGE_API_KEY=pa-xxxxx"
            )

        try:
            self._client = _voyageai_module.Client(api_key=api_key)
            # V42 FIX: Wrap sync Voyage SDK call with asyncio.to_thread() for async safety
            # Verify with a small test embedding
            test_result = await asyncio.to_thread(
                self._client.embed,
                texts=["test"],
                model=self.config.model,
                input_type="document",
                truncation=True,
            )
            if not test_result.embeddings:
                raise SDKConfigurationError(
                    sdk_name="voyage_embedding",
                    missing_config=["valid_api_key"],
                    example="Check API key at https://dash.voyageai.com/"
                )

            self._initialized = True
            logger.info(
                "embedding_layer_initialized",
                model=self.config.model,
                cache_enabled=self.config.cache_enabled,
            )

        except Exception as e:
            if "Invalid API key" in str(e) or "401" in str(e):
                raise SDKConfigurationError(
                    sdk_name="voyage_embedding",
                    missing_config=["valid_api_key"],
                    example="Check API key at https://dash.voyageai.com/"
                )
            raise

    async def close(self) -> None:
        """
        V43: Close resources including persistent HTTPx client.

        Call this when done with the embedding layer to release connections.
        """
        if self._httpx_client is not None:
            try:
                self._httpx_client.close()
                logger.debug("httpx_client_closed")
            except Exception as e:
                logger.debug(f"httpx_client_close_error: {e}")
            finally:
                self._httpx_client = None
        self._initialized = False

    async def __aenter__(self) -> "EmbeddingLayer":
        """V43: Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """V43: Async context manager exit with cleanup."""
        await self.close()

    async def embed(
        self,
        texts: Sequence[str],
        input_type: InputType = InputType.DOCUMENT,
        model: Optional[str | EmbeddingModel] = None,
        detect_code: bool = False,
    ) -> EmbeddingResult:
        """
        Embed texts with specified configuration.

        Args:
            texts: Texts to embed
            input_type: "document" for storage, "query" for retrieval
            model: Override default model
            detect_code: Auto-switch to code model if code detected

        Returns:
            EmbeddingResult with embeddings and metadata

        Raises:
            SDKNotAvailableError: If not initialized
        """
        # V39.11: Start timing for tracing
        start_time = time.perf_counter()

        if not self._initialized:
            await self.initialize()

        texts_list = list(texts)
        if not texts_list:
            return EmbeddingResult(
                embeddings=[],
                model=model or self.config.model,
                input_type=input_type.value,
                dimension=0,
            )

        # Get effective model using model mixing logic
        effective_model = self._get_effective_model(input_type, model)

        # Auto-detect code if requested (overrides model mixing)
        if detect_code and self._looks_like_code(texts_list):
            effective_model = EmbeddingModel.VOYAGE_CODE_3.value
            logger.debug("auto_detected_code", switched_to=effective_model)

        # Check cache
        uncached_texts, uncached_indices, cached = self._check_cache(
            texts_list, effective_model, input_type.value
        )

        cached_count = len(cached)
        new_embeddings: list[list[float]] = []
        total_tokens = 0

        if uncached_texts:
            # Calculate dynamic batch size based on text lengths
            batch_size = self._calculate_dynamic_batch_size(uncached_texts)

            # Process in batches
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i + batch_size]
                batch_result = await self._embed_batch(
                    batch, effective_model, input_type.value
                )
                new_embeddings.extend(batch_result["embeddings"])
                total_tokens += batch_result.get("total_tokens", 0)

            # Update cache
            self._update_cache(
                uncached_texts, new_embeddings, effective_model, input_type.value
            )

        # Reconstruct full results in original order
        embeddings: list[list[float]] = [[] for _ in range(len(texts_list))]

        # Fill cached embeddings
        for idx, emb in cached.items():
            embeddings[idx] = emb

        # Fill new embeddings
        for idx, emb in zip(uncached_indices, new_embeddings):
            embeddings[idx] = emb

        dimension = len(embeddings[0]) if embeddings and embeddings[0] else 0

        logger.info(
            "embeddings_generated",
            count=len(texts_list),
            cached=cached_count,
            model=effective_model,
            dimension=dimension,
        )

        result = EmbeddingResult(
            embeddings=embeddings,
            model=effective_model,
            input_type=input_type.value,
            total_tokens=total_tokens,
            cached_count=cached_count,
            dimension=dimension,
        )

        # V39.11: Log tracing metadata to Opik
        latency_ms = (time.perf_counter() - start_time) * 1000
        self._trace_embedding_operation(
            operation_name="embed",
            result=result,
            latency_ms=latency_ms,
            extra_metadata={
                "detect_code": detect_code,
                "cache_enabled": self.config.cache_enabled,
            },
        )

        return result

    async def _embed_batch(
        self,
        texts: list[str],
        model: str,
        input_type: str,
    ) -> dict[str, Any]:
        """Embed a single batch with retry logic and rate limiting."""
        last_error: Optional[Exception] = None

        # Estimate tokens (rough: ~1.3 tokens per word, ~4 chars per token)
        estimated_tokens = sum(len(t) // 4 for t in texts)

        for attempt in range(self.config.max_retries):
            try:
                # Apply rate limiting before API call
                await self._rate_limiter.acquire(tokens=estimated_tokens)

                # Note: voyageai SDK 0.2.x doesn't support output_dimension yet
                # Voyage 4 models default to 1024d; when SDK updates, add:
                # output_dimension=self.config.output_dimension
                # V42 FIX: Wrap sync Voyage SDK call with asyncio.to_thread() for async safety
                result = await asyncio.to_thread(
                    self._client.embed,
                    texts=texts,
                    model=model,
                    input_type=input_type,
                    truncation=self.config.truncation,
                )
                return {
                    "embeddings": result.embeddings,
                    "total_tokens": getattr(result, "total_tokens", 0),
                }

            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                # Handle rate limit errors specifically
                if "rate" in error_str or "429" in error_str:
                    wait_time = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        "rate_limit_hit",
                        attempt=attempt + 1,
                        wait_seconds=wait_time,
                    )
                    await asyncio.sleep(wait_time)
                elif attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    logger.warning(
                        "embed_retry",
                        attempt=attempt + 1,
                        error=str(e),
                    )

        raise last_error or RuntimeError("Embedding failed")

    def _looks_like_code(self, texts: list[str]) -> bool:
        """Simple heuristic to detect if texts contain code."""
        code_indicators = [
            "def ", "class ", "import ", "from ", "return ",  # Python
            "function ", "const ", "let ", "var ", "=>",      # JavaScript
            "public ", "private ", "void ", "int ",           # Java/C++
            "fn ", "impl ", "use ", "mod ",                   # Rust
            "{", "}", ";", "->", "==", "!=",                  # Common syntax
        ]

        sample_text = " ".join(texts[:5])[:2000]  # Sample first 5 texts
        indicator_count = sum(1 for ind in code_indicators if ind in sample_text)

        return indicator_count >= 3

    async def embed_documents(
        self,
        documents: Sequence[str],
        model: Optional[str] = None,
    ) -> EmbeddingResult:
        """
        Embed documents for storage/indexing.

        Uses input_type="document" for optimal retrieval performance.

        Args:
            documents: Document texts to embed
            model: Override default model

        Returns:
            EmbeddingResult with document embeddings
        """
        return await self.embed(
            documents,
            input_type=InputType.DOCUMENT,
            model=model,
        )

    async def embed_query(
        self,
        query: str,
        model: Optional[str] = None,
    ) -> list[float]:
        """
        Embed a single query for retrieval.

        Uses input_type="query" for asymmetric retrieval optimization.

        Args:
            query: Query text to embed
            model: Override default model

        Returns:
            Query embedding vector
        """
        result = await self.embed(
            [query],
            input_type=InputType.QUERY,
            model=model,
        )
        return result.embeddings[0] if result.embeddings else []

    async def embed_queries(
        self,
        queries: Sequence[str],
        model: Optional[str] = None,
    ) -> EmbeddingResult:
        """
        Embed multiple queries for batch retrieval.

        Args:
            queries: Query texts to embed
            model: Override default model

        Returns:
            EmbeddingResult with query embeddings
        """
        return await self.embed(
            queries,
            input_type=InputType.QUERY,
            model=model,
        )

    async def embed_code(
        self,
        code_snippets: Sequence[str],
        input_type: InputType = InputType.DOCUMENT,
    ) -> EmbeddingResult:
        """
        Embed code snippets using the code-specialized model.

        Args:
            code_snippets: Code texts to embed
            input_type: "document" for storage, "query" for search

        Returns:
            EmbeddingResult with code embeddings
        """
        return await self.embed(
            code_snippets,
            input_type=input_type,
            model=EmbeddingModel.VOYAGE_CODE_3.value,
        )

    def clear_cache(self) -> int:
        """Clear embedding cache. Returns number of entries cleared."""
        count = len(self._cache)
        self._cache.clear()
        self._cache_stats = CacheStats()  # Reset stats
        logger.info("embedding_cache_cleared", entries=count)
        return count

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get detailed cache statistics for monitoring and optimization.

        Returns:
            Dictionary with comprehensive cache metrics including:
            - hits/misses: Raw access counts
            - hit_rate_percent: Cache efficiency metric
            - memory_mb: Estimated memory usage
            - embedding_count: Number of cached embeddings
            - cache_size: Current/max cache entries
            - ttl_seconds: Time-to-live configuration
        """
        return {
            **self._cache_stats.to_dict(),
            "cache_size": len(self._cache),
            "cache_max_size": self.config.cache_size,
            "cache_enabled": self.config.cache_enabled,
            "ttl_seconds": self.config.cache_ttl,
            "estimated_dimension": (
                len(next(iter(self._cache.values())).embedding)
                if self._cache
                else 0
            ),
        }

    async def warm_cache(
        self,
        texts: list[str],
        model: EmbeddingModel | str = EmbeddingModel.VOYAGE_4_LARGE,
        input_type: InputType = InputType.DOCUMENT,
    ) -> dict[str, Any]:
        """
        Pre-warm cache with frequently used embeddings.

        This is useful for:
        - Application startup to reduce cold-start latency
        - Pre-loading known query patterns
        - Warming after cache clear

        Args:
            texts: List of texts to pre-embed and cache
            model: Embedding model to use
            input_type: Type of input (document/query)

        Returns:
            Statistics about the warming operation
        """
        if not texts:
            return {"warmed": 0, "already_cached": 0, "errors": 0}

        model_str = model.value if isinstance(model, EmbeddingModel) else model
        input_type_str = input_type.value if isinstance(input_type, InputType) else str(input_type)

        # Check which texts are already cached
        uncached_texts, uncached_indices, cached = self._check_cache(
            texts, model_str, input_type_str
        )

        already_cached = len(cached)
        warmed = 0
        errors = 0

        if uncached_texts:
            try:
                # Embed uncached texts
                result = await self.embed(
                    texts=uncached_texts,
                    model=model,
                    input_type=input_type,
                )
                warmed = len(result.embeddings)
            except Exception as e:
                logger.error("cache_warm_error", error=str(e))
                errors = len(uncached_texts)

        logger.info(
            "cache_warmed",
            warmed=warmed,
            already_cached=already_cached,
            errors=errors,
        )

        return {
            "warmed": warmed,
            "already_cached": already_cached,
            "errors": errors,
            "cache_stats": self.get_cache_stats(),
        }

    def get_cache_efficiency_report(self) -> dict[str, Any]:
        """
        Generate a detailed efficiency report for cache optimization decisions.

        Returns:
            Comprehensive report with recommendations
        """
        stats = self._cache_stats
        cache_size = len(self._cache)
        max_size = self.config.cache_size

        # Calculate utilization
        utilization = (cache_size / max_size * 100) if max_size > 0 else 0

        # Calculate efficiency score (0-100)
        efficiency_score = stats.hit_rate * 0.7 + (100 - utilization) * 0.3

        # Generate recommendations
        recommendations: list[str] = []
        if stats.hit_rate < 30:
            recommendations.append("Consider increasing cache_ttl for longer retention")
        if utilization > 90 and stats.evictions > stats.hits:
            recommendations.append("Consider increasing cache_size to reduce evictions")
        if stats.expired > stats.evictions:
            recommendations.append("TTL is causing more evictions than capacity - consider longer TTL")
        if utilization < 20 and cache_size > 100:
            recommendations.append("Cache is underutilized - consider smaller cache_size to save memory")

        return {
            "efficiency_score": round(efficiency_score, 1),
            "utilization_percent": round(utilization, 1),
            "stats": stats.to_dict(),
            "capacity": {
                "current": cache_size,
                "maximum": max_size,
                "available": max_size - cache_size,
            },
            "recommendations": recommendations,
        }

    def export_cache(self) -> dict[str, Any]:
        """
        Export cache state for persistence or transfer.

        Returns:
            Dictionary with cache entries that can be serialized to JSON.
        """
        entries = []
        for key, entry in self._cache.items():
            entries.append({
                "key": key,
                "embedding": entry.embedding,
                "timestamp": entry.timestamp,
                "hit_count": entry.hit_count,
            })

        return {
            "version": "V39.1",
            "exported_at": time.time(),
            "entries": entries,
            "stats": self._cache_stats.to_dict(),
            "config": {
                "cache_size": self.config.cache_size,
                "cache_ttl": self.config.cache_ttl,
            },
        }

    def import_cache(
        self,
        cache_data: dict[str, Any],
        validate_ttl: bool = True,
    ) -> dict[str, Any]:
        """
        Import cache state from exported data.

        Args:
            cache_data: Previously exported cache data
            validate_ttl: If True, skip expired entries based on current time

        Returns:
            Import statistics
        """
        if "entries" not in cache_data:
            return {"imported": 0, "skipped": 0, "error": "Invalid cache data format"}

        imported = 0
        skipped = 0
        now = time.time()

        for entry_data in cache_data.get("entries", []):
            try:
                key = entry_data["key"]
                timestamp = entry_data["timestamp"]

                # Skip expired entries if validation is enabled
                if validate_ttl and (now - timestamp > self.config.cache_ttl):
                    skipped += 1
                    continue

                # Skip if cache is full
                if len(self._cache) >= self.config.cache_size:
                    # Evict oldest entry
                    _, evicted = self._cache.popitem(last=False)
                    self._cache_stats.evictions += 1
                    self._cache_stats.embedding_count -= 1
                    self._cache_stats.total_bytes -= self._cache_stats.estimate_embedding_bytes(
                        evicted.embedding
                    )

                embedding = entry_data["embedding"]
                self._cache[key] = CacheEntry(
                    embedding=embedding,
                    timestamp=timestamp,
                    hit_count=entry_data.get("hit_count", 0),
                )
                self._cache_stats.embedding_count += 1
                self._cache_stats.total_bytes += self._cache_stats.estimate_embedding_bytes(embedding)
                imported += 1

            except (KeyError, TypeError) as e:
                logger.warning("cache_import_entry_error", error=str(e))
                skipped += 1

        logger.info("cache_imported", imported=imported, skipped=skipped)
        return {
            "imported": imported,
            "skipped": skipped,
            "cache_stats": self.get_cache_stats(),
        }

    async def save_cache_to_file(self, file_path: str) -> dict[str, Any]:
        """
        Save cache to a JSON file for persistence across sessions.

        Args:
            file_path: Path to save the cache file

        Returns:
            Save operation statistics
        """
        import json
        from pathlib import Path

        try:
            cache_data = self.export_cache()
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w") as f:
                json.dump(cache_data, f)

            logger.info("cache_saved_to_file", path=str(path), entries=len(cache_data["entries"]))
            return {
                "success": True,
                "path": str(path),
                "entries_saved": len(cache_data["entries"]),
                "size_bytes": path.stat().st_size,
            }
        except Exception as e:
            logger.error("cache_save_error", error=str(e))
            return {"success": False, "error": str(e)}

    async def load_cache_from_file(
        self,
        file_path: str,
        validate_ttl: bool = True,
    ) -> dict[str, Any]:
        """
        Load cache from a JSON file.

        Args:
            file_path: Path to the cache file
            validate_ttl: Skip expired entries if True

        Returns:
            Load operation statistics
        """
        import json
        from pathlib import Path

        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        try:
            with open(path) as f:
                cache_data = json.load(f)

            result = self.import_cache(cache_data, validate_ttl=validate_ttl)
            result["success"] = True
            result["path"] = str(path)
            logger.info("cache_loaded_from_file", path=str(path), imported=result["imported"])
            return result

        except Exception as e:
            logger.error("cache_load_error", error=str(e))
            return {"success": False, "error": str(e)}

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive layer statistics including cache performance."""
        # Determine if model mixing is active
        model_mixing_active = bool(
            self.config.query_model
            and self.config.query_model != self.config.model
        )

        return {
            "version": "V39.1",  # Updated version with cache enhancements
            "initialized": self._initialized,
            "model": self.config.model,
            "query_model": self.config.query_model,
            "model_mixing_active": model_mixing_active,
            "cache_enabled": self.config.cache_enabled,
            "cache_size": len(self._cache),
            "cache_max_size": self.config.cache_size,
            "cache_ttl_seconds": self.config.cache_ttl,
            "cache_stats": self._cache_stats.to_dict(),
            "rate_limits": {
                "rpm": self.config.rate_limit_rpm,
                "tpm": self.config.rate_limit_tpm,
            },
            "dynamic_batch_sizing": {
                "enabled": self.config.dynamic_batch_sizing,
                "target_tokens": self.config.target_tokens_per_batch,
                "min_batch": self.config.min_batch_size,
                "max_batch": self.config.max_batch_size,
            },
            "voyage_available": VOYAGE_AVAILABLE,
        }

    # =========================================================================
    # Semantic Search Helpers
    # =========================================================================

    async def semantic_search(
        self,
        query: str,
        documents: list[str],
        doc_embeddings: Optional[list[list[float]]] = None,
        top_k: int = 5,
    ) -> list[tuple[int, float, str]]:
        """
        Perform semantic search over documents.

        Args:
            query: Search query
            documents: List of document texts
            doc_embeddings: Pre-computed embeddings (optional, will compute if None)
            top_k: Number of results to return

        Returns:
            List of (index, similarity_score, document) tuples, sorted by score
        """
        # V39.11: Start timing for tracing
        start_time = time.perf_counter()

        if not self._initialized:
            await self.initialize()

        # Get query embedding
        query_emb = await self.embed_query(query)

        # Get or compute document embeddings
        if doc_embeddings is None:
            result = await self.embed_documents(documents)
            doc_embeddings = result.embeddings

        # Compute similarities
        similarities = [
            (i, self.cosine_similarity(query_emb, doc_emb), doc)
            for i, (doc_emb, doc) in enumerate(zip(doc_embeddings, documents))
        ]

        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = similarities[:top_k]

        # V39.11: Log tracing metadata
        latency_ms = (time.perf_counter() - start_time) * 1000
        self._trace_search_operation(
            operation_name="semantic_search",
            query=query,
            num_docs=len(documents),
            num_results=len(results),
            latency_ms=latency_ms,
            extra_metadata={"top_k": top_k},
        )

        return results

    @staticmethod
    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    @staticmethod
    def batch_cosine_similarity(
        query_emb: list[float],
        doc_embeddings: list[list[float]],
    ) -> list[float]:
        """Compute cosine similarity between query and multiple documents."""
        return [
            EmbeddingLayer.cosine_similarity(query_emb, doc_emb)
            for doc_emb in doc_embeddings
        ]

    # =========================================================================
    # Reranking (V37.0 - Cross-Encoder Refinement)
    # =========================================================================

    async def rerank(
        self,
        query: str,
        documents: list[str],
        model: str = RerankModel.RERANK_2_5.value,
        top_k: Optional[int] = None,
    ) -> RerankResult:
        """
        Rerank documents using Voyage AI cross-encoder models.

        Reranking uses a cross-encoder that jointly processes query and document
        together, providing more accurate relevance scores than embedding-based
        similarity. Use this to refine top candidates from initial embedding search.

        Best Practice:
        1. Use embeddings to get top 50-100 candidates (fast, scalable)
        2. Use reranking to refine to top 5-10 (more accurate, expensive)

        Args:
            query: Search query
            documents: List of candidate documents to rerank
            model: Reranking model (default: rerank-2.5 - best quality)
            top_k: Number of results to return (default: all)

        Returns:
            RerankResult with scored and sorted documents

        Raises:
            SDKNotAvailableError: If not initialized
        """
        if not self._initialized:
            await self.initialize()

        if not documents:
            return RerankResult(
                results=[],
                model=model,
                query=query,
                total_documents=0,
            )

        # Apply rate limiting
        estimated_tokens = len(query.split()) + sum(len(d.split()) for d in documents)
        await self._rate_limiter.acquire(tokens=estimated_tokens)

        try:
            # V42 FIX: Wrap sync Voyage SDK call with asyncio.to_thread() for async safety
            # Voyage AI SDK rerank method
            result = await asyncio.to_thread(
                self._client.rerank,
                query=query,
                documents=documents,
                model=model,
                top_k=top_k or len(documents),
            )

            # Convert to RerankResult format
            scored_results = [
                (item.index, item.relevance_score, documents[item.index])
                for item in result.results
            ]

            logger.info(
                "rerank_completed",
                query_preview=query[:50],
                total_docs=len(documents),
                returned=len(scored_results),
                model=model,
            )

            return RerankResult(
                results=scored_results,
                model=model,
                query=query,
                total_documents=len(documents),
            )

        except Exception as e:
            logger.error("rerank_failed", error=str(e))
            raise

    async def semantic_search_with_rerank(
        self,
        query: str,
        documents: list[str],
        doc_embeddings: Optional[list[list[float]]] = None,
        initial_k: int = 50,
        final_k: int = 5,
        rerank_model: str = RerankModel.RERANK_2_5.value,
    ) -> RerankResult:
        """
        Two-stage retrieval: embedding search + reranking.

        This is the recommended approach for high-quality semantic search:
        1. Embeddings: Fast filtering to top initial_k candidates
        2. Reranking: Cross-encoder refinement to top final_k results

        The two-stage approach provides:
        - Scalability: Embeddings can search millions of documents
        - Accuracy: Reranking provides more nuanced relevance scoring
        - Efficiency: Only rerank a small candidate set

        Args:
            query: Search query
            documents: All documents to search
            doc_embeddings: Pre-computed embeddings (optional)
            initial_k: Number of candidates from embedding search
            final_k: Number of final results after reranking
            rerank_model: Model for reranking stage

        Returns:
            RerankResult with top final_k documents

        Example:
            >>> result = await layer.semantic_search_with_rerank(
            ...     query="machine learning optimization",
            ...     documents=all_docs,
            ...     initial_k=50,
            ...     final_k=5,
            ... )
            >>> for idx, score, doc in result.results:
            ...     print(f"[{score:.4f}] {doc[:100]}...")
        """
        if not self._initialized:
            await self.initialize()

        # Stage 1: Embedding-based retrieval
        initial_results = await self.semantic_search(
            query=query,
            documents=documents,
            doc_embeddings=doc_embeddings,
            top_k=min(initial_k, len(documents)),
        )

        # Extract candidate documents for reranking
        candidates = [doc for _, _, doc in initial_results]

        if not candidates:
            return RerankResult(
                results=[],
                model=rerank_model,
                query=query,
                total_documents=len(documents),
            )

        # Stage 2: Cross-encoder reranking
        rerank_result = await self.rerank(
            query=query,
            documents=candidates,
            model=rerank_model,
            top_k=final_k,
        )

        # Map back to original document indices
        candidate_to_original = {doc: idx for idx, _, doc in initial_results}
        final_results = [
            (candidate_to_original[doc], score, doc)
            for _, score, doc in rerank_result.results
        ]

        logger.info(
            "two_stage_search_completed",
            total_docs=len(documents),
            initial_k=len(candidates),
            final_k=len(final_results),
        )

        return RerankResult(
            results=final_results,
            model=rerank_model,
            query=query,
            total_documents=len(documents),
        )

    # =========================================================================
    # HTTP API Features (V38.0 - Advanced Options)
    # =========================================================================

    async def embed_with_options(
        self,
        texts: Sequence[str],
        input_type: InputType = InputType.DOCUMENT,
        model: Optional[str] = None,
        output_dimension: Optional[int] = None,
        output_dtype: Optional[str] = None,
    ) -> EmbeddingResult:
        """
        Embed texts with advanced HTTP API options not in SDK.

        This method uses direct HTTP API calls to access features that the
        voyageai SDK doesn't expose yet:
        - output_dimension: Reduce dimensions for storage savings
        - output_dtype: Quantization for massive storage reduction

        Requires httpx: `pip install httpx`

        Args:
            texts: Texts to embed
            input_type: "document" for storage, "query" for retrieval
            model: Override default model
            output_dimension: Target dimension (2048, 1024, 512, 256)
                            - Lower = less storage, faster search
                            - Slight accuracy trade-off
            output_dtype: Output format (float, int8, uint8, binary, ubinary)
                        - float: Full precision (default, 4 bytes/dim)
                        - int8/uint8: 4x reduction (1 byte/dim)
                        - binary/ubinary: 32x reduction (1 bit/dim)

        Returns:
            EmbeddingResult with embeddings and metadata

        Raises:
            SDKNotAvailableError: If httpx not installed
            RuntimeError: If API request fails

        Example:
            >>> # High-compression for large-scale search
            >>> result = await layer.embed_with_options(
            ...     texts=documents,
            ...     output_dimension=256,  # 4x smaller vectors
            ...     output_dtype="int8",   # 4x more compression
            ... )  # Total: 16x storage reduction!
        """
        if not self._initialized:
            await self.initialize()

        # If no advanced options, use standard SDK path
        if output_dimension is None and output_dtype is None:
            return await self.embed(texts, input_type=input_type, model=model)

        # Check httpx availability
        if not HTTPX_AVAILABLE or _httpx_module is None:
            raise SDKNotAvailableError(
                sdk_name="httpx",
                install_cmd="pip install httpx",
                docs_url="https://www.python-httpx.org/"
            )

        texts_list = list(texts)
        if not texts_list:
            return EmbeddingResult(
                embeddings=[],
                model=model or self.config.model,
                input_type=input_type.value,
                dimension=0,
            )

        effective_model = self._get_effective_model(input_type, model)

        # Estimate tokens for rate limiting
        estimated_tokens = sum(len(t) // 4 for t in texts_list)
        await self._rate_limiter.acquire(tokens=estimated_tokens)

        # Build HTTP request
        api_key = self._get_api_key()
        url = "https://api.voyageai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "input": texts_list,
            "model": effective_model,
            "input_type": input_type.value,
        }

        # Add optional parameters
        if output_dimension is not None:
            payload["output_dimension"] = output_dimension
        if output_dtype is not None:
            payload["output_dtype"] = output_dtype
        if self.config.truncation:
            payload["truncation"] = True

        last_error: Optional[Exception] = None
        # V43: Use persistent client with connection pooling
        http_client = self._httpx_client

        def _sync_http_call() -> dict[str, Any]:
            """Sync HTTP call to run in thread pool (avoids async detection issues)."""
            # V43: Use persistent connection pool instead of per-request client
            if http_client is not None:
                response = http_client.post(url, headers=headers, json=payload, timeout=60.0)
                response.raise_for_status()
                return response.json()
            else:
                # Fallback if no persistent client
                with _httpx_module.Client(timeout=60.0) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    return response.json()

        for attempt in range(self.config.max_retries):
            try:
                # Use asyncio.to_thread to avoid async library detection issues
                # This is more robust with nest_asyncio patched event loops
                data = await asyncio.to_thread(_sync_http_call)

                embeddings = [item["embedding"] for item in data.get("data", [])]
                total_tokens = data.get("usage", {}).get("total_tokens", 0)
                dimension = len(embeddings[0]) if embeddings else 0

                logger.info(
                    "http_embeddings_generated",
                    count=len(embeddings),
                    model=effective_model,
                    dimension=dimension,
                    output_dtype=output_dtype,
                )

                return EmbeddingResult(
                    embeddings=embeddings,
                    model=effective_model,
                    input_type=input_type.value,
                    total_tokens=total_tokens,
                    cached_count=0,  # No caching for advanced options
                    dimension=dimension,
                )

            except _httpx_module.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Rate limited - backoff and retry
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    logger.warning("http_rate_limit", attempt=attempt + 1, wait=wait_time)
                    await asyncio.sleep(wait_time)
                    last_error = e
                else:
                    last_error = e
                    if attempt < self.config.max_retries - 1:
                        wait_time = self.config.retry_delay * (attempt + 1)
                        logger.warning("http_embed_retry", attempt=attempt + 1, error=str(e))
                        await asyncio.sleep(wait_time)

            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    wait_time = self.config.retry_delay * (attempt + 1)
                    logger.warning("http_embed_retry", attempt=attempt + 1, error=str(e))
                    await asyncio.sleep(wait_time)

        raise RuntimeError(f"HTTP embedding failed after {self.config.max_retries} attempts: {last_error}")

    async def embed_quantized(
        self,
        texts: Sequence[str],
        dtype: str = OutputDType.INT8.value,
        dimension: int = OutputDimension.D1024.value,
        input_type: InputType = InputType.DOCUMENT,
        model: Optional[str] = None,
    ) -> EmbeddingResult:
        """
        Convenience method for quantized embeddings with dimension reduction.

        Combines output_dimension and output_dtype for maximum storage savings.

        Storage Comparison (per embedding, 1024d baseline):
        - float 1024d: 4096 bytes (baseline)
        - int8 1024d: 1024 bytes (4x reduction)
        - int8 256d: 256 bytes (16x reduction)
        - binary 256d: 32 bytes (128x reduction!)

        Args:
            texts: Texts to embed
            dtype: Quantization type (int8, uint8, binary, ubinary)
            dimension: Target dimension (2048, 1024, 512, 256)
            input_type: "document" or "query"
            model: Override default model

        Returns:
            EmbeddingResult with quantized embeddings

        Example:
            >>> # Ultra-compact for billion-scale search
            >>> result = await layer.embed_quantized(
            ...     texts=documents,
            ...     dtype="binary",
            ...     dimension=256,
            ... )  # 128x storage reduction!
        """
        return await self.embed_with_options(
            texts=texts,
            input_type=input_type,
            model=model,
            output_dimension=dimension,
            output_dtype=dtype,
        )

    # =========================================================================
    # V39.0: Multimodal Embeddings
    # =========================================================================

    async def embed_multimodal(
        self,
        inputs: Sequence[Any],
        input_type: InputType = InputType.DOCUMENT,
        model: str = EmbeddingModel.VOYAGE_MULTIMODAL_3_5.value,
    ) -> EmbeddingResult:
        """
        Embed multimodal inputs (text + images + videos) using voyage-multimodal-3.5.

        Uses the official Voyage AI multimodal embeddings API endpoint:
        POST https://api.voyageai.com/v1/multimodalembeddings

        Supports interleaved text and images for unified embedding space.
        Unlike CLIP, processes text and images together through single backbone.

        API Format (per official docs):
        {
          "inputs": [
            {"content": [{"type": "text", "text": "..."}]},
            {"content": [{"type": "image_url", "image_url": "..."}]}
          ]
        }

        Input Formats:
        - Text: {"type": "text", "text": "content"}
        - Image URL: {"type": "image_url", "image_url": "https://..."}
        - Image Base64: {"type": "image_base64", "image_base64": "data:image/png;base64,..."}
        - Video URL: {"type": "video_url", "video_url": "https://..."}
        - Video Base64: {"type": "video_base64", "video_base64": "data:video/mp4;base64,..."}

        For convenience, also accepts:
        - Plain strings (converted to text type)
        - Lists of content objects

        Constraints:
        - Max 1000 inputs per request
        - Images: ≤16 million pixels, ≤20MB
        - Per input: ≤32,000 tokens (560 pixels = 1 token)
        - Total: ≤320,000 tokens

        Args:
            inputs: List of content objects or strings
            input_type: "document" or "query" for retrieval optimization
            model: Multimodal model (voyage-multimodal-3 or voyage-multimodal-3.5)

        Returns:
            EmbeddingResult with multimodal embeddings

        Example:
            >>> result = await layer.embed_multimodal([
            ...     {"type": "text", "text": "A red sports car"},
            ...     {"type": "image_base64", "image_base64": car_base64},
            ... ])
        """
        if not self._initialized:
            await self.initialize()

        # Multimodal embeddings require httpx for direct API call
        if not HTTPX_AVAILABLE or _httpx_module is None:
            raise SDKNotAvailableError(
                sdk_name="httpx",
                install_cmd="pip install httpx",
                docs_url="https://docs.voyageai.com/docs/multimodal-embeddings"
            )

        # Ensure multimodal model
        if "multimodal" not in model:
            logger.warning(
                "multimodal_model_override",
                requested=model,
                using=EmbeddingModel.VOYAGE_MULTIMODAL_3_5.value
            )
            model = EmbeddingModel.VOYAGE_MULTIMODAL_3_5.value

        # Convert inputs to API format: {"content": [...]} wrapper
        formatted_inputs: list[dict[str, Any]] = []
        for item in inputs:
            if isinstance(item, str):
                # Plain string -> text content wrapped
                formatted_inputs.append({"content": [{"type": "text", "text": item}]})
            elif isinstance(item, dict):
                # Already a content object - wrap it
                formatted_inputs.append({"content": [item]})
            elif isinstance(item, (list, tuple)):
                # List of content objects for a single input
                formatted_inputs.append({"content": list(item)})
            else:
                # Try to handle PIL Image or other types
                try:
                    import base64
                    from io import BytesIO
                    # Attempt to serialize as image
                    buffer = BytesIO()
                    item.save(buffer, format="PNG")
                    img_b64 = base64.b64encode(buffer.getvalue()).decode()
                    formatted_inputs.append({"content": [{
                        "type": "image_base64",
                        "image_base64": f"data:image/png;base64,{img_b64}"
                    }]})
                except Exception:
                    raise ValueError(f"Unsupported input type: {type(item)}")

        start_time = time.time()

        # Build HTTP request
        api_key = self._get_api_key()
        url = "https://api.voyageai.com/v1/multimodalembeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "inputs": formatted_inputs,
            "model": model,
        }

        if input_type:
            payload["input_type"] = input_type.value
        if self.config.truncation:
            payload["truncation"] = True

        # Estimate tokens for rate limiting (rough estimate)
        estimated_tokens = len(formatted_inputs) * 100
        await self._rate_limiter.acquire(tokens=estimated_tokens)

        # V43: Use persistent client with connection pooling
        http_client = self._httpx_client

        def _sync_http_call() -> dict[str, Any]:
            """Sync HTTP call to run in thread pool."""
            # V43: Use persistent connection pool instead of per-request client
            if http_client is not None:
                response = http_client.post(url, headers=headers, json=payload, timeout=120.0)  # Longer for images/videos
                response.raise_for_status()
                return response.json()
            else:
                with _httpx_module.Client(timeout=120.0) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    return response.json()

        last_error: Optional[Exception] = None

        for attempt in range(self.config.max_retries):
            try:
                data = await asyncio.to_thread(_sync_http_call)

                embeddings = [item["embedding"] for item in data.get("data", [])]
                total_tokens = data.get("usage", {}).get("total_tokens", 0)
                dimension = len(embeddings[0]) if embeddings else 1024

                elapsed = time.time() - start_time
                logger.info(
                    "multimodal_embeddings_generated",
                    count=len(embeddings),
                    dimension=dimension,
                    model=model,
                    elapsed_ms=round(elapsed * 1000, 2),
                    api_endpoint="multimodalembeddings",
                )

                return EmbeddingResult(
                    embeddings=embeddings,
                    model=model,
                    input_type=input_type.value if input_type else "document",
                    dimension=dimension,
                    total_tokens=total_tokens,
                )

            except _httpx_module.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    logger.warning("multimodal_rate_limit", attempt=attempt + 1, wait=wait_time)
                    await asyncio.sleep(wait_time)
                    last_error = e
                else:
                    last_error = e
                    if attempt < self.config.max_retries - 1:
                        wait_time = self.config.retry_delay * (attempt + 1)
                        logger.warning("multimodal_retry", attempt=attempt + 1, error=str(e))
                        await asyncio.sleep(wait_time)

            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    wait_time = self.config.retry_delay * (attempt + 1)
                    logger.warning("multimodal_retry", attempt=attempt + 1, error=str(e))
                    await asyncio.sleep(wait_time)

        raise RuntimeError(f"Multimodal embedding failed after {self.config.max_retries} attempts: {last_error}")

    # -------------------------------------------------------------------------
    # V39.0: Contextualized Chunk Embeddings (voyage-context-3)
    # -------------------------------------------------------------------------

    async def embed_contextualized(
        self,
        documents: Sequence[Sequence[str]],
        input_type: InputType = InputType.DOCUMENT,
        output_dimension: Optional[int] = None,
        context_window: int = 1,
    ) -> list[EmbeddingResult]:
        """
        Embed document chunks with inter-chunk context awareness.
        
        Uses the official Voyage AI contextualized embeddings API endpoint:
        POST https://api.voyageai.com/v1/contextualizedembeddings
        
        Unlike regular embeddings, contextualized embeddings understand the
        relationship between chunks in the same document. Each chunk's embedding
        captures not just its own content, but also context from surrounding chunks.
        
        This is CRITICAL for:
        - Long documents split into sections
        - Code files with related functions
        - Conversations with multiple turns
        - Trading strategies with interconnected components
        
        Args:
            documents: List of documents, each document is a list of chunks.
                       Example: [["intro", "body", "conclusion"], ["chapter1", "chapter2"]]
            input_type: "document" or "query"
            output_dimension: Optional dimension (256, 512, 1024, 2048)
            context_window: Number of surrounding chunks for context (ignored by API, kept for compatibility)
        
        Returns:
            List of EmbeddingResults, one per document, each containing
            embeddings for all chunks in that document.
        
        Example:
            >>> # Trading strategy with related sections
            >>> strategy_chunks = [
            ...     "Risk management overview...",
            ...     "Position sizing rules...",
            ...     "Exit conditions...",
            ... ]
            >>> results = await layer.embed_contextualized(
            ...     documents=[strategy_chunks],
            ...     input_type=InputType.DOCUMENT,
            ... )
            >>> # Each chunk embedding understands its role in the strategy
        """
        if not self._initialized:
            await self.initialize()
        
        # Contextualized embeddings require httpx for direct API call
        if not HTTPX_AVAILABLE or _httpx_module is None:
            raise SDKNotAvailableError(
                sdk_name="httpx",
                install_cmd="pip install httpx",
                docs_url="https://docs.voyageai.com/docs/contextualized-chunk-embeddings",
            )
        
        # Official model for contextualized embeddings
        model = EmbeddingModel.VOYAGE_CONTEXT_3.value
        results: list[EmbeddingResult] = []

        # Build HTTP request components
        api_key = self._get_api_key()
        url = "https://api.voyageai.com/v1/contextualizedembeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # V43: Use persistent client with connection pooling
        http_client = self._httpx_client

        for doc_chunks in documents:
            if not doc_chunks:
                continue
            
            start_time = time.time()
            chunks_list = list(doc_chunks)
            
            # Estimate tokens for rate limiting
            estimated_tokens = sum(len(chunk) // 4 for chunk in chunks_list)
            await self._rate_limiter.acquire(tokens=estimated_tokens)
            
            # Build payload per Voyage AI API spec
            # NOTE: Contextualized endpoint does NOT accept input_type or truncation
            payload: dict[str, Any] = {
                "inputs": [chunks_list],  # Array of arrays per API spec
                "model": model,
            }
            
            # Only add optional parameters that the API accepts
            # Note: output_dimension is supported per official docs
            if output_dimension is not None:
                payload["output_dimension"] = output_dimension
            # Note: truncation is NOT supported by contextualized endpoint

            def _sync_http_call() -> dict[str, Any]:
                """Sync HTTP call to run in thread pool."""
                # V43: Use persistent connection pool instead of per-request client
                if http_client is not None:
                    response = http_client.post(url, headers=headers, json=payload, timeout=60.0)
                    response.raise_for_status()
                    return response.json()
                else:
                    with _httpx_module.Client(timeout=60.0) as client:
                        response = client.post(url, headers=headers, json=payload)
                        response.raise_for_status()
                        return response.json()

            last_error: Optional[Exception] = None
            
            for attempt in range(self.config.max_retries):
                try:
                    data = await asyncio.to_thread(_sync_http_call)
                    
                    # API returns nested structure: data[0].data[i].embedding
                    # Structure: {"data": [{"data": [{"embedding": [...], "index": 0}, ...]}]}
                    doc_data = data.get("data", [])
                    embeddings: list[list[float]] = []
                    if doc_data and len(doc_data) > 0:
                        # Each document has a nested "data" array containing chunk embeddings
                        chunk_data = doc_data[0].get("data", [])
                        for chunk in chunk_data:
                            if isinstance(chunk, dict) and "embedding" in chunk:
                                embeddings.append(chunk["embedding"])
                    
                    total_tokens = data.get("usage", {}).get("total_tokens", 0)
                    dimension = len(embeddings[0]) if embeddings else 1024
                    
                    elapsed = time.time() - start_time
                    logger.info(
                        "contextualized_embeddings_generated",
                        chunks=len(embeddings),
                        dimension=dimension,
                        elapsed_ms=round(elapsed * 1000, 2),
                        model=model,
                        api_endpoint="contextualizedembeddings",
                    )
                    
                    results.append(EmbeddingResult(
                        embeddings=embeddings,
                        model=model,
                        input_type=input_type.value if input_type else "document",
                        dimension=dimension,
                        total_tokens=total_tokens,
                    ))
                    break  # Success, exit retry loop
                    
                except _httpx_module.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        wait_time = self.config.retry_delay * (2 ** attempt)
                        logger.warning("contextualized_rate_limit", attempt=attempt + 1, wait=wait_time)
                        await asyncio.sleep(wait_time)
                        last_error = e
                    else:
                        last_error = e
                        if attempt < self.config.max_retries - 1:
                            wait_time = self.config.retry_delay * (attempt + 1)
                            logger.warning("contextualized_retry", attempt=attempt + 1, error=str(e))
                            await asyncio.sleep(wait_time)
                        
                except Exception as e:
                    last_error = e
                    if attempt < self.config.max_retries - 1:
                        wait_time = self.config.retry_delay * (attempt + 1)
                        logger.warning("contextualized_retry", attempt=attempt + 1, error=str(e))
                        await asyncio.sleep(wait_time)
            else:
                # All retries exhausted
                raise RuntimeError(f"Contextualized embedding failed after {self.config.max_retries} attempts: {last_error}")
        
        return results

    # -------------------------------------------------------------------------
    # V39.0: Matryoshka Embedding Utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def truncate_matryoshka(
        embeddings: list[list[float]],
        target_dimension: int,
    ) -> list[list[float]]:
        """
        Truncate Matryoshka embeddings to a smaller dimension.
        
        Voyage 4 series embeddings are trained with Matryoshka learning,
        meaning the first N dimensions form a valid N-dimensional embedding.
        This allows post-hoc dimension reduction without re-embedding.
        
        Storage savings:
        - 2048d → 1024d: 2x savings
        - 2048d → 512d:  4x savings
        - 2048d → 256d:  8x savings
        
        Args:
            embeddings: Full-dimension embeddings
            target_dimension: Desired dimension (256, 512, 1024)
        
        Returns:
            Truncated embeddings
            
        Example:
            >>> # Embed once at full resolution
            >>> result = await layer.embed(texts, output_dimension=2048)
            >>> 
            >>> # Store different resolutions for different use cases
            >>> high_res = result.embeddings  # 2048d for archival
            >>> search_res = layer.truncate_matryoshka(result.embeddings, 1024)
            >>> compact = layer.truncate_matryoshka(result.embeddings, 256)
        """
        if not embeddings:
            return []
        
        current_dim = len(embeddings[0])
        if target_dimension >= current_dim:
            return embeddings
        
        # Matryoshka: just take first N dimensions
        return [emb[:target_dimension] for emb in embeddings]

    @staticmethod
    def normalize_embeddings(
        embeddings: list[list[float]],
    ) -> list[list[float]]:
        """
        L2-normalize embeddings to unit vectors.
        
        Required after Matryoshka truncation to maintain cosine similarity quality.
        Voyage embeddings are normalized at creation, but truncation breaks this.
        
        Args:
            embeddings: Embeddings to normalize
            
        Returns:
            L2-normalized embeddings
        """
        import math
        
        normalized = []
        for emb in embeddings:
            magnitude = math.sqrt(sum(x * x for x in emb))
            if magnitude > 0:
                normalized.append([x / magnitude for x in emb])
            else:
                normalized.append(emb)
        
        return normalized

    # =========================================================================
    # V39.0: Intelligent Chunking for Long Documents
    # =========================================================================

    async def embed_with_chunking(
        self,
        texts: Sequence[str],
        chunk_size: int = 8000,
        chunk_overlap: int = 200,
        aggregation: str = "mean",
        input_type: InputType = InputType.DOCUMENT,
        model: Optional[str] = None,
    ) -> EmbeddingResult:
        """
        Embed long documents with intelligent chunking and aggregation.

        Splits long texts into overlapping chunks, embeds each, and aggregates
        into a single embedding per document. Preserves semantic continuity.

        Strategies:
        - mean: Average of chunk embeddings (default, most stable)
        - first: Use first chunk only (fastest, for title-heavy docs)
        - max: Max-pool across chunks (captures strongest signals)
        - weighted: More weight to early chunks (for structured docs)

        Args:
            texts: Long documents to embed
            chunk_size: Characters per chunk (default 8000 ≈ 2000 tokens)
            chunk_overlap: Overlap between chunks (default 200 chars)
            aggregation: How to combine chunks: "mean", "first", "max", "weighted"
            input_type: "document" or "query"
            model: Override default model

        Returns:
            EmbeddingResult with one embedding per input document

        Example:
            >>> # Long legal documents
            >>> result = await layer.embed_with_chunking(
            ...     texts=long_contracts,
            ...     chunk_size=10000,
            ...     aggregation="weighted",  # Prioritize early sections
            ... )
        """
        if not self._initialized:
            await self.initialize()

        all_embeddings = []
        total_tokens = 0

        for text in texts:
            if len(text) <= chunk_size:
                # Short text, embed directly
                result = await self.embed([text], input_type=input_type, model=model)
                all_embeddings.append(result.embeddings[0])
                total_tokens += result.total_tokens
            else:
                # Split into chunks
                chunks = self._split_into_chunks(text, chunk_size, chunk_overlap)

                # Embed all chunks
                chunk_result = await self.embed(chunks, input_type=input_type, model=model)
                chunk_embeddings = chunk_result.embeddings
                total_tokens += chunk_result.total_tokens

                # Aggregate based on strategy
                aggregated = self._aggregate_embeddings(chunk_embeddings, aggregation)
                all_embeddings.append(aggregated)

        model_used = model or self.config.model

        return EmbeddingResult(
            embeddings=all_embeddings,
            model=model_used,
            input_type=input_type.value if input_type else "document",
            dimension=len(all_embeddings[0]) if all_embeddings else 1024,
            total_tokens=total_tokens,
        )

    def _split_into_chunks(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end near chunk boundary
                for sep in [". ", ".\n", "! ", "? ", "\n\n"]:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > chunk_size * 0.7:  # Only if reasonably far in
                        end = start + last_sep + len(sep)
                        break

            chunks.append(text[start:end].strip())
            start = end - overlap

        return [c for c in chunks if c]  # Filter empty

    def _aggregate_embeddings(
        self,
        embeddings: list[list[float]],
        method: str,
    ) -> list[float]:
        """Aggregate multiple chunk embeddings into one."""
        if not embeddings:
            return []

        if method == "first":
            return embeddings[0]

        elif method == "max":
            # Max pooling across dimensions
            result = []
            for i in range(len(embeddings[0])):
                result.append(max(e[i] for e in embeddings))
            return result

        elif method == "weighted":
            # Weighted average (more weight to early chunks)
            weights = [1.0 / (i + 1) for i in range(len(embeddings))]
            total_weight = sum(weights)
            result = [0.0] * len(embeddings[0])
            for i, emb in enumerate(embeddings):
                w = weights[i] / total_weight
                for j in range(len(emb)):
                    result[j] += emb[j] * w
            return result

        else:  # "mean" default
            result = [0.0] * len(embeddings[0])
            for emb in embeddings:
                for j in range(len(emb)):
                    result[j] += emb[j]
            return [x / len(embeddings) for x in result]

    # =========================================================================
    # V39.0: Batch Optimization for Large-Scale Indexing
    # =========================================================================

    async def embed_batch_optimized(
        self,
        texts: Sequence[str],
        target_batch_tokens: int = 50000,
        max_concurrent: int = 5,
        input_type: InputType = InputType.DOCUMENT,
        model: Optional[str] = None,
        progress_callback: Optional[Any] = None,
    ) -> EmbeddingResult:
        """
        Embed large collections with token-aware batching and concurrency.

        Optimizes for throughput by:
        1. Estimating tokens per text
        2. Creating batches that approach token limits
        3. Processing multiple batches concurrently
        4. Respecting rate limits

        Args:
            texts: Large collection to embed
            target_batch_tokens: Token budget per batch (default 50K, conservative)
            max_concurrent: Max concurrent API calls (default 5)
            input_type: "document" or "query"
            model: Override default model
            progress_callback: Optional callback(completed, total) for progress

        Returns:
            EmbeddingResult with all embeddings

        Example:
            >>> # Index 1 million documents
            >>> result = await layer.embed_batch_optimized(
            ...     texts=million_docs,
            ...     target_batch_tokens=100000,
            ...     max_concurrent=10,
            ...     progress_callback=lambda done, total: print(f"{done}/{total}"),
            ... )
        """
        if not self._initialized:
            await self.initialize()

        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model=model or self.config.model,
                input_type=input_type.value if input_type else "document",
                dimension=1024,
                total_tokens=0,
            )

        start_time = time.time()

        # Create token-optimized batches
        batches = self._create_token_optimized_batches(texts, target_batch_tokens)
        total_batches = len(batches)

        logger.info(
            "batch_embed_starting",
            total_texts=len(texts),
            num_batches=total_batches,
            max_concurrent=max_concurrent,
        )

        all_embeddings: list[list[float]] = []
        total_tokens = 0
        completed = 0

        # Process batches with controlled concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_batch(batch: list[str], batch_idx: int) -> EmbeddingResult:
            async with semaphore:
                result = await self.embed(batch, input_type=input_type, model=model)
                return result

        # Create tasks for all batches
        tasks = [
            process_batch(batch, i)
            for i, batch in enumerate(batches)
        ]

        # Gather results in order
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("batch_embed_error", batch_idx=i, error=str(result))
                raise result

            # Type narrowing: after exception check, result is EmbeddingResult
            embed_result: EmbeddingResult = result  # type: ignore[assignment]
            all_embeddings.extend(embed_result.embeddings)
            total_tokens += embed_result.total_tokens
            completed += 1

            if progress_callback:
                try:
                    progress_callback(completed, total_batches)
                except Exception:
                    pass

        elapsed = time.time() - start_time
        model_used = model or self.config.model

        logger.info(
            "batch_embed_completed",
            total_texts=len(texts),
            total_tokens=total_tokens,
            elapsed_s=round(elapsed, 2),
            texts_per_second=round(len(texts) / elapsed, 2) if elapsed > 0 else 0,
        )

        return EmbeddingResult(
            embeddings=all_embeddings,
            model=model_used,
            input_type=input_type.value if input_type else "document",
            dimension=len(all_embeddings[0]) if all_embeddings else 1024,
            total_tokens=total_tokens,
        )

    def _create_token_optimized_batches(
        self,
        texts: Sequence[str],
        target_tokens: int,
    ) -> list[list[str]]:
        """Create batches optimized for token budget."""
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0

        for text in texts:
            # Estimate tokens (rough: 1 token ≈ 4 chars for English)
            estimated_tokens = len(text) // 4 + 1

            if current_tokens + estimated_tokens > target_tokens and current_batch:
                # Start new batch
                batches.append(current_batch)
                current_batch = [text]
                current_tokens = estimated_tokens
            else:
                current_batch.append(text)
                current_tokens += estimated_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    # =========================================================================
    # V39.5: STREAMING & PERFORMANCE OPTIMIZATIONS
    # =========================================================================

    async def embed_stream(
        self,
        texts: Sequence[str],
        batch_size: int = 50,
        input_type: InputType = InputType.DOCUMENT,
        model: Optional[str] = None,
    ) -> AsyncGenerator[tuple[int, list[float]], None]:
        """
        Stream embeddings as they are computed - yields without waiting for all.

        Useful for large collections where you want to:
        1. Start processing results before all embeddings are ready
        2. Avoid memory spikes from holding all embeddings at once
        3. Show real-time progress to users

        Args:
            texts: Sequence of texts to embed
            batch_size: Number of texts per API call (default 50)
            input_type: Type hint for Voyage AI optimization
            model: Override default model

        Yields:
            Tuple of (original_index, embedding_vector) in order

        Example:
            >>> async for idx, embedding in layer.embed_stream(large_corpus):
            ...     # Process immediately without waiting for all
            ...     await store_embedding(idx, embedding)
            ...     print(f"Processed {idx + 1}/{len(large_corpus)}")
        """
        if not self._initialized:
            await self.initialize()

        if not texts:
            return

        # Process in batches, yielding each result immediately
        texts_list = list(texts)
        for batch_start in range(0, len(texts_list), batch_size):
            batch_end = min(batch_start + batch_size, len(texts_list))
            batch = texts_list[batch_start:batch_end]

            # Get embeddings for this batch
            result = await self.embed(batch, input_type=input_type, model=model)

            # Yield each embedding with its original index
            for i, embedding in enumerate(result.embeddings):
                yield (batch_start + i, embedding)

    async def embed_batch_streaming(
        self,
        texts: Sequence[str],
        target_batch_tokens: int = 50000,
        max_concurrent: int = 5,
        input_type: InputType = InputType.DOCUMENT,
        model: Optional[str] = None,
        on_batch_complete: Optional[Callable[[int, int, list[list[float]]], None]] = None,
    ) -> EmbeddingResult:
        """
        Progressive batch processing with streaming callback.

        Like embed_batch_optimized but calls a callback after each batch,
        enabling progressive processing without waiting for all batches.

        Args:
            texts: Large collection to embed
            target_batch_tokens: Token budget per batch
            max_concurrent: Max concurrent API calls
            input_type: Type hint for optimization
            model: Override default model
            on_batch_complete: Callback(batch_idx, total_batches, batch_embeddings)

        Returns:
            EmbeddingResult with all embeddings (also available via callback)

        Example:
            >>> embeddings_so_far = []
            >>> def on_batch(idx, total, batch_embs):
            ...     embeddings_so_far.extend(batch_embs)
            ...     print(f"Batch {idx}/{total}: {len(batch_embs)} embeddings")
            ...
            >>> result = await layer.embed_batch_streaming(
            ...     texts=million_docs,
            ...     on_batch_complete=on_batch,
            ... )
        """
        if not self._initialized:
            await self.initialize()

        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model=model or self.config.model,
                input_type=input_type.value if input_type else "document",
                dimension=1024,
                total_tokens=0,
            )

        start_time = time.time()
        batches = self._create_token_optimized_batches(texts, target_batch_tokens)
        total_batches = len(batches)

        all_embeddings: list[list[float]] = []
        total_tokens = 0

        # Process with concurrency but call callback for each batch as it completes
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_batch(batch: list[str], batch_idx: int) -> tuple[int, EmbeddingResult]:
            async with semaphore:
                result = await self.embed(batch, input_type=input_type, model=model)
                return (batch_idx, result)

        # Use asyncio.as_completed for progressive results
        tasks = [
            asyncio.create_task(process_batch(batch, i))
            for i, batch in enumerate(batches)
        ]

        # Results may arrive out of order, so we collect and sort
        batch_results: dict[int, EmbeddingResult] = {}

        for coro in asyncio.as_completed(tasks):
            batch_idx, result = await coro
            batch_results[batch_idx] = result

            # Call streaming callback if provided
            if on_batch_complete:
                try:
                    on_batch_complete(batch_idx, total_batches, result.embeddings)
                except Exception:
                    pass

        # Reconstruct in order
        for i in range(total_batches):
            result = batch_results[i]
            all_embeddings.extend(result.embeddings)
            total_tokens += result.total_tokens

        elapsed = time.time() - start_time

        logger.info(
            "batch_embed_streaming_completed",
            total_texts=len(texts),
            total_tokens=total_tokens,
            elapsed_s=round(elapsed, 2),
        )

        return EmbeddingResult(
            embeddings=all_embeddings,
            model=model or self.config.model,
            input_type=input_type.value if input_type else "document",
            dimension=len(all_embeddings[0]) if all_embeddings else 1024,
            total_tokens=total_tokens,
        )

    def analyze_query_characteristics(self, query: str) -> dict[str, Any]:
        """
        Analyze query to determine optimal search parameters.

        Used by adaptive_hybrid_search to auto-tune alpha.

        Returns:
            dict with: keyword_density, avg_word_length, has_special_terms,
                      semantic_complexity, recommended_alpha
        """
        words = query.lower().split()
        word_count = len(words) if words else 1

        # Count keyword-like patterns
        keyword_patterns = sum(1 for w in words if len(w) <= 4 or w.isalpha())

        # Detect semantic phrases (longer descriptive terms)
        semantic_words = sum(1 for w in words if len(w) > 6)

        # Special terms that suggest keyword search
        special_terms = {"api", "sdk", "v2", "v3", "http", "sql", "css", "html", "json"}
        has_special = any(w in special_terms for w in words)

        # Code-like patterns suggest keyword matching
        has_code_patterns = any(
            c in query for c in ["()", "[]", "{}", ".", "_", "->", "::"]
        )

        # Calculate metrics
        keyword_density = keyword_patterns / word_count
        avg_word_length = sum(len(w) for w in words) / word_count if words else 0
        semantic_complexity = semantic_words / word_count

        # Recommend alpha based on analysis
        # Higher alpha = more semantic, Lower alpha = more keyword
        if has_code_patterns or has_special:
            recommended_alpha = 0.3  # Favor keyword matching
        elif semantic_complexity > 0.5:
            recommended_alpha = 0.8  # Favor semantic matching
        elif keyword_density > 0.7:
            recommended_alpha = 0.4  # Slightly favor keywords
        else:
            recommended_alpha = 0.6  # Balanced with slight semantic bias

        return {
            "keyword_density": round(keyword_density, 3),
            "avg_word_length": round(avg_word_length, 2),
            "has_special_terms": has_special,
            "has_code_patterns": has_code_patterns,
            "semantic_complexity": round(semantic_complexity, 3),
            "recommended_alpha": recommended_alpha,
        }

    async def adaptive_hybrid_search(
        self,
        query: str,
        documents: list[str],
        doc_embeddings: Optional[list[list[float]]] = None,
        top_k: int = 5,
        alpha_override: Optional[float] = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> tuple[list[tuple[int, float, str]], float]:
        """
        Hybrid search with automatic alpha tuning based on query analysis.

        Analyzes the query to determine optimal balance between semantic
        and keyword matching, then performs hybrid search.

        Args:
            query: Search query
            documents: List of document texts
            doc_embeddings: Pre-computed embeddings (optional)
            top_k: Number of results
            alpha_override: Manual override for alpha (skips auto-tuning)
            bm25_k1: BM25 term frequency saturation
            bm25_b: BM25 length normalization

        Returns:
            Tuple of (results, alpha_used) where results is
            list of (index, score, document) and alpha_used is the
            automatically determined or overridden alpha value.

        Example:
            >>> # Let the system choose optimal alpha
            >>> results, alpha = await layer.adaptive_hybrid_search(
            ...     query="async python caching patterns",
            ...     documents=docs,
            ... )
            >>> print(f"Used alpha={alpha:.2f} for this query")
        """
        if alpha_override is not None:
            alpha = alpha_override
        else:
            analysis = self.analyze_query_characteristics(query)
            alpha = analysis["recommended_alpha"]

            logger.debug(
                "adaptive_alpha_computed",
                query_preview=query[:50],
                analysis=analysis,
                alpha=alpha,
            )

        # V39.11: Start timing for tracing
        start_time = time.perf_counter()

        results = await self.hybrid_search(
            query=query,
            documents=documents,
            doc_embeddings=doc_embeddings,
            top_k=top_k,
            alpha=alpha,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
        )

        # V39.11: Log adaptive search tracing
        latency_ms = (time.perf_counter() - start_time) * 1000
        self._trace_search_operation(
            operation_name="adaptive_hybrid_search",
            query=query,
            num_docs=len(documents),
            num_results=len(results),
            latency_ms=latency_ms,
            extra_metadata={
                "alpha_auto": alpha_override is None,
                "alpha_used": alpha,
            },
        )

        return (results, alpha)

    async def prefetch_cache(
        self,
        recent_queries: list[str],
        candidate_texts: list[str],
        similarity_threshold: float = 0.7,
        max_prefetch: int = 50,
        input_type: InputType = InputType.DOCUMENT,
        model: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Predictive cache warming based on query patterns.

        Analyzes recent queries and prefetches embeddings for texts
        likely to be queried next (based on similarity to recent patterns).

        Args:
            recent_queries: List of recent query strings
            candidate_texts: Texts to consider for prefetching
            similarity_threshold: Minimum similarity to prefetch (0-1)
            max_prefetch: Maximum number of texts to prefetch
            input_type: Type for candidate embeddings
            model: Model to use

        Returns:
            Dict with prefetched count, skipped count, candidates analyzed

        Example:
            >>> # Prefetch based on user's recent search patterns
            >>> result = await layer.prefetch_cache(
            ...     recent_queries=user_recent_queries[-10:],
            ...     candidate_texts=popular_documents[:100],
            ...     similarity_threshold=0.6,
            ... )
            >>> print(f"Prefetched {result['prefetched']} likely queries")
        """
        if not self._initialized:
            await self.initialize()

        if not recent_queries or not candidate_texts:
            return {"prefetched": 0, "skipped": 0, "analyzed": 0}

        # Embed recent queries to understand query patterns
        query_result = await self.embed(
            recent_queries,
            input_type=InputType.QUERY,
            model=model,
        )

        # Compute average query embedding (centroid)
        query_centroid = [
            sum(embs) / len(embs)
            for embs in zip(*query_result.embeddings)
        ]
        # Normalize centroid
        centroid_norm = sum(x**2 for x in query_centroid) ** 0.5
        if centroid_norm > 0:
            query_centroid = [x / centroid_norm for x in query_centroid]

        # Check which candidates are already cached
        uncached_candidates = []
        for text in candidate_texts:
            cache_key = self._get_cache_key(text, model or self.config.model, input_type)
            if cache_key not in self._cache:
                uncached_candidates.append(text)

        if not uncached_candidates:
            return {
                "prefetched": 0,
                "skipped": len(candidate_texts),
                "analyzed": len(candidate_texts),
                "reason": "all_already_cached",
            }

        # Limit to max_prefetch uncached candidates
        candidates_to_score = uncached_candidates[:max_prefetch * 2]

        # Embed candidates and compute similarity to query centroid
        candidate_result = await self.embed(
            candidates_to_score,
            input_type=input_type,
            model=model,
        )

        # Score by similarity to query patterns
        scored = []
        for i, emb in enumerate(candidate_result.embeddings):
            sim = self.cosine_similarity(query_centroid, emb)
            if sim >= similarity_threshold:
                scored.append((candidates_to_score[i], sim))

        # Sort by similarity and take top
        scored.sort(key=lambda x: x[1], reverse=True)
        to_prefetch = [text for text, _ in scored[:max_prefetch]]

        # These are now in cache from the embed call above
        prefetched_count = len(to_prefetch)

        logger.info(
            "prefetch_cache_completed",
            total_candidates=len(candidate_texts),
            analyzed=len(candidates_to_score),
            above_threshold=len(scored),
            prefetched=prefetched_count,
        )

        return {
            "prefetched": prefetched_count,
            "skipped": len(candidate_texts) - len(candidates_to_score),
            "analyzed": len(candidates_to_score),
            "above_threshold": len(scored),
            "similarity_threshold": similarity_threshold,
        }

    # =========================================================================
    # V39.6: Multi-Person & Temporal Embedding Methods
    # =========================================================================

    async def embed_multi_pose(
        self,
        poses: list[dict],
        include_velocity: bool = True,
        max_performers: int = 8,
        identity_cache: Optional["IdentityCache"] = None,
        model: Optional[EmbeddingModel] = None,
    ) -> dict[str, list[float]]:
        """
        Embed multiple simultaneous poses with identity preservation.

        V39.6 Feature: Multi-person tracking support for State of Witness.
        Each pose is serialized to text description and embedded in a single batch call.

        Args:
            poses: List of pose dictionaries, each containing:
                - "identity": str - Performer identifier
                - "keypoints": list[float] - 33 pose keypoints (x, y, z per joint)
                - "confidence": float - Detection confidence (0.0-1.0)
            include_velocity: If True and identity_cache provided, include velocity in text
            max_performers: Maximum number of performers to process (capped at 8)
            identity_cache: Optional IdentityCache for velocity calculation
            model: Embedding model (defaults to config model)

        Returns:
            Dict mapping identity → embedding (1024d for voyage-4, 512d for lite)

        Example:
            result = await layer.embed_multi_pose([
                {"identity": "performer_1", "keypoints": [...], "confidence": 0.95},
                {"identity": "performer_2", "keypoints": [...], "confidence": 0.87},
            ])
            # result = {"performer_1": [0.12, ...], "performer_2": [0.34, ...]}
        """
        if not poses:
            return {}

        # Cap at max performers
        poses = poses[:max_performers]

        # Build text descriptions for each pose
        texts = []
        identities = []

        for pose in poses:
            identity = pose.get("identity", f"unknown_{len(identities)}")
            keypoints = pose.get("keypoints", [])
            confidence = pose.get("confidence", 1.0)

            # Skip low-confidence poses
            if confidence < 0.5:
                continue

            # Serialize keypoints to descriptive text
            text_parts = [f"Pose for {identity}:"]

            # Extract key joint positions (simplified for embedding)
            if len(keypoints) >= 99:  # 33 joints × 3 coords
                # Head position (nose - joint 0)
                head_x, head_y, head_z = keypoints[0:3]
                text_parts.append(f"head at ({head_x:.2f}, {head_y:.2f}, {head_z:.2f})")

                # Shoulder positions (joints 11, 12)
                l_shoulder = keypoints[33:36]  # joint 11
                r_shoulder = keypoints[36:39]  # joint 12
                text_parts.append(f"shoulders at L({l_shoulder[0]:.2f}, {l_shoulder[1]:.2f}) R({r_shoulder[0]:.2f}, {r_shoulder[1]:.2f})")

                # Hip positions (joints 23, 24)
                l_hip = keypoints[69:72]  # joint 23
                r_hip = keypoints[72:75]  # joint 24
                text_parts.append(f"hips at L({l_hip[0]:.2f}, {l_hip[1]:.2f}) R({r_hip[0]:.2f}, {r_hip[1]:.2f})")

                # Wrist positions (joints 15, 16)
                l_wrist = keypoints[45:48]  # joint 15
                r_wrist = keypoints[48:51]  # joint 16
                text_parts.append(f"wrists at L({l_wrist[0]:.2f}, {l_wrist[1]:.2f}) R({r_wrist[0]:.2f}, {r_wrist[1]:.2f})")

            # Add velocity if cache available
            if include_velocity and identity_cache:
                velocity = identity_cache.get_velocity(identity)
                if velocity:
                    vel_mag = sum(v**2 for v in velocity) ** 0.5
                    text_parts.append(f"moving with velocity {vel_mag:.2f}")

            text_parts.append(f"confidence {confidence:.2f}")
            texts.append(" ".join(text_parts))
            identities.append(identity)

        if not texts:
            return {}

        # Single batch API call for all poses
        result = await self.embed(
            texts=texts,
            input_type=InputType.DOCUMENT,
            model=model or self.config.model,
        )

        # Map identities to embeddings
        identity_embeddings = {}
        for i, identity in enumerate(identities):
            identity_embeddings[identity] = result.embeddings[i]

            # Update cache if provided
            if identity_cache:
                pose = poses[i] if i < len(poses) else {}
                identity_cache.update(
                    identity=identity,
                    embedding=result.embeddings[i],
                    keypoints=pose.get("keypoints", []),
                    confidence=pose.get("confidence", 1.0),
                )

        logger.info(
            "embed_multi_pose_completed",
            num_poses=len(poses),
            num_embedded=len(identity_embeddings),
            identities=list(identity_embeddings.keys()),
        )

        return identity_embeddings

    async def embed_pose_sequence(
        self,
        sequence: list[list[float]],
        window_size: int = 30,
        aggregation: str = "mean",
        model: Optional[EmbeddingModel] = None,
    ) -> list[float]:
        """
        Embed a temporal sequence of poses as a single gesture vector.

        V39.6 Feature: Temporal gesture recognition for State of Witness.
        Supports rolling window for streaming gesture detection.

        Args:
            sequence: List of pose keypoints over time (each element is 99 floats for 33 joints)
            window_size: Number of frames to consider (default 30 = ~1 second at 30fps)
            aggregation: How to combine frame embeddings:
                - "mean": Average all frame embeddings (stable, smooth)
                - "attention": Weight by motion magnitude (emphasizes key moments)
                - "last": Weighted average favoring recent frames (responsive)
            model: Embedding model to use

        Returns:
            Single embedding vector representing the motion sequence

        Example:
            gesture_embedding = await layer.embed_pose_sequence(
                sequence=pose_buffer[-30:],
                aggregation="attention",
            )
        """
        if not sequence:
            raise ValueError("Sequence cannot be empty")

        # Truncate to window size
        sequence = sequence[-window_size:]

        if len(sequence) < 2:
            # Single frame - just embed it directly
            text = self._pose_to_text(sequence[0])
            result = await self.embed(
                texts=[text],
                input_type=InputType.DOCUMENT,
                model=model or self.config.model,
            )
            return result.embeddings[0]

        # Build text descriptions for each frame
        texts = []
        motion_magnitudes = []

        for i, keypoints in enumerate(sequence):
            # Build frame description
            text = self._pose_to_text(keypoints)

            # Calculate motion from previous frame
            if i > 0:
                prev = sequence[i - 1]
                if len(keypoints) == len(prev):
                    motion = sum((a - b) ** 2 for a, b in zip(keypoints, prev)) ** 0.5
                    motion_magnitudes.append(motion)
                    text += f" motion={motion:.3f}"
                else:
                    motion_magnitudes.append(0.0)
            else:
                motion_magnitudes.append(0.0)

            texts.append(text)

        # Embed all frames in single batch
        result = await self.embed(
            texts=texts,
            input_type=InputType.DOCUMENT,
            model=model or self.config.model,
        )

        embeddings = result.embeddings
        dim = len(embeddings[0])

        # Aggregate embeddings based on strategy
        if aggregation == "mean":
            # Simple average of all frames
            aggregated = [
                sum(emb[d] for emb in embeddings) / len(embeddings)
                for d in range(dim)
            ]

        elif aggregation == "attention":
            # Weight by motion magnitude (normalized)
            total_motion = sum(motion_magnitudes) or 1.0
            weights = [m / total_motion for m in motion_magnitudes]

            aggregated = [0.0] * dim
            for i, emb in enumerate(embeddings):
                for d in range(dim):
                    aggregated[d] += emb[d] * weights[i]

        elif aggregation == "last":
            # Exponential weighting toward recent frames
            decay = 0.9
            weights = [decay ** (len(embeddings) - 1 - i) for i in range(len(embeddings))]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            aggregated = [0.0] * dim
            for i, emb in enumerate(embeddings):
                for d in range(dim):
                    aggregated[d] += emb[d] * weights[i]

        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        # Normalize the result
        norm = sum(x**2 for x in aggregated) ** 0.5
        if norm > 0:
            aggregated = [x / norm for x in aggregated]

        logger.info(
            "embed_pose_sequence_completed",
            sequence_length=len(sequence),
            window_size=window_size,
            aggregation=aggregation,
        )

        return aggregated

    def _pose_to_text(self, keypoints: list[float]) -> str:
        """Convert pose keypoints to text description for embedding."""
        if not keypoints:
            return "Empty pose"

        parts = ["Pose:"]

        if len(keypoints) >= 99:
            # Head (nose - joint 0)
            parts.append(f"head({keypoints[0]:.2f},{keypoints[1]:.2f},{keypoints[2]:.2f})")
            # Left shoulder (joint 11)
            parts.append(f"lshoulder({keypoints[33]:.2f},{keypoints[34]:.2f})")
            # Right shoulder (joint 12)
            parts.append(f"rshoulder({keypoints[36]:.2f},{keypoints[37]:.2f})")
            # Left hip (joint 23)
            parts.append(f"lhip({keypoints[69]:.2f},{keypoints[70]:.2f})")
            # Right hip (joint 24)
            parts.append(f"rhip({keypoints[72]:.2f},{keypoints[73]:.2f})")
            # Left wrist (joint 15)
            parts.append(f"lwrist({keypoints[45]:.2f},{keypoints[46]:.2f})")
            # Right wrist (joint 16)
            parts.append(f"rwrist({keypoints[48]:.2f},{keypoints[49]:.2f})")
        elif len(keypoints) >= 6:
            # Minimal representation
            parts.append(f"p({keypoints[0]:.2f},{keypoints[1]:.2f},{keypoints[2]:.2f})")

        return " ".join(parts)

    async def embed_pose_with_emotion(
        self,
        pose_keypoints: list[float],
        face_emotion: dict,
        fusion_weight: float = 0.3,
        model: Optional[EmbeddingModel] = None,
    ) -> list[float]:
        """
        Combine body pose with facial emotion for richer archetype matching.

        V39.6 Feature: Emotion-pose fusion for State of Witness.
        Creates composite embedding that captures both physical posture and emotional state.

        Args:
            pose_keypoints: 99 floats (33 joints × 3 coords) for body pose
            face_emotion: Dictionary with:
                - "emotion": str - Detected emotion (joy, sadness, anger, fear, surprise, neutral)
                - "confidence": float - Detection confidence (0.0-1.0)
                - "landmarks": list[float] - Optional face landmarks
            fusion_weight: How much emotion affects final embedding (0.0-1.0, default 0.3)
                - 0.0 = pure pose embedding
                - 1.0 = pure emotion embedding
            model: Embedding model to use

        Returns:
            Fused embedding vector (1024d or 512d depending on model)

        Example:
            embedding = await layer.embed_pose_with_emotion(
                pose_keypoints=body_pose,
                face_emotion={"emotion": "joy", "confidence": 0.9},
                fusion_weight=0.4,
            )
        """
        # Build pose text
        pose_text = self._pose_to_text(pose_keypoints)

        # Build emotion text
        emotion = face_emotion.get("emotion", "neutral")
        emotion_confidence = face_emotion.get("confidence", 0.5)
        emotion_text = f"Facial expression: {emotion} (confidence {emotion_confidence:.2f})"

        # Get separate embeddings for pose and emotion
        result = await self.embed(
            texts=[pose_text, emotion_text],
            input_type=InputType.DOCUMENT,
            model=model or self.config.model,
        )

        pose_emb = result.embeddings[0]
        emotion_emb = result.embeddings[1]

        # Weighted fusion
        pose_weight = 1.0 - fusion_weight
        dim = len(pose_emb)

        fused = [
            pose_emb[d] * pose_weight + emotion_emb[d] * fusion_weight
            for d in range(dim)
        ]

        # Normalize
        norm = sum(x**2 for x in fused) ** 0.5
        if norm > 0:
            fused = [x / norm for x in fused]

        logger.info(
            "embed_pose_with_emotion_completed",
            emotion=emotion,
            emotion_confidence=emotion_confidence,
            fusion_weight=fusion_weight,
        )

        return fused

    async def save_session_state(
        self,
        filepath: str,
        identity_cache: Optional["IdentityCache"] = None,
        gesture_library: Optional["GestureEmbeddingLibrary"] = None,
        include_cache: bool = True,
        incremental: bool = False,
    ) -> dict:
        """
        Save complete embedding session state for recording/playback.

        V39.6 Feature: Session persistence for State of Witness.
        Enables recording/playback of performer tracking sessions.

        Args:
            filepath: Path to save session state (JSON format)
            identity_cache: IdentityCache to serialize
            gesture_library: GestureEmbeddingLibrary to serialize
            include_cache: Whether to include embedding cache (can be large)
            incremental: If True, append to existing file; if False, overwrite

        Returns:
            Dict with save metadata: filepath, entries count, size in MB

        Example:
            result = await layer.save_session_state(
                "session_2026_01_25.json",
                identity_cache=cache,
                gesture_library=gestures,
            )
            # {"saved": "session_2026_01_25.json", "entries": 245, "size_mb": 1.2}
        """
        import json
        import os
        from datetime import datetime, timezone

        session_data = {
            "version": "39.6",
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "model": self.config.model.value if hasattr(self.config.model, 'value') else str(self.config.model) if self.config.model else None,
                "cache_enabled": self.config.cache_enabled,
                "cache_size": self.config.cache_size,
            },
        }

        entries = 0

        # Save identity cache
        if identity_cache:
            session_data["identity_cache"] = identity_cache.to_dict()
            entries += len(identity_cache._histories)

        # Save gesture library
        if gesture_library:
            session_data["gesture_library"] = gesture_library.to_dict()
            entries += len(gesture_library._embeddings)

        # Optionally save embedding cache
        if include_cache:
            # Convert cache to serializable format
            cache_data = {}
            for key, embedding in self._cache.items():
                cache_data[key] = embedding
            session_data["embedding_cache"] = cache_data
            entries += len(cache_data)

        # Handle incremental saves
        if incremental and os.path.exists(filepath):
            with open(filepath, "r") as f:
                existing = json.load(f)
            # Merge: newer data takes precedence
            if "identity_cache" in session_data and "identity_cache" in existing:
                # Merge identities
                existing_ids = existing["identity_cache"].get("identities", {})
                new_ids = session_data["identity_cache"].get("identities", {})
                existing_ids.update(new_ids)
                session_data["identity_cache"]["identities"] = existing_ids
            if "embedding_cache" in session_data and "embedding_cache" in existing:
                existing["embedding_cache"].update(session_data["embedding_cache"])
                session_data["embedding_cache"] = existing["embedding_cache"]

        # Write to file
        with open(filepath, "w") as f:
            json.dump(session_data, f)

        size_bytes = os.path.getsize(filepath)
        size_mb = size_bytes / (1024 * 1024)

        logger.info(
            "save_session_state_completed",
            filepath=filepath,
            entries=entries,
            size_mb=round(size_mb, 2),
            incremental=incremental,
        )

        return {
            "saved": filepath,
            "entries": entries,
            "size_mb": round(size_mb, 2),
        }

    async def load_session_state(
        self,
        filepath: str,
        merge: bool = False,
    ) -> dict:
        """
        Load session state for playback or continuation.

        V39.6 Feature: Session restoration for State of Witness.
        Enables resuming performer tracking from saved state.

        Args:
            filepath: Path to saved session JSON
            merge: If True, merge with existing state; if False, replace

        Returns:
            Dict with load metadata: filepath, entries count, identities list

        Example:
            result = await layer.load_session_state(
                "session_2026_01_25.json",
                merge=False,
            )
            # {"loaded": "...", "entries": 245, "identities": ["performer_1", ...]}
        """
        import json

        with open(filepath, "r") as f:
            session_data = json.load(f)

        entries = 0
        identities = []
        identity_cache = None
        gesture_library = None

        # Load identity cache
        if "identity_cache" in session_data:
            cache_data = session_data["identity_cache"]
            identity_cache = IdentityCache.from_dict(cache_data)
            identities = list(identity_cache._histories.keys())
            entries += len(identities)

        # Load gesture library
        if "gesture_library" in session_data:
            lib_data = session_data["gesture_library"]
            gesture_library = await GestureEmbeddingLibrary.from_dict(lib_data, self)
            entries += len(gesture_library._embeddings)

        # Load embedding cache
        if "embedding_cache" in session_data:
            cache_data = session_data["embedding_cache"]
            if merge:
                self._cache.update(cache_data)
            else:
                self._cache = cache_data.copy()
            entries += len(cache_data)

        logger.info(
            "load_session_state_completed",
            filepath=filepath,
            entries=entries,
            identities=identities,
            merge=merge,
        )

        return {
            "loaded": filepath,
            "entries": entries,
            "identities": identities,
            "identity_cache": identity_cache,
            "gesture_library": gesture_library,
            "version": session_data.get("version", "unknown"),
        }

    async def cross_project_search(
        self,
        query: str,
        collections: list[str],
        top_k_per_collection: int = 5,
        merge_strategy: str = "rrf",
    ) -> list[dict[str, Any]]:
        """
        Search across multiple project collections with result fusion.

        Enables unified search across Witness + Trading or other
        project-specific vector stores.

        Args:
            query: Search query
            collections: List of collection names to search
            top_k_per_collection: Results per collection before fusion
            merge_strategy: How to merge results ("rrf", "max", "sum")

        Returns:
            List of merged results with collection source metadata

        Example:
            >>> # Search across both Witness and Trading
            >>> results = await layer.cross_project_search(
            ...     query="pattern recognition momentum",
            ...     collections=["witness_poses", "trading_signals"],
            ...     merge_strategy="rrf",
            ... )
            >>> for r in results:
            ...     print(f"[{r['collection']}] {r['payload']}")
        """
        if not self._initialized:
            await self.initialize()

        # Get query embedding
        query_emb = await self.embed_query(query)

        all_results: list[dict[str, Any]] = []

        # Search each collection (would typically use QdrantVectorStore)
        # This is a template - actual implementation depends on store
        for collection in collections:
            # Placeholder for actual vector store search
            logger.debug(
                "cross_project_search_collection",
                collection=collection,
                query_preview=query[:30],
            )
            # Results would come from: await self.store.search(collection, query_emb, top_k_per_collection)

        # Apply merge strategy
        if merge_strategy == "rrf":
            # Reciprocal Rank Fusion across collections
            pass
        elif merge_strategy == "max":
            # Take max score per unique document
            pass
        else:
            # Sum scores
            pass

        return all_results

    # =========================================================================
    # V39.3: Advanced Semantic Search Patterns
    # =========================================================================

    async def semantic_search_mmr(
        self,
        query: str,
        documents: list[str],
        doc_embeddings: Optional[list[list[float]]] = None,
        top_k: int = 5,
        lambda_mult: float = 0.5,
        fetch_k: int = 20,
    ) -> list[tuple[int, float, str]]:
        """
        Maximal Marginal Relevance (MMR) search for diverse results.

        MMR balances relevance and diversity by iteratively selecting documents
        that are similar to the query but dissimilar to already-selected documents.

        Formula: MMR = λ * Sim(d, q) - (1-λ) * max(Sim(d, d_selected))

        Higher lambda = more relevance-focused
        Lower lambda = more diversity-focused

        Args:
            query: Search query
            documents: List of document texts
            doc_embeddings: Pre-computed embeddings (optional)
            top_k: Number of diverse results to return
            lambda_mult: Balance between relevance (1.0) and diversity (0.0)
            fetch_k: Initial candidates to consider before MMR filtering

        Returns:
            List of (index, mmr_score, document) tuples

        Example:
            >>> # Get diverse search results
            >>> results = await layer.semantic_search_mmr(
            ...     query="machine learning applications",
            ...     documents=corpus,
            ...     top_k=5,
            ...     lambda_mult=0.7,  # More relevance-focused
            ... )
        """
        # V39.11: Start timing for tracing
        start_time = time.perf_counter()

        if not self._initialized:
            await self.initialize()

        # Get query embedding
        query_emb = await self.embed_query(query)

        # Get or compute document embeddings
        if doc_embeddings is None:
            result = await self.embed_documents(documents)
            doc_embeddings = result.embeddings

        # Calculate query-document similarities
        query_sims = [
            (i, self.cosine_similarity(query_emb, doc_emb))
            for i, doc_emb in enumerate(doc_embeddings)
        ]

        # Sort by similarity and get top fetch_k candidates
        query_sims.sort(key=lambda x: x[1], reverse=True)
        candidates = query_sims[:fetch_k]

        # MMR selection
        selected: list[int] = []
        selected_embeddings: list[list[float]] = []
        mmr_results: list[tuple[int, float, str]] = []

        while len(selected) < top_k and candidates:
            best_mmr = float("-inf")
            best_idx = -1
            best_candidate_pos = -1

            for pos, (doc_idx, query_sim) in enumerate(candidates):
                if doc_idx in selected:
                    continue

                # Calculate max similarity to already selected documents
                if selected_embeddings:
                    max_div_sim = max(
                        self.cosine_similarity(doc_embeddings[doc_idx], sel_emb)
                        for sel_emb in selected_embeddings
                    )
                else:
                    max_div_sim = 0.0

                # MMR score
                mmr = lambda_mult * query_sim - (1 - lambda_mult) * max_div_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = doc_idx
                    best_candidate_pos = pos

            if best_idx >= 0:
                selected.append(best_idx)
                selected_embeddings.append(doc_embeddings[best_idx])
                mmr_results.append((best_idx, best_mmr, documents[best_idx]))
                # Remove from candidates
                if best_candidate_pos >= 0:
                    candidates.pop(best_candidate_pos)

        logger.info(
            "mmr_search_completed",
            query_length=len(query),
            total_docs=len(documents),
            fetch_k=fetch_k,
            top_k=len(mmr_results),
            lambda_mult=lambda_mult,
        )

        # V39.11: Log tracing metadata
        latency_ms = (time.perf_counter() - start_time) * 1000
        self._trace_search_operation(
            operation_name="semantic_search_mmr",
            query=query,
            num_docs=len(documents),
            num_results=len(mmr_results),
            latency_ms=latency_ms,
            extra_metadata={"lambda_mult": lambda_mult, "fetch_k": fetch_k, "top_k": top_k},
        )

        return mmr_results

    async def semantic_search_multi_query(
        self,
        query: str,
        documents: list[str],
        doc_embeddings: Optional[list[list[float]]] = None,
        top_k: int = 5,
        num_sub_queries: int = 3,
        fusion_method: str = "rrf",
    ) -> list[tuple[int, float, str]]:
        """
        Multi-query retrieval for complex searches.

        Decomposes the query into multiple sub-queries and fuses results.
        Uses query expansion to capture different aspects of the search intent.

        Fusion methods:
        - rrf: Reciprocal Rank Fusion (default, robust)
        - sum: Sum of relevance scores
        - max: Maximum relevance score

        Args:
            query: Complex search query
            documents: List of document texts
            doc_embeddings: Pre-computed embeddings (optional)
            top_k: Number of final results
            num_sub_queries: Number of query variations to generate
            fusion_method: How to combine results ("rrf", "sum", "max")

        Returns:
            List of (index, fused_score, document) tuples

        Example:
            >>> # Complex query broken into sub-queries
            >>> results = await layer.semantic_search_multi_query(
            ...     query="How does the caching system improve performance?",
            ...     documents=codebase_docs,
            ...     num_sub_queries=4,
            ... )
        """
        if not self._initialized:
            await self.initialize()

        # Get or compute document embeddings once
        if doc_embeddings is None:
            result = await self.embed_documents(documents)
            doc_embeddings = result.embeddings

        # Generate sub-queries through query expansion
        sub_queries = self._generate_sub_queries(query, num_sub_queries)

        # Collect ranked lists from each sub-query
        all_rankings: list[list[tuple[int, float]]] = []

        for sub_query in sub_queries:
            query_emb = await self.embed_query(sub_query)
            similarities = [
                (i, self.cosine_similarity(query_emb, doc_emb))
                for i, doc_emb in enumerate(doc_embeddings)
            ]
            similarities.sort(key=lambda x: x[1], reverse=True)
            all_rankings.append(similarities[:top_k * 2])  # Get more for fusion

        # Fuse results
        if fusion_method == "rrf":
            fused_scores = self._reciprocal_rank_fusion(all_rankings)
        elif fusion_method == "sum":
            fused_scores = self._sum_fusion(all_rankings)
        elif fusion_method == "max":
            fused_scores = self._max_fusion(all_rankings)
        else:
            fused_scores = self._reciprocal_rank_fusion(all_rankings)

        # Sort by fused score and return top_k
        results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        logger.info(
            "multi_query_search_completed",
            original_query=query[:50],
            num_sub_queries=len(sub_queries),
            fusion_method=fusion_method,
            top_k=len(results),
        )

        return [(idx, score, documents[idx]) for idx, score in results]

    def _generate_sub_queries(self, query: str, num_queries: int) -> list[str]:
        """Generate query variations for multi-query retrieval."""
        sub_queries = [query]  # Always include original

        # Simple query expansion strategies
        if num_queries >= 2:
            # Add keyword-focused variant
            keywords = [w for w in query.split() if len(w) > 3]
            if keywords:
                sub_queries.append(" ".join(keywords[:5]))

        if num_queries >= 3:
            # Add question-style variant
            if not query.endswith("?"):
                sub_queries.append(f"What is {query}?")
            else:
                sub_queries.append(query.replace("?", "").strip())

        if num_queries >= 4:
            # Add definition-style variant
            sub_queries.append(f"explain {query}")

        return sub_queries[:num_queries]

    def _reciprocal_rank_fusion(
        self,
        rankings: list[list[tuple[int, float]]],
        k: int = 60,
    ) -> dict[int, float]:
        """Reciprocal Rank Fusion: 1/(k + rank)."""
        fused: dict[int, float] = {}

        for ranking in rankings:
            for rank, (doc_idx, _score) in enumerate(ranking):
                if doc_idx not in fused:
                    fused[doc_idx] = 0.0
                fused[doc_idx] += 1.0 / (k + rank + 1)

        return fused

    def _sum_fusion(
        self,
        rankings: list[list[tuple[int, float]]],
    ) -> dict[int, float]:
        """Sum fusion: sum of relevance scores."""
        fused: dict[int, float] = {}

        for ranking in rankings:
            for doc_idx, score in ranking:
                if doc_idx not in fused:
                    fused[doc_idx] = 0.0
                fused[doc_idx] += score

        return fused

    def _max_fusion(
        self,
        rankings: list[list[tuple[int, float]]],
    ) -> dict[int, float]:
        """Max fusion: maximum relevance score."""
        fused: dict[int, float] = {}

        for ranking in rankings:
            for doc_idx, score in ranking:
                if doc_idx not in fused:
                    fused[doc_idx] = score
                else:
                    fused[doc_idx] = max(fused[doc_idx], score)

        return fused

    async def hybrid_search(
        self,
        query: str,
        documents: list[str],
        doc_embeddings: Optional[list[list[float]]] = None,
        top_k: int = 5,
        alpha: float = 0.5,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> list[tuple[int, float, str]]:
        """
        Hybrid search combining vector similarity with BM25 keyword matching.

        Uses linear interpolation: score = α * vector_score + (1-α) * bm25_score

        Higher alpha = more semantic/vector-focused
        Lower alpha = more keyword/BM25-focused

        Args:
            query: Search query
            documents: List of document texts
            doc_embeddings: Pre-computed embeddings (optional)
            top_k: Number of results
            alpha: Balance between vector (1.0) and BM25 (0.0)
            bm25_k1: BM25 term frequency saturation parameter
            bm25_b: BM25 document length normalization parameter

        Returns:
            List of (index, hybrid_score, document) tuples

        Example:
            >>> # Balance semantic and keyword matching
            >>> results = await layer.hybrid_search(
            ...     query="async python cache implementation",
            ...     documents=code_docs,
            ...     alpha=0.6,  # Slightly more semantic
            ... )
        """
        # V39.11: Start timing for tracing
        start_time = time.perf_counter()

        if not self._initialized:
            await self.initialize()

        # Get query embedding and compute vector similarities
        query_emb = await self.embed_query(query)

        if doc_embeddings is None:
            result = await self.embed_documents(documents)
            doc_embeddings = result.embeddings

        vector_scores = [
            self.cosine_similarity(query_emb, doc_emb)
            for doc_emb in doc_embeddings
        ]

        # Normalize vector scores to [0, 1]
        max_vec = max(vector_scores) if vector_scores else 1.0
        min_vec = min(vector_scores) if vector_scores else 0.0
        if max_vec > min_vec:
            vector_scores = [(s - min_vec) / (max_vec - min_vec) for s in vector_scores]

        # Compute BM25 scores
        bm25_scores = self._compute_bm25_scores(query, documents, bm25_k1, bm25_b)

        # Normalize BM25 scores to [0, 1]
        max_bm25 = max(bm25_scores) if bm25_scores else 1.0
        min_bm25 = min(bm25_scores) if bm25_scores else 0.0
        if max_bm25 > min_bm25:
            bm25_scores = [(s - min_bm25) / (max_bm25 - min_bm25) for s in bm25_scores]

        # Hybrid fusion
        hybrid_scores = [
            alpha * vec + (1 - alpha) * bm25
            for vec, bm25 in zip(vector_scores, bm25_scores)
        ]

        # Sort and return top_k
        results = sorted(
            [(i, score, documents[i]) for i, score in enumerate(hybrid_scores)],
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        logger.info(
            "hybrid_search_completed",
            query_length=len(query),
            total_docs=len(documents),
            alpha=alpha,
            top_k=len(results),
        )

        # V39.11: Log tracing metadata
        latency_ms = (time.perf_counter() - start_time) * 1000
        self._trace_search_operation(
            operation_name="hybrid_search",
            query=query,
            num_docs=len(documents),
            num_results=len(results),
            latency_ms=latency_ms,
            extra_metadata={"alpha": alpha, "bm25_k1": bm25_k1, "bm25_b": bm25_b},
        )

        return results

    def _compute_bm25_scores(
        self,
        query: str,
        documents: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[float]:
        """Compute BM25 scores for keyword matching."""
        import math
        import re

        # Simple tokenization
        def tokenize(text: str) -> list[str]:
            return re.findall(r"\b\w+\b", text.lower())

        query_terms = set(tokenize(query))
        doc_tokens = [tokenize(doc) for doc in documents]

        # Calculate average document length
        avg_dl = sum(len(doc) for doc in doc_tokens) / len(doc_tokens) if doc_tokens else 1

        # Calculate IDF for query terms
        n_docs = len(documents)
        idf: dict[str, float] = {}
        for term in query_terms:
            df = sum(1 for doc in doc_tokens if term in doc)
            idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1) if df > 0 else 0

        # Calculate BM25 scores
        scores = []
        for doc_idx, tokens in enumerate(doc_tokens):
            doc_len = len(tokens)
            term_freqs: dict[str, int] = {}
            for token in tokens:
                term_freqs[token] = term_freqs.get(token, 0) + 1

            score = 0.0
            for term in query_terms:
                tf = term_freqs.get(term, 0)
                if tf > 0 and term in idf:
                    # BM25 formula
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_len / avg_dl))
                    score += idf[term] * (numerator / denominator)

            scores.append(score)

        return scores

    async def semantic_search_with_filters(
        self,
        query: str,
        documents: list[str],
        metadata: list[dict[str, Any]],
        doc_embeddings: Optional[list[list[float]]] = None,
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
        threshold: float = 0.0,
    ) -> list[tuple[int, float, str, dict[str, Any]]]:
        """
        Semantic search with metadata filtering.

        Applies metadata filters before or after vector search for
        efficient filtered retrieval.

        Filter syntax:
        - {"field": "value"} - Exact match
        - {"field": {"$gt": value}} - Greater than
        - {"field": {"$lt": value}} - Less than
        - {"field": {"$in": [values]}} - In list
        - {"field": {"$contains": substring}} - Contains substring

        Args:
            query: Search query
            documents: List of document texts
            metadata: List of metadata dicts corresponding to documents
            doc_embeddings: Pre-computed embeddings (optional)
            top_k: Number of results
            filters: Metadata filter conditions
            threshold: Minimum similarity score

        Returns:
            List of (index, similarity, document, metadata) tuples

        Example:
            >>> results = await layer.semantic_search_with_filters(
            ...     query="authentication patterns",
            ...     documents=docs,
            ...     metadata=doc_metadata,
            ...     filters={"category": "security", "year": {"$gt": 2023}},
            ...     top_k=10,
            ... )
        """
        if not self._initialized:
            await self.initialize()

        # Apply pre-filters to reduce search space
        if filters:
            valid_indices = [
                i for i, meta in enumerate(metadata)
                if self._matches_filters(meta, filters)
            ]
            filtered_docs = [documents[i] for i in valid_indices]
            filtered_metadata = [metadata[i] for i in valid_indices]
            if doc_embeddings:
                filtered_embeddings = [doc_embeddings[i] for i in valid_indices]
            else:
                filtered_embeddings = None
        else:
            valid_indices = list(range(len(documents)))
            filtered_docs = documents
            filtered_metadata = metadata
            filtered_embeddings = doc_embeddings

        if not filtered_docs:
            return []

        # Perform semantic search on filtered set
        query_emb = await self.embed_query(query)

        if filtered_embeddings is None:
            result = await self.embed_documents(filtered_docs)
            filtered_embeddings = result.embeddings

        # Compute similarities
        similarities = [
            (i, self.cosine_similarity(query_emb, doc_emb))
            for i, doc_emb in enumerate(filtered_embeddings)
        ]

        # Filter by threshold and sort
        similarities = [
            (i, score) for i, score in similarities if score >= threshold
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Map back to original indices and return
        results = []
        for local_idx, score in similarities[:top_k]:
            original_idx = valid_indices[local_idx]
            results.append((
                original_idx,
                score,
                documents[original_idx],
                metadata[original_idx],
            ))

        logger.info(
            "filtered_search_completed",
            query_length=len(query),
            total_docs=len(documents),
            filtered_docs=len(filtered_docs),
            results=len(results),
        )

        return results

    def _matches_filters(self, meta: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check if metadata matches all filter conditions."""
        for field, condition in filters.items():
            if field not in meta:
                return False

            value = meta[field]

            if isinstance(condition, dict):
                # Operator conditions
                for op, target in condition.items():
                    if op == "$gt" and not value > target:
                        return False
                    elif op == "$gte" and not value >= target:
                        return False
                    elif op == "$lt" and not value < target:
                        return False
                    elif op == "$lte" and not value <= target:
                        return False
                    elif op == "$in" and value not in target:
                        return False
                    elif op == "$nin" and value in target:
                        return False
                    elif op == "$contains" and target not in str(value):
                        return False
                    elif op == "$ne" and value == target:
                        return False
            else:
                # Exact match
                if value != condition:
                    return False

        return True

    # =========================================================================
    # V39.7: Batch API Methods (33% Cost Savings)
    # =========================================================================

    async def upload_batch_file(
        self,
        texts: list[str],
        filename: Optional[str] = None,
    ) -> BatchFile:
        """
        Upload a JSONL file for batch processing.

        V39.7 Feature: Prepares input file for Batch API operations.

        Args:
            texts: List of texts to embed
            filename: Optional custom filename

        Returns:
            BatchFile with file_id for batch creation

        Example:
            >>> file = await layer.upload_batch_file(gesture_texts)
            >>> print(f"Uploaded: {file.id}")
        """
        if not self._initialized:
            await self.initialize()

        if not HTTPX_AVAILABLE or _httpx_module is None:
            raise SDKNotAvailableError(
                sdk_name="httpx",
                install_cmd="pip install httpx",
                docs_url="https://docs.voyageai.com/docs/batch-inference",
            )

        import json
        import tempfile
        from pathlib import Path

        # Generate JSONL content
        lines = []
        for i, text in enumerate(texts):
            lines.append(json.dumps({
                "custom_id": f"request_{i}",
                "body": {"input": [text]}
            }))
        jsonl_content = "\n".join(lines)

        # Create temp file
        filename = filename or f"batch_input_{int(time.time())}.jsonl"

        # Upload via API
        api_key = self.config.api_key or os.getenv("VOYAGE_API_KEY", _DEFAULT_VOYAGE_API_KEY)

        async with _httpx_module.AsyncClient() as client:
            files = {"file": (filename, jsonl_content.encode(), "application/jsonl")}
            data = {"purpose": "batch"}

            response = await client.post(
                "https://api.voyageai.com/v1/files",
                headers={"Authorization": f"Bearer {api_key}"},
                files=files,
                data=data,
                timeout=120.0,
            )

            if response.status_code != 200:
                logger.error("batch_file_upload_failed", status=response.status_code, body=response.text)
                raise SDKConfigurationError(
                    sdk_name="Voyage AI Batch API",
                    missing_config=["valid API response"],
                    example=f"Status {response.status_code}: {response.text[:100]}",
                )

            file_data = response.json()

        logger.info(
            "batch_file_uploaded",
            file_id=file_data.get("id"),
            filename=filename,
            texts_count=len(texts),
            bytes=file_data.get("bytes", 0),
        )

        return BatchFile.from_api_response(file_data)

    async def create_batch_embedding_job(
        self,
        texts: list[str],
        model: EmbeddingModel = EmbeddingModel.VOYAGE_4_LARGE,
        input_type: InputType = InputType.DOCUMENT,
        output_dimension: Optional[int] = None,
        output_dtype: str = "float",
        metadata: Optional[dict[str, str]] = None,
    ) -> BatchJob:
        """
        Create an async batch embedding job for large-scale operations.

        V39.7 Feature: 33% cost savings via official Voyage AI Batch API.

        The Batch API is ideal for non-real-time operations:
        - Pre-computing gesture library embeddings
        - Embedding recorded pose sessions
        - Creating training datasets for archetype clustering
        - Populating vector databases

        Args:
            texts: Up to 100K texts to embed
            model: Embedding model (voyage-4-large, voyage-4, voyage-4-lite)
            input_type: query or document
            output_dimension: 256, 512, 1024, 2048 (optional)
            output_dtype: float, int8, uint8, binary, ubinary
            metadata: Up to 16 key-value pairs for job tracking

        Returns:
            BatchJob with id, status, and tracking info

        Example:
            >>> # Pre-compute gesture library (one-time operation)
            >>> job = await layer.create_batch_embedding_job(
            ...     texts=all_gesture_descriptions,
            ...     metadata={"corpus": "gesture_library_v2"}
            ... )
            >>> print(f"Job created: {job.id}, status: {job.status}")

            >>> # Check status later
            >>> status = await layer.get_batch_status(job.id)
            >>> if status.is_successful:
            ...     embeddings = await layer.download_batch_results(job.id)
        """
        if not self._initialized:
            await self.initialize()

        if not HTTPX_AVAILABLE or _httpx_module is None:
            raise SDKNotAvailableError(
                sdk_name="httpx",
                install_cmd="pip install httpx",
                docs_url="https://docs.voyageai.com/docs/batch-inference",
            )

        # Validate inputs
        if len(texts) > 100000:
            raise ValueError(f"Batch API supports max 100K inputs, got {len(texts)}")

        if len(texts) == 0:
            raise ValueError("texts cannot be empty")

        # Upload batch input file
        batch_file = await self.upload_batch_file(texts)

        # Create batch job
        api_key = self.config.api_key or os.getenv("VOYAGE_API_KEY", _DEFAULT_VOYAGE_API_KEY)

        request_params: dict[str, Any] = {
            "model": model.value if isinstance(model, EmbeddingModel) else model,
            "input_type": input_type.value if isinstance(input_type, InputType) else input_type,
        }

        if output_dimension:
            request_params["output_dimension"] = output_dimension
        if output_dtype != "float":
            request_params["output_dtype"] = output_dtype

        batch_request = {
            "endpoint": "/v1/embeddings",
            "completion_window": "12h",
            "request_params": request_params,
            "input_file_id": batch_file.id,
        }

        if metadata:
            batch_request["metadata"] = metadata

        async with _httpx_module.AsyncClient() as client:
            response = await client.post(
                "https://api.voyageai.com/v1/batches",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=batch_request,
                timeout=60.0,
            )

            if response.status_code != 200:
                logger.error("batch_job_create_failed", status=response.status_code, body=response.text)
                raise SDKConfigurationError(
                    sdk_name="Voyage AI Batch API",
                    missing_config=["valid batch creation"],
                    example=f"Status {response.status_code}: {response.text[:100]}",
                )

            job_data = response.json()

        job = BatchJob.from_api_response(job_data)

        logger.info(
            "batch_job_created",
            job_id=job.id,
            status=job.status.value,
            texts_count=len(texts),
            model=request_params["model"],
            metadata=metadata,
        )

        return job

    async def create_batch_contextualized_job(
        self,
        documents: list[list[str]],
        output_dimension: Optional[int] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> BatchJob:
        """
        Create batch job for contextualized chunk embeddings.

        V39.8 Phase 3 Feature: Batch processing for large document corpora.

        Contextualized embeddings understand relationships between chunks in the
        same document. Each chunk's embedding captures not just its own content,
        but also context from surrounding chunks.

        Perfect for:
        - Large document corpus embedding (33% cheaper than real-time)
        - State of Witness session recordings
        - Archetype training datasets
        - Long legal or technical documents

        Args:
            documents: List of documents, each document is a list of chunks.
                       Example: [["intro", "body", "conclusion"], ["ch1", "ch2"]]
            output_dimension: Optional dimension (256, 512, 1024, 2048)
            metadata: Up to 16 key-value pairs for job tracking

        Returns:
            BatchJob with id, status, and tracking info

        Example:
            >>> # Pre-compute session recordings (33% cheaper)
            >>> session_chunks = [
            ...     ["Frame 1: Warrior stance...", "Frame 2: Transition..."],
            ...     ["Frame 1: Sage meditation...", "Frame 2: Stillness..."],
            ... ]
            >>> job = await layer.create_batch_contextualized_job(
            ...     documents=session_chunks,
            ...     metadata={"session": "performance_2026_01_26"}
            ... )
            >>> await layer.wait_for_batch_completion(job.id)
            >>> results = await layer.download_batch_results(job.id)
        """
        if not self._initialized:
            await self.initialize()

        if not HTTPX_AVAILABLE or _httpx_module is None:
            raise SDKNotAvailableError(
                sdk_name="httpx",
                install_cmd="pip install httpx",
                docs_url="https://docs.voyageai.com/docs/batch-inference",
            )

        # Validate inputs
        if len(documents) == 0:
            raise ValueError("documents cannot be empty")

        total_chunks = sum(len(doc) for doc in documents)
        if total_chunks > 100000:
            raise ValueError(f"Batch API supports max 100K inputs, got {total_chunks} chunks")

        # Create JSONL content for contextualized embeddings
        # Each line is a JSON array of chunks (one document per line)
        import json
        import io

        jsonl_content = io.BytesIO()
        for doc_chunks in documents:
            # Write each document as a JSON array of chunks on a single line
            line = json.dumps(doc_chunks, ensure_ascii=False) + "\n"
            jsonl_content.write(line.encode("utf-8"))

        jsonl_content.seek(0)
        file_content = jsonl_content.getvalue()

        # Upload batch file
        api_key = self.config.api_key or os.getenv("VOYAGE_API_KEY", _DEFAULT_VOYAGE_API_KEY)

        async with _httpx_module.AsyncClient() as client:
            # Upload file
            upload_response = await client.post(
                "https://api.voyageai.com/v1/files",
                headers={"Authorization": f"Bearer {api_key}"},
                files={
                    "file": ("contextualized_batch.jsonl", file_content, "application/jsonl")
                },
                data={"purpose": "batch"},
                timeout=120.0,
            )

            if upload_response.status_code != 200:
                logger.error(
                    "contextualized_batch_file_upload_failed",
                    status=upload_response.status_code,
                    body=upload_response.text,
                )
                raise SDKConfigurationError(
                    sdk_name="Voyage AI Batch API",
                    missing_config=["valid file upload"],
                    example=f"Status {upload_response.status_code}: {upload_response.text[:100]}",
                )

            file_data = upload_response.json()
            input_file_id = file_data.get("id")

        # Create batch job for contextualized embeddings
        model = EmbeddingModel.VOYAGE_CONTEXT_3.value

        request_params: dict[str, Any] = {
            "model": model,
        }

        if output_dimension:
            request_params["output_dimension"] = output_dimension

        batch_request = {
            "endpoint": "/v1/contextualizedembeddings",
            "completion_window": "12h",
            "request_params": request_params,
            "input_file_id": input_file_id,
        }

        if metadata:
            batch_request["metadata"] = metadata

        async with _httpx_module.AsyncClient() as client:
            response = await client.post(
                "https://api.voyageai.com/v1/batches",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=batch_request,
                timeout=60.0,
            )

            if response.status_code != 200:
                logger.error(
                    "contextualized_batch_job_create_failed",
                    status=response.status_code,
                    body=response.text,
                )
                raise SDKConfigurationError(
                    sdk_name="Voyage AI Batch API",
                    missing_config=["valid contextualized batch creation"],
                    example=f"Status {response.status_code}: {response.text[:100]}",
                )

            job_data = response.json()

        job = BatchJob.from_api_response(job_data)

        logger.info(
            "batch_contextualized_job_created",
            job_id=job.id,
            status=job.status.value,
            documents_count=len(documents),
            total_chunks=total_chunks,
            model=model,
            metadata=metadata,
        )

        return job

    async def create_batch_rerank_job(
        self,
        queries: list[str],
        documents_per_query: list[list[str]],
        model: RerankModel = RerankModel.RERANK_2_5,
        top_k: Optional[int] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> BatchJob:
        """
        Create batch reranking job for large-scale retrieval refinement.

        V39.8 Phase 3 Feature: Cost-effective reranking for large query sets.

        The Batch API offers 33% cost savings compared to real-time reranking,
        making it ideal for:
        - Offline evaluation of retrieval systems
        - Pre-computing reranked results for common queries
        - Large-scale archetype matching
        - Batch processing of session recordings

        Args:
            queries: List of query strings
            documents_per_query: List of document lists, one per query.
                                 Each inner list contains documents to rerank for that query.
            model: Reranking model (rerank-2.5, rerank-2.5-lite, etc.)
            top_k: Return top_k documents per query (optional)
            metadata: Up to 16 key-value pairs for job tracking

        Returns:
            BatchJob with id, status, and tracking info

        Example:
            >>> # Rerank archetype descriptions for each session
            >>> job = await layer.create_batch_rerank_job(
            ...     queries=["warrior pose", "sage meditation", "jester dance"],
            ...     documents_per_query=[
            ...         archetype_descriptions,  # Same docs for each query
            ...         archetype_descriptions,
            ...         archetype_descriptions,
            ...     ],
            ...     model=RerankModel.RERANK_2_5,
            ...     top_k=5,
            ...     metadata={"purpose": "archetype_matching"}
            ... )
            >>> await layer.wait_for_batch_completion(job.id)
            >>> results = await layer.download_batch_results(job.id)
        """
        if not self._initialized:
            await self.initialize()

        if not HTTPX_AVAILABLE or _httpx_module is None:
            raise SDKNotAvailableError(
                sdk_name="httpx",
                install_cmd="pip install httpx",
                docs_url="https://docs.voyageai.com/docs/batch-inference",
            )

        # Validate inputs
        if len(queries) == 0:
            raise ValueError("queries cannot be empty")

        if len(queries) != len(documents_per_query):
            raise ValueError(
                f"queries and documents_per_query must have same length. "
                f"Got {len(queries)} queries and {len(documents_per_query)} document sets."
            )

        total_pairs = sum(len(docs) for docs in documents_per_query)
        if total_pairs > 100000:
            raise ValueError(f"Batch API supports max 100K inputs, got {total_pairs} query-doc pairs")

        # Create JSONL content for reranking
        # Each line is a JSON object with "query" and "documents" keys
        import json
        import io

        jsonl_content = io.BytesIO()
        for query, docs in zip(queries, documents_per_query):
            line_obj = {
                "query": query,
                "documents": docs,
            }
            line = json.dumps(line_obj, ensure_ascii=False) + "\n"
            jsonl_content.write(line.encode("utf-8"))

        jsonl_content.seek(0)
        file_content = jsonl_content.getvalue()

        # Upload batch file
        api_key = self.config.api_key or os.getenv("VOYAGE_API_KEY", _DEFAULT_VOYAGE_API_KEY)

        async with _httpx_module.AsyncClient() as client:
            # Upload file
            upload_response = await client.post(
                "https://api.voyageai.com/v1/files",
                headers={"Authorization": f"Bearer {api_key}"},
                files={
                    "file": ("rerank_batch.jsonl", file_content, "application/jsonl")
                },
                data={"purpose": "batch"},
                timeout=120.0,
            )

            if upload_response.status_code != 200:
                logger.error(
                    "rerank_batch_file_upload_failed",
                    status=upload_response.status_code,
                    body=upload_response.text,
                )
                raise SDKConfigurationError(
                    sdk_name="Voyage AI Batch API",
                    missing_config=["valid file upload"],
                    example=f"Status {upload_response.status_code}: {upload_response.text[:100]}",
                )

            file_data = upload_response.json()
            input_file_id = file_data.get("id")

        # Create batch job for reranking
        model_value = model.value if isinstance(model, RerankModel) else model

        request_params: dict[str, Any] = {
            "model": model_value,
        }

        if top_k is not None:
            request_params["top_k"] = top_k

        batch_request = {
            "endpoint": "/v1/rerank",
            "completion_window": "12h",
            "request_params": request_params,
            "input_file_id": input_file_id,
        }

        if metadata:
            batch_request["metadata"] = metadata

        async with _httpx_module.AsyncClient() as client:
            response = await client.post(
                "https://api.voyageai.com/v1/batches",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=batch_request,
                timeout=60.0,
            )

            if response.status_code != 200:
                logger.error(
                    "rerank_batch_job_create_failed",
                    status=response.status_code,
                    body=response.text,
                )
                raise SDKConfigurationError(
                    sdk_name="Voyage AI Batch API",
                    missing_config=["valid rerank batch creation"],
                    example=f"Status {response.status_code}: {response.text[:100]}",
                )

            job_data = response.json()

        job = BatchJob.from_api_response(job_data)

        logger.info(
            "batch_rerank_job_created",
            job_id=job.id,
            status=job.status.value,
            queries_count=len(queries),
            total_pairs=total_pairs,
            model=model_value,
            top_k=top_k,
            metadata=metadata,
        )

        return job

    async def get_batch_status(self, batch_id: str) -> BatchJob:
        """
        Get current status of a batch job.

        V39.7 Feature: Poll for batch job completion.

        Args:
            batch_id: The batch job ID

        Returns:
            Updated BatchJob with current status

        Example:
            >>> job = await layer.get_batch_status("batch-abc123")
            >>> print(f"Status: {job.status}, Progress: {job.request_counts.progress_percent}%")
        """
        if not HTTPX_AVAILABLE or _httpx_module is None:
            raise SDKNotAvailableError(
                sdk_name="httpx",
                install_cmd="pip install httpx",
                docs_url="https://docs.voyageai.com/docs/batch-inference",
            )

        api_key = self.config.api_key or os.getenv("VOYAGE_API_KEY", _DEFAULT_VOYAGE_API_KEY)

        async with _httpx_module.AsyncClient() as client:
            response = await client.get(
                f"https://api.voyageai.com/v1/batches/{batch_id}",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )

            if response.status_code != 200:
                logger.error("batch_status_failed", batch_id=batch_id, status=response.status_code)
                raise SDKConfigurationError(
                    sdk_name="Voyage AI Batch API",
                    missing_config=["valid batch ID"],
                    example=f"Status {response.status_code}: {response.text[:100]}",
                )

            job_data = response.json()

        job = BatchJob.from_api_response(job_data)

        logger.debug(
            "batch_status_retrieved",
            job_id=job.id,
            status=job.status.value,
            progress=job.request_counts.progress_percent,
        )

        return job

    async def list_batch_jobs(
        self,
        limit: int = 20,
        after: Optional[str] = None,
    ) -> list[BatchJob]:
        """
        List all batch jobs.

        V39.7 Feature: Retrieve batch job history.

        Args:
            limit: Maximum jobs to return (default 20)
            after: Cursor for pagination

        Returns:
            List of BatchJob objects
        """
        if not HTTPX_AVAILABLE or _httpx_module is None:
            raise SDKNotAvailableError(
                sdk_name="httpx",
                install_cmd="pip install httpx",
                docs_url="https://docs.voyageai.com/docs/batch-inference",
            )

        api_key = self.config.api_key or os.getenv("VOYAGE_API_KEY", _DEFAULT_VOYAGE_API_KEY)

        params: dict[str, Any] = {"limit": limit}
        if after:
            params["after"] = after

        async with _httpx_module.AsyncClient() as client:
            response = await client.get(
                "https://api.voyageai.com/v1/batches",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                params=params,
                timeout=30.0,
            )

            if response.status_code != 200:
                logger.error("batch_list_failed", status=response.status_code)
                raise SDKConfigurationError(
                    sdk_name="Voyage AI Batch API",
                    missing_config=["valid API key"],
                    example=f"Status {response.status_code}: {response.text[:100]}",
                )

            data = response.json()

        jobs = [BatchJob.from_api_response(j) for j in data.get("data", [])]

        logger.debug("batch_jobs_listed", count=len(jobs))

        return jobs

    async def cancel_batch_job(self, batch_id: str) -> BatchJob:
        """
        Cancel an in-progress batch job.

        V39.7 Feature: Stop batch processing early (no penalty).

        Cancellation may take up to 10 minutes. Any completed results
        will be available in the output file.

        Args:
            batch_id: The batch job ID to cancel

        Returns:
            Updated BatchJob with cancelling/cancelled status
        """
        if not HTTPX_AVAILABLE or _httpx_module is None:
            raise SDKNotAvailableError(
                sdk_name="httpx",
                install_cmd="pip install httpx",
                docs_url="https://docs.voyageai.com/docs/batch-inference",
            )

        api_key = self.config.api_key or os.getenv("VOYAGE_API_KEY", _DEFAULT_VOYAGE_API_KEY)

        async with _httpx_module.AsyncClient() as client:
            response = await client.post(
                f"https://api.voyageai.com/v1/batches/{batch_id}/cancel",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )

            if response.status_code != 200:
                logger.error("batch_cancel_failed", batch_id=batch_id, status=response.status_code)
                raise SDKConfigurationError(
                    sdk_name="Voyage AI Batch API",
                    missing_config=["valid batch ID"],
                    example=f"Status {response.status_code}: {response.text[:100]}",
                )

            job_data = response.json()

        job = BatchJob.from_api_response(job_data)

        logger.info(
            "batch_job_cancelled",
            job_id=job.id,
            status=job.status.value,
            completed=job.request_counts.completed,
        )

        return job

    async def download_batch_results(
        self,
        batch_id: str,
        output_path: Optional[str] = None,
        populate_cache: bool = False,
        original_texts: Optional[list[str]] = None,
        input_type: InputType = InputType.DOCUMENT,
    ) -> list[EmbeddingResult]:
        """
        Download and parse batch job results.

        V39.7 Feature: Retrieve embeddings from completed batch job.
        V39.8 Enhancement: Auto-populate cache with downloaded embeddings.

        Args:
            batch_id: The batch job ID
            output_path: Optional path to save raw JSONL results
            populate_cache: If True, add embeddings to cache for future use
            original_texts: Original texts for cache key generation (required if populate_cache=True)
            input_type: Input type for cache keys (query or document)

        Returns:
            List of EmbeddingResult objects from the batch

        Example:
            >>> # V39.7: Basic download
            >>> results = await layer.download_batch_results("batch-abc123")
            >>> for result in results:
            ...     print(f"Embedding dimension: {result.dimension}")

            >>> # V39.8: Download and cache for instant future retrieval
            >>> results = await layer.download_batch_results(
            ...     batch_id="batch-abc123",
            ...     populate_cache=True,
            ...     original_texts=gesture_descriptions,  # Same texts used for batch
            ... )
            >>> # Future calls with same texts will hit cache instantly!
        """
        if not HTTPX_AVAILABLE or _httpx_module is None:
            raise SDKNotAvailableError(
                sdk_name="httpx",
                install_cmd="pip install httpx",
                docs_url="https://docs.voyageai.com/docs/batch-inference",
            )

        import json

        # V39.8: Validate cache population parameters
        if populate_cache and not original_texts:
            raise ValueError(
                "original_texts required when populate_cache=True. "
                "Provide the same texts used in create_batch_embedding_job()."
            )

        # Get batch status to find output file
        job = await self.get_batch_status(batch_id)

        if not job.is_successful:
            raise ValueError(f"Batch job not completed. Status: {job.status.value}")

        if not job.output_file_id:
            raise ValueError("Batch job has no output file")

        # Download output file
        api_key = self.config.api_key or os.getenv("VOYAGE_API_KEY", _DEFAULT_VOYAGE_API_KEY)

        async with _httpx_module.AsyncClient() as client:
            response = await client.get(
                f"https://api.voyageai.com/v1/files/{job.output_file_id}/content",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=300.0,  # Large files may take time
            )

            if response.status_code != 200:
                logger.error("batch_download_failed", batch_id=batch_id, status=response.status_code)
                raise SDKConfigurationError(
                    sdk_name="Voyage AI Batch API",
                    missing_config=["valid output file"],
                    example=f"Status {response.status_code}: {response.text[:100]}",
                )

            content = response.text

        # Optionally save raw content
        if output_path:
            from pathlib import Path
            Path(output_path).write_text(content)
            logger.info("batch_results_saved", path=output_path)

        # Parse JSONL results
        results: list[EmbeddingResult] = []
        lines = content.strip().split("\n")

        for line in lines:
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                response_body = data.get("response", {}).get("body", {})

                if response_body.get("object") == "list":
                    embeddings_data = response_body.get("data", [])
                    embeddings = [e.get("embedding", []) for e in embeddings_data]

                    result = EmbeddingResult(
                        embeddings=embeddings,
                        model=response_body.get("model", ""),
                        input_type="document",
                        total_tokens=response_body.get("usage", {}).get("total_tokens", 0),
                        dimension=len(embeddings[0]) if embeddings else 0,
                    )
                    results.append(result)

            except json.JSONDecodeError as e:
                logger.warning("batch_result_parse_error", error=str(e), line=line[:50])

        logger.info(
            "batch_results_downloaded",
            batch_id=batch_id,
            result_count=len(results),
            total_embeddings=sum(r.count for r in results),
        )

        # V39.8: Populate cache with downloaded embeddings
        if populate_cache and original_texts and results:
            cache_populated = 0
            text_idx = 0
            now = time.time()
            input_type_str = input_type.value if isinstance(input_type, InputType) else input_type

            # Flatten all embeddings and match to original texts
            for result in results:
                model_str = result.model
                for embedding in result.embeddings:
                    if text_idx < len(original_texts):
                        text = original_texts[text_idx]
                        key = self._get_cache_key(text, model_str, input_type_str)

                        # Add to cache if not already present
                        if key not in self._cache:
                            self._cache[key] = CacheEntry(
                                embedding=embedding,
                                timestamp=now,
                                hit_count=0,
                            )
                            self._cache_stats.embedding_count += 1
                            cache_populated += 1

                        text_idx += 1

            logger.info(
                "batch_cache_populated",
                batch_id=batch_id,
                cache_populated=cache_populated,
                total_texts=len(original_texts),
                cache_size=len(self._cache),
            )

        return results

    async def wait_for_batch_completion(
        self,
        batch_id: str,
        poll_interval: float = 30.0,
        timeout: float = 43200.0,  # 12 hours default
        on_progress: Optional[Callable[[BatchJob], None]] = None,
    ) -> BatchJob:
        """
        Wait for batch job to complete with polling.

        V39.7 Feature: Convenience method for batch job completion.

        Args:
            batch_id: The batch job ID
            poll_interval: Seconds between status checks (default 30s)
            timeout: Maximum wait time in seconds (default 12h)
            on_progress: Optional callback for progress updates

        Returns:
            Completed BatchJob

        Raises:
            TimeoutError: If job doesn't complete within timeout

        Example:
            >>> job = await layer.wait_for_batch_completion(
            ...     "batch-abc123",
            ...     on_progress=lambda j: print(f"Progress: {j.request_counts.progress_percent}%")
            ... )
        """
        start_time = time.time()

        while True:
            job = await self.get_batch_status(batch_id)

            if on_progress:
                on_progress(job)

            if job.is_terminal:
                return job

            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Batch job {batch_id} did not complete within {timeout}s. "
                    f"Current status: {job.status.value}"
                )

            await asyncio.sleep(poll_interval)

    async def stream_batch_progress(
        self,
        batch_id: str,
        poll_interval: float = 5.0,
        include_metrics: bool = True,
    ) -> AsyncGenerator[BatchProgressEvent, None]:
        """
        Stream batch progress updates using async generator.

        V39.9 Feature: Real-time progress visibility with rate and ETA.

        This method yields BatchProgressEvent objects as the batch job
        progresses, allowing for real-time monitoring with calculated
        processing rate and estimated time to completion.

        Args:
            batch_id: The batch job ID to monitor
            poll_interval: Seconds between status checks (default 5s)
            include_metrics: Include processing rate and ETA calculations

        Yields:
            BatchProgressEvent with progress, rate, ETA

        Example:
            >>> async for event in layer.stream_batch_progress(job.id):
            ...     print(f"Progress: {event.completed}/{event.total} ({event.percent:.1f}%)")
            ...     print(f"Rate: {event.rate:.1f} embeds/sec, ETA: {event.eta_seconds:.0f}s")
            ...     if event.is_complete:
            ...         break

        State of Witness Use Case:
            >>> # Monitor large session recording embedding
            >>> async for event in layer.stream_batch_progress(job.id):
            ...     await update_td_progress(event.percent, event.eta_seconds)
            ...     if event.is_complete:
            ...         await notify_td_complete()
        """
        # Initialize metrics tracker if enabled
        metrics = BatchProgressMetrics(start_time=datetime.now()) if include_metrics else None

        while True:
            # Get current job status
            job = await self.get_batch_status(batch_id)

            # Update metrics with new sample
            if metrics:
                metrics.add_sample(job.request_counts.completed)

            # Create and yield progress event
            event = BatchProgressEvent.from_batch_job(job, metrics)
            yield event

            # Exit if job is complete
            if event.is_complete:
                return

            # Wait before next poll
            await asyncio.sleep(poll_interval)

    async def wait_for_batch_completion_with_progress(
        self,
        batch_id: str,
        on_progress: Callable[[BatchProgressEvent], Awaitable[None]],
        poll_interval: float = 5.0,
        max_wait: int = 43200,  # 12 hours
    ) -> BatchJob:
        """
        Wait for batch completion with async progress callbacks.

        V39.9 Feature: Progress-aware batch waiting with rate/ETA.

        This method provides callback-based progress monitoring for
        batch jobs, invoking the callback with each progress update
        including calculated rate and ETA metrics.

        Args:
            batch_id: The batch job ID
            on_progress: Async callback for each progress update
            poll_interval: Seconds between status checks (default 5s)
            max_wait: Maximum seconds to wait (default 12 hours)

        Returns:
            Completed BatchJob

        Raises:
            TimeoutError: If job doesn't complete within max_wait

        Example:
            >>> async def log_progress(event: BatchProgressEvent):
            ...     print(f"[{event.percent:.1f}%] {event.completed}/{event.total}")
            ...     if event.rate > 0:
            ...         print(f"Rate: {event.rate:.1f}/s, ETA: {event.eta_seconds:.0f}s")
            ...
            >>> job = await layer.wait_for_batch_completion_with_progress(
            ...     batch_id=job.id,
            ...     on_progress=log_progress,
            ... )

        State of Witness Use Case:
            >>> # Track archetype training batch with logging
            >>> async def on_training_progress(event: BatchProgressEvent):
            ...     if event.percent % 10 == 0:  # Log every 10%
            ...         logger.info(f"Training: {event.percent:.0f}% complete")
            ...
            >>> await layer.wait_for_batch_completion_with_progress(
            ...     batch_id=training_job.id,
            ...     on_progress=on_training_progress,
            ... )
        """
        start_time = time.time()
        final_job: Optional[BatchJob] = None

        async for event in self.stream_batch_progress(
            batch_id=batch_id,
            poll_interval=poll_interval,
            include_metrics=True,
        ):
            # Invoke the progress callback
            await on_progress(event)

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= max_wait:
                raise TimeoutError(
                    f"Batch job {batch_id} did not complete within {max_wait}s. "
                    f"Current progress: {event.percent:.1f}%"
                )

            # Store job for return when complete
            if event.is_complete:
                final_job = await self.get_batch_status(batch_id)
                break

        if final_job is None:
            # Should not happen, but handle gracefully
            final_job = await self.get_batch_status(batch_id)

        return final_job


# =============================================================================
# V39.0: Qdrant Vector Database Integration
# =============================================================================

# Try to import qdrant-client
try:
    from qdrant_client import AsyncQdrantClient, models
    from qdrant_client.http.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        SearchRequest,
        UpdateResult,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    AsyncQdrantClient = None  # type: ignore


@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector database connection."""
    host: str = "localhost"
    port: int = 6333
    api_key: Optional[str] = None
    https: bool = False
    prefer_grpc: bool = True
    timeout: int = 30
    
    # Collection settings
    default_dimension: int = 1024
    distance: str = "cosine"  # cosine, euclid, dot
    on_disk: bool = False  # Store vectors on disk for large collections


@dataclass
class QdrantSearchResult:
    """Result from Qdrant search operation."""
    id: str
    score: float
    payload: dict[str, Any]
    vector: Optional[list[float]] = None


class QdrantVectorStore:
    """
    V39.0: Qdrant Vector Database Integration for production-grade vector storage.
    
    Integrates seamlessly with EmbeddingLayer for end-to-end vector workflows:
    1. Generate embeddings with Voyage AI
    2. Store in Qdrant with rich metadata
    3. Search with hybrid filters
    4. Manage collections per project
    
    Project Collections:
    - witness_poses: Pose embeddings for archetype matching
    - witness_shaders: GLSL shader semantic search
    - trading_signals: Market signal embeddings
    - trading_strategies: Strategy documentation vectors
    - unleash_skills: Claude skill embeddings
    - unleash_conversations: Episodic memory vectors
    
    Example:
        >>> store = QdrantVectorStore(config)
        >>> await store.initialize()
        >>> 
        >>> # Store embeddings from EmbeddingLayer
        >>> result = await embedding_layer.embed(texts)
        >>> await store.upsert(
        ...     collection="witness_shaders",
        ...     embeddings=result.embeddings,
        ...     payloads=[{"name": "noise", "type": "fragment"}],
        ... )
        >>> 
        >>> # Semantic search
        >>> matches = await store.search(
        ...     collection="witness_shaders",
        ...     query_vector=query_embedding,
        ...     filter={"type": "fragment"},
        ...     limit=10,
        ... )
    """
    
    def __init__(
        self,
        config: Optional[QdrantConfig] = None,
        embedding_layer: Optional["EmbeddingLayer"] = None,
    ):
        """
        Initialize Qdrant vector store.
        
        Args:
            config: Qdrant connection configuration
            embedding_layer: Optional EmbeddingLayer for auto-embedding
        """
        self.config = config or QdrantConfig()
        self.embedding_layer = embedding_layer
        self._client: Optional[Any] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize Qdrant client connection."""
        if self._initialized:
            return
        
        if not QDRANT_AVAILABLE:
            raise SDKNotAvailableError(
                sdk_name="qdrant-client",
                install_cmd="pip install qdrant-client",
                docs_url="https://qdrant.tech/documentation/",
            )
        
        self._client = AsyncQdrantClient(  # type: ignore[misc]
            host=self.config.host,
            port=self.config.port,
            api_key=self.config.api_key,
            https=self.config.https,
            prefer_grpc=self.config.prefer_grpc,
            timeout=self.config.timeout,
        )
        
        self._initialized = True
        logger.info(
            "qdrant_initialized",
            host=self.config.host,
            port=self.config.port,
        )
    
    async def create_collection(
        self,
        name: str,
        dimension: Optional[int] = None,
        distance: Optional[str] = None,
        on_disk: Optional[bool] = None,
        recreate: bool = False,
    ) -> bool:
        """
        Create a vector collection.
        
        Args:
            name: Collection name
            dimension: Vector dimension (default: config.default_dimension)
            distance: Distance metric (cosine, euclid, dot)
            on_disk: Store vectors on disk
            recreate: Delete and recreate if exists
            
        Returns:
            True if created, False if already exists
        """
        if not self._initialized:
            await self.initialize()
        
        assert self._client is not None, "Qdrant client not initialized"
        
        dim = dimension or self.config.default_dimension
        dist = distance or self.config.distance
        disk = on_disk if on_disk is not None else self.config.on_disk
        
        # Map distance string to Qdrant enum (type: ignore for optional import)
        distance_map = {
            "cosine": Distance.COSINE,  # type: ignore[possibly-undefined]
            "euclid": Distance.EUCLID,  # type: ignore[possibly-undefined]
            "dot": Distance.DOT,  # type: ignore[possibly-undefined]
        }
        qdrant_distance = distance_map.get(dist.lower(), Distance.COSINE)  # type: ignore[possibly-undefined]
        
        # Check if exists
        collections = await self._client.get_collections()
        exists = any(c.name == name for c in collections.collections)
        
        if exists:
            if recreate:
                await self._client.delete_collection(name)
                logger.info("qdrant_collection_deleted", name=name)
            else:
                logger.debug("qdrant_collection_exists", name=name)
                return False
        
        # Create collection
        await self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(  # type: ignore[possibly-undefined]
                size=dim,
                distance=qdrant_distance,
                on_disk=disk,
            ),
        )
        
        logger.info(
            "qdrant_collection_created",
            name=name,
            dimension=dim,
            distance=dist,
        )
        return True
    
    async def upsert(
        self,
        collection: str,
        embeddings: list[list[float]],
        ids: Optional[list[str]] = None,
        payloads: Optional[list[dict[str, Any]]] = None,
        batch_size: int = 100,
    ) -> int:
        """
        Insert or update vectors in collection.
        
        Args:
            collection: Collection name
            embeddings: List of embedding vectors
            ids: Optional IDs (auto-generated if not provided)
            payloads: Optional metadata for each vector
            batch_size: Batch size for upsert operations
            
        Returns:
            Number of vectors upserted
        """
        if not self._initialized:
            await self.initialize()
        
        if not embeddings:
            return 0
        
        # Generate IDs if not provided
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in embeddings]
        
        assert self._client is not None, "Qdrant client not initialized"
        
        # Default empty payloads
        if payloads is None:
            payloads = [{} for _ in embeddings]
        
        # Create points
        points = [
            PointStruct(  # type: ignore[possibly-undefined]
                id=id_,
                vector=vector,
                payload=payload,
            )
            for id_, vector, payload in zip(ids, embeddings, payloads)
        ]
        
        # Batch upsert
        total_upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            await self._client.upsert(
                collection_name=collection,
                points=batch,
            )
            total_upserted += len(batch)
        
        logger.info(
            "qdrant_upsert_completed",
            collection=collection,
            count=total_upserted,
        )
        return total_upserted
    
    async def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 10,
        filter: Optional[dict[str, Any]] = None,
        with_vectors: bool = False,
        score_threshold: Optional[float] = None,
    ) -> list[QdrantSearchResult]:
        """
        Search for similar vectors.
        
        Args:
            collection: Collection name
            query_vector: Query embedding vector
            limit: Maximum results
            filter: Payload filter conditions ({"field": "value"})
            with_vectors: Include vectors in results
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results with scores and payloads
        """
        if not self._initialized:
            await self.initialize()
        
        assert self._client is not None, "Qdrant client not initialized"
        
        # Build filter if provided
        qdrant_filter = None
        if filter:
            conditions = [
                FieldCondition(  # type: ignore[possibly-undefined]
                    key=key,
                    match=MatchValue(value=value),  # type: ignore[possibly-undefined]
                )
                for key, value in filter.items()
            ]
            qdrant_filter = Filter(must=conditions)  # type: ignore[possibly-undefined, arg-type]
        
        # Execute search
        results = await self._client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
            query_filter=qdrant_filter,
            with_vectors=with_vectors,
            score_threshold=score_threshold,
        )
        
        # Convert to our result type
        search_results = [
            QdrantSearchResult(
                id=str(hit.id),
                score=hit.score,
                payload=hit.payload or {},
                vector=hit.vector if with_vectors else None,
            )
            for hit in results
        ]
        
        logger.debug(
            "qdrant_search_completed",
            collection=collection,
            results=len(search_results),
        )
        return search_results
    
    async def search_with_text(
        self,
        collection: str,
        query_text: str,
        limit: int = 10,
        filter: Optional[dict[str, Any]] = None,
        input_type: InputType = InputType.QUERY,
    ) -> list[QdrantSearchResult]:
        """
        Search using text query (auto-embedded).
        
        Requires embedding_layer to be set.
        
        Args:
            collection: Collection name
            query_text: Text to search for
            limit: Maximum results
            filter: Payload filter conditions
            input_type: Query or document type
            
        Returns:
            List of search results
        """
        if self.embedding_layer is None:
            raise ValueError("embedding_layer required for text search")
        
        # Generate query embedding
        result = await self.embedding_layer.embed(
            [query_text],
            input_type=input_type,
        )
        
        return await self.search(
            collection=collection,
            query_vector=result.embeddings[0],
            limit=limit,
            filter=filter,
        )
    
    async def delete(
        self,
        collection: str,
        ids: Optional[list[str]] = None,
        filter: Optional[dict[str, Any]] = None,
    ) -> int:
        """
        Delete vectors by ID or filter.
        
        Args:
            collection: Collection name
            ids: Specific IDs to delete
            filter: Filter conditions for bulk delete
            
        Returns:
            Approximate count of deleted vectors
        """
        if not self._initialized:
            await self.initialize()
        
        assert self._client is not None, "Qdrant client not initialized"
        
        if ids:
            await self._client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=ids),  # type: ignore[possibly-undefined, arg-type]
            )
            logger.info("qdrant_delete_by_ids", collection=collection, count=len(ids))
            return len(ids)
        
        if filter:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))  # type: ignore[possibly-undefined]
                for k, v in filter.items()
            ]
            await self._client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(  # type: ignore[possibly-undefined]
                    filter=Filter(must=conditions)  # type: ignore[possibly-undefined, arg-type]
                ),
            )
            logger.info("qdrant_delete_by_filter", collection=collection, filter=filter)
            return -1  # Unknown count for filter delete
        
        return 0
    
    async def get_collection_info(self, name: str) -> dict[str, Any]:
        """Get collection statistics and configuration."""
        if not self._initialized:
            await self.initialize()
        
        assert self._client is not None, "Qdrant client not initialized"
        info = await self._client.get_collection(name)
        return {
            "name": name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
            "dimension": info.config.params.vectors.size,
            "distance": str(info.config.params.vectors.distance),
        }


# =============================================================================
# V39.0: Project-Specific Adapters
# =============================================================================

class WitnessVectorAdapter:
    """
    State of Witness project adapter for creative AI vector operations.
    
    Specialized for:
    - Pose archetype embeddings and matching
    - Shader semantic search
    - Particle behavior vectors
    - TouchDesigner parameter spaces
    
    Collections:
    - witness_poses: 33-keypoint pose embeddings → archetype matching
    - witness_shaders: GLSL code semantic vectors
    - witness_particles: Particle system configurations
    - witness_archetypes: 8 archetype reference embeddings
    """
    
    COLLECTIONS = {
        "poses": "witness_poses",
        "shaders": "witness_shaders", 
        "particles": "witness_particles",
        "archetypes": "witness_archetypes",
    }
    
    def __init__(
        self,
        embedding_layer: "EmbeddingLayer",
        qdrant_store: QdrantVectorStore,
    ):
        self.embedding_layer = embedding_layer
        self.qdrant = qdrant_store
        self.qdrant.embedding_layer = embedding_layer
    
    async def initialize_collections(self) -> None:
        """Create all Witness collections with appropriate dimensions."""
        await self.qdrant.initialize()
        
        # Poses: voyage-4-large (1024d)
        await self.qdrant.create_collection(
            name=self.COLLECTIONS["poses"],
            dimension=1024,
            distance="cosine",
        )
        
        # Shaders: voyage-code-3 (1024d)
        await self.qdrant.create_collection(
            name=self.COLLECTIONS["shaders"],
            dimension=1024,
            distance="cosine",
        )
        
        # Particles: voyage-4-large (1024d)
        await self.qdrant.create_collection(
            name=self.COLLECTIONS["particles"],
            dimension=1024,
        )
        
        # Archetypes: 8 reference embeddings
        await self.qdrant.create_collection(
            name=self.COLLECTIONS["archetypes"],
            dimension=1024,
        )
        
        logger.info("witness_collections_initialized")
    
    async def embed_pose(
        self,
        pose_description: str,
        archetype_hint: Optional[str] = None,
    ) -> list[float]:
        """Embed a pose description for archetype matching."""
        prompt = pose_description
        if archetype_hint:
            prompt = f"[{archetype_hint}] {pose_description}"
        
        result = await self.embedding_layer.embed(
            [prompt],
            input_type=InputType.DOCUMENT,
        )
        return result.embeddings[0]
    
    async def find_similar_poses(
        self,
        query: str,
        archetype: Optional[str] = None,
        limit: int = 5,
    ) -> list[QdrantSearchResult]:
        """Find poses similar to the query."""
        filter_dict = {"archetype": archetype} if archetype else None
        
        return await self.qdrant.search_with_text(
            collection=self.COLLECTIONS["poses"],
            query_text=query,
            limit=limit,
            filter=filter_dict,
            input_type=InputType.QUERY,
        )
    
    async def index_shader(
        self,
        shader_code: str,
        name: str,
        shader_type: str = "fragment",
        tags: Optional[list[str]] = None,
    ) -> str:
        """Index a GLSL shader for semantic search."""
        import uuid
        
        # Use code-3 model for shader embeddings
        result = await self.embedding_layer.embed(
            [shader_code],
            input_type=InputType.DOCUMENT,
            model=EmbeddingModel.VOYAGE_CODE_3.value,
        )
        
        shader_id = str(uuid.uuid4())
        await self.qdrant.upsert(
            collection=self.COLLECTIONS["shaders"],
            embeddings=result.embeddings,
            ids=[shader_id],
            payloads=[{
                "name": name,
                "type": shader_type,
                "tags": tags or [],
                "preview": shader_code[:500],
            }],
        )
        
        return shader_id
    
    async def search_shaders(
        self,
        query: str,
        shader_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[QdrantSearchResult]:
        """Search shaders by description or code pattern."""
        # Embed query with code-3
        result = await self.embedding_layer.embed(
            [query],
            input_type=InputType.QUERY,
            model=EmbeddingModel.VOYAGE_CODE_3.value,
        )
        
        filter_dict = {"type": shader_type} if shader_type else None
        
        return await self.qdrant.search(
            collection=self.COLLECTIONS["shaders"],
            query_vector=result.embeddings[0],
            limit=limit,
            filter=filter_dict,
        )

    async def find_similar_poses_mmr(
        self,
        query: str,
        archetype: Optional[str] = None,
        top_k: int = 5,
        lambda_mult: float = 0.5,
        fetch_k: int = 20,
    ) -> list[QdrantSearchResult]:
        """
        Find diverse poses using Maximal Marginal Relevance.
        
        MMR balances relevance with diversity:
        - lambda_mult=1.0: Pure relevance (most similar first)
        - lambda_mult=0.0: Maximum diversity
        - lambda_mult=0.5: Balanced (default)
        
        Useful for archetype exploration - finds poses across different archetypes
        even when query is specific to one.
        """
        # Get all poses from collection
        all_poses = await self.qdrant.scroll(
            collection=self.COLLECTIONS["poses"],
            limit=fetch_k,
            filter={"archetype": archetype} if archetype else None,
        )
        
        if not all_poses:
            return []
        
        # Extract texts and embeddings
        pose_texts = [p.payload.get("description", "") for p in all_poses]
        pose_embeddings = [p.vector for p in all_poses if p.vector]
        
        if not pose_embeddings:
            # Fallback to standard search if no embeddings available
            return await self.find_similar_poses(query, archetype, top_k)
        
        # Use MMR from embedding layer
        mmr_results = await self.embedding_layer.semantic_search_mmr(
            query=query,
            documents=pose_texts,
            doc_embeddings=pose_embeddings,
            top_k=top_k,
            lambda_mult=lambda_mult,
            fetch_k=fetch_k,
        )
        
        # Map back to QdrantSearchResult format
        return [
            QdrantSearchResult(
                id=all_poses[idx].id,
                score=score,
                payload=all_poses[idx].payload,
                vector=all_poses[idx].vector,
            )
            for idx, score, _ in mmr_results
        ]
    
    async def hybrid_shader_search(
        self,
        query: str,
        shader_type: Optional[str] = None,
        alpha: float = 0.5,
        top_k: int = 10,
    ) -> list[QdrantSearchResult]:
        """
        Search shaders using hybrid vector + keyword matching.
        
        Args:
            query: Search query (can be description or code pattern)
            shader_type: Filter by 'fragment', 'vertex', 'compute', etc.
            alpha: Balance between vector (1.0) and BM25 (0.0)
            top_k: Number of results
            
        Returns:
            List of shader results ranked by hybrid score
        
        Example:
            # Find noise-related shaders with keyword focus
            results = await adapter.hybrid_shader_search(
                "simplex noise fractal pattern",
                shader_type="fragment",
                alpha=0.3,  # More keyword focus
            )
        """
        # Get all shaders from collection
        all_shaders = await self.qdrant.scroll(
            collection=self.COLLECTIONS["shaders"],
            limit=100,
            filter={"type": shader_type} if shader_type else None,
        )
        
        if not all_shaders:
            return []
        
        # Extract code and embeddings
        shader_codes = [s.payload.get("preview", "") for s in all_shaders]
        shader_embeddings = [s.vector for s in all_shaders if s.vector]
        
        if not shader_embeddings:
            # Fallback to standard search
            return await self.search_shaders(query, shader_type, top_k)
        
        # Use hybrid search from embedding layer
        hybrid_results = await self.embedding_layer.hybrid_search(
            query=query,
            documents=shader_codes,
            doc_embeddings=shader_embeddings,
            top_k=top_k,
            alpha=alpha,
        )
        
        # Map back to QdrantSearchResult format
        return [
            QdrantSearchResult(
                id=all_shaders[idx].id,
                score=score,
                payload=all_shaders[idx].payload,
                vector=all_shaders[idx].vector,
            )
            for idx, score, _ in hybrid_results
        ]
    
    async def search_particles_with_filters(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
    ) -> list[QdrantSearchResult]:
        """
        Search particle system configurations with metadata filters.
        
        Supported filter operators:
        - Exact match: {"mass": 1.0}
        - Range: {"lifetime": {"$gte": 2.0, "$lte": 10.0}}
        - List membership: {"archetype": {"$in": ["WARRIOR", "JESTER"]}}
        
        Example:
            # Find high-energy particle systems for WARRIOR archetype
            results = await adapter.search_particles_with_filters(
                "explosive burst pattern",
                filters={
                    "archetype": "WARRIOR",
                    "gravity": {"$gte": 8.0},
                    "mass": {"$lte": 1.5},
                },
            )
        """
        # Get filtered particles from collection
        all_particles = await self.qdrant.scroll(
            collection=self.COLLECTIONS["particles"],
            limit=100,
            filter=None,  # We'll filter manually for complex conditions
        )
        
        if not all_particles:
            return []
        
        # Extract descriptions and embeddings
        particle_descs = [p.payload.get("description", "") for p in all_particles]
        particle_embeddings = [p.vector for p in all_particles if p.vector]
        particle_metadata = [p.payload for p in all_particles]
        
        if not particle_embeddings:
            # Fallback to standard search
            return await self.qdrant.search_with_text(
                collection=self.COLLECTIONS["particles"],
                query_text=query,
                limit=top_k,
            )
        
        # Use filtered search from embedding layer
        filtered_results = await self.embedding_layer.semantic_search_with_filters(
            query=query,
            documents=particle_descs,
            metadata=particle_metadata,
            doc_embeddings=particle_embeddings,
            top_k=top_k,
            filters=filters or {},
        )
        
        # Map back to QdrantSearchResult format
        return [
            QdrantSearchResult(
                id=all_particles[idx].id,
                score=score,
                payload=meta,
                vector=all_particles[idx].vector,
            )
            for idx, score, _, meta in filtered_results
        ]
    
    async def discover_archetypes_mmr(
        self,
        seed_pose: str,
        diversity: float = 0.7,
        num_archetypes: int = 4,
    ) -> list[tuple[str, float, dict]]:
        """
        Discover diverse archetypes from a seed pose description.
        
        Uses MMR with high diversity to find poses across different archetypes
        even when starting from a specific pose.
        
        Args:
            seed_pose: Description of starting pose
            diversity: How diverse results should be (0.0-1.0, higher = more diverse)
            num_archetypes: Number of different archetypes to discover
            
        Returns:
            List of (archetype_name, confidence, pose_info) tuples
        """
        # Use MMR with low lambda for maximum diversity
        lambda_mult = 1.0 - diversity
        
        results = await self.find_similar_poses_mmr(
            query=seed_pose,
            top_k=num_archetypes * 2,  # Get more results to ensure diversity
            lambda_mult=lambda_mult,
            fetch_k=50,
        )
        
        # Group by archetype, keep highest confidence per archetype
        archetype_map: dict[str, tuple[float, dict]] = {}
        
        for result in results:
            archetype = result.payload.get("archetype", "UNKNOWN")
            if archetype not in archetype_map or result.score > archetype_map[archetype][0]:
                archetype_map[archetype] = (result.score, result.payload)
        
        # Return top N archetypes
        sorted_archetypes = sorted(
            archetype_map.items(),
            key=lambda x: x[1][0],
            reverse=True,
        )[:num_archetypes]
        
        return [
            (archetype, score, payload)
            for archetype, (score, payload) in sorted_archetypes
        ]


class TradingVectorAdapter:
    """
    AlphaForge Trading System adapter for market intelligence vectors.
    
    Specialized for:
    - Market signal embeddings
    - Strategy documentation search
    - Risk pattern matching
    - Sentiment analysis vectors
    
    Collections:
    - trading_signals: Market event embeddings
    - trading_strategies: Strategy documentation
    - trading_risk: Risk pattern vectors
    - trading_sentiment: News/social sentiment
    """
    
    COLLECTIONS = {
        "signals": "trading_signals",
        "strategies": "trading_strategies",
        "risk": "trading_risk",
        "sentiment": "trading_sentiment",
    }
    
    def __init__(
        self,
        embedding_layer: "EmbeddingLayer",
        qdrant_store: QdrantVectorStore,
    ):
        self.embedding_layer = embedding_layer
        self.qdrant = qdrant_store
        self.qdrant.embedding_layer = embedding_layer
    
    async def initialize_collections(self) -> None:
        """Create all Trading collections."""
        await self.qdrant.initialize()
        
        for name in self.COLLECTIONS.values():
            await self.qdrant.create_collection(
                name=name,
                dimension=1024,
                distance="cosine",
            )
        
        logger.info("trading_collections_initialized")
    
    async def embed_signal(
        self,
        signal_description: str,
        signal_type: str,
        timestamp: Optional[str] = None,
    ) -> str:
        """Embed and store a market signal."""
        import uuid
        
        result = await self.embedding_layer.embed(
            [signal_description],
            input_type=InputType.DOCUMENT,
        )
        
        signal_id = str(uuid.uuid4())
        await self.qdrant.upsert(
            collection=self.COLLECTIONS["signals"],
            embeddings=result.embeddings,
            ids=[signal_id],
            payloads=[{
                "type": signal_type,
                "timestamp": timestamp,
                "description": signal_description[:200],
            }],
        )
        
        return signal_id
    
    async def find_similar_signals(
        self,
        query: str,
        signal_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[QdrantSearchResult]:
        """Find similar historical market signals."""
        filter_dict = {"type": signal_type} if signal_type else None
        
        return await self.qdrant.search_with_text(
            collection=self.COLLECTIONS["signals"],
            query_text=query,
            limit=limit,
            filter=filter_dict,
        )
    
    async def index_strategy(
        self,
        strategy_doc: str,
        name: str,
        category: str = "general",
    ) -> str:
        """Index strategy documentation for semantic search."""
        import uuid
        
        # Use chunking for long strategy documents
        result = await self.embedding_layer.embed_with_chunking(
            [strategy_doc],
            chunk_size=4000,
            aggregation="weighted",  # Prioritize early sections (abstract/summary)
        )
        
        strategy_id = str(uuid.uuid4())
        await self.qdrant.upsert(
            collection=self.COLLECTIONS["strategies"],
            embeddings=result.embeddings,
            ids=[strategy_id],
            payloads=[{
                "name": name,
                "category": category,
                "preview": strategy_doc[:300],
            }],
        )
        
        return strategy_id

    async def find_similar_signals_mmr(
        self,
        query: str,
        signal_type: Optional[str] = None,
        top_k: int = 5,
        lambda_mult: float = 0.5,
        fetch_k: int = 20,
    ) -> list[QdrantSearchResult]:
        """
        Find diverse market signals using Maximal Marginal Relevance.
        
        Useful for discovering varied signal patterns rather than clustering
        on similar ones - helps identify different market conditions.
        
        Args:
            query: Signal description or pattern
            signal_type: Filter by 'momentum', 'mean_reversion', 'breakout', etc.
            top_k: Number of diverse results
            lambda_mult: 0.0=max diversity, 1.0=max relevance
            fetch_k: Candidate pool size
        """
        all_signals = await self.qdrant.scroll(
            collection=self.COLLECTIONS["signals"],
            limit=fetch_k,
            filter={"type": signal_type} if signal_type else None,
        )
        
        if not all_signals:
            return []
        
        signal_descs = [s.payload.get("description", "") for s in all_signals]
        signal_embeddings = [s.vector for s in all_signals if s.vector]
        
        if not signal_embeddings:
            return await self.find_similar_signals(query, signal_type, top_k)
        
        mmr_results = await self.embedding_layer.semantic_search_mmr(
            query=query,
            documents=signal_descs,
            doc_embeddings=signal_embeddings,
            top_k=top_k,
            lambda_mult=lambda_mult,
            fetch_k=fetch_k,
        )
        
        return [
            QdrantSearchResult(
                id=all_signals[idx].id,
                score=score,
                payload=all_signals[idx].payload,
                vector=all_signals[idx].vector,
            )
            for idx, score, _ in mmr_results
        ]
    
    async def hybrid_strategy_search(
        self,
        query: str,
        strategy_type: Optional[str] = None,
        alpha: float = 0.5,
        top_k: int = 10,
    ) -> list[QdrantSearchResult]:
        """
        Search trading strategies using hybrid vector + keyword matching.
        
        Combines semantic understanding with exact keyword matching,
        useful for finding strategies that match both concept and terminology.
        
        Args:
            query: Strategy description or keywords
            strategy_type: Filter by strategy type
            alpha: Balance (0.0=keywords, 1.0=semantic)
            top_k: Number of results
        """
        all_strategies = await self.qdrant.scroll(
            collection=self.COLLECTIONS["strategies"],
            limit=100,
            filter={"type": strategy_type} if strategy_type else None,
        )
        
        if not all_strategies:
            return []
        
        strategy_descs = [s.payload.get("description", "") for s in all_strategies]
        strategy_embeddings = [s.vector for s in all_strategies if s.vector]
        
        if not strategy_embeddings:
            return []
        
        hybrid_results = await self.embedding_layer.hybrid_search(
            query=query,
            documents=strategy_descs,
            doc_embeddings=strategy_embeddings,
            top_k=top_k,
            alpha=alpha,
        )
        
        return [
            QdrantSearchResult(
                id=all_strategies[idx].id,
                score=score,
                payload=all_strategies[idx].payload,
                vector=all_strategies[idx].vector,
            )
            for idx, score, _ in hybrid_results
        ]
    
    async def search_signals_with_filters(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[QdrantSearchResult]:
        """
        Search signals with metadata filtering.
        
        Useful for finding signals matching specific market conditions:
        - Time-based: {"timestamp": {"$gte": "2024-01-01"}}
        - Numeric: {"confidence": {"$gte": 0.8}}
        - Categorical: {"market": {"$in": ["BTC", "ETH"]}}
        
        Example:
            results = await adapter.search_signals_with_filters(
                "bullish momentum divergence",
                filters={
                    "confidence": {"$gte": 0.7},
                    "market": {"$in": ["BTC", "ETH", "SOL"]},
                    "timeframe": "4h",
                },
            )
        """
        all_signals = await self.qdrant.scroll(
            collection=self.COLLECTIONS["signals"],
            limit=100,
        )
        
        if not all_signals:
            return []
        
        signal_descs = [s.payload.get("description", "") for s in all_signals]
        signal_embeddings = [s.vector for s in all_signals if s.vector]
        signal_metadata = [s.payload for s in all_signals]
        
        if not signal_embeddings:
            return []
        
        filtered_results = await self.embedding_layer.semantic_search_with_filters(
            query=query,
            documents=signal_descs,
            metadata=signal_metadata,
            doc_embeddings=signal_embeddings,
            top_k=top_k,
            filters=filters or {},
            threshold=threshold,
        )
        
        return [
            QdrantSearchResult(
                id=all_signals[idx].id,
                score=score,
                payload=meta,
                vector=all_signals[idx].vector,
            )
            for idx, score, _, meta in filtered_results
        ]


class UnleashVectorAdapter:
    """
    Unleash Meta-Project adapter for Claude enhancement vectors.
    
    Specialized for:
    - Skill embeddings and retrieval
    - Conversation memory (episodic)
    - Research synthesis
    - Pattern library
    
    Collections:
    - unleash_skills: Claude skill embeddings
    - unleash_memory: Episodic conversation memory
    - unleash_research: Research synthesis vectors
    - unleash_patterns: Code/architecture pattern library
    """
    
    COLLECTIONS = {
        "skills": "unleash_skills",
        "memory": "unleash_memory",
        "research": "unleash_research",
        "patterns": "unleash_patterns",
    }
    
    def __init__(
        self,
        embedding_layer: "EmbeddingLayer",
        qdrant_store: QdrantVectorStore,
    ):
        self.embedding_layer = embedding_layer
        self.qdrant = qdrant_store
        self.qdrant.embedding_layer = embedding_layer
    
    async def initialize_collections(self) -> None:
        """Create all Unleash collections."""
        await self.qdrant.initialize()
        
        for name in self.COLLECTIONS.values():
            await self.qdrant.create_collection(
                name=name,
                dimension=1024,
                distance="cosine",
            )
        
        logger.info("unleash_collections_initialized")
    
    async def index_skill(
        self,
        skill_content: str,
        name: str,
        category: str,
        tags: Optional[list[str]] = None,
    ) -> str:
        """Index a Claude skill for semantic retrieval."""
        import uuid
        
        result = await self.embedding_layer.embed(
            [skill_content],
            input_type=InputType.DOCUMENT,
        )
        
        skill_id = str(uuid.uuid4())
        await self.qdrant.upsert(
            collection=self.COLLECTIONS["skills"],
            embeddings=result.embeddings,
            ids=[skill_id],
            payloads=[{
                "name": name,
                "category": category,
                "tags": tags or [],
                "preview": skill_content[:500],
            }],
        )
        
        return skill_id
    
    async def find_relevant_skills(
        self,
        task_description: str,
        category: Optional[str] = None,
        limit: int = 5,
    ) -> list[QdrantSearchResult]:
        """Find skills relevant to a task."""
        filter_dict = {"category": category} if category else None
        
        return await self.qdrant.search_with_text(
            collection=self.COLLECTIONS["skills"],
            query_text=task_description,
            limit=limit,
            filter=filter_dict,
            input_type=InputType.QUERY,
        )
    
    async def store_conversation_memory(
        self,
        conversation_summary: str,
        session_id: str,
        topics: list[str],
        importance: float = 0.5,
    ) -> str:
        """Store conversation summary for episodic memory."""
        import uuid
        
        result = await self.embedding_layer.embed(
            [conversation_summary],
            input_type=InputType.DOCUMENT,
        )
        
        memory_id = str(uuid.uuid4())
        await self.qdrant.upsert(
            collection=self.COLLECTIONS["memory"],
            embeddings=result.embeddings,
            ids=[memory_id],
            payloads=[{
                "session_id": session_id,
                "topics": topics,
                "importance": importance,
                "preview": conversation_summary[:300],
            }],
        )
        
        return memory_id
    
    async def recall_memories(
        self,
        query: str,
        min_importance: float = 0.3,
        limit: int = 10,
    ) -> list[QdrantSearchResult]:
        """Recall relevant conversation memories."""
        results = await self.qdrant.search_with_text(
            collection=self.COLLECTIONS["memory"],
            query_text=query,
            limit=limit * 2,  # Over-fetch to filter
        )
        
        # Filter by importance
        return [
            r for r in results
            if r.payload.get("importance", 0) >= min_importance
        ][:limit]
    
    async def index_pattern(
        self,
        pattern_code: str,
        name: str,
        pattern_type: str,
        language: str = "python",
    ) -> str:
        """Index a code/architecture pattern."""
        import uuid
        
        # Use code-3 for code patterns
        result = await self.embedding_layer.embed(
            [pattern_code],
            input_type=InputType.DOCUMENT,
            model=EmbeddingModel.VOYAGE_CODE_3.value,
        )
        
        pattern_id = str(uuid.uuid4())
        await self.qdrant.upsert(
            collection=self.COLLECTIONS["patterns"],
            embeddings=result.embeddings,
            ids=[pattern_id],
            payloads=[{
                "name": name,
                "type": pattern_type,
                "language": language,
                "preview": pattern_code[:500],
            }],
        )
        
        return pattern_id


# =============================================================================
# V39.6: Identity-Aware Cache for Multi-Person Tracking
# =============================================================================

@dataclass
class IdentityFrame:
    """Single frame of data for a tracked identity."""
    embedding: list[float]
    keypoints: list[float]
    timestamp: float
    confidence: float = 1.0


class IdentityCache:
    """
    Per-performer embedding cache with temporal consistency.

    V39.6 Feature: Supports multi-person pose tracking with:
    - Per-identity frame history for velocity calculation
    - Hungarian algorithm matching for identity reassignment
    - Automatic identity expiration after timeout
    - Velocity and acceleration computation

    Example:
        cache = IdentityCache(max_identities=8, history_length=30)

        # Update with new frame
        cache.update(
            identity="performer_1",
            embedding=[0.1, 0.2, ...],
            keypoints=[...],
            confidence=0.95,
        )

        # Get velocity for motion analysis
        velocity = cache.get_velocity("performer_1")

        # Match new detections to existing identities
        matched = cache.match_identities(
            new_embeddings=[...],
            method="hungarian",
        )
    """

    def __init__(
        self,
        max_identities: int = 8,
        history_length: int = 30,
        expiration_seconds: float = 5.0,
    ):
        self.max_identities = max_identities
        self.history_length = history_length
        self.expiration_seconds = expiration_seconds

        # Identity -> deque of IdentityFrame
        self._histories: dict[str, deque] = {}
        self._last_seen: dict[str, float] = {}
        self._creation_time: dict[str, float] = {}

    def update(
        self,
        identity: str,
        embedding: list[float],
        keypoints: list[float],
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Update cache with new frame for given identity.

        Args:
            identity: Unique identifier for this performer
            embedding: 1024d embedding vector
            keypoints: Raw pose keypoints (33 points × 3 coords = 99 values)
            confidence: Detection confidence (0.0-1.0)
            timestamp: Optional timestamp (defaults to current time)
        """
        import time
        ts = timestamp if timestamp is not None else time.time()

        # Initialize history if new identity
        if identity not in self._histories:
            if len(self._histories) >= self.max_identities:
                self._evict_oldest()
            self._histories[identity] = deque(maxlen=self.history_length)
            self._creation_time[identity] = ts

        # Add frame to history
        frame = IdentityFrame(
            embedding=embedding,
            keypoints=keypoints,
            timestamp=ts,
            confidence=confidence,
        )
        self._histories[identity].append(frame)
        self._last_seen[identity] = ts

    def _evict_oldest(self) -> None:
        """Evict the least recently seen identity."""
        if not self._last_seen:
            return
        oldest = min(self._last_seen.keys(), key=lambda k: self._last_seen[k])
        self.remove_identity(oldest)

    def remove_identity(self, identity: str) -> None:
        """Remove an identity from the cache."""
        self._histories.pop(identity, None)
        self._last_seen.pop(identity, None)
        self._creation_time.pop(identity, None)

    def get_velocity(self, identity: str) -> Optional[list[float]]:
        """
        Calculate velocity from recent frames.

        Returns:
            Velocity vector (same dimension as keypoints), or None if insufficient history
        """
        history = self._histories.get(identity)
        if not history or len(history) < 2:
            return None

        # Get last two frames
        current = history[-1]
        previous = history[-2]

        dt = current.timestamp - previous.timestamp
        if dt <= 0:
            return None

        # Calculate velocity for each keypoint dimension
        velocity = [
            (c - p) / dt
            for c, p in zip(current.keypoints, previous.keypoints)
        ]
        return velocity

    def get_acceleration(self, identity: str) -> Optional[list[float]]:
        """
        Calculate acceleration from velocity history.

        Returns:
            Acceleration vector, or None if insufficient history
        """
        history = self._histories.get(identity)
        if not history or len(history) < 3:
            return None

        # Get velocities at t-1 and t
        frames = list(history)[-3:]

        dt1 = frames[1].timestamp - frames[0].timestamp
        dt2 = frames[2].timestamp - frames[1].timestamp

        if dt1 <= 0 or dt2 <= 0:
            return None

        # Velocity at t-1
        v1 = [(b - a) / dt1 for a, b in zip(frames[0].keypoints, frames[1].keypoints)]
        # Velocity at t
        v2 = [(b - a) / dt2 for a, b in zip(frames[1].keypoints, frames[2].keypoints)]

        # Acceleration
        dt_avg = (dt1 + dt2) / 2
        acceleration = [(v2i - v1i) / dt_avg for v1i, v2i in zip(v1, v2)]
        return acceleration

    def get_history(self, identity: str) -> list[IdentityFrame]:
        """Get full frame history for an identity."""
        return list(self._histories.get(identity, []))

    def get_latest_embedding(self, identity: str) -> Optional[list[float]]:
        """Get the most recent embedding for an identity."""
        history = self._histories.get(identity)
        if not history:
            return None
        return history[-1].embedding

    def get_average_embedding(self, identity: str, window: int = 5) -> Optional[list[float]]:
        """
        Get averaged embedding over recent frames.

        Useful for stable archetype matching.
        """
        history = self._histories.get(identity)
        if not history:
            return None

        frames = list(history)[-window:]
        if not frames:
            return None

        # Average embeddings
        n = len(frames)
        dim = len(frames[0].embedding)
        avg = [0.0] * dim

        for frame in frames:
            for i, val in enumerate(frame.embedding):
                avg[i] += val / n

        return avg

    def match_identities(
        self,
        new_embeddings: list[list[float]],
        method: str = "greedy",
        threshold: float = 0.7,
    ) -> dict[int, Optional[str]]:
        """
        Match new embeddings to existing identities.

        Args:
            new_embeddings: List of embeddings from new detections
            method: "greedy" or "hungarian"
            threshold: Minimum similarity to match (0.0-1.0)

        Returns:
            Dict mapping new_embedding_index -> matched_identity (or None)
        """
        if not self._histories or not new_embeddings:
            return {i: None for i in range(len(new_embeddings))}

        # Get current embeddings for all identities
        identity_embeddings = {}
        for identity in list(self._histories.keys()):
            emb = self.get_latest_embedding(identity)
            if emb:
                identity_embeddings[identity] = emb

        if not identity_embeddings:
            return {i: None for i in range(len(new_embeddings))}

        # Compute similarity matrix
        identities = list(identity_embeddings.keys())
        n_new = len(new_embeddings)
        n_existing = len(identities)

        # Use cosine similarity
        def cosine_sim(a: list[float], b: list[float]) -> float:
            dot = sum(ai * bi for ai, bi in zip(a, b))
            norm_a = sum(ai ** 2 for ai in a) ** 0.5
            norm_b = sum(bi ** 2 for bi in b) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        sim_matrix = [
            [cosine_sim(new_embeddings[i], identity_embeddings[identities[j]])
             for j in range(n_existing)]
            for i in range(n_new)
        ]

        matches: dict[int, Optional[str]] = {}
        used_identities: set[str] = set()

        if method == "greedy":
            # Greedy matching: assign each new embedding to best unused identity
            for i in range(n_new):
                best_j = -1
                best_sim = threshold
                for j in range(n_existing):
                    if identities[j] not in used_identities and sim_matrix[i][j] > best_sim:
                        best_sim = sim_matrix[i][j]
                        best_j = j

                if best_j >= 0:
                    matches[i] = identities[best_j]
                    used_identities.add(identities[best_j])
                else:
                    matches[i] = None

        elif method == "hungarian":
            # Hungarian algorithm for optimal assignment
            try:
                from scipy.optimize import linear_sum_assignment
                # Convert similarity to cost (negative similarity)
                cost_matrix = [[-sim_matrix[i][j] for j in range(n_existing)]
                               for i in range(n_new)]

                # Pad if necessary
                if n_new != n_existing:
                    max_dim = max(n_new, n_existing)
                    # Pad with zeros (high cost)
                    for row in cost_matrix:
                        row.extend([0] * (max_dim - n_existing))
                    for _ in range(max_dim - n_new):
                        cost_matrix.append([0] * max_dim)

                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                for i, j in zip(row_ind, col_ind):
                    if i < n_new and j < n_existing:
                        if sim_matrix[i][j] >= threshold:
                            matches[i] = identities[j]
                        else:
                            matches[i] = None

                # Fill in any missing
                for i in range(n_new):
                    if i not in matches:
                        matches[i] = None

            except ImportError:
                # Fallback to greedy if scipy not available
                return self.match_identities(new_embeddings, method="greedy", threshold=threshold)

        return matches

    def cleanup_expired(self, current_time: Optional[float] = None) -> list[str]:
        """
        Remove identities that haven't been seen recently.

        Returns:
            List of removed identity IDs
        """
        import time
        now = current_time if current_time is not None else time.time()

        expired = []
        for identity, last_seen in list(self._last_seen.items()):
            if now - last_seen > self.expiration_seconds:
                expired.append(identity)
                self.remove_identity(identity)

        return expired

    def get_active_identities(self) -> list[str]:
        """Get list of currently tracked identities."""
        return list(self._histories.keys())

    def get_identity_stats(self, identity: str) -> Optional[dict]:
        """Get statistics for an identity."""
        if identity not in self._histories:
            return None

        history = self._histories[identity]
        return {
            "identity": identity,
            "frame_count": len(history),
            "duration_seconds": history[-1].timestamp - history[0].timestamp if len(history) > 1 else 0,
            "avg_confidence": sum(f.confidence for f in history) / len(history) if history else 0,
            "created_at": self._creation_time.get(identity, 0),
            "last_seen": self._last_seen.get(identity, 0),
        }

    def to_dict(self) -> dict:
        """Serialize cache state for persistence."""
        return {
            "max_identities": self.max_identities,
            "history_length": self.history_length,
            "expiration_seconds": self.expiration_seconds,
            "identities": {
                identity: {
                    "frames": [
                        {
                            "embedding": f.embedding,
                            "keypoints": f.keypoints,
                            "timestamp": f.timestamp,
                            "confidence": f.confidence,
                        }
                        for f in history
                    ],
                    "created_at": self._creation_time.get(identity, 0),
                    "last_seen": self._last_seen.get(identity, 0),
                }
                for identity, history in self._histories.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IdentityCache":
        """Deserialize cache state from persistence."""
        cache = cls(
            max_identities=data.get("max_identities", 8),
            history_length=data.get("history_length", 30),
            expiration_seconds=data.get("expiration_seconds", 5.0),
        )

        for identity, id_data in data.get("identities", {}).items():
            cache._histories[identity] = deque(maxlen=cache.history_length)
            cache._creation_time[identity] = id_data.get("created_at", 0)
            cache._last_seen[identity] = id_data.get("last_seen", 0)

            for frame_data in id_data.get("frames", []):
                frame = IdentityFrame(
                    embedding=frame_data["embedding"],
                    keypoints=frame_data["keypoints"],
                    timestamp=frame_data["timestamp"],
                    confidence=frame_data.get("confidence", 1.0),
                )
                cache._histories[identity].append(frame)

        return cache


# =============================================================================
# V39.6: Gesture Embedding Library
# =============================================================================

class GestureEmbeddingLibrary:
    """
    Pre-computed embeddings for predefined gestures.

    V39.6 Feature: Enables fast gesture recognition via cosine similarity.

    Gestures:
    - WAVE_HELLO, WAVE_GOODBYE
    - POINT_UP, POINT_DOWN, POINT_LEFT, POINT_RIGHT
    - STOP_PALM, THUMBS_UP, THUMBS_DOWN
    - CLAP, ARMS_CROSSED, HANDS_ON_HIPS
    - BOW, JUMP, SPIN

    Example:
        library = GestureEmbeddingLibrary(layer)
        await library.initialize()

        # Recognize gesture from pose embedding
        matches = await library.recognize_gesture(
            pose_embedding=current_embedding,
            confidence_threshold=0.7,
            top_k=3,
        )
        # matches = [("WAVE_HELLO", 0.92), ("POINT_UP", 0.34), ...]
    """

    # Gesture descriptions for embedding
    GESTURES = {
        "WAVE_HELLO": "Person standing with right arm raised above head, hand waving side to side in greeting motion",
        "WAVE_GOODBYE": "Person with arm extended forward at shoulder height, hand waving forward and backward",
        "POINT_UP": "Person with arm raised, index finger extended pointing straight upward toward ceiling",
        "POINT_DOWN": "Person with arm lowered, index finger extended pointing downward toward floor",
        "POINT_LEFT": "Person with left arm extended horizontally, index finger pointing left",
        "POINT_RIGHT": "Person with right arm extended horizontally, index finger pointing right",
        "STOP_PALM": "Person with arm extended forward at shoulder height, palm facing outward in stop gesture",
        "THUMBS_UP": "Person with arm bent at elbow, fist closed with thumb extended upward in approval",
        "THUMBS_DOWN": "Person with arm bent at elbow, fist closed with thumb extended downward in disapproval",
        "CLAP": "Person with both hands raised in front of chest, palms coming together in applause motion",
        "ARMS_CROSSED": "Person standing with both arms folded across chest, hands tucked under opposite elbows",
        "HANDS_ON_HIPS": "Person standing with both hands resting on hip bones, elbows pointing outward",
        "BOW": "Person bending forward from waist, upper body tilting down, head lowered in respect",
        "JUMP": "Person airborne with both feet off ground, arms raised overhead, body fully extended",
        "SPIN": "Person rotating body around vertical axis, one leg pivoting, arms extended for balance",
    }

    def __init__(
        self,
        embedding_layer: "EmbeddingLayer",
        model: EmbeddingModel = EmbeddingModel.VOYAGE_4_LITE,
    ):
        self.embedding_layer = embedding_layer
        self.model = model
        self._embeddings: dict[str, list[float]] = {}
        self._initialized = False
        self._batch_job_id: Optional[str] = None

    async def initialize(
        self,
        use_batch: bool = False,
        batch_wait: bool = True,
    ) -> Optional[str]:
        """
        Pre-compute embeddings for all gestures.

        V39.8 Enhancement: Optional batch mode for 33% cost savings.

        Args:
            use_batch: If True, use Batch API (async, 33% cheaper)
            batch_wait: If True, wait for batch completion (up to 12h)

        Returns:
            None if using real-time API or batch_wait=True
            batch_job_id if use_batch=True and batch_wait=False

        Example (real-time, immediate):
            >>> await library.initialize()  # Full price, instant

        Example (batch, 33% cheaper, wait):
            >>> await library.initialize(use_batch=True)  # Wait for completion

        Example (batch, fire-and-forget):
            >>> batch_id = await library.initialize(use_batch=True, batch_wait=False)
            >>> # Poll later with: await library.initialize_from_batch_job(batch_id)
        """
        if self._initialized:
            return None

        descriptions = list(self.GESTURES.values())
        names = list(self.GESTURES.keys())

        if use_batch:
            # V39.8: Use Batch API for 33% cost savings
            job = await self.embedding_layer.create_batch_embedding_job(
                texts=descriptions,
                model=self.model,
                input_type=InputType.DOCUMENT,
                metadata={"purpose": "gesture_library", "version": "v39.8"},
            )
            self._batch_job_id = job.id

            if batch_wait:
                # Wait for batch completion
                completed_job = await self.embedding_layer.wait_for_batch_completion(
                    batch_id=job.id,
                    poll_interval=30.0,  # Check every 30 seconds
                )

                if completed_job.is_successful:
                    # Download and cache results
                    results = await self.embedding_layer.download_batch_results(
                        batch_id=job.id,
                        populate_cache=True,
                        original_texts=descriptions,
                        input_type=InputType.DOCUMENT,
                    )

                    # Extract embeddings
                    all_embeddings: list[list[float]] = []
                    for result in results:
                        all_embeddings.extend(result.embeddings)

                    for name, embedding in zip(names, all_embeddings):
                        self._embeddings[name] = embedding

                    self._initialized = True
                    return None
                else:
                    raise RuntimeError(f"Batch job failed: {completed_job.status.value}")
            else:
                # Return job ID for later polling
                return job.id
        else:
            # Real-time API (original behavior)
            result = await self.embedding_layer.embed(
                texts=descriptions,
                model=self.model,
                input_type=InputType.DOCUMENT,
            )

            for name, embedding in zip(names, result.embeddings):
                self._embeddings[name] = embedding

            self._initialized = True
            return None

    async def initialize_from_batch_job(self, batch_id: str) -> None:
        """
        Initialize gestures from a previously created batch job.

        V39.8 Feature: Load pre-computed gesture embeddings.

        Useful for loading pre-computed embeddings after fire-and-forget batch.

        Args:
            batch_id: The batch job ID from initialize(use_batch=True, batch_wait=False)

        Example:
            >>> # Create batch job
            >>> batch_id = await library.initialize(use_batch=True, batch_wait=False)
            >>> # ... hours later ...
            >>> await library.initialize_from_batch_job(batch_id)
        """
        if self._initialized:
            return

        descriptions = list(self.GESTURES.values())
        names = list(self.GESTURES.keys())

        # Download and cache results
        results = await self.embedding_layer.download_batch_results(
            batch_id=batch_id,
            populate_cache=True,
            original_texts=descriptions,
            input_type=InputType.DOCUMENT,
        )

        # Extract embeddings
        all_embeddings: list[list[float]] = []
        for result in results:
            all_embeddings.extend(result.embeddings)

        for name, embedding in zip(names, all_embeddings):
            self._embeddings[name] = embedding

        self._batch_job_id = batch_id
        self._initialized = True

    async def recognize_gesture(
        self,
        pose_embedding: list[float],
        confidence_threshold: float = 0.6,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """
        Match a pose embedding to known gestures.

        Args:
            pose_embedding: Embedding of current pose/sequence
            confidence_threshold: Minimum similarity to return
            top_k: Number of top matches to return

        Returns:
            List of (gesture_name, confidence) tuples, sorted by confidence
        """
        if not self._initialized:
            await self.initialize()

        # Compute similarities
        similarities = []
        for gesture_name, gesture_emb in self._embeddings.items():
            sim = self._cosine_similarity(pose_embedding, gesture_emb)
            if sim >= confidence_threshold:
                similarities.append((gesture_name, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(ai * bi for ai, bi in zip(a, b))
        norm_a = sum(ai ** 2 for ai in a) ** 0.5
        norm_b = sum(bi ** 2 for bi in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def add_custom_gesture(
        self,
        name: str,
        description: str,
    ) -> None:
        """
        Add a custom gesture to the library.

        Args:
            name: Unique gesture name (uppercase recommended)
            description: Natural language description of the gesture
        """
        result = await self.embedding_layer.embed(
            texts=[description],
            model=self.model,
            input_type=InputType.DOCUMENT,
        )
        self._embeddings[name] = result.embeddings[0]

    def get_gesture_names(self) -> list[str]:
        """Get list of all recognized gesture names."""
        return list(self._embeddings.keys())

    def get_gesture_embedding(self, name: str) -> Optional[list[float]]:
        """Get the embedding for a specific gesture."""
        return self._embeddings.get(name)

    def to_dict(self) -> dict:
        """Serialize library for persistence."""
        return {
            "gestures": dict(self.GESTURES),
            "embeddings": dict(self._embeddings),
            "model": self.model.value if hasattr(self.model, 'value') else str(self.model),
        }

    @classmethod
    async def from_dict(
        cls,
        data: dict,
        embedding_layer: "EmbeddingLayer",
    ) -> "GestureEmbeddingLibrary":
        """Deserialize library from persistence."""
        model = EmbeddingModel(data.get("model", "voyage-3-lite"))
        library = cls(embedding_layer, model=model)
        library._embeddings = data.get("embeddings", {})
        library._initialized = bool(library._embeddings)
        return library


# =============================================================================
# V39.12: Archetype-Based Cache Warming (State of Witness Integration)
# =============================================================================

# The 8 archetypes from State of Witness
ARCHETYPE_NAMES: list[str] = [
    "WARRIOR",
    "NURTURER",
    "SAGE",
    "JESTER",
    "LOVER",
    "MAGICIAN",
    "INNOCENT",
    "EVERYMAN",
]

# Archetype gesture variations for cache warming
# Each archetype has 5 descriptive variations for better coverage
ARCHETYPE_VARIATIONS: dict[str, list[str]] = {
    "WARRIOR": [
        "Person in aggressive forward stance, weight on front foot, arms raised in fighting position",
        "Figure performing quick forceful punch with rotated torso and extended arm",
        "Combative pose with feet wide apart, fists clenched, intense forward lean",
        "Person executing rapid defensive blocks with both arms crossing body",
        "Athletic figure in dynamic lunge position, body coiled for explosive movement",
    ],
    "NURTURER": [
        "Person in centered grounded posture with arms open wide in welcoming embrace",
        "Figure with smooth flowing movements, hands at heart level, gentle swaying",
        "Calm standing pose with one hand on chest, other arm extended supportively",
        "Person in stable nurturing stance, slight forward lean, arms cradling gesture",
        "Figure demonstrating protective posture, shoulders relaxed, palms facing outward softly",
    ],
    "SAGE": [
        "Person with perfect vertical spine alignment, hands in contemplative mudra position",
        "Figure in deliberate slow movement, one hand raised as if imparting wisdom",
        "Meditative standing pose with eyes focused, minimal but intentional gestures",
        "Scholar-like posture with one hand stroking chin, head slightly tilted in thought",
        "Person in balanced stillness, weight evenly distributed, serene expression",
    ],
    "JESTER": [
        "Person in erratic dancing motion, limbs moving in unpredictable directions",
        "Figure with high acceleration bursts, sudden direction changes, playful energy",
        "Exaggerated theatrical pose with arms and legs at odd angles, head tilted",
        "Person performing bouncy jumps with asymmetric arm movements",
        "Clownish stance with one leg raised, arms waving frantically, body swaying",
    ],
    "LOVER": [
        "Person in fluid undulating movements, hips leading the motion, sensual energy",
        "Figure with slow hip-centric dance, arms tracing curves through space",
        "Graceful pose with weight shifting between feet, body in gentle S-curve",
        "Person reaching outward with longing gesture, body leaning toward focal point",
        "Romantic partner dance position, one arm extended, body open and receptive",
    ],
    "MAGICIAN": [
        "Person performing precise hand gestures, fingers articulating complex patterns",
        "Figure with controlled mystical movements, hands drawing symbols in air",
        "Conjurer pose with one hand raised palm up, other making intricate finger signs",
        "Person in focused stance, both hands manipulating invisible energy between them",
        "Theatrical presentation posture, cape-flourish motion with dramatic arm sweep",
    ],
    "INNOCENT": [
        "Person with light bouncy movements, high energy, childlike enthusiasm",
        "Figure skipping with arms swinging freely, head turning with curiosity",
        "Joyful jumping pose with both feet leaving ground, arms reaching upward",
        "Wonder-filled stance, hands clasped in delight, slight upward gaze",
        "Playful spinning motion, arms extended for balance, carefree expression",
    ],
    "EVERYMAN": [
        "Person in neutral balanced standing posture, weight evenly distributed",
        "Figure with consistent measured movements, neither fast nor slow",
        "Standard walking pose with natural arm swing and regular stride",
        "Relaxed casual stance, hands in pockets or at sides, easy expression",
        "Person in everyday working position, practical and functional movements",
    ],
}

# Archetype color mappings (RGB) for visualization
ARCHETYPE_COLORS: dict[str, tuple[int, int, int]] = {
    "WARRIOR": (255, 0, 0),       # Pure Red
    "NURTURER": (255, 105, 180),  # Hot Pink
    "SAGE": (0, 255, 255),        # Cyan
    "JESTER": (255, 255, 0),      # Yellow
    "LOVER": (255, 20, 147),      # Deep Pink
    "MAGICIAN": (138, 43, 226),   # Blue Violet
    "INNOCENT": (0, 255, 127),    # Spring Green
    "EVERYMAN": (192, 192, 192),  # Silver/Gray
}


@dataclass
class ArchetypeCacheStats:
    """
    V39.12: Per-archetype cache statistics for monitoring.

    Tracks cache hits, misses, and warming effectiveness by archetype.
    """
    archetype: str
    warmed_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_queries: int = 0
    last_warmed: Optional[str] = None  # ISO timestamp

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as percentage."""
        if self.total_queries == 0:
            return 0.0
        return (self.cache_hits / self.total_queries) * 100.0

    @property
    def effectiveness(self) -> float:
        """Warming effectiveness: proportion of cache used."""
        if self.warmed_count == 0:
            return 0.0
        return min(1.0, self.cache_hits / self.warmed_count) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize stats to dictionary."""
        return {
            "archetype": self.archetype,
            "warmed_count": self.warmed_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_queries": self.total_queries,
            "hit_rate_percent": round(self.hit_rate, 2),
            "effectiveness_percent": round(self.effectiveness, 2),
            "last_warmed": self.last_warmed,
        }


class ArchetypeEmbeddingLibrary:
    """
    V39.12: Pre-computed embeddings for archetype gesture variations.

    Enables real-time archetype probability distribution for State of Witness.
    Uses predictive cache warming based on performance context.

    The 8 Archetypes:
    - WARRIOR: Aggressive, forceful, high-intensity movements
    - NURTURER: Centered, grounded, protective gestures
    - SAGE: Vertical, deliberate, contemplative poses
    - JESTER: Erratic, unpredictable, high-acceleration
    - LOVER: Fluid, undulating, hip-centric movement
    - MAGICIAN: Precise, complex hand patterns
    - INNOCENT: Light, bouncy, high-energy
    - EVERYMAN: Neutral, balanced, consistent

    Example:
        library = ArchetypeEmbeddingLibrary(layer)
        await library.initialize()

        # Get archetype probabilities from pose embedding
        probs = await library.compute_archetype_probabilities(pose_embedding)
        # probs = {"WARRIOR": 0.15, "SAGE": 0.45, ...}

        # Find dominant archetype
        dominant = library.get_dominant_archetype(probs)
        # dominant = ("SAGE", 0.45)
    """

    def __init__(
        self,
        embedding_layer: "EmbeddingLayer",
        model: EmbeddingModel = EmbeddingModel.VOYAGE_4_LARGE,
    ):
        self.embedding_layer = embedding_layer
        self.model = model
        self._embeddings: dict[str, list[list[float]]] = {}  # archetype -> list of embeddings
        self._centroid_embeddings: dict[str, list[float]] = {}  # archetype -> centroid embedding
        self._initialized = False
        self._stats: dict[str, ArchetypeCacheStats] = {}
        self._batch_job_id: Optional[str] = None

    async def initialize(
        self,
        use_batch: bool = False,
        batch_wait: bool = True,
    ) -> Optional[str]:
        """
        Pre-compute embeddings for all archetype variations.

        V39.12 Feature: Warm cache with archetype gesture descriptions.

        Args:
            use_batch: If True, use Batch API (async, 33% cheaper)
            batch_wait: If True, wait for batch completion

        Returns:
            None if using real-time API or batch_wait=True
            batch_job_id if use_batch=True and batch_wait=False
        """
        if self._initialized:
            return None

        # Collect all descriptions with archetype tracking
        all_descriptions: list[str] = []
        archetype_indices: dict[str, list[int]] = {}

        idx = 0
        for archetype in ARCHETYPE_NAMES:
            variations = ARCHETYPE_VARIATIONS.get(archetype, [])
            archetype_indices[archetype] = list(range(idx, idx + len(variations)))
            all_descriptions.extend(variations)
            idx += len(variations)

        if use_batch:
            # Use Batch API for cost savings
            job = await self.embedding_layer.create_batch_embedding_job(
                texts=all_descriptions,
                model=self.model,
                input_type=InputType.DOCUMENT,
                metadata={"purpose": "archetype_library", "version": "v39.12"},
            )
            self._batch_job_id = job.id

            if batch_wait:
                completed_job = await self.embedding_layer.wait_for_batch_completion(
                    batch_id=job.id,
                    poll_interval=30.0,
                )

                if completed_job.is_successful:
                    results = await self.embedding_layer.download_batch_results(
                        batch_id=job.id,
                        populate_cache=True,
                        original_texts=all_descriptions,
                        input_type=InputType.DOCUMENT,
                    )

                    all_embeddings: list[list[float]] = []
                    for result in results:
                        all_embeddings.extend(result.embeddings)

                    self._distribute_embeddings(all_embeddings, archetype_indices)
                    self._initialized = True
                    return None
                else:
                    raise RuntimeError(f"Batch job failed: {completed_job.status.value}")
            else:
                return job.id
        else:
            # Real-time API
            result = await self.embedding_layer.embed(
                texts=all_descriptions,
                model=self.model,
                input_type=InputType.DOCUMENT,
            )

            self._distribute_embeddings(result.embeddings, archetype_indices)
            self._initialized = True
            return None

    def _distribute_embeddings(
        self,
        embeddings: list[list[float]],
        archetype_indices: dict[str, list[int]],
    ) -> None:
        """Distribute embeddings to archetypes and compute centroids."""
        from datetime import datetime

        for archetype, indices in archetype_indices.items():
            archetype_embeddings = [embeddings[i] for i in indices]
            self._embeddings[archetype] = archetype_embeddings

            # Compute centroid (average embedding)
            if archetype_embeddings:
                dim = len(archetype_embeddings[0])
                centroid = [0.0] * dim
                for emb in archetype_embeddings:
                    for i, val in enumerate(emb):
                        centroid[i] += val
                n = len(archetype_embeddings)
                centroid = [c / n for c in centroid]
                self._centroid_embeddings[archetype] = centroid

            # Initialize stats
            self._stats[archetype] = ArchetypeCacheStats(
                archetype=archetype,
                warmed_count=len(indices),
                last_warmed=datetime.utcnow().isoformat(),
            )

    async def compute_archetype_probabilities(
        self,
        pose_embedding: list[float],
        temperature: float = 1.0,
    ) -> dict[str, float]:
        """
        Compute probability distribution over archetypes for a pose.

        Uses softmax over cosine similarities to centroid embeddings.

        Args:
            pose_embedding: Embedding of current pose/sequence
            temperature: Softmax temperature (higher = more uniform)

        Returns:
            Dictionary mapping archetype names to probabilities
        """
        if not self._initialized:
            await self.initialize()

        import math

        # Compute similarities to each archetype centroid
        similarities: dict[str, float] = {}
        for archetype, centroid in self._centroid_embeddings.items():
            sim = self._cosine_similarity(pose_embedding, centroid)
            similarities[archetype] = sim
            # Update stats
            self._stats[archetype].total_queries += 1
            self._stats[archetype].cache_hits += 1

        # Apply softmax with temperature
        exp_sims = {k: math.exp(v / temperature) for k, v in similarities.items()}
        total = sum(exp_sims.values())

        if total == 0:
            # Uniform distribution fallback
            uniform = 1.0 / len(ARCHETYPE_NAMES)
            return {name: uniform for name in ARCHETYPE_NAMES}

        return {k: v / total for k, v in exp_sims.items()}

    async def classify_to_archetype(
        self,
        pose_embedding: list[float],
        use_variations: bool = True,
        top_k: int = 1,
    ) -> list[tuple[str, float]]:
        """
        Classify a pose to the most similar archetype(s).

        Args:
            pose_embedding: Embedding of current pose
            use_variations: If True, match against all variations (more accurate)
                           If False, match against centroids only (faster)
            top_k: Number of top matches to return

        Returns:
            List of (archetype_name, confidence) tuples
        """
        if not self._initialized:
            await self.initialize()

        if use_variations:
            # Match against all variations, pick best per archetype
            archetype_scores: dict[str, float] = {}
            for archetype, embeddings in self._embeddings.items():
                max_sim = 0.0
                for emb in embeddings:
                    sim = self._cosine_similarity(pose_embedding, emb)
                    max_sim = max(max_sim, sim)
                archetype_scores[archetype] = max_sim
        else:
            # Match against centroids only
            archetype_scores = {}
            for archetype, centroid in self._centroid_embeddings.items():
                archetype_scores[archetype] = self._cosine_similarity(
                    pose_embedding, centroid
                )

        # Sort by score descending
        sorted_scores = sorted(
            archetype_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_scores[:top_k]

    def get_dominant_archetype(
        self,
        probabilities: dict[str, float],
    ) -> tuple[str, float]:
        """Get the archetype with highest probability."""
        if not probabilities:
            return ("EVERYMAN", 0.125)  # Default fallback
        return max(probabilities.items(), key=lambda x: x[1])

    def get_archetype_color(self, archetype: str) -> tuple[int, int, int]:
        """Get RGB color for archetype visualization."""
        return ARCHETYPE_COLORS.get(archetype, (192, 192, 192))

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(ai * bi for ai, bi in zip(a, b))
        norm_a = sum(ai ** 2 for ai in a) ** 0.5
        norm_b = sum(bi ** 2 for bi in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get_cache_stats(self) -> dict[str, ArchetypeCacheStats]:
        """Get per-archetype cache statistics."""
        return self._stats

    def get_aggregate_stats(self) -> dict[str, Any]:
        """Get aggregate statistics across all archetypes."""
        total_warmed = sum(s.warmed_count for s in self._stats.values())
        total_hits = sum(s.cache_hits for s in self._stats.values())
        total_queries = sum(s.total_queries for s in self._stats.values())

        return {
            "total_archetypes": len(ARCHETYPE_NAMES),
            "total_variations_warmed": total_warmed,
            "total_cache_hits": total_hits,
            "total_queries": total_queries,
            "overall_hit_rate": (total_hits / total_queries * 100) if total_queries > 0 else 0.0,
            "per_archetype": {name: stats.to_dict() for name, stats in self._stats.items()},
        }

    def to_dict(self) -> dict:
        """Serialize library for persistence."""
        return {
            "embeddings": dict(self._embeddings),
            "centroids": dict(self._centroid_embeddings),
            "model": self.model.value if hasattr(self.model, 'value') else str(self.model),
            "stats": {name: stats.to_dict() for name, stats in self._stats.items()},
        }

    @classmethod
    async def from_dict(
        cls,
        data: dict,
        embedding_layer: "EmbeddingLayer",
    ) -> "ArchetypeEmbeddingLibrary":
        """Deserialize library from persistence."""
        model = EmbeddingModel(data.get("model", "voyage-4-large"))
        library = cls(embedding_layer, model=model)
        library._embeddings = data.get("embeddings", {})
        library._centroid_embeddings = data.get("centroids", {})
        library._initialized = bool(library._embeddings)

        # Restore stats
        stats_data = data.get("stats", {})
        for name, stat_dict in stats_data.items():
            library._stats[name] = ArchetypeCacheStats(
                archetype=stat_dict.get("archetype", name),
                warmed_count=stat_dict.get("warmed_count", 0),
                cache_hits=stat_dict.get("cache_hits", 0),
                cache_misses=stat_dict.get("cache_misses", 0),
                total_queries=stat_dict.get("total_queries", 0),
                last_warmed=stat_dict.get("last_warmed"),
            )

        return library


async def warm_archetype_cache(
    layer: "EmbeddingLayer",
    archetypes: Optional[list[str]] = None,
    model: EmbeddingModel = EmbeddingModel.VOYAGE_4_LARGE,
) -> dict[str, Any]:
    """
    V39.12: Pre-warm cache with archetype gesture descriptions.

    This is the recommended entry point for cache warming in State of Witness.
    Pre-warms the embedding cache with all archetype variation descriptions,
    reducing API calls during real-time archetype classification.

    Args:
        layer: EmbeddingLayer to warm cache for
        archetypes: Optional subset of archetypes to warm (defaults to all 8)
        model: Embedding model to use

    Returns:
        Dictionary with warming statistics:
        - total_warmed: Total embeddings cached
        - per_archetype: Dict of archetype -> count warmed
        - errors: Any errors encountered
        - estimated_savings: Estimated API calls saved

    Example:
        layer = await get_embedding_layer()
        stats = await warm_archetype_cache(layer)
        print(f"Warmed {stats['total_warmed']} archetype variations")
    """
    target_archetypes = archetypes or ARCHETYPE_NAMES
    results: dict[str, int] = {}
    total_warmed = 0
    errors: list[str] = []

    for archetype in target_archetypes:
        if archetype not in ARCHETYPE_VARIATIONS:
            errors.append(f"Unknown archetype: {archetype}")
            continue

        variations = ARCHETYPE_VARIATIONS[archetype]
        try:
            warm_result = await layer.warm_cache(
                texts=variations,
                model=model,
                input_type=InputType.DOCUMENT,
            )
            warmed = warm_result.get("warmed", 0) + warm_result.get("already_cached", 0)
            results[archetype] = warmed
            total_warmed += warmed
        except Exception as e:
            errors.append(f"{archetype}: {str(e)}")
            results[archetype] = 0

    return {
        "total_warmed": total_warmed,
        "per_archetype": results,
        "errors": errors if errors else None,
        "estimated_api_savings": total_warmed,  # Each cached = 1 API call saved
        "archetypes_warmed": len([a for a in results.values() if a > 0]),
        "total_archetypes": len(ARCHETYPE_NAMES),
    }


# =============================================================================
# V39.12 Phase 2: Streaming Classification Infrastructure
# =============================================================================
# Domain-agnostic streaming classification with temporal smoothing.
# Reusable for ANY real-time classification pipeline.


@dataclass
class StreamingClassificationResult:
    """
    V39.12: Result from streaming classification with temporal context.

    Domain-agnostic container for real-time classification results.
    Works with any CategoryEmbeddingLibrary-based classifier.
    """
    timestamp: float
    raw_probabilities: dict[str, float]
    smoothed_probabilities: dict[str, float]
    dominant_category: str
    dominant_confidence: float
    temporal_stability: float  # How consistent over window
    window_size: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API/protocol transmission."""
        return {
            "timestamp": self.timestamp,
            "raw": self.raw_probabilities,
            "smoothed": self.smoothed_probabilities,
            "dominant": self.dominant_category,
            "confidence": self.dominant_confidence,
            "stability": self.temporal_stability,
            "window": self.window_size,
        }


class StreamingClassifier:
    """
    V39.12: Streaming classification with temporal smoothing.

    Domain-agnostic infrastructure for real-time classification.
    Uses a sliding window (deque) to smooth probabilities over time.

    Example:
        library = ArchetypeEmbeddingLibrary(layer)
        await library.initialize()

        classifier = StreamingClassifier(library, window_size=5)

        async for embedding in embedding_stream:
            result = await classifier.classify(embedding)
            print(f"Category: {result.dominant_category} ({result.dominant_confidence:.2%})")
    """

    def __init__(
        self,
        library: ArchetypeEmbeddingLibrary,
        window_size: int = 5,
        smoothing_factor: float = 0.85,
    ):
        """
        Initialize streaming classifier.

        Args:
            library: Initialized CategoryEmbeddingLibrary
            window_size: Number of frames to keep for temporal smoothing
            smoothing_factor: Exponential smoothing alpha (0-1, higher = more smoothing)
        """
        self.library = library
        self.window_size = window_size
        self.smoothing_factor = smoothing_factor
        self._history: deque = deque(maxlen=window_size)
        self._last_result: Optional[StreamingClassificationResult] = None

    def _compute_smoothed_probabilities(
        self,
        raw_probs: dict[str, float],
    ) -> dict[str, float]:
        """
        Apply temporal smoothing to probabilities.

        Uses exponential moving average with configurable alpha.
        Falls back to raw probabilities if no history.
        """
        if not self._history:
            return raw_probs.copy()

        # Exponential smoothing with history
        smoothed = {}
        for category in raw_probs:
            raw_val = raw_probs[category]

            # Weight recent history more heavily
            weighted_sum = 0.0
            weight_total = 0.0

            for i, hist_probs in enumerate(self._history):
                # More recent = higher weight
                weight = (i + 1) / len(self._history)
                weighted_sum += hist_probs.get(category, 0) * weight
                weight_total += weight

            hist_avg = weighted_sum / weight_total if weight_total > 0 else 0

            # Blend current with history
            smoothed[category] = (
                self.smoothing_factor * hist_avg +
                (1 - self.smoothing_factor) * raw_val
            )

        # Normalize to sum to 1
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v / total for k, v in smoothed.items()}

        return smoothed

    def _compute_temporal_stability(self) -> float:
        """
        Compute how stable the dominant category has been over the window.

        Returns 1.0 if same category dominated entire window, 0.0 if all different.
        """
        if len(self._history) < 2:
            return 1.0

        dominants = []
        for hist_probs in self._history:
            if hist_probs:
                dominant = max(hist_probs, key=lambda k: hist_probs[k])
                dominants.append(dominant)

        if not dominants:
            return 1.0

        # Count most common
        from collections import Counter
        counter = Counter(dominants)
        most_common_count = counter.most_common(1)[0][1]

        return most_common_count / len(dominants)

    async def classify(
        self,
        embedding: list[float],
        timestamp: Optional[float] = None,
    ) -> StreamingClassificationResult:
        """
        Classify embedding with temporal smoothing.

        Args:
            embedding: Input embedding vector
            timestamp: Optional timestamp (defaults to current time)

        Returns:
            StreamingClassificationResult with smoothed probabilities
        """
        import time
        ts = timestamp or time.time()

        # Get raw probabilities from library
        raw_probs = await self.library.compute_archetype_probabilities(embedding)

        # Add to history
        self._history.append(raw_probs)

        # Compute smoothed probabilities
        smoothed = self._compute_smoothed_probabilities(raw_probs)

        # Get dominant from smoothed
        dominant = max(smoothed, key=lambda k: smoothed[k])
        confidence = smoothed[dominant]

        # Compute temporal stability
        stability = self._compute_temporal_stability()

        result = StreamingClassificationResult(
            timestamp=ts,
            raw_probabilities=raw_probs,
            smoothed_probabilities=smoothed,
            dominant_category=dominant,
            dominant_confidence=confidence,
            temporal_stability=stability,
            window_size=len(self._history),
        )

        self._last_result = result
        return result

    async def classify_stream(
        self,
        embedding_stream: AsyncGenerator[tuple[float, list[float]], None],
    ) -> AsyncGenerator[StreamingClassificationResult, None]:
        """
        Classify a stream of embeddings with temporal smoothing.

        This is the main entry point for real-time classification pipelines.

        Args:
            embedding_stream: Async generator yielding (timestamp, embedding) tuples

        Yields:
            StreamingClassificationResult for each input

        Example:
            async def pose_embeddings():
                while True:
                    pose = await get_pose()
                    embedding = await layer.embed([pose.description])
                    yield (time.time(), embedding.embeddings[0])

            async for result in classifier.classify_stream(pose_embeddings()):
                send_to_visualization(result)
        """
        async for timestamp, embedding in embedding_stream:
            yield await self.classify(embedding, timestamp)

    def reset(self):
        """Clear history and reset state."""
        self._history.clear()
        self._last_result = None

    @property
    def history_size(self) -> int:
        """Current number of frames in history."""
        return len(self._history)

    @property
    def last_result(self) -> Optional[StreamingClassificationResult]:
        """Most recent classification result."""
        return self._last_result


# =============================================================================
# V39.13: Classification Enhancement & Metrics
# =============================================================================
# Comprehensive metrics, thresholds, transitions, and adaptive smoothing.
# Builds on V39.12 StreamingClassifier infrastructure.


@dataclass
class ClassificationMetrics:
    """
    V39.13: Track classification performance over time.

    Provides running averages and distributions for monitoring
    classification quality in production pipelines.

    Example:
        metrics = ClassificationMetrics()
        for result in classifications:
            metrics.record(result, latency_ms=15.0)
        print(f"Accuracy: {metrics.high_confidence_rate:.2%}")
    """
    total_classifications: int = 0
    high_confidence_count: int = 0
    low_confidence_count: int = 0
    transitions_detected: int = 0
    _confidence_sum: float = 0.0
    _stability_sum: float = 0.0
    _latency_sum: float = 0.0
    category_distribution: dict[str, int] = field(default_factory=dict)

    # Threshold for "high confidence" (can be overridden)
    high_confidence_threshold: float = 0.7

    def record(
        self,
        result: StreamingClassificationResult,
        latency_ms: float,
        is_transition: bool = False,
    ):
        """
        Record metrics from a classification result.

        Args:
            result: Classification result to record
            latency_ms: Classification latency in milliseconds
            is_transition: Whether a category transition was detected
        """
        self.total_classifications += 1

        # Track confidence
        if result.dominant_confidence >= self.high_confidence_threshold:
            self.high_confidence_count += 1
        else:
            self.low_confidence_count += 1

        # Update running sums
        self._confidence_sum += result.dominant_confidence
        self._stability_sum += result.temporal_stability
        self._latency_sum += latency_ms

        # Track transitions
        if is_transition:
            self.transitions_detected += 1

        # Update distribution
        cat = result.dominant_category
        self.category_distribution[cat] = self.category_distribution.get(cat, 0) + 1

    @property
    def avg_confidence(self) -> float:
        """Average confidence across all classifications."""
        if self.total_classifications == 0:
            return 0.0
        return self._confidence_sum / self.total_classifications

    @property
    def avg_stability(self) -> float:
        """Average temporal stability across all classifications."""
        if self.total_classifications == 0:
            return 0.0
        return self._stability_sum / self.total_classifications

    @property
    def avg_latency_ms(self) -> float:
        """Average classification latency in milliseconds."""
        if self.total_classifications == 0:
            return 0.0
        return self._latency_sum / self.total_classifications

    @property
    def high_confidence_rate(self) -> float:
        """Proportion of high-confidence classifications."""
        if self.total_classifications == 0:
            return 0.0
        return self.high_confidence_count / self.total_classifications

    @property
    def transition_rate(self) -> float:
        """Proportion of classifications that triggered transitions."""
        if self.total_classifications == 0:
            return 0.0
        return self.transitions_detected / self.total_classifications

    def to_dict(self) -> dict[str, Any]:
        """Export metrics for monitoring/logging."""
        return {
            "total": self.total_classifications,
            "high_confidence_count": self.high_confidence_count,
            "high_confidence_rate": round(self.high_confidence_rate, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_stability": round(self.avg_stability, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "transitions": self.transitions_detected,
            "transition_rate": round(self.transition_rate, 4),
            "distribution": self.category_distribution.copy(),
        }

    def reset(self):
        """Reset all metrics to initial state."""
        self.total_classifications = 0
        self.high_confidence_count = 0
        self.low_confidence_count = 0
        self.transitions_detected = 0
        self._confidence_sum = 0.0
        self._stability_sum = 0.0
        self._latency_sum = 0.0
        self.category_distribution = {}


@dataclass
class ClassificationThresholds:
    """
    V39.13: Configurable thresholds for classification filtering.

    Allows filtering out low-confidence or unstable classifications
    to reduce noise in downstream applications.

    Example:
        thresholds = ClassificationThresholds(min_confidence=0.4)
        if thresholds.is_valid(result):
            process(result)
    """
    min_confidence: float = 0.3       # Minimum to accept classification
    high_confidence: float = 0.7      # Mark as high-confidence
    stability_weight: float = 0.3     # Factor stability into filtering (0-1)

    def is_valid(self, result: StreamingClassificationResult) -> bool:
        """
        Check if classification meets minimum threshold.

        Args:
            result: Classification result to validate

        Returns:
            True if classification confidence >= min_confidence
        """
        return result.dominant_confidence >= self.min_confidence

    def is_high_confidence(self, result: StreamingClassificationResult) -> bool:
        """
        Check if classification is high-confidence.

        Uses weighted combination of confidence and stability.

        Args:
            result: Classification result to check

        Returns:
            True if weighted score >= high_confidence threshold
        """
        weighted = (
            result.dominant_confidence * (1 - self.stability_weight) +
            result.temporal_stability * self.stability_weight
        )
        return weighted >= self.high_confidence

    def get_weighted_score(self, result: StreamingClassificationResult) -> float:
        """
        Get the weighted confidence/stability score.

        Args:
            result: Classification result

        Returns:
            Weighted score between 0 and 1
        """
        return (
            result.dominant_confidence * (1 - self.stability_weight) +
            result.temporal_stability * self.stability_weight
        )


@dataclass
class TransitionEvent:
    """
    V39.13: Represents a category transition.

    Emitted when the dominant category changes after
    meeting persistence requirements.

    Example:
        if transition := detector.check_transition(result):
            print(f"Transition: {transition.from_category} -> {transition.to_category}")
    """
    timestamp: float
    from_category: str
    to_category: str
    from_confidence: float
    to_confidence: float
    stability_at_transition: float
    was_stable: bool  # True if previous category was stable

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API/protocol transmission."""
        return {
            "timestamp": self.timestamp,
            "from": self.from_category,
            "to": self.to_category,
            "from_confidence": self.from_confidence,
            "to_confidence": self.to_confidence,
            "confidence_change": self.to_confidence - self.from_confidence,
            "stability": self.stability_at_transition,
            "was_stable": self.was_stable,
        }


class TransitionDetector:
    """
    V39.13: Detect meaningful category transitions.

    Uses persistence filtering to avoid triggering transitions
    on momentary fluctuations. Only emits a transition event when
    the new category persists for a configurable number of frames.

    Example:
        detector = TransitionDetector(persistence_frames=3)
        for result in classification_stream:
            if transition := detector.check_transition(result):
                handle_transition(transition)
    """

    def __init__(
        self,
        stability_threshold: float = 0.6,
        persistence_frames: int = 3,
    ):
        """
        Initialize transition detector.

        Args:
            stability_threshold: Minimum stability to consider "stable"
            persistence_frames: Frames new category must persist before confirmed
        """
        self.stability_threshold = stability_threshold
        self.persistence_frames = persistence_frames

        self._pending_category: Optional[str] = None
        self._pending_count: int = 0
        self._pending_confidence: float = 0.0
        self._last_stable_category: Optional[str] = None
        self._last_stable_confidence: float = 0.0

    def check_transition(
        self,
        result: StreamingClassificationResult,
    ) -> Optional[TransitionEvent]:
        """
        Check if a confirmed transition occurred.

        Args:
            result: Current classification result

        Returns:
            TransitionEvent if confirmed transition, None otherwise
        """
        current = result.dominant_category
        confidence = result.dominant_confidence
        stability = result.temporal_stability

        # First classification - establish baseline
        if self._last_stable_category is None:
            if stability >= self.stability_threshold:
                self._last_stable_category = current
                self._last_stable_confidence = confidence
            return None

        # Same category - reset pending
        if current == self._last_stable_category:
            self._pending_category = None
            self._pending_count = 0
            # Update confidence if still same
            self._last_stable_confidence = confidence
            return None

        # Different category - track persistence
        if current == self._pending_category:
            self._pending_count += 1
            self._pending_confidence = confidence
        else:
            # New pending category
            self._pending_category = current
            self._pending_count = 1
            self._pending_confidence = confidence

        # Check if persistence threshold met
        if self._pending_count >= self.persistence_frames:
            # Confirmed transition
            transition = TransitionEvent(
                timestamp=result.timestamp,
                from_category=self._last_stable_category,
                to_category=current,
                from_confidence=self._last_stable_confidence,
                to_confidence=confidence,
                stability_at_transition=stability,
                was_stable=stability >= self.stability_threshold,
            )

            # Update stable category
            self._last_stable_category = current
            self._last_stable_confidence = confidence
            self._pending_category = None
            self._pending_count = 0

            return transition

        return None

    def reset(self):
        """Reset detector state."""
        self._pending_category = None
        self._pending_count = 0
        self._pending_confidence = 0.0
        self._last_stable_category = None
        self._last_stable_confidence = 0.0

    @property
    def current_category(self) -> Optional[str]:
        """Current stable category."""
        return self._last_stable_category

    @property
    def pending_transition(self) -> Optional[tuple[str, int]]:
        """Pending transition info (category, frame count) or None."""
        if self._pending_category:
            return (self._pending_category, self._pending_count)
        return None


class AdaptiveSmoother:
    """
    V39.13: Dynamically adjust smoothing factor based on stability.

    When classifications are stable, increases smoothing for smoother output.
    When classifications are volatile, decreases smoothing for responsiveness.

    Example:
        smoother = AdaptiveSmoother(base_factor=0.85)
        for result in classification_stream:
            current_factor = smoother.adapt(result.temporal_stability)
            # Use current_factor for next classification
    """

    def __init__(
        self,
        base_factor: float = 0.85,
        min_factor: float = 0.5,
        max_factor: float = 0.95,
        adaptation_rate: float = 0.1,
    ):
        """
        Initialize adaptive smoother.

        Args:
            base_factor: Starting/default smoothing factor
            min_factor: Minimum smoothing (more responsive)
            max_factor: Maximum smoothing (more stable)
            adaptation_rate: How quickly to adapt (0-1)
        """
        self.base_factor = base_factor
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.adaptation_rate = adaptation_rate
        self._current_factor = base_factor

    def adapt(self, stability: float) -> float:
        """
        Adjust smoothing factor based on current stability.

        High stability -> increase smoothing (more stable output)
        Low stability -> decrease smoothing (more responsive)

        Args:
            stability: Current temporal stability (0-1)

        Returns:
            Adapted smoothing factor
        """
        # Target factor based on stability
        target = self.min_factor + stability * (self.max_factor - self.min_factor)

        # Exponential moving toward target
        self._current_factor += (target - self._current_factor) * self.adaptation_rate

        # Clamp to bounds
        self._current_factor = max(self.min_factor, min(self.max_factor, self._current_factor))

        return self._current_factor

    @property
    def current_factor(self) -> float:
        """Current smoothing factor."""
        return self._current_factor

    def reset(self):
        """Reset to base factor."""
        self._current_factor = self.base_factor


@dataclass
class EnhancedClassificationResult:
    """
    V39.13: Enhanced classification result with metrics and metadata.

    Wraps StreamingClassificationResult with additional V39.13 fields
    for filtering, transitions, and performance tracking.

    Example:
        result = await enhanced_classifier.classify(embedding)
        if result.is_valid and result.is_high_confidence:
            if result.transition:
                handle_transition(result.transition)
            display(result)
    """
    base_result: StreamingClassificationResult
    is_valid: bool                              # Meets min confidence
    is_high_confidence: bool                    # Above high confidence threshold
    transition: Optional[TransitionEvent]       # Transition if detected
    current_smoothing_factor: float             # Active smoothing factor
    latency_ms: float                           # Classification latency

    # Delegate common properties to base result
    @property
    def timestamp(self) -> float:
        return self.base_result.timestamp

    @property
    def raw_probabilities(self) -> dict[str, float]:
        return self.base_result.raw_probabilities

    @property
    def smoothed_probabilities(self) -> dict[str, float]:
        return self.base_result.smoothed_probabilities

    @property
    def dominant_category(self) -> str:
        return self.base_result.dominant_category

    @property
    def dominant_confidence(self) -> float:
        return self.base_result.dominant_confidence

    @property
    def temporal_stability(self) -> float:
        return self.base_result.temporal_stability

    @property
    def window_size(self) -> int:
        return self.base_result.window_size

    def to_dict(self) -> dict[str, Any]:
        """Protocol-ready serialization with enhanced fields."""
        data = self.base_result.to_dict()
        data.update({
            "valid": self.is_valid,
            "high_confidence": self.is_high_confidence,
            "smoothing_factor": round(self.current_smoothing_factor, 4),
            "latency_ms": round(self.latency_ms, 2),
            "transition": self.transition.to_dict() if self.transition else None,
        })
        return data


class EnhancedStreamingClassifier:
    """
    V39.13: Enhanced streaming classifier with metrics, thresholds, and transitions.

    Extends V39.12 StreamingClassifier with:
    - Classification metrics tracking
    - Confidence threshold filtering
    - Category transition detection
    - Adaptive smoothing factor

    Example:
        classifier = EnhancedStreamingClassifier(
            library,
            thresholds=ClassificationThresholds(min_confidence=0.4),
            enable_transition_detection=True,
            enable_adaptive_smoothing=True,
        )

        async for result in classifier.classify_stream(embedding_stream):
            if result.is_valid:
                if result.transition:
                    handle_transition(result.transition)
                display(result)

        # Export metrics
        print(classifier.metrics.to_dict())
    """

    def __init__(
        self,
        library: ArchetypeEmbeddingLibrary,
        window_size: int = 5,
        base_smoothing_factor: float = 0.85,
        # V39.13 enhancements
        thresholds: Optional[ClassificationThresholds] = None,
        enable_transition_detection: bool = True,
        enable_adaptive_smoothing: bool = False,
        enable_metrics: bool = True,
    ):
        """
        Initialize enhanced streaming classifier.

        Args:
            library: Initialized ArchetypeEmbeddingLibrary
            window_size: Number of frames for temporal smoothing
            base_smoothing_factor: Starting smoothing factor
            thresholds: Classification thresholds (uses defaults if None)
            enable_transition_detection: Track category transitions
            enable_adaptive_smoothing: Dynamically adjust smoothing
            enable_metrics: Track classification metrics
        """
        self.library = library
        self.window_size = window_size
        self._base_smoothing_factor = base_smoothing_factor
        self._history: deque = deque(maxlen=window_size)
        self._last_result: Optional[EnhancedClassificationResult] = None

        # V39.13 components
        self.thresholds = thresholds or ClassificationThresholds()
        self.transition_detector = TransitionDetector() if enable_transition_detection else None
        self.adaptive_smoother = AdaptiveSmoother(base_smoothing_factor) if enable_adaptive_smoothing else None
        self.metrics = ClassificationMetrics() if enable_metrics else None

    @property
    def current_smoothing_factor(self) -> float:
        """Get current (possibly adapted) smoothing factor."""
        if self.adaptive_smoother:
            return self.adaptive_smoother.current_factor
        return self._base_smoothing_factor

    def _compute_smoothed_probabilities(
        self,
        raw_probs: dict[str, float],
    ) -> dict[str, float]:
        """Apply temporal smoothing using current smoothing factor."""
        if not self._history:
            return raw_probs.copy()

        smoothing = self.current_smoothing_factor
        smoothed = {}

        for category in raw_probs:
            raw_val = raw_probs[category]

            # Weight recent history more heavily
            weighted_sum = 0.0
            weight_total = 0.0

            for i, hist_probs in enumerate(self._history):
                weight = (i + 1) / len(self._history)
                weighted_sum += hist_probs.get(category, 0) * weight
                weight_total += weight

            hist_avg = weighted_sum / weight_total if weight_total > 0 else 0

            # Blend with current smoothing factor
            smoothed[category] = smoothing * hist_avg + (1 - smoothing) * raw_val

        # Normalize
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v / total for k, v in smoothed.items()}

        return smoothed

    def _compute_temporal_stability(self) -> float:
        """Compute how stable the dominant category has been."""
        if len(self._history) < 2:
            return 1.0

        dominants = []
        for hist_probs in self._history:
            if hist_probs:
                dominant = max(hist_probs, key=lambda k: hist_probs[k])
                dominants.append(dominant)

        if not dominants:
            return 1.0

        from collections import Counter
        counter = Counter(dominants)
        most_common_count = counter.most_common(1)[0][1]

        return most_common_count / len(dominants)

    async def classify(
        self,
        embedding: list[float],
        timestamp: Optional[float] = None,
    ) -> EnhancedClassificationResult:
        """
        Classify embedding with enhanced tracking.

        Args:
            embedding: Input embedding vector
            timestamp: Optional timestamp (defaults to current time)

        Returns:
            EnhancedClassificationResult with metrics and filtering info
        """
        import time as time_module
        start_time = time_module.perf_counter()
        ts = timestamp or time_module.time()

        # Get raw probabilities from library
        raw_probs = await self.library.compute_archetype_probabilities(embedding)

        # Add to history
        self._history.append(raw_probs)

        # Compute smoothed probabilities
        smoothed = self._compute_smoothed_probabilities(raw_probs)

        # Get dominant from smoothed
        dominant = max(smoothed, key=lambda k: smoothed[k])
        confidence = smoothed[dominant]

        # Compute temporal stability
        stability = self._compute_temporal_stability()

        # Build base result
        base_result = StreamingClassificationResult(
            timestamp=ts,
            raw_probabilities=raw_probs,
            smoothed_probabilities=smoothed,
            dominant_category=dominant,
            dominant_confidence=confidence,
            temporal_stability=stability,
            window_size=len(self._history),
        )

        # Apply adaptive smoothing if enabled
        if self.adaptive_smoother:
            self.adaptive_smoother.adapt(stability)

        # Check for transitions
        transition = None
        if self.transition_detector:
            transition = self.transition_detector.check_transition(base_result)

        # Calculate latency
        latency_ms = (time_module.perf_counter() - start_time) * 1000

        # Record metrics
        if self.metrics:
            self.metrics.record(base_result, latency_ms, is_transition=(transition is not None))

        # Build enhanced result
        result = EnhancedClassificationResult(
            base_result=base_result,
            is_valid=self.thresholds.is_valid(base_result),
            is_high_confidence=self.thresholds.is_high_confidence(base_result),
            transition=transition,
            current_smoothing_factor=self.current_smoothing_factor,
            latency_ms=latency_ms,
        )

        self._last_result = result
        return result

    async def classify_stream(
        self,
        embedding_stream: AsyncGenerator[tuple[float, list[float]], None],
    ) -> AsyncGenerator[EnhancedClassificationResult, None]:
        """
        Classify a stream of embeddings with enhanced tracking.

        Args:
            embedding_stream: Async generator yielding (timestamp, embedding) tuples

        Yields:
            EnhancedClassificationResult for each input
        """
        async for timestamp, embedding in embedding_stream:
            yield await self.classify(embedding, timestamp)

    def reset(self):
        """Clear history and reset all state."""
        self._history.clear()
        self._last_result = None
        if self.transition_detector:
            self.transition_detector.reset()
        if self.adaptive_smoother:
            self.adaptive_smoother.reset()
        if self.metrics:
            self.metrics.reset()

    @property
    def history_size(self) -> int:
        """Current number of frames in history."""
        return len(self._history)

    @property
    def last_result(self) -> Optional[EnhancedClassificationResult]:
        """Most recent classification result."""
        return self._last_result


# =============================================================================
# Factory Functions
# =============================================================================

# Module-level singleton
_default_layer: Optional[EmbeddingLayer] = None


def create_embedding_layer(
    model: str = EmbeddingModel.VOYAGE_4_LARGE.value,
    api_key: Optional[str] = None,
    cache_enabled: bool = True,
    **kwargs,
) -> EmbeddingLayer:
    """
    Factory function to create embedding layer.

    Args:
        model: Voyage AI model to use
        api_key: API key (uses env var or default if not provided)
        cache_enabled: Enable embedding cache
        **kwargs: Additional config options

    Returns:
        Configured EmbeddingLayer
    """
    config = EmbeddingConfig(
        model=model,
        api_key=api_key,
        cache_enabled=cache_enabled,
        **kwargs,
    )
    return EmbeddingLayer(config)


def create_model_mixing_layer(
    document_model: str = EmbeddingModel.VOYAGE_4_LARGE.value,
    query_model: str = EmbeddingModel.VOYAGE_4_LITE.value,
    api_key: Optional[str] = None,
    **kwargs,
) -> EmbeddingLayer:
    """
    Factory function for model mixing (Voyage 4 shared embedding space).

    Uses a high-quality model for documents and a lighter model for queries.
    This reduces query latency and costs by ~40% while maintaining retrieval quality.

    Best combinations:
    - voyage-4-large + voyage-4-lite: Best quality, ~40% cheaper queries
    - voyage-4 + voyage-4-lite: Good quality, ~25% cheaper queries

    Args:
        document_model: Model for embedding documents (stored/indexed)
        query_model: Model for embedding queries (faster, cheaper)
        api_key: API key (uses env var or default if not provided)
        **kwargs: Additional config options

    Returns:
        Configured EmbeddingLayer with model mixing

    Example:
        >>> layer = create_model_mixing_layer()
        >>> await layer.initialize()
        >>> # Documents use voyage-4-large
        >>> docs = await layer.embed_documents(["doc1", "doc2"])
        >>> # Queries use voyage-4-lite (faster, cheaper)
        >>> query = await layer.embed_query("search term")
    """
    config = EmbeddingConfig(
        model=document_model,
        query_model=query_model,
        api_key=api_key,
        **kwargs,
    )
    return EmbeddingLayer(config)


async def get_embedding_layer() -> EmbeddingLayer:
    """
    Get the default embedding layer singleton.

    Returns:
        Initialized EmbeddingLayer

    Raises:
        SDKNotAvailableError: If voyageai not installed
    """
    global _default_layer
    if _default_layer is None:
        _default_layer = create_embedding_layer()
        await _default_layer.initialize()
    return _default_layer


# =============================================================================
# Convenience Functions
# =============================================================================

async def embed_texts(
    texts: Sequence[str],
    model: Optional[str] = None,
    input_type: str = "document",
) -> list[list[float]]:
    """
    Quick embedding function using default layer.

    Args:
        texts: Texts to embed
        model: Override default model
        input_type: "document" or "query"

    Returns:
        List of embedding vectors
    """
    layer = await get_embedding_layer()
    it = InputType(input_type) if isinstance(input_type, str) else input_type
    result = await layer.embed(texts, input_type=it, model=model)
    return result.embeddings


async def embed_for_search(
    query: str,
    documents: Sequence[str],
    model: Optional[str] = None,
) -> tuple[list[float], list[list[float]]]:
    """
    Embed query and documents with correct input types for search.

    Args:
        query: Search query
        documents: Documents to search over
        model: Override default model

    Returns:
        (query_embedding, document_embeddings)
    """
    layer = await get_embedding_layer()

    # Embed query with "query" type
    query_result = await layer.embed(
        [query],
        input_type=InputType.QUERY,
        model=model,
    )

    # Embed documents with "document" type
    doc_result = await layer.embed(
        documents,
        input_type=InputType.DOCUMENT,
        model=model,
    )

    return query_result.embeddings[0], doc_result.embeddings


def get_embedding_layer_sync() -> EmbeddingLayer:
    """
    Get embedding layer synchronously (for non-async contexts).
    Note: Layer will not be initialized - call initialize() before use.

    Returns:
        EmbeddingLayer (not initialized)
    """
    global _default_layer
    if _default_layer is None:
        _default_layer = create_embedding_layer()
    return _default_layer


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Availability
    "VOYAGE_AVAILABLE",
    "HTTPX_AVAILABLE",  # V38.0: HTTP client for advanced features

    # Classes
    "EmbeddingLayer",
    "EmbeddingConfig",
    "EmbeddingResult",
    "RerankResult",  # V37.0: Reranking result
    "CacheEntry",
    "CacheStats",
    "RateLimiter",

    # Enums
    "EmbeddingModel",
    "RerankModel",  # V37.0: Reranking models
    "InputType",
    "OutputDType",  # V38.0: Quantization types
    "OutputDimension",  # V38.0: Output dimensions
    "BatchStatus",  # V39.7: Batch job status lifecycle

    # V39.7: Batch API Classes
    "BatchRequestCounts",  # V39.7: Batch request counts tracking
    "BatchJob",  # V39.7: Batch job representation
    "BatchFile",  # V39.7: Uploaded batch file metadata

    # Factory
    "create_embedding_layer",
    "create_model_mixing_layer",  # V36.0: Model mixing factory
    "get_embedding_layer",
    "get_embedding_layer_sync",

    # Convenience
    "embed_texts",
    "embed_for_search",

    # V39.12: Archetype-Based Cache Warming (Phase 1)
    "ARCHETYPE_NAMES",  # V39.12: List of 8 archetypes
    "ARCHETYPE_VARIATIONS",  # V39.12: Gesture descriptions per archetype
    "ARCHETYPE_COLORS",  # V39.12: RGB colors for visualization
    "ArchetypeCacheStats",  # V39.12: Per-archetype cache statistics
    "ArchetypeEmbeddingLibrary",  # V39.12: Archetype gesture library
    "warm_archetype_cache",  # V39.12: Pre-warm cache with archetypes

    # V39.12: Streaming Classification Infrastructure (Phase 2)
    "StreamingClassificationResult",  # V39.12: Result with temporal context
    "StreamingClassifier",  # V39.12: Streaming classifier with smoothing

    # V39.13: Classification Enhancement & Metrics Infrastructure
    "ClassificationMetrics",  # V39.13: Track classification performance
    "ClassificationThresholds",  # V39.13: Configurable filtering thresholds
    "TransitionEvent",  # V39.13: Category transition event dataclass
    "TransitionDetector",  # V39.13: Detect meaningful category transitions
    "AdaptiveSmoother",  # V39.13: Dynamic smoothing factor adjustment
    "EnhancedClassificationResult",  # V39.13: Rich result with metrics
    "EnhancedStreamingClassifier",  # V39.13: Full enhanced classifier
]


if __name__ == "__main__":
    async def main():
        """Test the embedding layer."""
        print("Embedding Layer Status")
        print("=" * 40)
        print(f"  Voyage AI Available: {VOYAGE_AVAILABLE}")

        if not VOYAGE_AVAILABLE:
            print("\n  Install with: pip install voyageai")
            return

        # Create and initialize layer
        layer = create_embedding_layer()
        await layer.initialize()

        print(f"  Initialized: {layer.is_initialized}")
        print(f"  Model: {layer.config.model}")

        # Test embeddings
        texts = [
            "Hello, this is a test document.",
            "Another document for testing embeddings.",
        ]

        result = await layer.embed_documents(texts)
        print(f"\nEmbedded {result.count} documents")
        print(f"  Dimension: {result.dimension}")
        print(f"  Cached: {result.cached_count}")

        # Test query
        query_emb = await layer.embed_query("test query")
        print(f"\nQuery embedding dimension: {len(query_emb)}")

        # Test code
        code = [
            "def hello():\n    return 'world'",
            "async function fetchData() { return await fetch(url); }",
        ]
        code_result = await layer.embed_code(code)
        print(f"\nCode embeddings dimension: {code_result.dimension}")

        print(f"\nStats: {layer.get_stats()}")

    asyncio.run(main())
