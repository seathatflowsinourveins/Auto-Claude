"""
Token Optimization Layer - UNLEASH Platform
============================================

Implements comprehensive token optimization strategies for cost efficiency
without degrading quality. Inspired by everything-claude-code patterns.

Key Components:
1. Context Compression - Intelligent context pruning with semantic preservation
2. Prompt Caching - LRU cache for repeated queries
3. Response Streaming - Progressive output for better UX
4. Batch Processing - Group similar requests
5. Cost Tracking Dashboard - Real-time monitoring

Research Sources:
- everything-claude-code: Continuous learning, strategic compaction
- Anthropic Prompt Caching: 90% cost reduction for cached prompts
- SimpleMem: Semantic lossless compression patterns

Version: 1.0.0 (2026-01-30)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Generic, Iterator, Optional, TypeVar

logger = logging.getLogger(__name__)


# =============================================================================
# COMPRESSION STRATEGIES
# =============================================================================

class CompressionLevel(str, Enum):
    """Compression aggressiveness levels."""
    NONE = "none"           # No compression
    LIGHT = "light"         # Remove obvious redundancy (~10-20% reduction)
    MODERATE = "moderate"   # Structured summarization (~30-50% reduction)
    AGGRESSIVE = "aggressive"  # Heavy summarization (~50-70% reduction)
    EXTREME = "extreme"     # Maximum compression (~70-90% reduction)


@dataclass
class CompressionResult:
    """Result of context compression."""
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    content: str
    strategy_used: str
    quality_score: float  # 0-1, estimated quality preservation


class CompressionStrategy(ABC):
    """Base class for compression strategies."""

    @abstractmethod
    def compress(self, content: str, target_tokens: Optional[int] = None) -> CompressionResult:
        """Compress content."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass


class WhitespaceCompression(CompressionStrategy):
    """Remove excessive whitespace while preserving structure."""

    def name(self) -> str:
        return "whitespace"

    def compress(self, content: str, target_tokens: Optional[int] = None) -> CompressionResult:
        _ = target_tokens  # Not used for this strategy
        original_tokens = len(content) // 4

        # Normalize whitespace
        import re
        compressed = re.sub(r'\n{3,}', '\n\n', content)  # Max 2 newlines
        compressed = re.sub(r'[ \t]+', ' ', compressed)  # Single spaces
        compressed = re.sub(r' +\n', '\n', compressed)   # No trailing spaces

        compressed_tokens = len(compressed) // 4

        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / max(original_tokens, 1),
            content=compressed,
            strategy_used=self.name(),
            quality_score=0.99  # Almost lossless
        )


class CodeBlockCompression(CompressionStrategy):
    """Compress code blocks by removing comments and normalizing."""

    def name(self) -> str:
        return "code_block"

    def compress(self, content: str, target_tokens: Optional[int] = None) -> CompressionResult:
        _ = target_tokens
        original_tokens = len(content) // 4

        import re

        # Process code blocks
        def compress_code(match: re.Match[str]) -> str:
            lang = match.group(1) or ""
            code = match.group(2)

            # Remove single-line comments (be careful with strings)
            # This is a simplified version - production would use AST
            lines = code.split('\n')
            compressed_lines = []
            for line in lines:
                stripped = line.strip()
                # Skip pure comment lines (simplified)
                if stripped.startswith('#') and '"""' not in line and "'''" not in line:
                    continue
                if stripped.startswith('//'):
                    continue
                # Keep empty lines only if next to non-empty
                if stripped or (compressed_lines and compressed_lines[-1].strip()):
                    compressed_lines.append(line)

            return f"```{lang}\n{chr(10).join(compressed_lines)}\n```"

        compressed = re.sub(
            r'```(\w*)\n(.*?)\n```',
            compress_code,
            content,
            flags=re.DOTALL
        )

        compressed_tokens = len(compressed) // 4

        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / max(original_tokens, 1),
            content=compressed,
            strategy_used=self.name(),
            quality_score=0.95
        )


class SemanticCompression(CompressionStrategy):
    """
    Semantic compression using structured summarization.

    Preserves:
    - Key facts and decisions
    - Code patterns and signatures
    - Error messages and solutions

    Removes:
    - Verbose explanations
    - Repeated context
    - Conversational filler
    """

    def name(self) -> str:
        return "semantic"

    def compress(self, content: str, target_tokens: Optional[int] = None) -> CompressionResult:
        original_tokens = len(content) // 4

        # Split into semantic sections
        sections = self._identify_sections(content)

        # Score each section by importance
        scored_sections = [
            (section, self._importance_score(section))
            for section in sections
        ]

        # Sort by importance (descending)
        scored_sections.sort(key=lambda x: x[1], reverse=True)

        # Keep most important sections up to target
        if target_tokens:
            kept_content = []
            current_tokens = 0
            for section, score in scored_sections:
                section_tokens = len(section) // 4
                if current_tokens + section_tokens <= target_tokens:
                    kept_content.append(section)
                    current_tokens += section_tokens
                elif score > 0.8:  # Always keep very important content
                    kept_content.append(section)
                    current_tokens += section_tokens
            compressed = '\n\n'.join(kept_content)
        else:
            # Just remove low-importance sections
            compressed = '\n\n'.join(
                section for section, score in scored_sections
                if score > 0.3
            )

        compressed_tokens = len(compressed) // 4

        # Calculate quality score based on kept importance
        total_importance = sum(score for _, score in scored_sections)
        kept_importance = sum(
            score for section, score in scored_sections
            if section in compressed
        )
        quality_score = kept_importance / max(total_importance, 1)

        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / max(original_tokens, 1),
            content=compressed,
            strategy_used=self.name(),
            quality_score=quality_score
        )

    def _identify_sections(self, content: str) -> list[str]:
        """Split content into semantic sections."""
        import re

        # Split on headers, code blocks, paragraph breaks
        sections = re.split(r'\n(?=#{1,3} |\n\n|```)', content)
        return [s.strip() for s in sections if s.strip()]

    def _importance_score(self, section: str) -> float:
        """Score section importance (0-1)."""
        score = 0.5  # Base score

        # Code blocks are important
        if '```' in section:
            score += 0.3

        # Error messages are important
        if any(kw in section.lower() for kw in ['error', 'exception', 'fail', 'bug']):
            score += 0.2

        # Headers indicate structure
        if section.startswith('#'):
            score += 0.1

        # Short sections might be key points
        if len(section) < 200:
            score += 0.1

        # Very long sections might be verbose
        if len(section) > 2000:
            score -= 0.2

        # Decisions and actions
        if any(kw in section.lower() for kw in ['decided', 'chosen', 'implemented', 'created', 'fixed']):
            score += 0.15

        return min(1.0, max(0.0, score))


class ContextCompressor:
    """
    Main context compression orchestrator.

    Applies multiple strategies based on compression level.
    """

    def __init__(self):
        self.strategies: dict[CompressionLevel, list[CompressionStrategy]] = {
            CompressionLevel.NONE: [],
            CompressionLevel.LIGHT: [WhitespaceCompression()],
            CompressionLevel.MODERATE: [WhitespaceCompression(), CodeBlockCompression()],
            CompressionLevel.AGGRESSIVE: [
                WhitespaceCompression(),
                CodeBlockCompression(),
                SemanticCompression()
            ],
            CompressionLevel.EXTREME: [
                WhitespaceCompression(),
                CodeBlockCompression(),
                SemanticCompression()
            ]
        }

    def compress(
        self,
        content: str,
        level: CompressionLevel = CompressionLevel.MODERATE,
        target_tokens: Optional[int] = None
    ) -> CompressionResult:
        """
        Compress content using specified level.

        Args:
            content: Text to compress
            level: How aggressive to be
            target_tokens: Optional target token count

        Returns:
            CompressionResult with compressed content
        """
        if level == CompressionLevel.NONE:
            tokens = len(content) // 4
            return CompressionResult(
                original_tokens=tokens,
                compressed_tokens=tokens,
                compression_ratio=1.0,
                content=content,
                strategy_used="none",
                quality_score=1.0
            )

        result = CompressionResult(
            original_tokens=len(content) // 4,
            compressed_tokens=len(content) // 4,
            compression_ratio=1.0,
            content=content,
            strategy_used="",
            quality_score=1.0
        )

        strategies_used = []
        for strategy in self.strategies[level]:
            result = strategy.compress(result.content, target_tokens)
            strategies_used.append(strategy.name())

            # Check if we've hit target
            if target_tokens and result.compressed_tokens <= target_tokens:
                break

        # Update with combined strategy info
        result.strategy_used = "+".join(strategies_used)
        result.original_tokens = len(content) // 4
        result.compression_ratio = result.compressed_tokens / max(result.original_tokens, 1)

        return result


# =============================================================================
# PROMPT CACHING
# =============================================================================

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Entry in the prompt cache."""
    key: str
    value: T
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1


class LRUCache(Generic[T]):
    """
    LRU cache with TTL support.

    Used for caching:
    - Prompt embeddings
    - Common responses
    - Research results
    """

    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        if key not in self._cache:
            self._misses += 1
            return None

        entry = self._cache[key]

        # Check expiration
        if entry.is_expired():
            del self._cache[key]
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.touch()
        self._hits += 1

        return entry.value

    def put(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Put value in cache."""
        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # Remove oldest

        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(timezone.utc),
            last_accessed=datetime.now(timezone.utc),
            ttl_seconds=ttl or self.default_ttl
        )
        self._cache[key] = entry

    def invalidate(self, key: str) -> bool:
        """Remove entry from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all entries."""
        self._cache.clear()

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "default_ttl": self.default_ttl
        }


class PromptCache:
    """
    Specialized cache for prompt/response pairs.

    Supports:
    - Exact match caching
    - Semantic similarity matching (with embeddings)
    - Prefix caching for Anthropic API
    """

    def __init__(
        self,
        max_entries: int = 500,
        ttl_seconds: int = 3600,
        enable_semantic: bool = False
    ):
        self._exact_cache: LRUCache[str] = LRUCache(max_entries, ttl_seconds)
        self._prefix_cache: LRUCache[str] = LRUCache(max_entries // 2, ttl_seconds * 2)
        self.enable_semantic = enable_semantic
        self._embeddings_cache: dict[str, list[float]] = {}

    def _hash_prompt(self, prompt: str) -> str:
        """Create hash for prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    def _extract_prefix(self, prompt: str, length: int = 1024) -> str:
        """Extract cacheable prefix from prompt."""
        # For Anthropic API, system prompts and static context can be cached
        # Extract the first N tokens worth of content
        return prompt[:length * 4]  # Approximate chars

    def get(self, prompt: str) -> Optional[str]:
        """Get cached response for prompt."""
        # Try exact match first
        key = self._hash_prompt(prompt)
        cached = self._exact_cache.get(key)
        if cached:
            logger.debug(f"Prompt cache hit (exact): {key[:8]}...")
            return cached

        return None

    def get_with_prefix(self, prompt: str) -> tuple[Optional[str], bool]:
        """
        Get cached response, with prefix cache indicator.

        Returns:
            (response, is_prefix_cached): Response and whether prefix was cached
        """
        # Try exact match
        exact = self.get(prompt)
        if exact:
            return exact, False

        # Check prefix cache
        prefix = self._extract_prefix(prompt)
        prefix_key = self._hash_prompt(prefix)
        if self._prefix_cache.get(prefix_key):
            logger.debug(f"Prompt cache hit (prefix): {prefix_key[:8]}...")
            return None, True  # Prefix is cached, but need full response

        return None, False

    def put(self, prompt: str, response: str) -> None:
        """Cache prompt-response pair."""
        key = self._hash_prompt(prompt)
        self._exact_cache.put(key, response)

        # Also cache the prefix
        prefix = self._extract_prefix(prompt)
        prefix_key = self._hash_prompt(prefix)
        self._prefix_cache.put(prefix_key, prefix)

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "exact_cache": self._exact_cache.stats(),
            "prefix_cache": self._prefix_cache.stats(),
            "semantic_enabled": self.enable_semantic,
            "embeddings_count": len(self._embeddings_cache)
        }


# =============================================================================
# RESPONSE STREAMING
# =============================================================================

@dataclass
class StreamChunk:
    """A chunk of streamed response."""
    content: str
    index: int
    is_final: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tokens_so_far: int = 0


class ResponseStreamer:
    """
    Stream responses for better UX and early termination.

    Benefits:
    - User sees results immediately
    - Can stop generation early if off-track
    - Reduces perceived latency
    """

    def __init__(self, chunk_size: int = 100):
        self.chunk_size = chunk_size
        self._buffer: list[str] = []
        self._chunk_index = 0
        self._total_tokens = 0
        self._callbacks: list[Callable[[StreamChunk], None]] = []

    def register_callback(self, callback: Callable[[StreamChunk], None]) -> None:
        """Register callback for new chunks."""
        self._callbacks.append(callback)

    def _emit_chunk(self, content: str, is_final: bool) -> StreamChunk:
        """Emit a chunk to all callbacks."""
        chunk = StreamChunk(
            content=content,
            index=self._chunk_index,
            is_final=is_final,
            tokens_so_far=self._total_tokens
        )
        self._chunk_index += 1

        for callback in self._callbacks:
            try:
                callback(chunk)
            except Exception as e:
                logger.error(f"Stream callback error: {e}")

        return chunk

    def stream_text(self, text: str) -> Iterator[StreamChunk]:
        """
        Stream text in chunks.

        Yields chunks as they're "generated" (for simulation).
        In production, integrate with actual API streaming.
        """
        self._chunk_index = 0
        self._total_tokens = 0

        # Split into sentences or chunks
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for i, sentence in enumerate(sentences):
            is_final = (i == len(sentences) - 1)
            self._total_tokens += len(sentence) // 4

            chunk = self._emit_chunk(sentence, is_final)
            yield chunk

    async def stream_text_async(self, text: str) -> AsyncIterator[StreamChunk]:
        """Async version of stream_text."""
        for chunk in self.stream_text(text):
            yield chunk
            await asyncio.sleep(0)  # Yield to event loop


# Make async iterator protocol work
from typing import AsyncIterator


# =============================================================================
# BATCH PROCESSING
# =============================================================================

@dataclass
class BatchRequest:
    """A request in a batch."""
    id: str
    prompt: str
    priority: int = 0  # Higher = more urgent
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    callback: Optional[Callable[[str], None]] = None


@dataclass
class BatchResult:
    """Result of batch processing."""
    request_id: str
    response: str
    tokens_used: int
    processing_time_ms: float


class BatchProcessor:
    """
    Batch similar requests for efficiency.

    Groups requests by:
    - Similar prompt prefixes
    - Same task type
    - Common context

    Benefits:
    - Reduced API calls
    - Better throughput
    - Cost savings from batching
    """

    def __init__(
        self,
        batch_size: int = 5,
        max_wait_ms: int = 500,
        group_by_prefix: bool = True
    ):
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.group_by_prefix = group_by_prefix
        self._queue: list[BatchRequest] = []
        self._processing = False
        self._results: dict[str, BatchResult] = {}

    async def add_request(self, request: BatchRequest) -> str:
        """Add request to batch queue. Returns request ID."""
        self._queue.append(request)

        # Auto-trigger if batch is full
        if len(self._queue) >= self.batch_size:
            await self._process_batch()

        return request.id

    async def wait_for_result(self, request_id: str, timeout_ms: int = 5000) -> Optional[BatchResult]:
        """Wait for result of a request."""
        start = time.time()
        while (time.time() - start) * 1000 < timeout_ms:
            if request_id in self._results:
                return self._results.pop(request_id)
            await asyncio.sleep(0.05)
        return None

    async def _process_batch(self) -> None:
        """Process current batch."""
        if self._processing or not self._queue:
            return

        self._processing = True
        batch = self._queue[:self.batch_size]
        self._queue = self._queue[self.batch_size:]

        try:
            # Group by prefix similarity if enabled
            if self.group_by_prefix:
                groups = self._group_by_prefix(batch)
            else:
                groups = [batch]

            for group in groups:
                results = await self._process_group(group)
                for result in results:
                    self._results[result.request_id] = result

                    # Call callbacks
                    request = next((r for r in group if r.id == result.request_id), None)
                    if request and request.callback:
                        request.callback(result.response)
        finally:
            self._processing = False

    def _group_by_prefix(self, requests: list[BatchRequest]) -> list[list[BatchRequest]]:
        """Group requests by common prefix."""
        groups: dict[str, list[BatchRequest]] = {}

        for request in requests:
            # Use first 256 chars as grouping key
            prefix = request.prompt[:256]
            key = hashlib.md5(prefix.encode()).hexdigest()[:8]

            if key not in groups:
                groups[key] = []
            groups[key].append(request)

        return list(groups.values())

    async def _process_group(self, requests: list[BatchRequest]) -> list[BatchResult]:
        """Process a group of similar requests."""
        results = []
        start = time.time()

        for request in requests:
            # In production, this would batch to the actual API
            # For now, simulate processing
            response = f"[Batched response for: {request.prompt[:50]}...]"

            results.append(BatchResult(
                request_id=request.id,
                response=response,
                tokens_used=len(response) // 4,
                processing_time_ms=(time.time() - start) * 1000
            ))

        return results


# =============================================================================
# COST TRACKING
# =============================================================================

@dataclass
class TokenUsage:
    """Token usage for a single operation."""
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    model: str = "unknown"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost(self, rates: dict[str, dict[str, float]]) -> float:
        """Calculate cost based on model rates."""
        if self.model not in rates:
            return 0.0

        model_rates = rates[self.model]
        input_cost = (self.input_tokens - self.cached_tokens) * model_rates.get("input", 0)
        cached_cost = self.cached_tokens * model_rates.get("cached", 0)
        output_cost = self.output_tokens * model_rates.get("output", 0)

        return (input_cost + cached_cost + output_cost) / 1_000_000  # Per million tokens


class CostTracker:
    """
    Track token usage and costs in real-time.

    Features:
    - Per-model tracking
    - Budget alerts
    - Usage patterns
    - Cost projections
    """

    # Model rates per million tokens (January 2026)
    DEFAULT_RATES: dict[str, dict[str, float]] = {
        "claude-3-opus": {"input": 15.0, "output": 75.0, "cached": 1.5},
        "claude-3-sonnet": {"input": 3.0, "output": 15.0, "cached": 0.3},
        "claude-3-haiku": {"input": 0.25, "output": 1.25, "cached": 0.025},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0, "cached": 1.0},
        "gpt-4o": {"input": 5.0, "output": 15.0, "cached": 0.5},
        "ollama-local": {"input": 0.0, "output": 0.0, "cached": 0.0},
    }

    def __init__(
        self,
        daily_budget: float = 50.0,
        alert_threshold: float = 0.8,
        rates: Optional[dict[str, dict[str, float]]] = None
    ):
        self.daily_budget = daily_budget
        self.alert_threshold = alert_threshold
        self.rates = rates or self.DEFAULT_RATES

        self._usage_history: list[TokenUsage] = []
        self._daily_totals: dict[str, float] = {}  # date -> cost
        self._alert_callbacks: list[Callable[[str, float], None]] = []

    def track(self, usage: TokenUsage) -> float:
        """Track token usage. Returns cost."""
        self._usage_history.append(usage)

        cost = usage.cost(self.rates)

        # Update daily total
        date_key = usage.timestamp.strftime("%Y-%m-%d")
        self._daily_totals[date_key] = self._daily_totals.get(date_key, 0) + cost

        # Check budget
        if self._daily_totals[date_key] >= self.daily_budget * self.alert_threshold:
            self._trigger_alert(date_key, self._daily_totals[date_key])

        return cost

    def register_alert_callback(self, callback: Callable[[str, float], None]) -> None:
        """Register callback for budget alerts."""
        self._alert_callbacks.append(callback)

    def _trigger_alert(self, date: str, current_spend: float) -> None:
        """Trigger budget alert."""
        for callback in self._alert_callbacks:
            try:
                callback(date, current_spend)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        logger.warning(
            f"Budget alert: ${current_spend:.2f} of ${self.daily_budget:.2f} "
            f"({current_spend/self.daily_budget*100:.1f}%) on {date}"
        )

    def get_daily_spend(self, date: Optional[str] = None) -> float:
        """Get spending for a date (default: today)."""
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._daily_totals.get(date, 0.0)

    def get_usage_stats(self, hours: int = 24) -> dict[str, Any]:
        """Get usage statistics for recent period."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent = [u for u in self._usage_history if u.timestamp > cutoff]

        if not recent:
            return {
                "period_hours": hours,
                "total_tokens": 0,
                "total_cost": 0.0,
                "by_model": {}
            }

        by_model: dict[str, dict[str, Any]] = {}
        total_tokens = 0
        total_cost = 0.0

        for usage in recent:
            total_tokens += usage.total_tokens
            cost = usage.cost(self.rates)
            total_cost += cost

            if usage.model not in by_model:
                by_model[usage.model] = {
                    "tokens": 0,
                    "cost": 0.0,
                    "cached_tokens": 0,
                    "calls": 0
                }

            by_model[usage.model]["tokens"] += usage.total_tokens
            by_model[usage.model]["cost"] += cost
            by_model[usage.model]["cached_tokens"] += usage.cached_tokens
            by_model[usage.model]["calls"] += 1

        return {
            "period_hours": hours,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "by_model": by_model,
            "cache_savings": sum(
                u.cached_tokens * (self.rates.get(u.model, {}).get("input", 0) -
                                   self.rates.get(u.model, {}).get("cached", 0))
                for u in recent
            ) / 1_000_000
        }

    def project_daily_cost(self) -> float:
        """Project daily cost based on current usage pattern."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        current_spend = self._daily_totals.get(today, 0.0)

        # Calculate hours elapsed today
        now = datetime.now(timezone.utc)
        hours_elapsed = now.hour + now.minute / 60

        if hours_elapsed < 1:
            return current_spend

        # Project to 24 hours
        return current_spend * (24 / hours_elapsed)


# =============================================================================
# UNIFIED TOKEN OPTIMIZER
# =============================================================================

class TokenOptimizer:
    """
    Unified token optimization layer.

    Combines all optimization strategies:
    - Context compression
    - Prompt caching
    - Response streaming
    - Batch processing
    - Cost tracking

    Usage:
        optimizer = TokenOptimizer(daily_budget=50.0)

        # Compress context
        compressed = optimizer.compress(large_context, CompressionLevel.MODERATE)

        # Check cache
        cached_response = optimizer.cache.get(prompt)
        if cached_response:
            return cached_response

        # Process and track
        response = await model.generate(compressed.content)
        optimizer.track_usage(input_tokens, output_tokens, model="claude-3-sonnet")
        optimizer.cache.put(prompt, response)
    """

    def __init__(
        self,
        daily_budget: float = 50.0,
        cache_size: int = 500,
        batch_size: int = 5,
        enable_streaming: bool = True
    ):
        self.compressor = ContextCompressor()
        self.cache = PromptCache(max_entries=cache_size)
        self.streamer = ResponseStreamer() if enable_streaming else None
        self.batcher = BatchProcessor(batch_size=batch_size)
        self.cost_tracker = CostTracker(daily_budget=daily_budget)

        self._compression_stats: list[CompressionResult] = []

    def compress(
        self,
        content: str,
        level: CompressionLevel = CompressionLevel.MODERATE,
        target_tokens: Optional[int] = None
    ) -> CompressionResult:
        """Compress content using specified level."""
        result = self.compressor.compress(content, level, target_tokens)
        self._compression_stats.append(result)

        if result.compression_ratio < 0.5:
            logger.info(
                f"Compressed {result.original_tokens} → {result.compressed_tokens} tokens "
                f"({result.compression_ratio:.1%}) using {result.strategy_used}"
            )

        return result

    def track_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cached_tokens: int = 0
    ) -> float:
        """Track token usage. Returns cost."""
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            model=model
        )
        return self.cost_tracker.track(usage)

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive optimization statistics."""
        # Compression stats
        if self._compression_stats:
            total_original = sum(r.original_tokens for r in self._compression_stats)
            total_compressed = sum(r.compressed_tokens for r in self._compression_stats)
            avg_ratio = total_compressed / max(total_original, 1)
        else:
            total_original = 0
            total_compressed = 0
            avg_ratio = 1.0

        return {
            "compression": {
                "operations": len(self._compression_stats),
                "total_original_tokens": total_original,
                "total_compressed_tokens": total_compressed,
                "average_ratio": avg_ratio,
                "tokens_saved": total_original - total_compressed
            },
            "cache": self.cache.stats(),
            "cost": self.cost_tracker.get_usage_stats(24),
            "daily_spend": self.cost_tracker.get_daily_spend(),
            "projected_daily": self.cost_tracker.project_daily_cost(),
            "budget_remaining": self.cost_tracker.daily_budget - self.cost_tracker.get_daily_spend()
        }

    def should_use_cheaper_model(self) -> bool:
        """Check if we should prefer cheaper models due to budget."""
        spend_ratio = self.cost_tracker.get_daily_spend() / self.cost_tracker.daily_budget
        return spend_ratio >= 0.7  # Switch at 70% budget usage

    def get_recommended_compression_level(self, token_count: int) -> CompressionLevel:
        """Get recommended compression level based on token count."""
        if token_count < 2000:
            return CompressionLevel.NONE
        elif token_count < 8000:
            return CompressionLevel.LIGHT
        elif token_count < 32000:
            return CompressionLevel.MODERATE
        elif token_count < 100000:
            return CompressionLevel.AGGRESSIVE
        else:
            return CompressionLevel.EXTREME


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Compression
    "CompressionLevel",
    "CompressionResult",
    "CompressionStrategy",
    "WhitespaceCompression",
    "CodeBlockCompression",
    "SemanticCompression",
    "ContextCompressor",
    # Caching
    "CacheEntry",
    "LRUCache",
    "PromptCache",
    # Streaming
    "StreamChunk",
    "ResponseStreamer",
    # Batching
    "BatchRequest",
    "BatchResult",
    "BatchProcessor",
    # Cost Tracking
    "TokenUsage",
    "CostTracker",
    # Unified
    "TokenOptimizer",
]
