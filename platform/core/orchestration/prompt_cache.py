"""
Prompt Cache Manager - V68 Cost Optimization

Wraps Anthropic's prompt caching to reduce per-iteration costs by ~90%.
Manages cacheable prompt segments (system prompts, tool definitions, context),
tracks hit/miss statistics, and estimates cost savings.

V68 Enhancements (2026-02-05):
- TTL parameter support: 5m (default) or 1h cache duration
- Model-specific minimum token thresholds
- Updated pricing for all models including 1h cache
- Enhanced metrics from API response (cache_creation_input_tokens, cache_read_input_tokens)
- Integration with model_routing.py

Anthropic prompt caching pricing (as of 2026-02):
| Model             | Base Input | 5m Write | 1h Write | Cache Read |
|-------------------|------------|----------|----------|------------|
| Claude Opus 4.5   | $5/MTok    | $6.25    | $10      | $0.50      |
| Claude Opus 4/4.1 | $15/MTok   | $18.75   | $30      | $1.50      |
| Claude Sonnet 4/4.5| $3/MTok   | $3.75    | $6       | $0.30      |
| Claude Haiku 4.5  | $1/MTok    | $1.25    | $2       | $0.10      |
| Claude Haiku 3.5  | $0.80/MTok | $1       | $1.6     | $0.08      |
| Claude Haiku 3    | $0.25/MTok | $0.30    | $0.50    | $0.03      |

Cache Rules:
- 5m cache writes: 1.25x base input price
- 1h cache writes: 2x base input price
- Cache reads: 0.1x base input price
- TTL refreshed on each hit (no additional cost)
- Up to 4 cache breakpoints allowed per request

Usage:
    from core.orchestration.prompt_cache import PromptCacheManager, CacheTTL

    manager = PromptCacheManager()
    manager.mark_static(system_prompt, ttl=CacheTTL.ONE_HOUR)  # 1h for infrequent access
    messages = manager.prepare_messages(user_messages, system_prompt=system_prompt)
    # Pass messages to Anthropic API - cache_control blocks added automatically

Reference: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Literal

logger = logging.getLogger(__name__)


# =============================================================================
# CACHE TTL AND MODEL-SPECIFIC CONFIGURATIONS
# =============================================================================

class CacheTTL(str, Enum):
    """Cache Time-To-Live options for Anthropic prompt caching."""
    FIVE_MINUTES = "5m"
    ONE_HOUR = "1h"


# Model-specific minimum token requirements for caching
# Models with higher context windows have different thresholds
MODEL_CACHE_THRESHOLDS = {
    # Opus models: 4096 minimum
    "claude-opus-4-5-20251101": 4096,
    "claude-opus-4-20250514": 1024,
    "claude-opus-4-1-20250312": 1024,
    # Sonnet models: 1024 minimum
    "claude-sonnet-4-5-20251101": 1024,
    "claude-sonnet-4-20250514": 1024,
    "claude-3-7-sonnet-20250219": 1024,  # deprecated but supported
    # Haiku models: varies
    "claude-haiku-4-5-20251101": 4096,
    "claude-3-5-haiku-20241022": 2048,
    "claude-3-haiku-20240307": 2048,
}

# Model-specific pricing (per MTok)
MODEL_PRICING = {
    "claude-opus-4-5-20251101": {
        "input": 5.0, "write_5m": 6.25, "write_1h": 10.0, "read": 0.50, "output": 25.0
    },
    "claude-opus-4-20250514": {
        "input": 15.0, "write_5m": 18.75, "write_1h": 30.0, "read": 1.50, "output": 75.0
    },
    "claude-opus-4-1-20250312": {
        "input": 15.0, "write_5m": 18.75, "write_1h": 30.0, "read": 1.50, "output": 75.0
    },
    "claude-sonnet-4-5-20251101": {
        "input": 3.0, "write_5m": 3.75, "write_1h": 6.0, "read": 0.30, "output": 15.0
    },
    "claude-sonnet-4-20250514": {
        "input": 3.0, "write_5m": 3.75, "write_1h": 6.0, "read": 0.30, "output": 15.0
    },
    "claude-3-7-sonnet-20250219": {
        "input": 3.0, "write_5m": 3.75, "write_1h": 6.0, "read": 0.30, "output": 15.0
    },
    "claude-haiku-4-5-20251101": {
        "input": 1.0, "write_5m": 1.25, "write_1h": 2.0, "read": 0.10, "output": 5.0
    },
    "claude-3-5-haiku-20241022": {
        "input": 0.80, "write_5m": 1.0, "write_1h": 1.6, "read": 0.08, "output": 4.0
    },
    "claude-3-haiku-20240307": {
        "input": 0.25, "write_5m": 0.30, "write_1h": 0.50, "read": 0.03, "output": 1.25
    },
}

# Default fallback values
_ANTHROPIC_MIN_CACHE_TOKENS = 1024
_DEFAULT_MODEL = "claude-sonnet-4-20250514"

# Rough estimate: 1 token ~= 4 characters
_CHARS_PER_TOKEN = 4


def _estimate_tokens(content: str) -> int:
    """Estimate token count from content length (1 token ~ 4 chars)."""
    if not content:
        return 0
    return max(1, len(content) // _CHARS_PER_TOKEN)


def _content_hash(content: str) -> str:
    """SHA-256 hash of content, truncated to 32 hex chars."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]


def get_min_tokens_for_model(model: str) -> int:
    """Get minimum cacheable token count for a specific model."""
    return MODEL_CACHE_THRESHOLDS.get(model, _ANTHROPIC_MIN_CACHE_TOKENS)


def get_pricing_for_model(model: str) -> Dict[str, float]:
    """Get pricing configuration for a specific model."""
    return MODEL_PRICING.get(model, MODEL_PRICING[_DEFAULT_MODEL])


@dataclass
class PromptCacheConfig:
    """Configuration for prompt caching behavior and cost tracking.

    Attributes:
        ttl: Cache TTL - either CacheTTL.FIVE_MINUTES (default) or CacheTTL.ONE_HOUR
        ttl_seconds: Local tracking TTL (for internal cache expiration tracking)
        max_cached_prompts: Maximum prompts to track locally
        min_token_threshold: Minimum tokens for caching (overridden by model-specific)
        model: Model ID for model-specific pricing and thresholds
        enable_cost_tracking: Whether to track cost savings
        input_cost_per_mtok: Base input token cost (model-specific if model set)
        cached_cost_per_mtok: Cache read cost (model-specific if model set)
        write_cost_per_mtok_5m: 5-minute cache write cost
        write_cost_per_mtok_1h: 1-hour cache write cost
    """
    ttl: CacheTTL = CacheTTL.FIVE_MINUTES
    ttl_seconds: int = 300  # Local tracking TTL
    max_cached_prompts: int = 100
    min_token_threshold: int = _ANTHROPIC_MIN_CACHE_TOKENS
    model: str = _DEFAULT_MODEL
    enable_cost_tracking: bool = True
    input_cost_per_mtok: float = 3.0
    cached_cost_per_mtok: float = 0.30
    write_cost_per_mtok_5m: float = 3.75
    write_cost_per_mtok_1h: float = 6.0

    def __post_init__(self) -> None:
        if self.ttl_seconds < 0:
            raise ValueError(f"ttl_seconds must be non-negative, got {self.ttl_seconds}")
        if self.max_cached_prompts < 1:
            raise ValueError(f"max_cached_prompts must be >= 1, got {self.max_cached_prompts}")
        if self.min_token_threshold < 0:
            raise ValueError(
                f"min_token_threshold must be non-negative, got {self.min_token_threshold}"
            )
        if self.input_cost_per_mtok < 0:
            raise ValueError("input_cost_per_mtok must be non-negative")
        if self.cached_cost_per_mtok < 0:
            raise ValueError("cached_cost_per_mtok must be non-negative")
        if self.write_cost_per_mtok_5m < 0:
            raise ValueError("write_cost_per_mtok_5m must be non-negative")
        if self.write_cost_per_mtok_1h < 0:
            raise ValueError("write_cost_per_mtok_1h must be non-negative")

        # Apply model-specific settings
        if self.model in MODEL_PRICING:
            pricing = MODEL_PRICING[self.model]
            self.input_cost_per_mtok = pricing["input"]
            self.cached_cost_per_mtok = pricing["read"]
            self.write_cost_per_mtok_5m = pricing["write_5m"]
            self.write_cost_per_mtok_1h = pricing["write_1h"]

        if self.model in MODEL_CACHE_THRESHOLDS:
            self.min_token_threshold = MODEL_CACHE_THRESHOLDS[self.model]

        # Update local TTL based on cache TTL
        if self.ttl == CacheTTL.ONE_HOUR:
            self.ttl_seconds = 3600

    @property
    def write_cost_per_mtok(self) -> float:
        """Get write cost based on TTL setting."""
        if self.ttl == CacheTTL.ONE_HOUR:
            return self.write_cost_per_mtok_1h
        return self.write_cost_per_mtok_5m


@dataclass
class CacheablePrompt:
    """A prompt segment eligible for Anthropic prompt caching."""

    content: str
    cache_key: str
    token_count: int
    last_used: float = field(default_factory=time.time)
    hit_count: int = 0
    is_static: bool = False
    created_at: float = field(default_factory=time.time)
    ttl: CacheTTL = CacheTTL.FIVE_MINUTES

    def is_expired(self, ttl_seconds: int, now: Optional[float] = None) -> bool:
        """Check if this cache entry has exceeded its TTL."""
        current = now if now is not None else time.time()
        return (current - self.last_used) > ttl_seconds

    def touch(self, now: Optional[float] = None) -> None:
        """Update last_used timestamp and increment hit count."""
        self.last_used = now if now is not None else time.time()
        self.hit_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "token_count": self.token_count,
            "hit_count": self.hit_count,
            "is_static": self.is_static,
            "last_used": self.last_used,
            "ttl": self.ttl.value,
        }


@dataclass
class CacheStats:
    """Aggregate cache performance statistics.

    V68 Enhancement: Now tracks separate 5m/1h write tokens from API responses.
    """

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    estimated_savings_usd: float = 0.0
    tokens_cached: int = 0
    tokens_uncached: int = 0
    # V68: API response tracking
    api_cache_read_tokens: int = 0
    api_cache_creation_tokens: int = 0
    api_cache_creation_5m_tokens: int = 0
    api_cache_creation_1h_tokens: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction [0.0, 1.0]."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(self.hit_rate, 4),
            "estimated_savings_usd": round(self.estimated_savings_usd, 6),
            "tokens_cached": self.tokens_cached,
            "tokens_uncached": self.tokens_uncached,
            "api_metrics": {
                "cache_read_tokens": self.api_cache_read_tokens,
                "cache_creation_tokens": self.api_cache_creation_tokens,
                "cache_creation_5m_tokens": self.api_cache_creation_5m_tokens,
                "cache_creation_1h_tokens": self.api_cache_creation_1h_tokens,
            },
        }

    def update_from_api_response(self, usage: Dict[str, Any]) -> None:
        """Update stats from Anthropic API response usage field.

        The usage object contains:
        - input_tokens: tokens after last cache breakpoint (not cached)
        - cache_read_input_tokens: tokens read from cache
        - cache_creation_input_tokens: tokens written to cache
        - cache_creation: {ephemeral_5m_input_tokens, ephemeral_1h_input_tokens}
        """
        self.api_cache_read_tokens += usage.get("cache_read_input_tokens", 0)
        self.api_cache_creation_tokens += usage.get("cache_creation_input_tokens", 0)

        # Detailed cache creation breakdown (if available)
        cache_creation = usage.get("cache_creation", {})
        if cache_creation:
            self.api_cache_creation_5m_tokens += cache_creation.get(
                "ephemeral_5m_input_tokens", 0
            )
            self.api_cache_creation_1h_tokens += cache_creation.get(
                "ephemeral_1h_input_tokens", 0
            )


class PromptCacheManager:
    """Manages Anthropic prompt caching for cost optimization.

    Thread-safe manager that tracks cacheable prompt segments, adds
    cache_control breakpoints to API messages, and estimates cost savings.
    """

    def __init__(self, config: Optional[PromptCacheConfig] = None) -> None:
        self._config = config or PromptCacheConfig()
        self._lock = threading.Lock()
        self._cache: Dict[str, CacheablePrompt] = {}
        self._stats = CacheStats()
        self._static_keys: set = set()
        logger.info(
            "PromptCacheManager initialized (ttl=%ds, max=%d, min_tokens=%d)",
            self._config.ttl_seconds,
            self._config.max_cached_prompts,
            self._config.min_token_threshold,
        )

    @property
    def config(self) -> PromptCacheConfig:
        return self._config

    def should_cache(self, content: str) -> bool:
        """Determine if content meets caching criteria.

        Content is cacheable if its estimated token count meets the minimum
        threshold (Anthropic requires >= 1024 tokens for cache eligibility).
        """
        if not content:
            return False
        token_count = _estimate_tokens(content)
        return token_count >= self._config.min_token_threshold

    def mark_static(
        self,
        content: str,
        ttl: CacheTTL = CacheTTL.FIVE_MINUTES,
    ) -> str:
        """Register content as static (e.g., system prompts that never change).

        Static prompts are always cached and never evicted by LRU.

        Args:
            content: The content to mark as static
            ttl: Cache TTL - use CacheTTL.ONE_HOUR for content accessed less
                 frequently than every 5 minutes but more than every hour.

        Returns:
            The cache key for the registered content.

        Best Practices:
            - Use 5m TTL (default) for frequently used prompts (>1x per 5min)
            - Use 1h TTL for infrequent access (>5min between uses)
            - 1h cache costs 2x base input but saves latency
        """
        if not content:
            raise ValueError("Cannot mark empty content as static")

        cache_key = _content_hash(content)
        token_count = _estimate_tokens(content)

        with self._lock:
            if cache_key in self._cache:
                self._cache[cache_key].is_static = True
                self._cache[cache_key].ttl = ttl
            else:
                entry = CacheablePrompt(
                    content=content,
                    cache_key=cache_key,
                    token_count=token_count,
                    is_static=True,
                    ttl=ttl,
                )
                self._cache[cache_key] = entry
            self._static_keys.add(cache_key)

        logger.debug(
            "Marked static prompt: key=%s, tokens=%d, ttl=%s",
            cache_key[:12], token_count, ttl.value
        )
        return cache_key

    def _evict_if_needed(self) -> None:
        """Evict least-recently-used non-static entries if over capacity.

        Must be called with self._lock held.
        """
        while len(self._cache) > self._config.max_cached_prompts:
            # Find oldest non-static entry
            evict_key = None
            oldest_time = float("inf")
            for key, entry in self._cache.items():
                if key in self._static_keys:
                    continue
                if entry.last_used < oldest_time:
                    oldest_time = entry.last_used
                    evict_key = key
            if evict_key is None:
                # All entries are static, cannot evict further
                logger.warning(
                    "Cache full with %d static entries, cannot evict",
                    len(self._cache),
                )
                break
            del self._cache[evict_key]
            logger.debug("Evicted cache entry: %s", evict_key[:12])

    def _purge_expired(self, now: Optional[float] = None) -> int:
        """Remove expired non-static entries. Must be called with self._lock held.

        Returns the number of entries purged.
        """
        current = now if now is not None else time.time()
        expired_keys = [
            key
            for key, entry in self._cache.items()
            if key not in self._static_keys
            and entry.is_expired(self._config.ttl_seconds, now=current)
        ]
        for key in expired_keys:
            del self._cache[key]
        if expired_keys:
            logger.debug("Purged %d expired cache entries", len(expired_keys))
        return len(expired_keys)

    def _lookup(self, content: str, now: Optional[float] = None) -> Optional[CacheablePrompt]:
        """Look up content in cache. Must be called with self._lock held.

        Returns the CacheablePrompt if found and not expired, else None.
        """
        cache_key = _content_hash(content)
        entry = self._cache.get(cache_key)
        if entry is None:
            return None

        current = now if now is not None else time.time()
        # Static entries never expire
        if cache_key not in self._static_keys and entry.is_expired(
            self._config.ttl_seconds, now=current
        ):
            del self._cache[cache_key]
            return None

        entry.touch(now=current)
        return entry

    def _register(
        self,
        content: str,
        now: Optional[float] = None,
        ttl: CacheTTL = CacheTTL.FIVE_MINUTES,
    ) -> CacheablePrompt:
        """Register new content in cache. Must be called with self._lock held."""
        cache_key = _content_hash(content)
        current = now if now is not None else time.time()
        entry = CacheablePrompt(
            content=content,
            cache_key=cache_key,
            token_count=_estimate_tokens(content),
            last_used=current,
            created_at=current,
            hit_count=0,
            is_static=cache_key in self._static_keys,
            ttl=ttl,
        )
        self._cache[cache_key] = entry
        self._evict_if_needed()
        return entry

    def _record_hit(self, token_count: int) -> None:
        """Record a cache hit in stats. Must be called with self._lock held."""
        self._stats.total_requests += 1
        self._stats.cache_hits += 1
        self._stats.tokens_cached += token_count
        if self._config.enable_cost_tracking:
            # Per-hit savings: difference between uncached and cached read cost
            mtok = token_count / 1_000_000.0
            per_hit_savings = mtok * (
                self._config.input_cost_per_mtok - self._config.cached_cost_per_mtok
            )
            self._stats.estimated_savings_usd += per_hit_savings

    def _record_miss(self, token_count: int) -> None:
        """Record a cache miss in stats. Must be called with self._lock held."""
        self._stats.total_requests += 1
        self._stats.cache_misses += 1
        self._stats.tokens_uncached += token_count

    def estimate_savings(self, token_count: int, num_calls: int) -> float:
        """Estimate USD savings from caching a prompt over num_calls invocations.

        The first call pays the write premium, subsequent calls get the read discount.

        Args:
            token_count: Number of tokens in the cached content.
            num_calls: Total number of API calls using this content.

        Returns:
            Estimated savings in USD (can be negative if num_calls == 1).
        """
        if num_calls <= 0 or token_count <= 0:
            return 0.0
        mtok = token_count / 1_000_000.0
        uncached_cost = mtok * self._config.input_cost_per_mtok * num_calls
        # First call: write cost; subsequent calls: cached read cost
        cached_cost = (mtok * self._config.write_cost_per_mtok) + (
            mtok * self._config.cached_cost_per_mtok * max(0, num_calls - 1)
        )
        return uncached_cost - cached_cost

    def prepare_messages(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        now: Optional[float] = None,
        system_ttl: Optional[CacheTTL] = None,
        messages_ttl: Optional[CacheTTL] = None,
    ) -> Dict[str, Any]:
        """Format messages with Anthropic cache_control breakpoints.

        Adds cache_control: {"type": "ephemeral"} (or with ttl) to eligible
        content blocks (system prompt and large user messages).

        Args:
            messages: List of message dicts (role/content).
            system_prompt: Optional system prompt to cache.
            now: Optional timestamp for testing.
            system_ttl: TTL for system prompt (default: config TTL).
                        Use CacheTTL.ONE_HOUR for large static context.
            messages_ttl: TTL for message content (default: config TTL).

        Returns:
            Dict with "system" and "messages" keys ready for the Anthropic API.

        Best Practices:
            - Use 1h TTL for large static content (RAG context, documentation)
            - Use 5m TTL for frequently changing conversation context
            - System prompts benefit most from 1h caching if used <1x per 5min
        """
        result: Dict[str, Any] = {"messages": [], "system": None}

        with self._lock:
            self._purge_expired(now=now)

            # Process system prompt
            if system_prompt:
                system_block = self._prepare_content_block(
                    system_prompt, role="system", now=now, ttl=system_ttl
                )
                result["system"] = system_block

            # Process messages
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    block = self._prepare_content_block(
                        content, role=role, now=now, ttl=messages_ttl
                    )
                    result["messages"].append({"role": role, "content": block})
                elif isinstance(content, list):
                    # Already structured content blocks - pass through, add cache_control
                    # to the last block if total is large enough
                    processed = self._prepare_structured_blocks(
                        content, now=now, ttl=messages_ttl
                    )
                    result["messages"].append({"role": role, "content": processed})
                else:
                    result["messages"].append(msg)

        return result

    def _prepare_content_block(
        self,
        content: str,
        role: str = "user",
        now: Optional[float] = None,
        ttl: Optional[CacheTTL] = None,
    ) -> List[Dict[str, Any]]:
        """Prepare a single content string as cacheable block(s).

        Must be called with self._lock held.

        Args:
            content: Text content to prepare
            role: Message role (user, assistant, system)
            now: Optional timestamp for testing
            ttl: Cache TTL override (default uses config TTL)
        """
        block: Dict[str, Any] = {"type": "text", "text": content}
        token_count = _estimate_tokens(content)
        effective_ttl = ttl or self._config.ttl

        if token_count >= self._config.min_token_threshold:
            entry = self._lookup(content, now=now)
            if entry is not None:
                self._record_hit(token_count)
                effective_ttl = entry.ttl  # Use stored TTL for consistency
                logger.debug(
                    "Cache HIT: role=%s, tokens=%d, key=%s, ttl=%s",
                    role,
                    token_count,
                    entry.cache_key[:12],
                    effective_ttl.value,
                )
            else:
                self._record_miss(token_count)
                self._register(content, now=now, ttl=effective_ttl)
                logger.debug(
                    "Cache MISS: role=%s, tokens=%d, ttl=%s",
                    role, token_count, effective_ttl.value
                )

            # Build cache_control with optional TTL
            cache_control: Dict[str, str] = {"type": "ephemeral"}
            if effective_ttl != CacheTTL.FIVE_MINUTES:
                cache_control["ttl"] = effective_ttl.value
            block["cache_control"] = cache_control

        return [block]

    def _prepare_structured_blocks(
        self,
        blocks: List[Dict[str, Any]],
        now: Optional[float] = None,
        ttl: Optional[CacheTTL] = None,
    ) -> List[Dict[str, Any]]:
        """Process pre-structured content blocks. Must be called with self._lock held.

        Adds cache_control to the last text block if total tokens exceed threshold.
        """
        result = []
        total_tokens = 0
        last_text_idx = -1
        effective_ttl = ttl or self._config.ttl

        for i, block in enumerate(blocks):
            copied = dict(block)
            if copied.get("type") == "text":
                text = copied.get("text", "")
                total_tokens += _estimate_tokens(text)
                last_text_idx = i
            result.append(copied)

        if total_tokens >= self._config.min_token_threshold and last_text_idx >= 0:
            # Build cache_control with optional TTL
            cache_control: Dict[str, str] = {"type": "ephemeral"}
            if effective_ttl != CacheTTL.FIVE_MINUTES:
                cache_control["ttl"] = effective_ttl.value
            result[last_text_idx]["cache_control"] = cache_control

            # Track stats for the aggregate
            combined = " ".join(
                b.get("text", "") for b in blocks if b.get("type") == "text"
            )
            entry = self._lookup(combined, now=now)
            if entry is not None:
                self._record_hit(total_tokens)
                effective_ttl = entry.ttl
            else:
                self._record_miss(total_tokens)
                self._register(combined, now=now, ttl=effective_ttl)

        return result

    def get_stats(self) -> CacheStats:
        """Return a snapshot of current cache statistics."""
        with self._lock:
            return CacheStats(
                total_requests=self._stats.total_requests,
                cache_hits=self._stats.cache_hits,
                cache_misses=self._stats.cache_misses,
                estimated_savings_usd=self._stats.estimated_savings_usd,
                tokens_cached=self._stats.tokens_cached,
                tokens_uncached=self._stats.tokens_uncached,
                api_cache_read_tokens=self._stats.api_cache_read_tokens,
                api_cache_creation_tokens=self._stats.api_cache_creation_tokens,
                api_cache_creation_5m_tokens=self._stats.api_cache_creation_5m_tokens,
                api_cache_creation_1h_tokens=self._stats.api_cache_creation_1h_tokens,
            )

    def update_from_api_response(self, usage: Dict[str, Any]) -> None:
        """Update cache stats from Anthropic API response.

        Call this after each API call to track actual cache performance
        from the server's perspective.

        Args:
            usage: The 'usage' field from the Anthropic API response containing:
                   - input_tokens: uncached tokens after last breakpoint
                   - cache_read_input_tokens: tokens read from cache
                   - cache_creation_input_tokens: tokens written to cache
                   - cache_creation: {ephemeral_5m_input_tokens, ephemeral_1h_input_tokens}

        Example:
            response = client.messages.create(**params)
            manager.update_from_api_response(response.usage.model_dump())
        """
        with self._lock:
            self._stats.update_from_api_response(usage)

    def get_cached_entries(self) -> List[Dict[str, Any]]:
        """Return info about all currently cached entries."""
        with self._lock:
            return [entry.to_dict() for entry in self._cache.values()]

    def clear(self) -> None:
        """Clear all cache entries and reset statistics."""
        with self._lock:
            self._cache.clear()
            self._static_keys.clear()
            self._stats = CacheStats()
        logger.info("Prompt cache cleared")

    def size(self) -> int:
        """Return the number of entries currently in cache."""
        with self._lock:
            return len(self._cache)


# =============================================================================
# Global instance and factory functions
# =============================================================================

_global_cache_manager: Optional[PromptCacheManager] = None


def get_prompt_cache_manager() -> PromptCacheManager:
    """Get or create the global prompt cache manager singleton."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = PromptCacheManager()
    return _global_cache_manager


def set_prompt_cache_manager(manager: PromptCacheManager) -> None:
    """Set the global prompt cache manager (useful for testing)."""
    global _global_cache_manager
    _global_cache_manager = manager


def create_cached_messages_params(
    messages: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096,
    tools: Optional[List[Dict[str, Any]]] = None,
    system_ttl: Optional[CacheTTL] = None,
    tools_ttl: Optional[CacheTTL] = None,
    messages_ttl: Optional[CacheTTL] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Create Anthropic API parameters with prompt caching enabled.

    This is the main integration point for prompt caching. Use this function
    to prepare message parameters before calling the Anthropic API.

    V68 Enhancements:
    - TTL support: Use 5m (default) or 1h cache duration
    - Model-aware: Uses model-specific minimum token thresholds
    - Up to 4 cache breakpoints supported

    Args:
        messages: List of message dicts (role/content)
        system_prompt: Optional system prompt (highly recommended to cache)
        model: Claude model ID (default: claude-sonnet-4-20250514)
        max_tokens: Maximum output tokens
        tools: Optional list of tool definitions (large tools benefit from caching)
        system_ttl: TTL for system prompt cache (default: 5m)
        tools_ttl: TTL for tools cache (default: 5m)
        messages_ttl: TTL for message content cache (default: 5m)
        **kwargs: Additional Anthropic API parameters

    Returns:
        Dict ready to pass to anthropic.messages.create(**params)

    Example:
        import anthropic
        from core.orchestration.prompt_cache import (
            create_cached_messages_params,
            CacheTTL,
        )

        client = anthropic.Anthropic()
        params = create_cached_messages_params(
            messages=[{"role": "user", "content": "Analyze this code..."}],
            system_prompt=LARGE_SYSTEM_PROMPT,  # Will be cached
            system_ttl=CacheTTL.ONE_HOUR,       # 1h cache for static content
            tools=TOOL_DEFINITIONS,             # Will be cached if large
            tools_ttl=CacheTTL.ONE_HOUR,        # Tools rarely change
        )
        response = client.messages.create(**params)

    Cost Savings (Claude Sonnet):
        - 5m cache: Write $3.75/MTok, Read $0.30/MTok (90% discount)
        - 1h cache: Write $6.00/MTok, Read $0.30/MTok (90% discount)
        - Break-even: 2 calls (5m), 3 calls (1h)
        - At 10+ calls: ~90% savings regardless of TTL

    When to use 1h TTL:
        - Prompts used less frequently than every 5 minutes
        - Agentic workflows taking >5 minutes per iteration
        - Long-running conversations with >5 min user response times
        - When latency reduction is more important than cost
    """
    manager = get_prompt_cache_manager()
    effective_system_ttl = system_ttl or CacheTTL.FIVE_MINUTES
    effective_tools_ttl = tools_ttl or CacheTTL.FIVE_MINUTES

    # Get model-specific minimum token threshold
    min_tokens = get_min_tokens_for_model(model)

    # Mark system prompt as static if provided (always cached)
    if system_prompt:
        manager.mark_static(system_prompt, ttl=effective_system_ttl)

    # Prepare messages with cache_control breakpoints
    prepared = manager.prepare_messages(
        messages,
        system_prompt=system_prompt,
        system_ttl=effective_system_ttl,
        messages_ttl=messages_ttl,
    )

    # Build API parameters
    params: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": prepared["messages"],
    }

    # Add system as content blocks with cache_control
    if prepared["system"]:
        params["system"] = prepared["system"]

    # Add tools with cache_control on large definitions
    if tools:
        total_tool_chars = sum(
            len(str(tool)) for tool in tools
        )
        total_tool_tokens = total_tool_chars // _CHARS_PER_TOKEN

        if total_tool_tokens >= min_tokens:
            # Cache tools by adding cache_control to the last tool
            cached_tools = list(tools)
            if cached_tools:
                # Add cache_control to last tool with TTL
                last_tool = dict(cached_tools[-1])
                cache_control: Dict[str, str] = {"type": "ephemeral"}
                if effective_tools_ttl != CacheTTL.FIVE_MINUTES:
                    cache_control["ttl"] = effective_tools_ttl.value
                last_tool["cache_control"] = cache_control
                cached_tools[-1] = last_tool
            params["tools"] = cached_tools
        else:
            params["tools"] = tools

    # Add any additional parameters
    params.update(kwargs)

    return params


async def create_cached_message_async(
    messages: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096,
    tools: Optional[List[Dict[str, Any]]] = None,
    client: Optional[Any] = None,
    system_ttl: Optional[CacheTTL] = None,
    tools_ttl: Optional[CacheTTL] = None,
    messages_ttl: Optional[CacheTTL] = None,
    track_usage: bool = True,
    **kwargs,
) -> Any:
    """
    Create a message using the Anthropic API with prompt caching enabled.

    This is a convenience function that combines parameter preparation
    with the actual API call, and optionally tracks cache metrics.

    Args:
        messages: List of message dicts
        system_prompt: Optional system prompt to cache
        model: Claude model ID
        max_tokens: Maximum output tokens
        tools: Optional tool definitions
        client: Optional anthropic.AsyncAnthropic client (created if not provided)
        system_ttl: TTL for system prompt (5m or 1h)
        tools_ttl: TTL for tools cache
        messages_ttl: TTL for message content
        track_usage: If True, updates cache stats from API response
        **kwargs: Additional API parameters

    Returns:
        Anthropic Message response

    Example:
        response = await create_cached_message_async(
            messages=[{"role": "user", "content": "Hello"}],
            system_prompt="You are a helpful assistant.",
            system_ttl=CacheTTL.ONE_HOUR,  # Use 1h cache for static system prompt
        )
        print(response.content[0].text)
    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "anthropic package required. Install with: pip install anthropic"
        ) from exc

    # Prepare cached parameters
    params = create_cached_messages_params(
        messages=messages,
        system_prompt=system_prompt,
        model=model,
        max_tokens=max_tokens,
        tools=tools,
        system_ttl=system_ttl,
        tools_ttl=tools_ttl,
        messages_ttl=messages_ttl,
        **kwargs,
    )

    # Create or use provided client
    if client is None:
        client = anthropic.AsyncAnthropic()

    # Make API call
    response = await client.messages.create(**params)

    # Track usage metrics from API response
    if track_usage and hasattr(response, "usage"):
        manager = get_prompt_cache_manager()
        manager.update_from_api_response(response.usage.model_dump())

    return response


def create_cached_message_sync(
    messages: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096,
    tools: Optional[List[Dict[str, Any]]] = None,
    client: Optional[Any] = None,
    system_ttl: Optional[CacheTTL] = None,
    tools_ttl: Optional[CacheTTL] = None,
    messages_ttl: Optional[CacheTTL] = None,
    track_usage: bool = True,
    **kwargs,
) -> Any:
    """
    Synchronous version of create_cached_message_async.

    See create_cached_message_async for full documentation.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "anthropic package required. Install with: pip install anthropic"
        ) from exc

    params = create_cached_messages_params(
        messages=messages,
        system_prompt=system_prompt,
        model=model,
        max_tokens=max_tokens,
        tools=tools,
        system_ttl=system_ttl,
        tools_ttl=tools_ttl,
        messages_ttl=messages_ttl,
        **kwargs,
    )

    if client is None:
        client = anthropic.Anthropic()

    response = client.messages.create(**params)

    # Track usage metrics from API response
    if track_usage and hasattr(response, "usage"):
        manager = get_prompt_cache_manager()
        manager.update_from_api_response(response.usage.model_dump())

    return response


# =============================================================================
# Integration with model_routing.py
# =============================================================================

def create_routed_cached_message(
    task: str,
    messages: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
    context: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 4096,
    system_ttl: Optional[CacheTTL] = None,
    tools_ttl: Optional[CacheTTL] = None,
    force_model: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Create cached message parameters with automatic model routing.

    Combines prompt caching with 3-tier model routing to optimize both
    cost (via caching) and model selection (via complexity analysis).

    Args:
        task: Task description for routing decision
        messages: List of message dicts
        system_prompt: Optional system prompt to cache
        context: Optional context for routing analysis
        tools: Optional tool definitions
        max_tokens: Maximum output tokens
        system_ttl: TTL for system prompt cache
        tools_ttl: TTL for tools cache
        force_model: Force specific model (bypasses routing)
        **kwargs: Additional API parameters

    Returns:
        Dict with API parameters including routed model ID

    Example:
        # Automatically routes to optimal model with caching
        params = create_routed_cached_message(
            task="Design a secure authentication system",
            messages=[{"role": "user", "content": "Design OAuth2 + PKCE flow"}],
            system_prompt=AUTH_SYSTEM_PROMPT,
            system_ttl=CacheTTL.ONE_HOUR,
        )
        # params["model"] will be "claude-opus-4-5-20251101" (Tier 3)

        # Simple task routes to cheaper model
        params = create_routed_cached_message(
            task="Fix this typo",
            messages=[{"role": "user", "content": "Fix typo in config"}],
        )
        # params["model"] will be "claude-3-5-haiku-20241022" (Tier 1)
    """
    if force_model:
        model = force_model
    else:
        # Lazy import to avoid circular dependency
        from .model_routing import get_claude_model_for_task
        model = get_claude_model_for_task(task, context)

    return create_cached_messages_params(
        messages=messages,
        system_prompt=system_prompt,
        model=model,
        max_tokens=max_tokens,
        tools=tools,
        system_ttl=system_ttl,
        tools_ttl=tools_ttl,
        **kwargs,
    )


async def create_routed_cached_message_async(
    task: str,
    messages: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
    context: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 4096,
    system_ttl: Optional[CacheTTL] = None,
    tools_ttl: Optional[CacheTTL] = None,
    force_model: Optional[str] = None,
    client: Optional[Any] = None,
    track_usage: bool = True,
    **kwargs,
) -> Any:
    """
    Create and execute a cached message with automatic model routing.

    Combines routing + caching + API call in one function.

    Args:
        task: Task description for routing
        messages: List of message dicts
        system_prompt: Optional system prompt
        context: Optional routing context
        tools: Optional tool definitions
        max_tokens: Maximum output tokens
        system_ttl: TTL for system prompt
        tools_ttl: TTL for tools
        force_model: Force specific model
        client: Optional AsyncAnthropic client
        track_usage: Track cache stats from response
        **kwargs: Additional API parameters

    Returns:
        Anthropic Message response

    Example:
        response = await create_routed_cached_message_async(
            task="Explain quantum computing",
            messages=[{"role": "user", "content": "What is quantum computing?"}],
            system_prompt=SCIENCE_PROMPT,
            system_ttl=CacheTTL.ONE_HOUR,
        )
    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "anthropic package required. Install with: pip install anthropic"
        ) from exc

    params = create_routed_cached_message(
        task=task,
        messages=messages,
        system_prompt=system_prompt,
        context=context,
        tools=tools,
        max_tokens=max_tokens,
        system_ttl=system_ttl,
        tools_ttl=tools_ttl,
        force_model=force_model,
        **kwargs,
    )

    if client is None:
        client = anthropic.AsyncAnthropic()

    response = await client.messages.create(**params)

    if track_usage and hasattr(response, "usage"):
        manager = get_prompt_cache_manager()
        manager.update_from_api_response(response.usage.model_dump())

    return response


def get_cache_savings_report(model: Optional[str] = None) -> Dict[str, Any]:
    """
    Get a comprehensive report of prompt caching statistics and estimated savings.

    V68 Enhancement: Now includes API-tracked metrics and model-specific recommendations.

    Args:
        model: Optional model ID for model-specific cost calculations

    Returns:
        Dict with cache statistics including:
        - hit_rate: Fraction of requests that were cache hits
        - estimated_savings_usd: Total estimated savings in USD
        - tokens_cached: Total tokens served from cache
        - tokens_uncached: Total tokens not cached
        - api_metrics: Actual metrics from API responses
        - recommendation: Suggestion for improving cache efficiency
        - cost_breakdown: Detailed cost analysis
    """
    manager = get_prompt_cache_manager()
    stats = manager.get_stats()
    config = manager.config

    # Calculate potential savings
    entries = manager.get_cached_entries()
    static_entries = [e for e in entries if e.get("is_static")]
    one_hour_entries = [e for e in entries if e.get("ttl") == "1h"]

    # Get model-specific pricing if provided
    pricing = get_pricing_for_model(model or config.model)

    report: Dict[str, Any] = {
        "statistics": stats.to_dict(),
        "cache_size": manager.size(),
        "static_prompts": len(static_entries),
        "one_hour_cache_entries": len(one_hour_entries),
        "model": model or config.model,
        "pricing_per_mtok": pricing,
        "recommendation": None,
        "cost_breakdown": None,
    }

    # Calculate cost breakdown if we have API metrics
    if stats.api_cache_read_tokens > 0 or stats.api_cache_creation_tokens > 0:
        read_tokens = stats.api_cache_read_tokens
        write_5m = stats.api_cache_creation_5m_tokens
        write_1h = stats.api_cache_creation_1h_tokens

        # Calculate actual costs
        read_cost = (read_tokens / 1_000_000) * pricing["read"]
        write_5m_cost = (write_5m / 1_000_000) * pricing["write_5m"]
        write_1h_cost = (write_1h / 1_000_000) * pricing["write_1h"]

        # Calculate what it would have cost without caching
        total_tokens = read_tokens + write_5m + write_1h
        uncached_cost = (total_tokens / 1_000_000) * pricing["input"]

        actual_cost = read_cost + write_5m_cost + write_1h_cost
        savings = uncached_cost - actual_cost

        report["cost_breakdown"] = {
            "uncached_cost_usd": round(uncached_cost, 6),
            "cached_cost_usd": round(actual_cost, 6),
            "actual_savings_usd": round(savings, 6),
            "savings_percentage": round((savings / uncached_cost * 100) if uncached_cost > 0 else 0, 2),
            "breakdown": {
                "cache_read_cost": round(read_cost, 6),
                "cache_write_5m_cost": round(write_5m_cost, 6),
                "cache_write_1h_cost": round(write_1h_cost, 6),
            },
        }

    # Provide recommendations
    if stats.hit_rate < 0.3 and stats.total_requests > 10:
        min_tokens = get_min_tokens_for_model(model or config.model)
        report["recommendation"] = (
            f"Low cache hit rate ({stats.hit_rate:.1%}) detected. Consider:\n"
            f"1. Mark system prompts as static with manager.mark_static(prompt, ttl=CacheTTL.ONE_HOUR)\n"
            f"2. Ensure cached content is at least {min_tokens} tokens (model requirement)\n"
            "3. Reuse identical prompts across requests\n"
            "4. Use 1h TTL for content accessed less frequently than every 5 minutes"
        )
    elif stats.total_requests < 5:
        report["recommendation"] = (
            "Not enough requests to evaluate caching. "
            "Prompt caching breaks even at 2 requests (5m) or 3 requests (1h) "
            "and saves ~90% at 10+ requests."
        )
    elif stats.hit_rate > 0.7:
        savings_str = (
            f"${report['cost_breakdown']['actual_savings_usd']:.4f}"
            if report.get("cost_breakdown")
            else f"${stats.estimated_savings_usd:.4f}"
        )
        report["recommendation"] = (
            f"Excellent cache hit rate ({stats.hit_rate:.1%}). "
            f"Actual savings: {savings_str}"
        )
    elif 0.3 <= stats.hit_rate <= 0.7:
        report["recommendation"] = (
            f"Moderate cache hit rate ({stats.hit_rate:.1%}). "
            "Consider:\n"
            "1. Moving more static content to system prompt\n"
            "2. Using longer TTL (1h) for RAG context or documentation"
        )

    return report
