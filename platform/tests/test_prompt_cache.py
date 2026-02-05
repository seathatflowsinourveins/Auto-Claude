"""
Tests for prompt_cache.py - Anthropic prompt caching wrapper.

Covers: cache hit/miss tracking, cost estimation, TTL expiration,
thread safety, token counting, static vs dynamic prompts, config validation.
"""
import sys
import os
import threading
import time
from unittest.mock import patch

import pytest

# Handle platform module shadowing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.orchestration.prompt_cache import (
    PromptCacheManager,
    PromptCacheConfig,
    CacheablePrompt,
    CacheStats,
    _estimate_tokens,
    _content_hash,
    _CHARS_PER_TOKEN,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_content(token_count: int) -> str:
    """Generate content string with approximately token_count tokens."""
    return "x" * (token_count * _CHARS_PER_TOKEN)


SMALL_CONTENT = _make_content(100)  # Below 1024 threshold
LARGE_CONTENT = _make_content(2000)  # Above 1024 threshold
LARGE_CONTENT_2 = _make_content(3000)


# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_empty_string(self):
        assert _estimate_tokens("") == 0

    def test_short_string(self):
        assert _estimate_tokens("hi") == 1  # max(1, 2//4)

    def test_known_length(self):
        content = "a" * 4000
        assert _estimate_tokens(content) == 1000

    def test_large_content(self):
        content = "b" * 40000
        assert _estimate_tokens(content) == 10000


# ---------------------------------------------------------------------------
# _content_hash
# ---------------------------------------------------------------------------

class TestContentHash:
    def test_deterministic(self):
        assert _content_hash("hello") == _content_hash("hello")

    def test_different_content_different_hash(self):
        assert _content_hash("hello") != _content_hash("world")

    def test_length_is_32(self):
        assert len(_content_hash("test")) == 32


# ---------------------------------------------------------------------------
# PromptCacheConfig validation
# ---------------------------------------------------------------------------

class TestPromptCacheConfig:
    def test_defaults(self):
        cfg = PromptCacheConfig()
        assert cfg.ttl_seconds == 300
        assert cfg.max_cached_prompts == 100
        assert cfg.min_token_threshold == 1024
        assert cfg.enable_cost_tracking is True
        assert cfg.input_cost_per_mtok == 3.0
        assert cfg.cached_cost_per_mtok == 0.30
        assert cfg.write_cost_per_mtok == 3.75

    def test_custom_values(self):
        cfg = PromptCacheConfig(ttl_seconds=60, max_cached_prompts=10)
        assert cfg.ttl_seconds == 60
        assert cfg.max_cached_prompts == 10

    def test_negative_ttl_raises(self):
        with pytest.raises(ValueError, match="ttl_seconds"):
            PromptCacheConfig(ttl_seconds=-1)

    def test_zero_max_cached_raises(self):
        with pytest.raises(ValueError, match="max_cached_prompts"):
            PromptCacheConfig(max_cached_prompts=0)

    def test_negative_min_token_raises(self):
        with pytest.raises(ValueError, match="min_token_threshold"):
            PromptCacheConfig(min_token_threshold=-5)

    def test_negative_input_cost_raises(self):
        with pytest.raises(ValueError, match="input_cost_per_mtok"):
            PromptCacheConfig(input_cost_per_mtok=-1.0)

    def test_negative_cached_cost_raises(self):
        with pytest.raises(ValueError, match="cached_cost_per_mtok"):
            PromptCacheConfig(cached_cost_per_mtok=-0.5)

    def test_negative_write_cost_5m_raises(self):
        with pytest.raises(ValueError, match="write_cost_per_mtok_5m"):
            PromptCacheConfig(write_cost_per_mtok_5m=-2.0)

    def test_negative_write_cost_1h_raises(self):
        with pytest.raises(ValueError, match="write_cost_per_mtok_1h"):
            PromptCacheConfig(write_cost_per_mtok_1h=-2.0)


# ---------------------------------------------------------------------------
# CacheablePrompt
# ---------------------------------------------------------------------------

class TestCacheablePrompt:
    def test_is_expired_false(self):
        entry = CacheablePrompt(
            content="test", cache_key="k", token_count=100, last_used=1000.0
        )
        assert entry.is_expired(300, now=1100.0) is False

    def test_is_expired_true(self):
        entry = CacheablePrompt(
            content="test", cache_key="k", token_count=100, last_used=1000.0
        )
        assert entry.is_expired(300, now=1400.0) is True

    def test_touch_updates(self):
        entry = CacheablePrompt(
            content="test", cache_key="k", token_count=100, last_used=1000.0
        )
        entry.touch(now=2000.0)
        assert entry.last_used == 2000.0
        assert entry.hit_count == 1

    def test_to_dict(self):
        entry = CacheablePrompt(
            content="test", cache_key="abc123", token_count=500, is_static=True
        )
        d = entry.to_dict()
        assert d["cache_key"] == "abc123"
        assert d["token_count"] == 500
        assert d["is_static"] is True


# ---------------------------------------------------------------------------
# CacheStats
# ---------------------------------------------------------------------------

class TestCacheStats:
    def test_hit_rate_zero_requests(self):
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        stats = CacheStats(total_requests=10, cache_hits=7, cache_misses=3)
        assert abs(stats.hit_rate - 0.7) < 1e-9

    def test_to_dict(self):
        stats = CacheStats(total_requests=5, cache_hits=3, cache_misses=2)
        d = stats.to_dict()
        assert d["total_requests"] == 5
        assert d["hit_rate"] == 0.6


# ---------------------------------------------------------------------------
# PromptCacheManager - should_cache
# ---------------------------------------------------------------------------

class TestShouldCache:
    def test_small_content_not_cached(self):
        mgr = PromptCacheManager()
        assert mgr.should_cache(SMALL_CONTENT) is False

    def test_large_content_cached(self):
        mgr = PromptCacheManager()
        assert mgr.should_cache(LARGE_CONTENT) is True

    def test_empty_content_not_cached(self):
        mgr = PromptCacheManager()
        assert mgr.should_cache("") is False

    def test_custom_threshold(self):
        # Note: model= must not be in MODEL_CACHE_THRESHOLDS to avoid override
        cfg = PromptCacheConfig(min_token_threshold=10, model="unknown-model")
        mgr = PromptCacheManager(config=cfg)
        assert mgr.should_cache("a" * 50) is True  # 50 chars = ~12 tokens


# ---------------------------------------------------------------------------
# PromptCacheManager - mark_static
# ---------------------------------------------------------------------------

class TestMarkStatic:
    def test_returns_cache_key(self):
        mgr = PromptCacheManager()
        key = mgr.mark_static("system prompt content")
        assert isinstance(key, str)
        assert len(key) == 32

    def test_empty_raises(self):
        mgr = PromptCacheManager()
        with pytest.raises(ValueError, match="empty"):
            mgr.mark_static("")

    def test_idempotent(self):
        mgr = PromptCacheManager()
        key1 = mgr.mark_static("same content")
        key2 = mgr.mark_static("same content")
        assert key1 == key2
        assert mgr.size() == 1

    def test_static_entry_in_cache(self):
        mgr = PromptCacheManager()
        mgr.mark_static("static data")
        entries = mgr.get_cached_entries()
        assert len(entries) == 1
        assert entries[0]["is_static"] is True


# ---------------------------------------------------------------------------
# PromptCacheManager - prepare_messages (core functionality)
# ---------------------------------------------------------------------------

class TestPrepareMessages:
    def test_system_prompt_gets_cache_control(self):
        mgr = PromptCacheManager()
        result = mgr.prepare_messages([], system_prompt=LARGE_CONTENT)
        system_blocks = result["system"]
        assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}

    def test_small_system_prompt_no_cache_control(self):
        mgr = PromptCacheManager()
        result = mgr.prepare_messages([], system_prompt="short")
        system_blocks = result["system"]
        assert "cache_control" not in system_blocks[0]

    def test_large_user_message_gets_cache_control(self):
        mgr = PromptCacheManager()
        messages = [{"role": "user", "content": LARGE_CONTENT}]
        result = mgr.prepare_messages(messages)
        content_blocks = result["messages"][0]["content"]
        assert content_blocks[0]["cache_control"] == {"type": "ephemeral"}

    def test_small_user_message_no_cache_control(self):
        mgr = PromptCacheManager()
        messages = [{"role": "user", "content": "hello"}]
        result = mgr.prepare_messages(messages)
        content_blocks = result["messages"][0]["content"]
        assert "cache_control" not in content_blocks[0]

    def test_no_system_prompt(self):
        mgr = PromptCacheManager()
        result = mgr.prepare_messages([{"role": "user", "content": "hi"}])
        assert result["system"] is None

    def test_empty_messages(self):
        mgr = PromptCacheManager()
        result = mgr.prepare_messages([])
        assert result["messages"] == []
        assert result["system"] is None

    def test_structured_blocks_passthrough(self):
        mgr = PromptCacheManager()
        blocks = [
            {"type": "text", "text": LARGE_CONTENT},
            {"type": "text", "text": "more text"},
        ]
        messages = [{"role": "user", "content": blocks}]
        result = mgr.prepare_messages(messages)
        out_blocks = result["messages"][0]["content"]
        # Last text block should get cache_control since total tokens > threshold
        assert "cache_control" in out_blocks[-1]


# ---------------------------------------------------------------------------
# Cache hit/miss tracking
# ---------------------------------------------------------------------------

class TestHitMissTracking:
    def test_first_call_is_miss(self):
        mgr = PromptCacheManager()
        mgr.prepare_messages(
            [{"role": "user", "content": LARGE_CONTENT}], now=1000.0
        )
        stats = mgr.get_stats()
        assert stats.cache_misses == 1
        assert stats.cache_hits == 0

    def test_second_call_is_hit(self):
        mgr = PromptCacheManager()
        mgr.prepare_messages(
            [{"role": "user", "content": LARGE_CONTENT}], now=1000.0
        )
        mgr.prepare_messages(
            [{"role": "user", "content": LARGE_CONTENT}], now=1001.0
        )
        stats = mgr.get_stats()
        assert stats.cache_hits == 1
        assert stats.cache_misses == 1
        assert stats.total_requests == 2

    def test_different_content_both_miss(self):
        mgr = PromptCacheManager()
        mgr.prepare_messages(
            [{"role": "user", "content": LARGE_CONTENT}], now=1000.0
        )
        mgr.prepare_messages(
            [{"role": "user", "content": LARGE_CONTENT_2}], now=1001.0
        )
        stats = mgr.get_stats()
        assert stats.cache_misses == 2
        assert stats.cache_hits == 0


# ---------------------------------------------------------------------------
# TTL expiration
# ---------------------------------------------------------------------------

class TestTTLExpiration:
    def test_entry_expires_after_ttl(self):
        cfg = PromptCacheConfig(ttl_seconds=10)
        mgr = PromptCacheManager(config=cfg)

        # First call: miss, registers entry
        mgr.prepare_messages(
            [{"role": "user", "content": LARGE_CONTENT}], now=1000.0
        )
        # Second call after TTL: should be miss (expired)
        mgr.prepare_messages(
            [{"role": "user", "content": LARGE_CONTENT}], now=1020.0
        )
        stats = mgr.get_stats()
        assert stats.cache_misses == 2
        assert stats.cache_hits == 0

    def test_entry_alive_within_ttl(self):
        cfg = PromptCacheConfig(ttl_seconds=10)
        mgr = PromptCacheManager(config=cfg)

        mgr.prepare_messages(
            [{"role": "user", "content": LARGE_CONTENT}], now=1000.0
        )
        mgr.prepare_messages(
            [{"role": "user", "content": LARGE_CONTENT}], now=1005.0
        )
        stats = mgr.get_stats()
        assert stats.cache_hits == 1

    def test_static_never_expires(self):
        cfg = PromptCacheConfig(ttl_seconds=1)
        mgr = PromptCacheManager(config=cfg)

        mgr.mark_static(LARGE_CONTENT)
        # First call: hit (already registered via mark_static)
        mgr.prepare_messages(
            [{"role": "user", "content": LARGE_CONTENT}], now=1000.0
        )
        # Way past TTL
        mgr.prepare_messages(
            [{"role": "user", "content": LARGE_CONTENT}], now=9999.0
        )
        stats = mgr.get_stats()
        # Both should be hits since mark_static pre-registers
        assert stats.cache_hits == 2

    def test_ttl_refresh_on_hit(self):
        cfg = PromptCacheConfig(ttl_seconds=10)
        mgr = PromptCacheManager(config=cfg)

        mgr.prepare_messages(
            [{"role": "user", "content": LARGE_CONTENT}], now=1000.0
        )
        # Hit at t=1008, refreshes last_used
        mgr.prepare_messages(
            [{"role": "user", "content": LARGE_CONTENT}], now=1008.0
        )
        # At t=1015, 7s since last hit (within TTL)
        mgr.prepare_messages(
            [{"role": "user", "content": LARGE_CONTENT}], now=1015.0
        )
        stats = mgr.get_stats()
        assert stats.cache_hits == 2  # t=1008 and t=1015 are hits


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

class TestCostEstimation:
    def test_zero_calls(self):
        mgr = PromptCacheManager()
        assert mgr.estimate_savings(1000, 0) == 0.0

    def test_zero_tokens(self):
        mgr = PromptCacheManager()
        assert mgr.estimate_savings(0, 10) == 0.0

    def test_single_call_negative_savings(self):
        mgr = PromptCacheManager()
        savings = mgr.estimate_savings(1_000_000, 1)
        # 1 call: uncached=$3.00, cached=write=$3.75 -> savings=-0.75
        assert savings < 0

    def test_two_calls_positive_savings(self):
        mgr = PromptCacheManager()
        savings = mgr.estimate_savings(1_000_000, 2)
        # uncached = $3.00 * 2 = $6.00
        # cached = $3.75 (write) + $0.30 (read) = $4.05
        # savings = $1.95
        assert abs(savings - 1.95) < 0.01

    def test_ten_calls_large_savings(self):
        mgr = PromptCacheManager()
        savings = mgr.estimate_savings(1_000_000, 10)
        # uncached = $3.00 * 10 = $30.00
        # cached = $3.75 + $0.30 * 9 = $6.45
        # savings = $23.55
        assert abs(savings - 23.55) < 0.01

    def test_stats_accumulate_savings(self):
        cfg = PromptCacheConfig(enable_cost_tracking=True)
        mgr = PromptCacheManager(config=cfg)
        # First call: miss, no savings recorded
        mgr.prepare_messages(
            [{"role": "user", "content": LARGE_CONTENT}], now=1000.0
        )
        # Second call: hit, savings recorded
        mgr.prepare_messages(
            [{"role": "user", "content": LARGE_CONTENT}], now=1001.0
        )
        stats = mgr.get_stats()
        assert stats.estimated_savings_usd > 0


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------

class TestEviction:
    def test_evicts_lru_when_full(self):
        # Use unknown model to avoid min_token_threshold override
        cfg = PromptCacheConfig(max_cached_prompts=2, min_token_threshold=1, model="unknown-model")
        mgr = PromptCacheManager(config=cfg)

        c1 = _make_content(10) + "AAA"
        c2 = _make_content(10) + "BBB"
        c3 = _make_content(10) + "CCC"

        mgr.prepare_messages([{"role": "user", "content": c1}], now=1000.0)
        mgr.prepare_messages([{"role": "user", "content": c2}], now=1001.0)
        assert mgr.size() == 2

        # Adding c3 should evict c1 (oldest)
        mgr.prepare_messages([{"role": "user", "content": c3}], now=1002.0)
        assert mgr.size() == 2

        # c1 should be miss again (evicted)
        mgr.prepare_messages([{"role": "user", "content": c1}], now=1003.0)
        stats = mgr.get_stats()
        # c1: miss, c2: miss, c3: miss, c1 again: miss (evicted)
        assert stats.cache_misses == 4

    def test_static_not_evicted(self):
        # Use unknown model to avoid min_token_threshold override
        cfg = PromptCacheConfig(max_cached_prompts=2, min_token_threshold=1, model="unknown-model")
        mgr = PromptCacheManager(config=cfg)

        static = _make_content(10) + "STATIC"
        mgr.mark_static(static)

        c2 = _make_content(10) + "BBB"
        c3 = _make_content(10) + "CCC"

        mgr.prepare_messages([{"role": "user", "content": c2}], now=1001.0)
        mgr.prepare_messages([{"role": "user", "content": c3}], now=1002.0)

        # Static entry should still be there
        mgr.prepare_messages([{"role": "user", "content": static}], now=1003.0)
        stats = mgr.get_stats()
        # static: hit (mark_static pre-registered), c2+c3 misses, static again: hit
        assert stats.cache_hits >= 1


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_prepare(self):
        mgr = PromptCacheManager()
        errors = []

        def worker(content: str, worker_id: int):
            try:
                for _ in range(50):
                    mgr.prepare_messages(
                        [{"role": "user", "content": content}]
                    )
            except Exception as e:
                errors.append((worker_id, e))

        threads = [
            threading.Thread(target=worker, args=(LARGE_CONTENT, i))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Thread errors: {errors}"
        stats = mgr.get_stats()
        assert stats.total_requests == 200  # 4 threads * 50 calls

    def test_concurrent_mark_static(self):
        mgr = PromptCacheManager()
        errors = []

        def worker(i: int):
            try:
                content = _make_content(2000) + f"_worker_{i}"
                mgr.mark_static(content)
            except Exception as e:
                errors.append((i, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Thread errors: {errors}"
        assert mgr.size() == 10


# ---------------------------------------------------------------------------
# Clear and size
# ---------------------------------------------------------------------------

class TestClearAndSize:
    def test_clear_resets_everything(self):
        mgr = PromptCacheManager()
        mgr.mark_static("content")
        mgr.prepare_messages(
            [{"role": "user", "content": LARGE_CONTENT}], now=1000.0
        )
        mgr.clear()
        assert mgr.size() == 0
        stats = mgr.get_stats()
        assert stats.total_requests == 0
        assert stats.cache_hits == 0

    def test_size_tracks_entries(self):
        # Use unknown model to avoid min_token_threshold override
        cfg = PromptCacheConfig(min_token_threshold=1, model="unknown-model")
        mgr = PromptCacheManager(config=cfg)
        assert mgr.size() == 0
        mgr.prepare_messages(
            [{"role": "user", "content": "hello world test"}], now=1000.0
        )
        assert mgr.size() == 1


# ---------------------------------------------------------------------------
# Token tracking in stats
# ---------------------------------------------------------------------------

class TestTokenTracking:
    def test_tokens_cached_on_hit(self):
        mgr = PromptCacheManager()
        mgr.prepare_messages(
            [{"role": "user", "content": LARGE_CONTENT}], now=1000.0
        )
        mgr.prepare_messages(
            [{"role": "user", "content": LARGE_CONTENT}], now=1001.0
        )
        stats = mgr.get_stats()
        assert stats.tokens_cached > 0
        assert stats.tokens_uncached > 0

    def test_tokens_uncached_on_miss(self):
        mgr = PromptCacheManager()
        mgr.prepare_messages(
            [{"role": "user", "content": LARGE_CONTENT}], now=1000.0
        )
        stats = mgr.get_stats()
        assert stats.tokens_uncached == _estimate_tokens(LARGE_CONTENT)
        assert stats.tokens_cached == 0


# ---------------------------------------------------------------------------
# Cost tracking disabled
# ---------------------------------------------------------------------------

class TestCostTrackingDisabled:
    def test_no_savings_when_disabled(self):
        cfg = PromptCacheConfig(enable_cost_tracking=False)
        mgr = PromptCacheManager(config=cfg)
        mgr.prepare_messages(
            [{"role": "user", "content": LARGE_CONTENT}], now=1000.0
        )
        mgr.prepare_messages(
            [{"role": "user", "content": LARGE_CONTENT}], now=1001.0
        )
        stats = mgr.get_stats()
        assert stats.estimated_savings_usd == 0.0
        # Hits still tracked
        assert stats.cache_hits == 1


# ---------------------------------------------------------------------------
# V68 - Convenience functions for Anthropic API integration
# ---------------------------------------------------------------------------

from core.orchestration.prompt_cache import (
    get_prompt_cache_manager,
    set_prompt_cache_manager,
    create_cached_messages_params,
    get_cache_savings_report,
)


class TestGlobalCacheManager:
    def test_get_returns_singleton(self):
        mgr1 = get_prompt_cache_manager()
        mgr2 = get_prompt_cache_manager()
        assert mgr1 is mgr2

    def test_set_replaces_singleton(self):
        original = get_prompt_cache_manager()
        new_mgr = PromptCacheManager()
        set_prompt_cache_manager(new_mgr)
        assert get_prompt_cache_manager() is new_mgr
        # Restore original
        set_prompt_cache_manager(original)


class TestCreateCachedMessagesParams:
    def test_basic_params(self):
        params = create_cached_messages_params(
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
        )
        assert params["model"] == "claude-sonnet-4-20250514"
        assert params["max_tokens"] == 1024
        assert len(params["messages"]) == 1

    def test_system_prompt_added(self):
        params = create_cached_messages_params(
            messages=[{"role": "user", "content": "Hello"}],
            system_prompt=LARGE_CONTENT,
        )
        assert "system" in params
        assert params["system"] is not None
        # System should have cache_control
        assert params["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_small_system_prompt_no_cache(self):
        params = create_cached_messages_params(
            messages=[{"role": "user", "content": "Hello"}],
            system_prompt="Be helpful.",
        )
        assert "system" in params
        # Small system prompts don't get cache_control
        assert "cache_control" not in params["system"][0]

    def test_tools_caching(self):
        # Create large tool definitions
        large_tool = {
            "name": "search",
            "description": "x" * 5000,  # Large description
            "input_schema": {"type": "object", "properties": {}},
        }
        params = create_cached_messages_params(
            messages=[{"role": "user", "content": "Search for X"}],
            tools=[large_tool],
        )
        assert "tools" in params
        # Last tool should have cache_control
        assert params["tools"][-1].get("cache_control") == {"type": "ephemeral"}

    def test_small_tools_no_caching(self):
        small_tool = {
            "name": "ping",
            "description": "Ping",
            "input_schema": {"type": "object"},
        }
        params = create_cached_messages_params(
            messages=[{"role": "user", "content": "Ping"}],
            tools=[small_tool],
        )
        assert "tools" in params
        # Small tools don't get cache_control
        assert "cache_control" not in params["tools"][-1]

    def test_extra_kwargs_passed(self):
        params = create_cached_messages_params(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.5,
            top_p=0.9,
        )
        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.9


class TestCacheSavingsReport:
    def test_report_structure(self):
        # Reset global manager
        mgr = PromptCacheManager()
        set_prompt_cache_manager(mgr)

        report = get_cache_savings_report()
        assert "statistics" in report
        assert "cache_size" in report
        assert "static_prompts" in report
        assert "recommendation" in report

    def test_recommendation_for_few_requests(self):
        mgr = PromptCacheManager()
        set_prompt_cache_manager(mgr)

        report = get_cache_savings_report()
        assert "Not enough requests" in report["recommendation"]

    def test_recommendation_for_low_hit_rate(self):
        # Use unknown model to avoid min_token_threshold override
        cfg = PromptCacheConfig(min_token_threshold=1, model="unknown-model")
        mgr = PromptCacheManager(config=cfg)
        set_prompt_cache_manager(mgr)

        # Generate many misses (all unique content)
        for i in range(15):
            mgr.prepare_messages(
                [{"role": "user", "content": f"unique content {i}"}], now=1000.0 + i
            )

        report = get_cache_savings_report()
        assert report["recommendation"] is not None
        assert "Low cache hit rate" in report["recommendation"]

    def test_recommendation_for_good_hit_rate(self):
        # Use unknown model to avoid min_token_threshold override
        cfg = PromptCacheConfig(min_token_threshold=1, model="unknown-model")
        mgr = PromptCacheManager(config=cfg)
        set_prompt_cache_manager(mgr)

        # Generate many hits (same content)
        for i in range(15):
            mgr.prepare_messages(
                [{"role": "user", "content": "same content repeated"}], now=1000.0 + i
            )

        report = get_cache_savings_report()
        assert report["recommendation"] is not None
        assert "Excellent cache hit rate" in report["recommendation"]


# ---------------------------------------------------------------------------
# V68 - TTL Support Tests
# ---------------------------------------------------------------------------

from core.orchestration.prompt_cache import (
    CacheTTL,
    get_min_tokens_for_model,
    get_pricing_for_model,
    MODEL_CACHE_THRESHOLDS,
    MODEL_PRICING,
    create_routed_cached_message,
)


class TestCacheTTL:
    def test_ttl_enum_values(self):
        assert CacheTTL.FIVE_MINUTES.value == "5m"
        assert CacheTTL.ONE_HOUR.value == "1h"

    def test_default_ttl_is_five_minutes(self):
        cfg = PromptCacheConfig()
        assert cfg.ttl == CacheTTL.FIVE_MINUTES

    def test_one_hour_ttl_updates_ttl_seconds(self):
        cfg = PromptCacheConfig(ttl=CacheTTL.ONE_HOUR)
        assert cfg.ttl_seconds == 3600


class TestModelSpecificThresholds:
    def test_opus_4_5_threshold(self):
        assert get_min_tokens_for_model("claude-opus-4-5-20251101") == 4096

    def test_sonnet_threshold(self):
        assert get_min_tokens_for_model("claude-sonnet-4-20250514") == 1024

    def test_haiku_3_threshold(self):
        assert get_min_tokens_for_model("claude-3-haiku-20240307") == 2048

    def test_unknown_model_uses_default(self):
        assert get_min_tokens_for_model("unknown-model") == 1024


class TestModelSpecificPricing:
    def test_sonnet_pricing(self):
        pricing = get_pricing_for_model("claude-sonnet-4-20250514")
        assert pricing["input"] == 3.0
        assert pricing["write_5m"] == 3.75
        assert pricing["write_1h"] == 6.0
        assert pricing["read"] == 0.30

    def test_opus_pricing(self):
        pricing = get_pricing_for_model("claude-opus-4-5-20251101")
        assert pricing["input"] == 5.0
        assert pricing["write_5m"] == 6.25
        assert pricing["write_1h"] == 10.0
        assert pricing["read"] == 0.50

    def test_haiku_pricing(self):
        pricing = get_pricing_for_model("claude-3-haiku-20240307")
        assert pricing["input"] == 0.25
        assert pricing["read"] == 0.03

    def test_config_uses_model_pricing(self):
        cfg = PromptCacheConfig(model="claude-opus-4-5-20251101")
        assert cfg.input_cost_per_mtok == 5.0
        assert cfg.cached_cost_per_mtok == 0.50


class TestTTLInCacheControl:
    def test_five_minute_ttl_no_explicit_ttl_field(self):
        mgr = PromptCacheManager()
        result = mgr.prepare_messages([], system_prompt=LARGE_CONTENT)
        system_block = result["system"][0]
        # 5m is default, so ttl field should be absent (only "type": "ephemeral")
        assert system_block["cache_control"] == {"type": "ephemeral"}

    def test_one_hour_ttl_includes_ttl_field(self):
        cfg = PromptCacheConfig(ttl=CacheTTL.ONE_HOUR)
        mgr = PromptCacheManager(config=cfg)
        result = mgr.prepare_messages([], system_prompt=LARGE_CONTENT)
        system_block = result["system"][0]
        assert system_block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_mark_static_with_one_hour_ttl(self):
        mgr = PromptCacheManager()
        mgr.mark_static(LARGE_CONTENT, ttl=CacheTTL.ONE_HOUR)
        entries = mgr.get_cached_entries()
        assert len(entries) == 1
        assert entries[0]["ttl"] == "1h"

    def test_system_ttl_override(self):
        mgr = PromptCacheManager()
        result = mgr.prepare_messages(
            [], system_prompt=LARGE_CONTENT, system_ttl=CacheTTL.ONE_HOUR
        )
        system_block = result["system"][0]
        assert system_block["cache_control"]["ttl"] == "1h"


class TestCacheStatsAPIMetrics:
    def test_update_from_api_response(self):
        stats = CacheStats()
        usage = {
            "input_tokens": 100,
            "cache_read_input_tokens": 5000,
            "cache_creation_input_tokens": 2000,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 1500,
                "ephemeral_1h_input_tokens": 500,
            },
        }
        stats.update_from_api_response(usage)

        assert stats.api_cache_read_tokens == 5000
        assert stats.api_cache_creation_tokens == 2000
        assert stats.api_cache_creation_5m_tokens == 1500
        assert stats.api_cache_creation_1h_tokens == 500

    def test_api_metrics_in_to_dict(self):
        stats = CacheStats(
            api_cache_read_tokens=1000,
            api_cache_creation_tokens=500,
        )
        d = stats.to_dict()
        assert "api_metrics" in d
        assert d["api_metrics"]["cache_read_tokens"] == 1000

    def test_manager_update_from_api(self):
        mgr = PromptCacheManager()
        usage = {
            "cache_read_input_tokens": 10000,
            "cache_creation_input_tokens": 0,
        }
        mgr.update_from_api_response(usage)
        stats = mgr.get_stats()
        assert stats.api_cache_read_tokens == 10000


class TestEnhancedSavingsReport:
    def test_report_includes_model_info(self):
        mgr = PromptCacheManager()
        set_prompt_cache_manager(mgr)
        report = get_cache_savings_report(model="claude-opus-4-5-20251101")
        assert report["model"] == "claude-opus-4-5-20251101"
        assert "pricing_per_mtok" in report

    def test_report_includes_one_hour_entries(self):
        mgr = PromptCacheManager()
        mgr.mark_static(LARGE_CONTENT, ttl=CacheTTL.ONE_HOUR)
        set_prompt_cache_manager(mgr)
        report = get_cache_savings_report()
        assert report["one_hour_cache_entries"] == 1

    def test_cost_breakdown_from_api_metrics(self):
        mgr = PromptCacheManager()
        # Simulate API response tracking
        usage = {
            "cache_read_input_tokens": 1_000_000,  # 1M tokens read
            "cache_creation_input_tokens": 100_000,  # 100K tokens written
            "cache_creation": {
                "ephemeral_5m_input_tokens": 100_000,
                "ephemeral_1h_input_tokens": 0,
            },
        }
        mgr.update_from_api_response(usage)
        set_prompt_cache_manager(mgr)

        report = get_cache_savings_report(model="claude-sonnet-4-20250514")

        assert "cost_breakdown" in report
        assert report["cost_breakdown"] is not None
        # 1.1M tokens at $3/MTok uncached = $3.30
        # 1M read at $0.30/MTok = $0.30, 100K write at $3.75 = $0.375
        # Savings = $3.30 - $0.675 = $2.625
        assert report["cost_breakdown"]["actual_savings_usd"] > 0


class TestCreateCachedMessagesParamsWithTTL:
    def test_system_ttl_parameter(self):
        params = create_cached_messages_params(
            messages=[{"role": "user", "content": "Hello"}],
            system_prompt=LARGE_CONTENT,
            system_ttl=CacheTTL.ONE_HOUR,
        )
        assert params["system"][0]["cache_control"]["ttl"] == "1h"

    def test_tools_ttl_parameter(self):
        large_tool = {
            "name": "search",
            "description": "x" * 5000,
            "input_schema": {"type": "object"},
        }
        params = create_cached_messages_params(
            messages=[{"role": "user", "content": "Search"}],
            tools=[large_tool],
            tools_ttl=CacheTTL.ONE_HOUR,
        )
        assert params["tools"][-1]["cache_control"]["ttl"] == "1h"

    def test_model_specific_threshold(self):
        # Opus 4.5 requires 4096 tokens minimum
        small_for_opus = _make_content(3000)  # Below 4096
        # Reset global manager to ensure it uses the model's threshold
        mgr = PromptCacheManager(config=PromptCacheConfig(model="claude-opus-4-5-20251101"))
        set_prompt_cache_manager(mgr)

        params = create_cached_messages_params(
            messages=[{"role": "user", "content": small_for_opus}],
            model="claude-opus-4-5-20251101",
        )
        # Should not have cache_control since below threshold
        msg_content = params["messages"][0]["content"]
        if isinstance(msg_content, list):
            assert "cache_control" not in msg_content[0]


class TestRoutedCachedMessage:
    def test_simple_task_routes_to_haiku(self):
        params = create_routed_cached_message(
            task="Fix this typo",
            messages=[{"role": "user", "content": "Fix the typo"}],
        )
        assert "claude" in params["model"].lower()
        # Simple task should route to tier 1 (Haiku)
        assert "haiku" in params["model"].lower()

    def test_complex_task_routes_to_opus(self):
        params = create_routed_cached_message(
            task="Design a distributed consensus algorithm for blockchain",
            messages=[{"role": "user", "content": "Design consensus"}],
        )
        # Complex task should route to tier 3 (Opus)
        assert "opus" in params["model"].lower()

    def test_force_model_overrides_routing(self):
        params = create_routed_cached_message(
            task="Design architecture",  # Would normally route to Opus
            messages=[{"role": "user", "content": "Design"}],
            force_model="claude-3-5-haiku-20241022",
        )
        assert params["model"] == "claude-3-5-haiku-20241022"

    def test_includes_cache_control(self):
        # Use a larger content to ensure it passes the 1024 token threshold
        very_large_content = _make_content(3000)  # ~3000 tokens
        # Reset manager to ensure clean state
        mgr = PromptCacheManager()
        set_prompt_cache_manager(mgr)

        params = create_routed_cached_message(
            task="Analyze code",
            messages=[{"role": "user", "content": "Analyze"}],
            system_prompt=very_large_content,
            system_ttl=CacheTTL.ONE_HOUR,
        )
        assert params["system"][0]["cache_control"]["ttl"] == "1h"


class TestWriteCostProperty:
    def test_five_minute_write_cost(self):
        cfg = PromptCacheConfig(model="claude-sonnet-4-20250514")
        assert cfg.write_cost_per_mtok == 3.75

    def test_one_hour_write_cost(self):
        cfg = PromptCacheConfig(
            model="claude-sonnet-4-20250514",
            ttl=CacheTTL.ONE_HOUR,
        )
        assert cfg.write_cost_per_mtok == 6.0
