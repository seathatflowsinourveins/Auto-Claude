"""
Tests for Ralph Loop Production Enhancements

Tests the V37 production features:
- Structured logging with correlation IDs
- Prometheus metrics
- Rate limiting
- Graceful shutdown
- State checkpointing
- V11 enhanced feature managers
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict
import pytest

# Import production modules
from core.ralph.production import (
    # Logging
    RalphLogger,
    StructuredLogFormatter,
    configure_production_logging,
    correlation_context,
    get_correlation_id,
    set_correlation_id,
    # Metrics
    RalphMetrics,
    get_metrics,
    # Rate Limiting
    RateLimitConfig,
    RateLimiter,
    TokenBucket,
    get_rate_limiter,
    # Shutdown
    ShutdownHandler,
    # Checkpointing
    CheckpointManager,
    CheckpointMetadata,
    # V11 Managers
    SpeculativeDecodingManager,
    ChainOfDraftManager,
    AdaptiveRAGManager,
    RewardHackingDetector,
    HypothesisTracker,
    # Config
    ProductionConfig,
    initialize_production,
)


class TestCorrelationContext:
    """Tests for correlation ID management."""

    def test_get_correlation_id_generates_new(self):
        """Should generate a new correlation ID if none exists."""
        # Clear any existing context
        import threading
        if hasattr(threading.local(), 'correlation_id'):
            delattr(threading.local(), 'correlation_id')

        cid = get_correlation_id()
        assert cid is not None
        assert len(cid) == 12

    def test_set_correlation_id(self):
        """Should set a specific correlation ID."""
        set_correlation_id("test123456")
        assert get_correlation_id() == "test123456"

    def test_correlation_context_manager(self):
        """Should manage correlation ID scope."""
        with correlation_context("ctx_abc123") as cid:
            assert cid == "ctx_abc123"
            assert get_correlation_id() == "ctx_abc123"

    def test_nested_correlation_context(self):
        """Should handle nested correlation contexts."""
        with correlation_context("outer123") as outer:
            assert outer == "outer123"
            with correlation_context("inner456") as inner:
                assert inner == "inner456"
                assert get_correlation_id() == "inner456"


class TestStructuredLogging:
    """Tests for structured logging."""

    def test_structured_log_formatter_json(self):
        """Should format logs as JSON."""
        import logging
        formatter = StructuredLogFormatter("test_service")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["service"] == "test_service"
        assert "timestamp" in parsed
        assert "correlation_id" in parsed

    def test_ralph_logger_context(self):
        """Should include context in logs."""
        logger = RalphLogger("test", loop_id="loop_123")
        logger.set_context(iteration=5, strategy="mcts")

        assert logger._current_iteration == 5
        assert logger._current_strategy == "mcts"


class TestRalphMetrics:
    """Tests for Prometheus-compatible metrics."""

    def test_counter_increment(self):
        """Should increment counter metrics."""
        metrics = RalphMetrics("test")
        metrics.inc_counter("test_counter")
        metrics.inc_counter("test_counter", 5.0)

        snapshot = metrics.get_snapshot()
        assert "test_test_counter" in snapshot["counters"]

    def test_gauge_set(self):
        """Should set gauge values."""
        metrics = RalphMetrics("test")
        metrics.set_gauge("test_gauge", 42.5)

        snapshot = metrics.get_snapshot()
        assert "test_test_gauge" in snapshot["gauges"]

    def test_histogram_observe(self):
        """Should record histogram observations."""
        metrics = RalphMetrics("test")
        for i in range(10):
            metrics.observe_histogram("test_hist", i * 0.1)

        snapshot = metrics.get_snapshot()
        assert "test_test_hist" in snapshot["histograms"]

    def test_timer_context_manager(self):
        """Should measure time with context manager."""
        metrics = RalphMetrics("test")

        with metrics.timer("test_timer"):
            time.sleep(0.01)

        snapshot = metrics.get_snapshot()
        assert "test_test_timer" in snapshot["histograms"]
        assert snapshot["histograms"]["test_test_timer"][""]["count"] == 1

    def test_prometheus_export(self):
        """Should export metrics in Prometheus format."""
        metrics = RalphMetrics("prom")
        metrics.inc_counter("requests", labels={"method": "GET"})
        metrics.set_gauge("active", 5)

        output = metrics.export_prometheus()
        assert "prom_requests" in output
        assert "prom_active" in output


class TestRateLimiter:
    """Tests for rate limiting."""

    def test_token_bucket_acquire(self):
        """Should acquire tokens from bucket."""
        bucket = TokenBucket(rate=10.0, capacity=5)

        success, wait = bucket.acquire(1)
        assert success is True
        assert wait == 0.0

    def test_token_bucket_exhausted(self):
        """Should report wait time when exhausted."""
        bucket = TokenBucket(rate=1.0, capacity=2)

        # Exhaust tokens
        bucket.acquire(2)

        success, wait = bucket.acquire(1)
        assert success is False
        assert wait > 0

    @pytest.mark.asyncio
    async def test_rate_limiter_async_acquire(self):
        """Should acquire with async waiting."""
        config = RateLimitConfig(
            requests_per_second=100.0,
            burst_size=10,
            max_retries=3
        )
        limiter = RateLimiter(config)

        # Should succeed quickly
        success = await limiter.acquire("test")
        assert success is True


class TestShutdownHandler:
    """Tests for graceful shutdown."""

    def test_shutdown_handler_not_shutting_down(self):
        """Should report not shutting down initially."""
        handler = ShutdownHandler(timeout_seconds=5.0)
        assert handler.is_shutting_down is False

    def test_shutdown_handler_initiate(self):
        """Should initiate shutdown."""
        handler = ShutdownHandler(timeout_seconds=5.0)
        handler.initiate_shutdown()
        assert handler.is_shutting_down is True

    def test_cleanup_callback(self):
        """Should call cleanup callbacks on shutdown."""
        handler = ShutdownHandler(timeout_seconds=5.0)
        cleanup_called = []

        def cleanup():
            cleanup_called.append(True)

        handler.register_cleanup(cleanup)
        handler.initiate_shutdown()

        assert len(cleanup_called) == 1


class TestCheckpointManager:
    """Tests for state checkpointing."""

    def test_save_and_load_checkpoint(self):
        """Should save and load checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=Path(tmpdir),
                max_checkpoints=5
            )

            state = {"iteration": 10, "fitness": 0.85, "data": "test"}

            metadata = manager.save_checkpoint(
                loop_id="test_loop",
                state_dict=state,
                iteration=10,
                fitness=0.85,
                status="running"
            )

            assert metadata.iteration == 10
            assert metadata.fitness == 0.85

            # Load checkpoint
            result = manager.load_checkpoint("test_loop", metadata.checkpoint_id)
            assert result is not None

            loaded_meta, loaded_state = result
            assert loaded_state["iteration"] == 10
            assert loaded_state["fitness"] == 0.85

    def test_checkpoint_cleanup(self):
        """Should remove old checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=Path(tmpdir),
                max_checkpoints=3
            )

            # Create more checkpoints than max
            for i in range(5):
                manager.save_checkpoint(
                    loop_id="test_loop",
                    state_dict={"iteration": i},
                    iteration=i,
                    fitness=0.5,
                    status="running"
                )

            # Should only have max_checkpoints
            checkpoints = manager.list_checkpoints("test_loop")
            assert len(checkpoints) <= 3


class TestSpeculativeDecodingManager:
    """Tests for speculative decoding."""

    @pytest.mark.asyncio
    async def test_generate_hypotheses(self):
        """Should generate multiple hypotheses."""
        manager = SpeculativeDecodingManager(
            max_parallel_hypotheses=3,
            verification_timeout_seconds=5.0
        )

        async def generator(context: str) -> Dict[str, Any]:
            return {"content": f"Hypothesis for {context}", "confidence": 0.8}

        hypotheses = await manager.generate_hypotheses(
            "test context",
            generator,
            num_hypotheses=3
        )

        assert len(hypotheses) == 3
        for h in hypotheses:
            assert h.confidence == 0.8
            assert "test context" in h.content

    @pytest.mark.asyncio
    async def test_verify_hypothesis(self):
        """Should verify a hypothesis."""
        manager = SpeculativeDecodingManager()

        hypothesis = HypothesisTracker(
            hypothesis_id="test_1",
            content="Test hypothesis",
            confidence=0.8,
            generation_cost=100,
            generated_at="2025-01-01T00:00:00Z"
        )

        async def verifier(content: str) -> Dict[str, Any]:
            return {"verified": True, "reasoning": "Looks good"}

        result = await manager.verify_hypothesis(hypothesis, verifier)

        assert result.verification_status == "verified"
        assert result.verification_result is True

    def test_acceptance_rate(self):
        """Should track acceptance rate."""
        manager = SpeculativeDecodingManager()

        # Simulate some verifications
        manager._total_verified = 7
        manager._total_rejected = 3

        assert manager.acceptance_rate == 0.7


class TestChainOfDraftManager:
    """Tests for Chain-of-Draft compression."""

    @pytest.mark.asyncio
    async def test_compress_reasoning(self):
        """Should compress reasoning chains."""
        manager = ChainOfDraftManager(max_tokens_per_step=5)

        async def compressor(text: str) -> Dict[str, Any]:
            # Simulate compression
            words = text.split()
            return {
                "steps": [" ".join(words[i:i+5]) for i in range(0, len(words), 5)],
                "original_tokens": len(words)
            }

        steps, stats = await manager.compress_reasoning(
            "This is a longer reasoning chain that should be compressed into smaller steps",
            compressor
        )

        assert len(steps) > 0
        assert "compression_ratio" in stats
        assert stats["compression_ratio"] > 0

    def test_overall_compression(self):
        """Should track overall compression ratio."""
        manager = ChainOfDraftManager()
        manager._total_draft_tokens = 100
        manager._total_equivalent_cot_tokens = 500

        assert manager.get_overall_compression() == 0.8


class TestAdaptiveRAGManager:
    """Tests for adaptive RAG."""

    def test_should_retrieve_low_confidence(self):
        """Should recommend retrieval for low confidence."""
        manager = AdaptiveRAGManager(
            confidence_threshold=0.7,
            novelty_threshold=0.5
        )

        should, reason = manager.should_retrieve(confidence=0.3, novelty=0.2)
        assert should is True
        assert "confidence" in reason.lower()

    def test_should_retrieve_high_novelty(self):
        """Should recommend retrieval for high novelty."""
        manager = AdaptiveRAGManager(
            confidence_threshold=0.7,
            novelty_threshold=0.5
        )

        should, reason = manager.should_retrieve(confidence=0.9, novelty=0.8)
        assert should is True
        assert "novelty" in reason.lower()

    def test_should_not_retrieve_internal(self):
        """Should skip retrieval when internal knowledge sufficient."""
        manager = AdaptiveRAGManager(
            confidence_threshold=0.7,
            novelty_threshold=0.5
        )

        should, reason = manager.should_retrieve(confidence=0.9, novelty=0.2)
        assert should is False

    @pytest.mark.asyncio
    async def test_perform_retrieval(self):
        """Should perform retrieval when needed."""
        manager = AdaptiveRAGManager()

        async def retriever(query: str) -> Dict[str, Any]:
            return {"documents": ["Doc 1", "Doc 2"], "scores": [0.9, 0.8]}

        result = await manager.perform_retrieval(
            "test query",
            retriever,
            confidence=0.3,
            novelty=0.5
        )

        assert result["retrieved"] is True
        assert len(result["results"]) == 2


class TestRewardHackingDetector:
    """Tests for reward hacking detection."""

    def test_check_proxy_divergence(self):
        """Should detect proxy-true reward divergence."""
        detector = RewardHackingDetector(
            proxy_divergence_threshold=0.3,
            history_window=10
        )

        # Simulate history where proxy improves but true stagnates
        for i in range(10):
            detector.record_rewards(
                true_reward=0.5,  # Stagnant
                proxy_reward=0.5 + i * 0.05  # Improving
            )

        signal = detector.check_proxy_divergence(0.5, 0.95)
        assert signal is not None
        assert signal.signal_type == "proxy_divergence"

    def test_check_suspicious_improvement(self):
        """Should detect suspiciously large improvements."""
        detector = RewardHackingDetector(
            suspicious_improvement_threshold=0.5,
            history_window=20
        )

        # Build normal history
        for i in range(15):
            detector.record_rewards(0.5 + i * 0.01, 0.5 + i * 0.01)

        # Check for anomalous improvement
        signal = detector.check_suspicious_improvement(0.8)  # Huge jump
        assert signal is not None
        assert signal.signal_type == "suspicious_pattern"

    def test_check_specification_gaming(self):
        """Should detect specification gaming patterns."""
        detector = RewardHackingDetector()

        # Check empty solution
        signal = detector.check_specification_gaming("", "previous", 0.5)
        assert signal is not None
        assert signal.signal_type == "specification_gaming"

        # Check excessive repetition
        repetitive = "word " * 100
        signal = detector.check_specification_gaming(repetitive, "different", 0.3)
        assert signal is not None

    def test_get_statistics(self):
        """Should track detection statistics."""
        detector = RewardHackingDetector()

        # Create some signals
        detector._create_signal(
            "proxy_divergence", 0.7,
            "Test signal", {"test": True}
        )

        stats = detector.get_statistics()
        assert stats["total_signals"] == 1
        assert "proxy_divergence" in stats["signals_by_type"]


class TestProductionConfig:
    """Tests for production configuration."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = ProductionConfig()

        assert config.log_level == "INFO"
        assert config.api_requests_per_second == 10.0
        assert config.max_checkpoints == 10
        assert config.shutdown_timeout_seconds == 30.0

    def test_initialize_production(self):
        """Should initialize all production components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProductionConfig(
                checkpoint_dir=Path(tmpdir),
                log_level="DEBUG"
            )

            components = initialize_production(config)

            assert "logger" in components
            assert "metrics" in components
            assert "rate_limiter" in components
            assert "shutdown_handler" in components
            assert "checkpoint_manager" in components
            assert "speculative_manager" in components
            assert "cod_manager" in components
            assert "rag_manager" in components
            assert "reward_detector" in components


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
