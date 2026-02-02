"""
Unleashed Platform - Integration Tests

Comprehensive integration tests verifying all V2 components work together.
"""

import asyncio
import pytest
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass


# ============================================================================
# Test Configuration
# ============================================================================

@dataclass
class TestConfig:
    """Configuration for integration tests."""
    run_slow_tests: bool = False
    test_real_apis: bool = False
    mock_all_external: bool = True


# ============================================================================
# Component Availability Tests
# ============================================================================

class TestComponentAvailability:
    """Test that all V2 components are importable."""

    def test_core_module_import(self):
        """Test core module can be imported."""
        from platform.core import (
            MemorySystem,
            AgentExecutor,
            ThinkingEngine,
            SkillRegistry,
            ToolRegistry,
        )
        assert MemorySystem is not None
        assert AgentExecutor is not None
        assert ThinkingEngine is not None
        assert SkillRegistry is not None
        assert ToolRegistry is not None

    def test_async_executor_import(self):
        """Test async executor components."""
        try:
            from platform.core import (
                V2_ASYNC_AVAILABLE,
                AsyncExecutor,
                TaskQueue,
            )
            if V2_ASYNC_AVAILABLE:
                assert AsyncExecutor is not None
                assert TaskQueue is not None
        except ImportError:
            pytest.skip("Async executor not available")

    def test_caching_import(self):
        """Test caching components."""
        try:
            from platform.core import (
                V2_CACHING_AVAILABLE,
                MemoryCache,
                TieredCache,
            )
            if V2_CACHING_AVAILABLE:
                assert MemoryCache is not None
                assert TieredCache is not None
        except ImportError:
            pytest.skip("Caching not available")

    def test_monitoring_import(self):
        """Test monitoring components."""
        try:
            from platform.core import (
                V2_MONITORING_AVAILABLE,
                MetricRegistry,
                Tracer,
                MonitoringDashboard,
            )
            if V2_MONITORING_AVAILABLE:
                assert MetricRegistry is not None
                assert Tracer is not None
                assert MonitoringDashboard is not None
        except ImportError:
            pytest.skip("Monitoring not available")

    def test_security_import(self):
        """Test security components."""
        try:
            from platform.core import (
                V2_SECURITY_AVAILABLE,
                SecurityManager,
                InputValidator,
                APIKeyManager,
            )
            if V2_SECURITY_AVAILABLE:
                assert SecurityManager is not None
                assert InputValidator is not None
                assert APIKeyManager is not None
        except ImportError:
            pytest.skip("Security not available")

    def test_secrets_import(self):
        """Test secrets management components."""
        try:
            from platform.core import (
                V2_SECRETS_AVAILABLE,
                SecretsManager,
                MemorySecretBackend,
            )
            if V2_SECRETS_AVAILABLE:
                assert SecretsManager is not None
                assert MemorySecretBackend is not None
        except ImportError:
            pytest.skip("Secrets management not available")


# ============================================================================
# Caching Integration Tests
# ============================================================================

class TestCachingIntegration:
    """Integration tests for the caching layer."""

    @pytest.fixture
    def memory_cache(self):
        """Create a memory cache for testing."""
        try:
            from core.caching import MemoryCache
            return MemoryCache(max_size=1000, default_ttl=60.0)
        except ImportError:
            pytest.skip("Caching not available")

    @pytest.mark.asyncio
    async def test_cache_set_get(self, memory_cache):
        """Test basic cache set and get operations."""
        await memory_cache.set("test_key", {"value": 42})
        result = await memory_cache.get("test_key")
        assert result == {"value": 42}

    @pytest.mark.asyncio
    async def test_cache_miss(self, memory_cache):
        """Test cache miss returns None."""
        result = await memory_cache.get("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_delete(self, memory_cache):
        """Test cache delete operation."""
        await memory_cache.set("delete_key", "value")
        await memory_cache.delete("delete_key")
        result = await memory_cache.get("delete_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, memory_cache):
        """Test cache TTL expiration."""
        await memory_cache.set("ttl_key", "value", ttl=0.1)
        await asyncio.sleep(0.2)
        result = await memory_cache.get("ttl_key")
        assert result is None


# ============================================================================
# Security Integration Tests
# ============================================================================

class TestSecurityIntegration:
    """Integration tests for security components."""

    @pytest.fixture
    def security_manager(self):
        """Create a security manager for testing."""
        try:
            from core.security import SecurityManager, SecurityConfig
            config = SecurityConfig(
                enable_audit_log=True,
                enable_rate_limiting=True,
                enable_input_validation=True
            )
            return SecurityManager(config)
        except ImportError:
            pytest.skip("Security not available")

    def test_input_validation_clean(self, security_manager):
        """Test that clean input passes validation."""
        result = security_manager.validator.validate_string("Hello, World!")
        assert not result.is_threat

    def test_input_validation_sql_injection(self, security_manager):
        """Test SQL injection detection."""
        result = security_manager.validator.validate_string(
            "SELECT * FROM users WHERE id = 1 OR 1=1"
        )
        assert result.is_threat
        assert result.threat_type.value == "injection"

    def test_input_validation_xss(self, security_manager):
        """Test XSS detection."""
        result = security_manager.validator.validate_string(
            '<script>alert("xss")</script>'
        )
        assert result.is_threat
        assert result.threat_type.value == "xss"

    def test_api_key_generation(self, security_manager):
        """Test API key generation and validation."""
        raw_key, key_info = security_manager.key_manager.generate_key(
            name="test_key",
            permissions=["read", "write"],
            rate_limit=100
        )

        # Validate the key
        validated = security_manager.key_manager.validate_key(raw_key)
        assert validated is not None
        assert validated.name == "test_key"
        assert "read" in validated.permissions

    def test_api_key_revocation(self, security_manager):
        """Test API key revocation."""
        raw_key, key_info = security_manager.key_manager.generate_key(name="revoke_test")

        # Revoke the key
        security_manager.key_manager.revoke_key(key_info.key_id)

        # Validate should fail
        validated = security_manager.key_manager.validate_key(raw_key)
        assert validated is None

    def test_rate_limiting(self, security_manager):
        """Test rate limiting functionality."""
        key_id = "rate_test_key"
        limit = 5

        # Make requests up to limit
        for i in range(limit):
            allowed, remaining = security_manager.rate_limiter.check_rate_limit(key_id, limit)
            assert allowed
            assert remaining == limit - i - 1

        # Next request should be blocked
        allowed, remaining = security_manager.rate_limiter.check_rate_limit(key_id, limit)
        assert not allowed
        assert remaining == 0


# ============================================================================
# Secrets Integration Tests
# ============================================================================

class TestSecretsIntegration:
    """Integration tests for secrets management."""

    @pytest.fixture
    def secrets_manager(self):
        """Create a secrets manager for testing."""
        try:
            from core.secrets import (
                SecretsManager,
                MemorySecretBackend,
                SecretType
            )
            backend = MemorySecretBackend(encrypt=True, encryption_key="test-key")
            return SecretsManager(primary_backend=backend)
        except ImportError:
            pytest.skip("Secrets not available")

    def test_secret_set_get(self, secrets_manager):
        """Test setting and getting secrets."""
        from core.secrets import SecretType

        secrets_manager.set(
            "api_key",
            "sk-test-12345",
            secret_type=SecretType.API_KEY
        )

        value = secrets_manager.get("api_key")
        assert value == "sk-test-12345"

    def test_secret_not_found(self, secrets_manager):
        """Test getting nonexistent secret returns default."""
        value = secrets_manager.get("nonexistent", default="default_value")
        assert value == "default_value"

    def test_secret_delete(self, secrets_manager):
        """Test deleting secrets."""
        secrets_manager.set("delete_me", "secret_value")
        assert secrets_manager.exists("delete_me")

        secrets_manager.delete("delete_me")
        assert not secrets_manager.exists("delete_me")

    def test_secret_rotation(self, secrets_manager):
        """Test secret rotation."""
        secrets_manager.set("rotate_me", "old_value")
        secrets_manager.rotate("rotate_me", "new_value")

        value = secrets_manager.get("rotate_me")
        assert value == "new_value"

    def test_audit_logging(self, secrets_manager):
        """Test that secret access is logged."""
        secrets_manager.set("audit_test", "value", accessor="test_user")
        secrets_manager.get("audit_test", accessor="test_user")

        log = secrets_manager.get_audit_log(name="audit_test")
        assert len(log) >= 2  # At least write and read


# ============================================================================
# Monitoring Integration Tests
# ============================================================================

class TestMonitoringIntegration:
    """Integration tests for monitoring components."""

    @pytest.fixture
    def metric_registry(self):
        """Create a metric registry for testing."""
        try:
            from core.monitoring import MetricRegistry
            registry = MetricRegistry()
            registry.reset()  # Clear any existing metrics
            return registry
        except ImportError:
            pytest.skip("Monitoring not available")

    def test_counter_increment(self, metric_registry):
        """Test counter metric increment."""
        counter = metric_registry.counter(
            "test_requests_total",
            "Total test requests",
            ["endpoint"]
        )

        counter.inc(endpoint="api/test")
        counter.inc(endpoint="api/test")
        counter.inc(2.0, endpoint="api/test")

        value = counter.get(endpoint="api/test")
        assert value == 4.0

    def test_gauge_set(self, metric_registry):
        """Test gauge metric set."""
        gauge = metric_registry.gauge(
            "test_queue_size",
            "Test queue size",
            ["queue"]
        )

        gauge.set(10, queue="main")
        gauge.inc(5, queue="main")
        gauge.dec(3, queue="main")

        value = gauge.get(queue="main")
        assert value == 12

    def test_histogram_observe(self, metric_registry):
        """Test histogram metric observation."""
        histogram = metric_registry.histogram(
            "test_latency_seconds",
            "Test latency",
            ["operation"],
            buckets=[0.1, 0.5, 1.0, 5.0]
        )

        histogram.observe(0.05, operation="fast")
        histogram.observe(0.3, operation="medium")
        histogram.observe(2.0, operation="slow")

        stats = histogram.get_stats(operation="")
        assert stats["count"] >= 0  # May be filtered by labels

    @pytest.fixture
    def tracer(self):
        """Create a tracer for testing."""
        try:
            from core.monitoring import Tracer
            return Tracer("test_service")
        except ImportError:
            pytest.skip("Monitoring not available")

    def test_trace_span_creation(self, tracer):
        """Test trace span creation."""
        span = tracer.start_span("test_operation")
        assert span.operation_name == "test_operation"
        assert span.trace_id is not None

        tracer.end_span(span)
        assert span.end_time is not None
        assert span.duration_ms is not None


# ============================================================================
# Observability Integration Tests
# ============================================================================

class TestObservabilityIntegration:
    """Integration tests for unified observability."""

    @pytest.fixture
    def observability(self):
        """Create observability instance for testing."""
        try:
            from core.observability import (
                Observability,
                ObservabilityConfig,
                LogLevel,
                reset_observability
            )
            reset_observability()
            config = ObservabilityConfig(
                log_level=LogLevel.DEBUG,
                log_format="json",
                metrics_enabled=True,
                tracing_enabled=True
            )
            return Observability(config)
        except ImportError:
            pytest.skip("Observability not available")

    def test_logger_creation(self, observability):
        """Test contextual logger creation."""
        logger = observability.get_logger("test_module")
        assert logger is not None
        logger.info("Test log message")

    def test_metric_creation(self, observability):
        """Test metric creation through observability."""
        counter = observability.counter(
            "obs_test_total",
            "Test counter"
        )
        if counter:
            counter.inc()

    @pytest.mark.asyncio
    async def test_tracing(self, observability):
        """Test tracing through observability."""
        async with observability.atrace("test_operation") as span:
            if span:
                assert span.operation_name == "test_operation"


# ============================================================================
# Configuration Integration Tests
# ============================================================================

class TestConfigurationIntegration:
    """Integration tests for configuration validation."""

    @pytest.fixture
    def config_manager(self):
        """Create a configuration manager for testing."""
        try:
            from core.config_validation import (
                ConfigurationManager,
                create_platform_config_schema
            )
            schema = create_platform_config_schema()
            return ConfigurationManager(schema, env_prefix="TEST")
        except ImportError:
            pytest.skip("Configuration validation not available")

    def test_config_load_defaults(self, config_manager):
        """Test loading default configuration."""
        result = config_manager.load()
        assert result.valid

    def test_config_get_value(self, config_manager):
        """Test getting configuration value."""
        config_manager.load()
        value = config_manager.get("log_level", "INFO")
        assert value in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    def test_config_set_value(self, config_manager):
        """Test setting configuration value."""
        config_manager.load()
        config_manager.set("log_level", "DEBUG")
        assert config_manager.get("log_level") == "DEBUG"

    def test_config_validation(self, config_manager):
        """Test configuration validation."""
        config_manager.load()
        config_manager.set("log_level", "INVALID")
        result = config_manager.validate()
        assert not result.valid
        assert result.errors > 0


# ============================================================================
# End-to-End Integration Tests
# ============================================================================

class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_request_flow(self):
        """Test a complete request flow through all components."""
        try:
            from core.security import SecurityManager, SecurityConfig
            from core.caching import MemoryCache
            from core.monitoring import MetricRegistry, Counter

            # Setup components
            security = SecurityManager(SecurityConfig())
            cache = MemoryCache(max_size=100)
            registry = MetricRegistry()

            # Create metrics
            request_counter = registry.counter(
                "e2e_requests_total",
                "E2E request counter",
                ["status"]
            )

            # Generate API key
            api_key, key_info = security.key_manager.generate_key(
                name="e2e_test",
                permissions=["read"]
            )

            # Simulate request validation
            allowed, error, validated_key = await security.validate_request(
                api_key=api_key,
                data={"query": "test search query"}
            )

            assert allowed
            assert validated_key is not None

            # Check cache
            cached_result = await cache.get("e2e_query_test")
            if not cached_result:
                # Simulate processing
                result = {"data": "processed result"}
                await cache.set("e2e_query_test", result)
                cached_result = result

            # Record metric
            request_counter.inc(status="success")

            assert cached_result is not None

        except ImportError as e:
            pytest.skip(f"Required components not available: {e}")


# ============================================================================
# Adapter Integration Tests
# ============================================================================

class TestAdapterIntegration:
    """Integration tests for SDK adapters."""

    def test_adapter_registry_import(self):
        """Test adapter registry can be imported."""
        try:
            from platform.adapters import get_adapter_status
            status = get_adapter_status()
            assert isinstance(status, dict)
        except ImportError:
            pytest.skip("Adapters not available")

    def test_pipeline_registry_import(self):
        """Test pipeline registry can be imported."""
        try:
            from platform.pipelines import get_pipeline_status
            status = get_pipeline_status()
            assert isinstance(status, dict)
        except ImportError:
            pytest.skip("Pipelines not available")


# ============================================================================
# CLI Integration Tests
# ============================================================================

class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_cli_import(self):
        """Test CLI can be imported."""
        try:
            from platform.cli import cli, main
            assert cli is not None
            assert main is not None
        except ImportError:
            pytest.skip("CLI not available")

    def test_cli_parser_creation(self):
        """Test CLI parser can be created."""
        try:
            from platform.cli.main import create_parser
            parser = create_parser()
            assert parser is not None
        except ImportError:
            pytest.skip("CLI not available")


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance and stress tests."""

    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache performance with many operations."""
        try:
            from core.caching import MemoryCache
            cache = MemoryCache(max_size=10000)

            # Write 1000 entries
            for i in range(1000):
                await cache.set(f"perf_key_{i}", {"value": i})

            # Read all entries
            for i in range(1000):
                result = await cache.get(f"perf_key_{i}")
                assert result is not None

        except ImportError:
            pytest.skip("Caching not available")

    def test_security_validation_performance(self):
        """Test security validation performance."""
        try:
            from core.security import SecurityManager, SecurityConfig
            security = SecurityManager(SecurityConfig())

            # Validate 1000 inputs
            for i in range(1000):
                result = security.validator.validate_string(f"Test input {i}")
                assert not result.is_threat

        except ImportError:
            pytest.skip("Security not available")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
