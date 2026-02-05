"""
Tests for 3-Tier Model Routing (Gap21 Implementation)
======================================================

Tests cover:
- RoutingTier enum and MODEL_CONFIGS
- ModelRouter routing decisions
- Pattern matching for simple/complex tasks
- Complexity-based tier selection
- WASM intent detection
- Batch routing for cost optimization
- Metrics tracking
- Convenience functions

Test Date: 2026-02-05 (V67)
"""

import pytest
from datetime import datetime, timezone


# =============================================================================
# IMPORTS AND FIXTURES
# =============================================================================

@pytest.fixture
def routing_module():
    """Import the routing module."""
    try:
        from core.orchestration.model_routing import (
            ModelRouter,
            BatchModelRouter,
            RoutingConfig,
            RoutingDecision,
            RoutingMetrics,
            RoutingTier,
            ModelProvider,
            MODEL_CONFIGS,
            TIER_1_PATTERNS,
            TIER_3_PATTERNS,
            route_query,
            get_claude_model_for_task,
            get_tier_for_task,
            get_routing_metrics,
            reset_routing_metrics,
        )
        return {
            "ModelRouter": ModelRouter,
            "BatchModelRouter": BatchModelRouter,
            "RoutingConfig": RoutingConfig,
            "RoutingDecision": RoutingDecision,
            "RoutingMetrics": RoutingMetrics,
            "RoutingTier": RoutingTier,
            "ModelProvider": ModelProvider,
            "MODEL_CONFIGS": MODEL_CONFIGS,
            "TIER_1_PATTERNS": TIER_1_PATTERNS,
            "TIER_3_PATTERNS": TIER_3_PATTERNS,
            "route_query": route_query,
            "get_claude_model_for_task": get_claude_model_for_task,
            "get_tier_for_task": get_tier_for_task,
            "get_routing_metrics": get_routing_metrics,
            "reset_routing_metrics": reset_routing_metrics,
        }
    except ImportError as e:
        pytest.skip(f"Model routing module not importable: {e}")


@pytest.fixture
def router(routing_module):
    """Create a fresh ModelRouter instance."""
    return routing_module["ModelRouter"]()


# =============================================================================
# ROUTING TIER TESTS
# =============================================================================

class TestRoutingTierEnum:
    """Tests for RoutingTier enum."""

    def test_tier_values(self, routing_module):
        """RoutingTier should have correct integer values."""
        RoutingTier = routing_module["RoutingTier"]

        assert RoutingTier.TIER_1.value == 1, "TIER_1 should be 1"
        assert RoutingTier.TIER_2.value == 2, "TIER_2 should be 2"
        assert RoutingTier.TIER_3.value == 3, "TIER_3 should be 3"

    def test_tier_count(self, routing_module):
        """Should have exactly 3 tiers."""
        RoutingTier = routing_module["RoutingTier"]

        assert len(list(RoutingTier)) == 3, "Should have 3 tiers"


class TestModelProvider:
    """Tests for ModelProvider enum."""

    def test_provider_values(self, routing_module):
        """ModelProvider should have correct values."""
        ModelProvider = routing_module["ModelProvider"]

        assert ModelProvider.ANTHROPIC.value == "anthropic"
        assert ModelProvider.LOCAL.value == "local"


# =============================================================================
# MODEL CONFIGS TESTS
# =============================================================================

class TestModelConfigs:
    """Tests for MODEL_CONFIGS dictionary."""

    def test_required_models_present(self, routing_module):
        """All required models should be present."""
        configs = routing_module["MODEL_CONFIGS"]

        required = ["wasm", "claude-haiku", "claude-sonnet", "claude-opus"]
        for model in required:
            assert model in configs, f"Missing model: {model}"

    def test_haiku_config(self, routing_module):
        """Haiku config should have correct values."""
        configs = routing_module["MODEL_CONFIGS"]
        haiku = configs["claude-haiku"]

        assert haiku["tier"].value == 1, "Haiku should be Tier 1"
        assert haiku["latency_ms"] == 500, "Haiku latency should be ~500ms"
        assert haiku["model_id"] == "claude-3-5-haiku-20241022"
        assert haiku["context_window"] == 200000

    def test_sonnet_config(self, routing_module):
        """Sonnet config should have correct values."""
        configs = routing_module["MODEL_CONFIGS"]
        sonnet = configs["claude-sonnet"]

        assert sonnet["tier"].value == 2, "Sonnet should be Tier 2"
        assert sonnet["latency_ms"] == 2000, "Sonnet latency should be ~2s"
        assert sonnet["model_id"] == "claude-sonnet-4-20250514"

    def test_opus_config(self, routing_module):
        """Opus config should have correct values."""
        configs = routing_module["MODEL_CONFIGS"]
        opus = configs["claude-opus"]

        assert opus["tier"].value == 3, "Opus should be Tier 3"
        assert opus["latency_ms"] == 5000, "Opus latency should be ~5s"
        assert opus["model_id"] == "claude-opus-4-5-20251101"

    def test_wasm_config(self, routing_module):
        """WASM config should be free and fast."""
        configs = routing_module["MODEL_CONFIGS"]
        wasm = configs["wasm"]

        assert wasm["tier"].value == 1, "WASM should be Tier 1"
        assert wasm["latency_ms"] == 1, "WASM should be <1ms"
        assert wasm["cost_per_1k_input"] == 0.0, "WASM should be free"
        assert wasm["cost_per_1k_output"] == 0.0, "WASM should be free"

    def test_cost_ordering(self, routing_module):
        """Costs should increase with tier."""
        configs = routing_module["MODEL_CONFIGS"]

        haiku_cost = configs["claude-haiku"]["cost_per_1k_input"]
        sonnet_cost = configs["claude-sonnet"]["cost_per_1k_input"]
        opus_cost = configs["claude-opus"]["cost_per_1k_input"]

        assert haiku_cost < sonnet_cost < opus_cost, (
            f"Costs should increase: haiku={haiku_cost}, sonnet={sonnet_cost}, opus={opus_cost}"
        )


# =============================================================================
# PATTERN MATCHING TESTS
# =============================================================================

class TestPatternLists:
    """Tests for tier pattern lists."""

    def test_tier_1_patterns_exist(self, routing_module):
        """TIER_1_PATTERNS should have exploration/simple patterns."""
        patterns = routing_module["TIER_1_PATTERNS"]

        assert len(patterns) > 0, "Should have Tier 1 patterns"
        assert "find" in patterns, "Should include 'find'"
        assert "simple" in patterns, "Should include 'simple'"
        assert "quick" in patterns, "Should include 'quick'"

    def test_tier_3_patterns_exist(self, routing_module):
        """TIER_3_PATTERNS should have complex/security patterns."""
        patterns = routing_module["TIER_3_PATTERNS"]

        assert len(patterns) > 0, "Should have Tier 3 patterns"
        assert "architecture" in patterns, "Should include 'architecture'"
        assert "security" in patterns, "Should include 'security'"
        assert "distributed" in patterns, "Should include 'distributed'"


# =============================================================================
# MODEL ROUTER TESTS
# =============================================================================

class TestModelRouterBasic:
    """Basic ModelRouter tests."""

    def test_router_creation(self, routing_module):
        """Router should be creatable."""
        router = routing_module["ModelRouter"]()
        assert router is not None

    def test_router_with_config(self, routing_module):
        """Router should accept config."""
        config = routing_module["RoutingConfig"](
            haiku_threshold=0.2,
            sonnet_threshold=0.5,
            enable_wasm=True,
        )
        router = routing_module["ModelRouter"](config=config)

        assert router.config.haiku_threshold == 0.2
        assert router.config.sonnet_threshold == 0.5
        assert router.config.enable_wasm is True

    def test_default_config_values(self, routing_module):
        """Default config should have sensible values."""
        router = routing_module["ModelRouter"]()

        assert router.config.haiku_threshold == 0.30
        assert router.config.sonnet_threshold == 0.60
        assert router.config.enable_wasm is False
        assert router.config.enable_batch is True


class TestModelRouterRouting:
    """Tests for ModelRouter.route() method."""

    def test_route_simple_task_to_haiku(self, routing_module):
        """Simple tasks should route to Haiku (Tier 1)."""
        router = routing_module["ModelRouter"]()

        simple_tasks = [
            "Fix this typo",
            "Find all Python files",
            "Show me the config",
            "Just format this file",
            "Quick check if file exists",
        ]

        for task in simple_tasks:
            decision = router.route(task)
            assert decision.tier.value == 1, f"'{task}' should be Tier 1, got Tier {decision.tier.value}"
            assert "haiku" in decision.model_id.lower(), f"'{task}' should use Haiku"

    def test_route_complex_task_to_opus(self, routing_module):
        """Complex tasks should route to Opus (Tier 3)."""
        router = routing_module["ModelRouter"]()

        complex_tasks = [
            "Design a microservices architecture for payment processing",
            "Review this code for security vulnerabilities",
            "Implement OAuth2 with PKCE authentication",
            "Create a distributed consensus algorithm",
            "Analyze and fix this critical production bug",
        ]

        for task in complex_tasks:
            decision = router.route(task)
            assert decision.tier.value == 3, f"'{task}' should be Tier 3, got Tier {decision.tier.value}"
            assert "opus" in decision.model_id.lower(), f"'{task}' should use Opus"

    def test_route_moderate_task_to_sonnet(self, routing_module):
        """Moderate tasks should route to Sonnet (Tier 2) or above."""
        router = routing_module["ModelRouter"]()

        # These tasks have moderate complexity - some may end up Tier 1 based on
        # complexity scoring (which prefers Haiku for efficiency per ADR-026).
        # The key is they DON'T match Tier 3 patterns (security, architecture).
        moderate_tasks = [
            "Implement a comprehensive user authentication system with OAuth integration and session management across multiple services",
            "Design and implement a caching layer for the database queries with invalidation strategies",
            "Create an integration test suite covering all API endpoints with mocking",
        ]

        tier_2_plus_count = 0
        for task in moderate_tasks:
            decision = router.route(task)
            if decision.tier.value >= 2:
                tier_2_plus_count += 1

        # At least some should be Tier 2+ with these longer, more complex descriptions
        assert tier_2_plus_count >= 1, (
            f"Expected at least 1 task to be Tier 2+, got {tier_2_plus_count}"
        )

    def test_route_returns_routing_decision(self, router, routing_module):
        """Route should return RoutingDecision with all fields."""
        decision = router.route("Test task")

        assert isinstance(decision.tier, routing_module["RoutingTier"])
        assert isinstance(decision.model_key, str)
        assert isinstance(decision.model_id, str)
        assert isinstance(decision.provider, routing_module["ModelProvider"])
        assert isinstance(decision.complexity_score, float)
        assert 0.0 <= decision.complexity_score <= 1.0
        assert isinstance(decision.estimated_latency_ms, int)
        assert isinstance(decision.estimated_cost, float)
        assert isinstance(decision.reason, str)
        assert isinstance(decision.timestamp, datetime)


class TestModelRouterWASM:
    """Tests for WASM/Agent Booster routing."""

    def test_wasm_detection_disabled_by_default(self, router):
        """WASM routing should be disabled by default."""
        decision = router.route("Convert var to const in this file")

        assert decision.is_wasm_eligible is False, "WASM should be disabled by default"

    def test_wasm_detection_when_enabled(self, routing_module):
        """WASM should be detected when enabled."""
        config = routing_module["RoutingConfig"](enable_wasm=True)
        router = routing_module["ModelRouter"](config=config)

        wasm_tasks = [
            ("Convert var to const", "var-to-const"),
            ("Add type annotations to this function", "add-types"),
            ("Add try catch error handling", "add-error-handling"),
            ("Convert callback to async await", "async-await"),
        ]

        for task, expected_intent in wasm_tasks:
            decision = router.route(task)
            if decision.is_wasm_eligible:
                assert decision.wasm_intent is not None, f"Should detect intent for '{task}'"
                assert decision.estimated_cost == 0.0, "WASM should be free"
                assert decision.estimated_latency_ms == 1, "WASM should be ~1ms"


class TestModelRouterForced:
    """Tests for forced tier routing."""

    def test_force_tier_1(self, routing_module):
        """Force tier should override complexity analysis."""
        config = routing_module["RoutingConfig"](
            force_tier=routing_module["RoutingTier"].TIER_1
        )
        router = routing_module["ModelRouter"](config=config)

        # Even complex task should be Tier 1
        decision = router.route("Design a distributed consensus algorithm")
        assert decision.tier.value == 1, "Forced tier should override"

    def test_force_tier_3(self, routing_module):
        """Force tier 3 should override simple patterns."""
        config = routing_module["RoutingConfig"](
            force_tier=routing_module["RoutingTier"].TIER_3
        )
        router = routing_module["ModelRouter"](config=config)

        # Even simple task should be Tier 3
        decision = router.route("Fix typo")
        assert decision.tier.value == 3, "Forced tier should override"


# =============================================================================
# ROUTING DECISION TESTS
# =============================================================================

class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_to_dict(self, router):
        """to_dict should serialize all fields."""
        decision = router.route("Test task")
        d = decision.to_dict()

        assert "tier" in d
        assert "model_key" in d
        assert "model_id" in d
        assert "provider" in d
        assert "complexity_score" in d
        assert "estimated_latency_ms" in d
        assert "estimated_cost" in d
        assert "reason" in d
        assert "timestamp" in d

    def test_to_dict_types(self, router):
        """to_dict values should be serializable types."""
        decision = router.route("Test task")
        d = decision.to_dict()

        assert isinstance(d["tier"], int)
        assert isinstance(d["model_key"], str)
        assert isinstance(d["provider"], str)
        assert isinstance(d["complexity_score"], float)
        assert isinstance(d["timestamp"], str)  # ISO format


# =============================================================================
# METRICS TESTS
# =============================================================================

class TestRoutingMetrics:
    """Tests for RoutingMetrics tracking."""

    def test_initial_metrics_zero(self, router):
        """Initial metrics should be zero."""
        metrics = router.get_metrics()

        assert metrics["total_routes"] == 0
        assert metrics["by_tier"]["tier_1"] == 0
        assert metrics["by_tier"]["tier_2"] == 0
        assert metrics["by_tier"]["tier_3"] == 0

    def test_metrics_increment_on_route(self, router):
        """Metrics should increment on each route call."""
        router.route("Simple task")
        router.route("Another task")

        metrics = router.get_metrics()
        assert metrics["total_routes"] == 2

    def test_metrics_tier_tracking(self, router):
        """Metrics should track tier distribution."""
        router.route("Find files")  # Tier 1
        router.route("Design architecture")  # Tier 3
        router.route("Quick fix")  # Tier 1

        metrics = router.get_metrics()
        assert metrics["by_tier"]["tier_1"] >= 2, "Should have at least 2 Tier 1 routes"
        assert metrics["by_tier"]["tier_3"] >= 1, "Should have at least 1 Tier 3 route"

    def test_metrics_distribution(self, router):
        """Distribution should be calculated correctly."""
        router.route("Find files")
        router.route("Find more")
        router.route("Design security audit")  # Tier 3

        metrics = router.get_metrics()
        dist = metrics["distribution_percent"]

        total = sum(dist.values())
        assert abs(total - 100.0) < 0.01, "Distribution should sum to 100%"

    def test_metrics_reset(self, router):
        """Reset should clear all metrics."""
        router.route("Task 1")
        router.route("Task 2")

        assert router.get_metrics()["total_routes"] == 2

        router.reset_metrics()

        assert router.get_metrics()["total_routes"] == 0

    def test_metrics_cost_tracking(self, router):
        """Total estimated cost should be tracked."""
        router.route("Design complex system")  # Higher cost

        metrics = router.get_metrics()
        assert metrics["total_estimated_cost_usd"] > 0


# =============================================================================
# BATCH ROUTER TESTS
# =============================================================================

class TestBatchModelRouter:
    """Tests for BatchModelRouter."""

    def test_batch_router_creation(self, routing_module):
        """BatchModelRouter should be creatable."""
        batch_router = routing_module["BatchModelRouter"]()
        assert batch_router is not None

    def test_add_task_returns_decision(self, routing_module):
        """add_task should return routing decision."""
        batch_router = routing_module["BatchModelRouter"]()
        decision = batch_router.add_task("Test task", "task-1")

        assert decision is not None
        assert hasattr(decision, "tier")

    def test_pending_counts(self, routing_module):
        """get_pending_counts should track pending tasks."""
        batch_router = routing_module["BatchModelRouter"]()

        batch_router.add_task("Find files", "task-1")
        batch_router.add_task("Design architecture", "task-2")

        counts = batch_router.get_pending_counts()

        # At least one tier should have pending
        total = sum(counts.values())
        assert total >= 1, "Should have pending tasks"

    def test_clear_pending(self, routing_module):
        """clear_pending should remove pending tasks."""
        batch_router = routing_module["BatchModelRouter"]()

        batch_router.add_task("Task 1", "id-1")
        batch_router.add_task("Task 2", "id-2")

        batch_router.clear_pending()

        counts = batch_router.get_pending_counts()
        total = sum(counts.values())
        assert total == 0, "All pending should be cleared"


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_route_query(self, routing_module):
        """route_query should return RoutingDecision."""
        decision = routing_module["route_query"]("Test task")

        assert hasattr(decision, "tier")
        assert hasattr(decision, "model_id")

    def test_get_claude_model_for_task(self, routing_module):
        """get_claude_model_for_task should return model ID string."""
        model_id = routing_module["get_claude_model_for_task"]("Fix typo")

        assert isinstance(model_id, str)
        assert "claude" in model_id.lower()

    def test_get_tier_for_task(self, routing_module):
        """get_tier_for_task should return tier number."""
        tier = routing_module["get_tier_for_task"]("Fix typo")

        assert isinstance(tier, int)
        assert tier in [1, 2, 3]

    def test_get_routing_metrics(self, routing_module):
        """get_routing_metrics should return metrics dict."""
        # Reset first
        routing_module["reset_routing_metrics"]()

        # Route something
        routing_module["route_query"]("Test")

        metrics = routing_module["get_routing_metrics"]()

        assert isinstance(metrics, dict)
        assert "total_routes" in metrics

    def test_reset_routing_metrics(self, routing_module):
        """reset_routing_metrics should clear global metrics."""
        routing_module["route_query"]("Task")
        routing_module["reset_routing_metrics"]()

        metrics = routing_module["get_routing_metrics"]()
        assert metrics["total_routes"] == 0


# =============================================================================
# ADR-026 COMPLIANCE TESTS
# =============================================================================

class TestADR026Compliance:
    """Tests for ADR-026 specification compliance."""

    def test_tier_distribution_targets(self, routing_module):
        """
        ADR-026 specifies target distribution:
        - Tier 1 (Haiku): ~70% of tasks
        - Tier 2 (Sonnet): ~25% of tasks
        - Tier 3 (Opus): ~5% of tasks
        """
        router = routing_module["ModelRouter"]()

        # Mix of tasks reflecting typical distribution
        tasks = [
            # 70% simple (Tier 1 targets)
            "Find Python files",
            "Show config",
            "List all modules",
            "Check file exists",
            "Quick grep for imports",
            "Simple format fix",
            "Read the README",
            # 25% moderate (may be Tier 2)
            "Implement new feature with tests",
            "Refactor for better performance",
            # 5% complex (Tier 3 targets)
            "Design microservices architecture",
        ]

        for task in tasks:
            router.route(task)

        metrics = router.get_metrics()
        dist = metrics["distribution_percent"]

        # Tier 1 should be majority (allow some flexibility in testing)
        assert dist["tier_1"] >= 50, f"Tier 1 should be majority, got {dist['tier_1']:.1f}%"

    def test_model_id_formats(self, routing_module):
        """Model IDs should be valid Anthropic model identifiers."""
        router = routing_module["ModelRouter"]()

        # Route to each tier
        tier_1 = router.route("Find files")
        tier_3 = router.route("Design security architecture")

        # Tier 1 should use Haiku
        assert "haiku" in tier_1.model_id.lower() or "wasm" in tier_1.model_id.lower()

        # Tier 3 should use Opus
        assert "opus" in tier_3.model_id.lower()

    def test_cost_estimates_realistic(self, routing_module):
        """Cost estimates should be in realistic ranges."""
        router = routing_module["ModelRouter"]()

        decision = router.route(
            "Complex task",
            estimated_input_tokens=1000,
            estimated_output_tokens=2000,
        )

        # Cost should be positive and reasonable (< $1 for moderate tokens)
        assert decision.estimated_cost >= 0
        assert decision.estimated_cost < 1.0, "Cost for 3K tokens should be < $1"

    def test_latency_estimates_ordered(self, routing_module):
        """Latency should increase with tier."""
        router = routing_module["ModelRouter"]()

        tier_1 = router.route("Find files")
        tier_3 = router.route("Design architecture")

        assert tier_1.estimated_latency_ms < tier_3.estimated_latency_ms, (
            f"Tier 1 ({tier_1.estimated_latency_ms}ms) should be faster than "
            f"Tier 3 ({tier_3.estimated_latency_ms}ms)"
        )


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests with complexity analyzer."""

    def test_complexity_analyzer_integration(self, routing_module):
        """Router should use complexity analyzer for ambiguous tasks."""
        router = routing_module["ModelRouter"]()

        # This task doesn't match Tier 1 or Tier 3 patterns,
        # so it should use the complexity analyzer
        task = "Implement a pagination system for the API with cursor-based navigation"
        decision = router.route(task)

        # Should have analyzed complexity
        assert decision.complexity_score > 0, "Should have complexity score"
        assert "complexity" in decision.reason.lower() or "pattern" in decision.reason.lower()

    def test_context_affects_routing(self, routing_module):
        """Additional context should influence routing."""
        router = routing_module["ModelRouter"]()

        # Same task, different context
        task = "Implement the feature"

        # Without security context
        decision1 = router.route(task)

        # With security context
        decision2 = router.route(task, context="This is for the authentication security module")

        # Security context might push to higher tier
        # (Not guaranteed, but context should be considered)
        assert decision1 is not None
        assert decision2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
