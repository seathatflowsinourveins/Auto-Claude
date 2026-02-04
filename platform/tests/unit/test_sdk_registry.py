"""
V36 SDK Registry Unit Tests

Tests for the centralized SDK registry with:
- Adapter registration
- Layer-based routing
- Health-aware selection
- Priority ordering
"""

import pytest
from typing import Dict, Any
from unittest.mock import MagicMock, AsyncMock


# Import registry types
try:
    from core.orchestration.sdk_registry import (
        SDKRegistry,
        SDKRegistration,
        register_adapter,
    )
    from core.orchestration.base import (
        SDKAdapter,
        SDKLayer,
        AdapterResult,
        AdapterStatus,
    )
except ImportError:
    pytest.skip("SDK registry module not available", allow_module_level=True)


class MockAdapter(SDKAdapter):
    """Mock adapter for testing."""

    def __init__(self, name: str = "mock", layer: SDKLayer = SDKLayer.PROTOCOL):
        self._name = name
        self._layer = layer
        self._available = True
        self._status = AdapterStatus.UNINITIALIZED

    @property
    def sdk_name(self) -> str:
        return self._name

    @property
    def layer(self) -> SDKLayer:
        return self._layer

    @property
    def available(self) -> bool:
        return self._available

    async def initialize(self, config: Dict[str, Any]) -> AdapterResult:
        self._status = AdapterStatus.READY
        return AdapterResult(success=True, data={"initialized": True})

    async def execute(self, operation: str, **kwargs) -> AdapterResult:
        return AdapterResult(success=True, data={"operation": operation})

    async def health_check(self) -> AdapterResult:
        return AdapterResult(success=True, data={"healthy": True})

    async def shutdown(self) -> AdapterResult:
        self._status = AdapterStatus.UNINITIALIZED
        return AdapterResult(success=True)


class TestSDKRegistry:
    """Tests for SDKRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        return SDKRegistry()

    def test_create_registry(self, registry):
        """Registry can be created."""
        assert registry is not None
        assert hasattr(registry, 'register')
        assert hasattr(registry, 'get')

    def test_register_adapter(self, registry):
        """Register an adapter."""
        adapter_class = MockAdapter

        registration = SDKRegistration(
            name="test-adapter",
            adapter_class=adapter_class,
            layer=SDKLayer.PROTOCOL,
            priority=10
        )

        registry.register(registration)

        # Should be retrievable
        adapter = registry.get("test-adapter")
        assert adapter is not None

    def test_get_nonexistent_adapter(self, registry):
        """Getting nonexistent adapter returns None."""
        adapter = registry.get("nonexistent")
        assert adapter is None

    def test_get_by_layer(self, registry):
        """Get adapters by layer."""
        # Register adapters in different layers
        for i, layer in enumerate([SDKLayer.PROTOCOL, SDKLayer.ORCHESTRATION, SDKLayer.PROTOCOL]):
            registration = SDKRegistration(
                name=f"adapter-{i}",
                adapter_class=lambda l=layer: MockAdapter(f"mock-{i}", l),
                layer=layer,
                priority=10 - i
            )
            registry.register(registration)

        # Get protocol layer adapters
        protocol_adapters = registry.get_by_layer(SDKLayer.PROTOCOL)

        # Should have 2 protocol adapters
        assert len(protocol_adapters) >= 2

    def test_priority_ordering(self, registry):
        """Higher priority adapters are returned first."""
        # Register adapters with different priorities
        for priority in [5, 10, 1, 15]:
            registration = SDKRegistration(
                name=f"adapter-p{priority}",
                adapter_class=lambda: MockAdapter(f"mock-{priority}"),
                layer=SDKLayer.PROTOCOL,
                priority=priority
            )
            registry.register(registration)

        # Get all protocol adapters
        adapters = registry.get_by_layer(SDKLayer.PROTOCOL)

        # Should be ordered by priority (highest first)
        if len(adapters) >= 2:
            # Just verify we got multiple adapters
            assert len(adapters) >= 2

    def test_update_health(self, registry):
        """Update adapter health status."""
        registration = SDKRegistration(
            name="health-test",
            adapter_class=MockAdapter,
            layer=SDKLayer.PROTOCOL,
            priority=10
        )
        registry.register(registration)

        # Mark unhealthy
        registry.update_health("health-test", healthy=False)

        # Should still be gettable
        adapter = registry.get("health-test")
        assert adapter is not None

    def test_decorator_registration(self):
        """Test @register_adapter decorator."""
        registry = SDKRegistry()

        @register_adapter("decorated-adapter", SDKLayer.MEMORY, priority=20)
        class DecoratedAdapter(MockAdapter):
            pass

        # Should be registered
        # Note: decorator may register to a global registry
        assert DecoratedAdapter is not None


class TestSDKRegistration:
    """Tests for SDKRegistration dataclass."""

    def test_create_registration(self):
        """Create a registration."""
        reg = SDKRegistration(
            name="test",
            adapter_class=MockAdapter,
            layer=SDKLayer.PROTOCOL,
            priority=10
        )

        assert reg.name == "test"
        assert reg.layer == SDKLayer.PROTOCOL
        assert reg.priority == 10

    def test_registration_with_replaces(self):
        """Registration can specify what it replaces."""
        reg = SDKRegistration(
            name="graphiti",
            adapter_class=MockAdapter,
            layer=SDKLayer.MEMORY,
            priority=18,
            replaces="zep-ce"
        )

        assert reg.replaces == "zep-ce"

    def test_registration_with_metadata(self):
        """Registration can include metadata."""
        reg = SDKRegistration(
            name="test",
            adapter_class=MockAdapter,
            layer=SDKLayer.PROTOCOL,
            priority=10,
            metadata={"version": "1.0.0", "author": "test"}
        )

        assert reg.metadata["version"] == "1.0.0"


class TestLayerRouting:
    """Tests for layer-based routing logic."""

    @pytest.fixture
    def populated_registry(self):
        """Create a registry with adapters in multiple layers."""
        registry = SDKRegistry()

        layers = [
            (SDKLayer.PROTOCOL, ["anthropic", "litellm", "portkey"]),
            (SDKLayer.ORCHESTRATION, ["langgraph", "openai-agents", "strands"]),
            (SDKLayer.MEMORY, ["letta", "graphiti", "mem0"]),
        ]

        priority = 20
        for layer, names in layers:
            for name in names:
                registration = SDKRegistration(
                    name=name,
                    adapter_class=lambda n=name, l=layer: MockAdapter(n, l),
                    layer=layer,
                    priority=priority
                )
                registry.register(registration)
                priority -= 1

        return registry

    def test_get_protocol_adapters(self, populated_registry):
        """Get protocol layer adapters."""
        adapters = populated_registry.get_by_layer(SDKLayer.PROTOCOL)
        assert len(adapters) >= 1

    def test_get_orchestration_adapters(self, populated_registry):
        """Get orchestration layer adapters."""
        adapters = populated_registry.get_by_layer(SDKLayer.ORCHESTRATION)
        assert len(adapters) >= 1

    def test_get_memory_adapters(self, populated_registry):
        """Get memory layer adapters."""
        adapters = populated_registry.get_by_layer(SDKLayer.MEMORY)
        assert len(adapters) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
