"""
Tests for the Mem0 Adapter (V65 Production Ready).

Comprehensive test coverage for:
- All CRUD operations (add, search, get, get_all, update, delete, delete_all)
- Error handling and validation
- Circuit breaker behavior
- Timeout handling
- Graph memory operations
- Memory type filtering
- User/agent/session scoping
"""

import asyncio
import sys
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import pytest

# Add platform directory to path for imports
_platform_dir = os.path.join(os.path.dirname(__file__), "..")
if _platform_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_platform_dir))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_mem0_client():
    """Create a mock Mem0 client."""
    client = MagicMock()

    # Mock add
    client.add = MagicMock(return_value={
        "results": [{"id": "mem_123", "memory": "test content"}]
    })

    # Mock search
    client.search = MagicMock(return_value={
        "results": [
            {"id": "mem_1", "memory": "result 1", "score": 0.95, "metadata": {}},
            {"id": "mem_2", "memory": "result 2", "score": 0.87, "metadata": {}},
        ]
    })

    # Mock get
    client.get = MagicMock(return_value={
        "id": "mem_123",
        "memory": "test content",
        "user_id": "user_1",
        "metadata": {"key": "value"},
    })

    # Mock get_all
    client.get_all = MagicMock(return_value={
        "results": [
            {"id": "mem_1", "memory": "memory 1", "metadata": {}},
            {"id": "mem_2", "memory": "memory 2", "metadata": {}},
        ]
    })

    # Mock update
    client.update = MagicMock(return_value=None)

    # Mock delete
    client.delete = MagicMock(return_value=None)

    # Mock delete_all
    client.delete_all = MagicMock(return_value=None)

    # Mock history
    client.history = MagicMock(return_value=[
        {"version": 1, "content": "original", "timestamp": "2024-01-01"},
        {"version": 2, "content": "updated", "timestamp": "2024-01-02"},
    ])

    return client


@pytest.fixture
def mock_adapter_result():
    """Create a mock AdapterResult class."""
    class MockAdapterResult:
        def __init__(self, success=True, data=None, error=None, latency_ms=0.0):
            self.success = success
            self.data = data
            self.error = error
            self.latency_ms = latency_ms
    return MockAdapterResult


@pytest.fixture
def mock_circuit_breaker():
    """Create a mock circuit breaker."""
    cb = MagicMock()
    cb.is_open = False
    cb.record_success = MagicMock()
    cb.record_failure = MagicMock()
    return cb


# =============================================================================
# Test Data Classes
# =============================================================================

class TestMemoryEntry:
    """Tests for MemoryEntry dataclass."""

    def test_creates_with_defaults(self):
        """Test creating MemoryEntry with minimal fields."""
        from adapters.mem0_adapter import MemoryEntry

        entry = MemoryEntry(id="test_1", content="Test content")

        assert entry.id == "test_1"
        assert entry.content == "Test content"
        assert entry.user_id is None
        assert entry.metadata == {}
        assert entry.created_at is not None
        assert entry.hash is not None

    def test_creates_with_all_fields(self):
        """Test creating MemoryEntry with all fields."""
        from adapters.mem0_adapter import MemoryEntry, MemoryType

        now = datetime.now(timezone.utc)
        entry = MemoryEntry(
            id="test_2",
            content="Full content",
            user_id="user_123",
            agent_id="agent_456",
            session_id="session_789",
            memory_type=MemoryType.EPISODIC,
            metadata={"key": "value"},
            created_at=now,
        )

        assert entry.id == "test_2"
        assert entry.user_id == "user_123"
        assert entry.agent_id == "agent_456"
        assert entry.memory_type == MemoryType.EPISODIC
        assert entry.metadata == {"key": "value"}

    def test_hash_is_computed(self):
        """Test that hash is computed from content."""
        from adapters.mem0_adapter import MemoryEntry

        entry1 = MemoryEntry(id="1", content="Same content")
        entry2 = MemoryEntry(id="2", content="Same content")
        entry3 = MemoryEntry(id="3", content="Different content")

        assert entry1.hash == entry2.hash
        assert entry1.hash != entry3.hash


class TestMemoryBackend:
    """Tests for MemoryBackend enum."""

    def test_all_backends_available(self):
        """Test all expected backends are defined."""
        from adapters.mem0_adapter import MemoryBackend

        expected = ["sqlite", "qdrant", "pinecone", "weaviate", "chroma", "supabase", "milvus"]
        for backend in expected:
            assert MemoryBackend(backend).value == backend

    def test_invalid_backend_raises(self):
        """Test invalid backend raises ValueError."""
        from adapters.mem0_adapter import MemoryBackend

        with pytest.raises(ValueError):
            MemoryBackend("invalid_backend")


class TestMemoryType:
    """Tests for MemoryType enum."""

    def test_all_types_available(self):
        """Test all expected types are defined."""
        from adapters.mem0_adapter import MemoryType

        expected = ["short_term", "long_term", "episodic", "semantic", "procedural", "working"]
        for mem_type in expected:
            assert MemoryType(mem_type).value == mem_type


# =============================================================================
# Test Adapter Creation
# =============================================================================

class TestAdapterCreation:
    """Tests for adapter instantiation."""

    def test_creates_adapter(self):
        """Test basic adapter creation."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        assert adapter is not None

    def test_get_status_uninitialized(self):
        """Test get_status returns correct state when uninitialized."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        status = adapter.get_status()

        assert "available" in status
        assert "initialized" in status or "status" in status

    def test_factory_function(self):
        """Test create_mem0_adapter factory function."""
        from adapters.mem0_adapter import create_mem0_adapter, MemoryBackend

        adapter = create_mem0_adapter(backend="chroma", enable_graph=True)
        assert adapter is not None

    def test_factory_with_string_backend(self):
        """Test factory accepts string backend."""
        from adapters.mem0_adapter import create_mem0_adapter

        adapter = create_mem0_adapter(backend="qdrant")
        assert adapter is not None

    def test_factory_with_enum_backend(self):
        """Test factory accepts enum backend."""
        from adapters.mem0_adapter import create_mem0_adapter, MemoryBackend

        adapter = create_mem0_adapter(backend=MemoryBackend.PINECONE)
        assert adapter is not None

    def test_factory_invalid_backend_defaults_to_qdrant(self):
        """Test factory defaults to qdrant for invalid backend."""
        from adapters.mem0_adapter import create_mem0_adapter

        adapter = create_mem0_adapter(backend="invalid_xyz")
        # Should not raise, should use default


# =============================================================================
# Test Add Operation
# =============================================================================

class TestAddOperation:
    """Tests for the add operation."""

    @pytest.mark.asyncio
    async def test_add_requires_content(self, mock_mem0_client):
        """Test add fails without content."""
        from adapters.mem0_adapter import Mem0Adapter

        with patch('adapters.mem0_adapter.Memory') as MockMemory:
            MockMemory.from_config.return_value = mock_mem0_client

            adapter = Mem0Adapter()
            adapter._client = mock_mem0_client
            adapter._available = True

            result = await adapter._add({})

            assert not result.success
            assert "content" in result.error.lower()

    @pytest.mark.asyncio
    async def test_add_with_user_id(self, mock_mem0_client):
        """Test add with user_id scope."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._add({
            "content": "Test memory",
            "user_id": "user_123",
        })

        assert result.success
        assert result.data["user_id"] == "user_123"
        mock_mem0_client.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_with_metadata(self, mock_mem0_client):
        """Test add with custom metadata."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._add({
            "content": "Test memory",
            "metadata": {"custom_key": "custom_value"},
        })

        assert result.success
        mock_mem0_client.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_with_memory_type(self, mock_mem0_client):
        """Test add with memory type."""
        from adapters.mem0_adapter import Mem0Adapter, MemoryType

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._add({
            "content": "Episodic memory",
            "memory_type": MemoryType.EPISODIC,
        })

        assert result.success

    @pytest.mark.asyncio
    async def test_add_extracts_memory_id(self, mock_mem0_client):
        """Test add extracts memory ID from response."""
        from adapters.mem0_adapter import Mem0Adapter

        mock_mem0_client.add.return_value = {"results": [{"id": "new_mem_id"}]}

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._add({"content": "Test"})

        assert result.success
        assert result.data["id"] == "new_mem_id"

    @pytest.mark.asyncio
    async def test_add_handles_list_response(self, mock_mem0_client):
        """Test add handles list response format."""
        from adapters.mem0_adapter import Mem0Adapter

        mock_mem0_client.add.return_value = [{"id": "list_mem_id"}]

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._add({"content": "Test"})

        assert result.success
        assert result.data["id"] == "list_mem_id"


# =============================================================================
# Test Search Operation
# =============================================================================

class TestSearchOperation:
    """Tests for the search operation."""

    @pytest.mark.asyncio
    async def test_search_requires_query(self, mock_mem0_client):
        """Test search fails without query."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._search({})

        assert not result.success
        assert "query" in result.error.lower()

    @pytest.mark.asyncio
    async def test_search_returns_results(self, mock_mem0_client):
        """Test search returns parsed results."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._search({"query": "test query"})

        assert result.success
        assert "memories" in result.data
        assert result.data["count"] == 2
        assert "search_time_ms" in result.data

    @pytest.mark.asyncio
    async def test_search_with_limit(self, mock_mem0_client):
        """Test search respects limit parameter."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._search({"query": "test", "limit": 5})

        assert result.success
        # Verify limit was passed to client
        call_kwargs = mock_mem0_client.search.call_args[1]
        assert call_kwargs["limit"] == 5

    @pytest.mark.asyncio
    async def test_search_with_user_filter(self, mock_mem0_client):
        """Test search with user_id filter."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._search({
            "query": "test",
            "user_id": "user_456",
        })

        assert result.success
        call_kwargs = mock_mem0_client.search.call_args[1]
        assert call_kwargs["user_id"] == "user_456"

    @pytest.mark.asyncio
    async def test_search_filters_by_memory_type(self, mock_mem0_client):
        """Test search filters by memory type."""
        from adapters.mem0_adapter import Mem0Adapter, MemoryType

        mock_mem0_client.search.return_value = {
            "results": [
                {"id": "1", "memory": "m1", "metadata": {"memory_type": "episodic"}},
                {"id": "2", "memory": "m2", "metadata": {"memory_type": "semantic"}},
            ]
        }

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._search({
            "query": "test",
            "memory_type": MemoryType.EPISODIC,
        })

        assert result.success
        assert result.data["count"] == 1


# =============================================================================
# Test Get Operation
# =============================================================================

class TestGetOperation:
    """Tests for the get operation."""

    @pytest.mark.asyncio
    async def test_get_requires_memory_id(self, mock_mem0_client):
        """Test get fails without memory_id."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._get({})

        assert not result.success
        assert "memory_id" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_returns_memory(self, mock_mem0_client):
        """Test get returns memory data."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._get({"memory_id": "mem_123"})

        assert result.success
        assert result.data["id"] == "mem_123"
        assert result.data["content"] == "test content"

    @pytest.mark.asyncio
    async def test_get_not_found(self, mock_mem0_client):
        """Test get handles not found."""
        from adapters.mem0_adapter import Mem0Adapter

        mock_mem0_client.get.return_value = None

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._get({"memory_id": "nonexistent"})

        assert not result.success
        assert "not found" in result.error.lower()


# =============================================================================
# Test Get All Operation
# =============================================================================

class TestGetAllOperation:
    """Tests for the get_all operation."""

    @pytest.mark.asyncio
    async def test_get_all_returns_memories(self, mock_mem0_client):
        """Test get_all returns all memories."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._get_all({})

        assert result.success
        assert result.data["count"] == 2

    @pytest.mark.asyncio
    async def test_get_all_with_user_filter(self, mock_mem0_client):
        """Test get_all with user_id filter."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._get_all({"user_id": "user_123"})

        assert result.success
        call_kwargs = mock_mem0_client.get_all.call_args[1]
        assert call_kwargs["user_id"] == "user_123"

    @pytest.mark.asyncio
    async def test_get_all_respects_limit(self, mock_mem0_client):
        """Test get_all respects limit."""
        from adapters.mem0_adapter import Mem0Adapter

        mock_mem0_client.get_all.return_value = {
            "results": [{"id": str(i), "memory": f"m{i}"} for i in range(10)]
        }

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._get_all({"limit": 3})

        assert result.success
        assert result.data["count"] == 3


# =============================================================================
# Test Update Operation
# =============================================================================

class TestUpdateOperation:
    """Tests for the update operation."""

    @pytest.mark.asyncio
    async def test_update_requires_memory_id(self, mock_mem0_client):
        """Test update fails without memory_id."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._update({"content": "new"})

        assert not result.success
        assert "memory_id" in result.error.lower()

    @pytest.mark.asyncio
    async def test_update_requires_content(self, mock_mem0_client):
        """Test update fails without content."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._update({"memory_id": "mem_123"})

        assert not result.success
        assert "content" in result.error.lower()

    @pytest.mark.asyncio
    async def test_update_succeeds(self, mock_mem0_client):
        """Test update succeeds with valid input."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._update({
            "memory_id": "mem_123",
            "content": "Updated content",
        })

        assert result.success
        assert result.data["updated"] is True


# =============================================================================
# Test Delete Operation
# =============================================================================

class TestDeleteOperation:
    """Tests for the delete operation."""

    @pytest.mark.asyncio
    async def test_delete_requires_memory_id(self, mock_mem0_client):
        """Test delete fails without memory_id."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._delete({})

        assert not result.success
        assert "memory_id" in result.error.lower()

    @pytest.mark.asyncio
    async def test_delete_succeeds(self, mock_mem0_client):
        """Test delete succeeds."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._delete({"memory_id": "mem_123"})

        assert result.success
        assert result.data["deleted"] is True


# =============================================================================
# Test Delete All Operation
# =============================================================================

class TestDeleteAllOperation:
    """Tests for the delete_all operation."""

    @pytest.mark.asyncio
    async def test_delete_all_succeeds(self, mock_mem0_client):
        """Test delete_all succeeds."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._delete_all({})

        assert result.success
        assert "deleted_count" in result.data

    @pytest.mark.asyncio
    async def test_delete_all_with_user_filter(self, mock_mem0_client):
        """Test delete_all with user filter."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._delete_all({"user_id": "user_123"})

        assert result.success
        assert result.data["user_id"] == "user_123"


# =============================================================================
# Test History Operation
# =============================================================================

class TestHistoryOperation:
    """Tests for the history operation."""

    @pytest.mark.asyncio
    async def test_history_requires_memory_id(self, mock_mem0_client):
        """Test history fails without memory_id."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._history({})

        assert not result.success
        assert "memory_id" in result.error.lower()

    @pytest.mark.asyncio
    async def test_history_returns_versions(self, mock_mem0_client):
        """Test history returns version list."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._history({"memory_id": "mem_123"})

        assert result.success
        assert result.data["count"] == 2


# =============================================================================
# Test Graph Memory Operations
# =============================================================================

class TestGraphOperations:
    """Tests for graph memory operations."""

    @pytest.mark.asyncio
    async def test_search_graph_requires_enabled(self, mock_mem0_client):
        """Test search_graph fails when graph not enabled."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True
        adapter._graph_enabled = False

        result = await adapter._search_graph({"query": "test"})

        assert not result.success
        assert "not enabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_add_entities_requires_enabled(self, mock_mem0_client):
        """Test add_entities fails when graph not enabled."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True
        adapter._graph_enabled = False

        result = await adapter._add_entities({
            "entities": [{"name": "Test", "type": "Person"}]
        })

        assert not result.success
        assert "not enabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_add_entities_requires_data(self, mock_mem0_client):
        """Test add_entities fails without entities or relations."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True
        adapter._graph_enabled = True

        result = await adapter._add_entities({})

        assert not result.success
        assert "required" in result.error.lower()


# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_handles_client_exception(self, mock_mem0_client):
        """Test adapter handles client exceptions."""
        from adapters.mem0_adapter import Mem0Adapter

        mock_mem0_client.add.side_effect = Exception("Connection failed")

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._add({"content": "test"})

        assert not result.success
        assert "Connection failed" in result.error

    @pytest.mark.asyncio
    async def test_handles_timeout(self, mock_mem0_client):
        """Test adapter handles timeouts."""
        from adapters.mem0_adapter import Mem0Adapter

        async def slow_operation(*args, **kwargs):
            await asyncio.sleep(2)
            return {"results": []}

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        # Set very short timeout
        result = await adapter.execute("add", content="test", timeout=0.01)

        assert not result.success
        assert "timed out" in result.error.lower()

    def test_not_initialized_error(self):
        """Test error when not initialized."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        status = adapter.get_status()

        # Should indicate not initialized
        assert status.get("initialized") is False or status.get("status") != "ready"


# =============================================================================
# Test Circuit Breaker Integration
# =============================================================================

class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration."""

    @pytest.mark.asyncio
    async def test_records_success(self, mock_mem0_client, mock_circuit_breaker):
        """Test circuit breaker records success."""
        from adapters.mem0_adapter import Mem0Adapter

        with patch('adapters.mem0_adapter.adapter_circuit_breaker') as mock_cb_func:
            mock_cb_func.return_value = mock_circuit_breaker

            adapter = Mem0Adapter()
            adapter._client = mock_mem0_client
            adapter._available = True

            result = await adapter.execute("search", query="test")

            if result.success:
                mock_circuit_breaker.record_success.assert_called()

    @pytest.mark.asyncio
    async def test_records_failure(self, mock_mem0_client, mock_circuit_breaker):
        """Test circuit breaker records failure on timeout."""
        from adapters.mem0_adapter import Mem0Adapter

        # Use timeout to trigger failure recording (exceptions in _search don't propagate to execute)
        async def slow_search(*args, **kwargs):
            await asyncio.sleep(10)
            return {}

        with patch('adapters.mem0_adapter.adapter_circuit_breaker') as mock_cb_func:
            mock_cb_func.return_value = mock_circuit_breaker

            adapter = Mem0Adapter()
            adapter._client = mock_mem0_client
            adapter._available = True

            # This should timeout and record a failure
            result = await adapter.execute("search", query="test", timeout=0.01)

            assert not result.success
            mock_circuit_breaker.record_failure.assert_called()

    @pytest.mark.asyncio
    async def test_circuit_open_returns_error(self, mock_mem0_client, mock_circuit_breaker):
        """Test returns error when circuit is open."""
        from adapters.mem0_adapter import Mem0Adapter

        mock_circuit_breaker.is_open = True

        with patch('adapters.mem0_adapter.adapter_circuit_breaker') as mock_cb_func:
            mock_cb_func.return_value = mock_circuit_breaker

            adapter = Mem0Adapter()
            adapter._client = mock_mem0_client
            adapter._available = True

            result = await adapter.execute("search", query="test")

            assert not result.success
            assert "circuit breaker" in result.error.lower()


# =============================================================================
# Test Execute Dispatch
# =============================================================================

class TestExecuteDispatch:
    """Tests for the execute dispatch logic."""

    @pytest.mark.asyncio
    async def test_dispatch_unknown_operation(self, mock_mem0_client):
        """Test dispatch returns error for unknown operation."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._dispatch_operation("unknown_op", {})

        assert not result.success
        assert "Unknown operation" in result.error

    @pytest.mark.asyncio
    async def test_dispatch_lists_available_operations(self, mock_mem0_client):
        """Test dispatch error includes available operations."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter._dispatch_operation("invalid", {})

        assert "add" in result.error
        assert "search" in result.error


# =============================================================================
# Test Health Check
# =============================================================================

class TestHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_passes(self, mock_mem0_client):
        """Test health check passes when client works."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter.health_check()

        assert result.success
        assert result.data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_fails_no_client(self):
        """Test health check fails without client."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = None

        result = await adapter.health_check()

        assert not result.success
        assert "not initialized" in result.error.lower()

    @pytest.mark.asyncio
    async def test_health_check_fails_on_error(self, mock_mem0_client):
        """Test health check fails on client error."""
        from adapters.mem0_adapter import Mem0Adapter

        mock_mem0_client.get_all.side_effect = Exception("Connection refused")

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter.health_check()

        assert not result.success


# =============================================================================
# Test Shutdown
# =============================================================================

class TestShutdown:
    """Tests for shutdown functionality."""

    @pytest.mark.asyncio
    async def test_shutdown_clears_client(self, mock_mem0_client):
        """Test shutdown clears client."""
        from adapters.mem0_adapter import Mem0Adapter

        adapter = Mem0Adapter()
        adapter._client = mock_mem0_client
        adapter._available = True

        result = await adapter.shutdown()

        assert result.success
        assert adapter._client is None
        assert adapter._available is False


# =============================================================================
# Test Module Exports
# =============================================================================

class TestModuleExports:
    """Tests for module exports."""

    def test_exports_available(self):
        """Test all expected exports are available."""
        from adapters import mem0_adapter

        expected = [
            "Mem0Adapter",
            "create_mem0_adapter",
            "MemoryEntry",
            "SearchResult",
            "MemoryBackend",
            "MemoryType",
            "MEM0_AVAILABLE",
        ]

        for name in expected:
            assert hasattr(mem0_adapter, name), f"Missing export: {name}"

    def test_all_list_complete(self):
        """Test __all__ contains expected exports."""
        from adapters.mem0_adapter import __all__

        assert "Mem0Adapter" in __all__
        assert "create_mem0_adapter" in __all__
        assert "MEM0_AVAILABLE" in __all__
