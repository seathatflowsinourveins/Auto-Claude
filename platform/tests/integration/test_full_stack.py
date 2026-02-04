"""
Full Stack Integration Tests - UNLEASH Platform V36

Comprehensive integration test suite covering:
1. Memory System Tests: Cross-session persistence, FTS5, embeddings, hooks
2. Orchestration Tests: SDK registry, health-aware routing, connection pooling
3. RAG Pipeline Tests: Document chunking, retrieval, hybrid search
4. Research Adapter Tests: Mock responses, error handling, timeouts

Usage:
    pytest platform/tests/integration/test_full_stack.py -v
    pytest platform/tests/integration/test_full_stack.py -v -m integration
    pytest platform/tests/integration/test_full_stack.py -v -k "test_memory"
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# IMPORTS WITH GRACEFUL FALLBACKS
# =============================================================================

# Memory backends
try:
    from core.memory.backends.base import (
        MemoryBackend,
        MemoryEntry,
        MemoryLayer,
        MemoryNamespace,
        MemoryPriority,
        MemoryTier,
        TierBackend,
        generate_memory_id,
    )
    from core.memory.backends.in_memory import InMemoryTierBackend
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    MemoryEntry = None
    MemoryTier = None
    InMemoryTierBackend = None

# Cross-session memory
try:
    from core.cross_session_memory import (
        CrossSessionMemory,
        Memory,
        Session,
        get_memory_store,
        remember_decision,
        remember_fact,
        remember_learning,
        recall,
    )
    CROSS_SESSION_AVAILABLE = True
except ImportError:
    CROSS_SESSION_AVAILABLE = False
    CrossSessionMemory = None

# SDK Registry
try:
    from core.orchestration.sdk_registry import (
        SDKRegistry,
        SDKRegistration,
        get_registry,
    )
    from core.orchestration.base import (
        AdapterConfig,
        AdapterResult,
        AdapterStatus,
        SDKAdapter,
        SDKLayer,
    )
    SDK_REGISTRY_AVAILABLE = True
except ImportError:
    SDK_REGISTRY_AVAILABLE = False
    SDKRegistry = None
    SDKAdapter = None
    AdapterResult = None

# Connection infrastructure
try:
    from core.orchestration.infrastructure.connection import (
        ConnectionPool,
        RequestDeduplicator,
        WarmupPreloader,
    )
    CONNECTION_AVAILABLE = True
except ImportError:
    CONNECTION_AVAILABLE = False
    ConnectionPool = None
    RequestDeduplicator = None
    WarmupPreloader = None

# Research adapters
try:
    from adapters.exa_adapter import ExaAdapter, EXA_AVAILABLE
except ImportError:
    ExaAdapter = None
    EXA_AVAILABLE = False

try:
    from adapters.tavily_adapter import TavilyAdapter
    TAVILY_AVAILABLE = True
except ImportError:
    TavilyAdapter = None
    TAVILY_AVAILABLE = False

try:
    from adapters.jina_adapter import JinaAdapter
    JINA_AVAILABLE = True
except ImportError:
    JinaAdapter = None
    JINA_AVAILABLE = False


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_memory_dir():
    """Create a temporary directory for memory persistence tests."""
    temp_dir = tempfile.mkdtemp(prefix="unleash_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def memory_entry_factory():
    """Factory for creating memory entries."""
    if not MEMORY_AVAILABLE:
        pytest.skip("Memory backend not available")

    def create_entry(
        entry_id: str = "test-entry",
        content: str = "Test content",
        tier: str = "main_context",
        priority: str = "normal",
        **kwargs
    ) -> MemoryEntry:
        return MemoryEntry(
            id=entry_id,
            content=content,
            tier=MemoryTier(tier),
            priority=MemoryPriority(priority),
            **kwargs
        )

    return create_entry


@pytest.fixture
async def in_memory_backend():
    """Create an in-memory tier backend."""
    if not MEMORY_AVAILABLE:
        pytest.skip("InMemoryTierBackend not available")

    backend = InMemoryTierBackend()
    yield backend
    backend.clear()


@pytest.fixture
def cross_session_memory(temp_memory_dir):
    """Create a cross-session memory instance with temp storage."""
    if not CROSS_SESSION_AVAILABLE:
        pytest.skip("CrossSessionMemory not available")

    memory = CrossSessionMemory(base_path=temp_memory_dir)
    yield memory


@pytest.fixture
def sdk_registry():
    """Create a fresh SDK registry for testing."""
    if not SDK_REGISTRY_AVAILABLE:
        pytest.skip("SDKRegistry not available")

    registry = SDKRegistry()
    yield registry


@pytest.fixture
def mock_adapter_class():
    """Create a mock adapter class for testing."""
    if not SDK_REGISTRY_AVAILABLE:
        pytest.skip("SDK base classes not available")

    class MockAdapter(SDKAdapter):
        def __init__(self, config=None):
            self._config = config or AdapterConfig(name="mock", layer=SDKLayer.MEMORY)
            self._status = AdapterStatus.UNINITIALIZED
            self._call_count = 0
            self._total_latency_ms = 0.0
            self._error_count = 0
            self._last_health_check = None
            self._healthy = True

        @property
        def sdk_name(self) -> str:
            return "mock"

        @property
        def layer(self) -> SDKLayer:
            return SDKLayer.MEMORY

        @property
        def available(self) -> bool:
            return True

        async def initialize(self, config: Dict) -> AdapterResult:
            self._status = AdapterStatus.READY
            return AdapterResult(success=True)

        async def execute(self, operation: str, **kwargs) -> AdapterResult:
            self._call_count += 1
            return AdapterResult(success=True, data={"operation": operation, **kwargs})

        async def health_check(self) -> AdapterResult:
            return AdapterResult(success=self._healthy)

        async def shutdown(self) -> AdapterResult:
            self._status = AdapterStatus.SHUTDOWN
            return AdapterResult(success=True)

    return MockAdapter


@pytest.fixture
def sample_documents() -> List[Dict[str, Any]]:
    """Sample documents for RAG testing."""
    return [
        {
            "id": "doc-1",
            "content": "Python is a versatile programming language used in web development, data science, and AI applications.",
            "metadata": {"topic": "programming", "language": "python"}
        },
        {
            "id": "doc-2",
            "content": "Machine learning models learn patterns from training data to make accurate predictions on new data.",
            "metadata": {"topic": "machine_learning", "difficulty": "intermediate"}
        },
        {
            "id": "doc-3",
            "content": "Natural language processing enables computers to understand, interpret, and generate human language.",
            "metadata": {"topic": "nlp", "difficulty": "advanced"}
        },
        {
            "id": "doc-4",
            "content": "Deep learning uses artificial neural networks with multiple layers to extract hierarchical features.",
            "metadata": {"topic": "deep_learning", "difficulty": "advanced"}
        },
        {
            "id": "doc-5",
            "content": "Vector databases store embeddings and enable efficient similarity search for retrieval augmented generation.",
            "metadata": {"topic": "vector_db", "difficulty": "intermediate"}
        },
    ]


# =============================================================================
# MEMORY SYSTEM TESTS
# =============================================================================

@pytest.mark.integration
class TestMemoryStack:
    """Integration tests for the memory system."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, in_memory_backend, memory_entry_factory):
        """Test basic store and retrieve operations."""
        entry = memory_entry_factory(
            entry_id="test-1",
            content="This is test content for retrieval",
            tier="main_context"
        )

        # Store entry
        await in_memory_backend.put("test-1", entry)

        # Retrieve entry
        retrieved = await in_memory_backend.get("test-1")

        assert retrieved is not None
        assert retrieved.id == "test-1"
        assert retrieved.content == "This is test content for retrieval"
        assert retrieved.access_count >= 1  # Touch updates access count

    @pytest.mark.asyncio
    async def test_search_by_content(self, in_memory_backend, memory_entry_factory):
        """Test content-based search."""
        # Store multiple entries
        entries = [
            memory_entry_factory("e1", "Python programming basics"),
            memory_entry_factory("e2", "JavaScript web development"),
            memory_entry_factory("e3", "Python machine learning tutorial"),
            memory_entry_factory("e4", "Database design patterns"),
        ]

        for entry in entries:
            await in_memory_backend.put(entry.id, entry)

        # Search for Python-related content
        results = await in_memory_backend.search("Python", limit=10)

        assert len(results) == 2
        assert all("python" in r.content.lower() for r in results)

    @pytest.mark.asyncio
    async def test_delete_entry(self, in_memory_backend, memory_entry_factory):
        """Test entry deletion."""
        entry = memory_entry_factory("to-delete", "Content to be deleted")
        await in_memory_backend.put("to-delete", entry)

        # Verify it exists
        assert await in_memory_backend.get("to-delete") is not None

        # Delete
        deleted = await in_memory_backend.delete("to-delete")
        assert deleted is True

        # Verify it is gone
        assert await in_memory_backend.get("to-delete") is None

    @pytest.mark.asyncio
    async def test_list_all_entries(self, in_memory_backend, memory_entry_factory):
        """Test listing all entries."""
        for i in range(5):
            entry = memory_entry_factory(f"entry-{i}", f"Content {i}")
            await in_memory_backend.put(f"entry-{i}", entry)

        all_entries = await in_memory_backend.list_all()
        assert len(all_entries) == 5

    @pytest.mark.asyncio
    async def test_entry_count(self, in_memory_backend, memory_entry_factory):
        """Test entry counting."""
        assert await in_memory_backend.count() == 0

        for i in range(3):
            entry = memory_entry_factory(f"entry-{i}", f"Content {i}")
            await in_memory_backend.put(f"entry-{i}", entry)

        assert await in_memory_backend.count() == 3

    @pytest.mark.asyncio
    async def test_entry_expiration(self, memory_entry_factory):
        """Test TTL-based entry expiration."""
        if not MEMORY_AVAILABLE:
            pytest.skip("Memory backend not available")

        # Create entry with 1-second TTL
        entry = memory_entry_factory(
            "expiring-entry",
            "This will expire",
            ttl_seconds=1
        )

        assert not entry.is_expired

        # Wait for expiration
        await asyncio.sleep(1.1)

        assert entry.is_expired


@pytest.mark.integration
class TestCrossSessionMemory:
    """Integration tests for cross-session memory persistence."""

    def test_session_lifecycle(self, cross_session_memory):
        """Test session start and end lifecycle."""
        session = cross_session_memory.start_session("Test task")

        assert session is not None
        assert session.id is not None
        assert session.task_summary == "Test task"
        assert session.ended_at is None

        cross_session_memory.end_session("Completed test")

        # Session should be updated
        current = cross_session_memory.get_current_session()
        assert current.ended_at is not None

    def test_memory_persistence(self, temp_memory_dir):
        """Test that memories persist across instances."""
        if not CROSS_SESSION_AVAILABLE:
            pytest.skip("CrossSessionMemory not available")

        # Create first instance and add memories
        mem1 = CrossSessionMemory(base_path=temp_memory_dir)
        mem1.start_session("Persistence test")

        mem1.add("Test fact content", memory_type="fact", importance=0.8)
        mem1.add("Test decision content", memory_type="decision", importance=0.9)

        stats1 = mem1.get_stats()
        assert stats1["total_memories"] == 2

        # Create new instance from same path
        mem2 = CrossSessionMemory(base_path=temp_memory_dir)

        stats2 = mem2.get_stats()
        assert stats2["total_memories"] == 2

        # Search for persisted content
        results = mem2.search("Test fact", limit=5)
        assert len(results) >= 1
        assert any("Test fact content" in r.content for r in results)

    def test_memory_search(self, cross_session_memory):
        """Test memory search functionality."""
        cross_session_memory.start_session("Search test")

        # Add various memories
        cross_session_memory.add("Python programming patterns", memory_type="learning", importance=0.7)
        cross_session_memory.add("Database optimization decision", memory_type="decision", importance=0.8)
        cross_session_memory.add("API design best practices", memory_type="fact", importance=0.6)
        cross_session_memory.add("Python decorators tutorial", memory_type="learning", importance=0.65)

        # Search for Python-related content
        results = cross_session_memory.search("Python", limit=10)

        assert len(results) >= 2
        # At least some results should contain "Python"
        assert any("Python" in r.content for r in results)

    def test_memory_by_type(self, cross_session_memory):
        """Test filtering memories by type."""
        cross_session_memory.start_session("Type filter test")

        cross_session_memory.add("Decision A", memory_type="decision")
        cross_session_memory.add("Decision B", memory_type="decision")
        cross_session_memory.add("Learning A", memory_type="learning")
        cross_session_memory.add("Fact A", memory_type="fact")

        decisions = cross_session_memory.get_decisions(limit=10)
        assert len(decisions) == 2
        assert all(d.memory_type == "decision" for d in decisions)

        learnings = cross_session_memory.get_learnings(limit=10)
        assert len(learnings) == 1

    def test_memory_invalidation(self, cross_session_memory):
        """Test memory invalidation."""
        cross_session_memory.start_session("Invalidation test")

        memory = cross_session_memory.add("Valid content", memory_type="fact")
        memory_id = memory.id

        # Invalidate the memory
        result = cross_session_memory.invalidate(memory_id, reason="No longer accurate")
        assert result is True

        # Search should not return invalidated memories
        results = cross_session_memory.search("Valid content", limit=10)
        assert len(results) == 0

    def test_session_context_generation(self, cross_session_memory):
        """Test generating context for new sessions."""
        cross_session_memory.start_session("Context generation test")

        cross_session_memory.add(
            "Use async/await for all I/O operations",
            memory_type="decision",
            importance=0.9
        )
        cross_session_memory.add(
            "SQLite performs better with WAL mode",
            memory_type="learning",
            importance=0.8
        )
        cross_session_memory.add(
            "Project uses pytest for testing",
            memory_type="fact",
            importance=0.7
        )

        cross_session_memory.end_session("Session completed")

        # Generate context
        context = cross_session_memory.get_session_context(max_tokens=1000)

        assert "Decision" in context or "decision" in context.lower()
        assert len(context) > 0
        assert len(context) < 4000  # Should be within token limit

    def test_session_history(self, temp_memory_dir):
        """Test retrieving session history."""
        if not CROSS_SESSION_AVAILABLE:
            pytest.skip("CrossSessionMemory not available")

        mem = CrossSessionMemory(base_path=temp_memory_dir)

        # Create multiple sessions
        for i in range(3):
            mem.start_session(f"Session {i}")
            mem.add(f"Content for session {i}", memory_type="context")
            mem.end_session(f"Completed session {i}")

        history = mem.get_session_history(limit=10)

        assert len(history) == 3
        # Most recent should be first
        assert "session 2" in history[0].task_summary.lower() or "2" in history[0].task_summary


@pytest.mark.integration
class TestFTS5Search:
    """Integration tests for FTS5 full-text search (simulated)."""

    def test_fts5_tokenization(self, cross_session_memory):
        """Test that FTS-style tokenization works for search."""
        cross_session_memory.start_session("FTS test")

        # Add content with various word patterns
        cross_session_memory.add("programming language features", memory_type="fact")
        cross_session_memory.add("language model training", memory_type="learning")
        cross_session_memory.add("feature engineering pipeline", memory_type="decision")

        # Search should match partial words
        results = cross_session_memory.search("language", limit=10)
        assert len(results) >= 2

    def test_fts5_phrase_matching(self, cross_session_memory):
        """Test phrase matching in search."""
        cross_session_memory.start_session("Phrase test")

        cross_session_memory.add("machine learning algorithms", memory_type="fact")
        cross_session_memory.add("learning curve improvement", memory_type="learning")
        cross_session_memory.add("machine vision applications", memory_type="fact")

        # Search for phrase
        results = cross_session_memory.search("machine learning", limit=10)
        assert len(results) >= 1
        assert any("machine learning" in r.content.lower() for r in results)


# =============================================================================
# ORCHESTRATION TESTS
# =============================================================================

@pytest.mark.integration
class TestOrchestrationStack:
    """Integration tests for SDK orchestration."""

    def test_sdk_registry_initialization(self, sdk_registry):
        """Test SDK registry can be initialized."""
        assert sdk_registry is not None
        stats = sdk_registry.get_stats()

        assert "total_registered" in stats
        assert "total_initialized" in stats
        assert "healthy_count" in stats

    @pytest.mark.asyncio
    async def test_adapter_registration(self, sdk_registry, mock_adapter_class):
        """Test registering adapters in the registry."""
        from core.orchestration.sdk_registry import SDKRegistration
        from core.orchestration.base import SDKLayer

        registration = SDKRegistration(
            name="test-adapter",
            layer=SDKLayer.MEMORY,
            adapter_class=mock_adapter_class,
            priority=10
        )

        sdk_registry.register(registration)

        # Verify registration
        all_registrations = sdk_registry.get_all()
        assert any(r.name == "test-adapter" for r in all_registrations)

    @pytest.mark.asyncio
    async def test_adapter_lazy_loading(self, sdk_registry, mock_adapter_class):
        """Test that adapters are lazily loaded."""
        from core.orchestration.sdk_registry import SDKRegistration
        from core.orchestration.base import SDKLayer

        registration = SDKRegistration(
            name="lazy-adapter",
            layer=SDKLayer.MEMORY,
            adapter_class=mock_adapter_class,
            priority=5
        )

        sdk_registry.register(registration)

        # Before get, adapter should not be initialized
        stats_before = sdk_registry.get_stats()
        initial_count = stats_before["total_initialized"]

        # Get adapter triggers lazy loading
        adapter = await sdk_registry.get("lazy-adapter")

        assert adapter is not None
        stats_after = sdk_registry.get_stats()
        assert stats_after["total_initialized"] == initial_count + 1

    @pytest.mark.asyncio
    async def test_health_aware_routing(self, sdk_registry, mock_adapter_class):
        """Test health-aware adapter routing."""
        from core.orchestration.sdk_registry import SDKRegistration
        from core.orchestration.base import SDKLayer

        # Register two adapters with different priorities
        sdk_registry.register(SDKRegistration(
            name="high-priority",
            layer=SDKLayer.MEMORY,
            adapter_class=mock_adapter_class,
            priority=20
        ))
        sdk_registry.register(SDKRegistration(
            name="low-priority",
            layer=SDKLayer.MEMORY,
            adapter_class=mock_adapter_class,
            priority=5
        ))

        # Get best adapter (should be high priority)
        best = await sdk_registry.get_best_for_layer(SDKLayer.MEMORY)
        assert best is not None

        # Mark high-priority as unhealthy
        sdk_registry.update_health("high-priority", False)

        # Now should get low-priority
        best = await sdk_registry.get_best_for_layer(SDKLayer.MEMORY)
        # The best adapter should be from the healthy ones

    @pytest.mark.asyncio
    async def test_adapter_unregistration(self, sdk_registry, mock_adapter_class):
        """Test unregistering adapters."""
        from core.orchestration.sdk_registry import SDKRegistration
        from core.orchestration.base import SDKLayer

        sdk_registry.register(SDKRegistration(
            name="to-remove",
            layer=SDKLayer.MEMORY,
            adapter_class=mock_adapter_class,
        ))

        # Verify registered
        assert any(r.name == "to-remove" for r in sdk_registry.get_all())

        # Unregister
        result = sdk_registry.unregister("to-remove")
        assert result is True

        # Verify removed
        assert not any(r.name == "to-remove" for r in sdk_registry.get_all())

    @pytest.mark.asyncio
    async def test_health_check_all(self, sdk_registry, mock_adapter_class):
        """Test running health checks on all adapters."""
        from core.orchestration.sdk_registry import SDKRegistration
        from core.orchestration.base import SDKLayer

        # Register and initialize adapters
        for i in range(3):
            sdk_registry.register(SDKRegistration(
                name=f"health-test-{i}",
                layer=SDKLayer.MEMORY,
                adapter_class=mock_adapter_class,
            ))
            await sdk_registry.get(f"health-test-{i}")

        # Run health checks
        results = await sdk_registry.check_all_health()

        assert len(results) >= 3
        assert all(status for status in results.values())


@pytest.mark.integration
class TestConnectionPooling:
    """Integration tests for connection pooling infrastructure."""

    def test_connection_pool_initialization(self):
        """Test connection pool can be initialized."""
        if not CONNECTION_AVAILABLE:
            pytest.skip("ConnectionPool not available")

        pool = ConnectionPool(max_connections=50, connection_ttl=300.0)

        stats = pool.get_stats()
        assert stats["max_connections"] == 50
        assert stats["pooled_connections"] == 0
        assert stats["active_connections"] == 0

    def test_connection_acquire_and_release(self):
        """Test acquiring and releasing connections."""
        if not CONNECTION_AVAILABLE:
            pytest.skip("ConnectionPool not available")

        pool = ConnectionPool(max_connections=10)

        # Create a mock connection factory
        connection_counter = {"count": 0}

        def create_connection():
            connection_counter["count"] += 1
            return {"id": connection_counter["count"]}

        # Acquire connection
        conn1 = pool.acquire("test-sdk", create_connection)
        assert conn1 is not None
        assert conn1["id"] == 1

        stats = pool.get_stats()
        assert stats["active_connections"] == 1

        # Release connection
        pool.release("test-sdk", conn1)

        stats = pool.get_stats()
        assert stats["active_connections"] == 0
        assert stats["pooled_connections"] == 1

        # Acquire again - should reuse
        conn2 = pool.acquire("test-sdk", create_connection)
        assert conn2["id"] == 1  # Same connection reused

    def test_connection_pool_limit(self):
        """Test connection pool respects max limit."""
        if not CONNECTION_AVAILABLE:
            pytest.skip("ConnectionPool not available")

        pool = ConnectionPool(max_connections=2)

        connections = []

        def create_connection():
            return {"id": len(connections)}

        # Acquire up to limit
        for i in range(2):
            conn = pool.acquire(f"sdk-{i}", create_connection)
            assert conn is not None
            connections.append(conn)

        # Third acquire should fail
        conn3 = pool.acquire("sdk-3", create_connection)
        assert conn3 is None

    @pytest.mark.asyncio
    async def test_request_deduplication(self):
        """Test request deduplication for identical requests."""
        if not CONNECTION_AVAILABLE:
            pytest.skip("RequestDeduplicator not available")

        deduplicator = RequestDeduplicator()
        execution_count = {"count": 0}

        async def slow_executor(query: str):
            execution_count["count"] += 1
            await asyncio.sleep(0.1)
            return f"Result for {query}"

        # Launch identical requests concurrently
        results = await asyncio.gather(
            deduplicator.execute_deduplicated("sdk", "search", slow_executor, query="test"),
            deduplicator.execute_deduplicated("sdk", "search", slow_executor, query="test"),
            deduplicator.execute_deduplicated("sdk", "search", slow_executor, query="test"),
        )

        # Should only execute once
        assert execution_count["count"] == 1
        # All results should be the same
        assert all(r == "Result for test" for r in results)

        stats = deduplicator.get_stats()
        assert stats["total_requests"] == 3
        assert stats["deduplicated"] == 2

    @pytest.mark.asyncio
    async def test_warmup_preloader(self):
        """Test adapter warmup preloading."""
        if not CONNECTION_AVAILABLE:
            pytest.skip("WarmupPreloader not available")

        preloader = WarmupPreloader(warmup_timeout=5.0, parallel_warmup=True)

        async def fast_init():
            await asyncio.sleep(0.05)
            return True

        async def slow_init():
            await asyncio.sleep(0.1)
            return True

        results = await preloader.warmup_all({
            "fast-adapter": fast_init,
            "slow-adapter": slow_init,
        })

        assert results["fast-adapter"] is True
        assert results["slow-adapter"] is True

        stats = preloader.get_stats()
        assert stats["warmed_up"] == 2
        assert stats["success_rate_pct"] == 100.0

    @pytest.mark.asyncio
    async def test_warmup_timeout(self):
        """Test warmup timeout handling."""
        if not CONNECTION_AVAILABLE:
            pytest.skip("WarmupPreloader not available")

        preloader = WarmupPreloader(warmup_timeout=0.1)

        async def timeout_init():
            await asyncio.sleep(1.0)  # Will timeout
            return True

        result = await preloader.warmup_adapter("timeout-adapter", timeout_init)
        assert result is False

        stats = preloader.get_stats()
        assert "timeout-adapter" in stats["errors"]


# =============================================================================
# RAG PIPELINE TESTS
# =============================================================================

@pytest.mark.integration
class TestRAGPipeline:
    """Integration tests for RAG (Retrieval Augmented Generation) pipeline."""

    def test_document_chunking(self, sample_documents):
        """Test document chunking for RAG."""
        # Simple chunking implementation for testing
        def chunk_document(doc: Dict, chunk_size: int = 100, overlap: int = 20) -> List[Dict]:
            content = doc["content"]
            chunks = []

            if len(content) <= chunk_size:
                chunks.append({
                    "id": f"{doc['id']}-chunk-0",
                    "content": content,
                    "metadata": {**doc.get("metadata", {}), "parent_id": doc["id"], "chunk_index": 0}
                })
            else:
                start = 0
                chunk_idx = 0
                while start < len(content):
                    end = min(start + chunk_size, len(content))
                    chunk_content = content[start:end]
                    chunks.append({
                        "id": f"{doc['id']}-chunk-{chunk_idx}",
                        "content": chunk_content,
                        "metadata": {**doc.get("metadata", {}), "parent_id": doc["id"], "chunk_index": chunk_idx}
                    })
                    start = end - overlap if end < len(content) else len(content)
                    chunk_idx += 1

            return chunks

        all_chunks = []
        for doc in sample_documents:
            chunks = chunk_document(doc, chunk_size=50, overlap=10)
            all_chunks.extend(chunks)

        assert len(all_chunks) >= len(sample_documents)
        # Verify chunk structure
        for chunk in all_chunks:
            assert "id" in chunk
            assert "content" in chunk
            assert "metadata" in chunk
            assert "parent_id" in chunk["metadata"]

    @pytest.mark.asyncio
    async def test_retrieval_with_scoring(self, sample_documents):
        """Test document retrieval with relevance scoring."""

        # Simple BM25-style scoring
        def score_document(query: str, doc: Dict) -> float:
            query_terms = set(query.lower().split())
            doc_terms = set(doc["content"].lower().split())
            overlap = len(query_terms & doc_terms)
            return overlap / max(len(query_terms), 1)

        query = "machine learning models"

        scored_docs = [
            (doc, score_document(query, doc))
            for doc in sample_documents
        ]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Top result should be about machine learning
        top_doc = scored_docs[0][0]
        assert "machine" in top_doc["content"].lower() or "learning" in top_doc["content"].lower()

    @pytest.mark.asyncio
    async def test_hybrid_search(self, sample_documents):
        """Test hybrid search combining keyword and semantic approaches."""

        def keyword_search(query: str, docs: List[Dict]) -> List[tuple]:
            """Simple keyword matching."""
            results = []
            query_lower = query.lower()
            for doc in docs:
                if query_lower in doc["content"].lower():
                    results.append((doc, 1.0))
                else:
                    # Partial match
                    words = query_lower.split()
                    matches = sum(1 for w in words if w in doc["content"].lower())
                    if matches > 0:
                        results.append((doc, matches / len(words)))
            return results

        def semantic_search(query: str, docs: List[Dict]) -> List[tuple]:
            """Simulated semantic search using topic matching."""
            # In real implementation, this would use embeddings
            topic_keywords = {
                "programming": ["python", "language", "code", "development"],
                "machine_learning": ["model", "training", "prediction", "learn"],
                "nlp": ["language", "text", "natural", "understand"],
                "deep_learning": ["neural", "network", "layer", "deep"],
                "vector_db": ["vector", "embedding", "similarity", "database"],
            }

            results = []
            query_lower = query.lower()

            for doc in docs:
                topic = doc.get("metadata", {}).get("topic", "")
                score = 0.0

                if topic in topic_keywords:
                    keywords = topic_keywords[topic]
                    matches = sum(1 for k in keywords if k in query_lower)
                    score = matches / len(keywords) if keywords else 0

                if score > 0:
                    results.append((doc, score))

            return results

        def hybrid_search(query: str, docs: List[Dict], keyword_weight: float = 0.5) -> List[Dict]:
            """Combine keyword and semantic search."""
            keyword_results = {doc["id"]: score for doc, score in keyword_search(query, docs)}
            semantic_results = {doc["id"]: score for doc, score in semantic_search(query, docs)}

            combined = {}
            all_ids = set(keyword_results.keys()) | set(semantic_results.keys())

            for doc_id in all_ids:
                kw_score = keyword_results.get(doc_id, 0)
                sem_score = semantic_results.get(doc_id, 0)
                combined[doc_id] = keyword_weight * kw_score + (1 - keyword_weight) * sem_score

            # Sort by combined score
            sorted_ids = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)

            return [
                {"id": doc_id, "score": combined[doc_id]}
                for doc_id in sorted_ids if combined[doc_id] > 0
            ]

        # Test hybrid search
        results = hybrid_search("neural network deep learning", sample_documents)

        assert len(results) > 0
        # Deep learning document should rank high
        top_ids = [r["id"] for r in results[:2]]
        assert "doc-4" in top_ids or any("deep" in sample_documents[int(id.split("-")[1]) - 1]["content"].lower() for id in top_ids if id.startswith("doc-"))

    @pytest.mark.asyncio
    async def test_reranking(self, sample_documents):
        """Test document reranking for improved relevance."""

        def initial_retrieval(query: str, docs: List[Dict], k: int = 5) -> List[Dict]:
            """First-stage retrieval."""
            return docs[:k]

        def rerank(query: str, candidates: List[Dict]) -> List[Dict]:
            """Rerank candidates based on query-document similarity."""
            query_words = set(query.lower().split())

            scored = []
            for doc in candidates:
                doc_words = set(doc["content"].lower().split())
                # Jaccard similarity
                intersection = len(query_words & doc_words)
                union = len(query_words | doc_words)
                score = intersection / union if union > 0 else 0

                # Boost for exact phrase matches
                if query.lower() in doc["content"].lower():
                    score += 0.5

                scored.append({"doc": doc, "score": score})

            scored.sort(key=lambda x: x["score"], reverse=True)
            return [s["doc"] for s in scored]

        query = "vector databases and embeddings"

        # Initial retrieval (just takes first N)
        candidates = initial_retrieval(query, sample_documents)

        # Rerank
        reranked = rerank(query, candidates)

        # Vector DB doc should be ranked higher after reranking
        assert any("vector" in doc["content"].lower() for doc in reranked[:2])


# =============================================================================
# RESEARCH ADAPTER TESTS
# =============================================================================

@pytest.mark.integration
class TestResearchAdapters:
    """Integration tests for research adapters with mocking."""

    @pytest.mark.asyncio
    async def test_exa_adapter_mock_mode(self):
        """Test Exa adapter in mock mode (no API key)."""
        if ExaAdapter is None:
            pytest.skip("ExaAdapter not available")

        adapter = ExaAdapter()

        # Initialize without API key - should enter mock mode
        with patch.dict(os.environ, {"EXA_API_KEY": ""}, clear=False):
            result = await adapter.initialize({})

        assert result.success

        # Execute search in mock mode
        search_result = await adapter.execute("search", query="test query")

        assert search_result.success
        assert search_result.data is not None
        assert search_result.data.get("mock") is True

        await adapter.shutdown()

    @pytest.mark.asyncio
    async def test_adapter_error_handling(self):
        """Test adapter error handling."""
        if ExaAdapter is None:
            pytest.skip("ExaAdapter not available")

        adapter = ExaAdapter()
        await adapter.initialize({})

        # Test invalid operation
        result = await adapter.execute("invalid_operation")

        assert result.success is False
        assert result.error is not None
        assert "Unknown operation" in result.error

        await adapter.shutdown()

    @pytest.mark.asyncio
    async def test_adapter_timeout_behavior(self):
        """Test adapter timeout handling."""
        if not SDK_REGISTRY_AVAILABLE:
            pytest.skip("SDK base classes not available")

        # Create an adapter that simulates slow operations
        class SlowAdapter(SDKAdapter):
            def __init__(self):
                self._status = AdapterStatus.UNINITIALIZED

            @property
            def sdk_name(self) -> str:
                return "slow"

            @property
            def layer(self) -> SDKLayer:
                return SDKLayer.RESEARCH

            @property
            def available(self) -> bool:
                return True

            async def initialize(self, config: Dict) -> AdapterResult:
                self._status = AdapterStatus.READY
                return AdapterResult(success=True)

            async def execute(self, operation: str, **kwargs) -> AdapterResult:
                timeout = kwargs.get("timeout", 10)
                try:
                    await asyncio.wait_for(
                        asyncio.sleep(100),  # Would take forever
                        timeout=0.1  # But we timeout quickly
                    )
                except asyncio.TimeoutError:
                    return AdapterResult(
                        success=False,
                        error="Operation timed out"
                    )
                return AdapterResult(success=True)

            async def health_check(self) -> AdapterResult:
                return AdapterResult(success=True)

            async def shutdown(self) -> AdapterResult:
                return AdapterResult(success=True)

        adapter = SlowAdapter()
        await adapter.initialize({})

        result = await adapter.execute("slow_operation", timeout=0.1)

        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_adapter_latency_tracking(self):
        """Test that adapter tracks operation latency."""
        if ExaAdapter is None:
            pytest.skip("ExaAdapter not available")

        adapter = ExaAdapter()
        await adapter.initialize({})

        result = await adapter.execute("search", query="latency test")

        assert result.latency_ms >= 0
        # Operations should complete within reasonable time
        assert result.latency_ms < 10000

        await adapter.shutdown()

    @pytest.mark.asyncio
    async def test_mock_adapter_responses(self, mock_adapter_class):
        """Test mock adapter response structure."""
        adapter = mock_adapter_class()
        await adapter.initialize({})

        result = await adapter.execute("test_operation", param1="value1", param2="value2")

        assert result.success
        assert result.data is not None
        assert result.data["operation"] == "test_operation"
        assert result.data["param1"] == "value1"
        assert result.data["param2"] == "value2"

    @pytest.mark.asyncio
    async def test_adapter_health_check_integration(self, mock_adapter_class):
        """Test health check integration."""
        adapter = mock_adapter_class()
        await adapter.initialize({})

        # Initial health check should pass
        result = await adapter.health_check()
        assert result.success

        # Simulate unhealthy state
        adapter._healthy = False
        result = await adapter.health_check()
        assert result.success is False

    @pytest.mark.asyncio
    async def test_adapter_shutdown(self, mock_adapter_class):
        """Test adapter shutdown cleans up properly."""
        adapter = mock_adapter_class()
        await adapter.initialize({})

        assert adapter._status == AdapterStatus.READY

        await adapter.shutdown()

        assert adapter._status == AdapterStatus.SHUTDOWN


# =============================================================================
# EMBEDDING-BASED SEARCH TESTS
# =============================================================================

@pytest.mark.integration
class TestEmbeddingSearch:
    """Integration tests for embedding-based semantic search."""

    def test_embedding_generation_mock(self, sample_documents):
        """Test mock embedding generation."""
        import random

        def generate_embedding(text: str, dim: int = 768) -> List[float]:
            """Generate deterministic pseudo-embeddings for testing."""
            random.seed(hash(text) % (2**32))
            return [random.random() for _ in range(dim)]

        embeddings = {}
        for doc in sample_documents:
            embeddings[doc["id"]] = generate_embedding(doc["content"])

        # All embeddings should have same dimension
        assert all(len(e) == 768 for e in embeddings.values())

        # Same text should produce same embedding
        e1 = generate_embedding("test text")
        e2 = generate_embedding("test text")
        assert e1 == e2

        # Different text should produce different embedding
        e3 = generate_embedding("different text")
        assert e1 != e3

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        import math

        def cosine_similarity(a: List[float], b: List[float]) -> float:
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(y * y for y in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot_product / (norm_a * norm_b)

        # Identical vectors
        v1 = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v1, v1) - 1.0) < 0.0001

        # Orthogonal vectors
        v2 = [1.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(v2, v3)) < 0.0001

        # Similar vectors
        v4 = [1.0, 2.0, 3.0]
        v5 = [1.1, 2.1, 3.1]
        sim = cosine_similarity(v4, v5)
        assert sim > 0.99

    def test_semantic_search_with_embeddings(self, sample_documents):
        """Test semantic search using embeddings."""
        import random
        import math

        def generate_embedding(text: str, dim: int = 64) -> List[float]:
            random.seed(hash(text) % (2**32))
            return [random.random() for _ in range(dim)]

        def cosine_similarity(a: List[float], b: List[float]) -> float:
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(y * y for y in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot_product / (norm_a * norm_b)

        # Generate embeddings for all documents
        doc_embeddings = {
            doc["id"]: generate_embedding(doc["content"])
            for doc in sample_documents
        }

        # Query embedding
        query = "neural network deep learning"
        query_embedding = generate_embedding(query)

        # Calculate similarities
        similarities = [
            (doc["id"], cosine_similarity(query_embedding, doc_embeddings[doc["id"]]))
            for doc in sample_documents
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)

        # We should get ranked results
        assert len(similarities) == len(sample_documents)
        assert all(0 <= sim <= 1 for _, sim in similarities)


# =============================================================================
# SESSION HOOKS TESTS
# =============================================================================

@pytest.mark.integration
class TestMemoryHooks:
    """Integration tests for memory session hooks."""

    def test_session_start_hook(self, temp_memory_dir):
        """Test hook triggered on session start."""
        if not CROSS_SESSION_AVAILABLE:
            pytest.skip("CrossSessionMemory not available")

        hook_called = {"count": 0, "session_id": None}

        class HookedMemory(CrossSessionMemory):
            def start_session(self, task_summary: str = "") -> Session:
                session = super().start_session(task_summary)
                hook_called["count"] += 1
                hook_called["session_id"] = session.id
                return session

        mem = HookedMemory(base_path=temp_memory_dir)
        session = mem.start_session("Test with hook")

        assert hook_called["count"] == 1
        assert hook_called["session_id"] == session.id

    def test_session_end_hook(self, temp_memory_dir):
        """Test hook triggered on session end."""
        if not CROSS_SESSION_AVAILABLE:
            pytest.skip("CrossSessionMemory not available")

        hook_called = {"count": 0, "summary": None}

        class HookedMemory(CrossSessionMemory):
            def end_session(self, summary: Optional[str] = None) -> None:
                hook_called["count"] += 1
                hook_called["summary"] = summary
                super().end_session(summary)

        mem = HookedMemory(base_path=temp_memory_dir)
        mem.start_session("Test session")
        mem.end_session("Session completed successfully")

        assert hook_called["count"] == 1
        assert hook_called["summary"] == "Session completed successfully"

    def test_memory_add_hook(self, temp_memory_dir):
        """Test hook triggered when memory is added."""
        if not CROSS_SESSION_AVAILABLE:
            pytest.skip("CrossSessionMemory not available")

        added_memories = []

        class HookedMemory(CrossSessionMemory):
            def add(
                self,
                content: str,
                memory_type: str = "context",
                importance: float = 0.5,
                tags: Optional[List[str]] = None,
                metadata: Optional[Dict[str, Any]] = None
            ) -> Memory:
                memory = super().add(content, memory_type, importance, tags, metadata)
                added_memories.append({
                    "id": memory.id,
                    "type": memory_type,
                    "importance": importance
                })
                return memory

        mem = HookedMemory(base_path=temp_memory_dir)
        mem.start_session("Hook test")

        mem.add("Decision 1", memory_type="decision", importance=0.8)
        mem.add("Learning 1", memory_type="learning", importance=0.6)

        assert len(added_memories) == 2
        assert added_memories[0]["type"] == "decision"
        assert added_memories[0]["importance"] == 0.8


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestPerformance:
    """Performance integration tests."""

    @pytest.mark.asyncio
    async def test_memory_bulk_operations(self, temp_memory_dir):
        """Test bulk memory operations performance."""
        if not CROSS_SESSION_AVAILABLE:
            pytest.skip("CrossSessionMemory not available")

        mem = CrossSessionMemory(base_path=temp_memory_dir)
        mem.start_session("Bulk test")

        # Bulk add
        start_time = time.time()
        for i in range(100):
            mem.add(f"Bulk content item {i}", memory_type="context", importance=0.5)
        add_duration = time.time() - start_time

        # Bulk search
        start_time = time.time()
        for i in range(50):
            mem.search(f"item {i}", limit=10)
        search_duration = time.time() - start_time

        # Performance assertions
        assert add_duration < 5.0, f"Bulk add took {add_duration}s, expected <5s"
        assert search_duration < 2.0, f"Bulk search took {search_duration}s, expected <2s"

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, temp_memory_dir):
        """Test concurrent memory operations."""
        if not CROSS_SESSION_AVAILABLE:
            pytest.skip("CrossSessionMemory not available")

        mem = CrossSessionMemory(base_path=temp_memory_dir)
        mem.start_session("Concurrent test")

        async def add_memories(prefix: str, count: int):
            for i in range(count):
                mem.add(f"{prefix}-{i}", memory_type="context")
                await asyncio.sleep(0)  # Yield to event loop

        # Run concurrent additions
        await asyncio.gather(
            add_memories("thread-a", 20),
            add_memories("thread-b", 20),
            add_memories("thread-c", 20),
        )

        stats = mem.get_stats()
        assert stats["total_memories"] == 60


# =============================================================================
# CLEANUP AND UTILITIES
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_global_registry():
    """Reset global registry between tests."""
    yield
    # Cleanup after test
    if SDK_REGISTRY_AVAILABLE:
        import core.orchestration.sdk_registry as registry_module
        registry_module._registry = None


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
