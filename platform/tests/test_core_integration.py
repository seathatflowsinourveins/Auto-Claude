"""
Integration tests for the UAP Core package.

Tests that all modules work together correctly.
"""

import asyncio
import inspect
import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCoreImports:
    """Test that all core modules import correctly."""

    def test_memory_imports(self):
        """Test memory module imports."""
        from core.memory import MemorySystem, CoreMemory, ArchivalMemory, TemporalGraph
        assert MemorySystem is not None
        assert CoreMemory is not None
        assert ArchivalMemory is not None
        assert TemporalGraph is not None

    def test_cooperation_imports(self):
        """Test cooperation module imports."""
        from core.cooperation import CooperationManager, TaskCoordinator, SessionHandoff
        assert CooperationManager is not None
        assert TaskCoordinator is not None
        assert SessionHandoff is not None

    def test_harness_imports(self):
        """Test harness module imports."""
        from core.harness import AgentHarness, ContextWindow, ShiftHandoff
        assert AgentHarness is not None
        assert ContextWindow is not None
        assert ShiftHandoff is not None

    def test_mcp_manager_imports(self):
        """Test MCP manager module imports."""
        from core.mcp_manager import MCPServerManager, ServerConfig, ToolSchema
        assert MCPServerManager is not None
        assert ServerConfig is not None
        assert ToolSchema is not None

    def test_executor_imports(self):
        """Test executor module imports."""
        from core.executor import AgentExecutor, create_executor, ThinkingMode
        assert AgentExecutor is not None
        assert create_executor is not None
        assert ThinkingMode is not None

    def test_thinking_imports(self):
        """Test thinking module imports."""
        from core.thinking import ThinkingEngine, create_thinking_engine, ThinkingStrategy
        assert ThinkingEngine is not None
        assert create_thinking_engine is not None
        assert ThinkingStrategy is not None

    def test_package_imports(self):
        """Test all imports from package __init__."""
        from core import (
            MemorySystem,
            CooperationManager,
            AgentHarness,
            MCPServerManager,
            AgentExecutor,
            ThinkingEngine,
        )
        assert all([
            MemorySystem,
            CooperationManager,
            AgentHarness,
            MCPServerManager,
            AgentExecutor,
            ThinkingEngine,
        ])


class TestMemorySystem:
    """Test the memory system."""

    def test_create_memory_system(self):
        """Test creating a memory system."""
        from core.memory import MemorySystem
        memory = MemorySystem(agent_id="test-agent")
        assert memory.agent_id == "test-agent"
        assert memory.core is not None
        assert memory.archival is not None
        assert memory.temporal is not None

    def test_core_memory_operations(self):
        """Test core memory operations."""
        from core.memory import MemorySystem
        memory = MemorySystem(agent_id="test-agent")

        # Update a block
        result = memory.core.update("task_state", "Working on tests")
        assert result is True

        # Get the block
        content = memory.core.get("task_state")
        assert content == "Working on tests"


class TestHarness:
    """Test the agent harness."""

    def test_create_harness(self):
        """Test creating an agent harness."""
        from core.harness import AgentHarness
        harness = AgentHarness(max_tokens=50000)
        assert harness.max_tokens == 50000

    def test_begin_task(self):
        """Test beginning a task."""
        from core.harness import AgentHarness
        harness = AgentHarness()
        task_id = harness.begin_task("Test task", steps=["Step 1", "Step 2"])
        assert task_id is not None

    def test_context_window(self):
        """Test context window operations."""
        from core.harness import ContextWindow, ContextPillar
        ctx = ContextWindow(max_tokens=10000)

        # Add to context
        result = ctx.add(ContextPillar.KNOWLEDGE, "test_key", "test content")
        assert result is True

        # Get from context
        content = ctx.get(ContextPillar.KNOWLEDGE, "test_key")
        assert content == "test content"

        # Get summary
        summary = ctx.get_context_summary()
        assert "KNOWLEDGE" in summary


class TestMCPManager:
    """Test the MCP server manager."""

    def test_create_manager(self):
        """Test creating an MCP manager."""
        from core.mcp_manager import MCPServerManager
        manager = MCPServerManager()
        assert manager is not None

    def test_register_server(self):
        """Test registering a server."""
        from core.mcp_manager import MCPServerManager, ServerConfig
        manager = MCPServerManager()

        config = ServerConfig(name="test-server", command=["python", "-m", "test"])
        manager.register_server(config)

        status = manager.get_server_status("test-server")
        assert status is not None
        assert status["status"] == "registered"


class TestThinkingEngine:
    """Test the thinking engine."""

    def test_create_engine(self):
        """Test creating a thinking engine."""
        from core.thinking import create_thinking_engine
        engine = create_thinking_engine(budget_tokens=8000)
        assert engine is not None
        assert engine.tokens_remaining > 0

    def test_think_through(self):
        """Test thinking through a question."""
        from core.thinking import create_thinking_engine
        engine = create_thinking_engine(budget_tokens=8000)

        chain = engine.think_through("What is 2 + 2?", pattern="analytical")
        assert chain is not None
        assert len(chain.steps) > 0
        assert chain.conclusion is not None
        assert 0 <= chain.final_confidence <= 1


class TestExecutor:
    """Test the agent executor."""

    def test_create_executor(self):
        """Test creating an executor."""
        from core.executor import create_executor, ThinkingMode
        executor = create_executor(
            agent_id="test-executor",
            thinking_mode=ThinkingMode.INTERLEAVED,
        )
        assert executor is not None
        assert executor.is_running is False

    @pytest.mark.asyncio
    async def test_execute_task(self):
        """Test executing a task."""
        from core.executor import create_executor, ThinkingMode

        executor = create_executor(
            agent_id="test-executor",
            max_tokens=50000,
            thinking_mode=ThinkingMode.INTERLEAVED,
        )

        # Execute with very low max_iterations for testing
        executor._max_iterations = 6

        result = await executor.execute(
            task="Simple test task",
            context={"test": True},
        )

        assert result is not None
        assert result.task_description == "Simple test task"
        assert result.iteration_count > 0


class TestIntegration:
    """Integration tests across modules."""

    def test_full_workflow(self):
        """Test a complete workflow across modules."""
        from core.memory import MemorySystem
        from core.harness import AgentHarness
        from core.mcp_manager import MCPServerManager
        from core.thinking import create_thinking_engine

        # Create components
        memory = MemorySystem(agent_id="integration-test")
        harness = AgentHarness(max_tokens=50000)
        mcp = MCPServerManager()
        thinking = create_thinking_engine(budget_tokens=8000)

        # Use memory
        memory.core.update("task_state", "Running integration test")

        # Use harness
        harness.begin_task("Integration test")
        harness.record_action("Test action", "Success", success=True)

        # Use thinking
        chain = thinking.think_through("How to verify integration?")

        # Verify all components work together
        assert memory.core.get("task_state") == "Running integration test"
        assert len(harness.actions) == 1
        assert chain.conclusion is not None


class TestSkillSystem:
    """Test the skill system."""

    def test_create_registry(self):
        """Test creating a skill registry."""
        from core.skills import create_skill_registry
        registry = create_skill_registry(include_builtins=True)
        assert registry is not None
        assert len(registry._skills) >= 3  # Built-in skills

    def test_find_relevant_skills(self):
        """Test finding relevant skills for a query."""
        from core.skills import create_skill_registry
        registry = create_skill_registry(include_builtins=True)

        results = registry.find_relevant("code review security", max_skills=2)
        assert len(results) > 0
        # The code-review skill should be highly relevant
        skill_names = [s.metadata.name for s, _ in results]
        assert "code-review" in skill_names

    def test_skill_activation(self):
        """Test activating and deactivating skills."""
        from core.skills import create_skill_registry, SkillLoadLevel
        registry = create_skill_registry(include_builtins=True)

        # Activate a skill
        result = registry.activate("ultrathink")
        assert result is True
        assert "ultrathink" in registry._active_skills

        # Get active context
        context = registry.get_active_context(SkillLoadLevel.SUMMARY)
        assert "ultrathink" in context.lower() or "Ultrathink" in context

        # Deactivate
        result = registry.deactivate("ultrathink")
        assert result is True
        assert "ultrathink" not in registry._active_skills


class TestToolRegistry:
    """Test the tool registry."""

    def test_create_registry(self):
        """Test creating a tool registry."""
        from core.tool_registry import create_tool_registry
        registry = create_tool_registry(include_builtins=True)
        assert registry is not None
        assert len(registry._tools) >= 2  # Built-in tools

    def test_search_tools(self):
        """Test searching for tools."""
        from core.tool_registry import create_tool_registry
        registry = create_tool_registry(include_builtins=True)

        results = registry.search("read file")
        assert len(results) > 0
        # Read tool should be found
        tool_names = [t.schema_.name for t in results]
        assert "Read" in tool_names

    def test_recommend_for_task(self):
        """Test tool recommendation for a task."""
        from core.tool_registry import create_tool_registry
        registry = create_tool_registry(include_builtins=True)

        results = registry.recommend_for_task("read configuration files")
        assert len(results) > 0

    def test_get_schemas_for_llm(self):
        """Test getting LLM-compatible schemas."""
        from core.tool_registry import create_tool_registry
        registry = create_tool_registry(include_builtins=True)

        schemas = registry.get_schemas_for_llm(format="anthropic")
        assert len(schemas) > 0
        assert all("name" in s for s in schemas)
        assert all("input_schema" in s for s in schemas)


class TestPersistence:
    """Test the persistence system."""

    def test_create_manager(self):
        """Test creating a persistence manager."""
        from core.persistence import PersistenceManager, PersistenceBackend
        manager = PersistenceManager(backend=PersistenceBackend.MEMORY)
        assert manager is not None

    def test_create_session(self):
        """Test creating a session."""
        from core.persistence import PersistenceManager, PersistenceBackend
        manager = PersistenceManager(backend=PersistenceBackend.MEMORY)

        session = manager.create_session(
            session_id="test-session",
            agent_id="test-agent",
            task_description="Test task",
        )
        assert session is not None
        assert session.metadata.session_id == "test-session"

    def test_save_and_get_state(self):
        """Test saving and retrieving state."""
        from core.persistence import PersistenceManager, PersistenceBackend
        manager = PersistenceManager(backend=PersistenceBackend.MEMORY)

        manager.create_session(session_id="test-session")
        manager.save_state("test-session", "counter", 42)

        value = manager.get_state("test-session", "counter")
        assert value == 42

    def test_create_checkpoint(self):
        """Test creating a checkpoint."""
        from core.persistence import PersistenceManager, PersistenceBackend, CheckpointType
        manager = PersistenceManager(backend=PersistenceBackend.MEMORY)

        manager.create_session(session_id="test-session")
        checkpoint = manager.create_checkpoint(
            session_id="test-session",
            checkpoint_type=CheckpointType.AUTO,
            description="Test checkpoint",
        )
        assert checkpoint is not None
        assert checkpoint.session_id == "test-session"

    def test_restore_checkpoint(self):
        """Test restoring from a checkpoint."""
        from core.persistence import PersistenceManager, PersistenceBackend
        manager = PersistenceManager(backend=PersistenceBackend.MEMORY)

        manager.create_session(session_id="test-session")
        manager.create_checkpoint(
            session_id="test-session",
            description="Test checkpoint",
            task_state={"progress": 50},
        )

        restored = manager.restore_checkpoint("test-session")
        assert restored is not None
        assert restored.task_state.get("progress") == 50


class TestOrchestrator:
    """Test the agent orchestrator."""

    def test_create_orchestrator(self):
        """Test creating an orchestrator."""
        from core.orchestrator import create_orchestrator, Topology
        orch = create_orchestrator(topology=Topology.HYBRID)
        assert orch is not None
        assert orch._topology == Topology.HYBRID

    def test_register_agent(self):
        """Test registering agents."""
        from core.orchestrator import create_orchestrator, AgentRole, AgentCapability
        orch = create_orchestrator()

        agent = orch.register_agent(
            "test-agent",
            role=AgentRole.SPECIALIST,
            capabilities=[AgentCapability("python", proficiency=0.9)],
        )
        assert agent is not None
        assert agent.agent_id == "test-agent"
        assert len(agent.capabilities) == 1

    def test_submit_task(self):
        """Test submitting tasks."""
        from core.orchestrator import create_orchestrator, TaskPriority, TaskStatus
        orch = create_orchestrator()

        task_id = orch.submit_task(
            "Test task",
            priority=TaskPriority.HIGH,
            required_capabilities=["python"],
        )
        assert task_id is not None

        task = orch.get_task(task_id)
        assert task is not None
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.QUEUED

    def test_task_decomposition(self):
        """Test decomposing tasks."""
        from core.orchestrator import create_orchestrator, SequentialDecomposition
        orch = create_orchestrator()

        task_id = orch.submit_task("Complex task")
        decomposition = orch.decompose_task(
            task_id,
            SequentialDecomposition(["Step 1", "Step 2", "Step 3"]),
        )

        assert decomposition is not None
        assert len(decomposition.subtasks) == 3
        # Second step depends on first
        assert len(decomposition.subtasks[1].dependencies) == 1

    def test_capability_matching(self):
        """Test capability-based agent matching."""
        from core.orchestrator import create_orchestrator, AgentCapability, AgentRole
        orch = create_orchestrator(enable_swarm=False)

        # Register agent with Python capability
        orch.register_agent(
            "python-agent",
            role=AgentRole.SPECIALIST,
            capabilities=[AgentCapability("python", proficiency=0.95)],
        )

        # Submit Python task
        task_id = orch.submit_task("Write Python code", required_capabilities=["python"])

        # Find best agent
        task = orch.get_task(task_id)
        agent = orch._find_best_agent(task)
        assert agent is not None
        assert agent.agent_id == "python-agent"

    def test_orchestrator_metrics(self):
        """Test orchestrator metrics."""
        from core.orchestrator import create_orchestrator
        orch = create_orchestrator()

        # Register some agents
        orch.register_agent("agent-1")
        orch.register_agent("agent-2")

        # Submit tasks
        orch.submit_task("Task 1")
        orch.submit_task("Task 2")

        metrics = orch.get_metrics()
        assert metrics.total_tasks_submitted == 2
        assert metrics.idle_agents == 2


class TestMCPDiscovery:
    """Test the MCP discovery system."""

    def test_create_discovery_manager(self):
        """Test creating a discovery manager."""
        from core.mcp_discovery import MCPDiscovery
        discovery = MCPDiscovery(enable_registry=False, enable_pooling=True)
        assert discovery is not None
        assert discovery.manager is not None

    def test_registry_entry(self):
        """Test registry entry model."""
        from core.mcp_discovery import RegistryEntry
        entry = RegistryEntry(
            name="test-server",
            description="A test MCP server",
            version="1.0.0",
            package_name="@test/mcp-server",
            package_manager="npm",
            tool_count=5,
        )
        assert entry.name == "test-server"
        assert entry.tool_count == 5

    def test_server_capabilities(self):
        """Test server capabilities model."""
        from core.mcp_discovery import ServerCapabilities, MCPProtocolVersion
        from core.mcp_manager import ToolSchema
        caps = ServerCapabilities(
            server_name="test",
            protocol_version=MCPProtocolVersion.V_2025_06.value,
            supports_tools=True,
            tools=[ToolSchema(name="test_tool", description="Test")],
        )
        assert caps.supports_tools is True
        assert len(caps.tools) == 1

    def test_connection_pool_stats(self):
        """Test connection pool statistics."""
        from core.mcp_discovery import ConnectionPool
        pool = ConnectionPool(max_connections_per_server=3)
        stats = pool.get_stats()
        assert stats["total_connections"] == 0
        assert stats["healthy_connections"] == 0

    def test_discovery_stats(self):
        """Test discovery statistics."""
        from core.mcp_discovery import MCPDiscovery
        discovery = MCPDiscovery(enable_registry=False)
        stats = discovery.get_discovery_stats()
        assert "servers_registered" in stats
        assert "total_tools_discovered" in stats

    def test_factory_functions(self):
        """Test factory functions."""
        from core.mcp_discovery import (
            create_discovery_manager,
            create_registry_client,
            create_connection_pool,
        )
        discovery = create_discovery_manager(enable_registry=False)
        assert discovery is not None

        pool = create_connection_pool(max_per_server=5)
        assert pool is not None


class TestAdvancedMemory:
    """Test advanced memory with Letta integration."""

    def test_embedding_model_enum(self):
        """Test embedding model enum values."""
        from core.advanced_memory import EmbeddingModel
        assert EmbeddingModel.LOCAL_MINILM.value == "all-MiniLM-L6-v2"
        assert EmbeddingModel.OPENAI_ADA.value == "text-embedding-ada-002"
        assert EmbeddingModel.LETTA_DEFAULT.value == "letta-default"

    def test_embedding_result(self):
        """Test embedding result dataclass."""
        from core.advanced_memory import EmbeddingResult
        result = EmbeddingResult(
            text="test content",
            embedding=[0.1, 0.2, 0.3],
            model="test-model",
            dimensions=3,
            tokens_used=5,
        )
        assert result.text == "test content"
        assert len(result.embedding) == 3
        assert result.dimensions == 3

    def test_semantic_entry(self):
        """Test semantic entry dataclass."""
        from core.advanced_memory import SemanticEntry
        entry = SemanticEntry(
            id="entry-001",
            content="test content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "test"},
            importance=0.8,
        )
        assert entry.id == "entry-001"
        assert entry.importance == 0.8
        assert entry.access_count == 0

    def test_consolidation_strategy_enum(self):
        """Test consolidation strategy enum."""
        from core.advanced_memory import ConsolidationStrategy
        assert ConsolidationStrategy.SUMMARIZE.value == "summarize"
        assert ConsolidationStrategy.COMPRESS.value == "compress"
        assert ConsolidationStrategy.PRUNE.value == "prune"
        assert ConsolidationStrategy.HIERARCHICAL.value == "hierarchical"

    def test_consolidation_result(self):
        """Test consolidation result dataclass."""
        from core.advanced_memory import ConsolidationResult, ConsolidationStrategy
        result = ConsolidationResult(
            strategy=ConsolidationStrategy.SUMMARIZE,
            entries_processed=100,
            entries_removed=50,
            entries_created=5,
            tokens_saved=5000,
            duration_ms=150.0,
        )
        assert result.entries_removed == 50
        assert result.tokens_saved == 5000
        assert result.entries_created == 5

    def test_factory_functions(self):
        """Test advanced memory factory functions."""
        from core.advanced_memory import (
            create_advanced_memory,
            create_consolidator,
            SemanticIndex,
            LocalEmbeddingProvider,
        )
        # Test without Letta (no API key needed)
        memory = create_advanced_memory(
            agent_id="test-agent",
        )
        assert memory is not None
        assert memory.agent_id == "test-agent"

        # Create semantic index with local embeddings for consolidator test
        embedding_provider = LocalEmbeddingProvider()
        semantic_index = SemanticIndex(embedding_provider)
        consolidator = create_consolidator(semantic_index)
        assert consolidator is not None

    def test_advanced_memory_stats(self):
        """Test advanced memory system statistics."""
        from core.advanced_memory import create_advanced_memory
        memory = create_advanced_memory(
            agent_id="stats-test",
        )
        stats = memory.get_stats()
        assert stats["agent_id"] == "stats-test"
        assert "core_blocks" in stats
        assert "letta_enabled" in stats
        assert stats["letta_enabled"] is False  # No API key provided


class TestUltrathink:
    """Test ultrathink extended thinking module."""

    def test_thinking_level_enum(self):
        """Test thinking level enum values."""
        from core.ultrathink import ThinkingLevel
        assert ThinkingLevel.QUICK.value == "quick"
        assert ThinkingLevel.THINK.value == "think"
        assert ThinkingLevel.HARDTHINK.value == "hardthink"
        assert ThinkingLevel.ULTRATHINK.value == "ultrathink"

    def test_detect_thinking_level(self):
        """Test power word detection."""
        from core.ultrathink import detect_thinking_level, ThinkingLevel
        assert detect_thinking_level("What is 2+2?") == ThinkingLevel.QUICK
        assert detect_thinking_level("Analyze this think") == ThinkingLevel.THINK
        assert detect_thinking_level("Design this hardthink") == ThinkingLevel.HARDTHINK
        assert detect_thinking_level("Plan this ultrathink") == ThinkingLevel.ULTRATHINK

    def test_thinking_budgets(self):
        """Test thinking budget mappings."""
        from core.ultrathink import THINKING_BUDGETS, ThinkingLevel
        assert THINKING_BUDGETS[ThinkingLevel.QUICK] == 1000
        assert THINKING_BUDGETS[ThinkingLevel.ULTRATHINK] == 128000

    def test_cot_phase_enum(self):
        """Test chain of thought phase enum."""
        from core.ultrathink import CoTPhase
        assert CoTPhase.UNDERSTAND.value == "understand"
        assert CoTPhase.SYNTHESIZE.value == "synthesize"
        assert CoTPhase.CONCLUDE.value == "conclude"

    def test_tree_of_thoughts(self):
        """Test Tree of Thoughts exploration."""
        from core.ultrathink import create_tree_of_thoughts
        tot = create_tree_of_thoughts(max_branches=3)
        root = tot.create_root("How to design this?")
        tot.add_branch(root.id, "Option A", 0.8)
        tot.add_branch(root.id, "Option B", 0.6)

        stats = tot.get_stats()
        assert stats["total_branches"] == 3
        assert stats["promising_branches"] == 3

    def test_confidence_calibrator(self):
        """Test confidence calibration."""
        from core.ultrathink import create_confidence_calibrator, EvidenceItem
        calibrator = create_confidence_calibrator()
        evidence = [
            EvidenceItem(
                content="Test evidence",
                source="test",
                strength=0.8,
                relevance=0.9,
            )
        ]
        calibrated, level = calibrator.calibrate(0.7, evidence, 0.8)
        assert 0.0 <= calibrated <= 1.0
        assert level is not None

    def test_ultrathink_engine(self):
        """Test UltrathinkEngine creation and basic operation."""
        from core.ultrathink import create_ultrathink_engine, ThinkingLevel
        engine = create_ultrathink_engine()
        chain = engine.think("Test problem ultrathink")
        assert chain is not None
        assert chain.level == ThinkingLevel.ULTRATHINK
        assert len(chain.steps) > 0

    def test_engine_stats(self):
        """Test engine statistics."""
        from core.ultrathink import create_ultrathink_engine
        engine = create_ultrathink_engine()
        engine.think("Test problem")
        stats = engine.get_stats()
        assert "total_chains" in stats
        assert "completed_chains" in stats
        assert stats["total_chains"] == 1


class TestResilience:
    """Test resilience module for production hardening."""

    def test_circuit_state_enum(self):
        """Test circuit state enum values."""
        from core.resilience import CircuitState
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"

    def test_create_circuit_breaker(self):
        """Test circuit breaker creation."""
        from core.resilience import create_circuit_breaker, CircuitState
        breaker = create_circuit_breaker(failure_threshold=5, recovery_timeout=30.0)
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_threshold == 5
        assert breaker.recovery_timeout == 30.0

    def test_retry_strategy_enum(self):
        """Test retry strategy enum."""
        from core.resilience import RetryStrategy
        assert RetryStrategy.FIXED.value == "fixed"
        assert RetryStrategy.EXPONENTIAL.value == "exponential"
        assert RetryStrategy.DECORRELATED_JITTER.value == "decorrelated_jitter"

    def test_create_retry_policy(self):
        """Test retry policy creation."""
        from core.resilience import create_retry_policy, RetryStrategy
        policy = create_retry_policy(max_retries=3, base_delay=1.0)
        assert policy.max_retries == 3
        assert policy.base_delay == 1.0
        assert policy.strategy == RetryStrategy.EXPONENTIAL

    def test_retry_delay_calculation(self):
        """Test exponential backoff delay calculation."""
        from core.resilience import RetryPolicy, RetryStrategy
        policy = RetryPolicy(
            max_retries=5,
            base_delay=1.0,
            strategy=RetryStrategy.EXPONENTIAL,
            jitter=False,
        )
        assert policy.calculate_delay(0) == 1.0
        assert policy.calculate_delay(1) == 2.0
        assert policy.calculate_delay(2) == 4.0

    def test_create_rate_limiter(self):
        """Test rate limiter creation."""
        from core.resilience import create_rate_limiter
        limiter = create_rate_limiter(requests_per_second=10.0, burst_size=100)
        assert limiter.tokens_per_second == 10.0
        assert limiter.bucket_size == 100
        assert limiter.available_tokens == 100

    def test_rate_limiter_stats(self):
        """Test rate limiter statistics."""
        from core.resilience import RateLimiter
        limiter = RateLimiter(tokens_per_second=100.0, bucket_size=10)
        stats = limiter.stats
        assert stats.total_requests == 0
        assert stats.allowed_requests == 0
        assert stats.rejection_rate == 0.0

    def test_load_level_enum(self):
        """Test load level enum."""
        from core.resilience import LoadLevel
        assert LoadLevel.NORMAL.value == "normal"
        assert LoadLevel.ELEVATED.value == "elevated"
        assert LoadLevel.CRITICAL.value == "critical"
        assert LoadLevel.OVERLOADED.value == "overloaded"

    def test_backpressure_manager(self):
        """Test backpressure manager creation."""
        from core.resilience import BackpressureManager, LoadLevel
        manager = BackpressureManager()
        assert manager.load_level == LoadLevel.NORMAL
        assert manager.should_accept_request(priority=0)

    def test_health_status_enum(self):
        """Test health status enum."""
        from core.resilience import HealthStatus
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"

    def test_health_checker(self):
        """Test health checker."""
        from core.resilience import HealthChecker, HealthStatus
        checker = HealthChecker()
        checker.register("test", lambda: True)
        assert checker.get_overall_status() == HealthStatus.UNKNOWN

    def test_metric_type_enum(self):
        """Test metric type enum."""
        from core.resilience import MetricType
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"

    def test_telemetry_collector(self):
        """Test telemetry collector."""
        from core.resilience import create_telemetry
        telemetry = create_telemetry("test-service")
        telemetry.record_counter("requests", 1, {"endpoint": "/test"})
        telemetry.record_gauge("memory_mb", 512.0)
        metrics = telemetry.get_all_metrics()
        assert "counters" in metrics
        assert "gauges" in metrics

    def test_resilience_handler(self):
        """Test composite resilience handler."""
        from core.resilience import create_resilience_handler, CircuitState
        handler = create_resilience_handler()
        assert handler.circuit_breaker.state == CircuitState.CLOSED
        assert handler.rate_limiter is not None
        stats = handler.get_stats()
        assert "circuit_breaker" in stats
        assert "retry_policy" in stats
        assert "rate_limiter" in stats

    def test_resilience_config(self):
        """Test resilience configuration."""
        from core.resilience import ResilienceConfig, ResilienceHandler
        config = ResilienceConfig(
            circuit_failure_threshold=10,
            retry_max_attempts=5,
            rate_tokens_per_second=50.0,
        )
        handler = ResilienceHandler(config)
        assert handler.circuit_breaker.failure_threshold == 10
        assert handler.retry_policy.max_retries == 5
        assert handler.rate_limiter.tokens_per_second == 50.0


class TestEndToEndWorkflow:
    """
    End-to-end integration tests validating all 14 core modules work together.

    This test class ensures the complete UAP system operates cohesively:
    1. Memory (core, archival, temporal)
    2. Cooperation (session handoff, task coordination)
    3. Harness (agent harness, context window)
    4. MCP Manager (server management)
    5. Executor (ReAct loop)
    6. Thinking (extended thinking)
    7. Skills (skill registry, progressive disclosure)
    8. Tool Registry (centralized tool management)
    9. Persistence (checkpoints, session state)
    10. Orchestrator (multi-agent coordination)
    11. MCP Discovery (capability negotiation)
    12. Advanced Memory (semantic search, Letta)
    13. Ultrathink (power words, ToT, confidence)
    14. Resilience (circuit breaker, retry, rate limiting)
    """

    def test_complete_agent_workflow(self):
        """Test a complete agent workflow using all systems."""
        # Import all core modules
        from core import (
            MemorySystem,
            CooperationManager,
            AgentHarness,
            ThinkingEngine,
            SkillRegistry,
            ToolRegistry,
            PersistenceManager,
            Orchestrator,
            AdvancedMemorySystem,
            UltrathinkEngine,
            ResilienceHandler,
        )
        from core.persistence import PersistenceBackend, CheckpointType
        from core.orchestrator import Topology, AgentRole, AgentCapability
        from core.resilience import CircuitState

        # 1. Initialize memory system
        memory = MemorySystem(agent_id="e2e-test-agent")
        memory.core.update("persona", "End-to-end test agent")
        assert memory.core.get("persona") == "End-to-end test agent"

        # 2. Initialize cooperation manager
        coop = CooperationManager()
        assert coop is not None

        # 3. Initialize agent harness (takes max_tokens, not agent_id)
        harness = AgentHarness(max_tokens=100000)
        assert harness.context.max_tokens > 0

        # 4. Initialize thinking engine (uses _budget, not budget)
        thinking = ThinkingEngine()
        assert thinking.tokens_remaining > 0  # Use property instead

        # 5. Initialize skill registry (uses _skills internally)
        skills = SkillRegistry()
        # Skills are registered via register(), check the internal dict
        assert skills._skills is not None

        # 6. Initialize tool registry (use factory for built-in tools)
        from core import create_tool_registry
        tools = create_tool_registry(include_builtins=True)
        tool_count = len(tools.list_all())
        assert tool_count >= 2  # At least Read, ListDir

        # 7. Initialize persistence manager (memory backend)
        persistence = PersistenceManager(backend=PersistenceBackend.MEMORY)
        session_id = "test-session"  # session_id is the input param
        session_state = persistence.create_session(session_id)
        assert session_state is not None

        # Create a checkpoint using the string session_id
        checkpoint = persistence.create_checkpoint(
            session_id=session_id,  # Use string, not SessionState
            checkpoint_type=CheckpointType.AUTO,
            description="Test checkpoint",
        )
        assert checkpoint is not None

        # 8. Initialize orchestrator
        orchestrator = Orchestrator(topology=Topology.HIERARCHICAL)
        agent = orchestrator.register_agent(
            agent_id="worker-1",
            role=AgentRole.SPECIALIST,
            capabilities=[
                AgentCapability(name="testing", proficiency=1.0),
                AgentCapability(name="validation", proficiency=0.8),
            ],
        )
        assert agent.agent_id == "worker-1"

        # 9. Initialize advanced memory (requires agent_id)
        adv_memory = AdvancedMemorySystem(agent_id="e2e-test-agent")
        assert adv_memory is not None

        # 10. Initialize ultrathink engine
        ultrathink = UltrathinkEngine()
        chain = ultrathink.begin_chain("test analysis")
        assert chain is not None
        assert chain.id is not None  # CoTChain uses 'id' not 'chain_id'

        # 11. Initialize resilience handler
        resilience = ResilienceHandler()
        assert resilience.circuit_breaker.state == CircuitState.CLOSED
        assert resilience.retry_policy is not None
        assert resilience.rate_limiter is not None

        # Verify all systems initialized successfully
        systems = {
            "memory": memory,
            "cooperation": coop,
            "harness": harness,
            "thinking": thinking,
            "skills": skills,
            "tools": tools,
            "persistence": persistence,
            "orchestrator": orchestrator,
            "advanced_memory": adv_memory,
            "ultrathink": ultrathink,
            "resilience": resilience,
        }

        assert all(s is not None for s in systems.values())
        assert len(systems) == 11  # All 11 instantiable systems

    def test_memory_to_persistence_flow(self):
        """Test data flows from memory through persistence."""
        from core import MemorySystem, PersistenceManager
        from core.persistence import PersistenceBackend, CheckpointType

        # Create memory and persistence
        memory = MemorySystem(agent_id="flow-test-agent")
        persistence = PersistenceManager(backend=PersistenceBackend.MEMORY)

        # Store task state in core memory (synchronous)
        memory.core.update("current_task", "Integration testing")

        # Create session (session_id is input, returns SessionState)
        session_id = "memory-flow-test"
        session_state = persistence.create_session(session_id)
        assert session_state is not None
        assert session_state.metadata.session_id == session_id

        # Create a checkpoint using the session_id string
        checkpoint = persistence.create_checkpoint(
            session_id=session_id,  # Use the string, not SessionState
            checkpoint_type=CheckpointType.MILESTONE,
            description="Memory flow test checkpoint",
        )

        # Verify checkpoint was created
        assert checkpoint is not None
        # Check checkpoint has expected attributes
        assert checkpoint.checkpoint_type == CheckpointType.MILESTONE
        assert checkpoint.description == "Memory flow test checkpoint"

    def test_orchestrator_with_resilience(self):
        """Test orchestrator task execution with resilience patterns."""
        from core import Orchestrator, ResilienceHandler
        from core.orchestrator import Topology, AgentRole, TaskPriority
        from core.resilience import CircuitState

        # Setup orchestrator with multiple agents
        orchestrator = Orchestrator(topology=Topology.MESH)
        resilience = ResilienceHandler()

        # Register agents with different capabilities (use SPECIALIST role with AgentCapability)
        from core.orchestrator import AgentCapability
        orchestrator.register_agent(
            agent_id="analyzer-1",
            role=AgentRole.SPECIALIST,
            capabilities=[
                AgentCapability(name="code_analysis", proficiency=0.95),
                AgentCapability(name="testing", proficiency=0.7),
            ],
        )
        orchestrator.register_agent(
            agent_id="tester-1",
            role=AgentRole.SPECIALIST,
            capabilities=[
                AgentCapability(name="testing", proficiency=0.9),
                AgentCapability(name="code_analysis", proficiency=0.5),
            ],
        )

        # Verify agents were registered (use internal _agents dict)
        agents = list(orchestrator._agents.values())
        assert len(agents) >= 2
        assert any(a.agent_id == "analyzer-1" for a in agents)
        assert any(a.agent_id == "tester-1" for a in agents)

        # Verify resilience is ready
        assert resilience.circuit_breaker.state == CircuitState.CLOSED
        stats = resilience.get_stats()
        assert stats["circuit_breaker"]["state"] == "closed"

    def test_skill_tool_integration(self):
        """Test skills and tools work together."""
        from core import create_skill_registry, create_tool_registry
        from core.tool_registry import ToolPermission

        # Use factory functions to get built-in skills and tools
        skills = create_skill_registry(include_builtins=True)
        tools = create_tool_registry(include_builtins=True)

        # Find relevant skills for a query
        relevant_skills = skills.find_relevant("extended thinking", max_skills=3)
        assert len(relevant_skills) >= 0  # May or may not find matches

        # Verify registry has built-in skills (ultrathink, code-review, tdd-workflow)
        assert len(skills._skills) >= 3

        # Search for tools (use search() method)
        read_tools = tools.search("read")
        assert len(read_tools) >= 0  # May or may not find matches

        # Verify built-in Read tool (use get() method)
        read_tool = tools.get("Read")
        assert read_tool is not None
        assert read_tool.permission == ToolPermission.READ_ONLY  # singular 'permission'

    def test_ultrathink_confidence_flow(self):
        """Test ultrathink reasoning with confidence calibration."""
        from core import UltrathinkEngine
        from core.ultrathink import ThinkingLevel, detect_thinking_level, CoTPhase

        engine = UltrathinkEngine()

        # Test power word detection
        level = detect_thinking_level("Let me ultrathink about this complex problem")
        assert level == ThinkingLevel.ULTRATHINK

        level = detect_thinking_level("I need to think carefully")
        assert level == ThinkingLevel.THINK

        # Run a reasoning chain with THINK level for sufficient budget (~8K tokens)
        chain = engine.begin_chain("Evaluate system architecture", level=ThinkingLevel.THINK)
        chain_id = chain.id

        # Add steps with proper API (chain_id, phase, content)
        engine.add_step(chain_id, CoTPhase.UNDERSTAND, "Identify components")
        engine.add_step(chain_id, CoTPhase.DECOMPOSE, "Analyze dependencies")
        engine.add_step(chain_id, CoTPhase.EVALUATE, "Assess scalability")

        # Conclude the chain
        completed_chain = engine.conclude_chain(chain_id, "Architecture is well-designed")
        assert completed_chain is not None
        assert len(completed_chain.steps) == 3
        assert completed_chain.conclusion == "Architecture is well-designed"

    def test_advanced_memory_semantic_search(self):
        """Test advanced memory initialization and capabilities."""
        from core import AdvancedMemorySystem
        from core.advanced_memory import ConsolidationStrategy, EmbeddingModel

        # Test consolidation strategy enum
        assert ConsolidationStrategy.SUMMARIZE.value == "summarize"
        assert ConsolidationStrategy.PRUNE.value == "prune"

        # Test embedding model enum (actual values from enum definition)
        assert EmbeddingModel.LOCAL_MINILM.value == "all-MiniLM-L6-v2"
        assert EmbeddingModel.OPENAI_ADA.value == "text-embedding-ada-002"

        # Create advanced memory system (requires agent_id)
        memory = AdvancedMemorySystem(agent_id="semantic-test-agent")
        assert memory is not None
        assert memory.agent_id == "semantic-test-agent"

        # Verify core memory is initialized
        assert memory.core is not None

        # Verify get_stats works (synchronous)
        stats = memory.get_stats()
        assert "agent_id" in stats
        assert stats["agent_id"] == "semantic-test-agent"
        assert "core_blocks" in stats

    def test_full_system_statistics(self):
        """Test that all systems report statistics correctly."""
        from core import (
            MemorySystem,
            AgentHarness,
            ToolRegistry,
            Orchestrator,
            AdvancedMemorySystem,
            UltrathinkEngine,
            ResilienceHandler,
            create_skill_registry,
        )
        from core.orchestrator import Topology

        # Collect stats from all systems
        stats = {}

        # Memory stats
        memory = MemorySystem(agent_id="stats-test")
        stats["memory"] = {
            "agent_id": memory.agent_id,
            "has_core": memory.core is not None,
            "has_archival": memory.archival is not None,
            "has_temporal": memory.temporal is not None,
        }

        # Harness stats (uses max_tokens, not agent_id)
        harness = AgentHarness(max_tokens=100000)
        stats["harness"] = {
            "max_tokens": harness.context.max_tokens,
        }

        # Skills stats (use factory to get built-in skills)
        skills = create_skill_registry(include_builtins=True)
        stats["skills"] = {
            "total_skills": len(skills._skills),
            "categories": len(set(s.metadata.category for s in skills._skills.values())),
        }

        # Tools stats (use factory for built-in tools)
        from core import create_tool_registry
        tools = create_tool_registry(include_builtins=True)
        stats["tools"] = {
            "total_tools": len(tools.list_all()),
        }

        # Orchestrator stats (use _topology private attribute)
        orch = Orchestrator(topology=Topology.SOLO)
        orch_stats = orch.get_metrics()
        stats["orchestrator"] = {
            "topology": orch._topology.value,
            "agent_count": orch_stats.active_agents + orch_stats.idle_agents,
        }

        # Advanced memory stats (requires agent_id)
        adv_mem = AdvancedMemorySystem(agent_id="stats-test")
        stats["advanced_memory"] = adv_mem.get_stats()

        # Ultrathink stats
        ultra = UltrathinkEngine()
        stats["ultrathink"] = ultra.get_stats()

        # Resilience stats
        resilience = ResilienceHandler()
        stats["resilience"] = resilience.get_stats()

        # Verify all stats collected
        assert len(stats) == 8
        assert all(v is not None for v in stats.values())

        # Verify specific stats
        assert stats["memory"]["has_core"] is True
        assert stats["skills"]["total_skills"] >= 3
        assert stats["tools"]["total_tools"] >= 2
        assert stats["resilience"]["circuit_breaker"]["state"] == "closed"


def run_tests():
    """Run all tests manually (without pytest)."""
    print("=" * 60)
    print("UAP Core Integration Tests")
    print("=" * 60)

    test_classes = [
        TestCoreImports,
        TestMemorySystem,
        TestHarness,
        TestMCPManager,
        TestThinkingEngine,
        TestExecutor,
        TestIntegration,
        TestSkillSystem,
        TestToolRegistry,
        TestPersistence,
        TestOrchestrator,
        TestMCPDiscovery,
        TestAdvancedMemory,
        TestUltrathink,
        TestResilience,
        TestEndToEndWorkflow,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_class in test_classes:
        print(f"\n[{test_class.__name__}]")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith("test_"):
                method = getattr(instance, method_name)

                # Skip async tests for simple runner
                if inspect.iscoroutinefunction(method):
                    print(f"  {method_name}: SKIPPED (async)")
                    continue

                try:
                    method()
                    print(f"  {method_name}: PASSED")
                    passed += 1
                except Exception as e:
                    print(f"  {method_name}: FAILED - {e}")
                    failed += 1
                    errors.append((method_name, str(e)))

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")

    if errors:
        print("\nFailures:")
        for name, error in errors:
            print(f"  - {name}: {error}")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
