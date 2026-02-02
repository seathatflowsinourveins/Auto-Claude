"""
Platform Orchestrator V2 - Enhanced Multi-Agent Architecture
=============================================================

Major upgrades from V1:
1. Intelligent Model Routing - 3-tier system (LOCAL/HYBRID/PREMIUM)
2. Agent Mesh Communication - Direct pub/sub without bottleneck
3. Speculative Execution - Pre-compute likely paths
4. Connection Pre-warming - Reduce latency spikes
5. Quality Gates - Automatic tier escalation
6. Cost Tracking - Real-time budget monitoring

Architecture:
    INPUT (query/task)
        |
    MODEL ROUTER (classify task → select tier)
        |
    ┌──────────────────────────────────────┐
    │          AGENT MESH                   │
    │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  │
    │  │Agent│──│Agent│──│Agent│──│Agent│  │
    │  │ 1   │  │ 2   │  │ 3   │  │ 4   │  │
    │  └─────┘  └─────┘  └─────┘  └─────┘  │
    │     │         │         │       │     │
    │     └─────────┴─────────┴───────┘     │
    │              Redis Pub/Sub            │
    └──────────────────────────────────────┘
        |
    SPECULATIVE EXECUTION (parallel paths)
        |
    QUALITY GATE (escalate if needed)
        |
    VERIFICATION (6-phase)
        |
    OUTPUT (verified result)

Version: 2.0.0 (2026-01-30)
Philosophy: Performance-first, cost-aware, production-grade
"""

from __future__ import annotations

import asyncio
import time
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

# Add parent directories to path for imports
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

# Try to import from integrations directory
_integrations_dir = Path.home() / ".claude" / "integrations"
if _integrations_dir.exists() and str(_integrations_dir) not in sys.path:
    sys.path.insert(0, str(_integrations_dir))


def _safe_import(module_name: str, names: list[str], fallback_path: Optional[Path] = None):  # noqa: ARG001
    """Safely import with fallback."""
    result = {}
    try:
        mod = __import__(module_name, fromlist=names)
        for name in names:
            result[name] = getattr(mod, name, None)
    except ImportError:
        for name in names:
            result[name] = None
    return result


# Import V1 components for backward compatibility
_v1 = _safe_import("platform_orchestrator", [
    "PlatformOrchestrator", "PlatformState", "PlatformResult",
    "PlatformPhase", "TaskType", "MCPExecutor", "mock_mcp_executor"
])
PlatformOrchestratorV1 = _v1.get("PlatformOrchestrator")
PlatformState = _v1.get("PlatformState")
PlatformResult = _v1.get("PlatformResult")
TaskType = _v1.get("TaskType")
MCPExecutor = _v1.get("MCPExecutor")
mock_mcp_executor = _v1.get("mock_mcp_executor")

# Import new V2 components
_router = _safe_import("model_router", [
    "ModelRouter", "ModelTier", "TaskComplexity",
    "TaskDomain", "RoutingDecision", "OllamaClient"
])
ModelRouter = _router.get("ModelRouter")
ModelTier = _router.get("ModelTier")
TaskComplexity = _router.get("TaskComplexity")
TaskDomain = _router.get("TaskDomain")
RoutingDecision = _router.get("RoutingDecision")
OllamaClient = _router.get("OllamaClient")

_mesh = _safe_import("agent_mesh", [
    "AgentMesh", "AgentRole", "MessageType", "MeshMessage", "TaskState"
])
AgentMesh = _mesh.get("AgentMesh")
AgentRole = _mesh.get("AgentRole")
MessageType = _mesh.get("MessageType")
MeshMessage = _mesh.get("MeshMessage")
TaskState = _mesh.get("TaskState")

# Import Token Optimizer (new in V2.1)
_token_opt = _safe_import("token_optimizer", [
    "TokenOptimizer", "CompressionLevel", "CompressionStrategy",
    "ContextCompressor", "PromptCache", "ResponseStreamer",
    "BatchProcessor", "CostTracker"
])
TokenOptimizer = _token_opt.get("TokenOptimizer")
CompressionLevel = _token_opt.get("CompressionLevel")
CompressionStrategy = _token_opt.get("CompressionStrategy")
CostTrackerV2 = _token_opt.get("CostTracker")  # Renamed to avoid conflict


# =============================================================================
# Fallback Type Definitions (when imports fail)
# =============================================================================

class _ModelTierFallback(Enum):
    """Fallback ModelTier if import fails."""
    LOCAL = "local"
    HYBRID = "hybrid"
    PREMIUM = "premium"


class _TaskTypeFallback(Enum):
    """Fallback TaskType if import fails."""
    RESEARCH = "research"
    IMPLEMENT = "implement"
    DECIDE = "decide"
    FULL = "full"
    VERIFY_ONLY = "verify"
    QUICK_LOOKUP = "quick"


class _AgentRoleFallback(Enum):
    """Fallback AgentRole if import fails."""
    ROUTER = "router"
    CODER = "coder"
    REVIEWER = "reviewer"
    RESEARCHER = "researcher"
    TESTER = "tester"


class _CompressionLevelFallback(Enum):
    """Fallback CompressionLevel if import fails."""
    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"


# Use fallbacks if imports failed - ensure these are always available
ModelTier = ModelTier if ModelTier is not None else _ModelTierFallback  # type: ignore[misc]
TaskType = TaskType if TaskType is not None else _TaskTypeFallback  # type: ignore[misc]
AgentRole = AgentRole if AgentRole is not None else _AgentRoleFallback  # type: ignore[misc]
CompressionLevel = CompressionLevel if CompressionLevel is not None else _CompressionLevelFallback  # type: ignore[misc]

# Re-export to ensure type checker knows they're available
ModelTier = ModelTier  # type: ignore[has-type]
TaskType = TaskType  # type: ignore[has-type]
AgentRole = AgentRole  # type: ignore[has-type]
CompressionLevel = CompressionLevel  # type: ignore[has-type]

logger = logging.getLogger(__name__)


# =============================================================================
# V2 Enums and Types
# =============================================================================

class ExecutionMode(Enum):
    """Execution modes for the V2 orchestrator."""
    STANDARD = "standard"           # Normal execution
    SPECULATIVE = "speculative"     # Pre-compute multiple paths
    PARALLEL_AGENTS = "parallel"    # Multiple agents work concurrently
    CASCADE = "cascade"             # Escalate through tiers
    FASTEST = "fastest"             # Race multiple approaches


class QualityLevel(Enum):
    """Quality thresholds for responses."""
    DRAFT = 0.6          # Acceptable for internal use
    STANDARD = 0.75      # Normal production threshold
    HIGH = 0.85          # Important decisions
    CRITICAL = 0.95      # Security, financial, compliance


class CostTier(Enum):
    """Cost awareness tiers."""
    MINIMAL = "minimal"      # Use LOCAL only unless critical
    BALANCED = "balanced"    # Mix of tiers based on task
    PERFORMANCE = "performance"  # Use best model for task
    UNLIMITED = "unlimited"  # Always use premium


# =============================================================================
# V2 State Models
# =============================================================================

@dataclass
class V2ExecutionState:
    """
    Extended execution state for V2 orchestrator.

    Tracks model routing, agent mesh activity, and cost metrics.
    """
    # Inherited from PlatformState (using Any for dynamic import)
    base_state: Any  # PlatformState

    # Model routing
    routing_decision: Any = None  # RoutingDecision
    model_used: str = ""
    tier_used: Any = None  # ModelTier - set in __post_init__

    def __post_init__(self) -> None:
        if self.tier_used is None and ModelTier is not None:
            self.tier_used = ModelTier.HYBRID

    # Agent mesh
    mesh_agents_used: list[str] = field(default_factory=list)
    mesh_messages_sent: int = 0
    mesh_context_shared: int = 0

    # Speculative execution
    speculative_paths: int = 0
    speculative_hits: int = 0

    # Quality tracking
    initial_quality: float = 0.0
    final_quality: float = 0.0
    quality_gate_passed: bool = False
    escalations: int = 0

    # Cost tracking
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0

    # Token optimization tracking (V2.1)
    tokens_before_compression: int = 0
    tokens_after_compression: int = 0
    compression_ratio: float = 1.0
    cache_hit: bool = False
    cache_key: str = ""

    # Timing
    model_routing_ms: float = 0.0
    mesh_setup_ms: float = 0.0
    execution_ms: float = 0.0
    quality_check_ms: float = 0.0
    compression_ms: float = 0.0


@dataclass
class CostMetrics:
    """Real-time cost tracking."""
    total_spent_today: float = 0.0
    daily_budget: float = 50.0
    tier_breakdown: dict[str, float] = field(default_factory=dict)
    model_breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def remaining_budget(self) -> float:
        return self.daily_budget - self.total_spent_today

    @property
    def budget_utilization(self) -> float:
        return self.total_spent_today / self.daily_budget if self.daily_budget > 0 else 1.0

    def should_prefer_local(self) -> bool:
        """Check if we should prefer local models due to budget."""
        return self.budget_utilization >= 0.8


# =============================================================================
# Speculative Execution Engine
# =============================================================================

class SpeculativeExecutor:
    """
    Pre-computes likely execution paths to reduce latency.

    Key Insight: While waiting for user input or between phases,
    speculatively execute likely next steps in parallel.
    """

    def __init__(self, max_speculative_paths: int = 3):
        self.max_paths = max_speculative_paths
        self._active_speculations: dict[str, asyncio.Task] = {}
        self._speculation_results: dict[str, Any] = {}
        self._hit_rate_history: list[bool] = []

    async def speculate(
        self,
        paths: Sequence[tuple[str, Callable[[], Any]]],
    ) -> dict[str, asyncio.Task[Any]]:
        """
        Start speculative execution of multiple paths.

        Args:
            paths: List of (path_id, coroutine_factory) tuples

        Returns:
            Dict of path_id -> Task
        """
        tasks: dict[str, asyncio.Task[Any]] = {}
        for path_id, coro_factory in list(paths)[:self.max_paths]:
            coro = coro_factory()
            # Ensure we have a proper coroutine
            if asyncio.iscoroutine(coro):
                task = asyncio.create_task(coro)
                tasks[path_id] = task
                self._active_speculations[path_id] = task
        return tasks

    async def await_path(self, path_id: str, timeout: float = 30.0) -> Optional[Any]:
        """Get result from a speculative path."""
        if path_id in self._speculation_results:
            self._hit_rate_history.append(True)
            return self._speculation_results[path_id]

        task = self._active_speculations.get(path_id)
        if task:
            try:
                result = await asyncio.wait_for(task, timeout=timeout)
                self._speculation_results[path_id] = result
                self._hit_rate_history.append(True)
                return result
            except asyncio.TimeoutError:
                self._hit_rate_history.append(False)
                return None

        self._hit_rate_history.append(False)
        return None

    def cancel_unused(self, used_path: str) -> None:
        """Cancel all speculative tasks except the one used."""
        for path_id, task in list(self._active_speculations.items()):
            if path_id != used_path and not task.done():
                task.cancel()
            del self._active_speculations[path_id]

    @property
    def hit_rate(self) -> float:
        """Calculate speculation hit rate."""
        if not self._hit_rate_history:
            return 0.0
        return sum(self._hit_rate_history) / len(self._hit_rate_history)


# =============================================================================
# Quality Gate System
# =============================================================================

class QualityGate:
    """
    Automatic quality assessment and tier escalation.

    If response quality falls below threshold, automatically
    escalate to a higher-tier model and retry.
    """

    def __init__(
        self,
        min_quality: QualityLevel = QualityLevel.STANDARD,
        max_escalations: int = 2,
    ):
        self.min_quality = min_quality
        self.max_escalations = max_escalations
        self._escalation_history: list[tuple[str, str, float]] = []

    def assess_quality(
        self,
        response: str,
        task: str,
        confidence: float = 0.0,
    ) -> tuple[float, list[str]]:
        """
        Assess response quality using multiple heuristics.

        Returns (quality_score, issues_found)
        """
        issues = []
        score = 1.0

        # Check response length
        if len(response) < 50:
            issues.append("Response too short")
            score -= 0.2

        # Check for error indicators
        error_markers = ["i don't know", "i cannot", "error:", "failed to"]
        for marker in error_markers:
            if marker in response.lower():
                issues.append(f"Contains error marker: {marker}")
                score -= 0.15

        # Check for hallucination indicators
        hallucination_markers = ["as an ai", "i'm not able to", "hypothetically"]
        for marker in hallucination_markers:
            if marker in response.lower():
                issues.append(f"Possible hallucination: {marker}")
                score -= 0.1

        # Use provided confidence
        if confidence > 0:
            score = min(score, confidence)

        # Check relevance (simple keyword match)
        task_words = set(task.lower().split())
        response_words = set(response.lower().split())
        overlap = len(task_words & response_words) / len(task_words) if task_words else 0
        if overlap < 0.2:
            issues.append("Low relevance to task")
            score -= 0.2

        return max(0.0, min(1.0, score)), issues

    def should_escalate(
        self,
        quality_score: float,
        current_tier: Any,  # ModelTier
        escalation_count: int,
    ) -> tuple[bool, Any]:  # tuple[bool, Optional[ModelTier]]
        """
        Determine if we should escalate to a higher tier.

        Returns (should_escalate, next_tier)
        """
        if quality_score >= self.min_quality.value:
            return False, None

        if escalation_count >= self.max_escalations:
            return False, None

        # Determine next tier (handle dynamic imports)
        if ModelTier is None:
            return False, None

        tier_order = [ModelTier.LOCAL, ModelTier.HYBRID, ModelTier.PREMIUM]
        try:
            current_idx = tier_order.index(current_tier)
            if current_idx < len(tier_order) - 1:
                next_tier = tier_order[current_idx + 1]
                self._escalation_history.append((
                    str(current_tier),
                    str(next_tier),
                    quality_score,
                ))
                return True, next_tier
        except ValueError:
            pass

        return False, None


# =============================================================================
# Connection Pre-Warmer
# =============================================================================

class ConnectionPrewarmer:
    """
    Pre-warms connections to reduce latency spikes.

    Maintains warm connections to frequently-used services.
    """

    def __init__(self):
        self._warm_connections: dict[str, datetime] = {}
        self._warmup_interval = 300  # 5 minutes
        self._background_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start background warmup loop."""
        self._background_task = asyncio.create_task(self._warmup_loop())

    async def stop(self) -> None:
        """Stop background warmup loop."""
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass

    async def _warmup_loop(self) -> None:
        """Background loop to keep connections warm."""
        while True:
            try:
                await self._warmup_all()
                await asyncio.sleep(self._warmup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Warmup error: {e}")
                await asyncio.sleep(60)

    async def _warmup_all(self) -> None:
        """Warm up all known connections."""
        # Warm Ollama (local models)
        if OllamaClient is not None:
            try:
                client = OllamaClient()
                await client.warmup()
                self._warm_connections["ollama"] = datetime.now(timezone.utc)
            except Exception as e:
                logger.debug(f"Ollama warmup failed: {e}")

        # Add more services as needed

    def is_warm(self, service: str) -> bool:
        """Check if a connection is warm."""
        if service not in self._warm_connections:
            return False
        age = (datetime.now(timezone.utc) - self._warm_connections[service]).total_seconds()
        return age < self._warmup_interval


# =============================================================================
# Platform Orchestrator V2
# =============================================================================

class PlatformOrchestratorV2:
    """
    Enhanced platform orchestrator with intelligent model routing
    and agent mesh communication.

    Key Improvements:
    1. 3-tier model routing for cost optimization
    2. Direct agent-to-agent communication via mesh
    3. Speculative execution for reduced latency
    4. Automatic quality gates with escalation
    5. Real-time cost tracking

    Usage:
        orchestrator = PlatformOrchestratorV2(
            cost_tier=CostTier.BALANCED,
            min_quality=QualityLevel.STANDARD,
        )

        result = await orchestrator.execute(
            "Implement authentication",
            TaskType.IMPLEMENT,
            execution_mode=ExecutionMode.SPECULATIVE,
        )
    """

    def __init__(
        self,
        mcp_executor: Any = None,  # MCPExecutor | None
        project_id: str = "default",
        cost_tier: CostTier = CostTier.BALANCED,
        min_quality: QualityLevel = QualityLevel.STANDARD,
        daily_budget: float = 50.0,
        enable_speculation: bool = True,
        enable_mesh: bool = True,
        enable_prewarm: bool = True,
        enable_token_optimization: bool = True,
        compression_level: Any = None,  # CompressionLevel - set in body
    ):
        """
        Initialize V2 orchestrator.

        Args:
            mcp_executor: MCP tool executor
            project_id: Project identifier
            cost_tier: Cost awareness level
            min_quality: Minimum acceptable quality
            daily_budget: Daily spending limit
            enable_speculation: Enable speculative execution
            enable_mesh: Enable agent mesh
            enable_prewarm: Enable connection pre-warming
            enable_token_optimization: Enable token optimization (compression, caching)
            compression_level: Default compression level for prompts
        """
        # V1 base (for backward compatibility)
        self._v1: Any = None
        if PlatformOrchestratorV1 is not None:
            self._v1 = PlatformOrchestratorV1(
                mcp_executor=mcp_executor,
                project_id=project_id,
            )

        # V2 components
        self._model_router: Any = None
        if ModelRouter is not None:
            self._model_router = ModelRouter()
        self._agent_mesh: Any = None
        if enable_mesh and AgentMesh is not None:
            self._agent_mesh = AgentMesh()
        self._speculative = SpeculativeExecutor() if enable_speculation else None
        self._quality_gate = QualityGate(min_quality=min_quality)
        self._prewarmer = ConnectionPrewarmer() if enable_prewarm else None

        # Token Optimizer (V2.1) - compression, caching, cost tracking
        self._token_optimizer: Any = None
        if enable_token_optimization and TokenOptimizer is not None:
            self._token_optimizer = TokenOptimizer(
                daily_budget=daily_budget,
                cache_size=500,
                batch_size=5,
                enable_streaming=True,
            )

        # Default compression level based on cost tier
        if compression_level is None and CompressionLevel is not None:
            if cost_tier == CostTier.MINIMAL:
                compression_level = CompressionLevel.AGGRESSIVE
            elif cost_tier == CostTier.BALANCED:
                compression_level = CompressionLevel.MODERATE
            elif cost_tier == CostTier.PERFORMANCE:
                compression_level = CompressionLevel.LIGHT
            else:  # UNLIMITED
                compression_level = CompressionLevel.NONE
        self._compression_level = compression_level

        # Configuration
        self._cost_tier = cost_tier
        self._min_quality = min_quality
        self._daily_budget = daily_budget

        # Cost tracking
        self._cost_metrics = CostMetrics(daily_budget=daily_budget)

        # V2 statistics
        self._v2_stats = {
            "total_executions": 0,
            "local_executions": 0,
            "hybrid_executions": 0,
            "premium_executions": 0,
            "escalations": 0,
            "speculation_hits": 0,
            "mesh_messages": 0,
            "quality_gate_passes": 0,
            "quality_gate_failures": 0,
            # Token optimization stats (V2.1)
            "cache_hits": 0,
            "cache_misses": 0,
            "tokens_saved": 0,
            "compression_ratio_avg": 1.0,
        }

        # Background tasks
        self._started = False

    async def start(self) -> None:
        """Start background services."""
        if self._started:
            return

        if self._prewarmer:
            await self._prewarmer.start()

        if self._agent_mesh:
            await self._agent_mesh.start()
            # Register default agents
            await self._register_default_agents()

        self._started = True
        logger.info("PlatformOrchestratorV2 started")

    async def stop(self) -> None:
        """Stop background services."""
        if not self._started:
            return

        if self._prewarmer:
            await self._prewarmer.stop()

        if self._agent_mesh:
            await self._agent_mesh.stop()

        self._started = False
        logger.info("PlatformOrchestratorV2 stopped")

    async def _register_default_agents(self) -> None:
        """Register default agents in the mesh."""
        if not self._agent_mesh or AgentRole is None:
            return

        # Router agent (LOCAL tier - fast classification)
        await self._agent_mesh.register_agent(
            role=AgentRole.ROUTER,
            model_tier="local",
            capabilities=["task_classification", "routing"],
        )

        # Coder agent (HYBRID tier - main development)
        await self._agent_mesh.register_agent(
            role=AgentRole.CODER,
            model_tier="hybrid",
            capabilities=["code_generation", "debugging", "refactoring"],
        )

        # Reviewer agent (PREMIUM tier - quality assurance)
        await self._agent_mesh.register_agent(
            role=AgentRole.REVIEWER,
            model_tier="premium",
            capabilities=["code_review", "security_audit", "architecture"],
        )

        # Researcher agent (HYBRID tier - information gathering)
        await self._agent_mesh.register_agent(
            role=AgentRole.RESEARCHER,
            model_tier="hybrid",
            capabilities=["web_search", "documentation", "analysis"],
        )

    # =========================================================================
    # Main Execution Methods
    # =========================================================================

    async def execute(
        self,
        task: str,
        task_type: Any = None,  # TaskType - set default in body
        execution_mode: ExecutionMode = ExecutionMode.STANDARD,
        project_path: str | None = None,
        context: str | None = None,
        required_quality: QualityLevel | None = None,
    ) -> Any:  # PlatformResult
        """
        Execute a task with V2 enhancements.

        Args:
            task: The task/query to execute
            task_type: Type of task
            execution_mode: How to execute (standard, speculative, etc.)
            project_path: Path to project (for verification)
            context: Additional context
            required_quality: Override minimum quality

        Returns:
            PlatformResult with all outputs
        """
        if not self._started:
            await self.start()

        self._v2_stats["total_executions"] += 1
        start_time = time.time()

        # Set default task_type if None
        if task_type is None:
            task_type = TaskType.FULL  # type: ignore[union-attr]

        # Initialize V2 state (PlatformState may be None if import failed)
        base_state: Any = None
        if PlatformState is not None:
            base_state = PlatformState(query=task, task_type=task_type)  # type: ignore[misc]
        v2_state = V2ExecutionState(base_state=base_state)

        try:
            # Phase -1: Token Optimization (new in V2.1)
            optimized_task, v2_state = await self._phase_token_optimization(task, v2_state)

            # Phase 0: Model Routing (new in V2)
            v2_state = await self._phase_model_routing(v2_state, context)

            # Phase 1: Execute based on mode (use optimized task)
            task = optimized_task  # Use compressed version
            if execution_mode == ExecutionMode.SPECULATIVE:
                v2_state = await self._execute_speculative(v2_state, task_type, context)
            elif execution_mode == ExecutionMode.PARALLEL_AGENTS:
                v2_state = await self._execute_parallel_agents(v2_state, task_type, context)
            elif execution_mode == ExecutionMode.CASCADE:
                v2_state = await self._execute_cascade(v2_state, task_type, context)
            elif execution_mode == ExecutionMode.FASTEST:
                v2_state = await self._execute_fastest(v2_state, task_type, context)
            else:
                v2_state = await self._execute_standard(v2_state, task_type, context)

            # Phase 2: Quality Gate
            quality_level = required_quality or self._min_quality
            v2_state = await self._phase_quality_gate(v2_state, quality_level)

            # Phase 3: Verification (if needed)
            verify_types = ["implement", "full", "verify"]
            task_type_str = task_type.value if hasattr(task_type, 'value') else str(task_type)
            if task_type_str in verify_types:
                if project_path and self._v1 is not None:
                    result = await self._v1.verify(project_path)
                    v2_state.base_state.verification_state = result

            # Update costs
            self._update_cost_metrics(v2_state)

            # Build result
            v2_state.base_state.success = True
            v2_state.base_state.completed_at = datetime.now(timezone.utc)

        except Exception as e:
            v2_state.base_state.error = str(e)
            v2_state.base_state.success = False
            v2_state.base_state.completed_at = datetime.now(timezone.utc)
            logger.error(f"Execution failed: {e}")

        v2_state.execution_ms = (time.time() - start_time) * 1000
        return self._build_v2_result(v2_state)

    # =========================================================================
    # Phase Implementations
    # =========================================================================

    async def _phase_token_optimization(
        self,
        task: str,
        state: V2ExecutionState,
    ) -> tuple[str, V2ExecutionState]:
        """
        Optimize tokens before execution (V2.1).

        - Check cache for identical/similar prompts
        - Compress context if needed
        - Track token savings

        Returns:
            Tuple of (optimized_task, updated_state)
        """
        start_time = time.time()

        # If token optimizer not available, pass through
        if self._token_optimizer is None:
            return task, state

        # Record original token count (rough estimate: 4 chars per token)
        state.tokens_before_compression = len(task) // 4

        # Check cache first
        cache = self._token_optimizer.cache
        cached = cache.get(task)
        if cached is not None:
            state.cache_hit = True
            state.cache_key = cached.get("cache_key", "")
            self._v2_stats["cache_hits"] += 1
            state.compression_ms = (time.time() - start_time) * 1000
            # Return cached response info (the task itself doesn't change for cache)
            return task, state

        self._v2_stats["cache_misses"] += 1

        # Compress if we have a compression level set
        if self._compression_level is not None:
            compressor = self._token_optimizer.compressor
            compressed = compressor.compress(
                task,
                level=self._compression_level,
            )
            optimized_task = compressed["text"]
            state.tokens_after_compression = len(optimized_task) // 4
            state.compression_ratio = compressed["ratio"]

            # Track tokens saved
            tokens_saved = state.tokens_before_compression - state.tokens_after_compression
            self._v2_stats["tokens_saved"] += tokens_saved

            # Update running average of compression ratio
            total_execs = self._v2_stats["total_executions"] or 1
            prev_avg = self._v2_stats["compression_ratio_avg"]
            self._v2_stats["compression_ratio_avg"] = (
                (prev_avg * (total_execs - 1) + state.compression_ratio) / total_execs
            )
        else:
            optimized_task = task
            state.tokens_after_compression = state.tokens_before_compression
            state.compression_ratio = 1.0

        state.compression_ms = (time.time() - start_time) * 1000
        return optimized_task, state

    async def _phase_model_routing(
        self,
        state: V2ExecutionState,
        context: str | None,
    ) -> V2ExecutionState:
        """Route task to optimal model tier."""
        start_time = time.time()

        # Get routing decision
        decision = self._model_router.route(
            task=state.base_state.query,
            context=context,
            budget_utilization=self._cost_metrics.budget_utilization,
            prefer_performance=self._cost_tier == CostTier.PERFORMANCE,
        )

        state.routing_decision = decision
        state.tier_used = decision.tier
        state.model_used = decision.model_name

        # Track by tier
        tier_name = decision.tier.value
        self._v2_stats[f"{tier_name}_executions"] += 1

        state.model_routing_ms = (time.time() - start_time) * 1000
        logger.info(f"Routed to {decision.tier.value}: {decision.model_name}")

        return state

    async def _execute_standard(
        self,
        state: V2ExecutionState,
        task_type: Any,  # TaskType
        context: str | None,  # Reserved for future context-aware execution
    ) -> V2ExecutionState:
        """Standard execution using routed model."""
        _ = context  # Reserved for future use
        # Use V1 research with the routed model
        research_types = ["research", "implement", "decide", "full"]
        task_str = task_type.value if hasattr(task_type, 'value') else str(task_type)
        if task_str in research_types and self._v1 is not None:
            result = await self._v1.research(state.base_state.query)
            state.base_state.research_results = result
            state.initial_quality = result.confidence if result else 0.0

        return state

    async def _execute_speculative(
        self,
        state: V2ExecutionState,
        task_type: Any,  # TaskType
        context: str | None,
    ) -> V2ExecutionState:
        """Execute with speculative paths."""
        if not self._speculative:
            return await self._execute_standard(state, task_type, context)

        # Get tier values (handle potential None)
        local_tier = getattr(ModelTier, 'LOCAL', _ModelTierFallback.LOCAL) if ModelTier else _ModelTierFallback.LOCAL
        hybrid_tier = getattr(ModelTier, 'HYBRID', _ModelTierFallback.HYBRID) if ModelTier else _ModelTierFallback.HYBRID
        premium_tier = getattr(ModelTier, 'PREMIUM', _ModelTierFallback.PREMIUM) if ModelTier else _ModelTierFallback.PREMIUM

        # Define speculative paths
        paths: list[tuple[str, Callable[[], Any]]] = [
            ("local", lambda: self._execute_with_tier(state, local_tier, context)),
            ("hybrid", lambda: self._execute_with_tier(state, hybrid_tier, context)),
        ]

        # If high quality required, also speculate premium
        if self._min_quality.value >= QualityLevel.HIGH.value:
            paths.append(
                ("premium", lambda: self._execute_with_tier(state, premium_tier, context))
            )

        state.speculative_paths = len(paths)

        # Start speculation
        await self._speculative.speculate(paths)

        # Wait for the routed tier first
        tier_name = state.tier_used.value
        result = await self._speculative.await_path(tier_name, timeout=30.0)

        if result:
            state = result
            state.speculative_hits = 1
            self._v2_stats["speculation_hits"] += 1
        else:
            # Fall back to standard execution
            state = await self._execute_standard(state, task_type, context)

        # Cancel unused paths
        self._speculative.cancel_unused(tier_name)

        return state

    async def _execute_parallel_agents(
        self,
        state: V2ExecutionState,
        task_type: Any,  # TaskType
        context: str | None,
    ) -> V2ExecutionState:
        """Execute with multiple agents working in parallel."""
        if not self._agent_mesh:
            return await self._execute_standard(state, task_type, context)

        start_time = time.time()

        # Assign task to mesh
        task_id = f"task-{int(time.time() * 1000)}"

        # Determine preferred role based on task type (using string comparison)
        task_str = task_type.value if hasattr(task_type, 'value') else str(task_type)

        # Get role values safely (handle potential None)
        researcher_role = getattr(AgentRole, 'RESEARCHER', _AgentRoleFallback.RESEARCHER) if AgentRole else _AgentRoleFallback.RESEARCHER
        coder_role = getattr(AgentRole, 'CODER', _AgentRoleFallback.CODER) if AgentRole else _AgentRoleFallback.CODER
        reviewer_role = getattr(AgentRole, 'REVIEWER', _AgentRoleFallback.REVIEWER) if AgentRole else _AgentRoleFallback.REVIEWER

        role_mapping: dict[str, Any] = {
            "research": researcher_role,
            "implement": coder_role,
            "decide": reviewer_role,
            "full": coder_role,
            "verify": reviewer_role,
        }
        preferred_role = role_mapping.get(task_str, coder_role)

        # Share context with mesh
        if context:
            await self._agent_mesh.share_context(
                sender_id="orchestrator",
                context_key="task_context",
                context_value=context,
                recipients=None,  # Broadcast to all
            )
            state.mesh_context_shared += 1

        # Assign to agent
        agent_id = await self._agent_mesh.assign_task(
            task_id=task_id,
            description=state.base_state.query,
            preferred_role=preferred_role,
            complexity=state.routing_decision.complexity.value if state.routing_decision else "moderate",
        )

        if agent_id:
            state.mesh_agents_used.append(agent_id)

        state.mesh_setup_ms = (time.time() - start_time) * 1000
        self._v2_stats["mesh_messages"] += state.mesh_context_shared

        # Execute through V1 for actual work
        state = await self._execute_standard(state, task_type, context)

        return state

    async def _execute_cascade(
        self,
        state: V2ExecutionState,
        task_type: Any,  # TaskType - reserved for future tier hints
        context: str | None,
    ) -> V2ExecutionState:
        """Execute with tier escalation cascade."""
        _ = task_type  # Reserved for future tier hints based on task type
        # Get tier values safely (handle potential None)
        local_tier = getattr(ModelTier, 'LOCAL', _ModelTierFallback.LOCAL) if ModelTier else _ModelTierFallback.LOCAL
        hybrid_tier = getattr(ModelTier, 'HYBRID', _ModelTierFallback.HYBRID) if ModelTier else _ModelTierFallback.HYBRID
        premium_tier = getattr(ModelTier, 'PREMIUM', _ModelTierFallback.PREMIUM) if ModelTier else _ModelTierFallback.PREMIUM
        tiers = [local_tier, hybrid_tier, premium_tier]

        for tier in tiers:
            state = await self._execute_with_tier(state, tier, context)

            # Check quality
            quality, _ = self._quality_gate.assess_quality(
                response=state.base_state.research_results.primary_answer if state.base_state.research_results else "",
                task=state.base_state.query,
                confidence=state.initial_quality,
            )

            if quality >= self._min_quality.value:
                state.final_quality = quality
                state.quality_gate_passed = True
                break
            else:
                state.escalations += 1
                self._v2_stats["escalations"] += 1
                logger.info(f"Escalating from {tier.value} (quality: {quality:.2f})")

        return state

    async def _execute_fastest(
        self,
        state: V2ExecutionState,
        task_type: Any,  # TaskType - reserved for future complexity hints
        context: str | None,
    ) -> V2ExecutionState:
        """Race multiple approaches, use fastest acceptable result."""
        _ = (task_type, context)  # Reserved for future use
        # Get tier values safely (handle potential None)
        local_tier = getattr(ModelTier, 'LOCAL', _ModelTierFallback.LOCAL) if ModelTier else _ModelTierFallback.LOCAL
        hybrid_tier = getattr(ModelTier, 'HYBRID', _ModelTierFallback.HYBRID) if ModelTier else _ModelTierFallback.HYBRID

        # Create concurrent tasks
        tasks = [
            asyncio.create_task(self._execute_with_tier(state, local_tier, context)),
            asyncio.create_task(self._execute_with_tier(state, hybrid_tier, context)),
        ]

        # Wait for first to complete
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Get result from first completed
        result_state = None
        for task in done:
            try:
                result_state = await task
                # Check if quality is acceptable
                quality, _ = self._quality_gate.assess_quality(
                    response=result_state.base_state.research_results.primary_answer if result_state.base_state.research_results else "",
                    task=state.base_state.query,
                )
                if quality >= self._min_quality.value:
                    break
            except Exception:
                continue

        # Cancel pending
        for task in pending:
            task.cancel()

        return result_state if result_state else state

    async def _execute_with_tier(
        self,
        state: V2ExecutionState,
        tier: Any,  # ModelTier
        context: str | None,  # Reserved for tier-specific context
    ) -> V2ExecutionState:
        """Execute task with a specific tier."""
        _ = context  # Reserved for future use
        # Create base_state safely
        if PlatformState is not None:
            base_state_obj = PlatformState(
                query=state.base_state.query,
                task_type=state.base_state.task_type,
            )
        else:
            base_state_obj = state.base_state  # Reuse existing

        new_state = V2ExecutionState(
            base_state=base_state_obj,
            tier_used=tier,
        )

        # Get model for tier
        model = self._model_router.get_best_model_for_tier(tier)
        new_state.model_used = model.model_id if model else tier.value

        # Execute research
        result = await self._v1.research(state.base_state.query)
        new_state.base_state.research_results = result
        new_state.initial_quality = result.confidence if result else 0.0

        return new_state

    async def _phase_quality_gate(
        self,
        state: V2ExecutionState,
        required_quality: QualityLevel,
    ) -> V2ExecutionState:
        """Check quality and escalate if needed."""
        start_time = time.time()

        response = ""
        if state.base_state.research_results:
            response = state.base_state.research_results.primary_answer

        quality, _issues = self._quality_gate.assess_quality(
            response=response,
            task=state.base_state.query,
            confidence=state.initial_quality,
        )

        state.final_quality = quality

        if quality >= required_quality.value:
            state.quality_gate_passed = True
            self._v2_stats["quality_gate_passes"] += 1
        else:
            # Check if we should escalate
            should_escalate, next_tier = self._quality_gate.should_escalate(
                quality,
                state.tier_used,
                state.escalations,
            )

            if should_escalate and next_tier:
                state.escalations += 1
                self._v2_stats["escalations"] += 1
                logger.info(f"Quality gate escalating to {next_tier.value}")

                # Re-execute with higher tier
                state = await self._execute_with_tier(state, next_tier, None)
                state.tier_used = next_tier

                # Re-check quality
                if state.base_state.research_results:
                    quality, _ = self._quality_gate.assess_quality(
                        response=state.base_state.research_results.primary_answer,
                        task=state.base_state.query,
                    )
                    state.final_quality = quality
                    state.quality_gate_passed = quality >= required_quality.value
            else:
                self._v2_stats["quality_gate_failures"] += 1

        state.quality_check_ms = (time.time() - start_time) * 1000
        return state

    # =========================================================================
    # Cost Management
    # =========================================================================

    def _update_cost_metrics(self, state: V2ExecutionState) -> None:
        """Update cost tracking from execution."""
        # Get tier values safely (handle potential None)
        local_tier = getattr(ModelTier, 'LOCAL', _ModelTierFallback.LOCAL) if ModelTier else _ModelTierFallback.LOCAL
        hybrid_tier = getattr(ModelTier, 'HYBRID', _ModelTierFallback.HYBRID) if ModelTier else _ModelTierFallback.HYBRID
        premium_tier = getattr(ModelTier, 'PREMIUM', _ModelTierFallback.PREMIUM) if ModelTier else _ModelTierFallback.PREMIUM

        # Estimate cost based on tier and tokens
        tier_costs: dict[Any, float] = {
            local_tier: 0.0,  # Free
            hybrid_tier: 0.001,  # ~$1/1M tokens
            premium_tier: 0.015,  # ~$15/1M tokens
        }

        # Rough estimate: 1000 tokens per request
        estimated_tokens = 1000
        tier_cost = tier_costs.get(state.tier_used, 0.001)
        state.estimated_cost = (estimated_tokens / 1_000_000) * tier_cost * 1000  # Convert to per-1K

        self._cost_metrics.total_spent_today += state.estimated_cost

        tier_name = state.tier_used.value
        self._cost_metrics.tier_breakdown[tier_name] = (
            self._cost_metrics.tier_breakdown.get(tier_name, 0.0) + state.estimated_cost
        )

    def get_cost_report(self) -> dict[str, Any]:
        """Get cost tracking report."""
        return {
            "total_spent_today": self._cost_metrics.total_spent_today,
            "daily_budget": self._cost_metrics.daily_budget,
            "remaining_budget": self._cost_metrics.remaining_budget,
            "budget_utilization": f"{self._cost_metrics.budget_utilization:.1%}",
            "tier_breakdown": self._cost_metrics.tier_breakdown,
            "should_prefer_local": self._cost_metrics.should_prefer_local(),
        }

    # =========================================================================
    # Result Building
    # =========================================================================

    def _build_v2_result(self, state: V2ExecutionState) -> Any:  # PlatformResult
        """Build result with V2 enhancements."""
        # Use V1 result building
        result = self._v1._build_result(state.base_state)

        # Add V2 metadata to warnings
        v2_info = [
            f"[V2] Model: {state.model_used} ({state.tier_used.value})",
            f"[V2] Quality: {state.final_quality:.1%} (gate: {'PASS' if state.quality_gate_passed else 'FAIL'})",
            f"[V2] Escalations: {state.escalations}",
        ]

        # Add token optimization info (V2.1)
        if state.compression_ratio < 1.0:
            v2_info.append(
                f"[V2.1] Tokens: {state.tokens_before_compression}→{state.tokens_after_compression} "
                f"({state.compression_ratio:.1%} ratio, saved {state.tokens_before_compression - state.tokens_after_compression})"
            )
        if state.cache_hit:
            v2_info.append(f"[V2.1] Cache HIT (key: {state.cache_key[:8]}...)")

        if state.mesh_agents_used:
            v2_info.append(f"[V2] Mesh agents: {len(state.mesh_agents_used)}")

        if state.speculative_hits > 0:
            v2_info.append(f"[V2] Speculative hits: {state.speculative_hits}/{state.speculative_paths}")

        result.warnings = v2_info + result.warnings

        return result

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_stats(self) -> dict[str, Any]:
        """Get comprehensive V2 statistics."""
        v1_stats = await self._v1.get_stats()

        # Token optimization stats
        token_opt_stats = {}
        if self._token_optimizer is not None:
            token_opt_stats = {
                "cache_hits": self._v2_stats.get("cache_hits", 0),
                "cache_misses": self._v2_stats.get("cache_misses", 0),
                "cache_hit_rate": (
                    self._v2_stats.get("cache_hits", 0) /
                    max(1, self._v2_stats.get("cache_hits", 0) + self._v2_stats.get("cache_misses", 0))
                ),
                "tokens_saved": self._v2_stats.get("tokens_saved", 0),
                "compression_ratio_avg": self._v2_stats.get("compression_ratio_avg", 1.0),
                "compression_level": self._compression_level.value if self._compression_level else "none",
                "cost_tracker": self._token_optimizer.cost_tracker.get_summary() if hasattr(self._token_optimizer, 'cost_tracker') else {},
            }

        return {
            "v1": v1_stats,
            "v2": {
                **self._v2_stats,
                "speculation_hit_rate": self._speculative.hit_rate if self._speculative else 0.0,
                "mesh_active_agents": len(self._agent_mesh._agents) if self._agent_mesh else 0,
            },
            "v2.1_token_optimization": token_opt_stats,
            "cost": self.get_cost_report(),
            "model_router": {
                "total_routes": self._model_router._stats["total_routes"],
                "tier_distribution": self._model_router._stats["tier_distribution"],
            },
        }

    def format_v2_report(self, result: Any) -> str:  # result: PlatformResult
        """Format a V2-enhanced report."""
        v1_report = self._v1.format_platform_report(result)

        # Token optimization section
        token_opt_section = ""
        if self._token_optimizer is not None:
            cache_hits = self._v2_stats.get("cache_hits", 0)
            cache_misses = self._v2_stats.get("cache_misses", 0)
            total_cache = cache_hits + cache_misses
            cache_rate = (cache_hits / total_cache * 100) if total_cache > 0 else 0

            token_opt_section = f"""
### Token Optimization (V2.1)
- Compression level: {self._compression_level.value if self._compression_level else 'none'}
- Average compression ratio: {self._v2_stats.get('compression_ratio_avg', 1.0):.1%}
- Tokens saved: {self._v2_stats.get('tokens_saved', 0):,}
- Cache hit rate: {cache_rate:.1f}% ({cache_hits}/{total_cache})
"""

        v2_section = """
## V2 Enhancements

### Model Routing
- Tier selection based on task complexity
- Automatic escalation on quality issues

### Cost Tracking
{}
{}
### Performance
- Speculative execution enabled
- Agent mesh for parallel work
""".format(
            "\n".join(f"- {k}: {v}" for k, v in self.get_cost_report().items()),
            token_opt_section
        )

        return v1_report + v2_section


# =============================================================================
# Factory Functions
# =============================================================================

def create_v2_platform(
    mcp_executor: Any = None,  # MCPExecutor | None
    cost_tier: CostTier = CostTier.BALANCED,
    min_quality: QualityLevel = QualityLevel.STANDARD,
    daily_budget: float = 50.0,
) -> PlatformOrchestratorV2:
    """
    Create a fully configured V2 platform orchestrator.

    Args:
        mcp_executor: MCP tool executor
        cost_tier: Cost awareness level
        min_quality: Minimum quality threshold
        daily_budget: Daily spending limit

    Returns:
        Configured PlatformOrchestratorV2
    """
    return PlatformOrchestratorV2(
        mcp_executor=mcp_executor,
        cost_tier=cost_tier,
        min_quality=min_quality,
        daily_budget=daily_budget,
        enable_speculation=True,
        enable_mesh=True,
        enable_prewarm=True,
    )


def create_minimal_v2_platform() -> PlatformOrchestratorV2:
    """Create a minimal V2 platform (no background services)."""
    return PlatformOrchestratorV2(
        mcp_executor=None,
        cost_tier=CostTier.MINIMAL,
        enable_speculation=False,
        enable_mesh=False,
        enable_prewarm=False,
    )


def create_performance_v2_platform(
    mcp_executor: Any = None,  # MCPExecutor | None
    daily_budget: float = 100.0,
) -> PlatformOrchestratorV2:
    """Create a performance-focused V2 platform."""
    return PlatformOrchestratorV2(
        mcp_executor=mcp_executor,
        cost_tier=CostTier.PERFORMANCE,
        min_quality=QualityLevel.HIGH,
        daily_budget=daily_budget,
        enable_speculation=True,
        enable_mesh=True,
        enable_prewarm=True,
    )


# =============================================================================
# Demo
# =============================================================================

async def demo():
    """Comprehensive V2 platform demonstration."""
    print("=" * 70)
    print("Platform Orchestrator V2 - Demo")
    print("=" * 70)
    print()

    # Create V2 platform
    platform = create_v2_platform(
        cost_tier=CostTier.BALANCED,
        min_quality=QualityLevel.STANDARD,
        daily_budget=50.0,
    )

    await platform.start()

    try:
        # =================================
        # Demo 1: Standard Execution
        # =================================
        print("1. STANDARD EXECUTION")
        print("-" * 40)

        # Get TaskType values safely
        research_type = getattr(TaskType, 'RESEARCH', _TaskTypeFallback.RESEARCH) if TaskType else _TaskTypeFallback.RESEARCH

        result = await platform.execute(
            "How to implement OAuth2 authentication",
            research_type,
            execution_mode=ExecutionMode.STANDARD,
        )

        print(f"Status: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"Duration: {result.total_duration_ms:.0f}ms")
        print(f"V2 Info: {result.warnings[:3]}")
        print()

        # =================================
        # Demo 2: Speculative Execution
        # =================================
        print("2. SPECULATIVE EXECUTION")
        print("-" * 40)

        result = await platform.execute(
            "Design a microservices architecture",
            research_type,
            execution_mode=ExecutionMode.SPECULATIVE,
        )

        print(f"Status: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"V2 Info: {result.warnings[:4]}")
        print()

        # =================================
        # Demo 3: Cascade Execution
        # =================================
        print("3. CASCADE EXECUTION")
        print("-" * 40)

        implement_type = getattr(TaskType, 'IMPLEMENT', _TaskTypeFallback.IMPLEMENT) if TaskType else _TaskTypeFallback.IMPLEMENT

        result = await platform.execute(
            "Implement secure payment processing",
            implement_type,
            execution_mode=ExecutionMode.CASCADE,
        )

        print(f"Status: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"V2 Info: {result.warnings[:4]}")
        print()

        # =================================
        # Demo 4: Cost Report
        # =================================
        print("4. COST REPORT")
        print("-" * 40)

        cost_report = platform.get_cost_report()
        for key, value in cost_report.items():
            print(f"  {key}: {value}")
        print()

        # =================================
        # Demo 5: Full Statistics
        # =================================
        print("5. V2 STATISTICS")
        print("-" * 40)

        stats = await platform.get_stats()

        print("V2 Metrics:")
        for key, value in stats["v2"].items():
            print(f"  {key}: {value}")

        print("\nModel Router:")
        for key, value in stats["model_router"].items():
            print(f"  {key}: {value}")

    finally:
        await platform.stop()

    print()
    print("=" * 70)
    print("V2 Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo())
