"""
Proactive Agent Orchestration Layer - UNLEASH Platform.

Implements 10 proactive agents (9 from Everything-Claude-Code + Verification Subagent):
1. planner - Complex feature implementation planning
2. architect - System design and architecture decisions
3. tdd-guide - Test-driven development enforcement
4. code-reviewer - Quality, security, and maintainability review
5. security-reviewer - Vulnerability analysis and security audit
6. build-error-resolver - Build failure diagnosis and fixing
7. e2e-runner - End-to-end test generation and execution
8. refactor-cleaner - Dead code removal and refactoring
9. doc-updater - Documentation synchronization
10. verification-agent - Quality gates & 6-phase verification (P0 Anthropic pattern)

Key Principles:
- Agents trigger AUTOMATICALLY based on task type detection
- Each agent uses optimal model (Haiku for speed, Opus for critical)
- Parallel execution for independent tasks
- Max 6-8 concurrent agents to avoid O(n²) coordination overhead
- Verification agent runs LAST as final quality gate

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                ProactiveAgentOrchestrator                        │
    │  ┌──────────────────────────────────────────────────────────┐   │
    │  │           Task Type Detector                              │   │
    │  │  • Keyword analysis                                       │   │
    │  │  • File pattern matching                                  │   │
    │  │  • Context inference                                      │   │
    │  └──────────────────────────────────────────────────────────┘   │
    │                           │                                      │
    │                           ▼                                      │
    │  ┌──────────────────────────────────────────────────────────┐   │
    │  │           Agent Router                                    │   │
    │  │  • Trigger matching                                       │   │
    │  │  • Model selection (Haiku/Sonnet/Opus)                   │   │
    │  │  • Load balancing                                         │   │
    │  └──────────────────────────────────────────────────────────┘   │
    │                           │                                      │
    │                           ▼                                      │
    │  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐ │
    │  │Plan │Arch │TDD  │Revw │Sec  │Build│E2E  │Refac│Docs │VERFY│ │
    │  │ner  │itect│Guide│er   │Rvw  │Fix  │Run  │Clean│Upd  │GATE │ │
    │  │     │     │     │     │     │     │     │     │     │     │ │
    │  │Sonn.│Opus │Sonn.│Sonn.│Opus │Sonn.│Haiku│Haiku│Haiku│Haiku│ │
    │  │ P1  │ P1  │ P2  │ P3  │ P2  │ P0  │ P4  │ P5  │ P6  │ P7  │ │
    │  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘ │
    │         ← Higher Priority (runs first)    Lower Priority →      │
    │                                                      ↑          │
    │                                         FINAL QUALITY GATE      │
    └─────────────────────────────────────────────────────────────────┘

Version: V1.1.0 (January 2026) - Added Verification Subagent (P0 Optimization)
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class AgentType(str, Enum):
    """Types of proactive agents."""
    PLANNER = "planner"
    ARCHITECT = "architect"
    TDD_GUIDE = "tdd-guide"
    CODE_REVIEWER = "code-reviewer"
    SECURITY_REVIEWER = "security-reviewer"
    BUILD_ERROR_RESOLVER = "build-error-resolver"
    E2E_RUNNER = "e2e-runner"
    REFACTOR_CLEANER = "refactor-cleaner"
    DOC_UPDATER = "doc-updater"
    VERIFICATION_AGENT = "verification-agent"  # P0: Anthropic-recommended verification pattern


class ModelTier(str, Enum):
    """Model tiers for cost/performance optimization."""
    HAIKU = "haiku"      # Fast, cheap - exploration, docs, simple tasks
    SONNET = "sonnet"    # Balanced - main development
    OPUS = "opus"        # Best - architecture, security, critical decisions


class TaskCategory(str, Enum):
    """Categories of development tasks."""
    FEATURE = "feature"           # New feature implementation
    BUGFIX = "bugfix"            # Bug fixing
    REFACTOR = "refactor"        # Code refactoring
    SECURITY = "security"        # Security-related changes
    PERFORMANCE = "performance"  # Performance optimization
    DOCUMENTATION = "documentation"  # Documentation
    TESTING = "testing"          # Test writing
    BUILD = "build"              # Build/CI issues
    ARCHITECTURE = "architecture"  # Architecture decisions
    VERIFICATION = "verification"  # Quality gates and verification


# =============================================================================
# TRIGGER PATTERNS
# =============================================================================

# Keyword patterns that trigger specific agents
TRIGGER_PATTERNS: Dict[AgentType, Dict[str, Any]] = {
    AgentType.PLANNER: {
        "keywords": [
            "implement", "add feature", "create", "build", "develop",
            "design", "plan", "complex", "multi-step", "milestone"
        ],
        "file_patterns": [],
        "conditions": lambda ctx: ctx.get("estimated_complexity", 0) > 3,
        "model": ModelTier.SONNET,
        "priority": 1,  # High priority, runs first
    },
    AgentType.ARCHITECT: {
        "keywords": [
            "architecture", "system design", "scalability", "microservice",
            "database schema", "api design", "refactor large", "restructure"
        ],
        "file_patterns": [],
        "conditions": lambda ctx: ctx.get("affects_architecture", False),
        "model": ModelTier.OPUS,
        "priority": 1,
    },
    AgentType.TDD_GUIDE: {
        "keywords": [
            "test", "tdd", "coverage", "spec", "unit test", "integration test",
            "fix bug", "implement", "new feature"
        ],
        "file_patterns": [r"\.test\.", r"\.spec\.", r"_test\.py$", r"test_.*\.py$"],
        "conditions": lambda ctx: True,  # Always applicable for code changes
        "model": ModelTier.SONNET,
        "priority": 2,
    },
    AgentType.CODE_REVIEWER: {
        "keywords": [
            "review", "check", "quality", "code change", "pr", "pull request"
        ],
        "file_patterns": [r"\.(py|ts|tsx|js|jsx|go|rs)$"],
        "conditions": lambda ctx: ctx.get("code_changed", False),
        "model": ModelTier.SONNET,
        "priority": 3,  # Runs after code changes
    },
    AgentType.SECURITY_REVIEWER: {
        "keywords": [
            "security", "auth", "authentication", "authorization", "password",
            "token", "api key", "secret", "credential", "encrypt", "vulnerability",
            "input validation", "sanitize", "injection", "xss", "csrf"
        ],
        "file_patterns": [
            r"auth", r"login", r"session", r"token", r"password",
            r"security", r"crypto", r"secret"
        ],
        "conditions": lambda ctx: ctx.get("touches_auth", False),
        "model": ModelTier.OPUS,  # Security needs best model
        "priority": 2,
    },
    AgentType.BUILD_ERROR_RESOLVER: {
        "keywords": [
            "build failed", "compile error", "type error", "syntax error",
            "import error", "module not found", "dependency", "build"
        ],
        "file_patterns": [],
        "conditions": lambda ctx: ctx.get("build_failed", False),
        "model": ModelTier.SONNET,
        "priority": 0,  # Highest priority when triggered
    },
    AgentType.E2E_RUNNER: {
        "keywords": [
            "e2e", "end to end", "playwright", "cypress", "selenium",
            "integration test", "user flow", "critical path"
        ],
        "file_patterns": [r"e2e", r"\.e2e\.", r"playwright", r"cypress"],
        "conditions": lambda ctx: ctx.get("affects_user_flow", False),
        "model": ModelTier.HAIKU,  # E2E tests are execution-heavy, not reasoning-heavy
        "priority": 4,
    },
    AgentType.REFACTOR_CLEANER: {
        "keywords": [
            "refactor", "cleanup", "dead code", "remove unused", "simplify",
            "consolidate", "deprecate", "delete", "prune"
        ],
        "file_patterns": [],
        "conditions": lambda ctx: ctx.get("cleanup_requested", False),
        "model": ModelTier.HAIKU,
        "priority": 5,
    },
    AgentType.DOC_UPDATER: {
        "keywords": [
            "document", "readme", "api doc", "update docs", "changelog",
            "comment", "docstring", "jsdoc", "typedoc"
        ],
        "file_patterns": [r"\.md$", r"README", r"CHANGELOG", r"docs/"],
        "conditions": lambda ctx: ctx.get("api_changed", False),
        "model": ModelTier.HAIKU,
        "priority": 6,  # Lowest priority, runs last
    },
    # P0 Optimization: Anthropic-recommended Verification Subagent Pattern
    # Source: RESEARCH_SYNTHESIS_2026.md, arXiv 2507.14928 (DecentLLMs)
    AgentType.VERIFICATION_AGENT: {
        "keywords": [
            "verify", "validate", "check", "confirm", "quality gate",
            "pre-commit", "regression", "pass", "complete", "done",
            "ready to merge", "pr ready", "ship it"
        ],
        "file_patterns": [],  # Triggers on context, not file patterns
        "conditions": lambda ctx: (
            ctx.get("code_changed", False) or
            ctx.get("pre_commit", False) or
            ctx.get("task_complete", False) or
            ctx.get("run_verification", False)
        ),
        "model": ModelTier.HAIKU,  # Fast, cheap - per Anthropic guidance
        "priority": 7,  # Runs AFTER all other agents complete (final gate)
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for a proactive agent."""
    agent_type: AgentType
    model: ModelTier
    triggers: List[str]
    file_patterns: List[str]
    priority: int
    max_concurrent: int = 1
    timeout_seconds: int = 300
    enabled: bool = True

    @classmethod
    def from_type(cls, agent_type: AgentType) -> "AgentConfig":
        """Create config from agent type using default trigger patterns."""
        patterns = TRIGGER_PATTERNS.get(agent_type, {})
        return cls(
            agent_type=agent_type,
            model=patterns.get("model", ModelTier.SONNET),
            triggers=patterns.get("keywords", []),
            file_patterns=patterns.get("file_patterns", []),
            priority=patterns.get("priority", 5),
        )


@dataclass
class AgentTask:
    """A task assigned to a proactive agent."""
    task_id: str
    agent_type: AgentType
    description: str
    context: Dict[str, Any]
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    model_used: Optional[ModelTier] = None

    @property
    def duration_ms(self) -> Optional[float]:
        """Calculate execution duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None


@dataclass
class TaskAnalysis:
    """Result of analyzing a task for agent routing."""
    original_description: str
    category: TaskCategory
    detected_keywords: List[str]
    matched_files: List[str]
    triggered_agents: List[AgentType]
    context: Dict[str, Any]
    confidence: float


# =============================================================================
# TASK TYPE DETECTOR
# =============================================================================

class TaskTypeDetector:
    """
    Detects task types and determines which agents should be triggered.

    Uses:
    - Keyword matching in task description
    - File pattern analysis
    - Context conditions (build status, code changes, etc.)
    """

    def __init__(self):
        self.keyword_cache: Dict[str, Set[str]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for efficiency."""
        self.compiled_patterns: Dict[AgentType, List[re.Pattern]] = {}

        for agent_type, config in TRIGGER_PATTERNS.items():
            patterns = config.get("file_patterns", [])
            self.compiled_patterns[agent_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def analyze(
        self,
        description: str,
        files: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskAnalysis:
        """
        Analyze a task and determine which agents should handle it.

        Args:
            description: Task description text
            files: List of files being modified
            context: Additional context (build status, etc.)

        Returns:
            TaskAnalysis with triggered agents and confidence
        """
        files = files or []
        context = context or {}
        description_lower = description.lower()

        triggered_agents: List[Tuple[AgentType, float, int]] = []  # (type, score, priority)
        all_keywords: List[str] = []
        matched_files: List[str] = []

        for agent_type, config in TRIGGER_PATTERNS.items():
            score = 0.0
            agent_keywords = []

            # Check keywords
            for keyword in config.get("keywords", []):
                if keyword.lower() in description_lower:
                    score += 1.0
                    agent_keywords.append(keyword)

            # Check file patterns
            for pattern in self.compiled_patterns.get(agent_type, []):
                for file_path in files:
                    if pattern.search(file_path):
                        score += 0.5
                        if file_path not in matched_files:
                            matched_files.append(file_path)

            # Check conditions
            condition = config.get("conditions", lambda ctx: True)
            if condition(context):
                score += 0.3

            # If score is significant, add to triggered agents
            if score > 0.5 or agent_keywords:
                all_keywords.extend(agent_keywords)
                priority = config.get("priority", 5)
                triggered_agents.append((agent_type, score, priority))

        # Sort by priority (lower is higher priority), then by score (higher is better)
        triggered_agents.sort(key=lambda x: (x[2], -x[1]))

        # Determine task category
        category = self._infer_category(description_lower, triggered_agents)

        # Calculate overall confidence
        confidence = min(1.0, sum(s for _, s, _ in triggered_agents) / max(len(triggered_agents), 1))

        return TaskAnalysis(
            original_description=description,
            category=category,
            detected_keywords=list(set(all_keywords)),
            matched_files=matched_files,
            triggered_agents=[t[0] for t in triggered_agents],
            context=context,
            confidence=confidence
        )

    def _infer_category(
        self,
        description: str,
        triggered: List[Tuple[AgentType, float, int]]
    ) -> TaskCategory:
        """Infer the task category from description and triggered agents."""
        # Priority checks
        if "bug" in description or "fix" in description:
            return TaskCategory.BUGFIX
        if "security" in description or AgentType.SECURITY_REVIEWER in [t[0] for t in triggered]:
            return TaskCategory.SECURITY
        if "build" in description or "compile" in description:
            return TaskCategory.BUILD
        if "refactor" in description:
            return TaskCategory.REFACTOR
        if "test" in description:
            return TaskCategory.TESTING
        if "doc" in description or "readme" in description:
            return TaskCategory.DOCUMENTATION
        if "performance" in description or "optimize" in description:
            return TaskCategory.PERFORMANCE
        if "architect" in description or "design" in description:
            return TaskCategory.ARCHITECTURE

        return TaskCategory.FEATURE


# =============================================================================
# PROACTIVE AGENT ORCHESTRATOR
# =============================================================================

class ProactiveAgentOrchestrator:
    """
    Orchestrates proactive agents based on task analysis.

    Key features:
    - Automatic agent triggering based on task type
    - Optimal model selection per agent
    - Parallel execution for independent agents
    - Max 6-8 concurrent to avoid O(n²) overhead
    """

    def __init__(
        self,
        max_concurrent: int = 6,
        enable_auto_trigger: bool = True
    ):
        self.max_concurrent = max_concurrent
        self.enable_auto_trigger = enable_auto_trigger

        # Initialize components
        self.detector = TaskTypeDetector()
        self.agent_configs: Dict[AgentType, AgentConfig] = {
            t: AgentConfig.from_type(t) for t in AgentType
        }

        # Task tracking
        self.pending_tasks: List[AgentTask] = []
        self.running_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: List[AgentTask] = []

        # Agent handlers (to be set by consumers)
        self.handlers: Dict[AgentType, Callable] = {}

        # Metrics
        self.metrics = {
            "tasks_processed": 0,
            "agents_triggered": 0,
            "avg_task_duration_ms": 0.0,
            "model_usage": {tier.value: 0 for tier in ModelTier},
        }

        self._lock = asyncio.Lock()
        self._task_counter = 0

        logger.info(
            "ProactiveAgentOrchestrator initialized: max_concurrent=%d, auto_trigger=%s",
            max_concurrent, enable_auto_trigger
        )

    def register_handler(
        self,
        agent_type: AgentType,
        handler: Callable[[AgentTask], Any]
    ) -> None:
        """Register a handler function for an agent type."""
        self.handlers[agent_type] = handler
        logger.debug("Registered handler for %s", agent_type.value)

    def analyze_task(
        self,
        description: str,
        files: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskAnalysis:
        """Analyze a task and get triggered agents."""
        return self.detector.analyze(description, files, context)

    async def process_task(
        self,
        description: str,
        files: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a task through the proactive agent system.

        1. Analyze task to determine triggered agents
        2. Create agent tasks for each triggered agent
        3. Execute agents respecting priority and concurrency
        4. Aggregate results

        Returns:
            Dict with analysis, agent results, and metrics
        """
        start_time = time.perf_counter()

        # Analyze the task
        analysis = self.analyze_task(description, files, context)

        if not self.enable_auto_trigger:
            return {
                "analysis": analysis,
                "agents_triggered": [],
                "results": {},
                "auto_trigger_disabled": True
            }

        # Create agent tasks
        agent_tasks = []
        for agent_type in analysis.triggered_agents:
            config = self.agent_configs.get(agent_type)
            if config and config.enabled:
                task = self._create_agent_task(
                    agent_type=agent_type,
                    description=description,
                    context={
                        **analysis.context,
                        "analysis": {
                            "category": analysis.category.value,
                            "keywords": analysis.detected_keywords,
                            "files": analysis.matched_files,
                        }
                    },
                    model=config.model
                )
                agent_tasks.append(task)

        # Execute agents (respecting concurrency)
        results = await self._execute_agents(agent_tasks)

        # Update metrics
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._update_metrics(agent_tasks, elapsed_ms)

        return {
            "analysis": {
                "category": analysis.category.value,
                "triggered_agents": [a.value for a in analysis.triggered_agents],
                "keywords": analysis.detected_keywords,
                "files": analysis.matched_files,
                "confidence": analysis.confidence,
            },
            "agents_triggered": [t.agent_type.value for t in agent_tasks],
            "results": results,
            "elapsed_ms": elapsed_ms,
            "metrics": self.get_metrics()
        }

    def _create_agent_task(
        self,
        agent_type: AgentType,
        description: str,
        context: Dict[str, Any],
        model: ModelTier
    ) -> AgentTask:
        """Create a new agent task."""
        self._task_counter += 1
        task_id = f"task_{self._task_counter}_{agent_type.value}"

        return AgentTask(
            task_id=task_id,
            agent_type=agent_type,
            description=description,
            context=context,
            model_used=model
        )

    async def _execute_agents(
        self,
        tasks: List[AgentTask]
    ) -> Dict[str, Any]:
        """Execute agent tasks respecting concurrency limits."""
        results: Dict[str, Any] = {}

        # Group tasks by priority
        priority_groups: Dict[int, List[AgentTask]] = {}
        for task in tasks:
            config = self.agent_configs.get(task.agent_type)
            priority = config.priority if config else 5
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(task)

        # Execute by priority order
        for priority in sorted(priority_groups.keys()):
            group_tasks = priority_groups[priority]

            # Execute within concurrency limit
            for batch_start in range(0, len(group_tasks), self.max_concurrent):
                batch = group_tasks[batch_start:batch_start + self.max_concurrent]

                # Execute batch in parallel
                batch_results = await asyncio.gather(
                    *[self._execute_single_agent(task) for task in batch],
                    return_exceptions=True
                )

                # Collect results
                for task, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        results[task.agent_type.value] = {
                            "success": False,
                            "error": str(result)
                        }
                    else:
                        results[task.agent_type.value] = result

        return results

    async def _execute_single_agent(
        self,
        task: AgentTask
    ) -> Dict[str, Any]:
        """Execute a single agent task."""
        task.status = "running"
        task.started_at = datetime.now(timezone.utc)

        async with self._lock:
            self.running_tasks[task.task_id] = task

        try:
            # Get handler for this agent type
            handler = self.handlers.get(task.agent_type)

            if handler:
                # Execute the handler
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(task)
                else:
                    result = handler(task)

                task.result = result
                task.status = "completed"

                return {
                    "success": True,
                    "result": result,
                    "model": task.model_used.value if task.model_used else None,
                    "duration_ms": task.duration_ms
                }
            else:
                # No handler - return placeholder
                return {
                    "success": True,
                    "result": f"Agent {task.agent_type.value} triggered (no handler registered)",
                    "model": task.model_used.value if task.model_used else None,
                    "triggered_only": True
                }

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            logger.error("Agent %s failed: %s", task.agent_type.value, e)
            raise

        finally:
            task.completed_at = datetime.now(timezone.utc)

            async with self._lock:
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
                self.completed_tasks.append(task)

    def _update_metrics(
        self,
        tasks: List[AgentTask],
        total_duration_ms: float
    ) -> None:
        """Update orchestrator metrics."""
        self.metrics["tasks_processed"] += 1
        self.metrics["agents_triggered"] += len(tasks)

        # Update model usage
        for task in tasks:
            if task.model_used:
                self.metrics["model_usage"][task.model_used.value] += 1

        # Update average duration
        alpha = 0.1
        self.metrics["avg_task_duration_ms"] = (
            alpha * total_duration_ms +
            (1 - alpha) * self.metrics["avg_task_duration_ms"]
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            **self.metrics,
            "running_count": len(self.running_tasks),
            "completed_count": len(self.completed_tasks),
            "registered_handlers": list(self.handlers.keys()),
        }

    def enable_agent(self, agent_type: AgentType) -> None:
        """Enable a specific agent."""
        if agent_type in self.agent_configs:
            self.agent_configs[agent_type].enabled = True

    def disable_agent(self, agent_type: AgentType) -> None:
        """Disable a specific agent."""
        if agent_type in self.agent_configs:
            self.agent_configs[agent_type].enabled = False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_orchestrator_instance: Optional[ProactiveAgentOrchestrator] = None


def get_orchestrator(
    max_concurrent: int = 6,
    enable_auto_trigger: bool = True
) -> ProactiveAgentOrchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = ProactiveAgentOrchestrator(
            max_concurrent=max_concurrent,
            enable_auto_trigger=enable_auto_trigger
        )
    return _orchestrator_instance


def reset_orchestrator() -> None:
    """Reset the global orchestrator instance (for testing)."""
    global _orchestrator_instance
    _orchestrator_instance = None


# =============================================================================
# AGENT PROMPT TEMPLATES
# =============================================================================

AGENT_PROMPTS: Dict[AgentType, str] = {
    AgentType.PLANNER: """You are a senior planning specialist for complex feature implementations.

Your role:
1. Break down complex tasks into step-by-step implementation plans
2. Identify critical files and dependencies
3. Consider architectural trade-offs
4. Estimate complexity and risk

Focus on actionable, concrete steps. Don't over-plan - identify the minimum viable approach.""",

    AgentType.ARCHITECT: """You are a senior software architect specializing in system design.

Your role:
1. Make architecture decisions that scale
2. Design APIs and data models
3. Consider security, performance, and maintainability
4. Document architectural decisions (ADRs)

Focus on simplicity and proven patterns. Avoid over-engineering.""",

    AgentType.TDD_GUIDE: """You are a TDD (Test-Driven Development) specialist.

Your workflow:
1. Define interfaces first (types, contracts)
2. Write failing tests (RED)
3. Implement minimal code to pass (GREEN)
4. Refactor for quality (IMPROVE)
5. Verify 80%+ coverage

Enforce the discipline: tests BEFORE implementation.""",

    AgentType.CODE_REVIEWER: """You are a senior code reviewer focused on quality and maintainability.

Review criteria:
1. Code correctness and logic
2. Security vulnerabilities (OWASP Top 10)
3. Performance issues
4. Code style and readability
5. Test coverage

Provide specific, actionable feedback. Approve only when ready.""",

    AgentType.SECURITY_REVIEWER: """You are a security specialist focused on vulnerability detection.

Your checklist:
1. Input validation at boundaries
2. Authentication and authorization
3. Secrets and credentials handling
4. Injection vulnerabilities (SQL, XSS, command)
5. OWASP Top 10 compliance

Flag issues by severity. Never approve security-critical code without thorough review.""",

    AgentType.BUILD_ERROR_RESOLVER: """You are a build error specialist.

Your approach:
1. Analyze the error message carefully
2. Identify the root cause (not just symptoms)
3. Fix with minimal changes
4. Verify the fix doesn't break other things

Focus on getting builds green quickly. Don't refactor while fixing.""",

    AgentType.E2E_RUNNER: """You are an E2E testing specialist using Playwright.

Your approach:
1. Identify critical user journeys
2. Write reliable, non-flaky tests
3. Use proper waiting and assertions
4. Handle test data setup/cleanup
5. Quarantine flaky tests

Focus on user-visible behavior, not implementation details.""",

    AgentType.REFACTOR_CLEANER: """You are a dead code cleanup specialist.

Your tools:
- knip (TypeScript)
- depcheck (dependencies)
- ts-prune (exports)
- pyright (Python)

Your approach:
1. Identify unused code with static analysis
2. Verify no dynamic references
3. Remove safely with minimal blast radius
4. Update imports and exports

Don't refactor while cleaning. Focus on removal only.""",

    AgentType.DOC_UPDATER: """You are a documentation specialist.

Your scope:
1. Update README files when APIs change
2. Generate JSDoc/docstrings for public APIs
3. Keep CHANGELOG current
4. Update architecture docs for significant changes

Focus on useful, accurate docs. Don't over-document obvious code.""",

    # P0 Optimization: Verification Subagent (Anthropic-recommended pattern)
    AgentType.VERIFICATION_AGENT: """You are a verification specialist ensuring quality gates pass before completion.

Your 6-Phase Verification Loop:
1. BUILD   → Run build command, verify 0 errors
2. TYPES   → Run type checker (pyright/tsc), 0 errors required
3. LINT    → Run linter, minimal warnings acceptable
4. TEST    → Run test suite, verify 80%+ coverage
5. SECRETS → Grep for hardcoded secrets, 0 matches required
6. DIFF    → Review all changes before approving

Confidence Levels:
- HIGH: Direct test execution (run the actual component)
- MEDIUM: Proxy test (check via indicator)
- LOW: Existence check (file/function exists)

Critical Rules:
- NEVER claim "COMPLETE" or "SUCCESS" until all 6 phases pass
- NEVER approve with "basically done" or "works except" caveats
- ALWAYS run tests before claiming code works
- For critical paths: require pass^3 = 100% (all 3 attempts succeed)

Quality Gates (ALL must pass):
□ All spec deliverables EXIST and FUNCTION
□ End-to-end test PASSES (not just unit tests)
□ User can use the feature RIGHT NOW
□ No "known issues" breaking core functionality

Output Format:
Return JSON: {"verified": true|false, "phases": {...}, "blockers": [...], "confidence": "high"|"medium"|"low"}""",
}


# =============================================================================
# LAMaS-INSPIRED PARALLEL EXECUTION ENHANCEMENT
# =============================================================================
# Based on research: LAMaS achieves 38-46% latency reduction through:
# 1. Speculative parallel execution across priority levels
# 2. First-response-wins for validation tasks
# 3. Latency-aware routing based on historical performance
# 4. Dependency graph analysis for true independence

@dataclass
class AgentLatencyStats:
    """Track latency statistics for intelligent routing."""
    agent_type: AgentType
    total_executions: int = 0
    total_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    success_rate: float = 1.0
    _latencies: List[float] = field(default_factory=list)

    def record(self, latency_ms: float, success: bool = True) -> None:
        """Record a new latency measurement."""
        self._latencies.append(latency_ms)
        self.total_executions += 1
        self.total_latency_ms += latency_ms

        # Keep only last 100 measurements
        if len(self._latencies) > 100:
            self._latencies = self._latencies[-100:]

        # Update percentiles
        sorted_latencies = sorted(self._latencies)
        n = len(sorted_latencies)
        self.p50_latency_ms = sorted_latencies[int(n * 0.5)] if n > 0 else 0
        self.p95_latency_ms = sorted_latencies[int(n * 0.95)] if n > 0 else 0
        self.p99_latency_ms = sorted_latencies[int(n * 0.99)] if n > 0 else 0

        # Update success rate (EMA)
        alpha = 0.1
        self.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate


@dataclass
class DependencyEdge:
    """Represents a dependency between agents."""
    from_agent: AgentType
    to_agent: AgentType
    dependency_type: str  # "hard" (must wait) or "soft" (can speculate)


class LAMaSEnhancedOrchestrator(ProactiveAgentOrchestrator):
    """
    LAMaS-inspired orchestrator with advanced parallel execution.

    Research basis: LAMaS achieves 38-46% latency reduction through intelligent
    parallel execution. This implementation adds:

    1. Speculative Execution: Start lower-priority agents before higher-priority
       ones complete, cancel if results invalidate the speculation.

    2. First-Response-Wins: For validation/verification tasks, use whichever
       agent completes first with a valid response.

    3. Latency-Aware Routing: Track agent performance and route time-critical
       tasks to faster agents.

    4. Dependency Analysis: Build actual dependency graph rather than relying
       solely on priority levels.
    """

    def __init__(
        self,
        max_concurrent: int = 8,  # Increased from 6 based on research
        enable_auto_trigger: bool = True,
        enable_speculation: bool = True,
        speculation_budget_pct: float = 0.2,  # 20% extra compute for speculation
    ):
        super().__init__(max_concurrent, enable_auto_trigger)

        self.enable_speculation = enable_speculation
        self.speculation_budget_pct = speculation_budget_pct

        # Latency tracking per agent
        self.latency_stats: Dict[AgentType, AgentLatencyStats] = {
            agent_type: AgentLatencyStats(agent_type=agent_type)
            for agent_type in AgentType
        }

        # Dependency graph (which agents must wait for others)
        self.dependencies: List[DependencyEdge] = self._build_dependency_graph()

        # Speculative execution tracking
        self.speculative_tasks: Dict[str, asyncio.Task] = {}

        logger.info(
            "LAMaSEnhancedOrchestrator initialized: speculation=%s, budget=%.1f%%",
            enable_speculation, speculation_budget_pct * 100
        )

    def _build_dependency_graph(self) -> List[DependencyEdge]:
        """
        Build the agent dependency graph based on logical dependencies.

        Hard dependencies (must wait):
        - verification-agent depends on all code-changing agents
        - code-reviewer depends on tdd-guide (tests should exist first)

        Soft dependencies (can speculate):
        - doc-updater can start early, cancel if code changes
        - e2e-runner can start with current code
        """
        return [
            # Hard: verification must wait for all others
            DependencyEdge(AgentType.VERIFICATION_AGENT, AgentType.CODE_REVIEWER, "hard"),
            DependencyEdge(AgentType.VERIFICATION_AGENT, AgentType.TDD_GUIDE, "hard"),
            DependencyEdge(AgentType.VERIFICATION_AGENT, AgentType.BUILD_ERROR_RESOLVER, "hard"),

            # Hard: code review should have tests
            DependencyEdge(AgentType.CODE_REVIEWER, AgentType.TDD_GUIDE, "soft"),

            # Soft: docs can speculate
            DependencyEdge(AgentType.DOC_UPDATER, AgentType.CODE_REVIEWER, "soft"),

            # Soft: e2e can start early
            DependencyEdge(AgentType.E2E_RUNNER, AgentType.TDD_GUIDE, "soft"),
        ]

    def _get_independent_agents(
        self,
        tasks: List[AgentTask],
        completed: Set[AgentType]
    ) -> List[AgentTask]:
        """
        Identify agents that can run in parallel (no hard dependencies blocking).

        LAMaS insight: Instead of strict priority ordering, identify truly
        independent tasks and run them together.
        """
        independent = []

        for task in tasks:
            can_run = True

            # Check hard dependencies
            for dep in self.dependencies:
                if dep.to_agent == task.agent_type and dep.dependency_type == "hard":
                    # This agent depends on another
                    if dep.from_agent not in completed:
                        # The dependency hasn't completed yet
                        # Check if it's in our task list
                        dep_in_tasks = any(t.agent_type == dep.from_agent for t in tasks)
                        if dep_in_tasks:
                            can_run = False
                            break

            if can_run:
                independent.append(task)

        return independent

    def _get_speculative_candidates(
        self,
        tasks: List[AgentTask],
        running: Set[AgentType],
        completed: Set[AgentType]
    ) -> List[AgentTask]:
        """
        Identify tasks that can run speculatively (soft dependencies only).

        These tasks might need to be cancelled if their soft dependencies
        produce results that invalidate the speculation.
        """
        if not self.enable_speculation:
            return []

        candidates = []

        for task in tasks:
            if task.agent_type in running or task.agent_type in completed:
                continue

            is_speculative = False
            blocked_hard = False

            for dep in self.dependencies:
                if dep.to_agent == task.agent_type:
                    if dep.from_agent not in completed:
                        if dep.dependency_type == "hard":
                            blocked_hard = True
                            break
                        else:  # soft dependency
                            is_speculative = True

            if not blocked_hard and is_speculative:
                candidates.append(task)

        return candidates

    async def _execute_agents(
        self,
        tasks: List[AgentTask]
    ) -> Dict[str, Any]:
        """
        Execute agents with LAMaS-inspired parallel optimization.

        Strategy:
        1. Build dependency-aware execution groups
        2. Run independent agents in parallel
        3. Speculatively start soft-dependent agents
        4. Cancel speculative tasks if dependencies invalidate them
        """
        results: Dict[str, Any] = {}
        completed: Set[AgentType] = set()
        running: Set[AgentType] = set()
        remaining = list(tasks)

        start_time = time.perf_counter()

        while remaining:
            # Get independent tasks that can run now
            independent = self._get_independent_agents(remaining, completed)

            if not independent:
                # All remaining tasks are blocked - shouldn't happen with proper graph
                logger.warning("All tasks blocked, forcing sequential execution")
                independent = remaining[:1]

            # Add speculative tasks if within budget
            speculative = self._get_speculative_candidates(remaining, running, completed)
            speculation_slots = int(self.max_concurrent * self.speculation_budget_pct)
            speculative = speculative[:speculation_slots]

            # Batch = independent + speculative (up to max_concurrent)
            batch = independent + speculative
            batch = batch[:self.max_concurrent]

            # Mark as running
            for task in batch:
                running.add(task.agent_type)

            # Execute batch in parallel
            async_tasks = []
            for task in batch:
                is_speculative = task in speculative
                async_task = asyncio.create_task(
                    self._execute_with_tracking(task, is_speculative)
                )
                async_tasks.append((task, async_task, is_speculative))

                if is_speculative:
                    self.speculative_tasks[task.task_id] = async_task

            # Wait for all batch tasks
            for task, async_task, is_speculative in async_tasks:
                try:
                    result = await async_task
                    results[task.agent_type.value] = result
                    completed.add(task.agent_type)
                    running.discard(task.agent_type)

                    # Remove from remaining
                    remaining = [t for t in remaining if t.agent_type != task.agent_type]

                    # Check if we need to cancel speculative tasks
                    if not is_speculative and result.get("success"):
                        await self._maybe_cancel_speculative(task.agent_type, results)

                except asyncio.CancelledError:
                    logger.info("Speculative task %s cancelled", task.agent_type.value)
                    running.discard(task.agent_type)
                    # Don't remove from remaining - it wasn't truly completed

                except Exception as exc:
                    results[task.agent_type.value] = {
                        "success": False,
                        "error": str(exc)
                    }
                    completed.add(task.agent_type)
                    running.discard(task.agent_type)
                    remaining = [t for t in remaining if t.agent_type != task.agent_type]

        # Log LAMaS optimization metrics
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        sequential_estimate = sum(
            self.latency_stats[t.agent_type].p50_latency_ms or 100
            for t in tasks
        )
        if sequential_estimate > 0:
            speedup = (sequential_estimate - elapsed_ms) / sequential_estimate * 100
            if speedup > 0:
                logger.info(
                    "LAMaS optimization: %.1fms actual vs %.1fms sequential (%.1f%% faster)",
                    elapsed_ms, sequential_estimate, speedup
                )

        return results

    async def _execute_with_tracking(
        self,
        task: AgentTask,
        is_speculative: bool = False
    ) -> Dict[str, Any]:
        """Execute agent with latency tracking for LAMaS optimization."""
        start_time = time.perf_counter()

        try:
            result = await self._execute_single_agent(task)

            # Record latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.latency_stats[task.agent_type].record(latency_ms, success=True)

            result["latency_ms"] = latency_ms
            result["speculative"] = is_speculative

            return result

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.latency_stats[task.agent_type].record(latency_ms, success=False)
            raise

    async def _maybe_cancel_speculative(
        self,
        completed_agent: AgentType,
        results: Dict[str, Any]
    ) -> None:
        """
        Cancel speculative tasks that are invalidated by a completed agent.

        For example, if planner completes with a significantly different
        approach, we might cancel speculative doc updates.
        """
        # For now, simple heuristic: don't cancel unless planner/architect changed things
        if completed_agent not in [AgentType.PLANNER, AgentType.ARCHITECT]:
            return

        result = results.get(completed_agent.value, {})
        if not result.get("success"):
            return

        # Check if result indicates significant changes
        result_data = result.get("result", {})
        if isinstance(result_data, dict) and result_data.get("major_changes"):
            # Cancel speculative doc/e2e tasks
            for task_id, async_task in list(self.speculative_tasks.items()):
                if "doc" in task_id or "e2e" in task_id:
                    async_task.cancel()
                    del self.speculative_tasks[task_id]

    def get_latency_report(self) -> Dict[str, Any]:
        """Get latency statistics for all agents."""
        return {
            agent_type.value: {
                "p50_ms": stats.p50_latency_ms,
                "p95_ms": stats.p95_latency_ms,
                "p99_ms": stats.p99_latency_ms,
                "success_rate": stats.success_rate,
                "total_executions": stats.total_executions,
            }
            for agent_type, stats in self.latency_stats.items()
            if stats.total_executions > 0
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get enhanced metrics including LAMaS-specific stats."""
        base_metrics = super().get_metrics()
        return {
            **base_metrics,
            "lamas_enabled": True,
            "speculation_enabled": self.enable_speculation,
            "speculation_budget_pct": self.speculation_budget_pct,
            "latency_stats": self.get_latency_report(),
            "active_speculative": len(self.speculative_tasks),
        }


# =============================================================================
# FIRST-RESPONSE-WINS PATTERN (Validation Optimization)
# =============================================================================

class FirstResponseWinsValidator:
    """
    For validation tasks, run multiple validators and use first success.

    LAMaS research shows this reduces p99 latency significantly for
    validation-heavy workflows.
    """

    def __init__(
        self,
        orchestrator: ProactiveAgentOrchestrator,
        validator_agents: Optional[List[AgentType]] = None
    ):
        self.orchestrator = orchestrator
        self.validator_agents = validator_agents or [
            AgentType.CODE_REVIEWER,
            AgentType.VERIFICATION_AGENT,
        ]

    async def validate_first_wins(
        self,
        description: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run validation agents in parallel, return first successful result.
        """
        tasks = []

        for agent_type in self.validator_agents:
            task = AgentTask(
                task_id=f"frw_{agent_type.value}_{time.time()}",
                agent_type=agent_type,
                description=description,
                context=context,
                model_used=TRIGGER_PATTERNS[agent_type].get("model", ModelTier.HAIKU)
            )
            tasks.append(task)

        # Create async tasks
        async_tasks = [
            asyncio.create_task(
                self.orchestrator._execute_single_agent(task)
            )
            for task in tasks
        ]

        # Return first successful result
        done, pending = await asyncio.wait(
            async_tasks,
            return_when=asyncio.FIRST_COMPLETED
        )

        # Get the first result
        first_result = None
        for completed_task in done:
            try:
                result = completed_task.result()
                if result.get("success"):
                    first_result = result
                    break
            except Exception:
                continue

        # Cancel remaining tasks
        for pending_task in pending:
            pending_task.cancel()

        # If no success, wait for all and return best effort
        if first_result is None:
            # All failed or were cancelled
            results = []
            for t in done:
                try:
                    results.append(t.result())
                except Exception:
                    pass
            first_result = results[0] if results else {"success": False, "error": "All validators failed"}

        first_result["first_response_wins"] = True
        return first_result


# =============================================================================
# GLOBAL INSTANCE (Enhanced)
# =============================================================================

_lamas_orchestrator_instance: Optional[LAMaSEnhancedOrchestrator] = None


def get_lamas_orchestrator(
    max_concurrent: int = 8,
    enable_auto_trigger: bool = True,
    enable_speculation: bool = True
) -> LAMaSEnhancedOrchestrator:
    """Get or create the global LAMaS-enhanced orchestrator instance."""
    global _lamas_orchestrator_instance
    if _lamas_orchestrator_instance is None:
        _lamas_orchestrator_instance = LAMaSEnhancedOrchestrator(
            max_concurrent=max_concurrent,
            enable_auto_trigger=enable_auto_trigger,
            enable_speculation=enable_speculation
        )
    return _lamas_orchestrator_instance


def reset_lamas_orchestrator() -> None:
    """Reset the global LAMaS orchestrator instance (for testing)."""
    global _lamas_orchestrator_instance
    _lamas_orchestrator_instance = None


# =============================================================================
# CLI INTERFACE
# =============================================================================

async def main():
    """Demo the proactive agent orchestrator."""

    orchestrator = get_orchestrator()

    print("Proactive Agent Orchestrator Demo")
    print("=" * 50)

    # Test cases
    test_cases = [
        "Implement user authentication with JWT tokens",
        "Fix the build error in the login component",
        "Refactor and clean up dead code in utils/",
        "Update the API documentation for the new endpoints",
        "Add end-to-end tests for the checkout flow",
    ]

    for description in test_cases:
        print(f"\nTask: {description}")
        print("-" * 40)

        result = await orchestrator.process_task(
            description=description,
            context={"code_changed": True}
        )

        print(f"Category: {result['analysis']['category']}")
        print(f"Triggered: {result['agents_triggered']}")
        print(f"Keywords: {result['analysis']['keywords']}")
        print(f"Confidence: {result['analysis']['confidence']:.2f}")

    print("\n" + "=" * 50)
    print("Final Metrics:", orchestrator.get_metrics())


if __name__ == "__main__":
    asyncio.run(main())
