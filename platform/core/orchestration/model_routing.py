"""
3-Tier Model Routing - Gap21 Implementation (V67)
==================================================

Unified model routing based on ADR-026 specifications:
- TIER 1 (WASM/Haiku): <1ms, $0-$0.0002 - Simple transforms, exploration
- TIER 2 (Sonnet): ~2s, $0.003 - Multi-file implementation, moderate complexity
- TIER 3 (Opus): ~5s, $0.015 - Security, architecture, complex reasoning

Wires the QueryComplexityAnalyzer to actual model selection for:
- RAG pipeline routing
- LLM generation routing
- Batch processing with cost optimization

Integration:
    from core.orchestration.model_routing import (
        ModelRouter,
        RoutingConfig,
        route_query,
        get_claude_model_for_task,
    )

    # Create router
    router = ModelRouter()

    # Route a query
    decision = router.route("Design a microservices architecture")
    print(f"Model: {decision.model_id}")  # claude-opus-4-5-20251101
    print(f"Tier: {decision.tier}")       # 3
    print(f"Est. cost: ${decision.estimated_cost:.4f}")

    # Quick function for simple routing
    model = get_claude_model_for_task("Fix this typo")
    # Returns: claude-3-5-haiku-20241022

Reference: ADR-026-agent-booster-model-routing.md
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ..rag.complexity_analyzer import QueryAnalysis

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class RoutingTier(int, Enum):
    """Model routing tiers per ADR-026."""
    TIER_1 = 1  # WASM/Haiku - Simple transforms, exploration (~70% of tasks)
    TIER_2 = 2  # Sonnet - Multi-file, moderate complexity (~25% of tasks)
    TIER_3 = 3  # Opus - Security, architecture, complex (~5% of tasks)


class ModelProvider(str, Enum):
    """Supported model providers."""
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # For WASM/Agent Booster transforms


# Model configurations per ADR-026 and model_router.py
MODEL_CONFIGS = {
    # Tier 1: Fast, cheap (or free for WASM)
    "wasm": {
        "tier": RoutingTier.TIER_1,
        "provider": ModelProvider.LOCAL,
        "latency_ms": 1,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "model_id": "agent-booster-wasm",
        "context_window": 4096,
        "max_output": 4096,
    },
    "claude-haiku": {
        "tier": RoutingTier.TIER_1,
        "provider": ModelProvider.ANTHROPIC,
        "latency_ms": 500,
        "cost_per_1k_input": 0.00025,
        "cost_per_1k_output": 0.00125,
        "model_id": "claude-3-5-haiku-20241022",
        "context_window": 200000,
        "max_output": 8192,
    },
    # Tier 2: Balanced
    "claude-sonnet": {
        "tier": RoutingTier.TIER_2,
        "provider": ModelProvider.ANTHROPIC,
        "latency_ms": 2000,
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "model_id": "claude-sonnet-4-20250514",
        "context_window": 200000,
        "max_output": 16384,
    },
    # Tier 3: Premium
    "claude-opus": {
        "tier": RoutingTier.TIER_3,
        "provider": ModelProvider.ANTHROPIC,
        "latency_ms": 5000,
        "cost_per_1k_input": 0.015,
        "cost_per_1k_output": 0.075,
        "model_id": "claude-opus-4-5-20251101",
        "context_window": 200000,
        "max_output": 16384,
    },
}

# Task patterns for quick classification (from ADR-026)
TIER_1_PATTERNS = [
    "var-to-const", "add-types", "add-error-handling", "async-await",
    "add-logging", "remove-console", "fix typo", "format", "lint",
    "find", "search", "where is", "list files", "show me", "glob",
    "grep", "explore", "look for", "check if", "read file", "show content",
    "what files", "directory", "quick", "simple", "just", "only", "single file",
]

TIER_3_PATTERNS = [
    "architecture", "microservices", "distributed", "system design",
    "oauth", "pkce", "jwt", "rbac", "authentication system", "security",
    "consensus", "byzantine", "raft", "paxos", "machine learning",
    "neural", "optimization", "schema design", "data model", "normalization",
    "low latency", "high throughput", "concurrent", "vulnerability",
    "penetration", "pentest", "audit", "critical", "production",
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RoutingConfig:
    """Configuration for model routing.

    Attributes:
        enable_wasm: Allow WASM/Agent Booster for Tier 1 (requires agentic-flow)
        prefer_cost: Prioritize cost savings over quality
        prefer_quality: Prioritize quality over cost
        haiku_threshold: Complexity score below this uses Haiku (default: 0.3)
        sonnet_threshold: Complexity score below this uses Sonnet (default: 0.6)
        enable_batch: Enable Anthropic Batch API for 50% cost savings
        batch_threshold: Minimum requests to trigger batch mode
        force_tier: Force a specific tier (for testing)
    """
    enable_wasm: bool = False  # Disabled by default (requires setup)
    prefer_cost: bool = False
    prefer_quality: bool = False
    haiku_threshold: float = 0.30
    sonnet_threshold: float = 0.60
    enable_batch: bool = True
    batch_threshold: int = 10
    force_tier: Optional[RoutingTier] = None


@dataclass
class RoutingDecision:
    """Result of model routing decision.

    Attributes:
        tier: The routing tier (1, 2, or 3)
        model_key: Internal model key (e.g., "claude-haiku")
        model_id: API model identifier (e.g., "claude-3-5-haiku-20241022")
        provider: Model provider (anthropic, local)
        complexity_score: The analyzed complexity score (0-1)
        complexity_level: The complexity level string
        estimated_latency_ms: Estimated latency in milliseconds
        estimated_cost: Estimated cost in USD
        reason: Human-readable explanation
        can_use_batch: Whether this can be batched for cost savings
        is_wasm_eligible: Whether WASM/Agent Booster can handle this
        wasm_intent: The detected WASM intent type if applicable
        timestamp: When the decision was made
    """
    tier: RoutingTier
    model_key: str
    model_id: str
    provider: ModelProvider
    complexity_score: float
    complexity_level: str
    estimated_latency_ms: int
    estimated_cost: float
    reason: str
    can_use_batch: bool = True
    is_wasm_eligible: bool = False
    wasm_intent: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tier": self.tier.value,
            "model_key": self.model_key,
            "model_id": self.model_id,
            "provider": self.provider.value,
            "complexity_score": round(self.complexity_score, 4),
            "complexity_level": self.complexity_level,
            "estimated_latency_ms": self.estimated_latency_ms,
            "estimated_cost": round(self.estimated_cost, 6),
            "reason": self.reason,
            "can_use_batch": self.can_use_batch,
            "is_wasm_eligible": self.is_wasm_eligible,
            "wasm_intent": self.wasm_intent,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RoutingMetrics:
    """Metrics tracking for routing decisions."""
    total_routes: int = 0
    tier_1_count: int = 0
    tier_2_count: int = 0
    tier_3_count: int = 0
    wasm_count: int = 0
    batch_eligible_count: int = 0
    total_estimated_cost: float = 0.0
    total_estimated_latency_ms: int = 0

    def record(self, decision: RoutingDecision) -> None:
        """Record a routing decision."""
        self.total_routes += 1

        if decision.tier == RoutingTier.TIER_1:
            self.tier_1_count += 1
        elif decision.tier == RoutingTier.TIER_2:
            self.tier_2_count += 1
        else:
            self.tier_3_count += 1

        if decision.is_wasm_eligible:
            self.wasm_count += 1
        if decision.can_use_batch:
            self.batch_eligible_count += 1

        self.total_estimated_cost += decision.estimated_cost
        self.total_estimated_latency_ms += decision.estimated_latency_ms

    def get_distribution(self) -> Dict[str, float]:
        """Get tier distribution percentages."""
        if self.total_routes == 0:
            return {"tier_1": 0.0, "tier_2": 0.0, "tier_3": 0.0}
        return {
            "tier_1": self.tier_1_count / self.total_routes * 100,
            "tier_2": self.tier_2_count / self.total_routes * 100,
            "tier_3": self.tier_3_count / self.total_routes * 100,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        dist = self.get_distribution()
        return {
            "total_routes": self.total_routes,
            "by_tier": {
                "tier_1": self.tier_1_count,
                "tier_2": self.tier_2_count,
                "tier_3": self.tier_3_count,
            },
            "distribution_percent": dist,
            "wasm_eligible": self.wasm_count,
            "batch_eligible": self.batch_eligible_count,
            "total_estimated_cost_usd": round(self.total_estimated_cost, 4),
            "total_estimated_latency_ms": self.total_estimated_latency_ms,
            "avg_cost_per_route": (
                round(self.total_estimated_cost / self.total_routes, 6)
                if self.total_routes > 0 else 0
            ),
        }


# =============================================================================
# MODEL ROUTER
# =============================================================================

class ModelRouter:
    """
    3-Tier Model Router implementing ADR-026 specifications.

    Routes queries to optimal Claude models based on complexity analysis:
    - Tier 1 (Haiku/WASM): 70% of tasks - exploration, simple edits
    - Tier 2 (Sonnet): 25% of tasks - multi-file, moderate complexity
    - Tier 3 (Opus): 5% of tasks - security, architecture, complex reasoning

    Example:
        router = ModelRouter()

        # Route based on task description
        decision = router.route("Fix this typo in config.py")
        print(f"Use: {decision.model_id}")  # claude-3-5-haiku-20241022

        decision = router.route("Design a distributed consensus algorithm")
        print(f"Use: {decision.model_id}")  # claude-opus-4-5-20251101

        # Get metrics
        print(router.get_metrics())
    """

    def __init__(self, config: Optional[RoutingConfig] = None):
        self.config = config or RoutingConfig()
        self._metrics = RoutingMetrics()
        self._analyzer: Optional["QueryComplexityAnalyzer"] = None

    def _get_analyzer(self) -> "QueryComplexityAnalyzer":
        """Lazy load the complexity analyzer."""
        if self._analyzer is None:
            from ..rag.complexity_analyzer import QueryComplexityAnalyzer
            self._analyzer = QueryComplexityAnalyzer()
        return self._analyzer

    def route(
        self,
        task: str,
        context: Optional[str] = None,
        estimated_input_tokens: int = 1000,
        estimated_output_tokens: int = 2000,
    ) -> RoutingDecision:
        """
        Route a task to the optimal model tier.

        Args:
            task: Task description or query
            context: Optional additional context
            estimated_input_tokens: Estimated input tokens
            estimated_output_tokens: Estimated output tokens

        Returns:
            RoutingDecision with model selection and metadata
        """
        # Check for forced tier
        if self.config.force_tier is not None:
            return self._create_decision_for_tier(
                self.config.force_tier,
                task,
                0.5,  # Default complexity
                "forced",
                estimated_input_tokens,
                estimated_output_tokens,
            )

        # Quick pattern matching for obvious cases
        task_lower = task.lower()

        # Check for Tier 1 patterns (simple tasks)
        if self._matches_patterns(task_lower, TIER_1_PATTERNS):
            # Check if WASM eligible
            wasm_intent = self._detect_wasm_intent(task_lower)
            if wasm_intent and self.config.enable_wasm:
                decision = self._create_wasm_decision(
                    task, wasm_intent, estimated_input_tokens, estimated_output_tokens
                )
            else:
                decision = self._create_decision_for_tier(
                    RoutingTier.TIER_1,
                    task,
                    0.15,
                    f"Pattern match: simple task",
                    estimated_input_tokens,
                    estimated_output_tokens,
                )
            self._metrics.record(decision)
            return decision

        # Check for Tier 3 patterns (complex tasks)
        if self._matches_patterns(task_lower, TIER_3_PATTERNS):
            decision = self._create_decision_for_tier(
                RoutingTier.TIER_3,
                task,
                0.85,
                f"Pattern match: complex/security/architecture task",
                estimated_input_tokens,
                estimated_output_tokens,
            )
            self._metrics.record(decision)
            return decision

        # Use full complexity analysis for ambiguous cases
        full_text = f"{task} {context or ''}"
        analyzer = self._get_analyzer()
        analysis = analyzer.analyze(full_text)

        # Map complexity to tier
        tier = self._complexity_to_tier(analysis.complexity_score, analysis)

        decision = self._create_decision_for_tier(
            tier,
            task,
            analysis.complexity_score,
            (
                f"Complexity {analysis.complexity_level.value} "
                f"({analysis.complexity_score:.2f}), "
                f"question_type={analysis.question_type.value}, "
                f"domain={analysis.domain.value}"
            ),
            estimated_input_tokens,
            estimated_output_tokens,
        )

        self._metrics.record(decision)
        return decision

    def _matches_patterns(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any pattern."""
        return any(p in text for p in patterns)

    def _detect_wasm_intent(self, text: str) -> Optional[str]:
        """Detect if task can be handled by WASM/Agent Booster."""
        wasm_intents = {
            "var-to-const": ["var to const", "var to let", "convert var"],
            "add-types": ["add types", "add type annotations", "typescript types"],
            "add-error-handling": ["add try catch", "error handling", "wrap in try"],
            "async-await": ["async await", "convert to async", "callback to async"],
            "add-logging": ["add logging", "add console.log", "add print"],
            "remove-console": ["remove console", "remove log", "clean up logs"],
        }

        for intent, triggers in wasm_intents.items():
            if any(t in text for t in triggers):
                return intent
        return None

    def _complexity_to_tier(
        self,
        score: float,
        analysis: Optional["QueryAnalysis"] = None,
    ) -> RoutingTier:
        """Map complexity score to routing tier."""
        # Security/architecture always goes to Tier 3
        if analysis:
            from ..rag.complexity_analyzer import DomainType
            if analysis.domain in (DomainType.LEGAL, DomainType.MEDICINE):
                return RoutingTier.TIER_3
            if analysis.requires_reasoning and score > 0.4:
                return RoutingTier.TIER_3

        # Apply thresholds
        if score < self.config.haiku_threshold:
            return RoutingTier.TIER_1
        elif score < self.config.sonnet_threshold:
            return RoutingTier.TIER_2
        else:
            return RoutingTier.TIER_3

    def _create_wasm_decision(
        self,
        task: str,
        wasm_intent: str,
        input_tokens: int,
        output_tokens: int,
    ) -> RoutingDecision:
        """Create a WASM/Agent Booster routing decision."""
        config = MODEL_CONFIGS["wasm"]

        return RoutingDecision(
            tier=RoutingTier.TIER_1,
            model_key="wasm",
            model_id=config["model_id"],
            provider=config["provider"],
            complexity_score=0.05,
            complexity_level="trivial",
            estimated_latency_ms=config["latency_ms"],
            estimated_cost=0.0,
            reason=f"WASM Agent Booster can handle '{wasm_intent}' - 352x faster, $0",
            can_use_batch=False,  # WASM doesn't need batching
            is_wasm_eligible=True,
            wasm_intent=wasm_intent,
        )

    def _create_decision_for_tier(
        self,
        tier: RoutingTier,
        task: str,
        complexity_score: float,
        reason: str,
        input_tokens: int,
        output_tokens: int,
    ) -> RoutingDecision:
        """Create routing decision for a specific tier."""
        # Select model for tier
        if tier == RoutingTier.TIER_1:
            model_key = "claude-haiku"
        elif tier == RoutingTier.TIER_2:
            model_key = "claude-sonnet"
        else:
            model_key = "claude-opus"

        config = MODEL_CONFIGS[model_key]

        # Calculate estimated cost
        cost = (
            (input_tokens / 1000) * config["cost_per_1k_input"] +
            (output_tokens / 1000) * config["cost_per_1k_output"]
        )

        # Complexity level string
        if complexity_score < 0.25:
            complexity_level = "low"
        elif complexity_score < 0.5:
            complexity_level = "medium"
        elif complexity_score < 0.75:
            complexity_level = "high"
        else:
            complexity_level = "very_high"

        return RoutingDecision(
            tier=tier,
            model_key=model_key,
            model_id=config["model_id"],
            provider=config["provider"],
            complexity_score=complexity_score,
            complexity_level=complexity_level,
            estimated_latency_ms=config["latency_ms"],
            estimated_cost=cost,
            reason=reason,
            can_use_batch=tier != RoutingTier.TIER_1,  # Batch for non-trivial
            is_wasm_eligible=False,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get routing metrics."""
        return self._metrics.to_dict()

    def reset_metrics(self) -> None:
        """Reset routing metrics."""
        self._metrics = RoutingMetrics()

    def get_model_id(self, tier: RoutingTier) -> str:
        """Get model ID for a specific tier."""
        if tier == RoutingTier.TIER_1:
            return MODEL_CONFIGS["claude-haiku"]["model_id"]
        elif tier == RoutingTier.TIER_2:
            return MODEL_CONFIGS["claude-sonnet"]["model_id"]
        else:
            return MODEL_CONFIGS["claude-opus"]["model_id"]


# =============================================================================
# BATCH ROUTER FOR COST OPTIMIZATION
# =============================================================================

class BatchModelRouter:
    """
    Batch-aware model router for Anthropic Message Batches API integration.

    Groups requests by tier and batches compatible ones for 50% cost savings.

    Example:
        batch_router = BatchModelRouter()

        # Add tasks for batching
        batch_router.add_task("Fix typo", "task-1")
        batch_router.add_task("Design architecture", "task-2")
        batch_router.add_task("Simple edit", "task-3")

        # Get batch groups
        batches = batch_router.get_batch_groups()
        # Returns: {
        #     "tier_1": [("Fix typo", "task-1"), ("Simple edit", "task-3")],
        #     "tier_3": [("Design architecture", "task-2")],
        # }
    """

    def __init__(
        self,
        router: Optional[ModelRouter] = None,
        min_batch_size: int = 10,
    ):
        self.router = router or ModelRouter()
        self.min_batch_size = min_batch_size
        self._pending_tasks: Dict[RoutingTier, List[Tuple[str, str, RoutingDecision]]] = {
            RoutingTier.TIER_1: [],
            RoutingTier.TIER_2: [],
            RoutingTier.TIER_3: [],
        }

    def add_task(
        self,
        task: str,
        custom_id: str,
        context: Optional[str] = None,
    ) -> RoutingDecision:
        """Add a task for potential batching."""
        decision = self.router.route(task, context)

        if decision.can_use_batch:
            self._pending_tasks[decision.tier].append((task, custom_id, decision))

        return decision

    def get_batch_groups(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get tasks grouped by tier for batching."""
        groups = {}

        for tier, tasks in self._pending_tasks.items():
            if len(tasks) >= self.min_batch_size:
                tier_key = f"tier_{tier.value}"
                model_id = self.router.get_model_id(tier)

                groups[tier_key] = {
                    "model_id": model_id,
                    "count": len(tasks),
                    "estimated_savings": self._calculate_batch_savings(tasks),
                    "tasks": [
                        {
                            "custom_id": custom_id,
                            "task": task,
                            "complexity_score": decision.complexity_score,
                        }
                        for task, custom_id, decision in tasks
                    ],
                }

        return groups

    def _calculate_batch_savings(
        self,
        tasks: List[Tuple[str, str, RoutingDecision]],
    ) -> float:
        """Calculate cost savings from batching (50% discount)."""
        total_cost = sum(d.estimated_cost for _, _, d in tasks)
        return total_cost * 0.5  # 50% savings

    def clear_pending(self, tier: Optional[RoutingTier] = None) -> None:
        """Clear pending tasks."""
        if tier:
            self._pending_tasks[tier] = []
        else:
            for t in RoutingTier:
                self._pending_tasks[t] = []

    def get_pending_counts(self) -> Dict[str, int]:
        """Get counts of pending tasks per tier."""
        return {
            f"tier_{tier.value}": len(tasks)
            for tier, tasks in self._pending_tasks.items()
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global router instance for convenience functions
_global_router: Optional[ModelRouter] = None


def get_router() -> ModelRouter:
    """Get or create the global model router."""
    global _global_router
    if _global_router is None:
        _global_router = ModelRouter()
    return _global_router


def route_query(
    task: str,
    context: Optional[str] = None,
) -> RoutingDecision:
    """
    Route a query to the optimal model.

    Args:
        task: Task description
        context: Optional context

    Returns:
        RoutingDecision with model selection
    """
    return get_router().route(task, context)


def get_claude_model_for_task(
    task: str,
    context: Optional[str] = None,
) -> str:
    """
    Get the Claude model ID for a task.

    This is the main entry point for simple routing.

    Args:
        task: Task description
        context: Optional context

    Returns:
        Model ID string (e.g., "claude-3-5-haiku-20241022")

    Example:
        model = get_claude_model_for_task("Fix this typo")
        # Returns: "claude-3-5-haiku-20241022"

        model = get_claude_model_for_task("Design secure auth system")
        # Returns: "claude-opus-4-5-20251101"
    """
    decision = get_router().route(task, context)
    return decision.model_id


def get_tier_for_task(
    task: str,
    context: Optional[str] = None,
) -> int:
    """
    Get the routing tier (1, 2, or 3) for a task.

    Args:
        task: Task description
        context: Optional context

    Returns:
        Tier number (1, 2, or 3)
    """
    decision = get_router().route(task, context)
    return decision.tier.value


def get_routing_metrics() -> Dict[str, Any]:
    """Get global routing metrics."""
    return get_router().get_metrics()


def reset_routing_metrics() -> None:
    """Reset global routing metrics."""
    get_router().reset_metrics()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    "ModelRouter",
    "BatchModelRouter",
    "RoutingConfig",
    "RoutingDecision",
    "RoutingMetrics",
    # Enums
    "RoutingTier",
    "ModelProvider",
    # Constants
    "MODEL_CONFIGS",
    "TIER_1_PATTERNS",
    "TIER_3_PATTERNS",
    # Convenience functions
    "route_query",
    "get_claude_model_for_task",
    "get_tier_for_task",
    "get_routing_metrics",
    "reset_routing_metrics",
    "get_router",
]
