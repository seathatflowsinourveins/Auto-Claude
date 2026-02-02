"""
P4 Integration Bridge - Connects P4 Components to UNLEASH Platform.

This module bridges the P4 components (Performance Maximizer, Metrics, Continuous Learning)
with the existing UNLEASH platform infrastructure.

P4 Components (from ~/.claude/integrations/):
- performance_maximizer.py: SemanticCache, ComplexityAnalyzer, ParallelExecutor
- metrics.py: Prometheus-compatible metrics collection
- continuous_learning.py: Pattern extraction and skill learning

UNLEASH Platform Components:
- observability.py: Unified logging, metrics, tracing
- proactive_agents.py: 10 auto-triggered agents
- unified_memory_gateway.py: 5-layer memory stack

Integration Strategy:
1. P4 metrics → Platform observability
2. P4 semantic cache → Platform caching layer
3. P4 complexity analyzer → Proactive agent routing
4. P4 continuous learning → Ralph Loop pattern extraction

Version: V1.0.0 (2026-01-29)
"""

from __future__ import annotations

import asyncio
import sys
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Add integrations directory to path
INTEGRATIONS_PATH = Path.home() / ".claude" / "integrations"
if str(INTEGRATIONS_PATH) not in sys.path:
    sys.path.insert(0, str(INTEGRATIONS_PATH))

# =============================================================================
# LAZY IMPORTS - Load P4 components on demand
# =============================================================================

_p4_performance_maximizer = None
_p4_metrics = None
_p4_continuous_learning = None


def _load_p4_components() -> Dict[str, Any]:
    """Lazy-load P4 components from ~/.claude/integrations/."""
    global _p4_performance_maximizer, _p4_metrics, _p4_continuous_learning

    loaded = {}

    try:
        if _p4_performance_maximizer is None:
            from performance_maximizer import (
                PerformanceMaximizer,
                PerformanceConfig,
                SemanticCache,
                ComplexityAnalyzer,
                SemanticToolSelector,
                ParallelExecutor,
                ContextOptimizer,
                TaskComplexity,
                ModelTier,
                get_maximizer,
            )
            _p4_performance_maximizer = {
                "PerformanceMaximizer": PerformanceMaximizer,
                "PerformanceConfig": PerformanceConfig,
                "SemanticCache": SemanticCache,
                "ComplexityAnalyzer": ComplexityAnalyzer,
                "SemanticToolSelector": SemanticToolSelector,
                "ParallelExecutor": ParallelExecutor,
                "ContextOptimizer": ContextOptimizer,
                "TaskComplexity": TaskComplexity,
                "ModelTier": ModelTier,
                "get_maximizer": get_maximizer,
            }
        loaded["performance_maximizer"] = _p4_performance_maximizer

    except ImportError as e:
        logger.warning("P4 performance_maximizer not available: %s", e)

    try:
        if _p4_metrics is None:
            from metrics import (
                MetricsRegistry,
                MetricType,
                record_mcp_request,
                record_research_query,
                metrics as p4_metrics_instance,
            )
            _p4_metrics = {
                "MetricsRegistry": MetricsRegistry,
                "MetricType": MetricType,
                "record_mcp_request": record_mcp_request,
                "record_research_query": record_research_query,
                "instance": p4_metrics_instance,
            }
        loaded["metrics"] = _p4_metrics

    except ImportError as e:
        logger.warning("P4 metrics not available: %s", e)

    try:
        if _p4_continuous_learning is None:
            from continuous_learning import (
                ContinuousLearningSystem,
                PatternDetector,
                PatternType,
                LearningConfig,
                get_learning_summary,
                evaluate_session_file,
            )
            _p4_continuous_learning = {
                "ContinuousLearningSystem": ContinuousLearningSystem,
                "PatternDetector": PatternDetector,
                "PatternType": PatternType,
                "LearningConfig": LearningConfig,
                "get_learning_summary": get_learning_summary,
                "evaluate_session_file": evaluate_session_file,
            }
        loaded["continuous_learning"] = _p4_continuous_learning

    except ImportError as e:
        logger.warning("P4 continuous_learning not available: %s", e)

    return loaded


# =============================================================================
# P4 BRIDGE CLASSES
# =============================================================================

@dataclass
class P4IntegrationConfig:
    """Configuration for P4 integration."""
    enable_semantic_cache: bool = True
    enable_complexity_routing: bool = True
    enable_parallel_execution: bool = True
    enable_metrics_bridge: bool = True
    enable_continuous_learning: bool = True

    # Performance tuning
    cache_similarity_threshold: float = 0.92
    max_parallel_agents: int = 6
    context_compression_threshold: int = 30000


class P4MetricsBridge:
    """
    Bridge between P4 metrics and platform observability.

    Forwards P4 metrics to the platform's observability layer.
    """

    def __init__(self, observability=None):
        self.observability = observability
        self._p4_metrics = None

    def connect(self) -> bool:
        """Connect to P4 metrics system."""
        components = _load_p4_components()
        if "metrics" in components:
            self._p4_metrics = components["metrics"]["instance"]
            return True
        return False

    def record_request(
        self,
        tool_name: str,
        status: str = "success",
        duration_seconds: float = 0.0
    ) -> None:
        """Record a tool request in both systems."""
        if self._p4_metrics:
            # Record in P4 metrics
            components = _load_p4_components()
            if "metrics" in components:
                components["metrics"]["record_mcp_request"](
                    tool_name, status, duration_seconds
                )

        # Also record in platform observability if available
        if self.observability:
            histogram = self.observability.histogram(
                "tool_request_duration_seconds",
                "Tool request duration",
                ["tool", "status"]
            )
            if histogram:
                histogram.observe(duration_seconds, tool=tool_name, status=status)

    def get_combined_metrics(self) -> Dict[str, Any]:
        """Get combined metrics from both systems."""
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "p4_metrics": {},
            "platform_metrics": {},
        }

        if self._p4_metrics:
            result["p4_metrics"] = self._p4_metrics.to_json()

        if self.observability:
            result["platform_metrics"] = self.observability.collect_metrics()

        return result

    def export_prometheus(self) -> str:
        """Export combined metrics in Prometheus format."""
        lines = []

        if self._p4_metrics:
            lines.append("# P4 Metrics")
            lines.append(self._p4_metrics.to_prometheus())

        if self.observability:
            lines.append("\n# Platform Metrics")
            lines.append(self.observability.export_prometheus())

        return "\n".join(lines)


class P4CacheBridge:
    """
    Bridge between P4 semantic cache and platform caching.

    Provides unified caching across both systems.
    """

    def __init__(self, config: P4IntegrationConfig):
        self.config = config
        self._p4_cache = None
        self._initialized = False

    def connect(self) -> bool:
        """Connect to P4 semantic cache."""
        if not self.config.enable_semantic_cache:
            return False

        components = _load_p4_components()
        if "performance_maximizer" in components:
            SemanticCache = components["performance_maximizer"]["SemanticCache"]
            self._p4_cache = SemanticCache(
                similarity_threshold=self.config.cache_similarity_threshold
            )
            self._initialized = True
            return True
        return False

    def get(self, query: str, embedding: Optional[List[float]] = None) -> Optional[Any]:
        """Get from cache (P4 semantic matching)."""
        if not self._initialized or not self._p4_cache:
            return None
        return self._p4_cache.get(query, embedding)

    def set(
        self,
        query: str,
        response: Any,
        embedding: Optional[List[float]] = None
    ) -> None:
        """Set in cache."""
        if not self._initialized or not self._p4_cache:
            return
        self._p4_cache.set(query, response, embedding)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._initialized or not self._p4_cache:
            return {"status": "not_initialized"}
        return self._p4_cache.get_stats()

    def clear(self) -> None:
        """Clear the cache."""
        if self._p4_cache:
            self._p4_cache.clear()


class P4ComplexityBridge:
    """
    Bridge for complexity analysis to inform agent routing.

    Uses P4 ComplexityAnalyzer to determine task complexity
    and recommend model tiers for proactive agents.
    """

    def __init__(self, config: P4IntegrationConfig):
        self.config = config
        self._analyzer = None
        self._initialized = False

    def connect(self) -> bool:
        """Connect to P4 complexity analyzer."""
        if not self.config.enable_complexity_routing:
            return False

        components = _load_p4_components()
        if "performance_maximizer" in components:
            ComplexityAnalyzer = components["performance_maximizer"]["ComplexityAnalyzer"]
            self._analyzer = ComplexityAnalyzer()
            self._initialized = True
            return True
        return False

    def analyze(self, task_description: str) -> Dict[str, Any]:
        """Analyze task complexity."""
        if not self._initialized or not self._analyzer:
            return {"complexity": "unknown", "model": "sonnet"}

        components = _load_p4_components()
        TaskComplexity = components["performance_maximizer"]["TaskComplexity"]
        PerformanceConfig = components["performance_maximizer"]["PerformanceConfig"]

        complexity = self._analyzer.analyze(task_description)
        perf_config = PerformanceConfig()
        model = self._analyzer.recommend_model(complexity, perf_config)

        return {
            "complexity": complexity.value,
            "recommended_model": model.value,
            "details": {
                "word_count": len(task_description.split()),
                "has_complex_keywords": complexity in [TaskComplexity.COMPLEX, TaskComplexity.CRITICAL],
            }
        }


class P4LearningBridge:
    """
    Bridge between P4 continuous learning and platform learning systems.

    Integrates pattern extraction with Ralph Loop and other learning components.

    Integration Points:
    1. P4 patterns → Ralph Loop experience replay buffer
    2. P4 ExtractedPattern → Ralph Loop ProceduralSkill
    3. Ralph Loop iteration hooks → P4 session evaluation
    """

    def __init__(self, config: P4IntegrationConfig):
        self.config = config
        self._learning_system = None
        self._pattern_detector = None
        self._initialized = False
        self._ralph_loop_ref = None  # Optional reference to active Ralph Loop

    def connect(self) -> bool:
        """Connect to P4 continuous learning system."""
        if not self.config.enable_continuous_learning:
            return False

        components = _load_p4_components()
        if "continuous_learning" in components:
            ContinuousLearningSystem = components["continuous_learning"]["ContinuousLearningSystem"]
            PatternDetector = components["continuous_learning"]["PatternDetector"]
            self._learning_system = ContinuousLearningSystem()
            self._pattern_detector = PatternDetector()
            self._initialized = True
            return True
        return False

    def set_ralph_loop(self, ralph_loop: Any) -> None:
        """Set reference to active Ralph Loop for bidirectional integration."""
        self._ralph_loop_ref = ralph_loop
        logger.info("P4LearningBridge connected to Ralph Loop")

    def evaluate_session(self, session_content: str, session_id: str = "") -> Dict[str, Any]:
        """Evaluate a session for learnable patterns."""
        if not self._initialized or not self._learning_system:
            return {"evaluated": False, "reason": "Learning system not initialized"}
        return self._learning_system.evaluate_session(session_content, session_id)

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        if not self._initialized or not self._learning_system:
            return {"status": "not_initialized"}
        return self._learning_system.get_learning_stats()

    def get_learned_skills(self) -> List[Dict[str, Any]]:
        """Get list of learned skills."""
        if not self._initialized or not self._learning_system:
            return []
        return self._learning_system.get_learned_skills()

    # =========================================================================
    # RALPH LOOP INTEGRATION
    # =========================================================================

    def detect_patterns(self, content: str) -> List[Dict[str, Any]]:
        """
        Detect extractable patterns from content.

        Returns patterns in a format compatible with Ralph Loop.
        """
        if not self._initialized or not self._pattern_detector:
            return []
        return self._pattern_detector.analyze_session(content)

    def patterns_to_experience_replay(
        self,
        patterns: List[Dict[str, Any]],
        context: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Convert P4 patterns to Ralph Loop experience replay format.

        Ralph Loop experience format:
        {
            "state": str,        # Context/state when pattern was learned
            "action": str,       # What was done (the pattern/solution)
            "reward": float,     # How valuable (-1 to 1)
            "next_state": str,   # Resulting state
            "timestamp": float,  # When it occurred
        }
        """
        experiences = []
        timestamp = datetime.now(timezone.utc).timestamp()

        for pattern in patterns:
            pattern_type = pattern.get("type")
            pattern_context = pattern.get("context", "")

            # Map pattern types to reward values
            reward_map = {
                "error_resolution": 0.8,      # High value - solved a problem
                "user_correction": 0.9,       # Very high - learned from feedback
                "workaround": 0.5,            # Medium - useful but hacky
                "debugging_technique": 0.7,   # Good - reusable technique
                "api_signature": 0.95,        # Critical - API accuracy
                "best_practice": 0.6,         # Good general knowledge
            }

            # Get pattern type value (handle both Enum and string types)
            if pattern_type is None:
                ptype_str = "unknown"
            elif hasattr(pattern_type, "value"):
                ptype_str = pattern_type.value
            else:
                ptype_str = str(pattern_type)
            reward = reward_map.get(ptype_str, 0.5)

            experience = {
                "state": context or "session_learning",
                "action": f"{ptype_str}: {pattern_context[:200]}",
                "reward": reward,
                "next_state": "pattern_extracted",
                "timestamp": timestamp,
                "source": "p4_continuous_learning",
                "pattern_type": ptype_str,
            }
            experiences.append(experience)

        return experiences

    def feed_to_ralph_loop(
        self,
        session_content: str,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Extract patterns from session and feed to Ralph Loop experience replay.

        This is the main integration point for continuous learning.
        """
        result = {
            "patterns_detected": 0,
            "experiences_added": 0,
            "ralph_loop_connected": self._ralph_loop_ref is not None,
        }

        # Detect patterns
        patterns = self.detect_patterns(session_content)
        result["patterns_detected"] = len(patterns)

        if not patterns:
            return result

        # Convert to experience replay format
        experiences = self.patterns_to_experience_replay(patterns, context)

        # If Ralph Loop is connected, feed experiences directly
        if self._ralph_loop_ref is not None:
            try:
                state = getattr(self._ralph_loop_ref, "state", None)
                if state and hasattr(state, "experience_replay"):
                    replay = state.experience_replay
                    if replay:
                        for exp in experiences:
                            # Priority based on reward
                            priority = exp["reward"] + 0.1
                            replay.add(exp, priority)
                            result["experiences_added"] += 1
            except Exception as e:
                logger.warning("Failed to feed to Ralph Loop: %s", e)

        return result

    def on_ralph_iteration_complete(
        self,
        iteration_result: Dict[str, Any],
        iteration_content: str = ""
    ) -> Dict[str, Any]:
        """
        Hook called after each Ralph Loop iteration.

        Evaluates the iteration for learnable patterns and stores them.
        """
        feedback = {
            "iteration": iteration_result.get("iteration", 0),
            "patterns_found": [],
            "skills_extracted": 0,
        }

        if not iteration_content:
            return feedback

        # Evaluate for patterns
        eval_result = self.evaluate_session(
            iteration_content,
            session_id=f"ralph_iter_{iteration_result.get('iteration', 0)}"
        )

        if eval_result.get("evaluated"):
            patterns = eval_result.get("patterns", [])
            feedback["patterns_found"] = [
                {"type": p.get("type", "unknown"), "context": p.get("context", "")[:100]}
                for p in patterns
            ]

            # Feed back to Ralph Loop
            feed_result = self.feed_to_ralph_loop(iteration_content)
            feedback["experiences_added"] = feed_result.get("experiences_added", 0)

        return feedback

    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of Ralph Loop integration."""
        return {
            "p4_initialized": self._initialized,
            "pattern_detector_ready": self._pattern_detector is not None,
            "ralph_loop_connected": self._ralph_loop_ref is not None,
            "learning_system_ready": self._learning_system is not None,
        }


# =============================================================================
# UNIFIED P4 INTEGRATION
# =============================================================================

class P4Integration:
    """
    Unified P4 integration layer for the UNLEASH platform.

    Provides single entry point for all P4 component interactions.
    """

    _instance: Optional["P4Integration"] = None

    def __new__(cls, config: Optional[P4IntegrationConfig] = None) -> "P4Integration":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[P4IntegrationConfig] = None):
        if self._initialized:
            return

        self.config = config or P4IntegrationConfig()

        # Initialize bridges
        self.metrics = P4MetricsBridge()
        self.cache = P4CacheBridge(self.config)
        self.complexity = P4ComplexityBridge(self.config)
        self.learning = P4LearningBridge(self.config)

        # Track connection status
        self._connections: Dict[str, bool] = {}

        self._initialized = True
        logger.info("P4Integration initialized")

    def connect_all(self, observability=None) -> Dict[str, bool]:
        """Connect all P4 bridges."""
        self._connections = {
            "metrics": self.metrics.connect(),
            "cache": self.cache.connect(),
            "complexity": self.complexity.connect(),
            "learning": self.learning.connect(),
        }

        if observability:
            self.metrics.observability = observability

        logger.info("P4 connections: %s", self._connections)
        return self._connections

    def get_status(self) -> Dict[str, Any]:
        """Get P4 integration status."""
        return {
            "initialized": self._initialized,
            "connections": self._connections,
            "config": {
                "semantic_cache": self.config.enable_semantic_cache,
                "complexity_routing": self.config.enable_complexity_routing,
                "parallel_execution": self.config.enable_parallel_execution,
                "metrics_bridge": self.config.enable_metrics_bridge,
                "continuous_learning": self.config.enable_continuous_learning,
            },
            "cache_stats": self.cache.get_stats(),
            "learning_stats": self.learning.get_learning_stats(),
        }

    async def optimize_request(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize a request through P4 components.

        1. Check semantic cache
        2. Analyze complexity
        3. Return optimized context
        """
        result = {
            "query": query,
            "optimizations_applied": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Check cache first
        if self._connections.get("cache"):
            cached = self.cache.get(query)
            if cached:
                result["cached_response"] = cached
                result["optimizations_applied"].append("semantic_cache_hit")
                return result

        # Analyze complexity
        if self._connections.get("complexity"):
            analysis = self.complexity.analyze(query)
            result["complexity_analysis"] = analysis
            result["optimizations_applied"].append("complexity_analysis")

        # Optimize context if provided
        if context:
            result["optimized_context"] = context
            result["optimizations_applied"].append("context_passed")

        return result

    def cache_response(self, query: str, response: Any) -> None:
        """Cache a response for future queries."""
        if self._connections.get("cache"):
            self.cache.set(query, response)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_p4_instance: Optional[P4Integration] = None


def get_p4_integration(
    config: Optional[P4IntegrationConfig] = None
) -> P4Integration:
    """Get or create the global P4 integration instance."""
    global _p4_instance
    if _p4_instance is None:
        _p4_instance = P4Integration(config)
    return _p4_instance


def reset_p4_integration() -> None:
    """Reset the global P4 integration instance."""
    global _p4_instance
    _p4_instance = None
    P4Integration._instance = None


async def initialize_p4_with_platform() -> Dict[str, Any]:
    """
    Initialize P4 integration with platform components.

    Call this from the platform startup to enable P4 features.
    """
    try:
        # Import platform observability
        from .observability import get_observability
        observability = get_observability()
    except ImportError:
        observability = None
        logger.warning("Platform observability not available")

    # Initialize P4
    p4 = get_p4_integration()
    connections = p4.connect_all(observability)

    return {
        "p4_status": p4.get_status(),
        "connections": connections,
        "observability_available": observability is not None,
    }


# =============================================================================
# CLI INTERFACE
# =============================================================================

async def main():
    """Demo the P4 integration."""

    print("P4 Integration Demo")
    print("=" * 50)

    # Initialize
    result = await initialize_p4_with_platform()

    print("\n1. Initialization Status:")
    print(f"   Connections: {result['connections']}")

    # Get P4 instance
    p4 = get_p4_integration()

    # Test complexity analysis
    print("\n2. Complexity Analysis:")
    test_queries = [
        "What is Python?",
        "Implement user authentication with JWT",
        "Critical production security audit needed",
    ]

    for query in test_queries:
        analysis = p4.complexity.analyze(query)
        print(f"   '{query[:40]}...' → {analysis['complexity']} ({analysis['recommended_model']})")

    # Test caching
    print("\n3. Semantic Cache:")
    p4.cache_response("test query", {"answer": "test response"})
    cached = p4.cache.get("test query")
    print(f"   Cache set/get: {'OK' if cached else 'MISS'}")
    print(f"   Stats: {p4.cache.get_stats()}")

    # Full status
    print("\n4. Full Status:")
    status = p4.get_status()
    print(f"   Initialized: {status['initialized']}")
    print(f"   Connections: {status['connections']}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
