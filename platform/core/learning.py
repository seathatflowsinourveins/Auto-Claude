"""
Continuous Learning V2 - Instinct-Based Pattern Extraction

This module implements the everything-claude-code pattern for continuous learning:
1. Automatic pattern extraction from successful completions
2. Confidence scoring (0.0-1.0) with decay over time
3. pass@k / pass^k metrics for capability vs reliability measurement
4. Instinct-based learning that improves agent performance over sessions

Key Concepts:
- pass@k: At least 1 of k attempts succeeds (measures capability)
- pass^k: ALL k attempts succeed (measures reliability)
- Instinct: A learned pattern with confidence score that decays without reinforcement

Usage:
    from platform.core.learning import LearningEngine, Pattern

    engine = LearningEngine()

    # Record a successful pattern
    engine.record_success(
        task_type="api_integration",
        pattern="Use Context7 + Exa + Tavily in parallel",
        context={"sdk": "letta", "method": "agents.messages.create"}
    )

    # Query for relevant patterns
    patterns = engine.get_patterns("letta sdk api call")
    for p in patterns:
        print(f"{p.pattern} (confidence: {p.confidence:.2f})")
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import hashlib
import json
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# Core Types
# =============================================================================

class PatternType(Enum):
    """Types of learned patterns."""
    API_SIGNATURE = "api_signature"      # Correct API usage
    ERROR_FIX = "error_fix"              # How to fix specific errors
    ARCHITECTURE = "architecture"         # Architectural decisions
    TOOL_USAGE = "tool_usage"            # When to use which tool
    WORKFLOW = "workflow"                # Multi-step workflows
    OPTIMIZATION = "optimization"        # Performance patterns
    SECURITY = "security"                # Security best practices
    CODE_PATTERN = "code_pattern"        # Coding patterns


@dataclass
class Pattern:
    """
    A learned pattern with confidence scoring.

    Confidence decays over time without reinforcement,
    modeling the "forgetting curve" of unused knowledge.
    """
    id: str
    pattern_type: PatternType
    pattern: str                         # The actual learned pattern
    context: Dict[str, Any]              # Context where pattern applies
    confidence: float = 0.5              # Initial confidence (0.0-1.0)
    success_count: int = 0               # Number of successful uses
    failure_count: int = 0               # Number of failed uses
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    last_reinforced: datetime = field(default_factory=datetime.now)
    tags: Set[str] = field(default_factory=set)

    def reinforce(self, success: bool = True):
        """
        Reinforce the pattern based on usage outcome.

        Success increases confidence, failure decreases it.
        Uses a bounded update formula to prevent extremes.
        """
        self.last_used = datetime.now()

        if success:
            self.success_count += 1
            self.last_reinforced = datetime.now()
            # Increase confidence, bounded by 0.99
            self.confidence = min(0.99, self.confidence + (1 - self.confidence) * 0.1)
        else:
            self.failure_count += 1
            # Decrease confidence, bounded by 0.01
            self.confidence = max(0.01, self.confidence - self.confidence * 0.2)

    def decay(self, days_since_use: float) -> float:
        """
        Calculate decayed confidence based on time since last use.

        Uses exponential decay with half-life of 30 days.
        """
        half_life = 30.0  # Days
        decay_factor = math.pow(0.5, days_since_use / half_life)
        return self.confidence * decay_factor

    @property
    def current_confidence(self) -> float:
        """Get confidence with time decay applied."""
        days_since = (datetime.now() - self.last_reinforced).days
        return self.decay(days_since)

    @property
    def reliability(self) -> float:
        """
        Calculate reliability score (pass^k metric).

        Higher reliability means consistent success.
        """
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Unknown reliability
        return self.success_count / total

    def to_dict(self) -> Dict[str, Any]:
        """Serialize pattern to dictionary."""
        return {
            "id": self.id,
            "pattern_type": self.pattern_type.value,
            "pattern": self.pattern,
            "context": self.context,
            "confidence": self.confidence,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat(),
            "last_reinforced": self.last_reinforced.isoformat(),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pattern":
        """Deserialize pattern from dictionary."""
        return cls(
            id=data["id"],
            pattern_type=PatternType(data["pattern_type"]),
            pattern=data["pattern"],
            context=data.get("context", {}),
            confidence=data.get("confidence", 0.5),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            last_used=datetime.fromisoformat(data.get("last_used", datetime.now().isoformat())),
            last_reinforced=datetime.fromisoformat(data.get("last_reinforced", datetime.now().isoformat())),
            tags=set(data.get("tags", [])),
        )


# =============================================================================
# Pass@k Metrics
# =============================================================================

@dataclass
class TaskAttempt:
    """Record of a task execution attempt."""
    task_id: str
    task_type: str
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: int = 0
    error: Optional[str] = None
    patterns_used: List[str] = field(default_factory=list)


class PassMetrics:
    """
    Track pass@k and pass^k metrics for capability and reliability.

    pass@k = P(at least 1 of k attempts succeeds) - measures CAPABILITY
    pass^k = P(all k attempts succeed) - measures RELIABILITY

    Target requirements (from verification protocol):
    - Standard features: pass@3 = 100%
    - Critical paths: pass^3 = 100%
    """

    def __init__(self):
        self._attempts: Dict[str, List[TaskAttempt]] = {}

    def record_attempt(self, attempt: TaskAttempt):
        """Record a task execution attempt."""
        if attempt.task_type not in self._attempts:
            self._attempts[attempt.task_type] = []
        self._attempts[attempt.task_type].append(attempt)

    def pass_at_k(self, task_type: str, k: int = 3) -> float:
        """
        Calculate pass@k metric (capability).

        Returns probability that at least 1 of k attempts succeeds.
        """
        attempts = self._attempts.get(task_type, [])
        if len(attempts) < k:
            return 0.0  # Not enough data

        # Use sliding window of size k
        windows_passed = 0
        total_windows = len(attempts) - k + 1

        for i in range(total_windows):
            window = attempts[i:i + k]
            if any(a.success for a in window):
                windows_passed += 1

        return windows_passed / total_windows if total_windows > 0 else 0.0

    def pass_power_k(self, task_type: str, k: int = 3) -> float:
        """
        Calculate pass^k metric (reliability).

        Returns probability that ALL k attempts succeed.
        """
        attempts = self._attempts.get(task_type, [])
        if len(attempts) < k:
            return 0.0  # Not enough data

        # Use sliding window of size k
        windows_passed = 0
        total_windows = len(attempts) - k + 1

        for i in range(total_windows):
            window = attempts[i:i + k]
            if all(a.success for a in window):
                windows_passed += 1

        return windows_passed / total_windows if total_windows > 0 else 0.0

    def get_summary(self, task_type: str) -> Dict[str, Any]:
        """Get summary metrics for a task type."""
        attempts = self._attempts.get(task_type, [])
        if not attempts:
            return {"task_type": task_type, "total_attempts": 0}

        successes = sum(1 for a in attempts if a.success)
        return {
            "task_type": task_type,
            "total_attempts": len(attempts),
            "successes": successes,
            "success_rate": successes / len(attempts),
            "pass@3": self.pass_at_k(task_type, 3),
            "pass^3": self.pass_power_k(task_type, 3),
            "avg_duration_ms": sum(a.duration_ms for a in attempts) / len(attempts),
        }


# =============================================================================
# Learning Engine
# =============================================================================

class LearningEngine:
    """
    Continuous Learning V2 Engine.

    Implements instinct-based pattern extraction with confidence scoring
    and automatic decay for unused patterns.

    Features:
    - Pattern extraction from successful completions
    - Confidence scoring with time-based decay
    - pass@k metrics for capability tracking
    - pass^k metrics for reliability tracking
    - Persistent storage for cross-session learning
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        min_confidence_threshold: float = 0.3,
        auto_save: bool = True,
    ):
        """
        Initialize the learning engine.

        Args:
            storage_path: Path for persistent storage
            min_confidence_threshold: Minimum confidence to return patterns
            auto_save: Automatically save after updates
        """
        self.storage_path = storage_path or Path.home() / ".claude" / "learning" / "patterns.json"
        self.min_confidence_threshold = min_confidence_threshold
        self.auto_save = auto_save

        self._patterns: Dict[str, Pattern] = {}
        self._metrics = PassMetrics()
        self._keyword_index: Dict[str, Set[str]] = {}  # keyword -> pattern_ids

        # Load existing patterns
        self._load()

    def _generate_id(self, pattern: str, context: Dict[str, Any]) -> str:
        """Generate unique ID for a pattern."""
        content = f"{pattern}:{json.dumps(context, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _index_pattern(self, pattern: Pattern):
        """Index pattern by keywords for fast retrieval."""
        # Extract keywords from pattern and context
        text = f"{pattern.pattern} {json.dumps(pattern.context)}"
        keywords = set(text.lower().split())

        for keyword in keywords:
            if len(keyword) >= 3:  # Skip very short words
                if keyword not in self._keyword_index:
                    self._keyword_index[keyword] = set()
                self._keyword_index[keyword].add(pattern.id)

    def record_success(
        self,
        task_type: str,
        pattern: str,
        context: Optional[Dict[str, Any]] = None,
        pattern_type: PatternType = PatternType.CODE_PATTERN,
        tags: Optional[Set[str]] = None,
    ) -> Pattern:
        """
        Record a successful pattern.

        Call this when a task completes successfully to reinforce
        the patterns that led to success.

        Args:
            task_type: Type of task (e.g., "api_integration")
            pattern: The pattern/approach that worked
            context: Additional context (e.g., {"sdk": "letta"})
            pattern_type: Category of pattern
            tags: Optional tags for categorization

        Returns:
            The created or updated Pattern
        """
        context = context or {}
        tags = tags or set()

        pattern_id = self._generate_id(pattern, context)

        if pattern_id in self._patterns:
            # Reinforce existing pattern
            existing = self._patterns[pattern_id]
            existing.reinforce(success=True)
            logger.info(f"Reinforced pattern {pattern_id}: confidence={existing.confidence:.2f}")
        else:
            # Create new pattern
            new_pattern = Pattern(
                id=pattern_id,
                pattern_type=pattern_type,
                pattern=pattern,
                context=context,
                tags=tags,
            )
            new_pattern.reinforce(success=True)
            self._patterns[pattern_id] = new_pattern
            self._index_pattern(new_pattern)
            logger.info(f"Created new pattern {pattern_id}: {pattern[:50]}...")

        # Record attempt for metrics
        self._metrics.record_attempt(TaskAttempt(
            task_id=pattern_id,
            task_type=task_type,
            success=True,
            patterns_used=[pattern_id],
        ))

        if self.auto_save:
            self._save()

        return self._patterns[pattern_id]

    def record_failure(
        self,
        task_type: str,
        pattern: str,
        context: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        """
        Record a failed pattern.

        Call this when a pattern fails to decrease its confidence.

        Args:
            task_type: Type of task
            pattern: The pattern that failed
            context: Additional context
            error: Error message if available
        """
        context = context or {}
        pattern_id = self._generate_id(pattern, context)

        if pattern_id in self._patterns:
            existing = self._patterns[pattern_id]
            existing.reinforce(success=False)
            logger.info(f"Pattern {pattern_id} failed: confidence={existing.confidence:.2f}")

        # Record attempt for metrics
        self._metrics.record_attempt(TaskAttempt(
            task_id=pattern_id,
            task_type=task_type,
            success=False,
            error=error,
            patterns_used=[pattern_id] if pattern_id in self._patterns else [],
        ))

        if self.auto_save:
            self._save()

    def get_patterns(
        self,
        query: str,
        pattern_type: Optional[PatternType] = None,
        min_confidence: Optional[float] = None,
        limit: int = 10,
    ) -> List[Pattern]:
        """
        Get relevant patterns for a query.

        Returns patterns sorted by relevance (confidence * match score).

        Args:
            query: Search query
            pattern_type: Filter by pattern type
            min_confidence: Minimum confidence (uses default if not specified)
            limit: Maximum number of patterns to return

        Returns:
            List of relevant patterns sorted by relevance
        """
        min_conf = min_confidence or self.min_confidence_threshold
        keywords = set(query.lower().split())

        # Find candidate patterns
        candidate_ids: Set[str] = set()
        for keyword in keywords:
            if keyword in self._keyword_index:
                candidate_ids.update(self._keyword_index[keyword])

        # Score and filter candidates
        scored_patterns: List[tuple[float, Pattern]] = []

        for pattern_id in candidate_ids:
            pattern = self._patterns.get(pattern_id)
            if not pattern:
                continue

            # Filter by type
            if pattern_type and pattern.pattern_type != pattern_type:
                continue

            # Get current confidence with decay
            confidence = pattern.current_confidence
            if confidence < min_conf:
                continue

            # Calculate match score (keyword overlap)
            pattern_text = f"{pattern.pattern} {json.dumps(pattern.context)}".lower()
            pattern_keywords = set(pattern_text.split())
            match_score = len(keywords & pattern_keywords) / len(keywords) if keywords else 0

            # Combined relevance score
            relevance = confidence * (0.5 + 0.5 * match_score) * pattern.reliability

            scored_patterns.append((relevance, pattern))

        # Sort by relevance and return top results
        scored_patterns.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored_patterns[:limit]]

    def get_metrics(self, task_type: str) -> Dict[str, Any]:
        """Get pass@k metrics for a task type."""
        return self._metrics.get_summary(task_type)

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all task types."""
        return {
            task_type: self._metrics.get_summary(task_type)
            for task_type in self._metrics._attempts.keys()
        }

    def prune_low_confidence(self, threshold: float = 0.1):
        """
        Remove patterns with confidence below threshold.

        Used for cleanup of patterns that have decayed or
        repeatedly failed.
        """
        to_remove = [
            pid for pid, p in self._patterns.items()
            if p.current_confidence < threshold
        ]

        for pid in to_remove:
            del self._patterns[pid]
            logger.info(f"Pruned low-confidence pattern: {pid}")

        # Rebuild index
        self._keyword_index.clear()
        for pattern in self._patterns.values():
            self._index_pattern(pattern)

        if self.auto_save:
            self._save()

        return len(to_remove)

    def export_patterns(self) -> List[Dict[str, Any]]:
        """Export all patterns as dictionaries."""
        return [p.to_dict() for p in self._patterns.values()]

    def import_patterns(self, patterns: List[Dict[str, Any]]):
        """Import patterns from dictionaries."""
        for data in patterns:
            pattern = Pattern.from_dict(data)
            self._patterns[pattern.id] = pattern
            self._index_pattern(pattern)

        if self.auto_save:
            self._save()

    def _save(self):
        """Save patterns to persistent storage."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "2.0",
            "patterns": self.export_patterns(),
        }
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved {len(self._patterns)} patterns to {self.storage_path}")

    def _load(self):
        """Load patterns from persistent storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                patterns = data.get("patterns", [])
                self.import_patterns(patterns)
                logger.info(f"Loaded {len(self._patterns)} patterns from {self.storage_path}")
            except Exception as e:
                logger.warning(f"Failed to load patterns: {e}")


# =============================================================================
# Convenience Functions
# =============================================================================

# Global engine instance
_engine: Optional[LearningEngine] = None


def get_engine() -> LearningEngine:
    """Get the global learning engine instance."""
    global _engine
    if _engine is None:
        _engine = LearningEngine()
    return _engine


def record_pattern(
    task_type: str,
    pattern: str,
    context: Optional[Dict[str, Any]] = None,
    success: bool = True,
    pattern_type: PatternType = PatternType.CODE_PATTERN,
) -> Optional[Pattern]:
    """
    Convenience function to record a pattern.

    Args:
        task_type: Type of task
        pattern: The pattern/approach
        context: Additional context
        success: Whether the pattern succeeded
        pattern_type: Category of pattern

    Returns:
        Pattern if success, None if failure
    """
    engine = get_engine()
    if success:
        return engine.record_success(task_type, pattern, context, pattern_type)
    else:
        engine.record_failure(task_type, pattern, context)
        return None


def query_patterns(query: str, limit: int = 5) -> List[Pattern]:
    """
    Convenience function to query patterns.

    Args:
        query: Search query
        limit: Maximum results

    Returns:
        List of relevant patterns
    """
    return get_engine().get_patterns(query, limit=limit)


# =============================================================================
# Pre-loaded Patterns (from CLAUDE.md "What Claude Gets Wrong")
# =============================================================================

def load_known_patterns():
    """
    Load known correct patterns from CLAUDE.md.

    These are verified API signatures and approaches that
    have been documented as corrections.
    """
    engine = get_engine()

    # Letta SDK patterns
    engine.record_success(
        task_type="api_integration",
        pattern="Use client.agents.messages.create(agentId, {messages: [{role: 'user', content: msg}]})",
        context={"sdk": "letta", "operation": "send_message"},
        pattern_type=PatternType.API_SIGNATURE,
        tags={"letta", "api", "verified"},
    )

    # LangGraph patterns
    engine.record_success(
        task_type="api_integration",
        pattern="Use workflow.add_conditional_edges(source, routing_fn, path_map)",
        context={"sdk": "langgraph", "operation": "conditional_routing"},
        pattern_type=PatternType.API_SIGNATURE,
        tags={"langgraph", "api", "verified"},
    )

    engine.record_success(
        task_type="api_integration",
        pattern="graph = workflow.compile(checkpointer=checkpointer); config={'configurable': {'thread_id': '...'}}",
        context={"sdk": "langgraph", "operation": "checkpointing"},
        pattern_type=PatternType.API_SIGNATURE,
        tags={"langgraph", "checkpointer", "verified"},
    )

    # Research workflow patterns
    engine.record_success(
        task_type="research",
        pattern="Use Context7 + Exa + Tavily in PARALLEL, never as fallback chain",
        context={"workflow": "api_research"},
        pattern_type=PatternType.WORKFLOW,
        tags={"research", "parallel", "best_practice"},
    )

    # Testing patterns
    engine.record_success(
        task_type="testing",
        pattern="Always include at least ONE real integration test against actual service",
        context={"workflow": "testing"},
        pattern_type=PatternType.WORKFLOW,
        tags={"testing", "integration", "best_practice"},
    )

    logger.info("Loaded known patterns from CLAUDE.md")


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate the continuous learning system."""
    print("=" * 70)
    print("UNLEASH V2 Continuous Learning - Demo")
    print("=" * 70)
    print()

    engine = LearningEngine(auto_save=False)  # Don't save demo data

    # Record some patterns
    print("Recording patterns...")
    engine.record_success(
        task_type="api_integration",
        pattern="Use Context7 to verify API signatures before implementation",
        context={"sdk": "any"},
        pattern_type=PatternType.WORKFLOW,
    )

    engine.record_success(
        task_type="api_integration",
        pattern="client.agents.messages.create(agent_id, {messages: [...]}})",
        context={"sdk": "letta", "method": "send_message"},
        pattern_type=PatternType.API_SIGNATURE,
    )

    # Simulate some failures
    engine.record_failure(
        task_type="api_integration",
        pattern="client.sendMessage(agentId, msg)",
        context={"sdk": "letta"},
        error="Method not found",
    )

    print("\nQuerying patterns for 'letta sdk'...")
    patterns = engine.get_patterns("letta sdk api")
    for p in patterns:
        print(f"  • {p.pattern[:60]}...")
        print(f"    Confidence: {p.current_confidence:.2f}, Reliability: {p.reliability:.2f}")

    print("\nMetrics for api_integration:")
    metrics = engine.get_metrics("api_integration")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("=" * 70)


if __name__ == "__main__":
    demo()
