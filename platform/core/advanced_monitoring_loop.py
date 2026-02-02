#!/usr/bin/env python3
"""
Advanced Monitoring Loop - V25 Production Engine
================================================

The MISSING PIECE: An actual iteration engine that:
1. Runs autonomously for hours with consistent updates
2. Monitors direction dynamically via chi-squared drift detection
3. Uses dual-gate exit conditions (Ralph Wiggum pattern)
4. Integrates with Letta for cross-session state persistence
5. Provides GOAP planning for autonomous correction

This synthesizes research from:
- everything-claude-code (instinct-based learning, stream-chaining)
- claude-flow (GOAP planning, autonomous correction)
- letta-ai (cross-session memory, conversations API)
- opik (trajectory evaluation, cost tracking)

Created: 2026-01-31 (V25 Iteration)
"""

from __future__ import annotations

import asyncio
import json
import time
import os
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from enum import Enum
from collections import defaultdict
import statistics

# Type hints for optional imports
stats: Any = None
SCIPY_AVAILABLE = False
try:
    from scipy import stats as scipy_stats
    stats = scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("advanced_monitoring_loop")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LoopConfig:
    """Configuration for the advanced monitoring loop."""
    # Timing
    iteration_interval_seconds: float = 60.0  # 1 minute between iterations
    max_runtime_hours: float = 8.0  # Maximum total runtime
    checkpoint_interval: int = 5  # Checkpoint every N iterations

    # Direction monitoring
    drift_detection_window: int = 10  # Recent iterations for drift analysis
    drift_threshold_p_value: float = 0.05  # Statistical significance for drift
    goal_alignment_threshold: float = 0.7  # Minimum alignment score

    # Exit conditions (dual-gate - Ralph Wiggum pattern)
    completion_indicators: List[str] = field(default_factory=lambda: [
        "OBJECTIVE_ACHIEVED",
        "ALL_TASKS_COMPLETE",
        "GOALS_MET",
        "SUCCESS",
    ])
    exit_signal_keywords: List[str] = field(default_factory=lambda: [
        "EXIT_NOW",
        "TERMINATE_LOOP",
        "STOP_ITERATION",
        "MISSION_COMPLETE",
    ])
    require_both_gates: bool = True  # Dual-gate: require BOTH completion AND exit signal

    # Budget management
    max_cost_usd: float = 50.0  # Maximum spend before stopping
    cost_warning_threshold: float = 0.8  # Warn at 80% of budget

    # Model pricing (per 1M tokens)
    model_pricing: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "claude-opus-4": {"input": 15.0, "output": 75.0},
        "claude-sonnet-4": {"input": 3.0, "output": 15.0},
        "claude-haiku-4": {"input": 0.25, "output": 1.25},
    })

    # Letta integration
    letta_agent_id: Optional[str] = None
    letta_enabled: bool = True

    # Paths
    state_path: Path = field(default_factory=lambda: Path.home() / ".claude" / "state" / "loop_state.json")
    checkpoint_path: Path = field(default_factory=lambda: Path.home() / ".claude" / "checkpoints")
    instincts_path: Path = field(default_factory=lambda: Path.home() / ".claude" / "homunculus" / "instincts")


# =============================================================================
# DIRECTION MONITORING - Chi-Squared Drift Detection
# =============================================================================

@dataclass
class DriftMetrics:
    """Results from direction drift analysis."""
    chi2_statistic: float
    p_value: float
    is_drifting: bool
    drift_direction: str  # "on_track", "minor_drift", "major_drift"
    goal_alignment_score: float
    recommendation: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DirectionMonitor:
    """
    Monitor goal direction using chi-squared drift detection.

    Research basis: everything-claude-code patterns + statistical process control

    Detects when task distribution deviates from baseline goals,
    enabling autonomous course correction.
    """

    def __init__(self, config: LoopConfig):
        self.config = config
        self.baseline_distribution: Dict[str, float] = {}
        self.task_history: List[Dict[str, Any]] = []
        self.goal_categories: List[str] = [
            "research", "implementation", "testing", "documentation",
            "optimization", "debugging", "integration", "review"
        ]

    def set_baseline_goals(self, goals: Dict[str, float]) -> None:
        """
        Set the expected distribution of task types.

        Example: {"research": 0.3, "implementation": 0.4, "testing": 0.2, "documentation": 0.1}
        """
        total = sum(goals.values())
        self.baseline_distribution = {k: v / total for k, v in goals.items()}
        logger.info(f"Baseline goals set: {self.baseline_distribution}")

    def record_task(self, task_type: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a task execution for drift analysis."""
        self.task_history.append({
            "type": task_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        })

    def compute_goal_drift(self) -> DriftMetrics:
        """
        Compute goal drift using chi-squared test.

        Compares recent task distribution against baseline goals.
        """
        if not self.baseline_distribution:
            return DriftMetrics(
                chi2_statistic=0.0,
                p_value=1.0,
                is_drifting=False,
                drift_direction="on_track",
                goal_alignment_score=1.0,
                recommendation="Set baseline goals first"
            )

        # Get recent tasks within window
        window = self.config.drift_detection_window
        recent_tasks = self.task_history[-window:] if len(self.task_history) >= window else self.task_history

        if len(recent_tasks) < 5:
            return DriftMetrics(
                chi2_statistic=0.0,
                p_value=1.0,
                is_drifting=False,
                drift_direction="on_track",
                goal_alignment_score=1.0,
                recommendation="Insufficient data for drift detection"
            )

        # Count task types
        observed_counts = defaultdict(int)
        for task in recent_tasks:
            observed_counts[task["type"]] += 1

        # Build observed and expected arrays
        categories = list(self.baseline_distribution.keys())
        observed = [observed_counts.get(cat, 0) for cat in categories]
        total_observed = sum(observed)
        expected = [self.baseline_distribution[cat] * total_observed for cat in categories]

        # Ensure no zero expected values (add small epsilon)
        expected = [max(e, 0.001) for e in expected]

        # Chi-squared test
        if SCIPY_AVAILABLE and stats is not None:
            chi2_result = stats.chisquare(observed, expected)
            chi2 = float(chi2_result.statistic)
            p_value = float(chi2_result.pvalue)
        else:
            # Manual chi-squared calculation
            chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
            # Approximate p-value (degrees of freedom = len(categories) - 1)
            p_value = 0.5 if chi2 < 5 else 0.1 if chi2 < 10 else 0.01

        # Determine drift level
        is_drifting = p_value < self.config.drift_threshold_p_value

        if p_value >= 0.1:
            drift_direction = "on_track"
        elif p_value >= 0.01:
            drift_direction = "minor_drift"
        else:
            drift_direction = "major_drift"

        # Calculate goal alignment score
        alignment_scores = []
        for cat in categories:
            expected_pct = self.baseline_distribution[cat]
            observed_pct = observed_counts.get(cat, 0) / total_observed if total_observed > 0 else 0
            alignment = 1.0 - abs(expected_pct - observed_pct)
            alignment_scores.append(alignment)

        goal_alignment_score = statistics.mean(alignment_scores) if alignment_scores else 1.0

        # Generate recommendation
        if drift_direction == "on_track":
            recommendation = "Continue current approach"
        elif drift_direction == "minor_drift":
            # Find most over/under represented categories
            deviations = {cat: observed_counts.get(cat, 0) / total_observed - self.baseline_distribution[cat]
                         for cat in categories}
            over = max(deviations.keys(), key=lambda k: deviations[k])
            under = min(deviations.keys(), key=lambda k: deviations[k])
            recommendation = f"Minor drift detected. Consider less '{over}' and more '{under}'"
        else:
            recommendation = "Major drift! Recommend goal realignment or explicit course correction"

        return DriftMetrics(
            chi2_statistic=float(chi2),
            p_value=float(p_value),
            is_drifting=is_drifting,
            drift_direction=drift_direction,
            goal_alignment_score=goal_alignment_score,
            recommendation=recommendation
        )


# =============================================================================
# BUDGET MANAGEMENT - Multi-Tier Cost Tracking
# =============================================================================

@dataclass
class CostEntry:
    """A single cost entry."""
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BudgetManager:
    """
    Multi-tier cost tracking and budget management.

    Research basis: Opik cost tracking patterns
    """

    def __init__(self, config: LoopConfig):
        self.config = config
        self.cost_history: List[CostEntry] = []
        self.total_spent_usd: float = 0.0
        self.session_start = datetime.now(timezone.utc)

    def record_usage(self, model: str, input_tokens: int, output_tokens: int) -> CostEntry:
        """Record token usage and calculate cost."""
        pricing = self.config.model_pricing.get(model, {"input": 3.0, "output": 15.0})

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        entry = CostEntry(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=total_cost
        )

        self.cost_history.append(entry)
        self.total_spent_usd += total_cost

        return entry

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        remaining = self.config.max_cost_usd - self.total_spent_usd
        pct_used = (self.total_spent_usd / self.config.max_cost_usd) * 100 if self.config.max_cost_usd > 0 else 0

        return {
            "total_spent_usd": round(self.total_spent_usd, 4),
            "remaining_usd": round(remaining, 4),
            "max_budget_usd": self.config.max_cost_usd,
            "percent_used": round(pct_used, 2),
            "is_warning": pct_used >= self.config.cost_warning_threshold * 100,
            "is_exceeded": self.total_spent_usd >= self.config.max_cost_usd,
            "entries_count": len(self.cost_history)
        }

    def should_continue(self) -> Tuple[bool, str]:
        """Check if budget allows continuation."""
        status = self.get_budget_status()

        if status["is_exceeded"]:
            return False, f"Budget exceeded: ${status['total_spent_usd']:.2f} / ${status['max_budget_usd']:.2f}"

        if status["is_warning"]:
            logger.warning(f"Budget warning: {status['percent_used']:.1f}% used")

        return True, "Budget OK"


# =============================================================================
# STATE PERSISTENCE - Cross-Session with Letta Integration
# =============================================================================

@dataclass
class LoopState:
    """Complete loop state for persistence."""
    iteration_count: int = 0
    total_runtime_seconds: float = 0.0
    goals: Dict[str, float] = field(default_factory=dict)
    task_history: List[Dict[str, Any]] = field(default_factory=list)
    cost_history: List[Dict[str, Any]] = field(default_factory=list)
    total_spent_usd: float = 0.0
    drift_history: List[Dict[str, Any]] = field(default_factory=list)
    instincts: List[Dict[str, Any]] = field(default_factory=list)
    last_checkpoint: Optional[str] = None
    session_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:12])
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoopState":
        return cls(**data)


class StatePersistence:
    """
    Persist loop state for crash recovery and cross-session continuity.

    Integrates with Letta Cloud for cross-session memory.
    """

    def __init__(self, config: LoopConfig):
        self.config = config
        self.state = LoopState()
        self.letta_client = None

        # Ensure directories exist
        self.config.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Initialize Letta if enabled
        if config.letta_enabled and config.letta_agent_id:
            self._init_letta()

    def _init_letta(self) -> None:
        """Initialize Letta client for cross-session memory."""
        try:
            from letta_client import Letta
            api_key = os.environ.get("LETTA_API_KEY")
            if api_key:
                self.letta_client = Letta(api_key=api_key)
                logger.info("Letta client initialized for cross-session memory")
        except ImportError:
            logger.warning("letta-client not installed. Cross-session memory disabled.")
        except Exception as e:
            logger.error(f"Letta initialization failed: {e}")

    def save_local(self) -> None:
        """Save state to local file."""
        self.state.updated_at = datetime.now(timezone.utc).isoformat()
        with open(self.config.state_path, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)
        logger.debug(f"State saved locally: {self.config.state_path}")

    def load_local(self) -> bool:
        """Load state from local file. Returns True if loaded successfully."""
        if self.config.state_path.exists():
            try:
                with open(self.config.state_path) as f:
                    data = json.load(f)
                self.state = LoopState.from_dict(data)
                logger.info(f"State loaded: iteration {self.state.iteration_count}")
                return True
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        return False

    async def save_to_letta(self, summary: str) -> bool:
        """Save state summary to Letta archival memory."""
        if not self.letta_client or not self.config.letta_agent_id:
            return False

        try:
            content = json.dumps({
                "type": "loop_state",
                "session_id": self.state.session_id,
                "iteration": self.state.iteration_count,
                "total_runtime_hours": self.state.total_runtime_seconds / 3600,
                "total_spent_usd": self.state.total_spent_usd,
                "summary": summary
            })

            # Use passages API (V23+ verified)
            created = self.letta_client.agents.passages.create(
                self.config.letta_agent_id,
                text=content,
                tags=["loop_state", f"session:{self.state.session_id}"]
            )
            logger.info(f"State saved to Letta: {created[0].id}")
            return True
        except Exception as e:
            logger.error(f"Letta save failed: {e}")
            return False

    async def load_from_letta(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load previous session state from Letta."""
        if not self.letta_client or not self.config.letta_agent_id:
            return None

        try:
            # Search for loop states (V23+ verified: query= and top_k=)
            search_result = self.letta_client.agents.passages.search(
                self.config.letta_agent_id,
                query="loop_state session",
                top_k=5,
                tags=["loop_state"]
            )

            # V23: Use .results and .content
            for result in search_result.results:
                try:
                    data = json.loads(result.content)
                    if session_id is None or data.get("session_id") == session_id:
                        return data
                except json.JSONDecodeError:
                    continue

            return None
        except Exception as e:
            logger.error(f"Letta load failed: {e}")
            return None

    def create_checkpoint(self, checkpoint_name: str) -> Path:
        """Create a named checkpoint."""
        checkpoint_file = self.config.checkpoint_path / f"{checkpoint_name}_{self.state.session_id}.json"
        with open(checkpoint_file, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)
        self.state.last_checkpoint = str(checkpoint_file)
        logger.info(f"Checkpoint created: {checkpoint_file}")
        return checkpoint_file


# =============================================================================
# INSTINCT-BASED LEARNING - Compound Learning Patterns
# =============================================================================

@dataclass
class Instinct:
    """
    An atomic learned pattern (instinct) from everything-claude-code research.

    Instincts are trigger/action pairs with confidence scoring (0.3-0.9).
    They evolve into skills/commands/agents via clustering.
    """
    instinct_id: str
    trigger_pattern: str  # When to activate
    action_pattern: str   # What to do
    confidence: float     # 0.3 (tentative) to 0.9 (near-certain)
    activation_count: int = 0
    success_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_activated: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Instinct":
        return cls(**data)

    def activate(self, success: bool = True) -> None:
        """Record an activation of this instinct."""
        self.activation_count += 1
        if success:
            self.success_count += 1
        self.last_activated = datetime.now(timezone.utc).isoformat()

        # Update confidence based on success rate
        if self.activation_count >= 3:
            success_rate = self.success_count / self.activation_count
            # Blend toward success rate with momentum
            self.confidence = 0.7 * self.confidence + 0.3 * success_rate
            self.confidence = max(0.3, min(0.9, self.confidence))


class ContinuousLearning:
    """
    Instinct-based continuous learning system.

    Research basis: everything-claude-code v1.2.0 instinct architecture
    """

    def __init__(self, config: LoopConfig):
        self.config = config
        self.instincts: Dict[str, Instinct] = {}
        self.pending_observations: List[Dict[str, Any]] = []

        # Load existing instincts
        self._load_instincts()

    def _load_instincts(self) -> None:
        """Load instincts from disk."""
        personal_path = self.config.instincts_path / "personal"
        personal_path.mkdir(parents=True, exist_ok=True)

        for file in personal_path.glob("*.json"):
            try:
                with open(file) as f:
                    data = json.load(f)
                instinct = Instinct.from_dict(data)
                self.instincts[instinct.instinct_id] = instinct
            except Exception as e:
                logger.warning(f"Failed to load instinct {file}: {e}")

        logger.info(f"Loaded {len(self.instincts)} instincts")

    def _save_instinct(self, instinct: Instinct) -> None:
        """Save a single instinct to disk."""
        personal_path = self.config.instincts_path / "personal"
        personal_path.mkdir(parents=True, exist_ok=True)

        file_path = personal_path / f"{instinct.instinct_id}.json"
        with open(file_path, "w") as f:
            json.dump(instinct.to_dict(), f, indent=2)

    def observe(self, trigger: str, action: str, outcome: bool) -> None:
        """Record an observation for potential instinct formation."""
        self.pending_observations.append({
            "trigger": trigger,
            "action": action,
            "outcome": outcome,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def extract_patterns(self) -> List[Instinct]:
        """Extract instincts from pending observations."""
        if len(self.pending_observations) < 3:
            return []

        # Group observations by trigger pattern (simplified)
        trigger_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for obs in self.pending_observations:
            # Simple trigger normalization
            trigger_key = obs["trigger"].lower().strip()[:50]
            trigger_groups[trigger_key].append(obs)

        new_instincts = []

        for trigger_pattern, observations in trigger_groups.items():
            if len(observations) < 2:
                continue

            # Find most common successful action
            successful_actions = [obs["action"] for obs in observations if obs["outcome"]]
            if not successful_actions:
                continue

            # Simple majority action
            action_counts = defaultdict(int)
            for action in successful_actions:
                action_counts[action] += 1

            best_action = max(action_counts.keys(), key=lambda k: action_counts[k])
            success_rate = len(successful_actions) / len(observations)

            # Create instinct if confidence high enough
            if success_rate >= 0.5:
                instinct_id = hashlib.md5(f"{trigger_pattern}:{best_action}".encode()).hexdigest()[:12]

                if instinct_id not in self.instincts:
                    instinct = Instinct(
                        instinct_id=instinct_id,
                        trigger_pattern=trigger_pattern,
                        action_pattern=best_action,
                        confidence=0.3 + (success_rate * 0.3),  # 0.3-0.6 initial
                        activation_count=len(observations),
                        success_count=len(successful_actions)
                    )
                    self.instincts[instinct_id] = instinct
                    self._save_instinct(instinct)
                    new_instincts.append(instinct)
                    logger.info(f"New instinct created: {instinct_id} (confidence: {instinct.confidence:.2f})")

        # Clear processed observations
        self.pending_observations = []

        return new_instincts

    def get_applicable_instincts(self, context: str, min_confidence: float = 0.5) -> List[Instinct]:
        """Find instincts applicable to current context."""
        applicable = []
        context_lower = context.lower()

        for instinct in self.instincts.values():
            if instinct.confidence >= min_confidence:
                # Simple substring matching (could be enhanced with embeddings)
                if instinct.trigger_pattern.lower() in context_lower:
                    applicable.append(instinct)

        return sorted(applicable, key=lambda i: i.confidence, reverse=True)


# =============================================================================
# GOAP PLANNER - Autonomous Correction
# =============================================================================

@dataclass
class GOAPAction:
    """A GOAP action with preconditions and effects."""
    name: str
    preconditions: Dict[str, Any]  # Required world state
    effects: Dict[str, Any]        # State changes after execution
    cost: float = 1.0

    def is_applicable(self, world_state: Dict[str, Any]) -> bool:
        """Check if action preconditions are met."""
        for key, required_value in self.preconditions.items():
            if world_state.get(key) != required_value:
                return False
        return True

    def apply(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply action effects to world state."""
        new_state = world_state.copy()
        new_state.update(self.effects)
        return new_state


class GOAPPlanner:
    """
    Goal-Oriented Action Planning for autonomous correction.

    Research basis: claude-flow V3 patterns

    Uses A* search to find optimal action sequences from current state to goal.
    """

    def __init__(self):
        self.actions: List[GOAPAction] = []
        self._init_default_actions()

    def _init_default_actions(self) -> None:
        """Initialize default corrective actions."""
        self.actions = [
            GOAPAction(
                name="refocus_on_goals",
                preconditions={"is_drifting": True},
                effects={"is_drifting": False, "goal_alignment": "high"},
                cost=1.0
            ),
            GOAPAction(
                name="reduce_scope",
                preconditions={"is_drifting": True, "scope": "large"},
                effects={"is_drifting": False, "scope": "focused"},
                cost=2.0
            ),
            GOAPAction(
                name="request_clarification",
                preconditions={"goal_clarity": "low"},
                effects={"goal_clarity": "high"},
                cost=1.5
            ),
            GOAPAction(
                name="checkpoint_and_review",
                preconditions={"iterations_since_checkpoint": "high"},
                effects={"iterations_since_checkpoint": "low", "reviewed": True},
                cost=1.0
            ),
            GOAPAction(
                name="budget_optimization",
                preconditions={"budget_status": "warning"},
                effects={"budget_status": "optimized", "model_tier": "haiku"},
                cost=0.5
            ),
            GOAPAction(
                name="consolidate_learnings",
                preconditions={"pending_observations": "high"},
                effects={"pending_observations": "low", "instincts_updated": True},
                cost=1.0
            ),
        ]

    def plan(self, current_state: Dict[str, Any], goal_state: Dict[str, Any], max_depth: int = 10) -> List[GOAPAction]:
        """
        Find action sequence from current to goal state using A*.
        """
        def heuristic(state: Dict[str, Any]) -> float:
            """Estimate cost to reach goal."""
            mismatches = sum(1 for k, v in goal_state.items() if state.get(k) != v)
            return float(mismatches)

        def state_key(state: Dict[str, Any]) -> str:
            """Create hashable key for state."""
            return json.dumps(sorted(state.items()))

        # A* search
        open_set: List[Tuple[float, float, Dict[str, Any], List[GOAPAction]]] = [
            (0.0 + heuristic(current_state), 0.0, current_state, [])
        ]
        closed_set: Set[str] = set()

        while open_set:
            open_set.sort(key=lambda x: x[0])
            _, g_cost, state, path = open_set.pop(0)

            state_hash = state_key(state)
            if state_hash in closed_set:
                continue
            closed_set.add(state_hash)

            # Check if goal reached
            if all(state.get(k) == v for k, v in goal_state.items()):
                return path

            # Expand with applicable actions
            if len(path) < max_depth:
                for action in self.actions:
                    if action.is_applicable(state):
                        new_state = action.apply(state)
                        new_path = path + [action]
                        new_g: float = g_cost + action.cost
                        new_f: float = new_g + heuristic(new_state)
                        open_set.append((new_f, new_g, new_state, new_path))

        return []  # No plan found


# =============================================================================
# ADVANCED MONITORING LOOP - Main Engine
# =============================================================================

class AdvancedMonitoringLoop:
    """
    Production-ready autonomous monitoring loop.

    Synthesizes:
    - DirectionMonitor (chi-squared drift detection)
    - BudgetManager (multi-tier cost tracking)
    - StatePersistence (local + Letta cross-session)
    - ContinuousLearning (instinct-based patterns)
    - GOAPPlanner (autonomous correction)

    Features dual-gate exit conditions (Ralph Wiggum pattern):
    - Gate 1: Completion indicators present
    - Gate 2: Explicit exit signal received
    Both must be satisfied to terminate (configurable).
    """

    def __init__(self, config: Optional[LoopConfig] = None):
        self.config = config or LoopConfig()

        # Initialize components
        self.direction_monitor = DirectionMonitor(self.config)
        self.budget_manager = BudgetManager(self.config)
        self.state_persistence = StatePersistence(self.config)
        self.continuous_learning = ContinuousLearning(self.config)
        self.goap_planner = GOAPPlanner()

        # Runtime state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.iteration_count = 0

        # Callbacks
        self.on_iteration: Optional[Callable[[int, Dict[str, Any]], None]] = None
        self.on_drift_detected: Optional[Callable[[DriftMetrics], None]] = None
        self.on_correction: Optional[Callable[[List[GOAPAction]], None]] = None

    def set_goals(self, goals: Dict[str, float]) -> None:
        """Set baseline goal distribution for drift monitoring."""
        self.direction_monitor.set_baseline_goals(goals)
        self.state_persistence.state.goals = goals

    def _check_exit_conditions(self, iteration_output: str) -> Tuple[bool, str]:
        """
        Check dual-gate exit conditions.

        Returns (should_exit, reason)
        """
        has_completion = any(
            indicator in iteration_output.upper()
            for indicator in self.config.completion_indicators
        )

        has_exit_signal = any(
            signal in iteration_output.upper()
            for signal in self.config.exit_signal_keywords
        )

        if self.config.require_both_gates:
            # Dual-gate: require BOTH
            if has_completion and has_exit_signal:
                return True, "Dual-gate exit: completion + exit signal"
        else:
            # Single-gate: either is sufficient
            if has_completion:
                return True, "Completion indicator detected"
            if has_exit_signal:
                return True, "Exit signal received"

        return False, ""

    def _check_runtime_limits(self) -> Tuple[bool, str]:
        """Check if runtime limits exceeded."""
        if self.start_time is None:
            return True, ""

        elapsed_hours = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600

        if elapsed_hours >= self.config.max_runtime_hours:
            return False, f"Maximum runtime exceeded: {elapsed_hours:.2f}h / {self.config.max_runtime_hours}h"

        return True, ""

    def _get_world_state(self, drift_metrics: DriftMetrics) -> Dict[str, Any]:
        """Get current world state for GOAP planning."""
        budget_status = self.budget_manager.get_budget_status()

        return {
            "is_drifting": drift_metrics.is_drifting,
            "drift_direction": drift_metrics.drift_direction,
            "goal_alignment": "high" if drift_metrics.goal_alignment_score >= 0.7 else "low",
            "budget_status": "warning" if budget_status["is_warning"] else "ok",
            "iterations_since_checkpoint": "high" if self.iteration_count % self.config.checkpoint_interval == 0 else "low",
            "pending_observations": "high" if len(self.continuous_learning.pending_observations) >= 5 else "low",
        }

    async def run_iteration(self, task_executor: Callable[[int], str]) -> Dict[str, Any]:
        """
        Run a single iteration of the loop.

        Args:
            task_executor: Async function that executes the iteration task
                          Takes iteration number, returns output string

        Returns:
            Iteration result dictionary
        """
        self.iteration_count += 1
        iteration_start = time.time()

        # Execute task
        try:
            output = await asyncio.to_thread(task_executor, self.iteration_count)
        except Exception as e:
            output = f"ERROR: {str(e)}"
            logger.error(f"Iteration {self.iteration_count} failed: {e}")

        iteration_duration = time.time() - iteration_start

        # Record task (extract type from output if possible)
        task_type = self._extract_task_type(output)
        self.direction_monitor.record_task(task_type, {"iteration": self.iteration_count})

        # Check drift
        drift_metrics = self.direction_monitor.compute_goal_drift()

        # Handle drift with GOAP planning
        corrections = []
        if drift_metrics.is_drifting:
            world_state = self._get_world_state(drift_metrics)
            goal_state = {"is_drifting": False, "goal_alignment": "high"}
            corrections = self.goap_planner.plan(world_state, goal_state)

            if corrections and self.on_correction:
                self.on_correction(corrections)

            if self.on_drift_detected:
                self.on_drift_detected(drift_metrics)

        # Update state
        self.state_persistence.state.iteration_count = self.iteration_count
        self.state_persistence.state.total_runtime_seconds += iteration_duration
        self.state_persistence.state.task_history.append({
            "iteration": self.iteration_count,
            "type": task_type,
            "duration": iteration_duration,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        self.state_persistence.state.drift_history.append(drift_metrics.to_dict())

        # Periodic checkpoint
        if self.iteration_count % self.config.checkpoint_interval == 0:
            self.state_persistence.save_local()
            checkpoint_name = f"iteration_{self.iteration_count}"
            self.state_persistence.create_checkpoint(checkpoint_name)

        # Extract patterns from observations
        new_instincts = self.continuous_learning.extract_patterns()

        result = {
            "iteration": self.iteration_count,
            "output": output,
            "duration_seconds": iteration_duration,
            "task_type": task_type,
            "drift": drift_metrics.to_dict(),
            "budget": self.budget_manager.get_budget_status(),
            "corrections": [a.name for a in corrections],
            "new_instincts": [i.instinct_id for i in new_instincts],
        }

        if self.on_iteration:
            self.on_iteration(self.iteration_count, result)

        return result

    def _extract_task_type(self, output: str) -> str:
        """Extract task type from output (simple heuristic)."""
        output_lower = output.lower()

        type_keywords = {
            "research": ["search", "research", "investigate", "explore", "analyze"],
            "implementation": ["implement", "create", "build", "write", "code"],
            "testing": ["test", "verify", "validate", "check"],
            "documentation": ["document", "readme", "explain", "describe"],
            "optimization": ["optimize", "improve", "enhance", "refactor"],
            "debugging": ["debug", "fix", "error", "bug"],
            "integration": ["integrate", "connect", "combine", "merge"],
            "review": ["review", "audit", "examine", "inspect"],
        }

        for task_type, keywords in type_keywords.items():
            if any(kw in output_lower for kw in keywords):
                return task_type

        return "general"

    async def run(self, task_executor: Callable[[int], str]) -> Dict[str, Any]:
        """
        Run the monitoring loop until exit conditions are met.

        Args:
            task_executor: Function that executes each iteration task
                          Takes iteration number, returns output string

        Returns:
            Final summary of the loop execution
        """
        self.is_running = True
        self.start_time = datetime.now(timezone.utc)

        # Try to load previous state
        if self.state_persistence.load_local():
            self.iteration_count = self.state_persistence.state.iteration_count
            logger.info(f"Resumed from iteration {self.iteration_count}")

        results = []
        exit_reason = ""

        try:
            while self.is_running:
                # Check runtime limits
                can_continue, reason = self._check_runtime_limits()
                if not can_continue:
                    exit_reason = reason
                    break

                # Check budget
                can_continue, reason = self.budget_manager.should_continue()
                if not can_continue:
                    exit_reason = reason
                    break

                # Run iteration
                result = await self.run_iteration(task_executor)
                results.append(result)

                # Check exit conditions
                should_exit, reason = self._check_exit_conditions(result["output"])
                if should_exit:
                    exit_reason = reason
                    break

                # Wait for next iteration
                await asyncio.sleep(self.config.iteration_interval_seconds)

        except KeyboardInterrupt:
            exit_reason = "User interrupt (Ctrl+C)"
        except Exception as e:
            exit_reason = f"Unexpected error: {e}"
            logger.exception("Loop terminated with error")
        finally:
            self.is_running = False

        # Final state save
        self.state_persistence.save_local()

        # Save to Letta if available
        summary = f"Completed {self.iteration_count} iterations. Exit: {exit_reason}"
        await self.state_persistence.save_to_letta(summary)

        return {
            "total_iterations": self.iteration_count,
            "total_runtime_hours": (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600,
            "exit_reason": exit_reason,
            "total_spent_usd": self.budget_manager.total_spent_usd,
            "final_drift": self.direction_monitor.compute_goal_drift().to_dict(),
            "instincts_created": len(self.continuous_learning.instincts),
            "results_count": len(results),
        }

    def stop(self) -> None:
        """Request graceful stop of the loop."""
        self.is_running = False
        logger.info("Stop requested")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_loop(
    max_runtime_hours: float = 8.0,
    max_cost_usd: float = 50.0,
    letta_agent_id: Optional[str] = None,
    goals: Optional[Dict[str, float]] = None
) -> AdvancedMonitoringLoop:
    """
    Create a pre-configured monitoring loop.

    Example:
        loop = create_loop(
            max_runtime_hours=4.0,
            max_cost_usd=25.0,
            goals={"research": 0.3, "implementation": 0.5, "testing": 0.2}
        )

        async def my_task(iteration: int) -> str:
            # Your task logic here
            return "Task completed"

        result = await loop.run(my_task)
    """
    config = LoopConfig(
        max_runtime_hours=max_runtime_hours,
        max_cost_usd=max_cost_usd,
        letta_agent_id=letta_agent_id,
    )

    loop = AdvancedMonitoringLoop(config)

    if goals:
        loop.set_goals(goals)

    return loop


async def quick_run(
    task_executor: Callable[[int], str],
    max_iterations: int = 10,
    goals: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Quick run for testing - runs for a fixed number of iterations.

    Example:
        def my_task(i):
            return f"Completed iteration {i}"

        result = await quick_run(my_task, max_iterations=5)
    """
    config = LoopConfig(
        max_runtime_hours=1.0,
        iteration_interval_seconds=1.0,  # Fast for testing
    )

    loop = AdvancedMonitoringLoop(config)

    if goals:
        loop.set_goals(goals)

    # Override exit check to stop after max_iterations
    original_check = loop._check_exit_conditions

    def limited_check(iteration_output: str) -> Tuple[bool, str]:
        if loop.iteration_count >= max_iterations:
            return True, f"Reached max iterations ({max_iterations})"
        return original_check(iteration_output)

    loop._check_exit_conditions = limited_check  # type: ignore

    return await loop.run(task_executor)


# =============================================================================
# MAIN - Demo / Test
# =============================================================================

async def _demo():
    """Demonstrate the advanced monitoring loop."""
    print("=" * 60)
    print("Advanced Monitoring Loop - V25 Demo")
    print("=" * 60)

    # Create loop with goals
    loop = create_loop(
        max_runtime_hours=0.1,  # 6 minutes for demo
        max_cost_usd=5.0,
        goals={
            "research": 0.3,
            "implementation": 0.4,
            "testing": 0.2,
            "documentation": 0.1
        }
    )

    # Set up callbacks
    def on_iteration(iteration: int, result: Dict[str, Any]) -> None:
        drift = result["drift"]
        print(f"\n[Iteration {iteration}]")
        print(f"  Task type: {result['task_type']}")
        print(f"  Duration: {result['duration_seconds']:.2f}s")
        print(f"  Drift: {drift['drift_direction']} (p={drift['p_value']:.3f})")
        print(f"  Goal alignment: {drift['goal_alignment_score']:.2f}")
        if result["corrections"]:
            print(f"  Corrections: {result['corrections']}")

    def on_drift_detected(metrics: DriftMetrics) -> None:
        print(f"\n⚠️  DRIFT DETECTED: {metrics.drift_direction}")
        print(f"   Recommendation: {metrics.recommendation}")

    loop.on_iteration = on_iteration
    loop.on_drift_detected = on_drift_detected

    # Demo task executor
    import random
    task_types = ["research", "implementation", "testing", "documentation", "debugging"]

    def demo_task(iteration: int) -> str:
        # Simulate different task types
        task_type = random.choices(
            task_types,
            weights=[0.2, 0.3, 0.2, 0.1, 0.2]  # Slightly misaligned from goals
        )[0]
        time.sleep(0.5)  # Simulate work

        # Occasionally signal completion
        if iteration >= 5:
            return f"Completed {task_type} task. OBJECTIVE_ACHIEVED. EXIT_NOW."

        return f"Completed {task_type} task in iteration {iteration}"

    # Run the loop
    print("\nStarting loop...")
    result = await loop.run(demo_task)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total iterations: {result['total_iterations']}")
    print(f"Total runtime: {result['total_runtime_hours']:.3f} hours")
    print(f"Exit reason: {result['exit_reason']}")
    print(f"Total spent: ${result['total_spent_usd']:.4f}")
    print(f"Instincts created: {result['instincts_created']}")


if __name__ == "__main__":
    asyncio.run(_demo())
