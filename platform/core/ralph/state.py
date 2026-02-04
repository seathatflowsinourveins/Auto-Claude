"""
Ralph Loop Core State Classes - V36 Architecture

Core state management classes for the Ralph Loop.
Extracted from ralph_loop.py for modularity.

Classes:
- IterationResult: Result of a single iteration
- LoopState: Persistent state of the entire loop (V6 Enhanced)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Import strategy classes for LoopState composition
from .strategies import (
    # V4
    Reflection,
    DebatePosition,
    ProceduralSkill,
    # V5
    ConsistencyPath,
    VerificationStep,
    OODAState,
    RISEAttempt,
    # V6
    StrategyArm,
    ConvergenceState,
    IterationMomentum,
    MetaIterationState,
    # V7
    CurriculumState,
    ExperienceReplay,
    STOPState,
    HierarchicalLoopState,
    # V8
    MCTSState,
    SelfPlayState,
    StrategistState,
)


@dataclass
class IterationResult:
    """
    Result of a single Ralph Loop iteration.

    Captures all metrics and artifacts from one iteration cycle.

    Attributes:
        iteration: The iteration number (1-indexed)
        started_at: ISO timestamp when iteration started
        completed_at: ISO timestamp when iteration completed
        latency_ms: Total duration in milliseconds
        fitness_score: Quality/fitness score achieved (0.0 to 1.0)
        improvements: List of improvements made
        artifacts_created: List of file paths or artifact IDs created
        errors: List of error messages encountered
        metadata: Additional iteration-specific data

    Example:
        result = IterationResult(
            iteration=5,
            started_at="2026-02-02T10:00:00Z",
            completed_at="2026-02-02T10:01:30Z",
            latency_ms=90000,
            fitness_score=0.85,
            improvements=["Added test coverage", "Fixed bug #123"],
            artifacts_created=["tests/test_new.py"],
            errors=[]
        )
    """
    iteration: int
    started_at: str
    completed_at: str
    latency_ms: float
    fitness_score: float
    improvements: List[str]
    artifacts_created: List[str]
    errors: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Whether iteration completed without errors."""
        return len(self.errors) == 0

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return self.latency_ms / 1000.0


@dataclass
class LoopState:
    """
    Persistent state of the Ralph Loop (V6 Enhanced - Meta-Iteration).

    This is the main state object that persists across iterations and sessions.
    It contains all strategy states and historical data.

    Attributes:
        loop_id: Unique identifier for this loop instance
        task: Description of the task being optimized
        current_iteration: Current iteration number
        max_iterations: Maximum iterations allowed
        best_fitness: Best fitness achieved so far
        best_solution: The best solution found
        history: List of all iteration results
        started_at: ISO timestamp when loop started
        status: Current status ("running", "completed", "paused", "failed", "converged")
        metadata: Additional loop-specific data

        # V4 Reflexion
        reflections: Episodic memory of failures
        debate_history: Multi-agent debate sessions
        procedural_skills: Extracted reusable skills

        # V5 Consistency
        consistency_paths: Self-consistency reasoning paths
        verification_history: Chain-of-Verification history
        ooda_states: OODA Loop states
        rise_attempts: RISE recursive introspection attempts

        # V6 Scheduling
        strategy_arms: Thompson Sampling bandit arms
        convergence_state: Early stopping tracker
        iteration_momentum: Pattern momentum tracker
        meta_iteration: Meta-learning state
        current_strategy: Selected strategy for current iteration

        # V7 Curriculum
        curriculum_state: Difficulty adaptation
        experience_replay: Experience buffer
        stop_state: Self-Taught Optimizer state
        hierarchical_state: Macro/micro loop coordination

        # V8 MCTS
        mcts_state: Monte Carlo Tree Search state
        self_play_state: Multi-agent self-play
        strategist_state: Bi-level MCTS

    Example:
        state = LoopState(
            loop_id="loop_123",
            task="Improve test coverage",
            current_iteration=0,
            max_iterations=50,
            best_fitness=0.0,
            best_solution=None,
            history=[],
            started_at="2026-02-02T10:00:00Z",
            status="running"
        )
    """
    loop_id: str
    task: str
    current_iteration: int
    max_iterations: int
    best_fitness: float
    best_solution: Any
    history: List[IterationResult]
    started_at: str
    status: str  # "running", "completed", "paused", "failed", "converged"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # V4: Reflexion pattern - episodic memory of failures
    reflections: List[Reflection] = field(default_factory=list)
    # V4: Multi-agent debate history
    debate_history: List[List[DebatePosition]] = field(default_factory=list)
    # V4: Extracted procedural skills with reliability tracking
    procedural_skills: List[ProceduralSkill] = field(default_factory=list)

    # V5: Self-consistency paths for ensemble reasoning
    consistency_paths: List[ConsistencyPath] = field(default_factory=list)
    # V5: Chain-of-Verification history
    verification_history: List[List[VerificationStep]] = field(default_factory=list)
    # V5: OODA Loop states
    ooda_states: List[OODAState] = field(default_factory=list)
    # V5: RISE recursive introspection attempts
    rise_attempts: List[RISEAttempt] = field(default_factory=list)

    # V6: Strategy arms for Thompson Sampling bandit
    strategy_arms: List[StrategyArm] = field(default_factory=list)
    # V6: Convergence tracking for early stopping
    convergence_state: Optional[ConvergenceState] = None
    # V6: Momentum tracking for successful patterns
    iteration_momentum: Optional[IterationMomentum] = None
    # V6: Meta-learning state for iteration optimization
    meta_iteration: Optional[MetaIterationState] = None
    # V6: Selected strategy for current/last iteration
    current_strategy: str = "dspy"

    # V7: Curriculum learning state
    curriculum_state: Optional[CurriculumState] = None
    # V7: Experience replay buffer
    experience_replay: Optional[ExperienceReplay] = None
    # V7: STOP (Self-Taught Optimizer) state
    stop_state: Optional[STOPState] = None
    # V7: Hierarchical loop state (macro/micro)
    hierarchical_state: Optional[HierarchicalLoopState] = None

    # V8: MCTS exploration state
    mcts_state: Optional[MCTSState] = None
    # V8: Multi-agent self-play state (MARSHAL pattern)
    self_play_state: Optional[SelfPlayState] = None
    # V8: Bi-level MCTS (Strategist pattern)
    strategist_state: Optional[StrategistState] = None

    def update_best(self, fitness: float, solution: Any) -> bool:
        """
        Update best solution if fitness improved.

        Args:
            fitness: New fitness score
            solution: New solution

        Returns:
            True if best was updated
        """
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_solution = solution
            return True
        return False

    def add_iteration_result(self, result: IterationResult) -> None:
        """Add an iteration result to history."""
        self.history.append(result)
        self.current_iteration = result.iteration

        # Update convergence state if present
        if self.convergence_state:
            should_continue = self.convergence_state.update(result.fitness_score)
            if not should_continue:
                self.status = "converged"

    def get_recent_fitness(self, n: int = 5) -> List[float]:
        """Get fitness scores from recent iterations."""
        return [r.fitness_score for r in self.history[-n:]]

    @property
    def progress(self) -> float:
        """Progress through max iterations (0.0 to 1.0)."""
        return self.current_iteration / self.max_iterations if self.max_iterations > 0 else 0.0

    @property
    def is_running(self) -> bool:
        """Check if loop is currently running."""
        return self.status == "running"

    @property
    def is_complete(self) -> bool:
        """Check if loop has completed (any terminal state)."""
        return self.status in ("completed", "failed", "converged")
