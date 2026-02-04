"""
V7 Curriculum Learning & STOP Pattern

Extracted from ralph_loop.py V7 enhancements.
Implements Curriculum Learning, Experience Replay, STOP, and Hierarchical Loops.

Classes:
- CurriculumState: Self-evolving curriculum for difficulty scaling (ICLR 2026)
- ExperienceReplay: Priority-weighted memory buffer for learning
- STOPState: Self-Taught Optimizer for recursive meta-improvement (ICLR 2024)
- HierarchicalLoopState: Macro/micro nested optimization

References:
- Self-Evolving Curriculum (ICLR 2026)
- STOP: Self-Taught Optimizer (arxiv:2310.02304, ICLR 2024)
- Experience Replay (DQN/Prioritized Experience Replay)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CurriculumState:
    """
    V7: Self-Evolving Curriculum for iteration difficulty scaling.

    Based on ICLR 2026 paper: Adapts problem difficulty based on agent capability.
    Implements competence-based progression with automatic difficulty adjustment.

    Attributes:
        current_difficulty: Current difficulty level (0.0 = easiest, 1.0 = hardest)
        competence_score: Agent's estimated capability (0.0 to 1.0)
        success_window: Recent success/fail history
        window_size: Number of recent results to track
        difficulty_delta: Step size for difficulty adjustment
        min_success_rate: Below this rate, decrease difficulty
        max_success_rate: Above this rate, increase difficulty

    Example:
        curriculum = CurriculumState()
        curriculum.update_from_result(success=True, improvement=0.1)
        params = curriculum.get_task_modifier()
    """
    current_difficulty: float = 0.3  # 0.0 (easiest) to 1.0 (hardest)
    competence_score: float = 0.5  # Agent's estimated capability
    success_window: List[bool] = field(default_factory=list)  # Recent success/fail
    window_size: int = 10
    difficulty_delta: float = 0.05  # How much to adjust difficulty
    min_success_rate: float = 0.4  # Below this = decrease difficulty
    max_success_rate: float = 0.8  # Above this = increase difficulty

    def update_from_result(self, success: bool, improvement: float) -> None:
        """
        Update curriculum state based on iteration result.

        Args:
            success: Whether the iteration succeeded
            improvement: Magnitude of improvement (0.0 to 1.0)
        """
        self.success_window.append(success)
        if len(self.success_window) > self.window_size:
            self.success_window.pop(0)

        # Update competence score with exponential smoothing
        result_score = 0.8 if success else 0.2
        if improvement > 0:
            result_score += min(0.2, improvement)
        self.competence_score = self.competence_score * 0.9 + result_score * 0.1

        # Adjust difficulty based on success rate
        if len(self.success_window) >= self.window_size // 2:
            success_rate = sum(self.success_window) / len(self.success_window)
            if success_rate < self.min_success_rate:
                self.current_difficulty = max(0.1, self.current_difficulty - self.difficulty_delta)
            elif success_rate > self.max_success_rate:
                self.current_difficulty = min(1.0, self.current_difficulty + self.difficulty_delta)

    def get_task_modifier(self) -> Dict[str, Any]:
        """
        Get task modification parameters based on current difficulty.

        Returns:
            Dict with:
                - complexity_multiplier: Scale for task complexity
                - exploration_scope: How wide to search
                - constraints_strictness: How strict validation should be
                - suggested_strategy: Exploration vs exploitation
        """
        return {
            "complexity_multiplier": 0.5 + self.current_difficulty * 0.5,
            "exploration_scope": 1.0 - self.current_difficulty * 0.3,
            "constraints_strictness": self.current_difficulty,
            "suggested_strategy": "exploit" if self.current_difficulty > 0.7 else "explore"
        }

    @property
    def success_rate(self) -> float:
        """Current success rate over the window."""
        if not self.success_window:
            return 0.5
        return sum(self.success_window) / len(self.success_window)


@dataclass
class ExperienceReplay:
    """
    V7: Experience replay buffer for iteration memory.

    Stores past iterations for learning. Enables:
    - Prioritized experience replay (higher rewards = more likely to sample)
    - Temporal difference learning from iteration sequences
    - Pattern recognition across similar states

    Attributes:
        buffer: List of stored experiences
        max_size: Maximum buffer capacity
        priorities: Priority weight for each experience

    Example:
        replay = ExperienceReplay(max_size=100)
        replay.add({"state": "...", "action": "...", "reward": 0.8}, priority=0.8)
        samples = replay.sample(n=5)
    """
    buffer: List[Dict[str, Any]] = field(default_factory=list)
    max_size: int = 100
    priorities: List[float] = field(default_factory=list)

    def add(self, experience: Dict[str, Any], priority: float = 1.0) -> None:
        """
        Add an experience to the replay buffer.

        Args:
            experience: Dict containing the experience data
            priority: Priority weight (higher = more likely to sample)
        """
        if len(self.buffer) >= self.max_size:
            # Remove lowest priority experience
            min_idx = self.priorities.index(min(self.priorities))
            self.buffer.pop(min_idx)
            self.priorities.pop(min_idx)

        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Sample experiences with priority weighting.

        Args:
            n: Number of experiences to sample

        Returns:
            List of sampled experiences
        """
        if not self.buffer:
            return []

        n = min(n, len(self.buffer))
        total_priority = sum(self.priorities)
        if total_priority == 0:
            # Uniform sampling if no priorities
            indices = random.sample(range(len(self.buffer)), n)
        else:
            # Priority-weighted sampling
            probs = [p / total_priority for p in self.priorities]
            indices = []
            for _ in range(n):
                r = random.random()
                cumsum = 0
                for i, p in enumerate(probs):
                    cumsum += p
                    if r <= cumsum and i not in indices:
                        indices.append(i)
                        break
            # Fill remaining if needed
            while len(indices) < n:
                idx = random.randint(0, len(self.buffer) - 1)
                if idx not in indices:
                    indices.append(idx)

        return [self.buffer[i] for i in indices]

    def get_similar(self, context: str, n: int = 3) -> List[Dict[str, Any]]:
        """
        Get experiences with similar context (simple keyword matching).

        Args:
            context: Context string to match against
            n: Number of similar experiences to return

        Returns:
            List of similar experiences sorted by relevance
        """
        keywords = set(context.lower().split()[:10])
        scored = []
        for exp in self.buffer:
            exp_keywords = set(str(exp.get("context", "")).lower().split()[:10])
            overlap = len(keywords & exp_keywords)
            if overlap > 0:
                scored.append((overlap, exp))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored[:n]]

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


@dataclass
class STOPState:
    """
    V7: Self-Taught Optimizer (STOP) state for recursive self-improvement.

    Based on ICLR 2024 paper (arxiv:2310.02304):
    LLM writes code that recursively improves its own improvement ability.

    Key insight: The optimizer itself can be optimized through self-reflection.

    Attributes:
        improvement_code: Current improvement strategy as pseudocode
        improvement_history: History of (strategy, score) pairs
        meta_improvement_attempts: Number of meta-improvement tries
        best_improvement_score: Best score achieved
        recursion_depth: Current recursion level
        max_recursion: Maximum recursion depth allowed

    Example:
        stop = STOPState()
        stop.record_improvement("Use gradient descent", 0.8)
        if stop.should_meta_improve():
            prompt = stop.get_meta_prompt()
    """
    improvement_code: str = ""  # Current improvement strategy as pseudocode
    improvement_history: List[Tuple[str, float]] = field(default_factory=list)
    meta_improvement_attempts: int = 0
    best_improvement_score: float = 0.0
    recursion_depth: int = 0
    max_recursion: int = 3

    def record_improvement(self, code: str, score: float) -> None:
        """
        Record an improvement attempt and its effectiveness.

        Args:
            code: The improvement strategy used
            score: Effectiveness score (0.0 to 1.0)
        """
        self.improvement_history.append((code, score))
        if score > self.best_improvement_score:
            self.best_improvement_score = score
            self.improvement_code = code
        self.meta_improvement_attempts += 1

    def should_meta_improve(self) -> bool:
        """
        Decide if we should try to improve the improvement strategy itself.

        Returns:
            True if meta-improvement is warranted
        """
        if self.recursion_depth >= self.max_recursion:
            return False
        if self.meta_improvement_attempts < 5:
            return False  # Need baseline data first

        # Check if recent improvements are stagnating
        if len(self.improvement_history) >= 5:
            recent_scores = [s for _, s in self.improvement_history[-5:]]
            avg_score = sum(recent_scores) / len(recent_scores)
            if avg_score < self.best_improvement_score * 0.7:
                return True  # Performance degraded, try meta-improvement

        return False

    def get_meta_prompt(self) -> str:
        """
        Generate prompt for meta-improvement (improving the improver).

        Returns:
            Prompt string for LLM to generate improved improvement strategy
        """
        recent = self.improvement_history[-5:] if len(self.improvement_history) >= 5 else self.improvement_history
        history_str = "\n".join([f"- Strategy: {c[:100]}... Score: {s:.3f}" for c, s in recent])

        return f"""You are a Self-Taught Optimizer (STOP). Your task is to improve the improvement strategy itself.

Current improvement strategy:
{self.improvement_code[:500] if self.improvement_code else "No strategy yet"}

Recent improvement attempts:
{history_str}

Best score achieved: {self.best_improvement_score:.3f}

Analyze what makes improvements effective and propose a better meta-strategy.
Focus on patterns that led to higher scores."""

    def enter_recursion(self) -> bool:
        """
        Enter a recursion level for meta-improvement.

        Returns:
            True if recursion was entered, False if at max depth
        """
        if self.recursion_depth >= self.max_recursion:
            return False
        self.recursion_depth += 1
        return True

    def exit_recursion(self) -> None:
        """Exit current recursion level."""
        if self.recursion_depth > 0:
            self.recursion_depth -= 1


@dataclass
class HierarchicalLoopState:
    """
    V7: Hierarchical iteration state for macro/micro loop coordination.

    Implements nested optimization:
    - Macro loop: High-level strategy selection, goal setting
    - Micro loop: Detailed iteration within strategy

    Enables multi-scale optimization with different time horizons.

    Attributes:
        macro_iteration: Current macro-level iteration
        micro_iteration: Current micro-level iteration
        micro_iterations_per_macro: Micro iterations before macro advancement
        macro_strategy: Current high-level strategy
        macro_goals: Goals for current macro phase
        micro_improvements: Recent micro-level improvements

    Example:
        hier = HierarchicalLoopState()
        while not done:
            improvement = run_micro_iteration()
            if hier.advance_micro(improvement):
                result = hier.advance_macro()
    """
    macro_iteration: int = 0
    micro_iteration: int = 0
    micro_iterations_per_macro: int = 10
    macro_strategy: str = "explore"  # Current macro-level strategy
    macro_goals: List[str] = field(default_factory=list)
    micro_improvements: List[float] = field(default_factory=list)

    def advance_micro(self, improvement: float) -> bool:
        """
        Advance micro iteration.

        Args:
            improvement: Improvement achieved in this micro iteration

        Returns:
            True if macro should advance (micro cycle complete)
        """
        self.micro_iteration += 1
        self.micro_improvements.append(improvement)

        if self.micro_iteration >= self.micro_iterations_per_macro:
            return True  # Time for macro advancement
        return False

    def advance_macro(self) -> Dict[str, Any]:
        """
        Advance macro iteration and decide new strategy.

        Returns:
            Dict with analysis of completed micro cycle
        """
        self.macro_iteration += 1

        # Analyze micro-loop performance
        avg_improvement = sum(self.micro_improvements) / len(self.micro_improvements) if self.micro_improvements else 0

        # Decide next macro strategy
        if avg_improvement < 0.01:
            self.macro_strategy = "explore"  # Stuck, need exploration
        elif avg_improvement > 0.1:
            self.macro_strategy = "exploit"  # Working well, keep exploiting
        else:
            self.macro_strategy = "balanced"

        # Reset micro state
        result = {
            "previous_micro_iterations": self.micro_iteration,
            "avg_improvement": avg_improvement,
            "new_strategy": self.macro_strategy
        }

        self.micro_iteration = 0
        self.micro_improvements = []

        return result

    def get_guidance(self) -> str:
        """
        Get guidance for current iteration based on hierarchical state.

        Returns:
            Human-readable guidance string
        """
        guidance = (
            f"Macro {self.macro_iteration}, Micro {self.micro_iteration}/{self.micro_iterations_per_macro}. "
            f"Strategy: {self.macro_strategy}."
        )
        if len(self.micro_improvements) >= 3:
            trend = sum(self.micro_improvements[-3:]) / 3
            guidance += f" Recent trend: {trend:.4f}"
        return guidance

    @property
    def progress_ratio(self) -> float:
        """Progress through current macro cycle (0.0 to 1.0)."""
        return self.micro_iteration / self.micro_iterations_per_macro
