"""
V6 Scheduling Pattern - Meta-Iteration Optimization

Extracted from ralph_loop.py V6 enhancements.
Implements Thompson Sampling, Adaptive Scheduling, and Meta-Learning.

Classes:
- StrategyArm: Strategy arm for Thompson Sampling bandit selection
- ConvergenceState: Track convergence for early stopping decisions
- IterationMomentum: Carry forward successful patterns with decay
- MetaIterationState: Meta-learning state for iteration optimization

Features:
- Thompson Sampling: Bayesian bandit for strategy selection
- Adaptive Scheduling: Dynamic iteration count based on convergence
- Early Stopping: Convergence detection with patience window
- Meta-Learning: Learn optimal iteration parameters (MAML-inspired)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StrategyArm:
    """
    V6: A strategy arm for Thompson Sampling bandit selection.

    Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.
    Each strategy is modeled with a Beta distribution that is updated based
    on observed rewards.

    Attributes:
        name: Strategy identifier (e.g., "map_elites", "dspy", "debate", "ooda")
        alpha: Beta distribution alpha parameter (successes + 1)
        beta: Beta distribution beta parameter (failures + 1)
        pulls: Total times this strategy was selected
        total_reward: Cumulative reward from this strategy

    Methods:
        sample(): Sample from Beta distribution for selection
        update(reward): Update parameters after observing reward
        mean_reward: Expected reward based on current statistics

    Example:
        arm = StrategyArm(name="dspy")
        arm.update(0.8)  # Observed 80% success
        arm.update(0.9)  # Observed 90% success
        print(arm.mean_reward)  # ~0.8
    """
    name: str  # Strategy identifier
    alpha: float = 1.0  # Beta distribution alpha (successes + 1)
    beta: float = 1.0  # Beta distribution beta (failures + 1)
    pulls: int = 0  # Total times this strategy was selected
    total_reward: float = 0.0  # Cumulative reward from this strategy

    def sample(self) -> float:
        """Sample from Beta distribution for Thompson Sampling."""
        return random.betavariate(self.alpha, self.beta)

    def update(self, reward: float) -> None:
        """
        Update arm statistics after observing reward.

        Args:
            reward: Observed reward (0-1 scale)
        """
        self.pulls += 1
        self.total_reward += reward
        # Update Beta distribution parameters
        self.alpha += reward
        self.beta += (1 - reward)

    @property
    def mean_reward(self) -> float:
        """Expected reward based on current statistics."""
        return self.alpha / (self.alpha + self.beta)


@dataclass
class ConvergenceState:
    """
    V6: Track convergence for early stopping decisions.

    Monitors fitness history to detect when improvement has plateaued,
    allowing early termination of the optimization loop.

    Attributes:
        window_size: Number of iterations to consider for trend
        patience: Iterations without improvement before stopping
        min_delta: Minimum improvement to count as progress
        fitness_history: History of fitness scores
        best_fitness: Best fitness observed
        iterations_without_improvement: Counter for early stopping

    Methods:
        update(fitness): Returns True if should continue, False if converged
        get_trend(): Calculate improvement trend over window

    Example:
        state = ConvergenceState(patience=20)
        for i in range(100):
            fitness = run_iteration()
            if not state.update(fitness):
                print("Converged!")
                break
    """
    window_size: int = 10  # Number of iterations to consider
    patience: int = 20  # Iterations without improvement before stopping
    min_delta: float = 0.001  # Minimum improvement to count as progress
    fitness_history: List[float] = field(default_factory=list)
    best_fitness: float = 0.0
    iterations_without_improvement: int = 0

    def update(self, fitness: float) -> bool:
        """
        Update convergence state.

        Args:
            fitness: Current iteration fitness

        Returns:
            True if should continue, False if should stop
        """
        self.fitness_history.append(fitness)

        if fitness > self.best_fitness + self.min_delta:
            self.best_fitness = fitness
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1

        # Check for convergence (early stopping)
        return self.iterations_without_improvement < self.patience

    def get_trend(self) -> float:
        """
        Calculate improvement trend over the window.

        Returns:
            Positive value = improving, negative = declining
        """
        if len(self.fitness_history) < 2:
            return 0.0

        window = self.fitness_history[-self.window_size:]
        if len(window) < 2:
            return 0.0

        # Simple linear regression slope
        n = len(window)
        x_mean = (n - 1) / 2
        y_mean = sum(window) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(window))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        return numerator / denominator if denominator > 0 else 0.0

    @property
    def is_converged(self) -> bool:
        """Check if convergence has been reached."""
        return self.iterations_without_improvement >= self.patience


@dataclass
class IterationMomentum:
    """
    V6: Carry forward successful patterns with decay.

    Tracks which patterns/techniques have been successful recently
    and applies momentum to continue using them with gradual decay.

    Attributes:
        pattern_weights: Current weight for each pattern
        decay_rate: How fast weights decay (0-1)
        min_weight: Minimum weight before pattern is dropped
        history: History of pattern selections and outcomes

    Example:
        momentum = IterationMomentum(decay_rate=0.9)
        momentum.boost("refactoring", 0.8)  # Successful refactoring
        momentum.apply_decay()  # Decay all weights
    """
    pattern_weights: Dict[str, float] = field(default_factory=dict)
    decay_rate: float = 0.9  # How fast weights decay
    min_weight: float = 0.1  # Minimum weight before dropping
    history: List[Dict[str, Any]] = field(default_factory=list)

    def boost(self, pattern: str, success_score: float) -> None:
        """
        Boost a pattern's weight based on success.

        Args:
            pattern: Pattern identifier
            success_score: Success score (0-1)
        """
        current = self.pattern_weights.get(pattern, 0.5)
        # Blend current weight with new success score
        self.pattern_weights[pattern] = 0.7 * current + 0.3 * success_score
        self.history.append({
            "pattern": pattern,
            "score": success_score,
            "new_weight": self.pattern_weights[pattern]
        })

    def apply_decay(self) -> None:
        """Apply decay to all pattern weights."""
        to_remove = []
        for pattern, weight in self.pattern_weights.items():
            new_weight = weight * self.decay_rate
            if new_weight < self.min_weight:
                to_remove.append(pattern)
            else:
                self.pattern_weights[pattern] = new_weight

        for pattern in to_remove:
            del self.pattern_weights[pattern]

    def get_weight(self, pattern: str) -> float:
        """Get current weight for a pattern."""
        return self.pattern_weights.get(pattern, 0.5)


@dataclass
class MetaIterationState:
    """
    V6: Meta-learning state for iteration optimization.

    Learns optimal iteration parameters (like iteration count, strategy mix)
    from historical performance data. Inspired by MAML.

    Attributes:
        optimal_iterations: Learned optimal iteration count
        strategy_mix: Learned optimal strategy mixing weights
        learning_rate: How fast to adapt parameters
        history: Historical performance data for learning

    Example:
        meta = MetaIterationState()
        meta.learn_from_history([
            {"iterations": 10, "fitness": 0.8},
            {"iterations": 20, "fitness": 0.95},
            {"iterations": 30, "fitness": 0.96}
        ])
    """
    optimal_iterations: int = 20
    strategy_mix: Dict[str, float] = field(default_factory=lambda: {
        "dspy": 0.4,
        "map_elites": 0.3,
        "debate": 0.2,
        "ooda": 0.1
    })
    learning_rate: float = 0.1
    history: List[Dict[str, Any]] = field(default_factory=list)

    def learn_from_history(self, results: List[Dict[str, Any]]) -> None:
        """
        Update optimal parameters based on historical results.

        Args:
            results: List of dicts with 'iterations', 'fitness', 'strategy'
        """
        if not results:
            return

        # Find best performing configuration
        best_result = max(results, key=lambda r: r.get("fitness", 0))

        # Update optimal iterations with learning rate
        if "iterations" in best_result:
            target = best_result["iterations"]
            self.optimal_iterations = int(
                self.optimal_iterations + self.learning_rate * (target - self.optimal_iterations)
            )

        # Update strategy mix if strategy info present
        if "strategy" in best_result:
            strategy = best_result["strategy"]
            for s in self.strategy_mix:
                if s == strategy:
                    self.strategy_mix[s] += self.learning_rate * (1 - self.strategy_mix[s])
                else:
                    self.strategy_mix[s] *= (1 - self.learning_rate)

            # Normalize
            total = sum(self.strategy_mix.values())
            if total > 0:
                self.strategy_mix = {s: w / total for s, w in self.strategy_mix.items()}

        self.history.extend(results)

    def suggest_iterations(self) -> int:
        """Suggest optimal number of iterations based on learned parameters."""
        return self.optimal_iterations

    def suggest_strategy(self) -> str:
        """Suggest a strategy based on learned mix weights."""
        r = random.random()
        cumulative = 0.0
        for strategy, weight in self.strategy_mix.items():
            cumulative += weight
            if r <= cumulative:
                return strategy
        return list(self.strategy_mix.keys())[0]
