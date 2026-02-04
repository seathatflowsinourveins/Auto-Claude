"""
V10 Process Reward Models, Constitutional AI & Test-Time Compute

Extracted from ralph_loop.py V10 enhancements.
Implements step-level verification, principled self-critique, and adaptive compute.

Classes:
- ProcessRewardStep: Single step with process-level reward
- PRMState: Process Reward Model verification state
- ConstitutionalPrinciple: A principle in the CAI constitution
- ConstitutionalCritique: Result of self-critique against principles
- CAIState: Constitutional AI self-correction state
- ThinkingBudget: Budget allocation for extended thinking
- TestTimeComputeState: Test-time compute scaling state

References:
- ThinkPRM: Process Reward Models (ICLR 2026)
- Constitutional AI (Anthropic)
- DeepSeek-R1 / OpenAI o1 test-time compute scaling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ProcessRewardStep:
    """
    V10: A single step in a reasoning chain with process-level reward.

    Based on OpenAI's "Let's Verify Step by Step" and ThinkPRM (ICLR 2026).
    Unlike outcome reward models (ORM) that only score final answers,
    PRMs evaluate each intermediate step for correctness.

    Attributes:
        step_index: Position in the reasoning chain
        step_content: The content of this reasoning step
        is_correct: Whether this step is correct
        reward: 1.0 = correct, 0.0 = incorrect, 0.5 = uncertain
        verification_reasoning: CoT explaining correctness assessment
        confidence: Model's confidence in this assessment
        created_at: ISO timestamp of creation
    """
    step_index: int
    step_content: str
    is_correct: bool
    reward: float
    verification_reasoning: str
    confidence: float
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def weighted_reward(self) -> float:
        """Reward weighted by confidence."""
        return self.reward * self.confidence


@dataclass
class PRMState:
    """
    V10: Process Reward Model state for step-level verification.

    Implements the key insight from ThinkPRM: generate verification CoT
    for each step rather than discriminative scoring. This provides:
    - Better interpretability (can see why each step is marked correct/incorrect)
    - Data efficiency (needs fewer labeled examples)
    - Improved generalization across problem types

    Attributes:
        verified_solutions: List of verified solution step chains
        verification_threshold: Minimum confidence to trust verification
        first_error_tracking: Track first error in each solution
        reflective_mode: Allow correct steps after errors
        total_steps_verified: Total steps processed
        correct_steps: Count of correct steps
        incorrect_steps: Count of incorrect steps
        first_error_positions: Where errors typically occur
    """
    verified_solutions: List[List[ProcessRewardStep]] = field(default_factory=list)
    verification_threshold: float = 0.7
    first_error_tracking: bool = True
    reflective_mode: bool = True
    total_steps_verified: int = 0
    correct_steps: int = 0
    incorrect_steps: int = 0
    first_error_positions: List[int] = field(default_factory=list)

    def add_verified_solution(self, steps: List[ProcessRewardStep]) -> Dict[str, Any]:
        """Add a verified solution and return analysis."""
        self.verified_solutions.append(steps)

        first_error_idx = -1
        for i, step in enumerate(steps):
            self.total_steps_verified += 1
            if step.is_correct:
                self.correct_steps += 1
            else:
                self.incorrect_steps += 1
                if first_error_idx == -1:
                    first_error_idx = i
                    self.first_error_positions.append(i)

        if not steps:
            solution_score = 0.0
        else:
            solution_score = sum(s.weighted_reward for s in steps) / len(steps)

        return {
            "solution_index": len(self.verified_solutions) - 1,
            "num_steps": len(steps),
            "solution_score": solution_score,
            "first_error_at": first_error_idx,
            "all_correct": first_error_idx == -1
        }

    def get_prm_score(self, solution_idx: int = -1) -> float:
        """Get PRM score for a solution (product of step probabilities)."""
        if not self.verified_solutions:
            return 0.0

        steps = self.verified_solutions[solution_idx]
        if not steps:
            return 0.0

        score = 1.0
        for step in steps:
            score *= step.weighted_reward
        return score

    def get_best_of_n_ranking(self) -> List[Tuple[int, float]]:
        """Rank all solutions by PRM score for Best-of-N selection."""
        rankings = []
        for i in range(len(self.verified_solutions)):
            score = self.get_prm_score(i)
            rankings.append((i, score))
        return sorted(rankings, key=lambda x: x[1], reverse=True)

    @property
    def accuracy(self) -> float:
        """Step-level accuracy."""
        if self.total_steps_verified == 0:
            return 0.0
        return self.correct_steps / self.total_steps_verified

    @property
    def avg_first_error_position(self) -> float:
        """Average position where first error occurs."""
        if not self.first_error_positions:
            return float('inf')
        return sum(self.first_error_positions) / len(self.first_error_positions)


@dataclass
class ConstitutionalPrinciple:
    """
    V10: A principle in the Constitutional AI constitution.

    Based on Anthropic's CAI paper: principles guide model behavior
    through self-critique and revision cycles.

    Attributes:
        principle_id: Unique principle identifier
        description: Description of the desired behavior
        priority: Priority level for conflict resolution (0-1)
        category: Category of principle
        activation_keywords: Keywords that trigger this principle
    """
    principle_id: str
    description: str
    priority: float
    category: str  # "harmlessness", "helpfulness", "honesty", "reasoning"
    activation_keywords: List[str] = field(default_factory=list)

    def should_activate(self, context: str) -> bool:
        """Check if this principle should be applied to given context."""
        if not self.activation_keywords:
            return True
        context_lower = context.lower()
        return any(kw.lower() in context_lower for kw in self.activation_keywords)


@dataclass
class ConstitutionalCritique:
    """
    V10: Result of a self-critique against constitutional principles.

    Attributes:
        principle: The principle used for critique
        original_response: The original response being critiqued
        critique: What's wrong with the response
        revised_response: The improved response
        improvement_score: How much better the revision is (0-1)
        created_at: ISO timestamp of creation
    """
    principle: ConstitutionalPrinciple
    original_response: str
    critique: str
    revised_response: str
    improvement_score: float
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class CAIState:
    """
    V10: Constitutional AI state for self-correction.

    Implements the CAI training loop at inference time:
    1. Generate initial response
    2. Critique response against principles
    3. Revise response based on critique
    4. Repeat until satisfactory or max iterations

    Attributes:
        constitution: List of constitutional principles
        critiques: History of critiques performed
        max_revision_rounds: Maximum revision attempts
        improvement_threshold: Minimum improvement to continue
        total_critiques: Total critiques performed
        successful_revisions: Revisions that improved score
        avg_improvement: Average improvement per critique
    """
    constitution: List[ConstitutionalPrinciple] = field(default_factory=list)
    critiques: List[ConstitutionalCritique] = field(default_factory=list)
    max_revision_rounds: int = 3
    improvement_threshold: float = 0.1
    total_critiques: int = 0
    successful_revisions: int = 0
    avg_improvement: float = 0.0

    def add_principle(self, principle_id: str, description: str,
                     priority: float = 0.5, category: str = "reasoning",
                     activation_keywords: Optional[List[str]] = None) -> ConstitutionalPrinciple:
        """Add a principle to the constitution."""
        principle = ConstitutionalPrinciple(
            principle_id=principle_id,
            description=description,
            priority=priority,
            category=category,
            activation_keywords=activation_keywords or []
        )
        self.constitution.append(principle)
        return principle

    def get_applicable_principles(self, context: str) -> List[ConstitutionalPrinciple]:
        """Get principles that apply to the given context, sorted by priority."""
        applicable = [p for p in self.constitution if p.should_activate(context)]
        return sorted(applicable, key=lambda p: p.priority, reverse=True)

    def add_critique(self, critique: ConstitutionalCritique) -> None:
        """Record a critique and update statistics."""
        self.critiques.append(critique)
        self.total_critiques += 1

        if critique.improvement_score > 0:
            self.successful_revisions += 1
            self.avg_improvement = (
                (self.avg_improvement * (self.total_critiques - 1) + critique.improvement_score)
                / self.total_critiques
            )

    def get_critique_effectiveness(self) -> float:
        """Ratio of critiques that led to improvements."""
        if self.total_critiques == 0:
            return 0.0
        return self.successful_revisions / self.total_critiques


@dataclass
class ThinkingBudget:
    """
    V10: Budget allocation for extended thinking (test-time compute).

    Manages the allocation of "thinking tokens" - the internal reasoning
    budget that models like o1 and DeepSeek-R1 use for complex problems.

    Attributes:
        total_budget: Max thinking tokens (Claude's extended thinking limit)
        used_budget: Tokens already used
        budget_per_step: Default allocation per reasoning step
        adaptive_scaling: Whether to scale based on difficulty
        easy_multiplier: Budget multiplier for easy problems
        medium_multiplier: Budget multiplier for medium problems
        hard_multiplier: Budget multiplier for hard problems
        ultrahard_multiplier: Budget multiplier for ultrahard problems
    """
    total_budget: int = 128000
    used_budget: int = 0
    budget_per_step: int = 8000
    adaptive_scaling: bool = True
    easy_multiplier: float = 0.25
    medium_multiplier: float = 0.5
    hard_multiplier: float = 1.0
    ultrahard_multiplier: float = 2.0

    def allocate(self, difficulty: str = "medium") -> int:
        """Allocate thinking budget based on difficulty."""
        multipliers = {
            "easy": self.easy_multiplier,
            "medium": self.medium_multiplier,
            "hard": self.hard_multiplier,
            "ultrahard": self.ultrahard_multiplier
        }
        multiplier = multipliers.get(difficulty, self.medium_multiplier)
        allocation = int(self.budget_per_step * multiplier)

        remaining = self.total_budget - self.used_budget
        allocation = min(allocation, remaining)

        self.used_budget += allocation
        return allocation

    @property
    def remaining(self) -> int:
        """Remaining thinking budget."""
        return max(0, self.total_budget - self.used_budget)

    @property
    def utilization(self) -> float:
        """Fraction of budget used."""
        return self.used_budget / self.total_budget if self.total_budget > 0 else 0.0

    def reset(self) -> None:
        """Reset budget for new problem."""
        self.used_budget = 0


@dataclass
class TestTimeComputeState:
    """
    V10: Test-Time Compute Scaling state.

    Implements the paradigm shift from train-time to inference-time scaling.
    Based on DeepSeek-R1 and OpenAI o1 approaches:
    - Allocate more compute to harder problems
    - Use extended CoT for complex reasoning
    - Dynamic budget allocation based on problem characteristics

    Attributes:
        thinking_budget: Budget allocation manager
        current_difficulty: Current problem difficulty estimate
        difficulty_history: History of (difficulty, fitness) pairs
        scaling_decisions: Log of scaling decisions made
        total_thinking_tokens_used: Total tokens used for thinking
        problems_solved: Number of problems processed
        easy_threshold: Fitness threshold for easy problems
        hard_threshold: Fitness threshold for hard problems
    """
    thinking_budget: ThinkingBudget = field(default_factory=ThinkingBudget)
    current_difficulty: str = "medium"
    difficulty_history: List[Tuple[str, float]] = field(default_factory=list)
    scaling_decisions: List[Dict[str, Any]] = field(default_factory=list)
    total_thinking_tokens_used: int = 0
    problems_solved: int = 0
    easy_threshold: float = 0.8
    hard_threshold: float = 0.4

    def estimate_difficulty(self, problem_features: Dict[str, Any]) -> str:
        """Estimate problem difficulty from features."""
        length = problem_features.get("length", 0)
        num_constraints = problem_features.get("constraints", 0)
        novelty = problem_features.get("novelty", 0.5)
        past_failures = problem_features.get("past_failures", 0)

        difficulty_score = (
            (length / 1000) * 0.2 +
            (num_constraints / 10) * 0.3 +
            novelty * 0.3 +
            (past_failures / 5) * 0.2
        )

        if difficulty_score < 0.25:
            return "easy"
        elif difficulty_score < 0.5:
            return "medium"
        elif difficulty_score < 0.75:
            return "hard"
        else:
            return "ultrahard"

    def allocate_compute(self, problem_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Allocate test-time compute for a problem."""
        if problem_features:
            self.current_difficulty = self.estimate_difficulty(problem_features)

        tokens_allocated = self.thinking_budget.allocate(self.current_difficulty)

        decision = {
            "difficulty": self.current_difficulty,
            "tokens_allocated": tokens_allocated,
            "budget_remaining": self.thinking_budget.remaining,
            "utilization": self.thinking_budget.utilization
        }

        self.scaling_decisions.append(decision)
        self.total_thinking_tokens_used += tokens_allocated

        return decision

    def record_outcome(self, fitness: float) -> None:
        """Record outcome to update difficulty calibration."""
        self.difficulty_history.append((self.current_difficulty, fitness))
        self.problems_solved += 1

        alpha = 0.1
        if self.current_difficulty == "easy" and fitness > 0.7:
            pass
        elif self.current_difficulty in ["hard", "ultrahard"] and fitness < 0.5:
            pass
        else:
            if fitness > 0.7:
                self.easy_threshold = alpha * fitness + (1 - alpha) * self.easy_threshold
            elif fitness < 0.4:
                self.hard_threshold = alpha * fitness + (1 - alpha) * self.hard_threshold

    def get_compute_efficiency(self) -> float:
        """Compute efficiency: fitness per thinking token."""
        if self.total_thinking_tokens_used == 0:
            return 0.0

        total_fitness = sum(f for _, f in self.difficulty_history)
        return total_fitness / (self.total_thinking_tokens_used / 1000)

    def reset_for_new_problem(self) -> None:
        """Reset state for a new problem."""
        self.thinking_budget.reset()
        self.current_difficulty = "medium"


__all__ = [
    "ProcessRewardStep",
    "PRMState",
    "ConstitutionalPrinciple",
    "ConstitutionalCritique",
    "CAIState",
    "ThinkingBudget",
    "TestTimeComputeState",
]
