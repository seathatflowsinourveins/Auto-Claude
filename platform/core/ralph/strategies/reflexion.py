"""
V4 Reflexion Pattern - Self-Enhancement Strategies

Extracted from ralph_loop.py V4 enhancements.
Implements the Reflexion pattern, Multi-Agent Debate, and Procedural Memory.

Classes:
- Reflection: Natural language reflection on failures (Reflexion pattern)
- DebatePosition: Position in multi-agent debate (DMAD pattern)
- ProceduralSkill: Reusable skill with Bayesian reliability (MACLA pattern)

References:
- Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al.)
- DMAD: Debating with More Agents (Du et al., ICLR 2025)
- MACLA: Multi-Agent Procedural Memory (arxiv)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Reflection:
    """
    V4: Natural language reflection on a failure (Reflexion pattern).

    The Reflexion pattern stores failures as episodic memory with natural
    language reflections that can be retrieved and applied in future iterations.

    Attributes:
        iteration: The iteration where the failure occurred
        failure_description: What went wrong
        reflection_text: What was learned from this failure
        corrective_action: What to do differently next time
        created_at: ISO timestamp of when the reflection was created
        applied_in_iteration: Iteration where this reflection was used (if any)

    Example:
        reflection = Reflection(
            iteration=5,
            failure_description="Test suite failed due to missing edge case",
            reflection_text="Need to consider empty input scenarios",
            corrective_action="Add empty input validation before processing",
            created_at="2026-02-02T10:00:00Z"
        )
    """
    iteration: int
    failure_description: str
    reflection_text: str  # What I learned from this failure
    corrective_action: str  # What to do differently next time
    created_at: str
    applied_in_iteration: Optional[int] = None  # When this reflection was used


@dataclass
class DebatePosition:
    """
    V4: A position in multi-agent debate (DMAD pattern).

    In multi-agent debate, different agent perspectives argue positions
    before reaching a consensus decision. This improves reasoning quality
    through adversarial verification.

    Attributes:
        agent_perspective: Role of the debating agent
            - "optimist": Focuses on potential benefits
            - "critic": Identifies risks and flaws
            - "pragmatist": Considers practical feasibility
            - "innovator": Proposes novel approaches
        argument: The position being argued
        confidence: Confidence level (0.0 to 1.0)
        supporting_evidence: Evidence supporting the position

    Example:
        position = DebatePosition(
            agent_perspective="critic",
            argument="This approach may not scale to large datasets",
            confidence=0.8,
            supporting_evidence=["O(n^2) complexity in core loop"]
        )
    """
    agent_perspective: str  # "optimist", "critic", "pragmatist", "innovator"
    argument: str
    confidence: float  # 0.0 to 1.0
    supporting_evidence: List[str] = field(default_factory=list)


@dataclass
class ProceduralSkill:
    """
    V4: Extracted reusable skill with Bayesian reliability (MACLA pattern).

    Procedural skills are extracted patterns that can be reused across
    iterations. Each skill has Bayesian reliability tracking to measure
    how often it succeeds vs fails.

    Attributes:
        name: Unique identifier for the skill
        description: What this skill does
        preconditions: When to apply this skill
        steps: How to execute the skill
        success_count: Number of successful applications
        failure_count: Number of failed applications
        last_used: ISO timestamp of last use

    Properties:
        reliability: Bayesian reliability estimate using Laplace smoothing
            Formula: (successes + 1) / (total + 2)

    Example:
        skill = ProceduralSkill(
            name="error_handling",
            description="Add proper error handling to function",
            preconditions=["Function lacks try/except", "API calls present"],
            steps=["Identify failure points", "Add try/except", "Log errors"],
            success_count=8,
            failure_count=2
        )
        print(skill.reliability)  # 0.75 = (8+1)/(10+2)
    """
    name: str
    description: str
    preconditions: List[str]  # When to apply this skill
    steps: List[str]  # How to execute
    success_count: int = 0
    failure_count: int = 0
    last_used: Optional[str] = None

    @property
    def reliability(self) -> float:
        """
        Bayesian reliability estimate: (successes + 1) / (total + 2).

        Uses Laplace smoothing to handle cases with few observations.
        Returns value between 0.33 (no successes) and 1.0 (all successes).
        """
        total = self.success_count + self.failure_count
        return (self.success_count + 1) / (total + 2)  # Laplace smoothing

    def record_success(self) -> None:
        """Record a successful application of this skill."""
        self.success_count += 1

    def record_failure(self) -> None:
        """Record a failed application of this skill."""
        self.failure_count += 1
