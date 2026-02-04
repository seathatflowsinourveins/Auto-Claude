"""
V5 Consistency Pattern - Advanced Self-Enhancement Strategies

Extracted from ralph_loop.py V5 enhancements.
Implements Self-Consistency, Chain-of-Verification, OODA Loop, and RISE.

Classes:
- ConsistencyPath: Reasoning path for self-consistency voting (Google 2023)
- VerificationStep: Step in Chain-of-Verification (Meta AI, +94% accuracy)
- OODAState: OODA Loop state (Observe-Orient-Decide-Act)
- RISEAttempt: Recursive IntroSpEction for multi-turn self-correction

References:
- Self-Consistency (Wang et al., Google 2023)
- Chain-of-Verification (Dhuliawala et al., Meta AI)
- OODA Loop (Boyd's Decision Framework)
- RISE: Recursive IntroSpEction (arxiv)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ConsistencyPath:
    """
    V5: A reasoning path for self-consistency voting (Google 2023 paper).

    Self-consistency samples multiple reasoning paths to the same question
    and takes a majority vote. This improves accuracy by aggregating
    diverse solutions.

    Attributes:
        path_id: Unique identifier for this path
        reasoning_chain: The reasoning steps taken
        answer: The final answer derived
        confidence: Model confidence in this path (0.0 to 1.0)
        created_at: ISO timestamp

    Example:
        path = ConsistencyPath(
            path_id=1,
            reasoning_chain="Step 1: Parse input... Step 2: Apply formula...",
            answer=42,
            confidence=0.85,
            created_at="2026-02-02T10:00:00Z"
        )
    """
    path_id: int
    reasoning_chain: str
    answer: Any
    confidence: float
    created_at: str


@dataclass
class VerificationStep:
    """
    V5: A step in Chain-of-Verification (CoVe) process.

    CoVe is a 4-step verification process:
    1. "plan": Plan the verification approach
    2. "execute": Execute the initial response
    3. "factor": Factor into verification questions
    4. "verify": Verify each factored question

    This achieves +94% accuracy improvement in factual tasks.

    Attributes:
        step_type: Type of step ("plan", "execute", "factor", "verify")
        question: The question being addressed
        answer: The answer produced
        verified: Whether this step passed verification
        created_at: ISO timestamp

    Example:
        step = VerificationStep(
            step_type="verify",
            question="Is the date correct?",
            answer="Yes, confirmed via authoritative source",
            verified=True,
            created_at="2026-02-02T10:00:00Z"
        )
    """
    step_type: str  # "plan", "execute", "factor", "verify"
    question: str
    answer: str
    verified: bool
    created_at: str


@dataclass
class OODAState:
    """
    V5: OODA Loop state (Observe-Orient-Decide-Act).

    Boyd's OODA loop is a decision framework that cycles through:
    1. Observe: Gather information
    2. Orient: Analyze and synthesize
    3. Decide: Determine action
    4. Act: Execute decision

    Attributes:
        phase: Current phase in the loop
        observations: List of observations gathered
        orientation: Current understanding/mental model
        decision: The decision made
        action_taken: The action executed
        outcome: Optional result of the action

    Example:
        state = OODAState(
            phase="decide",
            observations=["Tests failing", "Coverage at 60%"],
            orientation="Need to improve test coverage",
            decision="Add unit tests for uncovered functions",
            action_taken=""
        )
    """
    phase: str  # "observe", "orient", "decide", "act"
    observations: List[str]
    orientation: str  # Current understanding/model
    decision: str
    action_taken: str
    outcome: Optional[str] = None


@dataclass
class RISEAttempt:
    """
    V5: Recursive IntroSpEction attempt for multi-turn self-correction.

    RISE performs recursive self-correction by:
    1. Generating an initial response
    2. Getting feedback on the response
    3. Introspecting on what needs to change
    4. Producing a corrected response
    5. Measuring improvement

    Attributes:
        turn: The turn number (1, 2, 3, ...)
        previous_response: The response being corrected
        feedback: Feedback on what was wrong
        introspection: Analysis of what needs to change
        corrected_response: The improved response
        improvement_score: Quantified improvement (0.0 to 1.0)

    Example:
        attempt = RISEAttempt(
            turn=2,
            previous_response="The answer is 41",
            feedback="Off by one error in calculation",
            introspection="Need to add 1 to final sum",
            corrected_response="The answer is 42",
            improvement_score=0.9
        )
    """
    turn: int
    previous_response: str
    feedback: str
    introspection: str  # What needs to change
    corrected_response: str
    improvement_score: float  # Did this turn improve things?
