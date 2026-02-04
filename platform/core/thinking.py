"""
UAP Thinking Module - Extended thinking patterns for Claude agents.

Implements thinking patterns based on Anthropic's extended thinking research:
- Budget-based thinking with configurable token limits
- Interleaved thinking (think-act-think cycles)
- Structured reasoning with metacognition
- Confidence calibration through deliberate reasoning
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class ThinkingStrategy(str, Enum):
    """Strategies for extended thinking."""

    CHAIN_OF_THOUGHT = "chain_of_thought"  # Linear reasoning chain
    TREE_OF_THOUGHTS = "tree_of_thoughts"  # Branching exploration
    SELF_CONSISTENCY = "self_consistency"  # Multiple paths, vote
    METACOGNITIVE = "metacognitive"        # Think about thinking
    REFLEXION = "reflexion"                # Memory + self-critique


class ReasoningType(str, Enum):
    """Types of reasoning steps."""

    OBSERVATION = "observation"    # What do I see/know?
    ANALYSIS = "analysis"          # What does this mean?
    HYPOTHESIS = "hypothesis"      # What might be true?
    EVALUATION = "evaluation"      # Is this correct?
    SYNTHESIS = "synthesis"        # Combine insights
    DECISION = "decision"          # What should I do?
    REFLECTION = "reflection"      # What did I learn?


class ConfidenceLevel(str, Enum):
    """Calibrated confidence levels."""

    VERY_LOW = "very_low"      # <20% - Highly uncertain
    LOW = "low"                # 20-40% - Significant doubt
    MEDIUM = "medium"          # 40-60% - Uncertain
    HIGH = "high"              # 60-80% - Fairly confident
    VERY_HIGH = "very_high"    # >80% - Highly confident


# =============================================================================
# Data Models
# =============================================================================

class ReasoningStep(BaseModel):
    """A single step in the reasoning chain."""

    step_type: ReasoningType
    content: str
    evidence: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    token_count: int = 0
    timestamp: float = Field(default_factory=time.time)


class ThinkingChain(BaseModel):
    """A chain of reasoning steps."""

    id: str
    question: str
    steps: List[ReasoningStep] = Field(default_factory=list)
    conclusion: Optional[str] = None
    final_confidence: float = 0.5
    total_tokens: int = 0
    strategy: ThinkingStrategy = ThinkingStrategy.CHAIN_OF_THOUGHT
    started_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None


class ThinkingBudget(BaseModel):
    """Token budget configuration for thinking.

    Aligned with Anthropic's Extended Thinking API requirements:
    - budget_tokens: Minimum 1024 tokens (Anthropic requirement)
    - Supports up to 128K tokens for complex reasoning

    For Anthropic's extended thinking, use:
    - budget_tokens: Primary thinking budget (min 1024)
    - interleaved_thinking: Enable thinking between tool calls (beta)
    """

    # Primary budget for Anthropic extended thinking (min 1024 per API docs)
    budget_tokens: int = Field(default=8000, ge=1024, le=128000)

    # Legacy fields for backward compatibility
    total_tokens: int = Field(default=16000, ge=1000, le=128000)
    min_tokens: int = Field(default=1024, ge=1024)  # Updated minimum to 1024
    max_tokens_per_step: int = Field(default=4000, ge=500)
    reserved_for_output: int = Field(default=2000, ge=500)

    # Interleaved thinking support (beta: thinking-interleaved-2025-02-19)
    interleaved_thinking: bool = Field(
        default=False,
        description="Enable interleaved thinking between tool calls (beta feature)"
    )

    @property
    def available_for_thinking(self) -> int:
        """Tokens available for thinking (excluding output reserve)."""
        return self.total_tokens - self.reserved_for_output

    def to_anthropic_config(self) -> Dict[str, Any]:
        """Convert to Anthropic API thinking configuration.

        Returns:
            Dict suitable for the 'thinking' parameter in Anthropic API calls.
        """
        return {
            "type": "enabled",
            "budget_tokens": self.budget_tokens
        }

    def get_beta_headers(self) -> List[str]:
        """Get beta headers for Anthropic API call.

        Returns:
            List of beta feature headers to include.
        """
        headers = []
        if self.interleaved_thinking:
            # Beta header for interleaved thinking
            headers.append("interleaved-thinking-2025-05-14")
        return headers


class MetacognitiveState(BaseModel):
    """State for metacognitive reasoning."""

    current_understanding: str = ""
    knowledge_gaps: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    potential_biases: List[str] = Field(default_factory=list)
    confidence_calibration: float = 0.5
    reasoning_quality: float = 0.5


# =============================================================================
# Thinking Patterns
# =============================================================================

@dataclass
class ThinkingPattern:
    """Base class for thinking patterns."""

    name: str
    description: str
    strategy: ThinkingStrategy

    def generate_prompt_prefix(self) -> str:
        """Generate prompt prefix for this pattern."""
        return f"Using {self.name} thinking pattern: {self.description}"


# Pre-defined thinking patterns
PATTERNS: Dict[str, ThinkingPattern] = {
    "analytical": ThinkingPattern(
        name="Analytical",
        description="Break down problems systematically, identify components and relationships",
        strategy=ThinkingStrategy.CHAIN_OF_THOUGHT,
    ),
    "exploratory": ThinkingPattern(
        name="Exploratory",
        description="Consider multiple possibilities, branch out before converging",
        strategy=ThinkingStrategy.TREE_OF_THOUGHTS,
    ),
    "critical": ThinkingPattern(
        name="Critical",
        description="Question assumptions, seek disconfirming evidence, devil's advocate",
        strategy=ThinkingStrategy.SELF_CONSISTENCY,
    ),
    "creative": ThinkingPattern(
        name="Creative",
        description="Make unexpected connections, reframe problems, lateral thinking",
        strategy=ThinkingStrategy.TREE_OF_THOUGHTS,
    ),
    "metacognitive": ThinkingPattern(
        name="Metacognitive",
        description="Monitor own reasoning, identify biases, calibrate confidence",
        strategy=ThinkingStrategy.METACOGNITIVE,
    ),
    "reflective": ThinkingPattern(
        name="Reflective",
        description="Learn from experience, update beliefs, iterate on solutions",
        strategy=ThinkingStrategy.REFLEXION,
    ),
}


# =============================================================================
# Anthropic Extended Thinking Support
# =============================================================================

# Models that support extended thinking (as of 2025)
EXTENDED_THINKING_MODELS: Dict[str, Dict[str, Any]] = {
    "claude-opus-4-5-20250514": {
        "alias": "claude-opus-4.5",
        "max_thinking_tokens": 128000,
        "supports_interleaved": True,
    },
    "claude-sonnet-4-5-20250514": {
        "alias": "claude-sonnet-4.5",
        "max_thinking_tokens": 128000,
        "supports_interleaved": True,
    },
    "claude-3-7-sonnet-20250219": {
        "alias": "claude-3.7-sonnet",
        "max_thinking_tokens": 128000,
        "supports_interleaved": True,
    },
    "claude-haiku-4-5-20250514": {
        "alias": "claude-haiku-4.5",
        "max_thinking_tokens": 64000,
        "supports_interleaved": True,
    },
}


def supports_extended_thinking(model: str) -> bool:
    """Check if a model supports extended thinking.

    Args:
        model: Model name or alias

    Returns:
        True if the model supports extended thinking
    """
    model_lower = model.lower()
    for model_id, info in EXTENDED_THINKING_MODELS.items():
        if model_lower in model_id.lower() or model_lower in info["alias"].lower():
            return True
    return False


def get_max_thinking_tokens(model: str) -> int:
    """Get the maximum thinking tokens for a model.

    Args:
        model: Model name or alias

    Returns:
        Maximum thinking tokens (defaults to 64000 if unknown)
    """
    model_lower = model.lower()
    for model_id, info in EXTENDED_THINKING_MODELS.items():
        if model_lower in model_id.lower() or model_lower in info["alias"].lower():
            return info["max_thinking_tokens"]
    return 64000  # Conservative default


def supports_interleaved_thinking(model: str) -> bool:
    """Check if a model supports interleaved thinking (beta).

    Interleaved thinking allows thinking blocks between tool calls
    in agentic workflows.

    Args:
        model: Model name or alias

    Returns:
        True if the model supports interleaved thinking
    """
    model_lower = model.lower()
    for model_id, info in EXTENDED_THINKING_MODELS.items():
        if model_lower in model_id.lower() or model_lower in info["alias"].lower():
            return info.get("supports_interleaved", False)
    return False


def create_thinking_config(
    budget_tokens: int = 8000,
    interleaved: bool = False,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """Create a thinking configuration for Anthropic API calls.

    Args:
        budget_tokens: Token budget for thinking (min 1024)
        interleaved: Enable interleaved thinking (beta)
        model: Optional model name to validate against

    Returns:
        Dict with 'thinking' config and optional 'beta_headers'

    Example:
        config = create_thinking_config(budget_tokens=16000, interleaved=True)
        # Use config['thinking'] in API call
        # Use config['beta_headers'] for extra_headers
    """
    # Validate budget
    if budget_tokens < 1024:
        budget_tokens = 1024
        logger.warning("budget_tokens increased to minimum of 1024")

    if model:
        max_tokens = get_max_thinking_tokens(model)
        if budget_tokens > max_tokens:
            budget_tokens = max_tokens
            logger.warning(f"budget_tokens capped to {max_tokens} for model {model}")

    result = {
        "thinking": {
            "type": "enabled",
            "budget_tokens": budget_tokens
        }
    }

    if interleaved:
        result["beta_headers"] = ["interleaved-thinking-2025-05-14"]

    return result


# =============================================================================
# Thinking Engine
# =============================================================================

class ThinkingEngine:
    """
    Engine for extended thinking with Claude.

    Manages thinking chains, budget allocation, and pattern application.
    In production, this would integrate with Claude's extended thinking API.
    """

    def __init__(
        self,
        budget: Optional[ThinkingBudget] = None,
        default_pattern: str = "analytical",
    ):
        self._budget = budget or ThinkingBudget()
        self._default_pattern = PATTERNS.get(default_pattern, PATTERNS["analytical"])
        self._active_chains: Dict[str, ThinkingChain] = {}
        self._metacognitive_state = MetacognitiveState()
        self._total_tokens_used = 0

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def tokens_remaining(self) -> int:
        """Get remaining tokens in budget."""
        return self._budget.available_for_thinking - self._total_tokens_used

    @property
    def metacognitive_state(self) -> MetacognitiveState:
        """Get current metacognitive state."""
        return self._metacognitive_state

    # -------------------------------------------------------------------------
    # Chain Management
    # -------------------------------------------------------------------------

    def begin_thinking(
        self,
        question: str,
        chain_id: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> ThinkingChain:
        """
        Begin a new thinking chain.

        Args:
            question: The question or problem to think about
            chain_id: Optional ID (generated if not provided)
            pattern: Name of thinking pattern to use

        Returns:
            New ThinkingChain instance
        """
        import uuid

        chain_id = chain_id or str(uuid.uuid4())
        selected_pattern = PATTERNS.get(pattern or "", self._default_pattern)

        chain = ThinkingChain(
            id=chain_id,
            question=question,
            strategy=selected_pattern.strategy,
        )

        self._active_chains[chain_id] = chain
        logger.info(f"Started thinking chain {chain_id}: {question[:50]}...")

        return chain

    def add_reasoning_step(
        self,
        chain_id: str,
        step_type: ReasoningType,
        content: str,
        evidence: Optional[List[str]] = None,
        confidence: float = 0.5,
    ) -> bool:
        """
        Add a reasoning step to a chain.

        Args:
            chain_id: ID of the thinking chain
            step_type: Type of reasoning step
            content: The reasoning content
            evidence: Supporting evidence
            confidence: Confidence in this step

        Returns:
            True if added, False if chain not found or budget exceeded
        """
        if chain_id not in self._active_chains:
            logger.warning(f"Chain {chain_id} not found")
            return False

        # Estimate tokens (rough: 4 chars per token)
        token_estimate = len(content) // 4

        if token_estimate > self._budget.max_tokens_per_step:
            logger.warning(f"Step exceeds max tokens: {token_estimate}")
            return False

        if token_estimate > self.tokens_remaining:
            logger.warning(f"Budget exceeded: need {token_estimate}, have {self.tokens_remaining}")
            return False

        step = ReasoningStep(
            step_type=step_type,
            content=content,
            evidence=evidence or [],
            confidence=confidence,
            token_count=token_estimate,
        )

        chain = self._active_chains[chain_id]
        chain.steps.append(step)
        chain.total_tokens += token_estimate
        self._total_tokens_used += token_estimate

        return True

    def conclude_thinking(
        self,
        chain_id: str,
        conclusion: str,
        final_confidence: Optional[float] = None,
    ) -> Optional[ThinkingChain]:
        """
        Conclude a thinking chain.

        Args:
            chain_id: ID of the chain to conclude
            conclusion: The final conclusion
            final_confidence: Overall confidence (computed if not provided)

        Returns:
            Completed ThinkingChain or None if not found
        """
        if chain_id not in self._active_chains:
            return None

        chain = self._active_chains[chain_id]
        chain.conclusion = conclusion
        chain.completed_at = time.time()

        # Compute final confidence from steps if not provided
        if final_confidence is not None:
            chain.final_confidence = final_confidence
        elif chain.steps:
            # Weighted average based on step type importance
            weights = {
                ReasoningType.OBSERVATION: 0.6,
                ReasoningType.ANALYSIS: 0.8,
                ReasoningType.HYPOTHESIS: 0.7,
                ReasoningType.EVALUATION: 0.9,
                ReasoningType.SYNTHESIS: 0.85,
                ReasoningType.DECISION: 1.0,
                ReasoningType.REFLECTION: 0.75,
            }
            total_weight = 0.0
            weighted_confidence = 0.0
            for step in chain.steps:
                weight = weights.get(step.step_type, 0.5)
                weighted_confidence += step.confidence * weight
                total_weight += weight
            chain.final_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.5

        logger.info(f"Concluded chain {chain_id} with confidence {chain.final_confidence:.2f}")
        return chain

    # -------------------------------------------------------------------------
    # Metacognitive Methods
    # -------------------------------------------------------------------------

    def update_metacognitive_state(
        self,
        understanding: Optional[str] = None,
        gaps: Optional[List[str]] = None,
        assumptions: Optional[List[str]] = None,
        biases: Optional[List[str]] = None,
    ) -> None:
        """Update the metacognitive state."""
        if understanding:
            self._metacognitive_state.current_understanding = understanding
        if gaps:
            self._metacognitive_state.knowledge_gaps.extend(gaps)
        if assumptions:
            self._metacognitive_state.assumptions.extend(assumptions)
        if biases:
            self._metacognitive_state.potential_biases.extend(biases)

    def get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to calibrated level."""
        if confidence < 0.2:
            return ConfidenceLevel.VERY_LOW
        elif confidence < 0.4:
            return ConfidenceLevel.LOW
        elif confidence < 0.6:
            return ConfidenceLevel.MEDIUM
        elif confidence < 0.8:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    def think_through(
        self,
        question: str,
        pattern: str = "analytical",
    ) -> ThinkingChain:
        """
        Quick method to think through a question with default steps.

        Args:
            question: The question to think about
            pattern: Thinking pattern to use

        Returns:
            Completed thinking chain
        """
        chain = self.begin_thinking(question, pattern=pattern)

        # Add standard reasoning steps
        self.add_reasoning_step(
            chain.id,
            ReasoningType.OBSERVATION,
            f"Observing the question: {question}",
            confidence=0.9,
        )

        self.add_reasoning_step(
            chain.id,
            ReasoningType.ANALYSIS,
            "Analyzing the key components and requirements...",
            confidence=0.7,
        )

        self.add_reasoning_step(
            chain.id,
            ReasoningType.HYPOTHESIS,
            "Forming initial hypotheses about possible approaches...",
            confidence=0.6,
        )

        self.add_reasoning_step(
            chain.id,
            ReasoningType.EVALUATION,
            "Evaluating the hypotheses against available evidence...",
            confidence=0.75,
        )

        self.add_reasoning_step(
            chain.id,
            ReasoningType.SYNTHESIS,
            "Synthesizing insights into a coherent understanding...",
            confidence=0.8,
        )

        return self.conclude_thinking(
            chain.id,
            "Completed thinking process with synthesized understanding.",
        ) or chain

    def get_chain(self, chain_id: str) -> Optional[ThinkingChain]:
        """Get a thinking chain by ID."""
        return self._active_chains.get(chain_id)

    def get_all_chains(self) -> List[ThinkingChain]:
        """Get all active thinking chains."""
        return list(self._active_chains.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get thinking engine statistics."""
        return {
            "total_chains": len(self._active_chains),
            "total_tokens_used": self._total_tokens_used,
            "tokens_remaining": self.tokens_remaining,
            "budget_total": self._budget.total_tokens,
            "budget_utilization": self._total_tokens_used / self._budget.available_for_thinking
            if self._budget.available_for_thinking > 0
            else 0.0,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_thinking_engine(
    budget_tokens: int = 16000,
    pattern: str = "analytical",
) -> ThinkingEngine:
    """
    Factory function to create a ThinkingEngine.

    Args:
        budget_tokens: Total token budget for thinking
        pattern: Default thinking pattern

    Returns:
        Configured ThinkingEngine instance
    """
    budget = ThinkingBudget(total_tokens=budget_tokens)
    return ThinkingEngine(budget=budget, default_pattern=pattern)


def estimate_thinking_budget(task_complexity: str) -> int:
    """
    Estimate appropriate thinking budget based on task complexity.

    Args:
        task_complexity: One of "simple", "moderate", "complex", "ultrathink"

    Returns:
        Recommended token budget
    """
    budgets = {
        "simple": 4000,
        "moderate": 16000,
        "complex": 64000,
        "ultrathink": 128000,
    }
    return budgets.get(task_complexity.lower(), 16000)


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate the ThinkingEngine."""
    print("=" * 60)
    print("UAP Thinking Engine Demo")
    print("=" * 60)

    # Create engine
    engine = create_thinking_engine(budget_tokens=32000, pattern="analytical")
    print(f"\nCreated engine with {engine.tokens_remaining} tokens available")

    # Think through a question
    print("\n[Thinking Through Question]")
    chain = engine.think_through(
        "How should we architect a multi-agent system for autonomous trading?",
        pattern="analytical",
    )

    print(f"  Chain ID: {chain.id}")
    print(f"  Steps: {len(chain.steps)}")
    print(f"  Total tokens: {chain.total_tokens}")
    print(f"  Final confidence: {chain.final_confidence:.2f}")
    print(f"  Confidence level: {engine.get_confidence_level(chain.final_confidence).value}")

    # Show stats
    stats = engine.get_stats()
    print("\n[Engine Stats]")
    print(f"  Total chains: {stats['total_chains']}")
    print(f"  Tokens used: {stats['total_tokens_used']}")
    print(f"  Budget utilization: {stats['budget_utilization']:.1%}")

    # Demonstrate metacognitive state
    print("\n[Metacognitive State]")
    engine.update_metacognitive_state(
        understanding="Multi-agent trading requires careful coordination",
        gaps=["Market microstructure impact", "Latency requirements"],
        assumptions=["Agents can communicate quickly"],
        biases=["Over-optimism about coordination"],
    )
    state = engine.metacognitive_state
    print(f"  Understanding: {state.current_understanding}")
    print(f"  Knowledge gaps: {state.knowledge_gaps}")
    print(f"  Assumptions: {state.assumptions}")
    print(f"  Potential biases: {state.potential_biases}")


if __name__ == "__main__":
    demo()
