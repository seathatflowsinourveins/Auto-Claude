#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pydantic>=2.0.0",
# ]
# ///
"""
UAP Ultrathink Module - Advanced Extended Thinking for Claude Agents.

Implements Claude Code's thinking levels (think, hardthink, ultrathink) with:
- Power word detection and automatic budget escalation
- Structured chain-of-thought with phases
- Tree of Thoughts exploration with branch evaluation
- Self-consistency via multi-path voting
- Evidence-based confidence calibration

Based on:
- Claude Code's extended thinking: https://claudelog.com/faqs/what-is-ultrathink/
- Anthropic's thinking mode research
- Tree of Thoughts paper (Yao et al., 2023)

Usage:
    from ultrathink import UltrathinkEngine, detect_thinking_level

    # Detect power words in prompt
    level = detect_thinking_level("Analyze the architecture ultrathink")
    # Returns: ThinkingLevel.ULTRATHINK

    # Create engine with automatic budget
    engine = UltrathinkEngine()
    result = await engine.think(
        prompt="Design a distributed cache system",
        level=ThinkingLevel.ULTRATHINK,
    )
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .thinking import (
    ThinkingBudget,
    ConfidenceLevel,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Thinking Levels (Power Words)
# =============================================================================

class ThinkingLevel(str, Enum):
    """
    Claude Code thinking levels with associated token budgets.

    - QUICK: Fast responses, minimal reasoning (~1K tokens)
    - THINK: Standard extended thinking (~8K tokens)
    - HARDTHINK: Deep analysis for complex problems (~32K tokens)
    - ULTRATHINK: Maximum reasoning depth (~128K tokens)
    """
    QUICK = "quick"
    THINK = "think"
    HARDTHINK = "hardthink"
    ULTRATHINK = "ultrathink"


# Token budgets for each thinking level
THINKING_BUDGETS: Dict[ThinkingLevel, int] = {
    ThinkingLevel.QUICK: 1000,
    ThinkingLevel.THINK: 8000,
    ThinkingLevel.HARDTHINK: 32000,
    ThinkingLevel.ULTRATHINK: 128000,
}

# Power word patterns for detection
POWER_WORD_PATTERNS: Dict[ThinkingLevel, List[str]] = {
    ThinkingLevel.ULTRATHINK: [
        r"\bultrathink\b",
        r"\bultra-think\b",
        r"\bmaximum\s+thinking\b",
        r"\bdeep\s+analysis\b",
    ],
    ThinkingLevel.HARDTHINK: [
        r"\bhardthink\b",
        r"\bhard-think\b",
        r"\bthink\s+hard\b",
        r"\bcareful\s+analysis\b",
    ],
    ThinkingLevel.THINK: [
        r"\bthink\b",
        r"\bthink\s+about\b",
        r"\breason\s+through\b",
        r"\banalyze\b",
    ],
}


def detect_thinking_level(prompt: str) -> ThinkingLevel:
    """
    Detect thinking level from power words in prompt.

    Args:
        prompt: User prompt text

    Returns:
        Detected ThinkingLevel (defaults to QUICK if no power words)
    """
    prompt_lower = prompt.lower()

    # Check in order of highest to lowest level
    for level in [ThinkingLevel.ULTRATHINK, ThinkingLevel.HARDTHINK, ThinkingLevel.THINK]:
        for pattern in POWER_WORD_PATTERNS[level]:
            if re.search(pattern, prompt_lower):
                return level

    return ThinkingLevel.QUICK


def get_budget_for_level(level: ThinkingLevel) -> ThinkingBudget:
    """Create ThinkingBudget for a thinking level.

    Budget allocation strategy:
    - min_tokens: At least 500, or 5% of total (for baseline reasoning)
    - max_tokens_per_step: Between 500-8000, quarter of total (per-step cap)
    - reserved_for_output: At least 1000, or 10% (for final synthesis)
    """
    total_tokens = THINKING_BUDGETS[level]
    return ThinkingBudget(
        total_tokens=total_tokens,
        min_tokens=max(1024, total_tokens // 20),
        max_tokens_per_step=max(500, min(8000, total_tokens // 4)),
        reserved_for_output=max(1000, total_tokens // 10),
    )


# =============================================================================
# Chain of Thought Phases
# =============================================================================

class CoTPhase(str, Enum):
    """Phases in structured chain-of-thought reasoning."""
    UNDERSTAND = "understand"      # Parse and understand the problem
    DECOMPOSE = "decompose"        # Break into sub-problems
    EXPLORE = "explore"            # Explore solution approaches
    EVALUATE = "evaluate"          # Evaluate approaches
    SYNTHESIZE = "synthesize"      # Combine insights
    VERIFY = "verify"              # Verify the solution
    CONCLUDE = "conclude"          # Final conclusion


@dataclass
class CoTStep:
    """A step in the chain of thought."""
    phase: CoTPhase
    content: str
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.5
    tokens: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoTChain:
    """A complete chain of thought."""
    id: str
    prompt: str
    level: ThinkingLevel
    steps: List[CoTStep] = field(default_factory=list)
    branches: List["CoTChain"] = field(default_factory=list)  # For Tree of Thoughts
    conclusion: Optional[str] = None
    confidence: float = 0.5
    total_tokens: int = 0
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    @property
    def duration_ms(self) -> float:
        """Get chain duration in milliseconds."""
        end = self.completed_at or time.time()
        return (end - self.started_at) * 1000


# =============================================================================
# Confidence Calibration
# =============================================================================

@dataclass
class EvidenceItem:
    """An evidence item supporting a conclusion."""
    content: str
    source: str
    strength: float  # 0-1, how strong this evidence is
    relevance: float  # 0-1, how relevant to the conclusion
    contradicts: bool = False  # Does this contradict the conclusion?


class ConfidenceCalibrator:
    """
    Calibrates confidence based on evidence and reasoning quality.

    Uses Bayesian-inspired updating with:
    - Prior confidence from base reasoning
    - Evidence strength and relevance
    - Contradictory evidence penalty
    - Reasoning chain coherence
    """

    def __init__(
        self,
        prior_confidence: float = 0.5,
        evidence_weight: float = 0.3,
        coherence_weight: float = 0.2,
    ):
        self.prior_confidence = prior_confidence
        self.evidence_weight = evidence_weight
        self.coherence_weight = coherence_weight

    def calibrate(
        self,
        base_confidence: float,
        evidence: List[EvidenceItem],
        chain_coherence: float = 0.5,
    ) -> Tuple[float, ConfidenceLevel]:
        """
        Calibrate confidence based on evidence.

        Args:
            base_confidence: Initial confidence estimate
            evidence: Supporting/contradicting evidence
            chain_coherence: How coherent the reasoning chain is

        Returns:
            Tuple of (calibrated_confidence, confidence_level)
        """
        if not evidence:
            # No evidence, apply light penalty
            calibrated = base_confidence * 0.9
        else:
            # Calculate evidence contribution
            supporting_strength = sum(
                e.strength * e.relevance
                for e in evidence
                if not e.contradicts
            )
            contradicting_strength = sum(
                e.strength * e.relevance
                for e in evidence
                if e.contradicts
            )

            # Net evidence effect
            evidence_effect = (supporting_strength - contradicting_strength) / (len(evidence) + 1)

            # Combine with base confidence
            calibrated = (
                base_confidence * (1 - self.evidence_weight - self.coherence_weight)
                + evidence_effect * self.evidence_weight
                + chain_coherence * self.coherence_weight
            )

        # Clamp to valid range
        calibrated = max(0.0, min(1.0, calibrated))

        # Convert to level
        level = self._to_level(calibrated)

        return calibrated, level

    def _to_level(self, confidence: float) -> ConfidenceLevel:
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


# =============================================================================
# Tree of Thoughts Explorer
# =============================================================================

class ThoughtBranch(BaseModel):
    """A branch in the Tree of Thoughts."""
    id: str
    parent_id: Optional[str] = None
    thought: str
    evaluation_score: float = 0.0
    is_promising: bool = True
    depth: int = 0
    children: List[str] = Field(default_factory=list)


class TreeOfThoughts:
    """
    Tree of Thoughts implementation for exploration.

    Explores multiple reasoning paths and evaluates them to find
    the most promising direction.
    """

    def __init__(
        self,
        max_branches: int = 3,
        max_depth: int = 5,
        pruning_threshold: float = 0.3,
    ):
        self.max_branches = max_branches
        self.max_depth = max_depth
        self.pruning_threshold = pruning_threshold
        self._branches: Dict[str, ThoughtBranch] = {}
        self._root_id: Optional[str] = None

    def create_root(self, thought: str) -> ThoughtBranch:
        """Create the root thought."""
        import uuid
        branch_id = str(uuid.uuid4())[:8]
        branch = ThoughtBranch(
            id=branch_id,
            thought=thought,
            depth=0,
        )
        self._branches[branch_id] = branch
        self._root_id = branch_id
        return branch

    def add_branch(
        self,
        parent_id: str,
        thought: str,
        evaluation_score: float = 0.5,
    ) -> Optional[ThoughtBranch]:
        """
        Add a branch to the tree.

        Returns None if parent not found or max depth reached.
        """
        if parent_id not in self._branches:
            return None

        parent = self._branches[parent_id]
        if parent.depth >= self.max_depth:
            return None

        if len(parent.children) >= self.max_branches:
            return None

        import uuid
        branch_id = str(uuid.uuid4())[:8]
        branch = ThoughtBranch(
            id=branch_id,
            parent_id=parent_id,
            thought=thought,
            evaluation_score=evaluation_score,
            is_promising=evaluation_score >= self.pruning_threshold,
            depth=parent.depth + 1,
        )

        self._branches[branch_id] = branch
        parent.children.append(branch_id)

        return branch

    def get_best_path(self) -> List[ThoughtBranch]:
        """Get the best path from root to leaf."""
        if not self._root_id:
            return []

        path: List[ThoughtBranch] = []
        current = self._branches.get(self._root_id)

        while current:
            path.append(current)

            if not current.children:
                break

            # Find best child
            best_child = None
            best_score = -1.0
            for child_id in current.children:
                child = self._branches.get(child_id)
                if child and child.is_promising and child.evaluation_score > best_score:
                    best_child = child
                    best_score = child.evaluation_score

            current = best_child

        return path

    def get_promising_branches(self) -> List[ThoughtBranch]:
        """Get all promising branches."""
        return [b for b in self._branches.values() if b.is_promising]

    def prune(self) -> int:
        """
        Prune unpromising branches.

        Returns number of branches pruned.
        """
        pruned = 0
        for branch in list(self._branches.values()):
            if not branch.is_promising and branch.id != self._root_id:
                # Remove from parent's children
                if branch.parent_id and branch.parent_id in self._branches:
                    parent = self._branches[branch.parent_id]
                    if branch.id in parent.children:
                        parent.children.remove(branch.id)

                del self._branches[branch.id]
                pruned += 1

        return pruned

    def get_stats(self) -> Dict[str, Any]:
        """Get tree statistics."""
        depths = [b.depth for b in self._branches.values()]
        return {
            "total_branches": len(self._branches),
            "promising_branches": len(self.get_promising_branches()),
            "max_depth_reached": max(depths) if depths else 0,
            "avg_evaluation_score": (
                sum(b.evaluation_score for b in self._branches.values()) / len(self._branches)
                if self._branches else 0.0
            ),
        }


# =============================================================================
# Self-Consistency Checker
# =============================================================================

@dataclass
class ReasoningPath:
    """A single reasoning path for self-consistency."""
    id: str
    steps: List[str]
    conclusion: str
    confidence: float
    tokens_used: int


class SelfConsistencyChecker:
    """
    Self-consistency checking via multi-path voting.

    Generates multiple reasoning paths and votes on the final answer.
    """

    def __init__(
        self,
        num_paths: int = 3,
        agreement_threshold: float = 0.6,
    ):
        self.num_paths = num_paths
        self.agreement_threshold = agreement_threshold
        self._paths: List[ReasoningPath] = []

    def add_path(self, path: ReasoningPath) -> None:
        """Add a reasoning path."""
        self._paths.append(path)

    def check_consistency(self) -> Tuple[bool, float, Optional[str]]:
        """
        Check consistency across paths.

        Returns:
            Tuple of (is_consistent, agreement_ratio, majority_conclusion)
        """
        if not self._paths:
            return False, 0.0, None

        # Group by conclusion (simplified - in practice, use semantic similarity)
        conclusion_groups: Dict[str, List[ReasoningPath]] = {}
        for path in self._paths:
            # Normalize conclusion for grouping
            norm_conclusion = path.conclusion.lower().strip()
            if norm_conclusion not in conclusion_groups:
                conclusion_groups[norm_conclusion] = []
            conclusion_groups[norm_conclusion].append(path)

        # Find majority
        majority_conclusion = max(conclusion_groups.keys(), key=lambda k: len(conclusion_groups[k]))
        majority_count = len(conclusion_groups[majority_conclusion])
        agreement_ratio = majority_count / len(self._paths)

        is_consistent = agreement_ratio >= self.agreement_threshold

        return is_consistent, agreement_ratio, majority_conclusion

    def get_weighted_conclusion(self) -> Tuple[str, float]:
        """
        Get confidence-weighted conclusion.

        Returns:
            Tuple of (best_conclusion, weighted_confidence)
        """
        if not self._paths:
            return "", 0.0

        # Weight by confidence
        conclusion_scores: Dict[str, float] = {}
        for path in self._paths:
            norm_conclusion = path.conclusion.lower().strip()
            if norm_conclusion not in conclusion_scores:
                conclusion_scores[norm_conclusion] = 0.0
            conclusion_scores[norm_conclusion] += path.confidence

        best_conclusion = max(conclusion_scores.keys(), key=lambda k: conclusion_scores[k])
        weighted_confidence = conclusion_scores[best_conclusion] / sum(conclusion_scores.values())

        return best_conclusion, weighted_confidence


# =============================================================================
# Ultrathink Engine
# =============================================================================

class UltrathinkEngine:
    """
    Advanced thinking engine with ultrathink support.

    Combines:
    - Automatic thinking level detection
    - Structured chain-of-thought
    - Tree of Thoughts exploration
    - Self-consistency checking
    - Evidence-based confidence calibration
    """

    def __init__(
        self,
        default_level: ThinkingLevel = ThinkingLevel.THINK,
        enable_tot: bool = True,
        enable_self_consistency: bool = True,
        tot_branches: int = 3,
        consistency_paths: int = 3,
    ):
        self.default_level = default_level
        self.enable_tot = enable_tot
        self.enable_self_consistency = enable_self_consistency
        self.tot_branches = tot_branches
        self.consistency_paths = consistency_paths

        self._calibrator = ConfidenceCalibrator()
        self._chains: Dict[str, CoTChain] = {}
        self._total_tokens_used = 0
        self._total_chains_completed = 0

    def begin_chain(
        self,
        prompt: str,
        level: Optional[ThinkingLevel] = None,
    ) -> CoTChain:
        """
        Begin a new chain of thought.

        Args:
            prompt: The prompt to think about
            level: Thinking level (auto-detected if not provided)

        Returns:
            New CoTChain instance
        """
        import uuid

        # Detect level if not provided
        if level is None:
            level = detect_thinking_level(prompt)

        chain_id = str(uuid.uuid4())[:12]
        chain = CoTChain(
            id=chain_id,
            prompt=prompt,
            level=level,
        )

        self._chains[chain_id] = chain
        logger.info(f"Started ultrathink chain {chain_id} at level {level.value}")

        return chain

    def add_step(
        self,
        chain_id: str,
        phase: CoTPhase,
        content: str,
        evidence: Optional[List[str]] = None,
        confidence: float = 0.5,
    ) -> bool:
        """Add a step to the chain."""
        if chain_id not in self._chains:
            return False

        chain = self._chains[chain_id]
        budget = get_budget_for_level(chain.level)

        # Estimate tokens
        tokens = len(content) // 4
        if chain.total_tokens + tokens > budget.available_for_thinking:
            logger.warning(f"Budget exceeded for chain {chain_id}")
            return False

        step = CoTStep(
            phase=phase,
            content=content,
            evidence=evidence or [],
            confidence=confidence,
            tokens=tokens,
        )

        chain.steps.append(step)
        chain.total_tokens += tokens
        self._total_tokens_used += tokens

        return True

    def conclude_chain(
        self,
        chain_id: str,
        conclusion: str,
        evidence: Optional[List[EvidenceItem]] = None,
    ) -> Optional[CoTChain]:
        """
        Conclude a chain with confidence calibration.

        Args:
            chain_id: Chain to conclude
            conclusion: Final conclusion
            evidence: Evidence items for calibration

        Returns:
            Completed chain or None if not found
        """
        if chain_id not in self._chains:
            return None

        chain = self._chains[chain_id]
        chain.conclusion = conclusion
        chain.completed_at = time.time()

        # Calculate base confidence from steps
        if chain.steps:
            base_confidence = sum(s.confidence for s in chain.steps) / len(chain.steps)
        else:
            base_confidence = 0.5

        # Calculate chain coherence (simplified)
        coherence = self._calculate_coherence(chain)

        # Calibrate confidence
        calibrated, level = self._calibrator.calibrate(
            base_confidence,
            evidence or [],
            coherence,
        )
        chain.confidence = calibrated

        self._total_chains_completed += 1
        logger.info(
            f"Concluded chain {chain_id} with confidence {calibrated:.2f} ({level.value})"
        )

        return chain

    def _calculate_coherence(self, chain: CoTChain) -> float:
        """Calculate reasoning chain coherence."""
        if len(chain.steps) < 2:
            return 0.5

        # Check phase progression (should follow logical order)
        expected_order = [
            CoTPhase.UNDERSTAND,
            CoTPhase.DECOMPOSE,
            CoTPhase.EXPLORE,
            CoTPhase.EVALUATE,
            CoTPhase.SYNTHESIZE,
            CoTPhase.VERIFY,
            CoTPhase.CONCLUDE,
        ]

        actual_phases = [s.phase for s in chain.steps]

        # Calculate order score
        order_score = 0.0
        for i, phase in enumerate(actual_phases):
            if phase in expected_order:
                expected_idx = expected_order.index(phase)
                # Reward if phase appears in roughly expected position
                position_ratio = i / len(actual_phases)
                expected_ratio = expected_idx / len(expected_order)
                order_score += 1 - abs(position_ratio - expected_ratio)

        return order_score / len(actual_phases) if actual_phases else 0.5

    def explore_with_tot(
        self,
        chain_id: str,
        evaluator: Optional[Callable[[str], float]] = None,
    ) -> TreeOfThoughts:
        """
        Explore alternatives using Tree of Thoughts.

        Args:
            chain_id: Chain to explore from
            evaluator: Function to evaluate thoughts (default: simple heuristic)

        Returns:
            TreeOfThoughts instance with explored branches
        """
        if chain_id not in self._chains:
            return TreeOfThoughts()

        chain = self._chains[chain_id]
        tot = TreeOfThoughts(max_branches=self.tot_branches)

        # Create root from prompt
        tot.create_root(chain.prompt)

        # Store evaluator for use when generating branches
        # In practice, this would generate alternatives using LLM
        self._evaluator = evaluator or (lambda _: 0.5)

        return tot

    def check_self_consistency(
        self,
        paths: List[ReasoningPath],
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Check self-consistency across multiple reasoning paths.

        Args:
            paths: List of reasoning paths to check

        Returns:
            Tuple of (is_consistent, agreement_ratio, majority_conclusion)
        """
        checker = SelfConsistencyChecker(
            num_paths=self.consistency_paths,
        )
        for path in paths:
            checker.add_path(path)

        return checker.check_consistency()

    def think(
        self,
        prompt: str,
        level: Optional[ThinkingLevel] = None,
    ) -> CoTChain:
        """
        Complete thinking process with automatic phases.

        This is a convenience method that executes a full chain of thought.

        Args:
            prompt: The prompt to think about
            level: Thinking level (auto-detected if not provided)

        Returns:
            Completed CoTChain
        """
        chain = self.begin_chain(prompt, level)

        # Execute standard phases
        phases = [
            (CoTPhase.UNDERSTAND, f"Understanding the problem: {prompt[:100]}...", 0.9),
            (CoTPhase.DECOMPOSE, "Breaking down into components...", 0.8),
            (CoTPhase.EXPLORE, "Exploring potential approaches...", 0.7),
            (CoTPhase.EVALUATE, "Evaluating approaches against criteria...", 0.75),
            (CoTPhase.SYNTHESIZE, "Synthesizing insights...", 0.8),
            (CoTPhase.VERIFY, "Verifying the solution...", 0.85),
        ]

        for phase, content, confidence in phases:
            self.add_step(chain.id, phase, content, confidence=confidence)

        return self.conclude_chain(
            chain.id,
            "Completed thinking process with synthesized solution.",
        ) or chain

    def get_chain(self, chain_id: str) -> Optional[CoTChain]:
        """Get a chain by ID."""
        return self._chains.get(chain_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_chains": len(self._chains),
            "completed_chains": self._total_chains_completed,
            "total_tokens_used": self._total_tokens_used,
            "enable_tot": self.enable_tot,
            "enable_self_consistency": self.enable_self_consistency,
            "chains_by_level": {
                level.value: sum(1 for c in self._chains.values() if c.level == level)
                for level in ThinkingLevel
            },
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_ultrathink_engine(
    default_level: ThinkingLevel = ThinkingLevel.THINK,
    enable_tot: bool = True,
    enable_self_consistency: bool = True,
) -> UltrathinkEngine:
    """
    Create an UltrathinkEngine.

    Args:
        default_level: Default thinking level
        enable_tot: Enable Tree of Thoughts
        enable_self_consistency: Enable self-consistency checking

    Returns:
        Configured UltrathinkEngine
    """
    return UltrathinkEngine(
        default_level=default_level,
        enable_tot=enable_tot,
        enable_self_consistency=enable_self_consistency,
    )


def create_confidence_calibrator(
    prior: float = 0.5,
    evidence_weight: float = 0.3,
) -> ConfidenceCalibrator:
    """Create a ConfidenceCalibrator."""
    return ConfidenceCalibrator(
        prior_confidence=prior,
        evidence_weight=evidence_weight,
    )


def create_tree_of_thoughts(
    max_branches: int = 3,
    max_depth: int = 5,
    pruning_threshold: float = 0.3,
) -> TreeOfThoughts:
    """Create a TreeOfThoughts explorer."""
    return TreeOfThoughts(
        max_branches=max_branches,
        max_depth=max_depth,
        pruning_threshold=pruning_threshold,
    )


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate the Ultrathink module."""
    print("=" * 60)
    print("UAP Ultrathink Module Demo")
    print("=" * 60)

    # Test power word detection
    print("\n[Power Word Detection]")
    test_prompts = [
        "What is 2+2?",
        "Analyze the architecture think",
        "Design a distributed system hardthink",
        "Plan the migration strategy ultrathink",
    ]
    for prompt in test_prompts:
        level = detect_thinking_level(prompt)
        budget = THINKING_BUDGETS[level]
        print(f"  '{prompt[:40]}...' -> {level.value} ({budget:,} tokens)")

    # Create engine and think
    print("\n[Ultrathink Engine]")
    engine = create_ultrathink_engine()
    chain = engine.think(
        "Design a scalable microservices architecture for a trading platform ultrathink",
    )

    print(f"  Chain ID: {chain.id}")
    print(f"  Level: {chain.level.value}")
    print(f"  Steps: {len(chain.steps)}")
    print(f"  Total tokens: {chain.total_tokens}")
    print(f"  Confidence: {chain.confidence:.2f}")
    print(f"  Duration: {chain.duration_ms:.1f}ms")

    # Test Tree of Thoughts
    print("\n[Tree of Thoughts]")
    tot = create_tree_of_thoughts()
    root = tot.create_root("How should we handle authentication?")
    tot.add_branch(root.id, "Use JWT tokens", 0.8)
    tot.add_branch(root.id, "Use session cookies", 0.6)
    tot.add_branch(root.id, "Use OAuth2", 0.9)

    stats = tot.get_stats()
    print(f"  Branches: {stats['total_branches']}")
    print(f"  Promising: {stats['promising_branches']}")
    print(f"  Best path: {[b.thought[:30] for b in tot.get_best_path()]}")

    # Test confidence calibration
    print("\n[Confidence Calibration]")
    calibrator = create_confidence_calibrator()
    evidence = [
        EvidenceItem(
            content="Performance benchmarks show 10x improvement",
            source="internal testing",
            strength=0.9,
            relevance=0.8,
        ),
        EvidenceItem(
            content="Similar system failed at scale",
            source="case study",
            strength=0.7,
            relevance=0.6,
            contradicts=True,
        ),
    ]
    calibrated, level = calibrator.calibrate(0.7, evidence, 0.8)
    print(f"  Base confidence: 0.7")
    print(f"  Calibrated: {calibrated:.2f} ({level.value})")

    # Engine stats
    print("\n[Engine Stats]")
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    demo()
