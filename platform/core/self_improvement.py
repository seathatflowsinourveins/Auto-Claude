"""
Self-Improvement Loop Orchestrator - V1.0

Implements continuous learning patterns:
1. Factory Signals - Friction detection and resolution
2. Reflexion Pattern - Learning from failures
3. Evaluator-Optimizer Loop - Self-critique and refinement
4. SiriuS Framework - Rubric-based quality selection

Based on:
- Factory.ai Signals for automated friction detection
- NeurIPS 2025 Reflexion paper for learning from failures
- SiriuS Framework for self-improving through rubrics
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# Import unified confidence if available
try:
    from platform.core.unified_confidence import (
        PatternConfidence as _PatternConfidence,
        PatternStore as _PatternStore,
        PatternType as _PatternType,
        ConfidenceLevel,
        get_confidence_scorer as _get_confidence_scorer,
    )
    HAS_CONFIDENCE = True
    PatternStore = _PatternStore
    PatternType = _PatternType
    PatternConfidence = _PatternConfidence
    get_confidence_scorer = _get_confidence_scorer
except ImportError:
    HAS_CONFIDENCE = False
    PatternStore = None  # type: ignore
    PatternType = None  # type: ignore
    PatternConfidence = None  # type: ignore
    get_confidence_scorer = None  # type: ignore
    ConfidenceLevel = None  # type: ignore

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================

class FrictionType(str, Enum):
    """Types of friction that trigger self-improvement."""
    REPEATED_ERROR = "repeated_error"          # Same error 2+ times
    API_MISMATCH = "api_mismatch"              # Docs vs reality mismatch
    BUILD_FAILURE = "build_failure"            # Repeated build failures
    TEST_FAILURE = "test_failure"              # Test mismatch
    RESEARCH_CONFLICT = "research_conflict"    # Sources disagree
    USER_CORRECTION = "user_correction"        # User says "that's wrong"
    SLOW_PATH = "slow_path"                    # Taking too long
    CONFUSION = "confusion"                    # Unclear requirements


class ImprovementAction(str, Enum):
    """Actions taken for self-improvement."""
    DOCUMENT_CORRECTION = "document_correction"  # Add to CLAUDE.md
    STORE_PATTERN = "store_pattern"              # Save to pattern store
    CREATE_REFLECTION = "create_reflection"      # Generate reflection
    UPDATE_REFERENCE = "update_reference"        # Update reference file
    EXTRACT_PATTERN = "extract_pattern"          # Extract reusable pattern


class QualityDimension(str, Enum):
    """Dimensions for rubric-based quality scoring."""
    CORRECTNESS = "correctness"
    COMPLETENESS = "completeness"
    EFFICIENCY = "efficiency"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"
    DOCUMENTATION = "documentation"


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class FrictionEvent:
    """A detected friction point."""
    friction_type: FrictionType
    description: str
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    occurrence_count: int = 1
    resolved: bool = False
    resolution: Optional[str] = None

    def hash_key(self) -> str:
        """Generate hash for deduplication."""
        content = f"{self.friction_type}:{self.description}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @classmethod
    def create(
        cls,
        friction_type: FrictionType,
        description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> "FrictionEvent":
        return cls(
            friction_type=friction_type,
            description=description,
            context=context or {}
        )


@dataclass
class Reflection:
    """A reflection on a failure or learning opportunity."""
    task: str
    failure: str
    root_cause: str
    fix: str
    prevention: str
    keywords: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "failure": self.failure,
            "root_cause": self.root_cause,
            "fix": self.fix,
            "prevention": self.prevention,
            "keywords": self.keywords,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class Correction:
    """A documented correction."""
    wrong: str
    correct: str
    source: str
    category: str
    verified_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_markdown(self) -> str:
        return f"""**{self.category}** (verified {self.verified_date.strftime('%Y-%m-%d')}):
- ❌ `{self.wrong}` (incorrect)
- ✅ `{self.correct}` (verified from {self.source})"""


class RubricScore(BaseModel):
    """Score for a quality dimension."""
    dimension: QualityDimension
    score: float = Field(ge=0.0, le=1.0)
    feedback: str = ""


class QualityRubric(BaseModel):
    """Quality rubric for evaluating solutions."""
    task_type: str
    dimensions: List[QualityDimension]
    weights: Dict[QualityDimension, float] = {}
    threshold: float = 0.70  # Minimum score to pass

    def score(self, dimension_scores: List[RubricScore]) -> Tuple[float, bool]:
        """Calculate weighted score and pass/fail."""
        total_weight = sum(self.weights.get(d, 1.0) for d in self.dimensions)
        weighted_sum = 0.0

        for rs in dimension_scores:
            weight = self.weights.get(rs.dimension, 1.0)
            weighted_sum += rs.score * weight

        final_score = weighted_sum / max(total_weight, 1.0)
        return final_score, final_score >= self.threshold


class SessionMetrics(BaseModel):
    """Metrics tracked per session."""
    session_id: str
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    errors_encountered: int = 0
    errors_resolved: int = 0
    patterns_extracted: int = 0
    corrections_documented: int = 0
    memory_queries: int = 0
    memory_hits: int = 0
    first_attempt_successes: int = 0
    total_attempts: int = 0

    @property
    def error_resolution_rate(self) -> float:
        if self.errors_encountered == 0:
            return 1.0
        return self.errors_resolved / self.errors_encountered

    @property
    def memory_hit_rate(self) -> float:
        if self.memory_queries == 0:
            return 0.0
        return self.memory_hits / self.memory_queries

    @property
    def first_attempt_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.first_attempt_successes / self.total_attempts


# =============================================================================
# Friction Detector
# =============================================================================

class FrictionDetector:
    """Detects friction points that trigger self-improvement."""

    def __init__(self):
        self.error_history: Dict[str, List[FrictionEvent]] = defaultdict(list)
        self.friction_events: List[FrictionEvent] = []
        self.threshold_repeated = 2  # Same error N times = friction

    def detect_repeated_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[FrictionEvent]:
        """Detect if an error is repeating."""
        key = f"{error_type}:{hashlib.sha256(error_message.encode()).hexdigest()[:16]}"

        # Add to history
        event = FrictionEvent.create(
            friction_type=FrictionType.REPEATED_ERROR,
            description=error_message,
            context=context
        )

        self.error_history[key].append(event)

        # Check if threshold reached
        if len(self.error_history[key]) >= self.threshold_repeated:
            event.occurrence_count = len(self.error_history[key])
            self.friction_events.append(event)
            logger.warning(f"Friction detected: Repeated error ({event.occurrence_count}x): {error_message[:100]}")
            return event

        return None

    def detect_api_mismatch(
        self,
        expected_signature: str,
        actual_signature: str,
        api_name: str,
        source: str
    ) -> FrictionEvent:
        """Detect API signature mismatch."""
        event = FrictionEvent.create(
            friction_type=FrictionType.API_MISMATCH,
            description=f"API mismatch in {api_name}: expected {expected_signature}, got {actual_signature}",
            context={
                "api_name": api_name,
                "expected": expected_signature,
                "actual": actual_signature,
                "source": source
            }
        )
        self.friction_events.append(event)
        logger.warning(f"Friction detected: API mismatch in {api_name}")
        return event

    def detect_user_correction(
        self,
        wrong_info: str,
        correct_info: str,
        category: str
    ) -> FrictionEvent:
        """Detect user correction."""
        event = FrictionEvent.create(
            friction_type=FrictionType.USER_CORRECTION,
            description=f"User corrected: {wrong_info} → {correct_info}",
            context={
                "wrong": wrong_info,
                "correct": correct_info,
                "category": category
            }
        )
        self.friction_events.append(event)
        logger.info(f"Friction detected: User correction in {category}")
        return event

    def detect_research_conflict(
        self,
        topic: str,
        sources: Dict[str, str],  # source_name -> finding
        discrepancy: str
    ) -> FrictionEvent:
        """Detect research conflict between sources."""
        event = FrictionEvent.create(
            friction_type=FrictionType.RESEARCH_CONFLICT,
            description=f"Research conflict on {topic}: {discrepancy}",
            context={
                "topic": topic,
                "sources": sources,
                "discrepancy": discrepancy
            }
        )
        self.friction_events.append(event)
        logger.warning(f"Friction detected: Research conflict on {topic}")
        return event

    def get_unresolved(self) -> List[FrictionEvent]:
        """Get all unresolved friction events."""
        return [e for e in self.friction_events if not e.resolved]

    def resolve(self, event: FrictionEvent, resolution: str) -> None:
        """Mark a friction event as resolved."""
        event.resolved = True
        event.resolution = resolution
        logger.info(f"Friction resolved: {event.friction_type.value}")


# =============================================================================
# Reflexion Engine
# =============================================================================

class ReflexionEngine:
    """Generates reflections from failures for learning."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".claude" / "learnings" / "reflections.jsonl"
        self.reflections: List[Reflection] = []
        self._load()

    def _load(self) -> None:
        """Load existing reflections."""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self.reflections.append(Reflection(
                            task=data.get("task", ""),
                            failure=data.get("failure", ""),
                            root_cause=data.get("root_cause", ""),
                            fix=data.get("fix", ""),
                            prevention=data.get("prevention", ""),
                            keywords=data.get("keywords", []),
                            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now(timezone.utc).isoformat()))
                        ))
        except Exception as e:
            logger.warning(f"Failed to load reflections: {e}")

    def save(self) -> None:
        """Save reflections to storage."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            for reflection in self.reflections:
                f.write(json.dumps(reflection.to_dict()) + "\n")

    def generate_reflection(
        self,
        task: str,
        failure: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Reflection:
        """Generate a reflection from a failure."""
        # Analyze root cause
        root_cause = self._analyze_root_cause(failure, context or {})

        # Generate fix
        fix = self._generate_fix(failure, root_cause)

        # Generate prevention strategy
        prevention = self._generate_prevention(root_cause)

        # Extract keywords for search
        keywords = self._extract_keywords(task, failure, root_cause)

        reflection = Reflection(
            task=task,
            failure=failure,
            root_cause=root_cause,
            fix=fix,
            prevention=prevention,
            keywords=keywords
        )

        self.reflections.append(reflection)
        self.save()

        return reflection

    def _analyze_root_cause(self, failure: str, context: Dict[str, Any]) -> str:  # noqa: ARG002
        """Analyze root cause of failure."""
        # Pattern matching for common root causes
        patterns = [
            ("API", "Incorrect API signature or method name"),
            ("TypeError", "Type mismatch or missing type handling"),
            ("AttributeError", "Accessing non-existent attribute"),
            ("ImportError", "Missing or incorrect import"),
            ("timeout", "Operation took too long"),
            ("401", "Authentication issue"),
            ("403", "Authorization issue"),
            ("404", "Resource not found"),
            ("mock", "Mock behavior differs from real API"),
        ]

        for pattern, cause in patterns:
            if pattern.lower() in failure.lower():
                return cause

        return "Unidentified root cause - requires manual analysis"

    def _generate_fix(self, failure: str, root_cause: str) -> str:
        """Generate fix suggestion."""
        fixes = {
            "API signature": "Verify signature against official documentation",
            "Type mismatch": "Add proper type handling or conversion",
            "Accessing non-existent": "Check attribute exists before access",
            "Missing import": "Add required import statement",
            "too long": "Add timeout handling or optimize operation",
            "Authentication": "Check API key or credentials",
            "Authorization": "Verify permissions and scopes",
            "not found": "Verify resource exists and path is correct",
            "Mock behavior": "Test against real API endpoint",
        }

        for pattern, fix in fixes.items():
            if pattern.lower() in root_cause.lower():
                return fix

        return "Investigate failure context and apply targeted fix"

    def _generate_prevention(self, root_cause: str) -> str:
        """Generate prevention strategy."""
        return f"Before similar tasks: Check for {root_cause.lower()}. Add verification step."

    def _extract_keywords(self, task: str, failure: str, root_cause: str) -> List[str]:
        """Extract keywords for future search."""
        # Simple keyword extraction (could be enhanced with NLP)
        words = (task + " " + failure + " " + root_cause).lower().split()
        # Filter common words and keep significant ones
        stopwords = {"the", "a", "an", "is", "was", "were", "be", "been", "being",
                    "have", "has", "had", "do", "does", "did", "will", "would", "could",
                    "should", "to", "of", "in", "for", "on", "with", "at", "by", "from"}
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]
        return list(set(keywords))[:10]  # Limit to 10 keywords

    def search(self, query: str, limit: int = 5) -> List[Reflection]:
        """Search reflections by keywords."""
        query_words = set(query.lower().split())
        scored = []

        for reflection in self.reflections:
            reflection_words = set(reflection.keywords)
            # Count keyword overlap
            overlap = len(query_words & reflection_words)
            if overlap > 0:
                scored.append((overlap, reflection))

        # Sort by overlap score
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:limit]]


# =============================================================================
# Evaluator-Optimizer Loop
# =============================================================================

class EvaluatorOptimizer:
    """Self-critique and refinement loop."""

    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.rubrics: Dict[str, QualityRubric] = {}
        self._init_default_rubrics()

    def _init_default_rubrics(self) -> None:
        """Initialize default quality rubrics."""
        # API Integration rubric
        self.rubrics["api_integration"] = QualityRubric(
            task_type="api_integration",
            dimensions=[
                QualityDimension.CORRECTNESS,
                QualityDimension.COMPLETENESS,
                QualityDimension.SECURITY,
            ],
            weights={
                QualityDimension.CORRECTNESS: 2.0,  # Most important
                QualityDimension.COMPLETENESS: 1.5,
                QualityDimension.SECURITY: 1.5,
            },
            threshold=0.75
        )

        # Architecture rubric
        self.rubrics["architecture"] = QualityRubric(
            task_type="architecture",
            dimensions=[
                QualityDimension.CORRECTNESS,
                QualityDimension.MAINTAINABILITY,
                QualityDimension.SECURITY,
                QualityDimension.DOCUMENTATION,
            ],
            weights={
                QualityDimension.CORRECTNESS: 1.5,
                QualityDimension.MAINTAINABILITY: 2.0,
                QualityDimension.SECURITY: 1.5,
                QualityDimension.DOCUMENTATION: 1.0,
            },
            threshold=0.70
        )

        # Code change rubric
        self.rubrics["code_change"] = QualityRubric(
            task_type="code_change",
            dimensions=[
                QualityDimension.CORRECTNESS,
                QualityDimension.EFFICIENCY,
                QualityDimension.MAINTAINABILITY,
            ],
            weights={
                QualityDimension.CORRECTNESS: 2.0,
                QualityDimension.EFFICIENCY: 1.0,
                QualityDimension.MAINTAINABILITY: 1.5,
            },
            threshold=0.70
        )

    def evaluate(
        self,
        solution: Any,
        task_type: str,
        evaluator: Callable[[Any, QualityDimension], RubricScore]
    ) -> Tuple[float, bool, List[RubricScore]]:
        """
        Evaluate a solution against rubric.

        Args:
            solution: The solution to evaluate
            task_type: Type of task for rubric selection
            evaluator: Function that scores a dimension

        Returns:
            Tuple of (score, passed, dimension_scores)
        """
        rubric = self.rubrics.get(task_type, self.rubrics["code_change"])

        dimension_scores = []
        for dimension in rubric.dimensions:
            score = evaluator(solution, dimension)
            dimension_scores.append(score)

        final_score, passed = rubric.score(dimension_scores)
        return final_score, passed, dimension_scores

    def optimize(
        self,
        solution: Any,
        task_type: str,
        evaluator: Callable[[Any, QualityDimension], RubricScore],
        improver: Callable[[Any, List[RubricScore]], Any]
    ) -> Tuple[Any, int, float]:
        """
        Iteratively improve solution until it passes rubric.

        Args:
            solution: Initial solution
            task_type: Type of task
            evaluator: Function to evaluate solution
            improver: Function to improve solution based on feedback

        Returns:
            Tuple of (final_solution, iterations, final_score)
        """
        current_solution = solution

        for iteration in range(self.max_iterations):
            score, passed, dimension_scores = self.evaluate(
                current_solution, task_type, evaluator
            )

            if passed:
                logger.info(f"Solution passed rubric at iteration {iteration + 1} with score {score:.2f}")
                return current_solution, iteration + 1, score

            # Get feedback and improve
            current_solution = improver(current_solution, dimension_scores)

        # Return best effort after max iterations
        final_score, _, _ = self.evaluate(current_solution, task_type, evaluator)
        logger.warning(f"Max iterations reached. Final score: {final_score:.2f}")
        return current_solution, self.max_iterations, final_score


# =============================================================================
# Self-Improvement Orchestrator
# =============================================================================

class SelfImprovementOrchestrator:
    """
    Main orchestrator for continuous self-improvement.

    Integrates:
    - Friction detection
    - Reflexion for learning from failures
    - Evaluator-optimizer loops
    - Pattern storage via unified confidence
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        storage_dir: Optional[Path] = None
    ):
        self.session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.storage_dir = storage_dir or Path.home() / ".claude" / "learnings"

        # Initialize components
        self.friction_detector = FrictionDetector()
        self.reflexion_engine = ReflexionEngine(self.storage_dir / "reflections.jsonl")
        self.evaluator_optimizer = EvaluatorOptimizer()

        # Pattern store (from unified confidence if available)
        if HAS_CONFIDENCE and PatternStore is not None and get_confidence_scorer is not None:
            self.pattern_store = PatternStore(self.storage_dir / "patterns.jsonl")
            self.confidence_scorer = get_confidence_scorer()
        else:
            self.pattern_store = None
            self.confidence_scorer = None

        # Metrics
        self.metrics = SessionMetrics(session_id=self.session_id)

        # Corrections storage
        self.corrections: List[Correction] = []

    # -------------------------------------------------------------------------
    # Error Handling & Learning
    # -------------------------------------------------------------------------

    def on_error(
        self,
        error_type: str,
        error_message: str,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[FrictionEvent]:
        """
        Handle an error and potentially trigger learning.

        Returns FrictionEvent if this is a repeated/significant error.
        """
        self.metrics.errors_encountered += 1

        # Detect friction
        friction = self.friction_detector.detect_repeated_error(
            error_type, error_message, context
        )

        if friction:
            # Generate reflection
            _reflection = self.reflexion_engine.generate_reflection(
                task=task,
                failure=error_message,
                context=context or {}
            )

            # Store pattern if confidence system available
            if self.pattern_store and HAS_CONFIDENCE and PatternType is not None:
                _pattern = self.pattern_store.create(
                    pattern_id=f"error-{friction.hash_key()}",
                    pattern_type=PatternType.ERROR_RESOLUTION,
                    initial_confidence=0.60
                )
                logger.info(f"Created error pattern: {_pattern.pattern_id}")

            return friction

        return None

    def on_error_resolved(self, friction: FrictionEvent, resolution: str) -> None:
        """Mark an error as resolved and update patterns."""
        self.friction_detector.resolve(friction, resolution)
        self.metrics.errors_resolved += 1

        # Update pattern confidence
        if self.pattern_store:
            pattern_id = f"error-{friction.hash_key()}"
            new_conf = self.pattern_store.update(pattern_id, success=True)
            if new_conf:
                logger.info(f"Pattern {pattern_id} confidence updated to {new_conf:.2f}")

    # -------------------------------------------------------------------------
    # User Corrections
    # -------------------------------------------------------------------------

    def on_user_correction(
        self,
        wrong: str,
        correct: str,
        category: str,
        source: str = "user"
    ) -> Correction:
        """
        Record a user correction for learning.

        User corrections get HIGH initial confidence (0.80).
        """
        # Detect friction
        friction = self.friction_detector.detect_user_correction(wrong, correct, category)

        # Create correction record
        correction = Correction(
            wrong=wrong,
            correct=correct,
            source=source,
            category=category
        )
        self.corrections.append(correction)
        self.metrics.corrections_documented += 1

        # Store high-confidence pattern
        if self.pattern_store and HAS_CONFIDENCE and PatternType is not None:
            _pattern = self.pattern_store.create(
                pattern_id=f"correction-{friction.hash_key()}",
                pattern_type=PatternType.USER_CORRECTION,
                initial_confidence=0.80  # User corrections are highly trusted
            )

        return correction

    # -------------------------------------------------------------------------
    # Pattern Extraction
    # -------------------------------------------------------------------------

    def extract_pattern(
        self,
        pattern_id: str,
        pattern_type: str,
        description: str,  # noqa: ARG002 - kept for API compatibility
        initial_confidence: float = 0.50
    ) -> Optional[Any]:
        """Extract and store a reusable pattern."""
        if not self.pattern_store or not HAS_CONFIDENCE or PatternType is None:
            logger.warning("Pattern store not available")
            return None

        # Map string to PatternType
        type_map: Dict[str, Any] = {
            "error_resolution": PatternType.ERROR_RESOLUTION,
            "user_correction": PatternType.USER_CORRECTION,
            "verification_insight": PatternType.VERIFICATION_INSIGHT,
            "tool_usage": PatternType.TOOL_USAGE,
            "architecture_decision": PatternType.ARCHITECTURE_DECISION,
            "api_signature": PatternType.API_SIGNATURE,
            "research_finding": PatternType.RESEARCH_FINDING,
        }

        pt = type_map.get(pattern_type.lower(), PatternType.TOOL_USAGE)

        pattern = self.pattern_store.create(
            pattern_id=pattern_id,
            pattern_type=pt,
            initial_confidence=initial_confidence
        )

        self.metrics.patterns_extracted += 1
        logger.info(f"Extracted pattern: {pattern_id} ({pt.value})")

        return pattern

    def update_pattern(self, pattern_id: str, success: bool) -> Optional[float]:
        """Update pattern confidence based on outcome."""
        if not self.pattern_store:
            return None

        return self.pattern_store.update(pattern_id, success)

    # -------------------------------------------------------------------------
    # Memory Query Tracking
    # -------------------------------------------------------------------------

    def on_memory_query(self, found_relevant: bool) -> None:
        """Track memory query success."""
        self.metrics.memory_queries += 1
        if found_relevant:
            self.metrics.memory_hits += 1

    def on_task_attempt(self, first_attempt_success: bool) -> None:
        """Track task attempt success."""
        self.metrics.total_attempts += 1
        if first_attempt_success:
            self.metrics.first_attempt_successes += 1

    # -------------------------------------------------------------------------
    # Reflexion Search
    # -------------------------------------------------------------------------

    def search_past_failures(self, query: str, limit: int = 5) -> List[Reflection]:
        """Search past failures for relevant lessons."""
        results = self.reflexion_engine.search(query, limit)
        self.on_memory_query(len(results) > 0)
        return results

    # -------------------------------------------------------------------------
    # Session Summary
    # -------------------------------------------------------------------------

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of self-improvement activity this session."""
        return {
            "session_id": self.session_id,
            "metrics": {
                "errors_encountered": self.metrics.errors_encountered,
                "errors_resolved": self.metrics.errors_resolved,
                "error_resolution_rate": f"{self.metrics.error_resolution_rate:.1%}",
                "patterns_extracted": self.metrics.patterns_extracted,
                "corrections_documented": self.metrics.corrections_documented,
                "memory_hit_rate": f"{self.metrics.memory_hit_rate:.1%}",
                "first_attempt_rate": f"{self.metrics.first_attempt_rate:.1%}",
            },
            "unresolved_friction": len(self.friction_detector.get_unresolved()),
            "reflections_generated": len(self.reflexion_engine.reflections),
            "corrections": [c.to_markdown() for c in self.corrections[-5:]],  # Last 5
        }

    def export_corrections(self, format: str = "markdown") -> str:
        """Export corrections in specified format."""
        if format == "markdown":
            lines = ["## Corrections Documented This Session\n"]
            for correction in self.corrections:
                lines.append(correction.to_markdown())
                lines.append("")
            return "\n".join(lines)
        else:
            return json.dumps([{
                "wrong": c.wrong,
                "correct": c.correct,
                "source": c.source,
                "category": c.category,
                "verified_date": c.verified_date.isoformat()
            } for c in self.corrections], indent=2)


# =============================================================================
# Convenience Functions
# =============================================================================

# Module-level singleton
_orchestrator_instance: Optional[SelfImprovementOrchestrator] = None


def get_improvement_orchestrator() -> SelfImprovementOrchestrator:
    """Get singleton orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = SelfImprovementOrchestrator()
    return _orchestrator_instance


def on_error(
    error_type: str,
    error_message: str,
    task: str,
    context: Optional[Dict[str, Any]] = None
) -> Optional[FrictionEvent]:
    """Convenience function for error handling."""
    return get_improvement_orchestrator().on_error(
        error_type, error_message, task, context
    )


def on_correction(wrong: str, correct: str, category: str) -> Correction:
    """Convenience function for user corrections."""
    return get_improvement_orchestrator().on_user_correction(
        wrong, correct, category
    )


def search_lessons(query: str) -> List[Reflection]:
    """Convenience function for searching past lessons."""
    return get_improvement_orchestrator().search_past_failures(query)
