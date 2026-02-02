"""
Unified Confidence Scoring Framework - V1.0

Integrates three confidence dimensions:
1. Research Confidence - Multi-source verification
2. Memory Confidence - Letta retrieval relevance
3. Pattern Confidence - Learned pattern reliability

Based on:
- Anthropic's extended thinking research
- Bayesian confidence adjustment from continuous-learning protocol
- Factory.ai Signals self-improvement patterns
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Core Enums (aligned with thinking.py)
# =============================================================================

class ConfidenceLevel(str, Enum):
    """Calibrated confidence levels matching thinking.py."""
    VERY_LOW = "very_low"      # <20% - Highly uncertain
    LOW = "low"                # 20-40% - Significant doubt
    MEDIUM = "medium"          # 40-60% - Uncertain
    HIGH = "high"              # 60-80% - Fairly confident
    VERY_HIGH = "very_high"    # >80% - Highly confident

    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        """Convert numeric score (0-1) to ConfidenceLevel."""
        if score < 0.20:
            return cls.VERY_LOW
        elif score < 0.40:
            return cls.LOW
        elif score < 0.60:
            return cls.MEDIUM
        elif score < 0.80:
            return cls.HIGH
        else:
            return cls.VERY_HIGH

    def to_score_range(self) -> Tuple[float, float]:
        """Return the score range for this level."""
        ranges = {
            self.VERY_LOW: (0.0, 0.20),
            self.LOW: (0.20, 0.40),
            self.MEDIUM: (0.40, 0.60),
            self.HIGH: (0.60, 0.80),
            self.VERY_HIGH: (0.80, 1.0),
        }
        return ranges[self]


class ConfidenceSource(str, Enum):
    """Sources that contribute to confidence scoring."""
    RESEARCH = "research"          # Multi-source research verification
    MEMORY = "memory"              # Letta memory retrieval
    PATTERN = "pattern"            # Learned pattern history
    VERIFICATION = "verification"  # Real API/test verification
    USER = "user"                  # User feedback/correction


class PatternType(str, Enum):
    """Types of learned patterns."""
    ERROR_RESOLUTION = "error_resolution"
    USER_CORRECTION = "user_correction"
    VERIFICATION_INSIGHT = "verification_insight"
    TOOL_USAGE = "tool_usage"
    ARCHITECTURE_DECISION = "architecture_decision"
    API_SIGNATURE = "api_signature"
    RESEARCH_FINDING = "research_finding"
    # Backward compatibility with legacy patterns
    SUCCESS = "success"
    FAILURE = "failure"


# =============================================================================
# Data Models
# =============================================================================

class ConfidenceScore(BaseModel):
    """A confidence score with provenance."""
    score: float = Field(ge=0.0, le=1.0)
    level: ConfidenceLevel
    source: ConfidenceSource
    evidence: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResearchConfidence(BaseModel):
    """Confidence from multi-source research."""
    sources_queried: int = 0
    sources_agreeing: int = 0
    discrepancies_found: int = 0
    discrepancies_resolved: int = 0
    official_docs_verified: bool = False
    real_api_tested: bool = False

    @property
    def score(self) -> float:
        """Calculate research confidence score."""
        if self.sources_queried == 0:
            return 0.0

        base_score = self.sources_agreeing / max(self.sources_queried, 1)

        # Bonus for official docs verification
        if self.official_docs_verified:
            base_score = min(1.0, base_score + 0.15)

        # Bonus for real API testing
        if self.real_api_tested:
            base_score = min(1.0, base_score + 0.20)

        # Penalty for unresolved discrepancies
        unresolved = self.discrepancies_found - self.discrepancies_resolved
        if unresolved > 0:
            base_score = max(0.0, base_score - (unresolved * 0.10))

        return base_score


class MemoryConfidence(BaseModel):
    """Confidence from Letta memory retrieval."""
    passages_retrieved: int = 0
    relevance_scores: List[float] = Field(default_factory=list)
    recency_days: Optional[float] = None
    source_verified: bool = False

    @property
    def score(self) -> float:
        """Calculate memory confidence score."""
        if not self.relevance_scores:
            return 0.0

        # Average relevance with top-weighted
        sorted_scores = sorted(self.relevance_scores, reverse=True)
        if len(sorted_scores) >= 3:
            # Weight top scores more heavily
            weighted = (sorted_scores[0] * 0.5 +
                       sorted_scores[1] * 0.3 +
                       sum(sorted_scores[2:]) / len(sorted_scores[2:]) * 0.2)
        else:
            weighted = sum(sorted_scores) / len(sorted_scores)

        # Recency decay (halve confidence after 30 days)
        if self.recency_days is not None and self.recency_days > 0:
            decay = math.exp(-self.recency_days / 30)
            weighted *= (0.5 + 0.5 * decay)  # Min 50% of original

        # Bonus for source verification
        if self.source_verified:
            weighted = min(1.0, weighted + 0.10)

        return weighted


class PatternConfidence(BaseModel):
    """Confidence from learned pattern history."""
    pattern_id: str
    pattern_type: PatternType
    initial_confidence: float = 0.5
    current_confidence: float = 0.5
    times_applied: int = 0
    successes: int = 0
    failures: int = 0
    last_applied: Optional[datetime] = None

    # Bayesian adjustment parameters (from continuous-learning spec)
    SUCCESS_MULTIPLIER: float = 1.20
    FAILURE_MULTIPLIER: float = 0.85
    FLOOR: float = 0.05
    CEILING: float = 0.95
    PRUNE_THRESHOLD: float = 0.30
    PRUNE_MIN_USES: int = 10

    def update(self, success: bool) -> float:
        """Update confidence based on outcome. Returns new confidence."""
        self.times_applied += 1
        self.last_applied = datetime.now(timezone.utc)

        if success:
            self.successes += 1
            self.current_confidence = min(
                self.CEILING,
                self.current_confidence * self.SUCCESS_MULTIPLIER
            )
        else:
            self.failures += 1
            self.current_confidence = max(
                self.FLOOR,
                self.current_confidence * self.FAILURE_MULTIPLIER
            )

        return self.current_confidence

    @property
    def success_rate(self) -> float:
        """Calculate historical success rate."""
        if self.times_applied == 0:
            return 1.0
        return self.successes / self.times_applied

    @property
    def should_prune(self) -> bool:
        """Check if pattern should be archived."""
        return (self.current_confidence < self.PRUNE_THRESHOLD and
                self.times_applied >= self.PRUNE_MIN_USES)

    @property
    def score(self) -> float:
        """Return current confidence score."""
        return self.current_confidence


class UnifiedConfidence(BaseModel):
    """Combined confidence from all sources."""
    research: Optional[ResearchConfidence] = None
    memory: Optional[MemoryConfidence] = None
    pattern: Optional[PatternConfidence] = None
    verification_passed: bool = False
    user_confirmed: bool = False

    # Weights for combining sources
    RESEARCH_WEIGHT: float = 0.35
    MEMORY_WEIGHT: float = 0.25
    PATTERN_WEIGHT: float = 0.25
    VERIFICATION_WEIGHT: float = 0.15

    @property
    def score(self) -> float:
        """Calculate unified confidence score."""
        scores = []
        weights = []

        if self.research:
            scores.append(self.research.score)
            weights.append(self.RESEARCH_WEIGHT)

        if self.memory:
            scores.append(self.memory.score)
            weights.append(self.MEMORY_WEIGHT)

        if self.pattern:
            scores.append(self.pattern.score)
            weights.append(self.PATTERN_WEIGHT)

        if self.verification_passed:
            scores.append(1.0)
            weights.append(self.VERIFICATION_WEIGHT)
        elif any([self.research, self.memory, self.pattern]):
            # Penalty for not verifying
            scores.append(0.3)
            weights.append(self.VERIFICATION_WEIGHT)

        if not scores:
            return 0.0

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        base_score = weighted_sum / total_weight

        # User confirmation bonus
        if self.user_confirmed:
            base_score = min(1.0, base_score + 0.10)

        return base_score

    @property
    def level(self) -> ConfidenceLevel:
        """Get confidence level from score."""
        return ConfidenceLevel.from_score(self.score)

    def to_confidence_score(self) -> ConfidenceScore:
        """Convert to ConfidenceScore with full provenance."""
        evidence = []

        if self.research:
            evidence.append(f"Research: {self.research.sources_agreeing}/{self.research.sources_queried} sources agree")
            if self.research.official_docs_verified:
                evidence.append("Official docs verified")
            if self.research.real_api_tested:
                evidence.append("Real API tested")

        if self.memory:
            evidence.append(f"Memory: {self.memory.passages_retrieved} passages, avg relevance {sum(self.memory.relevance_scores)/max(1,len(self.memory.relevance_scores)):.2f}")

        if self.pattern:
            evidence.append(f"Pattern: {self.pattern.success_rate:.0%} success over {self.pattern.times_applied} uses")

        if self.verification_passed:
            evidence.append("Verification: PASSED")

        if self.user_confirmed:
            evidence.append("User: CONFIRMED")

        return ConfidenceScore(
            score=self.score,
            level=self.level,
            source=ConfidenceSource.RESEARCH if self.research else ConfidenceSource.PATTERN,
            evidence=evidence
        )


# =============================================================================
# Pattern Storage
# =============================================================================

class PatternStore:
    """Persistent storage for learned patterns."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".claude" / "learnings" / "patterns.jsonl"
        self.patterns: Dict[str, PatternConfidence] = {}
        self._load()

    def _load(self) -> None:
        """Load patterns from storage."""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        pattern = PatternConfidence(
                            pattern_id=data.get("id", ""),
                            pattern_type=PatternType(data.get("pattern_type", "tool_usage")),
                            initial_confidence=data.get("confidence", 0.5),
                            current_confidence=data.get("confidence", 0.5),
                            times_applied=data.get("times_applied", 0),
                            successes=int(data.get("success_rate", 1.0) * data.get("times_applied", 0)),
                            failures=int((1 - data.get("success_rate", 1.0)) * data.get("times_applied", 0)),
                        )
                        self.patterns[pattern.pattern_id] = pattern
        except Exception as e:
            logger.warning(f"Failed to load patterns: {e}")

    def save(self) -> None:
        """Save patterns to storage."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.storage_path, "w") as f:
            for pattern in self.patterns.values():
                data = {
                    "id": pattern.pattern_id,
                    "pattern_type": pattern.pattern_type.value,
                    "confidence": pattern.current_confidence,
                    "times_applied": pattern.times_applied,
                    "success_rate": pattern.success_rate,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "last_applied": pattern.last_applied.isoformat() if pattern.last_applied else None,
                }
                f.write(json.dumps(data) + "\n")

    def get(self, pattern_id: str) -> Optional[PatternConfidence]:
        """Get pattern by ID."""
        return self.patterns.get(pattern_id)

    def create(self, pattern_id: str, pattern_type: PatternType,
               initial_confidence: float = 0.5) -> PatternConfidence:
        """Create new pattern."""
        pattern = PatternConfidence(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            initial_confidence=initial_confidence,
            current_confidence=initial_confidence,
        )
        self.patterns[pattern_id] = pattern
        return pattern

    def update(self, pattern_id: str, success: bool) -> Optional[float]:
        """Update pattern confidence. Returns new confidence or None if not found."""
        pattern = self.patterns.get(pattern_id)
        if pattern:
            new_confidence = pattern.update(success)
            self.save()
            return new_confidence
        return None

    def prune_unreliable(self) -> List[str]:
        """Archive patterns that have fallen below threshold."""
        pruned = []
        for pattern_id, pattern in list(self.patterns.items()):
            if pattern.should_prune:
                pruned.append(pattern_id)
                # Move to archive instead of delete
                archive_path = self.storage_path.parent / "archived_patterns.jsonl"
                with open(archive_path, "a") as f:
                    data = {
                        "id": pattern.pattern_id,
                        "pattern_type": pattern.pattern_type.value,
                        "final_confidence": pattern.current_confidence,
                        "times_applied": pattern.times_applied,
                        "success_rate": pattern.success_rate,
                        "archived_at": datetime.now(timezone.utc).isoformat(),
                        "reason": "confidence_below_threshold"
                    }
                    f.write(json.dumps(data) + "\n")
                del self.patterns[pattern_id]

        if pruned:
            self.save()
        return pruned

    def get_high_confidence(self, min_confidence: float = 0.80) -> List[PatternConfidence]:
        """Get all high-confidence patterns."""
        return [p for p in self.patterns.values() if p.current_confidence >= min_confidence]


# =============================================================================
# Confidence Scorer
# =============================================================================

class UnifiedConfidenceScorer:
    """Main interface for unified confidence scoring."""

    def __init__(self, pattern_store: Optional[PatternStore] = None):
        self.pattern_store = pattern_store or PatternStore()

    def score_research(
        self,
        sources_queried: int,
        sources_agreeing: int,
        discrepancies_found: int = 0,
        discrepancies_resolved: int = 0,
        official_docs_verified: bool = False,
        real_api_tested: bool = False
    ) -> ResearchConfidence:
        """Score research confidence."""
        return ResearchConfidence(
            sources_queried=sources_queried,
            sources_agreeing=sources_agreeing,
            discrepancies_found=discrepancies_found,
            discrepancies_resolved=discrepancies_resolved,
            official_docs_verified=official_docs_verified,
            real_api_tested=real_api_tested
        )

    def score_memory(
        self,
        passages_retrieved: int,
        relevance_scores: List[float],
        recency_days: Optional[float] = None,
        source_verified: bool = False
    ) -> MemoryConfidence:
        """Score memory retrieval confidence."""
        return MemoryConfidence(
            passages_retrieved=passages_retrieved,
            relevance_scores=relevance_scores,
            recency_days=recency_days,
            source_verified=source_verified
        )

    def score_pattern(self, pattern_id: str) -> Optional[PatternConfidence]:
        """Get pattern confidence by ID."""
        return self.pattern_store.get(pattern_id)

    def create_pattern(
        self,
        pattern_id: str,
        pattern_type: PatternType,
        initial_confidence: float = 0.5
    ) -> PatternConfidence:
        """Create and track new pattern."""
        return self.pattern_store.create(pattern_id, pattern_type, initial_confidence)

    def update_pattern(self, pattern_id: str, success: bool) -> Optional[float]:
        """Update pattern based on outcome."""
        return self.pattern_store.update(pattern_id, success)

    def compute_unified(
        self,
        research: Optional[ResearchConfidence] = None,
        memory: Optional[MemoryConfidence] = None,
        pattern_id: Optional[str] = None,
        verification_passed: bool = False,
        user_confirmed: bool = False
    ) -> UnifiedConfidence:
        """Compute unified confidence from all sources."""
        pattern = None
        if pattern_id:
            pattern = self.pattern_store.get(pattern_id)

        return UnifiedConfidence(
            research=research,
            memory=memory,
            pattern=pattern,
            verification_passed=verification_passed,
            user_confirmed=user_confirmed
        )

    def should_proceed(
        self,
        confidence: UnifiedConfidence,
        min_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    ) -> Tuple[bool, str]:
        """Check if confidence is sufficient to proceed."""
        level = confidence.level
        score = confidence.score

        level_order = [
            ConfidenceLevel.VERY_LOW,
            ConfidenceLevel.LOW,
            ConfidenceLevel.MEDIUM,
            ConfidenceLevel.HIGH,
            ConfidenceLevel.VERY_HIGH
        ]

        current_idx = level_order.index(level)
        min_idx = level_order.index(min_level)

        if current_idx >= min_idx:
            return True, f"Confidence {level.value} ({score:.2f}) meets threshold {min_level.value}"
        else:
            return False, f"Confidence {level.value} ({score:.2f}) below threshold {min_level.value}"

    def get_recommendations(self, confidence: UnifiedConfidence) -> List[str]:
        """Get recommendations to improve confidence."""
        recommendations = []

        if not confidence.research:
            recommendations.append("Conduct multi-source research (Context7 + Exa + Tavily)")
        elif confidence.research.score < 0.6:
            if not confidence.research.official_docs_verified:
                recommendations.append("Verify against official documentation")
            if not confidence.research.real_api_tested:
                recommendations.append("Test against real API endpoint")
            if confidence.research.discrepancies_found > confidence.research.discrepancies_resolved:
                recommendations.append("Resolve remaining source discrepancies")

        if not confidence.memory:
            recommendations.append("Query Letta memory for related context")
        elif confidence.memory.score < 0.6:
            recommendations.append("Search for more relevant memory passages")

        if not confidence.verification_passed:
            recommendations.append("Run verification tests against real service")

        return recommendations


# =============================================================================
# Convenience Functions
# =============================================================================

# Module-level singleton instance
_confidence_scorer_instance: Optional[UnifiedConfidenceScorer] = None


def get_confidence_scorer() -> UnifiedConfidenceScorer:
    """Get singleton confidence scorer instance."""
    global _confidence_scorer_instance
    if _confidence_scorer_instance is None:
        _confidence_scorer_instance = UnifiedConfidenceScorer()
    return _confidence_scorer_instance


def quick_confidence_check(
    sources_agreeing: int = 0,
    sources_total: int = 0,
    real_api_tested: bool = False,
    verified: bool = False
) -> Tuple[ConfidenceLevel, float]:
    """Quick confidence check for common cases."""
    scorer = get_confidence_scorer()

    research = None
    if sources_total > 0:
        research = scorer.score_research(
            sources_queried=sources_total,
            sources_agreeing=sources_agreeing,
            official_docs_verified=verified,
            real_api_tested=real_api_tested
        )

    unified = scorer.compute_unified(
        research=research,
        verification_passed=verified
    )

    return unified.level, unified.score
