# Voyage AI V39.13 Implementation Plan
## Classification Enhancement & Metrics Infrastructure

**Status**: ✅ COMPLETE (2026-01-26)
**Target**: Enhanced classification with metrics, confidence filtering, and transition detection
**Dependencies**: V39.12 Phase 2 complete (StreamingClassifier, 25/25 tests)
**Priority**: HIGH (Ralph Loop directive: optimize caching, validate integration)
**Result**: 39/39 tests passing (V39.10: 8, V39.12 P1: 9, V39.12 P2: 8, V39.13: 14)

---

## Gap Analysis (Building on V39.12)

| Current State (V39.12) | V39.13 Enhancement |
|------------------------|-------------------|
| Fixed smoothing factor (0.85) | Adaptive smoothing based on stability |
| No confidence filtering | Configurable confidence thresholds |
| No transition detection | Emit events on category changes |
| No classification metrics | Track accuracy, latency, throughput |
| Basic result format | Rich metrics in result output |

---

## V39.13 Features

### 1. ClassificationMetrics Tracking

Comprehensive metrics for monitoring classification performance:

```python
@dataclass
class ClassificationMetrics:
    """Track classification performance over time."""
    total_classifications: int = 0
    high_confidence_count: int = 0  # Above threshold
    low_confidence_count: int = 0   # Below threshold
    transitions_detected: int = 0    # Category changes
    avg_confidence: float = 0.0
    avg_stability: float = 0.0
    avg_latency_ms: float = 0.0
    category_distribution: dict[str, int] = field(default_factory=dict)

    def record(self, result: StreamingClassificationResult, latency_ms: float):
        """Record metrics from a classification result."""
        self.total_classifications += 1
        # Update running averages, distribution, etc.

    def to_dict(self) -> dict:
        """Export metrics for monitoring/logging."""
        return {
            "total": self.total_classifications,
            "high_confidence_rate": self.high_confidence_rate,
            "transition_rate": self.transition_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "distribution": self.category_distribution,
        }
```

### 2. Confidence Threshold Filtering

Filter classifications based on confidence level:

```python
@dataclass
class ClassificationThresholds:
    """Configurable thresholds for classification filtering."""
    min_confidence: float = 0.3      # Minimum to accept classification
    high_confidence: float = 0.7     # Mark as high-confidence
    stability_weight: float = 0.3    # Factor stability into filtering

    def is_valid(self, result: StreamingClassificationResult) -> bool:
        """Check if classification meets minimum threshold."""
        return result.dominant_confidence >= self.min_confidence

    def is_high_confidence(self, result: StreamingClassificationResult) -> bool:
        """Check if classification is high-confidence."""
        weighted = (
            result.dominant_confidence * (1 - self.stability_weight) +
            result.temporal_stability * self.stability_weight
        )
        return weighted >= self.high_confidence
```

### 3. Transition Detection

Detect and emit events when category changes significantly:

```python
@dataclass
class TransitionEvent:
    """Represents a category transition."""
    timestamp: float
    from_category: str
    to_category: str
    from_confidence: float
    to_confidence: float
    stability_at_transition: float
    was_stable: bool  # True if previous category was stable

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "from": self.from_category,
            "to": self.to_category,
            "confidence_change": self.to_confidence - self.from_confidence,
            "was_stable": self.was_stable,
        }

class TransitionDetector:
    """Detect meaningful category transitions."""

    def __init__(
        self,
        stability_threshold: float = 0.6,  # Stability needed to register
        persistence_frames: int = 3,        # Frames before confirming transition
    ):
        self.stability_threshold = stability_threshold
        self.persistence_frames = persistence_frames
        self._pending_transition: str | None = None
        self._pending_count: int = 0
        self._last_stable_category: str | None = None
        self._last_stable_confidence: float = 0.0

    def check_transition(
        self,
        result: StreamingClassificationResult,
    ) -> TransitionEvent | None:
        """Check if a confirmed transition occurred."""
        # Transition logic with persistence filtering
```

### 4. Adaptive Smoothing Factor

Dynamically adjust smoothing based on classification stability:

```python
class AdaptiveSmoother:
    """Dynamically adjust smoothing factor based on stability."""

    def __init__(
        self,
        base_factor: float = 0.85,
        min_factor: float = 0.5,   # More responsive when unstable
        max_factor: float = 0.95,  # More stable when consistent
        adaptation_rate: float = 0.1,
    ):
        self.base_factor = base_factor
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.adaptation_rate = adaptation_rate
        self._current_factor = base_factor

    def adapt(self, stability: float) -> float:
        """Adjust smoothing factor based on current stability."""
        # High stability -> increase smoothing (more stable output)
        # Low stability -> decrease smoothing (more responsive)
        target = self.min_factor + stability * (self.max_factor - self.min_factor)
        self._current_factor += (target - self._current_factor) * self.adaptation_rate
        self._current_factor = max(self.min_factor, min(self.max_factor, self._current_factor))
        return self._current_factor

    @property
    def current_factor(self) -> float:
        return self._current_factor

    def reset(self):
        self._current_factor = self.base_factor
```

### 5. Enhanced StreamingClassifier

Updated StreamingClassifier with V39.13 features:

```python
class EnhancedStreamingClassifier:
    """V39.13 StreamingClassifier with metrics, thresholds, and transitions."""

    def __init__(
        self,
        library: ArchetypeEmbeddingLibrary,
        window_size: int = 5,
        base_smoothing_factor: float = 0.85,
        # V39.13 enhancements
        thresholds: ClassificationThresholds | None = None,
        enable_transition_detection: bool = True,
        enable_adaptive_smoothing: bool = False,
        enable_metrics: bool = True,
    ):
        self.library = library
        self.window_size = window_size

        # V39.13 components
        self.thresholds = thresholds or ClassificationThresholds()
        self.transition_detector = TransitionDetector() if enable_transition_detection else None
        self.adaptive_smoother = AdaptiveSmoother(base_smoothing_factor) if enable_adaptive_smoothing else None
        self.metrics = ClassificationMetrics() if enable_metrics else None

        # Fallback to fixed smoothing if adaptive disabled
        self._fixed_smoothing = base_smoothing_factor

    @property
    def current_smoothing_factor(self) -> float:
        if self.adaptive_smoother:
            return self.adaptive_smoother.current_factor
        return self._fixed_smoothing

    async def classify(
        self,
        embedding: list[float],
        timestamp: float | None = None,
    ) -> EnhancedClassificationResult:
        """Classify with enhanced metrics and filtering."""
        start_time = time.perf_counter()

        # Get base classification from parent
        result = await self._classify_internal(embedding, timestamp)

        # Apply adaptive smoothing if enabled
        if self.adaptive_smoother:
            self.adaptive_smoother.adapt(result.temporal_stability)

        # Check for transitions
        transition = None
        if self.transition_detector:
            transition = self.transition_detector.check_transition(result)

        # Record metrics
        latency_ms = (time.perf_counter() - start_time) * 1000
        if self.metrics:
            self.metrics.record(result, latency_ms)
            if transition:
                self.metrics.transitions_detected += 1

        # Build enhanced result
        return EnhancedClassificationResult(
            base_result=result,
            is_valid=self.thresholds.is_valid(result),
            is_high_confidence=self.thresholds.is_high_confidence(result),
            transition=transition,
            current_smoothing_factor=self.current_smoothing_factor,
            latency_ms=latency_ms,
        )
```

### 6. EnhancedClassificationResult

Rich result format with V39.13 metadata:

```python
@dataclass
class EnhancedClassificationResult:
    """V39.13 enhanced classification result with metrics."""
    base_result: StreamingClassificationResult
    is_valid: bool                              # Meets min confidence
    is_high_confidence: bool                    # Above high confidence threshold
    transition: TransitionEvent | None          # Transition if detected
    current_smoothing_factor: float             # Active smoothing factor
    latency_ms: float                           # Classification latency

    # Delegate properties to base result
    @property
    def dominant_category(self) -> str:
        return self.base_result.dominant_category

    @property
    def dominant_confidence(self) -> float:
        return self.base_result.dominant_confidence

    @property
    def temporal_stability(self) -> float:
        return self.base_result.temporal_stability

    def to_dict(self) -> dict:
        """Protocol-ready serialization with enhanced fields."""
        data = self.base_result.to_dict()
        data.update({
            "valid": self.is_valid,
            "high_confidence": self.is_high_confidence,
            "smoothing_factor": self.current_smoothing_factor,
            "latency_ms": self.latency_ms,
            "transition": self.transition.to_dict() if self.transition else None,
        })
        return data
```

---

## Implementation Order

### ✅ Phase 1: Core Infrastructure (COMPLETE)
1. [x] Add `ClassificationMetrics` dataclass with recording logic
2. [x] Add `ClassificationThresholds` dataclass with validation
3. [x] Add `TransitionEvent` dataclass
4. [x] Add `TransitionDetector` with persistence filtering
5. [x] Add `AdaptiveSmoother` with dynamic factor calculation

### ✅ Phase 2: Enhanced Classifier (COMPLETE)
6. [x] Add `EnhancedClassificationResult` dataclass
7. [x] Extend `StreamingClassifier` with V39.13 options → EnhancedStreamingClassifier
8. [x] Wire metrics recording into classification flow
9. [x] Wire transition detection into classification flow
10. [x] Wire adaptive smoothing into classification flow

### ✅ Phase 3: Testing (COMPLETE)
11. [x] Test `ClassificationMetrics` recording
12. [x] Test `ClassificationThresholds` filtering
13. [x] Test `TransitionDetector` persistence
14. [x] Test `AdaptiveSmoother` adaptation
15. [x] Test `EnhancedClassificationResult` serialization
16. [x] Run full test suite (V39.10 + V39.12 + V39.13) → 39/39 passing

---

## Success Criteria

- [x] Classification metrics tracked with running averages
- [x] Confidence thresholds filter invalid/low-confidence results
- [x] Transitions detected with persistence filtering
- [x] Adaptive smoothing responds to stability changes
- [x] All V39.10 tests still passing (8/8) ✅
- [x] All V39.12 Phase 1 tests still passing (9/9) ✅
- [x] All V39.12 Phase 2 tests still passing (8/8) ✅
- [x] V39.13 new tests passing (14/14) ✅

---

## Integration Patterns

### Pattern 1: Real-Time Monitoring Dashboard
```python
classifier = EnhancedStreamingClassifier(
    library,
    enable_metrics=True,
    enable_transition_detection=True,
)

async for result in classifier.classify_stream(embedding_stream):
    # Send to monitoring
    if result.transition:
        await log_transition(result.transition)
    if result.is_high_confidence:
        await update_primary_display(result)

    # Periodic metrics export
    if classifier.metrics.total_classifications % 100 == 0:
        await export_metrics(classifier.metrics.to_dict())
```

### Pattern 2: TouchDesigner Integration
```python
# Configuration for TD: stable output, clear transitions
classifier = EnhancedStreamingClassifier(
    library,
    thresholds=ClassificationThresholds(
        min_confidence=0.4,     # Filter noise
        high_confidence=0.75,   # Strong signals only
        stability_weight=0.4,   # Factor in stability
    ),
    enable_adaptive_smoothing=True,  # Respond to volatility
)

async for result in classifier.classify_stream(embedding_stream):
    if result.is_valid:
        # Only send valid classifications to TD
        await osc_send("/archetype", result.to_dict())
```

---

Document Version: 2.0
Created: 2026-01-26
Completed: 2026-01-26
Author: Claude (Ralph Loop V39.13)
Status: ✅ COMPLETE - All 3 Phases (39/39 tests passing)
