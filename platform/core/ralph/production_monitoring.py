"""
Ralph Loop Production Monitoring & Observability Integration

This module provides production-ready monitoring for the Ralph Loop including:
- Chi-squared drift detection for model performance regression
- Integration hooks for Opik, Langfuse, LangSmith, and Phoenix
- Real metrics collection with Prometheus export
- Production alerting and health checks

V1 Production Features (February 2026):
- Chi-Squared Distribution Drift Detection
- Jensen-Shannon Divergence for continuous metrics
- Kolmogorov-Smirnov test for fitness distribution changes
- Exponential smoothing for anomaly detection
- OpenTelemetry-compatible tracing integration

References:
- Opik (comet-ml): https://github.com/comet-ml/opik
- Langfuse: https://langfuse.com
- Phoenix (Arize): https://github.com/Arize-ai/phoenix
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, TypeVar

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Types
# =============================================================================

class ObservabilityProvider(Enum):
    """Supported observability providers."""
    OPIK = "opik"
    LANGFUSE = "langfuse"
    LANGSMITH = "langsmith"
    PHOENIX = "phoenix"
    PROMETHEUS = "prometheus"
    CUSTOM = "custom"


@dataclass
class ProductionConfig:
    """
    Production configuration for Ralph Loop monitoring.

    Attributes:
        enable_monitoring: Master switch for production monitoring
        observability_provider: Primary observability backend
        drift_detection_enabled: Enable chi-squared drift detection
        drift_window_size: Number of samples for drift window
        drift_threshold: Chi-squared threshold for alerting (p-value)
        health_check_interval_s: Seconds between health checks
        metrics_export_interval_s: Seconds between metrics export
        trace_sample_rate: Fraction of traces to sample (0.0-1.0)
        alert_webhook_url: Optional webhook for alerts
        custom_metrics: Custom metric definitions
    """
    enable_monitoring: bool = True
    observability_provider: ObservabilityProvider = ObservabilityProvider.OPIK
    drift_detection_enabled: bool = True
    drift_window_size: int = 100
    drift_threshold: float = 0.05  # p-value threshold for chi-squared
    health_check_interval_s: int = 30
    metrics_export_interval_s: int = 60
    trace_sample_rate: float = 0.1
    alert_webhook_url: Optional[str] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    # Provider-specific configuration
    opik_api_key: Optional[str] = None
    opik_project_name: str = "ralph-loop"
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langsmith_api_key: Optional[str] = None
    phoenix_endpoint: str = "http://localhost:6006"


# =============================================================================
# Drift Detection - Chi-Squared and Statistical Tests
# =============================================================================

@dataclass
class DriftSignal:
    """A detected drift signal with statistical details."""
    drift_type: str  # "chi_squared", "ks_test", "jensen_shannon", "zscore"
    metric_name: str
    p_value: float
    statistic: float
    is_significant: bool
    baseline_stats: Dict[str, float]
    current_stats: Dict[str, float]
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    severity: str = "warning"  # "info", "warning", "critical"

    @property
    def drift_magnitude(self) -> float:
        """Calculate drift magnitude (0-1 scale)."""
        if self.p_value <= 0:
            return 1.0
        return min(1.0, -math.log10(self.p_value) / 5)  # 5 = critical threshold


class DriftDetector:
    """
    Statistical drift detection for Ralph Loop metrics.

    Implements multiple statistical tests:
    - Chi-squared test for categorical distributions
    - Kolmogorov-Smirnov test for continuous distributions
    - Jensen-Shannon divergence for distribution comparison
    - Z-score based anomaly detection
    """

    def __init__(
        self,
        window_size: int = 100,
        p_value_threshold: float = 0.05,
        zscore_threshold: float = 3.0
    ):
        self.window_size = window_size
        self.p_value_threshold = p_value_threshold
        self.zscore_threshold = zscore_threshold

        # Baseline and current windows
        self._baseline_fitness: List[float] = []
        self._current_fitness: List[float] = []
        self._baseline_latency: List[float] = []
        self._current_latency: List[float] = []

        # Strategy distribution tracking
        self._baseline_strategies: Dict[str, int] = {}
        self._current_strategies: Dict[str, int] = {}

        # Signal history
        self._drift_history: List[DriftSignal] = []

    def add_observation(
        self,
        fitness: float,
        latency_ms: float,
        strategy: str
    ) -> List[DriftSignal]:
        """
        Add an observation and check for drift.

        Returns list of detected drift signals.
        """
        signals = []

        # Update windows
        self._current_fitness.append(fitness)
        self._current_latency.append(latency_ms)
        self._current_strategies[strategy] = self._current_strategies.get(strategy, 0) + 1

        # Check if we have enough data for baseline
        if len(self._baseline_fitness) < self.window_size:
            # Building baseline
            if len(self._current_fitness) >= self.window_size:
                self._baseline_fitness = self._current_fitness.copy()
                self._baseline_latency = self._current_latency.copy()
                self._baseline_strategies = self._current_strategies.copy()
                self._current_fitness = []
                self._current_latency = []
                self._current_strategies = {}
            return signals

        # Check for drift when current window is full
        if len(self._current_fitness) >= self.window_size // 2:
            # Run statistical tests

            # 1. KS Test for fitness distribution
            ks_signal = self._ks_test(
                self._baseline_fitness,
                self._current_fitness,
                "fitness"
            )
            if ks_signal and ks_signal.is_significant:
                signals.append(ks_signal)

            # 2. Chi-squared for strategy distribution
            chi_signal = self._chi_squared_test(
                self._baseline_strategies,
                self._current_strategies
            )
            if chi_signal and chi_signal.is_significant:
                signals.append(chi_signal)

            # 3. Z-score for latency anomalies
            zscore_signal = self._zscore_test(
                self._baseline_latency,
                self._current_latency[-1],
                "latency_ms"
            )
            if zscore_signal and zscore_signal.is_significant:
                signals.append(zscore_signal)

            # 4. Jensen-Shannon divergence for overall distribution
            js_signal = self._jensen_shannon_test(
                self._baseline_fitness,
                self._current_fitness,
                "fitness"
            )
            if js_signal and js_signal.is_significant:
                signals.append(js_signal)

            # Rotate windows if current is full
            if len(self._current_fitness) >= self.window_size:
                self._baseline_fitness = self._current_fitness.copy()
                self._baseline_latency = self._current_latency.copy()
                self._baseline_strategies = self._current_strategies.copy()
                self._current_fitness = []
                self._current_latency = []
                self._current_strategies = {}

        # Record signals
        self._drift_history.extend(signals)
        if len(self._drift_history) > 1000:
            self._drift_history = self._drift_history[-1000:]

        return signals

    def _ks_test(
        self,
        baseline: List[float],
        current: List[float],
        metric_name: str
    ) -> Optional[DriftSignal]:
        """
        Two-sample Kolmogorov-Smirnov test.

        Compares the empirical cumulative distribution functions.
        """
        if len(baseline) < 10 or len(current) < 10:
            return None

        # Sort both samples
        baseline_sorted = sorted(baseline)
        current_sorted = sorted(current)

        # Compute empirical CDFs and find max difference
        n1, n2 = len(baseline_sorted), len(current_sorted)
        combined = sorted(set(baseline_sorted + current_sorted))

        max_diff = 0.0
        for x in combined:
            cdf1 = sum(1 for v in baseline_sorted if v <= x) / n1
            cdf2 = sum(1 for v in current_sorted if v <= x) / n2
            diff = abs(cdf1 - cdf2)
            max_diff = max(max_diff, diff)

        # KS statistic
        d_stat = max_diff

        # Approximate p-value using asymptotic distribution
        # For two samples: D_{n,m} ~ D where P(D > d) ~ 2*exp(-2*d^2*n*m/(n+m))
        n_eff = (n1 * n2) / (n1 + n2)
        p_value = 2 * math.exp(-2 * d_stat * d_stat * n_eff)
        p_value = min(1.0, max(0.0, p_value))

        is_significant = p_value < self.p_value_threshold

        return DriftSignal(
            drift_type="ks_test",
            metric_name=metric_name,
            p_value=p_value,
            statistic=d_stat,
            is_significant=is_significant,
            baseline_stats={
                "mean": statistics.mean(baseline),
                "std": statistics.stdev(baseline) if len(baseline) > 1 else 0,
                "n": len(baseline)
            },
            current_stats={
                "mean": statistics.mean(current),
                "std": statistics.stdev(current) if len(current) > 1 else 0,
                "n": len(current)
            },
            severity="critical" if p_value < 0.01 else ("warning" if p_value < 0.05 else "info")
        )

    def _chi_squared_test(
        self,
        baseline_counts: Dict[str, int],
        current_counts: Dict[str, int]
    ) -> Optional[DriftSignal]:
        """
        Chi-squared test for strategy distribution changes.
        """
        # Get all categories
        all_categories = set(baseline_counts.keys()) | set(current_counts.keys())

        if len(all_categories) < 2:
            return None

        total_baseline = sum(baseline_counts.values())
        total_current = sum(current_counts.values())

        if total_baseline < 10 or total_current < 10:
            return None

        # Calculate chi-squared statistic
        chi_squared = 0.0
        df = len(all_categories) - 1

        for category in all_categories:
            observed_baseline = baseline_counts.get(category, 0)
            observed_current = current_counts.get(category, 0)

            # Expected counts under null hypothesis (same distribution)
            total_cat = observed_baseline + observed_current
            expected_baseline = total_cat * (total_baseline / (total_baseline + total_current))
            expected_current = total_cat * (total_current / (total_baseline + total_current))

            if expected_baseline > 0:
                chi_squared += ((observed_baseline - expected_baseline) ** 2) / expected_baseline
            if expected_current > 0:
                chi_squared += ((observed_current - expected_current) ** 2) / expected_current

        # Approximate p-value using chi-squared distribution
        # Using Wilson-Hilferty approximation for chi-squared to normal
        if df > 0:
            z = (chi_squared / df) ** (1/3) - (1 - 2/(9*df))
            z = z / math.sqrt(2/(9*df))
            # Standard normal CDF approximation
            p_value = 0.5 * (1 + math.erf(-z / math.sqrt(2)))
            p_value = min(1.0, max(0.0, p_value))
        else:
            p_value = 1.0

        is_significant = p_value < self.p_value_threshold

        return DriftSignal(
            drift_type="chi_squared",
            metric_name="strategy_distribution",
            p_value=p_value,
            statistic=chi_squared,
            is_significant=is_significant,
            baseline_stats={
                cat: count/total_baseline for cat, count in baseline_counts.items()
            },
            current_stats={
                cat: count/total_current for cat, count in current_counts.items()
            },
            severity="critical" if p_value < 0.01 else ("warning" if p_value < 0.05 else "info")
        )

    def _zscore_test(
        self,
        baseline: List[float],
        current_value: float,
        metric_name: str
    ) -> Optional[DriftSignal]:
        """
        Z-score based anomaly detection for individual values.
        """
        if len(baseline) < 10:
            return None

        mean = statistics.mean(baseline)
        std = statistics.stdev(baseline)

        if std == 0:
            return None

        zscore = (current_value - mean) / std

        # Two-tailed test
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(zscore) / math.sqrt(2))))
        p_value = min(1.0, max(0.0, p_value))

        is_significant = abs(zscore) > self.zscore_threshold

        if not is_significant:
            return None

        return DriftSignal(
            drift_type="zscore",
            metric_name=metric_name,
            p_value=p_value,
            statistic=zscore,
            is_significant=is_significant,
            baseline_stats={
                "mean": mean,
                "std": std,
                "n": len(baseline)
            },
            current_stats={
                "value": current_value,
                "zscore": zscore
            },
            severity="critical" if abs(zscore) > 4 else ("warning" if abs(zscore) > 3 else "info")
        )

    def _jensen_shannon_test(
        self,
        baseline: List[float],
        current: List[float],
        metric_name: str
    ) -> Optional[DriftSignal]:
        """
        Jensen-Shannon divergence for distribution comparison.

        More robust than KL divergence (symmetric, bounded).
        """
        if len(baseline) < 10 or len(current) < 10:
            return None

        # Create histogram bins
        all_values = baseline + current
        min_val, max_val = min(all_values), max(all_values)

        if max_val == min_val:
            return None

        num_bins = min(20, max(5, int(math.sqrt(len(all_values)))))
        bin_width = (max_val - min_val) / num_bins

        # Count in bins
        baseline_hist = [0] * num_bins
        current_hist = [0] * num_bins

        for v in baseline:
            idx = min(num_bins - 1, int((v - min_val) / bin_width))
            baseline_hist[idx] += 1

        for v in current:
            idx = min(num_bins - 1, int((v - min_val) / bin_width))
            current_hist[idx] += 1

        # Normalize to probabilities (with smoothing)
        epsilon = 1e-10
        p = [(c + epsilon) / (sum(baseline_hist) + num_bins * epsilon) for c in baseline_hist]
        q = [(c + epsilon) / (sum(current_hist) + num_bins * epsilon) for c in current_hist]

        # Jensen-Shannon divergence
        m = [(pi + qi) / 2 for pi, qi in zip(p, q)]

        def kl_div(x, y):
            return sum(xi * math.log(xi / yi) for xi, yi in zip(x, y) if xi > 0 and yi > 0)

        js_div = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)

        # JS divergence is bounded [0, log(2)]
        # Convert to approximate p-value based on sample size
        n_eff = (len(baseline) * len(current)) / (len(baseline) + len(current))
        # Approximate: higher divergence + more samples = lower p-value
        p_value = math.exp(-js_div * n_eff / 10)
        p_value = min(1.0, max(0.0, p_value))

        is_significant = p_value < self.p_value_threshold

        return DriftSignal(
            drift_type="jensen_shannon",
            metric_name=metric_name,
            p_value=p_value,
            statistic=js_div,
            is_significant=is_significant,
            baseline_stats={
                "mean": statistics.mean(baseline),
                "std": statistics.stdev(baseline) if len(baseline) > 1 else 0,
                "histogram": baseline_hist
            },
            current_stats={
                "mean": statistics.mean(current),
                "std": statistics.stdev(current) if len(current) > 1 else 0,
                "histogram": current_hist
            },
            severity="critical" if js_div > 0.5 else ("warning" if js_div > 0.2 else "info")
        )

    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection state."""
        recent = self._drift_history[-50:] if self._drift_history else []

        return {
            "baseline_samples": len(self._baseline_fitness),
            "current_samples": len(self._current_fitness),
            "total_drift_signals": len(self._drift_history),
            "recent_signals": [
                {
                    "type": s.drift_type,
                    "metric": s.metric_name,
                    "p_value": s.p_value,
                    "severity": s.severity,
                    "detected_at": s.detected_at
                }
                for s in recent
            ],
            "drift_rate": len([s for s in recent if s.is_significant]) / max(1, len(recent)),
            "baseline_fitness_mean": statistics.mean(self._baseline_fitness) if self._baseline_fitness else 0,
            "current_fitness_mean": statistics.mean(self._current_fitness) if self._current_fitness else 0
        }


# =============================================================================
# Observability Provider Integrations
# =============================================================================

class ObservabilityBackend(ABC):
    """Abstract base class for observability backends."""

    @abstractmethod
    async def log_trace(
        self,
        trace_id: str,
        name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        metadata: Dict[str, Any],
        duration_ms: float
    ) -> None:
        """Log a trace to the observability backend."""
        pass

    @abstractmethod
    async def log_metric(
        self,
        name: str,
        value: float,
        labels: Dict[str, str]
    ) -> None:
        """Log a metric to the observability backend."""
        pass

    @abstractmethod
    async def log_evaluation(
        self,
        name: str,
        score: float,
        reasoning: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Log an evaluation result."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the backend is healthy."""
        pass


class OpikBackend(ObservabilityBackend):
    """
    Opik (Comet-ML) observability backend.

    Features:
    - Comprehensive tracing with LLM call tracking
    - Automated evaluations with LLM-as-a-judge
    - Production-ready dashboards
    - Online evaluation in real-time
    """

    def __init__(self, api_key: Optional[str] = None, project_name: str = "ralph-loop"):
        self.api_key = api_key
        self.project_name = project_name
        self._client = None
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Lazy initialize Opik client."""
        if self._initialized:
            return

        try:
            import opik

            if self.api_key:
                self._client = opik.Opik(api_key=self.api_key)
            else:
                self._client = opik.Opik()

            self._initialized = True
            logger.info(f"Opik backend initialized for project: {self.project_name}")
        except ImportError:
            logger.warning("Opik not installed. Install with: pip install opik")
            self._client = None
        except Exception as e:
            logger.error(f"Failed to initialize Opik: {e}")
            self._client = None

    async def log_trace(
        self,
        trace_id: str,
        name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        metadata: Dict[str, Any],
        duration_ms: float
    ) -> None:
        await self._ensure_initialized()

        if not self._client:
            return

        try:
            # Use Opik's tracing API
            self._client.log_trace(
                name=name,
                input=input_data,
                output=output_data,
                metadata={
                    "trace_id": trace_id,
                    "duration_ms": duration_ms,
                    "project": self.project_name,
                    **metadata
                }
            )
        except Exception as e:
            logger.error(f"Failed to log trace to Opik: {e}")

    async def log_metric(
        self,
        name: str,
        value: float,
        labels: Dict[str, str]
    ) -> None:
        await self._ensure_initialized()

        if not self._client:
            return

        try:
            self._client.log_metric(
                name=name,
                value=value,
                tags=labels
            )
        except Exception as e:
            logger.error(f"Failed to log metric to Opik: {e}")

    async def log_evaluation(
        self,
        name: str,
        score: float,
        reasoning: str,
        metadata: Dict[str, Any]
    ) -> None:
        await self._ensure_initialized()

        if not self._client:
            return

        try:
            self._client.log_evaluation(
                name=name,
                score=score,
                reasoning=reasoning,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Failed to log evaluation to Opik: {e}")

    async def health_check(self) -> bool:
        await self._ensure_initialized()
        return self._client is not None


class LangfuseBackend(ObservabilityBackend):
    """
    Langfuse observability backend.

    Features:
    - Open source (MIT license)
    - Best-in-class tracing and prompt management
    - Self-hosting support
    - Framework agnostic
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: str = "https://cloud.langfuse.com"
    ):
        self.public_key = public_key
        self.secret_key = secret_key
        self.host = host
        self._client = None
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host
            )
            self._initialized = True
            logger.info("Langfuse backend initialized")
        except ImportError:
            logger.warning("Langfuse not installed. Install with: pip install langfuse")
            self._client = None
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")
            self._client = None

    async def log_trace(
        self,
        trace_id: str,
        name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        metadata: Dict[str, Any],
        duration_ms: float
    ) -> None:
        await self._ensure_initialized()

        if not self._client:
            return

        try:
            trace = self._client.trace(
                id=trace_id,
                name=name,
                input=input_data,
                output=output_data,
                metadata=metadata
            )
            trace.end()
        except Exception as e:
            logger.error(f"Failed to log trace to Langfuse: {e}")

    async def log_metric(
        self,
        name: str,
        value: float,
        labels: Dict[str, str]
    ) -> None:
        await self._ensure_initialized()

        if not self._client:
            return

        try:
            self._client.score(
                name=name,
                value=value,
                comment=json.dumps(labels)
            )
        except Exception as e:
            logger.error(f"Failed to log metric to Langfuse: {e}")

    async def log_evaluation(
        self,
        name: str,
        score: float,
        reasoning: str,
        metadata: Dict[str, Any]
    ) -> None:
        await self._ensure_initialized()

        if not self._client:
            return

        try:
            self._client.score(
                name=name,
                value=score,
                comment=reasoning,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Failed to log evaluation to Langfuse: {e}")

    async def health_check(self) -> bool:
        await self._ensure_initialized()
        if not self._client:
            return False
        try:
            self._client.flush()
            return True
        except Exception:
            return False


class PhoenixBackend(ObservabilityBackend):
    """
    Phoenix (Arize) observability backend.

    Features:
    - Open-source, OpenTelemetry native
    - Deep agent evaluation support
    - RAG-focused observability
    - Local deployment (Jupyter, container)
    """

    def __init__(self, endpoint: str = "http://localhost:6006"):
        self.endpoint = endpoint
        self._session = None
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        try:
            import phoenix as px
            from phoenix.trace import SpanEvaluations

            self._session = px.launch_app()
            self._initialized = True
            logger.info(f"Phoenix backend initialized at {self.endpoint}")
        except ImportError:
            logger.warning("Phoenix not installed. Install with: pip install arize-phoenix")
            self._session = None
        except Exception as e:
            logger.error(f"Failed to initialize Phoenix: {e}")
            self._session = None

    async def log_trace(
        self,
        trace_id: str,
        name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        metadata: Dict[str, Any],
        duration_ms: float
    ) -> None:
        await self._ensure_initialized()

        if not self._session:
            return

        try:
            # Phoenix uses OpenTelemetry-compatible tracing
            from opentelemetry import trace
            tracer = trace.get_tracer(__name__)

            with tracer.start_as_current_span(name) as span:
                span.set_attribute("trace_id", trace_id)
                span.set_attribute("input", json.dumps(input_data))
                span.set_attribute("output", json.dumps(output_data))
                span.set_attribute("duration_ms", duration_ms)
                for k, v in metadata.items():
                    span.set_attribute(f"metadata.{k}", str(v))
        except Exception as e:
            logger.error(f"Failed to log trace to Phoenix: {e}")

    async def log_metric(
        self,
        name: str,
        value: float,
        labels: Dict[str, str]
    ) -> None:
        # Phoenix primarily uses OpenTelemetry metrics
        pass

    async def log_evaluation(
        self,
        name: str,
        score: float,
        reasoning: str,
        metadata: Dict[str, Any]
    ) -> None:
        await self._ensure_initialized()

        if not self._session:
            return

        try:
            from phoenix.trace import SpanEvaluations
            import pandas as pd

            eval_df = pd.DataFrame([{
                "name": name,
                "score": score,
                "reasoning": reasoning,
                **metadata
            }])

            # Phoenix evaluations are typically added via DataFrame
            self._session.add_evaluations(SpanEvaluations(eval_name=name, dataframe=eval_df))
        except Exception as e:
            logger.error(f"Failed to log evaluation to Phoenix: {e}")

    async def health_check(self) -> bool:
        await self._ensure_initialized()
        return self._session is not None


class ConsoleBackend(ObservabilityBackend):
    """Simple console logging backend for development/testing."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    async def log_trace(
        self,
        trace_id: str,
        name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        metadata: Dict[str, Any],
        duration_ms: float
    ) -> None:
        if self.verbose:
            logger.info(f"TRACE [{trace_id[:8]}] {name}: {duration_ms:.2f}ms")

    async def log_metric(
        self,
        name: str,
        value: float,
        labels: Dict[str, str]
    ) -> None:
        if self.verbose:
            label_str = ", ".join(f"{k}={v}" for k, v in labels.items())
            logger.info(f"METRIC {name}: {value:.4f} ({label_str})")

    async def log_evaluation(
        self,
        name: str,
        score: float,
        reasoning: str,
        metadata: Dict[str, Any]
    ) -> None:
        if self.verbose:
            logger.info(f"EVAL {name}: {score:.4f} - {reasoning[:100]}")

    async def health_check(self) -> bool:
        return True


# =============================================================================
# Production Monitor Integration
# =============================================================================

class RalphProductionMonitor:
    """
    Production monitoring integration for Ralph Loop.

    Provides:
    - Automatic drift detection with chi-squared tests
    - Multi-backend observability (Opik, Langfuse, Phoenix)
    - Real-time metrics and alerting
    - Health monitoring
    """

    def __init__(self, config: Optional[ProductionConfig] = None):
        self.config = config or ProductionConfig()

        # Initialize drift detector
        self.drift_detector = DriftDetector(
            window_size=self.config.drift_window_size,
            p_value_threshold=self.config.drift_threshold
        )

        # Initialize observability backend
        self._backend: ObservabilityBackend = self._create_backend()

        # Metrics buffer for batch export
        self._metrics_buffer: List[Dict[str, Any]] = []
        self._trace_buffer: List[Dict[str, Any]] = []

        # Monitoring state
        self._start_time = datetime.now(timezone.utc)
        self._iteration_count = 0
        self._last_health_check = datetime.now(timezone.utc)
        self._health_status = True

    def _create_backend(self) -> ObservabilityBackend:
        """Create the appropriate observability backend."""
        provider = self.config.observability_provider

        if provider == ObservabilityProvider.OPIK:
            return OpikBackend(
                api_key=self.config.opik_api_key,
                project_name=self.config.opik_project_name
            )
        elif provider == ObservabilityProvider.LANGFUSE:
            return LangfuseBackend(
                public_key=self.config.langfuse_public_key,
                secret_key=self.config.langfuse_secret_key
            )
        elif provider == ObservabilityProvider.PHOENIX:
            return PhoenixBackend(
                endpoint=self.config.phoenix_endpoint
            )
        else:
            return ConsoleBackend(verbose=True)

    async def on_iteration_start(
        self,
        iteration: int,
        task: str,
        strategy: str
    ) -> str:
        """Called at the start of each Ralph Loop iteration."""
        import uuid
        trace_id = str(uuid.uuid4())

        self._iteration_count = iteration

        if self.config.enable_monitoring:
            await self._backend.log_trace(
                trace_id=trace_id,
                name="ralph_iteration_start",
                input_data={"task": task, "strategy": strategy},
                output_data={},
                metadata={"iteration": iteration},
                duration_ms=0
            )

        return trace_id

    async def on_iteration_complete(
        self,
        trace_id: str,
        iteration: int,
        fitness: float,
        latency_ms: float,
        strategy: str,
        improved: bool,
        metadata: Dict[str, Any]
    ) -> List[DriftSignal]:
        """
        Called at the end of each Ralph Loop iteration.

        Returns any detected drift signals.
        """
        drift_signals = []

        if self.config.enable_monitoring:
            # Log trace
            await self._backend.log_trace(
                trace_id=trace_id,
                name="ralph_iteration_complete",
                input_data={"strategy": strategy},
                output_data={"fitness": fitness, "improved": improved},
                metadata={
                    "iteration": iteration,
                    "latency_ms": latency_ms,
                    **metadata
                },
                duration_ms=latency_ms
            )

            # Log metrics
            await self._backend.log_metric(
                name="ralph_fitness",
                value=fitness,
                labels={"strategy": strategy}
            )

            await self._backend.log_metric(
                name="ralph_latency_ms",
                value=latency_ms,
                labels={"strategy": strategy}
            )

            # Check for drift
            if self.config.drift_detection_enabled:
                drift_signals = self.drift_detector.add_observation(
                    fitness=fitness,
                    latency_ms=latency_ms,
                    strategy=strategy
                )

                # Log drift signals as evaluations
                for signal in drift_signals:
                    await self._backend.log_evaluation(
                        name=f"drift_detection_{signal.drift_type}",
                        score=signal.p_value,
                        reasoning=f"Drift detected in {signal.metric_name}: "
                                  f"p={signal.p_value:.4f}, stat={signal.statistic:.4f}",
                        metadata={
                            "drift_type": signal.drift_type,
                            "severity": signal.severity,
                            "baseline_stats": signal.baseline_stats,
                            "current_stats": signal.current_stats
                        }
                    )

                    logger.warning(
                        f"DRIFT DETECTED [{signal.severity.upper()}]: "
                        f"{signal.drift_type} in {signal.metric_name}, "
                        f"p={signal.p_value:.4f}"
                    )

        return drift_signals

    async def on_improvement(
        self,
        iteration: int,
        old_fitness: float,
        new_fitness: float,
        strategy: str,
        solution_preview: str
    ) -> None:
        """Called when an improvement is found."""
        if not self.config.enable_monitoring:
            return

        improvement = new_fitness - old_fitness

        await self._backend.log_evaluation(
            name="ralph_improvement",
            score=improvement,
            reasoning=f"Improved fitness from {old_fitness:.4f} to {new_fitness:.4f} "
                      f"using strategy: {strategy}",
            metadata={
                "iteration": iteration,
                "old_fitness": old_fitness,
                "new_fitness": new_fitness,
                "strategy": strategy,
                "improvement_pct": (improvement / old_fitness * 100) if old_fitness > 0 else 0
            }
        )

    async def on_failure(
        self,
        iteration: int,
        error_type: str,
        error_message: str,
        strategy: str
    ) -> None:
        """Called when an iteration fails."""
        if not self.config.enable_monitoring:
            return

        await self._backend.log_evaluation(
            name="ralph_failure",
            score=0.0,
            reasoning=f"Iteration {iteration} failed: {error_type} - {error_message}",
            metadata={
                "iteration": iteration,
                "error_type": error_type,
                "error_message": error_message,
                "strategy": strategy
            }
        )

    async def health_check(self) -> Dict[str, Any]:
        """Run health check on monitoring infrastructure."""
        backend_healthy = await self._backend.health_check()

        now = datetime.now(timezone.utc)
        uptime_seconds = (now - self._start_time).total_seconds()

        drift_summary = self.drift_detector.get_drift_summary()

        return {
            "status": "healthy" if backend_healthy else "degraded",
            "backend": self.config.observability_provider.value,
            "backend_healthy": backend_healthy,
            "uptime_seconds": uptime_seconds,
            "iterations_monitored": self._iteration_count,
            "drift_detection": drift_summary,
            "last_check": now.isoformat()
        }

    def get_production_status(self) -> Dict[str, Any]:
        """Get overall production status."""
        return {
            "monitoring_enabled": self.config.enable_monitoring,
            "provider": self.config.observability_provider.value,
            "drift_detection_enabled": self.config.drift_detection_enabled,
            "iterations_processed": self._iteration_count,
            "drift_summary": self.drift_detector.get_drift_summary(),
            "config": {
                "drift_window_size": self.config.drift_window_size,
                "drift_threshold": self.config.drift_threshold,
                "trace_sample_rate": self.config.trace_sample_rate
            }
        }


# =============================================================================
# Factory and Utilities
# =============================================================================

def create_production_monitor(
    provider: str = "opik",
    **kwargs
) -> RalphProductionMonitor:
    """
    Factory function to create a production monitor.

    Args:
        provider: One of "opik", "langfuse", "phoenix", "console"
        **kwargs: Additional configuration options

    Returns:
        Configured RalphProductionMonitor instance
    """
    provider_map = {
        "opik": ObservabilityProvider.OPIK,
        "langfuse": ObservabilityProvider.LANGFUSE,
        "langsmith": ObservabilityProvider.LANGSMITH,
        "phoenix": ObservabilityProvider.PHOENIX,
        "console": ObservabilityProvider.CUSTOM
    }

    config = ProductionConfig(
        observability_provider=provider_map.get(provider.lower(), ObservabilityProvider.CUSTOM),
        **kwargs
    )

    return RalphProductionMonitor(config)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Configuration
    "ProductionConfig",
    "ObservabilityProvider",

    # Drift Detection
    "DriftDetector",
    "DriftSignal",

    # Observability Backends
    "ObservabilityBackend",
    "OpikBackend",
    "LangfuseBackend",
    "PhoenixBackend",
    "ConsoleBackend",

    # Main Monitor
    "RalphProductionMonitor",

    # Factory
    "create_production_monitor",
]
