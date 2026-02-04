"""
Adaptive Timeout Handler - Dynamic Timeout Calculation Based on Historical Latency
==================================================================================

Provides intelligent timeout management by tracking per-operation latency
percentiles and calculating dynamic timeouts using p99 * 1.5 formula.

Features:
- Per-operation latency tracking with sliding window
- Percentile calculations (p50, p90, p95, p99)
- Dynamic timeout calculation (p99 * 1.5 by default)
- Minimum and maximum bounds for safety
- Per-adapter timeout profiles
- Alerting when timeouts are adjusted
- Thread-safe and async-safe design
- Integration with HTTP pool and adapters

Performance Target: 30% reduction in timeout-related failures through
intelligent timeout adjustment based on actual latency patterns.

Usage:
    from core.adaptive_timeout import (
        AdaptiveTimeout,
        get_adaptive_timeout,
        TimeoutProfile,
    )

    # Get shared timeout handler for an operation
    timeout = await get_adaptive_timeout("exa_search")

    # Get current dynamic timeout
    current_timeout = timeout.get_timeout()

    # Record operation latency
    timeout.record_latency(350.0)  # 350ms

    # Use with context manager for automatic recording
    async with timeout.measure() as ctx:
        result = await my_operation()
    # Latency automatically recorded

    # Get statistics
    stats = timeout.get_stats()
    print(f"P99 latency: {stats['p99_ms']}ms")
    print(f"Current timeout: {stats['current_timeout_ms']}ms")
"""

from __future__ import annotations

import asyncio
import logging
import time
import threading
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TimeoutProfile:
    """
    Configuration profile for adaptive timeout behavior.

    Attributes:
        min_timeout_ms: Minimum allowed timeout (safety floor)
        max_timeout_ms: Maximum allowed timeout (safety ceiling)
        default_timeout_ms: Initial timeout before enough data is collected
        percentile_target: Which percentile to use for calculation (default: 99)
        multiplier: Multiplier applied to percentile (default: 1.5)
        window_size: Number of samples to retain for calculation
        warmup_samples: Minimum samples before using adaptive timeout
        adjustment_threshold: Min % change to trigger timeout adjustment
        cooldown_seconds: Min seconds between timeout adjustments
    """
    min_timeout_ms: float = 1000.0  # 1 second floor
    max_timeout_ms: float = 300000.0  # 5 minutes ceiling
    default_timeout_ms: float = 30000.0  # 30 seconds default
    percentile_target: float = 99.0  # Use P99
    multiplier: float = 1.5  # P99 * 1.5
    window_size: int = 1000  # Keep last 1000 samples
    warmup_samples: int = 10  # Need 10 samples before adaptive
    adjustment_threshold: float = 0.1  # 10% change triggers adjustment
    cooldown_seconds: float = 60.0  # 1 minute between adjustments


# Default profiles for known adapters/operations
DEFAULT_PROFILES: Dict[str, TimeoutProfile] = {
    # Research adapters - generally fast
    "exa_search": TimeoutProfile(
        min_timeout_ms=500.0,
        max_timeout_ms=60000.0,
        default_timeout_ms=10000.0,
        warmup_samples=5,
    ),
    "exa_fast": TimeoutProfile(
        min_timeout_ms=200.0,
        max_timeout_ms=5000.0,
        default_timeout_ms=1000.0,
        warmup_samples=5,
    ),
    "exa_deep": TimeoutProfile(
        min_timeout_ms=2000.0,
        max_timeout_ms=120000.0,
        default_timeout_ms=30000.0,
        warmup_samples=3,
    ),
    "tavily_search": TimeoutProfile(
        min_timeout_ms=1000.0,
        max_timeout_ms=120000.0,
        default_timeout_ms=30000.0,
    ),
    "perplexity_research": TimeoutProfile(
        min_timeout_ms=2000.0,
        max_timeout_ms=300000.0,
        default_timeout_ms=60000.0,
    ),
    "jina_search": TimeoutProfile(
        min_timeout_ms=500.0,
        max_timeout_ms=60000.0,
        default_timeout_ms=15000.0,
    ),
    "firecrawl_scrape": TimeoutProfile(
        min_timeout_ms=2000.0,
        max_timeout_ms=180000.0,
        default_timeout_ms=60000.0,
    ),

    # Memory operations - typically fast
    "memory_store": TimeoutProfile(
        min_timeout_ms=100.0,
        max_timeout_ms=10000.0,
        default_timeout_ms=2000.0,
        warmup_samples=20,
    ),
    "memory_retrieve": TimeoutProfile(
        min_timeout_ms=50.0,
        max_timeout_ms=5000.0,
        default_timeout_ms=1000.0,
        warmup_samples=20,
    ),
    "memory_search": TimeoutProfile(
        min_timeout_ms=100.0,
        max_timeout_ms=10000.0,
        default_timeout_ms=3000.0,
        warmup_samples=15,
    ),

    # HTTP pool operations
    "http_connect": TimeoutProfile(
        min_timeout_ms=500.0,
        max_timeout_ms=30000.0,
        default_timeout_ms=5000.0,
    ),
    "http_request": TimeoutProfile(
        min_timeout_ms=1000.0,
        max_timeout_ms=60000.0,
        default_timeout_ms=15000.0,
    ),

    # Default for unknown operations
    "_default": TimeoutProfile(),
}


# =============================================================================
# Latency Statistics
# =============================================================================

@dataclass
class LatencyStats:
    """Statistics calculated from latency samples."""
    sample_count: int = 0
    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    stddev_ms: float = 0.0
    current_timeout_ms: float = 0.0
    timeout_adjustments: int = 0
    last_adjustment: Optional[datetime] = None
    is_warmed_up: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for serialization."""
        return {
            "sample_count": self.sample_count,
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p90_ms": round(self.p90_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "stddev_ms": round(self.stddev_ms, 2),
            "current_timeout_ms": round(self.current_timeout_ms, 2),
            "timeout_adjustments": self.timeout_adjustments,
            "last_adjustment": (
                self.last_adjustment.isoformat()
                if self.last_adjustment else None
            ),
            "is_warmed_up": self.is_warmed_up,
        }


def calculate_percentile(sorted_values: List[float], percentile: float) -> float:
    """
    Calculate a percentile from sorted values.

    Args:
        sorted_values: Pre-sorted list of values
        percentile: Percentile to calculate (0-100)

    Returns:
        Percentile value
    """
    if not sorted_values:
        return 0.0

    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]

    # Use linear interpolation for percentile calculation
    k = (n - 1) * (percentile / 100.0)
    f = int(k)
    c = min(f + 1, n - 1)

    if f == c:
        return sorted_values[f]

    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)

    return d0 + d1


# =============================================================================
# Adaptive Timeout Handler
# =============================================================================

class AdaptiveTimeout:
    """
    Adaptive timeout handler that calculates dynamic timeouts based on
    historical latency patterns.

    Uses a sliding window of latency samples to calculate percentiles
    and adjusts timeouts using the formula: timeout = p99 * multiplier

    Thread-safe and async-safe implementation with proper locking.

    Example:
        timeout = AdaptiveTimeout("my_operation")

        # Record latencies
        timeout.record_latency(100.0)
        timeout.record_latency(150.0)
        timeout.record_latency(120.0)

        # Get current timeout (adaptive if warmed up)
        current = timeout.get_timeout()

        # Use with measurement context
        async with timeout.measure() as ctx:
            await do_work()
        # Latency recorded automatically
    """

    def __init__(
        self,
        operation_name: str,
        profile: Optional[TimeoutProfile] = None,
    ):
        """
        Initialize adaptive timeout handler.

        Args:
            operation_name: Name of the operation being tracked
            profile: Timeout profile configuration (uses default if not provided)
        """
        self._operation_name = operation_name
        self._profile = profile or self._get_default_profile(operation_name)

        # Latency sample storage
        self._samples: Deque[float] = deque(maxlen=self._profile.window_size)
        self._sorted_samples: List[float] = []
        self._samples_dirty = False

        # Current timeout state
        self._current_timeout_ms = self._profile.default_timeout_ms
        self._timeout_adjustments = 0
        self._last_adjustment: Optional[datetime] = None

        # Thread safety
        self._lock = threading.Lock()

        # Alert callbacks
        self._alert_callbacks: List[Callable[[str, float, float], None]] = []

        logger.debug(
            f"AdaptiveTimeout initialized: operation={operation_name}, "
            f"default_timeout={self._profile.default_timeout_ms}ms"
        )

    def _get_default_profile(self, operation_name: str) -> TimeoutProfile:
        """Get the default profile for an operation."""
        # Try exact match first
        if operation_name in DEFAULT_PROFILES:
            return DEFAULT_PROFILES[operation_name]

        # Try adapter prefix matching
        for prefix in ["exa_", "tavily_", "perplexity_", "jina_", "firecrawl_",
                       "memory_", "http_"]:
            if operation_name.startswith(prefix):
                # Find best matching profile
                for profile_name, profile in DEFAULT_PROFILES.items():
                    if profile_name.startswith(prefix):
                        return profile

        return DEFAULT_PROFILES["_default"]

    def record_latency(self, latency_ms: float) -> None:
        """
        Record a latency sample.

        Args:
            latency_ms: Operation latency in milliseconds
        """
        if latency_ms <= 0:
            return

        with self._lock:
            self._samples.append(latency_ms)
            self._samples_dirty = True

            # Check if we should recalculate timeout
            self._maybe_adjust_timeout()

    def _ensure_sorted(self) -> None:
        """Ensure sorted samples are up to date."""
        if self._samples_dirty:
            self._sorted_samples = sorted(self._samples)
            self._samples_dirty = False

    def _maybe_adjust_timeout(self) -> None:
        """Check if timeout should be adjusted and do so if needed."""
        # Check warmup
        if len(self._samples) < self._profile.warmup_samples:
            return

        # Check cooldown
        if self._last_adjustment:
            elapsed = (datetime.utcnow() - self._last_adjustment).total_seconds()
            if elapsed < self._profile.cooldown_seconds:
                return

        # Calculate new timeout
        self._ensure_sorted()
        target_percentile = calculate_percentile(
            self._sorted_samples,
            self._profile.percentile_target
        )

        new_timeout = target_percentile * self._profile.multiplier

        # Apply bounds
        new_timeout = max(self._profile.min_timeout_ms, new_timeout)
        new_timeout = min(self._profile.max_timeout_ms, new_timeout)

        # Check if change is significant
        if self._current_timeout_ms > 0:
            change_ratio = abs(new_timeout - self._current_timeout_ms) / self._current_timeout_ms
            if change_ratio < self._profile.adjustment_threshold:
                return

        # Apply adjustment
        old_timeout = self._current_timeout_ms
        self._current_timeout_ms = new_timeout
        self._timeout_adjustments += 1
        self._last_adjustment = datetime.utcnow()

        logger.info(
            f"AdaptiveTimeout adjusted: operation={self._operation_name}, "
            f"old={old_timeout:.0f}ms, new={new_timeout:.0f}ms, "
            f"p99={target_percentile:.0f}ms"
        )

        # Fire alert callbacks
        for callback in self._alert_callbacks:
            try:
                callback(self._operation_name, old_timeout, new_timeout)
            except Exception as e:
                logger.warning(f"Alert callback error: {e}")

    def get_timeout(self) -> float:
        """
        Get current timeout in milliseconds.

        Returns the adaptive timeout if warmed up, otherwise returns
        the default timeout from the profile.

        Returns:
            Timeout in milliseconds
        """
        with self._lock:
            return self._current_timeout_ms

    def get_timeout_seconds(self) -> float:
        """
        Get current timeout in seconds.

        Returns:
            Timeout in seconds
        """
        return self.get_timeout() / 1000.0

    def is_warmed_up(self) -> bool:
        """Check if enough samples have been collected for adaptive behavior."""
        with self._lock:
            return len(self._samples) >= self._profile.warmup_samples

    def get_stats(self) -> LatencyStats:
        """
        Get comprehensive latency statistics.

        Returns:
            LatencyStats object with all calculated metrics
        """
        with self._lock:
            if not self._samples:
                return LatencyStats(
                    current_timeout_ms=self._current_timeout_ms,
                    timeout_adjustments=self._timeout_adjustments,
                    last_adjustment=self._last_adjustment,
                )

            self._ensure_sorted()

            sample_list = list(self._samples)
            n = len(sample_list)
            mean = sum(sample_list) / n

            # Calculate standard deviation
            variance = sum((x - mean) ** 2 for x in sample_list) / n
            stddev = variance ** 0.5

            return LatencyStats(
                sample_count=n,
                min_ms=self._sorted_samples[0],
                max_ms=self._sorted_samples[-1],
                mean_ms=mean,
                p50_ms=calculate_percentile(self._sorted_samples, 50),
                p90_ms=calculate_percentile(self._sorted_samples, 90),
                p95_ms=calculate_percentile(self._sorted_samples, 95),
                p99_ms=calculate_percentile(self._sorted_samples, 99),
                stddev_ms=stddev,
                current_timeout_ms=self._current_timeout_ms,
                timeout_adjustments=self._timeout_adjustments,
                last_adjustment=self._last_adjustment,
                is_warmed_up=n >= self._profile.warmup_samples,
            )

    def reset(self) -> None:
        """Reset all latency data and return to default timeout."""
        with self._lock:
            self._samples.clear()
            self._sorted_samples.clear()
            self._samples_dirty = False
            self._current_timeout_ms = self._profile.default_timeout_ms
            self._timeout_adjustments = 0
            self._last_adjustment = None

        logger.debug(f"AdaptiveTimeout reset: operation={self._operation_name}")

    def add_alert_callback(
        self,
        callback: Callable[[str, float, float], None]
    ) -> None:
        """
        Add a callback to be called when timeout is adjusted.

        Callback receives: (operation_name, old_timeout_ms, new_timeout_ms)
        """
        self._alert_callbacks.append(callback)

    @asynccontextmanager
    async def measure(self):
        """
        Context manager that automatically measures and records latency.

        Usage:
            async with timeout.measure() as ctx:
                result = await my_operation()
            # Latency is automatically recorded

        Yields:
            MeasurementContext with latency_ms attribute after completion
        """
        ctx = MeasurementContext()
        ctx.start_time = time.perf_counter()

        try:
            yield ctx
        finally:
            ctx.end_time = time.perf_counter()
            ctx.latency_ms = (ctx.end_time - ctx.start_time) * 1000
            self.record_latency(ctx.latency_ms)

    @property
    def operation_name(self) -> str:
        """Get the operation name."""
        return self._operation_name

    @property
    def profile(self) -> TimeoutProfile:
        """Get the timeout profile."""
        return self._profile


class MeasurementContext:
    """Context object for latency measurement."""

    def __init__(self):
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.latency_ms: float = 0.0


# =============================================================================
# Global Registry
# =============================================================================

_timeout_registry: Dict[str, AdaptiveTimeout] = {}
_registry_lock = asyncio.Lock()


async def get_adaptive_timeout(
    operation_name: str,
    profile: Optional[TimeoutProfile] = None,
) -> AdaptiveTimeout:
    """
    Get or create an adaptive timeout handler for an operation.

    Timeout handlers are cached and reused for the same operation name.

    Args:
        operation_name: Name of the operation
        profile: Optional custom profile (only used when creating new handler)

    Returns:
        AdaptiveTimeout instance

    Example:
        timeout = await get_adaptive_timeout("exa_search")
        current = timeout.get_timeout()
    """
    if operation_name in _timeout_registry:
        return _timeout_registry[operation_name]

    async with _registry_lock:
        # Double-check after acquiring lock
        if operation_name in _timeout_registry:
            return _timeout_registry[operation_name]

        timeout = AdaptiveTimeout(operation_name, profile)
        _timeout_registry[operation_name] = timeout
        logger.debug(f"Created adaptive timeout: {operation_name}")

    return timeout


def get_adaptive_timeout_sync(
    operation_name: str,
    profile: Optional[TimeoutProfile] = None,
) -> AdaptiveTimeout:
    """
    Synchronous version of get_adaptive_timeout.

    Creates a timeout handler without async locking. Safe for
    single-threaded initialization.
    """
    if operation_name in _timeout_registry:
        return _timeout_registry[operation_name]

    timeout = AdaptiveTimeout(operation_name, profile)
    _timeout_registry[operation_name] = timeout

    return timeout


def get_all_timeout_stats() -> Dict[str, Dict[str, Any]]:
    """
    Get statistics for all registered timeout handlers.

    Returns:
        Dictionary mapping operation names to their stats
    """
    stats = {}
    for name, timeout in _timeout_registry.items():
        stats[name] = timeout.get_stats().to_dict()
    return stats


def reset_all_timeouts() -> int:
    """
    Reset all timeout handlers to defaults.

    Returns:
        Number of handlers reset
    """
    count = 0
    for timeout in _timeout_registry.values():
        timeout.reset()
        count += 1
    logger.info(f"Reset {count} adaptive timeout handlers")
    return count


# =============================================================================
# Integration Helpers
# =============================================================================

class AdaptiveTimeoutMiddleware:
    """
    Middleware that wraps async operations with adaptive timeout tracking.

    Usage:
        middleware = AdaptiveTimeoutMiddleware("my_adapter")

        @middleware.wrap
        async def my_api_call():
            ...

        # Or use directly
        result = await middleware.execute(my_api_call, arg1, arg2)
    """

    def __init__(
        self,
        adapter_name: str,
        profile: Optional[TimeoutProfile] = None,
    ):
        """
        Initialize middleware.

        Args:
            adapter_name: Base name for adapter operations
            profile: Optional default profile for operations
        """
        self._adapter_name = adapter_name
        self._default_profile = profile
        self._timeouts: Dict[str, AdaptiveTimeout] = {}

    def get_timeout_handler(self, method_name: str) -> AdaptiveTimeout:
        """Get or create timeout handler for a method."""
        operation_name = f"{self._adapter_name}_{method_name}"

        if operation_name not in self._timeouts:
            self._timeouts[operation_name] = AdaptiveTimeout(
                operation_name,
                self._default_profile
            )

        return self._timeouts[operation_name]

    async def execute(
        self,
        method_name: str,
        func: Callable,
        *args: Any,
        timeout_override: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a function with adaptive timeout tracking.

        Args:
            method_name: Name of the method being called
            func: Async function to execute
            *args: Positional arguments for func
            timeout_override: Optional timeout override in seconds
            **kwargs: Keyword arguments for func

        Returns:
            Result from the function

        Raises:
            asyncio.TimeoutError: If operation times out
        """
        timeout_handler = self.get_timeout_handler(method_name)

        # Determine timeout to use
        if timeout_override is not None:
            timeout_seconds = timeout_override
        else:
            timeout_seconds = timeout_handler.get_timeout_seconds()

        async with timeout_handler.measure():
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=timeout_seconds
            )

    def wrap(self, method_name: Optional[str] = None):
        """
        Decorator to wrap a method with adaptive timeout tracking.

        Args:
            method_name: Optional explicit method name (uses function name if not provided)
        """
        def decorator(func):
            name = method_name or func.__name__

            async def wrapper(*args, **kwargs):
                return await self.execute(name, func, *args, **kwargs)

            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__
            return wrapper

        return decorator

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all methods tracked by this middleware."""
        return {
            name: handler.get_stats().to_dict()
            for name, handler in self._timeouts.items()
        }


# =============================================================================
# Timeout Alerting
# =============================================================================

class TimeoutAlertManager:
    """
    Manages alerts when adaptive timeouts are significantly adjusted.

    Supports multiple alert channels and severity levels.
    """

    def __init__(self):
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._thresholds = {
            "warning": 0.25,  # 25% change
            "error": 0.50,    # 50% change
            "critical": 1.00, # 100% change
        }

    def register_callback(
        self,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Register an alert callback."""
        self._callbacks.append(callback)

    def on_timeout_adjusted(
        self,
        operation_name: str,
        old_timeout_ms: float,
        new_timeout_ms: float
    ) -> None:
        """
        Handle a timeout adjustment event.

        Calculates severity and fires appropriate alerts.
        """
        if old_timeout_ms <= 0:
            return

        change_ratio = abs(new_timeout_ms - old_timeout_ms) / old_timeout_ms
        direction = "increased" if new_timeout_ms > old_timeout_ms else "decreased"

        # Determine severity
        severity = "info"
        for level, threshold in sorted(
            self._thresholds.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if change_ratio >= threshold:
                severity = level
                break

        alert = {
            "type": "adaptive_timeout_adjustment",
            "operation": operation_name,
            "old_timeout_ms": old_timeout_ms,
            "new_timeout_ms": new_timeout_ms,
            "change_percent": round(change_ratio * 100, 1),
            "direction": direction,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.log(
            logging.WARNING if severity in ("warning", "error", "critical") else logging.INFO,
            f"Timeout alert [{severity}]: {operation_name} {direction} by {alert['change_percent']}% "
            f"({old_timeout_ms:.0f}ms -> {new_timeout_ms:.0f}ms)"
        )

        # Fire callbacks
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.warning(f"Alert callback error: {e}")


# Global alert manager
_alert_manager = TimeoutAlertManager()


def get_timeout_alert_manager() -> TimeoutAlertManager:
    """Get the global timeout alert manager."""
    return _alert_manager


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core classes
    "AdaptiveTimeout",
    "TimeoutProfile",
    "LatencyStats",
    "MeasurementContext",

    # Registry functions
    "get_adaptive_timeout",
    "get_adaptive_timeout_sync",
    "get_all_timeout_stats",
    "reset_all_timeouts",

    # Integration helpers
    "AdaptiveTimeoutMiddleware",

    # Alerting
    "TimeoutAlertManager",
    "get_timeout_alert_manager",

    # Constants
    "DEFAULT_PROFILES",

    # Utility functions
    "calculate_percentile",
]
