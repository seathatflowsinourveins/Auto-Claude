"""
Comprehensive Error Recovery Module - Production-Grade Resilience

Provides enterprise-level error recovery with:
- Automatic retry classification (transient vs permanent)
- Fallback chain execution with multiple strategies
- Partial result handling and aggregation
- Error aggregation and reporting with patterns
- Memory failure fallback to in-memory storage
- RAG failure with cached/degraded response

This module builds on existing resilience patterns to provide
a unified error recovery framework for the platform.

Usage:
    from core.error_recovery import ErrorRecovery, RecoveryConfig

    # Create recovery instance
    recovery = ErrorRecovery(
        config=RecoveryConfig(
            max_retries=3,
            enable_fallback=True,
            enable_partial_results=True,
        )
    )

    # Execute with automatic recovery
    result = await recovery.execute(
        operation=my_operation,
        fallback_chain=["primary", "secondary", "tertiary"],
        partial_handler=handle_partial,
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import time
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# ERROR CLASSIFICATION
# =============================================================================

class ErrorType(str, Enum):
    """Classification of error types for recovery decisions."""
    TRANSIENT = "transient"          # Network glitches, timeouts - retry
    RATE_LIMITED = "rate_limited"    # Rate limit - backoff and retry
    CAPACITY = "capacity"            # Server overload - retry with backoff
    PERMANENT = "permanent"          # Invalid input, auth - don't retry
    PARTIAL = "partial"              # Partial success - handle specially
    UNKNOWN = "unknown"              # Unclassified - conservative retry


class RecoveryAction(str, Enum):
    """Actions the recovery system can take."""
    RETRY = "retry"                  # Retry the same operation
    FALLBACK = "fallback"            # Use fallback adapter/method
    CACHE = "cache"                  # Return cached result
    DEGRADE = "degrade"              # Return degraded response
    PARTIAL = "partial"              # Return partial result
    FAIL = "fail"                    # Give up and propagate error


@dataclass
class ClassifiedError:
    """Error with classification metadata."""
    original_error: Exception
    error_type: ErrorType
    recommended_action: RecoveryAction
    retry_after_seconds: Optional[float] = None
    is_retriable: bool = True
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def error_message(self) -> str:
        return str(self.original_error)

    @property
    def error_class(self) -> str:
        return type(self.original_error).__name__


class ErrorClassifier:
    """
    Classifies errors to determine appropriate recovery strategy.

    Analyzes exception types, error messages, and HTTP status codes
    to categorize errors as transient, permanent, or partial.
    """

    # HTTP status codes that indicate transient errors
    TRANSIENT_STATUS_CODES: Set[int] = {
        408,  # Request Timeout
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
        520, 521, 522, 523, 524,  # Cloudflare errors
    }

    # HTTP status codes that indicate rate limiting
    RATE_LIMIT_STATUS_CODES: Set[int] = {429}

    # HTTP status codes that indicate permanent errors
    PERMANENT_STATUS_CODES: Set[int] = {
        400,  # Bad Request
        401,  # Unauthorized
        403,  # Forbidden
        404,  # Not Found
        405,  # Method Not Allowed
        410,  # Gone
        422,  # Unprocessable Entity
        451,  # Unavailable For Legal Reasons
    }

    # Exception types that are transient
    TRANSIENT_EXCEPTIONS: Tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
        ConnectionResetError,
        ConnectionRefusedError,
        BrokenPipeError,
    )

    # Exception types that are permanent
    PERMANENT_EXCEPTIONS: Tuple[Type[Exception], ...] = (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        NotImplementedError,
    )

    # Keywords in error messages that suggest transient issues
    TRANSIENT_KEYWORDS: Set[str] = {
        "timeout", "timed out", "connection reset", "connection refused",
        "network unreachable", "temporary", "unavailable",
        "retry", "overloaded", "busy", "capacity",
    }

    # Keywords that suggest permanent issues
    PERMANENT_KEYWORDS: Set[str] = {
        "invalid", "unauthorized", "forbidden", "not found",
        "bad request", "authentication", "permission denied",
        "quota exceeded", "billing",
    }

    def classify(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ClassifiedError:
        """
        Classify an error and determine recovery strategy.

        Args:
            error: The exception to classify
            context: Additional context about the operation

        Returns:
            ClassifiedError with classification and recommendations
        """
        context = context or {}
        error_message = str(error).lower()

        # Check for HTTP status code
        status_code = self._extract_status_code(error)

        if status_code:
            if status_code in self.RATE_LIMIT_STATUS_CODES:
                retry_after = self._extract_retry_after(error)
                return ClassifiedError(
                    original_error=error,
                    error_type=ErrorType.RATE_LIMITED,
                    recommended_action=RecoveryAction.RETRY,
                    retry_after_seconds=retry_after or 60.0,
                    is_retriable=True,
                    context=context,
                )
            elif status_code in self.TRANSIENT_STATUS_CODES:
                return ClassifiedError(
                    original_error=error,
                    error_type=ErrorType.TRANSIENT,
                    recommended_action=RecoveryAction.RETRY,
                    is_retriable=True,
                    context=context,
                )
            elif status_code in self.PERMANENT_STATUS_CODES:
                return ClassifiedError(
                    original_error=error,
                    error_type=ErrorType.PERMANENT,
                    recommended_action=RecoveryAction.FAIL,
                    is_retriable=False,
                    context=context,
                )

        # Check exception type
        if isinstance(error, self.TRANSIENT_EXCEPTIONS):
            return ClassifiedError(
                original_error=error,
                error_type=ErrorType.TRANSIENT,
                recommended_action=RecoveryAction.RETRY,
                is_retriable=True,
                context=context,
            )

        if isinstance(error, self.PERMANENT_EXCEPTIONS):
            return ClassifiedError(
                original_error=error,
                error_type=ErrorType.PERMANENT,
                recommended_action=RecoveryAction.FAIL,
                is_retriable=False,
                context=context,
            )

        # Check error message keywords
        for keyword in self.TRANSIENT_KEYWORDS:
            if keyword in error_message:
                return ClassifiedError(
                    original_error=error,
                    error_type=ErrorType.TRANSIENT,
                    recommended_action=RecoveryAction.RETRY,
                    is_retriable=True,
                    context=context,
                )

        for keyword in self.PERMANENT_KEYWORDS:
            if keyword in error_message:
                return ClassifiedError(
                    original_error=error,
                    error_type=ErrorType.PERMANENT,
                    recommended_action=RecoveryAction.FAIL,
                    is_retriable=False,
                    context=context,
                )

        # Check for partial success indicators
        if self._is_partial_error(error):
            return ClassifiedError(
                original_error=error,
                error_type=ErrorType.PARTIAL,
                recommended_action=RecoveryAction.PARTIAL,
                is_retriable=False,
                context=context,
            )

        # Default: unknown, try conservative retry
        return ClassifiedError(
            original_error=error,
            error_type=ErrorType.UNKNOWN,
            recommended_action=RecoveryAction.RETRY,
            is_retriable=True,
            context=context,
        )

    def _extract_status_code(self, error: Exception) -> Optional[int]:
        """Extract HTTP status code from error if present."""
        # Check common attribute names
        for attr in ["status_code", "status", "code"]:
            if hasattr(error, attr):
                val = getattr(error, attr)
                if isinstance(val, int):
                    return val

        # Check for response object
        if hasattr(error, "response"):
            response = getattr(error, "response")
            if hasattr(response, "status_code"):
                return getattr(response, "status_code")
            if hasattr(response, "status"):
                return getattr(response, "status")

        return None

    def _extract_retry_after(self, error: Exception) -> Optional[float]:
        """Extract Retry-After header value if present."""
        # Check for headers in response
        if hasattr(error, "response"):
            response = getattr(error, "response")
            headers = getattr(response, "headers", {})
            retry_after = headers.get("Retry-After") or headers.get("retry-after")
            if retry_after:
                try:
                    return float(retry_after)
                except (ValueError, TypeError):
                    pass

        # Check for retry_after attribute
        if hasattr(error, "retry_after"):
            return getattr(error, "retry_after")

        return None

    def _is_partial_error(self, error: Exception) -> bool:
        """Check if error indicates partial success."""
        error_str = str(error).lower()
        partial_indicators = ["partial", "incomplete", "some results"]
        return any(indicator in error_str for indicator in partial_indicators)


# =============================================================================
# RECOVERY STRATEGIES
# =============================================================================

class RecoveryStrategy(ABC, Generic[T]):
    """Base class for recovery strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging."""
        pass

    @abstractmethod
    async def can_recover(self, classified_error: ClassifiedError) -> bool:
        """Check if this strategy can handle the error."""
        pass

    @abstractmethod
    async def recover(
        self,
        classified_error: ClassifiedError,
        operation: Callable[..., Awaitable[T]],
        args: tuple,
        kwargs: dict,
    ) -> T:
        """Attempt to recover from the error."""
        pass


class RetryStrategy(RecoveryStrategy[T]):
    """
    Retry with exponential backoff.

    Handles transient errors by retrying with increasing delays.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: float = 0.5,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self._current_attempt = 0

    @property
    def name(self) -> str:
        return "retry"

    async def can_recover(self, classified_error: ClassifiedError) -> bool:
        return (
            classified_error.is_retriable and
            self._current_attempt < self.max_retries
        )

    async def recover(
        self,
        classified_error: ClassifiedError,
        operation: Callable[..., Awaitable[T]],
        args: tuple,
        kwargs: dict,
    ) -> T:
        import random

        self._current_attempt += 1

        # Calculate delay
        delay = self.base_delay * (self.exponential_base ** (self._current_attempt - 1))
        delay = min(delay, self.max_delay)

        # Apply jitter
        if self.jitter > 0:
            jitter_range = delay * self.jitter
            delay = delay + random.uniform(-jitter_range, jitter_range)
            delay = max(0.01, delay)

        # Respect retry-after if specified
        if classified_error.retry_after_seconds:
            delay = max(delay, classified_error.retry_after_seconds)

        logger.info(
            f"Retry {self._current_attempt}/{self.max_retries} "
            f"after {delay:.2f}s for {classified_error.error_class}"
        )

        await asyncio.sleep(delay)
        return await operation(*args, **kwargs)

    def reset(self) -> None:
        """Reset retry counter."""
        self._current_attempt = 0


class FallbackChainStrategy(RecoveryStrategy[T]):
    """
    Fallback to alternative adapters/methods.

    Tries a chain of fallback operations when primary fails.
    """

    def __init__(
        self,
        fallbacks: List[Callable[..., Awaitable[T]]],
        adapter_names: Optional[List[str]] = None,
    ):
        """
        Initialize with fallback chain.

        Args:
            fallbacks: List of fallback functions to try
            adapter_names: Names for logging (optional)
        """
        self.fallbacks = fallbacks
        self.adapter_names = adapter_names or [f"fallback_{i}" for i in range(len(fallbacks))]
        self._current_index = 0

    @property
    def name(self) -> str:
        return "fallback_chain"

    async def can_recover(self, classified_error: ClassifiedError) -> bool:
        return self._current_index < len(self.fallbacks)

    async def recover(
        self,
        classified_error: ClassifiedError,
        operation: Callable[..., Awaitable[T]],
        args: tuple,
        kwargs: dict,
    ) -> T:
        if self._current_index >= len(self.fallbacks):
            raise classified_error.original_error

        fallback = self.fallbacks[self._current_index]
        fallback_name = self.adapter_names[self._current_index]
        self._current_index += 1

        logger.info(f"Falling back to {fallback_name}")
        return await fallback(*args, **kwargs)

    def reset(self) -> None:
        """Reset fallback index."""
        self._current_index = 0


class CacheRecoveryStrategy(RecoveryStrategy[T]):
    """
    Return cached result on failure.

    Uses in-memory or external cache to return stale data
    when fresh data is unavailable.
    """

    def __init__(
        self,
        cache: Optional[Dict[str, Tuple[T, datetime]]] = None,
        max_age_seconds: float = 3600.0,
    ):
        """
        Initialize with cache.

        Args:
            cache: Cache dict mapping keys to (value, timestamp) tuples
            max_age_seconds: Maximum age for cached results
        """
        self._cache: Dict[str, Tuple[T, datetime]] = cache or {}
        self.max_age_seconds = max_age_seconds

    @property
    def name(self) -> str:
        return "cache"

    def _make_cache_key(self, args: tuple, kwargs: dict) -> str:
        """Create a cache key from arguments."""
        key_data = str((args, sorted(kwargs.items())))
        return hashlib.md5(key_data.encode()).hexdigest()

    def store(self, args: tuple, kwargs: dict, result: T) -> None:
        """Store a result in the cache."""
        key = self._make_cache_key(args, kwargs)
        self._cache[key] = (result, datetime.now())

    async def can_recover(self, classified_error: ClassifiedError) -> bool:
        # Can recover if we have cached data for the operation
        args = classified_error.context.get("args", ())
        kwargs = classified_error.context.get("kwargs", {})
        key = self._make_cache_key(args, kwargs)

        if key not in self._cache:
            return False

        _, cached_time = self._cache[key]
        age = (datetime.now() - cached_time).total_seconds()
        return age <= self.max_age_seconds

    async def recover(
        self,
        classified_error: ClassifiedError,
        operation: Callable[..., Awaitable[T]],
        args: tuple,
        kwargs: dict,
    ) -> T:
        key = self._make_cache_key(args, kwargs)
        if key in self._cache:
            result, cached_time = self._cache[key]
            age = (datetime.now() - cached_time).total_seconds()
            logger.info(f"Returning cached result (age: {age:.1f}s)")
            return result
        raise classified_error.original_error


class DegradedResponseStrategy(RecoveryStrategy[T]):
    """
    Return degraded response on failure.

    Returns a minimal or placeholder response when full
    response is unavailable.
    """

    def __init__(
        self,
        default_factory: Callable[[], T],
        degradation_marker: Optional[str] = "degraded",
    ):
        """
        Initialize with default factory.

        Args:
            default_factory: Function that creates default response
            degradation_marker: Marker to indicate degraded response
        """
        self.default_factory = default_factory
        self.degradation_marker = degradation_marker

    @property
    def name(self) -> str:
        return "degraded"

    async def can_recover(self, classified_error: ClassifiedError) -> bool:
        # Can always provide degraded response
        return True

    async def recover(
        self,
        classified_error: ClassifiedError,
        operation: Callable[..., Awaitable[T]],
        args: tuple,
        kwargs: dict,
    ) -> T:
        logger.warning(f"Returning degraded response for {classified_error.error_class}")
        result = self.default_factory()

        # Mark as degraded if possible
        if isinstance(result, dict) and self.degradation_marker:
            result["_degraded"] = True
            result["_error"] = classified_error.error_message

        return result


class MemoryFallbackStrategy(RecoveryStrategy[T]):
    """
    Fallback to in-memory storage on memory backend failure.

    When persistent memory fails, uses temporary in-memory storage
    to allow operations to continue.
    """

    def __init__(self):
        self._fallback_storage: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "memory_fallback"

    async def can_recover(self, classified_error: ClassifiedError) -> bool:
        # Can recover memory-related operations
        context = classified_error.context
        return context.get("operation_type") in ("memory_read", "memory_write", "memory_search")

    async def recover(
        self,
        classified_error: ClassifiedError,
        operation: Callable[..., Awaitable[T]],
        args: tuple,
        kwargs: dict,
    ) -> T:
        context = classified_error.context
        op_type = context.get("operation_type", "")

        logger.warning(f"Using in-memory fallback for {op_type}")

        if op_type == "memory_read":
            key = kwargs.get("key") or (args[0] if args else None)
            return self._fallback_storage.get(key)  # type: ignore

        elif op_type == "memory_write":
            key = kwargs.get("key") or (args[0] if args else None)
            value = kwargs.get("value") or (args[1] if len(args) > 1 else None)
            self._fallback_storage[key] = value
            return None  # type: ignore

        elif op_type == "memory_search":
            query = kwargs.get("query", "").lower()
            results = [
                v for k, v in self._fallback_storage.items()
                if query in str(v).lower()
            ]
            return results[:kwargs.get("limit", 10)]  # type: ignore

        raise classified_error.original_error


# =============================================================================
# PARTIAL RESULT HANDLING
# =============================================================================

@dataclass
class PartialResult(Generic[T]):
    """Container for partial results."""
    results: List[T]
    errors: List[ClassifiedError]
    total_requested: int
    total_succeeded: int
    is_complete: bool

    @property
    def success_rate(self) -> float:
        if self.total_requested == 0:
            return 0.0
        return self.total_succeeded / self.total_requested

    def merge(self, other: "PartialResult[T]") -> "PartialResult[T]":
        """Merge two partial results."""
        return PartialResult(
            results=self.results + other.results,
            errors=self.errors + other.errors,
            total_requested=self.total_requested + other.total_requested,
            total_succeeded=self.total_succeeded + other.total_succeeded,
            is_complete=self.is_complete and other.is_complete,
        )


class PartialResultHandler(Generic[T]):
    """
    Handles partial results from multi-item operations.

    Aggregates successful results while tracking failures,
    allowing operations to succeed partially.
    """

    def __init__(
        self,
        min_success_rate: float = 0.5,
        fail_on_empty: bool = True,
    ):
        """
        Initialize handler.

        Args:
            min_success_rate: Minimum success rate to consider successful
            fail_on_empty: Whether to fail if no results
        """
        self.min_success_rate = min_success_rate
        self.fail_on_empty = fail_on_empty
        self._results: List[T] = []
        self._errors: List[ClassifiedError] = []
        self._total_requested = 0

    def add_result(self, result: T) -> None:
        """Add a successful result."""
        self._results.append(result)

    def add_error(self, error: ClassifiedError) -> None:
        """Add a failed item."""
        self._errors.append(error)

    def set_total_requested(self, total: int) -> None:
        """Set total items requested."""
        self._total_requested = total

    def get_partial_result(self) -> PartialResult[T]:
        """Get the partial result."""
        return PartialResult(
            results=self._results.copy(),
            errors=self._errors.copy(),
            total_requested=self._total_requested,
            total_succeeded=len(self._results),
            is_complete=len(self._errors) == 0,
        )

    def should_accept_partial(self) -> bool:
        """Check if partial result is acceptable."""
        if self.fail_on_empty and not self._results:
            return False

        success_rate = len(self._results) / max(1, self._total_requested)
        return success_rate >= self.min_success_rate

    def reset(self) -> None:
        """Reset handler state."""
        self._results.clear()
        self._errors.clear()
        self._total_requested = 0


# =============================================================================
# ERROR AGGREGATION AND REPORTING
# =============================================================================

@dataclass
class ErrorPattern:
    """Pattern of recurring errors."""
    error_type: ErrorType
    error_class: str
    count: int
    first_seen: datetime
    last_seen: datetime
    sample_messages: List[str]
    affected_operations: Set[str]

    @property
    def frequency_per_hour(self) -> float:
        duration = (self.last_seen - self.first_seen).total_seconds() / 3600
        if duration < 0.001:
            return float(self.count)
        return self.count / duration


class ErrorAggregator:
    """
    Aggregates and analyzes errors for patterns.

    Tracks error frequencies, patterns, and provides
    insights for system health monitoring.
    """

    def __init__(
        self,
        max_errors: int = 10000,
        pattern_window_minutes: int = 60,
    ):
        self.max_errors = max_errors
        self.pattern_window = timedelta(minutes=pattern_window_minutes)
        self._errors: List[ClassifiedError] = []
        self._patterns: Dict[str, ErrorPattern] = {}
        self._operation_errors: Dict[str, List[ClassifiedError]] = defaultdict(list)

    async def record(
        self,
        classified_error: ClassifiedError,
        operation_name: str,
    ) -> None:
        """Record an error occurrence."""
        if True:  # Lock removed: non-reentrant asyncio.Lock caused deadlock
            self._errors.append(classified_error)
            self._operation_errors[operation_name].append(classified_error)

            # Update patterns
            pattern_key = f"{classified_error.error_type.value}:{classified_error.error_class}"
            if pattern_key in self._patterns:
                pattern = self._patterns[pattern_key]
                pattern.count += 1
                pattern.last_seen = datetime.now()
                pattern.affected_operations.add(operation_name)
                if len(pattern.sample_messages) < 5:
                    pattern.sample_messages.append(classified_error.error_message[:200])
            else:
                self._patterns[pattern_key] = ErrorPattern(
                    error_type=classified_error.error_type,
                    error_class=classified_error.error_class,
                    count=1,
                    first_seen=datetime.now(),
                    last_seen=datetime.now(),
                    sample_messages=[classified_error.error_message[:200]],
                    affected_operations={operation_name},
                )

            # Prune old errors
            if len(self._errors) > self.max_errors:
                self._errors = self._errors[-self.max_errors:]

    async def get_recent_errors(
        self,
        minutes: int = 60,
        error_type: Optional[ErrorType] = None,
    ) -> List[ClassifiedError]:
        """Get errors from recent time window."""
        if True:  # Lock removed: non-reentrant asyncio.Lock caused deadlock
            cutoff = datetime.now() - timedelta(minutes=minutes)
            errors = [e for e in self._errors if e.timestamp > cutoff]
            if error_type:
                errors = [e for e in errors if e.error_type == error_type]
            return errors

    async def get_error_patterns(
        self,
        min_count: int = 3,
    ) -> List[ErrorPattern]:
        """Get recurring error patterns."""
        if True:  # Lock removed: non-reentrant asyncio.Lock caused deadlock
            patterns = [p for p in self._patterns.values() if p.count >= min_count]
            return sorted(patterns, key=lambda p: p.count, reverse=True)

    async def get_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        if True:  # Lock removed: non-reentrant asyncio.Lock caused deadlock
            total = len(self._errors)
            by_type: Dict[ErrorType, int] = defaultdict(int)
            by_class: Dict[str, int] = defaultdict(int)

            for error in self._errors:
                by_type[error.error_type] += 1
                by_class[error.error_class] += 1

            recent = await self.get_recent_errors(minutes=60)

            return {
                "total_errors": total,
                "errors_last_hour": len(recent),
                "by_type": {k.value: v for k, v in by_type.items()},
                "by_class": dict(by_class),
                "pattern_count": len(self._patterns),
                "affected_operations": len(self._operation_errors),
            }

    async def get_health_assessment(self) -> Dict[str, Any]:
        """Get system health assessment based on errors."""
        stats = await self.get_statistics()
        patterns = await self.get_error_patterns()

        # Assess health level
        errors_per_hour = stats["errors_last_hour"]
        if errors_per_hour == 0:
            health_level = "healthy"
        elif errors_per_hour < 10:
            health_level = "warning"
        elif errors_per_hour < 100:
            health_level = "degraded"
        else:
            health_level = "critical"

        # Identify top issues
        top_patterns = patterns[:5]
        critical_patterns = [p for p in patterns if p.count > 100]

        return {
            "health_level": health_level,
            "errors_per_hour": errors_per_hour,
            "top_patterns": [
                {
                    "type": p.error_type.value,
                    "class": p.error_class,
                    "count": p.count,
                    "frequency_per_hour": round(p.frequency_per_hour, 2),
                }
                for p in top_patterns
            ],
            "critical_patterns_count": len(critical_patterns),
            "recommendations": self._generate_recommendations(stats, patterns),
        }

    def _generate_recommendations(
        self,
        stats: Dict[str, Any],
        patterns: List[ErrorPattern],
    ) -> List[str]:
        """Generate recommendations based on error patterns."""
        recommendations = []

        # Check for rate limiting issues
        rate_limit_patterns = [
            p for p in patterns if p.error_type == ErrorType.RATE_LIMITED
        ]
        if rate_limit_patterns:
            recommendations.append(
                "High rate of rate limiting errors. Consider implementing "
                "request batching or increasing rate limit backoff."
            )

        # Check for network issues
        transient_patterns = [
            p for p in patterns if p.error_type == ErrorType.TRANSIENT
        ]
        if len(transient_patterns) > 3:
            recommendations.append(
                "Multiple transient error patterns detected. "
                "Check network connectivity and upstream service health."
            )

        # Check for permanent errors (likely bugs)
        permanent_patterns = [
            p for p in patterns
            if p.error_type == ErrorType.PERMANENT and p.count > 10
        ]
        if permanent_patterns:
            recommendations.append(
                "Recurring permanent errors detected. "
                "Review input validation and API usage patterns."
            )

        return recommendations

    async def clear(self) -> None:
        """Clear all error records."""
        if True:  # Lock removed: non-reentrant asyncio.Lock caused deadlock
            self._errors.clear()
            self._patterns.clear()
            self._operation_errors.clear()


# =============================================================================
# RECOVERY CONFIGURATION
# =============================================================================

@dataclass
class RecoveryConfig:
    """Configuration for error recovery behavior."""

    # Retry settings
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.5

    # Fallback settings
    enable_fallback: bool = True
    fallback_timeout_seconds: float = 30.0

    # Cache settings
    enable_cache_recovery: bool = True
    cache_max_age_seconds: float = 3600.0

    # Degraded response settings
    enable_degraded_response: bool = True

    # Memory fallback settings
    enable_memory_fallback: bool = True

    # Partial result settings
    enable_partial_results: bool = True
    min_success_rate: float = 0.5

    # Aggregation settings
    max_aggregated_errors: int = 10000
    pattern_window_minutes: int = 60


# =============================================================================
# MAIN ERROR RECOVERY CLASS
# =============================================================================

@dataclass
class RecoveryResult(Generic[T]):
    """Result of a recovery operation."""
    success: bool
    result: Optional[T] = None
    recovery_used: Optional[str] = None
    attempts: int = 0
    total_time_ms: float = 0.0
    errors: List[ClassifiedError] = field(default_factory=list)
    partial_result: Optional[PartialResult[T]] = None

    @property
    def is_partial(self) -> bool:
        return self.partial_result is not None and not self.partial_result.is_complete


class ErrorRecovery(Generic[T]):
    """
    Comprehensive error recovery system.

    Combines multiple recovery strategies to provide robust
    error handling for operations.

    Usage:
        recovery = ErrorRecovery(config=RecoveryConfig())

        # Simple execution with retry
        result = await recovery.execute(my_operation, arg1, arg2)

        # With fallback chain
        result = await recovery.execute_with_fallback(
            primary=primary_adapter.search,
            fallbacks=[secondary.search, tertiary.search],
            query="search term",
        )

        # With partial result handling
        result = await recovery.execute_batch(
            items=items,
            process_fn=process_item,
        )
    """

    def __init__(
        self,
        config: Optional[RecoveryConfig] = None,
        classifier: Optional[ErrorClassifier] = None,
        aggregator: Optional[ErrorAggregator] = None,
    ):
        self.config = config or RecoveryConfig()
        self.classifier = classifier or ErrorClassifier()
        self.aggregator = aggregator or ErrorAggregator(
            max_errors=self.config.max_aggregated_errors,
            pattern_window_minutes=self.config.pattern_window_minutes,
        )

        # Initialize strategies
        self._retry_strategy = RetryStrategy(
            max_retries=self.config.max_retries,
            base_delay=self.config.base_delay,
            max_delay=self.config.max_delay,
            exponential_base=self.config.exponential_base,
            jitter=self.config.jitter,
        )

        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_strategy = CacheRecoveryStrategy(
            cache=self._cache,
            max_age_seconds=self.config.cache_max_age_seconds,
        )

        self._memory_fallback = MemoryFallbackStrategy()

    async def execute(
        self,
        operation: Callable[..., Awaitable[T]],
        *args,
        operation_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> RecoveryResult[T]:
        """
        Execute an operation with automatic error recovery.

        Args:
            operation: Async function to execute
            *args: Positional arguments for operation
            operation_name: Name for logging/tracking
            context: Additional context for recovery
            **kwargs: Keyword arguments for operation

        Returns:
            RecoveryResult containing the result or error information
        """
        start_time = time.time()
        operation_name = operation_name or getattr(operation, "__name__", "unknown")
        context = context or {}
        context["args"] = args
        context["kwargs"] = kwargs

        errors: List[ClassifiedError] = []
        recovery_used: Optional[str] = None
        attempts = 0

        self._retry_strategy.reset()

        while True:
            attempts += 1
            try:
                result = await operation(*args, **kwargs)

                # Cache successful result
                if self.config.enable_cache_recovery:
                    self._cache_strategy.store(args, kwargs, result)

                return RecoveryResult(
                    success=True,
                    result=result,
                    recovery_used=recovery_used,
                    attempts=attempts,
                    total_time_ms=(time.time() - start_time) * 1000,
                    errors=errors,
                )

            except Exception as e:
                classified = self.classifier.classify(e, context)
                errors.append(classified)
                await self.aggregator.record(classified, operation_name)

                # Try retry strategy
                if await self._retry_strategy.can_recover(classified):
                    try:
                        result = await self._retry_strategy.recover(
                            classified, operation, args, kwargs
                        )
                        recovery_used = "retry"
                        return RecoveryResult(
                            success=True,
                            result=result,
                            recovery_used=recovery_used,
                            attempts=attempts,
                            total_time_ms=(time.time() - start_time) * 1000,
                            errors=errors,
                        )
                    except Exception:
                        continue  # Try next retry

                # Try cache recovery
                if self.config.enable_cache_recovery:
                    if await self._cache_strategy.can_recover(classified):
                        try:
                            result = await self._cache_strategy.recover(
                                classified, operation, args, kwargs
                            )
                            recovery_used = "cache"
                            return RecoveryResult(
                                success=True,
                                result=result,
                                recovery_used=recovery_used,
                                attempts=attempts,
                                total_time_ms=(time.time() - start_time) * 1000,
                                errors=errors,
                            )
                        except Exception:
                            pass

                # Try memory fallback for memory operations
                if self.config.enable_memory_fallback:
                    if await self._memory_fallback.can_recover(classified):
                        try:
                            result = await self._memory_fallback.recover(
                                classified, operation, args, kwargs
                            )
                            recovery_used = "memory_fallback"
                            return RecoveryResult(
                                success=True,
                                result=result,
                                recovery_used=recovery_used,
                                attempts=attempts,
                                total_time_ms=(time.time() - start_time) * 1000,
                                errors=errors,
                            )
                        except Exception:
                            pass

                # All recovery strategies exhausted
                return RecoveryResult(
                    success=False,
                    result=None,
                    recovery_used=None,
                    attempts=attempts,
                    total_time_ms=(time.time() - start_time) * 1000,
                    errors=errors,
                )

    async def execute_with_fallback(
        self,
        primary: Callable[..., Awaitable[T]],
        fallbacks: List[Callable[..., Awaitable[T]]],
        *args,
        fallback_names: Optional[List[str]] = None,
        operation_name: Optional[str] = None,
        **kwargs,
    ) -> RecoveryResult[T]:
        """
        Execute with fallback chain.

        Args:
            primary: Primary operation to try first
            fallbacks: List of fallback operations
            *args: Arguments for all operations
            fallback_names: Names for fallbacks (for logging)
            operation_name: Name for the overall operation
            **kwargs: Keyword arguments for all operations

        Returns:
            RecoveryResult containing the result
        """
        start_time = time.time()
        operation_name = operation_name or getattr(primary, "__name__", "unknown")
        errors: List[ClassifiedError] = []
        attempts = 0

        # Create fallback chain strategy
        fallback_strategy = FallbackChainStrategy(
            fallbacks=fallbacks,
            adapter_names=fallback_names,
        )

        # Try primary with retry
        primary_result = await self.execute(
            primary, *args,
            operation_name=f"{operation_name}_primary",
            **kwargs
        )

        if primary_result.success:
            return primary_result

        errors.extend(primary_result.errors)
        attempts += primary_result.attempts

        # Try fallbacks
        context: Dict[str, Any] = {"args": args, "kwargs": kwargs}

        for i, fallback in enumerate(fallbacks):
            attempts += 1
            fallback_name = (
                fallback_names[i] if fallback_names and i < len(fallback_names)
                else f"fallback_{i}"
            )

            try:
                logger.info(f"Trying fallback: {fallback_name}")
                result = await fallback(*args, **kwargs)

                return RecoveryResult(
                    success=True,
                    result=result,
                    recovery_used=f"fallback:{fallback_name}",
                    attempts=attempts,
                    total_time_ms=(time.time() - start_time) * 1000,
                    errors=errors,
                )

            except Exception as e:
                classified = self.classifier.classify(e, context)
                errors.append(classified)
                await self.aggregator.record(classified, f"{operation_name}_{fallback_name}")
                continue

        # All fallbacks exhausted
        return RecoveryResult(
            success=False,
            result=None,
            recovery_used=None,
            attempts=attempts,
            total_time_ms=(time.time() - start_time) * 1000,
            errors=errors,
        )

    async def execute_batch(
        self,
        items: List[Any],
        process_fn: Callable[[Any], Awaitable[T]],
        operation_name: Optional[str] = None,
        concurrency: int = 5,
    ) -> RecoveryResult[List[T]]:
        """
        Execute batch operations with partial result handling.

        Args:
            items: Items to process
            process_fn: Function to process each item
            operation_name: Name for the operation
            concurrency: Maximum concurrent operations

        Returns:
            RecoveryResult with partial results
        """
        start_time = time.time()
        operation_name = operation_name or getattr(process_fn, "__name__", "batch")

        partial_handler: PartialResultHandler[T] = PartialResultHandler(
            min_success_rate=self.config.min_success_rate,
            fail_on_empty=True,
        )
        partial_handler.set_total_requested(len(items))

        errors: List[ClassifiedError] = []
        semaphore = asyncio.Semaphore(concurrency)

        async def process_with_semaphore(item: Any) -> Optional[T]:
            async with semaphore:
                try:
                    result = await process_fn(item)
                    partial_handler.add_result(result)
                    return result
                except Exception as e:
                    classified = self.classifier.classify(e, {"item": item})
                    errors.append(classified)
                    partial_handler.add_error(classified)
                    await self.aggregator.record(classified, operation_name)
                    return None

        # Process all items
        await asyncio.gather(*[process_with_semaphore(item) for item in items])

        partial_result = partial_handler.get_partial_result()
        success = partial_handler.should_accept_partial()

        return RecoveryResult(
            success=success,
            result=partial_result.results if success else None,
            recovery_used="partial" if success and errors else None,
            attempts=len(items),
            total_time_ms=(time.time() - start_time) * 1000,
            errors=errors,
            partial_result=partial_result,
        )

    async def get_health_status(self) -> Dict[str, Any]:
        """Get error recovery system health status."""
        return await self.aggregator.get_health_assessment()

    async def get_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return await self.aggregator.get_statistics()


# =============================================================================
# DECORATOR FOR AUTOMATIC RECOVERY
# =============================================================================

def with_recovery(
    config: Optional[RecoveryConfig] = None,
    operation_name: Optional[str] = None,
    fallbacks: Optional[List[Callable]] = None,
):
    """
    Decorator to add automatic error recovery to a function.

    Usage:
        @with_recovery(config=RecoveryConfig(max_retries=5))
        async def my_operation():
            ...

        @with_recovery(fallbacks=[backup_operation])
        async def primary_operation():
            ...
    """
    recovery = ErrorRecovery(config=config)

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[RecoveryResult[T]]]:
        @wraps(fn)
        async def wrapper(*args, **kwargs) -> RecoveryResult[T]:
            name = operation_name or fn.__name__

            if fallbacks:
                return await recovery.execute_with_fallback(
                    primary=fn,
                    fallbacks=fallbacks,
                    *args,
                    operation_name=name,
                    **kwargs,
                )
            else:
                return await recovery.execute(
                    fn, *args,
                    operation_name=name,
                    **kwargs,
                )

        return wrapper

    return decorator


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_recovery(config: Optional[RecoveryConfig] = None) -> ErrorRecovery:
    """Create an ErrorRecovery instance with optional config."""
    return ErrorRecovery(config=config)


def create_classifier() -> ErrorClassifier:
    """Create an ErrorClassifier instance."""
    return ErrorClassifier()


def create_aggregator(
    max_errors: int = 10000,
    pattern_window_minutes: int = 60,
) -> ErrorAggregator:
    """Create an ErrorAggregator instance."""
    return ErrorAggregator(
        max_errors=max_errors,
        pattern_window_minutes=pattern_window_minutes,
    )


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_global_recovery: Optional[ErrorRecovery] = None


def get_error_recovery(config: Optional[RecoveryConfig] = None) -> ErrorRecovery:
    """Get or create global error recovery instance."""
    global _global_recovery
    if _global_recovery is None:
        _global_recovery = ErrorRecovery(config=config)
    return _global_recovery


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Error classification
    "ErrorType",
    "RecoveryAction",
    "ClassifiedError",
    "ErrorClassifier",
    # Recovery strategies
    "RecoveryStrategy",
    "RetryStrategy",
    "FallbackChainStrategy",
    "CacheRecoveryStrategy",
    "DegradedResponseStrategy",
    "MemoryFallbackStrategy",
    # Partial results
    "PartialResult",
    "PartialResultHandler",
    # Error aggregation
    "ErrorPattern",
    "ErrorAggregator",
    # Configuration
    "RecoveryConfig",
    # Main classes
    "RecoveryResult",
    "ErrorRecovery",
    # Decorator
    "with_recovery",
    # Factory functions
    "create_recovery",
    "create_classifier",
    "create_aggregator",
    "get_error_recovery",
]
