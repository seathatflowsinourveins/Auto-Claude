"""
Comprehensive Error Handling for Unleashed Platform

Provides:
- Structured exception hierarchy for all SDK operations
- Error categorization and severity levels
- Automatic error recovery strategies
- Error aggregation and reporting
- Graceful degradation patterns

This module ensures robust error management across all adapters and pipelines.
"""

import asyncio
import inspect
import traceback
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# ERROR CATEGORIES AND SEVERITY
# ============================================================================

class ErrorSeverity(Enum):
    """Severity levels for errors."""
    DEBUG = "debug"       # Development/debugging issues
    INFO = "info"         # Informational, may indicate minor issues
    WARNING = "warning"   # Potential problems, system continues
    ERROR = "error"       # Significant problems, operation failed
    CRITICAL = "critical" # System-level failure, requires intervention


class ErrorCategory(Enum):
    """Categories of errors for routing and handling."""
    NETWORK = "network"           # Network connectivity issues
    AUTHENTICATION = "auth"        # Auth/API key issues
    RATE_LIMIT = "rate_limit"     # Rate limiting/quota exceeded
    TIMEOUT = "timeout"           # Operation timeouts
    VALIDATION = "validation"     # Input validation errors
    CONFIGURATION = "config"      # Configuration errors
    RESOURCE = "resource"         # Resource not found/unavailable
    DEPENDENCY = "dependency"     # External dependency failures
    INTERNAL = "internal"         # Internal logic errors
    DATA = "data"                 # Data format/parsing errors
    PERMISSION = "permission"     # Permission denied
    QUOTA = "quota"               # Usage quota exceeded
    UNKNOWN = "unknown"           # Unclassified errors


# ============================================================================
# EXCEPTION HIERARCHY
# ============================================================================

class UnleashedError(Exception):
    """
    Base exception for all Unleashed platform errors.

    Provides structured error information with category, severity,
    context, and recovery suggestions.
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = True,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.cause = cause
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.timestamp = datetime.now()
        self.traceback_str = traceback.format_exc()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "details": self.details,
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None,
        }

    def __str__(self):
        return f"[{self.category.value.upper()}] {self.message}"


# Network Errors
class NetworkError(UnleashedError):
    """Network-related errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.NETWORK, **kwargs)


class ConnectionError(NetworkError):
    """Failed to establish connection."""
    pass


class TimeoutError(NetworkError):
    """Operation timed out."""
    def __init__(self, message: str, timeout: float, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.TIMEOUT,
            details={"timeout": timeout},
            **kwargs
        )


# Authentication Errors
class AuthenticationError(UnleashedError):
    """Authentication-related errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUTHENTICATION,
            recoverable=False,
            **kwargs
        )


class InvalidAPIKeyError(AuthenticationError):
    """Invalid or expired API key."""
    pass


class PermissionDeniedError(UnleashedError):
    """Permission denied for operation."""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.PERMISSION,
            recoverable=False,
            **kwargs
        )


# Rate Limiting Errors
class RateLimitError(UnleashedError):
    """Rate limit exceeded."""
    def __init__(self, message: str, retry_after: Optional[float] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.RATE_LIMIT,
            retry_after=retry_after or 60.0,
            **kwargs
        )


class QuotaExceededError(UnleashedError):
    """Usage quota exceeded."""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.QUOTA,
            recoverable=False,
            **kwargs
        )


# Validation Errors
class ValidationError(UnleashedError):
    """Input validation errors."""
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        details = kwargs.pop("details", {})
        if field:
            details["field"] = field
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            recoverable=False,
            details=details,
            **kwargs
        )


# Resource Errors
class ResourceNotFoundError(UnleashedError):
    """Requested resource not found."""
    def __init__(self, message: str, resource_type: str, resource_id: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE,
            details={"resource_type": resource_type, "resource_id": resource_id},
            **kwargs
        )


# Dependency Errors
class DependencyError(UnleashedError):
    """External dependency failure."""
    def __init__(self, message: str, dependency: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DEPENDENCY,
            details={"dependency": dependency},
            **kwargs
        )


class SDKNotAvailableError(DependencyError):
    """SDK is not installed or available."""
    def __init__(self, sdk_name: str, install_command: Optional[str] = None):
        message = f"SDK '{sdk_name}' is not available"
        if install_command:
            message += f". Install with: {install_command}"
        super().__init__(message, dependency=sdk_name, recoverable=False)


# Configuration Errors
class ConfigurationError(UnleashedError):
    """Configuration-related errors."""
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        details = kwargs.pop("details", {})
        if config_key:
            details["config_key"] = config_key
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            recoverable=False,
            details=details,
            **kwargs
        )


# Data Errors
class DataError(UnleashedError):
    """Data format or parsing errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.DATA, **kwargs)


class ParseError(DataError):
    """Failed to parse data."""
    pass


class SerializationError(DataError):
    """Failed to serialize data."""
    pass


# ============================================================================
# ERROR CONTEXT AND AGGREGATION
# ============================================================================

@dataclass
class ErrorContext:
    """
    Context information for an error.

    Captures where and how an error occurred for debugging.
    """
    adapter_name: Optional[str] = None
    pipeline_name: Optional[str] = None
    operation: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorRecord:
    """
    Complete record of an error occurrence.

    Combines the error with its context for logging and analysis.
    """
    error: UnleashedError
    context: ErrorContext
    handled: bool = False
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "error": self.error.to_dict(),
            "context": {
                "adapter_name": self.context.adapter_name,
                "pipeline_name": self.context.pipeline_name,
                "operation": self.context.operation,
                "request_id": self.context.request_id,
                "timestamp": self.context.timestamp.isoformat(),
            },
            "handled": self.handled,
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful,
            "recovery_strategy": self.recovery_strategy,
        }


class ErrorAggregator:
    """
    Aggregates errors for analysis and reporting.

    Tracks error patterns and frequencies across the system.
    """

    def __init__(self, max_records: int = 1000):
        self.max_records = max_records
        self._records: List[ErrorRecord] = []
        self._category_counts: Dict[ErrorCategory, int] = {
            cat: 0 for cat in ErrorCategory
        }
        self._adapter_counts: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def record(self, error: UnleashedError, context: ErrorContext) -> ErrorRecord:
        """Record an error occurrence."""
        async with self._lock:
            record = ErrorRecord(error=error, context=context)
            self._records.append(record)

            # Update counts
            self._category_counts[error.category] += 1
            if context.adapter_name:
                self._adapter_counts[context.adapter_name] = (
                    self._adapter_counts.get(context.adapter_name, 0) + 1
                )

            # Trim if over limit
            if len(self._records) > self.max_records:
                self._records = self._records[-self.max_records:]

            return record

    async def get_recent(self, count: int = 100) -> List[ErrorRecord]:
        """Get most recent error records."""
        async with self._lock:
            return self._records[-count:]

    async def get_by_category(self, category: ErrorCategory) -> List[ErrorRecord]:
        """Get errors by category."""
        async with self._lock:
            return [r for r in self._records if r.error.category == category]

    async def get_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        async with self._lock:
            total = len(self._records)
            recoverable = sum(1 for r in self._records if r.error.recoverable)
            handled = sum(1 for r in self._records if r.handled)

            return {
                "total_errors": total,
                "recoverable_count": recoverable,
                "handled_count": handled,
                "recovery_rate": handled / total if total > 0 else 0.0,
                "by_category": {
                    cat.value: count
                    for cat, count in self._category_counts.items()
                    if count > 0
                },
                "by_adapter": dict(self._adapter_counts),
            }

    async def clear(self):
        """Clear all records."""
        async with self._lock:
            self._records.clear()
            self._category_counts = {cat: 0 for cat in ErrorCategory}
            self._adapter_counts.clear()


# ============================================================================
# RECOVERY STRATEGIES
# ============================================================================

class RecoveryStrategy(ABC):
    """Abstract base for error recovery strategies."""

    @abstractmethod
    async def can_recover(self, error: UnleashedError, context: ErrorContext) -> bool:
        """Check if this strategy can recover from the error."""
        pass

    @abstractmethod
    async def recover(
        self,
        error: UnleashedError,
        context: ErrorContext,
        retry_fn: Optional[Callable] = None,
    ) -> Optional[Any]:
        """Attempt to recover from the error."""
        pass


class RetryRecovery(RecoveryStrategy):
    """
    Recovery through retry with backoff.

    Suitable for transient errors like network issues.
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    async def can_recover(self, error: UnleashedError, context: ErrorContext) -> bool:
        """Can recover if error is recoverable and has retry function."""
        return error.recoverable and error.category in (
            ErrorCategory.NETWORK,
            ErrorCategory.TIMEOUT,
            ErrorCategory.RATE_LIMIT,
        )

    async def recover(
        self,
        error: UnleashedError,
        context: ErrorContext,
        retry_fn: Optional[Callable] = None,
    ) -> Optional[Any]:
        """Retry the operation with exponential backoff."""
        if not retry_fn:
            return None

        for attempt in range(self.max_retries):
            delay = min(
                self.initial_delay * (self.exponential_base ** attempt),
                self.max_delay,
            )

            # Respect retry_after if specified
            if error.retry_after:
                delay = max(delay, error.retry_after)

            await asyncio.sleep(delay)

            try:
                if inspect.iscoroutinefunction(retry_fn):
                    return await retry_fn()
                else:
                    return retry_fn()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"Retry {attempt + 1}/{self.max_retries} failed: {e}")

        return None


class FallbackRecovery(RecoveryStrategy):
    """
    Recovery through fallback to alternative implementation.

    Used when primary adapter/method is unavailable.
    """

    def __init__(self, fallback_adapters: Dict[str, Any]):
        """
        Initialize with fallback adapters.

        Args:
            fallback_adapters: Map of adapter_name -> fallback_adapter
        """
        self.fallback_adapters = fallback_adapters

    async def can_recover(self, error: UnleashedError, context: ErrorContext) -> bool:
        """Can recover if there's a fallback for the adapter."""
        if not context.adapter_name:
            return False
        return context.adapter_name in self.fallback_adapters

    async def recover(
        self,
        error: UnleashedError,
        context: ErrorContext,
        retry_fn: Optional[Callable] = None,
    ) -> Optional[Any]:
        """Use fallback adapter."""
        if not context.adapter_name:
            return None

        fallback = self.fallback_adapters.get(context.adapter_name)
        if not fallback:
            return None

        # Get the same operation on fallback
        if context.operation:
            method = getattr(fallback, context.operation, None)
            if method and context.input_data:
                if inspect.iscoroutinefunction(method):
                    return await method(**context.input_data)
                else:
                    return method(**context.input_data)

        return None


class CacheRecovery(RecoveryStrategy):
    """
    Recovery by returning cached results.

    Used when live data is unavailable.
    """

    def __init__(self, cache: Any):  # Accepts any cache backend
        self.cache = cache

    async def can_recover(self, error: UnleashedError, context: ErrorContext) -> bool:
        """Can recover if operation is cacheable."""
        return error.category in (
            ErrorCategory.NETWORK,
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.TIMEOUT,
        )

    async def recover(
        self,
        error: UnleashedError,
        context: ErrorContext,
        retry_fn: Optional[Callable] = None,
    ) -> Optional[Any]:
        """Return cached result if available."""
        if not context.operation or not context.input_data:
            return None

        # Generate cache key
        cache_key = f"{context.adapter_name}:{context.operation}:{hash(str(context.input_data))}"

        # Try to get from cache
        cached = await self.cache.get(cache_key)
        if cached:
            logger.info(f"Recovered from cache: {cache_key}")
            return cached

        return None


class GracefulDegradationRecovery(RecoveryStrategy):
    """
    Recovery through graceful degradation.

    Returns partial results or default values.
    """

    def __init__(self, defaults: Dict[str, Any]):
        """
        Initialize with default values.

        Args:
            defaults: Map of operation_name -> default_value
        """
        self.defaults = defaults

    async def can_recover(self, error: UnleashedError, context: ErrorContext) -> bool:
        """Can always attempt graceful degradation."""
        return context.operation in self.defaults

    async def recover(
        self,
        error: UnleashedError,
        context: ErrorContext,
        retry_fn: Optional[Callable] = None,
    ) -> Optional[Any]:
        """Return default value."""
        if context.operation:
            default = self.defaults.get(context.operation)
            if default is not None:
                logger.warning(
                    f"Graceful degradation: returning default for {context.operation}"
                )
                return default
        return None


# ============================================================================
# ERROR HANDLER
# ============================================================================

class ErrorHandler:
    """
    Central error handler for the platform.

    Coordinates error recording, recovery, and reporting.
    """

    def __init__(
        self,
        recovery_strategies: Optional[List[RecoveryStrategy]] = None,
        aggregator: Optional[ErrorAggregator] = None,
    ):
        """
        Initialize error handler.

        Args:
            recovery_strategies: Strategies to try in order
            aggregator: Error aggregator for tracking
        """
        self.recovery_strategies = recovery_strategies or [
            RetryRecovery(),
        ]
        self.aggregator = aggregator or ErrorAggregator()
        self._handlers: Dict[ErrorCategory, List[Callable]] = {
            cat: [] for cat in ErrorCategory
        }

    def register_handler(
        self,
        category: ErrorCategory,
        handler: Callable[[UnleashedError, ErrorContext], None],
    ):
        """Register a custom handler for an error category."""
        self._handlers[category].append(handler)

    async def handle(
        self,
        error: UnleashedError,
        context: Optional[ErrorContext] = None,
        retry_fn: Optional[Callable] = None,
    ) -> Optional[Any]:
        """
        Handle an error with recovery attempts.

        Args:
            error: The error to handle
            context: Error context
            retry_fn: Function to retry the failed operation

        Returns:
            Recovery result if successful, None otherwise
        """
        context = context or ErrorContext()

        # Record the error
        record = await self.aggregator.record(error, context)

        # Log based on severity
        log_message = f"{error} | Context: {context.adapter_name}/{context.operation}"
        if error.severity == ErrorSeverity.DEBUG:
            logger.debug(log_message)
        elif error.severity == ErrorSeverity.INFO:
            logger.info(log_message)
        elif error.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        elif error.severity == ErrorSeverity.ERROR:
            logger.error(log_message)
        else:
            logger.critical(log_message)

        # Call custom handlers
        for handler in self._handlers.get(error.category, []):
            try:
                await handler(error, context) if inspect.iscoroutinefunction(handler) else handler(error, context)
            except Exception as e:
                logger.error(f"Custom handler failed: {e}")

        # Attempt recovery
        if error.recoverable:
            for strategy in self.recovery_strategies:
                try:
                    if await strategy.can_recover(error, context):
                        record.recovery_attempted = True
                        record.recovery_strategy = strategy.__class__.__name__

                        result = await strategy.recover(error, context, retry_fn)
                        if result is not None:
                            record.recovery_successful = True
                            record.handled = True
                            return result
                except Exception as e:
                    logger.error(f"Recovery strategy {strategy.__class__.__name__} failed: {e}")

        return None

    async def wrap(
        self,
        fn: Callable[..., T],
        context: Optional[ErrorContext] = None,
        *args,
        **kwargs,
    ) -> T:
        """
        Wrap a function with error handling.

        Args:
            fn: Function to wrap
            context: Error context
            *args, **kwargs: Arguments to pass to function

        Returns:
            Function result

        Raises:
            UnleashedError: If error cannot be recovered
        """
        context = context or ErrorContext()

        try:
            if inspect.iscoroutinefunction(fn):
                return await fn(*args, **kwargs)
            else:
                return fn(*args, **kwargs)
        except UnleashedError as e:
            result = await self.handle(e, context, lambda: fn(*args, **kwargs))
            if result is not None:
                return result
            raise
        except Exception as e:
            # Convert to UnleashedError
            unleashed_error = UnleashedError(
                message=str(e),
                category=self._categorize_exception(e),
                cause=e,
            )
            result = await self.handle(unleashed_error, context, lambda: fn(*args, **kwargs))
            if result is not None:
                return result
            raise unleashed_error from e

    def _categorize_exception(self, e: Exception) -> ErrorCategory:
        """Categorize a standard exception."""
        error_type = type(e).__name__.lower()

        if any(x in error_type for x in ["connection", "network", "socket"]):
            return ErrorCategory.NETWORK
        elif any(x in error_type for x in ["timeout"]):
            return ErrorCategory.TIMEOUT
        elif any(x in error_type for x in ["auth", "credential", "permission"]):
            return ErrorCategory.AUTHENTICATION
        elif any(x in error_type for x in ["validation", "invalid"]):
            return ErrorCategory.VALIDATION
        elif any(x in error_type for x in ["notfound", "missing"]):
            return ErrorCategory.RESOURCE
        elif any(x in error_type for x in ["parse", "decode", "encode"]):
            return ErrorCategory.DATA

        return ErrorCategory.UNKNOWN

    async def get_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return await self.aggregator.get_statistics()


# ============================================================================
# DECORATORS AND UTILITIES
# ============================================================================

def handle_errors(
    handler: Optional[ErrorHandler] = None,
    adapter_name: Optional[str] = None,
    operation: Optional[str] = None,
):
    """
    Decorator to add error handling to a function.

    Usage:
        @handle_errors(adapter_name="dspy", operation="optimize")
        async def my_function():
            ...
    """
    _handler = handler or ErrorHandler()

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        async def async_wrapper(*args, **kwargs) -> T:
            context = ErrorContext(
                adapter_name=adapter_name,
                operation=operation or fn.__name__,
                input_data={"args": str(args)[:100], "kwargs": list(kwargs.keys())},
            )
            return await _handler.wrap(fn, context, *args, **kwargs)

        @wraps(fn)
        def sync_wrapper(*args, **kwargs) -> T:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                # Convert and re-raise
                raise UnleashedError(str(e), cause=e)

        if inspect.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    return decorator


def get_error_handler(**kwargs) -> ErrorHandler:
    """Get configured error handler."""
    return ErrorHandler(**kwargs)


def create_context(
    adapter_name: Optional[str] = None,
    pipeline_name: Optional[str] = None,
    operation: Optional[str] = None,
    **kwargs,
) -> ErrorContext:
    """Create an error context."""
    return ErrorContext(
        adapter_name=adapter_name,
        pipeline_name=pipeline_name,
        operation=operation,
        additional_context=kwargs,
    )
