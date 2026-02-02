# -*- coding: utf-8 -*-
"""
Unified Error Handling for Unleash Platform

Provides a comprehensive error handling system with:
- Hierarchical exception classes for different error types
- Error categorization (transient vs permanent)
- Structured error responses for API/logging
- SDK-specific error handling
- Integration with resilience patterns

Based on:
- RFC 7807 (Problem Details for HTTP APIs)
- Google Cloud error model
- AWS error handling best practices
"""

import logging
import traceback
import time
from enum import Enum
from typing import Any, Dict, Optional, Type, Set, Callable
from dataclasses import dataclass, field
from functools import wraps

logger = logging.getLogger(__name__)


# =============================================================================
# Error Categories
# =============================================================================

class ErrorCategory(str, Enum):
    """Categories for error classification."""
    # Transient (retriable) errors
    NETWORK = "network"           # Network connectivity issues
    TIMEOUT = "timeout"           # Operation timed out
    RATE_LIMIT = "rate_limit"     # Rate limiting / throttling
    SERVICE_UNAVAILABLE = "service_unavailable"  # Temporary service issue
    RESOURCE_EXHAUSTED = "resource_exhausted"    # Quota/resource limits

    # Permanent (non-retriable) errors
    AUTHENTICATION = "authentication"   # Auth failures
    AUTHORIZATION = "authorization"     # Permission denied
    VALIDATION = "validation"           # Invalid input
    NOT_FOUND = "not_found"             # Resource doesn't exist
    CONFIGURATION = "configuration"     # Bad configuration

    # Internal errors
    INTERNAL = "internal"         # Internal system error
    DEPENDENCY = "dependency"     # External dependency failure
    DATA = "data"                 # Data corruption/parsing


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""
    DEBUG = "debug"       # Debugging information
    INFO = "info"         # Informational
    WARNING = "warning"   # Potential issues
    ERROR = "error"       # Error but recoverable
    CRITICAL = "critical" # Critical failure


# Transient error categories that should be retried
TRANSIENT_CATEGORIES: Set[ErrorCategory] = {
    ErrorCategory.NETWORK,
    ErrorCategory.TIMEOUT,
    ErrorCategory.RATE_LIMIT,
    ErrorCategory.SERVICE_UNAVAILABLE,
    ErrorCategory.RESOURCE_EXHAUSTED,
}


# =============================================================================
# Structured Error Response
# =============================================================================

@dataclass
class ErrorDetails:
    """
    Structured error details following RFC 7807 pattern.

    Provides consistent error information for logging, APIs, and debugging.
    """
    # Core fields
    error_code: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity = ErrorSeverity.ERROR

    # Context
    source: Optional[str] = None        # SDK or component that raised error
    operation: Optional[str] = None     # Operation being performed
    request_id: Optional[str] = None    # Request/correlation ID

    # Technical details
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    traceback_str: Optional[str] = None

    # Metadata
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Retry guidance
    is_retriable: bool = False
    retry_after_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "is_retriable": self.is_retriable,
            "timestamp": self.timestamp,
        }

        if self.source:
            result["source"] = self.source
        if self.operation:
            result["operation"] = self.operation
        if self.request_id:
            result["request_id"] = self.request_id
        if self.exception_type:
            result["exception_type"] = self.exception_type
        if self.exception_message:
            result["exception_message"] = self.exception_message
        if self.retry_after_seconds is not None:
            result["retry_after_seconds"] = self.retry_after_seconds
        if self.metadata:
            result["metadata"] = self.metadata

        return result

    def to_log_dict(self) -> Dict[str, Any]:
        """Convert to dictionary optimized for structured logging."""
        result = self.to_dict()
        if self.traceback_str:
            result["traceback"] = self.traceback_str
        return result


# =============================================================================
# Base Exception Classes
# =============================================================================

class UnleashError(Exception):
    """
    Base exception for all Unleash platform errors.

    Provides structured error details and categorization.
    """

    default_code: str = "UNLEASH_ERROR"
    default_category: ErrorCategory = ErrorCategory.INTERNAL
    default_severity: ErrorSeverity = ErrorSeverity.ERROR
    default_retriable: bool = False

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        source: Optional[str] = None,
        operation: Optional[str] = None,
        request_id: Optional[str] = None,
        cause: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None,
        retry_after: Optional[float] = None,
    ):
        """
        Initialize error with structured details.

        Args:
            message: Human-readable error message
            code: Machine-readable error code
            category: Error category for classification
            severity: Error severity level
            source: Component/SDK that raised the error
            operation: Operation being performed
            request_id: Request/correlation ID
            cause: Original exception that caused this error
            metadata: Additional context data
            retry_after: Suggested retry delay in seconds
        """
        super().__init__(message)

        self.code = code or self.default_code
        self.category = category or self.default_category
        self.severity = severity or self.default_severity
        self.source = source
        self.operation = operation
        self.request_id = request_id
        self.cause = cause
        self.metadata = metadata or {}
        self.retry_after = retry_after

        # Capture traceback
        if cause:
            self._traceback = "".join(traceback.format_exception(
                type(cause), cause, cause.__traceback__
            ))
        else:
            self._traceback = None

    @property
    def is_retriable(self) -> bool:
        """Check if this error is retriable."""
        if self.category in TRANSIENT_CATEGORIES:
            return True
        return self.default_retriable

    def get_details(self) -> ErrorDetails:
        """Get structured error details."""
        return ErrorDetails(
            error_code=self.code,
            message=str(self),
            category=self.category,
            severity=self.severity,
            source=self.source,
            operation=self.operation,
            request_id=self.request_id,
            exception_type=type(self.cause).__name__ if self.cause else type(self).__name__,
            exception_message=str(self.cause) if self.cause else str(self),
            traceback_str=self._traceback,
            is_retriable=self.is_retriable,
            retry_after_seconds=self.retry_after,
            metadata=self.metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.get_details().to_dict()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, message={str(self)!r})"


# =============================================================================
# Network & Connectivity Errors
# =============================================================================

class NetworkError(UnleashError):
    """Network connectivity or communication error."""
    default_code = "NETWORK_ERROR"
    default_category = ErrorCategory.NETWORK
    default_retriable = True


class TimeoutError(UnleashError):
    """Operation timed out."""
    default_code = "TIMEOUT_ERROR"
    default_category = ErrorCategory.TIMEOUT
    default_retriable = True


class ConnectionError(NetworkError):
    """Failed to establish connection."""
    default_code = "CONNECTION_ERROR"


# =============================================================================
# Rate Limiting & Quota Errors
# =============================================================================

class RateLimitError(UnleashError):
    """Rate limit exceeded."""
    default_code = "RATE_LIMIT_ERROR"
    default_category = ErrorCategory.RATE_LIMIT
    default_retriable = True

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        retry_after: Optional[float] = None,
        limit: Optional[int] = None,
        remaining: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, retry_after=retry_after, **kwargs)
        self.metadata["limit"] = limit
        self.metadata["remaining"] = remaining


class QuotaExceededError(UnleashError):
    """Quota or resource limit exceeded."""
    default_code = "QUOTA_EXCEEDED"
    default_category = ErrorCategory.RESOURCE_EXHAUSTED
    default_retriable = True


# =============================================================================
# Authentication & Authorization Errors
# =============================================================================

class AuthenticationError(UnleashError):
    """Authentication failed."""
    default_code = "AUTH_ERROR"
    default_category = ErrorCategory.AUTHENTICATION
    default_severity = ErrorSeverity.WARNING
    default_retriable = False


class AuthorizationError(UnleashError):
    """Permission denied."""
    default_code = "FORBIDDEN"
    default_category = ErrorCategory.AUTHORIZATION
    default_severity = ErrorSeverity.WARNING
    default_retriable = False


class InvalidAPIKeyError(AuthenticationError):
    """Invalid or expired API key."""
    default_code = "INVALID_API_KEY"


# =============================================================================
# Validation & Input Errors
# =============================================================================

class ValidationError(UnleashError):
    """Input validation failed."""
    default_code = "VALIDATION_ERROR"
    default_category = ErrorCategory.VALIDATION
    default_severity = ErrorSeverity.WARNING
    default_retriable = False

    def __init__(
        self,
        message: str,
        *,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        if field:
            self.metadata["field"] = field
        if value is not None:
            self.metadata["value"] = str(value)[:100]  # Truncate long values
        if constraints:
            self.metadata["constraints"] = constraints


class NotFoundError(UnleashError):
    """Resource not found."""
    default_code = "NOT_FOUND"
    default_category = ErrorCategory.NOT_FOUND
    default_severity = ErrorSeverity.WARNING
    default_retriable = False


class ConfigurationError(UnleashError):
    """Configuration error."""
    default_code = "CONFIG_ERROR"
    default_category = ErrorCategory.CONFIGURATION
    default_severity = ErrorSeverity.ERROR
    default_retriable = False


# =============================================================================
# Service & Dependency Errors
# =============================================================================

class ServiceUnavailableError(UnleashError):
    """Service temporarily unavailable."""
    default_code = "SERVICE_UNAVAILABLE"
    default_category = ErrorCategory.SERVICE_UNAVAILABLE
    default_retriable = True


class DependencyError(UnleashError):
    """External dependency failure."""
    default_code = "DEPENDENCY_ERROR"
    default_category = ErrorCategory.DEPENDENCY
    default_retriable = True


# =============================================================================
# SDK-Specific Errors
# =============================================================================

class SDKError(UnleashError):
    """Base class for SDK-specific errors."""
    default_code = "SDK_ERROR"

    def __init__(self, message: str, *, sdk_name: str, **kwargs):
        super().__init__(message, source=sdk_name, **kwargs)


class ExaError(SDKError):
    """Exa SDK error."""
    default_code = "EXA_ERROR"

    def __init__(self, message: str, **kwargs):
        super().__init__(message, sdk_name="exa", **kwargs)


class FirecrawlError(SDKError):
    """Firecrawl SDK error."""
    default_code = "FIRECRAWL_ERROR"

    def __init__(self, message: str, **kwargs):
        super().__init__(message, sdk_name="firecrawl", **kwargs)


class GraphitiError(SDKError):
    """Graphiti SDK error."""
    default_code = "GRAPHITI_ERROR"

    def __init__(self, message: str, **kwargs):
        super().__init__(message, sdk_name="graphiti", **kwargs)


class LettaError(SDKError):
    """Letta SDK error."""
    default_code = "LETTA_ERROR"

    def __init__(self, message: str, **kwargs):
        super().__init__(message, sdk_name="letta", **kwargs)


class ClaudeError(SDKError):
    """Claude/Anthropic SDK error."""
    default_code = "CLAUDE_ERROR"

    def __init__(self, message: str, **kwargs):
        super().__init__(message, sdk_name="claude", **kwargs)


# =============================================================================
# Error Mapping & Conversion
# =============================================================================

# Map HTTP status codes to error categories
HTTP_STATUS_MAPPING: Dict[int, ErrorCategory] = {
    400: ErrorCategory.VALIDATION,
    401: ErrorCategory.AUTHENTICATION,
    403: ErrorCategory.AUTHORIZATION,
    404: ErrorCategory.NOT_FOUND,
    408: ErrorCategory.TIMEOUT,
    429: ErrorCategory.RATE_LIMIT,
    500: ErrorCategory.INTERNAL,
    502: ErrorCategory.DEPENDENCY,
    503: ErrorCategory.SERVICE_UNAVAILABLE,
    504: ErrorCategory.TIMEOUT,
}


def categorize_http_error(status_code: int) -> ErrorCategory:
    """Map HTTP status code to error category."""
    return HTTP_STATUS_MAPPING.get(status_code, ErrorCategory.INTERNAL)


def is_http_retriable(status_code: int) -> bool:
    """Check if HTTP status code indicates a retriable error."""
    category = categorize_http_error(status_code)
    return category in TRANSIENT_CATEGORIES


# Map common exception types to UnleashError subclasses
EXCEPTION_MAPPING: Dict[Type[Exception], Type[UnleashError]] = {
    ConnectionRefusedError: ConnectionError,
    TimeoutError: TimeoutError,
    PermissionError: AuthorizationError,
    FileNotFoundError: NotFoundError,
    ValueError: ValidationError,
    KeyError: NotFoundError,
}


def wrap_exception(
    exc: Exception,
    *,
    source: Optional[str] = None,
    operation: Optional[str] = None,
    default_class: Type[UnleashError] = UnleashError,
) -> UnleashError:
    """
    Wrap a standard exception in an UnleashError.

    Args:
        exc: Exception to wrap
        source: Source component/SDK
        operation: Operation being performed
        default_class: Default UnleashError class if no mapping found

    Returns:
        Wrapped UnleashError
    """
    # Already an UnleashError
    if isinstance(exc, UnleashError):
        return exc

    # Find mapped class
    error_class = default_class
    for exc_type, mapped_class in EXCEPTION_MAPPING.items():
        if isinstance(exc, exc_type):
            error_class = mapped_class
            break

    return error_class(
        str(exc),
        source=source,
        operation=operation,
        cause=exc,
    )


# =============================================================================
# Error Handler Decorator
# =============================================================================

def handle_errors(
    *,
    source: Optional[str] = None,
    operation: Optional[str] = None,
    reraise: bool = True,
    log_errors: bool = True,
    default_return: Any = None,
) -> Callable:
    """
    Decorator to handle and transform errors consistently.

    Args:
        source: Source component for error context
        operation: Operation name for error context
        reraise: Whether to reraise wrapped errors
        log_errors: Whether to log errors
        default_return: Value to return if not reraising

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__name__

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except UnleashError:
                if log_errors:
                    logger.exception(f"Error in {op_name}")
                if reraise:
                    raise
                return default_return
            except Exception as e:
                wrapped = wrap_exception(e, source=source, operation=op_name)
                if log_errors:
                    logger.exception(f"Error in {op_name}: {wrapped}")
                if reraise:
                    raise wrapped from e
                return default_return

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except UnleashError:
                if log_errors:
                    logger.exception(f"Error in {op_name}")
                if reraise:
                    raise
                return default_return
            except Exception as e:
                wrapped = wrap_exception(e, source=source, operation=op_name)
                if log_errors:
                    logger.exception(f"Error in {op_name}: {wrapped}")
                if reraise:
                    raise wrapped from e
                return default_return

        import asyncio
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# =============================================================================
# Retriable Exception Sets for Resilience Integration
# =============================================================================

# All retriable exception types for use with RetryPolicy
RETRIABLE_EXCEPTIONS: Set[Type[Exception]] = {
    NetworkError,
    TimeoutError,
    ConnectionError,
    RateLimitError,
    QuotaExceededError,
    ServiceUnavailableError,
    DependencyError,
}

# Non-retriable exceptions that should fail immediately
NON_RETRIABLE_EXCEPTIONS: Set[Type[Exception]] = {
    AuthenticationError,
    AuthorizationError,
    ValidationError,
    NotFoundError,
    ConfigurationError,
    InvalidAPIKeyError,
}


def get_retriable_exceptions() -> Set[Type[Exception]]:
    """Get set of retriable exception types for retry policies."""
    return RETRIABLE_EXCEPTIONS.copy()


def get_non_retriable_exceptions() -> Set[Type[Exception]]:
    """Get set of non-retriable exception types."""
    return NON_RETRIABLE_EXCEPTIONS.copy()


# =============================================================================
# Error Aggregation
# =============================================================================

@dataclass
class ErrorAggregator:
    """
    Aggregates multiple errors for batch operations.

    Useful when processing multiple items and collecting all errors
    rather than failing on first error.
    """
    errors: list = field(default_factory=list)
    successes: int = 0

    def add_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Add an error to the aggregation."""
        if isinstance(error, UnleashError):
            details = error.get_details()
        else:
            wrapped = wrap_exception(error)
            details = wrapped.get_details()

        if context:
            details.metadata.update(context)

        self.errors.append(details)

    def add_success(self):
        """Record a successful operation."""
        self.successes += 1

    @property
    def has_errors(self) -> bool:
        """Check if any errors were recorded."""
        return len(self.errors) > 0

    @property
    def total(self) -> int:
        """Total operations (successes + errors)."""
        return self.successes + len(self.errors)

    @property
    def error_rate(self) -> float:
        """Error rate as a fraction."""
        if self.total == 0:
            return 0.0
        return len(self.errors) / self.total

    def get_summary(self) -> Dict[str, Any]:
        """Get error summary."""
        return {
            "total_operations": self.total,
            "successes": self.successes,
            "errors": len(self.errors),
            "error_rate": self.error_rate,
            "error_categories": self._count_by_category(),
            "errors_detail": [e.to_dict() for e in self.errors],
        }

    def _count_by_category(self) -> Dict[str, int]:
        """Count errors by category."""
        counts: Dict[str, int] = {}
        for error in self.errors:
            cat = error.category.value
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def raise_if_errors(self, threshold: float = 0.0):
        """
        Raise an exception if error rate exceeds threshold.

        Args:
            threshold: Maximum acceptable error rate (0.0 = any errors, 1.0 = all errors)
        """
        if self.error_rate > threshold:
            summary = self.get_summary()
            raise UnleashError(
                f"Batch operation failed: {len(self.errors)}/{self.total} errors",
                code="BATCH_ERROR",
                metadata=summary,
            )
