# -*- coding: utf-8 -*-
"""
Unleash Platform Utilities

Cross-platform utilities for encoding, logging, error handling, and common operations.
"""

from .encoding import (
    configure_encoding,
    safe_print,
    safe_str,
    status,
    ok,
    fail,
    warn,
    info,
    STATUS,
    is_windows,
    supports_unicode,
    configure_safe_logging,
)

from .logging_config import (
    configure_logging,
    get_logger,
    log_operation,
    log_error,
    log_context,
    set_context,
    clear_context,
    get_context,
    LogContext,
)

from .errors import (
    # Base classes
    UnleashError,
    ErrorDetails,
    ErrorCategory,
    ErrorSeverity,
    # Network errors
    NetworkError,
    TimeoutError,
    ConnectionError,
    # Rate limiting
    RateLimitError,
    QuotaExceededError,
    # Auth errors
    AuthenticationError,
    AuthorizationError,
    InvalidAPIKeyError,
    # Validation errors
    ValidationError,
    NotFoundError,
    ConfigurationError,
    # Service errors
    ServiceUnavailableError,
    DependencyError,
    # SDK errors
    SDKError,
    ExaError,
    FirecrawlError,
    GraphitiError,
    LettaError,
    ClaudeError,
    # Utilities
    wrap_exception,
    handle_errors,
    ErrorAggregator,
    RETRIABLE_EXCEPTIONS,
    NON_RETRIABLE_EXCEPTIONS,
    get_retriable_exceptions,
    get_non_retriable_exceptions,
)

# Auto-configure encoding on import
configure_encoding()

__all__ = [
    # Encoding
    "configure_encoding",
    "safe_print",
    "safe_str",
    "status",
    "ok",
    "fail",
    "warn",
    "info",
    "STATUS",
    "is_windows",
    "supports_unicode",
    "configure_safe_logging",
    # Logging
    "configure_logging",
    "get_logger",
    "log_operation",
    "log_error",
    "log_context",
    "set_context",
    "clear_context",
    "get_context",
    "LogContext",
    # Errors - Base
    "UnleashError",
    "ErrorDetails",
    "ErrorCategory",
    "ErrorSeverity",
    # Errors - Network
    "NetworkError",
    "TimeoutError",
    "ConnectionError",
    # Errors - Rate Limiting
    "RateLimitError",
    "QuotaExceededError",
    # Errors - Auth
    "AuthenticationError",
    "AuthorizationError",
    "InvalidAPIKeyError",
    # Errors - Validation
    "ValidationError",
    "NotFoundError",
    "ConfigurationError",
    # Errors - Service
    "ServiceUnavailableError",
    "DependencyError",
    # Errors - SDK
    "SDKError",
    "ExaError",
    "FirecrawlError",
    "GraphitiError",
    "LettaError",
    "ClaudeError",
    # Errors - Utilities
    "wrap_exception",
    "handle_errors",
    "ErrorAggregator",
    "RETRIABLE_EXCEPTIONS",
    "NON_RETRIABLE_EXCEPTIONS",
    "get_retriable_exceptions",
    "get_non_retriable_exceptions",
]
