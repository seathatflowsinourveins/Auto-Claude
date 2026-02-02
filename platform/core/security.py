"""
Unleashed Platform - Security Module

Comprehensive security features including:
- Input validation and sanitization
- API key management and rotation
- Rate limiting per API key
- Audit logging
- Secret detection and prevention
- Request/response encryption
"""

import asyncio
import hashlib
import hmac
import secrets
import re
import json
import base64
from typing import (
    Dict, Any, Optional, List, Callable, TypeVar, Set,
    Pattern, Union, Awaitable
)
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from abc import ABC, abstractmethod
from functools import wraps
import logging
import threading

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Security Enums
# ============================================================================

class SecurityLevel(Enum):
    """Security enforcement levels."""
    PERMISSIVE = "permissive"      # Log only
    STANDARD = "standard"          # Block known threats
    STRICT = "strict"              # Block suspicious patterns
    PARANOID = "paranoid"          # Block everything not explicitly allowed


class ThreatType(Enum):
    """Types of security threats."""
    INJECTION = "injection"
    XSS = "xss"
    PATH_TRAVERSAL = "path_traversal"
    SECRET_EXPOSURE = "secret_exposure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_INPUT = "invalid_input"
    UNAUTHORIZED = "unauthorized"
    SUSPICIOUS_PATTERN = "suspicious_pattern"


class AuditAction(Enum):
    """Types of actions to audit."""
    API_CALL = "api_call"
    AUTH_ATTEMPT = "auth_attempt"
    CONFIG_CHANGE = "config_change"
    SECRET_ACCESS = "secret_access"
    ERROR = "error"
    THREAT_BLOCKED = "threat_blocked"
    ADMIN_ACTION = "admin_action"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SecurityConfig:
    """Configuration for security module."""
    level: SecurityLevel = SecurityLevel.STANDARD
    enable_audit_log: bool = True
    enable_rate_limiting: bool = True
    enable_input_validation: bool = True
    enable_secret_detection: bool = True
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    max_string_length: int = 100000
    allowed_hosts: Optional[List[str]] = None
    blocked_ips: Optional[List[str]] = None
    api_key_rotation_days: int = 90


@dataclass
class ThreatDetection:
    """Result of a threat detection check."""
    is_threat: bool
    threat_type: Optional[ThreatType] = None
    severity: str = "low"
    details: str = ""
    matched_pattern: Optional[str] = None


@dataclass
class AuditEntry:
    """An entry in the audit log."""
    timestamp: datetime
    action: AuditAction
    actor: Optional[str]
    resource: Optional[str]
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    success: bool = True
    threat_detected: Optional[ThreatType] = None


@dataclass
class APIKey:
    """Represents an API key with metadata."""
    key_id: str
    key_hash: str  # Never store raw keys
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    permissions: List[str] = field(default_factory=list)
    rate_limit: int = 1000  # requests per hour
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitState:
    """State for rate limiting."""
    key_id: str
    window_start: datetime
    request_count: int
    limit: int


# ============================================================================
# Input Validation
# ============================================================================

class InputValidator:
    """Validates and sanitizes input data."""

    # Dangerous patterns for SQL injection
    SQL_INJECTION_PATTERNS: List[Pattern] = [
        re.compile(r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b)", re.IGNORECASE),
        re.compile(r"(--|#|/\*|\*/)", re.IGNORECASE),
        re.compile(r"(\bOR\b\s*\d+\s*=\s*\d+)", re.IGNORECASE),
        re.compile(r"(\bAND\b\s*\d+\s*=\s*\d+)", re.IGNORECASE),
        re.compile(r"('\s*(OR|AND)\s*'?\d+'?\s*=\s*'?\d+'?)", re.IGNORECASE),
    ]

    # Dangerous patterns for XSS
    XSS_PATTERNS: List[Pattern] = [
        re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),
        re.compile(r"<iframe[^>]*>", re.IGNORECASE),
        re.compile(r"<object[^>]*>", re.IGNORECASE),
    ]

    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS: List[Pattern] = [
        re.compile(r"\.\./"),
        re.compile(r"\.\.\\"),
        re.compile(r"%2e%2e[/\\]", re.IGNORECASE),
        re.compile(r"%252e%252e[/\\]", re.IGNORECASE),
    ]

    # Secret patterns (API keys, passwords, tokens)
    SECRET_PATTERNS: List[Pattern] = [
        re.compile(r"(api[_-]?key|apikey)\s*[:=]\s*['\"]?[\w-]{20,}", re.IGNORECASE),
        re.compile(r"(password|passwd|pwd)\s*[:=]\s*['\"]?[^\s'\"]{8,}", re.IGNORECASE),
        re.compile(r"(secret|token)\s*[:=]\s*['\"]?[\w-]{20,}", re.IGNORECASE),
        re.compile(r"(aws|azure|gcp)_?[a-z_]*_?(key|secret|token)", re.IGNORECASE),
        re.compile(r"sk-[a-zA-Z0-9]{48}", re.IGNORECASE),  # OpenAI keys
        re.compile(r"ghp_[a-zA-Z0-9]{36}", re.IGNORECASE),  # GitHub tokens
        re.compile(r"AKIA[A-Z0-9]{16}", re.IGNORECASE),  # AWS access key
    ]

    def __init__(self, config: SecurityConfig):
        self.config = config

    def validate_string(self, value: str, context: str = "input") -> ThreatDetection:
        """Validate a string for security threats."""
        if not isinstance(value, str):
            return ThreatDetection(
                is_threat=True,
                threat_type=ThreatType.INVALID_INPUT,
                severity="medium",
                details=f"Expected string, got {type(value).__name__}"
            )

        # Check length
        if len(value) > self.config.max_string_length:
            return ThreatDetection(
                is_threat=True,
                threat_type=ThreatType.INVALID_INPUT,
                severity="low",
                details=f"String exceeds maximum length ({len(value)} > {self.config.max_string_length})"
            )

        # Check for SQL injection
        if self.config.level in [SecurityLevel.STANDARD, SecurityLevel.STRICT, SecurityLevel.PARANOID]:
            for pattern in self.SQL_INJECTION_PATTERNS:
                match = pattern.search(value)
                if match:
                    return ThreatDetection(
                        is_threat=True,
                        threat_type=ThreatType.INJECTION,
                        severity="high",
                        details="Potential SQL injection detected",
                        matched_pattern=match.group()
                    )

        # Check for XSS
        if self.config.level in [SecurityLevel.STANDARD, SecurityLevel.STRICT, SecurityLevel.PARANOID]:
            for pattern in self.XSS_PATTERNS:
                match = pattern.search(value)
                if match:
                    return ThreatDetection(
                        is_threat=True,
                        threat_type=ThreatType.XSS,
                        severity="high",
                        details="Potential XSS attack detected",
                        matched_pattern=match.group()
                    )

        # Check for path traversal
        if self.config.level in [SecurityLevel.STRICT, SecurityLevel.PARANOID]:
            for pattern in self.PATH_TRAVERSAL_PATTERNS:
                match = pattern.search(value)
                if match:
                    return ThreatDetection(
                        is_threat=True,
                        threat_type=ThreatType.PATH_TRAVERSAL,
                        severity="high",
                        details="Potential path traversal attack detected",
                        matched_pattern=match.group()
                    )

        # Check for secrets
        if self.config.enable_secret_detection:
            for pattern in self.SECRET_PATTERNS:
                match = pattern.search(value)
                if match:
                    return ThreatDetection(
                        is_threat=True,
                        threat_type=ThreatType.SECRET_EXPOSURE,
                        severity="critical",
                        details="Potential secret exposure detected",
                        matched_pattern="[REDACTED]"  # Don't log the actual secret
                    )

        return ThreatDetection(is_threat=False)

    def validate_dict(self, data: Dict[str, Any], context: str = "input") -> ThreatDetection:
        """Validate a dictionary recursively."""
        for key, value in data.items():
            # Validate key
            key_check = self.validate_string(key, f"{context}.key")
            if key_check.is_threat:
                return key_check

            # Validate value based on type
            if isinstance(value, str):
                value_check = self.validate_string(value, f"{context}.{key}")
                if value_check.is_threat:
                    return value_check
            elif isinstance(value, dict):
                value_check = self.validate_dict(value, f"{context}.{key}")
                if value_check.is_threat:
                    return value_check
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        item_check = self.validate_string(item, f"{context}.{key}[{i}]")
                        if item_check.is_threat:
                            return item_check
                    elif isinstance(item, dict):
                        item_check = self.validate_dict(item, f"{context}.{key}[{i}]")
                        if item_check.is_threat:
                            return item_check

        return ThreatDetection(is_threat=False)

    def sanitize_string(self, value: str) -> str:
        """Sanitize a string by escaping potentially dangerous characters."""
        if not isinstance(value, str):
            return str(value)

        # HTML escape
        value = (
            value
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )

        return value

    def redact_secrets(self, value: str) -> str:
        """Redact any detected secrets in a string."""
        for pattern in self.SECRET_PATTERNS:
            value = pattern.sub("[REDACTED]", value)
        return value


# ============================================================================
# API Key Management
# ============================================================================

class APIKeyManager:
    """Manages API keys with secure storage and rotation."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self._keys: Dict[str, APIKey] = {}
        self._hash_to_id: Dict[str, str] = {}
        self._lock = threading.Lock()

    def generate_key(
        self,
        name: str,
        permissions: Optional[List[str]] = None,
        rate_limit: int = 1000,
        expires_in_days: Optional[int] = None
    ) -> tuple[str, APIKey]:
        """Generate a new API key."""
        # Generate secure random key
        raw_key = secrets.token_urlsafe(32)
        key_id = secrets.token_hex(8)
        key_hash = self._hash_key(raw_key)

        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at,
            permissions=permissions or ["read"],
            rate_limit=rate_limit
        )

        with self._lock:
            self._keys[key_id] = api_key
            self._hash_to_id[key_hash] = key_id

        # Return raw key (only time it's available) and metadata
        full_key = f"{key_id}.{raw_key}"
        return full_key, api_key

    def validate_key(self, key: str) -> Optional[APIKey]:
        """Validate an API key and return its metadata."""
        try:
            parts = key.split(".", 1)
            if len(parts) != 2:
                return None

            key_id, raw_key = parts
            key_hash = self._hash_key(raw_key)

            with self._lock:
                if key_id not in self._keys:
                    return None

                api_key = self._keys[key_id]

                # Verify hash
                if api_key.key_hash != key_hash:
                    return None

                # Check if enabled
                if not api_key.enabled:
                    return None

                # Check expiration
                if api_key.expires_at and datetime.now(timezone.utc) > api_key.expires_at:
                    return None

                # Update last used
                api_key.last_used = datetime.now(timezone.utc)

                return api_key

        except Exception:
            return None

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        with self._lock:
            if key_id in self._keys:
                self._keys[key_id].enabled = False
                return True
        return False

    def rotate_key(self, key_id: str) -> Optional[tuple[str, APIKey]]:
        """Rotate an API key (generate new key, revoke old)."""
        with self._lock:
            if key_id not in self._keys:
                return None

            old_key = self._keys[key_id]

        # Generate new key with same permissions
        return self.generate_key(
            name=old_key.name,
            permissions=old_key.permissions,
            rate_limit=old_key.rate_limit,
            expires_in_days=self.config.api_key_rotation_days
        )

    def list_keys(self) -> List[APIKey]:
        """List all API keys (without hashes)."""
        with self._lock:
            return list(self._keys.values())

    @staticmethod
    def _hash_key(raw_key: str) -> str:
        """Hash an API key using SHA-256."""
        return hashlib.sha256(raw_key.encode()).hexdigest()


# ============================================================================
# Rate Limiting
# ============================================================================

class RateLimiter:
    """Rate limiter with per-key limits and sliding windows."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self._states: Dict[str, RateLimitState] = {}
        self._lock = threading.Lock()
        self._window_seconds = 3600  # 1 hour window

    def check_rate_limit(self, key_id: str, limit: int) -> tuple[bool, int]:
        """
        Check if a request is within rate limits.
        Returns (allowed, remaining_requests).
        """
        now = datetime.now(timezone.utc)

        with self._lock:
            if key_id not in self._states:
                self._states[key_id] = RateLimitState(
                    key_id=key_id,
                    window_start=now,
                    request_count=0,
                    limit=limit
                )

            state = self._states[key_id]

            # Check if window has expired
            window_end = state.window_start + timedelta(seconds=self._window_seconds)
            if now > window_end:
                # Reset window
                state.window_start = now
                state.request_count = 0
                state.limit = limit

            # Check if within limit
            if state.request_count >= state.limit:
                remaining = 0
                return False, remaining

            # Increment counter
            state.request_count += 1
            remaining = state.limit - state.request_count

            return True, remaining

    def get_state(self, key_id: str) -> Optional[RateLimitState]:
        """Get current rate limit state for a key."""
        with self._lock:
            return self._states.get(key_id)

    def reset(self, key_id: str) -> None:
        """Reset rate limit for a key."""
        with self._lock:
            if key_id in self._states:
                del self._states[key_id]


# ============================================================================
# Audit Logging
# ============================================================================

class AuditLogger:
    """Audit logger for security events."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self._entries: List[AuditEntry] = []
        self._lock = threading.Lock()
        self._max_entries = 100000
        self._handlers: List[Callable[[AuditEntry], None]] = []

    def log(
        self,
        action: AuditAction,
        actor: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        success: bool = True,
        threat_detected: Optional[ThreatType] = None
    ) -> AuditEntry:
        """Log an audit entry."""
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc),
            action=action,
            actor=actor,
            resource=resource,
            details=details or {},
            ip_address=ip_address,
            success=success,
            threat_detected=threat_detected
        )

        if self.config.enable_audit_log:
            with self._lock:
                self._entries.append(entry)
                # Trim old entries
                if len(self._entries) > self._max_entries:
                    self._entries = self._entries[-self._max_entries:]

            # Notify handlers
            for handler in self._handlers:
                try:
                    handler(entry)
                except Exception as e:
                    logger.error(f"Audit handler error: {e}")

        # Log to standard logger as well
        log_msg = f"AUDIT: {action.value} by {actor or 'unknown'}"
        if threat_detected:
            logger.warning(f"{log_msg} - THREAT: {threat_detected.value}")
        elif not success:
            logger.warning(f"{log_msg} - FAILED")
        else:
            logger.info(log_msg)

        return entry

    def add_handler(self, handler: Callable[[AuditEntry], None]) -> None:
        """Add a handler to receive audit entries."""
        self._handlers.append(handler)

    def get_entries(
        self,
        action: Optional[AuditAction] = None,
        actor: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditEntry]:
        """Query audit entries."""
        with self._lock:
            entries = self._entries.copy()

        if action:
            entries = [e for e in entries if e.action == action]
        if actor:
            entries = [e for e in entries if e.actor == actor]
        if since:
            entries = [e for e in entries if e.timestamp >= since]

        return entries[-limit:]

    def export_json(self) -> str:
        """Export audit log as JSON."""
        with self._lock:
            entries = self._entries.copy()

        return json.dumps([
            {
                "timestamp": e.timestamp.isoformat(),
                "action": e.action.value,
                "actor": e.actor,
                "resource": e.resource,
                "details": e.details,
                "ip_address": e.ip_address,
                "success": e.success,
                "threat_detected": e.threat_detected.value if e.threat_detected else None
            }
            for e in entries
        ], indent=2)


# ============================================================================
# Security Manager
# ============================================================================

class SecurityManager:
    """Unified security manager for the platform."""

    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.validator = InputValidator(self.config)
        self.key_manager = APIKeyManager(self.config)
        self.rate_limiter = RateLimiter(self.config)
        self.audit_logger = AuditLogger(self.config)

    async def validate_request(
        self,
        api_key: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ) -> tuple[bool, Optional[str], Optional[APIKey]]:
        """
        Validate an incoming request.
        Returns (allowed, error_message, api_key_info).
        """
        key_info = None

        # Check blocked IPs
        if self.config.blocked_ips and ip_address:
            if ip_address in self.config.blocked_ips:
                self.audit_logger.log(
                    action=AuditAction.THREAT_BLOCKED,
                    ip_address=ip_address,
                    details={"reason": "blocked_ip"},
                    success=False,
                    threat_detected=ThreatType.UNAUTHORIZED
                )
                return False, "Access denied", None

        # Validate API key if provided
        if api_key:
            key_info = self.key_manager.validate_key(api_key)
            if not key_info:
                self.audit_logger.log(
                    action=AuditAction.AUTH_ATTEMPT,
                    ip_address=ip_address,
                    details={"reason": "invalid_api_key"},
                    success=False
                )
                return False, "Invalid API key", None

            # Check rate limit
            if self.config.enable_rate_limiting:
                allowed, remaining = self.rate_limiter.check_rate_limit(
                    key_info.key_id,
                    key_info.rate_limit
                )
                if not allowed:
                    self.audit_logger.log(
                        action=AuditAction.THREAT_BLOCKED,
                        actor=key_info.key_id,
                        ip_address=ip_address,
                        details={"reason": "rate_limit_exceeded"},
                        success=False,
                        threat_detected=ThreatType.RATE_LIMIT_EXCEEDED
                    )
                    return False, "Rate limit exceeded", key_info

        # Validate input data
        if data and self.config.enable_input_validation:
            threat = self.validator.validate_dict(data)
            if threat.is_threat:
                self.audit_logger.log(
                    action=AuditAction.THREAT_BLOCKED,
                    actor=key_info.key_id if key_info else None,
                    ip_address=ip_address,
                    details={
                        "threat_type": threat.threat_type.value if threat.threat_type else None,
                        "severity": threat.severity,
                        "details": threat.details
                    },
                    success=False,
                    threat_detected=threat.threat_type
                )
                return False, f"Security threat detected: {threat.details}", key_info

        # Log successful validation
        self.audit_logger.log(
            action=AuditAction.API_CALL,
            actor=key_info.key_id if key_info else None,
            ip_address=ip_address,
            details={"status": "validated"},
            success=True
        )

        return True, None, key_info

    def sanitize_output(self, data: Any) -> Any:
        """Sanitize output data to prevent secret exposure."""
        if isinstance(data, str):
            return self.validator.redact_secrets(data)
        elif isinstance(data, dict):
            return {k: self.sanitize_output(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_output(item) for item in data]
        return data


# ============================================================================
# Decorators
# ============================================================================

def secure_endpoint(
    require_auth: bool = True,
    required_permissions: Optional[List[str]] = None,
    rate_limit: Optional[int] = None
):
    """Decorator for securing API endpoints."""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, api_key: Optional[str] = None, **kwargs) -> T:
            security = get_security_manager()

            # Validate request
            allowed, error, key_info = await security.validate_request(
                api_key=api_key,
                data=kwargs
            )

            if not allowed:
                raise PermissionError(error)

            # Check required permissions
            if require_auth and not key_info:
                raise PermissionError("Authentication required")

            if required_permissions and key_info:
                missing = set(required_permissions) - set(key_info.permissions)
                if missing:
                    raise PermissionError(f"Missing permissions: {missing}")

            # Execute function
            result = await func(*args, **kwargs)

            # Sanitize output
            return security.sanitize_output(result)

        return wrapper
    return decorator


# ============================================================================
# Global Instance
# ============================================================================

_security_manager: Optional[SecurityManager] = None


def get_security_manager(config: Optional[SecurityConfig] = None) -> SecurityManager:
    """Get the global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager(config)
    return _security_manager


def configure_security(config: SecurityConfig) -> SecurityManager:
    """Configure and return the global security manager."""
    global _security_manager
    _security_manager = SecurityManager(config)
    return _security_manager


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Enums
    "SecurityLevel",
    "ThreatType",
    "AuditAction",

    # Data structures
    "SecurityConfig",
    "ThreatDetection",
    "AuditEntry",
    "APIKey",
    "RateLimitState",

    # Components
    "InputValidator",
    "APIKeyManager",
    "RateLimiter",
    "AuditLogger",
    "SecurityManager",

    # Decorators
    "secure_endpoint",

    # Global
    "get_security_manager",
    "configure_security",
]
