"""
GitHub Security Infrastructure
==============================

Security components for the autonomous PR review system:
- InputSanitizer: Sanitizes untrusted inputs (prompt injection, path traversal, Unicode)
- PermissionManager: Authorization checks against allowlist
"""

from __future__ import annotations

from .input_sanitizer import InputSanitizer, SanitizeResult
from .permission_manager import (
    PermissionCheckResult,
    PermissionDeniedError,
    PermissionManager,
    can_trigger_auto_pr_review,
    get_permission_manager,
    require_auto_pr_review_permission,
)

__all__ = [
    "InputSanitizer",
    "SanitizeResult",
    "PermissionCheckResult",
    "PermissionDeniedError",
    "PermissionManager",
    "can_trigger_auto_pr_review",
    "get_permission_manager",
    "require_auto_pr_review_permission",
]
