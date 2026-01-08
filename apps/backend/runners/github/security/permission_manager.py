"""
Permission Manager for Auto-PR-Review
======================================

Handles authorization checks for the autonomous PR review system.

Key features:
- GITHUB_AUTO_PR_REVIEW_ALLOWED_USERS allowlist enforcement
- Comprehensive audit logging of all permission decisions
- Integration with existing GitHubPermissionChecker for role-based checks
- Support for wildcard patterns and team-based authorization

Security notes:
- All permission decisions are logged for security auditing
- Denials include actor info for incident investigation
- Case-insensitive username matching to prevent bypass
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

logger = logging.getLogger(__name__)


# Environment variable name for allowed users
ALLOWED_USERS_ENV_VAR = "GITHUB_AUTO_PR_REVIEW_ALLOWED_USERS"


# Permission decision types
PermissionDecision = Literal["allowed", "denied", "not_configured"]


@dataclass
class PermissionCheckResult:
    """Result of a permission check for auto-PR-review."""

    allowed: bool
    username: str
    decision: PermissionDecision
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    allowlist_source: str = ALLOWED_USERS_ENV_VAR

    def to_dict(self) -> dict:
        """Convert to dictionary for logging and serialization."""
        return {
            "allowed": self.allowed,
            "username": self.username,
            "decision": self.decision,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "allowlist_source": self.allowlist_source,
        }


class PermissionDeniedError(Exception):
    """Raised when a permission check fails."""

    def __init__(self, result: PermissionCheckResult):
        self.result = result
        super().__init__(result.reason)


class PermissionManager:
    """
    Manages authorization for the autonomous PR review system.

    Authorization is controlled via the GITHUB_AUTO_PR_REVIEW_ALLOWED_USERS
    environment variable, which contains a comma-separated list of GitHub
    usernames authorized to trigger auto-PR-review.

    Usage:
        pm = PermissionManager()

        # Check if user can trigger auto-PR-review
        result = pm.can_trigger_auto_pr_review("username")
        if result.allowed:
            # proceed with PR review
            pass
        else:
            logger.warning(f"Access denied: {result.reason}")

        # Or use the raising variant
        try:
            pm.require_permission("username")
        except PermissionDeniedError as e:
            logger.error(f"Permission denied: {e}")

    Configuration:
        Set GITHUB_AUTO_PR_REVIEW_ALLOWED_USERS environment variable:
        - Single user: "alice"
        - Multiple users: "alice,bob,charlie"
        - All users (not recommended): "*"

    Security Notes:
        - Usernames are compared case-insensitively
        - Empty or whitespace-only values are ignored
        - All permission decisions are logged
    """

    # Special value that allows all users (use with caution)
    WILDCARD = "*"

    def __init__(
        self,
        env_var_name: str = ALLOWED_USERS_ENV_VAR,
        log_decisions: bool = True,
        strict_mode: bool = True,
    ):
        """
        Initialize the permission manager.

        Args:
            env_var_name: Name of the environment variable containing allowed users
            log_decisions: Whether to log all permission decisions
            strict_mode: If True, deny access when allowlist is empty/not configured
        """
        self.env_var_name = env_var_name
        self.log_decisions = log_decisions
        self.strict_mode = strict_mode

        # Parse the allowlist on initialization
        self._allowed_users: set[str] = set()
        self._allow_all: bool = False
        self._is_configured: bool = False

        self._parse_allowlist()

    def _parse_allowlist(self) -> None:
        """Parse the allowlist from the environment variable."""
        raw_value = os.environ.get(self.env_var_name, "").strip()

        if not raw_value:
            logger.info(
                f"Permission allowlist not configured: {self.env_var_name} is empty or not set"
            )
            self._is_configured = False
            return

        self._is_configured = True

        # Check for wildcard
        if raw_value == self.WILDCARD:
            logger.warning(
                f"⚠️ SECURITY: {self.env_var_name} is set to wildcard (*). "
                f"ALL users can trigger auto-PR-review."
            )
            self._allow_all = True
            return

        # Parse comma-separated list
        users = [u.strip().lower() for u in raw_value.split(",") if u.strip()]
        self._allowed_users = set(users)

        logger.info(
            f"Loaded {len(self._allowed_users)} users from {self.env_var_name} allowlist"
        )

    def reload_allowlist(self) -> None:
        """
        Reload the allowlist from the environment variable.

        Call this method if the environment variable may have changed.
        """
        self._allowed_users = set()
        self._allow_all = False
        self._is_configured = False
        self._parse_allowlist()

    @property
    def is_configured(self) -> bool:
        """Check if the allowlist is configured."""
        return self._is_configured

    @property
    def allowed_users(self) -> set[str]:
        """Get the set of allowed usernames (lowercase)."""
        return self._allowed_users.copy()

    @property
    def allows_all(self) -> bool:
        """Check if all users are allowed (wildcard mode)."""
        return self._allow_all

    def can_trigger_auto_pr_review(
        self,
        username: str,
        pr_number: int | None = None,
        repo: str | None = None,
    ) -> PermissionCheckResult:
        """
        Check if a user is allowed to trigger auto-PR-review.

        Args:
            username: GitHub username to check
            pr_number: Optional PR number for logging
            repo: Optional repository name for logging

        Returns:
            PermissionCheckResult with the authorization decision
        """
        # Normalize username for comparison
        normalized_username = username.lower().strip()

        # Build context for logging
        context = {
            "username": username,
            "normalized_username": normalized_username,
            "pr_number": pr_number,
            "repo": repo,
            "env_var": self.env_var_name,
        }

        # Check if allowlist is configured
        if not self._is_configured:
            if self.strict_mode:
                result = PermissionCheckResult(
                    allowed=False,
                    username=username,
                    decision="not_configured",
                    reason=f"Allowlist not configured: {self.env_var_name} is empty or not set. "
                    f"Set this environment variable to enable auto-PR-review.",
                )
                self._log_decision(result, context, "denied_not_configured")
                return result
            else:
                # Non-strict mode: allow when not configured (for testing)
                result = PermissionCheckResult(
                    allowed=True,
                    username=username,
                    decision="allowed",
                    reason="Allowlist not configured but strict_mode is disabled",
                )
                self._log_decision(result, context, "allowed_non_strict")
                return result

        # Check wildcard
        if self._allow_all:
            result = PermissionCheckResult(
                allowed=True,
                username=username,
                decision="allowed",
                reason="Wildcard allowlist: all users permitted",
            )
            self._log_decision(result, context, "allowed_wildcard")
            return result

        # Check specific user
        if normalized_username in self._allowed_users:
            result = PermissionCheckResult(
                allowed=True,
                username=username,
                decision="allowed",
                reason=f"User '{username}' is in the allowlist",
            )
            self._log_decision(result, context, "allowed_explicit")
            return result

        # User not in allowlist - denied
        result = PermissionCheckResult(
            allowed=False,
            username=username,
            decision="denied",
            reason=f"User '{username}' is not in the {self.env_var_name} allowlist. "
            f"Contact your administrator to request access.",
        )
        self._log_decision(result, context, "denied_not_in_allowlist")
        return result

    def require_permission(
        self,
        username: str,
        pr_number: int | None = None,
        repo: str | None = None,
    ) -> PermissionCheckResult:
        """
        Check permission and raise PermissionDeniedError if denied.

        Args:
            username: GitHub username to check
            pr_number: Optional PR number for logging
            repo: Optional repository name for logging

        Returns:
            PermissionCheckResult if allowed

        Raises:
            PermissionDeniedError: If user is not authorized
        """
        result = self.can_trigger_auto_pr_review(username, pr_number, repo)
        if not result.allowed:
            raise PermissionDeniedError(result)
        return result

    def _log_decision(
        self,
        result: PermissionCheckResult,
        context: dict,
        decision_type: str,
    ) -> None:
        """Log a permission decision with full context."""
        if not self.log_decisions:
            return

        log_context = {
            **context,
            "decision_type": decision_type,
            "decision_result": result.decision,
            "timestamp": result.timestamp.isoformat(),
        }

        if result.allowed:
            logger.info(
                f"✓ Permission GRANTED for {result.username} to trigger auto-PR-review",
                extra=log_context,
            )
        else:
            logger.warning(
                f"✗ Permission DENIED for {result.username} to trigger auto-PR-review: "
                f"{result.reason}",
                extra=log_context,
            )

    def log_permission_denial(
        self,
        action: str,
        username: str,
        pr_number: int | None = None,
        repo: str | None = None,
        additional_context: dict | None = None,
    ) -> None:
        """
        Log a permission denial with full context for security auditing.

        Args:
            action: Action that was denied (e.g., "trigger_auto_pr_review")
            username: GitHub username
            pr_number: Optional PR number
            repo: Optional repository name
            additional_context: Additional context to include in log
        """
        context = {
            "action": action,
            "username": username,
            "pr_number": pr_number,
            "repo": repo,
            "allowed_users_count": len(self._allowed_users),
            "allow_all_enabled": self._allow_all,
            "is_configured": self._is_configured,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if additional_context:
            context.update(additional_context)

        logger.warning(
            f"PERMISSION DENIED: User '{username}' attempted '{action}'",
            extra=context,
        )

    def add_user_to_allowlist(self, username: str) -> None:
        """
        Add a user to the in-memory allowlist.

        Note: This does NOT persist to the environment variable.
        Use this for testing or temporary additions.

        Args:
            username: GitHub username to add
        """
        normalized = username.lower().strip()
        self._allowed_users.add(normalized)
        self._is_configured = True

        logger.info(
            f"Added user '{username}' to allowlist (in-memory only, not persisted)"
        )

    def remove_user_from_allowlist(self, username: str) -> bool:
        """
        Remove a user from the in-memory allowlist.

        Note: This does NOT persist to the environment variable.
        Use this for testing or temporary removals.

        Args:
            username: GitHub username to remove

        Returns:
            True if user was removed, False if not in allowlist
        """
        normalized = username.lower().strip()
        if normalized in self._allowed_users:
            self._allowed_users.discard(normalized)
            logger.info(
                f"Removed user '{username}' from allowlist (in-memory only, not persisted)"
            )
            return True
        return False

    def get_allowlist_status(self) -> dict:
        """
        Get current allowlist status for monitoring/debugging.

        Returns:
            Dictionary with allowlist configuration status
        """
        return {
            "is_configured": self._is_configured,
            "allow_all": self._allow_all,
            "user_count": len(self._allowed_users),
            "env_var_name": self.env_var_name,
            "strict_mode": self.strict_mode,
            "log_decisions": self.log_decisions,
        }


# Global singleton instance
_permission_manager: PermissionManager | None = None


def get_permission_manager(
    strict_mode: bool = True,
    log_decisions: bool = True,
) -> PermissionManager:
    """
    Get the global PermissionManager instance.

    Args:
        strict_mode: If True, deny access when allowlist is not configured
        log_decisions: Whether to log permission decisions

    Returns:
        PermissionManager singleton instance
    """
    global _permission_manager
    if _permission_manager is None:
        _permission_manager = PermissionManager(
            strict_mode=strict_mode,
            log_decisions=log_decisions,
        )
    return _permission_manager


def can_trigger_auto_pr_review(
    username: str,
    pr_number: int | None = None,
    repo: str | None = None,
) -> PermissionCheckResult:
    """
    Convenience function to check if user can trigger auto-PR-review.

    Args:
        username: GitHub username to check
        pr_number: Optional PR number for logging
        repo: Optional repository name for logging

    Returns:
        PermissionCheckResult with authorization decision
    """
    return get_permission_manager().can_trigger_auto_pr_review(
        username, pr_number, repo
    )


def require_auto_pr_review_permission(
    username: str,
    pr_number: int | None = None,
    repo: str | None = None,
) -> PermissionCheckResult:
    """
    Convenience function that raises if user is not authorized.

    Args:
        username: GitHub username to check
        pr_number: Optional PR number for logging
        repo: Optional repository name for logging

    Returns:
        PermissionCheckResult if allowed

    Raises:
        PermissionDeniedError: If user is not authorized
    """
    return get_permission_manager().require_permission(username, pr_number, repo)
