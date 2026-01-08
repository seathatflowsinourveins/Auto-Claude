"""
Bot Identity Verifier for Auto-PR-Review
=========================================

Verifies external bot identities by account ID to prevent spoofing attacks.

Key features:
- Trusted bot allowlist via environment variable
- Account ID verification (not just username matching)
- Support for common CI/review bots (CodeRabbit, Cursor, etc.)
- Comprehensive audit logging of verification decisions
- Spoofing detection and rejection

Security notes:
- NEVER trust bot comments by name alone - verify by account ID
- Unknown bots are rejected by default (fail-safe)
- All verification decisions are logged for security auditing
- Account IDs are immutable unlike usernames which can be changed
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

logger = logging.getLogger(__name__)


# Environment variable for trusted bot IDs
TRUSTED_BOTS_ENV_VAR = "GITHUB_TRUSTED_BOT_IDS"


# Well-known bot account IDs for common CI/review bots
# These are GitHub account IDs which are immutable (unlike usernames)
# Format: account_id -> (expected_username, description)
WELL_KNOWN_BOTS: dict[int, tuple[str, str]] = {
    # CodeRabbit AI
    136556919: ("coderabbitai[bot]", "CodeRabbit AI code review bot"),
    # GitHub Actions
    41898282: ("github-actions[bot]", "GitHub Actions automation"),
    # Dependabot
    49699333: ("dependabot[bot]", "Dependabot dependency updates"),
    # Renovate Bot
    29139614: ("renovate[bot]", "Renovate dependency updates"),
    # Codecov
    22429695: ("codecov[bot]", "Codecov coverage reports"),
    # SonarCloud
    37929162: ("sonarcloud[bot]", "SonarCloud code analysis"),
    # Vercel Bot
    35613825: ("vercel[bot]", "Vercel deployment previews"),
    # Netlify Bot
    36544832: ("netlify[bot]", "Netlify deployment previews"),
    # Imgbot
    21237556: ("imgbot[bot]", "Imgbot image optimization"),
    # Snyk Bot
    19733683: ("snyk-bot", "Snyk security scanning"),
    # Stale Bot
    26384082: ("stale[bot]", "GitHub Stale issue/PR management"),
}


# Verification decision types
VerificationDecision = Literal["trusted", "rejected", "spoofing_detected", "unknown"]


@dataclass
class BotVerificationResult:
    """Result of a bot identity verification."""

    is_trusted: bool
    account_id: int
    username: str
    decision: VerificationDecision
    reason: str
    expected_username: str | None = None
    bot_description: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Convert to dictionary for logging and serialization."""
        return {
            "is_trusted": self.is_trusted,
            "account_id": self.account_id,
            "username": self.username,
            "decision": self.decision,
            "reason": self.reason,
            "expected_username": self.expected_username,
            "bot_description": self.bot_description,
            "timestamp": self.timestamp.isoformat(),
        }


class BotVerificationError(Exception):
    """Raised when bot verification fails."""

    def __init__(self, result: BotVerificationResult):
        self.result = result
        super().__init__(result.reason)


class BotVerifier:
    """
    Verifies external bot identities by account ID.

    GitHub usernames can be changed, but account IDs are immutable. This class
    ensures that bot comments are actually from the expected bots by verifying
    the account ID matches the username.

    Usage:
        verifier = BotVerifier()

        # Check if a comment author is a trusted bot
        result = verifier.is_trusted_bot(
            account_id=136556919,
            username="coderabbitai[bot]"
        )
        if result.is_trusted:
            # Process the bot comment
            process_findings(comment)
        else:
            logger.warning(f"Untrusted bot comment: {result.reason}")

        # Or use the raising variant
        try:
            verifier.require_trusted_bot(account_id, username)
        except BotVerificationError as e:
            logger.error(f"Bot verification failed: {e}")

    Configuration:
        Set GITHUB_TRUSTED_BOT_IDS environment variable with additional bot IDs:
        - Single ID: "12345678"
        - Multiple IDs: "12345678,87654321,11111111"
        - Well-known bots are always included unless explicitly disabled

    Security Notes:
        - Verifies account ID matches expected username
        - Detects potential spoofing (username mismatch)
        - Unknown bots are rejected by default
        - All decisions are logged for auditing
    """

    def __init__(
        self,
        env_var_name: str = TRUSTED_BOTS_ENV_VAR,
        include_well_known: bool = True,
        log_decisions: bool = True,
        strict_mode: bool = True,
    ):
        """
        Initialize the bot verifier.

        Args:
            env_var_name: Name of env var containing additional trusted bot IDs
            include_well_known: Include well-known bots (CodeRabbit, etc.)
            log_decisions: Whether to log all verification decisions
            strict_mode: If True, reject unknown bots; if False, allow them
        """
        self.env_var_name = env_var_name
        self.include_well_known = include_well_known
        self.log_decisions = log_decisions
        self.strict_mode = strict_mode

        # Build trusted bot registry
        self._trusted_bots: dict[int, tuple[str, str]] = {}
        self._additional_ids: set[int] = set()

        self._build_trusted_registry()

    def _build_trusted_registry(self) -> None:
        """Build the trusted bot registry from well-known bots and env var."""
        # Start with well-known bots if enabled
        if self.include_well_known:
            self._trusted_bots = WELL_KNOWN_BOTS.copy()
            logger.info(
                f"Loaded {len(WELL_KNOWN_BOTS)} well-known bots into trusted registry"
            )

        # Parse additional IDs from environment variable
        raw_value = os.environ.get(self.env_var_name, "").strip()

        if not raw_value:
            logger.debug(f"No additional bot IDs configured in {self.env_var_name}")
            return

        # Parse comma-separated IDs
        for id_str in raw_value.split(","):
            id_str = id_str.strip()
            if not id_str:
                continue

            try:
                account_id = int(id_str)
                self._additional_ids.add(account_id)
                # Add to trusted bots with unknown username/description
                if account_id not in self._trusted_bots:
                    self._trusted_bots[account_id] = (
                        "unknown",
                        "User-configured trusted bot",
                    )
            except ValueError:
                logger.warning(
                    f"Invalid bot ID in {self.env_var_name}: '{id_str}' (must be integer)"
                )

        logger.info(
            f"Loaded {len(self._additional_ids)} additional bot IDs from {self.env_var_name}"
        )

    def reload_registry(self) -> None:
        """
        Reload the trusted bot registry.

        Call this if the environment variable may have changed.
        """
        self._trusted_bots = {}
        self._additional_ids = set()
        self._build_trusted_registry()

    @property
    def trusted_bot_ids(self) -> set[int]:
        """Get the set of trusted bot account IDs."""
        return set(self._trusted_bots.keys())

    def get_bot_info(self, account_id: int) -> tuple[str, str] | None:
        """
        Get info about a trusted bot.

        Args:
            account_id: GitHub account ID

        Returns:
            Tuple of (expected_username, description) or None if not trusted
        """
        return self._trusted_bots.get(account_id)

    def is_trusted_bot(
        self,
        account_id: int,
        username: str,
        pr_number: int | None = None,
        repo: str | None = None,
    ) -> BotVerificationResult:
        """
        Verify if a bot is trusted by checking account ID and username.

        Args:
            account_id: GitHub account ID of the bot
            username: Current username of the bot
            pr_number: Optional PR number for logging
            repo: Optional repository name for logging

        Returns:
            BotVerificationResult with the verification decision
        """
        # Build context for logging
        context = {
            "account_id": account_id,
            "username": username,
            "pr_number": pr_number,
            "repo": repo,
        }

        # Check if account ID is in trusted registry
        bot_info = self._trusted_bots.get(account_id)

        if bot_info is None:
            # Unknown bot
            if self.strict_mode:
                result = BotVerificationResult(
                    is_trusted=False,
                    account_id=account_id,
                    username=username,
                    decision="unknown",
                    reason=f"Bot with account ID {account_id} ({username}) is not in the trusted registry. "
                    f"Add it to {self.env_var_name} if this bot should be trusted.",
                )
                self._log_decision(result, context, "rejected_unknown")
                return result
            else:
                # Non-strict mode: allow unknown bots
                result = BotVerificationResult(
                    is_trusted=True,
                    account_id=account_id,
                    username=username,
                    decision="trusted",
                    reason="Unknown bot allowed in non-strict mode",
                )
                self._log_decision(result, context, "allowed_non_strict")
                return result

        expected_username, description = bot_info

        # Verify username matches (detect spoofing)
        # Account ID is trusted, but someone might be impersonating with wrong username
        if (
            expected_username != "unknown"
            and username.lower() != expected_username.lower()
        ):
            # Potential spoofing: account ID is trusted but username doesn't match
            result = BotVerificationResult(
                is_trusted=False,
                account_id=account_id,
                username=username,
                decision="spoofing_detected",
                reason=f"SECURITY: Account ID {account_id} is trusted for '{expected_username}' "
                f"but current username is '{username}'. Possible account takeover or spoofing.",
                expected_username=expected_username,
                bot_description=description,
            )
            self._log_decision(result, context, "spoofing_detected")
            return result

        # Trusted bot with matching username
        result = BotVerificationResult(
            is_trusted=True,
            account_id=account_id,
            username=username,
            decision="trusted",
            reason=f"Verified trusted bot: {description}",
            expected_username=expected_username,
            bot_description=description,
        )
        self._log_decision(result, context, "verified_trusted")
        return result

    def require_trusted_bot(
        self,
        account_id: int,
        username: str,
        pr_number: int | None = None,
        repo: str | None = None,
    ) -> BotVerificationResult:
        """
        Verify bot and raise BotVerificationError if not trusted.

        Args:
            account_id: GitHub account ID of the bot
            username: Current username of the bot
            pr_number: Optional PR number for logging
            repo: Optional repository name for logging

        Returns:
            BotVerificationResult if trusted

        Raises:
            BotVerificationError: If bot is not trusted
        """
        result = self.is_trusted_bot(account_id, username, pr_number, repo)
        if not result.is_trusted:
            raise BotVerificationError(result)
        return result

    def verify_comment_author(
        self,
        comment: dict,
        pr_number: int | None = None,
        repo: str | None = None,
    ) -> BotVerificationResult:
        """
        Verify a comment author is a trusted bot.

        Args:
            comment: GitHub comment object with 'user' field containing 'id' and 'login'
            pr_number: Optional PR number for logging
            repo: Optional repository name for logging

        Returns:
            BotVerificationResult with verification decision
        """
        user = comment.get("user", {})
        account_id = user.get("id")
        username = user.get("login", "unknown")

        if account_id is None:
            return BotVerificationResult(
                is_trusted=False,
                account_id=0,
                username=username,
                decision="rejected",
                reason="Comment has no user account ID - cannot verify bot identity",
            )

        return self.is_trusted_bot(account_id, username, pr_number, repo)

    def _log_decision(
        self,
        result: BotVerificationResult,
        context: dict,
        decision_type: str,
    ) -> None:
        """Log a verification decision with full context."""
        if not self.log_decisions:
            return

        log_context = {
            **context,
            "decision_type": decision_type,
            "verification_result": result.decision,
            "timestamp": result.timestamp.isoformat(),
        }

        if result.is_trusted:
            logger.info(
                f"Bot VERIFIED: {result.username} (ID: {result.account_id}) - {result.reason}",
                extra=log_context,
            )
        elif result.decision == "spoofing_detected":
            # High severity security event
            logger.error(
                f"SPOOFING DETECTED: {result.username} (ID: {result.account_id}) - {result.reason}",
                extra=log_context,
            )
        else:
            logger.warning(
                f"Bot REJECTED: {result.username} (ID: {result.account_id}) - {result.reason}",
                extra=log_context,
            )

    def log_bot_rejection(
        self,
        account_id: int,
        username: str,
        action: str,
        pr_number: int | None = None,
        repo: str | None = None,
        additional_context: dict | None = None,
    ) -> None:
        """
        Log a bot rejection with full context for security auditing.

        Args:
            account_id: Bot's GitHub account ID
            username: Bot's username
            action: Action that was rejected (e.g., "process_comment")
            pr_number: Optional PR number
            repo: Optional repository name
            additional_context: Additional context to include in log
        """
        context = {
            "action": action,
            "account_id": account_id,
            "username": username,
            "pr_number": pr_number,
            "repo": repo,
            "trusted_bot_count": len(self._trusted_bots),
            "strict_mode": self.strict_mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if additional_context:
            context.update(additional_context)

        logger.warning(
            f"BOT REJECTED: {username} (ID: {account_id}) attempted '{action}'",
            extra=context,
        )

    def add_trusted_bot(
        self,
        account_id: int,
        username: str = "unknown",
        description: str = "Dynamically added trusted bot",
    ) -> None:
        """
        Add a bot to the in-memory trusted registry.

        Note: This does NOT persist to the environment variable.
        Use this for testing or temporary additions.

        Args:
            account_id: GitHub account ID
            username: Expected username
            description: Bot description
        """
        self._trusted_bots[account_id] = (username, description)
        logger.info(
            f"Added bot {username} (ID: {account_id}) to trusted registry (in-memory only)"
        )

    def remove_trusted_bot(self, account_id: int) -> bool:
        """
        Remove a bot from the in-memory trusted registry.

        Note: This does NOT persist changes.
        Cannot remove well-known bots unless include_well_known was False.

        Args:
            account_id: GitHub account ID

        Returns:
            True if bot was removed, False if not in registry
        """
        if account_id in self._trusted_bots:
            # Don't allow removing well-known bots
            if self.include_well_known and account_id in WELL_KNOWN_BOTS:
                logger.warning(
                    f"Cannot remove well-known bot {account_id} from registry"
                )
                return False

            del self._trusted_bots[account_id]
            logger.info(f"Removed bot ID {account_id} from trusted registry")
            return True
        return False

    def get_registry_status(self) -> dict:
        """
        Get current registry status for monitoring/debugging.

        Returns:
            Dictionary with registry configuration status
        """
        return {
            "total_trusted_bots": len(self._trusted_bots),
            "well_known_bots_included": self.include_well_known,
            "additional_ids_count": len(self._additional_ids),
            "env_var_name": self.env_var_name,
            "strict_mode": self.strict_mode,
            "log_decisions": self.log_decisions,
        }


# Global singleton instance
_bot_verifier: BotVerifier | None = None


def get_bot_verifier(
    strict_mode: bool = True,
    log_decisions: bool = True,
    include_well_known: bool = True,
) -> BotVerifier:
    """
    Get the global BotVerifier instance.

    Args:
        strict_mode: If True, reject unknown bots
        log_decisions: Whether to log verification decisions
        include_well_known: Include well-known bots in registry

    Returns:
        BotVerifier singleton instance
    """
    global _bot_verifier
    if _bot_verifier is None:
        _bot_verifier = BotVerifier(
            strict_mode=strict_mode,
            log_decisions=log_decisions,
            include_well_known=include_well_known,
        )
    return _bot_verifier


def is_trusted_bot(
    account_id: int,
    username: str,
    pr_number: int | None = None,
    repo: str | None = None,
) -> BotVerificationResult:
    """
    Convenience function to verify if a bot is trusted.

    Args:
        account_id: GitHub account ID of the bot
        username: Current username of the bot
        pr_number: Optional PR number for logging
        repo: Optional repository name for logging

    Returns:
        BotVerificationResult with verification decision
    """
    return get_bot_verifier().is_trusted_bot(account_id, username, pr_number, repo)


def require_trusted_bot(
    account_id: int,
    username: str,
    pr_number: int | None = None,
    repo: str | None = None,
) -> BotVerificationResult:
    """
    Convenience function that raises if bot is not trusted.

    Args:
        account_id: GitHub account ID of the bot
        username: Current username of the bot
        pr_number: Optional PR number for logging
        repo: Optional repository name for logging

    Returns:
        BotVerificationResult if trusted

    Raises:
        BotVerificationError: If bot is not trusted
    """
    return get_bot_verifier().require_trusted_bot(account_id, username, pr_number, repo)
