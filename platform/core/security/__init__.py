"""
Unleashed Platform - Security Module
=====================================

Comprehensive security package providing:
- Bash command filtering (allowlist-based)
- Input validation and sanitization
- API key management
- Rate limiting
- Audit logging
- Secret detection

Quick Start:
    from core.security import check_command, BashSecurityFilter

    # Simple check
    allowed, reason = check_command("ls -la")

    # Full analysis
    filter = BashSecurityFilter(strict_mode=True)
    result = filter.analyze("rm -rf /")
    if not result.allowed:
        print(f"Blocked: {result.reason}")
"""

from .bash_filter import (
    # Classes
    BashSecurityFilter,
    CommandAnalysis,
    CommandRisk,
    # Functions
    check_command,
    analyze_command,
    get_filter,
    # Sets (for extension)
    ALLOWLIST_SAFE,
    ALLOWLIST_CONDITIONAL,
    BLOCKLIST,
    DANGEROUS_PATTERNS,
)

__all__ = [
    # Classes
    "BashSecurityFilter",
    "CommandAnalysis",
    "CommandRisk",
    # Functions
    "check_command",
    "analyze_command",
    "get_filter",
    # Sets (for extension)
    "ALLOWLIST_SAFE",
    "ALLOWLIST_CONDITIONAL",
    "BLOCKLIST",
    "DANGEROUS_PATTERNS",
]
