"""
Bash Command Security Filter
============================

Implements allowlist-based command filtering for safe Bash execution.
Based on patterns from anthropics/claude-quickstarts/autonomous-coding/security.py

This module provides comprehensive bash command filtering to prevent:
- Destructive file operations (rm, shred)
- Privilege escalation (sudo, su)
- Shell escapes (bash, sh, zsh)
- Command injection attacks
- Network attack tools
- System modification commands

Usage:
    from core.security.bash_filter import BashSecurityFilter, check_command

    # Quick check
    allowed, reason = check_command("ls -la")

    # Full analysis
    filter = BashSecurityFilter()
    result = filter.analyze("rm -rf /")
    if not result.allowed:
        print(f"Blocked: {result.reason}")
"""

import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Tuple


class CommandRisk(Enum):
    """Risk levels for commands."""
    SAFE = "safe"
    MODERATE = "moderate"
    DANGEROUS = "dangerous"
    BLOCKED = "blocked"


@dataclass
class CommandAnalysis:
    """Result of command analysis."""
    command: str
    risk: CommandRisk
    allowed: bool
    reason: str
    matched_pattern: Optional[str] = None


# Safe commands that can run without additional checks
ALLOWLIST_SAFE: Set[str] = {
    # Navigation and listing
    "ls", "pwd", "cd", "tree", "dir",
    # File viewing (read-only)
    "cat", "head", "tail", "less", "more", "wc", "file", "stat",
    # Search (read-only)
    "find", "grep", "rg", "ag", "fd", "locate", "which", "whereis",
    # Git (most operations)
    "git",
    # Node.js development
    "node", "npm", "npx", "yarn", "pnpm", "bun", "deno",
    # Python development
    "python", "python3", "pip", "pip3", "uv", "uvx", "poetry", "pdm", "ruff",
    # Build tools
    "make", "cmake", "cargo", "go", "rustc", "rustup", "gcc", "g++", "clang",
    # Testing
    "pytest", "jest", "vitest", "mocha", "nyc", "coverage",
    # Utilities
    "echo", "date", "which", "type", "env", "printenv", "whoami", "id",
    "sort", "uniq", "diff", "patch", "tr", "cut", "paste", "join",
    "xargs", "tee", "basename", "dirname", "realpath", "readlink",
    "printf", "seq", "yes", "true", "false", "test", "[",
    # Claude Flow
    "claude", "claude-flow",
    # Compression (read operations)
    "tar", "gzip", "gunzip", "zip", "unzip", "bzip2", "xz",
    # Text processing
    "awk", "sed", "jq", "yq",
    # Process info (read-only)
    "ps", "top", "htop", "pgrep", "lsof",
    # Disk info (read-only)
    "df", "du", "free",
    # Network info (read-only)
    "ping", "host", "dig", "nslookup", "ifconfig", "ip",
}

# Commands that require argument inspection
ALLOWLIST_CONDITIONAL: Set[str] = {
    # File operations - need to check paths
    "cp", "mv", "mkdir", "touch", "ln",
    # Network - need to check targets
    "curl", "wget", "http", "httpie",
    # Docker - need to check commands
    "docker", "docker-compose", "podman",
    # Kubernetes - need to check commands
    "kubectl", "helm", "k9s",
    # Cloud CLIs
    "aws", "gcloud", "az",
    # SSH (need to verify targets)
    "ssh", "scp", "sftp", "rsync",
}

# Always blocked commands
BLOCKLIST: Set[str] = {
    # Destructive file operations
    "rm", "rmdir", "shred", "wipe",
    # System modification
    "chmod", "chown", "chgrp", "chattr",
    "sudo", "su", "doas", "pkexec",
    # Network attack tools
    "nc", "netcat", "ncat", "socat",
    "nmap", "masscan", "nikto", "sqlmap",
    "hydra", "john", "hashcat", "aircrack-ng",
    # Crypto/mining
    "bitcoin", "ethereum", "monero", "xmrig", "ccminer",
    # Shells (prevent escape)
    "bash", "sh", "zsh", "fish", "tcsh", "csh", "ksh", "dash",
    # Process control (potentially dangerous)
    "kill", "killall", "pkill", "renice",
    # Disk operations
    "dd", "fdisk", "gdisk", "parted", "mkfs", "mount", "umount", "losetup",
    # User/Group management
    "passwd", "useradd", "userdel", "usermod", "groupadd", "groupdel", "groupmod",
    "adduser", "deluser", "addgroup", "delgroup",
    # System control
    "shutdown", "reboot", "poweroff", "halt", "init", "systemctl", "service",
    # Kernel/boot
    "insmod", "rmmod", "modprobe", "depmod", "dmesg",
    # Package managers (can install malware)
    "apt", "apt-get", "yum", "dnf", "pacman", "zypper", "apk",
    "snap", "flatpak", "brew",
    # Dangerous utilities
    "eval", "exec", "source", "xargs",
    # Cron (persistence)
    "crontab", "at", "batch",
    # SELinux/AppArmor
    "setenforce", "getenforce", "aa-enforce", "aa-complain",
    # Firewall
    "iptables", "ip6tables", "nft", "firewall-cmd", "ufw",
    # Reverse shells
    "telnet", "rsh", "rlogin",
}

# Dangerous patterns in arguments
DANGEROUS_PATTERNS: List[Tuple[str, str]] = [
    # Command injection with rm
    (r";\s*rm\s", "Command injection with rm"),
    (r"\|\s*rm\s", "Pipe to rm"),
    (r"&&\s*rm\s", "Chained rm command"),
    (r"\|\|\s*rm\s", "Fallback rm command"),

    # Redirect to sensitive paths
    (r">\s*/dev/", "Redirect to device"),
    (r">\s*/etc/", "Redirect to /etc"),
    (r">\s*/sys/", "Redirect to /sys"),
    (r">\s*/proc/", "Redirect to /proc"),
    (r">\s*/boot/", "Redirect to /boot"),
    (r">\s*~", "Redirect to home directory"),

    # Command substitution (potential injection)
    (r"`[^`]+`", "Backtick command substitution"),
    (r"\$\([^)]+\)", "Dollar-paren command substitution"),

    # Eval and exec
    (r"\beval\s", "Eval command"),
    (r"\bexec\s", "Exec command"),
    (r"\bsource\s", "Source command"),
    (r"^\.\s+/", "Dot source command"),

    # Curl/wget to shell (common malware pattern)
    (r"curl[^|]*\|\s*(ba)?sh", "Curl pipe to shell"),
    (r"wget[^|]*\|\s*(ba)?sh", "Wget pipe to shell"),
    (r"curl[^|]*\|\s*python", "Curl pipe to python"),
    (r"wget[^|]*\|\s*python", "Wget pipe to python"),

    # Base64 decode to shell (obfuscation)
    (r"base64\s+-d[^|]*\|\s*(ba)?sh", "Base64 decode to shell"),
    (r"base64\s+--decode[^|]*\|\s*(ba)?sh", "Base64 decode to shell"),

    # Hex decode to shell
    (r"xxd\s+-r[^|]*\|\s*(ba)?sh", "Hex decode to shell"),

    # Fork bomb patterns
    (r":\(\)\s*{\s*:\s*\|\s*:\s*&\s*}", "Fork bomb detected"),
    (r"\.\s*\(\)\s*{\s*\.\s*\|\s*\.\s*&\s*}", "Fork bomb detected"),

    # Recursive operations on root
    (r"-[rR]\s+/\s*$", "Recursive operation on root"),
    (r"-[rR]f\s+/\s*$", "Recursive force operation on root"),
    (r"-rf\s+/\s*$", "Recursive force operation on root"),

    # Dangerous redirects
    (r">\s*/dev/sd[a-z]", "Redirect to block device"),
    (r">\s*/dev/nvme", "Redirect to NVMe device"),

    # History manipulation (covering tracks)
    (r"history\s+-[cd]", "History manipulation"),
    (r"unset\s+HISTFILE", "Disabling history"),
    (r"HISTSIZE=0", "Disabling history"),

    # Environment manipulation
    (r"export\s+LD_PRELOAD", "LD_PRELOAD injection"),
    (r"export\s+LD_LIBRARY_PATH", "LD_LIBRARY_PATH manipulation"),

    # Null byte injection
    (r"\\x00", "Null byte injection"),
    (r"\\0", "Null byte injection"),
]


class BashSecurityFilter:
    """
    Bash command security filter with allowlist enforcement.

    Provides three modes of operation:
    1. Non-strict (default): Allow unknown commands with MODERATE risk
    2. Strict: Block all unknown commands
    3. Custom: User-defined allow/block lists

    Usage:
        filter = BashSecurityFilter()
        result = filter.analyze("ls -la")
        if result.allowed:
            # Execute command
        else:
            # Block with reason
            print(f"Blocked: {result.reason}")
    """

    def __init__(
        self,
        additional_safe: Optional[Set[str]] = None,
        additional_blocked: Optional[Set[str]] = None,
        allow_conditionals: bool = True,
        strict_mode: bool = False,
    ):
        """
        Initialize the bash security filter.

        Args:
            additional_safe: Extra commands to allow
            additional_blocked: Extra commands to block
            allow_conditionals: Whether to allow conditional commands
            strict_mode: If True, block unknown commands
        """
        self.safe_commands = ALLOWLIST_SAFE.copy()
        if additional_safe:
            self.safe_commands.update(additional_safe)

        self.blocked_commands = BLOCKLIST.copy()
        if additional_blocked:
            self.blocked_commands.update(additional_blocked)

        self.conditional_commands = ALLOWLIST_CONDITIONAL if allow_conditionals else set()
        self.strict_mode = strict_mode

        # Compile patterns for efficiency
        self._compiled_patterns: List[Tuple[re.Pattern, str]] = [
            (re.compile(pattern, re.IGNORECASE), description)
            for pattern, description in DANGEROUS_PATTERNS
        ]

    def analyze(self, command: str) -> CommandAnalysis:
        """
        Analyze a command for security risks.

        Args:
            command: The bash command to analyze

        Returns:
            CommandAnalysis with risk assessment
        """
        command = command.strip()

        if not command:
            return CommandAnalysis(
                command=command,
                risk=CommandRisk.SAFE,
                allowed=True,
                reason="Empty command",
            )

        # Check for dangerous patterns first (highest priority)
        for pattern, description in self._compiled_patterns:
            match = pattern.search(command)
            if match:
                return CommandAnalysis(
                    command=command,
                    risk=CommandRisk.BLOCKED,
                    allowed=False,
                    reason=f"Dangerous pattern detected: {description}",
                    matched_pattern=pattern.pattern,
                )

        # Parse command to get base command
        try:
            # Handle Windows-style paths
            normalized_command = command.replace("\\", "/")
            parts = shlex.split(normalized_command)
        except ValueError as e:
            # If shlex fails, be conservative
            return CommandAnalysis(
                command=command,
                risk=CommandRisk.DANGEROUS,
                allowed=False,
                reason=f"Failed to parse command: {str(e)}",
            )

        if not parts:
            return CommandAnalysis(
                command=command,
                risk=CommandRisk.SAFE,
                allowed=True,
                reason="Empty parsed command",
            )

        base_command = parts[0]

        # Handle path prefixes (e.g., /usr/bin/rm -> rm)
        if "/" in base_command:
            base_command = base_command.split("/")[-1]

        # Handle .exe suffix on Windows
        if base_command.endswith(".exe"):
            base_command = base_command[:-4]

        # Check blocklist first (highest priority after patterns)
        if base_command.lower() in self.blocked_commands:
            return CommandAnalysis(
                command=command,
                risk=CommandRisk.BLOCKED,
                allowed=False,
                reason=f"Command '{base_command}' is blocked",
                matched_pattern=base_command,
            )

        # Check safe list
        if base_command.lower() in self.safe_commands:
            # Additional safety check for safe commands with dangerous args
            arg_check = self._check_safe_command_args(base_command, parts[1:])
            if arg_check:
                return arg_check

            return CommandAnalysis(
                command=command,
                risk=CommandRisk.SAFE,
                allowed=True,
                reason=f"Command '{base_command}' is in safe list",
            )

        # Check conditional list
        if base_command.lower() in self.conditional_commands:
            # Additional checks for conditional commands
            risk = self._analyze_conditional(base_command.lower(), parts[1:])
            return CommandAnalysis(
                command=command,
                risk=risk,
                allowed=risk != CommandRisk.BLOCKED,
                reason=f"Conditional command '{base_command}' analyzed",
            )

        # Unknown command handling
        if self.strict_mode:
            return CommandAnalysis(
                command=command,
                risk=CommandRisk.DANGEROUS,
                allowed=False,
                reason=f"Unknown command '{base_command}' (strict mode)",
            )
        else:
            return CommandAnalysis(
                command=command,
                risk=CommandRisk.MODERATE,
                allowed=True,
                reason=f"Unknown command '{base_command}' allowed (non-strict)",
            )

    def _check_safe_command_args(
        self, command: str, args: List[str]
    ) -> Optional[CommandAnalysis]:
        """
        Check arguments of safe commands for dangerous patterns.

        Returns CommandAnalysis if blocked, None if safe.
        """
        command_lower = command.lower()

        # Git: check for potentially dangerous operations
        if command_lower == "git":
            dangerous_git_ops = {
                "push": ["--force", "-f", "--force-with-lease"],
                "reset": ["--hard"],
                "clean": ["-f", "-fd", "-ffd"],
                "checkout": ["."],
            }

            if args:
                subcommand = args[0].lower()
                if subcommand in dangerous_git_ops:
                    for i, arg in enumerate(args[1:], 1):
                        if arg in dangerous_git_ops[subcommand]:
                            # Allow but mark as moderate risk
                            return CommandAnalysis(
                                command=f"{command} {' '.join(args)}",
                                risk=CommandRisk.MODERATE,
                                allowed=True,
                                reason=f"Git {subcommand} with {arg} - use caution",
                            )

        # Tar: check for extraction to sensitive paths
        if command_lower == "tar":
            for arg in args:
                if arg.startswith("/etc") or arg.startswith("/sys"):
                    return CommandAnalysis(
                        command=f"{command} {' '.join(args)}",
                        risk=CommandRisk.BLOCKED,
                        allowed=False,
                        reason="Tar operation targeting sensitive path",
                    )

        return None

    def _analyze_conditional(self, command: str, args: List[str]) -> CommandRisk:
        """
        Analyze conditional commands based on arguments.

        Args:
            command: The base command (lowercase)
            args: Command arguments

        Returns:
            CommandRisk level
        """
        # Sensitive paths that should not be targeted
        sensitive_paths = (
            "/etc", "/sys", "/dev", "/boot", "/proc",
            "/usr/bin", "/usr/sbin", "/bin", "/sbin",
            "/lib", "/lib64", "/root", "C:\\Windows",
            "C:\\Program Files",
        )

        if command in ("cp", "mv"):
            # Check if writing to sensitive paths
            for arg in args:
                if any(arg.startswith(p) or arg.startswith(p.replace("/", "\\"))
                       for p in sensitive_paths):
                    return CommandRisk.BLOCKED
            return CommandRisk.MODERATE

        if command == "mkdir":
            # Generally safe, but check for sensitive paths
            for arg in args:
                if any(arg.startswith(p) for p in sensitive_paths):
                    return CommandRisk.BLOCKED
            return CommandRisk.SAFE

        if command == "touch":
            # Check for sensitive paths
            for arg in args:
                if any(arg.startswith(p) for p in sensitive_paths):
                    return CommandRisk.BLOCKED
            return CommandRisk.SAFE

        if command in ("curl", "wget", "http"):
            # Check for pipe to shell (already caught by patterns, but double-check)
            full_cmd = " ".join(args)
            if "|" in full_cmd and any(sh in full_cmd for sh in ("sh", "bash", "python")):
                return CommandRisk.BLOCKED
            return CommandRisk.SAFE

        if command in ("docker", "podman"):
            # Block privileged containers and host mounts
            if "--privileged" in args:
                return CommandRisk.BLOCKED
            if any("-v" in arg and (":/etc" in arg or ":/root" in arg) for arg in args):
                return CommandRisk.BLOCKED
            return CommandRisk.MODERATE

        if command == "kubectl":
            # Block exec into containers without review
            if "exec" in args:
                return CommandRisk.MODERATE
            return CommandRisk.SAFE

        if command in ("ssh", "scp", "sftp", "rsync"):
            # Allow but mark as moderate risk
            return CommandRisk.MODERATE

        if command == "ln":
            # Symlinks can be used for attacks
            for arg in args:
                if any(arg.startswith(p) for p in sensitive_paths):
                    return CommandRisk.BLOCKED
            return CommandRisk.MODERATE

        return CommandRisk.MODERATE

    def is_allowed(self, command: str) -> bool:
        """
        Quick check if command is allowed.

        Args:
            command: The bash command to check

        Returns:
            True if allowed, False if blocked
        """
        return self.analyze(command).allowed

    def get_reason(self, command: str) -> str:
        """
        Get reason for allow/block decision.

        Args:
            command: The bash command to check

        Returns:
            Explanation of the decision
        """
        return self.analyze(command).reason

    def batch_analyze(self, commands: List[str]) -> List[CommandAnalysis]:
        """
        Analyze multiple commands.

        Args:
            commands: List of bash commands

        Returns:
            List of CommandAnalysis results
        """
        return [self.analyze(cmd) for cmd in commands]

    def filter_allowed(self, commands: List[str]) -> List[str]:
        """
        Filter a list of commands to only allowed ones.

        Args:
            commands: List of bash commands

        Returns:
            List of allowed commands
        """
        return [cmd for cmd in commands if self.is_allowed(cmd)]


# Singleton for hook integration
_default_filter: Optional[BashSecurityFilter] = None


def get_filter(strict_mode: bool = False) -> BashSecurityFilter:
    """
    Get the default security filter.

    Args:
        strict_mode: If True, create/get a strict mode filter

    Returns:
        BashSecurityFilter instance
    """
    global _default_filter
    if _default_filter is None or _default_filter.strict_mode != strict_mode:
        _default_filter = BashSecurityFilter(strict_mode=strict_mode)
    return _default_filter


def check_command(command: str, strict_mode: bool = False) -> Tuple[bool, str]:
    """
    Check if a command is allowed.

    This is the main entry point for hook integration.

    Args:
        command: The bash command to check
        strict_mode: If True, block unknown commands

    Returns:
        Tuple of (allowed: bool, reason: str)

    Example:
        allowed, reason = check_command("rm -rf /")
        # allowed = False
        # reason = "Command 'rm' is blocked"
    """
    result = get_filter(strict_mode).analyze(command)
    return result.allowed, result.reason


def analyze_command(command: str, strict_mode: bool = False) -> CommandAnalysis:
    """
    Get full analysis of a command.

    Args:
        command: The bash command to analyze
        strict_mode: If True, block unknown commands

    Returns:
        CommandAnalysis with full details
    """
    return get_filter(strict_mode).analyze(command)


# Export commonly used sets for extension
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
