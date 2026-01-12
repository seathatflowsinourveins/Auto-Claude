"""
Authentication helpers for Auto Claude.

Provides centralized authentication token resolution with fallback support
for multiple environment variables, and SDK environment variable passthrough
for custom API endpoints.
"""

import json
import os
import platform
import subprocess

# Priority order for auth token resolution
# NOTE: We intentionally do NOT fall back to ANTHROPIC_API_KEY.
# Auto Claude is designed to use Claude Code OAuth tokens only.
# This prevents silent billing to user's API credits when OAuth fails.
AUTH_TOKEN_ENV_VARS = [
    "CLAUDE_CODE_OAUTH_TOKEN",  # OAuth token from Claude Code CLI
    "ANTHROPIC_AUTH_TOKEN",  # CCR/proxy token (for enterprise setups)
]

# Environment variables to pass through to SDK subprocess
# NOTE: ANTHROPIC_API_KEY is intentionally excluded to prevent silent API billing
SDK_ENV_VARS = [
    # API endpoint configuration
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    # Model overrides (from API Profile custom model mappings)
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    # SDK behavior configuration
    "NO_PROXY",
    "DISABLE_TELEMETRY",
    "DISABLE_COST_WARNINGS",
    "API_TIMEOUT_MS",
    # Windows-specific: Git Bash path for Claude Code CLI
    "CLAUDE_CODE_GIT_BASH_PATH",
]


def get_token_from_keychain() -> str | None:
    """
    Get authentication token from system credential store.

    Reads Claude Code credentials from:
    - macOS: Keychain
    - Windows: Credential Manager
    - Linux: Not yet supported (use env var)

    Returns:
        Token string if found, None otherwise
    """
    system = platform.system()

    if system == "Darwin":
        return _get_token_from_macos_keychain()
    elif system == "Windows":
        return _get_token_from_windows_credential_files()
    else:
        # Linux: secret-service not yet implemented
        return None


def _get_token_from_macos_keychain() -> str | None:
    """Get token from macOS Keychain."""
    try:
        result = subprocess.run(
            [
                "/usr/bin/security",
                "find-generic-password",
                "-s",
                "Claude Code-credentials",
                "-w",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return None

        credentials_json = result.stdout.strip()
        if not credentials_json:
            return None

        data = json.loads(credentials_json)
        token = data.get("claudeAiOauth", {}).get("accessToken")

        if not token:
            return None

        # Validate token format (Claude OAuth tokens start with sk-ant-oat01-)
        if not token.startswith("sk-ant-oat01-"):
            return None

        return token

    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, Exception):
        return None


def _get_token_from_windows_credential_files() -> str | None:
    """Get token from Windows credential files.

    Claude Code on Windows stores credentials in ~/.claude/.credentials.json
    """
    try:
        # Claude Code stores credentials in ~/.claude/.credentials.json
        cred_paths = [
            os.path.expandvars(r"%USERPROFILE%\.claude\.credentials.json"),
            os.path.expandvars(r"%USERPROFILE%\.claude\credentials.json"),
            os.path.expandvars(r"%LOCALAPPDATA%\Claude\credentials.json"),
            os.path.expandvars(r"%APPDATA%\Claude\credentials.json"),
        ]

        for cred_path in cred_paths:
            if os.path.exists(cred_path):
                with open(cred_path, encoding="utf-8") as f:
                    data = json.load(f)
                    token = data.get("claudeAiOauth", {}).get("accessToken")
                    if token and token.startswith("sk-ant-oat01-"):
                        return token

        return None

    except (json.JSONDecodeError, KeyError, FileNotFoundError, Exception):
        return None


def get_auth_token() -> str | None:
    """
    Get authentication token from environment variables or system credential store.

    Checks multiple sources in priority order:
    1. CLAUDE_CODE_OAUTH_TOKEN (env var)
    2. ANTHROPIC_AUTH_TOKEN (CCR/proxy env var for enterprise setups)
    3. System credential store (macOS Keychain, Windows Credential Manager)

    NOTE: ANTHROPIC_API_KEY is intentionally NOT supported to prevent
    silent billing to user's API credits when OAuth is misconfigured.

    Returns:
        Token string if found, None otherwise
    """
    # First check environment variables
    for var in AUTH_TOKEN_ENV_VARS:
        token = os.environ.get(var)
        if token:
            return token

    # Fallback to system credential store
    return get_token_from_keychain()


def get_auth_token_source() -> str | None:
    """Get the name of the source that provided the auth token."""
    # Check environment variables first
    for var in AUTH_TOKEN_ENV_VARS:
        if os.environ.get(var):
            return var

    # Check if token came from system credential store
    if get_token_from_keychain():
        system = platform.system()
        if system == "Darwin":
            return "macOS Keychain"
        elif system == "Windows":
            return "Windows Credential Files"
        else:
            return "System Credential Store"

    return None


def require_auth_token() -> str:
    """
    Get authentication token or raise ValueError.

    Raises:
        ValueError: If no auth token is found in any supported source
    """
    token = get_auth_token()
    if not token:
        error_msg = (
            "No OAuth token found.\n\n"
            "Auto Claude requires Claude Code OAuth authentication.\n"
            "Direct API keys (ANTHROPIC_API_KEY) are not supported.\n\n"
        )
        # Provide platform-specific guidance
        system = platform.system()
        if system == "Darwin":
            error_msg += (
                "To authenticate:\n"
                "  1. Run: claude\n"
                "  2. Type: /login\n"
                "  3. Press Enter to open browser\n"
                "  4. Complete OAuth login in browser\n\n"
                "The token will be saved to macOS Keychain automatically."
            )
        elif system == "Windows":
            error_msg += (
                "To authenticate:\n"
                "  1. Run: claude\n"
                "  2. Type: /login\n"
                "  3. Press Enter to open browser\n"
                "  4. Complete OAuth login in browser\n\n"
                "The token will be saved to Windows Credential Manager."
            )
        else:
            error_msg += (
                "To authenticate:\n"
                "  1. Run: claude\n"
                "  2. Type: /login\n"
                "  3. Press Enter to open browser\n"
                "  4. Complete OAuth login in browser\n\n"
                "Or set CLAUDE_CODE_OAUTH_TOKEN in your .env file."
            )
        raise ValueError(error_msg)
    return token


def _find_git_bash_path() -> str | None:
    """
    Find git-bash (bash.exe) path on Windows.

    Uses 'where git' to find git.exe, then derives bash.exe location from it.
    Git for Windows installs bash.exe in the 'bin' directory alongside git.exe
    or in the parent 'bin' directory when git.exe is in 'cmd'.

    Returns:
        Full path to bash.exe if found, None otherwise
    """
    if platform.system() != "Windows":
        return None

    # If already set in environment, use that
    existing = os.environ.get("CLAUDE_CODE_GIT_BASH_PATH")
    if existing and os.path.exists(existing):
        return existing

    git_path = None

    # Method 1: Use 'where' command to find git.exe
    try:
        # Use where.exe explicitly for reliability
        result = subprocess.run(
            ["where.exe", "git"],
            capture_output=True,
            text=True,
            timeout=5,
            shell=False,
        )

        if result.returncode == 0 and result.stdout.strip():
            git_paths = result.stdout.strip().splitlines()
            if git_paths:
                git_path = git_paths[0].strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        # Intentionally suppress errors - best-effort detection with fallback to common paths
        pass

    # Method 2: Check common installation paths if 'where' didn't work
    if not git_path:
        common_git_paths = [
            os.path.expandvars(r"%PROGRAMFILES%\Git\cmd\git.exe"),
            os.path.expandvars(r"%PROGRAMFILES%\Git\bin\git.exe"),
            os.path.expandvars(r"%PROGRAMFILES(X86)%\Git\cmd\git.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Git\cmd\git.exe"),
        ]
        for path in common_git_paths:
            if os.path.exists(path):
                git_path = path
                break

    if not git_path:
        return None

    # Derive bash.exe location from git.exe location
    # Git for Windows structure:
    #   C:\...\Git\cmd\git.exe     -> bash.exe is at C:\...\Git\bin\bash.exe
    #   C:\...\Git\bin\git.exe     -> bash.exe is at C:\...\Git\bin\bash.exe
    #   C:\...\Git\mingw64\bin\git.exe -> bash.exe is at C:\...\Git\bin\bash.exe
    git_dir = os.path.dirname(git_path)
    git_parent = os.path.dirname(git_dir)
    git_grandparent = os.path.dirname(git_parent)

    # Check common bash.exe locations relative to git installation
    possible_bash_paths = [
        os.path.join(git_parent, "bin", "bash.exe"),  # cmd -> bin
        os.path.join(git_dir, "bash.exe"),  # If git.exe is in bin
        os.path.join(git_grandparent, "bin", "bash.exe"),  # mingw64/bin -> bin
    ]

    for bash_path in possible_bash_paths:
        if os.path.exists(bash_path):
            return bash_path

    return None


def get_sdk_env_vars() -> dict[str, str]:
    """
    Get environment variables to pass to SDK.

    Collects relevant env vars (ANTHROPIC_BASE_URL, etc.) that should
    be passed through to the claude-agent-sdk subprocess.

    On Windows, auto-detects CLAUDE_CODE_GIT_BASH_PATH if not already set.

    Returns:
        Dict of env var name -> value for non-empty vars
    """
    env = {}
    for var in SDK_ENV_VARS:
        value = os.environ.get(var)
        if value:
            env[var] = value

    # On Windows, auto-detect git-bash path if not already set
    # Claude Code CLI requires bash.exe to run on Windows
    if platform.system() == "Windows" and "CLAUDE_CODE_GIT_BASH_PATH" not in env:
        bash_path = _find_git_bash_path()
        if bash_path:
            env["CLAUDE_CODE_GIT_BASH_PATH"] = bash_path

    return env


def ensure_claude_code_oauth_token() -> None:
    """
    Ensure CLAUDE_CODE_OAUTH_TOKEN is set (for SDK compatibility).

    If not set but other auth tokens are available, copies the value
    to CLAUDE_CODE_OAUTH_TOKEN so the underlying SDK can use it.
    """
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return

    token = get_auth_token()
    if token:
        os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = token


def trigger_login() -> bool:
    """
    Trigger Claude Code OAuth login flow.

    Opens the Claude Code CLI and sends /login command to initiate
    browser-based OAuth authentication. The token is automatically
    saved to the system credential store (macOS Keychain, Windows
    Credential Manager).

    Returns:
        True if login was successful, False otherwise
    """
    system = platform.system()

    if system == "Darwin":
        return _trigger_login_macos()
    elif system == "Windows":
        return _trigger_login_windows()
    else:
        # Linux: fall back to manual instructions
        print("\nTo authenticate, run 'claude' and type '/login'")
        return False


def _trigger_login_macos() -> bool:
    """Trigger login on macOS using expect."""
    import shutil
    import tempfile

    # Check if expect is available
    if not shutil.which("expect"):
        print("\nTo authenticate, run 'claude' and type '/login'")
        return False

    # Create expect script
    expect_script = """#!/usr/bin/expect -f
set timeout 120
spawn claude
expect {
    -re ".*" {
        send "/login\\r"
        expect {
            "Press Enter" {
                send "\\r"
            }
            -re ".*login.*" {
                send "\\r"
            }
            timeout {
                send "\\r"
            }
        }
    }
}
# Keep running until user completes login or exits
interact
"""

    try:
        # Write expect script to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".exp", delete=False) as f:
            f.write(expect_script)
            script_path = f.name

        os.chmod(script_path, 0o755)

        print("\n" + "=" * 60)
        print("CLAUDE CODE LOGIN")
        print("=" * 60)
        print("\nOpening Claude Code for authentication...")
        print("A browser window will open for OAuth login.")
        print("After completing login in the browser, press Ctrl+C to exit.\n")

        # Run expect script
        result = subprocess.run(
            ["expect", script_path],
            timeout=300,  # 5 minute timeout
        )

        # Clean up
        os.unlink(script_path)

        # Verify token was saved
        token = get_token_from_keychain()
        if token:
            print("\n✓ Login successful! Token saved to macOS Keychain.")
            return True
        else:
            print(
                "\n✗ Login may not have completed. Try running 'claude' and type '/login'"
            )
            return False

    except subprocess.TimeoutExpired:
        print("\nLogin timed out. Try running 'claude' manually and type '/login'")
        return False
    except KeyboardInterrupt:
        # User pressed Ctrl+C - check if login completed
        token = get_token_from_keychain()
        if token:
            print("\n✓ Login successful! Token saved to macOS Keychain.")
            return True
        return False
    except Exception as e:
        print(f"\nLogin failed: {e}")
        print("Try running 'claude' manually and type '/login'")
        return False


def _trigger_login_windows() -> bool:
    """Trigger login on Windows."""
    # Windows doesn't have expect by default, so we use a simpler approach
    # that just launches claude and tells the user what to type
    print("\n" + "=" * 60)
    print("CLAUDE CODE LOGIN")
    print("=" * 60)
    print("\nLaunching Claude Code...")
    print("Please type '/login' and press Enter.")
    print("A browser window will open for OAuth login.\n")

    try:
        # Launch claude interactively
        subprocess.run(["claude"], timeout=300)

        # Verify token was saved
        token = _get_token_from_windows_credential_files()
        if token:
            print("\n✓ Login successful!")
            return True
        else:
            print("\n✗ Login may not have completed.")
            return False

    except Exception as e:
        print(f"\nLogin failed: {e}")
        return False


def ensure_authenticated() -> str:
    """
    Ensure the user is authenticated, prompting for login if needed.

    Checks for existing token and triggers login flow if not found.

    Returns:
        The authentication token

    Raises:
        ValueError: If authentication fails after login attempt
    """
    # First check if already authenticated
    token = get_auth_token()
    if token:
        return token

    # No token found - trigger login
    print("\nNo OAuth token found. Starting login flow...")

    if trigger_login():
        # Re-check for token after login
        token = get_auth_token()
        if token:
            return token

    # Login failed or was cancelled
    raise ValueError(
        "Authentication required.\n\n"
        "To authenticate:\n"
        "  1. Run: claude\n"
        "  2. Type: /login\n"
        "  3. Press Enter to open browser\n"
        "  4. Complete OAuth login in browser"
    )
