# Python getpass Module - Production Patterns (Cycle 119)

## Overview
The `getpass` module provides portable password input without echoing and user identification. Essential for CLI tools requiring secure credential handling.

## Core Functions

### getpass() - Secure Password Input
```python
import getpass

# Basic usage
password = getpass.getpass()  # Prompts "Password: "

# Custom prompt
password = getpass.getpass(prompt="Enter API key: ")

# Python 3.14+ echo_char for visual feedback
password = getpass.getpass(prompt="Password: ", echo_char="*")
# Shows: Password: ******* (asterisks as user types)

# Custom stream (Unix only, ignored on Windows)
import sys
password = getpass.getpass(stream=sys.stderr)
```

### getuser() - Get Current Username
```python
import getpass

# Get login name (preferred over os.getlogin())
username = getpass.getuser()

# Environment variable priority:
# 1. LOGNAME
# 2. USER
# 3. LNAME
# 4. USERNAME
# Falls back to pwd module on Unix if none set
```

### GetPassWarning Exception
```python
import getpass
import warnings

# Catch when echo-free input unavailable
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    password = getpass.getpass()
    
    if w and issubclass(w[-1].category, getpass.GetPassWarning):
        print("Warning: Password may have been visible!")
```

## Production Patterns

### Pattern 1: Secure Credential Prompt with Validation
```python
import getpass
import sys
from typing import Callable

def prompt_credential(
    prompt: str = "Password: ",
    validator: Callable[[str], bool] | None = None,
    max_attempts: int = 3,
    min_length: int = 8,
    echo_char: str | None = None  # Python 3.14+
) -> str | None:
    """Securely prompt for credential with validation."""
    
    for attempt in range(max_attempts):
        try:
            # Python 3.14+ supports echo_char
            kwargs = {"prompt": prompt}
            if echo_char and sys.version_info >= (3, 14):
                kwargs["echo_char"] = echo_char
            
            credential = getpass.getpass(**kwargs)
            
            if not credential:
                print("Error: Empty input not allowed.", file=sys.stderr)
                continue
            
            if len(credential) < min_length:
                print(f"Error: Must be at least {min_length} characters.", 
                      file=sys.stderr)
                continue
            
            if validator and not validator(credential):
                print("Error: Validation failed.", file=sys.stderr)
                continue
            
            return credential
            
        except (KeyboardInterrupt, EOFError):
            print("\nInput cancelled.", file=sys.stderr)
            return None
    
    print(f"Error: Maximum attempts ({max_attempts}) exceeded.", file=sys.stderr)
    return None

# Usage with custom validator
def has_special_char(s: str) -> bool:
    return any(c in s for c in "!@#$%^&*()_+-=[]{}|;:,.<>?")

password = prompt_credential(
    prompt="New password: ",
    validator=has_special_char,
    min_length=12,
    echo_char="*"  # Python 3.14+
)
```

### Pattern 2: API Key Configuration CLI
```python
import getpass
import os
import json
from pathlib import Path
from dataclasses import dataclass

@dataclass
class APIConfig:
    api_key: str
    api_secret: str | None = None
    username: str | None = None

def configure_api(
    config_path: Path,
    service_name: str = "API"
) -> APIConfig:
    """Interactive API configuration with secure input."""
    
    print(f"=== {service_name} Configuration ===")
    print("Credentials will be stored securely.\n")
    
    # Get username (with default from system)
    default_user = getpass.getuser()
    username = input(f"Username [{default_user}]: ").strip() or default_user
    
    # Get API key (hidden)
    api_key = getpass.getpass(f"{service_name} API Key: ")
    if not api_key:
        raise ValueError("API key is required")
    
    # Optional API secret
    api_secret = getpass.getpass(f"{service_name} API Secret (optional): ")
    
    config = APIConfig(
        api_key=api_key,
        api_secret=api_secret if api_secret else None,
        username=username
    )
    
    # Save config (in practice, encrypt this!)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({
        "username": config.username,
        "api_key": config.api_key,  # Should encrypt!
        "api_secret": config.api_secret
    }))
    config_path.chmod(0o600)  # Restrict permissions
    
    print(f"\nConfiguration saved to {config_path}")
    return config

# Usage
config = configure_api(
    Path.home() / ".config" / "myapp" / "credentials.json",
    service_name="OpenAI"
)
```

### Pattern 3: SSH-Style Password Confirmation
```python
import getpass
import hashlib
import secrets

def set_password_with_confirmation(
    prompt1: str = "New password: ",
    prompt2: str = "Confirm password: ",
    hash_algorithm: str = "sha256"
) -> tuple[str, str] | None:
    """Set password with confirmation, returns (hash, salt)."""
    
    password1 = getpass.getpass(prompt1)
    if not password1:
        print("Error: Password cannot be empty.", file=sys.stderr)
        return None
    
    password2 = getpass.getpass(prompt2)
    
    if password1 != password2:
        print("Error: Passwords do not match.", file=sys.stderr)
        return None
    
    # Generate salt and hash
    salt = secrets.token_hex(32)
    hash_input = (salt + password1).encode('utf-8')
    password_hash = hashlib.new(hash_algorithm, hash_input).hexdigest()
    
    return password_hash, salt

def verify_password(
    password: str, 
    stored_hash: str, 
    salt: str,
    hash_algorithm: str = "sha256"
) -> bool:
    """Verify password against stored hash."""
    hash_input = (salt + password).encode('utf-8')
    computed_hash = hashlib.new(hash_algorithm, hash_input).hexdigest()
    return secrets.compare_digest(computed_hash, stored_hash)

# Usage
result = set_password_with_confirmation()
if result:
    password_hash, salt = result
    print(f"Password hash: {password_hash[:16]}...")
    
    # Later verification
    test_password = getpass.getpass("Enter password to verify: ")
    if verify_password(test_password, password_hash, salt):
        print("Password verified!")
```

### Pattern 4: Environment-Aware Credential Loading
```python
import getpass
import os
from typing import NamedTuple

class Credentials(NamedTuple):
    username: str
    password: str
    source: str  # Where credentials came from

def get_credentials(
    env_user: str = "APP_USERNAME",
    env_pass: str = "APP_PASSWORD",
    prompt_user: str = "Username: ",
    prompt_pass: str = "Password: ",
    allow_interactive: bool = True
) -> Credentials | None:
    """Get credentials from environment or interactive prompt."""
    
    # Try environment first (CI/CD, containers)
    env_username = os.environ.get(env_user)
    env_password = os.environ.get(env_pass)
    
    if env_username and env_password:
        return Credentials(
            username=env_username,
            password=env_password,
            source="environment"
        )
    
    # Check if running in interactive terminal
    if not allow_interactive:
        return None
    
    # Check if stdin is a TTY
    import sys
    if not sys.stdin.isatty():
        print("Error: No TTY available for interactive input.", 
              file=sys.stderr)
        return None
    
    try:
        # Get username (visible) with default
        default_user = env_username or getpass.getuser()
        username = input(f"{prompt_user}[{default_user}]: ").strip()
        username = username or default_user
        
        # Get password (hidden)
        password = getpass.getpass(prompt_pass)
        
        if not password:
            print("Error: Password required.", file=sys.stderr)
            return None
        
        return Credentials(
            username=username,
            password=password,
            source="interactive"
        )
        
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.", file=sys.stderr)
        return None

# Usage
creds = get_credentials(
    env_user="DB_USER",
    env_pass="DB_PASSWORD"
)
if creds:
    print(f"Got credentials from: {creds.source}")
```

### Pattern 5: Sudo-Style Re-authentication
```python
import getpass
import time
from functools import wraps
from typing import Callable, TypeVar

T = TypeVar('T')

class AuthSession:
    """Session with timed re-authentication requirement."""
    
    def __init__(
        self, 
        verify_func: Callable[[str, str], bool],
        timeout_seconds: int = 300  # 5 minutes
    ):
        self.verify_func = verify_func
        self.timeout = timeout_seconds
        self._last_auth: float = 0
        self._username: str | None = None
    
    def authenticate(self) -> bool:
        """Prompt for credentials and verify."""
        username = input("Username: ").strip()
        password = getpass.getpass()
        
        if self.verify_func(username, password):
            self._last_auth = time.time()
            self._username = username
            return True
        return False
    
    def is_authenticated(self) -> bool:
        """Check if session is still valid."""
        if not self._last_auth:
            return False
        return (time.time() - self._last_auth) < self.timeout
    
    def require_auth(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator requiring valid authentication."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.is_authenticated():
                print("Authentication required.")
                if not self.authenticate():
                    raise PermissionError("Authentication failed")
            return func(*args, **kwargs)
        return wrapper

# Usage
def verify_credentials(username: str, password: str) -> bool:
    # In practice, check against secure storage
    return username == "admin" and password == "secret"

session = AuthSession(verify_credentials, timeout_seconds=60)

@session.require_auth
def sensitive_operation():
    print(f"Performing sensitive operation as {session._username}")

# Will prompt for credentials if session expired
sensitive_operation()
```

## Platform Considerations

### Unix/Linux
- Uses `/dev/tty` for input (bypasses redirected stdin)
- Falls back to `sys.stdin` with warning if no TTY
- `stream` parameter controls where prompt is written

### Windows
- Uses `msvcrt` module for character-by-character input
- `stream` parameter is ignored
- Works in CMD, PowerShell, and Windows Terminal

### IDLE/IDE Limitations
- Input may appear in launching terminal, not IDE window
- Consider falling back to visible input with warning

```python
import sys
import getpass

def safe_getpass(prompt: str = "Password: ") -> str:
    """getpass with IDE fallback."""
    # Check if running in IDLE or similar
    if 'idlelib' in sys.modules:
        import warnings
        warnings.warn("Password will be visible (IDE environment)")
        return input(prompt)
    return getpass.getpass(prompt)
```

## Python 3.14 New Feature: echo_char

```python
import getpass

# Visual feedback while typing (new in 3.14)
password = getpass.getpass(
    prompt="Password: ",
    echo_char="*"  # Shows asterisks
)

# Alternative characters
password = getpass.getpass(echo_char="●")  # Bullets
password = getpass.getpass(echo_char="•")  # Dots

# Note: Disables line editing (Ctrl+U etc.) in noncanonical mode
```

## Security Best Practices

1. **Never log passwords**: Even masked, avoid logging credential-related operations
2. **Clear from memory**: In sensitive apps, overwrite password strings when done
3. **Use environment variables in CI/CD**: Never prompt in automated environments
4. **Check TTY availability**: Fail gracefully when no terminal available
5. **Validate input length**: Prevent buffer issues with reasonable limits
6. **Use secrets.compare_digest()**: Prevent timing attacks in password verification

## Key Insights

1. **getpass vs input**: getpass prevents shoulder surfing and terminal history exposure
2. **getuser() preferred over os.getlogin()**: More reliable across platforms
3. **echo_char (3.14+)**: Provides UX feedback without revealing actual characters
4. **Stream parameter Unix-only**: Windows ignores it, writes to console directly
5. **GetPassWarning**: Always handle - indicates potential security issue
