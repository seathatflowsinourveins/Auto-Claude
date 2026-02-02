# Python warnings Module - Production Patterns (Cycle 118)

## Overview
The `warnings` module controls warning message display, filtering, and conversion to exceptions. Essential for deprecation management, API evolution, and production alerting.

## Warning Categories Hierarchy

```python
# Built-in warning categories (all subclass Warning → Exception)
Warning                    # Base class for all warnings
├── UserWarning            # Default for warn(), user-issued warnings
├── DeprecationWarning     # For other developers (ignored by default except __main__)
├── PendingDeprecationWarning  # Future deprecations (ignored by default)
├── SyntaxWarning          # Dubious syntax (emitted at compile time)
├── RuntimeWarning         # Dubious runtime behavior
├── FutureWarning          # For end users of applications
├── ImportWarning          # Import process issues (ignored by default)
├── UnicodeWarning         # Unicode-related issues
├── BytesWarning           # bytes/bytearray issues
└── ResourceWarning        # Resource leaks (ignored by default)

# Custom warning category
class SecurityWarning(UserWarning):
    """Custom warning for security-related issues."""
    pass
```

## Filter Actions

| Action | Behavior |
|--------|----------|
| `"default"` | Print first occurrence per location (module + line) |
| `"error"` | Convert warnings to exceptions |
| `"ignore"` | Never print matching warnings |
| `"always"` / `"all"` | Always print matching warnings |
| `"module"` | Print first occurrence per module (ignore line) |
| `"once"` | Print only first occurrence globally |

## Core Functions

### warn() - Issue Warnings
```python
import warnings

# Basic warning
warnings.warn("This feature is deprecated", DeprecationWarning)

# With stacklevel for wrapper functions
def deprecated_api(message):
    warnings.warn(message, DeprecationWarning, stacklevel=2)

# Python 3.12+ skip_file_prefixes for package-level warnings
import os
_warn_skips = (os.path.dirname(__file__),)

def internal_function():
    warnings.warn(
        "Please use new_function() instead",
        DeprecationWarning,
        skip_file_prefixes=_warn_skips  # Attributes to caller outside package
    )
```

### filterwarnings() vs simplefilter()
```python
# filterwarnings - Full regex control
warnings.filterwarnings(
    "error",                    # action
    message=".*deprecated.*",   # regex for message
    category=DeprecationWarning,
    module="mypackage.*",       # regex for module
    lineno=0,                   # 0 = all lines
    append=False                # Insert at front (default)
)

# simplefilter - Quick setup (no regex)
warnings.simplefilter("ignore", ResourceWarning)
warnings.simplefilter("error")  # Turn all warnings into errors

# Reset all filters
warnings.resetwarnings()
```

### formatwarning() and showwarning()
```python
# Custom warning format
def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    formatted = f"[{category.__name__}] {filename}:{lineno}: {message}\n"
    if file is None:
        file = sys.stderr
    file.write(formatted)

# Override default
warnings.showwarning = custom_showwarning

# Get formatted string
formatted = warnings.formatwarning(
    "Something happened", 
    UserWarning, 
    "script.py", 
    42
)
```

## catch_warnings Context Manager

```python
import warnings

# Temporarily suppress warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    legacy_function()  # Warnings suppressed here

# Record warnings for testing
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    deprecated_function()
    
    assert len(w) == 1
    assert issubclass(w[0].category, DeprecationWarning)
    assert "deprecated" in str(w[0].message)

# Python 3.11+ shorthand with action parameter
with warnings.catch_warnings(action="ignore"):
    noisy_library_call()

# Turn warnings to errors temporarily
with warnings.catch_warnings(action="error", category=DeprecationWarning):
    try:
        old_api()
    except DeprecationWarning as e:
        handle_deprecation(e)
```

## @deprecated Decorator (Python 3.13+)

```python
from warnings import deprecated
from typing import overload

@deprecated("Use NewClass instead")
class OldClass:
    pass

@deprecated("Use new_function() instead")
def old_function():
    pass

# With overloads
@overload
@deprecated("int support is deprecated")
def process(x: int) -> int: ...

@overload
def process(x: str) -> str: ...

def process(x):
    return x

# Custom category and stacklevel
@deprecated("Legacy API", category=FutureWarning, stacklevel=2)
def legacy_api():
    pass

# Suppress runtime warning (static type checkers still warn)
@deprecated("Only for type checking", category=None)
def type_check_only():
    pass

# Access deprecation message
print(OldClass.__deprecated__)  # "Use NewClass instead"
```

## Python 3.14+ Thread Safety

```python
import sys

# Check if context-aware warnings enabled
if sys.flags.context_aware_warnings:
    # Thread-safe behavior using ContextVar
    # catch_warnings uses thread-local storage
    pass

# Enable via command line: python -X context_aware_warnings
# Enable via env: PYTHON_CONTEXT_AWARE_WARNINGS=1

# For thread inheritance (threads inherit warning context)
# python -X thread_inherit_context
```

## Default Filter Configuration

```python
# Python's default filters (in order):
# default::DeprecationWarning:__main__  <- Show in main script
# ignore::DeprecationWarning            <- Ignore elsewhere
# ignore::PendingDeprecationWarning
# ignore::ImportWarning
# ignore::ResourceWarning

# Check current filters
print(warnings.filters)

# Override for applications (hide warnings from users)
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Override for testing (show all warnings)
import os
if not sys.warnoptions:
    warnings.simplefilter("default")
    os.environ["PYTHONWARNINGS"] = "default"  # Affect subprocesses
```

## Production Patterns

### Pattern 1: Deprecation Manager
```python
import warnings
import functools
from typing import Callable, TypeVar

T = TypeVar('T', bound=Callable)

class DeprecationManager:
    """Centralized deprecation tracking and migration assistance."""
    
    def __init__(self, package_name: str):
        self.package_name = package_name
        self._deprecations: list[tuple[str, str]] = []
    
    def deprecated(
        self, 
        reason: str, 
        replacement: str | None = None,
        remove_in: str | None = None
    ) -> Callable[[T], T]:
        """Decorator for deprecated functions with migration guidance."""
        def decorator(func: T) -> T:
            msg = f"{func.__qualname__} is deprecated"
            if reason:
                msg += f": {reason}"
            if replacement:
                msg += f". Use {replacement} instead"
            if remove_in:
                msg += f". Will be removed in {remove_in}"
            
            self._deprecations.append((func.__qualname__, msg))
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
                return func(*args, **kwargs)
            
            wrapper.__deprecated__ = msg
            return wrapper
        return decorator
    
    def report(self) -> str:
        """Generate deprecation report for documentation."""
        lines = [f"# Deprecated APIs in {self.package_name}\n"]
        for name, msg in self._deprecations:
            lines.append(f"- `{name}`: {msg}")
        return "\n".join(lines)

# Usage
deprecation = DeprecationManager("mypackage")

@deprecation.deprecated(
    reason="Performance issues",
    replacement="fast_process()",
    remove_in="v3.0"
)
def slow_process(data):
    return data
```

### Pattern 2: Warning Aggregator for Logging
```python
import warnings
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

@dataclass
class WarningStats:
    count: int = 0
    locations: set = field(default_factory=set)
    first_message: str = ""

class WarningAggregator:
    """Aggregate warnings for structured logging instead of stderr spam."""
    
    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        self.stats: dict[type, WarningStats] = defaultdict(WarningStats)
        self._original_showwarning = None
    
    def _custom_showwarning(
        self, message, category, filename, lineno, file=None, line=None
    ):
        """Capture warnings instead of printing."""
        stats = self.stats[category]
        stats.count += 1
        stats.locations.add(f"{filename}:{lineno}")
        if not stats.first_message:
            stats.first_message = str(message)
        
        # Log at appropriate level
        if category in (DeprecationWarning, PendingDeprecationWarning):
            self.logger.warning(f"{category.__name__}: {message} at {filename}:{lineno}")
        elif category in (ResourceWarning,):
            self.logger.error(f"{category.__name__}: {message} at {filename}:{lineno}")
        else:
            self.logger.info(f"{category.__name__}: {message} at {filename}:{lineno}")
    
    def __enter__(self):
        self._original_showwarning = warnings.showwarning
        warnings.showwarning = self._custom_showwarning
        warnings.simplefilter("always")
        return self
    
    def __exit__(self, *args):
        warnings.showwarning = self._original_showwarning
    
    def summary(self) -> dict[str, Any]:
        """Return warning summary for monitoring."""
        return {
            category.__name__: {
                "count": stats.count,
                "unique_locations": len(stats.locations),
                "sample": stats.first_message
            }
            for category, stats in self.stats.items()
        }

# Usage
with WarningAggregator() as aggregator:
    run_application()

print(aggregator.summary())
# {'DeprecationWarning': {'count': 15, 'unique_locations': 3, 'sample': '...'}}
```

### Pattern 3: Test Warning Assertions
```python
import warnings
import pytest
from contextlib import contextmanager

@contextmanager
def assert_warns(
    expected_warning: type[Warning],
    match: str | None = None,
    count: int | None = None
):
    """Context manager for precise warning assertions in tests."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        yield recorded
    
    # Filter to expected category
    matching = [w for w in recorded if issubclass(w.category, expected_warning)]
    
    if not matching:
        raise AssertionError(
            f"Expected {expected_warning.__name__} but got: "
            f"{[w.category.__name__ for w in recorded]}"
        )
    
    if match:
        matched = [w for w in matching if match in str(w.message)]
        if not matched:
            messages = [str(w.message) for w in matching]
            raise AssertionError(f"No warning matched '{match}'. Got: {messages}")
    
    if count is not None and len(matching) != count:
        raise AssertionError(
            f"Expected {count} warnings, got {len(matching)}"
        )

# Usage in tests
def test_deprecation_warning():
    with assert_warns(DeprecationWarning, match="old_function", count=1):
        old_function()

# pytest integration
def test_with_pytest():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        deprecated_api()
```

### Pattern 4: Environment-Aware Warning Configuration
```python
import warnings
import os
import sys

def configure_warnings(environment: str = None):
    """Configure warnings based on environment."""
    env = environment or os.environ.get("ENVIRONMENT", "development")
    
    # Reset to clean state
    warnings.resetwarnings()
    
    if env == "production":
        # Production: Log critical, ignore noise
        warnings.filterwarnings("error", category=ResourceWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        
    elif env == "staging":
        # Staging: Surface deprecations for monitoring
        warnings.filterwarnings("default", category=DeprecationWarning)
        warnings.filterwarnings("error", category=ResourceWarning)
        
    elif env == "testing":
        # Testing: All warnings visible, some as errors
        warnings.filterwarnings("always")
        warnings.filterwarnings("error", category=ResourceWarning)
        warnings.filterwarnings("error", category=DeprecationWarning)
        
    else:  # development
        # Development: Everything visible
        warnings.filterwarnings("default")
        # But still error on resource leaks
        warnings.filterwarnings("error", category=ResourceWarning)
    
    # Always surface security-related warnings
    if 'SecurityWarning' in dir(__builtins__):
        warnings.filterwarnings("error", category=SecurityWarning)

# Auto-configure on import
if not sys.warnoptions:
    configure_warnings()
```

## Command Line and Environment

```bash
# Command line options
python -W default          # Show all warnings
python -W error            # Turn all warnings to errors
python -W ignore           # Ignore all warnings
python -W "error::DeprecationWarning"  # Specific category

# Environment variable
export PYTHONWARNINGS="default,error::ResourceWarning"

# Multiple filters (comma-separated, later takes precedence)
python -W "ignore" -W "error::DeprecationWarning"

# Debug builds show all warnings by default
python -X dev  # Development mode (shows warnings + resource tracking)
```

## Key Insights

1. **stacklevel matters**: Use `stacklevel=2` in wrapper functions so warnings point to caller
2. **skip_file_prefixes (3.12+)**: Better than stacklevel for package-wide deprecation
3. **catch_warnings isn't thread-safe** before Python 3.14 unless `context_aware_warnings` flag set
4. **DeprecationWarning ignored by default** except in `__main__` - use `-Wd` for testing
5. **ResourceWarning for leak detection** - extremely valuable in production with `error` action
6. **@deprecated decorator (3.13+)**: Integrates with static type checkers for better IDE support
