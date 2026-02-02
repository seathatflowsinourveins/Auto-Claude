# Python traceback Module - Production Patterns (January 2026)

## Overview
The `traceback` module provides tools for extracting, formatting, and printing stack traces. Essential for error logging, debugging, and building custom error handlers in production applications.

## Module-Level Functions

### print_exc() - Quick Exception Print
```python
import traceback

try:
    1/0
except:
    traceback.print_exc()  # Prints to stderr
    traceback.print_exc(file=sys.stdout)  # Custom file
    traceback.print_exc(limit=2)  # Limit stack depth
```

### format_exc() - Get Exception as String
```python
import traceback

try:
    risky_operation()
except Exception:
    error_str = traceback.format_exc()
    logger.error(f"Operation failed:\n{error_str}")
```

### print_exception() - Full Control
```python
import traceback
import sys

try:
    risky_operation()
except Exception as e:
    traceback.print_exception(
        e,              # Exception object (3.10+)
        limit=5,        # Stack depth limit
        file=sys.stderr,
        chain=True      # Show chained exceptions
    )
```

### format_exception() - Format as List
```python
import traceback

try:
    risky_operation()
except Exception as e:
    lines = traceback.format_exception(e)
    # Returns list of strings, each ending with \n
    full_traceback = ''.join(lines)
```

### print_tb() / format_tb() - Traceback Only
```python
import traceback

try:
    risky_operation()
except Exception as e:
    # Print just the traceback (no exception info)
    traceback.print_tb(e.__traceback__, limit=3)
    
    # Get as formatted strings
    tb_lines = traceback.format_tb(e.__traceback__)
```

### print_stack() / format_stack() - Current Stack
```python
import traceback

def debug_here():
    # Print current call stack (not from exception)
    traceback.print_stack()
    
    # Get as string list
    stack_lines = traceback.format_stack()
    return ''.join(stack_lines)
```

### extract_tb() / extract_stack() - Structured Data
```python
import traceback

try:
    risky_operation()
except Exception as e:
    # Get StackSummary (list of FrameSummary)
    frames = traceback.extract_tb(e.__traceback__)
    
    for frame in frames:
        print(f"{frame.filename}:{frame.lineno} in {frame.name}")
        print(f"  {frame.line}")
```

### walk_tb() / walk_stack() - Frame Iteration
```python
import traceback

try:
    risky_operation()
except Exception as e:
    # Iterate frames with line numbers
    for frame, lineno in traceback.walk_tb(e.__traceback__):
        print(f"{frame.f_code.co_filename}:{lineno}")

# Walk current stack
for frame, lineno in traceback.walk_stack(None):
    print(f"{frame.f_code.co_name} at line {lineno}")
```

### clear_frames() - Memory Management
```python
import traceback

try:
    risky_operation()
except Exception as e:
    tb_str = traceback.format_exc()
    # Clear locals from frames to free memory
    traceback.clear_frames(e.__traceback__)
    # Now e.__traceback__ frames have no locals
```

## TracebackException Class

### Basic Usage
```python
from traceback import TracebackException

try:
    risky_operation()
except Exception as e:
    # Capture exception for later formatting
    tb_exc = TracebackException.from_exception(e)
    
    # Can be pickled/stored (no frame references)
    
    # Format later
    tb_exc.print()  # To stderr
    lines = list(tb_exc.format())  # As strings
```

### With Local Variables
```python
from traceback import TracebackException

try:
    x = 42
    y = "test"
    1/0
except Exception as e:
    tb_exc = TracebackException.from_exception(
        e, 
        capture_locals=True  # Capture local variables
    )
    tb_exc.print()
    # Shows: x = 42, y = 'test' in output
```

### Controlling Output
```python
from traceback import TracebackException

try:
    try:
        original_error()
    except:
        raise ValueError("wrapper")
except Exception as e:
    tb_exc = TracebackException.from_exception(e)
    
    # Without chained exceptions
    tb_exc.print(chain=False)
    
    # Just the exception part
    for line in tb_exc.format_exception_only():
        print(line, end='')
```

### Exception Attributes
```python
from traceback import TracebackException

tb_exc = TracebackException.from_exception(e)

tb_exc.exc_type_str   # "ValueError" (3.13+)
tb_exc.__cause__      # TracebackException of __cause__
tb_exc.__context__    # TracebackException of __context__
tb_exc.__notes__      # Exception notes list
tb_exc.stack          # StackSummary of frames
```

## StackSummary and FrameSummary

### StackSummary
```python
from traceback import StackSummary, walk_tb

# Extract from traceback
summary = StackSummary.extract(walk_tb(e.__traceback__))

# With options
summary = StackSummary.extract(
    walk_tb(e.__traceback__),
    limit=10,
    lookup_lines=True,   # Fetch source lines
    capture_locals=True  # Include local vars
)

# Format to strings
for line in summary.format():
    print(line, end='')
```

### FrameSummary Attributes
```python
from traceback import extract_tb

for frame in extract_tb(e.__traceback__):
    frame.filename   # "/path/to/file.py"
    frame.lineno     # 42
    frame.name       # "function_name"
    frame.line       # "source code line"
    frame.end_lineno # Last line (multi-line)
    frame.colno      # Column offset
    frame.end_colno  # End column
```

## Production Patterns

### Pattern 1: Structured Error Logger
```python
import traceback
import json
import logging
from datetime import datetime

def log_exception(e: Exception, context: dict = None) -> str:
    """Log exception with structured metadata."""
    tb_exc = TracebackException.from_exception(e, capture_locals=True)
    
    error_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "error_type": tb_exc.exc_type_str,
        "message": str(e),
        "traceback": list(tb_exc.format()),
        "context": context or {},
        "frames": [
            {
                "file": f.filename,
                "line": f.lineno,
                "function": f.name,
                "code": f.line
            }
            for f in tb_exc.stack
        ]
    }
    
    logging.error(json.dumps(error_data, indent=2))
    return error_data["timestamp"]
```

### Pattern 2: Exception Capture for Later
```python
from traceback import TracebackException
from dataclasses import dataclass
from typing import Optional
import pickle

@dataclass
class CapturedError:
    """Serializable error capture."""
    exc_type: str
    message: str
    formatted: str
    frames: list
    
    @classmethod
    def from_exception(cls, e: Exception) -> 'CapturedError':
        tb = TracebackException.from_exception(e)
        return cls(
            exc_type=tb.exc_type_str,
            message=str(e),
            formatted=''.join(tb.format()),
            frames=[(f.filename, f.lineno, f.name, f.line) 
                    for f in tb.stack]
        )
    
    def to_bytes(self) -> bytes:
        return pickle.dumps(self)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'CapturedError':
        return pickle.loads(data)
```

### Pattern 3: Custom Traceback Formatter
```python
from traceback import StackSummary, FrameSummary

class CompactStackSummary(StackSummary):
    """Stack summary with compact single-line format."""
    
    def format_frame_summary(self, frame: FrameSummary) -> str:
        # Override to customize output
        short_file = frame.filename.split('/')[-1]
        return f"  → {short_file}:{frame.lineno} {frame.name}()\n"

def format_compact(e: Exception) -> str:
    """Format exception in compact form."""
    summary = CompactStackSummary.extract(
        traceback.walk_tb(e.__traceback__),
        limit=5
    )
    lines = [f"ERROR: {type(e).__name__}: {e}\n"]
    lines.extend(summary.format())
    return ''.join(lines)
```

### Pattern 4: Safe Exception Handler
```python
import traceback
import sys

def safe_exception_handler(exc_type, exc_val, exc_tb):
    """Global exception handler that never fails."""
    try:
        # Try structured logging
        tb_exc = TracebackException(exc_type, exc_val, exc_tb)
        error_msg = ''.join(tb_exc.format())
        logging.critical(f"Unhandled exception:\n{error_msg}")
    except Exception:
        # Fallback to basic print
        try:
            traceback.print_exception(exc_type, exc_val, exc_tb)
        except Exception:
            print(f"CRITICAL: {exc_type.__name__}: {exc_val}", 
                  file=sys.stderr)

sys.excepthook = safe_exception_handler
```

### Pattern 5: Request Error Context
```python
import traceback
from functools import wraps

def capture_errors(func):
    """Decorator that captures and enriches errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Add context as exception note
            e.add_note(f"Function: {func.__name__}")
            e.add_note(f"Args: {args[:3]}...")  # Truncated
            
            # Re-raise with original traceback
            raise
    return wrapper
```

### Pattern 6: Debug Inspection
```python
import traceback

def debug_point(message: str = "Debug checkpoint"):
    """Print current location and call stack."""
    stack = traceback.extract_stack()[:-1]  # Exclude this call
    caller = stack[-1]
    
    print(f"\n{'='*50}")
    print(f"DEBUG: {message}")
    print(f"Location: {caller.filename}:{caller.lineno}")
    print(f"Function: {caller.name}")
    print(f"Code: {caller.line}")
    print(f"\nCall stack:")
    for frame in stack[-5:]:  # Last 5 frames
        print(f"  {frame.name} at {frame.filename}:{frame.lineno}")
    print('='*50 + '\n')
```

## Python 3.13+ Features

### Colorized Output
```python
# Output is colorized by default in 3.13+
# Control via environment variables:
# PYTHON_COLORS=0  - disable
# PYTHON_COLORS=1  - enable
# NO_COLOR         - disable (standard)
# FORCE_COLOR      - enable (standard)

import traceback
traceback.print_exc()  # Colorized in terminal
```

### Exception Groups
```python
from traceback import TracebackException

try:
    raise ExceptionGroup("errors", [
        ValueError("first"),
        TypeError("second")
    ])
except ExceptionGroup as eg:
    tb = TracebackException.from_exception(
        eg,
        max_group_width=15,  # Max exceptions shown
        max_group_depth=10   # Max nesting depth
    )
    tb.print()
```

## Limit Parameter Behavior

```python
import traceback

# Positive limit: first N frames from top
traceback.print_exc(limit=2)   # 2 frames from caller

# Negative limit: last N frames 
traceback.print_exc(limit=-2)  # 2 frames nearest exception

# None: all frames (default)
traceback.print_exc(limit=None)
```

## Best Practices

1. **Use TracebackException for storage** - doesn't hold frame refs
2. **clear_frames() after capturing** - releases memory
3. **capture_locals sparingly** - can expose sensitive data
4. **Set reasonable limits** - deep stacks hurt readability
5. **Format for your audience** - compact for logs, detailed for devs

## Common Pitfalls

```python
# WRONG: Holding traceback reference (memory leak)
saved_tb = e.__traceback__  # Keeps frames alive

# RIGHT: Capture and release
tb_exc = TracebackException.from_exception(e)
traceback.clear_frames(e.__traceback__)

# WRONG: Printing in except block hides exception
try:
    risky()
except:
    print("Error occurred")  # Swallows exception

# RIGHT: Log and re-raise
try:
    risky()
except Exception as e:
    logging.error(traceback.format_exc())
    raise
```
