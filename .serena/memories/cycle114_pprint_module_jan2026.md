# Python pprint Module - Production Patterns (January 2026)

## Overview
The `pprint` module provides data structure pretty-printing for debugging, logging, and human-readable output. Essential for inspecting nested structures, config dumps, and debug output.

## Core Functions

### pp() vs pprint() - The Critical Difference
```python
from pprint import pp, pprint

data = {'zebra': 1, 'apple': 2, 'mango': 3}

# pprint() - sorts dict keys alphabetically (legacy behavior)
pprint(data)  # {'apple': 2, 'mango': 3, 'zebra': 1}

# pp() - preserves insertion order (Python 3.8+, recommended)
pp(data)  # {'zebra': 1, 'apple': 2, 'mango': 3}
```

### pformat() - Return String Instead of Print
```python
from pprint import pformat

data = {'config': {'host': 'localhost', 'port': 8080}}
formatted = pformat(data, indent=2, width=60)
# Use in logging, files, or further processing
```

### Function Signatures
```python
pprint(object, stream=None, indent=1, width=80, depth=None, *, 
       compact=False, sort_dicts=True, underscore_numbers=False)

pp(object, *args, sort_dicts=False, **kwargs)  # 3.8+

pformat(object, indent=1, width=80, depth=None, *, 
        compact=False, sort_dicts=True, underscore_numbers=False)
```

## PrettyPrinter Class

### Creating Custom Printers
```python
from pprint import PrettyPrinter

# Production-ready debug printer
debug_printer = PrettyPrinter(
    indent=4,              # 4-space indentation
    width=120,             # wider for terminals
    depth=3,               # limit nesting (None=unlimited)
    compact=True,          # pack small items on one line
    sort_dicts=False,      # preserve order (3.8+)
    underscore_numbers=True  # 1_000_000 format (3.10+)
)

debug_printer.pprint(complex_data)
formatted = debug_printer.pformat(complex_data)
```

### Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `indent` | 1 | Spaces per nesting level |
| `width` | 80 | Max line width before wrapping |
| `depth` | None | Max nesting depth (None=unlimited) |
| `compact` | False | Pack sequences on single lines when possible |
| `sort_dicts` | True | Alphabetize dict keys (False preserves order) |
| `underscore_numbers` | False | Format as 1_000_000 (3.10+) |

## Helper Functions

### saferepr() - Safe Representation
```python
from pprint import saferepr

# Handles recursive structures safely
circular = []
circular.append(circular)
saferepr(circular)  # '[[...]]' instead of RecursionError
```

### isreadable() - Check if eval-able
```python
from pprint import isreadable

isreadable({'a': 1})  # True - can be recreated with eval()
isreadable({'a': lambda x: x})  # False - contains non-readable object
```

### isrecursive() - Detect Circular References
```python
from pprint import isrecursive

circular = []
circular.append(circular)
isrecursive(circular)  # True
isrecursive([1, 2, 3])  # False
```

## Production Patterns

### Pattern 1: Debug Logger with pprint
```python
import logging
from pprint import pformat

logger = logging.getLogger(__name__)

def log_structured(level: str, msg: str, data: dict) -> None:
    """Log with pretty-printed data structure."""
    formatted = pformat(data, indent=2, width=100, sort_dicts=False)
    getattr(logger, level)(f"{msg}:\n{formatted}")

# Usage
log_structured('debug', 'API Response', response_data)
```

### Pattern 2: Config Display
```python
from pprint import PrettyPrinter

class ConfigPrinter:
    """Pretty-print configuration for startup logs."""
    
    def __init__(self):
        self._printer = PrettyPrinter(
            indent=2,
            width=100,
            depth=4,
            sort_dicts=False,
            underscore_numbers=True
        )
    
    def display(self, config: dict, mask_secrets: bool = True) -> str:
        """Format config, optionally masking sensitive values."""
        if mask_secrets:
            config = self._mask_secrets(config)
        return self._printer.pformat(config)
    
    def _mask_secrets(self, obj, keys={'password', 'secret', 'token', 'key'}):
        if isinstance(obj, dict):
            return {
                k: '***' if any(s in k.lower() for s in keys) else 
                   self._mask_secrets(v, keys)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._mask_secrets(item, keys) for item in obj]
        return obj
```

### Pattern 3: REPL-Style Object Inspector
```python
from pprint import pp
from dataclasses import dataclass, asdict
from types import SimpleNamespace

def inspect(obj, depth: int = 3) -> None:
    """Universal object inspector for debugging."""
    print(f"Type: {type(obj).__name__}")
    print(f"Repr: {repr(obj)[:100]}...")
    
    # Convert to dict-like for pretty printing
    if hasattr(obj, '__dict__'):
        pp(vars(obj), depth=depth)
    elif dataclass and hasattr(obj, '__dataclass_fields__'):
        pp(asdict(obj), depth=depth)
    elif isinstance(obj, SimpleNamespace):
        pp(vars(obj), depth=depth)
    else:
        pp(obj, depth=depth)
```

### Pattern 4: Test Assertion Diff
```python
from pprint import pformat

def assert_equal_structures(actual, expected, msg=""):
    """Pretty assertion error for complex structures."""
    if actual != expected:
        actual_fmt = pformat(actual, indent=2, sort_dicts=True)
        expected_fmt = pformat(expected, indent=2, sort_dicts=True)
        raise AssertionError(
            f"{msg}\n"
            f"Actual:\n{actual_fmt}\n\n"
            f"Expected:\n{expected_fmt}"
        )
```

### Pattern 5: JSON-Like Output
```python
from pprint import PrettyPrinter

class JSONLikePrinter(PrettyPrinter):
    """Printer with JSON-ish indentation style."""
    
    def __init__(self):
        super().__init__(indent=2, width=120, compact=False, sort_dicts=False)
    
    def format_for_log(self, obj) -> str:
        """Format with timestamp prefix."""
        from datetime import datetime
        ts = datetime.now().isoformat()
        return f"[{ts}]\n{self.pformat(obj)}"
```

## Python 3.8+ Improvements

### SimpleNamespace Support (3.9+)
```python
from types import SimpleNamespace
from pprint import pp

ns = SimpleNamespace(name="test", value=42, nested=SimpleNamespace(x=1))
pp(ns)  # namespace(name='test', value=42, nested=namespace(x=1))
```

### Dataclass Support (3.10+)
```python
from dataclasses import dataclass
from pprint import pp

@dataclass
class Config:
    host: str
    port: int
    options: dict

config = Config("localhost", 8080, {"timeout": 30})
pp(config)  # Pretty-printed with proper formatting
```

### Underscore Numbers (3.10+)
```python
from pprint import pp

large_data = {'population': 8_000_000_000, 'bytes': 1_073_741_824}
pp(large_data, underscore_numbers=True)
# {'population': 8_000_000_000, 'bytes': 1_073_741_824}
```

## Best Practices

1. **Use pp() over pprint()** for dict order preservation (3.8+)
2. **Set depth limit** to avoid overwhelming output on deep structures
3. **Use pformat()** for logging rather than print redirection
4. **Enable compact=True** for sequences to reduce vertical space
5. **Use saferepr()** when structures might be circular
6. **Set width appropriately** for your terminal/log viewer

## Common Pitfalls

```python
# WRONG: Printing to file with pprint
pprint(data)  # Goes to stdout, not file

# RIGHT: Use stream parameter or pformat
pprint(data, stream=open('out.txt', 'w'))
# OR
with open('out.txt', 'w') as f:
    f.write(pformat(data))

# WRONG: Expecting JSON output
pprint({'key': 'value'})  # Uses repr-style quotes

# RIGHT: Use json.dumps for JSON
import json
print(json.dumps(data, indent=2))
```

## Version History
- 3.8: Added `sort_dicts` parameter, introduced `pp()` function
- 3.9: SimpleNamespace support
- 3.10: Dataclass support, `underscore_numbers` parameter
- 3.13: Performance improvements for large structures
