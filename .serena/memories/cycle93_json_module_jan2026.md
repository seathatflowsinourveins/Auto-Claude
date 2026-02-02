# Python json Module - Production Patterns (Jan 2026)

## Overview
The `json` module provides JSON encoding and decoding per RFC 7159 and ECMA-404. It exposes an API similar to `pickle` and `marshal` but produces human-readable, language-agnostic output.

## Type Conversion Tables

### Python to JSON
| Python | JSON |
|--------|------|
| dict | object |
| list, tuple | array |
| str | string |
| int, float | number |
| True | true |
| False | false |
| None | null |

### JSON to Python
| JSON | Python |
|------|--------|
| object | dict |
| array | list |
| string | str |
| number (int) | int |
| number (real) | float |
| true | True |
| false | False |
| null | None |

## Basic Usage

### Encoding (Python → JSON)
```python
import json

# To string
data = {"name": "Alice", "age": 30, "active": True}
json_str = json.dumps(data)
# '{"name": "Alice", "age": 30, "active": true}'

# To file
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f)

# Pretty printing
json_str = json.dumps(data, indent=2, sort_keys=True)
"""
{
  "active": true,
  "age": 30,
  "name": "Alice"
}
"""

# Compact (no whitespace)
json_str = json.dumps(data, separators=(",", ":"))
# '{"name":"Alice","age":30,"active":true}'
```

### Decoding (JSON → Python)
```python
import json

# From string
data = json.loads('{"name": "Alice", "age": 30}')
# {'name': 'Alice', 'age': 30}

# From file
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# From bytes (UTF-8, UTF-16, UTF-32 supported)
data = json.loads(b'{"key": "value"}')
```

## dumps() Parameters

```python
json.dumps(
    obj,                    # Python object to serialize
    skipkeys=False,         # Skip non-string keys instead of raising TypeError
    ensure_ascii=True,      # Escape non-ASCII characters (default)
    check_circular=True,    # Check for circular references
    allow_nan=True,         # Allow NaN, Infinity, -Infinity
    cls=None,               # Custom JSONEncoder class
    indent=None,            # Indentation level (int or string like "\t")
    separators=None,        # (item_sep, key_sep) tuple
    default=None,           # Function for non-serializable objects
    sort_keys=False         # Sort dictionary keys
)
```

### Common Parameter Combinations
```python
# Development (readable)
json.dumps(data, indent=2, sort_keys=True)

# Production API (compact)
json.dumps(data, separators=(",", ":"))

# Logging (single line, sorted)
json.dumps(data, sort_keys=True)

# Non-ASCII content
json.dumps(data, ensure_ascii=False)

# With custom serializer
json.dumps(data, default=str)  # Convert unknown types to string
```

## loads() Parameters

```python
json.loads(
    s,                      # JSON string, bytes, or bytearray
    cls=None,               # Custom JSONDecoder class
    object_hook=None,       # Function called on each decoded dict
    parse_float=None,       # Function to parse floats (default: float)
    parse_int=None,         # Function to parse ints (default: int)
    parse_constant=None,    # Function for -Infinity, Infinity, NaN
    object_pairs_hook=None  # Function called with ordered pairs
)
```

## Custom Serialization

### Using default Parameter
```python
import json
from datetime import datetime, date
from decimal import Decimal
from uuid import UUID
from pathlib import Path
from enum import Enum

def json_serializer(obj):
    """Handle non-serializable types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return str(obj)  # or float(obj) for numeric
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8")
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Cannot serialize {type(obj).__name__}")

# Usage
data = {
    "timestamp": datetime.now(),
    "price": Decimal("19.99"),
    "id": UUID("550e8400-e29b-41d4-a716-446655440000"),
    "tags": {"python", "json"}
}
json_str = json.dumps(data, default=json_serializer)
```

### Custom JSONEncoder Class
```python
import json
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class User:
    name: str
    email: str
    created_at: datetime

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return {"__datetime__": True, "value": obj.isoformat()}
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        return super().default(obj)

# Usage
user = User("Alice", "alice@example.com", datetime.now())
json_str = json.dumps(user, cls=CustomEncoder, indent=2)

# Alternatively, encoder instance
encoder = CustomEncoder(indent=2)
json_str = encoder.encode(user)
```

## Custom Deserialization

### Using object_hook
```python
import json
from datetime import datetime

def datetime_decoder(dct):
    """Decode datetime from ISO format."""
    if "__datetime__" in dct:
        return datetime.fromisoformat(dct["value"])
    return dct

json_str = '{"__datetime__": true, "value": "2026-01-25T10:30:00"}'
data = json.loads(json_str, object_hook=datetime_decoder)
# data is now a datetime object
```

### Using object_pairs_hook
```python
import json
from collections import OrderedDict

# Preserve order (though dict is ordered in Python 3.7+)
data = json.loads(json_str, object_pairs_hook=OrderedDict)

# Detect duplicate keys
def check_duplicates(pairs):
    keys = [k for k, v in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate keys found: {keys}")
    return dict(pairs)

data = json.loads(json_str, object_pairs_hook=check_duplicates)
```

### Using parse_float and parse_int
```python
import json
from decimal import Decimal

# Parse floats as Decimal for precision
data = json.loads('{"price": 19.99}', parse_float=Decimal)
# data["price"] is Decimal('19.99')

# Parse large integers
data = json.loads('{"big": 123456789012345678901234567890}')
# Works correctly - Python int has no size limit
```

## Exception Handling

```python
import json

# JSONDecodeError (subclass of ValueError)
try:
    data = json.loads("invalid json")
except json.JSONDecodeError as e:
    print(f"Error: {e.msg}")
    print(f"Document: {e.doc}")
    print(f"Position: {e.pos}")
    print(f"Line: {e.lineno}, Column: {e.colno}")

# TypeError for non-serializable objects
try:
    json.dumps({"func": lambda x: x})
except TypeError as e:
    print(f"Serialization error: {e}")
```

## Streaming / Incremental Encoding

```python
import json

# Streaming large objects
encoder = json.JSONEncoder()
for chunk in encoder.iterencode(large_object):
    socket.write(chunk)

# Multiple objects (JSON Lines format)
with open("data.jsonl", "w") as f:
    for item in items:
        f.write(json.dumps(item) + "\n")

# Reading JSON Lines
with open("data.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        process(item)
```

## NaN and Infinity Handling

```python
import json
import math

# Default: allows NaN/Infinity (not strict JSON)
json.dumps({"value": float("nan")})      # '{"value": NaN}'
json.dumps({"value": float("inf")})      # '{"value": Infinity}'
json.dumps({"value": float("-inf")})     # '{"value": -Infinity}'

# Strict JSON compliance
try:
    json.dumps({"value": float("nan")}, allow_nan=False)
except ValueError as e:
    print("NaN not allowed in strict mode")

# Custom handling on decode
def handle_constants(val):
    if val == "NaN":
        return None  # or 0, or raise
    if val in ("Infinity", "-Infinity"):
        raise ValueError("Infinity not supported")
    return val

data = json.loads('{"x": NaN}', parse_constant=handle_constants)
```

## Production JSON Handler Class

```python
"""Production-ready JSON utilities."""

import json
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeVar
from uuid import UUID
import dataclasses

T = TypeVar("T")


class JSONHandler:
    """Production JSON encoder/decoder with type preservation."""
    
    # Type markers for round-trip serialization
    TYPE_MARKERS = {
        datetime: "__datetime__",
        date: "__date__",
        time: "__time__",
        timedelta: "__timedelta__",
        Decimal: "__decimal__",
        UUID: "__uuid__",
        bytes: "__bytes__",
        set: "__set__",
        frozenset: "__frozenset__",
        complex: "__complex__",
    }
    
    def __init__(
        self,
        indent: int | None = None,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
        preserve_types: bool = True
    ):
        self.indent = indent
        self.sort_keys = sort_keys
        self.ensure_ascii = ensure_ascii
        self.preserve_types = preserve_types
    
    def encode(self, obj: Any) -> str:
        """Encode Python object to JSON string."""
        return json.dumps(
            obj,
            default=self._serialize,
            indent=self.indent,
            sort_keys=self.sort_keys,
            ensure_ascii=self.ensure_ascii
        )
    
    def decode(self, s: str | bytes) -> Any:
        """Decode JSON string to Python object."""
        return json.loads(s, object_hook=self._deserialize)
    
    def dump(self, obj: Any, path: Path | str) -> None:
        """Write object to JSON file."""
        path = Path(path)
        path.write_text(self.encode(obj), encoding="utf-8")
    
    def load(self, path: Path | str) -> Any:
        """Load object from JSON file."""
        path = Path(path)
        return self.decode(path.read_text(encoding="utf-8"))
    
    def _serialize(self, obj: Any) -> Any:
        """Serialize non-standard types."""
        if self.preserve_types:
            # With type markers for round-trip
            if isinstance(obj, datetime):
                return {"__datetime__": obj.isoformat()}
            if isinstance(obj, date):
                return {"__date__": obj.isoformat()}
            if isinstance(obj, time):
                return {"__time__": obj.isoformat()}
            if isinstance(obj, timedelta):
                return {"__timedelta__": obj.total_seconds()}
            if isinstance(obj, Decimal):
                return {"__decimal__": str(obj)}
            if isinstance(obj, UUID):
                return {"__uuid__": str(obj)}
            if isinstance(obj, bytes):
                return {"__bytes__": obj.decode("utf-8", errors="replace")}
            if isinstance(obj, (set, frozenset)):
                marker = "__set__" if isinstance(obj, set) else "__frozenset__"
                return {marker: list(obj)}
            if isinstance(obj, complex):
                return {"__complex__": [obj.real, obj.imag]}
        else:
            # Simple conversion (no round-trip)
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            if isinstance(obj, time):
                return obj.isoformat()
            if isinstance(obj, timedelta):
                return obj.total_seconds()
            if isinstance(obj, Decimal):
                return float(obj)
            if isinstance(obj, UUID):
                return str(obj)
            if isinstance(obj, bytes):
                return obj.decode("utf-8", errors="replace")
            if isinstance(obj, (set, frozenset)):
                return list(obj)
            if isinstance(obj, complex):
                return [obj.real, obj.imag]
        
        # Common types
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, Path):
            return str(obj)
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        if hasattr(obj, "__iter__"):
            return list(obj)
        
        raise TypeError(f"Cannot serialize {type(obj).__name__}")
    
    def _deserialize(self, dct: dict) -> Any:
        """Deserialize type-marked objects."""
        if not self.preserve_types:
            return dct
        
        if "__datetime__" in dct:
            return datetime.fromisoformat(dct["__datetime__"])
        if "__date__" in dct:
            return date.fromisoformat(dct["__date__"])
        if "__time__" in dct:
            return time.fromisoformat(dct["__time__"])
        if "__timedelta__" in dct:
            return timedelta(seconds=dct["__timedelta__"])
        if "__decimal__" in dct:
            return Decimal(dct["__decimal__"])
        if "__uuid__" in dct:
            return UUID(dct["__uuid__"])
        if "__bytes__" in dct:
            return dct["__bytes__"].encode("utf-8")
        if "__set__" in dct:
            return set(dct["__set__"])
        if "__frozenset__" in dct:
            return frozenset(dct["__frozenset__"])
        if "__complex__" in dct:
            return complex(dct["__complex__"][0], dct["__complex__"][1])
        
        return dct


# Convenience instances
json_handler = JSONHandler()
json_pretty = JSONHandler(indent=2, sort_keys=True)
json_compact = JSONHandler(preserve_types=False)


# Usage
data = {
    "id": UUID("550e8400-e29b-41d4-a716-446655440000"),
    "created": datetime.now(),
    "price": Decimal("19.99"),
    "tags": {"python", "json"}
}

# Round-trip with type preservation
encoded = json_handler.encode(data)
decoded = json_handler.decode(encoded)
assert decoded["id"] == data["id"]  # UUID preserved!

# Pretty print for debugging
print(json_pretty.encode(data))
```

## Pydantic Integration

```python
from pydantic import BaseModel
from datetime import datetime
import json

class User(BaseModel):
    id: int
    name: str
    created_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Pydantic models have built-in JSON support
user = User(id=1, name="Alice", created_at=datetime.now())

# To JSON string
json_str = user.model_dump_json()

# To dict then JSON
json_str = json.dumps(user.model_dump(), default=str)

# From JSON
user = User.model_validate_json(json_str)
```

## Command Line Interface

```bash
# Validate and pretty-print
python -m json input.json

# From stdin
echo '{"key": "value"}' | python -m json

# Sort keys
python -m json --sort-keys data.json

# Compact output
python -m json --compact data.json

# JSON Lines format
python -m json --json-lines data.jsonl

# Preserve non-ASCII
python -m json --no-ensure-ascii data.json
```

## Performance Tips

```python
# 1. Use orjson or ujson for speed (10-50x faster)
# pip install orjson
import orjson
data = orjson.loads(json_bytes)
json_bytes = orjson.dumps(data)

# 2. Reuse encoder for multiple objects
encoder = json.JSONEncoder(separators=(",", ":"))
for item in items:
    output = encoder.encode(item)

# 3. Use separators for smaller output
json.dumps(data, separators=(",", ":"))  # No whitespace

# 4. Stream large files
def stream_json_array(file_path):
    with open(file_path) as f:
        # Skip opening bracket
        f.read(1)
        decoder = json.JSONDecoder()
        buffer = ""
        for line in f:
            buffer += line
            while buffer:
                try:
                    obj, idx = decoder.raw_decode(buffer)
                    yield obj
                    buffer = buffer[idx:].lstrip(" ,\n")
                except json.JSONDecodeError:
                    break
```

## Key Takeaways

1. **Use encoding="utf-8"** - Always specify when opening files
2. **ensure_ascii=False** - For non-English content
3. **separators=(",",":")** - Compact output for APIs
4. **default=str** - Quick fix for unknown types (but not round-trip safe)
5. **object_hook** - For custom deserialization
6. **JSONDecodeError** - Has useful attributes (msg, doc, pos, lineno, colno)
7. **allow_nan=False** - For strict JSON compliance
8. **sort_keys=True** - For reproducible output (testing, caching)
9. **JSON Lines (.jsonl)** - One object per line for streaming
10. **Consider orjson** - 10-50x faster for production workloads
