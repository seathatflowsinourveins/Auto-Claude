# Python tomllib Module - Production Patterns (January 2026)

## Overview

`tomllib` is Python's built-in TOML parser (Python 3.11+). It provides **read-only** parsing of TOML v1.0.0 files. For writing TOML, use third-party libraries like `tomli-w` or `tomlkit`.

## Core API

### tomllib.load() - Parse from File
```python
import tomllib

# Binary mode required (TOML is UTF-8)
with open("config.toml", "rb") as f:
    config = tomllib.load(f)
```

### tomllib.loads() - Parse from String
```python
import tomllib

toml_string = '''
[database]
host = "localhost"
port = 5432
'''

config = tomllib.loads(toml_string)
# {'database': {'host': 'localhost', 'port': 5432}}
```

### parse_float Parameter
```python
from decimal import Decimal

# Use Decimal instead of float for precision
config = tomllib.loads(data, parse_float=Decimal)
```

### TOMLDecodeError Exception
```python
import tomllib

try:
    config = tomllib.loads(invalid_toml)
except tomllib.TOMLDecodeError as e:
    print(f"Parse error: {e}")
    # Includes line number and position
```

## TOML to Python Type Mapping

| TOML Type | Python Type |
|-----------|-------------|
| String | `str` |
| Integer | `int` |
| Float | `float` (or `parse_float` type) |
| Boolean | `bool` |
| Offset Date-Time | `datetime.datetime` (tzinfo set) |
| Local Date-Time | `datetime.datetime` (tzinfo=None) |
| Local Date | `datetime.date` |
| Local Time | `datetime.time` |
| Array | `list` |
| Table | `dict` |

## TOML Syntax Patterns

### Keys
```toml
# Bare keys (A-Za-z0-9_-)
name = "value"
bare_key = "allowed"

# Quoted keys (any Unicode)
"quoted key" = "value"
'literal key' = "value"

# Dotted keys (nested tables inline)
physical.color = "orange"
physical.shape = "round"
# Equivalent to: [physical] color = "orange" shape = "round"
```

### Strings
```toml
# Basic strings (escape sequences work)
str1 = "Hello\nWorld"
str2 = "Path: C:\\Users\\name"

# Literal strings (no escaping)
path = 'C:\Users\name'
regex = '<\i\c*\s*>'

# Multi-line basic
multi = """
Line 1
Line 2
"""

# Multi-line literal
literal_multi = '''
No \escaping
here
'''

# Line ending backslash (trim newline)
trimmed = """\
    The quick brown \
    fox jumps over \
    the lazy dog."""
```

### Numbers
```toml
# Integers
int1 = +99
int2 = 42
int3 = 0
int4 = -17
large = 1_000_000  # Underscores for readability

# Hex, octal, binary
hex = 0xDEADBEEF
oct = 0o755
bin = 0b11010110

# Floats
flt1 = +1.0
flt2 = 3.1415
flt3 = -0.01
flt4 = 5e+22
flt5 = 1e06
flt6 = -2E-2
flt7 = 6.626e-34

# Special floats
inf1 = inf
inf2 = +inf
inf3 = -inf
nan1 = nan
```

### Booleans
```toml
bool1 = true
bool2 = false
```

### Date-Time Types
```toml
# Offset date-time (UTC or offset)
odt1 = 1979-05-27T07:32:00Z
odt2 = 1979-05-27T00:32:00-07:00
odt3 = 1979-05-27T00:32:00.999999-07:00

# Local date-time (no timezone)
ldt1 = 1979-05-27T07:32:00
ldt2 = 1979-05-27T00:32:00.999999

# Local date
ld1 = 1979-05-27

# Local time
lt1 = 07:32:00
lt2 = 00:32:00.999999
```

### Arrays
```toml
# Homogeneous arrays (recommended)
integers = [1, 2, 3]
colors = ["red", "yellow", "green"]

# Mixed types allowed
mixed = [1, "string", true, 1.5]

# Nested arrays
nested = [[1, 2], ["a", "b"]]

# Multi-line
hosts = [
    "alpha",
    "omega",  # Trailing comma OK
]
```

### Tables
```toml
[table]
key = "value"

[table.subtable]
key = "nested value"

# Equivalent to:
# table = {key = "value", subtable = {key = "nested value"}}
```

### Inline Tables
```toml
# Single line only, no trailing comma
point = {x = 1, y = 2}
animal = {type.name = "pug"}

# Can't span lines (use regular tables instead)
```

### Arrays of Tables
```toml
[[products]]
name = "Hammer"
sku = 738594937

[[products]]
name = "Nail"
sku = 284758393
color = "gray"

# Results in:
# {'products': [
#     {'name': 'Hammer', 'sku': 738594937},
#     {'name': 'Nail', 'sku': 284758393, 'color': 'gray'}
# ]}
```

## Production Pattern: TOMLConfigManager

```python
"""Production TOML configuration manager with validation."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class TOMLConfigManager:
    """Manages TOML configuration with validation and defaults."""
    
    config: dict[str, Any] = field(default_factory=dict)
    _loaded_from: Path | None = None
    
    @classmethod
    def from_file(
        cls,
        path: Path | str,
        *,
        use_decimal: bool = False,
    ) -> "TOMLConfigManager":
        """Load configuration from TOML file.
        
        Args:
            path: Path to TOML file
            use_decimal: Use Decimal for floats (financial apps)
        
        Returns:
            Configured TOMLConfigManager instance
        
        Raises:
            FileNotFoundError: If file doesn't exist
            tomllib.TOMLDecodeError: If TOML is invalid
        """
        path = Path(path)
        
        with path.open("rb") as f:
            parse_float = Decimal if use_decimal else float
            config = tomllib.load(f, parse_float=parse_float)
        
        instance = cls(config=config)
        instance._loaded_from = path
        return instance
    
    @classmethod
    def from_string(
        cls,
        content: str,
        *,
        use_decimal: bool = False,
    ) -> "TOMLConfigManager":
        """Load configuration from TOML string."""
        parse_float = Decimal if use_decimal else float
        config = tomllib.loads(content, parse_float=parse_float)
        return cls(config=config)
    
    def get(
        self,
        key_path: str,
        default: T = None,
        *,
        required: bool = False,
    ) -> T | Any:
        """Get nested value using dot notation.
        
        Args:
            key_path: Dot-separated path (e.g., "database.host")
            default: Default value if key not found
            required: Raise KeyError if key missing
        
        Returns:
            Value at key path or default
        
        Example:
            >>> config.get("database.connection.host")
            "localhost"
        """
        keys = key_path.split(".")
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            elif required:
                raise KeyError(f"Required config key missing: {key_path}")
            else:
                return default
        
        return value
    
    def get_int(self, key_path: str, default: int = 0) -> int:
        """Get integer value with type checking."""
        value = self.get(key_path, default)
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"Expected int at {key_path}, got {type(value)}")
        return value
    
    def get_str(self, key_path: str, default: str = "") -> str:
        """Get string value with type checking."""
        value = self.get(key_path, default)
        if not isinstance(value, str):
            raise TypeError(f"Expected str at {key_path}, got {type(value)}")
        return value
    
    def get_bool(self, key_path: str, default: bool = False) -> bool:
        """Get boolean value with type checking."""
        value = self.get(key_path, default)
        if not isinstance(value, bool):
            raise TypeError(f"Expected bool at {key_path}, got {type(value)}")
        return value
    
    def get_list(self, key_path: str, default: list | None = None) -> list:
        """Get list value with type checking."""
        value = self.get(key_path, default or [])
        if not isinstance(value, list):
            raise TypeError(f"Expected list at {key_path}, got {type(value)}")
        return value
    
    def get_datetime(
        self,
        key_path: str,
        default: datetime | None = None,
    ) -> datetime | None:
        """Get datetime value with type checking."""
        value = self.get(key_path, default)
        if value is None:
            return default
        if not isinstance(value, datetime):
            raise TypeError(f"Expected datetime at {key_path}, got {type(value)}")
        return value
    
    def section(self, section_path: str) -> dict[str, Any]:
        """Get entire section as dict.
        
        Example:
            >>> config.section("database")
            {"host": "localhost", "port": 5432}
        """
        value = self.get(section_path, {})
        if not isinstance(value, dict):
            raise TypeError(f"Expected table at {section_path}, got {type(value)}")
        return value
    
    def merge(self, other: dict[str, Any]) -> None:
        """Deep merge another dict into config (other wins conflicts)."""
        self._deep_merge(self.config, other)
    
    def _deep_merge(self, base: dict, override: dict) -> None:
        """Recursively merge override into base."""
        for key, value in override.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_merge(base[key], value)
            else:
                base[key] = value


# Usage example
if __name__ == "__main__":
    toml_content = '''
    [app]
    name = "MyService"
    debug = false
    version = "1.2.3"
    
    [database]
    host = "localhost"
    port = 5432
    pool_size = 10
    
    [database.credentials]
    user = "admin"
    
    [[servers]]
    name = "primary"
    ip = "10.0.0.1"
    
    [[servers]]
    name = "backup"
    ip = "10.0.0.2"
    '''
    
    config = TOMLConfigManager.from_string(toml_content)
    
    # Dot notation access
    print(config.get("app.name"))  # "MyService"
    print(config.get("database.host"))  # "localhost"
    print(config.get("database.credentials.user"))  # "admin"
    
    # Type-safe getters
    port = config.get_int("database.port")  # 5432
    debug = config.get_bool("app.debug")  # False
    
    # Get sections
    db_config = config.section("database")
    
    # Arrays of tables
    servers = config.get_list("servers")
    for server in servers:
        print(f"{server['name']}: {server['ip']}")
```

## Key Patterns

### 1. Always Use Binary Mode
```python
# CORRECT
with open("config.toml", "rb") as f:
    config = tomllib.load(f)

# WRONG - will raise TypeError
with open("config.toml", "r") as f:
    config = tomllib.load(f)  # TypeError!
```

### 2. Financial Applications Use Decimal
```python
from decimal import Decimal

# Prices, rates, monetary values
config = tomllib.load(f, parse_float=Decimal)
price = config["product"]["price"]  # Decimal('19.99')
```

### 3. Validate Required Keys
```python
def load_config(path: Path) -> dict:
    with path.open("rb") as f:
        config = tomllib.load(f)
    
    required = ["database.host", "database.port", "app.secret_key"]
    for key in required:
        keys = key.split(".")
        value = config
        for k in keys:
            if k not in value:
                raise ValueError(f"Missing required config: {key}")
            value = value[k]
    
    return config
```

### 4. Environment Variable Override
```python
import os
import tomllib

def load_with_env_override(path: Path) -> dict:
    """Load TOML with environment variable overrides.
    
    Env vars like APP_DATABASE_HOST override database.host
    """
    with path.open("rb") as f:
        config = tomllib.load(f)
    
    prefix = "APP_"
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # APP_DATABASE_HOST -> database.host
            config_path = key[len(prefix):].lower().replace("_", ".")
            _set_nested(config, config_path, value)
    
    return config

def _set_nested(d: dict, path: str, value: str) -> None:
    """Set nested dict value from dot path."""
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
```

### 5. Schema Validation with Pydantic
```python
import tomllib
from pydantic import BaseModel, Field

class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = Field(ge=1, le=65535, default=5432)
    pool_size: int = Field(ge=1, le=100, default=10)

class AppConfig(BaseModel):
    name: str
    debug: bool = False
    database: DatabaseConfig

# Load and validate
with open("config.toml", "rb") as f:
    raw = tomllib.load(f)

config = AppConfig(**raw)  # Validates all fields
```

## tomllib vs Third-Party Libraries

| Feature | tomllib | tomli | tomlkit |
|---------|---------|-------|---------|
| Read TOML | ✅ | ✅ | ✅ |
| Write TOML | ❌ | ❌ (use tomli-w) | ✅ |
| Preserve comments | ❌ | ❌ | ✅ |
| Round-trip editing | ❌ | ❌ | ✅ |
| Python version | 3.11+ | 3.7+ | 3.7+ |
| Stdlib | ✅ | ❌ | ❌ |

**Recommendation:**
- Read-only configs: Use `tomllib` (stdlib)
- Write configs: Use `tomli-w`
- Edit preserving formatting: Use `tomlkit`

## Source
- Python 3.13 tomllib documentation: https://docs.python.org/3/library/tomllib.html
- TOML v1.0.0 specification: https://toml.io/en/v1.0.0
- Research cycle: Ralph Loop Cycle 102 (January 2026)
