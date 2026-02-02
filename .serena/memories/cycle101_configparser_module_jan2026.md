# Cycle 101: configparser Module - Production Patterns (January 2026)

## Overview

The `configparser` module provides INI file parsing with sections, key-value pairs, interpolation, type conversion, and fallback values. Supports DEFAULT section for shared values and customizable parsing behavior.

## Core Classes

### ConfigParser - Main Configuration Interface

```python
import configparser
from pathlib import Path

# Basic reading
config = configparser.ConfigParser()
config.read('config.ini')

# Access sections and options
sections = config.sections()  # Excludes DEFAULT
if config.has_section('database'):
    host = config['database']['host']
    port = config.getint('database', 'port')
    
# Dictionary-like access
for key in config['server']:
    print(f"{key} = {config['server'][key]}")

# Writing configuration
config['app'] = {
    'name': 'MyApp',
    'version': '1.0.0',
    'debug': 'false'
}
with open('config.ini', 'w') as f:
    config.write(f)
```

### Type Conversion Methods

```python
# getint, getfloat, getboolean with fallback
port = config.getint('server', 'port', fallback=8080)
timeout = config.getfloat('server', 'timeout', fallback=30.0)
debug = config.getboolean('app', 'debug', fallback=False)

# Boolean values recognized:
# True:  '1', 'yes', 'true', 'on'
# False: '0', 'no', 'false', 'off'

# Custom converters
config = configparser.ConfigParser(
    converters={
        'list': lambda x: [i.strip() for i in x.split(',')],
        'path': lambda x: Path(x)
    }
)
config.read_string("""
[paths]
data_dir = /var/data
allowed_hosts = localhost, 127.0.0.1, example.com
""")

hosts = config.getlist('paths', 'allowed_hosts')
data_path = config.getpath('paths', 'data_dir')
```

## Interpolation

### BasicInterpolation (Default)

```python
# config.ini
# [paths]
# base_dir = /opt/myapp
# log_dir = %(base_dir)s/logs
# data_dir = %(base_dir)s/data

config = configparser.ConfigParser()
config.read('config.ini')
print(config['paths']['log_dir'])  # /opt/myapp/logs

# Escape % with %%
# percentage = 80%%  -> reads as "80%"
```

### ExtendedInterpolation

```python
# Cross-section references with ${section:option}
config = configparser.ConfigParser(
    interpolation=configparser.ExtendedInterpolation()
)
config.read_string("""
[common]
base_url = https://api.example.com

[endpoints]
users = ${common:base_url}/users
posts = ${common:base_url}/posts

[auth]
token_url = ${common:base_url}/oauth/token
""")

print(config['endpoints']['users'])  # https://api.example.com/users

# Escape $ with $$
# price = $$99.99  -> reads as "$99.99"
```

### No Interpolation

```python
# Disable interpolation entirely
config = configparser.ConfigParser(interpolation=None)
# or use RawConfigParser
config = configparser.RawConfigParser()
```

## DEFAULT Section

```python
# DEFAULT values apply to all sections
config.read_string("""
[DEFAULT]
timeout = 30
retry = 3

[database]
host = localhost
port = 5432

[cache]
host = redis.local
port = 6379
""")

# Both sections inherit DEFAULT values
print(config['database']['timeout'])  # 30
print(config['cache']['retry'])       # 3

# Section values override DEFAULT
config['database']['timeout'] = '60'
print(config['database']['timeout'])  # 60
print(config['cache']['timeout'])     # 30 (still DEFAULT)
```

## Reading Configuration

### Multiple Sources

```python
# Read from multiple files (later files override earlier)
config.read(['defaults.ini', 'local.ini', 'override.ini'])

# Read from string
config.read_string("""
[section]
key = value
""")

# Read from dictionary
config.read_dict({
    'section1': {'key1': 'value1'},
    'section2': {'key2': 'value2'}
})

# Read from file object
with open('config.ini', encoding='utf-8') as f:
    config.read_file(f)
```

### Fallback Values

```python
# get() with fallback
value = config.get('section', 'missing_key', fallback='default')

# Section-level get
value = config['section'].get('missing_key', 'default')

# Type-safe fallbacks
port = config.getint('server', 'port', fallback=8080)
enabled = config.getboolean('features', 'new_ui', fallback=True)
```

## Parser Customization

### Constructor Options

```python
config = configparser.ConfigParser(
    # Custom delimiters (default: '=' and ':')
    delimiters=('=',),
    
    # Comment prefixes (default: '#' and ';')
    comment_prefixes=('#',),
    inline_comment_prefixes=(';',),
    
    # Allow keys without values
    allow_no_value=True,
    
    # Strict mode - no duplicates (default: True)
    strict=True,
    
    # Empty lines in multiline values
    empty_lines_in_values=False,
    
    # Custom default section name
    default_section='GLOBAL',
    
    # Interpolation handler
    interpolation=configparser.ExtendedInterpolation(),
    
    # Allow unnamed first section (Python 3.13+)
    allow_unnamed_section=True
)
```

### Case Sensitivity

```python
# Keys are case-insensitive by default (stored lowercase)
config = configparser.ConfigParser()
config.read_string("[section]\nMyKey = value")
print(list(config['section'].keys()))  # ['mykey']

# Make keys case-sensitive
config = configparser.ConfigParser()
config.optionxform = str  # Don't transform option names
config.read_string("[section]\nMyKey = value")
print(list(config['section'].keys()))  # ['MyKey']
```

### Custom Boolean States

```python
config = configparser.ConfigParser()
config.BOOLEAN_STATES = {
    'enabled': True, 'disabled': False,
    'active': True, 'inactive': False,
    'yes': True, 'no': False
}

config.read_string("[app]\nfeature = enabled")
print(config.getboolean('app', 'feature'))  # True
```

## Unnamed Sections (Python 3.13+)

```python
# Allow configuration without section header
config = configparser.ConfigParser(allow_unnamed_section=True)
config.read_string("""
key1 = value1
key2 = value2

[named_section]
key3 = value3
""")

# Access unnamed section
from configparser import UNNAMED_SECTION
value = config.get(UNNAMED_SECTION, 'key1')
```

## Production Pattern: ConfigManager

```python
import configparser
from pathlib import Path
from typing import Any, TypeVar, Callable, Optional
from dataclasses import dataclass
import os

T = TypeVar('T')

@dataclass
class ConfigSource:
    """Configuration source with priority."""
    path: Path
    required: bool = False
    encoding: str = 'utf-8'

class ConfigManager:
    """Production-grade configuration management with layered loading."""
    
    def __init__(
        self,
        sources: list[ConfigSource] | None = None,
        env_prefix: str = '',
        interpolation: configparser.Interpolation | None = None
    ):
        self._parser = configparser.ConfigParser(
            interpolation=interpolation or configparser.ExtendedInterpolation(),
            converters={
                'list': self._parse_list,
                'path': lambda x: Path(x).expanduser(),
                'json': self._parse_json
            }
        )
        self._env_prefix = env_prefix
        self._sources = sources or []
        self._loaded = False
    
    @staticmethod
    def _parse_list(value: str) -> list[str]:
        """Parse comma-separated list."""
        return [item.strip() for item in value.split(',') if item.strip()]
    
    @staticmethod
    def _parse_json(value: str) -> Any:
        """Parse JSON value."""
        import json
        return json.loads(value)
    
    def load(self) -> 'ConfigManager':
        """Load configuration from all sources."""
        for source in self._sources:
            if source.path.exists():
                self._parser.read(source.path, encoding=source.encoding)
            elif source.required:
                raise FileNotFoundError(f"Required config not found: {source.path}")
        
        self._loaded = True
        return self
    
    def _env_key(self, section: str, option: str) -> str:
        """Generate environment variable name."""
        prefix = f"{self._env_prefix}_" if self._env_prefix else ""
        return f"{prefix}{section}_{option}".upper().replace('.', '_')
    
    def get(
        self,
        section: str,
        option: str,
        fallback: T = None,
        converter: Callable[[str], T] | None = None
    ) -> T | str | None:
        """Get configuration value with env override support."""
        # Check environment variable first
        env_key = self._env_key(section, option)
        env_value = os.environ.get(env_key)
        
        if env_value is not None:
            return converter(env_value) if converter else env_value
        
        # Fall back to config file
        try:
            value = self._parser.get(section, option)
            return converter(value) if converter else value
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    
    def get_int(self, section: str, option: str, fallback: int = 0) -> int:
        """Get integer configuration value."""
        return self.get(section, option, fallback, int)
    
    def get_float(self, section: str, option: str, fallback: float = 0.0) -> float:
        """Get float configuration value."""
        return self.get(section, option, fallback, float)
    
    def get_bool(self, section: str, option: str, fallback: bool = False) -> bool:
        """Get boolean configuration value."""
        def parse_bool(value: str) -> bool:
            return value.lower() in ('1', 'yes', 'true', 'on', 'enabled')
        return self.get(section, option, fallback, parse_bool)
    
    def get_list(self, section: str, option: str, fallback: list | None = None) -> list:
        """Get list configuration value."""
        return self.get(section, option, fallback or [], self._parse_list)
    
    def get_path(self, section: str, option: str, fallback: Path | None = None) -> Path | None:
        """Get path configuration value."""
        result = self.get(section, option, None, lambda x: Path(x).expanduser())
        return result if result else fallback
    
    def sections(self) -> list[str]:
        """Get all section names."""
        return self._parser.sections()
    
    def has_section(self, section: str) -> bool:
        """Check if section exists."""
        return self._parser.has_section(section)
    
    def items(self, section: str) -> list[tuple[str, str]]:
        """Get all items in a section."""
        return list(self._parser.items(section))
    
    def set(self, section: str, option: str, value: Any) -> None:
        """Set a configuration value."""
        if not self._parser.has_section(section):
            self._parser.add_section(section)
        self._parser.set(section, option, str(value))
    
    def save(self, path: Path, encoding: str = 'utf-8') -> None:
        """Save configuration to file."""
        with open(path, 'w', encoding=encoding) as f:
            self._parser.write(f)
    
    def validate(self, schema: dict[str, dict[str, type]]) -> list[str]:
        """Validate configuration against schema."""
        errors = []
        for section, options in schema.items():
            if not self.has_section(section):
                errors.append(f"Missing section: [{section}]")
                continue
            
            for option, expected_type in options.items():
                try:
                    value = self.get(section, option)
                    if value is None:
                        errors.append(f"Missing option: [{section}] {option}")
                    elif expected_type == int:
                        int(value)
                    elif expected_type == float:
                        float(value)
                    elif expected_type == bool:
                        if value.lower() not in ('1', '0', 'yes', 'no', 'true', 'false', 'on', 'off'):
                            errors.append(f"Invalid boolean: [{section}] {option} = {value}")
                except ValueError as e:
                    errors.append(f"Type error: [{section}] {option} - {e}")
        
        return errors

# Usage example
if __name__ == '__main__':
    config = ConfigManager(
        sources=[
            ConfigSource(Path('defaults.ini'), required=True),
            ConfigSource(Path('local.ini'), required=False),
            ConfigSource(Path.home() / '.myapp.ini', required=False),
        ],
        env_prefix='MYAPP'
    ).load()
    
    # Get with type conversion and env override
    db_host = config.get('database', 'host', fallback='localhost')
    db_port = config.get_int('database', 'port', fallback=5432)
    debug = config.get_bool('app', 'debug', fallback=False)
    allowed = config.get_list('security', 'allowed_hosts')
    
    # Validate configuration
    schema = {
        'database': {'host': str, 'port': int},
        'app': {'name': str, 'debug': bool}
    }
    errors = config.validate(schema)
    if errors:
        for error in errors:
            print(f"Config error: {error}")
```

## Exception Handling

```python
from configparser import (
    Error,                    # Base exception
    NoSectionError,           # Section not found
    NoOptionError,            # Option not found
    DuplicateSectionError,    # Duplicate section in strict mode
    DuplicateOptionError,     # Duplicate option in strict mode
    InterpolationError,       # Interpolation failed
    InterpolationDepthError,  # Too many interpolation levels
    InterpolationMissingOptionError,  # Referenced option missing
    ParsingError,             # Malformed config file
    MissingSectionHeaderError # No section header found
)

try:
    config.read('config.ini')
    value = config.get('section', 'option')
except NoSectionError as e:
    print(f"Section not found: {e.section}")
except NoOptionError as e:
    print(f"Option not found: {e.option} in {e.section}")
except ParsingError as e:
    print(f"Parse error in {e.source}: {e}")
except InterpolationError as e:
    print(f"Interpolation failed: {e}")
```

## Key Takeaways

| Feature | Recommendation |
|---------|----------------|
| Interpolation | Use ExtendedInterpolation for cross-section refs |
| Type conversion | Use getint/getfloat/getboolean with fallback |
| Environment override | Check env vars before config file values |
| Validation | Validate required sections/options at startup |
| DEFAULT section | Use for shared values across sections |
| Case sensitivity | Keep default (insensitive) unless needed |
| Strict mode | Keep enabled (default) to catch duplicates |
