# Python argparse Module - Production Patterns (Jan 2026)

## Overview
The `argparse` module is Python's recommended command-line parsing library. It handles positional and optional arguments, generates help messages, issues errors for invalid input, and supports subcommands.

## ArgumentParser Basics

### Creating a Parser
```python
import argparse

parser = argparse.ArgumentParser(
    prog="myapp",                      # Program name (default: sys.argv[0])
    description="What the program does",  # Shown in help
    epilog="Additional info at end",   # After argument help
    formatter_class=argparse.RawDescriptionHelpFormatter,  # Preserve formatting
    add_help=True,                     # Add -h/--help (default)
    allow_abbrev=True,                 # Allow abbreviated options (default)
    exit_on_error=True                 # Exit on parse error (default)
)
```

### Formatter Classes
```python
# Preserve newlines in description/epilog
argparse.RawDescriptionHelpFormatter

# Preserve newlines + show defaults
argparse.RawTextHelpFormatter

# Show default values in help
argparse.ArgumentDefaultsHelpFormatter

# Fix help text width
argparse.HelpFormatter  # Base class

# Combine formatters
class MyFormatter(argparse.ArgumentDefaultsHelpFormatter,
                  argparse.RawDescriptionHelpFormatter):
    pass
```

## Adding Arguments

### Positional Arguments
```python
# Required positional argument
parser.add_argument("filename", help="Input file path")

# With type conversion
parser.add_argument("count", type=int, help="Number of items")

# Multiple positional arguments
parser.add_argument("files", nargs="+", help="One or more files")
```

### Optional Arguments
```python
# Long option
parser.add_argument("--verbose", help="Enable verbose output")

# Short and long options
parser.add_argument("-v", "--verbose", help="Verbose mode")

# Flag (store_true action)
parser.add_argument("-q", "--quiet", action="store_true",
                    help="Suppress output")

# Flag with False default
parser.add_argument("--no-cache", action="store_false", dest="cache",
                    help="Disable caching")
```

### Common Parameters

```python
parser.add_argument(
    "-o", "--output",
    type=str,                    # Type converter (int, float, str, Path, etc.)
    default="output.txt",        # Default value if not provided
    required=True,               # Make optional argument required
    help="Output file path",     # Help text
    metavar="FILE",              # Name shown in usage/help
    dest="output_file",          # Attribute name in Namespace
    choices=["json", "xml", "csv"],  # Restrict to valid values
    nargs="?",                   # Number of arguments (see below)
    action="store",              # Action type (see below)
)
```

## nargs Values

```python
# N (integer): Exactly N arguments
parser.add_argument("--point", nargs=2, type=float, metavar=("X", "Y"))
# Usage: --point 1.0 2.0

# '?': Zero or one argument
parser.add_argument("--config", nargs="?", const="default.cfg", default=None)
# --config         → "default.cfg" (uses const)
# --config foo.cfg → "foo.cfg"
# (omitted)        → None (uses default)

# '*': Zero or more arguments
parser.add_argument("--files", nargs="*")
# --files          → []
# --files a b c    → ["a", "b", "c"]

# '+': One or more arguments
parser.add_argument("--files", nargs="+")
# --files          → Error (requires at least one)
# --files a b c    → ["a", "b", "c"]

# argparse.REMAINDER: All remaining arguments
parser.add_argument("command", nargs=argparse.REMAINDER)
# mycli run --verbose → command=["run", "--verbose"]
```

## Action Types

```python
# 'store' (default): Store the argument value
parser.add_argument("--name", action="store")

# 'store_const': Store a constant value
parser.add_argument("--version", action="store_const", const="1.0.0")

# 'store_true' / 'store_false': Boolean flags
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--no-color", action="store_false", dest="color")

# 'append': Collect values into a list
parser.add_argument("--include", action="append")
# --include foo --include bar → ["foo", "bar"]

# 'append_const': Append constant to list
parser.add_argument("-v", action="append_const", const=1, dest="verbosity")
# -vvv → [1, 1, 1]

# 'count': Count occurrences
parser.add_argument("-v", "--verbose", action="count", default=0)
# -vvv → 3

# 'extend': Extend list with nargs values
parser.add_argument("--files", action="extend", nargs="+")
# --files a b --files c → ["a", "b", "c"]

# 'help': Print help and exit (auto-added with -h)
# 'version': Print version and exit
parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

# 'BooleanOptionalAction' (Python 3.9+): --flag / --no-flag
parser.add_argument("--feature", action=argparse.BooleanOptionalAction)
# --feature → True, --no-feature → False
```

## Type Converters

```python
# Built-in types
parser.add_argument("--count", type=int)
parser.add_argument("--ratio", type=float)

# pathlib.Path
from pathlib import Path
parser.add_argument("--input", type=Path)

# Custom type function
def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue

parser.add_argument("--threads", type=positive_int, default=4)

# File types (auto-open files)
parser.add_argument("--input", type=argparse.FileType("r", encoding="utf-8"))
parser.add_argument("--output", type=argparse.FileType("w"))
# Note: Files opened immediately; prefer Path and open manually

# Enum type
from enum import Enum

class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

def log_level(value):
    try:
        return LogLevel(value.lower())
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid log level: {value}")

parser.add_argument("--log-level", type=log_level, default=LogLevel.INFO)
```

## Subcommands (Sub-parsers)

```python
import argparse

parser = argparse.ArgumentParser(prog="git")
subparsers = parser.add_subparsers(
    title="commands",
    description="Available commands",
    dest="command",      # Store subcommand name here
    required=True        # Make subcommand required
)

# Clone subcommand
clone_parser = subparsers.add_parser("clone", help="Clone a repository")
clone_parser.add_argument("url", help="Repository URL")
clone_parser.add_argument("--depth", type=int, help="Shallow clone depth")

# Commit subcommand
commit_parser = subparsers.add_parser("commit", help="Commit changes")
commit_parser.add_argument("-m", "--message", required=True, help="Commit message")
commit_parser.add_argument("-a", "--all", action="store_true", help="Stage all")

# Push subcommand
push_parser = subparsers.add_parser("push", help="Push changes")
push_parser.add_argument("remote", nargs="?", default="origin")
push_parser.add_argument("branch", nargs="?", default="main")
push_parser.add_argument("-f", "--force", action="store_true")

args = parser.parse_args()

# Handle subcommands
if args.command == "clone":
    print(f"Cloning {args.url}")
elif args.command == "commit":
    print(f"Committing with message: {args.message}")
elif args.command == "push":
    print(f"Pushing {args.branch} to {args.remote}")
```

### Subcommand with set_defaults
```python
def handle_clone(args):
    print(f"Cloning {args.url}")

def handle_commit(args):
    print(f"Committing: {args.message}")

clone_parser = subparsers.add_parser("clone")
clone_parser.add_argument("url")
clone_parser.set_defaults(func=handle_clone)

commit_parser = subparsers.add_parser("commit")
commit_parser.add_argument("-m", "--message", required=True)
commit_parser.set_defaults(func=handle_commit)

args = parser.parse_args()
if hasattr(args, "func"):
    args.func(args)  # Call the handler function
```

## Argument Groups

```python
parser = argparse.ArgumentParser()

# Logical grouping (for help organization)
input_group = parser.add_argument_group("Input options")
input_group.add_argument("--input", "-i", help="Input file")
input_group.add_argument("--format", choices=["json", "csv"])

output_group = parser.add_argument_group("Output options")
output_group.add_argument("--output", "-o", help="Output file")
output_group.add_argument("--compress", action="store_true")

# Mutually exclusive group
exclusive = parser.add_mutually_exclusive_group(required=False)
exclusive.add_argument("-v", "--verbose", action="store_true")
exclusive.add_argument("-q", "--quiet", action="store_true")
# Can use -v OR -q, but not both
```

## Parsing Arguments

```python
# Parse sys.argv (typical usage)
args = parser.parse_args()

# Parse specific list
args = parser.parse_args(["--verbose", "file.txt"])

# Parse known args (ignore unknown)
args, unknown = parser.parse_known_args()
# Useful for passing through to subprocess

# Access parsed values
print(args.verbose)      # Attribute access
print(vars(args))        # As dictionary
```

## Namespace Object

```python
# Default Namespace
args = parser.parse_args()
print(args)  # Namespace(verbose=True, file='input.txt')

# Custom namespace
class MyNamespace:
    debug = False

ns = MyNamespace()
args = parser.parse_args(namespace=ns)
print(ns.debug)  # Preserved default
print(ns.verbose)  # Added by parser

# Convert to dict
config = vars(args)

# Merge with defaults
defaults = {"timeout": 30, "retries": 3}
config = {**defaults, **vars(args)}
```

## Handling Errors

```python
import sys

parser = argparse.ArgumentParser(exit_on_error=False)
parser.add_argument("--port", type=int, required=True)

try:
    args = parser.parse_args()
except argparse.ArgumentError as e:
    print(f"Argument error: {e}", file=sys.stderr)
    sys.exit(2)
except SystemExit:
    # Raised by parse_args on error when exit_on_error=True
    sys.exit(2)

# Custom error handling
class GracefulParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f"Error: {message}\n")
        self.print_help()
        sys.exit(2)
```

## Environment Variable Fallback

```python
import os
import argparse

def env_or_default(env_var, default):
    """Get value from environment or use default."""
    return os.environ.get(env_var, default)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--api-key",
    default=env_or_default("API_KEY", None),
    help="API key (default: $API_KEY)"
)
parser.add_argument(
    "--timeout",
    type=int,
    default=int(env_or_default("TIMEOUT", "30")),
    help="Timeout in seconds (default: $TIMEOUT or 30)"
)
```

## Configuration File Support

```python
import argparse
import json
from pathlib import Path

def load_config(config_path: Path) -> dict:
    if config_path.exists():
        return json.loads(config_path.read_text())
    return {}

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("config.json"))
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--output", type=Path)

# First pass: get config file
args, remaining = parser.parse_known_args()

# Load config and set as defaults
config = load_config(args.config)
parser.set_defaults(**config)

# Second pass: CLI overrides config
args = parser.parse_args(remaining)
```

## Production CLI Pattern

```python
#!/usr/bin/env python3
"""Production CLI application pattern."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(verbose: bool, quiet: bool) -> logging.Logger:
    """Configure logging based on verbosity."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        prog="myapp",
        description="My production application",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global options
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )
    
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    verbosity.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress non-error output"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path.home() / ".myapp" / "config.json",
        help="Configuration file path"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True
    )
    
    # 'run' command
    run_parser = subparsers.add_parser(
        "run",
        help="Run the main process"
    )
    run_parser.add_argument(
        "input",
        type=Path,
        help="Input file path"
    )
    run_parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file path"
    )
    run_parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="Show what would be done"
    )
    run_parser.set_defaults(func=cmd_run)
    
    # 'init' command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize configuration"
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config"
    )
    init_parser.set_defaults(func=cmd_init)
    
    return parser


def cmd_run(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle 'run' command."""
    logger.info(f"Processing {args.input}")
    
    if args.dry_run:
        logger.info("Dry run - no changes made")
        return 0
    
    # Processing logic here
    output = args.output or args.input.with_suffix(".out")
    logger.debug(f"Writing to {output}")
    
    return 0


def cmd_init(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle 'init' command."""
    config_dir = args.config.parent
    
    if args.config.exists() and not args.force:
        logger.error(f"Config exists: {args.config}. Use --force to overwrite.")
        return 1
    
    config_dir.mkdir(parents=True, exist_ok=True)
    args.config.write_text("{}")
    logger.info(f"Created config: {args.config}")
    
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    logger = setup_logging(args.verbose, args.quiet)
    
    try:
        return args.func(args, logger)
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

## Testing CLI Applications

```python
import pytest
from myapp import create_parser, main

def test_parser_help():
    """Test help exits cleanly."""
    parser = create_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0

def test_parser_version():
    """Test version flag."""
    parser = create_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0

def test_run_command(tmp_path):
    """Test run command parsing."""
    parser = create_parser()
    input_file = tmp_path / "input.txt"
    input_file.touch()
    
    args = parser.parse_args(["run", str(input_file), "--dry-run"])
    assert args.command == "run"
    assert args.input == input_file
    assert args.dry_run is True

def test_main_integration(tmp_path, capsys):
    """Test full CLI execution."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("test data")
    
    exit_code = main(["run", str(input_file), "--dry-run", "-v"])
    assert exit_code == 0

def test_missing_required():
    """Test missing required argument."""
    parser = create_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["run"])  # Missing input
    assert exc.value.code == 2  # argparse error exit code
```

## Key Takeaways

1. **Use `__name__` pattern** - Keep parser creation in a function for testing
2. **ArgumentDefaultsHelpFormatter** - Shows defaults in help text
3. **set_defaults(func=handler)** - Clean subcommand dispatch
4. **type=Path** - Use pathlib.Path for file arguments
5. **BooleanOptionalAction** - Modern --flag/--no-flag (Python 3.9+)
6. **parse_known_args** - For pass-through to subprocesses
7. **Mutually exclusive groups** - For conflicting options like -v/-q
8. **exit_on_error=False** - For graceful error handling
9. **main(argv=None)** - Accept argv for testing
10. **Return exit codes** - 0=success, 1=error, 2=usage error
