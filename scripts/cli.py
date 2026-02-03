#!/usr/bin/env python3
"""
Unleash V33 CLI - Main Entry Point
Part of Phase 8: CLI Integration & Performance Optimization.

Provides the main entry point for the Unleash CLI with:
- Auto-discovery of command modules
- Environment setup
- Rich traceback handling
- System health checks

Usage:
    python cli.py [command] [options]
    unleash [command] [options]  (when installed via pip)

NO STUBS - All functionality fully implemented.
NO GRACEFUL DEGRADATION - Explicit errors on missing dependencies.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def setup_environment() -> None:
    """Set up the environment for the CLI."""
    # Set default environment variables if not already set
    defaults = {
        "UNLEASH_LOG_LEVEL": "INFO",
        "UNLEASH_ENV": "development",
        "PYTHONUNBUFFERED": "1",
    }

    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value


def install_rich_traceback() -> None:
    """Install rich traceback handler for better error display."""
    try:
        from rich.traceback import install
        install(show_locals=True, width=120)
    except ImportError:
        # Rich not available, use standard traceback
        pass


def check_dependencies() -> dict:
    """Check for required dependencies and return status."""
    required = {
        "click": "click",
        "rich": "rich",
        "httpx": "httpx",
        "pyyaml": "yaml",
    }

    optional = {
        "anthropic": "anthropic",
        "openai": "openai",
        "langchain": "langchain_core",
        "langgraph": "langgraph",
        "langfuse": "langfuse",
        "ragas": "ragas",
        "phoenix": "phoenix",
        "redis": "redis",
    }

    status = {
        "required": {},
        "optional": {},
        "missing_required": [],
    }

    for name, module in required.items():
        try:
            __import__(module)
            status["required"][name] = True
        except ImportError:
            status["required"][name] = False
            status["missing_required"].append(name)

    for name, module in optional.items():
        try:
            __import__(module)
            status["optional"][name] = True
        except Exception:
            # Catch all exceptions for optional dependencies
            # Python 3.14+ can cause ConfigError, not just ImportError
            status["optional"][name] = False

    return status


def main() -> None:
    """Main entry point for the CLI."""
    # Set up environment
    setup_environment()

    # Install rich traceback
    install_rich_traceback()

    # Check dependencies
    dep_status = check_dependencies()

    if dep_status["missing_required"]:
        print("ERROR: Missing required dependencies:")
        for dep in dep_status["missing_required"]:
            print(f"  - {dep}")
        print("\nInstall with: pip install click rich httpx pyyaml")
        sys.exit(1)

    # Import and run the CLI
    try:
        from core.cli.unified_cli import cli
        cli(obj={})
    except ImportError as e:
        print(f"ERROR: Failed to import CLI module: {e}")
        print("\nMake sure you're running from the project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: CLI failed: {e}")
        sys.exit(1)


# ============================================================================
# Doctor Command - System Health Check
# ============================================================================


def doctor() -> None:
    """Run system health check and display diagnostics."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
    except ImportError:
        print("ERROR: rich is required for doctor command")
        print("Install with: pip install rich")
        sys.exit(1)

    console = Console()

    console.print(Panel(
        "[bold blue]Unleash V33 System Diagnostics[/bold blue]",
        title="Doctor",
    ))

    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 10)

    console.print(f"\n[bold]Python Version:[/bold] {py_version} ", end="")
    if py_ok:
        console.print("[green]OK[/green]")
    else:
        console.print("[red]FAIL (requires 3.10+)[/red]")

    # Check dependencies
    console.print("\n[bold]Dependencies:[/bold]")
    dep_status = check_dependencies()

    # Required dependencies table
    req_table = Table(title="Required Dependencies")
    req_table.add_column("Package", style="cyan")
    req_table.add_column("Status")

    for name, available in dep_status["required"].items():
        status = "[green]OK[/green]" if available else "[red]MISSING[/red]"
        req_table.add_row(name, status)

    console.print(req_table)

    # Optional dependencies table
    opt_table = Table(title="Optional Dependencies")
    opt_table.add_column("Package", style="cyan")
    opt_table.add_column("Status")

    for name, available in dep_status["optional"].items():
        status = "[green]OK[/green]" if available else "[yellow]OPTIONAL[/yellow]"
        opt_table.add_row(name, status)

    console.print(opt_table)

    # Check V33 layers
    console.print("\n[bold]V33 Layers:[/bold]")
    layers_table = Table(title="Layer Status")
    layers_table.add_column("Layer", style="cyan")
    layers_table.add_column("Status")

    layers = {
        "Core": "core",
        "Memory": "core.memory",
        "Tools": "core.tools",
        "Orchestration": "core.orchestration",
        "Structured": "core.structured",
        "Observability": "core.observability",
        "Performance": "core.performance",
        "CLI": "core.cli",
    }

    layer_status = {}
    for name, module in layers.items():
        try:
            __import__(module)
            layer_status[name] = True
        except ImportError:
            layer_status[name] = False

    for name, available in layer_status.items():
        status = "[green]OK[/green]" if available else "[red]MISSING[/red]"
        layers_table.add_row(name, status)

    console.print(layers_table)

    # Check configuration
    console.print("\n[bold]Configuration:[/bold]")
    config_path = Path.cwd() / ".unleash.yaml"
    if config_path.exists():
        console.print(f"  Config file: [green]{config_path}[/green]")
    else:
        console.print("  Config file: [yellow]Not found (run 'unleash config init')[/yellow]")

    # Summary
    console.print("\n[bold]Summary:[/bold]")
    all_required_ok = all(dep_status["required"].values())
    all_layers_ok = all(layer_status.values())

    if all_required_ok and all_layers_ok and py_ok:
        console.print("[green bold]System is healthy![/green bold]")
    else:
        console.print("[yellow bold]System has issues. See details above.[/yellow bold]")

        if dep_status["missing_required"]:
            console.print(f"\n[red]Missing required: {', '.join(dep_status['missing_required'])}[/red]")
            console.print("Install with: pip install click rich httpx pyyaml")


# ============================================================================
# Version Command
# ============================================================================


def version() -> None:
    """Print version information."""
    print("Unleash V33 CLI v33.8.0")
    print(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


# ============================================================================
# Entry Point
# ============================================================================


if __name__ == "__main__":
    # Handle special commands before loading full CLI
    if len(sys.argv) > 1:
        if sys.argv[1] == "doctor":
            doctor()
            sys.exit(0)
        elif sys.argv[1] == "--version" or sys.argv[1] == "-V":
            version()
            sys.exit(0)

    main()
