#!/usr/bin/env python3
"""
Phase 1 Environment Validation Script
Validates all prerequisites for Unleash platform.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Tuple


def check_python_version() -> Tuple[bool, str]:
    """Check Python version >= 3.11"""
    version = sys.version_info
    if version >= (3, 11):
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor} (need 3.11+)"


def check_node_version() -> Tuple[bool, str]:
    """Check Node.js version >= 18"""
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True
        )
        version = result.stdout.strip()
        major = int(version.lstrip("v").split(".")[0])
        if major >= 18:
            return True, version
        return False, f"{version} (need v18+)"
    except FileNotFoundError:
        return False, "Node.js not found"


def check_uv_available() -> Tuple[bool, str]:
    """Check if uv is available"""
    try:
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True
        )
        return True, result.stdout.strip()
    except FileNotFoundError:
        return False, "uv not found"


def check_directory(path: str) -> Tuple[bool, str]:
    """Check if directory exists and is not empty"""
    p = Path(path)
    if not p.exists():
        return False, f"Directory {path} does not exist"
    if not p.is_dir():
        return False, f"{path} is not a directory"
    contents = list(p.iterdir())
    if len(contents) == 0:
        return False, f"Directory {path} is empty"
    return True, f"{len(contents)} items"


def check_env_file() -> Tuple[bool, str]:
    """Check if .env file exists"""
    if Path(".env").exists():
        return True, ".env exists"
    if Path(".env.template").exists():
        return False, ".env.template exists, copy to .env"
    return False, "No .env or .env.template found"


def check_imports() -> Tuple[bool, str]:
    """Check base Python imports"""
    try:
        import structlog
        import httpx
        import pydantic
        from dotenv import load_dotenv
        from rich.console import Console
        return True, "All base imports OK"
    except ImportError as e:
        return False, f"Import error: {e}"


def main():
    """Run all validation checks."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("\n[bold blue]Phase 1: Environment Validation[/bold blue]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")

    checks = [
        ("Python Version", check_python_version),
        ("Node.js Version", check_node_version),
        ("uv Package Manager", check_uv_available),
        ("sdks/ Directory", lambda: check_directory("sdks")),
        ("stack/ Directory", lambda: check_directory("stack")),
        ("core/ Directory", lambda: check_directory("core")),
        ("platform/ Directory", lambda: check_directory("platform")),
        ("docs/ Directory", lambda: check_directory("docs")),
        ("Environment File", check_env_file),
        ("Python Imports", check_imports),
    ]

    all_passed = True
    for name, check_fn in checks:
        passed, details = check_fn()
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(name, status, details)
        if not passed:
            all_passed = False

    console.print(table)
    console.print()

    if all_passed:
        console.print("[bold green]All checks passed! Ready for Phase 2.[/bold green]")
        return 0
    else:
        console.print("[bold red]Some checks failed. Please fix before proceeding.[/bold red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
