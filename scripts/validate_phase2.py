#!/usr/bin/env python3
"""
Phase 2 Protocol Layer Validation Script
Validates all Protocol Layer (Layer 0) components.
"""

import os
import sys
import json
from pathlib import Path
from typing import Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_sdk_import(name: str, package: str) -> Tuple[bool, str]:
    """Check if an SDK can be imported."""
    try:
        __import__(package)
        return True, f"{name} importable"
    except ImportError as e:
        return False, f"{name} import failed: {e}"


def check_core_module(module_path: str, description: str) -> Tuple[bool, str]:
    """Check if a core module exists and is importable."""
    try:
        # Check file exists
        full_path = project_root / module_path.replace(".", "/")
        if full_path.with_suffix(".py").exists():
            # Try importing
            __import__(module_path)
            return True, f"{description} OK"
        elif full_path.is_dir() and (full_path / "__init__.py").exists():
            __import__(module_path)
            return True, f"{description} OK"
        else:
            return False, f"{description} file not found"
    except Exception as e:
        return False, f"{description} error: {e}"


def check_api_key(name: str, env_var: str) -> Tuple[bool, str]:
    """Check if an API key is configured."""
    value = os.getenv(env_var, "")
    if value and not value.startswith("your-"):
        return True, f"{name} configured"
    return False, f"{name} not configured (optional)"


def check_mcp_config() -> Tuple[bool, str]:
    """Check if MCP configuration exists and has unleash server."""
    config_path = project_root / "platform" / "config" / "mcp_servers.json"
    if not config_path.exists():
        return False, "mcp_servers.json not found"

    try:
        with open(config_path) as f:
            config = json.load(f)

        if "mcpServers" in config and "unleash" in config["mcpServers"]:
            return True, "Unleash MCP server configured"
        return False, "Unleash server not in config"
    except Exception as e:
        return False, f"Config parse error: {e}"


def check_llm_gateway_classes() -> Tuple[bool, str]:
    """Check LLM Gateway exports required classes."""
    try:
        from core.llm_gateway import (
            LLMGateway,
            Provider,
            ModelConfig,
            Message,
            CompletionResponse,
        )
        return True, "All gateway classes exported"
    except ImportError as e:
        return False, f"Missing exports: {e}"


def check_mcp_server_tools() -> Tuple[bool, str]:
    """Check MCP Server can create tools."""
    try:
        from core.mcp_server import create_mcp_server, FASTMCP_AVAILABLE
        if not FASTMCP_AVAILABLE:
            return False, "FastMCP not available"

        # Try creating server (doesn't start it)
        server = create_mcp_server("test")
        return True, "MCP Server creates successfully"
    except Exception as e:
        return False, f"Server creation failed: {e}"


def check_providers() -> Tuple[bool, str]:
    """Check provider modules exist."""
    try:
        from core.providers import (
            AnthropicProvider,
            ClaudeMessage,
            OpenAIProvider,
            OpenAIMessage,
        )
        return True, "All providers exported"
    except ImportError as e:
        return False, f"Provider import failed: {e}"


def main():
    """Run all Phase 2 validation checks."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    # Force UTF-8 for Windows console compatibility
    import io
    import sys
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    console = Console(force_terminal=True)
    console.print(Panel.fit(
        "[bold blue]Phase 2: Protocol Layer Validation[/bold blue]\n"
        "[dim]Layer 0 - 5 Core SDKs[/dim]",
        border_style="blue"
    ))
    console.print()

    # SDK Checks
    sdk_table = Table(title="SDK Availability", show_header=True, header_style="bold magenta")
    sdk_table.add_column("SDK", style="cyan")
    sdk_table.add_column("Status", style="green")
    sdk_table.add_column("Details")

    sdk_checks = [
        ("MCP SDK", "mcp"),
        ("FastMCP", "fastmcp"),
        ("LiteLLM", "litellm"),
        ("Anthropic", "anthropic"),
        ("OpenAI", "openai"),
    ]

    sdk_passed = True
    for name, package in sdk_checks:
        passed, details = check_sdk_import(name, package)
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        sdk_table.add_row(name, status, details)
        if not passed:
            sdk_passed = False

    console.print(sdk_table)
    console.print()

    # Core Module Checks
    core_table = Table(title="Core Modules", show_header=True, header_style="bold magenta")
    core_table.add_column("Module", style="cyan")
    core_table.add_column("Status", style="green")
    core_table.add_column("Details")

    core_checks = [
        ("core.llm_gateway", "LLM Gateway"),
        ("core.mcp_server", "MCP Server"),
        ("core.providers", "Provider Package"),
    ]

    core_passed = True
    for module, description in core_checks:
        passed, details = check_core_module(module, description)
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        core_table.add_row(module, status, details)
        if not passed:
            core_passed = False

    console.print(core_table)
    console.print()

    # Functional Checks
    func_table = Table(title="Functional Validation", show_header=True, header_style="bold magenta")
    func_table.add_column("Check", style="cyan")
    func_table.add_column("Status", style="green")
    func_table.add_column("Details")

    func_checks = [
        ("LLM Gateway Classes", check_llm_gateway_classes),
        ("MCP Server Tools", check_mcp_server_tools),
        ("Provider Exports", check_providers),
        ("MCP Configuration", check_mcp_config),
    ]

    func_passed = True
    for name, check_fn in func_checks:
        passed, details = check_fn()
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        func_table.add_row(name, status, details)
        if not passed:
            func_passed = False

    console.print(func_table)
    console.print()

    # API Key Status (informational, not required)
    key_table = Table(title="API Key Status (Optional)", show_header=True, header_style="bold yellow")
    key_table.add_column("Provider", style="cyan")
    key_table.add_column("Status", style="green")
    key_table.add_column("Details")

    key_checks = [
        ("Anthropic", "ANTHROPIC_API_KEY"),
        ("OpenAI", "OPENAI_API_KEY"),
    ]

    for name, env_var in key_checks:
        configured, details = check_api_key(name, env_var)
        status = "[green]PASS[/green]" if configured else "[yellow]SKIP[/yellow]"
        key_table.add_row(name, status, details)

    console.print(key_table)
    console.print()

    # Summary
    all_required_passed = sdk_passed and core_passed and func_passed

    if all_required_passed:
        console.print(Panel.fit(
            "[bold green]Phase 2 Validation PASSED[/bold green]\n\n"
            "Protocol Layer (Layer 0) is fully operational.\n"
            "- 5 SDKs installed and importable\n"
            "- Core modules created and functional\n"
            "- MCP server configured\n\n"
            "[dim]Ready for Phase 3: Orchestration Layer[/dim]",
            border_style="green"
        ))
        return 0
    else:
        console.print(Panel.fit(
            "[bold red]Phase 2 Validation FAILED[/bold red]\n\n"
            "Some required checks did not pass.\n"
            "Please review the tables above and fix issues.",
            border_style="red"
        ))
        return 1


if __name__ == "__main__":
    sys.exit(main())
