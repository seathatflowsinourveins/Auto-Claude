#!/usr/bin/env python3
"""
UNLEASH Code Intelligence Validation Suite
==========================================
Part of the Unified Code Intelligence Architecture (5-Layer, 2026)

Validates that all components of the code intelligence stack are properly
installed and configured:
- L0 Real-time: LSP servers (pyright, typescript-language-server)
- L1 Deep Analysis: narsil-mcp
- L2 Semantic: Qdrant + DeepContext
- L3 AST Tools: ast-grep, semgrep
- L4 Indexing: code-index-mcp

Usage:
    python scripts/validate_code_intelligence.py [--verbose] [--json]

Exit codes:
    0 - All validations passed
    1 - Some validations failed
    2 - Critical components missing
"""

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    component: str
    layer: str
    status: str  # "pass", "warn", "fail"
    message: str
    version: str | None = None
    details: dict[str, Any] | None = None


def log(message: str, level: str = "INFO") -> None:
    """Structured logging with Windows-safe characters."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    # Use ASCII-safe symbols for Windows compatibility
    symbols = {"INFO": "[i]", "PASS": "[+]", "WARN": "[!]", "FAIL": "[X]", "SKIP": "[-]"}
    symbol = symbols.get(level, "[*]")
    # Force UTF-8 output
    import sys
    try:
        print(f"[{timestamp}] {symbol} {message}")
    except UnicodeEncodeError:
        print(f"[{timestamp}] {symbol} {message.encode('ascii', 'replace').decode()}")


def check_command(cmd: list[str], timeout: int = 10) -> tuple[bool, str]:
    """Run a command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout.strip() or result.stderr.strip()
    except FileNotFoundError:
        return False, "Command not found"
    except subprocess.TimeoutExpired:
        return False, "Command timed out"


def check_binary_in_path(name: str) -> tuple[bool, str | None]:
    """Check if a binary is in PATH."""
    path = shutil.which(name)
    return path is not None, path


def validate_l0_lsp() -> list[ValidationResult]:
    """Validate L0 Real-time LSP servers."""
    results = []

    # Check pyright
    found, path = check_binary_in_path("pyright")
    if found:
        success, output = check_command(["pyright", "--version"])
        version = output.split()[-1] if success else None
        results.append(ValidationResult(
            component="pyright",
            layer="L0",
            status="pass" if success else "warn",
            message=f"Python LSP available at {path}",
            version=version
        ))
    else:
        results.append(ValidationResult(
            component="pyright",
            layer="L0",
            status="fail",
            message="pyright not found. Install: pip install pyright"
        ))

    # Check typescript-language-server
    found, path = check_binary_in_path("typescript-language-server")
    if found:
        success, output = check_command(["typescript-language-server", "--version"])
        version = output if success else None
        results.append(ValidationResult(
            component="typescript-language-server",
            layer="L0",
            status="pass" if success else "warn",
            message=f"TypeScript LSP available at {path}",
            version=version
        ))
    else:
        results.append(ValidationResult(
            component="typescript-language-server",
            layer="L0",
            status="fail",
            message="typescript-language-server not found. Install: npm install -g typescript-language-server"
        ))

    # Check mcp-language-server bridge
    found, path = check_binary_in_path("mcp-language-server")
    if found:
        results.append(ValidationResult(
            component="mcp-language-server",
            layer="L0",
            status="pass",
            message=f"LSP→MCP bridge available at {path}"
        ))
    else:
        # Check in Go bin
        go_path = Path.home() / "go" / "bin" / "mcp-language-server.exe"
        if go_path.exists():
            results.append(ValidationResult(
                component="mcp-language-server",
                layer="L0",
                status="warn",
                message=f"Found at {go_path} but not in PATH"
            ))
        else:
            results.append(ValidationResult(
                component="mcp-language-server",
                layer="L0",
                status="fail",
                message="mcp-language-server not found. Install: go install github.com/nickcdryan/mcp-language-server@latest"
            ))

    return results


def validate_l1_deep() -> list[ValidationResult]:
    """Validate L1 Deep Analysis (narsil-mcp)."""
    results = []

    found, path = check_binary_in_path("narsil-mcp")
    if found:
        success, output = check_command(["narsil-mcp", "--version"])
        version = output.split()[-1] if success and output else None
        results.append(ValidationResult(
            component="narsil-mcp",
            layer="L1",
            status="pass" if success else "warn",
            message=f"76-tool Rust MCP available at {path}",
            version=version,
            details={"features": ["taint-analysis", "call-graphs", "security", "neural-search"]}
        ))
    else:
        # Check in cargo bin
        cargo_path = Path.home() / ".cargo" / "bin" / "narsil-mcp.exe"
        if cargo_path.exists():
            results.append(ValidationResult(
                component="narsil-mcp",
                layer="L1",
                status="warn",
                message=f"Found at {cargo_path} but not in PATH"
            ))
        else:
            results.append(ValidationResult(
                component="narsil-mcp",
                layer="L1",
                status="fail",
                message="narsil-mcp not found. Install: cargo install narsil-mcp"
            ))

    return results


def validate_l2_semantic() -> list[ValidationResult]:
    """Validate L2 Semantic Search (Qdrant)."""
    results = []

    # Check Qdrant
    try:
        import httpx
        response = httpx.get("http://localhost:6333/collections", timeout=5)
        if response.status_code == 200:
            collections = response.json().get("result", {}).get("collections", [])
            results.append(ValidationResult(
                component="qdrant",
                layer="L2",
                status="pass",
                message=f"Qdrant running at localhost:6333 ({len(collections)} collections)",
                details={"collections": [c.get("name") for c in collections]}
            ))
        else:
            results.append(ValidationResult(
                component="qdrant",
                layer="L2",
                status="warn",
                message=f"Qdrant returned status {response.status_code}"
            ))
    except ImportError:
        results.append(ValidationResult(
            component="qdrant",
            layer="L2",
            status="warn",
            message="Cannot check Qdrant (httpx not installed)"
        ))
    except Exception:
        results.append(ValidationResult(
            component="qdrant",
            layer="L2",
            status="fail",
            message="Qdrant not running. Start: docker run -d -p 6333:6333 qdrant/qdrant"
        ))

    return results


def validate_l3_ast() -> list[ValidationResult]:
    """Validate L3 AST Tools (ast-grep, semgrep)."""
    results = []

    # Check ast-grep
    found, path = check_binary_in_path("ast-grep")
    if not found:
        found, path = check_binary_in_path("sg")  # Alternative name
    if found:
        success, output = check_command(["ast-grep", "--version"] if "ast-grep" in (path or "") else ["sg", "--version"])
        version = output.split()[-1] if success else None
        results.append(ValidationResult(
            component="ast-grep",
            layer="L3",
            status="pass" if success else "warn",
            message=f"AST pattern matching available at {path}",
            version=version
        ))
    else:
        results.append(ValidationResult(
            component="ast-grep",
            layer="L3",
            status="fail",
            message="ast-grep not found. Install: cargo install ast-grep"
        ))

    # Check semgrep
    found, path = check_binary_in_path("semgrep")
    if found:
        success, output = check_command(["semgrep", "--version"])
        version = output.split()[0] if success else None
        results.append(ValidationResult(
            component="semgrep",
            layer="L3",
            status="pass" if success else "warn",
            message=f"Security scanning available at {path}",
            version=version
        ))
    else:
        results.append(ValidationResult(
            component="semgrep",
            layer="L3",
            status="warn",
            message="semgrep not found. Install: pip install semgrep"
        ))

    return results


def validate_l4_indexing() -> list[ValidationResult]:
    """Validate L4 Indexing (code-index-mcp)."""
    results = []

    # Check code-index-mcp via uvx
    success, output = check_command(["uvx", "code-index-mcp", "--help"], timeout=30)
    if success:
        results.append(ValidationResult(
            component="code-index-mcp",
            layer="L4",
            status="pass",
            message="48-language code indexer available via uvx"
        ))
    else:
        results.append(ValidationResult(
            component="code-index-mcp",
            layer="L4",
            status="warn",
            message="code-index-mcp not available. Install: uvx code-index-mcp"
        ))

    return results


def validate_mcp_config() -> list[ValidationResult]:
    """Validate MCP configuration files."""
    results = []

    # Check global config
    global_config = Path.home() / ".claude" / "mcp_servers_OPTIMAL.json"
    if global_config.exists():
        try:
            with open(global_config) as f:
                config = json.load(f)
            servers = config.get("mcpServers", {})
            code_intel_servers = ["narsil", "lsp-python", "lsp-typescript", "deepcontext", "code-index"]
            found = [s for s in code_intel_servers if s in servers]
            results.append(ValidationResult(
                component="mcp-config-global",
                layer="CONFIG",
                status="pass" if len(found) >= 3 else "warn",
                message=f"Global MCP config has {len(found)}/{len(code_intel_servers)} code intelligence servers",
                details={"configured": found, "missing": [s for s in code_intel_servers if s not in found]}
            ))
        except json.JSONDecodeError:
            results.append(ValidationResult(
                component="mcp-config-global",
                layer="CONFIG",
                status="fail",
                message="Invalid JSON in global MCP config"
            ))
    else:
        results.append(ValidationResult(
            component="mcp-config-global",
            layer="CONFIG",
            status="fail",
            message=f"Global MCP config not found at {global_config}"
        ))

    # Check project config
    project_config = Path("Z:/insider/AUTO CLAUDE/unleash/platform/.mcp.json")
    if project_config.exists():
        try:
            with open(project_config) as f:
                config = json.load(f)
            servers = config.get("mcpServers", {})
            results.append(ValidationResult(
                component="mcp-config-project",
                layer="CONFIG",
                status="pass",
                message=f"Project MCP config has {len(servers)} servers configured"
            ))
        except json.JSONDecodeError:
            results.append(ValidationResult(
                component="mcp-config-project",
                layer="CONFIG",
                status="fail",
                message="Invalid JSON in project MCP config"
            ))
    else:
        results.append(ValidationResult(
            component="mcp-config-project",
            layer="CONFIG",
            status="warn",
            message="Project MCP config not found"
        ))

    return results


def run_all_validations() -> list[ValidationResult]:
    """Run all validation checks."""
    all_results = []

    log("=" * 60)
    log("UNLEASH Code Intelligence Validation Suite")
    log("=" * 60)

    log("\n[L0 Real-time] LSP Servers")
    all_results.extend(validate_l0_lsp())

    log("\n[L1 Deep Analysis] narsil-mcp")
    all_results.extend(validate_l1_deep())

    log("\n[L2 Semantic] Qdrant + Embeddings")
    all_results.extend(validate_l2_semantic())

    log("\n[L3 AST Tools] ast-grep, semgrep")
    all_results.extend(validate_l3_ast())

    log("\n[L4 Indexing] code-index-mcp")
    all_results.extend(validate_l4_indexing())

    log("\n[CONFIG] MCP Configuration")
    all_results.extend(validate_mcp_config())

    return all_results


def print_results(results: list[ValidationResult], verbose: bool = False) -> None:
    """Print validation results."""
    log("\n" + "=" * 60)
    log("VALIDATION SUMMARY")
    log("=" * 60)

    by_layer: dict[str, list[ValidationResult]] = {}
    for r in results:
        by_layer.setdefault(r.layer, []).append(r)

    for layer in ["L0", "L1", "L2", "L3", "L4", "CONFIG"]:
        if layer not in by_layer:
            continue
        layer_results = by_layer[layer]
        pass_count = sum(1 for r in layer_results if r.status == "pass")
        warn_count = sum(1 for r in layer_results if r.status == "warn")
        fail_count = sum(1 for r in layer_results if r.status == "fail")

        status_str = []
        if pass_count:
            status_str.append(f"+{pass_count}")
        if warn_count:
            status_str.append(f"!{warn_count}")
        if fail_count:
            status_str.append(f"X{fail_count}")

        log(f"\n[{layer}] {' '.join(status_str)}")

        for r in layer_results:
            level = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[r.status]
            version_str = f" (v{r.version})" if r.version else ""
            log(f"  {r.component}{version_str}: {r.message}", level)

            if verbose and r.details:
                for k, v in r.details.items():
                    log(f"    {k}: {v}", "INFO")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate UNLEASH code intelligence stack")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    results = run_all_validations()

    if args.json:
        output = [
            {
                "component": r.component,
                "layer": r.layer,
                "status": r.status,
                "message": r.message,
                "version": r.version,
                "details": r.details
            }
            for r in results
        ]
        print(json.dumps(output, indent=2))
    else:
        print_results(results, verbose=args.verbose)

    # Determine exit code
    fail_count = sum(1 for r in results if r.status == "fail")
    critical_components = ["narsil-mcp", "pyright", "mcp-language-server"]
    critical_failures = sum(1 for r in results if r.status == "fail" and r.component in critical_components)

    if critical_failures > 0:
        log(f"\n{critical_failures} critical component(s) missing!", "FAIL")
        sys.exit(2)
    elif fail_count > 0:
        log(f"\n{fail_count} validation(s) failed", "WARN")
        sys.exit(1)
    else:
        log("\nAll validations passed!", "PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
