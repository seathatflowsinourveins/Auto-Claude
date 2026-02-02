#!/usr/bin/env python3
"""
Integration Validation Script for Ultimate Autonomous Architecture v3

This script validates that all components of the architecture can work together:
1. MCP servers are accessible
2. Memory systems (Letta, Graphiti) are configured
3. Claude-Flow orchestration is ready
4. Hooks are properly registered
5. QA pipeline is functional

Run with: python validate_integration.py
"""

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ANSI colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

@dataclass
class ValidationResult:
    component: str
    status: str  # "pass", "fail", "warn"
    message: str
    details: Optional[str] = None

class IntegrationValidator:
    """Validates the integration of all autonomous architecture components."""

    def __init__(self, config_path: str = "./mcp_config.json"):
        self.config_path = config_path
        self.results: List[ValidationResult] = []

    def log(self, result: ValidationResult):
        """Log a validation result."""
        self.results.append(result)
        icon = {"pass": f"{GREEN}✓{RESET}", "fail": f"{RED}✗{RESET}", "warn": f"{YELLOW}⚠{RESET}"}[result.status]
        print(f"  {icon} {result.component}: {result.message}")
        if result.details:
            print(f"      {BLUE}→ {result.details}{RESET}")

    # =====================
    # LAYER 1: ORCHESTRATION
    # =====================

    def validate_claude_flow(self) -> ValidationResult:
        """Check if Claude-Flow is installed and configured."""
        try:
            result = subprocess.run(
                ["npx", "claude-flow@v3alpha", "--version"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return ValidationResult(
                    "Claude-Flow", "pass",
                    "Claude-Flow v3 is installed",
                    f"Version: {result.stdout.strip()}"
                )
            else:
                return ValidationResult(
                    "Claude-Flow", "fail",
                    "Claude-Flow not found",
                    "Run: npx claude-flow@v3alpha init"
                )
        except FileNotFoundError:
            return ValidationResult(
                "Claude-Flow", "fail",
                "npx not found - Node.js required",
                "Install Node.js from https://nodejs.org"
            )
        except subprocess.TimeoutExpired:
            return ValidationResult(
                "Claude-Flow", "warn",
                "Claude-Flow check timed out",
                "May still work, but check manually"
            )

    def validate_auto_claude(self) -> ValidationResult:
        """Check Auto-Claude workspace setup."""
        auto_claude_path = Path(__file__).parent / "auto-claude"

        if not auto_claude_path.exists():
            return ValidationResult(
                "Auto-Claude", "warn",
                "Auto-Claude directory not found in unleash",
                "Expected at: unleash/auto-claude/"
            )

        claude_md = auto_claude_path / "CLAUDE.md"
        if not claude_md.exists():
            return ValidationResult(
                "Auto-Claude", "fail",
                "CLAUDE.md not found in Auto-Claude",
                "Auto-Claude needs CLAUDE.md for configuration"
            )

        return ValidationResult(
            "Auto-Claude", "pass",
            "Auto-Claude workspace found",
            f"Location: {auto_claude_path}"
        )

    def validate_hooks_config(self) -> ValidationResult:
        """Check Claude Code hooks configuration."""
        hooks_path = Path.home() / ".claude" / "hooks.json"

        if not hooks_path.exists():
            # Check alternative location
            hooks_path = Path(__file__).parent / "hooks" / "config.json"

        if hooks_path.exists():
            try:
                with open(hooks_path) as f:
                    hooks = json.load(f)
                hook_count = len(hooks.get("hooks", []))
                return ValidationResult(
                    "Claude Code Hooks", "pass",
                    f"{hook_count} hooks configured",
                    f"Config: {hooks_path}"
                )
            except json.JSONDecodeError:
                return ValidationResult(
                    "Claude Code Hooks", "fail",
                    "Hooks config is invalid JSON",
                    f"Check: {hooks_path}"
                )
        else:
            return ValidationResult(
                "Claude Code Hooks", "warn",
                "No hooks configuration found",
                "Hooks are optional but recommended for safety"
            )

    # =====================
    # LAYER 2: MEMORY
    # =====================

    async def validate_letta(self) -> ValidationResult:
        """Check Letta server connectivity."""
        import aiohttp

        letta_url = os.getenv("LETTA_URL", "http://localhost:8283")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{letta_url}/v1/health", timeout=5) as resp:
                    if resp.status == 200:
                        return ValidationResult(
                            "Letta Server", "pass",
                            "Letta server is running",
                            f"URL: {letta_url}"
                        )
                    else:
                        return ValidationResult(
                            "Letta Server", "fail",
                            f"Letta returned status {resp.status}",
                            "Check server logs"
                        )
        except aiohttp.ClientError:
            return ValidationResult(
                "Letta Server", "warn",
                "Letta server not reachable",
                "Start with: docker run -p 8283:8283 letta/letta"
            )
        except ImportError:
            return ValidationResult(
                "Letta Server", "warn",
                "aiohttp not installed - skipping HTTP check",
                "Install with: pip install aiohttp"
            )

    async def validate_graphiti(self) -> ValidationResult:
        """Check Graphiti/Neo4j connectivity."""
        import aiohttp

        neo4j_url = os.getenv("NEO4J_URL", "http://localhost:7474")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(neo4j_url, timeout=5) as resp:
                    if resp.status == 200:
                        return ValidationResult(
                            "Graphiti/Neo4j", "pass",
                            "Neo4j is running",
                            f"URL: {neo4j_url}"
                        )
                    else:
                        return ValidationResult(
                            "Graphiti/Neo4j", "fail",
                            f"Neo4j returned status {resp.status}",
                            "Check server configuration"
                        )
        except aiohttp.ClientError:
            return ValidationResult(
                "Graphiti/Neo4j", "warn",
                "Neo4j not reachable",
                "Start with: docker run -p 7474:7474 -p 7687:7687 neo4j:5"
            )
        except ImportError:
            return ValidationResult(
                "Graphiti/Neo4j", "warn",
                "aiohttp not installed - skipping HTTP check",
                "Install with: pip install aiohttp"
            )

    async def validate_qdrant(self) -> ValidationResult:
        """Check Qdrant vector store connectivity."""
        import aiohttp

        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{qdrant_url}/health", timeout=5) as resp:
                    if resp.status == 200:
                        return ValidationResult(
                            "Qdrant Vector Store", "pass",
                            "Qdrant is running",
                            f"URL: {qdrant_url}"
                        )
                    else:
                        return ValidationResult(
                            "Qdrant Vector Store", "fail",
                            f"Qdrant returned status {resp.status}",
                            "Check server configuration"
                        )
        except aiohttp.ClientError:
            return ValidationResult(
                "Qdrant Vector Store", "warn",
                "Qdrant not reachable",
                "Start with: docker run -p 6333:6333 qdrant/qdrant"
            )
        except ImportError:
            return ValidationResult(
                "Qdrant Vector Store", "warn",
                "aiohttp not installed - skipping HTTP check",
                "Install with: pip install aiohttp"
            )

    # =====================
    # LAYER 3: MCP
    # =====================

    def validate_mcp_config(self) -> ValidationResult:
        """Check MCP configuration file."""
        mcp_paths = [
            Path(self.config_path),
            Path.home() / ".claude" / "mcp.json",
            Path(__file__).parent / "mcp_config.json",
        ]

        for mcp_path in mcp_paths:
            if mcp_path.exists():
                try:
                    with open(mcp_path) as f:
                        config = json.load(f)
                    servers = config.get("mcpServers", {})
                    return ValidationResult(
                        "MCP Configuration", "pass",
                        f"{len(servers)} MCP servers configured",
                        f"Config: {mcp_path}"
                    )
                except json.JSONDecodeError:
                    return ValidationResult(
                        "MCP Configuration", "fail",
                        "MCP config is invalid JSON",
                        f"Check: {mcp_path}"
                    )

        return ValidationResult(
            "MCP Configuration", "warn",
            "No MCP configuration found",
            "Create mcp_config.json or ~/.claude/mcp.json"
        )

    def validate_mcp_servers(self) -> List[ValidationResult]:
        """Check individual MCP server availability."""
        results = []

        # Check for key MCP servers
        required_servers = {
            "memory": "@modelcontextprotocol/server-memory",
            "filesystem": "@modelcontextprotocol/server-filesystem",
            "git": "@modelcontextprotocol/server-git",
        }

        optional_servers = {
            "exa": "exa-mcp-server",
            "context7": "@upstash/context7-mcp",
            "graphiti": "@getzep/graphiti-mcp",
        }

        for name, package in required_servers.items():
            try:
                result = subprocess.run(
                    ["npm", "list", "-g", package],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    results.append(ValidationResult(
                        f"MCP:{name}", "pass",
                        f"{package} available"
                    ))
                else:
                    results.append(ValidationResult(
                        f"MCP:{name}", "warn",
                        f"{package} not globally installed",
                        f"Will install on demand with npx"
                    ))
            except:
                results.append(ValidationResult(
                    f"MCP:{name}", "warn",
                    "Could not check package",
                    "npm may not be in PATH"
                ))

        return results

    # =====================
    # LAYER 4: QA PIPELINE
    # =====================

    def validate_qa_tools(self) -> List[ValidationResult]:
        """Check QA pipeline tools."""
        results = []

        tools = {
            "pytest": "pytest --version",
            "pyright": "pyright --version",
            "ruff": "ruff --version",
            "semgrep": "semgrep --version",
            "bandit": "bandit --version",
        }

        for name, cmd in tools.items():
            try:
                result = subprocess.run(
                    cmd.split(), capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    version = result.stdout.strip().split('\n')[0]
                    results.append(ValidationResult(
                        f"QA:{name}", "pass",
                        f"Installed: {version}"
                    ))
                else:
                    results.append(ValidationResult(
                        f"QA:{name}", "warn",
                        f"{name} not found",
                        f"Install with: pip install {name}"
                    ))
            except FileNotFoundError:
                results.append(ValidationResult(
                    f"QA:{name}", "warn",
                    f"{name} not in PATH",
                    f"Install with: pip install {name}"
                ))
            except subprocess.TimeoutExpired:
                results.append(ValidationResult(
                    f"QA:{name}", "warn",
                    f"{name} check timed out"
                ))

        return results

    # =====================
    # ENVIRONMENT CHECKS
    # =====================

    def validate_environment(self) -> List[ValidationResult]:
        """Check required environment variables."""
        results = []

        env_vars = {
            "ANTHROPIC_API_KEY": ("required", "Claude API access"),
            "LETTA_API_KEY": ("optional", "Letta cloud access"),
            "EXA_API_KEY": ("optional", "Exa search access"),
            "GITHUB_TOKEN": ("optional", "GitHub MCP server"),
        }

        for var, (importance, description) in env_vars.items():
            value = os.getenv(var)
            if value:
                masked = value[:8] + "..." if len(value) > 8 else "***"
                results.append(ValidationResult(
                    f"ENV:{var}", "pass",
                    f"Set ({masked})",
                    description
                ))
            elif importance == "required":
                results.append(ValidationResult(
                    f"ENV:{var}", "fail",
                    "Not set (required)",
                    description
                ))
            else:
                results.append(ValidationResult(
                    f"ENV:{var}", "warn",
                    "Not set (optional)",
                    description
                ))

        return results

    # =====================
    # MAIN VALIDATION
    # =====================

    async def run_all_validations(self):
        """Run all validation checks."""
        print(f"\n{BLUE}═══════════════════════════════════════════════════════════════{RESET}")
        print(f"{BLUE}       ULTIMATE AUTONOMOUS ARCHITECTURE v3 - VALIDATION{RESET}")
        print(f"{BLUE}═══════════════════════════════════════════════════════════════{RESET}\n")

        # Layer 1: Orchestration
        print(f"{YELLOW}▸ LAYER 1: ORCHESTRATION{RESET}")
        self.log(self.validate_claude_flow())
        self.log(self.validate_auto_claude())
        self.log(self.validate_hooks_config())
        print()

        # Layer 2: Memory
        print(f"{YELLOW}▸ LAYER 2: MEMORY & KNOWLEDGE{RESET}")
        self.log(await self.validate_letta())
        self.log(await self.validate_graphiti())
        self.log(await self.validate_qdrant())
        print()

        # Layer 3: MCP
        print(f"{YELLOW}▸ LAYER 3: MCP TOOL ACCESS{RESET}")
        self.log(self.validate_mcp_config())
        for result in self.validate_mcp_servers():
            self.log(result)
        print()

        # Layer 4: QA Pipeline
        print(f"{YELLOW}▸ LAYER 4: QA PIPELINE{RESET}")
        for result in self.validate_qa_tools():
            self.log(result)
        print()

        # Environment
        print(f"{YELLOW}▸ ENVIRONMENT{RESET}")
        for result in self.validate_environment():
            self.log(result)
        print()

        # Summary
        self._print_summary()

    def _print_summary(self):
        """Print validation summary."""
        passed = sum(1 for r in self.results if r.status == "pass")
        failed = sum(1 for r in self.results if r.status == "fail")
        warned = sum(1 for r in self.results if r.status == "warn")
        total = len(self.results)

        print(f"{BLUE}═══════════════════════════════════════════════════════════════{RESET}")
        print(f"  {GREEN}PASSED: {passed}{RESET}  |  {RED}FAILED: {failed}{RESET}  |  {YELLOW}WARNINGS: {warned}{RESET}  |  TOTAL: {total}")
        print(f"{BLUE}═══════════════════════════════════════════════════════════════{RESET}")

        if failed == 0:
            print(f"\n  {GREEN}✓ All critical checks passed!{RESET}")
            if warned > 0:
                print(f"  {YELLOW}⚠ Review warnings for optimal configuration{RESET}")
        else:
            print(f"\n  {RED}✗ {failed} critical issue(s) need attention{RESET}")

        print()


async def main():
    validator = IntegrationValidator()
    await validator.run_all_validations()


if __name__ == "__main__":
    asyncio.run(main())
