#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.26.0",
#     "structlog>=24.1.0",
#     "pydantic>=2.0.0",
# ]
# [tool.uv]
# exclude-newer = "2026-02-01"
# ///
"""
Auto-Validation Workflow - Ultimate Autonomous Platform

Comprehensive validation pipeline for Ralph Loop iterations:
1. Ecosystem health check
2. Hook syntax validation
3. MCP server verification
4. Test suite execution
5. Report generation

Usage:
    uv run auto_validate.py              # Full validation
    uv run auto_validate.py --quick      # Quick validation (skip tests)
    uv run auto_validate.py --json       # JSON output for automation
    uv run auto_validate.py --fix        # Attempt auto-fixes

Platform: Windows 11 + Python 3.11+
Architecture: V10 Optimized (Verified, Minimal, Seamless)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
V10_DIR = SCRIPT_DIR.parent
UNLEASH_DIR = V10_DIR.parent
HOOKS_DIR = V10_DIR / "hooks"
TESTS_DIR = V10_DIR / "tests"

# Windows compatibility
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class ValidationStatus(str, Enum):
    """Status of a validation step."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class ValidationResult:
    """Result of a single validation step."""
    name: str
    status: ValidationStatus
    message: str
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report."""
    timestamp: str
    total_duration_ms: float
    overall_status: ValidationStatus
    results: List[ValidationResult] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


# =============================================================================
# Validation Steps
# =============================================================================

async def validate_hooks() -> ValidationResult:
    """Validate all hook scripts have correct Python syntax."""
    start = time.perf_counter()

    hook_files = list(HOOKS_DIR.glob("*.py"))
    errors = []

    for hook_file in hook_files:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(hook_file)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                errors.append(f"{hook_file.name}: {result.stderr[:100]}")
        except Exception as e:
            errors.append(f"{hook_file.name}: {e}")

    duration = (time.perf_counter() - start) * 1000

    if errors:
        return ValidationResult(
            name="Hook Syntax",
            status=ValidationStatus.FAIL,
            message=f"{len(errors)} hooks have syntax errors",
            duration_ms=duration,
            details={"errors": errors, "total_hooks": len(hook_files)},
        )

    return ValidationResult(
        name="Hook Syntax",
        status=ValidationStatus.PASS,
        message=f"{len(hook_files)} hooks validated",
        duration_ms=duration,
        details={"hooks": [f.name for f in hook_files]},
    )


async def validate_mcp_packages() -> ValidationResult:
    """Validate core MCP packages are available."""
    start = time.perf_counter()

    core_packages = [
        "@modelcontextprotocol/server-filesystem",
        "@modelcontextprotocol/server-memory",
        "@modelcontextprotocol/server-sequential-thinking",
    ]

    verified = []
    missing = []

    for package in core_packages:
        try:
            result = subprocess.run(
                ["npm", "view", package, "version"],
                capture_output=True,
                text=True,
                timeout=30,
                shell=True,
            )
            if result.returncode == 0:
                verified.append(f"{package}@{result.stdout.strip()}")
            else:
                missing.append(package)
        except Exception:
            missing.append(package)

    duration = (time.perf_counter() - start) * 1000

    if missing:
        return ValidationResult(
            name="MCP Packages",
            status=ValidationStatus.FAIL,
            message=f"{len(missing)} packages not found",
            duration_ms=duration,
            details={"verified": verified, "missing": missing},
        )

    return ValidationResult(
        name="MCP Packages",
        status=ValidationStatus.PASS,
        message=f"{len(verified)} packages verified",
        duration_ms=duration,
        details={"packages": verified},
    )


async def validate_infrastructure() -> ValidationResult:
    """Validate infrastructure components."""
    start = time.perf_counter()
    checks = {}

    # Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks["python"] = py_version

    # uv available
    try:
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            shell=True,
        )
        checks["uv"] = result.stdout.strip().split()[0] if result.returncode == 0 else None
    except Exception:
        checks["uv"] = None

    # Node.js available
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            shell=True,
        )
        checks["node"] = result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        checks["node"] = None

    # Kill switch check
    kill_switch = Path.home() / ".claude" / "KILL_SWITCH"
    checks["kill_switch_active"] = kill_switch.exists()

    duration = (time.perf_counter() - start) * 1000

    # Determine overall status
    if checks["kill_switch_active"]:
        status = ValidationStatus.FAIL
        message = "KILL SWITCH ACTIVE - Operations blocked!"
    elif not checks["uv"]:
        status = ValidationStatus.WARN
        message = "uv not available (optional)"
    else:
        status = ValidationStatus.PASS
        message = f"Python {py_version}, uv, node ready"

    return ValidationResult(
        name="Infrastructure",
        status=status,
        message=message,
        duration_ms=duration,
        details=checks,
    )


async def validate_letta() -> ValidationResult:
    """Validate Letta server connectivity."""
    start = time.perf_counter()
    letta_url = os.environ.get("LETTA_URL", "http://localhost:8500")

    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Try multiple health endpoints (letta-sovereign uses /health)
            for endpoint in ["/health", "/v1/health"]:
                try:
                    response = await client.get(f"{letta_url}{endpoint}")
                    duration = (time.perf_counter() - start) * 1000

                    if response.status_code == 200:
                        return ValidationResult(
                            name="Letta Server",
                            status=ValidationStatus.PASS,
                            message=f"Running at {letta_url}",
                            duration_ms=duration,
                        )
                except Exception:
                    pass
    except Exception:
        pass

    duration = (time.perf_counter() - start) * 1000
    return ValidationResult(
        name="Letta Server",
        status=ValidationStatus.WARN,
        message="Not running (optional for local dev)",
        duration_ms=duration,
        details={"url": letta_url, "note": "Start with: docker run -d -p 8500:8283 --name letta-sovereign letta/letta:latest"},
    )


async def run_tests(quick: bool = False) -> ValidationResult:
    """Run the test suite."""
    if quick:
        return ValidationResult(
            name="Test Suite",
            status=ValidationStatus.SKIP,
            message="Skipped in quick mode",
        )

    start = time.perf_counter()
    test_files = list(TESTS_DIR.glob("test_*.py"))

    if not test_files:
        return ValidationResult(
            name="Test Suite",
            status=ValidationStatus.WARN,
            message="No test files found",
            details={"test_dir": str(TESTS_DIR)},
        )

    try:
        # Run pytest with minimal output
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(TESTS_DIR), "-v", "--tb=short", "-q"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(V10_DIR),
        )
        duration = (time.perf_counter() - start) * 1000

        # Parse output for pass/fail counts
        output = result.stdout + result.stderr

        if result.returncode == 0:
            return ValidationResult(
                name="Test Suite",
                status=ValidationStatus.PASS,
                message=f"{len(test_files)} test files executed",
                duration_ms=duration,
                details={"output": output[-500:] if len(output) > 500 else output},
            )
        else:
            return ValidationResult(
                name="Test Suite",
                status=ValidationStatus.FAIL,
                message="Tests failed",
                duration_ms=duration,
                details={"output": output[-1000:] if len(output) > 1000 else output},
            )

    except subprocess.TimeoutExpired:
        duration = (time.perf_counter() - start) * 1000
        return ValidationResult(
            name="Test Suite",
            status=ValidationStatus.FAIL,
            message="Test execution timed out (120s)",
            duration_ms=duration,
        )
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        return ValidationResult(
            name="Test Suite",
            status=ValidationStatus.FAIL,
            message=f"Test execution failed: {e}",
            duration_ms=duration,
        )


async def validate_ecosystem_orchestrator() -> ValidationResult:
    """Run the ecosystem orchestrator for comprehensive check."""
    start = time.perf_counter()
    orchestrator_path = SCRIPT_DIR / "ecosystem_orchestrator.py"

    if not orchestrator_path.exists():
        return ValidationResult(
            name="Ecosystem Health",
            status=ValidationStatus.SKIP,
            message="Ecosystem orchestrator not found",
        )

    try:
        result = subprocess.run(
            [sys.executable, str(orchestrator_path), "--quick", "--json"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(SCRIPT_DIR),
        )
        duration = (time.perf_counter() - start) * 1000

        if result.returncode <= 1:  # 0=healthy, 1=degraded
            try:
                report = json.loads(result.stdout)
                summary = report.get("summary", {})
                overall = report.get("overall_status", "unknown")

                status = ValidationStatus.PASS if overall == "healthy" else ValidationStatus.WARN
                return ValidationResult(
                    name="Ecosystem Health",
                    status=status,
                    message=f"OK={summary.get('healthy', 0)}, WARN={summary.get('degraded', 0)}, ERR={summary.get('unavailable', 0)}",
                    duration_ms=duration,
                    details={"overall_status": overall, "summary": summary},
                )
            except json.JSONDecodeError:
                pass

        return ValidationResult(
            name="Ecosystem Health",
            status=ValidationStatus.WARN,
            message="Orchestrator returned non-JSON output",
            duration_ms=duration,
            details={"output": result.stdout[-500:]},
        )

    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        return ValidationResult(
            name="Ecosystem Health",
            status=ValidationStatus.FAIL,
            message=f"Orchestrator failed: {e}",
            duration_ms=duration,
        )


# =============================================================================
# Main Workflow
# =============================================================================

class AutoValidator:
    """Auto-validation workflow runner."""

    def __init__(self, quick: bool = False, verbose: bool = False):
        self.quick = quick
        self.verbose = verbose
        self.start_time = time.perf_counter()

    async def run(self) -> ValidationReport:
        """Run all validation steps."""
        results: List[ValidationResult] = []

        # Run validations
        print("=" * 60)
        print("AUTO-VALIDATION WORKFLOW - V10 Ultimate Platform")
        print("=" * 60)
        print(f"Mode: {'Quick' if self.quick else 'Full'}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)

        # Step 1: Hooks
        print("[1/6] Validating hooks...", end=" ", flush=True)
        result = await validate_hooks()
        results.append(result)
        print(f"[{result.status.value.upper()}] {result.message}")

        # Step 2: MCP Packages
        print("[2/6] Verifying MCP packages...", end=" ", flush=True)
        result = await validate_mcp_packages()
        results.append(result)
        print(f"[{result.status.value.upper()}] {result.message}")

        # Step 3: Infrastructure
        print("[3/6] Checking infrastructure...", end=" ", flush=True)
        result = await validate_infrastructure()
        results.append(result)
        print(f"[{result.status.value.upper()}] {result.message}")

        # Step 4: Letta
        print("[4/6] Testing Letta connectivity...", end=" ", flush=True)
        result = await validate_letta()
        results.append(result)
        print(f"[{result.status.value.upper()}] {result.message}")

        # Step 5: Ecosystem Orchestrator
        print("[5/6] Running ecosystem health check...", end=" ", flush=True)
        result = await validate_ecosystem_orchestrator()
        results.append(result)
        print(f"[{result.status.value.upper()}] {result.message}")

        # Step 6: Tests
        print("[6/6] Running test suite...", end=" ", flush=True)
        result = await run_tests(quick=self.quick)
        results.append(result)
        print(f"[{result.status.value.upper()}] {result.message}")

        # Calculate summary
        total_duration = (time.perf_counter() - self.start_time) * 1000
        summary = {status.value: 0 for status in ValidationStatus}
        for r in results:
            summary[r.status.value] += 1

        # Determine overall status
        if summary["fail"] > 0:
            overall = ValidationStatus.FAIL
        elif summary["warn"] > 1:
            overall = ValidationStatus.WARN
        else:
            overall = ValidationStatus.PASS

        # Generate recommendations
        recommendations = []
        for r in results:
            if r.status == ValidationStatus.FAIL:
                if "Letta" in r.name:
                    recommendations.append("Start Letta: docker run -d -p 8500:8283 --name letta-sovereign letta/letta:latest")
                elif "Hook" in r.name:
                    recommendations.append("Fix hook syntax errors in v10/hooks/")
                elif "MCP" in r.name:
                    recommendations.append("Install missing MCP packages with npm")

        # Print summary
        print("-" * 60)
        print(f"RESULT: [{overall.value.upper()}]")
        print(f"Summary: PASS={summary['pass']}, WARN={summary['warn']}, FAIL={summary['fail']}, SKIP={summary['skip']}")
        print(f"Duration: {total_duration:.0f}ms")

        if recommendations:
            print("\nRecommendations:")
            for rec in recommendations:
                print(f"  - {rec}")

        print("=" * 60)

        return ValidationReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_duration_ms=total_duration,
            overall_status=overall,
            results=results,
            summary=summary,
            recommendations=recommendations,
        )


# =============================================================================
# CLI Entry Point
# =============================================================================

def report_to_dict(report: ValidationReport) -> dict:
    """Convert report to JSON-serializable dict."""
    return {
        "timestamp": report.timestamp,
        "total_duration_ms": report.total_duration_ms,
        "overall_status": report.overall_status.value,
        "results": [
            {
                "name": r.name,
                "status": r.status.value,
                "message": r.message,
                "duration_ms": r.duration_ms,
                "details": r.details,
            }
            for r in report.results
        ],
        "summary": report.summary,
        "recommendations": report.recommendations,
    }


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Auto-Validation Workflow for Ultimate Autonomous Platform",
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick validation (skip test suite)",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    validator = AutoValidator(quick=args.quick, verbose=args.verbose)
    report = await validator.run()

    if args.json:
        print(json.dumps(report_to_dict(report), indent=2))

    # Exit code based on status
    if report.overall_status == ValidationStatus.FAIL:
        sys.exit(2)
    elif report.overall_status == ValidationStatus.WARN:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
