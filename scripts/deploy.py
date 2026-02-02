#!/usr/bin/env python3
"""
Production Deployment Script
Phase 15: V35 Production Deployment

Orchestrates the deployment validation and execution steps.
"""

import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class StepResult:
    """Result of a deployment step."""

    name: str
    passed: bool
    duration: float
    output: str = ""
    error: str = ""


def run_command(cmd: str, timeout: int = 300) -> tuple[bool, str, str]:
    """Run a shell command and return (success, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=timeout,
            env={**dict(__import__("os").environ), "PYTHONIOENCODING": "utf-8"},
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def step_validate_sdks() -> StepResult:
    """Validate V35 SDK installation."""
    start = time.perf_counter()
    success, stdout, stderr = run_command("python scripts/validate_v35_final.py")
    duration = time.perf_counter() - start

    # Check for 36/36 in output
    passed = success and "36/36" in stdout

    return StepResult(
        name="SDK Validation",
        passed=passed,
        duration=duration,
        output=stdout[-500:] if stdout else "",
        error=stderr[-200:] if stderr and not passed else "",
    )


def step_security_audit() -> StepResult:
    """Run security audit."""
    start = time.perf_counter()
    success, stdout, stderr = run_command("python scripts/security_audit.py")
    duration = time.perf_counter() - start

    return StepResult(
        name="Security Audit",
        passed=success,
        duration=duration,
        output=stdout[-500:] if stdout else "",
        error=stderr[-200:] if stderr and not success else "",
    )


def step_cli_tests() -> StepResult:
    """Run CLI command tests."""
    start = time.perf_counter()
    success, stdout, stderr = run_command("python tests/test_cli_commands.py", timeout=600)
    duration = time.perf_counter() - start

    # Check for 30/30 in output
    passed = success and "30/30" in stdout

    return StepResult(
        name="CLI Tests",
        passed=passed,
        duration=duration,
        output=stdout[-500:] if stdout else "",
        error=stderr[-200:] if stderr and not passed else "",
    )


def step_e2e_tests() -> StepResult:
    """Run E2E integration tests."""
    start = time.perf_counter()
    success, stdout, stderr = run_command("python tests/test_e2e_integration.py", timeout=600)
    duration = time.perf_counter() - start

    return StepResult(
        name="E2E Integration Tests",
        passed=success,
        duration=duration,
        output=stdout[-500:] if stdout else "",
        error=stderr[-200:] if stderr and not success else "",
    )


def step_health_check() -> StepResult:
    """Run health check."""
    start = time.perf_counter()
    success, stdout, stderr = run_command("python core/health.py")
    duration = time.perf_counter() - start

    # Degraded is acceptable
    passed = success or '"status": "degraded"' in stdout

    return StepResult(
        name="Health Check",
        passed=passed,
        duration=duration,
        output=stdout[-500:] if stdout else "",
        error=stderr[-200:] if stderr and not passed else "",
    )


def step_config_validation() -> StepResult:
    """Validate configuration files exist."""
    start = time.perf_counter()

    required_files = [
        "config/production.yaml",
        "config/env.template",
        "Dockerfile",
        "docker-compose.yaml",
    ]

    missing = []
    for f in required_files:
        if not (PROJECT_ROOT / f).exists():
            missing.append(f)

    duration = time.perf_counter() - start
    passed = len(missing) == 0

    return StepResult(
        name="Configuration Validation",
        passed=passed,
        duration=duration,
        output=f"Checked {len(required_files)} files" if passed else "",
        error=f"Missing: {', '.join(missing)}" if missing else "",
    )


def deploy(dry_run: bool = True, verbose: bool = False):
    """Execute deployment validation steps."""
    print("=" * 60)
    print("UNLEASH V35 PRODUCTION DEPLOYMENT")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Mode: {'DRY RUN' if dry_run else 'PRODUCTION'}")
    print("=" * 60)

    steps: list[tuple[str, Callable[[], StepResult]]] = [
        ("SDK Validation", step_validate_sdks),
        ("Security Audit", step_security_audit),
        ("CLI Tests", step_cli_tests),
        ("E2E Tests", step_e2e_tests),
        ("Health Check", step_health_check),
        ("Config Validation", step_config_validation),
    ]

    results: list[StepResult] = []

    for i, (name, step_fn) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {name}...")

        result = step_fn()
        results.append(result)

        status = "[PASS]" if result.passed else "[FAIL]"
        print(f"  {status} {result.name} ({result.duration:.1f}s)")

        if verbose and result.output:
            for line in result.output.strip().split("\n")[-5:]:
                print(f"    {line}")

        if not result.passed and result.error:
            print(f"    Error: {result.error[:100]}")

    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    total_time = sum(r.duration for r in results)

    print("\n" + "=" * 60)
    print("DEPLOYMENT SUMMARY")
    print("=" * 60)
    print(f"\nResults: {passed}/{total} checks passed")
    print(f"Total time: {total_time:.1f}s")

    print("\nDetails:")
    for r in results:
        status = "[PASS]" if r.passed else "[FAIL]"
        print(f"  {status} {r.name}")

    if passed == total:
        print("\n" + "=" * 60)
        print("[PASS] ALL CHECKS PASSED - PRODUCTION READY!")
        print("=" * 60)

        if not dry_run:
            print("\nNext steps:")
            print("  1. docker-compose build")
            print("  2. docker-compose up -d")
            print("  3. Monitor logs: docker-compose logs -f")
        return 0
    else:
        print("\n" + "=" * 60)
        print("[WARN] SOME CHECKS FAILED - REVIEW BEFORE DEPLOYMENT")
        print("=" * 60)

        failed = [r for r in results if not r.passed]
        print(f"\nFailed checks ({len(failed)}):")
        for r in failed:
            print(f"  - {r.name}")
            if r.error:
                print(f"    Error: {r.error[:200]}")

        return 1


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Unleash V35 Production Deployment")
    parser.add_argument("--execute", action="store_true", help="Execute deployment (not dry run)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed output")
    args = parser.parse_args()

    return deploy(dry_run=not args.execute, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
