#!/usr/bin/env python3
"""
Final Production Validation Script
Phase 15: V35 Production Deployment

Comprehensive validation before production deployment.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def run_check(name: str, cmd: list[str], timeout: int = 300) -> tuple[bool, str]:
    """Run a validation check."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=timeout,
            env={**dict(__import__("os").environ), "PYTHONIOENCODING": "utf-8"},
        )
        return result.returncode == 0, result.stdout
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def main():
    """Run final production validation."""
    print("=" * 60)
    print("UNLEASH V35 FINAL PRODUCTION VALIDATION")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)

    results = {}
    start_time = time.perf_counter()

    # 1. SDK Validation
    print("\n[1/6] SDK Validation...")
    passed, output = run_check(
        "SDK Validation",
        [sys.executable, "scripts/validate_v35_final.py"],
    )
    if passed and "36/36" in output:
        results["sdk_validation"] = "[PASS] 36/36 SDKs"
        print("  [PASS] 36/36 SDKs verified")
    else:
        results["sdk_validation"] = "[FAIL] SDK validation failed"
        print("  [FAIL] SDK validation failed")

    # 2. CLI Tests
    print("\n[2/6] CLI Tests...")
    passed, output = run_check(
        "CLI Tests",
        [sys.executable, "tests/test_cli_commands.py"],
        timeout=600,
    )
    if passed and "30/30" in output:
        results["cli_tests"] = "[PASS] 30/30 tests"
        print("  [PASS] 30/30 CLI tests passed")
    else:
        results["cli_tests"] = "[WARN] CLI tests incomplete"
        print("  [WARN] CLI tests incomplete")

    # 3. E2E Tests
    print("\n[3/6] E2E Integration Tests...")
    passed, output = run_check(
        "E2E Tests",
        [sys.executable, "tests/test_e2e_integration.py"],
        timeout=600,
    )
    if passed:
        results["e2e_tests"] = "[PASS] All passed"
        print("  [PASS] E2E tests passed")
    else:
        results["e2e_tests"] = "[WARN] E2E tests incomplete"
        print("  [WARN] E2E tests incomplete")

    # 4. Security Audit
    print("\n[4/6] Security Audit...")
    passed, output = run_check(
        "Security Audit",
        [sys.executable, "scripts/security_audit.py"],
    )
    if passed:
        results["security_audit"] = "[PASS] No issues"
        print("  [PASS] Security audit clean")
    else:
        results["security_audit"] = "[WARN] Review needed"
        print("  [WARN] Security audit needs review")

    # 5. Health Check
    print("\n[5/6] Health Check...")
    passed, output = run_check(
        "Health Check",
        [sys.executable, "core/health.py"],
    )
    try:
        health_data = json.loads(output)
        status = health_data.get("status", "unknown")
        if status in ("healthy", "degraded"):
            results["health_check"] = f"[PASS] {status.title()}"
            print(f"  [PASS] Health: {status}")
        else:
            results["health_check"] = f"[WARN] {status}"
            print(f"  [WARN] Health: {status}")
    except Exception:
        if passed:
            results["health_check"] = "[PASS] Operational"
            print("  [PASS] Health check operational")
        else:
            results["health_check"] = "[WARN] Health check failed"
            print("  [WARN] Health check failed")

    # 6. Configuration Check
    print("\n[6/6] Configuration Check...")
    config_files = [
        "config/production.yaml",
        "config/env.template",
        "Dockerfile",
        "docker-compose.yaml",
    ]
    missing = [f for f in config_files if not (PROJECT_ROOT / f).exists()]
    if not missing:
        results["config_check"] = "[PASS] All configs present"
        print("  [PASS] All configuration files present")
    else:
        results["config_check"] = f"[WARN] Missing: {', '.join(missing)}"
        print(f"  [WARN] Missing: {', '.join(missing)}")

    # Summary
    total_time = time.perf_counter() - start_time
    all_passed = all("[PASS]" in v for v in results.values())

    print("\n" + "=" * 60)
    print("FINAL VALIDATION RESULTS")
    print("=" * 60)

    for check, status in results.items():
        print(f"  {check}: {status}")

    print(f"\nTotal time: {total_time:.1f}s")

    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] PRODUCTION READY!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Copy config/env.template to .env and fill in values")
        print("  2. docker-compose build")
        print("  3. docker-compose up -d")
        print("  4. docker-compose logs -f unleash")
    else:
        print("[REVIEW] ADDRESS WARNINGS BEFORE DEPLOYMENT")
        print("=" * 60)
        print("\nReview the warnings above and address any issues.")

    # Save results to JSON
    results_file = PROJECT_ROOT / "validation_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "version": "35.0.0",
                "results": results,
                "all_passed": all_passed,
                "duration_seconds": round(total_time, 1),
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {results_file}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
