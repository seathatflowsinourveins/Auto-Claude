#!/usr/bin/env python3
"""
Health Check Endpoint for Production Monitoring
Phase 15: V35 Production Deployment

Provides health status for all critical layers of the Unleash platform.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

# Ensure project root is in path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

__version__ = "35.0.0"


@dataclass
class CheckResult:
    """Result of a single health check."""

    status: str  # healthy, degraded, unhealthy
    provider: str
    latency_ms: float = 0.0
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """Overall health status of the platform."""

    status: str  # healthy, degraded, unhealthy
    version: str
    timestamp: str
    checks: dict[str, dict[str, Any]]
    summary: dict[str, int] = field(default_factory=dict)


async def check_protocol() -> CheckResult:
    """Check L0 Protocol availability."""
    import time

    start = time.perf_counter()
    try:
        from anthropic import Anthropic

        client = Anthropic()
        # Light check - verify client initialization
        latency = (time.perf_counter() - start) * 1000
        return CheckResult(
            status="healthy",
            provider="anthropic",
            latency_ms=round(latency, 2),
            details={"client": "initialized"},
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return CheckResult(
            status="unhealthy",
            provider="anthropic",
            latency_ms=round(latency, 2),
            error=str(e),
        )


async def check_memory() -> CheckResult:
    """Check L2 Memory availability."""
    import time

    start = time.perf_counter()
    try:
        from mem0 import Memory

        Memory()
        latency = (time.perf_counter() - start) * 1000
        return CheckResult(
            status="healthy",
            provider="mem0",
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return CheckResult(
            status="degraded",
            provider="mem0",
            latency_ms=round(latency, 2),
            error=str(e),
        )


async def check_structured() -> CheckResult:
    """Check L3 Structured output availability."""
    import time

    start = time.perf_counter()
    try:
        import instructor  # noqa: F401
        from pydantic import BaseModel

        class TestModel(BaseModel):
            value: str

        TestModel(value="test")
        latency = (time.perf_counter() - start) * 1000
        return CheckResult(
            status="healthy",
            provider="instructor+pydantic",
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return CheckResult(
            status="degraded",
            provider="instructor+pydantic",
            latency_ms=round(latency, 2),
            error=str(e),
        )


async def check_safety() -> CheckResult:
    """Check L6 Safety availability."""
    import time

    start = time.perf_counter()
    try:
        from core.safety.scanner_compat import InputScanner

        scanner = InputScanner()
        result = scanner.scan("health check test")
        latency = (time.perf_counter() - start) * 1000
        return CheckResult(
            status="healthy",
            provider="scanner_compat",
            latency_ms=round(latency, 2),
            details={"is_safe": result.is_safe},
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return CheckResult(
            status="degraded",
            provider="scanner_compat",
            latency_ms=round(latency, 2),
            error=str(e),
        )


async def check_knowledge() -> CheckResult:
    """Check L8 Knowledge availability."""
    import time

    start = time.perf_counter()
    try:
        from llama_index.core import VectorStoreIndex  # noqa: F401

        latency = (time.perf_counter() - start) * 1000
        return CheckResult(
            status="healthy",
            provider="llama_index",
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return CheckResult(
            status="degraded",
            provider="llama_index",
            latency_ms=round(latency, 2),
            error=str(e),
        )


async def check_observability() -> CheckResult:
    """Check L5 Observability availability."""
    import time

    start = time.perf_counter()
    try:
        from core.observability.langfuse_compat import LangfuseCompat  # noqa: F401

        latency = (time.perf_counter() - start) * 1000
        return CheckResult(
            status="healthy",
            provider="langfuse_compat",
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return CheckResult(
            status="degraded",
            provider="langfuse_compat",
            latency_ms=round(latency, 2),
            error=str(e),
        )


async def get_health_status() -> HealthStatus:
    """Get overall health status of the platform.

    Runs all layer checks concurrently and aggregates results.

    Returns:
        HealthStatus with overall status, per-check details, and summary counts
    """
    # Run all checks concurrently
    results: list[CheckResult | BaseException] = await asyncio.gather(
        check_protocol(),
        check_memory(),
        check_structured(),
        check_safety(),
        check_knowledge(),
        check_observability(),
        return_exceptions=True,
    )

    check_names: list[str] = [
        "L0_protocol",
        "L2_memory",
        "L3_structured",
        "L6_safety",
        "L8_knowledge",
        "L5_observability",
    ]

    checks: dict[str, dict[str, Any]] = {}
    for name, result in zip(check_names, results):
        if isinstance(result, BaseException):
            checks[name] = {
                "status": "unhealthy",
                "provider": "unknown",
                "error": str(result),
            }
        else:
            checks[name] = {
                "status": result.status,
                "provider": result.provider,
                "latency_ms": result.latency_ms,
            }
            if result.error:
                checks[name]["error"] = result.error
            if result.details:
                checks[name]["details"] = result.details

    # Calculate summary
    statuses: list[str] = [c["status"] for c in checks.values()]
    summary: dict[str, int] = {
        "healthy": statuses.count("healthy"),
        "degraded": statuses.count("degraded"),
        "unhealthy": statuses.count("unhealthy"),
        "total": len(statuses),
    }

    # Determine overall status
    overall: str
    if all(s == "healthy" for s in statuses):
        overall = "healthy"
    elif any(s == "unhealthy" for s in statuses):
        overall = "unhealthy"
    else:
        overall = "degraded"

    return HealthStatus(
        status=overall,
        version=__version__,
        timestamp=datetime.now(timezone.utc).isoformat(),
        checks=checks,
        summary=summary,
    )


def format_health_output(status: HealthStatus, verbose: bool = False) -> str:
    """Format health status as JSON.

    Args:
        status: The HealthStatus object to format
        verbose: If True, include detailed check results

    Returns:
        JSON string representation of health status
    """
    output: dict[str, Any] = {
        "status": status.status,
        "version": status.version,
        "timestamp": status.timestamp,
        "summary": status.summary,
    }

    if verbose:
        output["checks"] = status.checks

    return json.dumps(output, indent=2)


def main() -> NoReturn:
    """Run health check and output results.

    Exits with:
        0: healthy or degraded status
        1: unhealthy status
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Unleash V35 Health Check"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Include detailed check results"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Only output status code"
    )
    args: argparse.Namespace = parser.parse_args()

    # Run health check
    status: HealthStatus = asyncio.run(get_health_status())

    if args.quiet:
        # Exit with appropriate code
        sys.exit(0 if status.status == "healthy" else 1)

    # Print full output
    print(format_health_output(status, verbose=args.verbose or True))

    # Exit with appropriate code
    if status.status == "healthy":
        sys.exit(0)
    elif status.status == "degraded":
        sys.exit(0)  # Degraded is still operational
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
