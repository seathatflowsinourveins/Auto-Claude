"""
API Validator - Research Stack Health Check
============================================

Validates all research API connections at startup to prevent silent failures.
Per gap analysis GAP-01: 99.8% of findings were from Exa only because
Tavily and Perplexity were silently failing.

Usage:
    from core.api_validator import validate_research_stack

    # At startup
    results = await validate_research_stack()
    for name, status in results.items():
        print(f"{name}: {status}")

    # In production
    validator = ResearchStackValidator()
    await validator.run_health_checks()

Version: 1.0.0 (2026-02-03)
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check httpx availability at module level
HTTPX_AVAILABLE = False
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore


@dataclass
class APIHealthResult:
    """Result of a single API health check."""
    name: str
    status: str  # "OK", "FAIL", "DEGRADED", "NO_KEY"
    latency_ms: float = 0.0
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ValidationReport:
    """Complete validation report for all APIs."""
    results: Dict[str, APIHealthResult]
    total_apis: int = 0
    healthy_apis: int = 0
    failed_apis: int = 0
    degraded_apis: int = 0
    no_key_apis: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def all_healthy(self) -> bool:
        return self.failed_apis == 0 and self.degraded_apis == 0

    @property
    def source_diversity(self) -> float:
        """Calculate source diversity percentage."""
        if self.total_apis == 0:
            return 0.0
        return (self.healthy_apis / self.total_apis) * 100

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "RESEARCH STACK VALIDATION REPORT",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            f"Total APIs: {self.total_apis}",
            f"Healthy: {self.healthy_apis} ({self.source_diversity:.1f}%)",
            f"Failed: {self.failed_apis}",
            f"Degraded: {self.degraded_apis}",
            f"No API Key: {self.no_key_apis}",
            "-" * 60,
        ]
        for name, result in self.results.items():
            status_icon = "✅" if result.status == "OK" else "❌" if result.status == "FAIL" else "⚠️"
            lines.append(f"{status_icon} {name}: {result.status} ({result.latency_ms:.0f}ms)")
            if result.error:
                lines.append(f"   Error: {result.error[:80]}")
        lines.append("=" * 60)
        return "\n".join(lines)


class ResearchStackValidator:
    """
    Validates all research APIs at startup.

    Checks:
    - Exa (neural search)
    - Tavily (factual search)
    - Perplexity (AI search)
    - Firecrawl (web scraping)
    - Jina (embeddings)
    - Context7 (SDK docs)
    - Serper (SERP)
    """

    # Required environment variables for each API
    API_KEYS = {
        "exa": "EXA_API_KEY",
        "tavily": "TAVILY_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY",
        "firecrawl": "FIRECRAWL_API_KEY",
        "jina": "JINA_API_KEY",
        "serper": "SERPER_API_KEY",
        "context7": "CONTEXT7_API_KEY",  # Optional - higher rate limits
    }

    def __init__(self):
        self._results: Dict[str, APIHealthResult] = {}

    def _check_api_key(self, api_name: str) -> Optional[str]:
        """Check if API key is configured."""
        env_var = self.API_KEYS.get(api_name)
        if not env_var:
            return None
        return os.environ.get(env_var)

    async def check_exa(self) -> APIHealthResult:
        """Check Exa API connectivity."""
        name = "exa"
        api_key = self._check_api_key(name)

        if not api_key:
            return APIHealthResult(name=name, status="NO_KEY", error="EXA_API_KEY not set")

        try:
            import time
            start = time.time()

            from exa_py import Exa
            client = Exa(api_key=api_key)
            result = client.search("test", num_results=1)

            latency = (time.time() - start) * 1000
            return APIHealthResult(
                name=name,
                status="OK",
                latency_ms=latency,
                details={"results_count": len(result.results) if result.results else 0}
            )
        except ImportError:
            return APIHealthResult(name=name, status="DEGRADED", error="exa-py not installed")
        except Exception as e:
            return APIHealthResult(name=name, status="FAIL", error=str(e)[:200])

    async def check_tavily(self) -> APIHealthResult:
        """Check Tavily API connectivity."""
        name = "tavily"
        api_key = self._check_api_key(name)

        if not api_key:
            return APIHealthResult(name=name, status="NO_KEY", error="TAVILY_API_KEY not set")

        if not HTTPX_AVAILABLE:
            return APIHealthResult(name=name, status="DEGRADED", error="httpx not installed")

        try:
            import time
            start = time.time()

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={"api_key": api_key, "query": "test", "max_results": 1}
                )
                response.raise_for_status()
                data = response.json()

            latency = (time.time() - start) * 1000
            return APIHealthResult(
                name=name,
                status="OK",
                latency_ms=latency,
                details={"has_results": "results" in data}
            )
        except httpx.HTTPStatusError as e:
            return APIHealthResult(name=name, status="FAIL", error=f"HTTP {e.response.status_code}")
        except Exception as e:
            return APIHealthResult(name=name, status="FAIL", error=str(e)[:200])

    async def check_perplexity(self) -> APIHealthResult:
        """Check Perplexity API connectivity."""
        name = "perplexity"
        api_key = self._check_api_key(name)

        if not api_key:
            return APIHealthResult(name=name, status="NO_KEY", error="PERPLEXITY_API_KEY not set")

        if not HTTPX_AVAILABLE:
            return APIHealthResult(name=name, status="DEGRADED", error="httpx not installed")

        try:
            import time
            start = time.time()

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": "sonar",
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 10
                    }
                )
                response.raise_for_status()

            latency = (time.time() - start) * 1000
            return APIHealthResult(name=name, status="OK", latency_ms=latency)
        except httpx.HTTPStatusError as e:
            return APIHealthResult(name=name, status="FAIL", error=f"HTTP {e.response.status_code}")
        except Exception as e:
            return APIHealthResult(name=name, status="FAIL", error=str(e)[:200])

    async def check_firecrawl(self) -> APIHealthResult:
        """Check Firecrawl API connectivity."""
        name = "firecrawl"
        api_key = self._check_api_key(name)

        if not api_key:
            return APIHealthResult(name=name, status="NO_KEY", error="FIRECRAWL_API_KEY not set")

        try:
            import time
            start = time.time()

            from firecrawl import FirecrawlApp
            client = FirecrawlApp(api_key=api_key)
            # Use map which is lighter than scrape
            result = client.map_url("https://example.com", params={"limit": 1})

            latency = (time.time() - start) * 1000
            return APIHealthResult(name=name, status="OK", latency_ms=latency)
        except ImportError:
            return APIHealthResult(name=name, status="DEGRADED", error="firecrawl not installed")
        except Exception as e:
            return APIHealthResult(name=name, status="FAIL", error=str(e)[:200])

    async def check_jina(self) -> APIHealthResult:
        """Check Jina API connectivity (via remote MCP)."""
        name = "jina"
        api_key = self._check_api_key(name)

        if not api_key:
            # Jina remote MCP may work without key at lower rate limits
            return APIHealthResult(name=name, status="DEGRADED", error="JINA_API_KEY not set (lower rate limits)")

        if not HTTPX_AVAILABLE:
            return APIHealthResult(name=name, status="DEGRADED", error="httpx not installed")

        try:
            import time
            start = time.time()

            async with httpx.AsyncClient(timeout=15.0) as client:
                # Use Jina reader API as health check
                response = await client.get(
                    "https://r.jina.ai/https://example.com",
                    headers={"Authorization": f"Bearer {api_key}"}
                )
                response.raise_for_status()

            latency = (time.time() - start) * 1000
            return APIHealthResult(name=name, status="OK", latency_ms=latency)
        except Exception as e:
            return APIHealthResult(name=name, status="FAIL", error=str(e)[:200])

    async def check_context7(self) -> APIHealthResult:
        """Check Context7 API connectivity."""
        name = "context7"
        # Context7 works without key but with lower rate limits

        if not HTTPX_AVAILABLE:
            return APIHealthResult(name=name, status="DEGRADED", error="httpx not installed")

        try:
            import time
            start = time.time()

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    "https://api.context7.com/api/v2/libs/search",
                    params={"libraryName": "react", "query": ""}
                )
                response.raise_for_status()

            latency = (time.time() - start) * 1000
            api_key = self._check_api_key(name)
            status = "OK" if api_key else "DEGRADED"
            return APIHealthResult(
                name=name,
                status=status,
                latency_ms=latency,
                details={"has_api_key": bool(api_key)}
            )
        except Exception as e:
            return APIHealthResult(name=name, status="FAIL", error=str(e)[:200])

    async def check_serper(self) -> APIHealthResult:
        """Check Serper API connectivity."""
        name = "serper"
        api_key = self._check_api_key(name)

        if not api_key:
            return APIHealthResult(name=name, status="NO_KEY", error="SERPER_API_KEY not set")

        if not HTTPX_AVAILABLE:
            return APIHealthResult(name=name, status="DEGRADED", error="httpx not installed")

        try:
            import time
            start = time.time()

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                    json={"q": "test"}
                )
                response.raise_for_status()

            latency = (time.time() - start) * 1000
            return APIHealthResult(name=name, status="OK", latency_ms=latency)
        except httpx.HTTPStatusError as e:
            return APIHealthResult(name=name, status="FAIL", error=f"HTTP {e.response.status_code}")
        except Exception as e:
            return APIHealthResult(name=name, status="FAIL", error=str(e)[:200])

    async def run_health_checks(self, parallel: bool = True) -> ValidationReport:
        """
        Run all health checks.

        Args:
            parallel: Run checks in parallel (faster) or sequentially

        Returns:
            ValidationReport with all results
        """
        checks = [
            self.check_exa(),
            self.check_tavily(),
            self.check_perplexity(),
            self.check_firecrawl(),
            self.check_jina(),
            self.check_context7(),
            self.check_serper(),
        ]

        if parallel:
            results = await asyncio.gather(*checks, return_exceptions=True)
        else:
            results = []
            for check in checks:
                try:
                    results.append(await check)
                except Exception as e:
                    results.append(e)

        # Process results
        report_results: Dict[str, APIHealthResult] = {}
        for result in results:
            if isinstance(result, Exception):
                # Handle unexpected exceptions
                report_results["unknown"] = APIHealthResult(
                    name="unknown",
                    status="FAIL",
                    error=str(result)[:200]
                )
            elif isinstance(result, APIHealthResult):
                report_results[result.name] = result
                self._results[result.name] = result

        # Calculate stats
        total = len(report_results)
        healthy = sum(1 for r in report_results.values() if r.status == "OK")
        failed = sum(1 for r in report_results.values() if r.status == "FAIL")
        degraded = sum(1 for r in report_results.values() if r.status == "DEGRADED")
        no_key = sum(1 for r in report_results.values() if r.status == "NO_KEY")

        return ValidationReport(
            results=report_results,
            total_apis=total,
            healthy_apis=healthy,
            failed_apis=failed,
            degraded_apis=degraded,
            no_key_apis=no_key,
        )


async def validate_research_stack() -> Dict[str, str]:
    """
    Convenience function to validate research stack at startup.

    Returns:
        Dict mapping API name to status string
    """
    validator = ResearchStackValidator()
    report = await validator.run_health_checks()

    # Log summary
    logger.info(report.summary())

    # Return simple dict for quick checks
    return {name: result.status for name, result in report.results.items()}


# CLI interface
if __name__ == "__main__":
    async def main():
        print("Validating Research Stack...")
        print()

        validator = ResearchStackValidator()
        report = await validator.run_health_checks()

        print(report.summary())

        if not report.all_healthy:
            print("\n⚠️  WARNING: Some APIs are not fully operational.")
            print("Research source diversity may be limited.")
        else:
            print("\n✅ All research APIs operational!")

    asyncio.run(main())
