"""
Research Bridge - Connects ~/.claude/integrations to UNLEASH Platform
=====================================================================

This bridge enables seamless integration between:
- ~/.claude/integrations/research_orchestrator.py (Parallel research)
- ~/.claude/integrations/mcp_executor.py (Real MCP connections)
- ~/.claude/integrations/seamless_integration.py (Session management)

And the UNLEASH platform:
- platform/core/research_engine.py (SDK wrappers)
- platform/core/ecosystem_orchestrator.py (Multi-SDK integration)

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                  RESEARCH BRIDGE                     │
    │  (This file - unified entry point)                   │
    └─────────────────────────┬───────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ ~/.claude/  │    │ UNLEASH     │    │ MCP Tools   │
    │ integrations│    │ Research    │    │ (Runtime)   │
    │ (Python)    │    │ Engine      │    │             │
    └─────────────┘    └─────────────┘    └─────────────┘

Version: 1.0.0 (2026-01-30)
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Awaitable

# Add integration paths
CLAUDE_INTEGRATIONS = Path.home() / ".claude" / "integrations"
if CLAUDE_INTEGRATIONS.exists() and str(CLAUDE_INTEGRATIONS) not in sys.path:
    sys.path.insert(0, str(CLAUDE_INTEGRATIONS))

# Import from ~/.claude/integrations/ if available
INTEGRATIONS_AVAILABLE = False
try:
    from research_orchestrator import (
        ComprehensiveResearchOrchestrator,
        ResearchIntent,
        SynthesizedResult,
        ToolResult,
        format_research_report,
    )
    INTEGRATIONS_AVAILABLE = True
except ImportError:
    ComprehensiveResearchOrchestrator = None  # type: ignore
    ResearchIntent = None  # type: ignore
    SynthesizedResult = None  # type: ignore

MCP_EXECUTOR_AVAILABLE = False
try:
    from mcp_executor import MCPExecutor, create_mcp_executor
    MCP_EXECUTOR_AVAILABLE = True
except ImportError:
    MCPExecutor = None  # type: ignore
    create_mcp_executor = None  # type: ignore

SEAMLESS_AVAILABLE = False
try:
    from seamless_integration import SeamlessIntegration, get_integration
    SEAMLESS_AVAILABLE = True
except ImportError:
    SeamlessIntegration = None  # type: ignore
    get_integration = None  # type: ignore

# Import UNLEASH components
UNLEASH_AVAILABLE = False
try:
    from .research_engine import ResearchEngine, get_engine
    from .ecosystem_orchestrator import EcosystemOrchestrator
    UNLEASH_AVAILABLE = True
except ImportError:
    try:
        from research_engine import ResearchEngine, get_engine
        from ecosystem_orchestrator import EcosystemOrchestrator
        UNLEASH_AVAILABLE = True
    except ImportError:
        ResearchEngine = None  # type: ignore
        get_engine = None  # type: ignore
        EcosystemOrchestrator = None  # type: ignore


# =============================================================================
# Unified Research Interface
# =============================================================================

class ResearchMode(str, Enum):
    """Research execution mode."""
    QUICK = "quick"           # Single best tool
    STANDARD = "standard"     # Parallel, 2-3 tools
    COMPREHENSIVE = "comprehensive"  # All tools
    DEEP = "deep"             # Comprehensive + academic


@dataclass
class UnifiedResearchResult:
    """Unified result from any research path."""
    query: str
    mode: ResearchMode
    source: str  # "integrations" | "unleash" | "mcp_direct"

    # Content
    answer: str
    evidence: list[str] = field(default_factory=list)
    discrepancies: list[str] = field(default_factory=list)

    # Metadata
    tools_used: list[str] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Raw data
    raw_data: dict[str, Any] = field(default_factory=dict)


class ResearchBridge:
    """
    Unified bridge for all research capabilities.

    Provides a single interface that works regardless of which
    backend is available (integrations, UNLEASH, or direct MCP).

    Usage:
        bridge = ResearchBridge()

        # Simple research
        result = await bridge.research("LangGraph StateGraph")

        # Comprehensive research
        result = await bridge.research(
            "distributed consensus algorithms",
            mode=ResearchMode.DEEP
        )

        # Get SDK docs specifically
        result = await bridge.get_docs("react", "useEffect cleanup")
    """

    def __init__(
        self,
        prefer_integrations: bool = True,
        enable_caching: bool = True,
    ):
        """
        Initialize the research bridge.

        Args:
            prefer_integrations: Prefer ~/.claude/integrations over UNLEASH
            enable_caching: Enable result caching
        """
        self._prefer_integrations = prefer_integrations
        self._enable_caching = enable_caching

        # Initialize backends
        self._integrations_orchestrator: Any = None
        self._mcp_executor: Any = None
        self._unleash_engine: Any = None
        self._seamless: Any = None

        # Stats
        self._stats = {
            "total_queries": 0,
            "integrations_used": 0,
            "unleash_used": 0,
            "mcp_direct_used": 0,
            "cache_hits": 0,
        }

        # Cache
        self._cache: dict[str, tuple[datetime, UnifiedResearchResult]] = {}
        self._cache_ttl = 300.0  # 5 minutes

    async def initialize(self) -> dict[str, bool]:
        """
        Initialize all available backends.

        Returns dict showing which backends are available.
        """
        status = {
            "integrations": False,
            "mcp_executor": False,
            "unleash": False,
            "seamless": False,
        }

        # Initialize integrations orchestrator
        if INTEGRATIONS_AVAILABLE and ComprehensiveResearchOrchestrator:
            try:
                # Create MCP executor for real tool calls
                if MCP_EXECUTOR_AVAILABLE and MCPExecutor:
                    self._mcp_executor = MCPExecutor()
                    await self._mcp_executor.__aenter__()
                    status["mcp_executor"] = True

                self._integrations_orchestrator = ComprehensiveResearchOrchestrator(
                    mcp_executor=self._mcp_executor if self._mcp_executor else None
                )
                status["integrations"] = True
            except Exception as e:
                print(f"Warning: Failed to initialize integrations: {e}")

        # Initialize UNLEASH engine
        if UNLEASH_AVAILABLE and get_engine:
            try:
                self._unleash_engine = get_engine()
                status["unleash"] = True
            except Exception as e:
                print(f"Warning: Failed to initialize UNLEASH engine: {e}")

        # Initialize seamless integration
        if SEAMLESS_AVAILABLE and get_integration:
            try:
                self._seamless = get_integration()
                await self._seamless.initialize()
                status["seamless"] = True
            except Exception as e:
                print(f"Warning: Failed to initialize seamless integration: {e}")

        return status

    async def close(self) -> None:
        """Close all resources."""
        if self._mcp_executor:
            await self._mcp_executor.__aexit__(None, None, None)

    async def __aenter__(self) -> "ResearchBridge":
        await self.initialize()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    # =========================================================================
    # Main Research Interface
    # =========================================================================

    async def research(
        self,
        query: str,
        mode: ResearchMode = ResearchMode.STANDARD,
        intent: str | None = None,
    ) -> UnifiedResearchResult:
        """
        Execute unified research across all available backends.

        Args:
            query: Research query
            mode: Research depth (quick/standard/comprehensive/deep)
            intent: Override intent detection (optional)

        Returns:
            UnifiedResearchResult with synthesized findings
        """
        self._stats["total_queries"] += 1
        start_time = datetime.now(timezone.utc)

        # Check cache
        cache_key = f"{mode}:{query}"
        if self._enable_caching and cache_key in self._cache:
            cached_time, cached_result = self._cache[cache_key]
            age = (datetime.now(timezone.utc) - cached_time).total_seconds()
            if age < self._cache_ttl:
                self._stats["cache_hits"] += 1
                return cached_result

        # Route to best available backend
        if self._prefer_integrations and self._integrations_orchestrator:
            result = await self._research_via_integrations(query, mode, intent)
            self._stats["integrations_used"] += 1
        elif self._unleash_engine:
            result = await self._research_via_unleash(query, mode)
            self._stats["unleash_used"] += 1
        elif self._mcp_executor:
            result = await self._research_via_mcp_direct(query, mode)
            self._stats["mcp_direct_used"] += 1
        else:
            # No backend available - return empty result
            result = UnifiedResearchResult(
                query=query,
                mode=mode,
                source="none",
                answer="No research backends available. Please configure integrations or UNLEASH.",
                confidence=0.0,
            )

        # Calculate latency
        result.latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        # Cache result
        if self._enable_caching:
            self._cache[cache_key] = (datetime.now(timezone.utc), result)

        return result

    async def _research_via_integrations(
        self,
        query: str,
        mode: ResearchMode,
        intent: str | None,
    ) -> UnifiedResearchResult:
        """Research using ~/.claude/integrations."""
        # Map mode to orchestrator method
        if mode == ResearchMode.DEEP:
            synthesized = await self._integrations_orchestrator.comprehensive_research(
                query,
                include_academic=True,
                include_site_crawl=True,
            )
        elif mode == ResearchMode.COMPREHENSIVE:
            synthesized = await self._integrations_orchestrator.comprehensive_research(query)
        else:
            # Detect intent if not provided
            detected_intent = None
            if intent and ResearchIntent:
                try:
                    detected_intent = ResearchIntent(intent)
                except ValueError:
                    detected_intent = None

            synthesized = await self._integrations_orchestrator.research(
                query,
                intent=detected_intent,
            )

        # Convert to unified result
        return UnifiedResearchResult(
            query=query,
            mode=mode,
            source="integrations",
            answer=synthesized.primary_answer,
            evidence=synthesized.supporting_evidence,
            discrepancies=synthesized.discrepancies,
            tools_used=synthesized.tools_queried,
            confidence=synthesized.confidence,
            latency_ms=synthesized.total_latency_ms,
            raw_data={
                "intent": synthesized.intent.value if synthesized.intent else "unknown",
                "source_agreement": synthesized.source_agreement,
                "tools_succeeded": synthesized.tools_succeeded,
                "synthesis_notes": synthesized.synthesis_notes,
            },
        )

    async def _research_via_unleash(
        self,
        query: str,
        mode: ResearchMode,
    ) -> UnifiedResearchResult:
        """Research using UNLEASH research engine."""
        # Use UNLEASH's Exa search
        try:
            exa_result = await self._unleash_engine.exa_search(
                query=query,
                search_type="auto",
                num_results=10 if mode in [ResearchMode.COMPREHENSIVE, ResearchMode.DEEP] else 5,
            )

            # Extract content
            content = ""
            sources = []
            if exa_result and hasattr(exa_result, "results"):
                for r in exa_result.results[:5]:
                    content += f"**{r.title}**\n{r.text[:500]}...\n\n"
                    sources.append(r.url)

            return UnifiedResearchResult(
                query=query,
                mode=mode,
                source="unleash",
                answer=content,
                evidence=sources,
                tools_used=["exa"],
                confidence=0.85,
                raw_data={"exa_results": len(exa_result.results) if exa_result else 0},
            )
        except Exception as e:
            return UnifiedResearchResult(
                query=query,
                mode=mode,
                source="unleash",
                answer=f"UNLEASH research failed: {e}",
                confidence=0.0,
            )

    async def _research_via_mcp_direct(
        self,
        query: str,
        mode: ResearchMode,
    ) -> UnifiedResearchResult:
        """Research using direct MCP tool calls."""
        tools_used = []
        contents = []
        sources = []

        # Query Tavily (most reliable)
        try:
            tavily_result = await self._mcp_executor.execute("tavily", query)
            contents.append(tavily_result.get("content", ""))
            sources.extend(tavily_result.get("sources", []))
            tools_used.append("tavily")
        except Exception:
            pass

        # Query Exa for comprehensive modes
        if mode in [ResearchMode.COMPREHENSIVE, ResearchMode.DEEP]:
            try:
                exa_result = await self._mcp_executor.execute("exa", query)
                contents.append(exa_result.get("content", ""))
                sources.extend(exa_result.get("sources", []))
                tools_used.append("exa")
            except Exception:
                pass

        # Combine results
        combined_content = "\n\n---\n\n".join(contents)

        return UnifiedResearchResult(
            query=query,
            mode=mode,
            source="mcp_direct",
            answer=combined_content,
            evidence=[s.get("url", "") for s in sources if isinstance(s, dict)],
            tools_used=tools_used,
            confidence=0.7 if tools_used else 0.0,
        )

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def get_docs(
        self,
        library: str,
        query: str,
    ) -> UnifiedResearchResult:
        """Get SDK documentation via Context7."""
        if self._mcp_executor:
            try:
                result = await self._mcp_executor.execute(
                    "context7",
                    f"{library}: {query}",
                    {"libraryId": f"/{library}"},
                )
                return UnifiedResearchResult(
                    query=f"{library}: {query}",
                    mode=ResearchMode.QUICK,
                    source="mcp_direct",
                    answer=result.get("content", ""),
                    tools_used=["context7"],
                    confidence=result.get("confidence", 0.95),
                )
            except Exception as e:
                return UnifiedResearchResult(
                    query=f"{library}: {query}",
                    mode=ResearchMode.QUICK,
                    source="mcp_direct",
                    answer=f"Context7 query failed: {e}",
                    confidence=0.0,
                )

        # Fallback to general research
        return await self.research(f"{library} {query}", ResearchMode.STANDARD, "sdk_docs")

    async def deep_research(
        self,
        topic: str,
        include_academic: bool = True,
    ) -> UnifiedResearchResult:
        """Execute deep research with all available sources."""
        return await self.research(
            topic,
            mode=ResearchMode.DEEP if include_academic else ResearchMode.COMPREHENSIVE,
        )

    async def quick_search(
        self,
        query: str,
    ) -> UnifiedResearchResult:
        """Quick search with single best tool."""
        return await self.research(query, mode=ResearchMode.QUICK)

    def get_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return dict(self._stats)


# =============================================================================
# Factory Functions
# =============================================================================

_bridge: ResearchBridge | None = None


def get_bridge() -> ResearchBridge:
    """Get or create the global ResearchBridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = ResearchBridge()
    return _bridge


async def research(query: str, mode: str = "standard") -> UnifiedResearchResult:
    """Quick research function."""
    bridge = get_bridge()
    await bridge.initialize()
    return await bridge.research(query, ResearchMode(mode))


async def get_docs(library: str, query: str) -> UnifiedResearchResult:
    """Quick documentation lookup."""
    bridge = get_bridge()
    await bridge.initialize()
    return await bridge.get_docs(library, query)


# =============================================================================
# CLI Interface
# =============================================================================

async def main():
    """CLI for testing the research bridge."""
    import argparse

    parser = argparse.ArgumentParser(description="Research Bridge CLI")
    parser.add_argument("query", nargs="?", help="Research query")
    parser.add_argument("--mode", choices=["quick", "standard", "comprehensive", "deep"],
                        default="standard", help="Research mode")
    parser.add_argument("--docs", metavar="LIBRARY", help="Get docs for library")
    parser.add_argument("--status", action="store_true", help="Show backend status")

    args = parser.parse_args()

    async with ResearchBridge() as bridge:
        status = await bridge.initialize()

        if args.status:
            print("\n" + "=" * 50)
            print("  RESEARCH BRIDGE STATUS")
            print("=" * 50)
            for backend, available in status.items():
                icon = "✅" if available else "❌"
                print(f"  {icon} {backend}: {'Available' if available else 'Not available'}")
            print("=" * 50)
            return

        if args.docs:
            query = args.query or "overview"
            result = await bridge.get_docs(args.docs, query)
        elif args.query:
            result = await bridge.research(args.query, ResearchMode(args.mode))
        else:
            print("Usage: research_bridge.py <query> [--mode MODE] [--docs LIBRARY]")
            return

        # Print result
        print("\n" + "=" * 60)
        print(f"  RESEARCH: {result.query}")
        print("=" * 60)
        print(f"Mode: {result.mode.value}")
        print(f"Source: {result.source}")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Tools: {', '.join(result.tools_used)}")
        print(f"Latency: {result.latency_ms:.0f}ms")
        print("-" * 60)
        print("ANSWER:")
        print(result.answer[:1000] if result.answer else "(No answer)")
        if result.discrepancies:
            print("-" * 60)
            print("DISCREPANCIES:")
            for d in result.discrepancies:
                print(f"  ⚠️ {d}")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
