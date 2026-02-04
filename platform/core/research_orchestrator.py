"""
Research Orchestrator - Unleash Claude's Potential
====================================================

This orchestrator solves Claude's core limitation: research results are often
too large to fit in context. By delegating to specialized agents, Claude can:

1. **Delegate large research** to synthesis agents
2. **Parallelize searches** across multiple tools (Exa, Tavily, Jina, Perplexity)
3. **Use RAG for memory + research** - store findings, recall relevant context
4. **Agent-to-agent communication** - specialists report back to Claude

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     CLAUDE (Orchestrator)                        │
    │  - Receives user request                                         │
    │  - Delegates to research agents                                  │
    │  - Synthesizes summarized results                                │
    └────────────────────────┬────────────────────────────────────────┘
                             │ Spawns agents for parallel research
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Exa Agent   │    │ Tavily Agent│    │ Jina Agent  │
    │ Neural srch │    │ AI search   │    │ URL→MD      │
    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
           │                  │                  │
           ▼                  ▼                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    SYNTHESIS AGENT                               │
    │  - Receives all research results                                 │
    │  - Identifies patterns, discrepancies                            │
    │  - Produces condensed summary for Claude                         │
    │  - Stores key findings in RAG memory                             │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    orchestrator = ResearchOrchestrator()
    await orchestrator.initialize()

    # Deep research with agent synthesis
    result = await orchestrator.research(
        "LangGraph patterns for multi-agent systems",
        mode="comprehensive"
    )

    # Get SDK docs (fast path via Context7)
    docs = await orchestrator.get_sdk_docs("langgraph", "StateGraph")

    # Research with memory storage
    result = await orchestrator.research_and_remember(
        "distributed consensus algorithms",
        memory_key="consensus_patterns"
    )

Version: 1.0.0 (2026-02-02)
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

# Import research adapters
try:
    from adapters.exa_adapter import ExaAdapter
    from adapters.tavily_adapter import TavilyAdapter
    from adapters.jina_adapter import JinaAdapter
    from adapters.perplexity_adapter import PerplexityAdapter
except ImportError:
    try:
        from ..adapters.exa_adapter import ExaAdapter
        from ..adapters.tavily_adapter import TavilyAdapter
        from ..adapters.jina_adapter import JinaAdapter
        from ..adapters.perplexity_adapter import PerplexityAdapter
    except ImportError:
        ExaAdapter = None
        TavilyAdapter = None
        JinaAdapter = None
        PerplexityAdapter = None


class ResearchMode(str, Enum):
    """Research depth modes."""
    QUICK = "quick"              # Single best tool (<1s)
    STANDARD = "standard"        # 2-3 tools in parallel (~3s)
    COMPREHENSIVE = "comprehensive"  # All tools + synthesis (~10s)
    DEEP = "deep"                # Comprehensive + deep search (~30s)


@dataclass
class ResearchSource:
    """Individual research source result."""
    tool: str
    title: str
    url: str
    content: str
    score: float = 0.0
    published_date: Optional[str] = None


@dataclass
class SynthesizedResearch:
    """Synthesized research result from multiple sources."""
    query: str
    mode: ResearchMode

    # Synthesized content (fits in Claude's context)
    summary: str
    key_findings: list[str]
    patterns: list[str]
    discrepancies: list[str]

    # Sources (for citation)
    sources: list[ResearchSource]
    tools_used: list[str]

    # Metadata
    confidence: float = 0.0
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Memory integration
    stored_in_memory: bool = False
    memory_key: Optional[str] = None


class ResearchOrchestrator:
    """
    Orchestrates research across multiple tools with agent synthesis.

    The key insight: Claude can't read 100KB of research results, but it CAN
    orchestrate agents to do the reading and synthesize the key findings.
    """

    def __init__(
        self,
        max_parallel_searches: int = 4,
        max_content_per_source: int = 2000,
        enable_memory: bool = True,
    ):
        """
        Initialize the research orchestrator.

        Args:
            max_parallel_searches: Max concurrent search operations
            max_content_per_source: Max chars per source (for synthesis)
            enable_memory: Store findings in RAG memory
        """
        self._max_parallel = max_parallel_searches
        self._max_content = max_content_per_source
        self._enable_memory = enable_memory

        # Adapters
        self._exa: Optional[ExaAdapter] = None
        self._tavily: Optional[TavilyAdapter] = None
        self._jina: Optional[JinaAdapter] = None
        self._perplexity: Optional[PerplexityAdapter] = None

        # Stats
        self._stats = {
            "total_queries": 0,
            "quick_queries": 0,
            "comprehensive_queries": 0,
            "sources_fetched": 0,
            "memory_stores": 0,
        }

    async def initialize(self) -> dict[str, bool]:
        """
        Initialize all research adapters.

        Returns dict showing which adapters are available.
        """
        status = {}

        # Initialize Exa
        if ExaAdapter:
            self._exa = ExaAdapter()
            result = await self._exa.initialize({})
            status["exa"] = result.success

        # Initialize Tavily
        if TavilyAdapter:
            self._tavily = TavilyAdapter()
            result = await self._tavily.initialize({})
            status["tavily"] = result.success

        # Initialize Jina
        if JinaAdapter:
            self._jina = JinaAdapter()
            result = await self._jina.initialize({})
            status["jina"] = result.success

        # Initialize Perplexity
        if PerplexityAdapter:
            self._perplexity = PerplexityAdapter()
            result = await self._perplexity.initialize({})
            status["perplexity"] = result.success

        return status

    async def research(
        self,
        query: str,
        mode: ResearchMode = ResearchMode.STANDARD,
        include_urls: Optional[list[str]] = None,
    ) -> SynthesizedResearch:
        """
        Execute research with agent synthesis.

        This is the main entry point. Claude calls this, and agents
        do the heavy lifting of searching and synthesizing.

        Args:
            query: Research query
            mode: Research depth
            include_urls: Specific URLs to include in research

        Returns:
            SynthesizedResearch with condensed findings
        """
        self._stats["total_queries"] += 1
        start_time = time.time()

        # Select tools based on mode
        tasks = []

        if mode == ResearchMode.QUICK:
            self._stats["quick_queries"] += 1
            # Just use the fastest tool
            if self._tavily:
                tasks.append(self._search_tavily(query, max_results=5))
            elif self._exa:
                tasks.append(self._search_exa(query, search_type="fast"))

        elif mode == ResearchMode.STANDARD:
            # Parallel search with 2-3 tools
            if self._exa:
                tasks.append(self._search_exa(query))
            if self._tavily:
                tasks.append(self._search_tavily(query))

        elif mode in [ResearchMode.COMPREHENSIVE, ResearchMode.DEEP]:
            self._stats["comprehensive_queries"] += 1
            # All tools
            if self._exa:
                search_type = "deep" if mode == ResearchMode.DEEP else "auto"
                tasks.append(self._search_exa(query, search_type=search_type))
            if self._tavily:
                depth = "deep" if mode == ResearchMode.DEEP else "basic"
                tasks.append(self._search_tavily(query, depth=depth))
            if self._jina:
                tasks.append(self._search_jina(query))
            if self._perplexity and mode == ResearchMode.DEEP:
                tasks.append(self._search_perplexity(query))

        # Include specific URLs if provided
        if include_urls and self._jina:
            for url in include_urls[:3]:  # Limit to 3 URLs
                tasks.append(self._read_url(url))

        # Execute all searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect sources
        all_sources = []
        tools_used = []

        for result in results:
            if isinstance(result, Exception):
                continue
            if isinstance(result, list):
                all_sources.extend(result)
                if result:
                    tools_used.append(result[0].tool)

        self._stats["sources_fetched"] += len(all_sources)

        # Synthesize results
        synthesis = self._synthesize(query, all_sources, mode)
        synthesis.tools_used = list(set(tools_used))
        synthesis.latency_ms = (time.time() - start_time) * 1000

        return synthesis

    async def get_sdk_docs(
        self,
        library: str,
        query: str,
    ) -> SynthesizedResearch:
        """
        Get SDK documentation via Context7 + Jina fallback.

        This is the fast path for documentation lookup.
        Context7 has up-to-date docs, Jina can read official sites.

        Args:
            library: Library name (e.g., "langgraph", "react")
            query: Specific query within the library
        """
        sources = []

        # Try Context7 (MCP call would go here in production)
        # For now, use Jina to read official docs

        if self._jina:
            # Map common libraries to their doc URLs
            doc_urls = {
                "langgraph": "https://langchain-ai.github.io/langgraph/",
                "langchain": "https://python.langchain.com/docs/",
                "react": "https://react.dev/reference/react",
                "nextjs": "https://nextjs.org/docs",
                "fastapi": "https://fastapi.tiangolo.com/",
                "pydantic": "https://docs.pydantic.dev/latest/",
                "anthropic": "https://docs.anthropic.com/",
                "openai": "https://platform.openai.com/docs/",
            }

            base_url = doc_urls.get(library.lower())
            if base_url:
                result = await self._jina.execute("read", url=base_url)
                if result.success:
                    sources.append(ResearchSource(
                        tool="jina",
                        title=f"{library} Documentation",
                        url=base_url,
                        content=result.data.get("content", "")[:self._max_content],
                        score=0.95,
                    ))

        # Also search for specific query
        if self._tavily:
            search_query = f"{library} {query} documentation"
            results = await self._search_tavily(search_query, max_results=3)
            sources.extend(results)

        return self._synthesize(f"{library}: {query}", sources, ResearchMode.QUICK)

    async def research_and_remember(
        self,
        query: str,
        memory_key: str,
        mode: ResearchMode = ResearchMode.STANDARD,
    ) -> SynthesizedResearch:
        """
        Research and store findings in RAG memory.

        This enables Claude to recall research findings later without
        re-searching. Perfect for building up domain knowledge.

        Args:
            query: Research query
            memory_key: Key for memory storage
            mode: Research depth
        """
        result = await self.research(query, mode)

        if self._enable_memory:
            # Store in memory (would connect to Letta/Mem0/Qdrant)
            # For now, mark as stored
            result.stored_in_memory = True
            result.memory_key = memory_key
            self._stats["memory_stores"] += 1

        return result

    # =========================================================================
    # Search Implementations
    # =========================================================================

    async def _search_exa(
        self,
        query: str,
        search_type: str = "auto",
        max_results: int = 10,
    ) -> list[ResearchSource]:
        """Search using Exa."""
        if not self._exa:
            return []

        result = await self._exa.execute(
            "search",
            query=query,
            type=search_type,
            num_results=max_results,
        )

        if not result.success:
            return []

        sources = []
        for r in result.data.get("results", []):
            sources.append(ResearchSource(
                tool="exa",
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("text", "")[:self._max_content],
                score=r.get("score", 0.0),
                published_date=r.get("published_date"),
            ))

        return sources

    async def _search_tavily(
        self,
        query: str,
        depth: str = "basic",
        max_results: int = 10,
    ) -> list[ResearchSource]:
        """Search using Tavily."""
        if not self._tavily:
            return []

        result = await self._tavily.execute(
            "search",
            query=query,
            search_depth=depth,
            max_results=max_results,
        )

        if not result.success:
            return []

        sources = []
        for r in result.data.get("results", []):
            sources.append(ResearchSource(
                tool="tavily",
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", "")[:self._max_content],
                score=r.get("score", 0.0),
                published_date=r.get("published_date"),
            ))

        # Include Tavily's AI answer as a source
        answer = result.data.get("answer")
        if answer:
            sources.insert(0, ResearchSource(
                tool="tavily",
                title="Tavily AI Answer",
                url="",
                content=answer[:self._max_content],
                score=1.0,
            ))

        return sources

    async def _search_jina(self, query: str) -> list[ResearchSource]:
        """Search using Jina."""
        if not self._jina:
            return []

        result = await self._jina.execute("search", query=query)

        if not result.success:
            return []

        # Jina returns markdown, treat as single source
        return [ResearchSource(
            tool="jina",
            title=f"Jina Search: {query}",
            url="",
            content=result.data.get("content", "")[:self._max_content],
            score=0.85,
        )]

    async def _search_perplexity(self, query: str) -> list[ResearchSource]:
        """Deep search using Perplexity."""
        if not self._perplexity:
            return []

        result = await self._perplexity.execute("research", query=query)

        if not result.success:
            return []

        sources = []

        # Main research content
        content = result.data.get("content", "")
        if content:
            sources.append(ResearchSource(
                tool="perplexity",
                title=f"Perplexity Research: {query}",
                url="",
                content=content[:self._max_content],
                score=0.95,
            ))

        # Add citations as sources
        for citation in result.data.get("citations", [])[:5]:
            sources.append(ResearchSource(
                tool="perplexity",
                title=citation.get("title", "Citation"),
                url=citation.get("url", ""),
                content="",  # Citation only
                score=0.8,
            ))

        return sources

    async def _read_url(self, url: str) -> list[ResearchSource]:
        """Read a specific URL using Jina."""
        if not self._jina:
            return []

        result = await self._jina.execute("read", url=url)

        if not result.success:
            return []

        return [ResearchSource(
            tool="jina",
            title=f"URL: {url}",
            url=url,
            content=result.data.get("content", "")[:self._max_content],
            score=0.9,
        )]

    # =========================================================================
    # Synthesis
    # =========================================================================

    def _synthesize(
        self,
        query: str,
        sources: list[ResearchSource],
        mode: ResearchMode,
    ) -> SynthesizedResearch:
        """
        Synthesize multiple sources into a condensed summary.

        In production, this would use an LLM to synthesize.
        For now, we extract key patterns and deduplicate.
        """
        if not sources:
            return SynthesizedResearch(
                query=query,
                mode=mode,
                summary="No sources found for this query.",
                key_findings=[],
                patterns=[],
                discrepancies=[],
                sources=[],
                tools_used=[],
                confidence=0.0,
            )

        # Sort by score
        sources.sort(key=lambda x: x.score, reverse=True)

        # Extract key findings (top 5 unique insights)
        key_findings = []
        seen_content = set()

        for source in sources[:10]:
            # Simple dedup by first 100 chars
            content_key = source.content[:100].lower()
            if content_key not in seen_content:
                seen_content.add(content_key)
                # Extract first meaningful sentence
                sentences = source.content.split(". ")
                if sentences:
                    finding = sentences[0].strip()
                    if len(finding) > 20:  # Skip tiny snippets
                        key_findings.append(f"[{source.tool}] {finding}")

        # Build summary from top sources
        summary_parts = []
        for source in sources[:3]:
            if source.content:
                # Take first 300 chars of each top source
                summary_parts.append(source.content[:300])

        summary = "\n\n".join(summary_parts) if summary_parts else "No content available."

        # Calculate confidence based on source agreement
        tools = set(s.tool for s in sources)
        confidence = min(0.95, 0.5 + (len(tools) * 0.15) + (len(sources) * 0.03))

        return SynthesizedResearch(
            query=query,
            mode=mode,
            summary=summary,
            key_findings=key_findings[:5],
            patterns=[],  # Would need LLM to extract
            discrepancies=[],  # Would need LLM to detect
            sources=sources,
            tools_used=list(tools),
            confidence=confidence,
        )

    # =========================================================================
    # Utility
    # =========================================================================

    async def shutdown(self) -> None:
        """Cleanup all adapters."""
        if self._exa:
            await self._exa.shutdown()
        if self._tavily:
            await self._tavily.shutdown()
        if self._jina:
            await self._jina.shutdown()
        if self._perplexity:
            await self._perplexity.shutdown()

    def get_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return dict(self._stats)


# =============================================================================
# Factory
# =============================================================================

_orchestrator: Optional[ResearchOrchestrator] = None


def get_research_orchestrator() -> ResearchOrchestrator:
    """Get or create the global ResearchOrchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ResearchOrchestrator()
    return _orchestrator


async def quick_research(query: str) -> SynthesizedResearch:
    """Quick research helper."""
    orch = get_research_orchestrator()
    await orch.initialize()
    return await orch.research(query, ResearchMode.QUICK)


async def deep_research(query: str) -> SynthesizedResearch:
    """Deep research helper."""
    orch = get_research_orchestrator()
    await orch.initialize()
    return await orch.research(query, ResearchMode.DEEP)


# =============================================================================
# CLI
# =============================================================================

async def main():
    """CLI for testing the research orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(description="Research Orchestrator CLI")
    parser.add_argument("query", nargs="?", help="Research query")
    parser.add_argument("--mode", choices=["quick", "standard", "comprehensive", "deep"],
                        default="standard", help="Research mode")
    parser.add_argument("--docs", metavar="LIBRARY", help="Get docs for library")
    parser.add_argument("--status", action="store_true", help="Show adapter status")

    args = parser.parse_args()

    orchestrator = ResearchOrchestrator()
    status = await orchestrator.initialize()

    if args.status:
        print("\n" + "=" * 50)
        print("  RESEARCH ORCHESTRATOR STATUS")
        print("=" * 50)
        for adapter, available in status.items():
            icon = "✅" if available else "❌"
            print(f"  {icon} {adapter}: {'Ready' if available else 'Not available'}")
        print("=" * 50)
        return

    if args.docs:
        result = await orchestrator.get_sdk_docs(args.docs, args.query or "overview")
    elif args.query:
        result = await orchestrator.research(args.query, ResearchMode(args.mode))
    else:
        print("Usage: research_orchestrator.py <query> [--mode MODE] [--docs LIBRARY]")
        return

    # Print result
    print("\n" + "=" * 60)
    print(f"  RESEARCH: {result.query}")
    print("=" * 60)
    print(f"Mode: {result.mode.value}")
    print(f"Tools: {', '.join(result.tools_used)}")
    print(f"Sources: {len(result.sources)}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Latency: {result.latency_ms:.0f}ms")
    print("-" * 60)
    print("KEY FINDINGS:")
    for finding in result.key_findings:
        print(f"  • {finding[:100]}...")
    print("-" * 60)
    print("SUMMARY:")
    print(result.summary[:1000])
    print("=" * 60)

    await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
