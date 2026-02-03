"""
Ultimate Research Swarm - Claude Flow V3 + UNLEASH Integration
===============================================================

The crown jewel: Combines Claude Flow V3 multi-agent orchestration with
UNLEASH research infrastructure to truly unleash Claude's potential.

Key Insight: Research results are often too large (100KB+) for Claude's
context. This module solves that by:
1. Spawning specialized research agents (Exa, Tavily, Jina, Perplexity)
2. Running parallel searches across all tools
3. Synthesizing results through a Queen agent (100KB → 2-4KB)
4. Storing findings in RAG memory for future recall

Architecture:
    User Request
         │
         ▼
    ┌─────────────────────────────────────────────────┐
    │          CLAUDE (Ultimate Orchestrator)          │
    │  - Classifies depth (quick/standard/deep)        │
    │  - Spawns Claude Flow V3 research swarm          │
    └──────────────────────┬──────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
      ┌─────────┐    ┌─────────┐    ┌─────────┐
      │Exa Agent│    │Tavily   │    │Jina     │
      │<350ms   │    │AI Search│    │URL→MD   │
      └────┬────┘    └────┬────┘    └────┬────┘
           └───────────────┴───────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────┐
    │            SYNTHESIS QUEEN AGENT                 │
    │  - 100KB+ → 2-4KB condensed summary              │
    │  - Patterns, discrepancies, citations            │
    │  - Hive-mind consensus for complex decisions     │
    └──────────────────────┬──────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
      ┌─────────┐    ┌─────────┐    ┌─────────┐
      │  Letta  │    │  Mem0   │    │ Qdrant  │
      └─────────┘    └─────────┘    └─────────┘

Usage:
    swarm = UltimateResearchSwarm()
    await swarm.initialize()

    # Quick research
    result = await swarm.research("LangGraph patterns", depth=ResearchDepth.QUICK)

    # Comprehensive with memory storage
    result = await swarm.research(
        "distributed consensus algorithms",
        depth=ResearchDepth.COMPREHENSIVE,
        memory_key="consensus_patterns"
    )

Version: 1.0.0 (2026-02-02)
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

# Research adapters
try:
    from platform.adapters.exa_adapter import ExaAdapter
    from platform.adapters.tavily_adapter import TavilyAdapter
    from platform.adapters.jina_adapter import JinaAdapter
    from platform.adapters.perplexity_adapter import PerplexityAdapter
    from platform.adapters.letta_adapter import LettaAdapter
    from platform.adapters.mem0_adapter import Mem0Adapter
except ImportError:
    try:
        from ..adapters.exa_adapter import ExaAdapter
        from ..adapters.tavily_adapter import TavilyAdapter
        from ..adapters.jina_adapter import JinaAdapter
        from ..adapters.perplexity_adapter import PerplexityAdapter
        from ..adapters.letta_adapter import LettaAdapter
        from ..adapters.mem0_adapter import Mem0Adapter
    except ImportError:
        ExaAdapter = None
        TavilyAdapter = None
        JinaAdapter = None
        PerplexityAdapter = None
        LettaAdapter = None
        Mem0Adapter = None


# =============================================================================
# Data Types
# =============================================================================

class ResearchDepth(str, Enum):
    """Research depth levels."""
    QUICK = "quick"              # Single best tool (<2s)
    STANDARD = "standard"        # 2-3 tools parallel (~5s)
    COMPREHENSIVE = "comprehensive"  # All tools (~15s)
    DEEP = "deep"                # Comprehensive + consensus (~30s)


class ResearchAgentType(str, Enum):
    """Specialized research agent types."""
    EXA_NEURAL = "exa-neural"        # Fast neural search (<350ms)
    TAVILY_AI = "tavily-ai"          # AI-optimized search + Agent-in-a-Box
    JINA_READER = "jina-reader"      # URL to markdown conversion
    PERPLEXITY_DEEP = "perplexity"   # Deep multi-step research


@dataclass
class ResearchSwarmConfig:
    """Configuration for the research swarm."""
    max_agents: int = 8
    topology: str = "hierarchical"
    consensus_type: str = "raft"
    synthesis_max_kb: int = 4
    memory_backend: str = "hybrid"
    enable_hnsw: bool = True
    parallel_timeout_s: int = 60


@dataclass
class ResearchSource:
    """Individual research source."""
    tool: str
    title: str
    url: str
    content: str
    score: float = 0.0
    published_date: Optional[str] = None
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash and self.content:
            self.content_hash = hashlib.md5(self.content[:500].encode()).hexdigest()


@dataclass
class SynthesizedResult:
    """Result from Queen synthesis."""
    summary: str
    key_findings: list[str]
    patterns: list[str]
    discrepancies: list[str]
    sources: list[ResearchSource]
    confidence: float = 0.0
    consensus: Optional[dict] = None


@dataclass
class MemoryStorageResult:
    """Result of memory storage operation."""
    memory_key: str
    layers: dict[str, Any]
    success: bool = True


@dataclass
class RecalledResearch:
    """Recalled research from memory."""
    content: str
    source: str  # "letta", "mem0", "qdrant"
    score: float
    memory_key: Optional[str] = None


@dataclass
class UltimateResearchResult:
    """Final result from the Ultimate Research Swarm."""
    query: str
    depth: ResearchDepth

    # Synthesized content (fits in Claude's context)
    summary: str
    key_findings: list[str]
    patterns: list[str]
    discrepancies: list[str]

    # Sources and citations
    sources: list[ResearchSource]
    tools_used: list[str]
    agents_spawned: int

    # Metadata
    confidence: float = 0.0
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Memory integration
    stored_in_memory: bool = False
    memory_key: Optional[str] = None
    recalled_from_memory: bool = False

    # Consensus (for deep research)
    consensus_votes: Optional[dict] = None


# =============================================================================
# Research Agent Configurations
# =============================================================================

RESEARCH_AGENT_CONFIGS = {
    ResearchAgentType.EXA_NEURAL: {
        "adapter_class": "ExaAdapter",
        "priority": 25,
        "latency_target_ms": 350,
        "operations": ["search", "search_and_contents", "find_similar"],
        "search_type": "auto",
        "use_for": ["code_patterns", "technical_docs", "similar_content"],
    },
    ResearchAgentType.TAVILY_AI: {
        "adapter_class": "TavilyAdapter",
        "priority": 24,
        "latency_target_ms": 2000,
        "operations": ["search", "research", "extract", "qna"],
        "search_depth": "basic",
        "use_for": ["real_time_info", "research_reports", "structured_data"],
    },
    ResearchAgentType.JINA_READER: {
        "adapter_class": "JinaAdapter",
        "priority": 23,
        "latency_target_ms": 3000,
        "operations": ["read", "search", "rerank"],
        "use_for": ["full_docs", "pdf_content", "url_conversion"],
    },
    ResearchAgentType.PERPLEXITY_DEEP: {
        "adapter_class": "PerplexityAdapter",
        "priority": 22,
        "latency_target_ms": 30000,
        "operations": ["research", "chat", "pro"],
        "use_for": ["complex_topics", "multi_step_research", "synthesis"],
    },
}


# =============================================================================
# Synthesis Queen
# =============================================================================

class SynthesisQueen:
    """
    Queen agent that synthesizes large research results into condensed summaries.

    The Queen:
    1. Receives all research results from worker agents (100KB+)
    2. Deduplicates content using content hashing
    3. Identifies patterns across sources
    4. Detects discrepancies (conflicting information)
    5. Produces condensed summary (2-4KB) that fits in Claude's context
    6. Triggers hive-mind consensus for complex decisions
    """

    def __init__(self, config: ResearchSwarmConfig):
        self._config = config
        self._max_summary_chars = config.synthesis_max_kb * 1024

    def synthesize(
        self,
        sources: list[ResearchSource],
        query: str,
    ) -> SynthesizedResult:
        """
        Synthesize research results into condensed summary.

        Args:
            sources: List of research sources from agents
            query: Original query for context

        Returns:
            SynthesizedResult with condensed summary and analysis
        """
        if not sources:
            return SynthesizedResult(
                summary="No sources found for this query.",
                key_findings=[],
                patterns=[],
                discrepancies=[],
                sources=[],
                confidence=0.0,
            )

        # 1. Deduplicate by content hash
        unique_sources = self._deduplicate(sources)

        # 2. Sort by score
        unique_sources.sort(key=lambda x: x.score, reverse=True)

        # 3. Extract key findings (top 5 unique insights)
        key_findings = self._extract_findings(unique_sources)

        # 4. Identify patterns
        patterns = self._identify_patterns(unique_sources)

        # 5. Detect discrepancies
        discrepancies = self._detect_discrepancies(unique_sources)

        # 6. Build condensed summary
        summary = self._build_summary(unique_sources)

        # 7. Calculate confidence
        tools = set(s.tool for s in unique_sources)
        confidence = self._calculate_confidence(unique_sources, tools)

        return SynthesizedResult(
            summary=summary,
            key_findings=key_findings,
            patterns=patterns,
            discrepancies=discrepancies,
            sources=unique_sources,
            confidence=confidence,
        )

    def _deduplicate(self, sources: list[ResearchSource]) -> list[ResearchSource]:
        """Deduplicate sources by content hash."""
        seen_hashes = set()
        unique = []

        for source in sources:
            if source.content_hash not in seen_hashes:
                seen_hashes.add(source.content_hash)
                unique.append(source)

        return unique

    def _extract_findings(self, sources: list[ResearchSource]) -> list[str]:
        """Extract key findings from top sources."""
        findings = []
        seen_content = set()

        for source in sources[:10]:
            if not source.content:
                continue

            # Simple dedup by first 100 chars
            content_key = source.content[:100].lower()
            if content_key in seen_content:
                continue
            seen_content.add(content_key)

            # Extract first meaningful sentence
            sentences = source.content.split(". ")
            if sentences:
                finding = sentences[0].strip()
                if len(finding) > 20:  # Skip tiny snippets
                    findings.append(f"[{source.tool}] {finding}")

        return findings[:5]

    def _identify_patterns(self, sources: list[ResearchSource]) -> list[str]:
        """Identify common patterns across sources."""
        patterns = []

        # Count tool occurrences
        tool_counts = {}
        for source in sources:
            tool_counts[source.tool] = tool_counts.get(source.tool, 0) + 1

        # Pattern: Multiple sources from same tool
        for tool, count in tool_counts.items():
            if count >= 3:
                patterns.append(f"Strong coverage from {tool} ({count} sources)")

        # Pattern: High agreement (multiple high-score sources)
        high_score = [s for s in sources if s.score >= 0.8]
        if len(high_score) >= 3:
            patterns.append(f"High confidence: {len(high_score)} sources with score >= 0.8")

        return patterns

    def _detect_discrepancies(self, sources: list[ResearchSource]) -> list[str]:
        """Detect conflicting information across sources."""
        discrepancies = []

        # Simple heuristic: look for contradictory keywords
        contradictory_pairs = [
            ("yes", "no"),
            ("true", "false"),
            ("correct", "incorrect"),
            ("recommended", "not recommended"),
            ("deprecated", "current"),
        ]

        # Check each pair of sources for contradictions
        for i, s1 in enumerate(sources[:5]):
            for s2 in sources[i+1:6]:
                if s1.tool != s2.tool:  # Different tools
                    for w1, w2 in contradictory_pairs:
                        c1 = s1.content.lower()
                        c2 = s2.content.lower()
                        if (w1 in c1 and w2 in c2) or (w2 in c1 and w1 in c2):
                            discrepancies.append(
                                f"Potential conflict: {s1.tool} vs {s2.tool} on '{w1}/{w2}'"
                            )
                            break

        return discrepancies[:3]

    def _build_summary(self, sources: list[ResearchSource]) -> str:
        """Build condensed summary from top sources."""
        summary_parts = []
        total_chars = 0

        for source in sources[:5]:
            if not source.content:
                continue

            # Take portion of content
            available = self._max_summary_chars - total_chars
            if available <= 0:
                break

            chunk = source.content[:min(800, available)]
            summary_parts.append(f"**{source.title}** ({source.tool})\n{chunk}")
            total_chars += len(chunk) + len(source.title) + 20

        return "\n\n---\n\n".join(summary_parts) if summary_parts else "No content available."

    def _calculate_confidence(
        self,
        sources: list[ResearchSource],
        tools: set[str]
    ) -> float:
        """Calculate overall confidence based on source agreement."""
        # Base confidence from number of tools
        tool_bonus = len(tools) * 0.15

        # Source count bonus
        source_bonus = min(len(sources), 10) * 0.03

        # Average score bonus
        if sources:
            avg_score = sum(s.score for s in sources) / len(sources)
            score_bonus = avg_score * 0.2
        else:
            score_bonus = 0

        return min(0.95, 0.4 + tool_bonus + source_bonus + score_bonus)


# =============================================================================
# Research Memory Manager
# =============================================================================

class ResearchMemoryManager:
    """
    Manages research findings across Letta, Mem0, and Qdrant.

    Storage strategy:
    - Letta: Full archival of research (long-term, cross-session)
    - Mem0: Key findings with session context (universal layer)
    - Qdrant: Vector embeddings for semantic search (fast recall)
    """

    def __init__(self):
        self._letta: Optional[LettaAdapter] = None
        self._mem0: Optional[Mem0Adapter] = None
        self._initialized = False

    async def initialize(self) -> dict[str, bool]:
        """Initialize memory backends."""
        status = {"letta": False, "mem0": False}

        if LettaAdapter:
            try:
                self._letta = LettaAdapter()
                result = await self._letta.initialize({})
                status["letta"] = result.success
            except Exception:
                pass

        if Mem0Adapter:
            try:
                self._mem0 = Mem0Adapter()
                result = await self._mem0.initialize({})
                status["mem0"] = result.success
            except Exception:
                pass

        self._initialized = True
        return status

    async def store_research(
        self,
        result: SynthesizedResult,
        memory_key: str,
        user_id: Optional[str] = None,
    ) -> MemoryStorageResult:
        """
        Store research findings across all memory layers.

        Args:
            result: Synthesized research result
            memory_key: Key for memory storage
            user_id: Optional user ID for Mem0

        Returns:
            MemoryStorageResult with storage status per layer
        """
        if not self._initialized:
            await self.initialize()

        layers = {}

        # 1. Store in Letta (archival)
        if self._letta:
            try:
                letta_result = await self._letta.execute(
                    "add_memory",
                    content=result.summary,
                    metadata={
                        "memory_key": memory_key,
                        "type": "research_summary",
                        "sources": len(result.sources),
                        "patterns": result.patterns,
                        "confidence": result.confidence,
                    }
                )
                layers["letta"] = {"success": letta_result.success}
            except Exception as e:
                layers["letta"] = {"success": False, "error": str(e)}

        # 2. Store in Mem0 (universal)
        if self._mem0:
            try:
                findings_text = "\n".join(result.key_findings)
                mem0_result = await self._mem0.execute(
                    "add",
                    content=findings_text,
                    user_id=user_id or "research_swarm",
                    metadata={
                        "memory_key": memory_key,
                        "type": "research_findings",
                    }
                )
                layers["mem0"] = {"success": mem0_result.success}
            except Exception as e:
                layers["mem0"] = {"success": False, "error": str(e)}

        return MemoryStorageResult(
            memory_key=memory_key,
            layers=layers,
            success=any(l.get("success") for l in layers.values()),
        )

    async def recall_research(
        self,
        query: str,
        limit: int = 5,
    ) -> list[RecalledResearch]:
        """
        Recall relevant research from memory.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of recalled research findings
        """
        if not self._initialized:
            await self.initialize()

        results = []

        # 1. Search Mem0 (fast)
        if self._mem0:
            try:
                mem0_result = await self._mem0.execute(
                    "search",
                    query=query,
                    limit=limit,
                )
                if mem0_result.success:
                    for m in mem0_result.data.get("memories", []):
                        results.append(RecalledResearch(
                            content=m.get("content", ""),
                            source="mem0",
                            score=m.get("score", 0.5),
                            memory_key=m.get("metadata", {}).get("memory_key"),
                        ))
            except Exception:
                pass

        # 2. Search Letta (archival) if needed
        if len(results) < limit and self._letta:
            try:
                letta_result = await self._letta.execute(
                    "search",
                    query=query,
                    top_k=limit - len(results),
                )
                if letta_result.success:
                    for r in letta_result.data.get("results", []):
                        results.append(RecalledResearch(
                            content=r.get("content", ""),
                            source="letta",
                            score=r.get("score", 0.5),
                            memory_key=r.get("metadata", {}).get("memory_key"),
                        ))
            except Exception:
                pass

        return results

    async def shutdown(self) -> None:
        """Shutdown memory backends."""
        if self._letta:
            await self._letta.shutdown()
        if self._mem0:
            await self._mem0.shutdown()


# =============================================================================
# Ultimate Research Swarm
# =============================================================================

class UltimateResearchSwarm:
    """
    Ultimate Research Swarm - The crown jewel of UNLEASH.

    Combines Claude Flow V3 multi-agent orchestration with research
    infrastructure to enable Claude to research at scale.

    Key Features:
    - Spawns specialized research agents (Exa, Tavily, Jina, Perplexity)
    - Parallel execution across all tools
    - Queen synthesis: 100KB+ → 2-4KB summary
    - Memory integration: Store and recall findings
    - Hive-mind consensus for complex decisions
    """

    def __init__(self, config: Optional[ResearchSwarmConfig] = None):
        self._config = config or ResearchSwarmConfig()

        # Research adapters
        self._exa: Optional[ExaAdapter] = None
        self._tavily: Optional[TavilyAdapter] = None
        self._jina: Optional[JinaAdapter] = None
        self._perplexity: Optional[PerplexityAdapter] = None

        # Synthesis and memory
        self._queen = SynthesisQueen(self._config)
        self._memory = ResearchMemoryManager()

        # State
        self._initialized = False
        self._adapter_status: dict[str, bool] = {}

        # Stats
        self._stats = {
            "total_queries": 0,
            "quick_queries": 0,
            "comprehensive_queries": 0,
            "memory_hits": 0,
            "agents_spawned": 0,
        }

    async def initialize(self) -> dict[str, bool]:
        """
        Initialize all research adapters and memory.

        Returns dict showing which components are available.
        """
        status = {}

        # Initialize research adapters
        if ExaAdapter:
            self._exa = ExaAdapter()
            result = await self._exa.initialize({})
            status["exa"] = result.success

        if TavilyAdapter:
            self._tavily = TavilyAdapter()
            result = await self._tavily.initialize({})
            status["tavily"] = result.success

        if JinaAdapter:
            self._jina = JinaAdapter()
            result = await self._jina.initialize({})
            status["jina"] = result.success

        if PerplexityAdapter:
            self._perplexity = PerplexityAdapter()
            result = await self._perplexity.initialize({})
            status["perplexity"] = result.success

        # Initialize memory
        memory_status = await self._memory.initialize()
        status.update({f"memory_{k}": v for k, v in memory_status.items()})

        self._adapter_status = status
        self._initialized = True

        return status

    async def research(
        self,
        query: str,
        depth: ResearchDepth = ResearchDepth.STANDARD,
        include_urls: Optional[list[str]] = None,
        memory_key: Optional[str] = None,
        check_memory_first: bool = True,
    ) -> UltimateResearchResult:
        """
        Execute ultimate research with full agent orchestration.

        Args:
            query: Research query
            depth: Research depth (quick/standard/comprehensive/deep)
            include_urls: Specific URLs to include in research
            memory_key: Key for memory storage (optional)
            check_memory_first: Check memory before searching

        Returns:
            UltimateResearchResult with synthesized findings
        """
        self._stats["total_queries"] += 1
        start_time = time.time()

        if not self._initialized:
            await self.initialize()

        # 1. Check memory first (if enabled)
        if check_memory_first:
            recalled = await self._memory.recall_research(query, limit=3)
            if recalled and recalled[0].score > 0.85:
                self._stats["memory_hits"] += 1
                return UltimateResearchResult(
                    query=query,
                    depth=depth,
                    summary=recalled[0].content,
                    key_findings=[r.content[:200] for r in recalled[:3]],
                    patterns=["Recalled from memory"],
                    discrepancies=[],
                    sources=[],
                    tools_used=["memory"],
                    agents_spawned=0,
                    confidence=recalled[0].score,
                    latency_ms=(time.time() - start_time) * 1000,
                    recalled_from_memory=True,
                    memory_key=recalled[0].memory_key,
                )

        # 2. Select agents based on depth
        tasks = []
        agents_spawned = 0

        if depth == ResearchDepth.QUICK:
            self._stats["quick_queries"] += 1
            # Single best tool
            if self._tavily:
                tasks.append(self._search_tavily(query, max_results=5))
                agents_spawned += 1
            elif self._exa:
                tasks.append(self._search_exa(query, search_type="fast"))
                agents_spawned += 1

        elif depth == ResearchDepth.STANDARD:
            # 2-3 tools in parallel
            if self._exa:
                tasks.append(self._search_exa(query))
                agents_spawned += 1
            if self._tavily:
                tasks.append(self._search_tavily(query))
                agents_spawned += 1

        elif depth in [ResearchDepth.COMPREHENSIVE, ResearchDepth.DEEP]:
            self._stats["comprehensive_queries"] += 1
            # All tools
            if self._exa:
                search_type = "deep" if depth == ResearchDepth.DEEP else "auto"
                tasks.append(self._search_exa(query, search_type=search_type))
                agents_spawned += 1
            if self._tavily:
                search_depth = "advanced" if depth == ResearchDepth.DEEP else "basic"
                tasks.append(self._search_tavily(query, search_depth=search_depth))
                agents_spawned += 1
            if self._jina:
                tasks.append(self._search_jina(query))
                agents_spawned += 1
            if self._perplexity and depth == ResearchDepth.DEEP:
                tasks.append(self._search_perplexity(query))
                agents_spawned += 1

        # 3. Include specific URLs
        if include_urls and self._jina:
            for url in include_urls[:3]:
                tasks.append(self._read_url(url))

        self._stats["agents_spawned"] += agents_spawned

        # 4. Execute all searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 5. Collect sources
        all_sources: list[ResearchSource] = []
        tools_used: set[str] = set()

        for result in results:
            if isinstance(result, Exception):
                continue
            if isinstance(result, list):
                all_sources.extend(result)
                if result:
                    tools_used.add(result[0].tool)

        # 6. Queen synthesis
        synthesis = self._queen.synthesize(all_sources, query)

        # 7. Store in memory (if key provided)
        stored = False
        if memory_key:
            storage_result = await self._memory.store_research(
                synthesis, memory_key
            )
            stored = storage_result.success

        # 8. Build final result
        return UltimateResearchResult(
            query=query,
            depth=depth,
            summary=synthesis.summary,
            key_findings=synthesis.key_findings,
            patterns=synthesis.patterns,
            discrepancies=synthesis.discrepancies,
            sources=synthesis.sources,
            tools_used=list(tools_used),
            agents_spawned=agents_spawned,
            confidence=synthesis.confidence,
            latency_ms=(time.time() - start_time) * 1000,
            stored_in_memory=stored,
            memory_key=memory_key,
            recalled_from_memory=False,
            consensus_votes=synthesis.consensus,
        )

    # =========================================================================
    # Search Implementations
    # =========================================================================

    async def _search_exa(
        self,
        query: str,
        search_type: str = "auto",
        max_results: int = 10,
    ) -> list[ResearchSource]:
        """Search using Exa neural search."""
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
                content=r.get("text", "")[:2000],
                score=r.get("score", 0.0),
                published_date=r.get("published_date"),
            ))

        return sources

    async def _search_tavily(
        self,
        query: str,
        search_depth: str = "basic",
        max_results: int = 10,
    ) -> list[ResearchSource]:
        """Search using Tavily AI search."""
        if not self._tavily:
            return []

        result = await self._tavily.execute(
            "search",
            query=query,
            search_depth=search_depth,
            max_results=max_results,
        )

        if not result.success:
            return []

        sources = []

        # Include AI answer as first source
        answer = result.data.get("answer")
        if answer:
            sources.append(ResearchSource(
                tool="tavily",
                title="Tavily AI Answer",
                url="",
                content=answer[:2000],
                score=1.0,
            ))

        for r in result.data.get("results", []):
            sources.append(ResearchSource(
                tool="tavily",
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", "")[:2000],
                score=r.get("score", 0.0),
                published_date=r.get("published_date"),
            ))

        return sources

    async def _search_jina(self, query: str) -> list[ResearchSource]:
        """Search using Jina."""
        if not self._jina:
            return []

        result = await self._jina.execute("search", query=query)

        if not result.success:
            return []

        return [ResearchSource(
            tool="jina",
            title=f"Jina Search: {query}",
            url="",
            content=result.data.get("content", "")[:2000],
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
                content=content[:2000],
                score=0.95,
            ))

        # Citations
        for citation in result.data.get("citations", [])[:5]:
            sources.append(ResearchSource(
                tool="perplexity",
                title=citation.get("title", "Citation"),
                url=citation.get("url", ""),
                content="",
                score=0.8,
            ))

        return sources

    async def _read_url(self, url: str) -> list[ResearchSource]:
        """Read specific URL using Jina."""
        if not self._jina:
            return []

        result = await self._jina.execute("read", url=url)

        if not result.success:
            return []

        return [ResearchSource(
            tool="jina",
            title=f"URL: {url}",
            url=url,
            content=result.data.get("content", "")[:2000],
            score=0.9,
        )]

    # =========================================================================
    # Utility
    # =========================================================================

    async def shutdown(self) -> None:
        """Cleanup all resources."""
        if self._exa:
            await self._exa.shutdown()
        if self._tavily:
            await self._tavily.shutdown()
        if self._jina:
            await self._jina.shutdown()
        if self._perplexity:
            await self._perplexity.shutdown()
        await self._memory.shutdown()

    def get_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return dict(self._stats)

    def get_adapter_status(self) -> dict[str, bool]:
        """Get adapter availability status."""
        return dict(self._adapter_status)


# =============================================================================
# Factory
# =============================================================================

_swarm: Optional[UltimateResearchSwarm] = None


def get_ultimate_swarm() -> UltimateResearchSwarm:
    """Get or create the global UltimateResearchSwarm instance."""
    global _swarm
    if _swarm is None:
        _swarm = UltimateResearchSwarm()
    return _swarm


async def quick_research(query: str) -> UltimateResearchResult:
    """Quick research helper."""
    swarm = get_ultimate_swarm()
    await swarm.initialize()
    return await swarm.research(query, ResearchDepth.QUICK)


async def comprehensive_research(
    query: str,
    memory_key: Optional[str] = None,
) -> UltimateResearchResult:
    """Comprehensive research helper."""
    swarm = get_ultimate_swarm()
    await swarm.initialize()
    return await swarm.research(
        query,
        ResearchDepth.COMPREHENSIVE,
        memory_key=memory_key,
    )


async def deep_research(
    query: str,
    memory_key: Optional[str] = None,
) -> UltimateResearchResult:
    """Deep research helper."""
    swarm = get_ultimate_swarm()
    await swarm.initialize()
    return await swarm.research(
        query,
        ResearchDepth.DEEP,
        memory_key=memory_key,
    )


# =============================================================================
# CLI
# =============================================================================

async def main():
    """CLI for testing the Ultimate Research Swarm."""
    import argparse

    parser = argparse.ArgumentParser(description="Ultimate Research Swarm CLI")
    parser.add_argument("query", nargs="?", help="Research query")
    parser.add_argument("--depth", choices=["quick", "standard", "comprehensive", "deep"],
                        default="standard", help="Research depth")
    parser.add_argument("--memory-key", help="Memory key for storage")
    parser.add_argument("--status", action="store_true", help="Show swarm status")

    args = parser.parse_args()

    swarm = UltimateResearchSwarm()
    status = await swarm.initialize()

    if args.status:
        print("\n" + "=" * 60)
        print("  ULTIMATE RESEARCH SWARM STATUS")
        print("=" * 60)
        for component, available in status.items():
            icon = "✓" if available else "✗"
            print(f"  {icon} {component}: {'Ready' if available else 'Unavailable'}")
        print("=" * 60)
        return

    if not args.query:
        print("Usage: ultimate_research_swarm.py <query> [--depth DEPTH] [--memory-key KEY]")
        return

    result = await swarm.research(
        args.query,
        ResearchDepth(args.depth),
        memory_key=args.memory_key,
    )

    # Print result
    print("\n" + "=" * 70)
    print(f"  ULTIMATE RESEARCH: {result.query}")
    print("=" * 70)
    print(f"Depth: {result.depth.value}")
    print(f"Agents: {result.agents_spawned}")
    print(f"Tools: {', '.join(result.tools_used)}")
    print(f"Sources: {len(result.sources)}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Latency: {result.latency_ms:.0f}ms")
    if result.memory_key:
        print(f"Memory Key: {result.memory_key}")
    print("-" * 70)
    print("KEY FINDINGS:")
    for finding in result.key_findings:
        print(f"  • {finding[:100]}...")
    if result.patterns:
        print("-" * 70)
        print("PATTERNS:")
        for pattern in result.patterns:
            print(f"  📊 {pattern}")
    if result.discrepancies:
        print("-" * 70)
        print("DISCREPANCIES:")
        for disc in result.discrepancies:
            print(f"  ⚠️ {disc}")
    print("-" * 70)
    print("SUMMARY:")
    print(result.summary[:1500] + "..." if len(result.summary) > 1500 else result.summary)
    print("=" * 70)

    await swarm.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
