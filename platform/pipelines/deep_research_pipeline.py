"""
Deep Research Pipeline for Unleashed Platform

Combines multiple SDKs for comprehensive research:
- Exa AI: Semantic search (96% accuracy on FRAMES benchmark)
- Firecrawl: LLM-driven extraction
- Crawl4AI: Deep crawling
- Mem0: Memory and caching
- llm-reasoners: Reasoning and synthesis

This pipeline implements the full research flow:
Query → Search → Extract → Reason → Synthesize → Remember
"""

import os
import asyncio
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Check component availability
PIPELINE_AVAILABLE = True
_missing_components = []

# Exa
try:
    from exa_py import Exa
    EXA_AVAILABLE = True
except ImportError:
    EXA_AVAILABLE = False
    _missing_components.append("exa_py")

# Firecrawl
try:
    from firecrawl import FirecrawlApp
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False
    _missing_components.append("firecrawl")

# Crawl4AI
try:
    from crawl4ai import AsyncWebCrawler
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    _missing_components.append("crawl4ai")

# Update pipeline availability
if not EXA_AVAILABLE:
    PIPELINE_AVAILABLE = False

# Register pipeline
from . import register_pipeline
register_pipeline(
    "deep_research",
    PIPELINE_AVAILABLE,
    dependencies=["exa", "firecrawl", "crawl4ai", "mem0", "llm_reasoners"]
)


class ResearchDepth(Enum):
    """Research depth levels."""
    QUICK = "quick"       # Search only
    STANDARD = "standard"  # Search + Extract
    DEEP = "deep"         # Search + Extract + Deep Crawl
    COMPREHENSIVE = "comprehensive"  # All + Reasoning


class ResearchStrategy(Enum):
    """Research strategy types."""
    SEMANTIC = "semantic"   # Neural/semantic search
    KEYWORD = "keyword"     # Keyword-based search
    HYBRID = "hybrid"       # Combined approach


@dataclass
class Source:
    """A research source."""
    url: str
    title: str
    content: str
    relevance_score: float
    extracted_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchResult:
    """Result from research pipeline."""
    query: str
    sources: List[Source]
    synthesis: str
    confidence: float
    reasoning_path: Optional[List[Dict[str, Any]]] = None
    total_sources: int = 0
    depth: ResearchDepth = ResearchDepth.STANDARD
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Configuration for research pipeline."""
    # Search
    max_search_results: int = 10
    search_strategy: ResearchStrategy = ResearchStrategy.SEMANTIC

    # Extraction
    max_extract_urls: int = 5
    extract_formats: List[str] = field(default_factory=lambda: ["markdown"])

    # Deep crawl
    max_crawl_depth: int = 2
    max_crawl_urls: int = 3

    # Reasoning
    enable_reasoning: bool = True
    reasoning_algorithm: str = "graph_of_thoughts"

    # Memory
    enable_memory: bool = True
    cache_results: bool = True

    # Performance
    timeout_seconds: int = 60
    parallel_requests: int = 5


class DeepResearchPipeline:
    """
    Comprehensive research pipeline combining Exa, Firecrawl, Crawl4AI,
    memory, and reasoning capabilities.

    Flow:
    1. Search (Exa): Find relevant sources via semantic search
    2. Extract (Firecrawl): Extract structured content from top sources
    3. Deep Crawl (Crawl4AI): Optionally crawl linked pages
    4. Reason (llm-reasoners): Apply reasoning to synthesize findings
    5. Remember (Mem0): Cache results and update memory

    Example:
        pipeline = DeepResearchPipeline()
        result = await pipeline.research(
            query="Best practices for distributed systems",
            depth=ResearchDepth.COMPREHENSIVE,
        )
        print(result.synthesis)
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        exa_api_key: Optional[str] = None,
        firecrawl_api_key: Optional[str] = None,
    ):
        """
        Initialize the deep research pipeline.

        Args:
            config: Pipeline configuration
            exa_api_key: Exa API key (or from EXA_API_KEY env var)
            firecrawl_api_key: Firecrawl API key (or from FIRECRAWL_API_KEY env var)
        """
        self.config = config or PipelineConfig()

        # Initialize Exa
        self._exa = None
        if EXA_AVAILABLE:
            api_key = exa_api_key or os.getenv("EXA_API_KEY")
            if api_key:
                self._exa = Exa(api_key=api_key)

        # Initialize Firecrawl
        self._firecrawl = None
        if FIRECRAWL_AVAILABLE:
            api_key = firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")
            if api_key:
                self._firecrawl = FirecrawlApp(api_key=api_key)

        # Crawl4AI (no API key needed)
        self._crawler_available = CRAWL4AI_AVAILABLE

        # Initialize adapters
        self._memory = None
        self._reasoner = None

        self._try_init_adapters()

    def _try_init_adapters(self):
        """Try to initialize memory and reasoning adapters."""
        # Memory adapter
        try:
            from ..adapters.mem0_adapter import Mem0Adapter, MEM0_AVAILABLE
            if MEM0_AVAILABLE:
                self._memory = Mem0Adapter()
                self._memory.initialize()
        except ImportError:
            pass

        # Reasoning adapter
        try:
            from ..adapters.llm_reasoners_adapter import (
                LLMReasonersAdapter,
                LLM_REASONERS_AVAILABLE,
            )
            if LLM_REASONERS_AVAILABLE:
                self._reasoner = LLMReasonersAdapter()
        except ImportError:
            pass

    async def research(
        self,
        query: str,
        depth: ResearchDepth = ResearchDepth.STANDARD,
        question: Optional[str] = None,
        strategy: Optional[ResearchStrategy] = None,
    ) -> ResearchResult:
        """
        Execute the full research pipeline.

        Args:
            query: Search query
            depth: Research depth level
            question: Optional question to answer (triggers reasoning)
            strategy: Search strategy override

        Returns:
            ResearchResult with sources, synthesis, and metadata
        """
        import time
        start_time = time.time()

        strategy = strategy or self.config.search_strategy
        sources: List[Source] = []
        synthesis = ""
        reasoning_path = None
        confidence = 0.0

        # Stage 1: Search
        search_results = await self._search(query, strategy)
        sources.extend(search_results)

        # Stage 2: Extract (if STANDARD or deeper)
        if depth in [ResearchDepth.STANDARD, ResearchDepth.DEEP, ResearchDepth.COMPREHENSIVE]:
            extract_results = await self._extract(sources[:self.config.max_extract_urls])
            # Update sources with extracted content
            for i, extracted in enumerate(extract_results):
                if i < len(sources):
                    sources[i].content = extracted.get("content", sources[i].content)
                    sources[i].metadata.update(extracted.get("metadata", {}))

        # Stage 3: Deep Crawl (if DEEP or COMPREHENSIVE)
        if depth in [ResearchDepth.DEEP, ResearchDepth.COMPREHENSIVE] and self._crawler_available:
            crawl_results = await self._deep_crawl(sources[:self.config.max_crawl_urls])
            # Add crawled pages as additional sources
            for crawled in crawl_results:
                sources.append(Source(
                    url=crawled.get("url", ""),
                    title=crawled.get("title", "Crawled Page"),
                    content=crawled.get("content", ""),
                    relevance_score=0.5,  # Lower score for crawled pages
                    metadata={"source": "deep_crawl"},
                ))

        # Stage 4: Reasoning (if COMPREHENSIVE or question provided)
        if (depth == ResearchDepth.COMPREHENSIVE or question) and self.config.enable_reasoning:
            reasoning_result = await self._reason(
                query=query,
                question=question or query,
                sources=sources,
            )
            synthesis = reasoning_result.get("synthesis", "")
            reasoning_path = reasoning_result.get("reasoning_path")
            confidence = reasoning_result.get("confidence", 0.0)
        else:
            # Simple synthesis without reasoning
            synthesis = self._simple_synthesis(sources)
            confidence = 0.7 if sources else 0.0

        # Stage 5: Remember (cache results)
        if self.config.enable_memory and self._memory:
            await self._remember(query, sources, synthesis)

        execution_time = time.time() - start_time

        return ResearchResult(
            query=query,
            sources=sources,
            synthesis=synthesis,
            confidence=confidence,
            reasoning_path=reasoning_path,
            total_sources=len(sources),
            depth=depth,
            execution_time=execution_time,
            metadata={
                "search_strategy": strategy.value,
                "exa_available": self._exa is not None,
                "firecrawl_available": self._firecrawl is not None,
                "crawler_available": self._crawler_available,
            },
        )

    async def _search(
        self,
        query: str,
        strategy: ResearchStrategy,
    ) -> List[Source]:
        """Execute semantic search using Exa."""
        sources = []

        if not self._exa:
            return sources

        try:
            # Determine search type
            search_type = "neural" if strategy == ResearchStrategy.SEMANTIC else "keyword"
            if strategy == ResearchStrategy.HYBRID:
                search_type = "auto"

            # Execute search
            results = self._exa.search_and_contents(
                query=query,
                type=search_type,
                num_results=self.config.max_search_results,
                text=True,
            )

            # Convert to Source objects
            for result in results.results:
                sources.append(Source(
                    url=result.url,
                    title=result.title or "Untitled",
                    content=result.text or "",
                    relevance_score=result.score if hasattr(result, 'score') else 0.8,
                    metadata={
                        "published_date": getattr(result, 'published_date', None),
                        "author": getattr(result, 'author', None),
                    },
                ))

        except Exception as e:
            # Log error but continue
            print(f"Search error: {e}")

        return sources

    async def _extract(
        self,
        sources: List[Source],
    ) -> List[Dict[str, Any]]:
        """Extract content using Firecrawl."""
        extracted = []

        if not self._firecrawl or not sources:
            return extracted

        try:
            urls = [s.url for s in sources]

            # Batch scrape
            results = self._firecrawl.batch_scrape_urls(
                urls=urls,
                params={
                    "formats": self.config.extract_formats,
                },
            )

            # Process results
            for result in results.get("data", []):
                extracted.append({
                    "url": result.get("metadata", {}).get("url", ""),
                    "content": result.get("markdown", result.get("content", "")),
                    "metadata": result.get("metadata", {}),
                })

        except Exception as e:
            print(f"Extraction error: {e}")

        return extracted

    async def _deep_crawl(
        self,
        sources: List[Source],
    ) -> List[Dict[str, Any]]:
        """Deep crawl using Crawl4AI."""
        crawled = []

        if not CRAWL4AI_AVAILABLE or not sources:
            return crawled

        try:
            async with AsyncWebCrawler() as crawler:
                for source in sources[:self.config.max_crawl_urls]:
                    result = await crawler.arun(
                        url=source.url,
                        max_depth=self.config.max_crawl_depth,
                    )

                    if result.success:
                        crawled.append({
                            "url": source.url,
                            "title": result.metadata.get("title", ""),
                            "content": result.markdown or result.cleaned_html or "",
                            "links": result.links[:10] if hasattr(result, 'links') else [],
                        })

        except Exception as e:
            print(f"Deep crawl error: {e}")

        return crawled

    async def _reason(
        self,
        query: str,
        question: str,
        sources: List[Source],
    ) -> Dict[str, Any]:
        """Apply reasoning to synthesize findings."""
        if not self._reasoner:
            return {"synthesis": self._simple_synthesis(sources), "confidence": 0.6}

        try:
            # Prepare context from sources
            context = "\n\n".join([
                f"Source: {s.title}\nURL: {s.url}\nContent: {s.content[:500]}..."
                for s in sources[:5]
            ])

            # Format problem for reasoning
            problem = f"""
Based on the following research sources, answer the question.

Question: {question}

Research Context:
{context}

Provide a comprehensive synthesis of the findings.
"""

            # Execute reasoning
            result = await self._reasoner.reason(
                problem=problem,
                algorithm=None,  # Use default
                max_depth=5,
            )

            return {
                "synthesis": result.answer,
                "confidence": result.confidence,
                "reasoning_path": [
                    {"id": n.id, "content": n.content, "score": n.score}
                    for n in result.reasoning_path[:10]
                ],
            }

        except Exception as e:
            print(f"Reasoning error: {e}")
            return {"synthesis": self._simple_synthesis(sources), "confidence": 0.5}

    def _simple_synthesis(self, sources: List[Source]) -> str:
        """Create a simple synthesis without reasoning."""
        if not sources:
            return "No sources found for the query."

        synthesis_parts = [
            f"Based on {len(sources)} sources:",
            "",
        ]

        for i, source in enumerate(sources[:5], 1):
            summary = source.content[:200] if source.content else "No content available"
            synthesis_parts.append(f"{i}. {source.title}: {summary}...")

        return "\n".join(synthesis_parts)

    async def _remember(
        self,
        query: str,
        sources: List[Source],
        synthesis: str,
    ):
        """Cache results in memory."""
        if not self._memory:
            return

        try:
            # Store the research result
            self._memory.add(
                content=f"Research on '{query}': {synthesis[:500]}",
                metadata={
                    "type": "research",
                    "query": query,
                    "num_sources": len(sources),
                    "timestamp": datetime.now().isoformat(),
                },
            )
        except Exception as e:
            print(f"Memory error: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline component status."""
        return {
            "exa": self._exa is not None,
            "firecrawl": self._firecrawl is not None,
            "crawler": self._crawler_available,
            "memory": self._memory is not None,
            "reasoner": self._reasoner is not None,
            "config": {
                "max_search_results": self.config.max_search_results,
                "max_extract_urls": self.config.max_extract_urls,
                "max_crawl_depth": self.config.max_crawl_depth,
                "enable_reasoning": self.config.enable_reasoning,
                "enable_memory": self.config.enable_memory,
            },
        }


# Convenience function
async def quick_research(query: str, depth: str = "standard") -> ResearchResult:
    """
    Quick research helper function.

    Args:
        query: Search query
        depth: Research depth (quick/standard/deep/comprehensive)

    Returns:
        ResearchResult
    """
    depth_map = {
        "quick": ResearchDepth.QUICK,
        "standard": ResearchDepth.STANDARD,
        "deep": ResearchDepth.DEEP,
        "comprehensive": ResearchDepth.COMPREHENSIVE,
    }

    pipeline = DeepResearchPipeline()
    return await pipeline.research(query, depth=depth_map.get(depth, ResearchDepth.STANDARD))
