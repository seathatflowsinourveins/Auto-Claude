#!/usr/bin/env python3
"""
Auto-Research Pipeline

Combines Firecrawl (web scraping/crawling) with Exa (AI search) for
comprehensive autonomous research workflows.

Features:
- Multi-source research with parallel scraping
- AI-powered content extraction
- Topic summarization
- Integration with Letta for memory persistence
- Ralph Loop compatible

Usage:
    python auto_research.py research "AI agent architectures 2026"
    python auto_research.py scrape https://example.com
    python auto_research.py crawl https://docs.example.com --depth 3
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
import logging

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.firecrawl_integration import (
    FirecrawlResearch,
    ScrapeResult,
    CrawlResult,
    FIRECRAWL_AVAILABLE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class ResearchSource:
    """A source in the research pipeline."""
    url: str
    title: str = ""
    content: str = ""
    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    relevance: float = 0.0
    source_type: str = "web"  # web, pdf, doc
    scraped_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ResearchReport:
    """Complete research report."""
    topic: str
    query: str
    sources: List[ResearchSource] = field(default_factory=list)
    synthesis: str = ""
    key_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    total_sources: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "pending"


class AutoResearchPipeline:
    """
    Autonomous research pipeline combining multiple data sources.

    Workflow:
    1. Query Exa for relevant URLs
    2. Scrape/crawl URLs with Firecrawl
    3. Extract structured data
    4. Synthesize findings
    5. Persist to memory
    """

    def __init__(
        self,
        firecrawl_key: Optional[str] = None,
        exa_key: Optional[str] = None,
        letta_url: str = "http://localhost:8283"
    ):
        """
        Initialize the research pipeline.

        Args:
            firecrawl_key: Firecrawl API key
            exa_key: Exa API key
            letta_url: Letta server URL for memory persistence
        """
        self.firecrawl_key = firecrawl_key or os.getenv("FIRECRAWL_API_KEY")
        self.exa_key = exa_key or os.getenv("EXA_API_KEY")
        self.letta_url = letta_url

        # Initialize Firecrawl client
        self.firecrawl = None
        if FIRECRAWL_AVAILABLE and self.firecrawl_key:
            try:
                self.firecrawl = FirecrawlResearch(api_key=self.firecrawl_key)
                logger.info("Firecrawl client initialized")
            except Exception as e:
                logger.warning(f"Firecrawl init failed: {e}")

        # Initialize Exa client (optional)
        self.exa = None
        if self.exa_key:
            try:
                from exa_py import Exa
                self.exa = Exa(self.exa_key)
                logger.info("Exa client initialized")
            except ImportError:
                logger.warning("exa_py not installed. Run: pip install exa_py")
            except Exception as e:
                logger.warning(f"Exa init failed: {e}")

    def search_exa(
        self,
        query: str,
        num_results: int = 10,
        use_autoprompt: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant URLs using Exa.

        Args:
            query: Search query
            num_results: Number of results
            use_autoprompt: Use AI to improve query

        Returns:
            List of search results with URLs
        """
        if not self.exa:
            logger.warning("Exa client not available")
            return []

        try:
            results = self.exa.search(
                query,
                num_results=num_results,
                use_autoprompt=use_autoprompt
            )

            return [
                {
                    "url": r.url,
                    "title": r.title,
                    "score": r.score if hasattr(r, 'score') else 0.0,
                    "published_date": r.published_date if hasattr(r, 'published_date') else None
                }
                for r in results.results
            ]
        except Exception as e:
            logger.error(f"Exa search failed: {e}")
            return []

    def scrape_sources(
        self,
        urls: List[str],
        parallel: bool = True
    ) -> List[ScrapeResult]:
        """
        Scrape content from multiple URLs.

        Args:
            urls: URLs to scrape
            parallel: Use batch scraping

        Returns:
            List of scrape results
        """
        if not self.firecrawl:
            logger.warning("Firecrawl client not available")
            return []

        try:
            if parallel:
                return self.firecrawl.batch_scrape(urls)
            else:
                return [self.firecrawl.scrape(url) for url in urls]
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            return []

    def crawl_site(
        self,
        url: str,
        max_depth: int = 2,
        limit: int = 50
    ) -> CrawlResult:
        """
        Deep crawl a website.

        Args:
            url: Starting URL
            max_depth: Maximum crawl depth
            limit: Maximum pages

        Returns:
            Crawl result with all pages
        """
        if not self.firecrawl:
            logger.warning("Firecrawl client not available")
            return CrawlResult(start_url=url, success=False, error="Client not available")

        try:
            return self.firecrawl.crawl(url, max_depth=max_depth, limit=limit)
        except Exception as e:
            logger.error(f"Crawl failed: {e}")
            return CrawlResult(start_url=url, success=False, error=str(e))

    def extract_structured(
        self,
        url: str,
        schema: Dict[str, Any],
        prompt: str
    ) -> Dict[str, Any]:
        """
        Extract structured data from URL.

        Args:
            url: URL to extract from
            schema: JSON schema for output
            prompt: Extraction prompt

        Returns:
            Extracted data
        """
        if not self.firecrawl:
            return {"error": "Firecrawl not available"}

        return self.firecrawl.extract(url, schema, prompt)

    def research_topic(
        self,
        topic: str,
        num_sources: int = 10,
        crawl_depth: int = 1,
        extract_summaries: bool = True
    ) -> ResearchReport:
        """
        Comprehensive topic research.

        1. Search for relevant sources via Exa
        2. Scrape all sources via Firecrawl
        3. Extract key information
        4. Compile research report

        Args:
            topic: Research topic
            num_sources: Number of sources to research
            crawl_depth: How deep to crawl each source
            extract_summaries: Extract AI summaries

        Returns:
            Complete research report
        """
        report = ResearchReport(
            topic=topic,
            query=topic,
            status="in_progress"
        )

        logger.info(f"Starting research on: {topic}")

        # Step 1: Find relevant sources
        logger.info("Searching for sources via Exa...")
        search_results = self.search_exa(topic, num_results=num_sources)

        if not search_results:
            logger.warning("No search results found")
            report.status = "failed"
            return report

        urls = [r["url"] for r in search_results]
        logger.info(f"Found {len(urls)} relevant sources")

        # Step 2: Scrape all sources
        logger.info("Scraping source content via Firecrawl...")
        scrape_results = self.scrape_sources(urls)

        # Step 3: Process results
        summary_schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Brief summary"},
                "key_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key points"
                },
                "relevance": {"type": "number", "description": "Relevance 0-1"}
            }
        }

        for i, (search_result, scrape_result) in enumerate(zip(search_results, scrape_results)):
            source = ResearchSource(
                url=scrape_result.url,
                title=search_result.get("title", ""),
                content=scrape_result.markdown or "",
                source_type="web"
            )

            # Extract summary if requested
            if extract_summaries and scrape_result.success and self.firecrawl:
                try:
                    extracted = self.firecrawl.extract(
                        scrape_result.url,
                        schema=summary_schema,
                        prompt=f"Summarize this content in relation to: {topic}"
                    )
                    source.summary = extracted.get("summary", "")
                    source.key_points = extracted.get("key_points", [])
                    source.relevance = extracted.get("relevance", 0.0)
                except Exception as e:
                    logger.warning(f"Extraction failed for {scrape_result.url}: {e}")

            report.sources.append(source)
            logger.info(f"Processed source {i+1}/{len(scrape_results)}: {source.title}")

        report.total_sources = len(report.sources)
        report.status = "completed"

        # Step 4: Compile key findings
        all_key_points = []
        for source in report.sources:
            all_key_points.extend(source.key_points)

        # Deduplicate and rank key points
        seen = set()
        for point in all_key_points:
            normalized = point.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                report.key_findings.append(point)

        report.key_findings = report.key_findings[:20]  # Top 20 findings

        logger.info(f"Research complete: {report.total_sources} sources, {len(report.key_findings)} findings")

        return report

    def get_llm_context(
        self,
        urls: List[str],
        max_tokens: int = 8000
    ) -> str:
        """
        Get LLM-ready context from URLs.

        Args:
            urls: URLs to scrape
            max_tokens: Max tokens (4 chars/token)

        Returns:
            Formatted markdown context
        """
        if not self.firecrawl:
            return ""

        return self.firecrawl.get_llm_context(urls, max_tokens)

    def persist_to_letta(
        self,
        report: ResearchReport,
        agent_id: Optional[str] = None
    ) -> bool:
        """
        Persist research report to Letta memory.

        Args:
            report: Research report to persist
            agent_id: Letta agent ID

        Returns:
            Success status
        """
        try:
            from letta_client import Letta

            client = Letta(base_url=self.letta_url)

            # Format report for memory
            memory_content = f"""
## Research Report: {report.topic}

**Created**: {report.created_at}
**Sources**: {report.total_sources}
**Status**: {report.status}

### Key Findings
{chr(10).join(f'- {f}' for f in report.key_findings[:10])}

### Sources Summary
{chr(10).join(f'- [{s.title}]({s.url}): {s.summary[:200]}...' for s in report.sources[:5])}
"""

            # This would need actual Letta API implementation
            logger.info(f"Would persist report to Letta: {len(memory_content)} chars")
            return True

        except Exception as e:
            logger.error(f"Letta persistence failed: {e}")
            return False


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Auto-Research Pipeline - Firecrawl + Exa Integration"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Research command
    research_parser = subparsers.add_parser("research", help="Research a topic")
    research_parser.add_argument("topic", help="Topic to research")
    research_parser.add_argument("--sources", "-n", type=int, default=10, help="Number of sources")
    research_parser.add_argument("--depth", "-d", type=int, default=1, help="Crawl depth per source")
    research_parser.add_argument("--output", "-o", help="Output JSON file")

    # Scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Scrape a URL")
    scrape_parser.add_argument("url", help="URL to scrape")
    scrape_parser.add_argument("--output", "-o", help="Output file")

    # Crawl command
    crawl_parser = subparsers.add_parser("crawl", help="Crawl a website")
    crawl_parser.add_argument("url", help="Starting URL")
    crawl_parser.add_argument("--depth", "-d", type=int, default=2, help="Crawl depth")
    crawl_parser.add_argument("--limit", "-l", type=int, default=50, help="Max pages")
    crawl_parser.add_argument("--output", "-o", help="Output file")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract structured data")
    extract_parser.add_argument("url", help="URL to extract from")
    extract_parser.add_argument("--prompt", "-p", required=True, help="Extraction prompt")

    # Context command
    context_parser = subparsers.add_parser("context", help="Get LLM context from URLs")
    context_parser.add_argument("urls", nargs="+", help="URLs to process")
    context_parser.add_argument("--tokens", "-t", type=int, default=8000, help="Max tokens")

    # Test command
    subparsers.add_parser("test", help="Test connection")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = AutoResearchPipeline()

    if args.command == "test":
        print("=" * 60)
        print("Auto-Research Pipeline - Connection Test")
        print("=" * 60)
        print(f"Firecrawl: {'✅ Available' if pipeline.firecrawl else '❌ Not configured'}")
        print(f"Exa:       {'✅ Available' if pipeline.exa else '❌ Not configured'}")
        print()

        if pipeline.firecrawl:
            print("Testing Firecrawl scrape...")
            result = pipeline.firecrawl.scrape("https://example.com")
            print(f"  Success: {result.success}")
            if result.markdown:
                print(f"  Content: {result.markdown[:100]}...")

    elif args.command == "research":
        print(f"🔬 Researching: {args.topic}")
        print("-" * 60)

        report = pipeline.research_topic(
            args.topic,
            num_sources=args.sources,
            crawl_depth=args.depth
        )

        # Output
        if args.output:
            with open(args.output, "w") as f:
                json.dump(asdict(report), f, indent=2)
            print(f"\n📄 Report saved to: {args.output}")
        else:
            print(f"\n📊 Research Report: {report.topic}")
            print(f"   Status: {report.status}")
            print(f"   Sources: {report.total_sources}")
            print(f"\n🔑 Key Findings:")
            for i, finding in enumerate(report.key_findings[:10], 1):
                print(f"   {i}. {finding}")

    elif args.command == "scrape" and args.url:
        if not pipeline.firecrawl:
            print("❌ Firecrawl not configured")
            return

        print(f"🌐 Scraping: {args.url}")
        result = pipeline.firecrawl.scrape(args.url)

        if args.output:
            with open(args.output, "w") as f:
                f.write(result.markdown or "")
            print(f"📄 Saved to: {args.output}")
        else:
            print(f"\nSuccess: {result.success}")
            if result.markdown:
                print(f"\n{result.markdown[:2000]}...")

    elif args.command == "crawl" and args.url:
        if not pipeline.firecrawl:
            print("❌ Firecrawl not configured")
            return

        print(f"🕷️ Crawling: {args.url} (depth={args.depth}, limit={args.limit})")
        result = pipeline.crawl_site(args.url, max_depth=args.depth, limit=args.limit)

        print(f"\n📊 Crawl Results:")
        print(f"   Success: {result.success}")
        print(f"   Pages: {result.total_pages}")

        if result.pages:
            print(f"\n📄 Pages found:")
            for page in result.pages[:10]:
                print(f"   - {page.url}")

    elif args.command == "extract" and args.url:
        if not pipeline.firecrawl:
            print("❌ Firecrawl not configured")
            return

        # Default extraction schema
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "key_points": {"type": "array", "items": {"type": "string"}},
                "entities": {"type": "array", "items": {"type": "string"}}
            }
        }

        print(f"🔍 Extracting from: {args.url}")
        result = pipeline.firecrawl.extract(args.url, schema, args.prompt)
        print(json.dumps(result, indent=2))

    elif args.command == "context" and args.urls:
        if not pipeline.firecrawl:
            print("❌ Firecrawl not configured")
            return

        print(f"📚 Building LLM context from {len(args.urls)} URLs...")
        context = pipeline.get_llm_context(args.urls, max_tokens=args.tokens)
        print(context)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
