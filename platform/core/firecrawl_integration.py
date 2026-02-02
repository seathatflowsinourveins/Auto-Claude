#!/usr/bin/env python3
"""
Firecrawl Integration Module for Auto-Research Pipeline

Provides comprehensive web scraping, crawling, and extraction capabilities
using Firecrawl's AI-powered API. Integrates with the orchestrator for
autonomous research workflows.

Features:
- Single URL scraping with markdown output
- Deep website crawling with configurable depth
- Batch scraping for multiple URLs
- Structured data extraction with JSON schemas
- Async operations for non-blocking workflows
- LLM-ready markdown conversion
- Automatic retry and error handling
"""

import asyncio
import os
import json
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Try to import firecrawl, handle gracefully if not installed
try:
    from firecrawl import Firecrawl, AsyncFirecrawl
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False
    logger.warning("Firecrawl SDK not installed. Run: pip install firecrawl-py")


class OutputFormat(Enum):
    """Supported output formats for scraped content."""
    MARKDOWN = "markdown"
    HTML = "html"
    RAW_HTML = "rawHtml"
    SCREENSHOT = "screenshot"
    LINKS = "links"
    EXTRACT = "extract"


@dataclass
class ScrapeResult:
    """Result of a scrape operation."""
    url: str
    success: bool
    content: Optional[str] = None
    markdown: Optional[str] = None
    html: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    links: List[str] = field(default_factory=list)
    extracted_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class CrawlResult:
    """Result of a crawl operation."""
    start_url: str
    success: bool
    pages: List[ScrapeResult] = field(default_factory=list)
    total_pages: int = 0
    crawl_depth: int = 0
    job_id: Optional[str] = None
    status: str = "unknown"
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ExtractionSchema:
    """Schema for structured data extraction."""
    name: str
    description: str
    schema: Dict[str, Any]
    prompt: Optional[str] = None


class FirecrawlResearch:
    """
    Firecrawl integration for autonomous research workflows.

    Provides methods for:
    - Web scraping with LLM-ready output
    - Deep website crawling
    - Batch URL processing
    - Structured data extraction
    - Site mapping
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.firecrawl.dev",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        default_formats: List[str] = None
    ):
        """
        Initialize Firecrawl client.

        Args:
            api_key: Firecrawl API key (or set FIRECRAWL_API_KEY env var)
            base_url: API base URL
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Base delay between retries (exponential backoff)
            default_formats: Default output formats (default: ["markdown"])
        """
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.default_formats = default_formats or ["markdown"]

        if not FIRECRAWL_AVAILABLE:
            raise ImportError(
                "Firecrawl SDK not installed. Install with: pip install firecrawl-py"
            )

        if not self.api_key:
            raise ValueError(
                "Firecrawl API key required. Set FIRECRAWL_API_KEY environment variable "
                "or pass api_key parameter. Get your key at: https://firecrawl.dev"
            )

        # Initialize sync and async clients
        self._client = Firecrawl(api_key=self.api_key)
        self._async_client = None  # Lazy initialization

        logger.info("Firecrawl client initialized successfully")

    @property
    def async_client(self) -> "AsyncFirecrawl":
        """Lazy initialization of async client."""
        if self._async_client is None:
            self._async_client = AsyncFirecrawl(api_key=self.api_key)
        return self._async_client

    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
        raise last_error

    def _extract_metadata(self, doc) -> Dict[str, Any]:
        """Extract metadata from a Document object, handling v2 API response format."""
        if not doc:
            return {}

        # Try to get metadata attribute
        metadata = getattr(doc, 'metadata', None)
        if metadata is None:
            return {}

        # If metadata is a dict, return it directly
        if isinstance(metadata, dict):
            return metadata

        # If metadata is a pydantic model or object, convert to dict
        if hasattr(metadata, 'model_dump'):
            return metadata.model_dump()
        elif hasattr(metadata, '__dict__'):
            return {k: v for k, v in metadata.__dict__.items() if not k.startswith('_')}

        return {}

    # =========================================================================
    # SINGLE URL SCRAPING
    # =========================================================================

    def scrape(
        self,
        url: str,
        formats: List[str] = None,
        include_tags: List[str] = None,
        exclude_tags: List[str] = None,
        only_main_content: bool = True,
        wait_for: int = 0,
        timeout: int = 30000
    ) -> ScrapeResult:
        """
        Scrape a single URL and return LLM-ready content.

        Args:
            url: URL to scrape
            formats: Output formats (markdown, html, rawHtml, screenshot, links)
            include_tags: CSS selectors to include
            exclude_tags: CSS selectors to exclude
            only_main_content: Extract only main content (skip nav, footer, etc.)
            wait_for: Wait time in ms for JavaScript rendering
            timeout: Request timeout in ms

        Returns:
            ScrapeResult with content in requested formats
        """
        formats = formats or self.default_formats

        try:
            # Use v2 API with keyword arguments
            result = self._retry_with_backoff(
                self._client.scrape,
                url,
                formats=formats,
                only_main_content=only_main_content,
                wait_for=wait_for if wait_for > 0 else None,
                timeout=timeout if timeout != 30000 else None,
                include_tags=include_tags,
                exclude_tags=exclude_tags
            )

            # Handle Document response from v2 API
            return ScrapeResult(
                url=url,
                success=True,
                content=getattr(result, 'content', None),
                markdown=getattr(result, 'markdown', None),
                html=getattr(result, 'html', None),
                metadata=getattr(result, 'metadata', {}) if hasattr(result, 'metadata') else {},
                links=getattr(result, 'links', []) if hasattr(result, 'links') else []
            )
        except Exception as e:
            logger.error(f"Scrape failed for {url}: {e}")
            return ScrapeResult(
                url=url,
                success=False,
                error=str(e)
            )

    async def scrape_async(
        self,
        url: str,
        formats: List[str] = None,
        only_main_content: bool = True
    ) -> ScrapeResult:
        """Async version of scrape."""
        formats = formats or self.default_formats

        try:
            # Use v2 async API with keyword arguments
            result = await self.async_client.scrape(
                url,
                formats=formats,
                only_main_content=only_main_content
            )

            # Handle Document response from v2 API
            return ScrapeResult(
                url=url,
                success=True,
                content=getattr(result, 'content', None),
                markdown=getattr(result, 'markdown', None),
                html=getattr(result, 'html', None),
                metadata=getattr(result, 'metadata', {}) if hasattr(result, 'metadata') else {},
                links=getattr(result, 'links', []) if hasattr(result, 'links') else []
            )
        except Exception as e:
            logger.error(f"Async scrape failed for {url}: {e}")
            return ScrapeResult(
                url=url,
                success=False,
                error=str(e)
            )

    # =========================================================================
    # BATCH SCRAPING
    # =========================================================================

    def batch_scrape(
        self,
        urls: List[str],
        formats: List[str] = None,
        only_main_content: bool = True,
        concurrent_limit: int = 5
    ) -> List[ScrapeResult]:
        """
        Scrape multiple URLs in batch.

        Args:
            urls: List of URLs to scrape
            formats: Output formats
            only_main_content: Extract only main content
            concurrent_limit: Max concurrent requests (reserved for future use)

        Returns:
            List of ScrapeResult for each URL
        """
        formats = formats or self.default_formats
        results = []

        try:
            # Use v2 API with keyword arguments (batch_scrape instead of batch_scrape_urls)
            batch_result = self._retry_with_backoff(
                self._client.batch_scrape,
                urls,
                formats=formats,
                only_main_content=only_main_content
            )

            # Handle BatchScrapeResponse from v2 API
            data = getattr(batch_result, 'data', []) or []
            for doc in data:
                results.append(ScrapeResult(
                    url=getattr(doc, 'url', '') or '',
                    success=True,
                    content=getattr(doc, 'content', None),
                    markdown=getattr(doc, 'markdown', None),
                    html=getattr(doc, 'html', None),
                    metadata=self._extract_metadata(doc),
                    links=getattr(doc, 'links', []) or []
                ))

        except Exception as e:
            logger.error(f"Batch scrape failed: {e}")
            # Return error results for all URLs
            for url in urls:
                results.append(ScrapeResult(
                    url=url,
                    success=False,
                    error=str(e)
                ))

        return results

    async def batch_scrape_async(
        self,
        urls: List[str],
        formats: List[str] = None
    ) -> str:
        """
        Start async batch scrape job, returns job ID.

        Use check_batch_status() to poll for completion.
        """
        formats = formats or self.default_formats

        try:
            # Use v2 API with start_batch_scrape (returns job ID for polling)
            result = await self.async_client.start_batch_scrape(
                urls,
                formats=formats
            )
            return getattr(result, 'id', None) or result.get("id") if isinstance(result, dict) else result.id
        except Exception as e:
            logger.error(f"Async batch scrape failed: {e}")
            raise

    # =========================================================================
    # WEBSITE CRAWLING
    # =========================================================================

    def crawl(
        self,
        url: str,
        max_depth: int = 2,
        limit: int = 100,
        include_paths: List[str] = None,
        exclude_paths: List[str] = None,
        allow_external_links: bool = False,
        formats: List[str] = None
    ) -> CrawlResult:
        """
        Crawl a website starting from the given URL.

        Args:
            url: Starting URL
            max_depth: Maximum crawl depth (maps to max_discovery_depth in v2 API)
            limit: Maximum pages to crawl
            include_paths: Glob patterns for paths to include
            exclude_paths: Glob patterns for paths to exclude
            allow_external_links: Follow external links
            formats: Output formats

        Returns:
            CrawlResult with all crawled pages
        """
        formats = formats or self.default_formats

        try:
            # Use v2 API with keyword arguments
            # Note: maxDepth is now max_discovery_depth in v2 API
            from firecrawl.v2.types import ScrapeOptions

            scrape_options = ScrapeOptions(formats=formats) if formats else None

            result = self._retry_with_backoff(
                self._client.crawl,
                url,
                max_discovery_depth=max_depth,
                limit=limit,
                allow_external_links=allow_external_links,
                include_paths=include_paths,
                exclude_paths=exclude_paths,
                scrape_options=scrape_options,
                poll_interval=2  # v2 default is 2 seconds
            )

            # Handle CrawlJob response from v2 API
            # CrawlJob has: status, total, completed, credits_used, data: List[Document]
            pages = []
            job_status = getattr(result, 'status', 'unknown')

            # Get documents from the crawl job
            data = getattr(result, 'data', []) or []
            for doc in data:
                # Document objects have direct attributes (not dict-style)
                pages.append(ScrapeResult(
                    url=getattr(doc, 'url', '') or '',
                    success=True,
                    content=getattr(doc, 'content', None),
                    markdown=getattr(doc, 'markdown', None),
                    html=getattr(doc, 'html', None),
                    metadata=self._extract_metadata(doc),
                    links=getattr(doc, 'links', []) or []
                ))

            return CrawlResult(
                start_url=url,
                success=job_status == 'completed',
                pages=pages,
                total_pages=getattr(result, 'total', len(pages)),
                crawl_depth=max_depth,
                job_id=getattr(result, 'id', None),
                status=job_status
            )

        except ImportError:
            # Fallback if ScrapeOptions not available
            logger.warning("ScrapeOptions not available, using basic crawl")
            result = self._retry_with_backoff(
                self._client.crawl,
                url,
                max_discovery_depth=max_depth,
                limit=limit,
                allow_external_links=allow_external_links,
                include_paths=include_paths,
                exclude_paths=exclude_paths,
                poll_interval=2
            )

            # Handle CrawlJob response
            pages = []
            job_status = getattr(result, 'status', 'unknown')
            data = getattr(result, 'data', []) or []

            for doc in data:
                pages.append(ScrapeResult(
                    url=getattr(doc, 'url', '') or '',
                    success=True,
                    markdown=getattr(doc, 'markdown', None),
                    metadata=self._extract_metadata(doc),
                ))

            return CrawlResult(
                start_url=url,
                success=job_status == 'completed',
                pages=pages,
                total_pages=getattr(result, 'total', len(pages)),
                crawl_depth=max_depth,
                job_id=getattr(result, 'id', None),
                status=job_status
            )

        except Exception as e:
            logger.error(f"Crawl failed for {url}: {e}")
            return CrawlResult(
                start_url=url,
                success=False,
                error=str(e),
                status="failed"
            )

    def start_crawl_async(
        self,
        url: str,
        max_depth: int = 2,
        limit: int = 100
    ) -> str:
        """
        Start async crawl job, returns job ID.

        Use check_crawl_status() to poll for completion.
        """
        try:
            # Use v2 API with start_crawl (renamed from async_crawl)
            result = self._client.start_crawl(
                url,
                max_discovery_depth=max_depth,
                limit=limit
            )
            return getattr(result, 'id', None) or (result.get("id") if isinstance(result, dict) else None)
        except Exception as e:
            logger.error(f"Async crawl start failed: {e}")
            raise

    def check_crawl_status(self, job_id: str) -> Dict[str, Any]:
        """Check status of an async crawl job."""
        try:
            # Use v2 API with get_crawl_status (renamed from check_crawl_status)
            result = self._client.get_crawl_status(job_id)
            # Convert response to dict if it's a dataclass/model
            if hasattr(result, '__dict__'):
                return {
                    'status': getattr(result, 'status', 'unknown'),
                    'total': getattr(result, 'total', 0),
                    'completed': getattr(result, 'completed', 0),
                    'data': getattr(result, 'data', [])
                }
            return result
        except Exception as e:
            logger.error(f"Crawl status check failed: {e}")
            raise

    # =========================================================================
    # SITE MAPPING
    # =========================================================================

    def map_site(
        self,
        url: str,
        search: Optional[str] = None,
        limit: int = 5000,
        include_subdomains: bool = False
    ) -> List[str]:
        """
        Map all URLs on a website without scraping content.

        Args:
            url: Website URL
            search: Optional search query to filter URLs
            limit: Maximum URLs to return
            include_subdomains: Include subdomain URLs

        Returns:
            List of discovered URLs
        """
        try:
            # Use v2 API with keyword arguments
            result = self._retry_with_backoff(
                self._client.map,
                url,
                search=search,
                limit=limit,
                include_subdomains=include_subdomains
            )
            # Handle MapResponse from v2 API
            if hasattr(result, 'links'):
                return result.links or []
            return result.get("links", []) if isinstance(result, dict) else []
        except Exception as e:
            logger.error(f"Site mapping failed for {url}: {e}")
            return []

    # =========================================================================
    # STRUCTURED DATA EXTRACTION
    # =========================================================================

    def extract(
        self,
        url: str,
        schema: Dict[str, Any],
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract structured data from a URL using AI.

        Args:
            url: URL to extract from
            schema: JSON schema defining expected output structure
            prompt: Extraction prompt describing what to extract
            system_prompt: Optional system prompt for the extraction LLM

        Returns:
            Extracted data matching the schema
        """
        try:
            # Try using v2 extract API directly first
            try:
                from firecrawl.v2.types import ExtractConfig

                extract_config = ExtractConfig(
                    schema=schema,
                    prompt=prompt,
                    system_prompt=system_prompt
                )

                result = self._retry_with_backoff(
                    self._client.scrape,
                    url,
                    formats=["extract"],
                    extract=extract_config
                )
            except ImportError:
                # Fallback: use scrape with extract format
                result = self._retry_with_backoff(
                    self._client.scrape,
                    url,
                    formats=["extract"]
                )

            # Handle Document response from v2 API
            if hasattr(result, 'extract'):
                return result.extract or {}
            return result.get("extract", {}) if isinstance(result, dict) else {}
        except Exception as e:
            logger.error(f"Extraction failed for {url}: {e}")
            return {"error": str(e)}

    def extract_batch(
        self,
        urls: List[str],
        schema: Dict[str, Any],
        prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract structured data from multiple URLs."""
        results = []

        try:
            # Use v2 API with batch_scrape and extract config
            try:
                from firecrawl.v2.types import ExtractConfig

                extract_config = ExtractConfig(
                    schema=schema,
                    prompt=prompt
                )

                batch_result = self._retry_with_backoff(
                    self._client.batch_scrape,
                    urls,
                    formats=["extract"],
                    extract=extract_config
                )
            except ImportError:
                # Fallback without ExtractConfig
                batch_result = self._retry_with_backoff(
                    self._client.batch_scrape,
                    urls,
                    formats=["extract"]
                )

            # Handle BatchScrapeResponse from v2 API
            data = getattr(batch_result, 'data', []) if hasattr(batch_result, 'data') else []
            for item in data:
                url = getattr(item, 'url', '') if hasattr(item, 'url') else item.get("url", "")
                extract = getattr(item, 'extract', {}) if hasattr(item, 'extract') else item.get("extract", {})
                results.append({
                    "url": url,
                    "data": extract
                })

        except Exception as e:
            logger.error(f"Batch extraction failed: {e}")
            for url in urls:
                results.append({"url": url, "error": str(e)})

        return results

    # =========================================================================
    # RESEARCH HELPERS
    # =========================================================================

    def research_topic(
        self,
        topic: str,
        search_urls: List[str],
        max_pages_per_site: int = 5,
        extract_summary: bool = True
    ) -> Dict[str, Any]:
        """
        Research a topic by crawling multiple sources.

        Args:
            topic: Research topic
            search_urls: List of URLs to research
            max_pages_per_site: Max pages to crawl per URL
            extract_summary: Extract AI summary for each source

        Returns:
            Research results with content and summaries
        """
        results = {
            "topic": topic,
            "sources": [],
            "total_pages": 0,
            "timestamp": datetime.utcnow().isoformat()
        }

        summary_schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "key_points": {"type": "array", "items": {"type": "string"}},
                "relevance_score": {"type": "number"}
            }
        }

        for url in search_urls:
            logger.info(f"Researching: {url}")

            # Crawl the site
            crawl_result = self.crawl(
                url,
                max_depth=1,
                limit=max_pages_per_site
            )

            source_data = {
                "url": url,
                "pages_found": crawl_result.total_pages,
                "content": []
            }

            for page in crawl_result.pages:
                page_data = {
                    "url": page.url,
                    "markdown": page.markdown,
                    "metadata": page.metadata
                }

                # Optional: Extract summary
                if extract_summary and page.markdown:
                    extracted = self.extract(
                        page.url,
                        schema=summary_schema,
                        prompt=f"Summarize this content in relation to: {topic}"
                    )
                    page_data["summary"] = extracted

                source_data["content"].append(page_data)

            results["sources"].append(source_data)
            results["total_pages"] += crawl_result.total_pages

        return results

    def get_llm_context(
        self,
        urls: List[str],
        max_tokens: int = 10000,
        include_metadata: bool = True
    ) -> str:
        """
        Get LLM-ready context from multiple URLs.

        Scrapes URLs and formats content for LLM consumption,
        respecting token limits.

        Args:
            urls: URLs to scrape
            max_tokens: Approximate max tokens (4 chars/token estimate)
            include_metadata: Include page metadata

        Returns:
            Formatted markdown context string
        """
        max_chars = max_tokens * 4
        context_parts = []
        current_chars = 0

        results = self.batch_scrape(urls)

        for result in results:
            if not result.success or not result.markdown:
                continue

            # Build context entry
            entry = f"\n## Source: {result.url}\n"

            if include_metadata and result.metadata:
                title = result.metadata.get("title", "")
                description = result.metadata.get("description", "")
                if title:
                    entry += f"**Title**: {title}\n"
                if description:
                    entry += f"**Description**: {description}\n"

            entry += f"\n{result.markdown}\n"
            entry += "\n---\n"

            # Check token limit
            if current_chars + len(entry) > max_chars:
                # Truncate if needed
                remaining = max_chars - current_chars
                if remaining > 500:
                    entry = entry[:remaining] + "\n[Content truncated...]\n"
                    context_parts.append(entry)
                break

            context_parts.append(entry)
            current_chars += len(entry)

        return "".join(context_parts)


# =============================================================================
# LETTA INTEGRATION
# =============================================================================

def setup_firecrawl_tools(
    api_key: Optional[str] = None,
    base_url: str = "http://localhost:8283"
) -> List[str]:
    """
    Set up Firecrawl tools for Letta agent integration.

    Args:
        api_key: Firecrawl API key
        base_url: Letta server base URL

    Returns:
        List of registered tool IDs
    """
    try:
        from letta_client import Letta
        from letta_client.types import StdioServerConfig
    except ImportError:
        logger.error("letta_client not installed")
        return []

    firecrawl_key = api_key or os.getenv("FIRECRAWL_API_KEY")
    if not firecrawl_key:
        logger.warning("FIRECRAWL_API_KEY not set")
        return []

    client = Letta(base_url=base_url)

    try:
        # Configure Firecrawl MCP server
        server_config = StdioServerConfig(
            server_name="firecrawl",
            command="npx",
            args=["-y", "firecrawl-mcp"],
            env={
                "FIRECRAWL_API_KEY": firecrawl_key,
                "FIRECRAWL_RETRY_MAX_ATTEMPTS": "5"
            }
        )

        # Add MCP server
        try:
            client.tools.add_mcp_server(request=server_config)
            logger.info("Successfully added Firecrawl MCP server")
        except Exception as e:
            if "already exists" in str(e):
                pass
            else:
                raise e

        # List and add tools
        mcp_tools = client.tools.list_mcp_tools_by_server(mcp_server_name="firecrawl")

        desired_tools = [
            'firecrawl_scrape',
            'firecrawl_crawl',
            'firecrawl_map',
            'firecrawl_extract'
        ]

        tool_ids = []
        for tool in mcp_tools:
            if tool.name in desired_tools:
                try:
                    added = client.tools.add_mcp_tool(
                        mcp_server_name="firecrawl",
                        mcp_tool_name=tool.name
                    )
                    tool_ids.append(added.id)
                    logger.info(f"Added tool: {tool.name}")
                except Exception:
                    pass

        return tool_ids

    except Exception as e:
        logger.error(f"Firecrawl setup error: {e}")
        return []


# =============================================================================
# ORCHESTRATOR INTEGRATION
# =============================================================================

def create_firecrawl_tools() -> List[Dict[str, Any]]:
    """
    Create tool definitions for orchestrator integration.

    Returns:
        List of tool configurations for CoreOrchestrator
    """
    return [
        {
            "name": "firecrawl_scrape",
            "description": "Scrape a single URL and return LLM-ready markdown content",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to scrape"
                    },
                    "only_main_content": {
                        "type": "boolean",
                        "description": "Extract only main content",
                        "default": True
                    }
                },
                "required": ["url"]
            }
        },
        {
            "name": "firecrawl_crawl",
            "description": "Crawl a website starting from URL, returning all pages",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Starting URL"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum crawl depth",
                        "default": 2
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum pages to crawl",
                        "default": 50
                    }
                },
                "required": ["url"]
            }
        },
        {
            "name": "firecrawl_map",
            "description": "Map all URLs on a website without scraping content",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Website URL"
                    },
                    "search": {
                        "type": "string",
                        "description": "Optional search query to filter URLs"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum URLs to return",
                        "default": 1000
                    }
                },
                "required": ["url"]
            }
        },
        {
            "name": "firecrawl_extract",
            "description": "Extract structured data from URL using AI",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to extract from"
                    },
                    "schema": {
                        "type": "object",
                        "description": "JSON schema for extraction"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Extraction prompt"
                    }
                },
                "required": ["url", "schema"]
            }
        },
        {
            "name": "firecrawl_batch_scrape",
            "description": "Scrape multiple URLs in batch",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of URLs to scrape"
                    }
                },
                "required": ["urls"]
            }
        },
        {
            "name": "firecrawl_research",
            "description": "Research a topic by crawling and summarizing multiple sources",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Research topic"
                    },
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Source URLs to research"
                    },
                    "max_pages_per_site": {
                        "type": "integer",
                        "description": "Max pages per source",
                        "default": 5
                    }
                },
                "required": ["topic", "urls"]
            }
        }
    ]


class FirecrawlToolExecutor:
    """Execute Firecrawl tools for the orchestrator."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = FirecrawlResearch(api_key=api_key) if FIRECRAWL_AVAILABLE else None

    def execute(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a Firecrawl tool by name."""
        if not self.client:
            return {"success": False, "error": "Firecrawl not available"}

        handlers = {
            "firecrawl_scrape": self._handle_scrape,
            "firecrawl_crawl": self._handle_crawl,
            "firecrawl_map": self._handle_map,
            "firecrawl_extract": self._handle_extract,
            "firecrawl_batch_scrape": self._handle_batch_scrape,
            "firecrawl_research": self._handle_research
        }

        handler = handlers.get(tool_name)
        if not handler:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        return handler(params)

    def _handle_scrape(self, params: Dict[str, Any]) -> Dict[str, Any]:
        result = self.client.scrape(
            url=params["url"],
            only_main_content=params.get("only_main_content", True)
        )
        return {
            "success": result.success,
            "url": result.url,
            "markdown": result.markdown,
            "metadata": result.metadata,
            "error": result.error
        }

    def _handle_crawl(self, params: Dict[str, Any]) -> Dict[str, Any]:
        result = self.client.crawl(
            url=params["url"],
            max_depth=params.get("max_depth", 2),
            limit=params.get("limit", 50)
        )
        return {
            "success": result.success,
            "total_pages": result.total_pages,
            "pages": [
                {"url": p.url, "markdown": p.markdown[:500] + "..." if p.markdown and len(p.markdown) > 500 else p.markdown}
                for p in result.pages
            ],
            "error": result.error
        }

    def _handle_map(self, params: Dict[str, Any]) -> Dict[str, Any]:
        urls = self.client.map_site(
            url=params["url"],
            search=params.get("search"),
            limit=params.get("limit", 1000)
        )
        return {
            "success": len(urls) > 0,
            "total_urls": len(urls),
            "urls": urls[:100]  # Return first 100
        }

    def _handle_extract(self, params: Dict[str, Any]) -> Dict[str, Any]:
        result = self.client.extract(
            url=params["url"],
            schema=params["schema"],
            prompt=params.get("prompt")
        )
        return {
            "success": "error" not in result,
            "data": result
        }

    def _handle_batch_scrape(self, params: Dict[str, Any]) -> Dict[str, Any]:
        results = self.client.batch_scrape(urls=params["urls"])
        return {
            "success": any(r.success for r in results),
            "total": len(results),
            "successful": sum(1 for r in results if r.success),
            "results": [
                {"url": r.url, "success": r.success, "markdown_preview": (r.markdown or "")[:300]}
                for r in results
            ]
        }

    def _handle_research(self, params: Dict[str, Any]) -> Dict[str, Any]:
        result = self.client.research_topic(
            topic=params["topic"],
            search_urls=params["urls"],
            max_pages_per_site=params.get("max_pages_per_site", 5)
        )
        return {
            "success": result["total_pages"] > 0,
            "topic": result["topic"],
            "total_pages": result["total_pages"],
            "sources_count": len(result["sources"]),
            "sources": result["sources"]
        }


# =============================================================================
# CLI / TESTING
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Firecrawl Research CLI")
    parser.add_argument("command", choices=["scrape", "crawl", "map", "extract", "test"])
    parser.add_argument("--url", "-u", help="URL to process")
    parser.add_argument("--depth", "-d", type=int, default=2, help="Crawl depth")
    parser.add_argument("--limit", "-l", type=int, default=50, help="Page limit")

    args = parser.parse_args()

    # Initialize client
    try:
        client = FirecrawlResearch()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

    if args.command == "test":
        print("Testing Firecrawl connection...")
        result = client.scrape("https://example.com")
        print(f"Success: {result.success}")
        if result.markdown:
            print(f"Content preview: {result.markdown[:200]}...")

    elif args.command == "scrape" and args.url:
        result = client.scrape(args.url)
        print(json.dumps({
            "url": result.url,
            "success": result.success,
            "markdown": result.markdown,
            "metadata": result.metadata
        }, indent=2))

    elif args.command == "crawl" and args.url:
        result = client.crawl(args.url, max_depth=args.depth, limit=args.limit)
        print(f"Crawled {result.total_pages} pages from {result.start_url}")
        for page in result.pages[:5]:
            print(f"  - {page.url}")

    elif args.command == "map" and args.url:
        urls = client.map_site(args.url, limit=args.limit)
        print(f"Found {len(urls)} URLs:")
        for url in urls[:20]:
            print(f"  - {url}")

    else:
        parser.print_help()
