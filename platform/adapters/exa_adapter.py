"""
Exa AI Search Adapter - Neural Search for UNLEASH (FULLY UNLEASHED)
====================================================================

Exa is the first search engine optimized for AI - uses neural/embeddings-based
search rather than keyword matching.

Exa 2.0+ Features (2026):
- Exa Fast: <350ms P50 latency, 30% faster than competitors
- Exa Auto: Higher quality default with intelligent mode selection
- Exa Deep: Agentic retrieval with multi-step search (3.5s P50)
- Research API: Async research with structured output schemas
- Company Research: Comprehensive company intelligence
- LinkedIn Search: People and company discovery
- Code Context: Code snippets and documentation search (/context endpoint)
- Stream Answer: Streaming responses for real-time UX
- Websets API: Structured data collection with criteria
- Entity Extraction: Company/person entities from results
- Cost Tracking: Response includes cost_dollars field

Official Docs: https://docs.exa.ai/
GitHub: https://github.com/exa-labs/exa-py
MCP Server: https://github.com/exa-labs/exa-mcp-server

Usage:
    adapter = ExaAdapter()
    await adapter.initialize({"api_key": "exa-xxx"})

    # Fast search (<350ms) - uses type="fast" directly
    result = await adapter.execute("search", query="LangGraph patterns", type="fast")

    # Deep agentic search (highest quality) - uses type="deep" directly
    result = await adapter.execute("search", query="distributed consensus", type="deep")

    # Company research
    result = await adapter.execute("company_research", domain="anthropic.com")

    # Code context search (uses /context endpoint)
    result = await adapter.execute("code_search", query="LangGraph StateGraph example")

    # Async research with schema
    result = await adapter.execute("research", instructions="Compare vector databases",
                                   output_schema={"type": "object", ...})

    # Websets API
    result = await adapter.execute("create_webset", search_query="AI startups",
                                   count=10, criteria=["Must have funding"])
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, AsyncGenerator, List, Dict

import httpx

# Structured logging
try:
    from core.logging_config import get_logger, generate_correlation_id
    _logger = get_logger("adapter.exa")
except (ImportError, ValueError):
    import logging
    _logger = logging.getLogger(__name__)
    generate_correlation_id = lambda: "corr-fallback"

# Retry utilities
try:
    from .retry import RetryConfig, with_retry, retry_async, http_request_with_retry
except ImportError:
    # Fallback for standalone testing
    RetryConfig = None
    with_retry = lambda f=None, **kw: (lambda fn: fn) if f is None else f
    retry_async = None
    http_request_with_retry = None

# HTTP connection pool
try:
    from .http_pool import (
        HTTPConnectionPool,
        get_shared_pool_sync,
        get_config_for_service,
        PoolMetrics,
    )
    HTTP_POOL_AVAILABLE = True
except ImportError:
    HTTPConnectionPool = None
    get_shared_pool_sync = None
    get_config_for_service = None
    PoolMetrics = None
    HTTP_POOL_AVAILABLE = False

# Circuit breaker imports
try:
    from .circuit_breaker_manager import adapter_circuit_breaker, get_adapter_circuit_manager
    from core.resilience import CircuitOpenError
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False
    CircuitOpenError = Exception
    def adapter_circuit_breaker(name):
        class DummyBreaker:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return False
        return DummyBreaker()
    def get_adapter_circuit_manager():
        return None

# SDK Layer imports
try:
    from core.orchestration.base import SDKAdapter, AdapterResult, AdapterStatus, SDKLayer, register_adapter
except (ImportError, ValueError):
        from enum import Enum, IntEnum
        from dataclasses import dataclass, field
        from datetime import datetime
        from typing import Dict, Any, Optional
        from abc import ABC, abstractmethod

        class SDKLayer(IntEnum):
            RESEARCH = 8

        class AdapterStatus(str, Enum):
            UNINITIALIZED = "uninitialized"
            READY = "ready"
            FAILED = "failed"
            ERROR = "error"
            DEGRADED = "degraded"

        @dataclass
        class AdapterResult:
            success: bool
            data: Optional[Dict[str, Any]] = None
            error: Optional[str] = None
            latency_ms: float = 0.0
            cached: bool = False
            metadata: Dict[str, Any] = field(default_factory=dict)
            timestamp: datetime = field(default_factory=datetime.utcnow)

        class SDKAdapter(ABC):
            @property
            @abstractmethod
            def sdk_name(self) -> str: ...
            @abstractmethod
            async def initialize(self, config: Dict) -> AdapterResult: ...
            @abstractmethod
            async def execute(self, operation: str, **kwargs) -> AdapterResult: ...
            @abstractmethod
            async def shutdown(self) -> None: ...

        def register_adapter(name, layer, priority=0):
            def decorator(cls):
                return cls
            return decorator

# Exa SDK
EXA_AVAILABLE = False
try:
    from exa_py import Exa
    EXA_AVAILABLE = True
except ImportError:
    Exa = None


class ExaSearchType(str, Enum):
    """Exa search types with different speed/quality tradeoffs."""
    FAST = "fast"          # <350ms, streamlined models
    AUTO = "auto"          # Default, intelligent hybrid
    NEURAL = "neural"      # Pure embeddings-based
    KEYWORD = "keyword"    # Traditional BM25-style
    DEEP = "deep"          # Agentic retrieval, highest quality (3.5s)


class ExaCategory(str, Enum):
    """Content categories for focused search.

    Note: 'company' and 'people' categories have limited filter support -
    date filters (startPublishedDate, endPublishedDate, startCrawlDate, endCrawlDate)
    and text filters (includeText, excludeText) are NOT supported.
    For 'people' category, includeDomains only accepts LinkedIn domains.
    """
    COMPANY = "company"
    RESEARCH_PAPER = "research paper"
    NEWS = "news"
    TWEET = "tweet"
    PERSONAL_SITE = "personal site"
    FINANCIAL_REPORT = "financial report"
    PEOPLE = "people"
    PDF = "pdf"
    GITHUB = "github"
    PAPERS = "papers"  # Alias for research paper
    NEWS_ARTICLE = "news article"  # Alternative format
    MOVIE = "movie"
    SONG = "song"


class ExaLivecrawl(str, Enum):
    """Livecrawl options (deprecated - use maxAgeHours instead).

    Migration guide:
    - 'always' -> maxAgeHours: 0
    - 'never' -> maxAgeHours: -1
    - 'fallback' -> omit parameter (default)
    """
    NEVER = "never"
    FALLBACK = "fallback"
    PREFERRED = "preferred"
    ALWAYS = "always"
    AUTO = "auto"
    FALLBACK_1_6 = "fallback1.6"


class ExaResearchModel(str, Enum):
    """Research API model options."""
    FAST = "exa-research-fast"
    STANDARD = "exa-research"
    PRO = "exa-research-pro"


@dataclass
class ExaHighlightsConfig:
    """Configuration for highlights extraction.

    Attributes:
        query: Custom query for directing highlight selection (different from search query)
        max_characters: Maximum total highlight length
        num_sentences: Sentences per snippet (deprecated, use max_characters)
        highlights_per_url: Max snippets per result (deprecated)
    """
    query: Optional[str] = None
    max_characters: Optional[int] = None
    num_sentences: Optional[int] = None  # Deprecated
    highlights_per_url: Optional[int] = None  # Deprecated

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-compatible dictionary."""
        config = {}
        if self.query:
            config["query"] = self.query
        if self.max_characters:
            config["maxCharacters"] = self.max_characters
        if self.num_sentences:
            config["numSentences"] = self.num_sentences
        if self.highlights_per_url:
            config["highlightsPerUrl"] = self.highlights_per_url
        return config if config else True


@dataclass
class ExaTextConfig:
    """Configuration for text extraction.

    Attributes:
        max_characters: Maximum characters per result
        include_html_tags: Include HTML formatting in text
        verbosity: Content verbosity - 'compact', 'standard', 'full'
        include_sections: Semantic sections to include (e.g., ['body', 'header'])
        exclude_sections: Semantic sections to exclude (e.g., ['navigation', 'footer'])
    """
    max_characters: Optional[int] = None
    include_html_tags: bool = False
    verbosity: Optional[str] = None  # compact, standard, full
    include_sections: Optional[List[str]] = None
    exclude_sections: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-compatible dictionary."""
        config: Dict[str, Any] = {"include": True}
        if self.max_characters:
            config["maxCharacters"] = self.max_characters
        if self.include_html_tags:
            config["includeHtmlTags"] = True
        if self.verbosity:
            config["verbosity"] = self.verbosity
        if self.include_sections:
            config["includeSections"] = self.include_sections
        if self.exclude_sections:
            config["excludeSections"] = self.exclude_sections
        return config


@dataclass
class ExaSummaryConfig:
    """Configuration for summary generation.

    Attributes:
        query: Custom query for directing summary generation
        schema: JSON schema for structured output extraction
    """
    query: Optional[str] = None
    schema: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-compatible dictionary."""
        config = {}
        if self.query:
            config["query"] = self.query
        if self.schema:
            config["schema"] = self.schema
        return config if config else True


@dataclass
class ExaExtrasConfig:
    """Configuration for extra content extraction.

    Attributes:
        links: Number of URLs to extract from each webpage
        image_links: Number of images to extract per result
    """
    links: int = 0
    image_links: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-compatible dictionary."""
        config = {}
        if self.links > 0:
            config["links"] = self.links
        if self.image_links > 0:
            config["imageLinks"] = self.image_links
        return config


@dataclass
class ExaSearchResult:
    """Individual search result from Exa."""
    title: str
    url: str
    id: Optional[str] = None
    text: str = ""
    highlights: list[str] = field(default_factory=list)
    highlight_scores: list[float] = field(default_factory=list)
    score: float = 0.0
    published_date: Optional[str] = None
    author: Optional[str] = None
    image: Optional[str] = None
    favicon: Optional[str] = None
    subpages: Optional[List[Dict[str, Any]]] = None
    extras: Optional[Dict[str, Any]] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    summary: Optional[str] = None
    links: Optional[List[str]] = None
    image_links: Optional[List[str]] = None


@register_adapter("exa", SDKLayer.RESEARCH, priority=25)
class ExaAdapter(SDKAdapter):
    """
    Exa AI neural search adapter - FULLY UNLEASHED with ALL 2026 Features.

    Search Types:
        - auto: Intelligent hybrid (default) - combines neural and keyword
        - neural: Pure embeddings-based semantic search
        - keyword: Traditional BM25-style matching
        - fast: Streamlined models, <350ms P50 latency
        - deep: Agentic retrieval with query expansion, highest quality (3.5s P50)

    Categories (use for focused search):
        - company: Company pages (limited filters, use with includeDomains)
        - people: People/LinkedIn profiles (limited filters, LinkedIn domains only)
        - research paper / papers: Academic papers
        - news / news article: News articles
        - tweet: Twitter/X posts
        - personal site: Personal websites/blogs
        - financial report: SEC filings, annual reports
        - pdf: PDF documents
        - github: GitHub repositories and code
        - movie: Movie information
        - song: Music/song information

    Content Options:
        - text: Full text with verbosity (compact/standard/full), sections filtering
        - highlights: Relevant snippets with custom query, max_characters
        - summary: AI-generated summary with custom query, JSON schema support
        - context: Combined content string for RAG (recommended 10000+ chars)
        - extras: Extract links and imageLinks from pages
        - subpages: Crawl up to 10 subpages with subpageTarget keywords

    Freshness Control (maxAgeHours replaces livecrawl):
        - 0: Always livecrawl (real-time)
        - positive: Use cache if fresher than N hours
        - -1: Cache only, never livecrawl
        - omit: Default fallback behavior

    Operations:
        - search: Neural/semantic search with full parameter support
        - get_contents: Extract content from URLs with all options
        - find_similar: Find similar content to a URL
        - find_similar_and_contents: Similar search with content extraction
        - search_and_contents: Combined search + content extraction
        - answer: AI-generated answers with citations
        - stream_answer: Streaming answer responses
        - research: Async research with structured output (3 model tiers)
        - company_research: Comprehensive company intelligence
        - linkedin_search: People and company discovery
        - code_search: Code snippets and documentation (/context endpoint)
        - create_webset: Create structured data collection
        - get_webset: Get Webset by ID
        - list_websets: List all Websets
        - optimize_query: Apply autoprompt optimization to query
    """

    # Base URL for direct API calls
    BASE_URL = "https://api.exa.ai"

    def __init__(self):
        self._client: Optional[Exa] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._api_key: Optional[str] = None
        self._status = AdapterStatus.UNINITIALIZED
        self._config: dict[str, Any] = {}
        self._mock_mode: bool = False
        self._stats = {
            "searches": 0,
            "contents_fetched": 0,
            "total_results": 0,
            "company_researches": 0,
            "code_searches": 0,
            "linkedin_searches": 0,
            "research_tasks": 0,
            "websets_created": 0,
            "total_cost_dollars": 0.0,
            "avg_latency_ms": 0.0,
            "retries": 0,
        }
        # Retry configuration for transient errors
        self._retry_config = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=60.0,
            jitter=0.5,
        ) if RetryConfig else None

    @property
    def sdk_name(self) -> str:
        return "exa"

    @property
    def layer(self) -> SDKLayer:
        return SDKLayer.RESEARCH

    @property
    def available(self) -> bool:
        return EXA_AVAILABLE

    async def initialize(self, config: dict[str, Any]) -> AdapterResult:
        """Initialize Exa client with API key."""
        _logger.info("Initializing Exa adapter", adapter="exa")

        if not EXA_AVAILABLE:
            self._status = AdapterStatus.ERROR
            _logger.error("Exa SDK not installed", adapter="exa")
            return AdapterResult(
                success=False,
                error="Exa SDK not installed. Run: pip install exa-py"
            )

        try:
            api_key = config.get("api_key") or os.getenv("EXA_API_KEY")
            if not api_key:
                self._status = AdapterStatus.DEGRADED
                self._mock_mode = True
                _logger.warning("No API key provided, running in mock mode", adapter="exa")
                return AdapterResult(
                    success=True,
                    data={"status": "degraded", "reason": "No API key - mock mode"},
                )

            self._api_key = api_key
            self._client = Exa(api_key=api_key)
            # Initialize async HTTP client for direct API calls (context, websets)
            self._http_client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
            self._config = config
            self._status = AdapterStatus.READY
            self._mock_mode = False

            _logger.info("Exa adapter initialized successfully", adapter="exa", status="ready")
            return AdapterResult(
                success=True,
                data={
                    "status": "ready",
                    "features": [
                        "search", "get_contents", "find_similar", "find_similar_and_contents",
                        "search_and_contents", "answer", "stream_answer", "research",
                        "company_research", "linkedin_search", "code_search", "create_webset",
                        "get_webset", "list_websets", "optimize_query", "get_context"
                    ],
                    "search_types": ["fast", "auto", "neural", "keyword", "deep"],
                    "categories": [c.value for c in ExaCategory],
                    "content_options": ["text", "highlights", "summary", "context", "subpages", "extras"],
                    "freshness_control": "maxAgeHours (0=always livecrawl, -1=cache only, N=use cache if <N hours old)",
                    "research_models": [m.value for m in ExaResearchModel],
                    "version": "2.0+",
                }
            )
        except Exception as e:
            self._status = AdapterStatus.ERROR
            _logger.exception("Failed to initialize Exa adapter", adapter="exa", error_type=type(e).__name__)
            return AdapterResult(success=False, error=str(e))

    async def execute(self, operation: str, **kwargs) -> AdapterResult:
        """Execute Exa operations with circuit breaker protection."""
        start_time = time.time()
        correlation_id = generate_correlation_id()

        operations = {
            "search": self._search,
            "get_contents": self._get_contents,
            "find_similar": self._find_similar,
            "find_similar_and_contents": self._find_similar_and_contents,
            "search_and_contents": self._search_and_contents,
            "answer": self._answer,
            "stream_answer": self._stream_answer,
            "research": self._research,
            "company_research": self._company_research,
            "linkedin_search": self._linkedin_search,
            "code_search": self._code_search,
            "create_webset": self._create_webset,
            "get_webset": self._get_webset,
            "list_websets": self._list_websets,
            "optimize_query": self._optimize_query,
            "get_context": self._get_context,
        }

        if operation not in operations:
            _logger.warning(
                "Unknown operation requested",
                adapter="exa",
                operation=operation,
                correlation_id=correlation_id,
            )
            return AdapterResult(
                success=False,
                error=f"Unknown operation: {operation}. Valid: {list(operations.keys())}"
            )

        # Log operation start (with sampling for high-volume operations)
        _logger.debug(
            "Executing Exa operation",
            adapter="exa",
            operation=operation,
            correlation_id=correlation_id,
            sample_rate=0.1,  # Sample 10% of debug logs
        )

        # Execute with circuit breaker protection
        try:
            async with adapter_circuit_breaker("exa_adapter"):
                result = await operations[operation](**kwargs)
                result.latency_ms = (time.time() - start_time) * 1000

                # Log successful completion
                _logger.info(
                    "Exa operation completed",
                    adapter="exa",
                    operation=operation,
                    correlation_id=correlation_id,
                    duration_ms=result.latency_ms,
                    success=result.success,
                    cached=result.cached,
                    sample_rate=0.5,  # Sample 50% of success logs
                )
                return result
        except CircuitOpenError as e:
            latency_ms = (time.time() - start_time) * 1000
            _logger.warning(
                "Circuit breaker open",
                adapter="exa",
                operation=operation,
                correlation_id=correlation_id,
                duration_ms=latency_ms,
            )
            # Circuit is open - return fallback response
            return AdapterResult(
                success=False,
                error=f"Circuit breaker open for exa_adapter: {e}",
                latency_ms=latency_ms,
                metadata={"circuit_breaker": "open", "adapter": "exa", "correlation_id": correlation_id},
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            _logger.exception(
                "Exa operation failed",
                adapter="exa",
                operation=operation,
                correlation_id=correlation_id,
                duration_ms=latency_ms,
                error_type=type(e).__name__,
            )
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=latency_ms,
                metadata={"correlation_id": correlation_id},
            )

    def _parse_entities(self, result: Any) -> List[Dict[str, Any]]:
        """Parse company/person entities from search results."""
        entities = getattr(result, "entities", [])
        if not entities:
            return []
        return [
            {
                "type": e.get("type") if isinstance(e, dict) else getattr(e, "type", None),
                "name": e.get("name") if isinstance(e, dict) else getattr(e, "name", None),
                "description": e.get("description") if isinstance(e, dict) else getattr(e, "description", None),
                "url": e.get("url") if isinstance(e, dict) else getattr(e, "url", None),
            }
            for e in entities
        ]

    def _track_cost(self, response: Any) -> Optional[float]:
        """Extract and track cost from response."""
        cost = getattr(response, "cost_dollars", None)
        if cost is not None:
            self._stats["total_cost_dollars"] += cost
        return cost

    async def _search(
        self,
        query: str,
        type: str = "auto",
        num_results: int = 10,
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        use_autoprompt: bool = True,
        category: Optional[str] = None,
        livecrawl: Optional[str] = None,
        include_text: Optional[list[str]] = None,
        exclude_text: Optional[list[str]] = None,
        moderation: bool = False,
        user_location: Optional[str] = None,
        additional_queries: Optional[list[str]] = None,
        max_age_hours: Optional[int] = None,
        max_characters: Optional[int] = None,
        verbosity: Optional[str] = None,
        include_sections: Optional[List[str]] = None,
        exclude_sections: Optional[List[str]] = None,
        # Advanced highlight options
        highlights: Optional[bool] = None,
        highlights_query: Optional[str] = None,
        highlights_max_chars: Optional[int] = None,
        highlights_num_sentences: Optional[int] = None,
        highlights_per_url: Optional[int] = None,
        # Summary options
        summary: Optional[bool] = None,
        summary_query: Optional[str] = None,
        summary_schema: Optional[Dict[str, Any]] = None,
        # Subpages options
        subpages: int = 0,
        subpage_target: Optional[str] = None,
        # Extras options
        extras_links: int = 0,
        extras_image_links: int = 0,
        # Livecrawl timeout
        livecrawl_timeout: Optional[int] = None,
        # Context mode (combines all results)
        context: Optional[bool] = None,
        context_max_chars: Optional[int] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Execute neural search with full Exa 2.0+ capabilities.

        Args:
            query: Search query (for neural, phrase as "Here is X:" for best results)
            type: Search type - "auto", "neural", "keyword", "fast", "deep"
            num_results: Number of results (1-100, Enterprise can go higher)
            include_domains: Only search these domains (max 1200, supports paths)
            exclude_domains: Exclude these domains (max 1200)
            start_published_date: Filter by publish date (ISO 8601: YYYY-MM-DD)
            end_published_date: Filter by publish date (ISO 8601)
            start_crawl_date: Filter by crawl/discovery date (ISO 8601)
            end_crawl_date: Filter by crawl date (ISO 8601)
            use_autoprompt: Let Exa optimize the query for neural search
            category: Focus category - company, research paper, news, tweet,
                      personal site, financial report, people, pdf, github, papers,
                      news article, movie, song. Note: company/people have filter limits
            livecrawl: Deprecated - use max_age_hours instead
            include_text: Strings that must appear (max 1 string, 5 words)
            exclude_text: Strings to exclude from results
            moderation: Filter unsafe content
            user_location: Two-letter ISO country code for localization
            additional_queries: Extra queries for deep search expansion (max 5)
            max_age_hours: Freshness control:
                - 0: Always livecrawl (real-time)
                - positive: Use cache if fresher than N hours
                - -1: Cache only, never livecrawl
                - None: Default fallback behavior
            max_characters: Maximum characters for text content
            verbosity: Content verbosity - "compact", "standard", "full"
            include_sections: Semantic sections to include - ["body", "header"]
            exclude_sections: Semantic sections to exclude - ["navigation", "footer"]
            highlights: Enable highlight extraction
            highlights_query: Custom query for highlight selection
            highlights_max_chars: Maximum characters for highlights
            highlights_num_sentences: Sentences per highlight (deprecated)
            highlights_per_url: Highlights per result (deprecated)
            summary: Enable AI summary generation
            summary_query: Custom query for summary direction
            summary_schema: JSON schema for structured summary output
            subpages: Number of subpages to crawl (0-10)
            subpage_target: Keywords to prioritize specific subpages
            extras_links: Number of URLs to extract from each page
            extras_image_links: Number of images to extract per result
            livecrawl_timeout: Timeout in ms for livecrawl (default 10000)
            context: Combine all results into one string (good for RAG)
            context_max_chars: Max chars for context (recommended 10000+)
        """
        self._stats["searches"] += 1

        if self._mock_mode or not self._client:
            return AdapterResult(
                success=True,
                data={
                    "results": [{
                        "title": f"Mock result for: {query}",
                        "url": "https://example.com/mock",
                        "text": f"This is a mock result for query: {query}",
                        "score": 0.95,
                    }],
                    "type": type,
                    "mock": True,
                }
            )

        # Build search parameters
        search_params: Dict[str, Any] = {
            "query": query,
            "num_results": min(num_results, 100),
        }

        # Pass search type directly to API (no remapping)
        # Valid types: "fast", "neural", "keyword", "auto", "deep"
        if type in ["fast", "neural", "keyword", "auto", "deep"]:
            search_params["type"] = type
        else:
            search_params["type"] = "auto"

        # Handle deep search with additional queries
        if type == "deep" and additional_queries:
            search_params["additional_queries"] = additional_queries[:5]

        # Autoprompt - only for neural/auto search
        if use_autoprompt and type in ["neural", "auto"]:
            search_params["use_autoprompt"] = True

        # Domain filtering (max 1200 each)
        if include_domains:
            search_params["include_domains"] = include_domains[:1200]
        if exclude_domains:
            search_params["exclude_domains"] = exclude_domains[:1200]

        # Date filtering (ISO 8601 format)
        # Note: Not supported for 'company' and 'people' categories
        if category not in ["company", "people"]:
            if start_published_date:
                search_params["start_published_date"] = start_published_date
            if end_published_date:
                search_params["end_published_date"] = end_published_date
            if start_crawl_date:
                search_params["start_crawl_date"] = start_crawl_date
            if end_crawl_date:
                search_params["end_crawl_date"] = end_crawl_date
            if include_text:
                search_params["include_text"] = include_text[:1]
            if exclude_text:
                search_params["exclude_text"] = exclude_text

        # Category filter
        if category:
            search_params["category"] = category

        # Content moderation
        if moderation:
            search_params["moderation"] = True

        # User location for localized results
        if user_location:
            search_params["user_location"] = user_location

        # Freshness control (maxAgeHours is preferred over livecrawl)
        if max_age_hours is not None:
            search_params["max_age_hours"] = max_age_hours
        elif livecrawl and livecrawl != "fallback":
            # Legacy livecrawl parameter (deprecated)
            search_params["livecrawl"] = livecrawl

        # Livecrawl timeout
        if livecrawl_timeout:
            search_params["livecrawl_timeout"] = livecrawl_timeout

        # Build contents options
        contents_options: Dict[str, Any] = {}

        # Text extraction options
        if any([max_characters, verbosity, include_sections, exclude_sections]):
            text_opts: Dict[str, Any] = {}
            if max_characters:
                text_opts["maxCharacters"] = max_characters
            if verbosity:
                text_opts["verbosity"] = verbosity
            if include_sections:
                text_opts["includeSections"] = include_sections
            if exclude_sections:
                text_opts["excludeSections"] = exclude_sections
            contents_options["text"] = text_opts if text_opts else True

        # Highlights options
        if highlights:
            if any([highlights_query, highlights_max_chars, highlights_num_sentences, highlights_per_url]):
                highlight_opts: Dict[str, Any] = {}
                if highlights_query:
                    highlight_opts["query"] = highlights_query
                if highlights_max_chars:
                    highlight_opts["maxCharacters"] = highlights_max_chars
                if highlights_num_sentences:
                    highlight_opts["numSentences"] = highlights_num_sentences
                if highlights_per_url:
                    highlight_opts["highlightsPerUrl"] = highlights_per_url
                contents_options["highlights"] = highlight_opts
            else:
                contents_options["highlights"] = True

        # Summary options
        if summary:
            if any([summary_query, summary_schema]):
                summary_opts: Dict[str, Any] = {}
                if summary_query:
                    summary_opts["query"] = summary_query
                if summary_schema:
                    summary_opts["schema"] = summary_schema
                contents_options["summary"] = summary_opts
            else:
                contents_options["summary"] = True

        # Context mode (combines all results into one string)
        if context:
            if context_max_chars:
                contents_options["context"] = {"maxCharacters": context_max_chars}
            else:
                contents_options["context"] = True

        # Subpages crawling
        if subpages > 0:
            contents_options["subpages"] = min(subpages, 10)
            if subpage_target:
                contents_options["subpageTarget"] = subpage_target

        # Extras (links and images)
        if extras_links > 0 or extras_image_links > 0:
            extras: Dict[str, int] = {}
            if extras_links > 0:
                extras["links"] = extras_links
            if extras_image_links > 0:
                extras["imageLinks"] = extras_image_links
            contents_options["extras"] = extras

        # Add contents options if any were set
        if contents_options:
            search_params["contents"] = contents_options

        # Execute search with retry logic (wrap sync call in executor)
        async def _do_search():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return await loop.run_in_executor(
                None,
                lambda: self._client.search(**search_params)
            )

        if retry_async and self._retry_config:
            def _on_retry(attempt, exc, delay):
                self._stats["retries"] += 1
            config = RetryConfig(
                max_retries=self._retry_config.max_retries,
                base_delay=self._retry_config.base_delay,
                max_delay=self._retry_config.max_delay,
                jitter=self._retry_config.jitter,
                on_retry=_on_retry,
            )
            response = await retry_async(_do_search, config=config)
        else:
            response = await _do_search()

        # Track cost
        cost_dollars = self._track_cost(response)

        # Parse results with all fields
        results = []
        for r in response.results:
            result_dict = {
                "id": getattr(r, "id", None),
                "title": r.title,
                "url": r.url,
                "text": getattr(r, "text", ""),
                "highlights": getattr(r, "highlights", []),
                "highlight_scores": getattr(r, "highlight_scores", []),
                "score": getattr(r, "score", 0.0),
                "published_date": getattr(r, "published_date", None),
                "author": getattr(r, "author", None),
                "image": getattr(r, "image", None),
                "favicon": getattr(r, "favicon", None),
                "subpages": getattr(r, "subpages", None),
                "extras": getattr(r, "extras", None),
                "entities": self._parse_entities(r),
                "summary": getattr(r, "summary", None),
            }
            # Extract links and imageLinks from extras if present
            extras = getattr(r, "extras", None)
            if extras:
                result_dict["links"] = extras.get("links", []) if isinstance(extras, dict) else getattr(extras, "links", [])
                result_dict["image_links"] = extras.get("imageLinks", []) if isinstance(extras, dict) else getattr(extras, "imageLinks", [])
            results.append(result_dict)

        self._stats["total_results"] += len(results)

        # Build response data
        response_data: Dict[str, Any] = {
            "results": results,
            "count": len(results),
            "type": type,
            "autoprompt_string": getattr(response, "autoprompt_string", None),
            "search_type": getattr(response, "search_type", None),  # For auto mode
            "cost_dollars": cost_dollars,
            "request_id": getattr(response, "request_id", None),
        }

        # Include context if requested
        if context:
            response_data["context"] = getattr(response, "context", None)

        return AdapterResult(
            success=True,
            data=response_data
        )

    async def _get_contents(
        self,
        urls: list[str],
        text: bool = True,
        highlights: bool = False,
        summary: bool = False,
        subpages: int = 0,
        subpage_target: Optional[str] = None,
        livecrawl: Optional[str] = None,
        max_age_hours: Optional[int] = None,
        livecrawl_timeout: Optional[int] = None,
        max_characters: Optional[int] = None,
        verbosity: Optional[str] = None,
        include_sections: Optional[List[str]] = None,
        exclude_sections: Optional[List[str]] = None,
        include_html_tags: bool = False,
        highlights_query: Optional[str] = None,
        highlights_max_chars: Optional[int] = None,
        summary_query: Optional[str] = None,
        summary_schema: Optional[Dict[str, Any]] = None,
        extras_links: int = 0,
        extras_image_links: int = 0,
        **kwargs,
    ) -> AdapterResult:
        """
        Get contents from URLs with full extraction options.

        Args:
            urls: List of URLs to fetch (can also pass Result objects)
            text: Include full text extraction
            highlights: Include highlight snippets
            summary: Generate AI summary
            subpages: Number of subpages to crawl (0-10)
            subpage_target: Keywords to prioritize specific subpages
            livecrawl: Deprecated - use max_age_hours instead
            max_age_hours: Freshness control (0=always livecrawl, -1=cache only)
            livecrawl_timeout: Timeout in ms for livecrawl (default 10000)
            max_characters: Maximum characters for text content
            verbosity: Content verbosity - "compact", "standard", "full"
            include_sections: Semantic sections to include - ["body", "header"]
            exclude_sections: Semantic sections to exclude - ["navigation", "footer"]
            include_html_tags: Include HTML formatting in text
            highlights_query: Custom query for highlight selection
            highlights_max_chars: Maximum characters for highlights
            summary_query: Custom query for summary direction
            summary_schema: JSON schema for structured summary output
            extras_links: Number of URLs to extract from each page
            extras_image_links: Number of images to extract per result
        """
        self._stats["contents_fetched"] += len(urls)

        if self._mock_mode or not self._client:
            return AdapterResult(
                success=True,
                data={
                    "contents": [{"url": url, "text": "Mock content"} for url in urls],
                    "mock": True,
                }
            )

        # Build content options
        contents_options: Dict[str, Any] = {}

        # Text extraction options
        if text:
            text_options: Dict[str, Any] = {"include": True}
            if max_characters:
                text_options["maxCharacters"] = max_characters
            if verbosity:
                text_options["verbosity"] = verbosity
            if include_sections:
                text_options["includeSections"] = include_sections
            if exclude_sections:
                text_options["excludeSections"] = exclude_sections
            if include_html_tags:
                text_options["includeHtmlTags"] = True
            contents_options["text"] = text_options if len(text_options) > 1 else True

        # Highlights options
        if highlights:
            if highlights_query or highlights_max_chars:
                highlight_opts: Dict[str, Any] = {}
                if highlights_query:
                    highlight_opts["query"] = highlights_query
                if highlights_max_chars:
                    highlight_opts["maxCharacters"] = highlights_max_chars
                contents_options["highlights"] = highlight_opts
            else:
                contents_options["highlights"] = {"num_sentences": 3}

        # Summary options
        if summary:
            if summary_query or summary_schema:
                summary_opts: Dict[str, Any] = {}
                if summary_query:
                    summary_opts["query"] = summary_query
                if summary_schema:
                    summary_opts["schema"] = summary_schema
                contents_options["summary"] = summary_opts
            else:
                contents_options["summary"] = True

        # Subpages crawling
        if subpages > 0:
            contents_options["subpages"] = min(subpages, 10)
            if subpage_target:
                contents_options["subpageTarget"] = subpage_target

        # Extras (links and images)
        if extras_links > 0 or extras_image_links > 0:
            extras: Dict[str, int] = {}
            if extras_links > 0:
                extras["links"] = extras_links
            if extras_image_links > 0:
                extras["imageLinks"] = extras_image_links
            contents_options["extras"] = extras

        # Freshness control (maxAgeHours is preferred over livecrawl)
        if max_age_hours is not None:
            contents_options["max_age_hours"] = max_age_hours
        elif livecrawl and livecrawl != "fallback":
            contents_options["livecrawl"] = livecrawl

        # Livecrawl timeout
        if livecrawl_timeout:
            contents_options["livecrawl_timeout"] = livecrawl_timeout

        # Wrap sync call in executor to avoid blocking event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        response = await loop.run_in_executor(
            None,
            lambda: self._client.get_contents(urls, **contents_options)
        )

        # Track cost
        cost_dollars = self._track_cost(response)

        contents = []
        for r in response.results:
            content_dict: Dict[str, Any] = {
                "id": getattr(r, "id", None),
                "url": r.url,
                "title": getattr(r, "title", ""),
                "text": getattr(r, "text", ""),
                "highlights": getattr(r, "highlights", []),
                "highlight_scores": getattr(r, "highlight_scores", []),
                "summary": getattr(r, "summary", ""),
                "published_date": getattr(r, "published_date", None),
                "author": getattr(r, "author", None),
                "image": getattr(r, "image", None),
                "favicon": getattr(r, "favicon", None),
                "subpages": getattr(r, "subpages", None),
                "entities": self._parse_entities(r),
            }
            # Extract links and imageLinks from extras if present
            extras_data = getattr(r, "extras", None)
            if extras_data:
                content_dict["links"] = extras_data.get("links", []) if isinstance(extras_data, dict) else getattr(extras_data, "links", [])
                content_dict["image_links"] = extras_data.get("imageLinks", []) if isinstance(extras_data, dict) else getattr(extras_data, "imageLinks", [])
            contents.append(content_dict)

        return AdapterResult(
            success=True,
            data={
                "contents": contents,
                "count": len(contents),
                "cost_dollars": cost_dollars,
                "request_id": getattr(response, "request_id", None),
            }
        )

    async def _find_similar(
        self,
        url: str,
        num_results: int = 10,
        exclude_source_domain: bool = True,
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
        category: Optional[str] = None,
        **kwargs,
    ) -> AdapterResult:
        """Find content similar to a given URL."""
        if self._mock_mode or not self._client:
            return AdapterResult(
                success=True,
                data={"results": [], "mock": True}
            )

        params = {
            "url": url,
            "num_results": num_results,
            "exclude_source_domain": exclude_source_domain,
        }
        if include_domains:
            params["include_domains"] = include_domains
        if exclude_domains:
            params["exclude_domains"] = exclude_domains
        if category:
            params["category"] = category

        # Wrap sync call in executor with retry logic
        async def _do_find_similar():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return await loop.run_in_executor(
                None,
                lambda: self._client.find_similar(**params)
            )

        if retry_async and self._retry_config:
            def _on_retry(attempt, exc, delay):
                self._stats["retries"] += 1
            config = RetryConfig(
                max_retries=self._retry_config.max_retries,
                base_delay=self._retry_config.base_delay,
                max_delay=self._retry_config.max_delay,
                jitter=self._retry_config.jitter,
                on_retry=_on_retry,
            )
            response = await retry_async(_do_find_similar, config=config)
        else:
            response = await _do_find_similar()

        # Track cost
        cost_dollars = self._track_cost(response)

        results = []
        for r in response.results:
            results.append({
                "title": r.title,
                "url": r.url,
                "score": getattr(r, "score", 0.0),
                "image": getattr(r, "image", None),
                "favicon": getattr(r, "favicon", None),
                "entities": self._parse_entities(r),
            })

        return AdapterResult(
            success=True,
            data={
                "results": results,
                "source_url": url,
                "cost_dollars": cost_dollars,
            }
        )

    async def _find_similar_and_contents(
        self,
        url: str,
        num_results: int = 10,
        exclude_source_domain: bool = True,
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
        category: Optional[str] = None,
        text: bool = True,
        highlights: bool = False,
        summary: bool = False,
        max_characters: Optional[int] = None,
        highlights_query: Optional[str] = None,
        highlights_max_chars: Optional[int] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Find similar content and extract contents in one call.

        Args:
            url: Source URL to find similar content for
            num_results: Number of results to return
            exclude_source_domain: Exclude the source URL's domain from results
            include_domains: Only include these domains
            exclude_domains: Exclude these domains
            category: Focus category for results
            text: Include full text extraction
            highlights: Include highlight snippets
            summary: Include AI-generated summary
            max_characters: Maximum characters for text
            highlights_query: Custom query for highlight selection
            highlights_max_chars: Maximum characters for highlights
        """
        if self._mock_mode or not self._client:
            return AdapterResult(
                success=True,
                data={"results": [], "source_url": url, "mock": True}
            )

        # Build params
        params: Dict[str, Any] = {
            "url": url,
            "num_results": num_results,
            "exclude_source_domain": exclude_source_domain,
        }
        if include_domains:
            params["include_domains"] = include_domains
        if exclude_domains:
            params["exclude_domains"] = exclude_domains
        if category:
            params["category"] = category

        # Build contents options
        contents: Dict[str, Any] = {}
        if text:
            if max_characters:
                contents["text"] = {"maxCharacters": max_characters}
            else:
                contents["text"] = True
        if highlights:
            if highlights_query or highlights_max_chars:
                highlight_opts: Dict[str, Any] = {}
                if highlights_query:
                    highlight_opts["query"] = highlights_query
                if highlights_max_chars:
                    highlight_opts["maxCharacters"] = highlights_max_chars
                contents["highlights"] = highlight_opts
            else:
                contents["highlights"] = True
        if summary:
            contents["summary"] = True

        if contents:
            params["contents"] = contents

        # Execute with SDK's find_similar_and_contents if available
        async def _do_find_similar():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            # Try find_similar_and_contents first, fallback to find_similar
            if hasattr(self._client, 'find_similar_and_contents'):
                return await loop.run_in_executor(
                    None,
                    lambda: self._client.find_similar_and_contents(**params)
                )
            else:
                # Fallback: find_similar with contents parameter
                return await loop.run_in_executor(
                    None,
                    lambda: self._client.find_similar(**params)
                )

        if retry_async and self._retry_config:
            def _on_retry(attempt, exc, delay):
                self._stats["retries"] += 1
            config = RetryConfig(
                max_retries=self._retry_config.max_retries,
                base_delay=self._retry_config.base_delay,
                max_delay=self._retry_config.max_delay,
                jitter=self._retry_config.jitter,
                on_retry=_on_retry,
            )
            response = await retry_async(_do_find_similar, config=config)
        else:
            response = await _do_find_similar()

        # Track cost
        cost_dollars = self._track_cost(response)

        results = []
        for r in response.results:
            results.append({
                "title": r.title,
                "url": r.url,
                "text": getattr(r, "text", ""),
                "highlights": getattr(r, "highlights", []),
                "summary": getattr(r, "summary", ""),
                "score": getattr(r, "score", 0.0),
                "published_date": getattr(r, "published_date", None),
                "image": getattr(r, "image", None),
                "favicon": getattr(r, "favicon", None),
                "entities": self._parse_entities(r),
            })

        return AdapterResult(
            success=True,
            data={
                "results": results,
                "source_url": url,
                "count": len(results),
                "cost_dollars": cost_dollars,
            }
        )

    async def _search_and_contents(
        self,
        query: str,
        num_results: int = 5,
        type: str = "auto",
        text: bool = True,
        highlights: bool = True,
        summary: bool = False,
        max_characters: Optional[int] = None,
        verbosity: Optional[str] = None,
        include_sections: Optional[List[str]] = None,
        exclude_sections: Optional[List[str]] = None,
        max_age_hours: Optional[int] = None,
        **kwargs,
    ) -> AdapterResult:
        """Combined search and content extraction."""
        if self._mock_mode or not self._client:
            return AdapterResult(
                success=True,
                data={"results": [], "mock": True}
            )

        # Build highlights config
        highlights_config = {"num_sentences": 3} if highlights else None

        # Build text options
        text_options: Any = text
        if text and any([max_characters, verbosity, include_sections, exclude_sections]):
            text_options = {"include": True}
            if max_characters:
                text_options["maxCharacters"] = max_characters
            if verbosity:
                text_options["verbosity"] = verbosity
            if include_sections:
                text_options["includeSections"] = include_sections
            if exclude_sections:
                text_options["excludeSections"] = exclude_sections

        params: Dict[str, Any] = {
            "query": query,
            "num_results": num_results,
            "type": type,  # Pass through directly
            "text": text_options,
            "highlights": highlights_config,
            "summary": summary,
        }
        if max_age_hours:
            params["max_age_hours"] = max_age_hours

        # Wrap sync call in executor to avoid blocking event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        response = await loop.run_in_executor(
            None,
            lambda: self._client.search_and_contents(**params)
        )

        # Track cost
        cost_dollars = self._track_cost(response)

        results = []
        for r in response.results:
            results.append({
                "title": r.title,
                "url": r.url,
                "text": getattr(r, "text", ""),
                "highlights": getattr(r, "highlights", []),
                "summary": getattr(r, "summary", ""),
                "score": getattr(r, "score", 0.0),
                "published_date": getattr(r, "published_date", None),
                "author": getattr(r, "author", None),
                "image": getattr(r, "image", None),
                "favicon": getattr(r, "favicon", None),
                "subpages": getattr(r, "subpages", None),
                "extras": getattr(r, "extras", None),
                "entities": self._parse_entities(r),
            })

        return AdapterResult(
            success=True,
            data={
                "results": results,
                "count": len(results),
                "cost_dollars": cost_dollars,
            }
        )

    async def _answer(
        self,
        query: str,
        output_schema: Optional[Dict[str, Any]] = None,
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
        category: Optional[str] = None,
        num_results: int = 10,
        max_age_hours: Optional[int] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Ask a question and get an AI-generated answer with citations.

        Uses Exa's search and LLM capabilities to generate answers from
        web content with proper citations.

        Args:
            query: The question to answer
            output_schema: JSON schema for structured answer output
            include_domains: Only search these domains for sources
            exclude_domains: Exclude these domains from sources
            category: Focus category for source search
            num_results: Number of source results to consider
            max_age_hours: Freshness control for sources
        """
        self._stats["searches"] += 1

        if self._mock_mode or not self._client:
            return AdapterResult(
                success=True,
                data={
                    "answer": f"Mock answer for: {query}",
                    "citations": [],
                    "mock": True,
                }
            )

        try:
            # Build answer parameters
            answer_params: Dict[str, Any] = {"query": query}

            if output_schema:
                answer_params["output_schema"] = output_schema
            if include_domains:
                answer_params["include_domains"] = include_domains
            if exclude_domains:
                answer_params["exclude_domains"] = exclude_domains
            if category:
                answer_params["category"] = category
            if num_results != 10:
                answer_params["num_results"] = num_results
            if max_age_hours is not None:
                answer_params["max_age_hours"] = max_age_hours

            # Wrap sync call in executor to avoid blocking event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            response = await loop.run_in_executor(
                None,
                lambda: self._client.answer(**answer_params)
            )
            cost_dollars = self._track_cost(response)

            # Parse answer - can be string or dict if schema provided
            answer = getattr(response, "answer", "")

            # Parse citations with full metadata
            citations = []
            for c in getattr(response, "citations", []):
                if hasattr(c, 'url'):
                    citations.append({
                        "url": c.url,
                        "title": getattr(c, "title", ""),
                        "text": getattr(c, "text", ""),
                        "highlights": getattr(c, "highlights", []),
                    })
                elif isinstance(c, dict):
                    citations.append(c)
                else:
                    citations.append({"url": str(c)})

            return AdapterResult(
                success=True,
                data={
                    "answer": answer,
                    "citations": citations,
                    "cost_dollars": cost_dollars,
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _stream_answer(
        self,
        query: str,
        **kwargs,
    ) -> AdapterResult:
        """
        Stream an answer with real-time token generation.

        Args:
            query: The question to answer

        Returns:
            Result with streaming generator in data["stream"]
        """
        self._stats["searches"] += 1

        if self._mock_mode or not self._client:
            return AdapterResult(
                success=True,
                data={
                    "answer": f"Mock streaming answer for: {query}",
                    "stream": None,
                    "mock": True,
                }
            )

        try:
            if hasattr(self._client, 'stream_answer'):
                # Collect streamed response
                full_answer = ""
                citations = []
                cost_dollars = None

                for chunk in self._client.stream_answer(query):
                    if hasattr(chunk, 'text'):
                        full_answer += chunk.text
                    if hasattr(chunk, 'citations'):
                        citations = chunk.citations
                    if hasattr(chunk, 'cost_dollars') and chunk.cost_dollars:
                        cost_dollars = chunk.cost_dollars
                        self._stats["total_cost_dollars"] += cost_dollars

                return AdapterResult(
                    success=True,
                    data={
                        "answer": full_answer,
                        "citations": citations,
                        "streamed": True,
                        "cost_dollars": cost_dollars,
                    }
                )
            else:
                # Fallback to regular answer
                return await self._answer(query)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _research(
        self,
        instructions: str,
        output_schema: Optional[dict] = None,
        infer_schema: bool = False,
        model: str = "exa-research",
        poll_interval: float = 2.0,
        max_wait_seconds: int = 60,
        **kwargs,
    ) -> AdapterResult:
        """
        Perform complex async research with structured output.

        This uses Exa's Research API for multi-step intelligent research.
        Supports three model tiers for different speed/quality tradeoffs.

        Args:
            instructions: Research instructions/question
            output_schema: JSON schema for structured output
            infer_schema: Let Exa infer the output schema
            model: Research model tier:
                - "exa-research-fast": Fastest, lower quality
                - "exa-research": Balanced (default)
                - "exa-research-pro": Highest quality, slowest
            poll_interval: Seconds between status checks (default 2.0)
            max_wait_seconds: Maximum wait time before timeout (default 60)
        """
        self._stats["research_tasks"] += 1

        if not self._client:
            return AdapterResult(
                success=True,
                data={
                    "result": f"Mock research for: {instructions}",
                    "task_id": "mock-task-id",
                    "mock": True,
                }
            )

        try:
            # Check if research API is available
            if hasattr(self._client, 'research'):
                params: Dict[str, Any] = {"instructions": instructions}

                # Add model selection
                if model in ["exa-research-fast", "exa-research", "exa-research-pro"]:
                    params["model"] = model

                # Add schema options
                if output_schema:
                    params["output_schema"] = output_schema
                elif infer_schema:
                    params["infer_schema"] = True

                # Start research task
                task = self._client.research.create(**params)
                task_id = getattr(task, 'id', None)

                # Poll for completion (with timeout)
                max_attempts = int(max_wait_seconds / poll_interval)
                for attempt in range(max_attempts):
                    status = self._client.research.get(task_id)
                    task_status = getattr(status, 'status', '')

                    if task_status == 'completed':
                        cost_dollars = getattr(status, 'cost_dollars', None)
                        if cost_dollars:
                            self._stats["total_cost_dollars"] += cost_dollars

                        return AdapterResult(
                            success=True,
                            data={
                                "result": getattr(status, 'data', {}),
                                "citations": getattr(status, 'citations', []),
                                "task_id": task_id,
                                "model": model,
                                "cost_dollars": cost_dollars,
                            }
                        )
                    elif task_status == 'failed':
                        return AdapterResult(
                            success=False,
                            error=f"Research task failed: {getattr(status, 'error', 'Unknown')}"
                        )
                    elif task_status in ['pending', 'running']:
                        await asyncio.sleep(poll_interval)
                    else:
                        # Unknown status, continue polling
                        await asyncio.sleep(poll_interval)

                return AdapterResult(
                    success=False,
                    error=f"Research task timed out after {max_wait_seconds}s"
                )
            else:
                # Fallback to search + answer
                search_result = await self._search_and_contents(
                    instructions, num_results=10, summary=True
                )
                if search_result.success:
                    answer_result = await self._answer(instructions)
                    return AdapterResult(
                        success=True,
                        data={
                            "result": answer_result.data.get("answer", ""),
                            "sources": search_result.data.get("results", []),
                            "citations": answer_result.data.get("citations", []),
                            "fallback": True,
                        }
                    )
                return search_result
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _company_research(
        self,
        domain: str,
        **kwargs,
    ) -> AdapterResult:
        """
        Comprehensive company research by domain.

        Crawls company website to gather detailed intelligence about the business.

        Args:
            domain: Company domain (e.g., "anthropic.com")
        """
        self._stats["company_researches"] += 1

        if not self._client:
            return AdapterResult(
                success=True,
                data={
                    "company": domain,
                    "info": f"Mock company info for {domain}",
                    "mock": True,
                }
            )

        try:
            # Use company category search
            search_result = await self._search(
                query=f"site:{domain}",
                category="company",
                num_results=20,
                livecrawl="preferred",
            )

            if not search_result.success:
                return search_result

            # Get contents from top results
            urls = [r["url"] for r in search_result.data.get("results", [])[:10]]
            if urls:
                contents_result = await self._get_contents(
                    urls=urls,
                    text=True,
                    summary=True,
                )

                return AdapterResult(
                    success=True,
                    data={
                        "company": domain,
                        "pages": contents_result.data.get("contents", []),
                        "search_results": search_result.data.get("results", []),
                    }
                )

            return AdapterResult(
                success=True,
                data={
                    "company": domain,
                    "pages": [],
                    "search_results": search_result.data.get("results", []),
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _linkedin_search(
        self,
        query: str,
        search_type: str = "people",
        num_results: int = 10,
        **kwargs,
    ) -> AdapterResult:
        """
        Search LinkedIn for companies and people.

        Args:
            query: Search query (name, title, company, etc.)
            search_type: "people" or "company"
            num_results: Number of results
        """
        self._stats["linkedin_searches"] += 1

        if not self._client:
            return AdapterResult(
                success=True,
                data={
                    "results": [{
                        "name": f"Mock LinkedIn result for: {query}",
                        "url": "https://linkedin.com/in/mock",
                    }],
                    "mock": True,
                }
            )

        try:
            # Search LinkedIn domain
            linkedin_query = f"{query} site:linkedin.com"
            if search_type == "people":
                linkedin_query = f"{query} site:linkedin.com/in/"
            elif search_type == "company":
                linkedin_query = f"{query} site:linkedin.com/company/"

            return await self._search(
                query=linkedin_query,
                num_results=num_results,
                include_domains=["linkedin.com"],
                category="people" if search_type == "people" else None,
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _code_search(
        self,
        query: str,
        language: Optional[str] = None,
        tokens_num: int = 5000,
        include_domains: Optional[list[str]] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Search for code snippets, examples, and documentation using /context endpoint.

        This uses Exa's /context endpoint which is optimized for code and documentation.

        Args:
            query: Code-related search query
            language: Programming language filter
            tokens_num: Number of tokens to return (1000-50000)
            include_domains: Specific domains (defaults to code-related sites)
        """
        self._stats["code_searches"] += 1

        if self._mock_mode or not self._http_client:
            return AdapterResult(
                success=True,
                data={
                    "results": [{
                        "title": f"Mock code result for: {query}",
                        "url": "https://github.com/mock/repo",
                        "text": f"// Mock code for {query}\nfunction example() {{}}"
                    }],
                    "mock": True,
                }
            )

        try:
            # Enhance query with language if specified
            enhanced_query = query
            if language:
                enhanced_query = f"{query} {language}"

            # Use /context endpoint for code search
            request_data: Dict[str, Any] = {
                "query": enhanced_query,
                "tokensNum": max(1000, min(tokens_num, 50000)),
            }

            # Add domain filtering if specified
            if include_domains:
                request_data["includeDomains"] = include_domains
            else:
                # Default code-focused domains
                request_data["includeDomains"] = [
                    "github.com",
                    "stackoverflow.com",
                    "docs.python.org",
                    "developer.mozilla.org",
                    "npmjs.com",
                    "pypi.org",
                    "crates.io",
                    "pkg.go.dev",
                ]

            response = await self._http_client.post("/context", json=request_data)
            response.raise_for_status()
            data = response.json()

            # Parse context results
            results = []
            for r in data.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "text": r.get("text", ""),
                    "score": r.get("score", 0.0),
                    "highlights": r.get("highlights", []),
                    "published_date": r.get("publishedDate"),
                    "author": r.get("author"),
                    "entities": self._parse_entities(r) if r.get("entities") else [],
                })

            cost_dollars = data.get("cost_dollars")
            if cost_dollars:
                self._stats["total_cost_dollars"] += cost_dollars

            return AdapterResult(
                success=True,
                data={
                    "results": results,
                    "count": len(results),
                    "tokens_used": data.get("tokensUsed"),
                    "cost_dollars": cost_dollars,
                    "endpoint": "/context",
                }
            )
        except httpx.HTTPStatusError as e:
            # Fallback to search_and_contents if /context fails
            return await self._code_search_fallback(query, language, include_domains)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _code_search_fallback(
        self,
        query: str,
        language: Optional[str] = None,
        include_domains: Optional[list[str]] = None,
    ) -> AdapterResult:
        """Fallback code search using search_and_contents."""
        enhanced_query = query
        if language:
            enhanced_query = f"{query} {language}"

        if not include_domains:
            include_domains = [
                "github.com",
                "stackoverflow.com",
                "docs.python.org",
                "developer.mozilla.org",
                "npmjs.com",
                "pypi.org",
                "crates.io",
                "pkg.go.dev",
            ]

        result = await self._search_and_contents(
            query=enhanced_query,
            num_results=10,
            type="neural",
            text=True,
            highlights=True,
        )

        # Filter for include_domains if needed
        if result.success and include_domains:
            filtered_results = []
            for r in result.data.get("results", []):
                url = r.get("url", "")
                if any(domain in url for domain in include_domains):
                    filtered_results.append(r)
            result.data["results"] = filtered_results
            result.data["count"] = len(filtered_results)
            result.data["fallback"] = True

        return result

    async def _create_webset(
        self,
        search_query: str,
        count: int = 10,
        criteria: Optional[List[str]] = None,
        entity_type: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Create a Webset for structured data collection.

        Websets allow you to define criteria and collect structured data from the web.

        Args:
            search_query: Query to find matching pages
            count: Number of results to collect (default 10)
            criteria: List of criteria that results must meet
            entity_type: Type of entity to extract (e.g., "company", "person")
            schema: JSON schema for structured extraction
        """
        self._stats["websets_created"] += 1

        if self._mock_mode or not self._http_client:
            return AdapterResult(
                success=True,
                data={
                    "webset_id": "mock-webset-id",
                    "status": "pending",
                    "search_query": search_query,
                    "mock": True,
                }
            )

        try:
            # Build webset request
            request_data: Dict[str, Any] = {
                "search": {
                    "query": search_query,
                    "count": min(count, 100),
                }
            }

            if criteria:
                request_data["search"]["criteria"] = criteria
            if entity_type:
                request_data["entityType"] = entity_type
            if schema:
                request_data["schema"] = schema

            response = await self._http_client.post("/v0/websets", json=request_data)
            response.raise_for_status()
            data = response.json()

            cost_dollars = data.get("cost_dollars")
            if cost_dollars:
                self._stats["total_cost_dollars"] += cost_dollars

            return AdapterResult(
                success=True,
                data={
                    "webset_id": data.get("id"),
                    "status": data.get("status", "pending"),
                    "search_query": search_query,
                    "created_at": data.get("createdAt"),
                    "cost_dollars": cost_dollars,
                }
            )
        except httpx.HTTPStatusError as e:
            return AdapterResult(
                success=False,
                error=f"Webset creation failed: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _get_webset(
        self,
        webset_id: str,
        **kwargs,
    ) -> AdapterResult:
        """
        Get a Webset by ID, including its results if completed.

        Args:
            webset_id: The ID of the webset to retrieve
        """
        if self._mock_mode or not self._http_client:
            return AdapterResult(
                success=True,
                data={
                    "webset_id": webset_id,
                    "status": "completed",
                    "results": [],
                    "mock": True,
                }
            )

        try:
            response = await self._http_client.get(f"/v0/websets/{webset_id}")
            response.raise_for_status()
            data = response.json()

            return AdapterResult(
                success=True,
                data={
                    "webset_id": data.get("id"),
                    "status": data.get("status"),
                    "results": data.get("results", []),
                    "search_query": data.get("search", {}).get("query"),
                    "count": len(data.get("results", [])),
                    "created_at": data.get("createdAt"),
                    "completed_at": data.get("completedAt"),
                    "cost_dollars": data.get("cost_dollars"),
                }
            )
        except httpx.HTTPStatusError as e:
            return AdapterResult(
                success=False,
                error=f"Webset retrieval failed: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _list_websets(
        self,
        limit: int = 20,
        offset: int = 0,
        **kwargs,
    ) -> AdapterResult:
        """
        List all Websets for the account.

        Args:
            limit: Maximum number of websets to return
            offset: Offset for pagination
        """
        if self._mock_mode or not self._http_client:
            return AdapterResult(
                success=True,
                data={
                    "websets": [],
                    "total": 0,
                    "mock": True,
                }
            )

        try:
            response = await self._http_client.get(
                "/v0/websets",
                params={"limit": limit, "offset": offset}
            )
            response.raise_for_status()
            data = response.json()

            websets = []
            for w in data.get("websets", []):
                websets.append({
                    "id": w.get("id"),
                    "status": w.get("status"),
                    "search_query": w.get("search", {}).get("query"),
                    "results_count": len(w.get("results", [])),
                    "created_at": w.get("createdAt"),
                    "completed_at": w.get("completedAt"),
                })

            return AdapterResult(
                success=True,
                data={
                    "websets": websets,
                    "total": data.get("total", len(websets)),
                    "limit": limit,
                    "offset": offset,
                }
            )
        except httpx.HTTPStatusError as e:
            return AdapterResult(
                success=False,
                error=f"Webset listing failed: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _optimize_query(
        self,
        query: str,
        search_type: str = "neural",
        **kwargs,
    ) -> AdapterResult:
        """
        Optimize a query using Exa's autoprompt feature.

        This performs a minimal search with autoprompt enabled and returns
        the optimized query string that Exa generates.

        Best practices for neural search queries:
        - Phrase as statements: "Here is a great article about X:" works better
        - Add context modifiers: "funny", "academic", specific websites
        - End with colon ":" to mimic natural link sharing
        - Avoid question format for neural search

        Args:
            query: The original query to optimize
            search_type: Search type context - "neural" or "keyword"

        Returns:
            Result with optimized_query and tips for improvement
        """
        if self._mock_mode or not self._client:
            return AdapterResult(
                success=True,
                data={
                    "original_query": query,
                    "optimized_query": f"Here is a great resource about {query}:",
                    "tips": [
                        "Phrase queries as statements, not questions",
                        "Add context modifiers (academic, tutorial, etc.)",
                        "End with colon (:) for best neural search results",
                    ],
                    "mock": True,
                }
            )

        try:
            # Execute a minimal search with autoprompt to get the optimized string
            result = await self._search(
                query=query,
                type=search_type,
                num_results=1,
                use_autoprompt=True,
            )

            if result.success:
                autoprompt_string = result.data.get("autoprompt_string", query)
                return AdapterResult(
                    success=True,
                    data={
                        "original_query": query,
                        "optimized_query": autoprompt_string or query,
                        "search_type_used": result.data.get("search_type", search_type),
                        "tips": [
                            "For neural search, phrase as statements ending with ':'",
                            "Add domain context: 'site:arxiv.org' for papers",
                            "Use category parameter for focused results",
                            "Consider 'type=deep' for complex research queries",
                        ],
                    }
                )
            return result
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _get_context(
        self,
        query: str,
        num_results: int = 10,
        max_characters: int = 10000,
        type: str = "auto",
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
        category: Optional[str] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Get combined context from search results - optimized for RAG.

        This performs a search and combines all result contents into a single
        context string, which is more efficient for RAG applications than
        processing individual results.

        Args:
            query: Search query
            num_results: Number of results to include (default 10)
            max_characters: Maximum total context characters (recommended 10000+)
            type: Search type - "auto", "neural", "keyword", "fast", "deep"
            include_domains: Only search these domains
            exclude_domains: Exclude these domains
            category: Focus category for results

        Returns:
            Result with combined context string and source metadata
        """
        self._stats["searches"] += 1

        if self._mock_mode or not self._client:
            return AdapterResult(
                success=True,
                data={
                    "context": f"Mock context for query: {query}. This is combined content from multiple sources.",
                    "sources": [],
                    "mock": True,
                }
            )

        try:
            # Use search with context option
            result = await self._search(
                query=query,
                type=type,
                num_results=num_results,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                category=category,
                context=True,
                context_max_chars=max_characters,
            )

            if result.success:
                # Extract sources for citation
                sources = []
                for r in result.data.get("results", []):
                    sources.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "score": r.get("score", 0.0),
                    })

                return AdapterResult(
                    success=True,
                    data={
                        "context": result.data.get("context", ""),
                        "sources": sources,
                        "count": len(sources),
                        "cost_dollars": result.data.get("cost_dollars"),
                    }
                )
            return result
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def health_check(self) -> AdapterResult:
        """Check Exa API health."""
        if not EXA_AVAILABLE:
            return AdapterResult(success=False, error="SDK not installed")

        if self._mock_mode or not self._client:
            return AdapterResult(
                success=True,
                data={"status": "degraded", "reason": "No API key - mock mode"}
            )

        try:
            result = await self._search("test", num_results=1)
            return AdapterResult(
                success=True,
                data={
                    "status": "healthy",
                    "stats": self._stats,
                    "total_cost_dollars": self._stats["total_cost_dollars"],
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def shutdown(self) -> AdapterResult:
        """Cleanup resources."""
        # Close HTTP client if open
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._client = None
        self._api_key = None
        self._mock_mode = False
        self._status = AdapterStatus.UNINITIALIZED
        return AdapterResult(success=True, data={"stats": self._stats})


def get_exa_adapter() -> type[ExaAdapter]:
    """Get the Exa adapter class."""
    return ExaAdapter


if __name__ == "__main__":
    async def test():
        adapter = ExaAdapter()
        await adapter.initialize({})
        result = await adapter.execute("search", query="LangGraph StateGraph patterns")
        print(f"Search result: {result}")
        result = await adapter.execute("code_search", query="async Python decorator")
        print(f"Code search result: {result}")
        await adapter.shutdown()

    asyncio.run(test())
