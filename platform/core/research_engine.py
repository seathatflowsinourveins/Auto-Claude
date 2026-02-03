#!/usr/bin/env python3
"""
Research Engine - ULTIMATE Auto-Research Capabilities (v3.0)

THE DEFINITIVE research engine with EVERY feature from BOTH SDKs exposed.

FIRECRAWL v2 - COMPLETE FEATURE LIST:
=====================================
Sync Methods:
- scrape(url, **options) - Single URL scraping with all options
- crawl(url, **options) - Deep crawl with polling
- batch_scrape(urls, **options) - Parallel URL processing
- map(url, **options) - Site structure discovery
- search(query, **options) - Web search with content
- extract(urls, **options) - AI-powered extraction
- agent(urls, prompt, **options) - Agentic browsing

Async Job Control:
- start_crawl(url, **options) - Start async crawl
- get_crawl_status(job_id) - Get crawl status
- cancel_crawl(job_id) - Cancel running crawl
- get_crawl_errors(job_id) - Get crawl errors
- get_active_crawls() - List active crawls

- start_batch_scrape(urls, **options) - Start async batch
- get_batch_scrape_status(job_id) - Get batch status
- cancel_batch_scrape(job_id) - Cancel running batch
- get_batch_scrape_errors(job_id) - Get batch errors

- start_extract(urls, **options) - Start async extract
- get_extract_status(job_id) - Get extract status

- start_agent(urls, prompt, **options) - Start async agent
- get_agent_status(job_id) - Get agent status
- cancel_agent(job_id) - Cancel running agent

Real-time Monitoring:
- watcher(job_type, job_id) - WebSocket/polling watcher for job updates

Account & Usage:
- get_concurrency() - Get concurrency limits
- get_credit_usage() - Get current credit usage
- get_token_usage() - Get current token usage
- get_credit_usage_historical() - Historical credit usage
- get_token_usage_historical() - Historical token usage
- get_queue_status() - Get queue status

Utilities:
- crawl_params_preview(url, prompt) - AI-guided crawl params

EXA - COMPLETE FEATURE LIST:
============================
Search:
- search(query, **options) - Neural/keyword/hybrid search
  * type: neural, keyword, hybrid, fast, deep, auto
  * category: company, research paper, news, pdf, github, tweet, personal site, financial report, people
  * user_location: Two-letter ISO country code
  * moderation: Safety filtering
  * additional_queries: For deep search

Content:
- search_and_contents(query, **options) - Search + content
- find_similar(url, **options) - Similarity search
- find_similar_and_contents(url, **options) - Similar + content
- get_contents(ids_or_urls, **options) - Extract content
  * text: Full text with max_characters
  * summary: AI summary
  * highlights: Key sentences
  * livecrawl: always, fallback, never, auto, preferred
  * subpages: Include linked pages
  * subpage_target: Target subpage pattern
  * extras: {links: bool, image_links: bool}

Answer API:
- answer(query, **options) - Direct question answering
- stream_answer(query, **options) - Streaming answers

Research Client:
- research.create(instructions, model, output_schema) - Start research
- research.get(research_id, stream, events) - Get status/results
- research.list(cursor, limit) - List research tasks
- research.poll_until_finished(research_id, **options) - Wait for completion

Models: exa-research-fast, exa-research

AUTO-INTEGRATION:
================
- Auto-load on import
- Auto-research triggers
- Session integration hooks
- Ralph loop iteration support
"""

import asyncio
import os
import sys
import json
import logging
import time
import threading

# Configure UTF-8 encoding early for Windows compatibility
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from typing import (
    Any, Dict, List, Optional, Union, Literal, Type,
    Callable, Iterator, Generator, AsyncIterator
)
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from enum import Enum
from contextlib import contextmanager
import weakref

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add local SDKs to path
UNLEASH_ROOT = Path(__file__).parent.parent.parent
SDK_ROOT = UNLEASH_ROOT / "sdks"
CONFIG_ROOT = UNLEASH_ROOT / ".config"

# ============================================================================
# AUTO-LOAD ENVIRONMENT VARIABLES FROM .env FILE
# ============================================================================
def _load_env_file():
    """Load environment variables from .config/.env file."""
    env_file = CONFIG_ROOT / ".env"
    if env_file.exists():
        logger.info(f"[ENV] Loading environment from {env_file}")
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    # Handle variable references like ${VAR}
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # For API keys: only override if new value is non-empty
                        # For other keys: only set if not already present
                        is_api_key = '_API_KEY' in key or '_TOKEN' in key
                        if is_api_key:
                            # Override API keys only if .env has a non-empty value
                            should_set = bool(value)
                        else:
                            # For non-API keys, only set if not present
                            should_set = key not in os.environ or not os.environ[key]
                        if should_set:
                            # Resolve ${VAR} references
                            while '${' in value and '}' in value:
                                start = value.find('${')
                                end = value.find('}', start)
                                ref_var = value[start+2:end]
                                ref_value = os.environ.get(ref_var, '')
                                value = value[:start] + ref_value + value[end+1:]
                            os.environ[key] = value
                            logger.debug(f"[ENV] Set {key}={value[:20]}..." if len(value) > 20 else f"[ENV] Set {key}={value}")
            logger.info("[ENV] Environment loaded successfully")
        except Exception as e:
            logger.warning(f"[ENV] Failed to load .env file: {e}")
    else:
        logger.warning(f"[ENV] No .env file found at {env_file}")

# Load environment immediately
_load_env_file()

# Import SDKs (pip-installed, no local path needed)
try:
    from firecrawl import Firecrawl, AsyncFirecrawl
    FIRECRAWL_AVAILABLE = True
    logger.info("[OK] Firecrawl SDK loaded (pip-installed)")
except ImportError as e:
    FIRECRAWL_AVAILABLE = False
    logger.warning(f"Firecrawl SDK not available: {e}")

# Try to import Firecrawl watcher
try:
    from firecrawl.v2.watcher import CrawlWatcher
    FIRECRAWL_WATCHER_AVAILABLE = True
    logger.info("[OK] Firecrawl Watcher available")
except ImportError:
    FIRECRAWL_WATCHER_AVAILABLE = False
    CrawlWatcher = None

try:
    from exa_py import Exa
    EXA_AVAILABLE = True
    logger.info("[OK] Exa SDK loaded from local path")
except ImportError as e:
    EXA_AVAILABLE = False
    logger.warning(f"Exa SDK not available: {e}")

# Try to import Exa Research client
try:
    from exa_py.research import ResearchClient as ExaResearchClient
    EXA_RESEARCH_AVAILABLE = True
    logger.info("[OK] Exa Research client available")
except ImportError:
    EXA_RESEARCH_AVAILABLE = False
    ExaResearchClient = None


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class SearchType(str, Enum):
    """Exa search types."""
    NEURAL = "neural"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    FAST = "fast"
    DEEP = "deep"
    AUTO = "auto"


class SearchCategory(str, Enum):
    """Exa search categories."""
    COMPANY = "company"
    RESEARCH_PAPER = "research paper"
    NEWS = "news"
    PDF = "pdf"
    GITHUB = "github"
    TWEET = "tweet"
    PERSONAL_SITE = "personal site"
    FINANCIAL_REPORT = "financial report"
    PEOPLE = "people"


class LivecrawlMode(str, Enum):
    """Exa livecrawl modes."""
    ALWAYS = "always"
    FALLBACK = "fallback"
    NEVER = "never"
    AUTO = "auto"
    PREFERRED = "preferred"


class FirecrawlAgentModel(str, Enum):
    """Firecrawl agent models."""
    SPARK_PRO = "spark-1-pro"
    SPARK_MINI = "spark-1-mini"


class FirecrawlSearchCategory(str, Enum):
    """Firecrawl search categories."""
    WEB = "web"
    NEWS = "news"
    IMAGES = "images"
    GITHUB = "github"
    RESEARCH = "research"
    PDF = "pdf"


class ExaResearchModel(str, Enum):
    """Exa research models."""
    FAST = "exa-research-fast"
    STANDARD = "exa-research"


class JobStatus(str, Enum):
    """Async job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ResearchResult:
    """Comprehensive research result."""
    query: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    content: List[str] = field(default_factory=list)
    extracted_data: Optional[Dict] = None
    agent_response: Optional[Dict] = None
    deep_research: Optional[Dict] = None
    answer: Optional[str] = None
    summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    success: bool = True
    errors: List[str] = field(default_factory=list)
    credits_used: int = 0


@dataclass
class ExaDeepResearchResult:
    """Exa deep research result."""
    research_id: str
    status: str
    output: Optional[str] = None
    structured_output: Optional[Dict] = None
    events: List[Dict] = field(default_factory=list)
    cost: Optional[float] = None


@dataclass
class JobInfo:
    """Async job information."""
    job_id: str
    job_type: str  # crawl, batch_scrape, extract, agent
    status: JobStatus
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    result: Optional[Dict] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class WatcherEvent:
    """Event from watcher."""
    event_type: str  # document, done, error
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# WATCHER WRAPPER
# =============================================================================

class JobWatcher:
    """
    Wrapper for Firecrawl Watcher with WebSocket + HTTP polling fallback.

    Provides real-time updates for crawl and batch scrape jobs.
    """

    def __init__(
        self,
        firecrawl_client,
        job_type: str,
        job_id: str,
        poll_interval: int = 2,
    ):
        self.client = firecrawl_client
        self.job_type = job_type
        self.job_id = job_id
        self.poll_interval = poll_interval
        self._handlers: Dict[str, List[Callable]] = {
            "document": [],
            "done": [],
            "error": [],
            "progress": [],
        }
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def on_document(self, handler: Callable[[Dict], None]) -> "JobWatcher":
        """Register handler for document events."""
        self._handlers["document"].append(handler)
        return self

    def on_done(self, handler: Callable[[Dict], None]) -> "JobWatcher":
        """Register handler for completion."""
        self._handlers["done"].append(handler)
        return self

    def on_error(self, handler: Callable[[str], None]) -> "JobWatcher":
        """Register handler for errors."""
        self._handlers["error"].append(handler)
        return self

    def on_progress(self, handler: Callable[[float], None]) -> "JobWatcher":
        """Register handler for progress updates."""
        self._handlers["progress"].append(handler)
        return self

    def _emit(self, event_type: str, data: Any):
        """Emit event to all handlers."""
        for handler in self._handlers.get(event_type, []):
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Handler error for {event_type}: {e}")

    def _poll_loop(self):
        """Polling loop for job status."""
        seen_docs = set()

        while self._running:
            try:
                if self.job_type == "crawl":
                    status = self.client.get_crawl_status(self.job_id)
                elif self.job_type == "batch_scrape":
                    status = self.client.get_batch_scrape_status(self.job_id)
                else:
                    break

                # Convert to dict if needed
                if hasattr(status, 'model_dump'):
                    status = status.model_dump()
                elif hasattr(status, '__dict__'):
                    status = vars(status)

                # Emit progress
                progress = status.get("progress", 0)
                self._emit("progress", progress)

                # Emit new documents
                docs = status.get("data", [])
                for doc in docs:
                    doc_id = doc.get("url") or doc.get("id") or str(hash(str(doc)))
                    if doc_id not in seen_docs:
                        seen_docs.add(doc_id)
                        self._emit("document", doc)

                # Check completion
                job_status = status.get("status", "").lower()
                if job_status in ("completed", "done", "finished"):
                    self._emit("done", status)
                    break
                elif job_status in ("failed", "error"):
                    self._emit("error", status.get("error", "Job failed"))
                    break

                time.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Poll error: {e}")
                self._emit("error", str(e))
                break

        self._running = False

    def start(self) -> "JobWatcher":
        """Start watching (non-blocking)."""
        if self._running:
            return self

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """Stop watching."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def wait(self, timeout: Optional[float] = None) -> Dict:
        """Wait for completion and return final result."""
        result = {"status": "unknown"}
        done_event = threading.Event()

        def on_done(data):
            nonlocal result
            result = data
            done_event.set()

        def on_error(error):
            nonlocal result
            result = {"status": "error", "error": error}
            done_event.set()

        self.on_done(on_done).on_error(on_error)

        if not self._running:
            self.start()

        done_event.wait(timeout=timeout)
        self.stop()

        return result


# =============================================================================
# RESEARCH ENGINE v3.0
# =============================================================================

class ResearchEngine:
    """
    ULTIMATE Auto-Research Engine v3.0

    THE definitive research engine with EVERY capability from BOTH SDKs:

    FIRECRAWL v2 - ALL METHODS:
    ---------------------------
    Sync: scrape, crawl, batch_scrape, map, search, extract, agent
    Async Control: start_*, get_*_status, cancel_*, get_*_errors
    Monitoring: watcher() for real-time updates
    Account: get_concurrency, get_credit_usage, get_token_usage, etc.

    EXA - ALL METHODS:
    ------------------
    Search: search, search_and_contents (6 types, 9 categories)
    Similar: find_similar, find_similar_and_contents
    Content: get_contents (livecrawl, subpages, extras)
    Answer: answer, stream_answer
    Research: create, get, list, poll_until_finished

    NEW IN v3.0:
    -----------
    - ALL async job control methods
    - Watcher with WebSocket + HTTP polling
    - Answer and stream_answer APIs
    - find_similar_and_contents
    - Enhanced research client with streaming
    - All missing parameters (user_location, moderation, extras, etc.)
    - Auto-load session integration
    - Auto-research triggers
    """

    # Singleton instance for auto-load
    _instance: Optional["ResearchEngine"] = None
    _auto_loaded: bool = False

    @classmethod
    def get_instance(cls) -> "ResearchEngine":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def auto_load(cls) -> "ResearchEngine":
        """Auto-load on session start."""
        if not cls._auto_loaded:
            cls._instance = cls()
            cls._auto_loaded = True
            logger.info("[AUTO-LOAD] Research Engine v3.0 initialized")
        return cls._instance

    def __init__(
        self,
        firecrawl_api_key: Optional[str] = None,
        exa_api_key: Optional[str] = None,
    ):
        """Initialize with API keys from env or params."""
        # Load from .env if available
        env_path = UNLEASH_ROOT / ".config" / ".env"
        if env_path.exists():
            self._load_env(env_path)

        self.firecrawl_api_key = firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")
        self.exa_api_key = exa_api_key or os.getenv("EXA_API_KEY")

        # Initialize clients
        self.firecrawl = None
        self.firecrawl_async = None
        self.exa = None
        self.exa_research = None

        # Track active jobs
        self._active_jobs: Dict[str, JobInfo] = {}

        if FIRECRAWL_AVAILABLE and self.firecrawl_api_key:
            self.firecrawl = Firecrawl(api_key=self.firecrawl_api_key)
            logger.info("[OK] Firecrawl client initialized")

            # Also create async client
            try:
                self.firecrawl_async = AsyncFirecrawl(api_key=self.firecrawl_api_key)
                logger.info("[OK] Firecrawl async client initialized")
            except Exception as e:
                logger.warning(f"Could not initialize async Firecrawl: {e}")

        if EXA_AVAILABLE and self.exa_api_key:
            self.exa = Exa(api_key=self.exa_api_key)
            logger.info("[OK] Exa client initialized")

            # Initialize research client - takes parent Exa client, not api_key
            if EXA_RESEARCH_AVAILABLE and ExaResearchClient:
                try:
                    self.exa_research = ExaResearchClient(self.exa)
                    logger.info("[OK] Exa Research client initialized")
                except Exception as e:
                    logger.warning(f"Could not initialize Exa Research: {e}")

    def _load_env(self, env_path: Path):
        """Load environment variables from .env file."""
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        if not os.getenv(key):
                            os.environ[key] = value
        except Exception as e:
            logger.warning(f"Could not load .env: {e}")

    def _to_dict(self, obj: Any) -> Any:
        """Convert SDK response to dict."""
        if obj is None:
            return None
        if isinstance(obj, (dict, list, str, int, float, bool)):
            return obj
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        if hasattr(obj, '__dict__'):
            return vars(obj)
        return str(obj)

    # =========================================================================
    # FIRECRAWL v2 - SYNC METHODS (COMPLETE)
    # =========================================================================

    def scrape(
        self,
        url: str,
        *,
        formats: Optional[List[str]] = None,
        headers: Optional[Dict[str, str]] = None,
        include_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        only_main_content: bool = True,
        wait_for: Optional[int] = None,
        timeout: int = 30000,
        mobile: bool = False,
        actions: Optional[List[Dict]] = None,
        location: Optional[Dict] = None,
        skip_tls_verification: bool = False,
        remove_base64_images: bool = False,
        fast_mode: bool = False,
        block_ads: bool = True,
        proxy: Optional[str] = None,
        parsers: Optional[List[str]] = None,
        max_age: Optional[int] = None,
        store_in_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Scrape a single URL with FULL options.

        Args:
            url: URL to scrape
            formats: Output formats (markdown, html, screenshot, links, rawHtml)
            headers: Custom HTTP headers
            include_tags: HTML tags to include
            exclude_tags: HTML tags to exclude
            only_main_content: Extract main content only
            wait_for: Wait for element (ms)
            timeout: Request timeout (ms)
            mobile: Use mobile user agent
            actions: Browser actions (click, scroll, wait, type, etc.)
            location: Geo-location {country, languages}
            skip_tls_verification: Skip SSL verification
            remove_base64_images: Remove base64 images
            fast_mode: Enable fast mode
            block_ads: Block advertisements
            proxy: Proxy server URL
            parsers: Custom parsers
            max_age: Max cache age (seconds)
            store_in_cache: Store in cache

        Returns:
            Scraped content with markdown, metadata
        """
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            kwargs = {}
            if formats:
                kwargs["formats"] = formats
            if headers:
                kwargs["headers"] = headers
            if include_tags:
                kwargs["include_tags"] = include_tags
            if exclude_tags:
                kwargs["exclude_tags"] = exclude_tags
            if not only_main_content:
                kwargs["only_main_content"] = False
            if wait_for:
                kwargs["wait_for"] = wait_for
            if timeout != 30000:
                kwargs["timeout"] = timeout
            if mobile:
                kwargs["mobile"] = True
            if actions:
                kwargs["actions"] = actions
            if location:
                kwargs["location"] = location
            if skip_tls_verification:
                kwargs["skip_tls_verification"] = True
            if remove_base64_images:
                kwargs["remove_base64_images"] = True
            if fast_mode:
                kwargs["fast_mode"] = True
            if block_ads:
                kwargs["block_ads"] = True
            if proxy:
                kwargs["proxy"] = proxy
            if parsers:
                kwargs["parsers"] = parsers
            if max_age:
                kwargs["max_age"] = max_age
            if not store_in_cache:
                kwargs["store_in_cache"] = False

            result = self.firecrawl.scrape(url, **kwargs)
            return {"success": True, "data": self._to_dict(result)}
        except Exception as e:
            logger.error(f"Scrape error: {e}")
            return {"success": False, "error": str(e)}

    def crawl(
        self,
        url: str,
        *,
        max_discovery_depth: int = 2,
        limit: int = 50,
        include_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        allow_external_links: bool = False,
        allow_subdomains: bool = False,
        ignore_sitemap: bool = False,
        ignore_query_parameters: bool = False,
        crawl_entire_domain: bool = False,
        delay: Optional[int] = None,
        max_concurrency: Optional[int] = None,
        scrape_options: Optional[Dict] = None,
        webhook: Optional[str] = None,
        zero_data_retention: bool = False,
        poll_interval: int = 2,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Deep crawl a website with FULL SDK options.

        Args:
            url: Starting URL
            max_discovery_depth: How deep to crawl links
            limit: Max pages to crawl
            include_paths: Paths to include (regex patterns)
            exclude_paths: Paths to exclude (regex patterns)
            allow_external_links: Follow links to external domains
            allow_subdomains: Crawl subdomains of the starting URL
            ignore_sitemap: Skip sitemap.xml processing
            ignore_query_parameters: Ignore URL query params
            crawl_entire_domain: Crawl entire domain
            delay: Delay between requests (seconds)
            max_concurrency: Max concurrent requests
            scrape_options: Options for each page scrape
            webhook: Webhook URL for results
            zero_data_retention: Delete data after 24h
            poll_interval: Polling interval (seconds)
            timeout: Max wait time (seconds)

        Returns:
            All crawled pages with content
        """
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            kwargs: Dict[str, Any] = {
                "max_discovery_depth": max_discovery_depth,
                "limit": limit,
            }
            if include_paths:
                kwargs["include_paths"] = include_paths
            if exclude_paths:
                kwargs["exclude_paths"] = exclude_paths
            if allow_external_links:
                kwargs["allow_external_links"] = True
            if allow_subdomains:
                kwargs["allow_subdomains"] = True
            if ignore_sitemap:
                kwargs["ignore_sitemap"] = True
            if ignore_query_parameters:
                kwargs["ignore_query_parameters"] = True
            if crawl_entire_domain:
                kwargs["crawl_entire_domain"] = True
            if delay is not None:
                kwargs["delay"] = delay
            if max_concurrency is not None:
                kwargs["max_concurrency"] = max_concurrency
            if scrape_options:
                kwargs["scrape_options"] = scrape_options
            if webhook:
                kwargs["webhook"] = webhook
            if zero_data_retention:
                kwargs["zero_data_retention"] = True

            result = self.firecrawl.crawl(
                url,
                poll_interval=poll_interval,
                timeout=timeout,
                **kwargs
            )
            return {"success": True, "data": self._to_dict(result)}
        except Exception as e:
            logger.error(f"Crawl error: {e}")
            return {"success": False, "error": str(e)}

    def batch_scrape(
        self,
        urls: List[str],
        *,
        formats: Optional[List[str]] = None,
        only_main_content: bool = True,
        poll_interval: int = 2,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Scrape multiple URLs in parallel."""
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            kwargs = {}
            if formats:
                kwargs["formats"] = formats
            if not only_main_content:
                kwargs["only_main_content"] = False

            result = self.firecrawl.batch_scrape(
                urls,
                poll_interval=poll_interval,
                timeout=timeout,
                **kwargs
            )
            return {"success": True, "data": self._to_dict(result)}
        except Exception as e:
            logger.error(f"Batch scrape error: {e}")
            return {"success": False, "error": str(e)}

    def map_site(
        self,
        url: str,
        *,
        search: Optional[str] = None,
        include_subdomains: bool = False,
        limit: int = 5000,
        ignore_sitemap: bool = False,
    ) -> Dict[str, Any]:
        """Map all URLs on a website (up to 5000)."""
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            kwargs = {"limit": limit}
            if search:
                kwargs["search"] = search
            if include_subdomains:
                kwargs["include_subdomains"] = True
            if ignore_sitemap:
                kwargs["ignore_sitemap"] = True

            result = self.firecrawl.map(url, **kwargs)
            data = self._to_dict(result)
            # Normalize: Firecrawl returns 'links', we also expose as 'urls' for consistency
            if isinstance(data, dict) and 'links' in data:
                data['urls'] = data['links']
            return {"success": True, "data": data}
        except Exception as e:
            logger.error(f"Map error: {e}")
            return {"success": False, "error": str(e)}

    # Alias for map_site for compatibility
    map_url = map_site

    def firecrawl_search(
        self,
        query: str,
        *,
        limit: int = 10,
        location: Optional[str] = None,
        tbs: Optional[str] = None,
        sources: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        timeout: Optional[int] = None,
        scrape_options: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Firecrawl web search with content.

        Args:
            query: Search query
            limit: Results count (1-100)
            location: Location string (e.g., "us", "uk", "Germany")
            tbs: Time filter (qdr:h=hour, qdr:d=day, qdr:w=week, qdr:m=month, qdr:y=year)
            sources: Source types ["web", "news", "images"]
            categories: Category types ["github", "research", "pdf"]
            timeout: Request timeout in ms (default 300000)
            scrape_options: Scraping options for results

        Returns:
            Search results with content (grouped by source type)
        """
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            kwargs = {"limit": limit}
            if location:
                kwargs["location"] = location
            if tbs:
                kwargs["tbs"] = tbs
            if sources:
                kwargs["sources"] = sources
            if categories:
                kwargs["categories"] = categories
            if timeout:
                kwargs["timeout"] = timeout
            if scrape_options:
                kwargs["scrape_options"] = scrape_options

            result = self.firecrawl.search(query, **kwargs)
            data = self._to_dict(result)
            # Normalize: combine all result types into 'results' for consistency
            if isinstance(data, dict):
                all_results = []
                for key in ['web', 'news', 'images', 'data']:
                    if key in data and isinstance(data[key], list):
                        all_results.extend(data[key])
                data['results'] = all_results
            return {"success": True, "data": data}
        except Exception as e:
            logger.error(f"Firecrawl search error: {e}")
            return {"success": False, "error": str(e)}

    def extract(
        self,
        urls: Union[str, List[str]],
        *,
        prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        allow_external_links: bool = False,
        enable_web_search: bool = False,
        show_sources: bool = True,
        scrape_options: Optional[Dict] = None,
        ignore_invalid_urls: bool = True,
        poll_interval: int = 2,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        AI-powered structured data extraction.

        Args:
            urls: URL(s) to extract from
            prompt: What to extract
            schema: JSON schema for structured output
            system_prompt: System instructions
            allow_external_links: Follow external links
            enable_web_search: Search for context
            show_sources: Include source URLs
            scrape_options: Scraping options
            ignore_invalid_urls: Skip invalid URLs
            poll_interval: Polling interval (s)
            timeout: Max wait (s)

        Returns:
            Extracted structured data
        """
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            if isinstance(urls, str):
                urls = [urls]

            kwargs = {"ignore_invalid_urls": ignore_invalid_urls, "poll_interval": poll_interval}
            if prompt:
                kwargs["prompt"] = prompt
            if schema:
                kwargs["schema"] = schema
            if system_prompt:
                kwargs["system_prompt"] = system_prompt
            if allow_external_links:
                kwargs["allow_external_links"] = True
            if enable_web_search:
                kwargs["enable_web_search"] = True
            if show_sources:
                kwargs["show_sources"] = True
            if scrape_options:
                kwargs["scrape_options"] = scrape_options
            if timeout:
                kwargs["timeout"] = timeout

            result = self.firecrawl.extract(urls, **kwargs)
            data = self._to_dict(result)
            # Normalize: add 'results' key for consistency with other methods
            if isinstance(data, dict):
                # The extracted data is in 'data' field, also expose as 'results'
                if 'data' in data and data['data']:
                    data['results'] = [data['data']] if not isinstance(data['data'], list) else data['data']
                else:
                    data['results'] = []
            return {"success": True, "data": data}
        except Exception as e:
            logger.error(f"Extract error: {e}")
            return {"success": False, "error": str(e)}

    def agent(
        self,
        urls: Union[str, List[str]],
        prompt: str,
        *,
        model: str = "spark-1-pro",
        schema: Optional[Dict[str, Any]] = None,
        max_credits: Optional[int] = None,
        strict_constrain_to_urls: bool = False,
        poll_interval: int = 2,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Firecrawl AI Agent for agentic web browsing.

        Args:
            urls: Starting URL(s)
            prompt: Agent instructions
            model: spark-1-pro (advanced) or spark-1-mini (fast)
            schema: JSON schema for output
            max_credits: Credit limit
            strict_constrain_to_urls: Stay on provided URLs
            poll_interval: Polling interval (s)
            timeout: Max wait (s)

        Returns:
            Agent results
        """
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            if isinstance(urls, str):
                urls = [urls]

            kwargs = {"prompt": prompt, "model": model, "poll_interval": poll_interval}
            if schema:
                kwargs["schema"] = schema
            if max_credits:
                kwargs["max_credits"] = max_credits
            if strict_constrain_to_urls:
                kwargs["strict_constrain_to_urls"] = True
            if timeout:
                kwargs["timeout"] = timeout

            result = self.firecrawl.agent(urls, **kwargs)
            return {"success": True, "data": self._to_dict(result)}
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # FIRECRAWL v2 - ASYNC JOB CONTROL (NEW IN v3.0)
    # =========================================================================

    def start_crawl(
        self,
        url: str,
        *,
        max_depth: int = 2,
        limit: int = 50,
        include_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        allow_subdomains: bool = False,
        scrape_options: Optional[Dict] = None,
        webhook: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Start async crawl job (non-blocking).

        Returns job_id for status checking.
        """
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            kwargs = {"max_depth": max_depth, "limit": limit}
            if include_paths:
                kwargs["include_paths"] = include_paths
            if exclude_paths:
                kwargs["exclude_paths"] = exclude_paths
            if allow_subdomains:
                kwargs["allow_subdomains"] = True
            if scrape_options:
                kwargs["scrape_options"] = scrape_options
            if webhook:
                kwargs["webhook"] = webhook

            # Use start_crawl if available, otherwise crawl_async
            if hasattr(self.firecrawl, 'start_crawl'):
                result = self.firecrawl.start_crawl(url, **kwargs)
            elif hasattr(self.firecrawl, 'crawl_async'):
                result = self.firecrawl.crawl_async(url, **kwargs)
            else:
                return {"error": "Async crawl not available", "success": False}

            result = self._to_dict(result)
            job_id = result.get("id") or result.get("job_id") or result.get("crawl_id")

            if job_id:
                self._active_jobs[job_id] = JobInfo(
                    job_id=job_id,
                    job_type="crawl",
                    status=JobStatus.RUNNING,
                )

            return {"success": True, "job_id": job_id, "data": result}
        except Exception as e:
            logger.error(f"Start crawl error: {e}")
            return {"success": False, "error": str(e)}

    def get_crawl_status(self, job_id: str) -> Dict[str, Any]:
        """Get crawl job status."""
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            if hasattr(self.firecrawl, 'get_crawl_status'):
                result = self.firecrawl.get_crawl_status(job_id)
            elif hasattr(self.firecrawl, 'check_crawl_status'):
                result = self.firecrawl.check_crawl_status(job_id)
            else:
                return {"error": "Get crawl status not available", "success": False}

            return {"success": True, "data": self._to_dict(result)}
        except Exception as e:
            logger.error(f"Get crawl status error: {e}")
            return {"success": False, "error": str(e)}

    def cancel_crawl(self, job_id: str) -> Dict[str, Any]:
        """Cancel running crawl job."""
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            if hasattr(self.firecrawl, 'cancel_crawl'):
                result = self.firecrawl.cancel_crawl(job_id)
            else:
                return {"error": "Cancel crawl not available", "success": False}

            if job_id in self._active_jobs:
                self._active_jobs[job_id].status = JobStatus.CANCELLED

            return {"success": True, "data": self._to_dict(result)}
        except Exception as e:
            logger.error(f"Cancel crawl error: {e}")
            return {"success": False, "error": str(e)}

    def get_crawl_errors(self, job_id: str) -> Dict[str, Any]:
        """Get crawl job errors."""
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            if hasattr(self.firecrawl, 'get_crawl_errors'):
                result = self.firecrawl.get_crawl_errors(job_id)
                return {"success": True, "data": self._to_dict(result)}
            else:
                # Fallback: get from status
                status = self.get_crawl_status(job_id)
                errors = status.get("data", {}).get("errors", [])
                return {"success": True, "errors": errors}
        except Exception as e:
            logger.error(f"Get crawl errors error: {e}")
            return {"success": False, "error": str(e)}

    def get_active_crawls(self) -> Dict[str, Any]:
        """Get list of active crawl jobs."""
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            if hasattr(self.firecrawl, 'get_active_crawls'):
                result = self.firecrawl.get_active_crawls()
                return {"success": True, "data": self._to_dict(result)}
            else:
                # Fallback: return tracked jobs
                active = [
                    j for j in self._active_jobs.values()
                    if j.job_type == "crawl" and j.status == JobStatus.RUNNING
                ]
                return {"success": True, "jobs": [vars(j) for j in active]}
        except Exception as e:
            logger.error(f"Get active crawls error: {e}")
            return {"success": False, "error": str(e)}

    def start_batch_scrape(
        self,
        urls: List[str],
        *,
        formats: Optional[List[str]] = None,
        only_main_content: bool = True,
        webhook: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start async batch scrape job."""
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            kwargs = {}
            if formats:
                kwargs["formats"] = formats
            if not only_main_content:
                kwargs["only_main_content"] = False
            if webhook:
                kwargs["webhook"] = webhook

            if hasattr(self.firecrawl, 'start_batch_scrape'):
                result = self.firecrawl.start_batch_scrape(urls, **kwargs)
            elif hasattr(self.firecrawl, 'batch_scrape_async'):
                result = self.firecrawl.batch_scrape_async(urls, **kwargs)
            else:
                return {"error": "Async batch scrape not available", "success": False}

            result = self._to_dict(result)
            job_id = result.get("id") or result.get("job_id") or result.get("batch_id")

            if job_id:
                self._active_jobs[job_id] = JobInfo(
                    job_id=job_id,
                    job_type="batch_scrape",
                    status=JobStatus.RUNNING,
                )

            return {"success": True, "job_id": job_id, "data": result}
        except Exception as e:
            logger.error(f"Start batch scrape error: {e}")
            return {"success": False, "error": str(e)}

    def get_batch_scrape_status(self, job_id: str) -> Dict[str, Any]:
        """Get batch scrape job status."""
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            if hasattr(self.firecrawl, 'get_batch_scrape_status'):
                result = self.firecrawl.get_batch_scrape_status(job_id)
            elif hasattr(self.firecrawl, 'check_batch_scrape_status'):
                result = self.firecrawl.check_batch_scrape_status(job_id)
            else:
                return {"error": "Get batch status not available", "success": False}

            return {"success": True, "data": self._to_dict(result)}
        except Exception as e:
            logger.error(f"Get batch scrape status error: {e}")
            return {"success": False, "error": str(e)}

    def cancel_batch_scrape(self, job_id: str) -> Dict[str, Any]:
        """Cancel running batch scrape job."""
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            if hasattr(self.firecrawl, 'cancel_batch_scrape'):
                result = self.firecrawl.cancel_batch_scrape(job_id)
                return {"success": True, "data": self._to_dict(result)}
            else:
                return {"error": "Cancel batch scrape not available", "success": False}
        except Exception as e:
            logger.error(f"Cancel batch scrape error: {e}")
            return {"success": False, "error": str(e)}

    def get_batch_scrape_errors(self, job_id: str) -> Dict[str, Any]:
        """Get batch scrape job errors."""
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            if hasattr(self.firecrawl, 'get_batch_scrape_errors'):
                result = self.firecrawl.get_batch_scrape_errors(job_id)
                return {"success": True, "data": self._to_dict(result)}
            else:
                status = self.get_batch_scrape_status(job_id)
                errors = status.get("data", {}).get("errors", [])
                return {"success": True, "errors": errors}
        except Exception as e:
            logger.error(f"Get batch scrape errors error: {e}")
            return {"success": False, "error": str(e)}

    def start_extract(
        self,
        urls: Union[str, List[str]],
        *,
        prompt: Optional[str] = None,
        schema: Optional[Dict] = None,
        enable_web_search: bool = False,
    ) -> Dict[str, Any]:
        """Start async extract job."""
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            if isinstance(urls, str):
                urls = [urls]

            kwargs = {}
            if prompt:
                kwargs["prompt"] = prompt
            if schema:
                kwargs["schema"] = schema
            if enable_web_search:
                kwargs["enable_web_search"] = True

            if hasattr(self.firecrawl, 'start_extract'):
                result = self.firecrawl.start_extract(urls, **kwargs)
            elif hasattr(self.firecrawl, 'extract_async'):
                result = self.firecrawl.extract_async(urls, **kwargs)
            else:
                return {"error": "Async extract not available", "success": False}

            result = self._to_dict(result)
            job_id = result.get("id") or result.get("job_id")

            return {"success": True, "job_id": job_id, "data": result}
        except Exception as e:
            logger.error(f"Start extract error: {e}")
            return {"success": False, "error": str(e)}

    def get_extract_status(self, job_id: str) -> Dict[str, Any]:
        """Get extract job status."""
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            if hasattr(self.firecrawl, 'get_extract_status'):
                result = self.firecrawl.get_extract_status(job_id)
            elif hasattr(self.firecrawl, 'check_extract_status'):
                result = self.firecrawl.check_extract_status(job_id)
            else:
                return {"error": "Get extract status not available", "success": False}

            return {"success": True, "data": self._to_dict(result)}
        except Exception as e:
            logger.error(f"Get extract status error: {e}")
            return {"success": False, "error": str(e)}

    def start_agent(
        self,
        urls: Union[str, List[str]],
        prompt: str,
        *,
        model: str = "spark-1-pro",
        schema: Optional[Dict] = None,
        max_credits: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Start async agent job."""
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            if isinstance(urls, str):
                urls = [urls]

            kwargs = {"prompt": prompt, "model": model}
            if schema:
                kwargs["schema"] = schema
            if max_credits:
                kwargs["max_credits"] = max_credits

            if hasattr(self.firecrawl, 'start_agent'):
                result = self.firecrawl.start_agent(urls, **kwargs)
            elif hasattr(self.firecrawl, 'agent_async'):
                result = self.firecrawl.agent_async(urls, **kwargs)
            else:
                return {"error": "Async agent not available", "success": False}

            result = self._to_dict(result)
            job_id = result.get("id") or result.get("job_id")

            return {"success": True, "job_id": job_id, "data": result}
        except Exception as e:
            logger.error(f"Start agent error: {e}")
            return {"success": False, "error": str(e)}

    def get_agent_status(self, job_id: str) -> Dict[str, Any]:
        """Get agent job status."""
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            if hasattr(self.firecrawl, 'get_agent_status'):
                result = self.firecrawl.get_agent_status(job_id)
            elif hasattr(self.firecrawl, 'check_agent_status'):
                result = self.firecrawl.check_agent_status(job_id)
            else:
                return {"error": "Get agent status not available", "success": False}

            return {"success": True, "data": self._to_dict(result)}
        except Exception as e:
            logger.error(f"Get agent status error: {e}")
            return {"success": False, "error": str(e)}

    def cancel_agent(self, job_id: str) -> Dict[str, Any]:
        """Cancel running agent job."""
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            if hasattr(self.firecrawl, 'cancel_agent'):
                result = self.firecrawl.cancel_agent(job_id)
                return {"success": True, "data": self._to_dict(result)}
            else:
                return {"error": "Cancel agent not available", "success": False}
        except Exception as e:
            logger.error(f"Cancel agent error: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # FIRECRAWL v2 - WATCHER (NEW IN v3.0)
    # =========================================================================

    def watcher(
        self,
        job_type: str,
        job_id: str,
        poll_interval: int = 2,
    ) -> JobWatcher:
        """
        Create a watcher for real-time job updates.

        Uses WebSocket if available, falls back to HTTP polling.

        Args:
            job_type: "crawl" or "batch_scrape"
            job_id: Job ID from start_* methods
            poll_interval: Polling interval in seconds

        Returns:
            JobWatcher with event handlers

        Example:
            watcher = engine.watcher("crawl", job_id)
            watcher.on_document(lambda d: print(f"Got: {d['url']}"))
            watcher.on_done(lambda r: print("Done!"))
            watcher.start()
            # or
            result = watcher.wait(timeout=300)
        """
        return JobWatcher(
            self.firecrawl,
            job_type=job_type,
            job_id=job_id,
            poll_interval=poll_interval,
        )

    # =========================================================================
    # FIRECRAWL v2 - ACCOUNT & USAGE (NEW IN v3.0)
    # =========================================================================

    def get_concurrency(self) -> Dict[str, Any]:
        """Get current concurrency limits."""
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            if hasattr(self.firecrawl, 'get_concurrency'):
                result = self.firecrawl.get_concurrency()
                return {"success": True, "data": self._to_dict(result)}
            else:
                return {"success": False, "error": "get_concurrency not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_credit_usage(self) -> Dict[str, Any]:
        """Get current credit usage."""
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            if hasattr(self.firecrawl, 'get_credit_usage'):
                result = self.firecrawl.get_credit_usage()
            elif hasattr(self.firecrawl, 'get_usage'):
                result = self.firecrawl.get_usage()
            else:
                return {"success": False, "error": "Credit usage API not available"}

            return {"success": True, "data": self._to_dict(result)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_token_usage(self) -> Dict[str, Any]:
        """Get current token usage."""
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            if hasattr(self.firecrawl, 'get_token_usage'):
                result = self.firecrawl.get_token_usage()
                return {"success": True, "data": self._to_dict(result)}
            else:
                return {"success": False, "error": "Token usage API not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_credit_usage_historical(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get historical credit usage."""
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            if hasattr(self.firecrawl, 'get_credit_usage_historical'):
                kwargs = {}
                if start_date:
                    kwargs["start_date"] = start_date
                if end_date:
                    kwargs["end_date"] = end_date
                result = self.firecrawl.get_credit_usage_historical(**kwargs)
                return {"success": True, "data": self._to_dict(result)}
            else:
                return {"success": False, "error": "Historical credit usage not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_token_usage_historical(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get historical token usage."""
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            if hasattr(self.firecrawl, 'get_token_usage_historical'):
                kwargs = {}
                if start_date:
                    kwargs["start_date"] = start_date
                if end_date:
                    kwargs["end_date"] = end_date
                result = self.firecrawl.get_token_usage_historical(**kwargs)
                return {"success": True, "data": self._to_dict(result)}
            else:
                return {"success": False, "error": "Historical token usage not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status."""
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            if hasattr(self.firecrawl, 'get_queue_status'):
                result = self.firecrawl.get_queue_status()
                return {"success": True, "data": self._to_dict(result)}
            else:
                return {"success": False, "error": "Queue status not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def crawl_params_preview(
        self,
        url: str,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        AI-guided crawl parameters preview.

        Analyzes the URL and suggests optimal crawl parameters.
        """
        if not self.firecrawl:
            return {"error": "Firecrawl not available", "success": False}

        try:
            if hasattr(self.firecrawl, 'crawl_params_preview'):
                kwargs = {}
                if prompt:
                    kwargs["prompt"] = prompt
                result = self.firecrawl.crawl_params_preview(url, **kwargs)
                return {"success": True, "data": self._to_dict(result)}
            else:
                return {"success": False, "error": "Crawl params preview not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # EXA - SEARCH METHODS (COMPLETE)
    # =========================================================================

    def exa_search(
        self,
        query: str,
        *,
        num_results: int = 10,
        type: Optional[str] = None,
        category: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        use_autoprompt: bool = False,
        additional_queries: Optional[List[str]] = None,
        user_location: Optional[str] = None,
        moderation: Optional[bool] = None,
        flags: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Exa neural/keyword search with FULL options.

        Args:
            query: Search query
            num_results: Number of results (1-100)
            type: neural, keyword, hybrid, fast, deep, auto
            category: company, research paper, news, pdf, github, tweet,
                      personal site, financial report, people
            include_domains: Only these domains
            exclude_domains: Exclude these domains
            start_published_date: Filter by date (YYYY-MM-DD)
            end_published_date: Filter by date (YYYY-MM-DD)
            start_crawl_date: Filter by crawl date
            end_crawl_date: Filter by crawl date
            include_text: Must contain strings
            exclude_text: Must not contain strings
            use_autoprompt: Auto-enhance query
            additional_queries: For deep search (max 5)
            user_location: Two-letter ISO country code (NEW)
            moderation: Safety filtering (NEW)
            flags: Additional flags (NEW)

        Returns:
            Search results
        """
        if not self.exa:
            return {"error": "Exa not available", "success": False}

        try:
            kwargs = {"num_results": num_results}

            if type:
                kwargs["type"] = type
            if category:
                kwargs["category"] = category
            if include_domains:
                kwargs["include_domains"] = include_domains
            if exclude_domains:
                kwargs["exclude_domains"] = exclude_domains
            if start_published_date:
                kwargs["start_published_date"] = start_published_date
            if end_published_date:
                kwargs["end_published_date"] = end_published_date
            if start_crawl_date:
                kwargs["start_crawl_date"] = start_crawl_date
            if end_crawl_date:
                kwargs["end_crawl_date"] = end_crawl_date
            if include_text:
                kwargs["include_text"] = include_text
            if exclude_text:
                kwargs["exclude_text"] = exclude_text
            if use_autoprompt:
                kwargs["use_autoprompt"] = True
            if additional_queries:
                kwargs["additional_queries"] = additional_queries
            if user_location:
                kwargs["user_location"] = user_location
            if moderation is not None:
                kwargs["moderation"] = moderation
            if flags:
                kwargs["flags"] = flags

            result = self.exa.search(query, **kwargs)

            results_list = [
                {
                    "url": r.url,
                    "title": r.title,
                    "id": getattr(r, 'id', None),
                    "score": getattr(r, 'score', None),
                    "published_date": getattr(r, 'published_date', None),
                    "author": getattr(r, 'author', None),
                }
                for r in result.results
            ]

            # Return both formats for compatibility
            return {
                "success": True,
                "autoprompt_string": getattr(result, 'autoprompt_string', None),
                "results": results_list,  # Top-level for backwards compat
                "data": {"results": results_list}  # Nested for consistency
            }
        except Exception as e:
            logger.error(f"Exa search error: {e}")
            return {"success": False, "error": str(e)}

    def exa_search_and_contents(
        self,
        query: str,
        *,
        num_results: int = 10,
        type: Optional[str] = None,
        category: Optional[str] = None,
        text_max_chars: int = 2000,
        highlights: bool = True,
        highlight_sentences: int = 3,
        summary: bool = False,
        livecrawl: str = "fallback",
        subpages: Optional[int] = None,
        subpage_target: Optional[str] = None,
        extras: Optional[Dict[str, bool]] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        user_location: Optional[str] = None,
        moderation: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Exa search with full content extraction.

        Args:
            query: Search query
            num_results: Number of results
            type: neural, keyword, hybrid, fast, deep, auto
            category: Content category
            text_max_chars: Max characters per result
            highlights: Include snippets
            highlight_sentences: Number of sentences
            summary: Include AI summary
            livecrawl: always, fallback, never, auto, preferred
            subpages: Number of subpages
            subpage_target: Target subpage pattern (NEW)
            extras: {links: bool, image_links: bool} (NEW)
            include_domains: Only these domains
            exclude_domains: Exclude these domains
            start_published_date: Filter by date
            end_published_date: Filter by date
            user_location: ISO country code (NEW)
            moderation: Safety filtering (NEW)

        Returns:
            Search results with content
        """
        if not self.exa:
            return {"error": "Exa not available", "success": False}

        try:
            # Search options
            search_kwargs = {"num_results": num_results}
            if type:
                search_kwargs["type"] = type
            if category:
                search_kwargs["category"] = category
            if include_domains:
                search_kwargs["include_domains"] = include_domains
            if exclude_domains:
                search_kwargs["exclude_domains"] = exclude_domains
            if start_published_date:
                search_kwargs["start_published_date"] = start_published_date
            if end_published_date:
                search_kwargs["end_published_date"] = end_published_date
            if user_location:
                search_kwargs["user_location"] = user_location
            if moderation is not None:
                search_kwargs["moderation"] = moderation

            # Contents options
            text_options = {"max_characters": text_max_chars}
            contents_kwargs = {"text": text_options}

            if highlights:
                contents_kwargs["highlights"] = {"num_sentences": highlight_sentences}
            if summary:
                contents_kwargs["summary"] = True
            if livecrawl != "fallback":
                contents_kwargs["livecrawl"] = livecrawl
            if subpages:
                contents_kwargs["subpages"] = subpages
            if subpage_target:
                contents_kwargs["subpage_target"] = subpage_target
            if extras:
                contents_kwargs["extras"] = extras

            result = self.exa.search_and_contents(query, **search_kwargs, **contents_kwargs)

            return {
                "success": True,
                "results": [
                    {
                        "url": r.url,
                        "title": r.title,
                        "text": getattr(r, 'text', ''),
                        "highlights": getattr(r, 'highlights', []),
                        "summary": getattr(r, 'summary', None),
                        "score": getattr(r, 'score', None),
                        "published_date": getattr(r, 'published_date', None),
                        "author": getattr(r, 'author', None),
                        "links": getattr(r, 'links', None),
                        "image_links": getattr(r, 'image_links', None),
                    }
                    for r in result.results
                ]
            }
        except Exception as e:
            logger.error(f"Exa search_and_contents error: {e}")
            return {"success": False, "error": str(e)}

    def exa_find_similar(
        self,
        url: str,
        *,
        num_results: int = 10,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        exclude_source_domain: bool = True,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Find pages similar to a URL."""
        if not self.exa:
            return {"error": "Exa not available", "success": False}

        try:
            kwargs = {"num_results": num_results}
            if include_domains:
                kwargs["include_domains"] = include_domains
            if exclude_domains:
                kwargs["exclude_domains"] = exclude_domains
            if exclude_source_domain:
                kwargs["exclude_source_domain"] = True
            if start_published_date:
                kwargs["start_published_date"] = start_published_date
            if end_published_date:
                kwargs["end_published_date"] = end_published_date
            if category:
                kwargs["category"] = category

            result = self.exa.find_similar(url, **kwargs)

            return {
                "success": True,
                "results": [
                    {
                        "url": r.url,
                        "title": r.title,
                        "score": getattr(r, 'score', None),
                        "published_date": getattr(r, 'published_date', None),
                    }
                    for r in result.results
                ]
            }
        except Exception as e:
            logger.error(f"Exa find_similar error: {e}")
            return {"success": False, "error": str(e)}

    def exa_find_similar_and_contents(
        self,
        url: str,
        *,
        num_results: int = 10,
        text_max_chars: int = 2000,
        highlights: bool = True,
        highlight_sentences: int = 3,
        summary: bool = False,
        livecrawl: str = "fallback",
        subpages: Optional[int] = None,
        subpage_target: Optional[str] = None,
        extras: Optional[Dict[str, bool]] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        exclude_source_domain: bool = True,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Find similar pages WITH content (NEW IN v3.0).

        Combines find_similar + get_contents in one call.
        """
        if not self.exa:
            return {"error": "Exa not available", "success": False}

        try:
            # Find similar options
            similar_kwargs = {"num_results": num_results}
            if include_domains:
                similar_kwargs["include_domains"] = include_domains
            if exclude_domains:
                similar_kwargs["exclude_domains"] = exclude_domains
            if exclude_source_domain:
                similar_kwargs["exclude_source_domain"] = True
            if category:
                similar_kwargs["category"] = category

            # Contents options
            text_options = {"max_characters": text_max_chars}
            contents_kwargs = {"text": text_options}

            if highlights:
                contents_kwargs["highlights"] = {"num_sentences": highlight_sentences}
            if summary:
                contents_kwargs["summary"] = True
            if livecrawl != "fallback":
                contents_kwargs["livecrawl"] = livecrawl
            if subpages:
                contents_kwargs["subpages"] = subpages
            if subpage_target:
                contents_kwargs["subpage_target"] = subpage_target
            if extras:
                contents_kwargs["extras"] = extras

            # Try find_similar_and_contents if available
            if hasattr(self.exa, 'find_similar_and_contents'):
                result = self.exa.find_similar_and_contents(
                    url,
                    **similar_kwargs,
                    **contents_kwargs
                )
            else:
                # Fallback: find_similar + get_contents
                similar = self.exa.find_similar(url, **similar_kwargs)
                urls = [r.url for r in similar.results]
                result = self.exa.get_contents(urls, **contents_kwargs)

            return {
                "success": True,
                "results": [
                    {
                        "url": r.url,
                        "title": r.title,
                        "text": getattr(r, 'text', ''),
                        "highlights": getattr(r, 'highlights', []),
                        "summary": getattr(r, 'summary', None),
                        "score": getattr(r, 'score', None),
                        "links": getattr(r, 'links', None),
                        "image_links": getattr(r, 'image_links', None),
                    }
                    for r in result.results
                ]
            }
        except Exception as e:
            logger.error(f"Exa find_similar_and_contents error: {e}")
            return {"success": False, "error": str(e)}

    def exa_get_contents(
        self,
        ids_or_urls: List[str],
        *,
        text_max_chars: int = 2000,
        highlights: bool = False,
        highlight_sentences: int = 3,
        summary: bool = False,
        livecrawl: str = "fallback",
        subpages: Optional[int] = None,
        subpage_target: Optional[str] = None,
        extras: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """
        Get contents for specific URLs or IDs.

        Args:
            ids_or_urls: List of Exa IDs or URLs
            text_max_chars: Max characters
            highlights: Include snippets
            highlight_sentences: Number of sentences
            summary: Include AI summary
            livecrawl: always, fallback, never, auto, preferred
            subpages: Number of subpages
            subpage_target: Target subpage pattern (NEW)
            extras: {links: bool, image_links: bool} (NEW)

        Returns:
            Content for each URL/ID
        """
        if not self.exa:
            return {"error": "Exa not available", "success": False}

        try:
            text_options = {"max_characters": text_max_chars}
            kwargs = {"text": text_options}

            if highlights:
                kwargs["highlights"] = {"num_sentences": highlight_sentences}
            if summary:
                kwargs["summary"] = True
            if livecrawl != "fallback":
                kwargs["livecrawl"] = livecrawl
            if subpages:
                kwargs["subpages"] = subpages
            if subpage_target:
                kwargs["subpage_target"] = subpage_target
            if extras:
                kwargs["extras"] = extras

            result = self.exa.get_contents(ids_or_urls, **kwargs)

            return {
                "success": True,
                "results": [
                    {
                        "url": r.url,
                        "title": r.title,
                        "text": getattr(r, 'text', ''),
                        "highlights": getattr(r, 'highlights', []),
                        "summary": getattr(r, 'summary', None),
                        "links": getattr(r, 'links', None),
                        "image_links": getattr(r, 'image_links', None),
                    }
                    for r in result.results
                ]
            }
        except Exception as e:
            logger.error(f"Exa get_contents error: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # EXA - ANSWER API (NEW IN v3.0)
    # =========================================================================

    def exa_answer(
        self,
        query: str,
        *,
        text: bool = True,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        output_schema: Optional[Dict] = None,
        user_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Direct question answering using Exa's LLM (NEW IN v3.0).

        Args:
            query: Question to answer
            text: Include full text in results (default True)
            system_prompt: Guide the LLM's behavior
            model: "exa" (faster) or "exa-pro" (more capable)
            output_schema: JSON schema for structured output
            user_location: Two-letter ISO country code

        Returns:
            Answer with sources and citations
        """
        if not self.exa:
            return {"error": "Exa not available", "success": False}

        try:
            if hasattr(self.exa, 'answer'):
                kwargs = {"text": text}
                if system_prompt:
                    kwargs["system_prompt"] = system_prompt
                if model:
                    kwargs["model"] = model
                if output_schema:
                    kwargs["output_schema"] = output_schema
                if user_location:
                    kwargs["user_location"] = user_location

                result = self.exa.answer(query, **kwargs)

                # Handle response
                answer_text = getattr(result, 'answer', None)
                if answer_text is None:
                    answer_text = getattr(result, 'text', str(result))

                citations = getattr(result, 'citations', [])
                sources = getattr(result, 'sources', [])

                return {
                    "success": True,
                    "answer": answer_text,
                    "citations": citations,
                    "sources": sources,
                    "data": self._to_dict(result),
                }
            else:
                # Fallback: search + summarize
                search = self.exa_search_and_contents(
                    query,
                    num_results=5,
                    type="auto",
                    text_max_chars=2000,
                    summary=True,
                )
                if search.get("success"):
                    summaries = [r.get("summary", "") for r in search["results"] if r.get("summary")]
                    return {
                        "success": True,
                        "answer": "\n\n".join(summaries[:3]) if summaries else "No answer found",
                        "sources": search["results"],
                        "fallback": True,
                    }
                return search
        except Exception as e:
            logger.error(f"Exa answer error: {e}")
            return {"success": False, "error": str(e)}

    def exa_stream_answer(
        self,
        query: str,
        *,
        text: bool = False,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        output_schema: Optional[Dict] = None,
        user_location: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Streaming answer (NEW IN v3.0).

        Yields answer chunks as they're generated.

        Args:
            query: Question to answer
            text: Include full text in results
            system_prompt: Guide the LLM's behavior
            model: "exa" or "exa-pro"
            output_schema: JSON schema for structured output
            user_location: ISO country code

        Yields:
            Answer text chunks

        Example:
            for chunk in engine.exa_stream_answer("What is RAG?"):
                print(chunk, end='', flush=True)
        """
        if not self.exa:
            yield "Error: Exa not available"
            return

        try:
            if hasattr(self.exa, 'stream_answer'):
                kwargs = {"text": text}
                if system_prompt:
                    kwargs["system_prompt"] = system_prompt
                if model:
                    kwargs["model"] = model
                if output_schema:
                    kwargs["output_schema"] = output_schema
                if user_location:
                    kwargs["user_location"] = user_location

                response = self.exa.stream_answer(query, **kwargs)

                # Handle streaming response
                for chunk in response:
                    if hasattr(chunk, 'text'):
                        yield chunk.text
                    elif isinstance(chunk, str):
                        yield chunk
                    else:
                        yield str(chunk)
            else:
                # Fallback: non-streaming answer
                result = self.exa_answer(query, text=text)
                if result.get("success"):
                    yield result.get("answer", "")
                else:
                    yield f"Error: {result.get('error', 'Unknown error')}"
        except Exception as e:
            logger.error(f"Exa stream_answer error: {e}")
            yield f"Error: {str(e)}"

    # =========================================================================
    # EXA - RESEARCH CLIENT (ENHANCED IN v3.0)
    # =========================================================================

    def exa_research_create(
        self,
        instructions: str,
        *,
        model: str = "exa-research-fast",
        output_schema: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Create a new research task.

        Args:
            instructions: Research instructions
            model: exa-research-fast or exa-research
            output_schema: JSON schema for output

        Returns:
            Research task info with ID
        """
        if not self.exa_research:
            return {"error": "Exa Research not available", "success": False}

        try:
            kwargs = {"instructions": instructions, "model": model}
            if output_schema:
                kwargs["output_schema"] = output_schema

            result = self.exa_research.create(**kwargs)

            return {
                "success": True,
                "research_id": getattr(result, 'id', str(result)),
                "status": getattr(result, 'status', 'created'),
                "data": self._to_dict(result),
            }
        except Exception as e:
            logger.error(f"Exa research create error: {e}")
            return {"success": False, "error": str(e)}

    def exa_research_get(
        self,
        research_id: str,
        *,
        stream: bool = False,
        events: bool = False,
        output_schema: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Get research task status/results.

        Args:
            research_id: Research task ID
            stream: Enable streaming (NEW)
            events: Include events (NEW)
            output_schema: Schema for output

        Returns:
            Research status and results
        """
        if not self.exa_research:
            return {"error": "Exa Research not available", "success": False}

        try:
            kwargs = {}
            if stream:
                kwargs["stream"] = True
            if events:
                kwargs["events"] = True
            if output_schema:
                kwargs["output_schema"] = output_schema

            result = self.exa_research.get(research_id, **kwargs)

            return {
                "success": True,
                "research_id": research_id,
                "status": getattr(result, 'status', 'unknown'),
                "output": getattr(result, 'output', None),
                "cost": getattr(result, 'cost', None),
                "events": getattr(result, 'events', []),
                "data": self._to_dict(result),
            }
        except Exception as e:
            logger.error(f"Exa research get error: {e}")
            return {"success": False, "error": str(e)}

    def exa_research_list(
        self,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        List research tasks (NEW IN v3.0).

        Args:
            cursor: Pagination cursor
            limit: Max results

        Returns:
            List of research tasks
        """
        if not self.exa_research:
            return {"error": "Exa Research not available", "success": False}

        try:
            kwargs = {}
            if cursor:
                kwargs["cursor"] = cursor
            if limit:
                kwargs["limit"] = limit

            result = self.exa_research.list(**kwargs)

            return {
                "success": True,
                "tasks": self._to_dict(result),
            }
        except Exception as e:
            logger.error(f"Exa research list error: {e}")
            return {"success": False, "error": str(e)}

    def exa_deep_research(
        self,
        instructions: str,
        *,
        model: str = "exa-research-fast",
        output_schema: Optional[Dict] = None,
        poll_interval: int = 2,
        timeout: int = 600,
    ) -> Dict[str, Any]:
        """
        Deep research with polling until complete.

        Args:
            instructions: Research question/instructions
            model: exa-research-fast or exa-research
            output_schema: JSON schema
            poll_interval: Polling interval (s)
            timeout: Max wait (s)

        Returns:
            Complete research results
        """
        if not self.exa_research:
            return {"error": "Exa Research not available", "success": False}

        try:
            # Create task
            create_kwargs = {"instructions": instructions, "model": model}
            if output_schema:
                create_kwargs["output_schema"] = output_schema

            research = self.exa_research.create(**create_kwargs)
            research_id = getattr(research, 'id', str(research))

            # Poll until finished
            start_time = time.time()
            while time.time() - start_time < timeout:
                status = self.exa_research.get(research_id)

                if hasattr(status, 'status'):
                    status_str = str(status.status).lower()
                    if status_str in ('completed', 'finished', 'done'):
                        return {
                            "success": True,
                            "research_id": research_id,
                            "status": status_str,
                            "output": getattr(status, 'output', None),
                            "cost": getattr(status, 'cost', None),
                        }
                    elif status_str in ('failed', 'error'):
                        return {
                            "success": False,
                            "research_id": research_id,
                            "status": status_str,
                            "error": getattr(status, 'error', 'Research failed'),
                        }

                time.sleep(poll_interval)

            return {
                "success": False,
                "research_id": research_id,
                "status": "timeout",
                "error": f"Research timed out after {timeout}s",
            }
        except Exception as e:
            logger.error(f"Exa deep research error: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # COMBINED RESEARCH METHODS
    # =========================================================================

    def comprehensive_research(
        self,
        topic: str,
        *,
        num_sources: int = 15,
        search_type: str = "deep",
        categories: Optional[List[str]] = None,
        crawl_top_sources: int = 5,
        extract_schema: Optional[Dict] = None,
        use_agent: bool = False,
        agent_prompt: Optional[str] = None,
        use_answer: bool = True,
    ) -> ResearchResult:
        """
        Comprehensive multi-source research.

        Workflow:
        1. Exa answer for direct response (NEW)
        2. Exa deep search across categories
        3. Firecrawl batch scrape top sources
        4. Optional: Firecrawl extract
        5. Optional: Firecrawl agent

        Args:
            topic: Research topic
            num_sources: Total sources
            search_type: Exa search type
            categories: Categories to search
            crawl_top_sources: Sources to deep crawl
            extract_schema: Extraction schema
            use_agent: Use Firecrawl agent
            agent_prompt: Agent instructions
            use_answer: Use Exa answer API (NEW)

        Returns:
            Comprehensive ResearchResult
        """
        result = ResearchResult(query=topic)
        all_sources = []
        all_content = []

        logger.info(f"[RESEARCH] Starting comprehensive research: {topic}")

        # Step 0: Quick answer (NEW)
        if use_answer and self.exa:
            logger.info("[ANSWER] Getting direct answer...")
            answer = self.exa_answer(topic, num_results=5)
            if answer.get("success"):
                result.answer = answer.get("answer")
                logger.info("[OK] Got direct answer")

        # Step 1: Multi-category search
        if self.exa:
            search_categories = categories or [None, "research paper", "news"]

            for cat in search_categories:
                logger.info(f"[SEARCH] Category: {cat or 'general'}")
                search = self.exa_search_and_contents(
                    topic,
                    num_results=num_sources // len(search_categories),
                    type=search_type,
                    category=cat,
                    text_max_chars=3000,
                    highlights=True,
                    livecrawl="preferred",
                )

                if search.get("success"):
                    sources = search.get("results", [])
                    all_sources.extend(sources)
                    for s in sources:
                        if s.get("text"):
                            all_content.append(s["text"])
                    logger.info(f"[OK] Found {len(sources)} in {cat or 'general'}")
                else:
                    result.errors.append(f"Exa search ({cat}) failed")

        # Step 2: Deep scrape
        if self.firecrawl and all_sources and crawl_top_sources > 0:
            urls = [s["url"] for s in all_sources[:crawl_top_sources]]
            logger.info(f"[CRAWL] Scraping {len(urls)} sources...")

            batch = self.batch_scrape(urls, formats=["markdown"])
            if batch.get("success"):
                data = batch.get("data", {})
                pages = data.get("data", []) if isinstance(data, dict) else data
                for page in pages:
                    if isinstance(page, dict) and page.get("markdown"):
                        all_content.append(page["markdown"])
                logger.info(f"[OK] Scraped {len(pages)} pages")

        # Step 3: Extraction
        if extract_schema and self.firecrawl and all_sources:
            logger.info("[EXTRACT] Extracting structured data...")
            extract = self.extract(
                [s["url"] for s in all_sources[:3]],
                schema=extract_schema,
                enable_web_search=True,
            )
            if extract.get("success"):
                result.extracted_data = extract.get("data")

        # Step 4: Agent
        if use_agent and self.firecrawl and all_sources:
            logger.info("[AGENT] Running agent...")
            prompt = agent_prompt or f"Research: {topic}"
            agent_result = self.agent(
                [all_sources[0]["url"]],
                prompt=prompt,
                model="spark-1-pro",
                max_credits=50,
            )
            if agent_result.get("success"):
                result.agent_response = agent_result.get("data")

        # Compile
        result.sources = all_sources
        result.content = all_content
        result.metadata = {
            "num_sources": len(all_sources),
            "total_content_length": sum(len(c) for c in all_content),
            "search_type": search_type,
            "categories": categories or ["general"],
            "crawled": crawl_top_sources,
            "used_agent": use_agent,
            "has_answer": result.answer is not None,
        }
        result.success = len(all_sources) > 0

        logger.info(f"[DONE] {len(all_sources)} sources, {result.metadata['total_content_length']} chars")
        return result

    def quick_research(
        self,
        query: str,
        num_results: int = 5,
    ) -> ResearchResult:
        """Quick research with minimal API calls."""
        result = ResearchResult(query=query)

        if self.exa:
            search = self.exa_search_and_contents(
                query,
                num_results=num_results,
                type="fast",
                text_max_chars=1500,
            )
            if search.get("success"):
                result.sources = search.get("results", [])
                result.content = [s.get("text", "") for s in result.sources if s.get("text")]

        result.metadata = {"num_sources": len(result.sources), "mode": "quick"}
        result.success = len(result.sources) > 0
        return result

    def deep_research(
        self,
        topic: str,
        *,
        use_exa_research: bool = True,
        use_firecrawl_agent: bool = True,
        additional_queries: Optional[List[str]] = None,
        use_answer: bool = True,
    ) -> ResearchResult:
        """Maximum depth research."""
        result = ResearchResult(query=topic)

        # Answer first
        if use_answer and self.exa:
            logger.info("[DEEP] Getting answer...")
            answer = self.exa_answer(topic)
            if answer.get("success"):
                result.answer = answer.get("answer")

        # Deep search
        if self.exa:
            logger.info("[DEEP] Deep search...")
            # Build query with additional queries if provided
            search_query = topic
            if additional_queries:
                search_query = f"{topic} | {' | '.join(additional_queries)}"
            search = self.exa_search_and_contents(
                search_query,
                num_results=20,
                type="deep",
                text_max_chars=4000,
                highlights=True,
                summary=True,
                livecrawl="always",
            )
            if search.get("success"):
                result.sources = search.get("results", [])
                result.content = [s.get("text", "") for s in result.sources if s.get("text")]

        # Exa Research
        if use_exa_research and self.exa_research:
            logger.info("[DEEP] Research agent...")
            research = self.exa_deep_research(
                f"Comprehensive analysis: {topic}",
                model="exa-research",
                timeout=300,
            )
            if research.get("success"):
                result.deep_research = research

        # Firecrawl agent
        if use_firecrawl_agent and self.firecrawl and result.sources:
            logger.info("[DEEP] Firecrawl agent...")
            agent = self.agent(
                [s["url"] for s in result.sources[:3]],
                prompt=f"Research: {topic}",
                model="spark-1-pro",
            )
            if agent.get("success"):
                result.agent_response = agent.get("data")

        result.metadata = {
            "num_sources": len(result.sources),
            "total_content_length": sum(len(c) for c in result.content),
            "used_exa_research": use_exa_research,
            "used_firecrawl_agent": use_firecrawl_agent,
            "has_answer": result.answer is not None,
            "mode": "deep",
        }
        result.success = len(result.sources) > 0
        return result

    def get_llm_context(
        self,
        topic: str,
        *,
        max_tokens: int = 50000,
        num_sources: int = 10,
        include_summaries: bool = True,
    ) -> str:
        """Get LLM-ready context for a topic."""
        result = self.comprehensive_research(topic, num_sources=num_sources)

        parts = [
            f"# Research: {topic}",
            f"*Generated: {result.timestamp}*",
            f"*Sources: {len(result.sources)}*",
            "",
        ]

        # Add answer if available
        if result.answer:
            parts.extend([
                "## Direct Answer",
                result.answer,
                "",
            ])

        parts.extend(["## Sources", ""])

        for i, source in enumerate(result.sources, 1):
            line = f"{i}. [{source.get('title', 'Unknown')}]({source.get('url', '')})"
            if include_summaries and source.get('summary'):
                line += f"\n   > {source['summary'][:200]}..."
            parts.append(line)

        parts.extend(["", "## Content", ""])

        max_chars = max_tokens * 4
        current_chars = sum(len(p) for p in parts)

        for i, content in enumerate(result.content):
            if current_chars + len(content) > max_chars:
                remaining = max_chars - current_chars
                if remaining > 100:
                    parts.append(f"### Source {i+1} (truncated)")
                    parts.append(content[:remaining] + "...")
                break
            parts.append(f"### Source {i+1}")
            parts.append(content)
            parts.append("")
            current_chars += len(content) + 50

        if result.extracted_data:
            parts.extend([
                "",
                "## Extracted Data",
                "```json",
                json.dumps(result.extracted_data, indent=2, default=str)[:5000],
                "```",
            ])

        return "\n".join(parts)

    # =========================================================================
    # AUTO-RESEARCH (NEW IN v3.0)
    # =========================================================================

    def auto_research(
        self,
        context: str,
        *,
        triggers: Optional[List[str]] = None,
        depth: str = "quick",
    ) -> Optional[ResearchResult]:
        """
        Auto-research based on context triggers.

        Analyzes context and determines if research is needed.

        Args:
            context: Current context/conversation
            triggers: Keywords that trigger research
            depth: quick, comprehensive, or deep

        Returns:
            ResearchResult if triggered, None otherwise
        """
        default_triggers = [
            "research", "find out", "look up", "search for",
            "what is", "how does", "explain", "learn about",
            "latest", "recent", "current", "news about",
        ]
        triggers = triggers or default_triggers

        context_lower = context.lower()
        should_research = any(t in context_lower for t in triggers)

        if not should_research:
            return None

        # Extract topic (simple heuristic)
        topic = context.strip()
        for trigger in triggers:
            if trigger in context_lower:
                idx = context_lower.find(trigger)
                topic = context[idx + len(trigger):].strip()
                break

        if not topic or len(topic) < 3:
            return None

        logger.info(f"[AUTO-RESEARCH] Triggered for: {topic}")

        if depth == "quick":
            return self.quick_research(topic)
        elif depth == "deep":
            return self.deep_research(topic)
        else:
            return self.comprehensive_research(topic)

    # =========================================================================
    # SESSION INTEGRATION (NEW IN v3.0)
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get engine status for session integration."""
        return {
            "version": "3.0",
            "firecrawl_available": self.firecrawl is not None,
            "firecrawl_async_available": self.firecrawl_async is not None,
            "exa_available": self.exa is not None,
            "exa_research_available": self.exa_research is not None,
            "watcher_available": FIRECRAWL_WATCHER_AVAILABLE,
            "active_jobs": len(self._active_jobs),
            "capabilities": {
                "firecrawl": [
                    "scrape", "crawl", "batch_scrape", "map", "search",
                    "extract", "agent", "watcher",
                    "start_crawl", "get_crawl_status", "cancel_crawl",
                    "start_batch_scrape", "get_batch_scrape_status",
                    "start_extract", "get_extract_status",
                    "start_agent", "get_agent_status", "cancel_agent",
                    "get_concurrency", "get_credit_usage", "get_token_usage",
                ] if self.firecrawl else [],
                "exa": [
                    "search", "search_and_contents",
                    "find_similar", "find_similar_and_contents",
                    "get_contents", "answer", "stream_answer",
                ] if self.exa else [],
                "exa_research": [
                    "create", "get", "list", "deep_research",
                ] if self.exa_research else [],
            },
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Research Engine v3.0 - ULTIMATE Auto-Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python research_engine.py test
  python research_engine.py status
  python research_engine.py search "AI agents 2026"
  python research_engine.py answer "What is RAG?"
  python research_engine.py scrape https://example.com
  python research_engine.py crawl https://docs.example.com --depth 2
  python research_engine.py research "quantum computing" --deep
  python research_engine.py agent https://example.com "Find pricing"
        """
    )
    parser.add_argument("command", choices=[
        "search", "scrape", "crawl", "map", "extract", "agent",
        "research", "deep", "quick", "answer", "test", "status", "credits"
    ])
    parser.add_argument("query", nargs="?", help="Query or URL")
    parser.add_argument("prompt", nargs="?", help="Prompt for agent/extract")
    parser.add_argument("--sources", type=int, default=10, help="Number of sources")
    parser.add_argument("--depth", type=int, default=2, help="Crawl depth")
    parser.add_argument("--type", default="auto", help="Search type")
    parser.add_argument("--category", help="Search category")
    parser.add_argument("--deep", action="store_true", help="Use deep research")
    parser.add_argument("--output", "-o", help="Output file")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    engine = ResearchEngine()

    if args.command == "test":
        print("=" * 60)
        print("RESEARCH ENGINE v3.0 - ULTIMATE TEST")
        print("=" * 60)
        print(f"Firecrawl: {'[OK]' if engine.firecrawl else '[NO]'}")
        print(f"Firecrawl Async: {'[OK]' if engine.firecrawl_async else '[NO]'}")
        print(f"Exa: {'[OK]' if engine.exa else '[NO]'}")
        print(f"Exa Research: {'[OK]' if engine.exa_research else '[NO]'}")
        print(f"Watcher: {'[OK]' if FIRECRAWL_WATCHER_AVAILABLE else '[NO]'}")
        print()

        if engine.firecrawl:
            print("[TEST] Firecrawl scrape...")
            result = engine.scrape("https://example.com")
            print(f"  Result: {'[OK]' if result.get('success') else '[FAIL] ' + result.get('error', '')}")

        if engine.exa:
            print("\n[TEST] Exa search...")
            result = engine.exa_search("AI agents 2026", num_results=3)
            print(f"  Result: {'[OK]' if result.get('success') else '[FAIL] ' + result.get('error', '')}")
            if result.get("results"):
                print(f"  Found: {len(result['results'])} results")
                for r in result['results'][:2]:
                    print(f"    - {r.get('title', 'No title')[:50]}")

            print("\n[TEST] Exa answer...")
            result = engine.exa_answer("What is RAG?", text=True)
            print(f"  Result: {'[OK]' if result.get('success') else '[FAIL] ' + result.get('error', '')}")
            if result.get("answer"):
                # Handle Unicode safely for Windows console
                answer_preview = str(result['answer'])[:100]
                answer_preview = answer_preview.encode('ascii', 'replace').decode('ascii')
                print(f"  Answer: {answer_preview}...")

        print("\n" + "=" * 60)
        print("[DONE] Test complete!")
        return

    if args.command == "status":
        status = engine.get_status()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.command == "credits":
        result = engine.get_credit_usage()
        print(json.dumps(result, indent=2, default=str))
        return

    if not args.query and args.command not in ["test", "status", "credits"]:
        parser.error(f"{args.command} requires a query/URL")

    result = None

    if args.command == "search":
        result = engine.exa_search_and_contents(
            args.query,
            num_results=args.sources,
            type=args.type,
            category=args.category,
        )
    elif args.command == "answer":
        result = engine.exa_answer(args.query, num_results=args.sources)
    elif args.command == "scrape":
        result = engine.scrape(args.query)
    elif args.command == "crawl":
        result = engine.crawl(args.query, max_depth=args.depth)
    elif args.command == "map":
        result = engine.map_site(args.query)
    elif args.command == "extract":
        result = engine.extract(args.query, prompt=args.prompt or "Extract key information")
    elif args.command == "agent":
        result = engine.agent(args.query, prompt=args.prompt or "Gather information")
    elif args.command == "quick":
        res = engine.quick_research(args.query, num_results=args.sources)
        result = {
            "query": res.query,
            "sources": res.sources,
            "answer": res.answer,
            "success": res.success,
        }
    elif args.command == "deep":
        res = engine.deep_research(args.query)
        result = {
            "query": res.query,
            "sources": len(res.sources),
            "content_length": sum(len(c) for c in res.content),
            "has_answer": res.answer is not None,
            "has_deep_research": res.deep_research is not None,
            "success": res.success,
        }
    elif args.command == "research":
        if args.deep:
            res = engine.deep_research(args.query)
        else:
            res = engine.comprehensive_research(args.query, num_sources=args.sources)
        result = {
            "query": res.query,
            "sources": res.sources,
            "answer": res.answer,
            "content_length": sum(len(c) for c in res.content),
            "metadata": res.metadata,
            "success": res.success,
        }

    if result:
        output = json.dumps(result, indent=2, default=str)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"[OK] Output written to {args.output}")
        else:
            print(output)


# Auto-load on import
_engine = ResearchEngine.auto_load()


def get_engine() -> ResearchEngine:
    """Get the auto-loaded research engine singleton."""
    global _engine
    if _engine is None:
        _engine = ResearchEngine.auto_load()
    return _engine


# Convenience aliases
research = get_engine  # Shorthand: from research_engine import research; r = research()


if __name__ == "__main__":
    main()
