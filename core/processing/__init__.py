#!/usr/bin/env python3
"""
Processing Layer - Web Crawling & Code Analysis
Part of the V33 Architecture (Layer 7) - Phase 9 Production Fix.

Provides unified access to four processing SDKs:
- crawl4ai: LLM-friendly async web crawler with structured output
- firecrawl: API-based web scraping with markdown conversion
- aider: AI-powered code editing with git integration
- ast-grep: Structural code search and refactoring

NO STUBS: All SDKs must be explicitly installed and configured.
Missing SDKs raise SDKNotAvailableError with install instructions.
Misconfigured SDKs raise SDKConfigurationError with missing config.

Usage:
    from core.processing import (
        # Exceptions
        SDKNotAvailableError,
        SDKConfigurationError,

        # Crawl4AI
        get_crawl4ai_crawler,
        Crawl4AIClient,
        CRAWL4AI_AVAILABLE,

        # Firecrawl
        get_firecrawl_client,
        FirecrawlClient,
        FIRECRAWL_AVAILABLE,

        # Aider
        get_aider_coder,
        AiderClient,
        AIDER_AVAILABLE,

        # AST-Grep
        get_astgrep_client,
        ASTGrepClient,
        ASTGREP_AVAILABLE,

        # Factory
        ProcessingFactory,
    )

    # Quick start with explicit error handling
    try:
        crawler = get_crawl4ai_crawler()
    except SDKNotAvailableError as e:
        print(f"Install: {e.install_cmd}")
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Optional, List, Dict, Union
from dataclasses import dataclass, field

# Import exceptions from observability layer
from core.observability import (
    SDKNotAvailableError,
    SDKConfigurationError,
)


# ============================================================================
# SDK Availability Checks - Import-time validation
# ============================================================================

# Crawl4AI - Async Web Crawler
CRAWL4AI_AVAILABLE = False
CRAWL4AI_ERROR = None
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    from crawl4ai.extraction_strategy import LLMExtractionStrategy, JsonCssExtractionStrategy
    CRAWL4AI_AVAILABLE = True
except Exception as e:
    CRAWL4AI_ERROR = str(e)

# Firecrawl - API Web Scraper
FIRECRAWL_AVAILABLE = False
FIRECRAWL_ERROR = None
try:
    from firecrawl import FirecrawlApp
    FIRECRAWL_AVAILABLE = True
except Exception as e:
    FIRECRAWL_ERROR = str(e)

# Aider - AI Code Editor
AIDER_AVAILABLE = False
AIDER_ERROR = None
try:
    from aider.coders import Coder
    from aider.models import Model as AiderModel
    from aider.io import InputOutput
    AIDER_AVAILABLE = True
except Exception as e:
    AIDER_ERROR = str(e)

# AST-Grep - Structural Code Search
ASTGREP_AVAILABLE = False
ASTGREP_ERROR = None
try:
    from ast_grep_py import SgRoot
    ASTGREP_AVAILABLE = True
except Exception as e:
    ASTGREP_ERROR = str(e)


# ============================================================================
# Crawl4AI Types and Implementation
# ============================================================================

class ExtractionMode(str, Enum):
    """Crawl4AI extraction modes."""
    RAW = "raw"              # Raw HTML/text
    MARKDOWN = "markdown"     # Converted to markdown
    LLM = "llm"              # LLM-structured extraction
    JSON_CSS = "json_css"    # CSS selector-based JSON


@dataclass
class CrawlConfig:
    """Configuration for Crawl4AI."""
    headless: bool = True
    browser_type: str = "chromium"  # chromium, firefox, webkit
    user_agent: Optional[str] = None
    timeout: int = 30000
    viewport_width: int = 1280
    viewport_height: int = 720
    wait_for: Optional[str] = None  # CSS selector to wait for
    remove_selectors: List[str] = field(default_factory=list)


@dataclass
class CrawlResult:
    """Result from a crawl operation."""
    url: str
    success: bool
    content: str
    markdown: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class Crawl4AIClient:
    """
    Crawl4AI client for LLM-friendly web crawling.

    Provides async web crawling with automatic content extraction
    optimized for LLM consumption.
    """

    def __init__(
        self,
        config: Optional[CrawlConfig] = None,
        **kwargs: Any,
    ):
        """
        Initialize Crawl4AI client.

        Args:
            config: Optional CrawlConfig
            **kwargs: Override config values
        """
        if not CRAWL4AI_AVAILABLE:
            raise SDKNotAvailableError(
                sdk_name="crawl4ai",
                install_cmd="pip install crawl4ai>=0.4.0",
                docs_url="https://docs.crawl4ai.com/"
            )

        self.config = config or CrawlConfig()
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self._crawler = None

    async def _get_crawler(self) -> "AsyncWebCrawler":
        """Get or create the crawler instance."""
        if self._crawler is None:
            browser_config = BrowserConfig(
                headless=self.config.headless,
                browser_type=self.config.browser_type,
                user_agent=self.config.user_agent,
                viewport_width=self.config.viewport_width,
                viewport_height=self.config.viewport_height,
            )
            self._crawler = AsyncWebCrawler(config=browser_config)
            await self._crawler.start()
        return self._crawler

    async def crawl(
        self,
        url: str,
        mode: ExtractionMode = ExtractionMode.MARKDOWN,
        schema: Optional[Dict[str, Any]] = None,
        css_selector: Optional[str] = None,
        llm_instructions: Optional[str] = None,
    ) -> CrawlResult:
        """
        Crawl a URL and extract content.

        Args:
            url: URL to crawl
            mode: Extraction mode
            schema: JSON schema for LLM extraction
            css_selector: CSS selector for JSON_CSS mode
            llm_instructions: Instructions for LLM extraction

        Returns:
            CrawlResult with extracted content
        """
        crawler = await self._get_crawler()

        # Build extraction strategy based on mode
        extraction_strategy = None
        if mode == ExtractionMode.LLM and schema:
            extraction_strategy = LLMExtractionStrategy(
                schema=schema,
                instruction=llm_instructions or "Extract the relevant data",
            )
        elif mode == ExtractionMode.JSON_CSS and css_selector:
            extraction_strategy = JsonCssExtractionStrategy(
                schema={
                    "name": "extracted",
                    "baseSelector": css_selector,
                }
            )

        run_config = CrawlerRunConfig(
            wait_for=self.config.wait_for,
            remove_selectors=self.config.remove_selectors,
            extraction_strategy=extraction_strategy,
        )

        try:
            result = await crawler.arun(url=url, config=run_config)

            return CrawlResult(
                url=url,
                success=result.success,
                content=result.html or "",
                markdown=result.markdown if hasattr(result, "markdown") else None,
                extracted_data=result.extracted_content if hasattr(result, "extracted_content") else None,
                links=result.links if hasattr(result, "links") else [],
                images=result.images if hasattr(result, "images") else [],
                metadata={
                    "status_code": getattr(result, "status_code", None),
                    "response_headers": getattr(result, "response_headers", {}),
                },
            )
        except Exception as e:
            return CrawlResult(
                url=url,
                success=False,
                content="",
                error=str(e),
            )

    async def crawl_many(
        self,
        urls: List[str],
        mode: ExtractionMode = ExtractionMode.MARKDOWN,
        max_concurrent: int = 5,
    ) -> List[CrawlResult]:
        """
        Crawl multiple URLs concurrently.

        Args:
            urls: List of URLs to crawl
            mode: Extraction mode
            max_concurrent: Maximum concurrent crawls

        Returns:
            List of CrawlResult
        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)

        async def crawl_with_semaphore(url: str) -> CrawlResult:
            async with semaphore:
                return await self.crawl(url, mode=mode)

        results = await asyncio.gather(
            *[crawl_with_semaphore(url) for url in urls],
            return_exceptions=True,
        )

        return [
            r if isinstance(r, CrawlResult)
            else CrawlResult(url=urls[i], success=False, content="", error=str(r))
            for i, r in enumerate(results)
        ]

    async def close(self) -> None:
        """Close the crawler."""
        if self._crawler:
            await self._crawler.close()
            self._crawler = None


# ============================================================================
# Firecrawl Types and Implementation
# ============================================================================

@dataclass
class FirecrawlConfig:
    """Configuration for Firecrawl."""
    api_key: Optional[str] = None
    timeout: int = 30
    wait_until: str = "networkidle"  # load, domcontentloaded, networkidle


@dataclass
class ScrapeResult:
    """Result from a Firecrawl scrape."""
    url: str
    success: bool
    markdown: str
    html: Optional[str] = None
    links: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class FirecrawlClient:
    """
    Firecrawl client for API-based web scraping.

    Provides clean markdown conversion from any webpage
    with automatic cleanup and formatting.
    """

    def __init__(
        self,
        config: Optional[FirecrawlConfig] = None,
        **kwargs: Any,
    ):
        """
        Initialize Firecrawl client.

        Args:
            config: Optional FirecrawlConfig
            **kwargs: Override config values
        """
        if not FIRECRAWL_AVAILABLE:
            raise SDKNotAvailableError(
                sdk_name="firecrawl",
                install_cmd="pip install firecrawl-py>=1.0.0",
                docs_url="https://docs.firecrawl.dev/"
            )

        self.config = config or FirecrawlConfig()
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Firecrawl client."""
        api_key = self.config.api_key or os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise SDKConfigurationError(
                sdk_name="firecrawl",
                missing_config=["FIRECRAWL_API_KEY"],
                example="FIRECRAWL_API_KEY=fc-..."
            )

        self._client = FirecrawlApp(api_key=api_key)

    def scrape(
        self,
        url: str,
        formats: Optional[List[str]] = None,
        only_main_content: bool = True,
    ) -> ScrapeResult:
        """
        Scrape a URL and convert to markdown.

        Args:
            url: URL to scrape
            formats: Output formats (markdown, html, links)
            only_main_content: Remove nav/footer/ads

        Returns:
            ScrapeResult with content
        """
        try:
            result = self._client.scrape_url(
                url,
                params={
                    "formats": formats or ["markdown"],
                    "onlyMainContent": only_main_content,
                    "waitFor": self.config.timeout * 1000,
                }
            )

            return ScrapeResult(
                url=url,
                success=True,
                markdown=result.get("markdown", ""),
                html=result.get("html"),
                links=result.get("links", []),
                metadata=result.get("metadata", {}),
            )
        except Exception as e:
            return ScrapeResult(
                url=url,
                success=False,
                markdown="",
                error=str(e),
            )

    def crawl_site(
        self,
        url: str,
        max_depth: int = 2,
        max_pages: int = 10,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[ScrapeResult]:
        """
        Crawl an entire site.

        Args:
            url: Starting URL
            max_depth: Maximum link depth
            max_pages: Maximum pages to crawl
            include_patterns: URL patterns to include
            exclude_patterns: URL patterns to exclude

        Returns:
            List of ScrapeResult for each page
        """
        try:
            result = self._client.crawl_url(
                url,
                params={
                    "limit": max_pages,
                    "maxDepth": max_depth,
                    "includePaths": include_patterns or [],
                    "excludePaths": exclude_patterns or [],
                },
                poll_interval=2,
            )

            results = []
            for page in result.get("data", []):
                results.append(ScrapeResult(
                    url=page.get("url", url),
                    success=True,
                    markdown=page.get("markdown", ""),
                    html=page.get("html"),
                    links=page.get("links", []),
                    metadata=page.get("metadata", {}),
                ))

            return results

        except Exception as e:
            return [ScrapeResult(
                url=url,
                success=False,
                markdown="",
                error=str(e),
            )]

    def map_site(self, url: str) -> List[str]:
        """
        Get all URLs from a site.

        Args:
            url: Site URL

        Returns:
            List of discovered URLs
        """
        try:
            result = self._client.map_url(url)
            return result.get("links", [])
        except Exception as e:
            return []


# ============================================================================
# Aider Types and Implementation
# ============================================================================

@dataclass
class AiderConfig:
    """Configuration for Aider."""
    model: str = "claude-sonnet-4-20250514"
    provider: str = "anthropic"
    auto_commits: bool = False
    pretty: bool = False
    stream: bool = True


@dataclass
class EditRequest:
    """Request for code editing."""
    instruction: str
    files: List[str]
    context_files: Optional[List[str]] = None


@dataclass
class CodeEditResult:
    """Result from an Aider edit."""
    success: bool
    files_modified: List[str]
    diff: Optional[str] = None
    commit_hash: Optional[str] = None
    message: str = ""
    error: Optional[str] = None


class AiderClient:
    """
    Aider client for AI-powered code editing.

    Provides intelligent code editing with git integration
    and multi-file context awareness.
    """

    def __init__(
        self,
        config: Optional[AiderConfig] = None,
        **kwargs: Any,
    ):
        """
        Initialize Aider client.

        Args:
            config: Optional AiderConfig
            **kwargs: Override config values
        """
        if not AIDER_AVAILABLE:
            raise SDKNotAvailableError(
                sdk_name="aider",
                install_cmd="pip install aider-chat>=0.50.0",
                docs_url="https://aider.chat/docs/"
            )

        self.config = config or AiderConfig()
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self._check_api_key()
        self._coder = None

    def _check_api_key(self) -> None:
        """Check for required API key."""
        if self.config.provider == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise SDKConfigurationError(
                    sdk_name="aider",
                    missing_config=["ANTHROPIC_API_KEY"],
                    example="ANTHROPIC_API_KEY=sk-ant-..."
                )
        elif self.config.provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise SDKConfigurationError(
                    sdk_name="aider",
                    missing_config=["OPENAI_API_KEY"],
                    example="OPENAI_API_KEY=sk-..."
                )

    def _get_coder(self, files: List[str]) -> "Coder":
        """Get or create the Aider coder instance."""
        model = AiderModel(self.config.model)
        io = InputOutput(
            pretty=self.config.pretty,
            yes=True,  # Auto-accept
        )

        return Coder.create(
            model=model,
            fnames=files,
            io=io,
            auto_commits=self.config.auto_commits,
            stream=self.config.stream,
        )

    def edit(self, request: EditRequest) -> CodeEditResult:
        """
        Edit code files based on instruction.

        Args:
            request: EditRequest with instruction and files

        Returns:
            CodeEditResult with changes
        """
        try:
            coder = self._get_coder(request.files)

            # Add context files if provided
            if request.context_files:
                for f in request.context_files:
                    coder.add_rel_fname(f)

            # Run the edit
            result = coder.run(request.instruction)

            return CodeEditResult(
                success=True,
                files_modified=list(coder.aider_edited_files),
                diff=getattr(coder, "last_diff", None),
                commit_hash=getattr(coder, "last_commit_hash", None),
                message=str(result) if result else "Edit completed",
            )
        except Exception as e:
            return CodeEditResult(
                success=False,
                files_modified=[],
                error=str(e),
            )

    def ask(self, question: str, files: List[str]) -> str:
        """
        Ask a question about code files.

        Args:
            question: Question to ask
            files: Files to analyze

        Returns:
            Answer string
        """
        try:
            coder = self._get_coder(files)
            result = coder.run(f"/ask {question}")
            return str(result) if result else ""
        except Exception as e:
            return f"Error: {e}"


# ============================================================================
# AST-Grep Types and Implementation
# ============================================================================

@dataclass
class ASTGrepConfig:
    """Configuration for AST-Grep."""
    language: str = "python"


@dataclass
class MatchResult:
    """Result from an AST-Grep match."""
    file_path: str
    line_start: int
    line_end: int
    column_start: int
    column_end: int
    matched_text: str
    captures: Dict[str, str] = field(default_factory=dict)


@dataclass
class RefactorResult:
    """Result from an AST-Grep refactor."""
    success: bool
    files_modified: int
    matches_replaced: int
    diff: Optional[str] = None
    error: Optional[str] = None


class ASTGrepClient:
    """
    AST-Grep client for structural code search.

    Provides AST-based pattern matching for precise
    code search and refactoring.
    """

    def __init__(
        self,
        config: Optional[ASTGrepConfig] = None,
        **kwargs: Any,
    ):
        """
        Initialize AST-Grep client.

        Args:
            config: Optional ASTGrepConfig
            **kwargs: Override config values
        """
        if not ASTGREP_AVAILABLE:
            raise SDKNotAvailableError(
                sdk_name="ast-grep",
                install_cmd="pip install ast-grep-py>=0.25.0",
                docs_url="https://ast-grep.github.io/"
            )

        self.config = config or ASTGrepConfig()
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def search(
        self,
        pattern: str,
        source: str,
        language: Optional[str] = None,
    ) -> List[MatchResult]:
        """
        Search for AST pattern in source code.

        Args:
            pattern: AST-grep pattern (e.g., "print($ARG)")
            source: Source code to search
            language: Programming language

        Returns:
            List of MatchResult
        """
        lang = language or self.config.language
        root = SgRoot(source, lang)
        node = root.root()

        matches = node.find_all(pattern=pattern)

        results = []
        for match in matches:
            # Get captured variables
            captures = {}
            for name, captured in match.get_env().items():
                captures[name] = captured.text()

            results.append(MatchResult(
                file_path="<source>",
                line_start=match.range().start.line,
                line_end=match.range().end.line,
                column_start=match.range().start.column,
                column_end=match.range().end.column,
                matched_text=match.text(),
                captures=captures,
            ))

        return results

    def search_file(
        self,
        pattern: str,
        file_path: str,
        language: Optional[str] = None,
    ) -> List[MatchResult]:
        """
        Search for AST pattern in a file.

        Args:
            pattern: AST-grep pattern
            file_path: Path to file
            language: Programming language

        Returns:
            List of MatchResult
        """
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()

        results = self.search(pattern, source, language)

        # Update file paths
        for r in results:
            r.file_path = file_path

        return results

    def replace(
        self,
        pattern: str,
        replacement: str,
        source: str,
        language: Optional[str] = None,
    ) -> str:
        """
        Replace AST pattern in source code.

        Args:
            pattern: AST-grep pattern to find
            replacement: Replacement pattern (supports $VAR captures)
            source: Source code
            language: Programming language

        Returns:
            Modified source code
        """
        lang = language or self.config.language
        root = SgRoot(source, lang)
        node = root.root()

        # Apply replacement
        edits = []
        for match in node.find_all(pattern=pattern):
            # Build replacement with captures
            repl = replacement
            for name, captured in match.get_env().items():
                repl = repl.replace(f"${name}", captured.text())

            edits.append({
                "start": match.range().start.index,
                "end": match.range().end.index,
                "text": repl,
            })

        # Apply edits in reverse order to preserve indices
        result = source
        for edit in sorted(edits, key=lambda e: e["start"], reverse=True):
            result = result[:edit["start"]] + edit["text"] + result[edit["end"]:]

        return result

    def refactor_file(
        self,
        pattern: str,
        replacement: str,
        file_path: str,
        language: Optional[str] = None,
        dry_run: bool = False,
    ) -> RefactorResult:
        """
        Refactor a file using AST pattern replacement.

        Args:
            pattern: Pattern to find
            replacement: Replacement pattern
            file_path: File to refactor
            language: Programming language
            dry_run: If True, don't write changes

        Returns:
            RefactorResult
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                original = f.read()

            # Count matches before
            matches = self.search(pattern, original, language)
            match_count = len(matches)

            if match_count == 0:
                return RefactorResult(
                    success=True,
                    files_modified=0,
                    matches_replaced=0,
                )

            # Apply replacement
            modified = self.replace(pattern, replacement, original, language)

            # Generate diff
            import difflib
            diff = "\n".join(difflib.unified_diff(
                original.splitlines(),
                modified.splitlines(),
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                lineterm="",
            ))

            if not dry_run:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(modified)

            return RefactorResult(
                success=True,
                files_modified=1 if not dry_run else 0,
                matches_replaced=match_count,
                diff=diff,
            )

        except Exception as e:
            return RefactorResult(
                success=False,
                files_modified=0,
                matches_replaced=0,
                error=str(e),
            )


# ============================================================================
# Explicit Getter Functions - Raise SDKNotAvailableError if unavailable
# ============================================================================

def get_crawl4ai_crawler(
    headless: bool = True,
    **kwargs: Any,
) -> Crawl4AIClient:
    """
    Get a Crawl4AI crawler client.

    Args:
        headless: Run browser headless
        **kwargs: Additional configuration

    Returns:
        Crawl4AIClient instance

    Raises:
        SDKNotAvailableError: If crawl4ai is not installed
    """
    return Crawl4AIClient(config=CrawlConfig(headless=headless, **kwargs))


def get_firecrawl_client(
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> FirecrawlClient:
    """
    Get a Firecrawl client.

    Args:
        api_key: Firecrawl API key
        **kwargs: Additional configuration

    Returns:
        FirecrawlClient instance

    Raises:
        SDKNotAvailableError: If firecrawl is not installed
        SDKConfigurationError: If API key is missing
    """
    return FirecrawlClient(config=FirecrawlConfig(api_key=api_key, **kwargs))


def get_aider_coder(
    model: str = "claude-sonnet-4-20250514",
    provider: str = "anthropic",
    **kwargs: Any,
) -> AiderClient:
    """
    Get an Aider code editor client.

    Args:
        model: Model to use
        provider: LLM provider
        **kwargs: Additional configuration

    Returns:
        AiderClient instance

    Raises:
        SDKNotAvailableError: If aider is not installed
        SDKConfigurationError: If API key is missing
    """
    return AiderClient(config=AiderConfig(model=model, provider=provider, **kwargs))


def get_astgrep_client(
    language: str = "python",
    **kwargs: Any,
) -> ASTGrepClient:
    """
    Get an AST-Grep client.

    Args:
        language: Default programming language
        **kwargs: Additional configuration

    Returns:
        ASTGrepClient instance

    Raises:
        SDKNotAvailableError: If ast-grep is not installed
    """
    return ASTGrepClient(config=ASTGrepConfig(language=language, **kwargs))


# ============================================================================
# Unified Factory
# ============================================================================

class ProcessingFactory:
    """
    Unified factory for creating processing clients.

    Provides a single entry point for all four SDKs with
    consistent configuration and V33 integration.
    """

    def __init__(self):
        """Initialize the factory."""
        self._crawl4ai: Optional[Crawl4AIClient] = None
        self._firecrawl: Optional[FirecrawlClient] = None
        self._aider: Optional[AiderClient] = None
        self._astgrep: Optional[ASTGrepClient] = None

    def get_availability(self) -> Dict[str, bool]:
        """Get availability status of all SDKs."""
        return {
            "crawl4ai": CRAWL4AI_AVAILABLE,
            "firecrawl": FIRECRAWL_AVAILABLE,
            "aider": AIDER_AVAILABLE,
            "astgrep": ASTGREP_AVAILABLE,
        }

    def create_crawler(self, **kwargs: Any) -> Crawl4AIClient:
        """Create a Crawl4AI client."""
        self._crawl4ai = Crawl4AIClient(**kwargs)
        return self._crawl4ai

    def create_scraper(self, **kwargs: Any) -> FirecrawlClient:
        """Create a Firecrawl client."""
        self._firecrawl = FirecrawlClient(**kwargs)
        return self._firecrawl

    def create_coder(self, **kwargs: Any) -> AiderClient:
        """Create an Aider client."""
        self._aider = AiderClient(**kwargs)
        return self._aider

    def create_ast_search(self, **kwargs: Any) -> ASTGrepClient:
        """Create an AST-Grep client."""
        self._astgrep = ASTGrepClient(**kwargs)
        return self._astgrep


# ============================================================================
# Module-level availability
# ============================================================================

PROCESSING_AVAILABLE = (
    CRAWL4AI_AVAILABLE or
    FIRECRAWL_AVAILABLE or
    AIDER_AVAILABLE or
    ASTGREP_AVAILABLE
)


def get_available_sdks() -> Dict[str, bool]:
    """Get availability status of all processing SDKs."""
    return {
        "crawl4ai": CRAWL4AI_AVAILABLE,
        "firecrawl": FIRECRAWL_AVAILABLE,
        "aider": AIDER_AVAILABLE,
        "astgrep": ASTGREP_AVAILABLE,
    }


# ============================================================================
# All Exports
# ============================================================================

__all__ = [
    # Exceptions (re-exported from observability)
    "SDKNotAvailableError",
    "SDKConfigurationError",

    # Availability flags
    "CRAWL4AI_AVAILABLE",
    "FIRECRAWL_AVAILABLE",
    "AIDER_AVAILABLE",
    "ASTGREP_AVAILABLE",
    "PROCESSING_AVAILABLE",

    # Getter functions (raise on unavailable)
    "get_crawl4ai_crawler",
    "get_firecrawl_client",
    "get_aider_coder",
    "get_astgrep_client",

    # Crawl4AI
    "Crawl4AIClient",
    "CrawlConfig",
    "CrawlResult",
    "ExtractionMode",

    # Firecrawl
    "FirecrawlClient",
    "FirecrawlConfig",
    "ScrapeResult",

    # Aider
    "AiderClient",
    "AiderConfig",
    "EditRequest",
    "CodeEditResult",

    # AST-Grep
    "ASTGrepClient",
    "ASTGrepConfig",
    "MatchResult",
    "RefactorResult",

    # Factory
    "ProcessingFactory",
    "get_available_sdks",
]
