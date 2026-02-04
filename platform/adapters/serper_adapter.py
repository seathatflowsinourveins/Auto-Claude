"""
Serper API Adapter - Google SERP API for UNLEASH
=================================================

Complete implementation of Serper.dev API for Google search results.
Serper provides fast, reliable Google SERP data for AI applications.

Endpoints:
- /search - Web search (organic results, knowledge graph, people also ask)
- /images - Image search
- /news - News articles
- /videos - Video search
- /places - Local business listings
- /maps - Google Maps results
- /shopping - E-commerce products
- /scholar - Academic papers (Google Scholar)
- /patents - Patent database (Google Patents)
- /autocomplete - Search suggestions

Rate Limits:
- 300 queries/second max
- 2,500 free credits/month
- Credits expire after 6 months

Pricing:
- Free: 2,500 queries/month
- Starter: $50/mo for 50K queries ($1.00/1K)
- Standard: $150/mo for 250K queries ($0.60/1K)
- Scale: $500/mo for 1M queries ($0.50/1K)
- Enterprise: Custom ($0.30/1K)

Official Docs: https://serper.dev/docs
Dashboard: https://serper.dev/dashboard

Usage:
    adapter = SerperAdapter()
    await adapter.initialize({"api_key": "serper-xxx"})

    # Web search with knowledge graph
    result = await adapter.execute("search", query="OpenAI GPT-4")

    # Knowledge graph extraction (convenience method)
    result = await adapter.execute("knowledge_graph", query="Elon Musk")

    # Image search
    result = await adapter.execute("images", query="neural network architecture")

    # News search with time filter
    result = await adapter.execute("news", query="AI regulation", tbs="qdr:w")

    # Video search
    result = await adapter.execute("videos", query="transformer explained")

    # Local places search
    result = await adapter.execute("places", query="coffee shops", location="San Francisco")

    # Google Scholar academic search
    result = await adapter.execute("scholar", query="attention mechanism")

    # Autocomplete suggestions
    result = await adapter.execute("autocomplete", query="how to train")
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

# HTTP client
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    HTTPX_AVAILABLE = False

# Circuit breaker imports
try:
    from .circuit_breaker_manager import adapter_circuit_breaker, get_adapter_circuit_manager
    from ..core.resilience import CircuitOpenError
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
    from ..core.orchestration.base import SDKAdapter, AdapterResult, AdapterStatus, SDKLayer, register_adapter
except ImportError:
    try:
        from core.orchestration.base import SDKAdapter, AdapterResult, AdapterStatus, SDKLayer, register_adapter
    except ImportError:
        from enum import IntEnum
        from abc import ABC, abstractmethod

        class SDKLayer(IntEnum):
            RESEARCH = 8

        class AdapterStatus(str, Enum):
            UNINITIALIZED = "uninitialized"
            READY = "ready"
            DEGRADED = "degraded"
            ERROR = "error"

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


# Serper availability
SERPER_AVAILABLE = HTTPX_AVAILABLE


class SerperSearchType(str, Enum):
    """Serper search types/endpoints."""
    SEARCH = "search"
    IMAGES = "images"
    NEWS = "news"
    VIDEOS = "videos"
    PLACES = "places"
    MAPS = "maps"
    SHOPPING = "shopping"
    SCHOLAR = "scholar"
    PATENTS = "patents"
    AUTOCOMPLETE = "autocomplete"


class SerperTimeRange(str, Enum):
    """Time range filters using Google's tbs parameter."""
    HOUR = "qdr:h"    # Past hour
    DAY = "qdr:d"     # Past 24 hours
    WEEK = "qdr:w"    # Past week
    MONTH = "qdr:m"   # Past month
    YEAR = "qdr:y"    # Past year


class SerperImageType(str, Enum):
    """Image type filters."""
    PHOTO = "photo"
    CLIPART = "clipart"
    LINEART = "lineart"
    GIF = "gif"
    ANIMATED = "animated"


class SerperImageSize(str, Enum):
    """Image size filters."""
    LARGE = "large"
    MEDIUM = "medium"
    ICON = "icon"


class SerperSafeSearch(str, Enum):
    """Safe search settings."""
    OFF = "off"
    ACTIVE = "active"
    MODERATE = "moderate"


@dataclass
class SerperStatistics:
    """Statistics for Serper operations."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    credits_used: int = 0
    requests_by_type: Dict[str, int] = field(default_factory=dict)

    def record_request(self, search_type: str, success: bool, latency_ms: float):
        """Record a request and its outcome."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
            self.credits_used += 1
        else:
            self.failed_requests += 1
        self.total_latency_ms += latency_ms
        self.requests_by_type[search_type] = self.requests_by_type.get(search_type, 0) + 1

    @property
    def avg_latency_ms(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests


@register_adapter("serper", SDKLayer.RESEARCH, priority=23)
class SerperAdapter(SDKAdapter):
    """
    Serper API Adapter for Google SERP results.

    Provides access to Google search results across 11 operations:
    - search: Web search with organic results, knowledge graph, PAA
    - knowledge_graph: Extract knowledge panel data for entities
    - images: Image search with size/type filters
    - news: News articles with time filters
    - videos: Video search results
    - places: Local business listings
    - maps: Google Maps results
    - shopping: E-commerce product results
    - scholar: Google Scholar academic papers
    - patents: Google Patents database
    - autocomplete: Search suggestions

    Features:
    - Advanced search operators (site:, filetype:, before:, after:, etc.)
    - Knowledge graph extraction
    - People Also Ask processing
    - Related searches
    - Sitelinks extraction
    - Location-based results
    - Time range filtering
    """

    BASE_URL = "https://google.serper.dev"

    def __init__(self):
        self._api_key: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._status = AdapterStatus.UNINITIALIZED
        self._config: Dict[str, Any] = {}
        self._stats = SerperStatistics()
        self._mock_mode = False

    @property
    def sdk_name(self) -> str:
        return "serper"

    @property
    def layer(self) -> SDKLayer:
        return SDKLayer.RESEARCH

    @property
    def available(self) -> bool:
        return SERPER_AVAILABLE

    async def initialize(self, config: Dict[str, Any]) -> AdapterResult:
        """Initialize the Serper client."""
        start = time.time()

        if not HTTPX_AVAILABLE:
            self._status = AdapterStatus.ERROR
            return AdapterResult(
                success=False,
                error="httpx not installed. Run: pip install httpx",
                latency_ms=(time.time() - start) * 1000,
            )

        try:
            self._api_key = config.get("api_key") or os.environ.get("SERPER_API_KEY")
            self._config = config

            if not self._api_key:
                self._mock_mode = True
                self._status = AdapterStatus.DEGRADED
                return AdapterResult(
                    success=True,
                    data={
                        "status": "degraded",
                        "reason": "No API key - running in mock mode",
                        "features": list(SerperSearchType.__members__.keys()),
                    },
                    latency_ms=(time.time() - start) * 1000,
                )

            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={
                    "X-API-KEY": self._api_key,
                    "Content-Type": "application/json",
                },
                timeout=config.get("timeout", 30.0),
            )
            self._status = AdapterStatus.READY

            return AdapterResult(
                success=True,
                data={
                    "status": "ready",
                    "features": [t.value for t in SerperSearchType],
                    "rate_limit": "300 req/s",
                    "version": "1.0",
                },
                latency_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            self._status = AdapterStatus.ERROR
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start) * 1000,
            )

    async def execute(self, operation: str, **kwargs) -> AdapterResult:
        """Execute Serper operations with circuit breaker protection."""
        start = time.time()

        operations = {
            "search": self._search,
            "knowledge_graph": self.knowledge_graph,
            "images": self._image_search,
            "news": self._news_search,
            "videos": self._video_search,
            "places": self._places_search,
            "maps": self._maps_search,
            "shopping": self._shopping_search,
            "scholar": self._scholar_search,
            "patents": self._patents_search,
            "autocomplete": self._autocomplete,
        }

        if operation not in operations:
            return AdapterResult(
                success=False,
                error=f"Unknown operation: {operation}. Valid: {list(operations.keys())}",
                latency_ms=(time.time() - start) * 1000,
            )

        # Execute with circuit breaker protection
        try:
            async with adapter_circuit_breaker("serper_adapter"):
                result = await operations[operation](**kwargs)
                result.latency_ms = (time.time() - start) * 1000
                return result
        except CircuitOpenError as e:
            # Circuit is open - return fallback response
            latency = (time.time() - start) * 1000
            self._stats.record_request(operation, False, latency)
            return AdapterResult(
                success=False,
                error=f"Circuit breaker open for serper_adapter: {e}",
                latency_ms=latency,
                metadata={"circuit_breaker": "open", "adapter": "serper"},
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            self._stats.record_request(operation, False, latency)
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=latency,
            )

    async def _make_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
    ) -> AdapterResult:
        """Make a request to the Serper API."""
        start = time.time()

        if self._mock_mode or not self._client:
            latency = (time.time() - start) * 1000
            self._stats.record_request(endpoint.strip("/"), True, latency)
            return AdapterResult(
                success=True,
                data={
                    "mock": True,
                    "endpoint": endpoint,
                    "query": data.get("q", ""),
                    "organic": [
                        {
                            "title": f"Mock result for: {data.get('q', 'query')}",
                            "link": "https://example.com/mock",
                            "snippet": f"This is a mock result for query: {data.get('q', '')}",
                            "position": 1,
                        }
                    ],
                    "searchParameters": data,
                },
                latency_ms=latency,
                metadata={"mock_mode": True},
            )

        try:
            response = await self._client.post(endpoint, json=data)
            response.raise_for_status()
            result = response.json()
            latency = (time.time() - start) * 1000
            self._stats.record_request(endpoint.strip("/"), True, latency)

            return AdapterResult(
                success=True,
                data=result,
                latency_ms=latency,
                metadata={
                    "endpoint": endpoint,
                    "credits_used": 1,
                    "total_credits": self._stats.credits_used,
                },
            )
        except httpx.HTTPStatusError as e:
            latency = (time.time() - start) * 1000
            self._stats.record_request(endpoint.strip("/"), False, latency)
            return AdapterResult(
                success=False,
                data=None,
                latency_ms=latency,
                error=f"HTTP {e.response.status_code}: {e.response.text}",
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            self._stats.record_request(endpoint.strip("/"), False, latency)
            return AdapterResult(
                success=False,
                data=None,
                latency_ms=latency,
                error=str(e),
            )

    async def knowledge_graph(
        self,
        query: str,
        gl: str = "us",
        hl: str = "en",
        **kwargs,
    ) -> AdapterResult:
        """
        Get knowledge graph data for a query.

        This is a convenience method that extracts knowledge graph
        information from a search result.

        Args:
            query: Search query (entities work best: "OpenAI", "Elon Musk")
            gl: Country code (default "us")
            hl: Language code (default "en")

        Returns:
            AdapterResult with knowledge graph data:
            - title: Entity name
            - type: Entity type
            - description: Short description
            - descriptionSource: Source of description
            - descriptionLink: Link to source
            - imageUrl: Entity image
            - website: Official website
            - attributes: Dict of key-value attributes
        """
        result = await self._search(query=query, num=1, gl=gl, hl=hl, **kwargs)

        if result.success and result.data:
            kg = result.data.get("knowledgeGraph")
            if kg:
                return AdapterResult(
                    success=True,
                    data={
                        "query": query,
                        "knowledgeGraph": kg,
                        "title": kg.get("title"),
                        "type": kg.get("type"),
                        "description": kg.get("description"),
                        "attributes": kg.get("attributes", {}),
                    },
                    latency_ms=result.latency_ms,
                )
            else:
                return AdapterResult(
                    success=True,
                    data={
                        "query": query,
                        "knowledgeGraph": None,
                        "message": "No knowledge graph data available for this query",
                    },
                    latency_ms=result.latency_ms,
                )

        return result

    async def _search(
        self,
        query: str,
        num: int = 10,
        page: int = 1,
        location: Optional[str] = None,
        gl: str = "us",
        hl: str = "en",
        autocorrect: bool = True,
        tbs: Optional[str] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Perform web search.

        Args:
            query: Search query (supports operators: site:, filetype:, before:, after:,
                   intitle:, inurl:, related:, cache:, define:, etc.)
            num: Number of results (1-100, default 10)
            page: Page number (default 1)
            location: Location for local results (e.g., "New York, NY")
            gl: Country code (default "us")
            hl: Language code (default "en")
            autocorrect: Enable autocorrection (default True)
            tbs: Time filter (qdr:h=hour, qdr:d=day, qdr:w=week, qdr:m=month, qdr:y=year)

        Returns:
            AdapterResult with:
            - organic: List of organic search results
            - knowledgeGraph: Knowledge graph data (if available)
            - peopleAlsoAsk: Related questions
            - relatedSearches: Related search queries
            - topStories: News results (if applicable)
            - sitelinks: Site links for main result
        """
        data: Dict[str, Any] = {
            "q": query,
            "num": min(num, 100),
            "page": page,
            "gl": gl,
            "hl": hl,
            "autocorrect": autocorrect,
        }
        if location:
            data["location"] = location
        if tbs:
            data["tbs"] = tbs

        result = await self._make_request("/search", data)

        # Post-process to standardize output
        if result.success and result.data and not result.data.get("mock"):
            processed = {
                "query": query,
                "organic": result.data.get("organic", []),
                "knowledgeGraph": result.data.get("knowledgeGraph"),
                "peopleAlsoAsk": result.data.get("peopleAlsoAsk", []),
                "relatedSearches": result.data.get("relatedSearches", []),
                "topStories": result.data.get("topStories", []),
                "searchParameters": result.data.get("searchParameters", {}),
                "credits": result.data.get("credits", 1),
            }
            result.data = processed

        return result

    async def _image_search(
        self,
        query: str,
        num: int = 10,
        page: int = 1,
        gl: str = "us",
        hl: str = "en",
        safe: str = "off",
        image_type: Optional[str] = None,
        image_size: Optional[str] = None,
        autocorrect: bool = True,
        **kwargs,
    ) -> AdapterResult:
        """
        Perform image search.

        Args:
            query: Search query
            num: Number of results (1-100)
            page: Page number
            gl: Country code
            hl: Language code
            safe: Safe search (off, active, moderate)
            image_type: Type filter (photo, clipart, lineart, gif, animated)
            image_size: Size filter (large, medium, icon)
            autocorrect: Enable autocorrection

        Returns:
            AdapterResult with images array containing:
            - title: Image title
            - imageUrl: Direct image URL
            - imageWidth/imageHeight: Dimensions
            - thumbnailUrl: Thumbnail URL
            - source: Source website
            - link: Source page URL
        """
        data: Dict[str, Any] = {
            "q": query,
            "num": min(num, 100),
            "page": page,
            "gl": gl,
            "hl": hl,
            "safe": safe,
            "autocorrect": autocorrect,
        }
        if image_type:
            data["type"] = image_type
        if image_size:
            data["size"] = image_size

        result = await self._make_request("/images", data)

        if result.success and result.data and not result.data.get("mock"):
            processed = {
                "query": query,
                "images": result.data.get("images", []),
                "searchParameters": result.data.get("searchParameters", {}),
                "credits": result.data.get("credits", 1),
            }
            result.data = processed

        return result

    async def _news_search(
        self,
        query: str,
        num: int = 10,
        page: int = 1,
        gl: str = "us",
        hl: str = "en",
        tbs: Optional[str] = None,
        autocorrect: bool = True,
        **kwargs,
    ) -> AdapterResult:
        """
        Perform news search.

        Args:
            query: Search query
            num: Number of results
            page: Page number
            gl: Country code
            hl: Language code
            tbs: Time filter (qdr:h=hour, qdr:d=day, qdr:w=week, qdr:m=month, qdr:y=year)
            autocorrect: Enable autocorrection

        Returns:
            AdapterResult with news array containing:
            - title: Article title
            - link: Article URL
            - snippet: Article snippet
            - date: Publication date
            - source: News source
            - imageUrl: Article image (if available)
        """
        data: Dict[str, Any] = {
            "q": query,
            "num": min(num, 100),
            "page": page,
            "gl": gl,
            "hl": hl,
            "autocorrect": autocorrect,
        }
        if tbs:
            data["tbs"] = tbs

        result = await self._make_request("/news", data)

        if result.success and result.data and not result.data.get("mock"):
            processed = {
                "query": query,
                "news": result.data.get("news", []),
                "searchParameters": result.data.get("searchParameters", {}),
                "credits": result.data.get("credits", 1),
            }
            result.data = processed

        return result

    async def _video_search(
        self,
        query: str,
        num: int = 10,
        page: int = 1,
        gl: str = "us",
        hl: str = "en",
        autocorrect: bool = True,
        **kwargs,
    ) -> AdapterResult:
        """
        Perform video search.

        Args:
            query: Search query
            num: Number of results
            page: Page number
            gl: Country code
            hl: Language code
            autocorrect: Enable autocorrection

        Returns:
            AdapterResult with videos array containing:
            - title: Video title
            - link: Video URL
            - snippet: Video description
            - imageUrl: Thumbnail URL
            - duration: Video duration
            - source: Video platform (YouTube, etc.)
            - channel: Channel name
            - date: Upload date
        """
        data: Dict[str, Any] = {
            "q": query,
            "num": min(num, 100),
            "page": page,
            "gl": gl,
            "hl": hl,
            "autocorrect": autocorrect,
        }

        result = await self._make_request("/videos", data)

        if result.success and result.data and not result.data.get("mock"):
            processed = {
                "query": query,
                "videos": result.data.get("videos", []),
                "searchParameters": result.data.get("searchParameters", {}),
                "credits": result.data.get("credits", 1),
            }
            result.data = processed

        return result

    async def _places_search(
        self,
        query: str,
        location: str,
        num: int = 10,
        gl: str = "us",
        hl: str = "en",
        autocorrect: bool = True,
        **kwargs,
    ) -> AdapterResult:
        """
        Perform local places search.

        Args:
            query: Search query (e.g., "coffee shops", "restaurants")
            location: Required location for local results (e.g., "San Francisco, CA")
            num: Number of results
            gl: Country code
            hl: Language code
            autocorrect: Enable autocorrection

        Returns:
            AdapterResult with places array containing:
            - title: Business name
            - address: Full address
            - latitude/longitude: Coordinates
            - rating: Star rating
            - ratingCount: Number of reviews
            - category: Business category
            - phoneNumber: Contact phone
            - website: Business website
            - cid: Google place ID
        """
        data: Dict[str, Any] = {
            "q": query,
            "location": location,
            "num": min(num, 100),
            "gl": gl,
            "hl": hl,
            "autocorrect": autocorrect,
        }

        result = await self._make_request("/places", data)

        if result.success and result.data and not result.data.get("mock"):
            processed = {
                "query": query,
                "location": location,
                "places": result.data.get("places", []),
                "searchParameters": result.data.get("searchParameters", {}),
                "credits": result.data.get("credits", 1),
            }
            result.data = processed

        return result

    async def _maps_search(
        self,
        query: str,
        location: str,
        num: int = 10,
        gl: str = "us",
        hl: str = "en",
        autocorrect: bool = True,
        **kwargs,
    ) -> AdapterResult:
        """
        Perform Google Maps search.

        Args:
            query: Search query
            location: Required location
            num: Number of results
            gl: Country code
            hl: Language code
            autocorrect: Enable autocorrection

        Returns:
            AdapterResult with maps data including places and map metadata
        """
        data: Dict[str, Any] = {
            "q": query,
            "location": location,
            "num": min(num, 100),
            "gl": gl,
            "hl": hl,
            "autocorrect": autocorrect,
        }

        result = await self._make_request("/maps", data)

        if result.success and result.data and not result.data.get("mock"):
            processed = {
                "query": query,
                "location": location,
                "places": result.data.get("places", []),
                "searchParameters": result.data.get("searchParameters", {}),
                "credits": result.data.get("credits", 1),
            }
            result.data = processed

        return result

    async def _shopping_search(
        self,
        query: str,
        num: int = 10,
        page: int = 1,
        gl: str = "us",
        hl: str = "en",
        autocorrect: bool = True,
        **kwargs,
    ) -> AdapterResult:
        """
        Perform shopping/product search.

        Args:
            query: Product search query
            num: Number of results
            page: Page number
            gl: Country code
            hl: Language code
            autocorrect: Enable autocorrection

        Returns:
            AdapterResult with shopping array containing:
            - title: Product name
            - source: Retailer name
            - link: Product page URL
            - price: Product price
            - delivery: Delivery info
            - imageUrl: Product image
            - rating: Product rating
            - ratingCount: Number of reviews
            - offers: Number of offers
            - productId: Product identifier
        """
        data: Dict[str, Any] = {
            "q": query,
            "num": min(num, 100),
            "page": page,
            "gl": gl,
            "hl": hl,
            "autocorrect": autocorrect,
        }

        result = await self._make_request("/shopping", data)

        if result.success and result.data and not result.data.get("mock"):
            processed = {
                "query": query,
                "shopping": result.data.get("shopping", []),
                "searchParameters": result.data.get("searchParameters", {}),
                "credits": result.data.get("credits", 1),
            }
            result.data = processed

        return result

    async def _scholar_search(
        self,
        query: str,
        num: int = 10,
        page: int = 1,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        autocorrect: bool = True,
        **kwargs,
    ) -> AdapterResult:
        """
        Perform Google Scholar academic search.

        Args:
            query: Academic search query
            num: Number of results
            page: Page number
            year_from: Filter by start year (e.g., 2020)
            year_to: Filter by end year (e.g., 2024)
            autocorrect: Enable autocorrection

        Returns:
            AdapterResult with organic array containing:
            - title: Paper title
            - link: Paper URL
            - snippet: Abstract snippet
            - publicationInfo: Journal/conference info
            - citedBy: Citation count
            - year: Publication year
            - pdfUrl: Direct PDF link (if available)
        """
        data: Dict[str, Any] = {
            "q": query,
            "num": min(num, 20),  # Scholar typically has lower limits
            "page": page,
            "autocorrect": autocorrect,
        }
        if year_from:
            data["as_ylo"] = year_from
        if year_to:
            data["as_yhi"] = year_to

        result = await self._make_request("/scholar", data)

        if result.success and result.data and not result.data.get("mock"):
            processed = {
                "query": query,
                "organic": result.data.get("organic", []),
                "searchParameters": result.data.get("searchParameters", {}),
                "credits": result.data.get("credits", 1),
            }
            result.data = processed

        return result

    async def _patents_search(
        self,
        query: str,
        num: int = 10,
        page: int = 1,
        autocorrect: bool = True,
        **kwargs,
    ) -> AdapterResult:
        """
        Perform patent database search.

        Args:
            query: Patent search query
            num: Number of results
            page: Page number
            autocorrect: Enable autocorrection

        Returns:
            AdapterResult with organic array containing:
            - title: Patent title
            - link: Patent URL
            - snippet: Patent abstract
            - publicationNumber: Patent number
            - inventor: Inventor name(s)
            - assignee: Patent assignee/owner
            - filingDate: Filing date
            - priorityDate: Priority date
            - publicationDate: Publication date
        """
        data: Dict[str, Any] = {
            "q": query,
            "num": min(num, 100),
            "page": page,
            "autocorrect": autocorrect,
        }

        result = await self._make_request("/patents", data)

        if result.success and result.data and not result.data.get("mock"):
            processed = {
                "query": query,
                "organic": result.data.get("organic", []),
                "searchParameters": result.data.get("searchParameters", {}),
                "credits": result.data.get("credits", 1),
            }
            result.data = processed

        return result

    async def _autocomplete(
        self,
        query: str,
        gl: str = "us",
        hl: str = "en",
        **kwargs,
    ) -> AdapterResult:
        """
        Get search suggestions/autocomplete.

        Args:
            query: Partial search query
            gl: Country code
            hl: Language code

        Returns:
            AdapterResult with suggestions array containing autocomplete suggestions
        """
        data: Dict[str, Any] = {
            "q": query,
            "gl": gl,
            "hl": hl,
        }

        result = await self._make_request("/autocomplete", data)

        if result.success and result.data and not result.data.get("mock"):
            processed = {
                "query": query,
                "suggestions": result.data.get("suggestions", []),
                "searchParameters": result.data.get("searchParameters", {}),
                "credits": result.data.get("credits", 1),
            }
            result.data = processed

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            "total_requests": self._stats.total_requests,
            "successful_requests": self._stats.successful_requests,
            "failed_requests": self._stats.failed_requests,
            "avg_latency_ms": self._stats.avg_latency_ms,
            "success_rate": self._stats.success_rate,
            "credits_used": self._stats.credits_used,
            "requests_by_type": self._stats.requests_by_type,
        }

    async def health_check(self) -> AdapterResult:
        """Check Serper API health."""
        if not SERPER_AVAILABLE:
            return AdapterResult(
                success=False,
                error="httpx not installed"
            )

        if self._mock_mode or not self._client:
            return AdapterResult(
                success=True,
                data={
                    "status": "degraded",
                    "reason": "No API key - mock mode",
                    "stats": self.get_statistics(),
                }
            )

        try:
            result = await self._search("test", num=1)
            return AdapterResult(
                success=True,
                data={
                    "status": "healthy",
                    "stats": self.get_statistics(),
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def shutdown(self) -> AdapterResult:
        """Shutdown the client and cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._status = AdapterStatus.UNINITIALIZED
        return AdapterResult(
            success=True,
            data={"stats": self.get_statistics()}
        )


def get_serper_adapter() -> type[SerperAdapter]:
    """Get the Serper adapter class."""
    return SerperAdapter


if __name__ == "__main__":
    async def test():
        adapter = SerperAdapter()
        await adapter.initialize({})

        # Test web search
        result = await adapter.execute("search", query="LangChain agents patterns")
        print(f"Search result: {result}")

        # Test knowledge graph extraction
        result = await adapter.execute("knowledge_graph", query="OpenAI")
        print(f"Knowledge graph result: {result}")

        # Test news search
        result = await adapter.execute("news", query="AI regulation 2026", tbs="qdr:w")
        print(f"News result: {result}")

        # Test image search
        result = await adapter.execute("images", query="neural network diagram")
        print(f"Image result: {result}")

        # Test scholar search
        result = await adapter.execute("scholar", query="transformer attention mechanism")
        print(f"Scholar result: {result}")

        # Test autocomplete
        result = await adapter.execute("autocomplete", query="how to train")
        print(f"Autocomplete result: {result}")

        print(f"\nStatistics: {adapter.get_statistics()}")
        await adapter.shutdown()

    asyncio.run(test())
