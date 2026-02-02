"""
SDK Integrations - Unified wrapper for all downloaded research/RAG SDKs.

Integrates:
- Microsoft GraphRAG: Knowledge graph construction from unstructured text
- LightRAG: Lightweight GraphRAG alternative with faster indexing
- LlamaIndex: RAG indexing, retrieval, and query engines
- DSPy: Declarative RAG optimization and prompt engineering
- Crawl4AI: LLM-optimized web crawling with markdown output
- Tavily: AI-powered web search, extraction, and research API
- MCP Python SDK: Model Context Protocol client/server library

This module provides:
1. Unified SDK loading with path management
2. Consistent wrapper interfaces for each SDK
3. Integration with EcosystemOrchestrator
4. Async/sync method parity across SDKs
"""

import os
import sys
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# =============================================================================
# Path Configuration
# =============================================================================

# Find SDK root relative to this file
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PLATFORM_DIR = os.path.dirname(_MODULE_DIR)
_UNLEASH_DIR = os.path.dirname(_PLATFORM_DIR)
SDKS_ROOT = os.path.join(_UNLEASH_DIR, "sdks")

# SDK paths
SDK_PATHS = {
    "graphrag": os.path.join(SDKS_ROOT, "graphrag"),
    "lightrag": os.path.join(SDKS_ROOT, "lightrag"),
    "llama_index": os.path.join(SDKS_ROOT, "llama-index", "llama-index-core"),
    "dspy": os.path.join(SDKS_ROOT, "dspy"),
    "crawl4ai": os.path.join(SDKS_ROOT, "crawl4ai"),
    "tavily": os.path.join(SDKS_ROOT, "tavily"),
    "langgraph": os.path.join(SDKS_ROOT, "langgraph", "libs", "langgraph"),
    "mcp": os.path.join(SDKS_ROOT, "mcp", "python-sdk", "src"),
    # New SDKs
    "graphiti": os.path.join(SDKS_ROOT, "graphiti"),
    "letta": os.path.join(SDKS_ROOT, "letta"),
    "firecrawl": os.path.join(SDKS_ROOT, "firecrawl"),
}


def _add_sdk_path(sdk_name: str) -> bool:
    """Add SDK to sys.path if not already present."""
    path = SDK_PATHS.get(sdk_name)
    if path and path not in sys.path and os.path.exists(path):
        sys.path.insert(0, path)
        logger.debug(f"Added {sdk_name} path: {path}")
        return True
    return False


# =============================================================================
# SDK Availability Flags
# =============================================================================

# GraphRAG - Partial support (indexer adapters and config work, full API needs graspologic)
GRAPHRAG_AVAILABLE = False
GraphRagConfig = None
graphrag_query = None
graphrag_read_indexer_entities = None
graphrag_create_config = None

try:
    _add_sdk_path("graphrag")
    # These imports work without graspologic dependency
    from graphrag.query.indexer_adapters import read_indexer_entities as graphrag_read_indexer_entities
    from graphrag.config import create_graphrag_config as graphrag_create_config
    GRAPHRAG_AVAILABLE = True  # Partial support available
    logger.info("GraphRAG SDK loaded (partial support - indexer adapters & config)")
except ImportError as e:
    logger.debug(f"GraphRAG not available: {e}")

# LightRAG
LIGHTRAG_AVAILABLE = False
LightRAG = None
QueryParam = None

try:
    _add_sdk_path("lightrag")
    from lightrag import LightRAG, QueryParam
    LIGHTRAG_AVAILABLE = True
    logger.info("LightRAG SDK loaded successfully")
except ImportError as e:
    logger.debug(f"LightRAG not available: {e}")

# LlamaIndex - Use PyPI installed version (not local SDK path)
LLAMAINDEX_AVAILABLE = False
VectorStoreIndex = None
SimpleDirectoryReader = None
Document = None
Settings = None
StorageContext = None

try:
    # Import from PyPI-installed llama-index (version 0.11.x doesn't have workflows issue)
    from llama_index.core import (
        VectorStoreIndex,
        SimpleDirectoryReader,
        Document,
        Settings,
        StorageContext,
    )
    LLAMAINDEX_AVAILABLE = True
    logger.info("LlamaIndex SDK loaded successfully (PyPI version)")
except ImportError as e:
    logger.debug(f"LlamaIndex not available: {e}")

# DSPy - Partial support (LM, Module, ChatAdapter work without gepa dependency)
DSPY_AVAILABLE = False
DSPY_PARTIAL = False  # Partial support flag
dspy = None
dspy_LM = None
dspy_Module = None
dspy_ChatAdapter = None

try:
    _add_sdk_path("dspy")
    import dspy
    DSPY_AVAILABLE = True
    logger.info("DSPy SDK loaded successfully (full)")
except ImportError as e:
    logger.debug(f"DSPy full import failed: {e}")
    # Try partial imports (these work without gepa)
    try:
        _add_sdk_path("dspy")
        from dspy.clients import LM as dspy_LM
        from dspy.primitives.module import Module as dspy_Module
        from dspy.adapters.chat_adapter import ChatAdapter as dspy_ChatAdapter
        DSPY_PARTIAL = True
        logger.info("DSPy SDK loaded (partial support - LM, Module, ChatAdapter)")
    except ImportError as e2:
        logger.debug(f"DSPy partial import also failed: {e2}")

# Crawl4AI
CRAWL4AI_AVAILABLE = False
AsyncWebCrawler = None
CrawlerRunConfig = None
BrowserConfig = None
CrawlResult = None

try:
    _add_sdk_path("crawl4ai")
    from crawl4ai import (
        AsyncWebCrawler,
        CrawlerRunConfig,
        BrowserConfig,
        CrawlResult,
        CacheMode,
    )
    CRAWL4AI_AVAILABLE = True
    logger.info("Crawl4AI SDK loaded successfully")
except ImportError as e:
    logger.debug(f"Crawl4AI not available: {e}")

# Tavily - AI-powered web search and research
TAVILY_AVAILABLE = False
TavilyClient = None
AsyncTavilyClient = None
TavilyHybridClient = None

try:
    _add_sdk_path("tavily")
    from tavily import TavilyClient, AsyncTavilyClient, TavilyHybridClient
    TAVILY_AVAILABLE = True
    logger.info("Tavily SDK loaded successfully")
except ImportError as e:
    logger.debug(f"Tavily not available: {e}")

# LangGraph - Stateful agent workflows (partial support - needs langchain_core)
LANGGRAPH_AVAILABLE = False
StateGraph = None
MessagesState = None
langgraph_START = None
langgraph_END = None

try:
    _add_sdk_path("langgraph")
    from langgraph.graph import StateGraph, MessagesState, START as langgraph_START, END as langgraph_END
    LANGGRAPH_AVAILABLE = True
    logger.info("LangGraph SDK loaded successfully")
except ImportError as e:
    logger.debug(f"LangGraph not available: {e}")

# MCP Python SDK - Model Context Protocol client/server library
MCP_AVAILABLE = False
MCPClient = None
MCPClientSession = None
MCPServer = None
FastMCP = None
mcp_stdio_client = None
mcp_stdio_server = None

try:
    _add_sdk_path("mcp")
    from mcp import Client as MCPClient, ClientSession as MCPClientSession
    from mcp import stdio_client as mcp_stdio_client, stdio_server as mcp_stdio_server
    from mcp.server import Server as MCPServer
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
    logger.info("MCP Python SDK loaded successfully")
except ImportError as e:
    logger.debug(f"MCP Python SDK not available: {e}")

# Graphiti - Temporal Knowledge Graph (Zep)
GRAPHITI_AVAILABLE = False
GraphitiClient = None
GraphitiEpisodeType = None

try:
    _add_sdk_path("graphiti")
    from graphiti_core import Graphiti as GraphitiClient
    from graphiti_core.graphiti_types import EpisodeType as GraphitiEpisodeType
    GRAPHITI_AVAILABLE = True
    logger.info("Graphiti SDK loaded successfully")
except ImportError as e:
    logger.debug(f"Graphiti not available: {e}")

# Letta - Memory/Agent Framework
LETTA_AVAILABLE = False
LettaClient = None

try:
    _add_sdk_path("letta")
    # Letta client is typically used via REST API
    # Check if letta package is available
    import letta
    LettaClient = letta
    LETTA_AVAILABLE = True
    logger.info("Letta SDK loaded successfully")
except ImportError as e:
    logger.debug(f"Letta not available: {e}")

# Firecrawl - Web Scraping
FIRECRAWL_AVAILABLE = False
FirecrawlApp = None
AsyncFirecrawlApp = None

try:
    _add_sdk_path("firecrawl")
    from firecrawl import Firecrawl as FirecrawlApp, AsyncFirecrawl as AsyncFirecrawlApp
    FIRECRAWL_AVAILABLE = True
    logger.info("Firecrawl SDK loaded successfully")
except ImportError as e:
    logger.debug(f"Firecrawl not available: {e}")


# =============================================================================
# Data Models
# =============================================================================

class SDKType(str, Enum):
    """Available SDK types."""
    GRAPHRAG = "graphrag"
    LIGHTRAG = "lightrag"
    LLAMAINDEX = "llamaindex"
    DSPY = "dspy"
    CRAWL4AI = "crawl4ai"
    TAVILY = "tavily"
    LANGGRAPH = "langgraph"
    MCP = "mcp"
    # New SDKs
    GRAPHITI = "graphiti"  # Temporal knowledge graph (Zep)
    LETTA = "letta"  # Memory/Agent framework
    FIRECRAWL = "firecrawl"  # Web scraping


@dataclass
class SDKResult:
    """Unified result from any SDK operation."""
    sdk: SDKType
    operation: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sdk": self.sdk.value,
            "operation": self.operation,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# SDK Wrapper Base Class
# =============================================================================

class SDKWrapper(ABC):
    """Base class for SDK wrappers providing consistent interface."""

    def __init__(self):
        self._initialized = False

    @property
    @abstractmethod
    def sdk_type(self) -> SDKType:
        """Return the SDK type."""
        pass

    @property
    @abstractmethod
    def available(self) -> bool:
        """Check if SDK is available."""
        pass

    def _ensure_available(self) -> None:
        """Raise error if SDK not available."""
        if not self.available:
            raise ImportError(f"{self.sdk_type.value} SDK is not available")

    def _result(
        self,
        operation: str,
        success: bool,
        data: Any = None,
        error: Optional[str] = None,
        **metadata
    ) -> SDKResult:
        """Create a standardized result."""
        return SDKResult(
            sdk=self.sdk_type,
            operation=operation,
            success=success,
            data=data,
            error=error,
            metadata=metadata,
        )


# =============================================================================
# Crawl4AI Wrapper - LLM-Optimized Web Crawling
# =============================================================================

class Crawl4AIWrapper(SDKWrapper):
    """
    Wrapper for Crawl4AI - LLM-optimized web crawling.

    Features:
    - Async crawling with browser automation
    - LLM-friendly markdown output
    - Deep crawling with BFS/DFS strategies
    - Content extraction with CSS/XPath/Regex
    - Adaptive crawling with rate limiting
    """

    def __init__(self, browser_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.browser_config = browser_config or {}
        self._crawler: Optional[AsyncWebCrawler] = None

    @property
    def sdk_type(self) -> SDKType:
        return SDKType.CRAWL4AI

    @property
    def available(self) -> bool:
        return CRAWL4AI_AVAILABLE

    async def _get_crawler(self) -> "AsyncWebCrawler":
        """Get or create crawler instance."""
        self._ensure_available()
        if self._crawler is None:
            config = BrowserConfig(**self.browser_config) if self.browser_config else BrowserConfig()
            self._crawler = AsyncWebCrawler(config=config)
        return self._crawler

    async def crawl(
        self,
        url: str,
        *,
        extract_markdown: bool = True,
        extract_links: bool = True,
        cache_mode: str = "enabled",
        **kwargs
    ) -> SDKResult:
        """
        Crawl a single URL and extract LLM-friendly content.

        Args:
            url: URL to crawl
            extract_markdown: Convert HTML to markdown
            extract_links: Extract all links from page
            cache_mode: Cache behavior ('enabled', 'disabled', 'read_only')
            **kwargs: Additional CrawlerRunConfig options

        Returns:
            SDKResult with crawled content
        """
        try:
            crawler = await self._get_crawler()

            # Build run config
            run_config = CrawlerRunConfig(
                cache_mode=getattr(CacheMode, cache_mode.upper(), CacheMode.ENABLED),
                **kwargs
            )

            async with crawler:
                result = await crawler.arun(url=url, config=run_config)

            return self._result(
                operation="crawl",
                success=result.success,
                data={
                    "url": url,
                    "markdown": result.markdown if extract_markdown else None,
                    "html": result.html[:10000] if result.html else None,  # Truncate HTML
                    "links": result.links if extract_links else None,
                    "metadata": {
                        "title": getattr(result, "title", None),
                        "status_code": getattr(result, "status_code", None),
                    }
                },
                url=url,
            )
        except Exception as e:
            return self._result(
                operation="crawl",
                success=False,
                error=str(e),
                url=url,
            )

    async def batch_crawl(
        self,
        urls: List[str],
        *,
        max_concurrent: int = 5,
        **kwargs
    ) -> SDKResult:
        """
        Crawl multiple URLs concurrently.

        Args:
            urls: List of URLs to crawl
            max_concurrent: Maximum concurrent crawls
            **kwargs: Options passed to crawl()

        Returns:
            SDKResult with list of crawl results
        """
        try:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def crawl_with_semaphore(url: str):
                async with semaphore:
                    return await self.crawl(url, **kwargs)

            results = await asyncio.gather(
                *[crawl_with_semaphore(url) for url in urls],
                return_exceptions=True
            )

            successful = [r for r in results if isinstance(r, SDKResult) and r.success]
            failed = [r for r in results if not isinstance(r, SDKResult) or not r.success]

            return self._result(
                operation="batch_crawl",
                success=len(successful) > 0,
                data={
                    "results": [r.to_dict() for r in results if isinstance(r, SDKResult)],
                    "total": len(urls),
                    "successful": len(successful),
                    "failed": len(failed),
                },
                urls_count=len(urls),
            )
        except Exception as e:
            return self._result(
                operation="batch_crawl",
                success=False,
                error=str(e),
                urls_count=len(urls),
            )

    async def close(self):
        """Close the crawler."""
        if self._crawler:
            await self._crawler.close()
            self._crawler = None


# =============================================================================
# LightRAG Wrapper - Lightweight GraphRAG
# =============================================================================

class LightRAGWrapper(SDKWrapper):
    """
    Wrapper for LightRAG - lightweight GraphRAG alternative.

    Features:
    - Fast knowledge graph construction
    - Multiple query modes (naive, local, global, hybrid)
    - Incremental indexing
    - Lower resource requirements than full GraphRAG
    """

    def __init__(
        self,
        working_dir: str = "./lightrag_data",
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
    ):
        super().__init__()
        self.working_dir = working_dir
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self._rag: Optional["LightRAG"] = None

    @property
    def sdk_type(self) -> SDKType:
        return SDKType.LIGHTRAG

    @property
    def available(self) -> bool:
        return LIGHTRAG_AVAILABLE

    def _get_rag(self) -> "LightRAG":
        """Get or create LightRAG instance."""
        self._ensure_available()
        if self._rag is None:
            os.makedirs(self.working_dir, exist_ok=True)
            self._rag = LightRAG(
                working_dir=self.working_dir,
                llm_model_name=self.llm_model,
                embedding_model_name=self.embedding_model,
            )
        return self._rag

    async def insert(
        self,
        text: str,
        *,
        source: Optional[str] = None,
    ) -> SDKResult:
        """
        Insert text into the knowledge graph.

        Args:
            text: Text content to index
            source: Optional source identifier

        Returns:
            SDKResult indicating success
        """
        try:
            rag = self._get_rag()
            await rag.ainsert(text)

            return self._result(
                operation="insert",
                success=True,
                data={"text_length": len(text)},
                source=source,
            )
        except Exception as e:
            return self._result(
                operation="insert",
                success=False,
                error=str(e),
            )

    async def query(
        self,
        query: str,
        *,
        mode: str = "hybrid",
    ) -> SDKResult:
        """
        Query the knowledge graph.

        Args:
            query: Natural language query
            mode: Query mode ('naive', 'local', 'global', 'hybrid')

        Returns:
            SDKResult with query response
        """
        try:
            rag = self._get_rag()

            # Map mode string to QueryParam if available
            if QueryParam:
                param = QueryParam(mode=mode)
                response = await rag.aquery(query, param=param)
            else:
                response = await rag.aquery(query)

            return self._result(
                operation="query",
                success=True,
                data={"response": response, "mode": mode},
                query=query,
            )
        except Exception as e:
            return self._result(
                operation="query",
                success=False,
                error=str(e),
                query=query,
            )

    def insert_sync(self, text: str, **kwargs) -> SDKResult:
        """Synchronous insert wrapper."""
        return asyncio.run(self.insert(text, **kwargs))

    def query_sync(self, query: str, **kwargs) -> SDKResult:
        """Synchronous query wrapper."""
        return asyncio.run(self.query(query, **kwargs))


# =============================================================================
# DSPy Wrapper - Declarative RAG Optimization
# =============================================================================

class DSPyWrapper(SDKWrapper):
    """
    Wrapper for DSPy - declarative RAG optimization.

    Features:
    - Declarative prompt programming
    - Automatic prompt optimization
    - Chain-of-thought reasoning
    - Multi-hop retrieval
    - Evaluation pipelines
    """

    def __init__(self, model: str = "openai/gpt-4o-mini"):
        super().__init__()
        self.model = model
        self._configured = False

    @property
    def sdk_type(self) -> SDKType:
        return SDKType.DSPY

    @property
    def available(self) -> bool:
        return DSPY_AVAILABLE

    def configure(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> SDKResult:
        """
        Configure DSPy with language model.

        Args:
            model: Model identifier (e.g., 'openai/gpt-4o-mini')
            api_key: Optional API key override
            **kwargs: Additional configuration options

        Returns:
            SDKResult indicating success
        """
        try:
            self._ensure_available()

            model = model or self.model

            # Configure DSPy settings
            lm = dspy.LM(model, api_key=api_key) if api_key else dspy.LM(model)
            dspy.configure(lm=lm, **kwargs)

            self._configured = True

            return self._result(
                operation="configure",
                success=True,
                data={"model": model},
            )
        except Exception as e:
            return self._result(
                operation="configure",
                success=False,
                error=str(e),
            )

    def create_signature(
        self,
        input_fields: List[str],
        output_fields: List[str],
        instructions: Optional[str] = None,
    ) -> SDKResult:
        """
        Create a DSPy signature for structured prompting.

        Args:
            input_fields: List of input field names
            output_fields: List of output field names
            instructions: Optional task instructions

        Returns:
            SDKResult with signature class
        """
        try:
            self._ensure_available()

            # Build signature dynamically
            fields = {}
            for field in input_fields:
                fields[field] = dspy.InputField()
            for field in output_fields:
                fields[field] = dspy.OutputField()

            # Create signature class
            sig_class = type(
                "DynamicSignature",
                (dspy.Signature,),
                {"__doc__": instructions or "", **fields}
            )

            return self._result(
                operation="create_signature",
                success=True,
                data={"signature": sig_class},
                input_fields=input_fields,
                output_fields=output_fields,
            )
        except Exception as e:
            return self._result(
                operation="create_signature",
                success=False,
                error=str(e),
            )

    def predict(
        self,
        signature_or_prompt: Union[str, Any],
        **inputs
    ) -> SDKResult:
        """
        Run prediction with DSPy.

        Args:
            signature_or_prompt: Signature class or prompt string
            **inputs: Input field values

        Returns:
            SDKResult with prediction output
        """
        try:
            self._ensure_available()
            if not self._configured:
                self.configure()

            if isinstance(signature_or_prompt, str):
                # Simple completion
                predictor = dspy.Predict(signature_or_prompt)
            else:
                # Signature-based
                predictor = dspy.Predict(signature_or_prompt)

            result = predictor(**inputs)

            return self._result(
                operation="predict",
                success=True,
                data={
                    "output": dict(result) if hasattr(result, "__dict__") else str(result),
                },
            )
        except Exception as e:
            return self._result(
                operation="predict",
                success=False,
                error=str(e),
            )

    def chain_of_thought(
        self,
        signature: Any,
        **inputs
    ) -> SDKResult:
        """
        Run chain-of-thought reasoning.

        Args:
            signature: DSPy signature class
            **inputs: Input field values

        Returns:
            SDKResult with reasoning chain and output
        """
        try:
            self._ensure_available()
            if not self._configured:
                self.configure()

            cot = dspy.ChainOfThought(signature)
            result = cot(**inputs)

            return self._result(
                operation="chain_of_thought",
                success=True,
                data={
                    "output": dict(result) if hasattr(result, "__dict__") else str(result),
                    "reasoning": getattr(result, "reasoning", None),
                },
            )
        except Exception as e:
            return self._result(
                operation="chain_of_thought",
                success=False,
                error=str(e),
            )


# =============================================================================
# LlamaIndex Wrapper - RAG Indexing & Retrieval
# =============================================================================

class LlamaIndexWrapper(SDKWrapper):
    """
    Wrapper for LlamaIndex - comprehensive RAG framework.

    Features:
    - Document loading from multiple sources
    - Vector store indexing
    - Query engines with response synthesis
    - Knowledge graph indices
    - Property graph support
    """

    def __init__(
        self,
        persist_dir: str = "./llamaindex_storage",
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        super().__init__()
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self._index: Optional["VectorStoreIndex"] = None

    @property
    def sdk_type(self) -> SDKType:
        return SDKType.LLAMAINDEX

    @property
    def available(self) -> bool:
        return LLAMAINDEX_AVAILABLE

    def load_documents(
        self,
        input_dir: Optional[str] = None,
        input_files: Optional[List[str]] = None,
        texts: Optional[List[str]] = None,
    ) -> SDKResult:
        """
        Load documents from directory, files, or raw text.

        Args:
            input_dir: Directory to load documents from
            input_files: List of specific file paths
            texts: List of raw text strings

        Returns:
            SDKResult with loaded documents
        """
        try:
            self._ensure_available()

            documents = []

            if input_dir:
                reader = SimpleDirectoryReader(input_dir=input_dir)
                documents.extend(reader.load_data())

            if input_files:
                reader = SimpleDirectoryReader(input_files=input_files)
                documents.extend(reader.load_data())

            if texts:
                for i, text in enumerate(texts):
                    doc = Document(text=text, doc_id=f"text_{i}")
                    documents.append(doc)

            return self._result(
                operation="load_documents",
                success=True,
                data={
                    "documents": documents,
                    "count": len(documents),
                },
            )
        except Exception as e:
            return self._result(
                operation="load_documents",
                success=False,
                error=str(e),
            )

    def create_index(
        self,
        documents: Optional[List["Document"]] = None,
        texts: Optional[List[str]] = None,
    ) -> SDKResult:
        """
        Create a vector store index from documents.

        Args:
            documents: List of Document objects
            texts: List of raw text strings (alternative to documents)

        Returns:
            SDKResult with index reference
        """
        try:
            self._ensure_available()

            if texts and not documents:
                # Convert texts to documents
                documents = [
                    Document(text=text, doc_id=f"doc_{i}")
                    for i, text in enumerate(texts)
                ]

            if not documents:
                return self._result(
                    operation="create_index",
                    success=False,
                    error="No documents provided",
                )

            self._index = VectorStoreIndex.from_documents(documents)

            # Persist to storage
            os.makedirs(self.persist_dir, exist_ok=True)
            self._index.storage_context.persist(persist_dir=self.persist_dir)

            return self._result(
                operation="create_index",
                success=True,
                data={
                    "document_count": len(documents),
                    "persist_dir": self.persist_dir,
                },
            )
        except Exception as e:
            return self._result(
                operation="create_index",
                success=False,
                error=str(e),
            )

    def load_index(self, persist_dir: Optional[str] = None) -> SDKResult:
        """
        Load an existing index from storage.

        Args:
            persist_dir: Directory containing persisted index

        Returns:
            SDKResult indicating success
        """
        try:
            self._ensure_available()

            persist_dir = persist_dir or self.persist_dir
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            self._index = VectorStoreIndex.from_storage_context(storage_context)

            return self._result(
                operation="load_index",
                success=True,
                data={"persist_dir": persist_dir},
            )
        except Exception as e:
            return self._result(
                operation="load_index",
                success=False,
                error=str(e),
            )

    def query(
        self,
        query: str,
        *,
        similarity_top_k: int = 5,
        response_mode: str = "compact",
    ) -> SDKResult:
        """
        Query the index.

        Args:
            query: Natural language query
            similarity_top_k: Number of similar documents to retrieve
            response_mode: Response synthesis mode

        Returns:
            SDKResult with query response
        """
        try:
            self._ensure_available()

            if self._index is None:
                return self._result(
                    operation="query",
                    success=False,
                    error="No index loaded. Call create_index() or load_index() first.",
                )

            query_engine = self._index.as_query_engine(
                similarity_top_k=similarity_top_k,
                response_mode=response_mode,
            )

            response = query_engine.query(query)

            return self._result(
                operation="query",
                success=True,
                data={
                    "response": str(response),
                    "source_nodes": [
                        {
                            "text": node.text[:500],  # Truncate
                            "score": node.score,
                        }
                        for node in response.source_nodes
                    ] if hasattr(response, "source_nodes") else [],
                },
                query=query,
            )
        except Exception as e:
            return self._result(
                operation="query",
                success=False,
                error=str(e),
                query=query,
            )


# =============================================================================
# GraphRAG Wrapper - Microsoft Knowledge Graph
# =============================================================================

class GraphRAGWrapper(SDKWrapper):
    """
    Wrapper for Microsoft GraphRAG - knowledge graph from unstructured text.

    Features:
    - Entity and relationship extraction
    - Community detection
    - Global and local search
    - DRIFT search for complex queries
    - Multi-index support

    Note: GraphRAG requires pre-built indices. This wrapper assumes
    indices have been created via the graphrag CLI.
    """

    def __init__(self, index_root: str = "./graphrag_index"):
        super().__init__()
        self.index_root = index_root

    @property
    def sdk_type(self) -> SDKType:
        return SDKType.GRAPHRAG

    @property
    def available(self) -> bool:
        return GRAPHRAG_AVAILABLE

    async def global_search(
        self,
        query: str,
        *,
        community_level: int = 2,
        response_type: str = "multiple paragraphs",
    ) -> SDKResult:
        """
        Perform global search across the knowledge graph.

        Args:
            query: Natural language query
            community_level: Community hierarchy level to search
            response_type: Expected response format

        Returns:
            SDKResult with search response
        """
        try:
            self._ensure_available()

            # Note: GraphRAG requires pre-loaded dataframes
            # This is a placeholder showing the API structure
            return self._result(
                operation="global_search",
                success=False,
                error="GraphRAG requires pre-built index. Use graphrag CLI to build index first.",
                query=query,
            )
        except Exception as e:
            return self._result(
                operation="global_search",
                success=False,
                error=str(e),
                query=query,
            )

    async def local_search(
        self,
        query: str,
        *,
        community_level: int = 2,
        response_type: str = "multiple paragraphs",
    ) -> SDKResult:
        """
        Perform local search for specific entities.

        Args:
            query: Natural language query
            community_level: Community hierarchy level
            response_type: Expected response format

        Returns:
            SDKResult with search response
        """
        try:
            self._ensure_available()

            return self._result(
                operation="local_search",
                success=False,
                error="GraphRAG requires pre-built index. Use graphrag CLI to build index first.",
                query=query,
            )
        except Exception as e:
            return self._result(
                operation="local_search",
                success=False,
                error=str(e),
                query=query,
            )


# =============================================================================
# Tavily Wrapper - AI-Powered Web Search & Research
# =============================================================================

class TavilyWrapper(SDKWrapper):
    """
    Wrapper for Tavily - AI-powered web search and research.

    Features:
    - Semantic web search with depth options
    - Content extraction from URLs
    - Site crawling and mapping
    - AI research agent for deep investigation
    - Hybrid RAG support
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key
        self._client: Optional["TavilyClient"] = None
        self._async_client: Optional["AsyncTavilyClient"] = None

    @property
    def sdk_type(self) -> SDKType:
        return SDKType.TAVILY

    @property
    def available(self) -> bool:
        return TAVILY_AVAILABLE

    def _get_client(self) -> "TavilyClient":
        """Get or create sync Tavily client."""
        self._ensure_available()
        if self._client is None:
            self._client = TavilyClient(api_key=self.api_key)
        return self._client

    def _get_async_client(self) -> "AsyncTavilyClient":
        """Get or create async Tavily client."""
        self._ensure_available()
        if self._async_client is None:
            self._async_client = AsyncTavilyClient(api_key=self.api_key)
        return self._async_client

    async def search(
        self,
        query: str,
        *,
        search_depth: str = "basic",
        topic: str = "general",
        max_results: int = 10,
        include_answer: bool = True,
        include_raw_content: bool = False,
        include_images: bool = False,
        **kwargs
    ) -> SDKResult:
        """
        Perform AI-powered web search.

        Args:
            query: Search query
            search_depth: 'basic', 'advanced', 'fast', or 'ultra-fast'
            topic: 'general', 'news', or 'finance'
            max_results: Maximum number of results
            include_answer: Include AI-generated answer
            include_raw_content: Include raw page content
            include_images: Include images
            **kwargs: Additional Tavily API options

        Returns:
            SDKResult with search results
        """
        try:
            client = self._get_async_client()
            result = await client.search(
                query=query,
                search_depth=search_depth,
                topic=topic,
                max_results=max_results,
                include_answer=include_answer,
                include_raw_content=include_raw_content,
                include_images=include_images,
                **kwargs
            )

            return self._result(
                operation="search",
                success=True,
                data={
                    "answer": result.get("answer"),
                    "results": result.get("results", []),
                    "query": query,
                    "result_count": len(result.get("results", [])),
                },
                query=query,
            )
        except Exception as e:
            return self._result(
                operation="search",
                success=False,
                error=str(e),
                query=query,
            )

    async def extract(
        self,
        urls: Union[str, List[str]],
        *,
        extract_depth: str = "basic",
        format: str = "markdown",
        **kwargs
    ) -> SDKResult:
        """
        Extract content from URLs.

        Args:
            urls: Single URL or list of URLs
            extract_depth: 'basic' or 'advanced'
            format: 'markdown' or 'text'
            **kwargs: Additional options

        Returns:
            SDKResult with extracted content
        """
        try:
            client = self._get_async_client()
            result = await client.extract(
                urls=urls if isinstance(urls, list) else [urls],
                extract_depth=extract_depth,
                format=format,
                **kwargs
            )

            return self._result(
                operation="extract",
                success=True,
                data={
                    "results": result.get("results", []),
                    "failed_results": result.get("failed_results", []),
                    "url_count": len(urls) if isinstance(urls, list) else 1,
                },
            )
        except Exception as e:
            return self._result(
                operation="extract",
                success=False,
                error=str(e),
            )

    async def crawl(
        self,
        url: str,
        *,
        max_depth: int = 2,
        max_breadth: int = 10,
        limit: int = 50,
        format: str = "markdown",
        **kwargs
    ) -> SDKResult:
        """
        Crawl a website with depth/breadth controls.

        Args:
            url: Starting URL
            max_depth: Maximum crawl depth
            max_breadth: Maximum breadth per level
            limit: Maximum pages to crawl
            format: Output format ('markdown' or 'text')
            **kwargs: Additional options

        Returns:
            SDKResult with crawled content
        """
        try:
            client = self._get_async_client()
            result = await client.crawl(
                url=url,
                max_depth=max_depth,
                max_breadth=max_breadth,
                limit=limit,
                format=format,
                **kwargs
            )

            return self._result(
                operation="crawl",
                success=True,
                data=result,
                url=url,
            )
        except Exception as e:
            return self._result(
                operation="crawl",
                success=False,
                error=str(e),
                url=url,
            )

    async def research(
        self,
        input: str,
        *,
        model: str = "auto",
        citation_format: str = "numbered",
        **kwargs
    ) -> SDKResult:
        """
        Perform AI-powered deep research.

        Args:
            input: Research task or question
            model: 'mini', 'pro', or 'auto'
            citation_format: 'numbered', 'mla', 'apa', or 'chicago'
            **kwargs: Additional options

        Returns:
            SDKResult with research findings
        """
        try:
            client = self._get_async_client()
            result = await client.research(
                input=input,
                model=model,
                citation_format=citation_format,
                stream=False,
                **kwargs
            )

            return self._result(
                operation="research",
                success=True,
                data=result,
                input=input,
            )
        except Exception as e:
            return self._result(
                operation="research",
                success=False,
                error=str(e),
                input=input,
            )

    def search_sync(self, query: str, **kwargs) -> SDKResult:
        """Synchronous search wrapper."""
        return asyncio.run(self.search(query, **kwargs))

    def extract_sync(self, urls: Union[str, List[str]], **kwargs) -> SDKResult:
        """Synchronous extract wrapper."""
        return asyncio.run(self.extract(urls, **kwargs))


# =============================================================================
# MCP Wrapper - Model Context Protocol
# =============================================================================

class MCPWrapper(SDKWrapper):
    """
    Wrapper for MCP Python SDK - Model Context Protocol client/server library.

    Features:
    - Create MCP servers with FastMCP high-level API
    - Connect to MCP servers as client
    - Tool registration and calling
    - Resource and prompt management
    - Stdio and HTTP transport support

    The MCP SDK enables building AI tool servers and clients following
    the Model Context Protocol specification.
    """

    def __init__(self):
        super().__init__()
        self._servers: Dict[str, Any] = {}
        self._clients: Dict[str, Any] = {}

    @property
    def sdk_type(self) -> SDKType:
        return SDKType.MCP

    @property
    def available(self) -> bool:
        return MCP_AVAILABLE

    def create_server(
        self,
        name: str,
        *,
        instructions: Optional[str] = None,
    ) -> SDKResult:
        """
        Create a new FastMCP server instance.

        Args:
            name: Server name identifier
            instructions: Optional instructions for the server

        Returns:
            SDKResult with the created server instance
        """
        try:
            self._ensure_available()

            if name in self._servers:
                return self._result(
                    operation="create_server",
                    success=True,
                    data=self._servers[name],
                    note="Server already exists, returning existing instance",
                    server_name=name,
                )

            server = FastMCP(name, instructions=instructions)
            self._servers[name] = server

            return self._result(
                operation="create_server",
                success=True,
                data=server,
                server_name=name,
            )
        except Exception as e:
            return self._result(
                operation="create_server",
                success=False,
                error=str(e),
                server_name=name,
            )

    def get_server(self, name: str) -> SDKResult:
        """
        Get an existing server by name.

        Args:
            name: Server name identifier

        Returns:
            SDKResult with the server instance or error
        """
        if name in self._servers:
            return self._result(
                operation="get_server",
                success=True,
                data=self._servers[name],
                server_name=name,
            )
        return self._result(
            operation="get_server",
            success=False,
            error=f"Server '{name}' not found",
            server_name=name,
        )

    def register_tool(
        self,
        server_name: str,
        func: Callable,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> SDKResult:
        """
        Register a tool function with an MCP server.

        Args:
            server_name: Name of the server to register with
            func: The function to register as a tool
            name: Optional tool name (defaults to function name)
            description: Optional tool description

        Returns:
            SDKResult with registration status
        """
        try:
            self._ensure_available()

            if server_name not in self._servers:
                return self._result(
                    operation="register_tool",
                    success=False,
                    error=f"Server '{server_name}' not found",
                    server_name=server_name,
                )

            server = self._servers[server_name]
            tool_name = name or func.__name__

            # Use the @server.tool() decorator pattern
            decorated = server.tool(name=tool_name, description=description)(func)

            return self._result(
                operation="register_tool",
                success=True,
                data={"tool_name": tool_name, "function": func.__name__},
                server_name=server_name,
                tool_name=tool_name,
            )
        except Exception as e:
            return self._result(
                operation="register_tool",
                success=False,
                error=str(e),
                server_name=server_name,
            )

    async def call_tool(
        self,
        client: Any,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> SDKResult:
        """
        Call a tool on an MCP server via client.

        Args:
            client: MCP client instance (from async context manager)
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            SDKResult with tool call result
        """
        try:
            self._ensure_available()

            result = await client.call_tool(tool_name, arguments or {})

            return self._result(
                operation="call_tool",
                success=True,
                data=result,
                tool_name=tool_name,
            )
        except Exception as e:
            return self._result(
                operation="call_tool",
                success=False,
                error=str(e),
                tool_name=tool_name,
            )

    async def list_tools(self, client: Any) -> SDKResult:
        """
        List available tools on an MCP server.

        Args:
            client: MCP client instance

        Returns:
            SDKResult with list of tools
        """
        try:
            self._ensure_available()

            result = await client.list_tools()

            return self._result(
                operation="list_tools",
                success=True,
                data=result,
            )
        except Exception as e:
            return self._result(
                operation="list_tools",
                success=False,
                error=str(e),
            )

    async def list_resources(self, client: Any) -> SDKResult:
        """
        List available resources on an MCP server.

        Args:
            client: MCP client instance

        Returns:
            SDKResult with list of resources
        """
        try:
            self._ensure_available()

            result = await client.list_resources()

            return self._result(
                operation="list_resources",
                success=True,
                data=result,
            )
        except Exception as e:
            return self._result(
                operation="list_resources",
                success=False,
                error=str(e),
            )

    def status(self) -> Dict[str, Any]:
        """Get MCP wrapper status."""
        return {
            "available": MCP_AVAILABLE,
            "servers_created": list(self._servers.keys()),
            "clients_connected": list(self._clients.keys()),
        }


# =============================================================================
# Unified SDK Manager
# =============================================================================

class SDKManager:
    """
    Unified manager for all SDK wrappers.

    Provides single entry point for accessing all research/RAG SDKs
    with consistent interface and error handling.
    """

    def __init__(self):
        self._wrappers: Dict[SDKType, SDKWrapper] = {}

    @property
    def available_sdks(self) -> List[SDKType]:
        """List all available SDKs."""
        available = []
        if CRAWL4AI_AVAILABLE:
            available.append(SDKType.CRAWL4AI)
        if LIGHTRAG_AVAILABLE:
            available.append(SDKType.LIGHTRAG)
        if DSPY_AVAILABLE:
            available.append(SDKType.DSPY)
        if LLAMAINDEX_AVAILABLE:
            available.append(SDKType.LLAMAINDEX)
        if GRAPHRAG_AVAILABLE:
            available.append(SDKType.GRAPHRAG)
        if TAVILY_AVAILABLE:
            available.append(SDKType.TAVILY)
        if LANGGRAPH_AVAILABLE:
            available.append(SDKType.LANGGRAPH)
        if MCP_AVAILABLE:
            available.append(SDKType.MCP)
        return available

    def get_crawl4ai(self, **kwargs) -> Crawl4AIWrapper:
        """Get or create Crawl4AI wrapper."""
        if SDKType.CRAWL4AI not in self._wrappers:
            self._wrappers[SDKType.CRAWL4AI] = Crawl4AIWrapper(**kwargs)
        return self._wrappers[SDKType.CRAWL4AI]

    def get_lightrag(self, **kwargs) -> LightRAGWrapper:
        """Get or create LightRAG wrapper."""
        if SDKType.LIGHTRAG not in self._wrappers:
            self._wrappers[SDKType.LIGHTRAG] = LightRAGWrapper(**kwargs)
        return self._wrappers[SDKType.LIGHTRAG]

    def get_dspy(self, **kwargs) -> DSPyWrapper:
        """Get or create DSPy wrapper."""
        if SDKType.DSPY not in self._wrappers:
            self._wrappers[SDKType.DSPY] = DSPyWrapper(**kwargs)
        return self._wrappers[SDKType.DSPY]

    def get_llamaindex(self, **kwargs) -> LlamaIndexWrapper:
        """Get or create LlamaIndex wrapper."""
        if SDKType.LLAMAINDEX not in self._wrappers:
            self._wrappers[SDKType.LLAMAINDEX] = LlamaIndexWrapper(**kwargs)
        return self._wrappers[SDKType.LLAMAINDEX]

    def get_graphrag(self, **kwargs) -> GraphRAGWrapper:
        """Get or create GraphRAG wrapper."""
        if SDKType.GRAPHRAG not in self._wrappers:
            self._wrappers[SDKType.GRAPHRAG] = GraphRAGWrapper(**kwargs)
        return self._wrappers[SDKType.GRAPHRAG]

    def get_tavily(self, **kwargs) -> TavilyWrapper:
        """Get or create Tavily wrapper."""
        if SDKType.TAVILY not in self._wrappers:
            self._wrappers[SDKType.TAVILY] = TavilyWrapper(**kwargs)
        return self._wrappers[SDKType.TAVILY]

    def get_mcp(self, **kwargs) -> MCPWrapper:
        """Get or create MCP wrapper."""
        if SDKType.MCP not in self._wrappers:
            self._wrappers[SDKType.MCP] = MCPWrapper(**kwargs)
        return self._wrappers[SDKType.MCP]

    def status(self) -> Dict[str, Any]:
        """Get status of all SDKs."""
        return {
            "available": [sdk.value for sdk in self.available_sdks],
            "sdks": {
                "crawl4ai": {
                    "available": CRAWL4AI_AVAILABLE,
                    "partial": False,
                    "description": "LLM-optimized web crawling with markdown output",
                },
                "lightrag": {
                    "available": LIGHTRAG_AVAILABLE,
                    "partial": False,
                    "description": "Lightweight GraphRAG alternative",
                },
                "dspy": {
                    "available": DSPY_AVAILABLE,
                    "partial": DSPY_PARTIAL,
                    "description": "Declarative RAG optimization" + (" (partial: LM, Module, ChatAdapter)" if DSPY_PARTIAL and not DSPY_AVAILABLE else ""),
                },
                "llamaindex": {
                    "available": LLAMAINDEX_AVAILABLE,
                    "partial": False,
                    "description": "RAG indexing and query engines",
                },
                "graphrag": {
                    "available": GRAPHRAG_AVAILABLE,
                    "partial": True,  # Indexer adapters & config only
                    "description": "Microsoft knowledge graph (partial: indexer adapters & config)",
                },
                "tavily": {
                    "available": TAVILY_AVAILABLE,
                    "partial": False,
                    "description": "AI-powered web search, extraction, and research",
                },
                "langgraph": {
                    "available": LANGGRAPH_AVAILABLE,
                    "partial": False,
                    "description": "Stateful agent workflows (needs langchain_core)",
                },
                "mcp": {
                    "available": MCP_AVAILABLE,
                    "partial": False,
                    "description": "Model Context Protocol client/server SDK",
                },
            }
        }


# =============================================================================
# Singleton Instance
# =============================================================================

_sdk_manager: Optional[SDKManager] = None


def get_sdk_manager() -> SDKManager:
    """Get the singleton SDK manager instance."""
    global _sdk_manager
    if _sdk_manager is None:
        _sdk_manager = SDKManager()
    return _sdk_manager


# =============================================================================
# Convenience Functions
# =============================================================================

def crawl4ai(**kwargs) -> Crawl4AIWrapper:
    """Get Crawl4AI wrapper."""
    return get_sdk_manager().get_crawl4ai(**kwargs)


def lightrag(**kwargs) -> LightRAGWrapper:
    """Get LightRAG wrapper."""
    return get_sdk_manager().get_lightrag(**kwargs)


def dspy_wrapper(**kwargs) -> DSPyWrapper:
    """Get DSPy wrapper."""
    return get_sdk_manager().get_dspy(**kwargs)


def llamaindex(**kwargs) -> LlamaIndexWrapper:
    """Get LlamaIndex wrapper."""
    return get_sdk_manager().get_llamaindex(**kwargs)


def graphrag(**kwargs) -> GraphRAGWrapper:
    """Get GraphRAG wrapper."""
    return get_sdk_manager().get_graphrag(**kwargs)


def tavily(**kwargs) -> TavilyWrapper:
    """Get Tavily wrapper."""
    return get_sdk_manager().get_tavily(**kwargs)


def mcp_sdk(**kwargs) -> MCPWrapper:
    """Get MCP SDK wrapper."""
    return get_sdk_manager().get_mcp(**kwargs)


def sdk_status() -> Dict[str, Any]:
    """Get status of all SDKs."""
    return get_sdk_manager().status()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # SDK Types
    "SDKType",
    "SDKResult",

    # Wrappers
    "Crawl4AIWrapper",
    "LightRAGWrapper",
    "DSPyWrapper",
    "LlamaIndexWrapper",
    "GraphRAGWrapper",
    "TavilyWrapper",
    "MCPWrapper",

    # Manager
    "SDKManager",
    "get_sdk_manager",

    # Convenience
    "crawl4ai",
    "lightrag",
    "dspy_wrapper",
    "llamaindex",
    "graphrag",
    "tavily",
    "mcp_sdk",
    "sdk_status",

    # Availability flags
    "CRAWL4AI_AVAILABLE",
    "LIGHTRAG_AVAILABLE",
    "DSPY_AVAILABLE",
    "DSPY_PARTIAL",
    "LLAMAINDEX_AVAILABLE",
    "GRAPHRAG_AVAILABLE",
    "TAVILY_AVAILABLE",
    "LANGGRAPH_AVAILABLE",
    "MCP_AVAILABLE",

    # Partial DSPy exports (for when gepa is missing)
    "dspy_LM",
    "dspy_Module",
    "dspy_ChatAdapter",

    # Partial GraphRAG exports
    "graphrag_read_indexer_entities",
    "graphrag_create_config",

    # Tavily exports
    "TavilyClient",
    "AsyncTavilyClient",
    "TavilyHybridClient",

    # LangGraph exports (if available)
    "StateGraph",
    "MessagesState",
    "langgraph_START",
    "langgraph_END",

    # MCP exports (if available)
    "MCPClient",
    "MCPClientSession",
    "MCPServer",
    "FastMCP",
    "mcp_stdio_client",
    "mcp_stdio_server",
]
