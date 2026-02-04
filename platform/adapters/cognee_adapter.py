"""
cognee_adapter.py - Knowledge Graph RAG via Cognee 0.5.1
Provides graph-augmented retrieval with entity-relationship context.

Verified API patterns (from Context7 official docs + real import introspection):
  - cognee.add(text, dataset_name=) -> adds data to knowledge graph
  - cognee.cognify() -> processes into knowledge graph (extract, cognify, load)
  - cognee.search(query_text=, query_type=SearchType.GRAPH_COMPLETION) -> graph results
  - cognee.prune.prune_data() -> reset all data
  - SearchType options: GRAPH_COMPLETION, TEMPORAL, SUMMARIES, CHUNKS,
    RAG_COMPLETION, TRIPLET_COMPLETION, GRAPH_SUMMARY_COMPLETION,
    GRAPH_COMPLETION_COT, FEELING_LUCKY, CHUNKS_LEXICAL

Based on: Cognee official docs (docs.cognee.ai), Context7 /websites/cognee_ai
Achieves ~90% accuracy on multi-hop reasoning vs 60% for flat RAG (Cognee benchmarks).

Usage:
    adapter = CogneeAdapter()
    await adapter.initialize({"dataset_name": "my_dataset"})

    # Ingest documents
    result = await adapter.execute("ingest", texts=["AI transforms work", "Graphs store relationships"])

    # Search knowledge graph
    result = await adapter.execute("search", query="How does AI use graphs?")

    # Multi-type search
    result = await adapter.execute("search_multi", query="AI patterns", search_types=["GRAPH_COMPLETION", "TEMPORAL"])
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# SDK Layer imports
try:
    from core.orchestration.base import SDKAdapter, AdapterResult, AdapterStatus, SDKLayer, register_adapter
except ImportError:
    try:
        from core.orchestration.base import SDKAdapter, AdapterResult, AdapterStatus, SDKLayer, register_adapter
    except ImportError:
        from enum import IntEnum
        from abc import ABC, abstractmethod

        class SDKLayer(IntEnum):
            KNOWLEDGE = 7

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
            async def shutdown(self) -> AdapterResult: ...

        def register_adapter(name, layer, priority=0):
            def decorator(cls):
                return cls
            return decorator


# Cognee SDK
COGNEE_AVAILABLE = False
SearchType = None

try:
    import cognee
    from cognee.api.v1.search import SearchType
    COGNEE_AVAILABLE = True
except ImportError:
    logger.info("Cognee not installed - graph RAG disabled")
    cognee = None


@dataclass
class GraphResult:
    """Result from Cognee graph search."""
    content: str
    score: float = 0.0
    search_type: str = ""
    metadata: dict = field(default_factory=dict)


@register_adapter("cognee", SDKLayer.KNOWLEDGE, priority=22)
class CogneeAdapter(SDKAdapter):
    """
    Graph-augmented RAG adapter using Cognee knowledge graph.

    Operations:
        - ingest: Add texts to knowledge graph and process them
        - search: Search the knowledge graph with specified search type
        - search_multi: Search using multiple search types and merge results
        - reset: Clear all data from the knowledge graph
    """

    def __init__(self):
        self._status = AdapterStatus.UNINITIALIZED
        self._config: Dict[str, Any] = {}
        self._dataset_name: str = "unleash"
        self._stats = {
            "ingested": 0,
            "searches": 0,
            "search_multi": 0,
            "resets": 0,
            "avg_latency_ms": 0.0,
        }

    @property
    def sdk_name(self) -> str:
        return "cognee"

    @property
    def layer(self) -> SDKLayer:
        return SDKLayer.KNOWLEDGE

    @property
    def available(self) -> bool:
        return COGNEE_AVAILABLE

    async def initialize(self, config: Dict[str, Any]) -> AdapterResult:
        """Initialize Cognee adapter with configuration."""
        start = time.time()

        if not COGNEE_AVAILABLE:
            self._status = AdapterStatus.ERROR
            return AdapterResult(
                success=False,
                error="Cognee SDK not installed. Run: pip install cognee",
                latency_ms=(time.time() - start) * 1000,
            )

        try:
            self._dataset_name = config.get("dataset_name", "unleash")
            llm_model = config.get("llm_model", "openai/gpt-4o-mini")

            # Configure Cognee LLM
            llm_config = {"llm_model": llm_model}
            api_key = config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
            if api_key:
                llm_config["llm_api_key"] = api_key

            try:
                cognee.config.set_llm_config(llm_config)
            except Exception as e:
                logger.warning(f"Cognee LLM config failed (using defaults): {e}")

            self._config = config
            self._status = AdapterStatus.READY

            # Get available search types
            search_types = []
            if SearchType:
                search_types = [st.name for st in SearchType]

            return AdapterResult(
                success=True,
                data={
                    "status": "ready",
                    "dataset_name": self._dataset_name,
                    "llm_model": llm_model,
                    "search_types": search_types,
                    "features": ["ingest", "search", "search_multi", "reset"],
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
        """Execute Cognee operations."""
        start = time.time()

        operations = {
            "ingest": self._ingest,
            "search": self._search,
            "search_multi": self._search_multi,
            "reset": self._reset,
        }

        if operation not in operations:
            return AdapterResult(
                success=False,
                error=f"Unknown operation: {operation}. Valid: {list(operations.keys())}",
                latency_ms=(time.time() - start) * 1000,
            )

        try:
            result = await operations[operation](**kwargs)
            result.latency_ms = (time.time() - start) * 1000
            self._update_avg_latency(result.latency_ms)
            return result
        except Exception as e:
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start) * 1000,
            )

    async def _ingest(
        self,
        texts: List[str],
        dataset_name: Optional[str] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Add texts to Cognee knowledge graph and process them.

        Args:
            texts: List of text strings to ingest
            dataset_name: Override default dataset name
        """
        if not COGNEE_AVAILABLE:
            return AdapterResult(
                success=False,
                error="Cognee not installed",
            )

        ds = dataset_name or self._dataset_name
        added = 0
        errors = []

        for text in texts:
            if not text or len(text.strip()) < 10:
                continue
            try:
                await cognee.add(text, dataset_name=ds)
                added += 1
            except Exception as e:
                errors.append(str(e))
                logger.warning(f"Failed to add text to Cognee: {e}")

        if added > 0:
            try:
                await cognee.cognify()
                logger.info(f"Cognified {added} texts into knowledge graph")
            except Exception as e:
                logger.error(f"Cognify failed: {e}")
                return AdapterResult(
                    success=False,
                    error=f"Cognify failed: {e}",
                    data={"added_before_error": added},
                )

        self._stats["ingested"] += added

        return AdapterResult(
            success=True,
            data={
                "added": added,
                "total_texts": len(texts),
                "dataset_name": ds,
                "errors": errors if errors else None,
            },
        )

    async def _search(
        self,
        query: str,
        search_type: str = "GRAPH_COMPLETION",
        top_k: int = 10,
        **kwargs,
    ) -> AdapterResult:
        """
        Search the knowledge graph.

        Args:
            query: Natural language query
            search_type: One of GRAPH_COMPLETION, TEMPORAL, SUMMARIES, CHUNKS,
                         RAG_COMPLETION, GRAPH_COMPLETION_COT, FEELING_LUCKY
            top_k: Maximum results to return
        """
        self._stats["searches"] += 1

        if not COGNEE_AVAILABLE:
            return AdapterResult(
                success=True,
                data={
                    "results": [{"content": f"Mock result for: {query}", "score": 0.9}],
                    "mock": True,
                },
            )

        # Map string to SearchType enum
        try:
            st = SearchType[search_type]
        except (KeyError, TypeError):
            logger.warning(f"Unknown search type '{search_type}', using GRAPH_COMPLETION")
            st = SearchType.GRAPH_COMPLETION

        try:
            results = await cognee.search(
                query_text=query,
                query_type=st,
            )
        except Exception as e:
            logger.error(f"Cognee search failed: {e}")
            return AdapterResult(
                success=False,
                error=f"Cognee search failed: {e}",
            )

        graph_results = []
        if results:
            for i, item in enumerate(results[:top_k]):
                # Cognee returns various formats depending on search type
                if isinstance(item, str):
                    content = item
                elif isinstance(item, dict):
                    content = item.get("text", item.get("content", str(item)))
                else:
                    content = str(item)

                graph_results.append({
                    "content": content,
                    "score": 1.0 / (i + 1),  # Rank-based score
                    "search_type": search_type,
                    "index": i,
                })

        return AdapterResult(
            success=True,
            data={
                "results": graph_results,
                "count": len(graph_results),
                "query": query,
                "search_type": search_type,
            },
        )

    async def _search_multi(
        self,
        query: str,
        search_types: Optional[List[str]] = None,
        top_k: int = 5,
        **kwargs,
    ) -> AdapterResult:
        """
        Search using multiple search types and merge results.
        Default: GRAPH_COMPLETION + TEMPORAL for comprehensive coverage.
        """
        self._stats["search_multi"] += 1

        if search_types is None:
            search_types = ["GRAPH_COMPLETION", "TEMPORAL"]

        all_results = []
        errors = []

        for st in search_types:
            try:
                result = await self._search(query, search_type=st, top_k=top_k)
                if result.success and result.data:
                    for r in result.data.get("results", []):
                        all_results.append(r)
            except Exception as e:
                errors.append(f"{st}: {e}")
                logger.warning(f"Search type {st} failed: {e}")

        # Deduplicate by content
        seen = set()
        unique = []
        for r in all_results:
            content = r.get("content", "")
            key = content[:100] if content else ""
            if key and key not in seen:
                seen.add(key)
                unique.append(r)

        return AdapterResult(
            success=True,
            data={
                "results": unique,
                "count": len(unique),
                "query": query,
                "search_types": search_types,
                "errors": errors if errors else None,
            },
        )

    async def _reset(self, **kwargs) -> AdapterResult:
        """Reset/clear data from knowledge graph."""
        self._stats["resets"] += 1

        if not COGNEE_AVAILABLE:
            return AdapterResult(
                success=False,
                error="Cognee not installed",
            )

        try:
            await cognee.prune.prune_data()
            logger.info("Cognee data pruned")
            return AdapterResult(
                success=True,
                data={"message": "Knowledge graph data cleared"},
            )
        except Exception as e:
            logger.error(f"Cognee prune failed: {e}")
            return AdapterResult(
                success=False,
                error=f"Cognee prune failed: {e}",
            )

    def _update_avg_latency(self, latency: float):
        """Update rolling average latency."""
        total_ops = sum([
            self._stats["searches"],
            self._stats["search_multi"],
            self._stats["resets"],
        ])
        if total_ops > 0:
            self._stats["avg_latency_ms"] = (
                (self._stats["avg_latency_ms"] * (total_ops - 1) + latency)
                / total_ops
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return dict(self._stats)

    async def health_check(self) -> AdapterResult:
        """Check Cognee API health."""
        if not COGNEE_AVAILABLE:
            return AdapterResult(success=False, error="SDK not installed")

        if self._status != AdapterStatus.READY:
            return AdapterResult(
                success=True,
                data={"status": "not_initialized"},
            )

        return AdapterResult(
            success=True,
            data={"status": "healthy", "stats": self.get_stats()},
        )

    async def shutdown(self) -> AdapterResult:
        """Cleanup resources."""
        self._status = AdapterStatus.UNINITIALIZED
        return AdapterResult(success=True, data={"stats": self.get_stats()})


# Legacy class for backwards compatibility
class CogneeAdapterLegacy:
    """
    Legacy Graph-augmented RAG adapter (deprecated).

    Use CogneeAdapter with SDKAdapter interface instead:
        adapter = CogneeAdapter()
        await adapter.initialize({"dataset_name": "my_dataset"})
        result = await adapter.execute("search", query="...")
    """

    def __init__(self, dataset_name: str = "unleash", llm_model: str = "openai/gpt-4o-mini"):
        self._adapter = CogneeAdapter()
        self.dataset_name = dataset_name
        self._llm_model = llm_model
        self._initialized = False

    async def _ensure_init(self):
        if not self._initialized:
            await self._adapter.initialize({
                "dataset_name": self.dataset_name,
                "llm_model": self._llm_model,
            })
            self._initialized = True

    @staticmethod
    def is_available() -> bool:
        """Check if Cognee is installed."""
        return COGNEE_AVAILABLE

    async def ingest(self, texts: List[str], dataset_name: Optional[str] = None) -> int:
        """Legacy ingest method."""
        await self._ensure_init()
        result = await self._adapter.execute("ingest", texts=texts, dataset_name=dataset_name)
        return result.data.get("added", 0) if result.success else 0

    async def search(self, query: str, search_type: str = "GRAPH_COMPLETION", top_k: int = 10) -> List[GraphResult]:
        """Legacy search method."""
        await self._ensure_init()
        result = await self._adapter.execute("search", query=query, search_type=search_type, top_k=top_k)
        if not result.success:
            return []
        return [
            GraphResult(
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                search_type=r.get("search_type", ""),
                metadata={"index": r.get("index", 0)},
            )
            for r in result.data.get("results", [])
        ]

    async def search_multi(self, query: str, search_types: Optional[List[str]] = None, top_k: int = 5) -> List[GraphResult]:
        """Legacy search_multi method."""
        await self._ensure_init()
        result = await self._adapter.execute("search_multi", query=query, search_types=search_types, top_k=top_k)
        if not result.success:
            return []
        return [
            GraphResult(
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                search_type=r.get("search_type", ""),
                metadata={"index": r.get("index", 0)},
            )
            for r in result.data.get("results", [])
        ]

    async def reset(self):
        """Legacy reset method."""
        await self._ensure_init()
        await self._adapter.execute("reset")


def get_cognee_adapter() -> type[CogneeAdapter]:
    """Get the Cognee adapter class."""
    return CogneeAdapter


if __name__ == "__main__":
    import asyncio

    async def test():
        print(f"Cognee: {'available' if COGNEE_AVAILABLE else 'NOT installed'}")
        if COGNEE_AVAILABLE:
            print(f"Version: {cognee.__version__}")
            print(f"Search types: {[st.name for st in SearchType]}")

        adapter = CogneeAdapter()
        result = await adapter.initialize({})
        print(f"Initialize: {result}")

        result = await adapter.execute("search", query="test query")
        print(f"Search: {result}")

        await adapter.shutdown()
        print("OK")

    asyncio.run(test())
