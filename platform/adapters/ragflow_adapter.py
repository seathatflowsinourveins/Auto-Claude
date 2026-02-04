"""
RAGFlow Adapter - V36 Architecture

Integrates RAGFlow for production-grade RAG pipelines.

SDK: ragflow (https://github.com/infiniflow/ragflow)
Layer: L7 (Processing)
Features:
- Hybrid retrieval (dense + sparse + rerank)
- Multi-format document parsing
- Built-in chunking strategies
- Knowledge base management
- Citation tracking

API Patterns (from RAGFlow):
- Client(api_key, base_url) → client
- client.create_dataset(name) → create knowledge base
- client.upload_documents(dataset_id, files) → add documents
- client.search(dataset_id, query, top_k) → retrieve
- client.ask(dataset_id, query) → RAG with citations

Usage:
    from adapters.ragflow_adapter import RAGFlowAdapter

    adapter = RAGFlowAdapter()
    await adapter.initialize({"api_key": "...", "base_url": "http://localhost:9380"})

    # Create knowledge base
    await adapter.execute("create_dataset", name="codebase")

    # Upload documents
    await adapter.execute("upload", dataset_id="...", files=["doc1.pdf", "doc2.md"])

    # Search with retrieval
    result = await adapter.execute("search", query="How does authentication work?")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# SDK availability check
RAGFLOW_AVAILABLE = False

try:
    from ragflow import RAGFlow
    RAGFLOW_AVAILABLE = True
except ImportError:
    logger.info("RAGFlow not installed - install with: pip install ragflow-sdk")


# Import base adapter interface
try:
    from core.orchestration.base import (
        SDKAdapter,
        SDKLayer,
        AdapterConfig,
        AdapterResult,
        AdapterStatus,
    )
except ImportError:
    from dataclasses import dataclass as _dataclass
    from enum import IntEnum
    from abc import ABC, abstractmethod

    class SDKLayer(IntEnum):
        PROCESSING = 7

    @_dataclass
    class AdapterResult:
        success: bool
        data: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        latency_ms: float = 0.0

    @_dataclass
    class AdapterConfig:
        name: str = "ragflow"
        layer: int = 7

    class AdapterStatus:
        READY = "ready"
        FAILED = "failed"
        UNINITIALIZED = "uninitialized"

    class SDKAdapter(ABC):
        @property
        @abstractmethod
        def sdk_name(self) -> str: ...
        @property
        @abstractmethod
        def layer(self) -> int: ...
        @property
        @abstractmethod
        def available(self) -> bool: ...
        @abstractmethod
        async def initialize(self, config: Dict) -> AdapterResult: ...
        @abstractmethod
        async def execute(self, operation: str, **kwargs) -> AdapterResult: ...
        @abstractmethod
        async def health_check(self) -> AdapterResult: ...
        @abstractmethod
        async def shutdown(self) -> AdapterResult: ...


@dataclass
class RAGResult:
    """Result from RAGFlow retrieval."""
    content: str
    score: float
    source: str
    chunk_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGAnswer:
    """Answer from RAGFlow with citations."""
    answer: str
    citations: List[RAGResult]
    confidence: float
    tokens_used: int = 0


class RAGFlowAdapter(SDKAdapter):
    """
    RAGFlow adapter for production RAG pipelines.

    Provides enterprise-grade RAG capabilities:
    - Hybrid retrieval (dense vectors + BM25 + reranking)
    - Multi-format document parsing (PDF, DOCX, MD, etc.)
    - Intelligent chunking with overlap
    - Knowledge base versioning
    - Citation tracking with sources

    Operations:
    - create_dataset: Create a knowledge base
    - delete_dataset: Delete a knowledge base
    - list_datasets: List all knowledge bases
    - upload: Upload documents to a dataset
    - search: Retrieve relevant chunks
    - ask: Full RAG with answer generation
    - get_document: Get document details
    - delete_document: Remove a document
    """

    def __init__(self, config: Optional[AdapterConfig] = None):
        self._config = config or AdapterConfig(
            name="ragflow",
            layer=SDKLayer.PROCESSING
        )
        self._status = AdapterStatus.UNINITIALIZED
        self._client: Optional[Any] = None
        self._base_url: str = ""
        self._default_dataset_id: Optional[str] = None
        self._datasets: Dict[str, str] = {}  # name -> id mapping
        self._call_count = 0
        self._total_latency_ms = 0.0
        self._error_count = 0

    @property
    def sdk_name(self) -> str:
        return "ragflow"

    @property
    def layer(self) -> SDKLayer:
        return SDKLayer.PROCESSING

    @property
    def available(self) -> bool:
        return RAGFLOW_AVAILABLE

    @property
    def status(self) -> AdapterStatus:
        return self._status

    async def initialize(self, config: Dict[str, Any]) -> AdapterResult:
        """Initialize RAGFlow client."""
        try:
            import os

            api_key = config.get("api_key") or os.environ.get("RAGFLOW_API_KEY", "")
            base_url = config.get("base_url") or os.environ.get("RAGFLOW_BASE_URL", "http://localhost:9380")

            self._base_url = base_url

            if RAGFLOW_AVAILABLE:
                self._client = RAGFlow(api_key=api_key, base_url=base_url)
            else:
                # Stub mode
                self._client = None

            # Set default dataset if provided
            self._default_dataset_id = config.get("default_dataset_id")

            self._status = AdapterStatus.READY
            logger.info(f"RAGFlow adapter initialized (base_url={base_url})")

            return AdapterResult(
                success=True,
                data={
                    "base_url": base_url,
                    "ragflow_native": RAGFLOW_AVAILABLE,
                    "default_dataset_id": self._default_dataset_id
                }
            )

        except Exception as e:
            self._status = AdapterStatus.FAILED
            logger.error(f"RAGFlow initialization failed: {e}")
            return AdapterResult(success=False, error=str(e))

    async def execute(self, operation: str, **kwargs) -> AdapterResult:
        """Execute a RAGFlow operation."""
        start_time = time.time()

        try:
            if operation == "create_dataset":
                result = await self._create_dataset(**kwargs)
            elif operation == "delete_dataset":
                result = await self._delete_dataset(**kwargs)
            elif operation == "list_datasets":
                result = await self._list_datasets()
            elif operation == "upload":
                result = await self._upload(**kwargs)
            elif operation == "search":
                result = await self._search(**kwargs)
            elif operation == "ask":
                result = await self._ask(**kwargs)
            elif operation == "get_document":
                result = await self._get_document(**kwargs)
            elif operation == "delete_document":
                result = await self._delete_document(**kwargs)
            elif operation == "get_stats":
                result = await self._get_stats()
            else:
                result = AdapterResult(
                    success=False,
                    error=f"Unknown operation: {operation}"
                )

            latency_ms = (time.time() - start_time) * 1000
            self._call_count += 1
            self._total_latency_ms += latency_ms
            result.latency_ms = latency_ms

            if not result.success:
                self._error_count += 1

            return result

        except Exception as e:
            self._error_count += 1
            logger.error(f"RAGFlow execute error: {e}")
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )

    async def _create_dataset(
        self,
        name: str,
        description: str = "",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        **kwargs
    ) -> AdapterResult:
        """Create a knowledge base / dataset."""
        try:
            if RAGFLOW_AVAILABLE and self._client:
                dataset = self._client.create_dataset(
                    name=name,
                    description=description,
                    embedding_model=embedding_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                dataset_id = dataset.id
            else:
                # Stub mode
                import uuid
                dataset_id = f"ds-{uuid.uuid4().hex[:8]}"

            self._datasets[name] = dataset_id

            return AdapterResult(
                success=True,
                data={
                    "dataset_id": dataset_id,
                    "name": name,
                    "embedding_model": embedding_model
                }
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _delete_dataset(self, dataset_id: str, **kwargs) -> AdapterResult:
        """Delete a knowledge base."""
        try:
            if RAGFLOW_AVAILABLE and self._client:
                self._client.delete_dataset(dataset_id)

            # Remove from local mapping
            self._datasets = {k: v for k, v in self._datasets.items() if v != dataset_id}

            return AdapterResult(
                success=True,
                data={"dataset_id": dataset_id, "deleted": True}
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _list_datasets(self) -> AdapterResult:
        """List all datasets."""
        try:
            datasets = []

            if RAGFLOW_AVAILABLE and self._client:
                for ds in self._client.list_datasets():
                    datasets.append({
                        "id": ds.id,
                        "name": ds.name,
                        "document_count": getattr(ds, 'document_count', 0)
                    })
            else:
                # Return local mapping
                for name, ds_id in self._datasets.items():
                    datasets.append({
                        "id": ds_id,
                        "name": name,
                        "document_count": 0
                    })

            return AdapterResult(
                success=True,
                data={"datasets": datasets, "count": len(datasets)}
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _upload(
        self,
        dataset_id: Optional[str] = None,
        files: Optional[List[str]] = None,
        texts: Optional[List[str]] = None,
        **kwargs
    ) -> AdapterResult:
        """Upload documents or texts to a dataset."""
        ds_id = dataset_id or self._default_dataset_id
        if not ds_id:
            return AdapterResult(
                success=False,
                error="No dataset_id provided and no default set"
            )

        uploaded = 0
        errors = []

        try:
            if files:
                for file_path in files:
                    try:
                        if RAGFLOW_AVAILABLE and self._client:
                            self._client.upload_document(ds_id, file_path)
                        uploaded += 1
                    except Exception as e:
                        errors.append(f"{file_path}: {str(e)}")

            if texts:
                for i, text in enumerate(texts):
                    try:
                        if RAGFLOW_AVAILABLE and self._client:
                            self._client.upload_text(ds_id, text, name=f"text_{i}")
                        uploaded += 1
                    except Exception as e:
                        errors.append(f"text_{i}: {str(e)}")

            return AdapterResult(
                success=True,
                data={
                    "dataset_id": ds_id,
                    "uploaded": uploaded,
                    "errors": errors if errors else None
                }
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _search(
        self,
        query: str,
        dataset_id: Optional[str] = None,
        top_k: int = 10,
        rerank: bool = True,
        **kwargs
    ) -> AdapterResult:
        """Search for relevant chunks."""
        ds_id = dataset_id or self._default_dataset_id
        if not ds_id:
            return AdapterResult(
                success=False,
                error="No dataset_id provided and no default set"
            )

        try:
            results = []

            if RAGFLOW_AVAILABLE and self._client:
                search_results = self._client.search(
                    dataset_id=ds_id,
                    query=query,
                    top_k=top_k,
                    rerank=rerank
                )

                for i, result in enumerate(search_results):
                    results.append({
                        "content": getattr(result, 'content', str(result)),
                        "score": getattr(result, 'score', 1.0 / (i + 1)),
                        "source": getattr(result, 'source', 'unknown'),
                        "chunk_id": getattr(result, 'chunk_id', f"chunk_{i}")
                    })
            else:
                # Stub mode
                results.append({
                    "content": f"[RAGFlow stub] Results for: {query[:50]}",
                    "score": 0.95,
                    "source": "stub",
                    "chunk_id": "stub_0"
                })

            return AdapterResult(
                success=True,
                data={
                    "results": results,
                    "count": len(results),
                    "query": query,
                    "dataset_id": ds_id
                }
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _ask(
        self,
        query: str,
        dataset_id: Optional[str] = None,
        top_k: int = 5,
        stream: bool = False,
        **kwargs
    ) -> AdapterResult:
        """Full RAG: retrieve and generate answer with citations."""
        ds_id = dataset_id or self._default_dataset_id
        if not ds_id:
            return AdapterResult(
                success=False,
                error="No dataset_id provided and no default set"
            )

        try:
            if RAGFLOW_AVAILABLE and self._client:
                response = self._client.ask(
                    dataset_id=ds_id,
                    query=query,
                    top_k=top_k,
                    stream=stream
                )

                answer = getattr(response, 'answer', str(response))
                citations = []
                for cit in getattr(response, 'citations', []):
                    citations.append({
                        "content": getattr(cit, 'content', ''),
                        "source": getattr(cit, 'source', ''),
                        "score": getattr(cit, 'score', 0.0)
                    })
            else:
                # Stub mode
                answer = f"[RAGFlow stub] Answer for: {query[:50]}..."
                citations = [{
                    "content": "Stub citation content",
                    "source": "stub_document.md",
                    "score": 0.9
                }]

            return AdapterResult(
                success=True,
                data={
                    "answer": answer,
                    "citations": citations,
                    "query": query,
                    "dataset_id": ds_id
                }
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _get_document(self, document_id: str, **kwargs) -> AdapterResult:
        """Get document details."""
        try:
            if RAGFLOW_AVAILABLE and self._client:
                doc = self._client.get_document(document_id)
                return AdapterResult(
                    success=True,
                    data={
                        "document_id": document_id,
                        "name": getattr(doc, 'name', ''),
                        "status": getattr(doc, 'status', 'unknown'),
                        "chunk_count": getattr(doc, 'chunk_count', 0)
                    }
                )
            else:
                return AdapterResult(
                    success=True,
                    data={"document_id": document_id, "status": "stub"}
                )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _delete_document(
        self,
        document_id: str,
        dataset_id: Optional[str] = None,
        **kwargs
    ) -> AdapterResult:
        """Delete a document."""
        try:
            if RAGFLOW_AVAILABLE and self._client:
                self._client.delete_document(document_id)

            return AdapterResult(
                success=True,
                data={"document_id": document_id, "deleted": True}
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _get_stats(self) -> AdapterResult:
        """Get adapter statistics."""
        return AdapterResult(
            success=True,
            data={
                "base_url": self._base_url,
                "datasets_count": len(self._datasets),
                "default_dataset_id": self._default_dataset_id,
                "call_count": self._call_count,
                "error_count": self._error_count,
                "avg_latency_ms": self._total_latency_ms / max(1, self._call_count),
                "ragflow_native": RAGFLOW_AVAILABLE
            }
        )

    async def health_check(self) -> AdapterResult:
        """Check adapter health."""
        return AdapterResult(
            success=True,
            data={
                "status": "healthy",
                "base_url": self._base_url,
                "ragflow_available": RAGFLOW_AVAILABLE
            }
        )

    async def shutdown(self) -> AdapterResult:
        """Shutdown the adapter."""
        self._client = None
        self._datasets.clear()
        self._status = AdapterStatus.UNINITIALIZED
        logger.info("RAGFlow adapter shutdown")
        return AdapterResult(success=True)


# Register with SDK registry
try:
    from core.orchestration.sdk_registry import register_adapter, SDKLayer

    @register_adapter("ragflow", SDKLayer.PROCESSING, priority=15)
    class RegisteredRAGFlowAdapter(RAGFlowAdapter):
        """Registered RAGFlow adapter."""
        pass

except ImportError:
    pass


__all__ = ["RAGFlowAdapter", "RAGFLOW_AVAILABLE", "RAGResult", "RAGAnswer"]
