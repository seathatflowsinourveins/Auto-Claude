"""
Streaming Response Support for RAG Pipeline

This module provides streaming capabilities for the RAG pipeline, enabling:
- Chunk-by-chunk LLM generation streaming
- Progressive multi-source retrieval (stream results as they arrive)
- Top-first reranking streaming (stream best results first)
- Early termination support for cancellation
- Progress callbacks for UI integration

Usage:
    from core.rag.streaming import (
        StreamingRAGResponse,
        StreamingConfig,
        StreamEvent,
        StreamEventType,
    )

    # Stream pipeline execution
    async for event in pipeline.run_streaming("What is RAG?"):
        if event.event_type == StreamEventType.GENERATION_CHUNK:
            print(event.chunk, end="", flush=True)
        elif event.event_type == StreamEventType.RETRIEVAL_RESULT:
            print(f"Retrieved from {event.source}: {event.document_count} docs")
        elif event.event_type == StreamEventType.COMPLETE:
            print(f"\nTotal latency: {event.total_latency_ms}ms")

    # With progress callbacks
    config = StreamingConfig(
        progress_callback=lambda p: print(f"Progress: {p.stage} - {p.percent}%"),
        stream_retrieval=True,
        stream_reranking=True,
    )
    async for event in pipeline.run_streaming("query", streaming_config=config):
        handle_event(event)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class StreamEventType(str, Enum):
    """Types of streaming events."""
    # Pipeline lifecycle events
    STARTED = "started"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"

    # Stage events
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETE = "stage_complete"

    # Retrieval events
    RETRIEVAL_STARTED = "retrieval_started"
    RETRIEVAL_RESULT = "retrieval_result"
    RETRIEVAL_COMPLETE = "retrieval_complete"

    # Reranking events
    RERANK_STARTED = "rerank_started"
    RERANK_RESULT = "rerank_result"
    RERANK_COMPLETE = "rerank_complete"

    # Generation events
    GENERATION_STARTED = "generation_started"
    GENERATION_CHUNK = "generation_chunk"
    GENERATION_COMPLETE = "generation_complete"

    # Progress events
    PROGRESS = "progress"


class StreamingStage(str, Enum):
    """Stages in the streaming pipeline."""
    QUERY_REWRITE = "query_rewrite"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    GENERATION = "generation"
    EVALUATION = "evaluation"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class StreamProgress:
    """Progress information for callbacks."""
    stage: StreamingStage
    percent: float  # 0.0 to 100.0
    message: str = ""
    elapsed_ms: float = 0.0
    items_processed: int = 0
    items_total: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamEvent:
    """A streaming event from the RAG pipeline.

    Attributes:
        event_type: Type of the streaming event
        timestamp: Event timestamp in milliseconds since pipeline start
        chunk: Text chunk for generation events
        document: Retrieved document for retrieval events
        documents: List of documents for batch retrieval/rerank events
        document_count: Number of documents in batch
        source: Source name for retrieval events
        score: Relevance score for reranked documents
        stage: Current pipeline stage
        progress: Progress information
        error: Error message for error events
        metadata: Additional event metadata
        total_latency_ms: Total pipeline latency (for COMPLETE event)
        response: Final response text (for COMPLETE event)
        confidence: Confidence score (for COMPLETE event)
    """
    event_type: StreamEventType
    timestamp: float = 0.0
    chunk: Optional[str] = None
    document: Optional[Dict[str, Any]] = None
    documents: Optional[List[Dict[str, Any]]] = None
    document_count: int = 0
    source: Optional[str] = None
    score: Optional[float] = None
    stage: Optional[StreamingStage] = None
    progress: Optional[StreamProgress] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    response: Optional[str] = None
    confidence: float = 0.0

    def is_terminal(self) -> bool:
        """Check if this is a terminal event."""
        return self.event_type in (
            StreamEventType.COMPLETE,
            StreamEventType.ERROR,
            StreamEventType.CANCELLED,
        )


@dataclass
class StreamingConfig:
    """Configuration for streaming behavior.

    Attributes:
        stream_retrieval: Stream retrieval results as they arrive
        stream_reranking: Stream reranked results (top-first)
        stream_generation: Stream LLM generation chunks
        chunk_size: Minimum characters per generation chunk
        retrieval_batch_size: Results per retrieval stream event
        rerank_batch_size: Results per rerank stream event
        progress_callback: Optional callback for progress updates
        progress_interval_ms: Minimum interval between progress callbacks
        enable_early_termination: Allow pipeline cancellation
        generation_timeout_seconds: Timeout for generation stage
        retrieval_timeout_seconds: Timeout for retrieval stage
    """
    stream_retrieval: bool = True
    stream_reranking: bool = True
    stream_generation: bool = True
    chunk_size: int = 10
    retrieval_batch_size: int = 5
    rerank_batch_size: int = 3
    progress_callback: Optional[Callable[[StreamProgress], None]] = None
    progress_interval_ms: float = 100.0
    enable_early_termination: bool = True
    generation_timeout_seconds: float = 120.0
    retrieval_timeout_seconds: float = 30.0


# =============================================================================
# PROTOCOLS
# =============================================================================

class StreamingLLMProvider(Protocol):
    """Protocol for LLM providers with streaming support."""

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate text with streaming response.

        Yields:
            Text chunks as they are generated
        """
        ...


class StreamingRetrieverProtocol(Protocol):
    """Protocol for retrievers with streaming support."""

    @property
    def name(self) -> str:
        """Retriever name."""
        ...

    async def retrieve_stream(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Retrieve documents with streaming.

        Yields:
            Documents as they are retrieved
        """
        ...


# =============================================================================
# CANCELLATION TOKEN
# =============================================================================

class CancellationToken:
    """Token for early termination of streaming operations."""

    def __init__(self):
        self._cancelled = False
        self._cancel_event = asyncio.Event()
        self._reason: Optional[str] = None

    def cancel(self, reason: str = "User cancelled") -> None:
        """Cancel the operation."""
        self._cancelled = True
        self._reason = reason
        self._cancel_event.set()

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancelled

    @property
    def reason(self) -> Optional[str]:
        """Get cancellation reason."""
        return self._reason

    async def wait_for_cancel(self, timeout: Optional[float] = None) -> bool:
        """Wait for cancellation.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if cancelled, False if timed out
        """
        try:
            await asyncio.wait_for(self._cancel_event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def raise_if_cancelled(self) -> None:
        """Raise CancellationError if cancelled."""
        if self._cancelled:
            raise CancellationError(self._reason or "Operation cancelled")


class CancellationError(Exception):
    """Exception raised when a streaming operation is cancelled."""
    pass


# =============================================================================
# STREAMING RAG RESPONSE
# =============================================================================

class StreamingRAGResponse:
    """
    Streaming response handler for RAG pipeline.

    Manages the streaming state and provides methods for emitting events,
    tracking progress, and handling cancellation.

    Usage:
        response = StreamingRAGResponse(config)
        response.start()

        # Emit events
        response.emit_retrieval_result(doc, "exa")
        response.emit_generation_chunk("Hello")

        # Check status
        if response.is_complete:
            print(response.full_response)
    """

    def __init__(
        self,
        config: Optional[StreamingConfig] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ):
        """Initialize streaming response.

        Args:
            config: Streaming configuration
            cancellation_token: Token for early termination
        """
        self.config = config or StreamingConfig()
        self.cancellation_token = cancellation_token or CancellationToken()

        self._start_time: float = 0.0
        self._events: List[StreamEvent] = []
        self._event_queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        self._is_started = False
        self._is_complete = False
        self._is_cancelled = False
        self._error: Optional[str] = None

        # Accumulated state
        self._full_response: str = ""
        self._retrieved_documents: List[Dict[str, Any]] = []
        self._reranked_documents: List[Dict[str, Any]] = []
        self._contexts_used: List[str] = []
        self._confidence: float = 0.0
        self._current_stage: Optional[StreamingStage] = None

        # Progress tracking
        self._last_progress_time: float = 0.0
        self._stage_start_times: Dict[StreamingStage, float] = {}

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def is_started(self) -> bool:
        """Check if streaming has started."""
        return self._is_started

    @property
    def is_complete(self) -> bool:
        """Check if streaming is complete."""
        return self._is_complete

    @property
    def is_cancelled(self) -> bool:
        """Check if streaming was cancelled."""
        return self._is_cancelled

    @property
    def full_response(self) -> str:
        """Get the accumulated full response."""
        return self._full_response

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self._start_time == 0:
            return 0.0
        return (time.time() - self._start_time) * 1000

    @property
    def retrieved_documents(self) -> List[Dict[str, Any]]:
        """Get all retrieved documents."""
        return self._retrieved_documents.copy()

    @property
    def reranked_documents(self) -> List[Dict[str, Any]]:
        """Get all reranked documents."""
        return self._reranked_documents.copy()

    @property
    def contexts_used(self) -> List[str]:
        """Get contexts used for generation."""
        return self._contexts_used.copy()

    @property
    def current_stage(self) -> Optional[StreamingStage]:
        """Get current pipeline stage."""
        return self._current_stage

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    def start(self) -> StreamEvent:
        """Start the streaming response.

        Returns:
            The STARTED event
        """
        self._start_time = time.time()
        self._is_started = True

        event = StreamEvent(
            event_type=StreamEventType.STARTED,
            timestamp=0.0,
            metadata={"config": self.config.__dict__}
        )
        self._emit_event(event)
        return event

    def complete(
        self,
        response: str,
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StreamEvent:
        """Complete the streaming response.

        Args:
            response: Final response text
            confidence: Confidence score
            metadata: Additional metadata

        Returns:
            The COMPLETE event
        """
        self._is_complete = True
        self._full_response = response
        self._confidence = confidence

        event = StreamEvent(
            event_type=StreamEventType.COMPLETE,
            timestamp=self.elapsed_ms,
            response=response,
            confidence=confidence,
            total_latency_ms=self.elapsed_ms,
            metadata=metadata or {},
        )
        self._emit_event(event)
        return event

    def error(self, error_message: str, metadata: Optional[Dict[str, Any]] = None) -> StreamEvent:
        """Emit an error event.

        Args:
            error_message: Error description
            metadata: Additional metadata

        Returns:
            The ERROR event
        """
        self._is_complete = True
        self._error = error_message

        event = StreamEvent(
            event_type=StreamEventType.ERROR,
            timestamp=self.elapsed_ms,
            error=error_message,
            total_latency_ms=self.elapsed_ms,
            metadata=metadata or {},
        )
        self._emit_event(event)
        return event

    def cancel(self, reason: str = "User cancelled") -> StreamEvent:
        """Cancel the streaming response.

        Args:
            reason: Cancellation reason

        Returns:
            The CANCELLED event
        """
        self._is_complete = True
        self._is_cancelled = True
        self.cancellation_token.cancel(reason)

        event = StreamEvent(
            event_type=StreamEventType.CANCELLED,
            timestamp=self.elapsed_ms,
            error=reason,
            total_latency_ms=self.elapsed_ms,
        )
        self._emit_event(event)
        return event

    # -------------------------------------------------------------------------
    # Stage Events
    # -------------------------------------------------------------------------

    def start_stage(self, stage: StreamingStage) -> StreamEvent:
        """Start a pipeline stage.

        Args:
            stage: The stage being started

        Returns:
            The STAGE_STARTED event
        """
        self._current_stage = stage
        self._stage_start_times[stage] = time.time()

        event = StreamEvent(
            event_type=StreamEventType.STAGE_STARTED,
            timestamp=self.elapsed_ms,
            stage=stage,
        )
        self._emit_event(event)
        return event

    def complete_stage(self, stage: StreamingStage, metadata: Optional[Dict[str, Any]] = None) -> StreamEvent:
        """Complete a pipeline stage.

        Args:
            stage: The stage being completed
            metadata: Stage completion metadata

        Returns:
            The STAGE_COMPLETE event
        """
        start_time = self._stage_start_times.get(stage, self._start_time)
        stage_latency = (time.time() - start_time) * 1000

        event = StreamEvent(
            event_type=StreamEventType.STAGE_COMPLETE,
            timestamp=self.elapsed_ms,
            stage=stage,
            metadata={
                "stage_latency_ms": stage_latency,
                **(metadata or {}),
            },
        )
        self._emit_event(event)
        return event

    # -------------------------------------------------------------------------
    # Retrieval Events
    # -------------------------------------------------------------------------

    def start_retrieval(self, sources: List[str]) -> StreamEvent:
        """Start retrieval stage.

        Args:
            sources: List of retriever source names

        Returns:
            The RETRIEVAL_STARTED event
        """
        self._current_stage = StreamingStage.RETRIEVAL
        self._stage_start_times[StreamingStage.RETRIEVAL] = time.time()

        event = StreamEvent(
            event_type=StreamEventType.RETRIEVAL_STARTED,
            timestamp=self.elapsed_ms,
            stage=StreamingStage.RETRIEVAL,
            metadata={"sources": sources},
        )
        self._emit_event(event)
        return event

    def emit_retrieval_result(
        self,
        document: Dict[str, Any],
        source: str,
        score: Optional[float] = None,
    ) -> StreamEvent:
        """Emit a single retrieval result.

        Args:
            document: Retrieved document
            source: Source retriever name
            score: Optional relevance score

        Returns:
            The RETRIEVAL_RESULT event
        """
        self._retrieved_documents.append(document)

        event = StreamEvent(
            event_type=StreamEventType.RETRIEVAL_RESULT,
            timestamp=self.elapsed_ms,
            document=document,
            source=source,
            score=score,
            document_count=len(self._retrieved_documents),
            stage=StreamingStage.RETRIEVAL,
        )

        if self.config.stream_retrieval:
            self._emit_event(event)

        return event

    def emit_retrieval_batch(
        self,
        documents: List[Dict[str, Any]],
        source: str,
    ) -> StreamEvent:
        """Emit a batch of retrieval results.

        Args:
            documents: Retrieved documents
            source: Source retriever name

        Returns:
            The RETRIEVAL_RESULT event
        """
        self._retrieved_documents.extend(documents)

        event = StreamEvent(
            event_type=StreamEventType.RETRIEVAL_RESULT,
            timestamp=self.elapsed_ms,
            documents=documents,
            source=source,
            document_count=len(documents),
            stage=StreamingStage.RETRIEVAL,
            metadata={"batch": True, "total_retrieved": len(self._retrieved_documents)},
        )

        if self.config.stream_retrieval:
            self._emit_event(event)

        return event

    def complete_retrieval(self, total_documents: int) -> StreamEvent:
        """Complete retrieval stage.

        Args:
            total_documents: Total documents retrieved

        Returns:
            The RETRIEVAL_COMPLETE event
        """
        start_time = self._stage_start_times.get(StreamingStage.RETRIEVAL, self._start_time)
        stage_latency = (time.time() - start_time) * 1000

        event = StreamEvent(
            event_type=StreamEventType.RETRIEVAL_COMPLETE,
            timestamp=self.elapsed_ms,
            document_count=total_documents,
            stage=StreamingStage.RETRIEVAL,
            metadata={"stage_latency_ms": stage_latency},
        )
        self._emit_event(event)
        return event

    # -------------------------------------------------------------------------
    # Reranking Events
    # -------------------------------------------------------------------------

    def start_reranking(self, document_count: int) -> StreamEvent:
        """Start reranking stage.

        Args:
            document_count: Number of documents to rerank

        Returns:
            The RERANK_STARTED event
        """
        self._current_stage = StreamingStage.RERANKING
        self._stage_start_times[StreamingStage.RERANKING] = time.time()

        event = StreamEvent(
            event_type=StreamEventType.RERANK_STARTED,
            timestamp=self.elapsed_ms,
            document_count=document_count,
            stage=StreamingStage.RERANKING,
        )
        self._emit_event(event)
        return event

    def emit_rerank_result(
        self,
        document: Dict[str, Any],
        score: float,
        rank: int,
    ) -> StreamEvent:
        """Emit a single reranked result (top-first streaming).

        Args:
            document: Reranked document
            score: Relevance score
            rank: Position in final ranking (1-indexed)

        Returns:
            The RERANK_RESULT event
        """
        self._reranked_documents.append(document)

        event = StreamEvent(
            event_type=StreamEventType.RERANK_RESULT,
            timestamp=self.elapsed_ms,
            document=document,
            score=score,
            stage=StreamingStage.RERANKING,
            metadata={"rank": rank},
        )

        if self.config.stream_reranking:
            self._emit_event(event)

        return event

    def complete_reranking(self, top_documents: List[Dict[str, Any]]) -> StreamEvent:
        """Complete reranking stage.

        Args:
            top_documents: Final reranked documents

        Returns:
            The RERANK_COMPLETE event
        """
        self._reranked_documents = top_documents
        start_time = self._stage_start_times.get(StreamingStage.RERANKING, self._start_time)
        stage_latency = (time.time() - start_time) * 1000

        event = StreamEvent(
            event_type=StreamEventType.RERANK_COMPLETE,
            timestamp=self.elapsed_ms,
            documents=top_documents,
            document_count=len(top_documents),
            stage=StreamingStage.RERANKING,
            metadata={"stage_latency_ms": stage_latency},
        )
        self._emit_event(event)
        return event

    # -------------------------------------------------------------------------
    # Generation Events
    # -------------------------------------------------------------------------

    def start_generation(self, contexts: List[str]) -> StreamEvent:
        """Start generation stage.

        Args:
            contexts: Contexts being used for generation

        Returns:
            The GENERATION_STARTED event
        """
        self._current_stage = StreamingStage.GENERATION
        self._stage_start_times[StreamingStage.GENERATION] = time.time()
        self._contexts_used = contexts.copy()

        event = StreamEvent(
            event_type=StreamEventType.GENERATION_STARTED,
            timestamp=self.elapsed_ms,
            stage=StreamingStage.GENERATION,
            metadata={"context_count": len(contexts)},
        )
        self._emit_event(event)
        return event

    def emit_generation_chunk(self, chunk: str) -> StreamEvent:
        """Emit a generation text chunk.

        Args:
            chunk: Text chunk from LLM

        Returns:
            The GENERATION_CHUNK event
        """
        self._full_response += chunk

        event = StreamEvent(
            event_type=StreamEventType.GENERATION_CHUNK,
            timestamp=self.elapsed_ms,
            chunk=chunk,
            stage=StreamingStage.GENERATION,
            metadata={"total_length": len(self._full_response)},
        )

        if self.config.stream_generation:
            self._emit_event(event)

        return event

    def complete_generation(self, response: str, confidence: float = 0.0) -> StreamEvent:
        """Complete generation stage.

        Args:
            response: Full generated response
            confidence: Confidence score

        Returns:
            The GENERATION_COMPLETE event
        """
        self._full_response = response
        self._confidence = confidence
        start_time = self._stage_start_times.get(StreamingStage.GENERATION, self._start_time)
        stage_latency = (time.time() - start_time) * 1000

        event = StreamEvent(
            event_type=StreamEventType.GENERATION_COMPLETE,
            timestamp=self.elapsed_ms,
            response=response,
            confidence=confidence,
            stage=StreamingStage.GENERATION,
            metadata={
                "stage_latency_ms": stage_latency,
                "response_length": len(response),
            },
        )
        self._emit_event(event)
        return event

    # -------------------------------------------------------------------------
    # Progress Events
    # -------------------------------------------------------------------------

    def emit_progress(
        self,
        stage: StreamingStage,
        percent: float,
        message: str = "",
        items_processed: int = 0,
        items_total: int = 0,
    ) -> Optional[StreamEvent]:
        """Emit a progress update.

        Args:
            stage: Current stage
            percent: Progress percentage (0-100)
            message: Progress message
            items_processed: Items processed so far
            items_total: Total items to process

        Returns:
            The PROGRESS event, or None if throttled
        """
        now = time.time() * 1000
        if now - self._last_progress_time < self.config.progress_interval_ms:
            return None

        self._last_progress_time = now

        progress = StreamProgress(
            stage=stage,
            percent=percent,
            message=message,
            elapsed_ms=self.elapsed_ms,
            items_processed=items_processed,
            items_total=items_total,
        )

        # Call progress callback if configured
        if self.config.progress_callback:
            try:
                self.config.progress_callback(progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

        event = StreamEvent(
            event_type=StreamEventType.PROGRESS,
            timestamp=self.elapsed_ms,
            progress=progress,
            stage=stage,
        )
        self._emit_event(event)
        return event

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _emit_event(self, event: StreamEvent) -> None:
        """Add event to queue and history."""
        self._events.append(event)
        self._event_queue.put_nowait(event)

    async def iter_events(self) -> AsyncGenerator[StreamEvent, None]:
        """Iterate over streaming events.

        Yields:
            StreamEvent objects as they are emitted
        """
        while True:
            # Check for cancellation
            if self.cancellation_token.is_cancelled and self._event_queue.empty():
                yield StreamEvent(
                    event_type=StreamEventType.CANCELLED,
                    timestamp=self.elapsed_ms,
                    error=self.cancellation_token.reason,
                )
                break

            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=0.1
                )
                yield event

                if event.is_terminal():
                    break

            except asyncio.TimeoutError:
                # Check if complete without terminal event
                if self._is_complete and self._event_queue.empty():
                    break
                continue

    def get_all_events(self) -> List[StreamEvent]:
        """Get all emitted events."""
        return self._events.copy()


# =============================================================================
# STREAMING GENERATORS
# =============================================================================

async def stream_multi_source_retrieval(
    retrievers: List[Any],
    query: str,
    top_k: int = 5,
    timeout_seconds: float = 30.0,
) -> AsyncGenerator[Tuple[str, List[Dict[str, Any]]], None]:
    """Stream retrieval results from multiple sources as they arrive.

    Args:
        retrievers: List of retriever instances
        query: Search query
        top_k: Results per retriever
        timeout_seconds: Timeout for all retrievals

    Yields:
        Tuples of (source_name, documents)
    """
    async def retrieve_from_source(retriever: Any) -> Tuple[str, List[Dict[str, Any]]]:
        """Retrieve from a single source."""
        name = getattr(retriever, 'name', type(retriever).__name__)
        try:
            # Check for streaming support
            if hasattr(retriever, 'retrieve_stream'):
                docs = []
                async for doc in retriever.retrieve_stream(query, top_k=top_k):
                    docs.append(doc)
                return (name, docs)
            else:
                docs = await retriever.retrieve(query, top_k=top_k)
                return (name, docs)
        except Exception as e:
            logger.warning(f"Retrieval from {name} failed: {e}")
            return (name, [])

    # Create tasks for all retrievers
    tasks = {
        asyncio.create_task(retrieve_from_source(r)): r
        for r in retrievers
    }

    # Use as_completed to yield results as they arrive
    try:
        for coro in asyncio.as_completed(list(tasks.keys()), timeout=timeout_seconds):
            try:
                source_name, documents = await coro
                if documents:
                    yield (source_name, documents)
            except Exception as e:
                logger.warning(f"Retrieval task error: {e}")

    except asyncio.TimeoutError:
        logger.warning("Multi-source retrieval timed out")
        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()


async def stream_reranking_top_first(
    documents: List[Dict[str, Any]],
    query: str,
    reranker: Any,
    top_k: int = 10,
) -> AsyncGenerator[Tuple[Dict[str, Any], float, int], None]:
    """Stream reranked results, yielding top results first.

    This allows streaming the best results before full reranking completes.

    Args:
        documents: Documents to rerank
        query: Search query
        reranker: Reranker instance
        top_k: Number of results to return

    Yields:
        Tuples of (document, score, rank)
    """
    if not documents:
        return

    try:
        # Perform reranking
        from .reranker import Document as RerankerDoc

        reranker_docs = [
            RerankerDoc(
                id=str(i),
                content=doc.get("content", str(doc)),
                metadata=doc.get("metadata", {}),
            )
            for i, doc in enumerate(documents)
        ]

        reranked = await reranker.rerank(query, reranker_docs, top_k=top_k)

        # Yield results top-first (already sorted by score)
        for rank, scored_doc in enumerate(reranked, start=1):
            original_idx = int(scored_doc.document.id) if scored_doc.document.id.isdigit() else 0
            original_doc = documents[original_idx] if original_idx < len(documents) else {}

            yield (
                {
                    "content": scored_doc.document.content,
                    "metadata": scored_doc.document.metadata,
                    "original": original_doc,
                },
                scored_doc.score,
                rank,
            )

    except Exception as e:
        logger.error(f"Reranking error: {e}")
        # Fallback: yield documents without reranking
        for rank, doc in enumerate(documents[:top_k], start=1):
            yield (doc, 0.5, rank)


async def stream_llm_generation(
    llm: Any,
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    chunk_size: int = 10,
    timeout_seconds: float = 120.0,
) -> AsyncGenerator[str, None]:
    """Stream LLM generation, handling both streaming and non-streaming providers.

    Args:
        llm: LLM provider instance
        prompt: Generation prompt
        max_tokens: Maximum tokens to generate
        temperature: Generation temperature
        chunk_size: Minimum characters per chunk (for non-streaming)
        timeout_seconds: Generation timeout

    Yields:
        Text chunks
    """
    try:
        # Check for streaming support
        if hasattr(llm, 'generate_stream'):
            async for chunk in asyncio.wait_for(
                _consume_stream(llm.generate_stream(prompt, max_tokens=max_tokens, temperature=temperature)),
                timeout=timeout_seconds
            ):
                yield chunk
        else:
            # Non-streaming fallback: generate full response and chunk it
            response = await asyncio.wait_for(
                llm.generate(prompt, max_tokens=max_tokens, temperature=temperature),
                timeout=timeout_seconds
            )

            # Simulate streaming by yielding chunks
            for i in range(0, len(response), chunk_size):
                yield response[i:i + chunk_size]
                await asyncio.sleep(0)  # Yield control

    except asyncio.TimeoutError:
        logger.error("LLM generation timed out")
        raise
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        raise


async def _consume_stream(stream: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    """Helper to consume an async generator with proper error handling."""
    async for item in stream:
        yield item


# =============================================================================
# PIPELINE INTEGRATION MIXIN
# =============================================================================

class StreamingPipelineMixin:
    """
    Mixin to add streaming support to RAGPipeline.

    Add this mixin to RAGPipeline to enable run_streaming():

        class RAGPipeline(StreamingPipelineMixin):
            ...

    Then use:
        async for event in pipeline.run_streaming("query"):
            ...
    """

    async def run_streaming(
        self,
        query: str,
        streaming_config: Optional[StreamingConfig] = None,
        cancellation_token: Optional[CancellationToken] = None,
        **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute RAG pipeline with streaming response.

        Args:
            query: User query
            streaming_config: Streaming configuration
            cancellation_token: Token for early termination
            **kwargs: Additional arguments passed to run()

        Yields:
            StreamEvent objects for each pipeline event
        """
        config = streaming_config or StreamingConfig()
        response = StreamingRAGResponse(config, cancellation_token)

        # Start streaming
        yield response.start()

        try:
            # Check for cancellation at each stage
            if response.cancellation_token.is_cancelled:
                yield response.cancel(response.cancellation_token.reason or "Cancelled")
                return

            # Stage 1: Query Classification/Rewrite
            yield response.start_stage(StreamingStage.QUERY_REWRITE)

            query_type = None
            if hasattr(self, 'classifier'):
                query_type = await self.classifier.classify(query)

            queries = [query]
            if hasattr(self, 'config') and getattr(self.config, 'enable_query_rewrite', True):
                if hasattr(self, 'rewriter'):
                    queries = await self.rewriter.rewrite(query, query_type)

            yield response.complete_stage(
                StreamingStage.QUERY_REWRITE,
                {"original_query": query, "rewritten_queries": queries}
            )

            # Stage 2: Retrieval (streaming)
            if response.cancellation_token.is_cancelled:
                yield response.cancel(response.cancellation_token.reason or "Cancelled")
                return

            retrievers = getattr(self, 'retrievers', [])
            if retrievers:
                sources = [getattr(r, 'name', type(r).__name__) for r in retrievers]
                yield response.start_retrieval(sources)

                all_documents = []
                timeout = getattr(config, 'retrieval_timeout_seconds', 30.0)

                async for source, docs in stream_multi_source_retrieval(
                    retrievers, query,
                    top_k=getattr(getattr(self, 'config', None), 'top_k_retrieve', 10),
                    timeout_seconds=timeout
                ):
                    all_documents.extend(docs)
                    yield response.emit_retrieval_batch(docs, source)

                    # Progress update
                    response.emit_progress(
                        StreamingStage.RETRIEVAL,
                        percent=min(100, len(all_documents) * 10),
                        message=f"Retrieved {len(all_documents)} documents",
                        items_processed=len(all_documents),
                    )

                yield response.complete_retrieval(len(all_documents))
            else:
                all_documents = []

            if not all_documents:
                yield response.complete(
                    "I could not find relevant information to answer your question.",
                    confidence=0.0
                )
                return

            # Stage 3: Reranking (streaming top-first)
            if response.cancellation_token.is_cancelled:
                yield response.cancel(response.cancellation_token.reason or "Cancelled")
                return

            reranker = getattr(self, 'reranker', None)
            reranked_docs = []

            if reranker and getattr(getattr(self, 'config', None), 'enable_reranking', True):
                # Convert to proper format for reranking
                doc_dicts = []
                for doc in all_documents:
                    if isinstance(doc, dict):
                        doc_dicts.append(doc)
                    elif hasattr(doc, 'content'):
                        doc_dicts.append({
                            'content': doc.content,
                            'metadata': getattr(doc, 'metadata', {}),
                            'score': getattr(doc, 'score', 0.5),
                        })
                    else:
                        doc_dicts.append({'content': str(doc)})

                yield response.start_reranking(len(doc_dicts))

                top_k = getattr(getattr(self, 'config', None), 'top_k_final', 5)

                async for doc, score, rank in stream_reranking_top_first(
                    doc_dicts, query, reranker, top_k=top_k
                ):
                    reranked_docs.append(doc)
                    yield response.emit_rerank_result(doc, score, rank)

                yield response.complete_reranking(reranked_docs)
            else:
                reranked_docs = all_documents[:getattr(getattr(self, 'config', None), 'top_k_final', 5)]

            # Stage 4: Generation (streaming)
            if response.cancellation_token.is_cancelled:
                yield response.cancel(response.cancellation_token.reason or "Cancelled")
                return

            contexts = [
                doc.get('content', str(doc)) if isinstance(doc, dict)
                else getattr(doc, 'content', str(doc))
                for doc in reranked_docs
            ]

            yield response.start_generation(contexts)

            # Build prompt
            context_text = "\n\n---\n\n".join(contexts) if contexts else "No relevant context available."
            prompt = f"""Answer the following question using the provided context.
Be accurate, comprehensive, and cite the context when relevant.
If the context is insufficient, acknowledge the limitations.

Question: {query}

Context:
{context_text}

Answer:"""

            llm = getattr(self, 'llm', None)
            if llm:
                gen_config = getattr(self, 'config', None)
                max_tokens = kwargs.get('max_tokens', getattr(gen_config, 'max_tokens', 2048))
                temperature = kwargs.get('temperature', getattr(gen_config, 'temperature', 0.7))

                full_response = ""
                async for chunk in stream_llm_generation(
                    llm, prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    chunk_size=config.chunk_size,
                    timeout_seconds=config.generation_timeout_seconds,
                ):
                    full_response += chunk
                    yield response.emit_generation_chunk(chunk)

                confidence = 0.7 if contexts else 0.3
                yield response.complete_generation(full_response, confidence)
                yield response.complete(full_response, confidence)
            else:
                yield response.error("No LLM provider configured")

        except CancellationError as e:
            yield response.cancel(str(e))

        except asyncio.TimeoutError:
            yield response.error("Pipeline timeout")

        except Exception as e:
            logger.error(f"Streaming pipeline error: {e}", exc_info=True)
            yield response.error(str(e))


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_streaming_config(
    stream_all: bool = True,
    chunk_size: int = 10,
    progress_callback: Optional[Callable[[StreamProgress], None]] = None,
    **kwargs
) -> StreamingConfig:
    """Create a streaming configuration.

    Args:
        stream_all: Enable all streaming features
        chunk_size: Characters per generation chunk
        progress_callback: Callback for progress updates
        **kwargs: Additional configuration options

    Returns:
        StreamingConfig instance
    """
    return StreamingConfig(
        stream_retrieval=kwargs.get('stream_retrieval', stream_all),
        stream_reranking=kwargs.get('stream_reranking', stream_all),
        stream_generation=kwargs.get('stream_generation', stream_all),
        chunk_size=chunk_size,
        progress_callback=progress_callback,
        **{k: v for k, v in kwargs.items() if k not in ['stream_retrieval', 'stream_reranking', 'stream_generation']}
    )


def create_cancellation_token() -> CancellationToken:
    """Create a new cancellation token.

    Returns:
        CancellationToken instance
    """
    return CancellationToken()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    "StreamingRAGResponse",
    "StreamingConfig",
    "StreamProgress",
    "StreamEvent",
    # Enums
    "StreamEventType",
    "StreamingStage",
    # Cancellation
    "CancellationToken",
    "CancellationError",
    # Protocols
    "StreamingLLMProvider",
    "StreamingRetrieverProtocol",
    # Generators
    "stream_multi_source_retrieval",
    "stream_reranking_top_first",
    "stream_llm_generation",
    # Mixin
    "StreamingPipelineMixin",
    # Factory functions
    "create_streaming_config",
    "create_cancellation_token",
]
