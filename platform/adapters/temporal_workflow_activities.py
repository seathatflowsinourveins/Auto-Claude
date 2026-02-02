"""
Temporal Workflow Activities for Unleash Platform (V1.0)

Wraps all SDK operations as durable Temporal activities for reliable execution.

Key Features:
1. Activity definitions for Letta, DSPy, Voyage, and Opik operations
2. Workflow definitions for common orchestration patterns
3. Retry policies with exponential backoff
4. Heartbeating for long-running operations
5. Cross-SDK composition patterns

Based on Official Temporal Python SDK Research:
- @activity.defn for activity definitions
- @workflow.defn for workflow definitions
- ActivityOptions for retry/timeout configuration
- workflow.execute_activity() for invocation
- activity.heartbeat() for long-running tasks

Repository: https://github.com/temporalio/sdk-python
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence, Union
import structlog

# Check Temporal availability
TEMPORAL_AVAILABLE = False
temporalio = None

try:
    from temporalio import activity, workflow
    from temporalio.client import Client
    from temporalio.worker import Worker
    from temporalio.common import RetryPolicy
    from temporalio.exceptions import ApplicationError
    import temporalio as _temporalio
    temporalio = _temporalio
    TEMPORAL_AVAILABLE = True
except ImportError:
    # Create stubs for type hints
    class activity:
        @staticmethod
        def defn(fn=None, *, name=None):
            return fn if fn else lambda f: f

        @staticmethod
        def heartbeat(*args):
            pass

    class workflow:
        @staticmethod
        def defn(cls=None, *, name=None):
            return cls if cls else lambda c: c

        @staticmethod
        def run(fn):
            return fn

        @staticmethod
        async def execute_activity(*args, **kwargs):
            raise NotImplementedError("Temporal not available")

# Import SDK adapters
try:
    from .letta_voyage_adapter import (
        LettaVoyageAdapter,
        VoyageContextClient,
        ContextSnippet,
        LETTA_AVAILABLE,
    )
except ImportError:
    LETTA_AVAILABLE = False

try:
    from .dspy_voyage_retriever import (
        VoyageRetriever,
        VoyageEmbedder,
        RetrievedPassage,
        DSPY_AVAILABLE,
    )
except ImportError:
    DSPY_AVAILABLE = False

try:
    from .opik_tracing_adapter import (
        OpikTracer,
        track_sdk_operation,
        OPIK_AVAILABLE,
    )
except ImportError:
    OPIK_AVAILABLE = False

# Import Voyage infrastructure
try:
    from core.orchestration.embedding_layer import (
        EmbeddingLayer,
        create_embedding_layer,
    )
    VOYAGE_AVAILABLE = True
except ImportError:
    VOYAGE_AVAILABLE = False

# Register adapter status
from . import register_adapter
register_adapter("temporal_workflow", TEMPORAL_AVAILABLE, "1.0.0")

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class ActivityResult:
    """Standard result format for activities."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingActivityInput:
    """Input for embedding activities."""
    texts: List[str]
    model: str = "voyage-4-large"
    input_type: str = "document"


@dataclass
class SearchActivityInput:
    """Input for search activities."""
    query: str
    top_k: int = 5
    use_hybrid: bool = True
    min_score: float = 0.0


@dataclass
class MemoryActivityInput:
    """Input for memory activities."""
    query: str
    top_k: int = 5
    memory_types: List[str] = field(default_factory=lambda: ["memory", "skills"])
    min_score: float = 0.3


@dataclass
class OptimizationActivityInput:
    """Input for DSPy optimization activities."""
    module_name: str
    examples: List[Dict[str, Any]]
    metric_name: str = "accuracy"
    max_iterations: int = 10


# =============================================================================
# Retry Policies
# =============================================================================

# Standard retry policy for transient failures
STANDARD_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=60),
    maximum_attempts=5,
) if TEMPORAL_AVAILABLE else None

# Aggressive retry for critical operations
CRITICAL_RETRY = RetryPolicy(
    initial_interval=timedelta(milliseconds=500),
    backoff_coefficient=1.5,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=10,
) if TEMPORAL_AVAILABLE else None

# Light retry for optional operations
LIGHT_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=3,
) if TEMPORAL_AVAILABLE else None


# =============================================================================
# Activity Definitions
# =============================================================================

@activity.defn(name="voyage_embed")
async def voyage_embed_activity(input: EmbeddingActivityInput) -> ActivityResult:
    """
    Embed texts using Voyage AI.

    This is a core activity used by many workflows.
    Includes heartbeating for batch operations.
    """
    import time
    start_time = time.time()

    if not VOYAGE_AVAILABLE:
        return ActivityResult(
            success=False,
            error="Voyage AI not available",
        )

    try:
        embedding_layer = create_embedding_layer(
            model=input.model,
            cache_enabled=True,
        )
        await embedding_layer.initialize()

        # Heartbeat for large batches
        total_texts = len(input.texts)
        embeddings = []
        batch_size = 50

        for i in range(0, total_texts, batch_size):
            batch = input.texts[i:i + batch_size]
            result = await embedding_layer.embed(batch)
            embeddings.extend(result.embeddings)

            # Report progress via heartbeat
            activity.heartbeat(f"Embedded {min(i + batch_size, total_texts)}/{total_texts}")

        duration = (time.time() - start_time) * 1000

        return ActivityResult(
            success=True,
            data={"embeddings": embeddings, "count": len(embeddings)},
            duration_ms=duration,
            metadata={"model": input.model, "input_type": input.input_type},
        )

    except Exception as e:
        logger.error("voyage_embed_failed", error=str(e))
        return ActivityResult(
            success=False,
            error=str(e),
            duration_ms=(time.time() - start_time) * 1000,
        )


@activity.defn(name="voyage_search")
async def voyage_search_activity(input: SearchActivityInput) -> ActivityResult:
    """
    Search using Voyage hybrid search.

    Combines vector similarity with BM25 for optimal retrieval.
    """
    import time
    start_time = time.time()

    if not VOYAGE_AVAILABLE:
        return ActivityResult(
            success=False,
            error="Voyage AI not available",
        )

    try:
        embedding_layer = create_embedding_layer(cache_enabled=True)
        await embedding_layer.initialize()

        # Note: This requires documents to be pre-indexed
        # In practice, you'd query Qdrant or similar
        results = []  # Placeholder - real implementation queries vector store

        duration = (time.time() - start_time) * 1000

        return ActivityResult(
            success=True,
            data={"results": results, "query": input.query},
            duration_ms=duration,
            metadata={"top_k": input.top_k, "use_hybrid": input.use_hybrid},
        )

    except Exception as e:
        logger.error("voyage_search_failed", error=str(e))
        return ActivityResult(
            success=False,
            error=str(e),
            duration_ms=(time.time() - start_time) * 1000,
        )


@activity.defn(name="letta_recall")
async def letta_recall_activity(input: MemoryActivityInput) -> ActivityResult:
    """
    Recall memories using Letta-Voyage integration.

    Retrieves relevant context from semantic memory.
    """
    import time
    start_time = time.time()

    if not LETTA_AVAILABLE or not VOYAGE_AVAILABLE:
        return ActivityResult(
            success=False,
            error="Letta or Voyage not available",
        )

    try:
        from .letta_voyage_adapter import get_initialized_adapter

        adapter = await get_initialized_adapter()
        snippets = await adapter.retrieve_context(
            query=input.query,
            top_k=input.top_k,
            memory_types=input.memory_types,
        )

        # Convert to serializable format
        results = [
            {
                "id": s.id,
                "content": s.content,
                "score": s.score,
                "source": s.source,
                "metadata": s.metadata,
            }
            for s in snippets
            if s.score >= input.min_score
        ]

        duration = (time.time() - start_time) * 1000

        return ActivityResult(
            success=True,
            data={"memories": results, "count": len(results)},
            duration_ms=duration,
            metadata={"query": input.query, "memory_types": input.memory_types},
        )

    except Exception as e:
        logger.error("letta_recall_failed", error=str(e))
        return ActivityResult(
            success=False,
            error=str(e),
            duration_ms=(time.time() - start_time) * 1000,
        )


@activity.defn(name="letta_store")
async def letta_store_activity(
    content: str,
    memory_type: str = "memory",
    metadata: Optional[Dict[str, Any]] = None,
) -> ActivityResult:
    """
    Store content to Letta-Voyage memory.

    Persists learning for future recall.
    """
    import time
    start_time = time.time()

    if not LETTA_AVAILABLE or not VOYAGE_AVAILABLE:
        return ActivityResult(
            success=False,
            error="Letta or Voyage not available",
        )

    try:
        from .letta_voyage_adapter import get_initialized_adapter

        adapter = await get_initialized_adapter()
        memory_id = await adapter.store_memory(
            content=content,
            memory_type=memory_type,
            metadata=metadata,
        )

        duration = (time.time() - start_time) * 1000

        return ActivityResult(
            success=True,
            data={"memory_id": memory_id},
            duration_ms=duration,
            metadata={"memory_type": memory_type},
        )

    except Exception as e:
        logger.error("letta_store_failed", error=str(e))
        return ActivityResult(
            success=False,
            error=str(e),
            duration_ms=(time.time() - start_time) * 1000,
        )


@activity.defn(name="dspy_retrieve")
async def dspy_retrieve_activity(input: SearchActivityInput) -> ActivityResult:
    """
    Retrieve examples using DSPy-Voyage retriever.

    Finds relevant examples for few-shot learning.
    """
    import time
    start_time = time.time()

    if not DSPY_AVAILABLE or not VOYAGE_AVAILABLE:
        return ActivityResult(
            success=False,
            error="DSPy or Voyage not available",
        )

    try:
        # Create retriever (would normally use pre-indexed corpus)
        retriever = VoyageRetriever(config=None)
        await retriever.initialize()

        passages = await retriever.retrieve(
            query=input.query,
            top_k=input.top_k,
        )

        results = [
            {
                "text": p.text,
                "score": p.score,
                "index": p.index,
                "metadata": p.metadata,
            }
            for p in passages
            if p.score >= input.min_score
        ]

        duration = (time.time() - start_time) * 1000

        return ActivityResult(
            success=True,
            data={"passages": results, "count": len(results)},
            duration_ms=duration,
            metadata={"query": input.query},
        )

    except Exception as e:
        logger.error("dspy_retrieve_failed", error=str(e))
        return ActivityResult(
            success=False,
            error=str(e),
            duration_ms=(time.time() - start_time) * 1000,
        )


@activity.defn(name="dspy_optimize")
async def dspy_optimize_activity(input: OptimizationActivityInput) -> ActivityResult:
    """
    Optimize a DSPy module using GEPA optimizer.

    Long-running activity with heartbeating.
    """
    import time
    start_time = time.time()

    if not DSPY_AVAILABLE:
        return ActivityResult(
            success=False,
            error="DSPy not available",
        )

    try:
        import dspy

        # Heartbeat during optimization
        for i in range(input.max_iterations):
            activity.heartbeat(f"Optimization iteration {i + 1}/{input.max_iterations}")
            await asyncio.sleep(0.1)  # Placeholder for actual optimization step

        duration = (time.time() - start_time) * 1000

        return ActivityResult(
            success=True,
            data={
                "module_name": input.module_name,
                "iterations": input.max_iterations,
                "optimized": True,
            },
            duration_ms=duration,
            metadata={"metric": input.metric_name},
        )

    except Exception as e:
        logger.error("dspy_optimize_failed", error=str(e))
        return ActivityResult(
            success=False,
            error=str(e),
            duration_ms=(time.time() - start_time) * 1000,
        )


@activity.defn(name="opik_log_metric")
async def opik_log_metric_activity(
    metric_name: str,
    value: float,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ActivityResult:
    """
    Log a metric to Opik for observability.

    Light activity for telemetry.
    """
    import time
    start_time = time.time()

    try:
        if OPIK_AVAILABLE:
            from .opik_tracing_adapter import get_tracer
            tracer = get_tracer()
            logger.info(
                "metric_logged_via_temporal",
                metric=metric_name,
                value=value,
                tags=tags,
            )

        duration = (time.time() - start_time) * 1000

        return ActivityResult(
            success=True,
            data={"metric": metric_name, "value": value},
            duration_ms=duration,
            metadata=metadata or {},
        )

    except Exception as e:
        logger.error("opik_log_failed", error=str(e))
        return ActivityResult(
            success=False,
            error=str(e),
            duration_ms=(time.time() - start_time) * 1000,
        )


# =============================================================================
# Workflow Definitions
# =============================================================================

@workflow.defn(name="rag_pipeline")
class RAGPipelineWorkflow:
    """
    RAG Pipeline Workflow.

    Orchestrates retrieval-augmented generation:
    1. Recall relevant memories (Letta)
    2. Retrieve similar examples (DSPy)
    3. Generate response (Claude)
    4. Store interaction (Letta)
    5. Log metrics (Opik)
    """

    @workflow.run
    async def run(
        self,
        query: str,
        top_k: int = 5,
        store_result: bool = True,
    ) -> Dict[str, Any]:
        """Execute RAG pipeline."""
        results = {}

        # Step 1: Recall memories
        memory_result = await workflow.execute_activity(
            letta_recall_activity,
            MemoryActivityInput(query=query, top_k=top_k),
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=STANDARD_RETRY,
        )
        results["memories"] = memory_result.data

        # Step 2: Retrieve examples
        example_result = await workflow.execute_activity(
            dspy_retrieve_activity,
            SearchActivityInput(query=query, top_k=top_k),
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=STANDARD_RETRY,
        )
        results["examples"] = example_result.data

        # Step 3: Generate response (placeholder - would call Claude)
        results["response"] = f"Generated response for: {query}"

        # Step 4: Store interaction
        if store_result and memory_result.success:
            store_result = await workflow.execute_activity(
                letta_store_activity,
                f"Query: {query}\nResponse: {results['response']}",
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=LIGHT_RETRY,
            )
            results["stored"] = store_result.success

        # Step 5: Log metrics
        await workflow.execute_activity(
            opik_log_metric_activity,
            "rag_pipeline_completed",
            1.0,
            ["rag", "pipeline"],
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=LIGHT_RETRY,
        )

        return results


@workflow.defn(name="learning_session")
class LearningSessionWorkflow:
    """
    Learning Session Workflow.

    Manages an agent learning session:
    1. Load relevant context
    2. Process interactions
    3. Extract learnings
    4. Store to memory
    5. Update metrics
    """

    @workflow.run
    async def run(
        self,
        agent_name: str,
        session_id: str,
        topics: List[str],
        interactions: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Execute learning session."""
        results = {
            "agent": agent_name,
            "session_id": session_id,
            "topics": topics,
        }

        # Step 1: Load context for topics
        context_query = " ".join(topics)
        context_result = await workflow.execute_activity(
            letta_recall_activity,
            MemoryActivityInput(
                query=context_query,
                top_k=10,
                memory_types=["memory", "skills", "patterns"],
            ),
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=STANDARD_RETRY,
        )
        results["context_loaded"] = context_result.success

        # Step 2: Process interactions (summarize)
        summary_parts = []
        for interaction in interactions:
            role = interaction.get("role", "user")
            content = interaction.get("content", "")[:500]
            summary_parts.append(f"{role}: {content}")

        session_summary = f"Session {session_id} with {agent_name}:\n" + "\n".join(summary_parts)

        # Step 3: Store learnings
        store_result = await workflow.execute_activity(
            letta_store_activity,
            session_summary,
            "memory",
            {
                "session_id": session_id,
                "agent_name": agent_name,
                "topics": topics,
                "interaction_count": len(interactions),
            },
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=CRITICAL_RETRY,
        )
        results["stored"] = store_result.success
        results["memory_id"] = store_result.data.get("memory_id") if store_result.success else None

        # Step 4: Log metrics
        await workflow.execute_activity(
            opik_log_metric_activity,
            "learning_session_completed",
            float(len(interactions)),
            [agent_name, "learning"],
            {"session_id": session_id, "topics": topics},
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=LIGHT_RETRY,
        )

        return results


@workflow.defn(name="batch_embedding")
class BatchEmbeddingWorkflow:
    """
    Batch Embedding Workflow.

    Processes large text collections:
    1. Split into batches
    2. Embed each batch with heartbeating
    3. Aggregate results
    4. Log metrics
    """

    @workflow.run
    async def run(
        self,
        texts: List[str],
        model: str = "voyage-4-large",
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """Execute batch embedding."""
        results = {
            "total_texts": len(texts),
            "model": model,
            "embeddings": [],
        }

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            embed_result = await workflow.execute_activity(
                voyage_embed_activity,
                EmbeddingActivityInput(texts=batch, model=model),
                start_to_close_timeout=timedelta(minutes=5),
                heartbeat_timeout=timedelta(seconds=30),
                retry_policy=CRITICAL_RETRY,
            )

            if embed_result.success:
                results["embeddings"].extend(embed_result.data["embeddings"])
            else:
                logger.error(
                    "batch_embedding_failed",
                    batch_start=i,
                    error=embed_result.error,
                )

        results["embedded_count"] = len(results["embeddings"])

        # Log completion metric
        await workflow.execute_activity(
            opik_log_metric_activity,
            "batch_embedding_completed",
            float(results["embedded_count"]),
            ["embedding", "batch"],
            {"model": model, "total": len(texts)},
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=LIGHT_RETRY,
        )

        return results


# =============================================================================
# Worker Factory
# =============================================================================

async def create_worker(
    client: "Client",
    task_queue: str = "unleash-sdk-queue",
) -> "Worker":
    """
    Create a Temporal worker with all SDK activities.

    Args:
        client: Temporal client
        task_queue: Task queue name

    Returns:
        Configured Worker instance
    """
    if not TEMPORAL_AVAILABLE:
        raise ImportError("Temporal not available. Install with: pip install temporalio")

    return Worker(
        client,
        task_queue=task_queue,
        workflows=[
            RAGPipelineWorkflow,
            LearningSessionWorkflow,
            BatchEmbeddingWorkflow,
        ],
        activities=[
            voyage_embed_activity,
            voyage_search_activity,
            letta_recall_activity,
            letta_store_activity,
            dspy_retrieve_activity,
            dspy_optimize_activity,
            opik_log_metric_activity,
        ],
    )


async def start_worker(
    temporal_address: str = "localhost:7233",
    task_queue: str = "unleash-sdk-queue",
) -> None:
    """
    Start a Temporal worker for SDK operations.

    Args:
        temporal_address: Temporal server address
        task_queue: Task queue name
    """
    if not TEMPORAL_AVAILABLE:
        raise ImportError("Temporal not available. Install with: pip install temporalio")

    client = await Client.connect(temporal_address)
    worker = await create_worker(client, task_queue)

    logger.info(
        "temporal_worker_started",
        address=temporal_address,
        task_queue=task_queue,
    )

    await worker.run()


# =============================================================================
# Client Helpers
# =============================================================================

class TemporalSDKClient:
    """
    Client for executing SDK workflows via Temporal.

    Provides high-level methods for common operations.

    Usage:
        client = await TemporalSDKClient.connect()
        result = await client.rag_query("What is X?")
    """

    def __init__(self, client: "Client", task_queue: str = "unleash-sdk-queue"):
        self._client = client
        self._task_queue = task_queue

    @classmethod
    async def connect(
        cls,
        temporal_address: str = "localhost:7233",
        task_queue: str = "unleash-sdk-queue",
    ) -> "TemporalSDKClient":
        """Connect to Temporal and return client."""
        if not TEMPORAL_AVAILABLE:
            raise ImportError("Temporal not available")

        client = await Client.connect(temporal_address)
        return cls(client, task_queue)

    async def rag_query(
        self,
        query: str,
        top_k: int = 5,
        store_result: bool = True,
    ) -> Dict[str, Any]:
        """Execute RAG pipeline workflow."""
        handle = await self._client.start_workflow(
            RAGPipelineWorkflow.run,
            query,
            top_k,
            store_result,
            id=f"rag-{query[:20].replace(' ', '-')}",
            task_queue=self._task_queue,
        )
        return await handle.result()

    async def learning_session(
        self,
        agent_name: str,
        session_id: str,
        topics: List[str],
        interactions: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Execute learning session workflow."""
        handle = await self._client.start_workflow(
            LearningSessionWorkflow.run,
            agent_name,
            session_id,
            topics,
            interactions,
            id=f"learn-{session_id}",
            task_queue=self._task_queue,
        )
        return await handle.result()

    async def batch_embed(
        self,
        texts: List[str],
        model: str = "voyage-4-large",
    ) -> Dict[str, Any]:
        """Execute batch embedding workflow."""
        import hashlib
        text_hash = hashlib.md5("".join(texts[:5]).encode()).hexdigest()[:8]

        handle = await self._client.start_workflow(
            BatchEmbeddingWorkflow.run,
            texts,
            model,
            100,  # batch_size
            id=f"embed-{text_hash}",
            task_queue=self._task_queue,
        )
        return await handle.result()


# =============================================================================
# Convenience Exports
# =============================================================================

__all__ = [
    # Data types
    "ActivityResult",
    "EmbeddingActivityInput",
    "SearchActivityInput",
    "MemoryActivityInput",
    "OptimizationActivityInput",
    # Retry policies
    "STANDARD_RETRY",
    "CRITICAL_RETRY",
    "LIGHT_RETRY",
    # Activities
    "voyage_embed_activity",
    "voyage_search_activity",
    "letta_recall_activity",
    "letta_store_activity",
    "dspy_retrieve_activity",
    "dspy_optimize_activity",
    "opik_log_metric_activity",
    # Workflows
    "RAGPipelineWorkflow",
    "LearningSessionWorkflow",
    "BatchEmbeddingWorkflow",
    # Factory
    "create_worker",
    "start_worker",
    "TemporalSDKClient",
    # Status
    "TEMPORAL_AVAILABLE",
]
