"""
Async Letta Client Wrapper

Provides async/await interface for the synchronous Letta client.
Uses ThreadPoolExecutor to run sync methods in async context without blocking.

Features:
- AsyncLettaClient class wrapping synchronous Letta client
- Thread pool executor for non-blocking async execution
- Async versions of all key methods
- Proper cleanup and context management
- Connection pooling via executor reuse

Usage:
    async with AsyncLettaClient(api_key="...") as client:
        agent = await client.create_agent(name="my-agent")
        response = await client.send_message(agent.id, "Hello!")
        memory = await client.get_memory(agent.id)
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, TypeVar, Callable, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class LettaAgentInfo:
    """Information about a Letta agent."""
    id: str
    name: str
    model: Optional[str] = None
    embedding_model: Optional[str] = None
    system_prompt: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LettaMessage:
    """A message in a Letta conversation."""
    role: str
    content: str
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LettaMemoryBlock:
    """A memory block from Letta."""
    id: str
    content: str
    label: Optional[str] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LettaResponse:
    """Response from a Letta message operation."""
    messages: List[LettaMessage]
    usage: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def last_message(self) -> Optional[str]:
        """Get the last assistant message content."""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return self.messages[-1].content if self.messages else None


class AsyncLettaClient:
    """
    Async wrapper for the synchronous Letta client.

    This class wraps the synchronous Letta client and provides async/await
    methods by running sync operations in a ThreadPoolExecutor.

    Attributes:
        api_key: Letta API key
        base_url: Letta API base URL
        max_workers: Maximum number of thread pool workers

    Example:
        async with AsyncLettaClient(api_key="key") as client:
            agents = await client.list_agents()
            for agent in agents:
                print(f"Agent: {agent.name}")
    """

    DEFAULT_BASE_URL = "https://api.letta.com"
    DEFAULT_MAX_WORKERS = 4

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        executor: Optional[ThreadPoolExecutor] = None,
    ):
        """
        Initialize the async Letta client.

        Args:
            api_key: Letta API key (falls back to LETTA_API_KEY env var)
            base_url: API base URL (falls back to LETTA_BASE_URL env var or default)
            max_workers: Maximum thread pool workers for concurrent operations
            executor: Optional external executor for connection pooling
        """
        self.api_key = api_key or os.environ.get("LETTA_API_KEY")
        self.base_url = base_url or os.environ.get("LETTA_BASE_URL", self.DEFAULT_BASE_URL)
        self.max_workers = max_workers

        self._client = None
        self._executor = executor
        self._owns_executor = executor is None
        self._initialized = False
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "AsyncLettaClient":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with cleanup."""
        await self.close()

    async def initialize(self) -> None:
        """
        Initialize the Letta client and thread pool executor.

        Raises:
            ImportError: If letta-client is not installed
            ValueError: If API key is not configured
            ConnectionError: If unable to connect to Letta API
        """
        async with self._lock:
            if self._initialized:
                return

            if not self.api_key:
                raise ValueError(
                    "LETTA_API_KEY not configured. "
                    "Set it via constructor argument or LETTA_API_KEY environment variable."
                )

            try:
                from letta_client import Letta
            except ImportError as e:
                raise ImportError(
                    "letta-client not installed. Run: pip install letta-client"
                ) from e

            # Create executor if we don't have one
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix="letta-async-"
                )

            # Initialize sync client
            self._client = Letta(api_key=self.api_key, base_url=self.base_url)

            # Verify connection
            try:
                await self._run_in_executor(self._client.agents.list, limit=1)
            except Exception as e:
                await self.close()
                raise ConnectionError(f"Failed to connect to Letta API: {e}") from e

            self._initialized = True
            logger.info(f"AsyncLettaClient initialized (base_url={self.base_url})")

    async def close(self) -> None:
        """
        Close the client and release resources.

        This method is idempotent and safe to call multiple times.
        """
        async with self._lock:
            if self._owns_executor and self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None

            self._client = None
            self._initialized = False
            logger.info("AsyncLettaClient closed")

    async def _run_in_executor(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """
        Run a synchronous function in the thread pool executor.

        Args:
            func: Synchronous function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The function's return value

        Raises:
            RuntimeError: If client is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Call initialize() first or use async context manager.")

        loop = asyncio.get_running_loop()

        if kwargs:
            func = partial(func, **kwargs)

        return await loop.run_in_executor(self._executor, func, *args)

    def _ensure_initialized(self) -> None:
        """Ensure client is initialized."""
        if not self._initialized or self._client is None:
            raise RuntimeError(
                "Client not initialized. Call initialize() first or use async context manager."
            )

    # -------------------------------------------------------------------------
    # Agent Management
    # -------------------------------------------------------------------------

    async def create_agent(
        self,
        name: str,
        model: str = "claude-3-5-sonnet-20241022",
        embedding_model: str = "text-embedding-3-small",
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LettaAgentInfo:
        """
        Create a new Letta agent.

        Args:
            name: Agent name
            model: LLM model to use
            embedding_model: Embedding model for memory
            system_prompt: Optional system prompt
            metadata: Optional metadata dictionary

        Returns:
            LettaAgentInfo with the created agent details

        Raises:
            RuntimeError: If client not initialized
            Exception: If agent creation fails
        """
        self._ensure_initialized()

        create_kwargs = {
            "name": name,
            "model": model,
            "embedding": embedding_model,
        }
        if system_prompt:
            create_kwargs["system"] = system_prompt

        agent = await self._run_in_executor(
            self._client.agents.create,
            **create_kwargs
        )

        return LettaAgentInfo(
            id=agent.id,
            name=agent.name,
            model=model,
            embedding_model=embedding_model,
            system_prompt=system_prompt,
            created_at=getattr(agent, "created_at", None),
            metadata=metadata or {},
        )

    async def get_agent(self, agent_id: str) -> LettaAgentInfo:
        """
        Get agent details by ID.

        Args:
            agent_id: The agent ID

        Returns:
            LettaAgentInfo with agent details

        Raises:
            RuntimeError: If client not initialized
            Exception: If agent not found
        """
        self._ensure_initialized()

        agent = await self._run_in_executor(
            self._client.agents.get,
            agent_id=agent_id
        )

        return LettaAgentInfo(
            id=agent.id,
            name=agent.name,
            model=getattr(agent, "model", None),
            created_at=getattr(agent, "created_at", None),
        )

    async def list_agents(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[LettaAgentInfo]:
        """
        List all agents.

        Args:
            limit: Maximum number of agents to return
            offset: Number of agents to skip

        Returns:
            List of LettaAgentInfo objects

        Raises:
            RuntimeError: If client not initialized
        """
        self._ensure_initialized()

        agents = await self._run_in_executor(
            self._client.agents.list,
            limit=limit
        )

        return [
            LettaAgentInfo(
                id=a.id,
                name=a.name,
                model=getattr(a, "model", None),
                created_at=getattr(a, "created_at", None),
            )
            for a in agents
        ]

    async def delete_agent(self, agent_id: str) -> bool:
        """
        Delete an agent.

        Args:
            agent_id: The agent ID to delete

        Returns:
            True if deletion was successful

        Raises:
            RuntimeError: If client not initialized
            Exception: If deletion fails
        """
        self._ensure_initialized()

        await self._run_in_executor(
            self._client.agents.delete,
            agent_id=agent_id
        )

        return True

    # -------------------------------------------------------------------------
    # Messaging
    # -------------------------------------------------------------------------

    async def send_message(
        self,
        agent_id: str,
        content: str,
        role: str = "user",
    ) -> LettaResponse:
        """
        Send a message to an agent.

        Args:
            agent_id: The agent ID
            content: Message content
            role: Message role (default: "user")

        Returns:
            LettaResponse with agent's response

        Raises:
            RuntimeError: If client not initialized
            Exception: If message fails
        """
        self._ensure_initialized()

        response = await self._run_in_executor(
            self._client.agents.messages.create,
            agent_id=agent_id,
            messages=[{"role": role, "content": content}]
        )

        # Parse response messages
        messages = []
        for msg in response.messages:
            if hasattr(msg, "assistant_message") and msg.assistant_message:
                messages.append(LettaMessage(
                    role="assistant",
                    content=msg.assistant_message,
                    metadata={"raw_type": "assistant_message"}
                ))
            elif hasattr(msg, "content"):
                messages.append(LettaMessage(
                    role=getattr(msg, "role", "assistant"),
                    content=msg.content,
                ))
            elif hasattr(msg, "text"):
                messages.append(LettaMessage(
                    role="assistant",
                    content=msg.text,
                ))

        return LettaResponse(
            messages=messages,
            usage=getattr(response, "usage", None),
        )

    # -------------------------------------------------------------------------
    # Memory Operations
    # -------------------------------------------------------------------------

    async def get_memory(
        self,
        agent_id: str,
    ) -> Dict[str, Any]:
        """
        Get agent's memory state.

        Args:
            agent_id: The agent ID

        Returns:
            Dictionary containing memory state

        Raises:
            RuntimeError: If client not initialized
            Exception: If retrieval fails
        """
        self._ensure_initialized()

        # Get agent to access memory
        agent = await self._run_in_executor(
            self._client.agents.get,
            agent_id=agent_id
        )

        memory_state = {}

        # Get core memory if available
        if hasattr(agent, "memory"):
            memory = agent.memory
            if hasattr(memory, "blocks"):
                memory_state["blocks"] = [
                    {
                        "label": getattr(b, "label", None),
                        "value": getattr(b, "value", ""),
                        "limit": getattr(b, "limit", None),
                    }
                    for b in memory.blocks
                ]
            if hasattr(memory, "prompt_template"):
                memory_state["prompt_template"] = memory.prompt_template

        return memory_state

    async def update_memory(
        self,
        agent_id: str,
        block_label: str,
        value: str,
    ) -> bool:
        """
        Update a memory block.

        Args:
            agent_id: The agent ID
            block_label: Label of the memory block to update
            value: New value for the block

        Returns:
            True if update was successful

        Raises:
            RuntimeError: If client not initialized
            Exception: If update fails
        """
        self._ensure_initialized()

        # Get the agent to find the block ID
        agent = await self._run_in_executor(
            self._client.agents.get,
            agent_id=agent_id
        )

        block_id = None
        if hasattr(agent, "memory") and hasattr(agent.memory, "blocks"):
            for block in agent.memory.blocks:
                if getattr(block, "label", None) == block_label:
                    block_id = block.id
                    break

        if block_id is None:
            raise ValueError(f"Memory block with label '{block_label}' not found")

        # Update the block
        await self._run_in_executor(
            self._client.blocks.update,
            block_id=block_id,
            value=value
        )

        return True

    async def search_memory(
        self,
        agent_id: str,
        query: str,
        top_k: int = 10,
    ) -> List[LettaMemoryBlock]:
        """
        Search agent's archival memory.

        Args:
            agent_id: The agent ID
            query: Search query
            top_k: Number of results to return

        Returns:
            List of LettaMemoryBlock objects

        Raises:
            RuntimeError: If client not initialized
            Exception: If search fails
        """
        self._ensure_initialized()

        results = await self._run_in_executor(
            self._client.agents.passages.search,
            agent_id=agent_id,
            query=query,
            top_k=top_k
        )

        passages = []
        for r in results.results:
            passages.append(LettaMemoryBlock(
                id=getattr(r, "id", ""),
                content=r.content,
                score=getattr(r, "score", None),
                metadata=getattr(r, "metadata", {}),
            ))

        return passages

    async def add_memory(
        self,
        agent_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LettaMemoryBlock:
        """
        Add content to agent's archival memory.

        Args:
            agent_id: The agent ID
            content: Content to add
            metadata: Optional metadata

        Returns:
            LettaMemoryBlock with the created passage

        Raises:
            RuntimeError: If client not initialized
            Exception: If addition fails
        """
        self._ensure_initialized()

        passage = await self._run_in_executor(
            self._client.agents.passages.create,
            agent_id=agent_id,
            text=content,
            metadata=metadata or {}
        )

        return LettaMemoryBlock(
            id=passage.id,
            content=content,
            metadata=metadata or {},
        )

    async def delete_memory(
        self,
        agent_id: str,
        passage_id: str,
    ) -> bool:
        """
        Delete a passage from archival memory.

        Args:
            agent_id: The agent ID
            passage_id: The passage ID to delete

        Returns:
            True if deletion was successful

        Raises:
            RuntimeError: If client not initialized
            Exception: If deletion fails
        """
        self._ensure_initialized()

        await self._run_in_executor(
            self._client.agents.passages.delete,
            agent_id=agent_id,
            passage_id=passage_id
        )

        return True

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    async def health_check(self) -> bool:
        """
        Check if the Letta API is reachable.

        Returns:
            True if API is healthy

        Raises:
            RuntimeError: If client not initialized
        """
        self._ensure_initialized()

        try:
            await self._run_in_executor(self._client.agents.list, limit=1)
            return True
        except Exception:
            return False

    @property
    def is_initialized(self) -> bool:
        """Check if client is initialized."""
        return self._initialized

    @property
    def executor(self) -> Optional[ThreadPoolExecutor]:
        """Get the thread pool executor for connection pooling."""
        return self._executor


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------

async def create_async_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_workers: int = AsyncLettaClient.DEFAULT_MAX_WORKERS,
) -> AsyncLettaClient:
    """
    Create and initialize an async Letta client.

    This is a convenience function that creates and initializes
    the client in one call.

    Args:
        api_key: Letta API key
        base_url: API base URL
        max_workers: Thread pool workers

    Returns:
        Initialized AsyncLettaClient

    Example:
        client = await create_async_client()
        agents = await client.list_agents()
        await client.close()
    """
    client = AsyncLettaClient(
        api_key=api_key,
        base_url=base_url,
        max_workers=max_workers,
    )
    await client.initialize()
    return client


# -----------------------------------------------------------------------------
# Connection Pool for Shared Executor
# -----------------------------------------------------------------------------

class AsyncLettaPool:
    """
    Connection pool for multiple AsyncLettaClient instances.

    Shares a single ThreadPoolExecutor across multiple clients
    for better resource utilization.

    Example:
        pool = AsyncLettaPool(max_workers=8)
        client1 = pool.get_client(api_key="key1")
        client2 = pool.get_client(api_key="key2")

        async with client1:
            await client1.list_agents()

        pool.shutdown()
    """

    def __init__(self, max_workers: int = 8):
        """
        Initialize the connection pool.

        Args:
            max_workers: Maximum workers for the shared executor
        """
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="letta-pool-"
        )
        self._clients: List[AsyncLettaClient] = []

    def get_client(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> AsyncLettaClient:
        """
        Get a client instance that uses the shared executor.

        Args:
            api_key: Letta API key
            base_url: API base URL

        Returns:
            AsyncLettaClient using the shared executor
        """
        client = AsyncLettaClient(
            api_key=api_key,
            base_url=base_url,
            executor=self._executor,
        )
        self._clients.append(client)
        return client

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the pool and all clients.

        Args:
            wait: Whether to wait for pending tasks
        """
        # Close all clients first (they won't shutdown the executor since they don't own it)
        for client in self._clients:
            client._initialized = False
            client._client = None

        self._clients.clear()

        # Now shutdown the shared executor
        self._executor.shutdown(wait=wait)

    async def close_all(self) -> None:
        """Async method to close all clients."""
        for client in self._clients:
            await client.close()
        self._clients.clear()

    def __enter__(self) -> "AsyncLettaPool":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
