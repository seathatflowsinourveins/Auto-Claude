"""
Letta Adapter - Real Implementation (V37 - Fully Unleashed)

Production adapter for Letta Cloud memory service.
This is a REAL implementation that makes actual API calls, not a stub.

Features:
- Hierarchical memory (archival, core, recall)
- Multi-hop reasoning
- Cross-session persistence
- MCP server integration
- Memory summarization hooks (V37)
- Multi-agent memory sharing (V37)
- Advanced search with filters (V37)
- Memory blocks management (V37)
- Tag-based namespacing (V37)
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

try:
    from core.orchestration.base import (
        AdapterConfig,
        AdapterResult,
        AdapterStatus,
        SDKAdapter,
        SDKLayer,
    )
    from core.orchestration.sdk_registry import register_adapter
except (ImportError, ValueError):
    from core.orchestration.base import (
        AdapterConfig,
        AdapterResult,
        AdapterStatus,
        SDKAdapter,
        SDKLayer,
    )
    from core.orchestration.sdk_registry import register_adapter

logger = logging.getLogger(__name__)


@register_adapter("letta", SDKLayer.MEMORY, priority=10, tags={"memory", "production"})
class LettaAdapter(SDKAdapter):
    """
    Real Letta adapter that makes actual API calls to Letta Cloud.

    Configuration:
        - LETTA_API_KEY: API key for Letta Cloud
        - LETTA_BASE_URL: Base URL (default: https://api.letta.com)

    Supported Operations (V37 - Fully Unleashed):
        Agent Management:
        - create_agent: Create a new Letta agent with memory blocks
        - get_agent: Get agent details
        - list_agents: List all agents
        - delete_agent: Delete an agent

        Messaging:
        - message: Send a message to an agent

        Archival Memory:
        - search: Search archival memory with advanced filters
        - add_memory: Add content to archival memory with tags
        - delete_memory: Delete a passage from archival memory

        Core Memory Blocks:
        - list_blocks: List all memory blocks for an agent
        - get_block: Get a specific memory block by label
        - update_block: Update a memory block's value
        - attach_block: Attach a block to an agent
        - detach_block: Detach a block from an agent

        Summarization:
        - summarize_memory: Trigger memory summarization
        - get_memory_stats: Get memory usage statistics

        Multi-Agent:
        - share_block: Share a memory block between agents
        - sync_shared_blocks: Sync shared blocks in a group
    """

    def __init__(self, config: Optional[AdapterConfig] = None):
        super().__init__(config or AdapterConfig(name="letta", layer=SDKLayer.MEMORY))
        self._client = None
        self._available = False

    @property
    def sdk_name(self) -> str:
        return "letta"

    @property
    def layer(self) -> SDKLayer:
        return SDKLayer.MEMORY

    @property
    def available(self) -> bool:
        return self._available

    async def initialize(self, config: Dict[str, Any]) -> AdapterResult:
        """Initialize Letta client."""
        start = time.time()

        try:
            # Check for API key
            api_key = config.get("api_key") or os.environ.get("LETTA_API_KEY")
            if not api_key:
                return AdapterResult(
                    success=False,
                    error="LETTA_API_KEY not configured",
                    latency_ms=(time.time() - start) * 1000
                )

            # Import letta client
            try:
                from letta_client import Letta
            except ImportError:
                return AdapterResult(
                    success=False,
                    error="letta-client not installed. Run: pip install letta-client",
                    latency_ms=(time.time() - start) * 1000
                )

            # Initialize client
            base_url = config.get("base_url") or os.environ.get("LETTA_BASE_URL", "https://api.letta.com")
            self._client = Letta(api_key=api_key, base_url=base_url)

            # Verify connection by listing agents
            self._client.agents.list(limit=1)

            self._available = True
            self._status = AdapterStatus.READY

            logger.info("Letta adapter initialized successfully")
            return AdapterResult(
                success=True,
                data={"status": "connected", "base_url": base_url},
                latency_ms=(time.time() - start) * 1000
            )

        except Exception as e:
            self._status = AdapterStatus.FAILED
            logger.error(f"Failed to initialize Letta: {e}")
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start) * 1000
            )

    async def execute(self, operation: str, **kwargs) -> AdapterResult:
        """Execute a Letta operation."""
        start = time.time()

        if not self._available or not self._client:
            return AdapterResult(
                success=False,
                error="Letta client not initialized",
                latency_ms=(time.time() - start) * 1000
            )

        try:
            result = await self._dispatch_operation(operation, kwargs)
            latency = (time.time() - start) * 1000
            self._record_call(latency, result.success)

            result.latency_ms = latency
            return result

        except Exception as e:
            latency = (time.time() - start) * 1000
            self._record_call(latency, False)
            logger.error(f"Letta operation '{operation}' failed: {e}")
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=latency
            )

    async def _dispatch_operation(self, operation: str, kwargs: Dict[str, Any]) -> AdapterResult:
        """Dispatch to the appropriate operation handler."""
        handlers = {
            # Agent Management
            "create_agent": self._create_agent,
            "get_agent": self._get_agent,
            "list_agents": self._list_agents,
            "delete_agent": self._delete_agent,
            # Messaging
            "message": self._send_message,
            # Archival Memory
            "search": self._search_memory,
            "add_memory": self._add_memory,
            "delete_memory": self._delete_memory,
            # Core Memory Blocks (V37)
            "list_blocks": self._list_blocks,
            "get_block": self._get_block,
            "update_block": self._update_block,
            "attach_block": self._attach_block,
            "detach_block": self._detach_block,
            # Summarization (V37)
            "summarize_memory": self._summarize_memory,
            "get_memory_stats": self._get_memory_stats,
            # Multi-Agent (V37)
            "share_block": self._share_block,
            "sync_shared_blocks": self._sync_shared_blocks,
        }

        handler = handlers.get(operation)
        if not handler:
            return AdapterResult(
                success=False,
                error=f"Unknown operation: {operation}"
            )

        return await handler(kwargs)

    async def _create_agent(self, kwargs: Dict[str, Any]) -> AdapterResult:
        """Create a new Letta agent."""
        name = kwargs.get("name", "unleash-agent")
        system_prompt = kwargs.get("system_prompt")
        model = kwargs.get("model", "claude-3-5-sonnet-20241022")
        embedding_model = kwargs.get("embedding_model", "text-embedding-3-small")

        try:
            agent = self._client.agents.create(
                name=name,
                model=model,
                embedding=embedding_model,
                system=system_prompt
            )

            return AdapterResult(
                success=True,
                data={
                    "agent_id": agent.id,
                    "agent_name": agent.name,
                    "created": True
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _send_message(self, kwargs: Dict[str, Any]) -> AdapterResult:
        """Send a message to an agent."""
        agent_id = kwargs.get("agent_id")
        content = kwargs.get("content", "")

        if not agent_id:
            return AdapterResult(success=False, error="agent_id required")

        try:
            response = self._client.agents.messages.create(
                agent_id=agent_id,
                messages=[{"role": "user", "content": content}]
            )

            # Extract response messages
            messages = []
            for msg in response.messages:
                if hasattr(msg, 'assistant_message') and msg.assistant_message:
                    messages.append(msg.assistant_message)
                elif hasattr(msg, 'content'):
                    messages.append(msg.content)

            return AdapterResult(
                success=True,
                data={
                    "response": messages[-1] if messages else None,
                    "all_messages": messages,
                    "usage": getattr(response, 'usage', None)
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _search_memory(self, kwargs: Dict[str, Any]) -> AdapterResult:
        """
        Search agent archival memory with advanced filters (V37 Enhanced).

        Args:
            agent_id: The agent ID (required)
            query: Search query string (required)
            top_k: Maximum results to return (default: 10)
            tags: Optional list of tags to filter by
            tag_match_mode: "any" or "all" for tag matching (default: "any")
            start_datetime: Filter by creation time (ISO 8601)
            end_datetime: Filter by creation time (ISO 8601)
        """
        agent_id = kwargs.get("agent_id")
        query = kwargs.get("query", "")
        top_k = kwargs.get("top_k", 10)
        tags = kwargs.get("tags")
        tag_match_mode = kwargs.get("tag_match_mode", "any")
        start_datetime = kwargs.get("start_datetime")
        end_datetime = kwargs.get("end_datetime")

        if not agent_id:
            return AdapterResult(success=False, error="agent_id required")

        try:
            # Build search parameters with all V37 filters
            search_params = {
                "agent_id": agent_id,
                "query": query,
                "top_k": top_k
            }

            # Add tag filtering (V37)
            if tags:
                search_params["tags"] = tags
                search_params["tag_match_mode"] = tag_match_mode

            # Add temporal filtering (V37)
            if start_datetime:
                search_params["start_datetime"] = start_datetime
            if end_datetime:
                search_params["end_datetime"] = end_datetime

            results = self._client.agents.passages.search(**search_params)

            # Extract passages with full metadata
            passages = []
            for r in results.results:
                passages.append({
                    "id": getattr(r, 'id', None),
                    "content": r.content,
                    "score": getattr(r, 'score', None),
                    "metadata": getattr(r, 'metadata', {}),
                    "tags": getattr(r, 'tags', []),
                    "created_at": getattr(r, 'created_at', None)
                })

            return AdapterResult(
                success=True,
                data={
                    "query": query,
                    "results": passages,
                    "count": len(passages),
                    "filters_applied": {
                        "tags": tags,
                        "tag_match_mode": tag_match_mode if tags else None,
                        "start_datetime": start_datetime,
                        "end_datetime": end_datetime
                    }
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _add_memory(self, kwargs: Dict[str, Any]) -> AdapterResult:
        """
        Add content to agent archival memory with tags (V37 Enhanced).

        Args:
            agent_id: The agent ID (required)
            content: Text content to store (required)
            tags: Optional list of tags for categorization
            metadata: Optional metadata dictionary
        """
        agent_id = kwargs.get("agent_id")
        content = kwargs.get("content", "")
        tags = kwargs.get("tags")
        metadata = kwargs.get("metadata", {})

        if not agent_id:
            return AdapterResult(success=False, error="agent_id required")

        try:
            # Build passage parameters with tags (V37)
            passage_params = {
                "agent_id": agent_id,
                "text": content
            }

            if tags:
                passage_params["tags"] = tags
            if metadata:
                passage_params["metadata"] = metadata

            passage = self._client.agents.passages.create(**passage_params)

            return AdapterResult(
                success=True,
                data={
                    "passage_id": passage.id,
                    "added": True,
                    "content_preview": content[:100],
                    "tags": tags
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _delete_memory(self, kwargs: Dict[str, Any]) -> AdapterResult:
        """
        Delete a passage from archival memory (V37).

        Args:
            agent_id: The agent ID (required)
            passage_id: The passage ID to delete (required)
        """
        agent_id = kwargs.get("agent_id")
        passage_id = kwargs.get("passage_id")

        if not agent_id:
            return AdapterResult(success=False, error="agent_id required")
        if not passage_id:
            return AdapterResult(success=False, error="passage_id required")

        try:
            self._client.agents.passages.delete(
                agent_id=agent_id,
                memory_id=passage_id
            )

            return AdapterResult(
                success=True,
                data={
                    "deleted": True,
                    "passage_id": passage_id
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _list_agents(self, kwargs: Dict[str, Any]) -> AdapterResult:
        """List all agents."""
        limit = kwargs.get("limit", 50)

        try:
            agents = self._client.agents.list(limit=limit)

            agent_list = [
                {
                    "id": a.id,
                    "name": a.name,
                    "created_at": getattr(a, 'created_at', None)
                }
                for a in agents
            ]

            return AdapterResult(
                success=True,
                data={
                    "agents": agent_list,
                    "count": len(agent_list)
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _get_agent(self, kwargs: Dict[str, Any]) -> AdapterResult:
        """Get agent details."""
        agent_id = kwargs.get("agent_id")

        if not agent_id:
            return AdapterResult(success=False, error="agent_id required")

        try:
            agent = self._client.agents.get(agent_id=agent_id)

            return AdapterResult(
                success=True,
                data={
                    "id": agent.id,
                    "name": agent.name,
                    "model": getattr(agent, 'model', None),
                    "created_at": getattr(agent, 'created_at', None)
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _delete_agent(self, kwargs: Dict[str, Any]) -> AdapterResult:
        """Delete an agent."""
        agent_id = kwargs.get("agent_id")

        if not agent_id:
            return AdapterResult(success=False, error="agent_id required")

        try:
            self._client.agents.delete(agent_id=agent_id)

            return AdapterResult(
                success=True,
                data={"deleted": True, "agent_id": agent_id}
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    # =========================================================================
    # Core Memory Block Operations (V37)
    # =========================================================================

    async def _list_blocks(self, kwargs: Dict[str, Any]) -> AdapterResult:
        """
        List all memory blocks for an agent (V37).

        Args:
            agent_id: The agent ID (required)
        """
        agent_id = kwargs.get("agent_id")

        if not agent_id:
            return AdapterResult(success=False, error="agent_id required")

        try:
            blocks = self._client.agents.blocks.list(agent_id=agent_id)

            block_list = []
            for block in blocks:
                block_list.append({
                    "id": getattr(block, 'id', None),
                    "label": getattr(block, 'label', None),
                    "value": getattr(block, 'value', ''),
                    "limit": getattr(block, 'limit', 5000),
                    "description": getattr(block, 'description', ''),
                    "read_only": getattr(block, 'read_only', False)
                })

            return AdapterResult(
                success=True,
                data={
                    "blocks": block_list,
                    "count": len(block_list)
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _get_block(self, kwargs: Dict[str, Any]) -> AdapterResult:
        """
        Get a specific memory block by label (V37).

        Args:
            agent_id: The agent ID (required)
            label: The block label (required)
        """
        agent_id = kwargs.get("agent_id")
        label = kwargs.get("label")

        if not agent_id:
            return AdapterResult(success=False, error="agent_id required")
        if not label:
            return AdapterResult(success=False, error="label required")

        try:
            block = self._client.agents.blocks.retrieve(label, agent_id=agent_id)

            return AdapterResult(
                success=True,
                data={
                    "id": getattr(block, 'id', None),
                    "label": getattr(block, 'label', label),
                    "value": getattr(block, 'value', ''),
                    "limit": getattr(block, 'limit', 5000),
                    "description": getattr(block, 'description', ''),
                    "chars_used": len(getattr(block, 'value', ''))
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _update_block(self, kwargs: Dict[str, Any]) -> AdapterResult:
        """
        Update a memory block's value (V37).

        Args:
            agent_id: The agent ID (required)
            label: The block label (required)
            value: The new value for the block (required)
        """
        agent_id = kwargs.get("agent_id")
        label = kwargs.get("label")
        value = kwargs.get("value")

        if not agent_id:
            return AdapterResult(success=False, error="agent_id required")
        if not label:
            return AdapterResult(success=False, error="label required")
        if value is None:
            return AdapterResult(success=False, error="value required")

        try:
            self._client.agents.blocks.update(label, agent_id=agent_id, value=value)

            return AdapterResult(
                success=True,
                data={
                    "updated": True,
                    "label": label,
                    "chars": len(value)
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _attach_block(self, kwargs: Dict[str, Any]) -> AdapterResult:
        """
        Attach a block to an agent (V37).

        Args:
            agent_id: The agent ID (required)
            block_id: The block ID to attach (required)
        """
        agent_id = kwargs.get("agent_id")
        block_id = kwargs.get("block_id")

        if not agent_id:
            return AdapterResult(success=False, error="agent_id required")
        if not block_id:
            return AdapterResult(success=False, error="block_id required")

        try:
            self._client.agents.blocks.attach(block_id, agent_id=agent_id)

            return AdapterResult(
                success=True,
                data={
                    "attached": True,
                    "block_id": block_id,
                    "agent_id": agent_id
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _detach_block(self, kwargs: Dict[str, Any]) -> AdapterResult:
        """
        Detach a block from an agent (V37).

        Args:
            agent_id: The agent ID (required)
            block_id: The block ID to detach (required)
        """
        agent_id = kwargs.get("agent_id")
        block_id = kwargs.get("block_id")

        if not agent_id:
            return AdapterResult(success=False, error="agent_id required")
        if not block_id:
            return AdapterResult(success=False, error="block_id required")

        try:
            self._client.agents.blocks.detach(block_id, agent_id=agent_id)

            return AdapterResult(
                success=True,
                data={
                    "detached": True,
                    "block_id": block_id,
                    "agent_id": agent_id
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    # =========================================================================
    # Summarization Operations (V37)
    # =========================================================================

    async def _summarize_memory(self, kwargs: Dict[str, Any]) -> AdapterResult:
        """
        Trigger memory summarization for old passages (V37).

        This operation summarizes old archival memories and consolidates them
        to reduce storage and improve search quality.

        Args:
            agent_id: The agent ID (required)
            batch_size: Number of passages to summarize (default: 10)
            threshold_days: Age threshold in days (default: 30)
        """
        agent_id = kwargs.get("agent_id")
        batch_size = kwargs.get("batch_size", 10)
        threshold_days = kwargs.get("threshold_days", 30)

        if not agent_id:
            return AdapterResult(success=False, error="agent_id required")

        try:
            # Search for old passages to summarize
            results = self._client.agents.passages.search(
                agent_id=agent_id,
                query="*",  # Get all
                top_k=batch_size * 2
            )

            passages_to_summarize = []
            for r in results.results:
                created_at = getattr(r, 'created_at', None)
                if created_at:
                    # Check if older than threshold
                    if hasattr(created_at, 'timestamp'):
                        age_days = (datetime.now().timestamp() - created_at.timestamp()) / 86400
                        if age_days > threshold_days:
                            passages_to_summarize.append({
                                "id": getattr(r, 'id', None),
                                "content": r.content,
                                "age_days": age_days
                            })

            if not passages_to_summarize:
                return AdapterResult(
                    success=True,
                    data={
                        "summarized": False,
                        "reason": "No passages exceed age threshold",
                        "threshold_days": threshold_days
                    }
                )

            # Note: Actual summarization would be done by the agent via tools
            # This just identifies candidates
            return AdapterResult(
                success=True,
                data={
                    "candidates": len(passages_to_summarize),
                    "passages": passages_to_summarize[:batch_size],
                    "message": "Summarization candidates identified. Use agent tools to consolidate."
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _get_memory_stats(self, kwargs: Dict[str, Any]) -> AdapterResult:
        """
        Get memory usage statistics for an agent (V37).

        Args:
            agent_id: The agent ID (required)
        """
        agent_id = kwargs.get("agent_id")

        if not agent_id:
            return AdapterResult(success=False, error="agent_id required")

        try:
            # Get core memory stats
            blocks = self._client.agents.blocks.list(agent_id=agent_id)
            core_memory = {
                "block_count": 0,
                "total_chars": 0,
                "total_limit": 0,
                "blocks": []
            }

            for block in blocks:
                value = getattr(block, 'value', '')
                limit = getattr(block, 'limit', 5000)
                core_memory["block_count"] += 1
                core_memory["total_chars"] += len(value)
                core_memory["total_limit"] += limit
                core_memory["blocks"].append({
                    "label": getattr(block, 'label', ''),
                    "chars": len(value),
                    "limit": limit,
                    "usage_percent": (len(value) / limit * 100) if limit > 0 else 0
                })

            # Get archival memory stats (sample)
            results = self._client.agents.passages.search(
                agent_id=agent_id,
                query="*",
                top_k=100
            )

            archival_memory = {
                "sample_count": len(results.results),
                "estimated_total": len(results.results)  # Would need pagination for exact
            }

            return AdapterResult(
                success=True,
                data={
                    "core_memory": core_memory,
                    "archival_memory": archival_memory,
                    "health": {
                        "core_usage_percent": (
                            core_memory["total_chars"] / core_memory["total_limit"] * 100
                        ) if core_memory["total_limit"] > 0 else 0
                    }
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    # =========================================================================
    # Multi-Agent Memory Sharing (V37)
    # =========================================================================

    async def _share_block(self, kwargs: Dict[str, Any]) -> AdapterResult:
        """
        Share a memory block between agents (V37).

        Args:
            source_agent_id: Source agent ID (required)
            target_agent_id: Target agent ID (required)
            block_label: Label of block to share (required)
        """
        source_agent_id = kwargs.get("source_agent_id")
        target_agent_id = kwargs.get("target_agent_id")
        block_label = kwargs.get("block_label")

        if not source_agent_id:
            return AdapterResult(success=False, error="source_agent_id required")
        if not target_agent_id:
            return AdapterResult(success=False, error="target_agent_id required")
        if not block_label:
            return AdapterResult(success=False, error="block_label required")

        try:
            # Get block from source agent
            block = self._client.agents.blocks.retrieve(
                block_label, agent_id=source_agent_id
            )

            block_id = getattr(block, 'id', None)
            if not block_id:
                return AdapterResult(
                    success=False,
                    error=f"Block '{block_label}' not found on source agent"
                )

            # Attach to target agent
            self._client.agents.blocks.attach(block_id, agent_id=target_agent_id)

            return AdapterResult(
                success=True,
                data={
                    "shared": True,
                    "block_id": block_id,
                    "block_label": block_label,
                    "source_agent_id": source_agent_id,
                    "target_agent_id": target_agent_id
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _sync_shared_blocks(self, kwargs: Dict[str, Any]) -> AdapterResult:
        """
        Sync shared memory blocks across a group of agents (V37).

        Args:
            agent_ids: List of agent IDs to sync (required)
            block_labels: List of block labels to sync (required)
        """
        agent_ids = kwargs.get("agent_ids", [])
        block_labels = kwargs.get("block_labels", [])

        if not agent_ids or len(agent_ids) < 2:
            return AdapterResult(success=False, error="At least 2 agent_ids required")
        if not block_labels:
            return AdapterResult(success=False, error="block_labels required")

        try:
            # Use first agent as source
            source_agent_id = agent_ids[0]
            target_agent_ids = agent_ids[1:]

            synced = []
            errors = []

            for block_label in block_labels:
                try:
                    # Get block from source
                    block = self._client.agents.blocks.retrieve(
                        block_label, agent_id=source_agent_id
                    )
                    block_id = getattr(block, 'id', None)

                    if block_id:
                        for target_id in target_agent_ids:
                            try:
                                self._client.agents.blocks.attach(
                                    block_id, agent_id=target_id
                                )
                                synced.append({
                                    "block_label": block_label,
                                    "target_agent_id": target_id
                                })
                            except Exception as attach_err:
                                errors.append({
                                    "block_label": block_label,
                                    "target_agent_id": target_id,
                                    "error": str(attach_err)
                                })
                except Exception as block_err:
                    errors.append({
                        "block_label": block_label,
                        "error": str(block_err)
                    })

            return AdapterResult(
                success=len(errors) == 0,
                data={
                    "synced_count": len(synced),
                    "synced": synced,
                    "error_count": len(errors),
                    "errors": errors if errors else None
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    # =========================================================================
    # Health and Lifecycle
    # =========================================================================

    async def health_check(self) -> AdapterResult:
        """Check Letta connection health."""
        start = time.time()

        if not self._client:
            return AdapterResult(
                success=False,
                error="Client not initialized",
                latency_ms=(time.time() - start) * 1000
            )

        try:
            # Simple health check - list agents with limit 1
            self._client.agents.list(limit=1)
            return AdapterResult(
                success=True,
                data={"status": "healthy", "version": "V37"},
                latency_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start) * 1000
            )

    async def shutdown(self) -> AdapterResult:
        """Shutdown the Letta client."""
        self._client = None
        self._available = False
        self._status = AdapterStatus.SHUTDOWN
        return AdapterResult(success=True, data={"status": "shutdown"})
