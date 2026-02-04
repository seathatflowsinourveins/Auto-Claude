"""
MCP Fallback Chain Manager - V38 Architecture (ADR-028)

Provides automatic fallback routing when primary MCP servers fail.
Implements multiple fallback strategies for different use cases.

Strategies:
- failover: Try primary, then fallbacks in order
- parallel_race: Execute all in parallel, return fastest success
- consensus: Execute all, return majority result
- weighted_random: Random selection based on weights

Usage:
    manager = FallbackChainManager()

    # Register fallback chain
    manager.register_chain(
        name="research",
        primary="exa",
        fallbacks=["tavily", "perplexity"],
        strategy="failover",
    )

    # Execute with fallback
    result = await manager.execute(
        chain="research",
        tool="search",
        args={"query": "AI patterns"},
    )
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FallbackStrategy(str, Enum):
    """Fallback execution strategies."""
    FAILOVER = "failover"           # Try in order until success
    PARALLEL_RACE = "parallel_race"  # Execute all, return fastest
    CONSENSUS = "consensus"          # Execute all, return majority
    WEIGHTED_RANDOM = "weighted_random"  # Random with weights


@dataclass
class ChainMember:
    """A server in a fallback chain."""
    server: str
    weight: float = 1.0
    timeout_seconds: float = 30.0
    max_retries: int = 1
    is_healthy: bool = True
    consecutive_failures: int = 0
    total_calls: int = 0
    total_failures: int = 0

    @property
    def failure_rate(self) -> float:
        """Current failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.total_failures / self.total_calls

    def record_call(self, success: bool) -> None:
        """Record a call result."""
        self.total_calls += 1
        if success:
            self.consecutive_failures = 0
        else:
            self.total_failures += 1
            self.consecutive_failures += 1
            # Mark unhealthy after 3 consecutive failures
            if self.consecutive_failures >= 3:
                self.is_healthy = False


@dataclass
class FallbackChain:
    """Configuration for a fallback chain."""
    name: str
    members: List[ChainMember]
    strategy: FallbackStrategy = FallbackStrategy.FAILOVER
    min_healthy_members: int = 1
    circuit_breaker_threshold: int = 5
    recovery_timeout_seconds: float = 60.0

    @property
    def healthy_members(self) -> List[ChainMember]:
        """Get healthy members."""
        return [m for m in self.members if m.is_healthy]

    @property
    def primary(self) -> Optional[ChainMember]:
        """Get primary (first) member."""
        return self.members[0] if self.members else None


@dataclass
class ChainResult:
    """Result from a fallback chain execution."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    server_used: Optional[str] = None
    servers_tried: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    strategy: Optional[FallbackStrategy] = None
    is_fallback: bool = False

    @classmethod
    def failure(cls, error: str, servers_tried: List[str]) -> "ChainResult":
        """Create a failure result."""
        return cls(
            success=False,
            error=error,
            servers_tried=servers_tried,
        )


class FallbackChainManager:
    """
    Manages fallback chains for MCP servers.

    Provides automatic routing and fallback when servers fail,
    with multiple strategies for different use cases.
    """

    # Default fallback chains for common research patterns
    DEFAULT_CHAINS = {
        "research": {
            "members": [
                ("exa", 1.0, 60.0),
                ("tavily", 0.8, 45.0),
                ("perplexity", 0.7, 90.0),
                ("serper", 0.5, 15.0),
            ],
            "strategy": FallbackStrategy.FAILOVER,
        },
        "content_extraction": {
            "members": [
                ("firecrawl", 1.0, 120.0),
                ("jina", 0.9, 30.0),
            ],
            "strategy": FallbackStrategy.FAILOVER,
        },
        "code_search": {
            "members": [
                ("exa", 1.0, 60.0),
                ("context7", 0.8, 15.0),
            ],
            "strategy": FallbackStrategy.PARALLEL_RACE,
        },
        "documentation": {
            "members": [
                ("context7", 1.0, 15.0),
                ("exa", 0.8, 60.0),
            ],
            "strategy": FallbackStrategy.FAILOVER,
        },
    }

    def __init__(self):
        self._chains: Dict[str, FallbackChain] = {}
        self._executors: Dict[str, Callable] = {}
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self, register_defaults: bool = True) -> None:
        """Initialize the manager."""
        if self._initialized:
            return

        if register_defaults:
            for name, config in self.DEFAULT_CHAINS.items():
                members = [
                    ChainMember(server=s, weight=w, timeout_seconds=t)
                    for s, w, t in config["members"]
                ]
                self._chains[name] = FallbackChain(
                    name=name,
                    members=members,
                    strategy=config["strategy"],
                )

        self._initialized = True
        logger.info(f"[FALLBACK] Initialized with {len(self._chains)} chains")

    def register_chain(
        self,
        name: str,
        primary: str,
        fallbacks: List[str],
        strategy: FallbackStrategy = FallbackStrategy.FAILOVER,
        weights: Optional[Dict[str, float]] = None,
        timeouts: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Register a new fallback chain.

        Args:
            name: Chain name
            primary: Primary server name
            fallbacks: Fallback server names in order
            strategy: Execution strategy
            weights: Server weights (for weighted_random)
            timeouts: Server timeouts
        """
        weights = weights or {}
        timeouts = timeouts or {}

        members = [
            ChainMember(
                server=primary,
                weight=weights.get(primary, 1.0),
                timeout_seconds=timeouts.get(primary, 30.0),
            )
        ]

        for fb in fallbacks:
            members.append(ChainMember(
                server=fb,
                weight=weights.get(fb, 0.8),
                timeout_seconds=timeouts.get(fb, 30.0),
            ))

        self._chains[name] = FallbackChain(
            name=name,
            members=members,
            strategy=strategy,
        )

        logger.info(f"[FALLBACK] Registered chain '{name}': {primary} -> {fallbacks}")

    def register_executor(
        self,
        server: str,
        executor: Callable[[str, Dict[str, Any]], Any],
    ) -> None:
        """
        Register an executor for a server.

        Args:
            server: Server name
            executor: Async function(tool, args) -> result
        """
        self._executors[server] = executor

    async def execute(
        self,
        chain: str,
        tool: str,
        args: Dict[str, Any],
        prefer_server: Optional[str] = None,
    ) -> ChainResult:
        """
        Execute a tool call with fallback support.

        Args:
            chain: Fallback chain name
            tool: Tool name
            args: Tool arguments
            prefer_server: Preferred server (bypasses chain order)

        Returns:
            ChainResult with the execution result
        """
        start = time.time()

        if chain not in self._chains:
            return ChainResult.failure(
                f"Unknown chain: {chain}",
                servers_tried=[],
            )

        chain_config = self._chains[chain]

        # Route based on strategy
        if chain_config.strategy == FallbackStrategy.FAILOVER:
            result = await self._execute_failover(chain_config, tool, args, prefer_server)
        elif chain_config.strategy == FallbackStrategy.PARALLEL_RACE:
            result = await self._execute_parallel_race(chain_config, tool, args)
        elif chain_config.strategy == FallbackStrategy.CONSENSUS:
            result = await self._execute_consensus(chain_config, tool, args)
        elif chain_config.strategy == FallbackStrategy.WEIGHTED_RANDOM:
            result = await self._execute_weighted_random(chain_config, tool, args)
        else:
            result = await self._execute_failover(chain_config, tool, args, prefer_server)

        result.latency_ms = (time.time() - start) * 1000
        result.strategy = chain_config.strategy

        return result

    async def _execute_failover(
        self,
        chain: FallbackChain,
        tool: str,
        args: Dict[str, Any],
        prefer_server: Optional[str] = None,
    ) -> ChainResult:
        """Execute with failover strategy."""
        servers_tried = []
        members = chain.healthy_members

        # Reorder if prefer_server specified
        if prefer_server:
            preferred = [m for m in members if m.server == prefer_server]
            others = [m for m in members if m.server != prefer_server]
            members = preferred + others

        for member in members:
            servers_tried.append(member.server)

            try:
                result = await self._call_server(member, tool, args)
                member.record_call(success=True)

                return ChainResult(
                    success=True,
                    result=result,
                    server_used=member.server,
                    servers_tried=servers_tried,
                    is_fallback=member.server != chain.primary.server if chain.primary else False,
                )

            except Exception as e:
                member.record_call(success=False)
                logger.warning(f"[FALLBACK] {member.server} failed: {e}")
                continue

        return ChainResult.failure(
            f"All servers in chain '{chain.name}' failed",
            servers_tried=servers_tried,
        )

    async def _execute_parallel_race(
        self,
        chain: FallbackChain,
        tool: str,
        args: Dict[str, Any],
    ) -> ChainResult:
        """Execute all in parallel, return fastest success."""
        servers_tried = []
        members = chain.healthy_members

        if not members:
            return ChainResult.failure(
                f"No healthy servers in chain '{chain.name}'",
                servers_tried=[],
            )

        # Create tasks for all healthy members
        tasks = []
        for member in members:
            servers_tried.append(member.server)
            task = asyncio.create_task(
                self._call_server_with_timeout(member, tool, args),
                name=member.server,
            )
            tasks.append((member, task))

        # Wait for first success
        done = set()
        pending = {t for _, t in tasks}

        while pending:
            finished, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in finished:
                member = next(m for m, t in tasks if t == task)

                try:
                    result = task.result()
                    member.record_call(success=True)

                    # Cancel remaining tasks
                    for p in pending:
                        p.cancel()

                    return ChainResult(
                        success=True,
                        result=result,
                        server_used=member.server,
                        servers_tried=servers_tried,
                        is_fallback=member.server != chain.primary.server if chain.primary else False,
                    )

                except Exception as e:
                    member.record_call(success=False)
                    logger.debug(f"[FALLBACK] {member.server} failed in race: {e}")
                    continue

        return ChainResult.failure(
            f"All parallel executions in chain '{chain.name}' failed",
            servers_tried=servers_tried,
        )

    async def _execute_consensus(
        self,
        chain: FallbackChain,
        tool: str,
        args: Dict[str, Any],
    ) -> ChainResult:
        """Execute all and return majority result."""
        servers_tried = []
        members = chain.healthy_members

        if not members:
            return ChainResult.failure(
                f"No healthy servers in chain '{chain.name}'",
                servers_tried=[],
            )

        # Execute all in parallel
        results: List[Tuple[ChainMember, Any]] = []
        errors: List[Tuple[ChainMember, Exception]] = []

        tasks = []
        for member in members:
            servers_tried.append(member.server)
            task = asyncio.create_task(
                self._call_server_with_timeout(member, tool, args),
            )
            tasks.append((member, task))

        await asyncio.gather(*[t for _, t in tasks], return_exceptions=True)

        for member, task in tasks:
            try:
                result = task.result()
                member.record_call(success=True)
                results.append((member, result))
            except Exception as e:
                member.record_call(success=False)
                errors.append((member, e))

        if not results:
            return ChainResult.failure(
                f"All servers in chain '{chain.name}' failed for consensus",
                servers_tried=servers_tried,
            )

        # Return result from highest-weight server
        results.sort(key=lambda x: x[0].weight, reverse=True)
        best = results[0]

        return ChainResult(
            success=True,
            result=best[1],
            server_used=best[0].server,
            servers_tried=servers_tried,
            is_fallback=best[0].server != chain.primary.server if chain.primary else False,
        )

    async def _execute_weighted_random(
        self,
        chain: FallbackChain,
        tool: str,
        args: Dict[str, Any],
    ) -> ChainResult:
        """Execute on weighted random selection with fallback."""
        servers_tried = []
        members = chain.healthy_members.copy()

        if not members:
            return ChainResult.failure(
                f"No healthy servers in chain '{chain.name}'",
                servers_tried=[],
            )

        # Keep trying random selection until success or exhausted
        while members:
            # Weighted random selection
            total_weight = sum(m.weight for m in members)
            r = random.uniform(0, total_weight)
            cumulative = 0

            selected = members[0]
            for member in members:
                cumulative += member.weight
                if r <= cumulative:
                    selected = member
                    break

            servers_tried.append(selected.server)
            members.remove(selected)

            try:
                result = await self._call_server(selected, tool, args)
                selected.record_call(success=True)

                return ChainResult(
                    success=True,
                    result=result,
                    server_used=selected.server,
                    servers_tried=servers_tried,
                    is_fallback=selected.server != chain.primary.server if chain.primary else False,
                )

            except Exception as e:
                selected.record_call(success=False)
                logger.debug(f"[FALLBACK] {selected.server} failed in weighted: {e}")
                continue

        return ChainResult.failure(
            f"All weighted random selections in chain '{chain.name}' failed",
            servers_tried=servers_tried,
        )

    async def _call_server(
        self,
        member: ChainMember,
        tool: str,
        args: Dict[str, Any],
    ) -> Any:
        """Call a server's tool."""
        if member.server not in self._executors:
            raise ValueError(f"No executor registered for {member.server}")

        executor = self._executors[member.server]
        return await executor(tool, args)

    async def _call_server_with_timeout(
        self,
        member: ChainMember,
        tool: str,
        args: Dict[str, Any],
    ) -> Any:
        """Call a server's tool with timeout."""
        return await asyncio.wait_for(
            self._call_server(member, tool, args),
            timeout=member.timeout_seconds,
        )

    def get_chain(self, name: str) -> Optional[FallbackChain]:
        """Get a chain configuration."""
        return self._chains.get(name)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all chains."""
        stats = {}
        for name, chain in self._chains.items():
            stats[name] = {
                "strategy": chain.strategy.value,
                "members": [
                    {
                        "server": m.server,
                        "weight": m.weight,
                        "is_healthy": m.is_healthy,
                        "failure_rate": round(m.failure_rate * 100, 2),
                        "total_calls": m.total_calls,
                    }
                    for m in chain.members
                ],
                "healthy_count": len(chain.healthy_members),
                "total_count": len(chain.members),
            }
        return stats

    async def recover_unhealthy(self) -> Dict[str, bool]:
        """
        Attempt to recover unhealthy servers.

        Returns:
            Dict mapping server name to recovery success
        """
        recovered = {}

        for chain in self._chains.values():
            for member in chain.members:
                if not member.is_healthy:
                    # Reset and try a health check
                    member.consecutive_failures = 0
                    member.is_healthy = True

                    if member.server in self._executors:
                        try:
                            # Simple health check - try a minimal call
                            await asyncio.wait_for(
                                self._executors[member.server]("health", {}),
                                timeout=5.0,
                            )
                            recovered[member.server] = True
                            logger.info(f"[FALLBACK] Recovered {member.server}")
                        except Exception:
                            member.is_healthy = False
                            recovered[member.server] = False
                            logger.debug(f"[FALLBACK] Failed to recover {member.server}")

        return recovered


# Global manager instance
_manager: Optional[FallbackChainManager] = None


def get_fallback_manager() -> FallbackChainManager:
    """Get global fallback chain manager."""
    global _manager
    if _manager is None:
        _manager = FallbackChainManager()
    return _manager


async def execute_with_fallback(
    chain: str,
    tool: str,
    args: Dict[str, Any],
    prefer_server: Optional[str] = None,
) -> ChainResult:
    """
    Convenience function for fallback execution.

    Usage:
        result = await execute_with_fallback(
            chain="research",
            tool="search",
            args={"query": "LangGraph patterns"},
        )
    """
    manager = get_fallback_manager()
    if not manager._initialized:
        await manager.initialize()

    return await manager.execute(chain, tool, args, prefer_server)
