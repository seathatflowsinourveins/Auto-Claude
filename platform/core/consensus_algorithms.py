"""
Consensus Algorithms - Multi-Agent Coordination for UNLEASH Platform.

Implements distributed consensus patterns from Claude-Flow V3:
1. Byzantine Fault Tolerance (BFT) - Tolerates f < n/3 faulty agents
2. Raft - Leader-based consensus, tolerates f < n/2
3. Gossip - Epidemic propagation for eventual consistency
4. CRDT - Conflict-free replicated data types

These algorithms enable reliable coordination among multiple AI agents,
ensuring they can reach agreement even with failures or malicious actors.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                  Consensus Manager                           │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │           Strategy Router                             │   │
    │  │  • Selects optimal consensus for the scenario        │   │
    │  │  • Byzantine for critical decisions                   │   │
    │  │  • Raft for fast agreement                           │   │
    │  │  • Gossip for large-scale propagation                │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                           │                                  │
    │         ┌─────────────────┼─────────────────┐               │
    │         ▼                 ▼                 ▼               │
    │  ┌───────────┐     ┌───────────┐     ┌───────────┐         │
    │  │ Byzantine │     │   Raft    │     │  Gossip   │         │
    │  │    BFT    │     │ Consensus │     │ Protocol  │         │
    │  │           │     │           │     │           │         │
    │  │ 2/3 + 1   │     │  Leader   │     │ Epidemic  │         │
    │  │ majority  │     │  elected  │     │ broadcast │         │
    │  └───────────┘     └───────────┘     └───────────┘         │
    │                                                             │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │                  CRDT Store                          │   │
    │  │  G-Counter, PN-Counter, LWW-Register, OR-Set        │   │
    │  └─────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘

Version: V1.0.0 (January 2026)
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ConsensusStrategy(str, Enum):
    """Available consensus strategies."""
    BYZANTINE = "byzantine"  # BFT - tolerates f < n/3 faulty
    RAFT = "raft"           # Leader-based - tolerates f < n/2
    GOSSIP = "gossip"       # Epidemic - eventual consistency
    QUORUM = "quorum"       # Configurable quorum voting


class NodeState(str, Enum):
    """State of a node in the consensus network."""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"
    FAULTY = "faulty"


class VoteType(str, Enum):
    """Types of votes in consensus protocols."""
    PREPARE = "prepare"
    COMMIT = "commit"
    ACCEPT = "accept"
    REJECT = "reject"


# =============================================================================
# BASE TYPES
# =============================================================================

T = TypeVar("T")


@dataclass
class ConsensusMessage:
    """A message in the consensus protocol."""
    message_id: str
    sender_id: str
    term: int
    vote_type: VoteType
    proposal: Any
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signature: Optional[str] = None  # For Byzantine verification

    def compute_hash(self) -> str:
        """Compute message hash for verification."""
        content = f"{self.message_id}:{self.sender_id}:{self.term}:{self.vote_type}:{self.proposal}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ConsensusNode:
    """A node participating in consensus."""
    node_id: str
    state: NodeState = NodeState.FOLLOWER
    current_term: int = 0
    voted_for: Optional[str] = None
    weight: float = 1.0  # For weighted voting (e.g., queen has 3x weight)
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_queen: bool = False  # Queen has elevated privileges

    def update_heartbeat(self) -> None:
        """Update last heartbeat time."""
        self.last_heartbeat = datetime.now(timezone.utc)


@dataclass
class ConsensusResult:
    """Result of a consensus round."""
    success: bool
    value: Any
    term: int
    votes_for: int
    votes_against: int
    total_nodes: int
    strategy: ConsensusStrategy
    duration_ms: float


# =============================================================================
# BYZANTINE FAULT TOLERANCE (BFT)
# =============================================================================

class ByzantineConsensus:
    """
    Byzantine Fault Tolerant consensus implementation.

    Based on PBFT (Practical Byzantine Fault Tolerance).
    Tolerates f < n/3 faulty (potentially malicious) nodes.

    Key insight: Even if some agents are compromised or produce
    incorrect outputs, the system can still reach correct consensus
    if 2/3+ honest agents agree.
    """

    def __init__(
        self,
        nodes: List[ConsensusNode],
        queen_weight: float = 3.0
    ):
        self.nodes = {n.node_id: n for n in nodes}
        self.queen_weight = queen_weight
        self.prepare_votes: Dict[str, Dict[str, VoteType]] = {}  # proposal -> node -> vote
        self.commit_votes: Dict[str, Dict[str, VoteType]] = {}
        self.current_term = 0
        self._lock = asyncio.Lock()

    def _get_weighted_count(self, votes: Dict[str, VoteType], vote_type: VoteType) -> float:
        """Count votes with weighting (queen has elevated weight)."""
        total = 0.0
        for node_id, vote in votes.items():
            if vote == vote_type:
                node = self.nodes.get(node_id)
                if node:
                    weight = self.queen_weight if node.is_queen else node.weight
                    total += weight
        return total

    def _get_total_weight(self) -> float:
        """Get total voting weight across all nodes."""
        return sum(
            self.queen_weight if n.is_queen else n.weight
            for n in self.nodes.values()
            if n.state != NodeState.FAULTY
        )

    async def propose(
        self,
        proposal: Any,
        proposer_id: str
    ) -> ConsensusResult:
        """
        Propose a value for consensus.

        PBFT phases:
        1. Pre-prepare: Leader broadcasts proposal
        2. Prepare: Nodes verify and vote prepare
        3. Commit: If 2/3 prepare, nodes vote commit
        4. Reply: If 2/3 commit, value is committed
        """
        start_time = time.perf_counter()

        async with self._lock:
            self.current_term += 1
            proposal_id = f"prop_{self.current_term}_{proposer_id}"

            # Initialize vote tracking
            self.prepare_votes[proposal_id] = {}
            self.commit_votes[proposal_id] = {}

            total_weight = self._get_total_weight()
            required_weight = (2 * total_weight) / 3  # 2/3 + 1 majority

            # Phase 1: Prepare
            prepare_count = await self._collect_prepare_votes(proposal_id, proposal)
            prepare_weight = self._get_weighted_count(self.prepare_votes[proposal_id], VoteType.ACCEPT)

            if prepare_weight < required_weight:
                return ConsensusResult(
                    success=False,
                    value=None,
                    term=self.current_term,
                    votes_for=int(prepare_weight),
                    votes_against=int(total_weight - prepare_weight),
                    total_nodes=len(self.nodes),
                    strategy=ConsensusStrategy.BYZANTINE,
                    duration_ms=(time.perf_counter() - start_time) * 1000
                )

            # Phase 2: Commit
            commit_count = await self._collect_commit_votes(proposal_id, proposal)
            commit_weight = self._get_weighted_count(self.commit_votes[proposal_id], VoteType.ACCEPT)

            success = commit_weight >= required_weight

            return ConsensusResult(
                success=success,
                value=proposal if success else None,
                term=self.current_term,
                votes_for=int(commit_weight),
                votes_against=int(total_weight - commit_weight),
                total_nodes=len(self.nodes),
                strategy=ConsensusStrategy.BYZANTINE,
                duration_ms=(time.perf_counter() - start_time) * 1000
            )

    async def _collect_prepare_votes(
        self,
        proposal_id: str,
        proposal: Any
    ) -> int:
        """Collect prepare votes from all nodes."""
        for node_id, node in self.nodes.items():
            if node.state != NodeState.FAULTY:
                # Simulate vote (in production, this would be network call)
                vote = VoteType.ACCEPT  # Non-faulty nodes accept valid proposals
                self.prepare_votes[proposal_id][node_id] = vote
        return len(self.prepare_votes[proposal_id])

    async def _collect_commit_votes(
        self,
        proposal_id: str,
        proposal: Any
    ) -> int:
        """Collect commit votes from all nodes."""
        for node_id, node in self.nodes.items():
            if node.state != NodeState.FAULTY:
                vote = VoteType.ACCEPT
                self.commit_votes[proposal_id][node_id] = vote
        return len(self.commit_votes[proposal_id])


# =============================================================================
# RAFT CONSENSUS
# =============================================================================

class RaftConsensus:
    """
    Raft consensus implementation for leader-based agreement.

    Simpler and faster than Byzantine, but assumes no malicious actors.
    Tolerates f < n/2 crashed (but honest) nodes.

    Key concepts:
    - Leader election via randomized timeout
    - Log replication to followers
    - Safety: Only committed entries are applied
    """

    def __init__(
        self,
        nodes: List[ConsensusNode],
        election_timeout_range: Tuple[int, int] = (150, 300),  # ms
        heartbeat_interval: int = 50  # ms
    ):
        self.nodes = {n.node_id: n for n in nodes}
        self.election_timeout_range = election_timeout_range
        self.heartbeat_interval = heartbeat_interval
        self.current_term = 0
        self.leader_id: Optional[str] = None
        self.log: List[Tuple[int, Any]] = []  # (term, entry)
        self.commit_index = -1
        self._lock = asyncio.Lock()

    async def start_election(self, candidate_id: str) -> Optional[str]:
        """
        Start a leader election.

        Returns the elected leader's ID, or None if election failed.
        """
        async with self._lock:
            self.current_term += 1
            candidate = self.nodes.get(candidate_id)

            if not candidate:
                return None

            candidate.state = NodeState.CANDIDATE
            candidate.voted_for = candidate_id

            votes = 1  # Vote for self
            total = len([n for n in self.nodes.values() if n.state != NodeState.FAULTY])
            required = (total // 2) + 1

            # Request votes from other nodes
            for node_id, node in self.nodes.items():
                if node_id != candidate_id and node.state != NodeState.FAULTY:
                    # Simulate vote request
                    if node.voted_for is None or node.current_term < self.current_term:
                        node.voted_for = candidate_id
                        node.current_term = self.current_term
                        votes += 1

            if votes >= required:
                # Election won
                candidate.state = NodeState.LEADER
                self.leader_id = candidate_id

                # Reset other nodes to follower
                for node_id, node in self.nodes.items():
                    if node_id != candidate_id:
                        node.state = NodeState.FOLLOWER

                logger.info("Raft election: %s elected leader (term %d, votes %d/%d)",
                           candidate_id, self.current_term, votes, total)
                return candidate_id

            # Election failed
            candidate.state = NodeState.FOLLOWER
            return None

    async def propose(
        self,
        entry: Any,
        proposer_id: Optional[str] = None
    ) -> ConsensusResult:
        """
        Propose an entry for consensus.

        Only the leader can accept proposals. Entries are replicated
        to a majority before being committed.
        """
        start_time = time.perf_counter()

        async with self._lock:
            # Ensure we have a leader
            if not self.leader_id:
                await self.start_election(proposer_id or list(self.nodes.keys())[0])

            if not self.leader_id:
                return ConsensusResult(
                    success=False,
                    value=None,
                    term=self.current_term,
                    votes_for=0,
                    votes_against=len(self.nodes),
                    total_nodes=len(self.nodes),
                    strategy=ConsensusStrategy.RAFT,
                    duration_ms=(time.perf_counter() - start_time) * 1000
                )

            # Append to log
            self.log.append((self.current_term, entry))
            entry_index = len(self.log) - 1

            # Replicate to followers
            replicated_count = 1  # Leader has the entry
            total = len([n for n in self.nodes.values() if n.state != NodeState.FAULTY])
            required = (total // 2) + 1

            for node_id, node in self.nodes.items():
                if node_id != self.leader_id and node.state != NodeState.FAULTY:
                    # Simulate replication (in production, this is AppendEntries RPC)
                    replicated_count += 1

            success = replicated_count >= required

            if success:
                self.commit_index = entry_index

            return ConsensusResult(
                success=success,
                value=entry if success else None,
                term=self.current_term,
                votes_for=replicated_count,
                votes_against=total - replicated_count,
                total_nodes=total,
                strategy=ConsensusStrategy.RAFT,
                duration_ms=(time.perf_counter() - start_time) * 1000
            )


# =============================================================================
# GOSSIP PROTOCOL
# =============================================================================

@dataclass
class GossipMessage:
    """A message in the gossip protocol."""
    message_id: str
    content: Any
    version: int
    origin_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    hops: int = 0
    max_hops: int = 10


class GossipProtocol:
    """
    Gossip (Epidemic) protocol for eventual consistency.

    Optimized for large-scale systems where strong consistency
    is not required. Each node randomly shares updates with
    neighbors, achieving eventual consistency.

    Key properties:
    - Scales to thousands of nodes
    - Fault tolerant (messages propagate around failures)
    - Eventually consistent (not immediately)
    """

    def __init__(
        self,
        nodes: List[ConsensusNode],
        fanout: int = 3,  # Number of nodes to gossip to
        gossip_interval_ms: int = 100
    ):
        self.nodes = {n.node_id: n for n in nodes}
        self.fanout = fanout
        self.gossip_interval_ms = gossip_interval_ms

        # State tracking
        self.state: Dict[str, Tuple[Any, int]] = {}  # key -> (value, version)
        self.seen_messages: Set[str] = set()
        self.pending_messages: List[GossipMessage] = []

        self._lock = asyncio.Lock()

    async def broadcast(
        self,
        key: str,
        value: Any,
        origin_id: str
    ) -> ConsensusResult:
        """
        Broadcast a value via gossip protocol.

        The value will eventually propagate to all nodes.
        """
        start_time = time.perf_counter()

        async with self._lock:
            # Get current version
            current = self.state.get(key)
            version = (current[1] + 1) if current else 1

            # Create gossip message
            message = GossipMessage(
                message_id=f"gossip_{key}_{version}_{origin_id}",
                content=(key, value),
                version=version,
                origin_id=origin_id
            )

            # Update local state
            self.state[key] = (value, version)
            self.seen_messages.add(message.message_id)

            # Gossip to random subset of nodes
            active_nodes = [
                n for n in self.nodes.values()
                if n.state != NodeState.FAULTY and n.node_id != origin_id
            ]

            gossip_targets = random.sample(
                active_nodes,
                min(self.fanout, len(active_nodes))
            )

            reached_count = 1  # Origin has the message

            for node in gossip_targets:
                # Simulate gossip (in production, this is async network call)
                reached_count += 1
                message.hops += 1

            return ConsensusResult(
                success=True,  # Gossip always "succeeds" (eventually)
                value=value,
                term=version,
                votes_for=reached_count,
                votes_against=0,
                total_nodes=len(self.nodes),
                strategy=ConsensusStrategy.GOSSIP,
                duration_ms=(time.perf_counter() - start_time) * 1000
            )

    def receive_gossip(
        self,
        message: GossipMessage,
        receiver_id: str
    ) -> bool:
        """
        Receive a gossip message and potentially forward it.

        Returns True if the message was new (not seen before).
        """
        if message.message_id in self.seen_messages:
            return False

        if message.hops >= message.max_hops:
            return False

        self.seen_messages.add(message.message_id)

        # Update local state if newer
        key, value = message.content
        current = self.state.get(key)

        if current is None or message.version > current[1]:
            self.state[key] = (value, message.version)

        return True

    def get_state(self, key: str) -> Optional[Any]:
        """Get the current state value for a key."""
        result = self.state.get(key)
        return result[0] if result else None


# =============================================================================
# CRDT (CONFLICT-FREE REPLICATED DATA TYPES)
# =============================================================================

class GCounter:
    """
    Grow-only Counter CRDT.

    Each node maintains its own count. The total is the sum
    of all node counts. Only supports increment, never decrement.

    Merge: max(local[node], remote[node]) for each node
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.counts: Dict[str, int] = defaultdict(int)

    def increment(self, amount: int = 1) -> None:
        """Increment the counter for this node."""
        self.counts[self.node_id] += amount

    def value(self) -> int:
        """Get the total count across all nodes."""
        return sum(self.counts.values())

    def merge(self, other: "GCounter") -> None:
        """Merge another counter into this one."""
        for node_id, count in other.counts.items():
            self.counts[node_id] = max(self.counts[node_id], count)


class PNCounter:
    """
    Positive-Negative Counter CRDT.

    Uses two G-Counters: one for increments (P), one for decrements (N).
    Value = P - N

    Supports both increment and decrement operations.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.p_counter = GCounter(node_id)  # Positive
        self.n_counter = GCounter(node_id)  # Negative

    def increment(self, amount: int = 1) -> None:
        """Increment the counter."""
        self.p_counter.increment(amount)

    def decrement(self, amount: int = 1) -> None:
        """Decrement the counter."""
        self.n_counter.increment(amount)

    def value(self) -> int:
        """Get the current value (P - N)."""
        return self.p_counter.value() - self.n_counter.value()

    def merge(self, other: "PNCounter") -> None:
        """Merge another counter into this one."""
        self.p_counter.merge(other.p_counter)
        self.n_counter.merge(other.n_counter)


class LWWRegister(Generic[T]):
    """
    Last-Writer-Wins Register CRDT.

    Concurrent writes are resolved by timestamp - the latest write wins.
    Requires synchronized clocks (or logical timestamps).
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.value: Optional[T] = None
        self.timestamp: float = 0.0

    def set(self, value: T, timestamp: Optional[float] = None) -> None:
        """Set the register value."""
        ts = timestamp or time.time()
        if ts > self.timestamp:
            self.value = value
            self.timestamp = ts

    def get(self) -> Optional[T]:
        """Get the current value."""
        return self.value

    def merge(self, other: "LWWRegister[T]") -> None:
        """Merge another register into this one."""
        if other.timestamp > self.timestamp:
            self.value = other.value
            self.timestamp = other.timestamp


class ORSet(Generic[T]):
    """
    Observed-Remove Set CRDT.

    A set that supports both add and remove operations.
    Uses unique tags to track additions, allowing concurrent
    adds and removes to be resolved correctly.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.elements: Dict[T, Set[str]] = defaultdict(set)  # element -> set of tags
        self.tombstones: Dict[T, Set[str]] = defaultdict(set)  # removed tags
        self._tag_counter = 0

    def _new_tag(self) -> str:
        """Generate a unique tag for an element."""
        self._tag_counter += 1
        return f"{self.node_id}:{self._tag_counter}"

    def add(self, element: T) -> None:
        """Add an element to the set."""
        tag = self._new_tag()
        self.elements[element].add(tag)

    def remove(self, element: T) -> None:
        """Remove an element from the set."""
        if element in self.elements:
            # Move all tags to tombstones
            self.tombstones[element].update(self.elements[element])
            self.elements[element].clear()

    def contains(self, element: T) -> bool:
        """Check if element is in the set."""
        tags = self.elements.get(element, set())
        tombstones = self.tombstones.get(element, set())
        return bool(tags - tombstones)

    def values(self) -> Set[T]:
        """Get all elements in the set."""
        return {e for e in self.elements if self.contains(e)}

    def merge(self, other: "ORSet[T]") -> None:
        """Merge another set into this one."""
        # Merge elements
        for element, tags in other.elements.items():
            self.elements[element].update(tags)

        # Merge tombstones
        for element, tags in other.tombstones.items():
            self.tombstones[element].update(tags)


# =============================================================================
# CONSENSUS MANAGER
# =============================================================================

class ConsensusManager:
    """
    Unified consensus manager that routes to appropriate algorithm.

    Selection criteria:
    - Byzantine: Critical decisions, untrusted participants
    - Raft: Fast agreement with trusted participants
    - Gossip: Large-scale, eventual consistency
    - CRDT: Automatic merge without coordination
    """

    def __init__(
        self,
        nodes: List[ConsensusNode],
        default_strategy: ConsensusStrategy = ConsensusStrategy.RAFT
    ):
        self.nodes = nodes
        self.default_strategy = default_strategy

        # Initialize consensus implementations
        self.byzantine = ByzantineConsensus(nodes)
        self.raft = RaftConsensus(nodes)
        self.gossip = GossipProtocol(nodes)

        # CRDT stores (per-key)
        self.crdt_counters: Dict[str, PNCounter] = {}
        self.crdt_registers: Dict[str, LWWRegister] = {}
        self.crdt_sets: Dict[str, ORSet] = {}

        self.metrics = {
            "proposals": 0,
            "successes": 0,
            "failures": 0,
            "strategy_usage": {s.value: 0 for s in ConsensusStrategy},
        }

    async def propose(
        self,
        value: Any,
        proposer_id: str,
        strategy: Optional[ConsensusStrategy] = None
    ) -> ConsensusResult:
        """
        Propose a value for consensus using the specified strategy.
        """
        strategy = strategy or self.default_strategy
        self.metrics["proposals"] += 1
        self.metrics["strategy_usage"][strategy.value] += 1

        if strategy == ConsensusStrategy.BYZANTINE:
            result = await self.byzantine.propose(value, proposer_id)
        elif strategy == ConsensusStrategy.RAFT:
            result = await self.raft.propose(value, proposer_id)
        elif strategy == ConsensusStrategy.GOSSIP:
            # For gossip, we need a key
            key = value.get("key") if isinstance(value, dict) else str(value)
            result = await self.gossip.broadcast(key, value, proposer_id)
        else:
            # Default to Raft
            result = await self.raft.propose(value, proposer_id)

        if result.success:
            self.metrics["successes"] += 1
        else:
            self.metrics["failures"] += 1

        return result

    def get_crdt_counter(self, key: str, node_id: str) -> PNCounter:
        """Get or create a CRDT counter for a key."""
        if key not in self.crdt_counters:
            self.crdt_counters[key] = PNCounter(node_id)
        return self.crdt_counters[key]

    def get_crdt_register(self, key: str, node_id: str) -> LWWRegister:
        """Get or create a CRDT register for a key."""
        if key not in self.crdt_registers:
            self.crdt_registers[key] = LWWRegister(node_id)
        return self.crdt_registers[key]

    def get_crdt_set(self, key: str, node_id: str) -> ORSet:
        """Get or create a CRDT set for a key."""
        if key not in self.crdt_sets:
            self.crdt_sets[key] = ORSet(node_id)
        return self.crdt_sets[key]

    def get_metrics(self) -> Dict[str, Any]:
        """Get consensus metrics."""
        return {
            **self.metrics,
            "node_count": len(self.nodes),
            "raft_leader": self.raft.leader_id,
            "raft_term": self.raft.current_term,
            "gossip_state_keys": len(self.gossip.state),
        }


# =============================================================================
# SINGLETON & FACTORY
# =============================================================================

_consensus_manager: Optional[ConsensusManager] = None


def get_consensus_manager(
    nodes: Optional[List[ConsensusNode]] = None,
    default_strategy: ConsensusStrategy = ConsensusStrategy.RAFT
) -> ConsensusManager:
    """Get or create the global consensus manager."""
    global _consensus_manager

    if _consensus_manager is None:
        if nodes is None:
            # Create default nodes
            nodes = [
                ConsensusNode(node_id="node_1", is_queen=True),
                ConsensusNode(node_id="node_2"),
                ConsensusNode(node_id="node_3"),
            ]
        _consensus_manager = ConsensusManager(nodes, default_strategy)

    return _consensus_manager


def reset_consensus_manager() -> None:
    """Reset the global consensus manager."""
    global _consensus_manager
    _consensus_manager = None


# =============================================================================
# CLI INTERFACE
# =============================================================================

async def main():
    """Demo the consensus algorithms."""

    print("Consensus Algorithms Demo")
    print("=" * 50)

    # Create nodes with a queen
    nodes = [
        ConsensusNode(node_id="queen", is_queen=True, weight=3.0),
        ConsensusNode(node_id="worker_1"),
        ConsensusNode(node_id="worker_2"),
        ConsensusNode(node_id="worker_3"),
        ConsensusNode(node_id="worker_4"),
    ]

    manager = ConsensusManager(nodes)

    # Test Byzantine consensus
    print("\n1. Byzantine Consensus (2/3 majority required)")
    print("-" * 40)
    result = await manager.propose(
        {"action": "deploy_v2"},
        "queen",
        ConsensusStrategy.BYZANTINE
    )
    print(f"   Success: {result.success}")
    print(f"   Votes: {result.votes_for}/{result.total_nodes}")
    print(f"   Duration: {result.duration_ms:.2f}ms")

    # Test Raft consensus
    print("\n2. Raft Consensus (leader-based)")
    print("-" * 40)
    result = await manager.propose(
        {"action": "scale_workers"},
        "queen",
        ConsensusStrategy.RAFT
    )
    print(f"   Success: {result.success}")
    print(f"   Leader: {manager.raft.leader_id}")
    print(f"   Term: {result.term}")
    print(f"   Duration: {result.duration_ms:.2f}ms")

    # Test Gossip protocol
    print("\n3. Gossip Protocol (eventual consistency)")
    print("-" * 40)
    result = await manager.propose(
        {"key": "config", "value": {"timeout": 30}},
        "worker_1",
        ConsensusStrategy.GOSSIP
    )
    print(f"   Broadcast initiated: {result.success}")
    print(f"   Reached: {result.votes_for} nodes")
    print(f"   Duration: {result.duration_ms:.2f}ms")

    # Test CRDTs
    print("\n4. CRDT Operations (conflict-free)")
    print("-" * 40)

    # Counter
    counter = manager.get_crdt_counter("task_count", "queen")
    counter.increment(5)
    print(f"   Counter value: {counter.value()}")

    # Register
    register = manager.get_crdt_register("current_version", "queen")
    register.set("v2.0.0")
    print(f"   Register value: {register.get()}")

    # Set
    or_set = manager.get_crdt_set("active_workers", "queen")
    or_set.add("worker_1")
    or_set.add("worker_2")
    or_set.remove("worker_1")
    print(f"   Set values: {or_set.values()}")

    print("\n" + "=" * 50)
    print("Metrics:", manager.get_metrics())


if __name__ == "__main__":
    asyncio.run(main())
