# SWARM INTELLIGENCE PATTERNS CATALOG

**Generated**: January 2026 | **Source**: ruvnet-claude-flow analysis | **Agents**: 96+

---

## EXECUTIVE SUMMARY

This catalog documents proven swarm coordination patterns discovered in the Claude-Flow ecosystem, synthesized for integration with the Ultimate Autonomous Platform.

### Pattern Categories

| Category | Agent Count | Primary Use Case |
|----------|-------------|------------------|
| **Hive Mind** | 5 | Collective intelligence, queen-led coordination |
| **Consensus** | 7 | CRDT, Raft, Byzantine fault tolerance |
| **Swarm** | 3 | Hierarchical, mesh, adaptive topologies |
| **Optimization** | 5 | Performance, load balancing, resources |
| **GitHub** | 12 | PR/Issue/Release automation |
| **SPARC** | 4 | Specification → Architecture → Pseudocode → Refinement |
| **Core** | 5 | Coder, planner, researcher, reviewer, tester |

---

## TIER 1: HIVE MIND PATTERNS (Production Ready)

### 1.1 Queen Coordinator 👑
**File**: `hive-mind/queen-coordinator.md`
**Priority**: Critical
**Color**: Gold

```
Purpose: Sovereign orchestrator for hierarchical operations
Key Functions:
├── Strategic Command & Control
├── Resource Allocation (compute units, memory quotas)
├── Succession Planning (heir: collective-intelligence)
└── Hive Coherence Maintenance

Governance Modes:
├── Hierarchical: Direct command chains, rapid decisions
├── Democratic: Weighted voting, consensus building
└── Emergency: Absolute authority, bypass consensus

Memory Pattern:
  namespace: "coordination"
  keys:
    - swarm/queen/status
    - swarm/shared/royal-directives
    - swarm/shared/resource-allocation
    - swarm/queen/hive-health
```

### 1.2 Collective Intelligence Coordinator 🧠
**File**: `hive-mind/collective-intelligence-coordinator.md`
**Priority**: Critical
**Color**: Purple

```
Purpose: Neural nexus for distributed cognitive processes
Key Functions:
├── Memory Synchronization Protocol
├── Consensus Building (Byzantine fault tolerance)
├── Cognitive Load Balancing
└── Knowledge Integration

Coordination Patterns:
├── Hierarchical: Command hierarchy, accountability
├── Mesh: Peer-to-peer, emergent consensus
└── Adaptive: Dynamic topology based on task

Memory Requirement:
  - Write collective state every 30 seconds
  - Update consensus metrics continuously
  - Share knowledge graph
  - Log decision history
```

### 1.3 Swarm Memory Manager 💾
**File**: `hive-mind/swarm-memory-manager.md`
**Priority**: Critical
**Color**: Blue

```
Purpose: Distributed memory management and consistency
Key Functions:
├── Multi-level Caching (L1/L2/L3)
├── Predictive Prefetching
├── CRDT for Conflict-free Replication
└── Vector Clocks for Causality

Operations:
├── Batch Read (cache results for other agents)
├── Atomic Write (with conflict detection)
└── Synchronization Protocol

Performance Metrics (every 60 seconds):
  - operations_per_second
  - cache_hit_rate
  - sync_latency_ms
  - memory_usage_mb
  - active_connections
```

---

## TIER 2: CONSENSUS PATTERNS (Distributed Systems)

### 2.1 Byzantine Coordinator 🛡️
**File**: `consensus/byzantine-coordinator.md`
**Priority**: High
**Color**: Purple (#9C27B0)

```
Purpose: Byzantine fault-tolerant consensus with malicious detection
Key Functions:
├── PBFT Protocol (Pre-prepare, Prepare, Commit)
├── Malicious Actor Detection
├── Cryptographic Message Authentication
├── View Change Coordination
└── Attack Mitigation

Security Features:
├── Threshold signature schemes
├── Zero-knowledge proofs for vote verification
├── Replay attack prevention (sequence numbers)
└── DoS protection (rate limiting)

Fault Tolerance: f < n/3 malicious nodes
```

### 2.2 Raft Manager 🗳️
**File**: `consensus/raft-manager.md`
**Priority**: High
**Color**: Blue (#2196F3)

```
Purpose: Strong consistency through leader election and log replication
Key Functions:
├── Leader Election (randomized timeouts)
├── Log Replication (append entries protocol)
├── Consistency Management
├── Membership Changes
└── Recovery Coordination

Protocol:
├── Follower → Candidate (timeout)
├── Candidate → Leader (majority votes)
├── Leader → Follower (higher term discovered)

Guarantees:
├── Election Safety: One leader per term
├── Leader Append-Only: Never overwrites
├── Log Matching: Same index = same entry
└── State Machine Safety: Applied entries committed
```

### 2.3 Mesh Coordinator 🌐
**File**: `swarm/mesh-coordinator.md`
**Priority**: High
**Color**: Cyan (#00BCD4)

```
Purpose: Peer-to-peer decentralized coordination
Key Functions:
├── Distributed Decision Making
├── Gossip Algorithm (information dissemination)
├── Work Stealing (load balancing)
└── DHT Task Distribution

Network Topology:
    A ←→ B ←→ C
    ↕     ↕     ↕
    D ←→ E ←→ F
    ↕     ↕     ↕
    G ←→ H ←→ I

Task Distribution Strategies:
├── Work Stealing: Idle nodes steal from busy
├── DHT (Consistent Hashing): Route by task ID
└── Auction-Based: Capability matching + scoring
```

---

## TIER 3: SPARC METHODOLOGY (Development Flow)

### 3.1 Specification Agent 📋
**File**: `sparc/specification.md`
**Priority**: High
**SPARC Phase**: 1

```
Purpose: Requirements analysis and testable specifications
Deliverables:
├── Functional Requirements (FR-xxx)
├── Non-Functional Requirements (NFR-xxx)
├── Use Case Definitions
├── Acceptance Criteria (Gherkin)
└── Data Model Specification

Validation Checklist:
├── All requirements are testable
├── Acceptance criteria are clear
├── Edge cases documented
├── Performance metrics defined
├── Security requirements specified
└── Stakeholders approved
```

### 3.2 Architecture Agent 🏗️
**File**: `sparc/architecture.md`
**Priority**: High
**SPARC Phase**: 2

```
Purpose: System design and component architecture
Deliverables:
├── Component Diagrams
├── Sequence Diagrams
├── Data Flow Architecture
├── API Contracts
└── Integration Points
```

### 3.3 Pseudocode Agent 📝
**File**: `sparc/pseudocode.md`
**Priority**: High
**SPARC Phase**: 3

```
Purpose: Algorithm design before implementation
Deliverables:
├── Algorithm Pseudocode
├── Edge Case Handling
├── Complexity Analysis
└── Test Strategy
```

### 3.4 Refinement Agent 🔧
**File**: `sparc/refinement.md`
**Priority**: High
**SPARC Phase**: 4

```
Purpose: Code review and iterative improvement
Deliverables:
├── Code Review Feedback
├── Refactoring Suggestions
├── Performance Optimizations
└── Security Hardening
```

---

## TIER 4: AUTO-CLAUDE INTEGRATION

### 4.1 Planner Agent (Auto-Claude)
**File**: `auto-claude/prompts/planner.md`
**Priority**: High

```
Purpose: Subtask-based implementation planning
Workflow Types:
├── FEATURE: Backend → Worker → Frontend → Integration
├── REFACTOR: Add New → Migrate → Remove Old → Cleanup
├── INVESTIGATION: Reproduce → Investigate → Fix → Harden
├── MIGRATION: Prepare → Test → Execute → Cleanup
└── SIMPLE: Minimal overhead, just subtasks

Key Outputs:
├── implementation_plan.json
├── project_index.json
├── context.json
├── init.sh
└── build-progress.txt

Verification Types:
├── command: CLI verification
├── api: REST endpoint testing
├── browser: UI rendering checks
├── e2e: Full flow verification
└── manual: Human judgment required
```

---

## SWARM TOPOLOGY SELECTION GUIDE

### When to Use Each Topology

| Scenario | Recommended Topology | Rationale |
|----------|---------------------|-----------|
| Complex feature development | Hierarchical (Queen) | Clear command chain, accountability |
| Research/exploration | Mesh (Distributed) | Parallel discovery, fault tolerance |
| Critical decisions | Byzantine | Malicious actor protection |
| Log/state replication | Raft | Strong consistency guarantees |
| High-load processing | Work Stealing | Dynamic load balancing |
| Knowledge synthesis | Collective Intelligence | Consensus building |

### Memory Namespace Convention

```
namespace: "coordination"

Key Patterns:
├── swarm/[agent-type]/status     - Individual agent status
├── swarm/[agent-type]/metrics    - Performance metrics
├── swarm/shared/*                - Cross-agent shared data
├── swarm/broadcast/*             - Pub/sub messages
└── swarm/worker-*/               - Individual worker states
```

---

## INTEGRATION CHECKLIST

### Minimum Viable Swarm
- [ ] Queen Coordinator deployed
- [ ] Memory Manager initialized
- [ ] Shared memory namespace configured
- [ ] Status heartbeat every 30 seconds
- [ ] Graceful shutdown protocol

### Production Swarm
- [ ] Byzantine fault tolerance enabled
- [ ] Load balancing configured
- [ ] Performance monitoring active
- [ ] Recovery procedures tested
- [ ] Security hardening complete

---

## PRACTICAL INTEGRATION EXAMPLES

### Example 1: Queen Coordinator with Letta Memory

```python
"""
Queen Coordinator Integration Pattern
Uses Letta SDK for persistent state management
"""
from letta_client import Letta
from typing import Dict, Any
import json

class QueenCoordinator:
    """
    Sovereign orchestrator implementing hierarchical command.
    Integrates with Letta memory for cross-session persistence.
    """

    def __init__(self, letta_url: str = "http://localhost:8283"):
        self.letta = Letta(base_url=letta_url)
        self.namespace = "coordination"
        self.agent_id = self._get_or_create_agent()

    def _get_or_create_agent(self) -> str:
        """Get or create the queen agent."""
        agents = self.letta.agents.list()
        for agent in agents:
            if agent.name == "queen-coordinator":
                return agent.id

        # Create with 4 memory blocks
        agent = self.letta.agents.create(
            name="queen-coordinator",
            memory_blocks=[
                {"label": "royal-directives", "value": "", "limit": 4000},
                {"label": "resource-allocation", "value": "", "limit": 3000},
                {"label": "hive-health", "value": "", "limit": 2000},
                {"label": "decision-log", "value": "", "limit": 5000},
            ],
            enable_sleeptime=True,
        )
        return agent.id

    def issue_directive(self, directive: Dict[str, Any]) -> str:
        """Issue a royal directive to the swarm."""
        message = f"[DIRECTIVE] {json.dumps(directive)}"
        response = self.letta.agents.messages.send(
            agent_id=self.agent_id,
            messages=[{"role": "user", "content": message}]
        )
        return response.messages[-1].content

    def allocate_resources(self, worker_id: str, quota: Dict[str, int]):
        """Allocate compute/memory quotas to a worker."""
        allocation = {
            "worker": worker_id,
            "compute_units": quota.get("compute", 1),
            "memory_mb": quota.get("memory", 512),
            "timestamp": datetime.now().isoformat()
        }
        return self.issue_directive({"type": "resource_allocation", **allocation})
```

### Example 2: Mesh Coordinator with Work Stealing

```python
"""
Mesh Coordinator with Work Stealing Pattern
Peer-to-peer task distribution with load balancing
"""
import asyncio
from dataclasses import dataclass
from typing import List, Optional
import random

@dataclass
class Task:
    id: str
    priority: int
    payload: dict
    assigned_to: Optional[str] = None

class MeshCoordinator:
    """
    Decentralized peer-to-peer coordinator.
    Implements work stealing for load balancing.
    """

    def __init__(self, node_id: str, peers: List[str]):
        self.node_id = node_id
        self.peers = peers
        self.task_queue: List[Task] = []
        self.stolen_tasks: int = 0

    async def gossip_state(self):
        """Broadcast state to random subset of peers (gossip protocol)."""
        # Select sqrt(n) random peers for gossip
        num_peers = max(1, int(len(self.peers) ** 0.5))
        selected = random.sample(self.peers, min(num_peers, len(self.peers)))

        state = {
            "node_id": self.node_id,
            "queue_depth": len(self.task_queue),
            "load": self._calculate_load()
        }

        # In production, this would send via websocket/OSC
        for peer in selected:
            await self._send_gossip(peer, state)

    async def work_stealing(self):
        """Steal work from overloaded peers when idle."""
        if len(self.task_queue) > 2:
            return  # Already have enough work

        # Find busiest peer
        peer_loads = await self._collect_peer_loads()
        busiest = max(peer_loads.items(), key=lambda x: x[1], default=(None, 0))

        if busiest[1] > 5:  # Threshold for stealing
            stolen = await self._steal_from(busiest[0])
            if stolen:
                self.task_queue.append(stolen)
                self.stolen_tasks += 1

    def _calculate_load(self) -> float:
        """Calculate current load factor."""
        return len(self.task_queue) + sum(t.priority for t in self.task_queue) * 0.1
```

### Example 3: Byzantine Consensus Integration

```python
"""
Byzantine Fault Tolerant Consensus
Implements PBFT-style voting for critical decisions
"""
from enum import Enum
from typing import Dict, Set
import hashlib

class VotePhase(Enum):
    PRE_PREPARE = 1
    PREPARE = 2
    COMMIT = 3
    EXECUTED = 4

class ByzantineConsensus:
    """
    PBFT consensus for malicious actor tolerance.
    Requires f < n/3 honest nodes.
    """

    def __init__(self, node_id: str, total_nodes: int):
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.f = (total_nodes - 1) // 3  # Max faulty nodes
        self.quorum = 2 * self.f + 1     # Votes needed

        self.view = 0
        self.sequence = 0
        self.votes: Dict[str, Set[str]] = {}

    def propose(self, request: dict) -> str:
        """Leader proposes a request (PRE-PREPARE)."""
        request_hash = hashlib.sha256(str(request).encode()).hexdigest()[:16]

        message = {
            "phase": VotePhase.PRE_PREPARE.name,
            "view": self.view,
            "sequence": self.sequence,
            "request_hash": request_hash,
            "request": request
        }

        self.votes[request_hash] = {self.node_id}
        self.sequence += 1
        return request_hash

    def vote(self, request_hash: str, voter_id: str, phase: VotePhase) -> bool:
        """Record a vote and check for quorum."""
        if request_hash not in self.votes:
            self.votes[request_hash] = set()

        self.votes[request_hash].add(voter_id)

        if len(self.votes[request_hash]) >= self.quorum:
            return True  # Quorum reached
        return False

    def is_committed(self, request_hash: str) -> bool:
        """Check if request has achieved consensus."""
        return len(self.votes.get(request_hash, set())) >= self.quorum
```

### Example 4: Hook Integration with Swarm

```python
"""
Integrating V10 Hooks with Swarm Coordination
Uses hooks to enforce swarm security policies
"""
import json
from pathlib import Path

class SwarmSecurityHook:
    """
    PreToolUse hook that validates swarm operations.
    Ensures agents only access their designated resources.
    """

    NAMESPACE_PERMISSIONS = {
        "queen-coordinator": ["swarm/queen/*", "swarm/shared/*", "swarm/broadcast/*"],
        "worker": ["swarm/worker-{id}/*", "swarm/shared/tasks"],
        "memory-manager": ["swarm/*"],  # Full access
    }

    def validate_memory_access(self, agent_id: str, key: str) -> tuple[str, str]:
        """Validate an agent's memory access request."""
        agent_type = self._get_agent_type(agent_id)
        allowed_patterns = self.NAMESPACE_PERMISSIONS.get(agent_type, [])

        for pattern in allowed_patterns:
            # Simple pattern matching (production would use proper glob)
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                if key.startswith(prefix):
                    return "allow", f"Access allowed by pattern: {pattern}"
            elif "{id}" in pattern:
                concrete = pattern.replace("{id}", agent_id)
                if key.startswith(concrete.replace("/*", "")):
                    return "allow", f"Access allowed for agent-specific key"

        return "deny", f"Agent {agent_type} not authorized for key: {key}"

    def _get_agent_type(self, agent_id: str) -> str:
        """Extract agent type from ID."""
        if "queen" in agent_id:
            return "queen-coordinator"
        elif "memory" in agent_id:
            return "memory-manager"
        return "worker"
```

### Example 5: SPARC Workflow Orchestration

```python
"""
SPARC Methodology Orchestration
Specification → Architecture → Pseudocode → Refinement → Coding
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class SPARCPhase(Enum):
    SPECIFICATION = 1
    ARCHITECTURE = 2
    PSEUDOCODE = 3
    REFINEMENT = 4
    CODING = 5

@dataclass
class SPARCWorkflow:
    """
    Orchestrates the SPARC development methodology.
    Each phase produces artifacts consumed by the next.
    """
    feature_name: str
    current_phase: SPARCPhase = SPARCPhase.SPECIFICATION

    # Artifacts
    spec_document: Optional[str] = None
    architecture_diagram: Optional[str] = None
    pseudocode: Optional[str] = None
    review_feedback: Optional[str] = None
    final_code: Optional[str] = None

    def advance_phase(self) -> bool:
        """Advance to next phase if current is complete."""
        validations = {
            SPARCPhase.SPECIFICATION: self.spec_document is not None,
            SPARCPhase.ARCHITECTURE: self.architecture_diagram is not None,
            SPARCPhase.PSEUDOCODE: self.pseudocode is not None,
            SPARCPhase.REFINEMENT: self.review_feedback is not None,
        }

        if validations.get(self.current_phase, True):
            next_phase = SPARCPhase(self.current_phase.value + 1)
            self.current_phase = next_phase
            return True
        return False

    def get_agent_for_phase(self) -> str:
        """Return the specialized agent for current phase."""
        agents = {
            SPARCPhase.SPECIFICATION: "@specification-agent",
            SPARCPhase.ARCHITECTURE: "@architecture-agent",
            SPARCPhase.PSEUDOCODE: "@pseudocode-agent",
            SPARCPhase.REFINEMENT: "@refinement-agent",
            SPARCPhase.CODING: "@coder-agent",
        }
        return agents[self.current_phase]
```

---

## REFERENCES

### Source Files
- `ruvnet-claude-flow/.claude/agents/hive-mind/`
- `ruvnet-claude-flow/.claude/agents/consensus/`
- `ruvnet-claude-flow/.claude/agents/swarm/`
- `ruvnet-claude-flow/.claude/agents/sparc/`
- `auto-claude/apps/backend/prompts/`

### Related Documentation
- `ultimate-autonomous-platform-architecture-v2.md`
- `GOALS_TRACKING.md`
- V10 Hook Implementations in `v10_optimized/hooks/`
