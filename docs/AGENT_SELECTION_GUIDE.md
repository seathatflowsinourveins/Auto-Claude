# Agent Selection Decision Tree

**Generated**: January 2026 | **Source**: SWARM_PATTERNS_CATALOG.md synthesis | **Version**: 1.0

---

## Quick Reference Matrix

| Task Type | Primary Agent | Backup Agent | Topology |
|-----------|--------------|--------------|----------|
| Feature Development | @queen-coordinator | @hierarchical-coordinator | Hierarchical |
| Code Review | @refinement-agent | @code-reviewer | Peer Review |
| Bug Investigation | @mesh-coordinator | @researcher | Mesh |
| Architecture Design | @architecture-agent | @specification-agent | SPARC |
| Performance Tuning | @optimization-coordinator | @performance-monitor | Optimization |
| Security Audit | @security-auditor | @mcp-guard | Byzantine |
| Research/Exploration | @researcher | @collective-intelligence | Mesh |
| Testing | @tester | @qa-agent | Hierarchical |
| Documentation | @technical-writer | @coder | Single |
| Critical Decisions | @byzantine-coordinator | @raft-manager | Consensus |

---

## Decision Tree

```
START: What is the task?
│
├─► Is it a CRITICAL decision with multiple stakeholders?
│   │
│   ├─► YES: Do you need protection against malicious actors?
│   │   │
│   │   ├─► YES → @byzantine-coordinator (PBFT consensus)
│   │   │         └── Backup: @raft-manager
│   │   │
│   │   └─► NO → @raft-manager (leader election + log replication)
│   │             └── Backup: @collective-intelligence
│   │
│   └─► NO: Continue...
│
├─► Is it a DEVELOPMENT task?
│   │
│   ├─► Is it a NEW FEATURE?
│   │   │
│   │   ├─► Complex (3+ components) → @queen-coordinator
│   │   │   │                         └── Uses: Hierarchical topology
│   │   │   │                         └── Workers: @coder, @tester, @reviewer
│   │   │   │
│   │   │   └─► Also consider SPARC workflow:
│   │   │       1. @specification-agent → Requirements
│   │   │       2. @architecture-agent → Design
│   │   │       3. @pseudocode-agent → Algorithm
│   │   │       4. @refinement-agent → Review
│   │   │       5. @coder → Implementation
│   │   │
│   │   └─► Simple (1-2 files) → @coder (single agent)
│   │
│   ├─► Is it a BUG FIX?
│   │   │
│   │   ├─► Root cause unknown → @mesh-coordinator
│   │   │   │                    └── Parallel investigation
│   │   │   │                    └── Work stealing for load balance
│   │   │   │
│   │   │   └─► Pattern: INVESTIGATION workflow
│   │   │       1. Reproduce → 2. Investigate → 3. Fix → 4. Harden
│   │   │
│   │   └─► Root cause known → @coder (direct fix)
│   │
│   └─► Is it a REFACTOR?
│       │
│       └─► @hierarchical-coordinator
│           └── Pattern: REFACTOR workflow
│           1. Add New → 2. Migrate → 3. Remove Old → 4. Cleanup
│
├─► Is it a RESEARCH/EXPLORATION task?
│   │
│   ├─► Open-ended exploration → @collective-intelligence
│   │   │                        └── Distributed cognitive processes
│   │   │                        └── Knowledge integration
│   │   │
│   │   └─► Uses: Mesh topology (peer-to-peer discovery)
│   │
│   └─► Targeted research → @researcher
│                          └── Single-agent deep dive
│
├─► Is it an OPTIMIZATION task?
│   │
│   ├─► Performance → @optimization-coordinator
│   │   │            └── Profiling, caching, algorithms
│   │   │
│   ├─► Load Balancing → @load-balancer
│   │   │               └── Work stealing, DHT routing
│   │   │
│   └─► Resource Allocation → @queen-coordinator
│                            └── Compute/memory quotas
│
├─► Is it a SECURITY task?
│   │
│   ├─► Code Audit → @security-auditor
│   │               └── OWASP checks, vulnerability scan
│   │
│   ├─► Access Control → @mcp-guard
│   │                   └── Path validation, block patterns
│   │
│   └─► Consensus Required → @byzantine-coordinator
│                           └── Multi-party verification
│
└─► Is it a GITHUB/CI task?
    │
    ├─► PR Management → @pr-manager
    │                  └── Review, merge, conflict resolution
    │
    ├─► Issue Triage → @issue-manager
    │                 └── Labeling, assignment, prioritization
    │
    └─► Release → @release-manager
                 └── Versioning, changelog, deployment
```

---

## Topology Selection Guide

### Hierarchical (Queen-led)
**Use when:**
- Clear ownership and accountability needed
- Task has natural decomposition
- Fast decision-making required
- Resource allocation is critical

**Agents:** @queen-coordinator, @hierarchical-coordinator
**Pattern:** Command → Delegate → Execute → Report

```
        Queen
       /  |  \
     W1  W2  W3  (Workers)
```

### Mesh (Peer-to-peer)
**Use when:**
- Parallel exploration beneficial
- No single point of failure needed
- Emergent consensus acceptable
- Work stealing for load balance

**Agents:** @mesh-coordinator, @collective-intelligence
**Pattern:** Gossip → Vote → Consensus → Execute

```
    A ←→ B ←→ C
    ↕     ↕     ↕
    D ←→ E ←→ F
```

### Byzantine (Fault-tolerant)
**Use when:**
- Untrusted participants possible
- Critical decisions (financial, security)
- Audit trail required
- f < n/3 tolerance needed

**Agents:** @byzantine-coordinator
**Pattern:** Pre-prepare → Prepare → Commit → Execute

```
    L → V1 → PREPARE (2f+1)
    ↓    ↓
   V2 → V3 → COMMIT (2f+1)
```

### Raft (Strong Consistency)
**Use when:**
- Log replication needed
- Leader election required
- State machine consistency
- Membership changes

**Agents:** @raft-manager
**Pattern:** Election → Heartbeat → Replicate → Commit

```
    Leader → Follower1
       ↓  ↘
    Follower2  Follower3
```

---

## Memory Namespace Patterns

### Agent-Specific Keys
```
swarm/queen/status           - Queen health/state
swarm/queen/directives       - Active directives
swarm/worker-{id}/status     - Worker status
swarm/worker-{id}/tasks      - Assigned tasks
```

### Shared Keys
```
swarm/shared/resource-pool   - Available resources
swarm/shared/task-queue      - Pending tasks
swarm/shared/consensus       - Voting state
swarm/broadcast/alerts       - System alerts
```

### Coordination Keys
```
coordination/lock/{resource} - Distributed locks
coordination/election/state  - Leader election
coordination/heartbeat/{id}  - Liveness checks
```

---

## Quick Decision Shortcuts

| Keyword in Task | Recommended Agent |
|-----------------|-------------------|
| "implement", "build", "create" | @queen-coordinator or @coder |
| "fix", "debug", "investigate" | @mesh-coordinator |
| "review", "audit", "check" | @refinement-agent |
| "optimize", "performance" | @optimization-coordinator |
| "secure", "vulnerability" | @security-auditor |
| "design", "architecture" | @architecture-agent |
| "test", "verify" | @tester |
| "research", "explore" | @collective-intelligence |
| "decide", "consensus" | @byzantine-coordinator |
| "plan", "spec" | @specification-agent |

---

## Integration with V10 Hooks

| Hook | Swarm Integration |
|------|-------------------|
| `letta_sync_v2.py` | Memory persistence for swarm state |
| `mcp_guard_v2.py` | Access control for agent resources |
| `bash_guard.py` | Command validation for workers |
| `audit_log.py` | Decision trail for consensus |
| `memory_consolidate.py` | Sleeptime learning from swarm |

---

## Example Workflows

### Feature Development (SPARC + Hierarchical)
```
1. User Request → @specification-agent
   └── Output: requirements.md

2. Requirements → @architecture-agent
   └── Output: architecture.md, diagrams

3. Architecture → @queen-coordinator
   └── Spawns: @coder (x3), @tester
   └── Coordinates: parallel implementation

4. Implementation → @refinement-agent
   └── Code review, security check

5. Approved → @tester
   └── Integration tests, e2e

6. Complete → @queen-coordinator
   └── Merge, deploy, document
```

### Bug Investigation (Mesh + Work Stealing)
```
1. Bug Report → @mesh-coordinator
   └── Distribute to available workers

2. Workers investigate in parallel:
   ├── Worker A: Check logs
   ├── Worker B: Reproduce locally
   └── Worker C: Search codebase

3. First finding → Gossip to all workers
   └── Consensus on root cause

4. Fix → @coder
   └── Implement + test

5. Verify → @mesh-coordinator
   └── Confirm across all paths
```

---

## References

- SWARM_PATTERNS_CATALOG.md - Full pattern documentation
- GOALS_TRACKING.md - Implementation status
- ultimate-autonomous-platform-architecture-v2.md - System design
- v10_optimized/hooks/ - Hook implementations
