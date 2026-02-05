# Gap14: Agent Mesh Dynamic Messaging Protocol

> **Status**: RESOLVED (V66)
> **Created**: 2026-02-05
> **Implementation**: `platform/adapters/agent_mesh.py`
> **Tests**: `platform/tests/test_agent_mesh_messaging.py` (40 tests)

---

## Problem Statement

The Agent Mesh at `platform/adapters/agent_mesh.py` had static topology with no dynamic messaging protocol. Agents could not communicate dynamically - messages were point-to-point with no routing, queuing, or delivery confirmation.

### Original Limitations

1. **No message routing** - Messages could only be sent to specific agent IDs
2. **No priority handling** - All messages treated equally
3. **No delivery confirmation** - No way to know if message was received
4. **No retry mechanism** - Failed deliveries were lost
5. **No dead letter queue** - No handling for undeliverable messages
6. **No consensus integration** - Critical decisions lacked validation

---

## Solution

Implemented a full dynamic messaging protocol with:

### 1. Priority Message Queues

```python
class MessagePriority(Enum):
    CRITICAL = 0  # System-level, consensus, emergencies
    HIGH = 1      # Task assignments, escalations
    NORMAL = 2    # Standard operations
    LOW = 3       # Heartbeats, status updates
    BACKGROUND = 4  # Non-urgent, batch operations
```

Messages are dequeued in priority order (CRITICAL first).

### 2. Routing Strategies

```python
class RoutingStrategy(Enum):
    DIRECT = "direct"       # Send to specific agent
    BROADCAST = "broadcast"  # Send to all agents
    ROUND_ROBIN = "round_robin"  # Distribute across agents of same role
    LEAST_LOADED = "least_loaded"  # Send to least busy agent
    PRIORITY = "priority"    # Based on agent priority/tier
    CONSENSUS = "consensus"  # Route through consensus protocol
```

### 3. Delivery Confirmation

```python
class DeliveryStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    IN_FLIGHT = "in_flight"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"
    EXPIRED = "expired"
```

Messages can require acknowledgment (`requires_ack=True`).

### 4. Dead Letter Queue

Failed messages after max retries are moved to dead letter queue with error context.

### 5. CVT Consensus Integration

Critical operations can be validated through CVT consensus protocol:

```python
await mesh.request_consensus(
    proposer_id="coder-1",
    action_type="deploy",
    payload={"version": "2.0"},
    critical=True,
)
```

---

## Key Components

### MeshMessage (Enhanced)

```python
@dataclass
class MeshMessage:
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str]
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    delivery_status: DeliveryStatus = DeliveryStatus.PENDING
    routing_strategy: RoutingStrategy = RoutingStrategy.DIRECT
    requires_ack: bool = False
    retry_count: int = 0
    max_retries: int = 3
    ttl_seconds: int = 300
    checksum: str  # For integrity verification
```

### PriorityMessageQueue

Per-agent queue with:
- Multi-level priority queues (5 levels)
- Automatic expiration handling
- Dead letter queue support
- Statistics tracking

### MessageRouter

Routes messages based on strategy:
- Tracks round-robin indices per role
- Calculates load scores for least-loaded routing
- Considers priority weights for priority routing
- Tracks routing statistics

### AgentMesh (Enhanced)

New methods:
- `send_routed()` - Send with automatic routing
- `acknowledge()` - ACK/NACK a message
- `request_consensus()` - Request CVT consensus
- `validate_with_consensus()` - Fast-path validation

---

## Usage Examples

### Basic Direct Send

```python
await mesh.send(
    sender_id="router-1",
    recipient_id="coder-1",
    message_type=MessageType.TASK_ASSIGN,
    payload={"task": "implement feature"},
    priority=MessagePriority.HIGH,
    requires_ack=True,
)
```

### Routed Send (Auto-select Agent)

```python
msg_id, targets = await mesh.send_routed(
    sender_id="router-1",
    message_type=MessageType.TASK_ASSIGN,
    payload={"task": "implement feature"},
    target_role=AgentRole.CODER,
    routing_strategy=RoutingStrategy.LEAST_LOADED,
)
```

### Task Assignment with Consensus

```python
assigned = await mesh.assign_task(
    task_id="task-001",
    description="Deploy to production",
    preferred_role=AgentRole.CODER,
    use_consensus=True,
    routing_strategy=RoutingStrategy.PRIORITY,
)
```

### Message Acknowledgment

```python
async def handler(msg):
    # Process message
    result = await process(msg)
    # Acknowledge
    await mesh.acknowledge(msg, success=True, response_payload={"result": result})
```

---

## Statistics

The mesh tracks comprehensive statistics:

```python
stats = mesh.get_stats()
# {
#     "messages_sent": 150,
#     "messages_received": 145,
#     "messages_acknowledged": 140,
#     "messages_failed": 5,
#     "messages_retried": 10,
#     "tasks_assigned": 50,
#     "tasks_completed": 48,
#     "escalations": 3,
#     "consensus_requests": 5,
#     "consensus_reached": 5,
#     "routing_stats": {DIRECT: 100, LEAST_LOADED: 30, ...},
#     "queue_stats": {...},
# }
```

---

## Test Coverage

40 tests covering:
- MeshMessage creation, serialization, checksum verification
- PriorityMessageQueue operations and priority ordering
- MessageRouter strategies (direct, broadcast, round-robin, least-loaded, priority)
- AgentMesh messaging (send, routed, broadcast, request-response)
- Task management (assignment, completion, escalation)
- Context sharing
- Statistics tracking
- Edge cases and error handling

---

## Integration with A2A Patterns

The implementation follows patterns from `docs/AGENT_TO_AGENT_PATTERNS.md`:

1. **Hybrid Communication**: Task tool for spawning, memory for coordination
2. **Memory-Based Coordination**: Messages stored in queues, results in memory
3. **Priority Handling**: Critical messages processed first
4. **Consensus for Critical Decisions**: CVT integration for validation

---

## Files Modified

- `platform/adapters/agent_mesh.py` - Main implementation (enhanced with ~1000 lines)
- `platform/adapters/__init__.py` - Added exports for new components
- `platform/tests/test_agent_mesh_messaging.py` - New test file (40 tests)
- `docs/gap-resolution/14-AGENT-MESH-MESSAGING.md` - This document

---

## Related Components

- `platform/core/orchestration/cvt_consensus.py` - CVT consensus protocol
- `docs/AGENT_TO_AGENT_PATTERNS.md` - A2A pattern reference
- `platform/adapters/a2a_protocol_adapter.py` - Google A2A adapter
