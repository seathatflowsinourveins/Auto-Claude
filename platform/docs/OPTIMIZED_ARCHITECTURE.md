# Ultimate Autonomous Platform - Optimized Architecture v10.1

## Executive Summary

This document synthesizes best practices from official Claude SDKs, MCP specifications, Letta memory systems, and Graphiti temporal knowledge graphs into an optimized architecture for the Ultimate Autonomous Platform.

**Key Principles:**
1. **Simplicity First** - Avoid over-engineering; complexity only when necessary
2. **Agent Loop Pattern** - Context → Thought → Action → Observation
3. **Memory-Aware** - Core memory (in-context) + External memory (retrieved)
4. **Temporal Awareness** - Track what changed and when
5. **Production-Ready** - Security, observability, graceful degradation

---

## 1. Core Architecture Pattern

### 1.1 The Agent Loop (Official Claude Pattern)

```
┌─────────────────────────────────────────────────────────────┐
│                       AGENT LOOP                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐ │
│   │ CONTEXT │ -> │  THINK  │ -> │   ACT   │ -> │ OBSERVE │ │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘ │
│        ^                                             │       │
│        └─────────────────────────────────────────────┘       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Context**: What the agent knows (system prompt + memory + task)
**Think**: Extended reasoning (ultrathink up to 128K tokens)
**Act**: Execute tools (MCP servers, bash, file operations)
**Observe**: Process results and update memory

### 1.2 Harness Architecture

The "Harness" wraps agents with:

```python
@dataclass
class AgentHarness:
    """Core agent orchestration harness."""

    # Identity
    agent_id: str
    agent_type: str  # "coordinator", "worker", "specialist"

    # Capabilities
    tools: List[Tool]
    prompts: Dict[str, str]
    skills: List[Skill]

    # Memory
    core_memory: MemoryBlock      # Always in context
    archival_memory: VectorStore  # Retrieved on demand
    temporal_graph: GraphitiClient  # Knowledge graph

    # State
    context_window: int = 200_000
    current_context_usage: int = 0
    checkpoints: List[Checkpoint] = field(default_factory=list)
```

---

## 2. Memory Architecture (Letta + Graphiti Pattern)

### 2.1 Three-Tier Memory System

```
┌────────────────────────────────────────────────────────────────┐
│                    MEMORY ARCHITECTURE                          │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TIER 1: CORE MEMORY (Always In-Context)                       │
│  ├── system_persona: Agent identity and behavior               │
│  ├── user_context: Current user preferences/state              │
│  ├── task_state: Current task progress and goals               │
│  └── working_memory: Recent observations (last 10 turns)       │
│                                                                 │
│  TIER 2: ARCHIVAL MEMORY (Vector-Retrieved)                    │
│  ├── episodic: Past conversations and interactions             │
│  ├── semantic: Domain knowledge and facts                      │
│  └── procedural: How-to knowledge and patterns                 │
│                                                                 │
│  TIER 3: TEMPORAL KNOWLEDGE GRAPH (Graphiti)                   │
│  ├── entities: People, projects, concepts                      │
│  ├── relationships: How entities connect                       │
│  ├── temporal_facts: What was true and when                    │
│  └── evolution: How knowledge changed over time                │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 Memory Block Pattern (Letta Best Practice)

```python
class MemoryBlock:
    """Structured section of agent's context window."""

    def __init__(self, label: str, initial_content: str, max_tokens: int = 2000):
        self.label = label
        self.content = initial_content
        self.max_tokens = max_tokens
        self.last_updated = datetime.utcnow()

    def update(self, new_content: str) -> bool:
        """Update block content within token limit."""
        if count_tokens(new_content) <= self.max_tokens:
            self.content = new_content
            self.last_updated = datetime.utcnow()
            return True
        return False

    def append(self, addition: str) -> bool:
        """Append to block, compressing if needed."""
        combined = f"{self.content}\n{addition}"
        if count_tokens(combined) <= self.max_tokens:
            self.content = combined
            self.last_updated = datetime.utcnow()
            return True
        # Trigger summarization
        self.content = summarize_for_tokens(combined, self.max_tokens)
        return True
```

### 2.3 Temporal Knowledge Graph (Graphiti Pattern)

```python
class TemporalFact:
    """A fact with temporal validity."""

    subject: str
    predicate: str
    object: str
    valid_from: datetime
    valid_to: Optional[datetime]  # None = still valid
    confidence: float
    source: str  # Where this fact came from

    def is_current(self) -> bool:
        """Check if fact is currently valid."""
        now = datetime.utcnow()
        return self.valid_from <= now and (self.valid_to is None or self.valid_to > now)


class TemporalGraph:
    """Knowledge graph with temporal awareness."""

    async def add_fact(self, fact: TemporalFact) -> None:
        """Add fact, potentially invalidating old facts."""
        # Check for contradictions
        existing = await self.find_facts(fact.subject, fact.predicate)
        for old_fact in existing:
            if old_fact.object != fact.object and old_fact.is_current():
                # Invalidate old fact
                old_fact.valid_to = fact.valid_from
                await self.update_fact(old_fact)

        await self.store_fact(fact)

    async def query_at_time(self, timestamp: datetime, subject: str) -> List[TemporalFact]:
        """Get all facts about subject valid at specific time."""
        return await self.query(
            subject=subject,
            valid_from_lte=timestamp,
            valid_to_gte_or_null=timestamp
        )
```

---

## 3. MCP Server Architecture (Official Best Practices)

### 3.1 Naming Conventions

| Component | Python | TypeScript |
|-----------|--------|------------|
| Server | `{service}_mcp` | `{service}-mcp-server` |
| Tool | `{service}_{action}` | `{service}_{action}` |
| Resource | `{service}://{path}` | `{service}://{path}` |

### 3.2 Tool Design Patterns

```python
# GOOD: Clear, specific, with annotations
@mcp_tool(
    name="github_create_issue",
    title="Create GitHub Issue",
    description="Create a new issue in a GitHub repository",
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def create_issue(
    repo: str,           # owner/repo format
    title: str,          # Issue title (required)
    body: str = "",      # Issue body (optional)
    labels: List[str] = None  # Labels to apply
) -> IssueResult:
    ...

# BAD: Vague, overpowered
@mcp_tool(name="do_github")
async def do_github(action: str, data: dict) -> Any:
    ...
```

### 3.3 Response Format (Hybrid JSON + Markdown)

```python
def format_tool_response(data: Any, for_human: bool = False) -> str:
    """Format response for both LLM and human consumption."""

    if for_human:
        # Markdown for readability
        return f"""## Result

**Status**: {data.status}
**Created**: {data.created_at}

### Details
{data.description}
"""
    else:
        # JSON for programmatic use
        return json.dumps(data, indent=2)


# CHARACTER_LIMIT with graceful truncation
CHARACTER_LIMIT = 25_000

def truncate_response(content: str) -> str:
    """Truncate long responses gracefully."""
    if len(content) <= CHARACTER_LIMIT:
        return content

    truncated = content[:CHARACTER_LIMIT - 100]
    return f"{truncated}\n\n[TRUNCATED: {len(content) - CHARACTER_LIMIT + 100} characters omitted]"
```

### 3.4 Pagination Pattern

```python
@dataclass
class PaginatedResponse(Generic[T]):
    """Standard pagination response."""

    items: List[T]
    total: int
    limit: int
    offset: int
    has_more: bool
    next_offset: Optional[int]

    @classmethod
    def create(cls, all_items: List[T], limit: int = 20, offset: int = 0) -> "PaginatedResponse[T]":
        items = all_items[offset:offset + limit]
        has_more = offset + limit < len(all_items)
        return cls(
            items=items,
            total=len(all_items),
            limit=limit,
            offset=offset,
            has_more=has_more,
            next_offset=offset + limit if has_more else None
        )
```

---

## 4. Transport Layer (2025-2026 Best Practices)

### 4.1 Transport Selection Matrix

| Transport | Use Case | Latency | Complexity |
|-----------|----------|---------|------------|
| **Stdio** | Local agents, IDE integration | ~1ms | Low |
| **Streamable HTTP** | Remote servers, cloud deployment | ~50ms | Medium |
| **HTTP+SSE (deprecated)** | Legacy compatibility only | ~50ms | Medium |

### 4.2 Streamable HTTP (Recommended)

```python
from mcp.server import McpServer
from mcp.transport import StreamableHttpTransport

# Stateful server with sessions
server = McpServer(
    transport=StreamableHttpTransport(
        host="0.0.0.0",
        port=8080,
        enable_sessions=True,
        session_timeout=3600,
        allowed_hosts=["localhost", "127.0.0.1", "myapp.local"]
    )
)

# DNS rebinding protection (critical for localhost)
server.configure_security(
    dns_rebind_protection=True,
    cors_origins=["http://localhost:3000"],
    rate_limit=100  # requests per minute
)
```

---

## 5. Auto-Claude Cooperation Pattern

### 5.1 Multi-Agent Coordination

```
┌─────────────────────────────────────────────────────────────────┐
│                    COORDINATOR AGENT                             │
│  (Owns task decomposition, progress tracking, result synthesis)  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│    │   WORKER 1   │  │   WORKER 2   │  │   WORKER 3   │        │
│    │  (Research)  │  │   (Build)    │  │   (Test)     │        │
│    └──────────────┘  └──────────────┘  └──────────────┘        │
│           │                 │                 │                  │
│           └─────────────────┼─────────────────┘                  │
│                             │                                    │
│                    ┌────────▼────────┐                          │
│                    │ SHARED MEMORY   │                          │
│                    │ (Graphiti KG)   │                          │
│                    └─────────────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Session Handoff Protocol

```python
@dataclass
class SessionHandoff:
    """Context for transferring work between agents/sessions."""

    # Task identity
    task_id: str
    parent_session: str

    # Progress state
    completed_steps: List[str]
    current_step: str
    remaining_steps: List[str]

    # Knowledge transfer
    key_decisions: List[Decision]
    discovered_constraints: List[str]
    open_questions: List[str]

    # File state
    modified_files: List[str]
    created_files: List[str]

    # Memory snapshot
    core_memory_export: Dict[str, str]
    relevant_facts: List[TemporalFact]

    def to_context_string(self) -> str:
        """Generate context string for next session."""
        return f"""## Session Handoff

### Progress
Completed: {', '.join(self.completed_steps)}
Current: {self.current_step}
Remaining: {', '.join(self.remaining_steps)}

### Key Decisions
{chr(10).join(f'- {d.description}: {d.choice} (reason: {d.rationale})' for d in self.key_decisions)}

### Constraints Discovered
{chr(10).join(f'- {c}' for c in self.discovered_constraints)}

### Open Questions
{chr(10).join(f'- {q}' for q in self.open_questions)}

### Files Changed
Modified: {', '.join(self.modified_files)}
Created: {', '.join(self.created_files)}
"""
```

---

## 6. Checkpoint Pattern (Claude Code 2.0)

### 6.1 Automatic Checkpointing

```python
class CheckpointManager:
    """Manage agent state checkpoints for safe experimentation."""

    def __init__(self, storage_path: Path, max_checkpoints: int = 10):
        self.storage_path = storage_path
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[Checkpoint] = []

    async def create_checkpoint(self, agent_state: AgentState, description: str) -> Checkpoint:
        """Create a checkpoint before risky operations."""
        checkpoint = Checkpoint(
            id=str(uuid4()),
            timestamp=datetime.utcnow(),
            description=description,
            file_snapshot=await self._snapshot_files(agent_state.modified_files),
            memory_snapshot=agent_state.core_memory.export(),
            git_commit=await self._get_git_head()
        )

        self.checkpoints.append(checkpoint)

        # Prune old checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            old = self.checkpoints.pop(0)
            await self._cleanup_checkpoint(old)

        await self._persist_checkpoint(checkpoint)
        return checkpoint

    async def rollback(self, checkpoint_id: str) -> bool:
        """Rollback to a specific checkpoint."""
        checkpoint = self._find_checkpoint(checkpoint_id)
        if not checkpoint:
            return False

        # Restore files
        await self._restore_files(checkpoint.file_snapshot)

        # Restore memory
        await self._restore_memory(checkpoint.memory_snapshot)

        # Git reset if needed
        if checkpoint.git_commit:
            await self._git_reset(checkpoint.git_commit)

        return True

    async def rewind(self, steps: int = 1) -> Optional[Checkpoint]:
        """Rewind to N checkpoints ago."""
        if steps > len(self.checkpoints):
            return None

        target = self.checkpoints[-steps]
        await self.rollback(target.id)
        return target
```

---

## 7. Security Architecture

### 7.1 Permission Levels

```python
class PermissionLevel(Enum):
    """Tool permission levels."""

    READ_ONLY = "read_only"      # Can read, cannot modify
    READ_WRITE = "read_write"    # Can read and modify
    DESTRUCTIVE = "destructive"  # Can delete, requires confirmation
    ADMIN = "admin"              # Full access, audit logged

@dataclass
class ToolPermission:
    """Permission configuration for a tool."""

    tool_name: str
    level: PermissionLevel
    allowed_paths: List[str] = field(default_factory=list)
    denied_paths: List[str] = field(default_factory=list)
    requires_confirmation: bool = False
    audit_log: bool = True
```

### 7.2 Input Validation

```python
def validate_tool_input(tool: Tool, input_data: Dict[str, Any]) -> ValidationResult:
    """Validate tool input against schema and security rules."""

    # Schema validation (Pydantic/Zod)
    try:
        validated = tool.input_schema.model_validate(input_data)
    except ValidationError as e:
        return ValidationResult(valid=False, errors=e.errors())

    # Path traversal check
    for key, value in input_data.items():
        if isinstance(value, str) and ".." in value:
            return ValidationResult(valid=False, errors=[f"Path traversal detected in {key}"])

    # Command injection check
    dangerous_chars = [";", "|", "&", "`", "$", "(", ")"]
    for key, value in input_data.items():
        if isinstance(value, str) and any(c in value for c in dangerous_chars):
            return ValidationResult(valid=False, errors=[f"Potential injection in {key}"])

    return ValidationResult(valid=True, validated_data=validated)
```

---

## 8. Observability

### 8.1 Structured Logging

```python
import structlog

logger = structlog.get_logger()

# Log all tool invocations
async def invoke_tool(tool: Tool, input_data: Dict[str, Any]) -> ToolResult:
    log = logger.bind(
        tool_name=tool.name,
        input_size=len(json.dumps(input_data)),
        session_id=current_session_id()
    )

    log.info("tool_invocation_start")
    start_time = time.time()

    try:
        result = await tool.execute(input_data)
        duration = time.time() - start_time

        log.info(
            "tool_invocation_success",
            duration_ms=duration * 1000,
            output_size=len(str(result))
        )
        return result

    except Exception as e:
        log.error(
            "tool_invocation_error",
            error_type=type(e).__name__,
            error_message=str(e),
            duration_ms=(time.time() - start_time) * 1000
        )
        raise
```

### 8.2 Metrics Collection

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

TOOL_INVOCATIONS = Counter(
    'uap_tool_invocations_total',
    'Total tool invocations',
    ['tool_name', 'status']
)

TOOL_LATENCY = Histogram(
    'uap_tool_latency_seconds',
    'Tool invocation latency',
    ['tool_name'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
)

MEMORY_USAGE = Gauge(
    'uap_memory_context_tokens',
    'Current context window token usage',
    ['agent_id']
)

ACTIVE_SESSIONS = Gauge(
    'uap_active_sessions',
    'Number of active agent sessions'
)
```

---

## 9. File Organization

```
unleash/
├── v10_optimized/
│   ├── core/                    # Core platform modules
│   │   ├── agent.py            # AgentHarness implementation
│   │   ├── memory.py           # Memory system (Letta pattern)
│   │   ├── temporal_graph.py   # Graphiti integration
│   │   └── coordinator.py      # Multi-agent coordination
│   │
│   ├── mcp_servers/            # MCP server implementations
│   │   ├── __init__.py
│   │   ├── base.py             # Base MCP server class
│   │   └── transports/         # Transport implementations
│   │
│   ├── tools/                  # Tool definitions
│   │   ├── file_ops.py
│   │   ├── bash.py
│   │   └── research.py
│   │
│   ├── scripts/                # Operational scripts
│   │   ├── install.py
│   │   ├── health.py
│   │   ├── autoscale.py
│   │   └── config.py
│   │
│   ├── docs/                   # Documentation
│   │   ├── OPTIMIZED_ARCHITECTURE.md  # This file
│   │   └── official/           # Official SDK docs
│   │
│   └── tests/                  # Test suites
│       ├── unit/
│       ├── integration/
│       └── e2e/
```

---

## 10. Implementation Checklist

### Phase 1: Foundation (Complete)
- [x] Health check system
- [x] Circuit breakers
- [x] Rate limiters
- [x] Configuration management
- [x] Secrets management
- [x] Autoscaling metrics
- [x] Platform installer

### Phase 2: Memory System (In Progress)
- [ ] Core memory blocks (Letta pattern)
- [ ] Archival memory (vector store)
- [ ] Temporal knowledge graph (Graphiti)
- [ ] Memory synchronization

### Phase 3: Agent Coordination
- [ ] Session handoff protocol
- [ ] Multi-agent coordinator
- [ ] Checkpoint/rollback system
- [ ] Shared memory access

### Phase 4: Production Hardening
- [ ] Security audit
- [ ] Load testing
- [ ] Chaos engineering
- [ ] Documentation completion

---

## References

1. [MCP Best Practices](./official/sdks/awesome-claude-skills/mcp-builder/reference/mcp_best_practices.md)
2. [MCP TypeScript SDK](./official/sdks/alternative-langs/mcp-typescript/docs/server.md)
3. [Letta Memory Documentation](https://docs.letta.com/guides/agents/memory)
4. [Graphiti Temporal Knowledge Graphs](https://github.com/getzep/graphiti)
5. [Claude Agent SDK Workshop](https://www.youtube.com/watch?v=TqC1qOfiVcQ)

---

*Document Version: 10.1*
*Last Updated: 2026-01-19*
*Author: Claude AI (Ralph Loop Iteration 11)*
