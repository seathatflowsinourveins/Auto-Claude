# Cross-Session Bootstrap (2026-01-25 Corrected)

**Auto-loaded at session start for instant SDK access**

> **IMPORTANT**: V30-V40 architecture was INCORRECT. Use 8-layer, 34 SDK architecture.

## Correct Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         UNLEASH SDK ARCHITECTURE                            │
│                              34 PRODUCTION SDKs                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  L8: KNOWLEDGE       │ graphrag, pyribs                                    │
│  L7: PROCESSING      │ aider, ast-grep, crawl4ai, firecrawl               │
│  L6: SAFETY          │ guardrails-ai, llm-guard, nemo-guardrails          │
│  L5: OBSERVABILITY   │ langfuse, opik, arize-phoenix, deepeval, ragas, promptfoo │
│  L4: REASONING       │ dspy, serena                                        │
│  L3: STRUCTURED      │ instructor, baml, outlines, pydantic-ai            │
│  L2: MEMORY          │ letta, zep, mem0                                    │
│  L1: ORCHESTRATION   │ temporal-python, langgraph, claude-flow, crewai, autogen │
│  L0: PROTOCOL        │ mcp-python-sdk, fastmcp, litellm, anthropic, openai-sdk │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Import Reference (8-Layer)

### P0 BACKBONE (Always Loaded)
```python
# L0: Protocol
from anthropic import Anthropic
import litellm
from fastmcp import FastMCP

# L1: Orchestration
from langgraph.graph import StateGraph
from temporalio import workflow

# L2: Memory  
from letta import create_client

# L4: Reasoning
import dspy

# L5: Observability
import langfuse
```

### P1 CORE (Primary Capabilities)
```python
# L1: Multi-agent
from crewai import Agent, Crew
from autogen import AssistantAgent

# L2: Memory extensions
from zep_cloud.client import AsyncZep
from mem0 import Memory

# L3: Structured output
from instructor import from_anthropic
from pydantic_ai import Agent as PydanticAgent

# L5: Evaluation
import opik
from deepeval.metrics import HallucinationMetric
```

### P2 ADVANCED (Specialized)
```python
# L6: Safety
from guardrails import Guard
from llm_guard import scan_prompt

# L7: Processing
from crawl4ai import AsyncWebCrawler

# L8: Knowledge
from graphrag import GraphRAG
from ribs.archives import GridArchive  # MAP-Elites
```

## Critical Integration Patterns

### Extended Thinking (128K ULTRATHINK)
```python
response = client.messages.create(
    model="claude-opus-4-5-20251101",
    thinking={"type": "enabled", "budget_tokens": 128000}
)
```

### LangGraph Production Checkpointing
```python
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
pool = ConnectionPool(conn_string, max_size=10)
checkpointer = PostgresSaver(pool)
# SECURITY: Use langgraph>=3.0 (CVE-2025-64439)
```

### QDAIF for LLM Creative Exploration
```python
from ribs.schedulers import Scheduler
for iteration in range(1000):
    solutions = scheduler.ask()
    objectives = llm_evaluate_quality(solutions)
    measures = llm_extract_diversity_features(solutions)
    scheduler.tell(objectives, measures)
```

## SDK Location
`Z:/insider/AUTO CLAUDE/unleash/sdks/` - **34 SDKs** (not 185+)

## Key Reference Documents
- `architecture_2026_definitive` ← Master reference (Serena memory)
- `sdk_cleanup_2026_01_24` ← Cleanup decisions
- `audit/SDK_KEEP_ARCHITECTURE_2026.md` ← Full layer definitions

## Project-Specific Configurations

| Project | Primary SDKs |
|---------|--------------|
| **WITNESS** | pyribs, langgraph, letta, opik, fastmcp, graphiti |
| **TRADING** | temporal-python, guardrails-ai, langfuse, deepeval, hindsight |
| **UNLEASH** | All P0 + P1 + serena + aider + agent-harness |

## Enhancement Loop Patterns (Cycles 2-6, Jan 2026 Week 4)

### Top Patterns Integrated

#### 1. Context Engineering > Prompt Engineering
Memory is a "core architectural primitive" - 2026 paradigm shift
```python
context = await memory.retrieve(task)
context = await tools.lazy_load(needed_tools)
context = await episodic.recall(similar_tasks)
response = llm.generate(engineered_context)
```

#### 2. Lazy MCP Loading (Claude Code 2.1)
```json
{"mcp_servers": {"touchdesigner": {"lazy": true}, "comfyui": {"lazy": true}}}
```

#### 3. Four-Layer Memory Stack
1. **Persona** - Agent behavior | 2. **Toolbox** - Functions
3. **Conversational** - History | 4. **Workflow** - Patterns

#### 4. QDAIF (Quality-Diversity through AI Feedback)
```python
fitness, measures = await llm_evaluate(candidate)
scheduler.tell(objectives, measures)
```

#### 5. Graphiti Temporal Knowledge Graphs
```python
await graph.add_episode(content, timestamp, entity_types)
results = await graph.search(query, temporal_filter)
```

#### 6. Agent Harness Pattern
```python
while not complete:
    thought, action, observation = think → act → observe
    context = update(context, thought, observation)
```

#### 7. Hindsight Memory
```python
await memory.record_action(action, outcome)
advice = await memory.advise_action(proposed)
```

### MCP Security (CRITICAL - Updated Jan 25, 2026)

**PATCHED (Safe to use)**:
- mcp-server-git >= Dec 18, 2025 (CVE-2025-68143/68144/68145 fixed)

**BLOCKED (Do NOT use)**:
- aws-mcp-server (CVE-2025-5277 - Command Injection RCE)

```json
// Secure MCP Configuration
{
  "git": {"sandbox": true, "require_version": ">=2025.12.18"},
  "aws": {"blocked": true, "reason": "CVE-2025-5277"},
  "filesystem": {"sandbox": true, "allowed_paths": ["project/"]}
}
```json
// CVE-2025-68143/68144/68145 - Never chain Git + Filesystem MCP
{"git": {"sandbox": true, "whitelist_repos": ["internal/*"]}}
{"filesystem": {"sandbox": true, "allowed_paths": ["project/"]}}
```

### Key Research References
- arXiv:2601.01082 - Discount Model Search for high-dim QD
- arXiv:2601.01743 - AI Agent Systems Survey (Jan 2026)
- arXiv:2601.04703 - M-ASK Framework (decoupled multi-agent)
- arXiv:2502.06975 - Episodic Memory Position Paper

### Related Memory Files
- `enhancement_loop_unified_jan2026_week4` - Full consolidation
- `mcp_security_findings_jan2026` - CVE mitigations
- `cycle3_qd_agent_memory_jan2026` - QD + Agent SDK
- `cycle4_letta_mem0_comfyui_jan2026` - Memory layers
- `cycle5_graphiti_multiagent_jan2026` - Temporal graphs
- `witness_creative_integration_jan2026` - WITNESS patterns

---

## Cycle 9 Patterns (January 25, 2026)

### Context Engineering Paradigm Shift
- **Memory as Core Primitive**: Not afterthought, but architectural foundation
- **Lazy Context Loading**: Load only what's needed, when needed
- **Progressive Disclosure**: Reveal complexity only on request

### Multi-Agent Orchestration (100% Actionable)
```python
# From arXiv:2511.15755 - Single-agent: 1.7% actionable, Multi-agent: 100%
pipeline = {
    "proposer": Agent(role="Generate solutions"),
    "critic": Agent(role="Find flaws"),
    "synthesizer": Agent(role="Merge insights"),
    "validator": Agent(role="Verify actionability")
}
```

### MCP Agent-to-Agent Protocol (arXiv:2601.13671)
- Peer-to-peer agent communication via shared MCP state
- Emergent coordination without central orchestrator
- Fault tolerance through MCP state recovery

### ### Cycle 11 Security Critical (January 25, 2026)

**MCP Vulnerability Status**:
- 43% of servers vulnerable to command injection
- OAuth 2.1 + PKCE **MANDATORY** for remote servers
- 437,000+ installations affected by CVEs

**Agent Harness Pattern** (Official Anthropic):
```
Context → Thought → Action → Observation → Update Context → Loop
```

**MAGMA Multi-Graph Memory** (arXiv:2601.03236):
- Semantic graph (meaning)
- Temporal graph (time)
- Causal graph (cause-effect)
- Entity graph (named entities)

---

### Cycle 10 Platform Patterns (January 25, 2026)

**Claude Code 2.1.17**:
- Task management with dependency tracking
- MCP lazy loading (dynamic tool fetching)
- Opus 4/4.1 deprecated → Use Opus 4.5

**LangGraph 1.0 Mental Model**:
```
LangGraph = State Machine + LLM Brain
State → Node → Edge → Conditional Edge → Checkpoint
```

**pyribs DMS**: Discount Model Search for high-dimensional QD

---

### Application Patterns
| Project | Pattern | Application |
|---------|---------|-------------|
| WITNESS | Multi-agent | Aesthetic critic separate from generator |
| TRADING | 100% Actionable | Multi-agent trade signal validation |
| UNLEASH | Meta-patterns | Apply to Claude Code architecture |

---

### Cycle 12 Architecture, Code Generation & Auditing (January 25, 2026)

**System Architecture Patterns (Official)**:
```
CQRS: Segregate read/write for independent optimization
Event Sourcing: State as events for complete audit trail
Event-Driven: 72% of organizations use EDA
Microservices: Saga, Outbox, CDC, Database per Service
```

**Code Generation Optimization (Addy Osmani 2026)**:
```python
# "90% of Claude Code is written by Claude Code itself"
# BUT: Critical thinking remains key
# Pattern: Context Engineering > Prompt Engineering
requirements = ["Clear direction", "Context", "Oversight"]
```

**Auditing/Validation 2026 Trends**:
```
The 72.8% Paradox:
- 72.8% prioritize AI-powered testing
- #1 concern: "Does AI-generated code demand MORE testing?"
- Answer: YES - non-deterministic outputs need statistical validation
```

**Opik Evaluation Metrics** (50+ metrics):
```python
# LLM-as-Judge
from opik.evaluation.metrics import (
    Hallucination, AnswerRelevance, ContextPrecision,
    AgentTaskCompletionJudge, TrajectoryAccuracy
)
# Heuristic (no LLM required)
from opik.evaluation.metrics import (
    ROUGE, BERTScore, PromptInjection, Readability
)
```

**Problem Decomposition Pattern**:
```python
# Multi-agent decomposition (100% actionable)
decomposition = {
    "analyzer": "Break problem into subproblems",
    "proposer": "Generate solutions for each",
    "critic": "Find flaws in solutions",
    "synthesizer": "Merge into cohesive solution",
    "validator": "Verify against requirements"
}
```

---

### Cycle 13 Reasoning, Decomposition & Strategic Planning (January 25, 2026)

**Society of Thought** (arXiv:2601.10825v1):
```
Enhanced reasoning = Multi-agent-like INTERNAL simulation
├── Deliberate diversification of perspectives
├── Internal debate among implicit agents
└── Convergence through simulated consensus
```

**Tree of Thoughts (ToT)** - IBM Official:
```python
# Structured exploration with backtracking
branches = expand_thoughts(problem)
for branch in sorted(branches, key=score, reverse=True):
    result = explore(branch)
    if result.is_valid(): return result
    # Backtrack and try next branch
```

**Framework of Thoughts (FoT)** - ICLR 2026:
```
Unifies: CoT + ToT + GoT
├── Dynamic (not static) reasoning structures
├── Adapts to unseen problem types
└── Optimized for cost/runtime
```

**Divide and Conquer Decomposition**:
```
1. DIVIDE: Break into independent subproblems
2. CONQUER: Solve recursively
3. COMBINE: Merge solutions
```

**ADR Template** (AWS/Google/UK Gov):
```markdown
# ADR-XXX: [Title]
Status: Proposed | Accepted | Deprecated
Context: [Issue motivating decision]
Options: [Alternatives considered]
Decision: [Chosen option + justification]
Consequences: [Positive and negative]
```

---

### Cycle 14 Code Optimization & Verification (January 25, 2026)

**LLM Code Optimization** (StarCoder2 + POLO):
```python
# StarCoder2: 20.1% better code smell reduction than humans
# Targets: long_methods, duplicates, dead_code, god_classes

# POLO Framework (IJCAI 2025): Project-Level Optimization
# 1. Project analysis (dependency graph, hot paths)
# 2. Candidate ranking (impact score, risk assessment)
# 3. Transformation with rollback capability
```

**Formal Verification Stack**:
```bash
# Miri: Rust undefined behavior detection (POPL 2026)
MIRIFLAGS="-Zmiri-strict-provenance" cargo miri test

# Hypothesis: Property-based testing
@given(st.lists(st.integers()))
def test_sort_idempotent(lst):
    assert sorted(sorted(lst)) == sorted(lst)
```

**Claude Code Quality Gates** (PostToolUse Hooks):
```python
# ~/.claude/hooks/post_edit_verification.py
checks = [
    ("TypeScript", ["npx", "tsc", "--noEmit"]),
    ("ESLint", ["npx", "eslint"]),
    ("Prettier", ["npx", "prettier", "--check"]),
]
```

**TDD in Agentic Coding**:
```
RED-GREEN-REFACTOR keeps changes small and reversible
├── RED: Test MUST fail before implementation
├── GREEN: ONLY code needed to pass the test
└── REFACTOR: Small changes with test safety net
```

**Verification Metrics Targets**:
| Metric | Target |
|--------|--------|
| Mutation Score | ≥80% |
| Branch Coverage | ≥90% |
| Property Tests | ≥50/module |
| Type Coverage | 100% strict |
| Miri Clean | 0 UB |

---

### Cycle 15 Profiling, Debugging & Observability (January 25, 2026)

**Python Production Profiling** (py-spy):
```bash
# No code modification, attach to running process
py-spy record -o profile.svg --pid 12345
py-spy top --pid 12345  # Live view
```

**Python 3.13+ sys.monitoring API**:
- Replaces cProfile's 2006-era pstat format
- Native hooks without monkey-patching

**AI-First Debugging Paradigm** (Jeffrey Ullman):
```
"You can't debug a billion parameters like software"
├── Cluster similar errors automatically
├── Explain stack traces in natural language
├── Generate minimal reproduction cases
└── Suggest fixes with confidence scores
```

**Observability 2026** (ClickHouse):
```
"Observability is a data analytics problem, not monitoring"
├── Three pillars UNIFIED (not siloed)
├── Focus: "unknown unknowns" (not just known failures)
└── OpenTelemetry as universal standard
```

**Trace-Aware Debugging** (DebuggAI 2026):
```
OpenTelemetry spans + eBPF probes → AI Analysis
→ Faulty commit identification
→ Automatic fix suggestion
```

**Key Tools 2026**:
| Tool | Purpose |
|------|---------|
| py-spy | Production profiling |
| OpenTelemetry | Unified observability |
| Dash0 | AI-native root cause |
| DebuggAI | Trace replay debugging |

---

### Cycle 16 Resilience, Fault Tolerance & Error Handling (January 25, 2026)

**Resilience Triad**:
```
RETRY + TIMEOUT + CIRCUIT BREAKER
  ↓        ↓           ↓
Handles  Prevents   Stops
temp     infinite   cascading
errors   waiting    failures
```

**Circuit Breaker States**:
```python
# pybreaker implementation
breaker = pybreaker.CircuitBreaker(
    fail_max=5,        # CLOSED → OPEN after 5 failures
    reset_timeout=30   # OPEN → HALF-OPEN after 30s
)
```

**Retry with Tenacity**:
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(ConnectionError)
)
```

**LLM Agent Error Handling** (Instructor):
```python
from instructor.core.exceptions import (
    IncompleteOutputException,  # Truncated response
    ValidationError,            # Pydantic failed
    InstructorRetryException    # Retry limit exceeded
)
```

**Saga Pattern** (Temporal Durable Execution):
```
"What if your code never failed?"
├── Automatic retries with policies
├── State persisted across failures
├── Compensation transactions for rollback
└── Works across service boundaries
```

**Graceful Degradation Hierarchy**:
```
Primary LLM → Fallback LLM → Cache → Rules → Safe Default
```

---

### Cycle 17 Configuration Management & Secrets (January 25, 2026)

**Pydantic Settings v2** (Type-safe Configuration):
```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_nested_delimiter='__'  # DATABASE__HOST → database.host
    )
    api_key: SecretStr  # Never logged, repr shows '**********'
    database_url: str
```

**Configuration Layering** (Priority Order):
```
1. Environment variables (highest)
2. .secrets.toml (git-ignored)
3. settings.local.toml
4. settings.toml
5. defaults.toml (lowest)
```

**Secrets Management Stack 2026**:
| Tool | Use Case |
|------|----------|
| Doppler | Universal secrets sync |
| AWS Secrets Manager | Cloud-native with rotation |
| HashiCorp Vault | Dynamic secrets (auto-rotating) |
| External Secrets Operator | K8s integration |

**Doppler CLI Pattern**:
```bash
doppler run -- python app.py  # Inject at runtime
doppler secrets download --format env > .env  # Offline sync
```

**Feature Flags** (OpenFeature CNCF Standard):
```python
from openfeature import api
from openfeature.contrib.provider.flagd import FlagdProvider

api.set_provider(FlagdProvider())
client = api.get_client("my-service")

enabled = client.get_boolean_value("new-feature", False, context)
```

**Configuration Anti-Patterns to Avoid**:
1. Hardcoded secrets in code
2. .env files committed to git
3. Long-lived credentials (prefer short TTL + rotation)
4. Logging SecretStr values
5. Feature flag debt (remove after rollout)

---

### Cycle 18 Dependency Injection & IoC (January 25, 2026)

**Python DI Libraries 2026**:
| Library | Style | Best For |
|---------|-------|----------|
| dependency-injector | Full IoC | Production apps |
| punq | Minimal | Simple projects |
| svcs | Service locator | Health checks |
| yedi/kink | Decorators | Type-safe DI |

**dependency-injector Pattern**:
```python
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    database = providers.Singleton(Database, url=config.db.url)
    user_repo = providers.Factory(UserRepository, db=database)
```

**FastAPI Depends (2026 Best Practice)**:
```python
from typing import Annotated
from fastapi import Depends

DbDep = Annotated[Session, Depends(get_db)]

@app.get("/users")
async def get_users(db: DbDep):
    return db.query(User).all()
```

**Modern Lifespan Management**:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await create_pool()
    yield
    await app.state.pool.close()
```

**Provider Lifetime Scopes**:
| Scope | Provider | Use Case |
|-------|----------|----------|
| Application | Singleton | DB pools, config |
| Request | Factory | Handlers, repos |
| Managed | Resource | Cleanup required |

**Testing Override Pattern**:
```python
# dependency-injector
with container.repo.override(MockRepo()):
    result = service.get_data()

# FastAPI
app.dependency_overrides[get_db] = lambda: mock_db
```

**Clean Architecture Rule**: Dependencies point INWARD only
```
Frameworks → Adapters → Use Cases → Domain
```

---

### Cycle 19 Async & Concurrency Patterns (January 25, 2026)

**Concurrency Selection Rule**:
```
I/O-Bound  → asyncio (thousands of connections, single thread)
CPU-Bound  → multiprocessing (true parallelism, bypasses GIL)
Mixed/Legacy → threading (simpler, GIL-limited)
```

**Structured Concurrency (Python 3.11+)**:
```python
async with asyncio.TaskGroup() as tg:
    task1 = tg.create_task(fetch("url1"))
    task2 = tg.create_task(fetch("url2"))
# ALL cancel on first exception - no zombie tasks
```

**AnyIO Cancel Scopes**:
```python
with anyio.move_on_after(5.0) as scope:
    await long_operation()
if scope.cancelled_caught:
    await cleanup()  # Graceful timeout handling
```

**Never Block the Event Loop**:
```python
# BAD: time.sleep(1)
# GOOD: await asyncio.sleep(1)

# Sync to async bridge
result = await asyncio.to_thread(blocking_function)
```

**Rate Limiting Pattern**:
```python
semaphore = asyncio.Semaphore(10)  # Max concurrent
async with semaphore:
    await api_call()
```

**Production Event Loop**:
```python
import uvloop
uvloop.install()  # 2-4x faster than default
```

**Backpressure with Queue**:
```python
queue = asyncio.Queue(maxsize=10)  # Bounded queue
await queue.put(item)  # Blocks when full
```

**Common Mistakes**:
1. Missing `await` on coroutine
2. Untracked `create_task()` (garbage collected)
3. Using `time.sleep()` instead of `asyncio.sleep()`

---

### Cycle 20 Caching Strategies & Patterns (January 25, 2026)

**In-Memory Memoization**:
```python
from functools import lru_cache, cache

@lru_cache(maxsize=1024)  # Bounded LRU
def expensive_computation(n: int) -> int:
    return sum(range(n))

@cache  # Unbounded (Python 3.9+)
def fibonacci(n: int) -> int:
    if n < 2: return n
    return fibonacci(n-1) + fibonacci(n-2)
```

**Redis Caching Patterns**:
| Pattern | Use Case |
|---------|----------|
| Cache-Aside | Read-heavy, simple, lazy loading |
| Write-Through | Consistency critical, update DB+cache together |
| Write-Behind | Write-heavy, async DB updates (risk: data loss) |
| Read-Through | Cache handles DB loading automatically |

**Cache-Aside Implementation**:
```python
def get_user(user_id: str) -> dict:
    cached = r.get(f"user:{user_id}")
    if cached: return json.loads(cached)
    user = database.get_user(user_id)
    r.setex(f"user:{user_id}", 3600, json.dumps(user))
    return user
```

**Stampede Prevention (Single-Flight)**:
```python
async with _locks.setdefault(key, Lock()):
    cached = r.get(key)
    if cached: return json.loads(cached)
    value = await loader()
    r.setex(key, 3600, json.dumps(value))
    return value
```

**Cache Invalidation Strategies**:
| Strategy | Description |
|----------|-------------|
| TTL-Based | Simple, may serve stale data |
| Event-Based | Pub/Sub for distributed invalidation |
| Version-Based | Increment version, old cache becomes orphan |

**Multi-Tier Cache Architecture**:
```
L1: Process Memory (fastest, per-instance)
L2: Redis (shared, sub-ms)
L3: Database (source of truth)
```

**Anti-Patterns to Avoid**:
1. Caching everything (wastes memory)
2. No cache size limits (OOM)
3. Ignoring cache coherence (stale data)
4. Too-long TTLs for changing data
5. Caching with unhashable keys

---

### Cycle 21 API Design Patterns (January 25, 2026)

**REST Best Practices**:
```python
# Resource-oriented: nouns for resources, HTTP verbs for actions
GET    /users/123    # Read
POST   /users        # Create
PUT    /users/123    # Replace
PATCH  /users/123    # Partial update
DELETE /users/123    # Delete

# Versioning: URL path for public, headers for internal
GET /v1/users
```

**Pagination Strategies**:
| Strategy | Best For |
|----------|----------|
| Offset-based | Simple, small datasets |
| Cursor-based | Stable, large datasets |
| Keyset-based | Best performance (uses index) |

**Idempotency Pattern**:
```python
# Client sends unique key
POST /payments
Idempotency-Key: abc-123-unique

# Server caches response by key
existing = await cache.get(f"idempotency:{key}")
if existing: return existing  # Return cached
```

**GraphQL Production Patterns**:
```graphql
# Relay-style pagination (standard)
type UserConnection {
  edges: [UserEdge!]!
  pageInfo: PageInfo!
}

# Global object identification
interface Node { id: ID! }
query { node(id: ID!): Node }
```

**N+1 Prevention**:
```python
from aiodataloader import DataLoader
loader = DataLoader(lambda ids: db.batch_get(ids))
await loader.load(user.id)  # Batched automatically
```

**gRPC Essentials**:
```python
# CRITICAL: Reuse channel and stub
channel = grpc.insecure_channel('localhost:50051')
stub = UserServiceStub(channel)

# Always set timeout
response = stub.GetUser(request, timeout=5.0)

# Keepalive for long connections
options=[('grpc.keepalive_time_ms', 30000)]
```

**Protocol Selection**:
| Criterion | REST | GraphQL | gRPC |
|-----------|------|---------|------|
| Use | Public APIs | Flexible queries | Internal services |
| Speed | Baseline | Baseline | 2-10x faster |
| Browser | Native | Native | gRPC-Web |

---

### Cycle 22 Event-Driven Architecture (January 25, 2026)

**Core EDA Patterns**:
| Pattern | Purpose |
|---------|---------|
| Event Sourcing | State = replay of events (audit trail) |
| CQRS | Separate read/write models |
| ECST | Event carries all needed data |
| Saga | Distributed transactions with compensation |
| Outbox | Atomic DB write + reliable publish |

**Kafka Producer (confluent-kafka)**:
```python
producer = Producer({'bootstrap.servers': 'localhost:9092', 'acks': 'all'})
producer.produce('topic', key=b'key', value=b'value', callback=cb)
producer.flush()  # CRITICAL: actually sends messages
```

**Kafka Consumer (Manual Commit)**:
```python
consumer = Consumer({
    'group.id': 'my-group',
    'enable.auto.commit': False  # Safety
})
msg = consumer.poll(1.0)
process(msg)
consumer.commit()  # After successful processing
```

**RabbitMQ Dead Letter Exchange**:
```python
channel.queue_declare('main_queue', arguments={
    'x-dead-letter-exchange': 'dlx',
    'x-dead-letter-routing-key': 'dead_letters'
})
# Failed messages route to DLX automatically
```

**Kafka vs RabbitMQ**:
| | Kafka | RabbitMQ |
|-|-------|----------|
| Best For | Streaming, logs, CDC | Tasks, RPC |
| Throughput | Millions/sec | Thousands/sec |
| Replay | Yes (offset reset) | No |
| Ordering | Per partition | Per queue |

**Delivery Guarantees**:
- At-most-once: Fast, may lose
- At-least-once: Safe, may duplicate (use idempotency)
- Exactly-once: Complex, transactional API

---

### Cycle 23 Authentication & Authorization (January 25, 2026)

**OAuth 2.1 Key Changes**:
- Implicit Flow DEPRECATED (23% of breaches)
- PKCE MANDATORY for ALL clients
- Password Grant (ROPC) REMOVED
- Enforcement: Google, Microsoft, Okta by Q2 2026

**PKCE Flow (Required)**:
```python
# Generate challenge
code_verifier = secrets.token_urlsafe(32)
code_challenge = base64.urlsafe_b64encode(
    hashlib.sha256(code_verifier.encode()).digest()
).rstrip(b'=').decode()

# Auth request includes: code_challenge, code_challenge_method='S256'
# Token exchange includes: code_verifier (proof)
```

**JWT Security (CRITICAL)**:
```python
payload = jwt.decode(
    token, public_key,
    algorithms=['RS256', 'ES256'],  # Explicit allowlist!
    audience='expected-audience',    # MUST validate
    issuer='expected-issuer',        # MUST validate
    options={'require': ['exp', 'iat', 'sub', 'iss', 'aud']}
)
```

**Algorithm Selection**:
| Algorithm | Use Case |
|-----------|----------|
| RS256 | Production APIs (asymmetric) |
| ES256 | High-security (ECDSA) |
| HS256 | Internal ONLY (AVOID in prod) |
| none | NEVER (attack vector) |

**Token Lifetime**:
- Access: 15 minutes
- Refresh: 7 days + rotation

**Authorization Models (Casbin)**:
```python
# RBAC
enforcer.enforce(\"alice\", \"data1\", \"read\")
enforcer.add_grouping_policy(\"alice\", \"admin\")

# ABAC (attributes in matchers)
m = r.sub.Department == r.obj.Department && r.sub.Level >= r.obj.RequiredLevel
```

**FastAPI Pattern**:
```python
def require_permission(resource: str, action: str):
    async def checker(user = Depends(get_current_user)):
        if not enforcer.enforce(user['sub'], resource, action):
            raise HTTPException(status_code=403)
        return user
    return checker
```

**Security Anti-Patterns**:
- NEVER use `alg: none` or allow algorithm switching
- NEVER store tokens in localStorage (XSS vulnerable)
- NEVER skip state parameter in OAuth (CSRF)
- NEVER use plain code_challenge_method (must be S256)

---

### Cycle 24 Structured Logging & Observability (January 25, 2026)

**structlog Production Setup**:
```python
import structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()  # Production
    ],
    cache_logger_on_first_use=True,
)
```

**Canonical Log Lines** (fewer, richer):
```python
# BAD: Many small logs
# GOOD: Single rich event
log.info("request_completed",
    user_id=42, endpoint="/api", duration_ms=45.2, status_code=200)
```

**Request-Scoped Context**:
```python
structlog.contextvars.bind_contextvars(request_id=req_id, user_id=user.id)
# All logs now include request_id and user_id
structlog.contextvars.clear_contextvars()  # End of request
```

**OpenTelemetry Tracing**:
```python
from opentelemetry import trace
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("process_order") as span:
    span.set_attribute("order.id", order_id)
    result = process(order)
```

**Trace ID in Logs** (Correlation):
```python
def add_trace_context(logger, method_name, event_dict):
    span = trace.get_current_span()
    if span.is_recording():
        ctx = span.get_span_context()
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict
# Add to structlog processors
```

**Context Propagation** (W3C):
```python
from opentelemetry.propagate import inject, extract
inject(headers)  # Outgoing: adds traceparent
context = extract(request.headers)  # Incoming
```

**Anti-Patterns**:
- String formatting in logs (use structured fields)
- Missing trace_id correlation
- Logging sensitive data (use redaction)
- Blocking log calls (use async handlers)

---

### Cycle 25 Database Patterns (January 25, 2026)

**SQLAlchemy 2.0 Async Setup**:
```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    pool_pre_ping=True,   # Health check before use
    pool_recycle=3600,    # Recycle after 1 hour
)
async_session = async_sessionmaker(engine, expire_on_commit=False)
```

**Eager Loading (Prevent N+1)**:
```python
from sqlalchemy.orm import selectinload, joinedload

# selectinload: Best for collections (1-to-many)
stmt = select(User).options(selectinload(User.orders))

# joinedload: Best for single objects (many-to-1)
stmt = select(Order).options(joinedload(Order.user))
```

**Alembic Migrations**:
```bash
alembic revision --autogenerate -m "add users table"
alembic upgrade head
alembic downgrade -1
```

**Cursor Pagination (Avoid OFFSET)**:
```python
# Much faster than OFFSET for large datasets
stmt = select(Order).where(Order.id < cursor).order_by(Order.id.desc()).limit(20)
next_cursor = orders[-1].id if len(orders) > limit else None
```

**Connection Pooling (PgBouncer)**:
```ini
pool_mode = transaction  # Recommended
default_pool_size = 20
max_client_conn = 1000
```

**Query Optimization**:
```python
# EXPLAIN ANALYZE
await session.execute(text(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"))

# Composite indexes for common patterns
Index('ix_orders_user_status', 'user_id', 'status')

# Partial indexes (filter subsets)
Index('ix_orders_active', 'created_at', postgresql_where=text("status = 'active'"))
```

**Transaction Patterns**:
```python
# Nested transactions (savepoints)
async with session.begin_nested():  # Creates savepoint
    await session.execute(update_stmt)  # Rolls back on exception

# Optimistic locking
__mapper_args__ = {"version_id_col": version}  # Auto-increments
```

---

### Cycle 26 Validation & Data Modeling (January 25, 2026)

**Pydantic v2 Validator Order**:
```
1. @field_validator(mode='before')  # Raw input
2. Type coercion                    # Automatic
3. @field_validator(mode='after')   # Post-coercion
4. @model_validator(mode='after')   # Cross-field
```

**Field Validators**:
```python
@field_validator('email')
@classmethod
def validate_email(cls, v: str) -> str:
    if '@' not in v: raise ValueError('Invalid email')
    return v.lower()

# Access other fields via info.data
@field_validator('field', mode='after')
@classmethod
def validate(cls, v, info: ValidationInfo) -> str:
    other = info.data.get('other_field')
```

**Model Validators (Cross-Field)**:
```python
@model_validator(mode='after')
def validate_dates(self) -> 'DateRange':
    if self.end < self.start: raise ValueError('Invalid range')
    return self
```

**Reusable Annotated Types**:
```python
CleanString = Annotated[str, BeforeValidator(str.strip), AfterValidator(validate_not_empty)]
```

**Strict Mode (No Coercion)**:
```python
model_config = ConfigDict(strict=True, extra='forbid')
```

**Schema Evolution Rules**:
- ✅ Add optional field with default
- ✅ Widen type (int → int | float)
- ❌ Remove required field
- ❌ Rename field (breaking)

**Security Validation**:
```python
# SQL injection - reject dangerous patterns
if re.search(r"(--|;|'|\"|\bUNION\b)", v, re.I):
    raise ValueError('Invalid input')

# Path traversal prevention
if '..' in filename or filename.startswith('/'):
    raise ValueError('Invalid filename')
```

---

### Cycle 27 Testing Patterns (January 25, 2026)

**pytest Fundamentals**:
```python
# Parametrize for multiple test cases
@pytest.mark.parametrize("input,expected", [
    (1, 1), (2, 4), (3, 9)
])
def test_square(input, expected):
    assert input ** 2 == expected
```

**Fixture Patterns**:
```python
# Scope levels: function (default) < class < module < session
@pytest.fixture(scope="module")
def db_connection():
    conn = connect_db()
    yield conn
    conn.close()

# Factory pattern for dynamic fixtures
@pytest.fixture
def make_user():
    created = []
    def _make(name="Test"):
        user = User(name=name)
        created.append(user)
        return user
    yield _make
    for u in created: u.delete()
```

**Hypothesis Property-Based Testing**:
```python
from hypothesis import given, strategies as st

@given(st.lists(st.integers()))
def test_sorted_idempotent(lst):
    assert sorted(sorted(lst)) == sorted(lst)

# Custom strategies
emails = st.from_regex(r"^[a-z]+@[a-z]+\\.com$", fullmatch=True)

# Stateful testing (RuleBasedStateMachine)
class ShoppingCart(RuleBasedStateMachine):
    @rule(item=st.text())
    def add_item(self, item): ...
```

**Mocking Best Practices**:
```python
# CRITICAL: Patch where USED, not where defined
# WRONG: mocker.patch('external.fetch_data')
# CORRECT: mocker.patch('mymodule.fetch_data')

# AsyncMock for async functions
mocker.patch('module.async_fn', new_callable=AsyncMock)

# Spying (track calls without replacing)
spy = mocker.spy(service, 'method')
assert spy.call_count == 1
```

**Async Testing (pytest-asyncio)**:
```python
@pytest.fixture
async def async_client():
    async with AsyncClient(app, base_url="http://test") as client:
        yield client

@pytest.mark.asyncio
async def test_endpoint(async_client):
    response = await async_client.get("/api/resource")
    assert response.status_code == 200
```

**Coverage Targets**:
| Type | Target |
|------|--------|
| Unit | 80%+ |
| Integration | Critical paths |
| Property | Core invariants |
| Mutation | 70%+ kill rate |

---

### Cycle 28 CLI & DevOps Patterns (January 25, 2026)

**Typer CLI Patterns**:
```python
from typing import Annotated
import typer

app = typer.Typer()

@app.command()
def process(
    input_file: Annotated[Path, typer.Argument()],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
):
    """Process with type-safe arguments."""

# Context object for shared state
@app.callback()
def main(ctx: typer.Context, config: Path = None):
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)

# Testing with CliRunner
from typer.testing import CliRunner
result = runner.invoke(app, ["process", "file.txt"])
assert result.exit_code == 0
```

**GitHub Actions Reusable Workflows**:
```yaml
# Define reusable workflow
on:
  workflow_call:
    inputs:
      python-version:
        type: string
        default: "3.12"
    secrets:
      deploy-key:
        required: true

# Call reusable workflow
jobs:
  build:
    uses: ./.github/workflows/reusable-build.yml
    with:
      python-version: "3.12"
    secrets:
      deploy-key: ${{ secrets.DEPLOY_KEY }}

# Concurrency control
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

**Docker Multi-Stage Build**:
```dockerfile
# Stage 1: Build
FROM python:3.12-slim AS builder
COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt

# Stage 2: Runtime (90% smaller)
FROM python:3.12-slim AS runtime
RUN useradd --create-home app
USER app
COPY --from=builder /wheels /wheels
RUN pip install --user /wheels/*
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
```

**BuildKit Cache Mounts**:
```dockerfile
# syntax=docker/dockerfile:1.6
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
```

---

### Cycle 29 Packaging & Distribution (January 25, 2026)

**pyproject.toml (PEP 621 Standard)**:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "mypackage"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["httpx>=0.25.0", "pydantic>=2.0"]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff", "mypy"]

[project.scripts]
myapp = "mypackage.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/mypackage"]
```

**Build Backend Comparison**:
| Backend | Best For |
|---------|----------|
| Hatchling | Modern projects, mono-repos |
| setuptools | Legacy, complex builds |
| Poetry-core | Poetry users |
| Flit-core | Simple pure-Python |

**Semantic Versioning Automation**:
```toml
# pyproject.toml
[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
branch = "main"
commit_parser = "angular"
major_on_zero = false
```

```yaml
# GitHub Action - auto-release on conventional commits
- uses: python-semantic-release/python-semantic-release@v9
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
```

**Conventional Commits**:
```
feat: Add new feature      → MINOR (0.X.0)
fix: Bug fix              → PATCH (0.0.X)
feat!: Breaking change    → MAJOR (X.0.0)
BREAKING CHANGE: in body  → MAJOR (X.0.0)
```

**PyPI Trusted Publishing (OIDC - No API Tokens!)**:
```yaml
# GitHub Actions - Secure publishing
jobs:
  publish:
    runs-on: ubuntu-latest
    permissions:
      id-token: write  # REQUIRED for OIDC
    environment: pypi  # Maps to PyPI trusted publisher
    steps:
      - uses: pypa/gh-action-pypi-publish@release/v1
      # No password/token needed!
```

**Package Structure (src layout)**:
```
mypackage/
├── src/mypackage/
│   ├── __init__.py      # version = "0.1.0"
│   ├── py.typed         # PEP 561 marker
│   └── cli.py
├── tests/
├── pyproject.toml
└── README.md
```

**Version Single Source of Truth**:
```python
# src/mypackage/__init__.py
from importlib.metadata import version
__version__ = version("mypackage")
```

---

### Cycle 30 Type System & Static Analysis (January 25, 2026)

**Python 3.12+ Type Parameter Syntax (PEP 695)**:
```python
# NEW: No imports needed!
class Container[T]:
    def __init__(self, value: T) -> None:
        self.value = value

def first[T](items: list[T]) -> T:
    return items[0]

# Type aliases
type Vector[T] = list[T]
type Callback[**P, R] = Callable[P, R]
```

**Protocol (Structural Subtyping)**:
```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

# Any class with draw() matches - no inheritance needed
```

**ParamSpec (Decorator Signature Preservation)**:
```python
from typing import ParamSpec, Callable

P = ParamSpec('P')

def logged[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper
```

**Ruff Production Config**:
```toml
[tool.ruff]
target-version = "py312"
line-length = 88

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP", "ARG", "SIM", "TCH", "PTH", "PL", "RUF"]
fixable = ["ALL"]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101", "ARG"]
```

**Type Checker Config**:
```toml
# mypy strict mode
[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
disallow_untyped_defs = true
```

**Best Practice**: Run BOTH mypy AND pyright - different edge case interpretations

**Key Patterns**:
| Pattern | Use Case |
|---------|----------|
| Protocol | Duck typing with type safety |
| ParamSpec | Preserve decorator signatures |
| TypeVarTuple | Variadic generics (shapes) |
| TypeGuard/TypeIs | Type narrowing |
| Self | Fluent interfaces |
| Annotated | Attach metadata to types |

---

### Cycle 31 Documentation & API Docs (January 25, 2026)

**MkDocs + Material Production Setup**:
```yaml
theme:
  name: material
  features:
    - navigation.instant
    - navigation.tabs
    - navigation.sections
    - search.suggest
    - content.code.copy

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: google
            show_source: true
            heading_level: 2
```

**Google Docstring Style (Recommended)**:
```python
def process_data(items: list[str], limit: int = 100) -> dict[str, int]:
    """Process a list of items and return counts.

    Args:
        items: List of items to process.
        limit: Maximum items to process. Defaults to 100.

    Returns:
        Dictionary mapping item names to their counts.

    Raises:
        ValueError: If items list is empty.

    Example:
        >>> process_data(["a", "b", "a"])
        {"a": 2, "b": 1}
    """
```

**FastAPI Auto Documentation**:
```python
@app.get("/users/{user_id}",
    response_model=User,
    summary="Get user by ID",
    description="Retrieve user details from database",
    tags=["Users"])
async def get_user(
    user_id: Annotated[int, Path(description="User ID", ge=1)],
    db: DbDep
) -> User:
    """Endpoint docstring becomes operation description."""
```

**Documentation Quadrant Framework**:
| Type | Purpose | Examples |
|------|---------|----------|
| Tutorial | Learning | Getting started guides |
| How-To | Goal-oriented | Step-by-step procedures |
| Reference | Information | API docs, configs |
| Explanation | Understanding | Architecture, decisions |

**Key Tools**:
- MkDocs + Material: Primary static site generator
- mkdocstrings: Auto-generate from docstrings
- Sphinx: Legacy/complex projects
- FastAPI: Built-in OpenAPI + Swagger UI

---

### Cycle 32 Performance & Profiling (January 25, 2026)

**Profiler Selection**:
| Tool | Use Case | Production Safe |
|------|----------|-----------------|
| **py-spy** | CPU profiling | YES (1% overhead) |
| **Scalene** | CPU + memory + GPU | Yes |
| **pyinstrument** | Web apps, call stacks | Yes |
| **memray** | Memory leaks | Yes |
| **tracemalloc** | Built-in memory | Yes |

**py-spy (Production CPU Profiling)**:
```bash
# Attach to running process (NO code changes)
py-spy record -o profile.svg --pid 12345
py-spy top --pid 12345  # Live view
```

**Memory Leak Detection**:
```python
import tracemalloc
tracemalloc.start()
snapshot1 = tracemalloc.take_snapshot()
# ... suspected leaky code ...
snapshot2 = tracemalloc.take_snapshot()
diff = snapshot2.compare_to(snapshot1, 'lineno')
for stat in diff[:10]:
    print(stat)
```

**Memray (Bloomberg)**:
```bash
memray run script.py
memray flamegraph memray-script.py.bin
pytest --memray tests/  # Regression testing
```

**FastAPI Profiling Middleware**:
```python
from pyinstrument import Profiler
from starlette.middleware.base import BaseHTTPMiddleware

class ProfilingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.query_params.get("profile"):
            with Profiler(async_mode="enabled") as p:
                response = await call_next(request)
            print(p.output_text())
            return response
        return await call_next(request)
```

**Quick Optimization Wins**:
```python
# Generators for large data
data = (process(x) for x in huge_list)  # Not list comprehension

# __slots__ for many instances (40% less memory)
class Point:
    __slots__ = ['x', 'y']

# Local variables in hot loops
sqrt = math.sqrt  # Avoid repeated attribute lookup
for i in range(1000000):
    result = sqrt(i)

# String joining
s = "".join(parts)  # Not s += part in loop
```

**Performance Hierarchy** (biggest impact first):
1. Algorithm complexity (O(n²) → O(n log n))
2. I/O patterns (batching, caching, async)
3. Memory allocation (pooling, generators)
4. CPU-bound (Cython, numba, multiprocessing)
5. Micro-optimizations (usually not worth it)

---

### Cycle 33 Monitoring & Alerting (January 25, 2026)

**SRE Monitoring Frameworks**:
| Framework | Signals |
|-----------|---------|
| **Four Golden Signals** | Latency, Traffic, Errors, Saturation |
| **RED Method** | Rate, Errors, Duration (request-focused) |
| **USE Method** | Utilization, Saturation, Errors (resources) |

**Prometheus Metric Types**:
| Type | Use Case |
|------|----------|
| Counter | Requests, errors (only goes up) |
| Gauge | Current values (memory, connections) |
| Histogram | Latency distributions, request sizes |
| Summary | Pre-computed quantiles (client-side) |

**FastAPI Prometheus Setup**:
```python
from prometheus_client import Counter, Histogram
from starlette_prometheus import PrometheusMiddleware, metrics

REQUEST_COUNT = Counter(
    'fastapi_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'fastapi_request_duration_seconds',
    'Request latency',
    ['method', 'endpoint']
)

app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", metrics)
```

**Alert Rule Structure**:
```yaml
- alert: HighErrorRate
  expr: |
    sum(rate(http_requests_total{status=~"5.."}[5m])) 
    / sum(rate(http_requests_total[5m])) > 0.01
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Error rate above 1%"
```

**SLO/Error Budget Pattern**:
```python
# SLO: 99.9% availability (43 min/month error budget)
SLO = 0.999
error_budget_total = 1 - SLO  # 0.001

# Current burn
error_rate = errors / total_requests
budget_remaining = error_budget_total - error_rate

# Multi-window burn rate alert
short_window_burn = error_rate_1h / (1 - SLO)  # >14x = page
long_window_burn = error_rate_6h / (1 - SLO)   # >1x = ticket
```

**Alerting Severity Levels**:
| Level | Response | Example |
|-------|----------|---------|
| Critical | Page immediately | Service down |
| Warning | Within hours | High latency |
| Info | Next business day | Disk 70% |

**Incident Management** (2026): PagerDuty, incident.io, Rootly, FireHydrant (OpsGenie deprecated)

**Key Anti-Patterns**: Alert fatigue, missing runbooks, no SLO, siloed metrics

---

### Cycle 34 Background Tasks & Queues (January 25, 2026)

**Task Queue Selection**:
| Library | Best For | Async Native |
|---------|----------|--------------|
| **Celery** | High scale, multi-broker | No (prefork) |
| **ARQ** | FastAPI, async-first | YES |
| **RQ** | Simple Redis queue | No |
| **Dramatiq** | Celery alternative | No |
| **Temporal** | Durable workflows | Yes |

**FastAPI BackgroundTasks** (Built-in):
```python
from fastapi import BackgroundTasks

@app.post("/signup")
async def signup(user: User, background_tasks: BackgroundTasks):
    create_user(user)
    background_tasks.add_task(send_email, user.email, "Welcome!")
    return {"status": "created"}  # Returns immediately
```

**ARQ (Async-First for FastAPI)**:
```python
from arq import create_pool, Retry

async def send_email(ctx, email: str):
    await smtp.send_async(email, "Welcome!")

# Enqueue from FastAPI
job = await request.app.state.arq_pool.enqueue_job('send_email', email)

# Retry with exponential backoff
retry_count = ctx.get('job_try', 1)
if retry_count < 5:
    raise Retry(defer=2 ** retry_count)
```

**Celery Production Config**:
```python
app.conf.update(
    task_time_limit=300,      # Hard limit
    task_soft_time_limit=240, # Soft limit
    task_acks_late=True,      # Fair distribution
    worker_prefetch_multiplier=1,  # For long tasks
)

@app.task(bind=True, max_retries=3)
def process(self, data):
    try: return do_work(data)
    except TransientError as e:
        raise self.retry(exc=e, countdown=60)
```

**APScheduler (Scheduling)**:
```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
scheduler = AsyncIOScheduler()

@scheduler.scheduled_job('cron', hour=0)  # Midnight
async def daily_cleanup(): ...
```

**Decision Guide**:
- **BackgroundTasks**: Simple, < 100/min, no persistence
- **ARQ**: FastAPI + async, Redis, moderate scale
- **Celery**: High scale, complex routing, Beat scheduler
- **Temporal**: Multi-step workflows, exactly-once, sagas

**Anti-Patterns**: Passing large objects (use IDs), no timeouts, sync in async, single queue bottleneck

---

### Cycle 35 Rate Limiting & Throttling (January 25, 2026)

**Algorithm Comparison**:
| Algorithm | Best For | Drawback |
|-----------|----------|----------|
| **Token Bucket** | Bursty + average control | Memory per client |
| **Sliding Window** | Precision | Memory overhead |
| **Fixed Window** | Simple | 2x edge bursts |
| **Leaky Bucket** | Smooth output | Delays under load |

**SlowAPI for FastAPI**:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/api")
@limiter.limit("100/minute")
async def endpoint(request: Request):
    return {"data": "value"}
```

**Redis Distributed Rate Limiting (Lua Atomic)**:
```python
RATE_LIMIT_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local current = redis.call('INCR', key)
if current == 1 then redis.call('EXPIRE', key, window) end
if current > limit then return {0, redis.call('TTL', key)} end
return {1, limit - current}
"""
```

**Tiered Rate Limiting**:
```python
TIER_LIMITS = {
    "free": "10/minute",
    "pro": "1000/minute",
    "enterprise": "10000/minute",
}

def get_tier_limit(request: Request) -> str:
    return TIER_LIMITS[get_user_tier(request)]

@limiter.limit(limit_value=get_tier_limit)
```

**Multi-Layer DDoS Defense**:
1. L1: Connection limit (nginx limit_conn)
2. L2: Request rate (SlowAPI)
3. L3: Expensive ops limit (5/min)
4. L4: Circuit breaker for downstream

**Response Headers (RFC 6585)**:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1706234567
Retry-After: 60  # When limited
```

**Anti-Patterns**: In-memory only multi-server, no headers, same limit everywhere, blocking limit checks

---

### Cycle 36 WebSockets & Real-Time (January 25, 2026)

**FastAPI WebSocket Basic**:
```python
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            await websocket.send_json({"echo": data})
    except WebSocketDisconnect:
        pass
```

**Connection Manager Pattern**:
```python
class ConnectionManager:
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.rooms: Dict[str, Set[str]] = {}
    
    async def broadcast_to_room(self, room: str, message: dict):
        for user_id in self.rooms.get(room, set()):
            if ws := self.connections.get(user_id):
                await ws.send_json(message)
```

**Redis Pub/Sub for Horizontal Scaling**:
```python
# Single server: messages only reach local clients
# Multi-server: Redis pub/sub broadcasts across all instances

async def publish(self, channel: str, message: dict):
    await self.redis.publish(channel, json.dumps(message))

async def subscribe_listener(self, channel: str):
    await self.pubsub.subscribe(channel)
    async for msg in self.pubsub.listen():
        if msg["type"] == "message":
            await self.local_broadcast(json.loads(msg["data"]))
```

**python-socketio (Rooms Built-in)**:
```python
import socketio
sio = socketio.AsyncServer(async_mode='asgi')

@sio.event
async def join_room(sid, room):
    sio.enter_room(sid, room)

# Broadcast to room
await sio.emit('message', data, room='general')

# Redis adapter for multi-server
mgr = socketio.AsyncRedisManager('redis://localhost:6379')
sio = socketio.AsyncServer(client_manager=mgr)
```

**Heartbeat Pattern**:
```python
async def heartbeat(websocket: WebSocket, interval: int = 30):
    while True:
        await asyncio.sleep(interval)
        await websocket.send_json({"type": "ping"})
```

**Backpressure (Slow Clients)**:
```python
await asyncio.wait_for(ws.send_json(msg), timeout=5.0)
# If timeout: client too slow, queue or disconnect
```

**Production Checklist**:
- Auth on connect (token or first-message)
- Heartbeat for dead connection detection
- Redis pub/sub for horizontal scaling
- Sticky sessions if not using Redis
- Connection limits per user
- Graceful shutdown notification

**Anti-Patterns**: No heartbeat (ghost connections), blocking event loop, unlimited connections, no auth timeout

---

### Cycle 37 File Uploads & Media Processing (January 25, 2026)

**FastAPI UploadFile Pattern**:
```python
from fastapi import UploadFile, File, HTTPException
import aiofiles

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    # Validate type from content, not header
    allowed = {"image/jpeg", "image/png"}
    if file.content_type not in allowed:
        raise HTTPException(400, "Invalid type")
    
    # Stream to avoid memory issues
    async with aiofiles.open(f"uploads/{file.filename}", "wb") as f:
        while chunk := await file.read(1024 * 1024):
            await f.write(chunk)
```

**S3 Presigned URL (Direct Browser Upload)**:
```python
presigned = s3_client.generate_presigned_post(
    Bucket="my-bucket",
    Key=f"uploads/{uuid.uuid4()}.{ext}",
    Conditions=[
        {"Content-Type": content_type},
        ["content-length-range", 1, 100 * 1024 * 1024],  # 100MB max
    ],
    ExpiresIn=3600
)
# Returns: {"url": "...", "fields": {...}}
```

**S3 Multipart (Files >100MB)**:
```python
# 1. Initiate: create_multipart_upload()
# 2. Upload parts: generate_presigned_url("upload_part", ...)
# 3. Complete: complete_multipart_upload(Parts=[{PartNumber, ETag}])
```

**Pillow Image Processing**:
```python
from PIL import Image
from io import BytesIO

def process_image(image_bytes: bytes) -> tuple[bytes, bytes]:
    img = Image.open(BytesIO(image_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    
    # Resize
    img.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
    resized = BytesIO()
    img.save(resized, format="JPEG", quality=85, optimize=True)
    
    # Thumbnail
    thumb = img.copy()
    thumb.thumbnail((300, 300))
    thumb_buf = BytesIO()
    thumb.save(thumb_buf, format="JPEG", quality=80)
    
    return resized.getvalue(), thumb_buf.getvalue()
```

**File Validation (Magic Bytes)**:
```python
import magic
mime = magic.Magic(mime=True)
actual_type = mime.from_buffer(file_bytes)  # Detect from content
if actual_type != claimed_type:
    raise ValueError("Type mismatch")
```

**Filename Sanitization**:
```python
import re, unicodedata
def sanitize(filename: str) -> str:
    filename = unicodedata.normalize("NFKD", filename)
    filename = filename.encode("ascii", "ignore").decode()
    return re.sub(r'[<>:"/\\|?*]', "_", filename).strip(". ")
```

**Decision Matrix**:
| Size | Approach |
|------|----------|
| <10MB | FastAPI UploadFile in-memory |
| 10-100MB | Streaming, disk buffer |
| 100MB-5GB | S3 presigned URL |
| >5GB | S3 multipart upload |

**Anti-Patterns**: Loading large files into memory, trusting Content-Type header, using original filename directly, no file size limits, sync video processing

---

### Cycle 38 API Versioning & Deprecation (January 25, 2026)

**Versioning Strategy Selection**:
| Strategy | Use Case |
|----------|----------|
| **URL Path** `/v1/` | Public APIs (visible, cacheable) |
| **Header** `X-API-Version` | Internal APIs (clean URLs) |
| **Query** `?version=1` | Testing only |

**FastAPI URL Path Versioning**:
```python
v1 = APIRouter(prefix="/v1")
v2 = APIRouter(prefix="/v2")
app.include_router(v1)
app.include_router(v2)
```

**RFC 9745 Deprecation Headers (March 2025 Standard)**:
```python
response.headers["Deprecation"] = "Sat, 01 Feb 2026 00:00:00 GMT"
response.headers["Sunset"] = "Sat, 01 Jul 2026 00:00:00 GMT"
response.headers["Link"] = '<https://api.example.com/migrate>; rel="deprecation"'
```

**Breaking vs Non-Breaking Changes**:
```
✅ NON-BREAKING (safe):
- Add optional fields with defaults
- Add new endpoints
- Widen accepted types

❌ BREAKING (new version required):
- Remove fields
- Change field types
- Rename fields
- Change response structure
- Change auth method
```

**Deprecation Timeline**:
```
T-12 months: Announce, new version available
T-6 months:  Add deprecation headers
T-3 months:  Reminder emails
T-1 month:   Brownouts (10% → 25% → 50% errors)
T-0:         Return HTTP 410 Gone
```

**Brownout Pattern**:
```python
if random.random() < brownout_probability:
    return JSONResponse(503, {"error": "Please migrate to v2"})
```

**HTTP 410 After Sunset**:
```python
if datetime.now() >= sunset_date:
    return JSONResponse(410, {
        "error": "Gone",
        "successor": "/v2/users"
    })
```

**Anti-Patterns**: No version from start, immediate deprecation, silent breaking changes, too many active versions (max 2-3), no migration path

---

### Cycle 39 GraphQL Advanced Patterns (January 25, 2026)

**Strawberry GraphQL (Modern Python)**:
```python
import strawberry
from strawberry.fastapi import GraphQLRouter

@strawberry.type
class User:
    id: strawberry.ID
    name: str

@strawberry.type
class Query:
    @strawberry.field
    async def user(self, id: strawberry.ID, info: strawberry.Info) -> User:
        return await info.context["db"].get_user(id)

schema = strawberry.Schema(query=Query)
router = GraphQLRouter(schema, context_getter=get_context)
```

**DataLoader (Solve N+1)**:
```python
from strawberry.dataloader import DataLoader

# CRITICAL: Return order must match input keys!
async def load_users_batch(keys: List[int]) -> List[User]:
    users = await db.fetch_users_by_ids(keys)
    user_map = {u.id: u for u in users}
    return [user_map.get(key) for key in keys]

# One DataLoader per request (in context)
loader = DataLoader(load_fn=load_users_batch)
await loader.load(user_id)  # Batched automatically
```

**Subscriptions (Real-Time)**:
```python
@strawberry.type
class Subscription:
    @strawberry.subscription
    async def post_created(self) -> AsyncGenerator[Post, None]:
        async for message in pubsub.subscribe("posts:created"):
            yield Post(**message)
```

**Apollo Federation (Distributed Schemas)**:
```python
from strawberry.federation import Schema

@strawberry.federation.type(keys=["id"])
class Product:
    id: strawberry.ID
    @classmethod
    def resolve_reference(cls, id: strawberry.ID, info) -> "Product":
        return info.context["db"].get_product(id)

schema = Schema(query=Query, enable_federation_2=True)
```

**Performance Patterns**:
- Query depth limiting (max 10)
- Persisted queries (hash instead of full query)
- Response caching with cache hints
- Complexity calculation for expensive fields

**Error Handling (Union Types)**:
```python
@strawberry.type
class UserNotFound:
    message: str = "User not found"

UserResult = strawberry.union("UserResult", [UserSuccess, UserNotFound])
```

**Anti-Patterns**: Global DataLoader (cache pollution), sync resolvers (blocks loop), unbounded queries, N+1 in nested fields

---

### Cycle 40 gRPC & Protocol Buffers (January 25, 2026)

**Protocol Buffers Schema**:
```protobuf
service OrderService {
  rpc CreateOrder(Request) returns (Order);           // Unary
  rpc StreamUpdates(Filter) returns (stream Update);  // Server stream
  rpc BatchCreate(stream Request) returns (Result);   // Client stream
  rpc TradeStream(stream Req) returns (stream Res);   // Bidirectional
}

// Schema Evolution - CRITICAL rules:
reserved 3, 4;           // Never reuse removed tags
reserved "old_field";    // Document removed names
optional double price = 5;  // Nullable in proto3
```

**AsyncIO Server**:
```python
from grpc import aio

class OrderServiceServicer(order_pb2_grpc.OrderServiceServicer):
    async def CreateOrder(self, request, context):
        metadata = dict(context.invocation_metadata())
        await context.send_initial_metadata([("x-id", order.id)])
        return order_pb2.Order(id=order.id)

server = aio.server(options=[
    ("grpc.keepalive_time_ms", 10000),
    ("grpc.max_send_message_length", 50 * 1024 * 1024),
])
```

**Interceptors (Cross-Cutting)**:
```python
class LoggingInterceptor(aio.ServerInterceptor):
    async def intercept_service(self, continuation, handler_call_details):
        start = time.perf_counter()
        response = await continuation(handler_call_details)
        logger.info("rpc", method=handler_call_details.method,
                    duration_ms=(time.perf_counter()-start)*1000)
        return response

server = aio.server(interceptors=[LoggingInterceptor(), AuthInterceptor()])
```

**Health Checking (grpc_health)**:
```python
from grpc_health.v1 import health, health_pb2_grpc
health_servicer = health.HealthServicer()
health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
health_servicer.set("trading.v1.OrderService", health.HealthCheckResponse.SERVING)
```

**Client Retry Policy**:
```python
channel = aio.insecure_channel("localhost:50051", options=[
    ("grpc.enable_retries", 1),
    ("grpc.service_config", '''{"methodConfig": [{
        "name": [{"service": "trading.v1.OrderService"}],
        "retryPolicy": {"maxAttempts": 3, "retryableStatusCodes": ["UNAVAILABLE"]}
    }]}''')
])
```

**Bidirectional Streaming**:
```python
async def TradeStream(self, request_iterator, context):
    async for request in request_iterator:
        if context.cancelled(): break
        response = await self.process(request)
        yield response
```

**Anti-Patterns**: Blocking in async (use to_thread), ignoring deadlines (context.time_remaining()), not checking context.cancelled() in streams, reusing tag numbers

---

### Cycle 41 Message Serialization & Data Formats (January 25, 2026)

**Serialization Library Selection**:
| Library | Speed | Size | Best For |
|---------|-------|------|----------|
| **orjson** | 5-10x json | Same | REST APIs (Rust-powered) |
| **MessagePack** | 2-3x json | 60% | Caching, WebSocket, internal |
| **Avro** | Fast | 35% | Kafka, schema evolution |
| **Parquet** | N/A | 17% | Analytics, data lakes |

**orjson (Rust-Powered JSON)**:
```python
import orjson

# Returns bytes (not str) - 5-10x faster
data = orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_UTC_Z)
parsed = orjson.loads(data)

# Pydantic v2 integration
def model_dump_json_bytes(self) -> bytes:
    return orjson.dumps(self.model_dump(mode="python"))
```

**MessagePack (Binary Compact)**:
```python
import msgpack

packed = msgpack.packb(data)  # 40% smaller than JSON
unpacked = msgpack.unpackb(packed)

# Custom type handling
def encode_datetime(obj):
    if isinstance(obj, datetime):
        return msgpack.ExtType(1, obj.isoformat().encode())
    raise TypeError(f"Unknown type: {type(obj)}")
```

**Apache Avro (Schema Registry)**:
```python
from confluent_kafka.schema_registry.avro import AvroSerializer

avro_serializer = AvroSerializer(
    schema_registry_client,
    schema_str,
    to_dict_fn
)
producer.produce(topic, value=avro_serializer(event, ctx))
```

**Parquet with Polars (Fastest)**:
```python
import polars as pl

df.write_parquet("data.parquet", compression="zstd", statistics=True)

# Lazy evaluation with predicate pushdown
df = (pl.scan_parquet("data/*.parquet")
      .filter(pl.col("symbol") == "AAPL")
      .select(["timestamp", "price"])
      .collect())
```

**Use Case Matrix**:
- REST API responses → orjson
- Redis/caching → MessagePack
- Kafka streaming → Avro
- Data lake storage → Parquet
- Inter-service RPC → Protobuf (Cycle 40)

**Anti-Patterns**: Mixing formats in hot paths, string datetime in binary formats, no content negotiation, caching with unhashable keys

---

### Cycle 42 Data Validation & Schema Evolution (January 25, 2026)

**Pydantic v2 Discriminated Unions (O(1) Lookup)**:
```python
from typing import Annotated, Union, Literal
from pydantic import BaseModel, Field

class CreditCard(BaseModel):
    type: Literal["credit_card"]
    number: str

class BankTransfer(BaseModel):
    type: Literal["bank_transfer"]
    iban: str

# O(1) validation via discriminator field
PaymentMethod = Annotated[
    Union[CreditCard, BankTransfer],
    Field(discriminator="type")
]
```

**Generic Models (Reusable Containers)**:
```python
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: T | None = None
    error: str | None = None

# Usage: ApiResponse[User], ApiResponse[list[Order]]
```

**Validator Execution Order**:
```
1. @field_validator(mode='before')  # Raw input
2. Type coercion (str → int)        # Automatic
3. @field_validator(mode='after')   # Post-coercion
4. @model_validator(mode='after')   # Cross-field validation
```

**Zod for TypeScript Runtime Validation**:
```typescript
import { z } from 'zod';

const UserSchema = z.object({
  id: z.string().uuid(),
  email: z.string().email(),
  role: z.enum(['admin', 'user']),
  createdAt: z.coerce.date(),
});

type User = z.infer<typeof UserSchema>;  // Auto type extraction
const user = UserSchema.parse(data);     // Runtime validation!
```

**Schema Evolution Compatibility Types**:
| Type | Rule |
|------|------|
| **Backward** | New reader, old data ✓ |
| **Forward** | Old reader, new data ✓ |
| **Full** | Both directions ✓ |

**Avro Schema Evolution Rules**:
```
✅ SAFE: Add optional field with default
✅ SAFE: Remove field with default (Avro fills default)
✅ SAFE: Widen type (int → long)
❌ BREAKING: Remove required field
❌ BREAKING: Rename field
❌ BREAKING: Change type incompatibly
```

**Boundary Validation Principle**:
```
Validate at system boundaries (API, file I/O, queue consumers)
Trust internal code (skip redundant validation in hot paths)
```

**Anti-Patterns**: Validation in hot loops, trusting external input, no schema versioning, string validation only (use Pydantic/Zod)

---

### Cycle 43 HTTP Client & Connection Patterns (January 25, 2026)

**HTTP Client Selection 2026**:
| Library | Best For | Async | HTTP/2 |
|---------|----------|-------|--------|
| **httpx** | Modern apps (recommended) | Yes + Sync | Yes |
| **aiohttp** | 10k+ concurrent | Async only | No |
| **requests** | Simple scripts | No | No |

**httpx Production Setup**:
```python
import httpx

# Reuse client for connection pooling
client = httpx.AsyncClient(
    timeout=httpx.Timeout(10.0, connect=5.0),
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    http2=True,  # 30% latency reduction
)

# FastAPI lifespan pattern
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(...)
    yield
    await app.state.http_client.aclose()
```

**Retry with Tenacity**:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
)
async def fetch_with_retry(client, url):
    response = await client.get(url)
    response.raise_for_status()
    return response
```

**Circuit Breaker (pybreaker)**:
```python
breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=30)

@breaker
async def protected_request(client, url):
    response = await client.get(url, timeout=5.0)
    response.raise_for_status()
    return response
```

**Timeout Strategy**:
| Scenario | Connect | Read | Total |
|----------|---------|------|-------|
| Fast API | 2s | 5s | 10s |
| File Upload | 5s | 60s | 120s |
| Internal | 1s | 3s | 5s |

**Performance**: `uvloop.install()` for 2-4x faster event loop

**Anti-Patterns**: Client per request (socket exhaustion), no timeouts, sync in async, retry without backoff

---

### Cycle 44 Database Connection Pooling & Async (January 25, 2026)

**SQLAlchemy 2.0 Async Engine**:
```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost:5432/db",
    pool_size=20,           # Base connections
    max_overflow=10,        # Extra under load
    pool_pre_ping=True,     # Validate before use
    pool_recycle=3600,      # Recycle after 1 hour
    connect_args={"prepared_statement_cache_size": 100},
)
async_session = async_sessionmaker(engine, expire_on_commit=False)
```

**Pool Sizing Formula**:
```python
# Optimal: (cpu_cores * 2) + effective_spindle_count
pool_size = min(cpu_cores * 4, 50)
max_overflow = pool_size // 2
```

**asyncpg Direct Pooling**:
```python
pool = await asyncpg.create_pool(
    dsn="postgresql://user:pass@localhost/db",
    min_size=10, max_size=50,
    max_inactive_connection_lifetime=300,
)
# Prepared statements (20-40% faster)
stmt = await conn.prepare("SELECT * FROM orders WHERE id = $1")
```

**PgBouncer Config**:
```ini
pool_mode = transaction   # Release on transaction end
default_pool_size = 25
max_client_conn = 1000
```

**FastAPI Lifespan**:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.async_session = async_sessionmaker(engine)
    yield
    await engine.dispose()
```

**Anti-Patterns**: Engine per request, unbounded max_overflow, holding connection during external calls

---

### Cycle 45 Message Queues & Task Distribution (January 25, 2026)

**Celery 5.6 Production Config**:
```python
app.conf.update(
    task_acks_late=True,              # Ack after completion
    task_reject_on_worker_lost=True,  # Requeue on worker death
    task_time_limit=3600,             # Hard limit: 1 hour
    task_soft_time_limit=3300,        # Soft limit: 55 mins
    worker_prefetch_multiplier=4,     # Fetch 4 tasks/worker
)
```

**Broker Selection**:
| Factor | RabbitMQ | Redis |
|--------|----------|-------|
| Reliability | AMQP guarantees | At-most-once |
| Performance | 10-20K msg/s | 100K+ msg/s |
| Persistence | Disk-backed | In-memory |

**Task Idempotency Pattern**:
```python
def idempotent_task(ttl=86400):
    def decorator(func):
        def wrapper(*args, **kwargs):
            key = f"task:{func.__name__}:{hash((args, tuple(kwargs.items())))}"
            if redis.get(f"result:{key}"):
                return redis.get(f"result:{key}")
            if not redis.set(f"lock:{key}", "1", nx=True, ex=300):
                raise Exception("Already in progress")
            try:
                result = func(*args, **kwargs)
                redis.setex(f"result:{key}", ttl, result)
                return result
            finally:
                redis.delete(f"lock:{key}")
        return wrapper
    return decorator
```

**Retry with Exponential Backoff**:
```python
@app.task(
    bind=True,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=2,           # Base: 2 seconds
    retry_backoff_max=300,     # Max: 5 minutes
    retry_jitter=True,         # Prevent thundering herd
    max_retries=5,
)
def reliable_task(self, data): ...
```

**Celery Primitives**:
- `chain()`: Sequential task execution, result passing
- `group()`: Parallel execution, collect all results
- `chord()`: Parallel + callback when all complete

**Dead Letter Queue**:
```python
Queue("main", queue_arguments={
    "x-dead-letter-exchange": "dead_letter",
    "x-dead-letter-routing-key": "dead",
})
```

**Temporal for Durable Workflows**:
```python
@workflow.defn
class OrderWorkflow:
    @workflow.run
    async def run(self, order):
        await workflow.execute_activity(reserve_inventory, ...)
        try:
            await workflow.execute_activity(charge_payment, ...)
        except Exception:
            await workflow.execute_activity(release_inventory, ...)  # Compensation
            raise
```

**Anti-Patterns**: No idempotency on retries, missing time limits, large payloads in messages, synchronous waits in tasks

---

### Cycle 46 Caching Strategies & Invalidation (January 25, 2026)

**Caching Strategies**:
| Strategy | When to Use |
|----------|-------------|
| Cache-aside | Read-heavy, lazy loading |
| Write-through | Strong consistency needed |
| Write-behind | Write-heavy, eventual consistency OK |

**Cache-Aside Pattern**:
```python
def cache_aside(ttl=300, prefix="cache"):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            key = f"{prefix}:{func.__name__}:{hash(args)}"
            if cached := await redis.get(key):
                return json.loads(cached)
            result = await func(*args, **kwargs)
            await redis.setex(key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator
```

**CDC Invalidation (PostgreSQL NOTIFY)**:
```python
# Trigger pushes changes to app via LISTEN
async def _handle_notification(conn, pid, channel, payload):
    data = json.loads(payload)
    await redis.delete(f"{data['table']}:{data['id']}")

# PostgreSQL trigger function
# PERFORM pg_notify('cache_invalidate', json_build_object(...))
```

**Stampede Prevention - Mutex Lock**:
```python
async def get_or_rebuild(key, rebuild_fn, ttl=300):
    if value := await redis.get(key):
        return json.loads(value)
    lock_key = f"lock:{key}"
    lock_id = str(uuid.uuid4())
    if await redis.set(lock_key, lock_id, nx=True, ex=10):
        try:
            value = await rebuild_fn()
            await redis.setex(key, ttl, json.dumps(value))
            return value
        finally:
            # Release only if we own it (Lua script)
            await redis.eval("if get()==id then del() end", 1, lock_key, lock_id)
    else:
        # Wait for rebuilder to finish
        for _ in range(50):
            await asyncio.sleep(0.1)
            if value := await redis.get(key): return json.loads(value)
```

**Probabilistic Early Expiration (XFetch)**:
```python
# Randomly refresh before TTL expires
ttl_remaining = expiry - time.time()
threshold = -delta * beta * math.log(random.random())
if ttl_remaining <= threshold:
    asyncio.create_task(background_refresh())  # Refresh in bg
return cached_value  # Return stale immediately
```

**Stale-While-Revalidate**:
- Soft TTL: After this, refresh in background
- Hard TTL: Soft + stale buffer (serve stale while refreshing)
- User gets fast response, cache refreshes asynchronously

**TTL Guidelines**:
| Data Type | TTL |
|-----------|-----|
| Session | 30min-24hr |
| User profiles | 5-15min |
| Product catalog | 1-24hr |
| Config | 5-60min |

**Anti-Patterns**: No TTL (memory leak), cache everything (low hit ratio), large objects, ignoring stampede

---

### Cycle 47 API Rate Limiting & Throttling (January 25, 2026)

**Rate Limiting Algorithms** (from Redis official docs):
| Algorithm | Memory | Accuracy | Use Case |
|-----------|--------|----------|----------|
| Fixed Window | O(1) | Low | Simple APIs |
| Sliding Window Counter | O(1) | High | **Recommended default** |
| Token Bucket | O(1) | N/A | Burst-tolerant APIs |
| Leaky Bucket | O(1) | N/A | Constant-rate processing |

**Sliding Window Counter (Production)**:
```python
async def sliding_window_limit(redis_client, key, limit, window_seconds=60):
    now = time.time()
    current_window = int(now) // window_seconds
    window_position = (now % window_seconds) / window_seconds

    current_key = f"ratelimit:{key}:{current_window}"
    previous_key = f"ratelimit:{key}:{current_window - 1}"

    pipe = redis_client.pipeline()
    pipe.get(previous_key)
    pipe.incr(current_key)
    pipe.expire(current_key, window_seconds * 2)
    results = pipe.execute()

    previous_count = int(results[0] or 0)
    current_count = results[1]
    weighted_count = previous_count * (1 - window_position) + current_count

    return weighted_count <= limit, max(0, int(limit - weighted_count))
```

**SlowAPI with FastAPI** (official pattern):
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address, storage_uri="redis://localhost:6379")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/api/search")
@limiter.limit("5/minute")
async def search(request: Request): ...

@app.get("/api/resource")
@limiter.limit("100/minute")    # Short-term
@limiter.limit("1000/hour")     # Long-term
async def get_resource(request: Request): ...
```

**Dynamic Limits by User Tier**:
```python
def get_rate_limit(request: Request) -> str:
    tier = request.state.user.tier
    limits = {"free": "10/minute", "pro": "100/minute", "enterprise": "1000/minute"}
    return limits.get(tier, "10/minute")

@limiter.limit(get_rate_limit)
async def api_endpoint(request: Request): ...
```

**Token Bucket (Lua Atomic Script)**:
```python
lua_script = """
local key, capacity, refill_rate, now, requested = KEYS[1], ARGV[1], ARGV[2], ARGV[3], ARGV[4]
local bucket = redis.call('HMGET', key, 'tokens', 'last_update')
local tokens = tonumber(bucket[1]) or capacity
local last = tonumber(bucket[2]) or now

tokens = math.min(capacity, tokens + (now - last) * refill_rate)
local allowed = tokens >= requested and 1 or 0
if allowed == 1 then tokens = tokens - requested end

redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
redis.call('EXPIRE', key, math.ceil(capacity / refill_rate) * 2)
return {allowed, tokens}
"""
```

**Response Headers (Standard)**:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1706234567
Retry-After: 30  (on 429 only)
```

**Key Identification Strategies**:
| Strategy | Use Case |
|----------|----------|
| IP-based | Public APIs, unauthenticated |
| API Key | Developer/partner APIs |
| User ID | Authenticated users |
| Endpoint+IP | Per-route limiting |

**Anti-Patterns**: Global rate limiting only, no Redis for distributed, fixed window boundary burst, no Retry-After header

---

## CYCLE 48: Logging & Structured Observability (Official Docs)
**Sources**: structlog.org v25.5.0, opentelemetry.io | January 2026

### Structlog Production Configuration
```python
import structlog
from structlog.contextvars import merge_contextvars, bind_contextvars, clear_contextvars

# Shared processors for unified output
shared_processors = [
    merge_contextvars,  # MUST be early - injects request context
    structlog.stdlib.filter_by_level,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    structlog.processors.CallsiteParameterAdder({
        structlog.processors.CallsiteParameter.FILENAME,
        structlog.processors.CallsiteParameter.FUNC_NAME,
        structlog.processors.CallsiteParameter.LINENO,
    }),
]

# Production JSON config
structlog.configure(
    processors=shared_processors + [structlog.processors.JSONRenderer()],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
```

### Request-Scoped Context Variables
```python
# Middleware pattern (FastAPI/Starlette)
async def logging_middleware(request, call_next):
    clear_contextvars()
    bind_contextvars(
        request_id=str(uuid.uuid4()),
        peer=request.client.host,
        path=request.url.path,
    )
    response = await call_next(request)
    bind_contextvars(status_code=response.status_code)
    return response
```

### Filtering with DropEvent
```python
from structlog import DropEvent

class ConditionalDropper:
    def __init__(self, drop_if_peer=None):
        self._drop_peer = drop_if_peer

    def __call__(self, logger, method_name, event_dict):
        if self._drop_peer and event_dict.get("peer") == self._drop_peer:
            raise DropEvent
        return event_dict

# Filter health check noise
structlog.configure(processors=[ConditionalDropper(drop_if_peer="127.0.0.1"), ...])
```

### OpenTelemetry Traces
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, OTLPSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("process_order") as span:
    span.set_attribute("order.id", order_id)
    span.add_event("validation_complete", {"items": 5})
```

### OpenTelemetry Metrics
```python
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider

meter = metrics.get_meter("my.service")

request_counter = meter.create_counter("http.request_count", unit="1")
request_duration = meter.create_histogram("http.request_duration", unit="ms")

request_counter.add(1, {"method": "GET", "route": "/api/users"})
request_duration.record(45.2, {"status_code": "200"})
```

### Unified Structlog + OTel (Trace Correlation)
```python
def add_trace_context(logger, method_name, event_dict):
    span = trace.get_current_span()
    if span.is_recording():
        ctx = span.get_span_context()
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict

structlog.configure(processors=[
    merge_contextvars,
    add_trace_context,  # Correlate logs with traces
    # ... other processors
    structlog.processors.JSONRenderer(),
])
```

### Processor Order (Critical)
1. `merge_contextvars` - Request context injection
2. `filter_by_level` - Early filtering
3. `add_log_level` - Level string
4. `TimeStamper` - ISO timestamp
5. `CallsiteParameterAdder` - file/func/line
6. `JSONRenderer` - Final output

**Anti-Patterns**: String formatting in log calls, missing trace correlation, no request context, `logger.info(f"x={x}")` defeats structuring

---

## CYCLE 49: Caching Patterns (Official Docs)
**Sources**: redis.io, cachetools 6.2.4, aiocache 0.12.2 | January 2026

### Cachetools - Python Memoization
```python
from cachetools import cached, cachedmethod, LRUCache, TTLCache
from threading import Lock

# Thread-safe LRU with decorator
@cached(cache=LRUCache(maxsize=128), lock=Lock())
def get_user(user_id: int):
    return db.query(User).get(user_id)

# TTL cache (auto-expiring)
@cached(cache=TTLCache(maxsize=100, ttl=300))
def get_config(key: str):
    return fetch_config(key)

# With cache info tracking
@cached(cache=LRUCache(maxsize=32), info=True)
def expensive_fn(x):
    return compute(x)

print(expensive_fn.cache_info())  # CacheInfo(hits=3, misses=1, ...)
expensive_fn.cache_clear()  # Clear cache
```

### Cachetools - Instance Method Caching
```python
from cachetools import cachedmethod, LRUCache
from threading import RLock

class UserService:
    def __init__(self):
        self.cache = LRUCache(maxsize=256)
        self._lock = RLock()

    @cachedmethod(lambda self: self.cache, lock=lambda self: self._lock)
    def get_user(self, user_id: int):
        return db.query(User).get(user_id)
```

### Aiocache - Async Caching
```python
from aiocache import Cache, cached
from aiocache.serializers import JsonSerializer

# Basic async cache
cache = Cache(Cache.REDIS, endpoint="localhost", port=6379)
await cache.set("key", {"data": "value"}, ttl=300)
value = await cache.get("key", default=None)

# Multi operations
await cache.multi_set([("k1", "v1"), ("k2", "v2")], ttl=60)
values = await cache.multi_get(["k1", "k2"])

# Decorator pattern
@cached(ttl=300, cache=Cache.REDIS, serializer=JsonSerializer())
async def get_user(user_id: int):
    return await db.fetch_user(user_id)
```

### Redis Client-Side Caching
```
# Enable tracking (server remembers accessed keys)
CLIENT TRACKING ON REDIRECT <client-id>

# Broadcasting mode (subscribe to prefixes)
CLIENT TRACKING ON BCAST PREFIX user: PREFIX order:

# NOLOOP - don't invalidate own writes
CLIENT TRACKING ON NOLOOP
```

### Cache Algorithm Selection
| Algorithm | Use Case |
|-----------|----------|
| **LRU** | General purpose (most common) |
| **LFU** | Frequency-based access |
| **TTL** | Time-sensitive data |
| **FIFO** | Simple, predictable |

### Cache Patterns
```python
# Cache-Aside (Lazy Loading)
async def get_user(user_id):
    cached = await cache.get(f"user:{user_id}")
    if cached:
        return cached
    user = await db.fetch_user(user_id)
    await cache.set(f"user:{user_id}", user, ttl=300)
    return user

# Write-Through
async def update_user(user_id, data):
    user = await db.update_user(user_id, data)
    await cache.set(f"user:{user_id}", user, ttl=300)
    return user
```

**Anti-Patterns**: Caching without TTL, no lock on shared caches, caching frequently-changing data, not handling cache failures

---

*Updated: 2026-01-25 | Cycles 9-49 integrated*
