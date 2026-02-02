# MCP Access Optimization Patterns V1

**Created**: 2026-01-23
**Purpose**: Maximize MCP server utilization and access efficiency

---

## Available MCP Servers (40+)

### Memory Servers
| Server | Tool Prefix | Primary Use |
|--------|-------------|-------------|
| **serena** | `mcp__serena__` | Code-aware memory, symbol search |
| **claude-mem** | `mcp__claude_mem__` | Observation tracking, timeline |
| **episodic-memory** | `mcp__episodic_memory__` | Conversation search |
| **qdrant** | `mcp__qdrant__` | Vector semantic search |
| **redis** | `mcp__redis__` | Fast key-value cache |
| **sqlite** | `mcp__sqlite__` | Structured data queries |

### Creative Servers
| Server | Tool Prefix | Primary Use |
|--------|-------------|-------------|
| **touchdesigner-creative** | `mcp__touchdesigner_creative__` | Real-time visualization |
| **comfyui** | `mcp__comfyui__` | Image generation workflows |
| **blender** | `mcp__blender__` | 3D modeling/rendering |
| **everart** | `mcp__everart__` | Art generation |

### Research Servers
| Server | Tool Prefix | Primary Use |
|--------|-------------|-------------|
| **exa** | `mcp__exa__` | Web search, deep research |
| **context7** | `mcp__context7__` | SDK documentation |
| **brave-search** | `mcp__brave__` | General web search |
| **tavily** | `mcp__tavily__` | Research-focused search |

### Development Servers
| Server | Tool Prefix | Primary Use |
|--------|-------------|-------------|
| **github** | `mcp__github__` | Repository management |
| **git** | `mcp__git__` | Version control |
| **sentry** | `mcp__sentry__` | Error monitoring |
| **sequentialthinking** | `mcp__sequentialthinking__` | Structured reasoning |

---

## Optimal Access Patterns

### Pattern 1: Memory Search Cascade
```
QUERY: "How did I solve X?"

1. Serena Memory (code-aware, fast)
   └─▶ mcp__serena__list_memories() 
   └─▶ mcp__serena__read_memory(relevant_name)

2. Claude-Mem (observations, decisions)
   └─▶ mcp__claude_mem__search(query, limit=10)
   └─▶ mcp__claude_mem__timeline(anchor=id)
   └─▶ mcp__claude_mem__get_observations([ids])

3. Episodic (conversation history)
   └─▶ mcp__episodic_memory__search(query)
   └─▶ mcp__episodic_memory__read(path)
```

### Pattern 2: Research Pipeline
```
QUERY: "Latest patterns for X"

1. Exa Web Search (cutting edge)
   └─▶ mcp__exa__web_search_exa(query)
   └─▶ mcp__exa__get_code_context_exa(query, tokensNum=8000)

2. Context7 Docs (official documentation)
   └─▶ mcp__context7__resolve_library_id(query, libraryName)
   └─▶ mcp__context7__query_docs(libraryId, query)

3. Deep Research (complex topics)
   └─▶ mcp__exa__deep_researcher_start(instructions)
   └─▶ mcp__exa__deep_researcher_check(taskId) [poll until complete]
```

### Pattern 3: Creative Generation
```
TASK: Generate visualization

1. TouchDesigner MCP
   └─▶ mcp__touchdesigner_creative__ping()
   └─▶ mcp__touchdesigner_creative__create_node(type, params)
   └─▶ mcp__touchdesigner_creative__set_parameters(path, params)
   └─▶ mcp__touchdesigner_creative__connect_nodes(source, dest)

2. Template-based
   └─▶ mcp__touchdesigner_creative__create_network_from_template(template)
   └─▶ mcp__touchdesigner_creative__execute_script(code)
```

### Pattern 4: Code Intelligence
```
TASK: Understand codebase

1. Serena Symbol Navigation
   └─▶ mcp__serena__get_symbols_overview(relative_path)
   └─▶ mcp__serena__find_symbol(name_path, include_body=True)
   └─▶ mcp__serena__find_referencing_symbols(name_path)

2. Pattern Search
   └─▶ mcp__serena__search_for_pattern(substring_pattern)
   └─▶ mcp__serena__list_dir(relative_path, recursive=True)
```

---

## Parallel Access Optimization

### Parallel Memory Search
```python
# Execute in parallel for speed
results = await asyncio.gather(
    mcp__serena__search_memories(query),
    mcp__claude_mem__search(query),
    mcp__episodic_memory__search(query)
)
# Merge and rank results
```

### Parallel Research
```python
# Multiple search providers in parallel
results = await asyncio.gather(
    mcp__exa__web_search_exa(query),
    mcp__brave__search(query),
    mcp__tavily__search(query)
)
# Cross-reference for reliability
```

---

## Caching Strategy

### Session-Level Cache
```python
# Cache expensive lookups
_memory_cache = {}

async def cached_memory_read(name: str) -> str:
    if name not in _memory_cache:
        _memory_cache[name] = await mcp__serena__read_memory(name)
    return _memory_cache[name]
```

### Cross-Session Persistence
```python
# Store frequently accessed patterns
FREQUENTLY_ACCESSED = [
    "sdk_code_patterns_v31",
    "sdk_ecosystem_v30",
    "self_improvement_loop_v1",
    "memory_architecture_v11"
]

# Pre-load at session start
async def preload_memories():
    for mem in FREQUENTLY_ACCESSED:
        await cached_memory_read(mem)
```

---

## Server Selection Guide

### By Task Type
| Task | Primary Server | Fallback |
|------|----------------|----------|
| **Find past solution** | serena → claude-mem | episodic |
| **Research new topic** | exa → context7 | brave |
| **Generate visual** | touchdesigner | comfyui |
| **Analyze code** | serena | github |
| **Store learning** | serena → claude-mem | sqlite |

### By Project
| Project | Primary Servers |
|---------|-----------------|
| **UNLEASH** | serena, claude-mem, exa, context7 |
| **WITNESS** | touchdesigner, serena, qdrant |
| **TRADING** | serena, redis, github, sentry |

---

## Error Handling

### Quota Management
```python
# Context7 quota exceeded → fallback to Exa
try:
    docs = await mcp__context7__query_docs(libraryId, query)
except QuotaExceededError:
    docs = await mcp__exa__get_code_context_exa(query, tokensNum=8000)
```

### Connection Resilience
```python
# TouchDesigner connection check
status = await mcp__touchdesigner_creative__ping()
if not status.connected:
    # Queue commands for later
    await queue_td_commands(commands)
```

---

## Performance Metrics

### Optimal Response Times
| Operation | Target | Max |
|-----------|--------|-----|
| Memory read | <100ms | 500ms |
| Symbol search | <200ms | 1s |
| Web search | <2s | 10s |
| Deep research | <60s | 5min |
| TD node create | <50ms | 200ms |

### Monitoring
```python
import time

async def timed_mcp_call(func, *args, **kwargs):
    start = time.time()
    result = await func(*args, **kwargs)
    elapsed = time.time() - start
    
    # Log slow calls
    if elapsed > 1.0:
        log_slow_call(func.__name__, elapsed)
    
    return result
```

---

## Quick Reference Commands

### Memory Operations
```python
# List all memories
await mcp__serena__list_memories()

# Read specific memory
await mcp__serena__read_memory("sdk_code_patterns_v31")

# Write new memory
await mcp__serena__write_memory(name, content)

# Search across systems
await unified_memory_search(query)  # Uses ImportanceScorer
```

### Research Operations
```python
# Quick web search
await mcp__exa__web_search_exa(query, numResults=10)

# Code examples
await mcp__exa__get_code_context_exa(query, tokensNum=8000)

# Deep research (async)
task_id = await mcp__exa__deep_researcher_start(instructions)
result = await mcp__exa__deep_researcher_check(task_id)  # Poll
```

### Creative Operations
```python
# Check TD connection
await mcp__touchdesigner_creative__ping()

# Create node
await mcp__touchdesigner_creative__create_node(type="noiseTOP", name="noise1")

# Batch operations
await mcp__touchdesigner_creative__batch(commands)
```

---

*Optimization Guide Version: 1.0 | 40+ MCP Servers Mapped*
