# Graphiti Integration Pattern for Claude Code

## Overview
Production integration connecting Claude Code to Neo4j via Graphiti temporal knowledge graph.

## Location
`C:\Users\42\.claude\hooks\graphiti_real_integration.py`

## Key Features
- Episode-based ingestion for decisions, patterns, failures, learnings
- Bi-temporal tracking (valid_from, valid_to)
- Hybrid search (semantic + BM25 + graph)
- Entity timeline retrieval

## Required Environment Variables
```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_PASSWORD="your_password"
export OPENAI_API_KEY="sk-..."  # Required for embeddings
export ANTHROPIC_API_KEY="sk-ant-..."  # Optional
```

## Start Neo4j
```bash
docker run -d -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

## Usage Patterns

### Store Session Knowledge
```python
from graphiti_real_integration import get_graphiti_integration

integration = get_graphiti_integration()
await integration.initialize()

result = await integration.store_session_knowledge(
    session_id="session_123",
    decisions=[{"title": "Use async", "rationale": "Better performance"}],
    patterns=[{"name": "Singleton", "description": "For integration instance"}],
    learnings=[{"topic": "Type safety", "insight": "Use TYPE_CHECKING"}]
)
```

### Search Knowledge
```python
results = await integration.search_knowledge(
    query="authentication decisions",
    limit=10,
    include_nodes=True
)
```

### Get Entity Timeline
```python
timeline = await integration.get_entity_timeline(
    entity_name="GraphitiRealIntegration",
    limit=20
)
```

## CLI Usage
```bash
python graphiti_real_integration.py health
python graphiti_real_integration.py search "architecture decisions"
python graphiti_real_integration.py store --session-id abc123 --decision '{"title": "..."}'
python graphiti_real_integration.py timeline "EntityName"
```

## Type Safety Pattern
For optional SDK imports:
```python
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graphiti_core import Graphiti as GraphitiType

try:
    from graphiti_core import Graphiti
    AVAILABLE = True
except ImportError:
    AVAILABLE = False
    Graphiti = None  # type: ignore

class MyIntegration:
    def __init__(self):
        self._graphiti: Any = None  # Use Any for runtime flexibility
    
    async def init(self):
        if not AVAILABLE or Graphiti is None:
            return False
        self._graphiti = Graphiti(...)  # Now type-safe
```

## Integration with V31.1 Architecture
- Used in memory-gateway-hook.py for pre-task search
- Complements claude-mem and episodic-memory
- Provides temporal entity relationships other systems lack
