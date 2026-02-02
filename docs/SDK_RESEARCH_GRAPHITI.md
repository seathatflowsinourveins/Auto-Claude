# Graphiti by Zep - Temporal Knowledge Graph Framework

*Deep Research Report - 2026-01-19*
*Research Time: 118 seconds*

---

## 1. Core Concepts

Graphiti is an open-source temporal knowledge graph framework developed by Zep AI for AI agents in dynamic environments.

### Data Types
| Type | Description |
|------|-------------|
| **Entity Nodes** | Real-world entities (people, places, concepts) with summaries |
| **Entity Edges** | Relationships between nodes with semantic facts |
| **Episodic Nodes** | Raw data inputs (conversations, documents, events) |

### Temporal Attributes
Each edge includes:
- **Valid From**: When the fact becomes valid
- **Valid Until**: When the fact becomes invalid
- Enables tracking how relationships change over time

---

## 2. Graph Construction and Updates

- Ingests episodic data (conversations, documents, events)
- Uses LLMs to extract structured knowledge automatically
- Smart graph updates evaluate new entities against existing schema
- Incremental, real-time updates while preserving history
- Hybrid search: semantic + full-text (BM25) + graph-based
- Sub-100ms retrieval performance

---

## 3. Temporal Queries and Reasoning

### Dual-Timestamp Model
- **Event Time**: When a fact is valid
- **Ingestion Time**: When the fact was recorded

### Capabilities
- Multi-hop reasoning across evolving relationships
- Query state at specific points in time
- Trace evolution of entity attributes
- Combine temporal, full-text, semantic, and graph queries

---

## 4. Integration with Neo4j

**Requirements**: Neo4j version 5.26+

```bash
# Docker deployment
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.26
```

Features:
- Cypher query language support
- Real-time updates and complex queries
- Graph visualization and management tools

---

## 5. Use Cases for AI Agent Memory

- **Personal AI Assistants**: Evolve with user interactions
- **Business Integration**: Merge personal knowledge with real-time data
- **Autonomous Tasks**: Process state changes from dynamic sources
- **Memory Consistency**: Identity coherence across sessions

### Addresses Challenges
- Temporal reasoning
- Entity disambiguation
- Relationship modeling
- Context evolution

---

## 6. Python Implementation

```python
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from datetime import datetime, timezone

# Initialize Graphiti client
client = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)

# Add an episode (conversation/document)
episode_id = client.add_episode(
    episode_type=EpisodeType.CONVERSATION,
    text="Kendra loves Adidas shoes.",
    timestamp=datetime.now(timezone.utc)
)

# Query the graph
results = client.query(
    "MATCH (e:Entity)-[r]->(f:Entity) RETURN e, r, f LIMIT 10"
)
for record in results:
    print(record)
```

---

## Integration with Unleash Platform

```python
# In ecosystem_orchestrator.py
from graphiti_core import Graphiti

async def init_graphiti(self):
    self.graphiti = Graphiti(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD")
    )

async def ingest_to_graph(self, content: str, source: str):
    await self.graphiti.add_episode(
        episode_type=EpisodeType.TEXT,
        text=content,
        source=source,
        timestamp=datetime.now(timezone.utc)
    )
```

---

*Sources: [help.getzep.com](https://help.getzep.com), [github.com/getzep/graphiti](https://github.com/getzep/graphiti), [neo4j.com](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory)*
