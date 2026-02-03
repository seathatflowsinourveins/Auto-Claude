"""
MEMORY INTEGRATION ITERATIONS - Letta, Mem0, Qdrant Unified Memory
===================================================================
Full memory system integration with persistence and recall patterns
"""

import asyncio
import os
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Any
from dotenv import load_dotenv
import httpx

load_dotenv('.config/.env')

# ============================================================================
# MEMORY INTEGRATION TOPICS
# ============================================================================

MEMORY_TOPICS = [
    # Memory Architecture
    {"topic": "Letta MemGPT: core_memory, recall_memory, archival_memory architecture", "focus": "letta"},
    {"topic": "Mem0 memory layer: add, search, update, delete operations with metadata", "focus": "mem0"},
    {"topic": "Qdrant vector memory: collections, points, filters, payloads", "focus": "qdrant"},
    {"topic": "LangGraph checkpointing: SqliteSaver, MemorySaver, PostgresSaver", "focus": "langgraph"},

    # Memory Patterns
    {"topic": "Working memory pattern: recent context, attention window, decay", "focus": "working"},
    {"topic": "Episodic memory pattern: event storage, temporal indexing, replay", "focus": "episodic"},
    {"topic": "Semantic memory pattern: knowledge graphs, concept relations, inference", "focus": "semantic"},
    {"topic": "Procedural memory pattern: skill storage, action sequences, habits", "focus": "procedural"},

    # Memory Operations
    {"topic": "Memory consolidation: summarization, importance scoring, compression", "focus": "consolidation"},
    {"topic": "Memory retrieval: similarity search, recency weighting, hybrid recall", "focus": "retrieval"},
    {"topic": "Memory forgetting: decay functions, capacity limits, relevance pruning", "focus": "forgetting"},
    {"topic": "Memory sharing: cross-agent memory, namespace isolation, synchronization", "focus": "sharing"},

    # Production Memory
    {"topic": "Persistent memory: SQLite, PostgreSQL, Redis backends for agents", "focus": "persistence"},
    {"topic": "Distributed memory: sharding, replication, consistency models", "focus": "distributed"},
    {"topic": "Memory caching: embedding cache, query cache, result cache", "focus": "caching"},
    {"topic": "Memory observability: usage tracking, recall metrics, latency profiling", "focus": "observability"},
]


@dataclass
class MemoryEntry:
    """Universal memory entry structure."""
    id: str
    content: str
    embedding: Optional[list] = None
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    importance: float = 1.0
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


class UnifiedMemorySystem:
    """Unified memory system integrating Letta, Mem0, and Qdrant patterns."""

    def __init__(self):
        self.qdrant = None
        self.keys = {
            "jina": os.getenv("JINA_API_KEY"),
            "tavily": os.getenv("TAVILY_API_KEY"),
            "perplexity": os.getenv("PERPLEXITY_API_KEY"),
        }
        self.exa = None

        # Memory layers
        self.working_memory = []          # Recent context (limited capacity)
        self.episodic_memory = []         # Events with temporal context
        self.semantic_memory = {}         # Concepts and relations
        self.procedural_memory = []       # Skills and procedures

        # Memory indices
        self.importance_index = {}
        self.temporal_index = {}
        self.embedding_cache = {}

        # Stats
        self.stats = {
            "stores": 0, "retrieves": 0, "consolidations": 0,
            "vectors": 0, "sources": 0, "insights": 0
        }

    async def initialize(self):
        """Initialize all memory backends."""
        from exa_py import Exa
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.exa = Exa(os.getenv("EXA_API_KEY"))
        self.qdrant = QdrantClient(":memory:")

        # Create memory collections
        collections = {
            "working": 1024,      # High-churn, recent items
            "episodic": 1024,     # Event memories
            "semantic": 1024,     # Conceptual knowledge
            "procedural": 1024,   # Skills and procedures
            "consolidated": 1024, # Compressed memories
        }

        for name, dim in collections.items():
            try:
                self.qdrant.create_collection(
                    name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
                )
            except:
                pass

        print("[OK] Unified Memory System initialized")
        print(f"    Collections: {list(collections.keys())}")

    async def store(self, content: str, memory_type: str = "working", metadata: dict = None) -> str:
        """Store content in appropriate memory layer."""
        entry = MemoryEntry(
            id=f"{memory_type}_{int(time.time() * 1000)}",
            content=content,
            metadata=metadata or {},
            importance=self._calculate_importance(content)
        )

        # Route to appropriate layer
        if memory_type == "working":
            self.working_memory.append(entry)
            # Enforce capacity limit
            if len(self.working_memory) > 20:
                await self._consolidate_working_memory()

        elif memory_type == "episodic":
            entry.metadata["timestamp"] = time.time()
            self.episodic_memory.append(entry)

        elif memory_type == "semantic":
            concepts = self._extract_concepts(content)
            for concept in concepts:
                self.semantic_memory[concept] = entry

        elif memory_type == "procedural":
            self.procedural_memory.append(entry)

        self.stats["stores"] += 1
        return entry.id

    async def retrieve(self, query: str, memory_type: str = "all", k: int = 5) -> list:
        """Retrieve relevant memories."""
        results = []

        async with httpx.AsyncClient(timeout=60) as client:
            # Get query embedding
            query_vec = await self._embed(client, [query])
            if not query_vec:
                return results

            # Search appropriate collections
            collections = [memory_type] if memory_type != "all" else ["working", "episodic", "semantic"]

            for coll in collections:
                try:
                    search_result = self.qdrant.query_points(
                        collection_name=coll,
                        query=query_vec[0],
                        limit=k
                    )
                    for point in search_result.points:
                        results.append({
                            "id": point.id,
                            "content": point.payload.get("content", ""),
                            "score": point.score,
                            "memory_type": coll
                        })
                except:
                    pass

        # Sort by score
        results.sort(key=lambda x: -x["score"])
        self.stats["retrieves"] += 1

        return results[:k]

    async def _consolidate_working_memory(self):
        """Consolidate working memory into long-term storage."""
        if len(self.working_memory) < 10:
            return

        # Take oldest entries
        to_consolidate = self.working_memory[:10]
        self.working_memory = self.working_memory[10:]

        # Summarize
        contents = [e.content for e in to_consolidate]
        summary = f"Consolidated: {' | '.join(c[:50] for c in contents)}"

        # Store in episodic with consolidation marker
        await self.store(summary, "episodic", {"consolidated": True, "source_count": len(to_consolidate)})
        self.stats["consolidations"] += 1

    def _calculate_importance(self, content: str) -> float:
        """Calculate importance score for content."""
        # Simple heuristics
        score = 0.5

        # Length factor
        if len(content) > 200:
            score += 0.1

        # Keywords factor
        important_keywords = ["error", "critical", "important", "key", "must", "required"]
        for kw in important_keywords:
            if kw in content.lower():
                score += 0.1

        return min(score, 1.0)

    def _extract_concepts(self, content: str) -> list:
        """Extract key concepts from content."""
        # Simple extraction - in production use NER
        words = content.lower().split()
        concepts = [w for w in words if len(w) > 5 and w.isalpha()]
        return concepts[:5]

    async def _embed(self, client: httpx.AsyncClient, texts: list) -> list:
        """Get embeddings via Jina."""
        try:
            r = await client.post(
                'https://api.jina.ai/v1/embeddings',
                headers={'Authorization': f'Bearer {self.keys["jina"]}', 'Content-Type': 'application/json'},
                json={'model': 'jina-embeddings-v3', 'input': texts[:10], 'task': 'retrieval.passage'}
            )
            return [d.get('embedding', []) for d in r.json().get('data', [])]
        except:
            return []


class MemoryIntegrationExecutor:
    """Execute memory integration research and implementation."""

    def __init__(self):
        self.memory = None
        self.exa = None
        self.keys = {
            "tavily": os.getenv("TAVILY_API_KEY"),
            "jina": os.getenv("JINA_API_KEY"),
            "perplexity": os.getenv("PERPLEXITY_API_KEY"),
        }
        self.stats = {"sources": 0, "vectors": 0, "insights": 0, "memory_ops": 0}

    async def initialize(self):
        from exa_py import Exa
        self.exa = Exa(os.getenv("EXA_API_KEY"))
        self.memory = UnifiedMemorySystem()
        await self.memory.initialize()
        print("[OK] Memory Integration Executor ready")

    async def research_and_integrate(self, topic: str, focus: str) -> dict:
        """Research topic and integrate findings into memory."""
        result = {"sources": [], "findings": [], "vectors": 0, "memory_ops": []}

        async with httpx.AsyncClient(timeout=90) as client:
            # Research
            tasks = [
                self._exa_search(topic, focus),
                self._tavily_search(client, topic),
                self._perplexity_deep(client, topic, focus),
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for r in responses:
                if isinstance(r, dict):
                    result["sources"].extend(r.get("sources", []))
                    result["findings"].extend(r.get("findings", []))

            # Integrate findings into memory
            for finding in result["findings"][:5]:
                # Determine memory type based on focus
                memory_type = self._determine_memory_type(focus)
                await self.memory.store(finding, memory_type, {"topic": topic, "focus": focus})
                result["memory_ops"].append({"op": "store", "type": memory_type})

            # Embed to vector memory
            if result["sources"]:
                texts = [s.get("text", "")[:500] for s in result["sources"] if s.get("text")]
                if texts:
                    result["vectors"] = await self._embed_to_memory(client, texts, focus)

        return result

    def _determine_memory_type(self, focus: str) -> str:
        """Determine appropriate memory type based on focus."""
        mapping = {
            "letta": "semantic",
            "mem0": "working",
            "qdrant": "semantic",
            "langgraph": "procedural",
            "working": "working",
            "episodic": "episodic",
            "semantic": "semantic",
            "procedural": "procedural",
            "consolidation": "episodic",
            "retrieval": "working",
            "forgetting": "working",
            "sharing": "semantic",
            "persistence": "episodic",
            "distributed": "semantic",
            "caching": "working",
            "observability": "procedural",
        }
        return mapping.get(focus, "working")

    async def _exa_search(self, topic: str, focus: str) -> dict:
        try:
            search = self.exa.search_and_contents(topic, type="neural", num_results=5, text=True, highlights=True)
            sources = [{"title": r.title, "url": r.url, "text": r.text[:500] if r.text else "", "sdk": "exa"} for r in search.results]
            findings = [f"[exa] {r.title}" for r in search.results[:2]]
            if search.results and hasattr(search.results[0], 'highlights') and search.results[0].highlights:
                findings.append(f"[exa-h] {search.results[0].highlights[0][:120]}")
            return {"sources": sources, "findings": findings}
        except:
            return {"sources": [], "findings": []}

    async def _tavily_search(self, client: httpx.AsyncClient, topic: str) -> dict:
        try:
            r = await client.post('https://api.tavily.com/search', json={
                'api_key': self.keys["tavily"],
                'query': topic,
                'search_depth': 'advanced',
                'include_answer': True,
                'max_results': 5
            })
            data = r.json()
            sources = [{"title": s.get("title", ""), "url": s.get("url", ""), "text": s.get("content", "")[:500], "sdk": "tavily"} for s in data.get("results", [])]
            findings = [f"[tavily] {data.get('answer', '')[:200]}"] if data.get("answer") else []
            return {"sources": sources, "findings": findings}
        except:
            return {"sources": [], "findings": []}

    async def _perplexity_deep(self, client: httpx.AsyncClient, topic: str, focus: str) -> dict:
        try:
            prompt = f"Memory system implementation for: {topic}. Focus on {focus} patterns and best practices."
            r = await client.post(
                'https://api.perplexity.ai/chat/completions',
                headers={'Authorization': f'Bearer {self.keys["perplexity"]}', 'Content-Type': 'application/json'},
                json={'model': 'sonar-pro', 'messages': [{'role': 'user', 'content': prompt}], 'return_citations': True}
            )
            data = r.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            citations = data.get('citations', [])
            sources = [{"title": f"Citation {i+1}", "url": c, "text": "", "sdk": "perplexity"} for i, c in enumerate(citations[:3])]
            findings = [f"[perplexity] {content[:220]}"] if content else []
            return {"sources": sources, "findings": findings}
        except:
            return {"sources": [], "findings": []}

    async def _embed_to_memory(self, client: httpx.AsyncClient, texts: list, focus: str) -> int:
        try:
            from qdrant_client.models import PointStruct

            r = await client.post(
                'https://api.jina.ai/v1/embeddings',
                headers={'Authorization': f'Bearer {self.keys["jina"]}', 'Content-Type': 'application/json'},
                json={'model': 'jina-embeddings-v3', 'input': texts[:10], 'task': 'retrieval.passage'}
            )
            embeddings = [d.get('embedding', []) for d in r.json().get('data', [])]

            if not embeddings or not embeddings[0]:
                return 0

            collection = self._determine_memory_type(focus)
            base_id = int(time.time() * 1000) % 1000000
            points = [PointStruct(id=base_id + i, vector=emb, payload={"content": texts[i][:300], "focus": focus}) for i, emb in enumerate(embeddings)]
            self.memory.qdrant.upsert(collection, points=points)

            return len(points)
        except:
            return 0

    async def run_iteration(self, topic_data: dict, index: int) -> dict:
        topic = topic_data["topic"]
        focus = topic_data["focus"]

        print(f"\n[{index:02d}] [{focus}] {topic[:50]}...")

        start = time.time()
        result = await self.research_and_integrate(topic, focus)
        latency = time.time() - start

        self.stats["sources"] += len(result["sources"])
        self.stats["vectors"] += result["vectors"]
        self.stats["insights"] += len(result["findings"])
        self.stats["memory_ops"] += len(result["memory_ops"])

        print(f"    Src:{len(result['sources'])} Vec:{result['vectors']} Find:{len(result['findings'])} MemOps:{len(result['memory_ops'])} [{latency:.1f}s]")

        if result["findings"]:
            f = result["findings"][0]
            clean = f.encode('ascii', 'replace').decode('ascii')
            print(f"    -> {clean[:60]}...")

        return {
            "topic": topic,
            "focus": focus,
            "sources": len(result["sources"]),
            "vectors": result["vectors"],
            "findings": result["findings"][:3],
            "memory_ops": len(result["memory_ops"]),
            "latency": latency
        }

    def get_memory_stats(self) -> dict:
        """Get unified memory system stats."""
        return {
            "working_memory_size": len(self.memory.working_memory),
            "episodic_memory_size": len(self.memory.episodic_memory),
            "semantic_memory_size": len(self.memory.semantic_memory),
            "procedural_memory_size": len(self.memory.procedural_memory),
            "total_stores": self.memory.stats["stores"],
            "total_retrieves": self.memory.stats["retrieves"],
            "consolidations": self.memory.stats["consolidations"],
            "qdrant_collections": {
                "working": self.memory.qdrant.count("working").count,
                "episodic": self.memory.qdrant.count("episodic").count,
                "semantic": self.memory.qdrant.count("semantic").count,
                "procedural": self.memory.qdrant.count("procedural").count,
            }
        }


async def main():
    print("="*70)
    print("MEMORY INTEGRATION ITERATIONS")
    print("="*70)
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Topics: {len(MEMORY_TOPICS)}")
    print("="*70)

    executor = MemoryIntegrationExecutor()
    await executor.initialize()

    all_results = []

    for i, topic_data in enumerate(MEMORY_TOPICS, 1):
        try:
            result = await executor.run_iteration(topic_data, i)
            all_results.append(result)
        except Exception as e:
            print(f"    [ERR] {str(e)[:50]}")

        await asyncio.sleep(0.3)

    # Summary
    print("\n" + "="*70)
    print("MEMORY INTEGRATION COMPLETE")
    print("="*70)

    print(f"\n  Iterations: {len(all_results)}")
    print(f"  Total Sources: {executor.stats['sources']}")
    print(f"  Total Vectors: {executor.stats['vectors']}")
    print(f"  Total Insights: {executor.stats['insights']}")
    print(f"  Memory Operations: {executor.stats['memory_ops']}")

    avg_latency = sum(r["latency"] for r in all_results) / len(all_results) if all_results else 0
    print(f"  Avg Latency: {avg_latency:.1f}s")

    # Memory system stats
    print("\n  UNIFIED MEMORY SYSTEM:")
    mem_stats = executor.get_memory_stats()
    print(f"    Working Memory: {mem_stats['working_memory_size']} entries")
    print(f"    Episodic Memory: {mem_stats['episodic_memory_size']} entries")
    print(f"    Semantic Memory: {mem_stats['semantic_memory_size']} concepts")
    print(f"    Procedural Memory: {mem_stats['procedural_memory_size']} entries")

    print("\n  QDRANT COLLECTIONS:")
    for coll, count in mem_stats["qdrant_collections"].items():
        print(f"    {coll}: {count} vectors")

    print("\n  BY FOCUS:")
    focus_stats = {}
    for r in all_results:
        focus_stats[r["focus"]] = focus_stats.get(r["focus"], 0) + 1
    for focus, count in sorted(focus_stats.items()):
        print(f"    {focus}: {count}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "stats": executor.stats,
        "memory_stats": mem_stats,
        "results": all_results
    }

    with open("memory_integration_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Results saved to memory_integration_results.json")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
