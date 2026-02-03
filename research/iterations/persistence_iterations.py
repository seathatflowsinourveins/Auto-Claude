"""
PERSISTENCE ITERATIONS - Database Backend Patterns
====================================================
Memory persistence with SQLite, Redis, PostgreSQL patterns
"""

import asyncio
import os
import json
import time
import sqlite3
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import httpx

load_dotenv('.config/.env')

# ============================================================================
# PERSISTENCE TOPICS
# ============================================================================

PERSISTENCE_TOPICS = [
    # SQLite Patterns
    {"topic": "SQLite for LLM: conversation history, checkpoints, embeddings", "backend": "sqlite"},
    {"topic": "SQLite FTS5: full-text search for agent memory", "backend": "sqlite"},
    {"topic": "SQLite WAL mode: concurrent reads for agent queries", "backend": "sqlite"},
    {"topic": "SQLite JSON1: storing structured agent state", "backend": "sqlite"},

    # Redis Patterns
    {"topic": "Redis for LLM caching: prompt cache, response cache, embedding cache", "backend": "redis"},
    {"topic": "Redis Streams: event sourcing for agent actions", "backend": "redis"},
    {"topic": "Redis Search: vector similarity with RediSearch", "backend": "redis"},
    {"topic": "Redis pub/sub: real-time agent coordination", "backend": "redis"},

    # PostgreSQL Patterns
    {"topic": "PostgreSQL pgvector: native vector similarity search", "backend": "postgresql"},
    {"topic": "PostgreSQL JSONB: flexible agent state storage", "backend": "postgresql"},
    {"topic": "PostgreSQL partitioning: scaling conversation history", "backend": "postgresql"},
    {"topic": "PostgreSQL triggers: automated memory consolidation", "backend": "postgresql"},

    # Hybrid Patterns
    {"topic": "Multi-tier storage: hot/warm/cold data strategies", "backend": "hybrid"},
    {"topic": "Write-through cache: Redis + PostgreSQL consistency", "backend": "hybrid"},
    {"topic": "Event sourcing: append-only logs for agent state", "backend": "hybrid"},
    {"topic": "CQRS: command query separation for agent operations", "backend": "hybrid"},

    # Distributed Patterns
    {"topic": "Distributed SQLite: LiteFS for multi-region agents", "backend": "distributed"},
    {"topic": "Redis Cluster: sharding for high-throughput caching", "backend": "distributed"},
    {"topic": "CockroachDB: distributed SQL for global agents", "backend": "distributed"},
    {"topic": "Vitess: MySQL sharding for conversation scale", "backend": "distributed"},
]


@dataclass
class PersistenceResult:
    topic: str
    backend: str
    sources: list
    findings: list
    implementation: dict
    latency: float


class PersistenceExecutor:
    """Research and implement persistence patterns."""

    def __init__(self):
        self.exa = None
        self.db = None
        self.keys = {
            "tavily": os.getenv("TAVILY_API_KEY"),
            "jina": os.getenv("JINA_API_KEY"),
            "perplexity": os.getenv("PERPLEXITY_API_KEY"),
        }
        self.stats = {"sources": 0, "findings": 0, "implementations": 0}
        self.implementations = []

    async def initialize(self):
        from exa_py import Exa
        self.exa = Exa(os.getenv("EXA_API_KEY"))

        # Initialize SQLite for demo
        self.db = sqlite3.connect(":memory:")
        self._setup_sqlite()

        print("[OK] Persistence Executor initialized")
        print("    SQLite: In-memory demo database ready")

    def _setup_sqlite(self):
        """Setup SQLite schema for persistence demo."""
        cursor = self.db.cursor()

        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                content TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Memory table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY,
                key TEXT UNIQUE,
                value TEXT,
                memory_type TEXT,
                importance REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Checkpoints table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY,
                thread_id TEXT,
                checkpoint_id TEXT,
                state BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.db.commit()

    async def research_and_implement(self, topic: str, backend: str) -> PersistenceResult:
        """Research persistence pattern and implement demo."""
        result = {"sources": [], "findings": [], "implementation": {}}

        async with httpx.AsyncClient(timeout=90) as client:
            # Research
            tasks = [
                self._exa_search(topic),
                self._tavily_search(client, topic),
                self._perplexity_search(client, topic, backend),
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for r in responses:
                if isinstance(r, dict):
                    result["sources"].extend(r.get("sources", []))
                    result["findings"].extend(r.get("findings", []))

            # Implement pattern
            result["implementation"] = self._implement_pattern(backend, topic)

        return result

    async def _exa_search(self, topic: str) -> dict:
        try:
            search = self.exa.search_and_contents(topic, type="auto", num_results=4, text=True, highlights=True)
            sources = [{"title": r.title, "url": r.url, "text": r.text[:400] if r.text else ""} for r in search.results]
            findings = [f"[exa] {r.title}" for r in search.results[:2]]
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
                'max_results': 4
            })
            data = r.json()
            sources = [{"title": s.get("title", ""), "url": s.get("url", ""), "text": s.get("content", "")[:400]} for s in data.get("results", [])]
            findings = [f"[tavily] {data.get('answer', '')[:180]}"] if data.get("answer") else []
            return {"sources": sources, "findings": findings}
        except:
            return {"sources": [], "findings": []}

    async def _perplexity_search(self, client: httpx.AsyncClient, topic: str, backend: str) -> dict:
        try:
            r = await client.post(
                'https://api.perplexity.ai/chat/completions',
                headers={'Authorization': f'Bearer {self.keys["perplexity"]}'},
                json={
                    'model': 'sonar-pro',
                    'messages': [{'role': 'user', 'content': f"Implementation guide for {topic} using {backend}. Include schema and code examples."}],
                    'return_citations': True
                }
            )
            data = r.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            findings = [f"[perplexity] {content[:200]}"] if content else []
            return {"sources": [], "findings": findings}
        except:
            return {"sources": [], "findings": []}

    def _implement_pattern(self, backend: str, topic: str) -> dict:
        """Implement persistence pattern demo."""
        implementation = {"backend": backend, "status": "implemented", "operations": []}

        if backend == "sqlite":
            # Demo SQLite operations
            cursor = self.db.cursor()

            # Insert conversation
            cursor.execute(
                "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
                ("demo_session", "user", f"Research: {topic[:50]}")
            )

            # Insert memory
            cursor.execute(
                "INSERT OR REPLACE INTO memory (key, value, memory_type, importance) VALUES (?, ?, ?, ?)",
                (f"topic_{hash(topic) % 10000}", topic[:100], "semantic", 0.8)
            )

            # Insert checkpoint
            cursor.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_id, state) VALUES (?, ?, ?)",
                ("thread_1", f"cp_{int(time.time())}", json.dumps({"topic": topic[:50]}).encode())
            )

            self.db.commit()

            # Query stats
            cursor.execute("SELECT COUNT(*) FROM conversations")
            conv_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM memory")
            mem_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM checkpoints")
            cp_count = cursor.fetchone()[0]

            implementation["operations"] = [
                {"table": "conversations", "count": conv_count},
                {"table": "memory", "count": mem_count},
                {"table": "checkpoints", "count": cp_count}
            ]
            implementation["schema"] = {
                "conversations": ["id", "session_id", "role", "content", "embedding", "created_at"],
                "memory": ["id", "key", "value", "memory_type", "importance", "access_count"],
                "checkpoints": ["id", "thread_id", "checkpoint_id", "state", "created_at"]
            }

        elif backend == "redis":
            implementation["patterns"] = {
                "cache": "SET/GET with TTL for prompt/response caching",
                "streams": "XADD/XREAD for event sourcing",
                "search": "FT.CREATE/FT.SEARCH for vector similarity",
                "pubsub": "PUBLISH/SUBSCRIBE for agent coordination"
            }
            implementation["config"] = {
                "max_memory": "2gb",
                "eviction_policy": "allkeys-lru",
                "persistence": "AOF"
            }

        elif backend == "postgresql":
            implementation["patterns"] = {
                "pgvector": "CREATE EXTENSION vector; CREATE INDEX USING ivfflat",
                "jsonb": "JSONB columns with GIN index for flexible state",
                "partitioning": "PARTITION BY RANGE on created_at",
                "triggers": "AFTER INSERT trigger for consolidation"
            }
            implementation["schema"] = {
                "embeddings": "id, content, embedding vector(1024), metadata JSONB",
                "conversations": "id, thread_id, messages JSONB, created_at TIMESTAMPTZ",
                "checkpoints": "thread_id, checkpoint_ns, state BYTEA"
            }

        elif backend == "hybrid":
            implementation["architecture"] = {
                "hot": "Redis - recent context, active sessions",
                "warm": "PostgreSQL - conversation history, searchable memory",
                "cold": "S3/GCS - archived conversations, embeddings backup"
            }
            implementation["sync_patterns"] = {
                "write_through": "Write to cache and DB simultaneously",
                "write_behind": "Write to cache, async sync to DB",
                "read_through": "Check cache, fallback to DB, populate cache"
            }

        elif backend == "distributed":
            implementation["patterns"] = {
                "sharding": "Consistent hashing by thread_id",
                "replication": "Primary-replica with async replication",
                "consensus": "Raft for leader election",
                "partitioning": "Geographic partitioning by user region"
            }

        self.stats["implementations"] += 1
        self.implementations.append(implementation)

        return implementation

    async def run_iteration(self, topic_data: dict, index: int) -> PersistenceResult:
        topic = topic_data["topic"]
        backend = topic_data["backend"]

        print(f"\n[{index:02d}] [{backend}] {topic[:50]}...")

        start = time.time()
        result = await self.research_and_implement(topic, backend)
        latency = time.time() - start

        self.stats["sources"] += len(result["sources"])
        self.stats["findings"] += len(result["findings"])

        print(f"    Src:{len(result['sources'])} Find:{len(result['findings'])} [{latency:.1f}s]")

        if result["implementation"].get("operations"):
            ops = result["implementation"]["operations"]
            print(f"    -> SQLite: {ops}")
        elif result["implementation"].get("patterns"):
            patterns = list(result["implementation"]["patterns"].keys())
            print(f"    -> Patterns: {patterns}")

        return PersistenceResult(
            topic=topic,
            backend=backend,
            sources=result["sources"],
            findings=result["findings"],
            implementation=result["implementation"],
            latency=latency
        )


async def main():
    print("="*70)
    print("PERSISTENCE ITERATIONS - DATABASE BACKENDS")
    print("="*70)
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Topics: {len(PERSISTENCE_TOPICS)}")
    print("="*70)

    executor = PersistenceExecutor()
    await executor.initialize()

    all_results = []

    # Group by backend
    by_backend = {}
    for t in PERSISTENCE_TOPICS:
        by_backend.setdefault(t["backend"], []).append(t)

    iteration = 0
    for backend, topics in by_backend.items():
        print(f"\n{'='*70}")
        print(f"BACKEND: {backend.upper()}")
        print(f"{'='*70}")

        for topic_data in topics:
            iteration += 1
            try:
                result = await executor.run_iteration(topic_data, iteration)
                all_results.append(result)
            except Exception as e:
                print(f"    [ERR] {str(e)[:50]}")

            await asyncio.sleep(0.3)

    # Summary
    print("\n" + "="*70)
    print("PERSISTENCE ITERATIONS COMPLETE")
    print("="*70)

    print(f"\n  Iterations: {len(all_results)}")
    print(f"  Total Sources: {executor.stats['sources']}")
    print(f"  Total Findings: {executor.stats['findings']}")
    print(f"  Implementations: {executor.stats['implementations']}")

    avg_latency = sum(r.latency for r in all_results) / len(all_results) if all_results else 0
    print(f"  Avg Latency: {avg_latency:.1f}s")

    # SQLite stats
    cursor = executor.db.cursor()
    cursor.execute("SELECT COUNT(*) FROM conversations")
    conv_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM memory")
    mem_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM checkpoints")
    cp_count = cursor.fetchone()[0]

    print("\n  SQLITE DEMO DATABASE:")
    print(f"    Conversations: {conv_count}")
    print(f"    Memory entries: {mem_count}")
    print(f"    Checkpoints: {cp_count}")

    print("\n  BY BACKEND:")
    backend_stats = {}
    for r in all_results:
        backend_stats[r.backend] = backend_stats.get(r.backend, 0) + 1
    for backend, count in sorted(backend_stats.items()):
        print(f"    {backend}: {count}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "stats": executor.stats,
        "sqlite_demo": {"conversations": conv_count, "memory": mem_count, "checkpoints": cp_count},
        "implementations": executor.implementations,
        "results": [
            {
                "topic": r.topic,
                "backend": r.backend,
                "sources": len(r.sources),
                "findings": r.findings[:2],
                "implementation": r.implementation,
                "latency": r.latency
            }
            for r in all_results
        ]
    }

    with open("persistence_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Results saved to persistence_results.json")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
