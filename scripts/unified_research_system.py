"""
UNIFIED RESEARCH SYSTEM - Complete Production Implementation
=============================================================
Combines all SDKs, self-improvement, and multi-agent coordination.
"""

import asyncio
import os
import json
import time
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum
from dotenv import load_dotenv
import httpx

load_dotenv('.config/.env')

# ============================================================================
# TYPES AND ENUMS
# ============================================================================

class QueryType(Enum):
    FACTUAL = "factual"
    CODE = "code"
    COMPARISON = "comparison"
    RESEARCH = "research"
    CREATIVE = "creative"
    DEBUG = "debug"


class ResearchDepth(Enum):
    QUICK = "quick"          # 1-2 sources, fast
    STANDARD = "standard"    # 3-5 sources
    DEEP = "deep"           # 5-10 sources, all SDKs
    COMPREHENSIVE = "comprehensive"  # 10+ sources, full analysis


@dataclass
class ResearchResult:
    query: str
    query_type: QueryType
    depth: ResearchDepth
    sources: list
    findings: list
    insights: list
    vectors_stored: int
    confidence: float
    latency_s: float
    sdks_used: list
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentMemory:
    """Persistent agent memory."""
    facts: list = field(default_factory=list)
    patterns: dict = field(default_factory=dict)
    query_history: list = field(default_factory=list)
    performance: dict = field(default_factory=dict)


# ============================================================================
# SEMANTIC CACHE
# ============================================================================

class SemanticCache:
    """Cache with semantic similarity matching."""

    def __init__(self, jina_key: str, threshold: float = 0.85):
        self.jina_key = jina_key
        self.threshold = threshold
        self.cache = {}  # query_hash -> (embedding, result)
        self.embeddings = []
        self.qdrant = None

    async def init(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        self.qdrant = QdrantClient(":memory:")
        self.qdrant.create_collection(
            "cache",
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

    async def get(self, query: str) -> Optional[ResearchResult]:
        """Check cache for similar query."""
        if not self.qdrant:
            return None

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    'https://api.jina.ai/v1/embeddings',
                    headers={'Authorization': f'Bearer {self.jina_key}', 'Content-Type': 'application/json'},
                    json={'model': 'jina-embeddings-v3', 'input': [query]}
                )
                emb = r.json().get('data', [{}])[0].get('embedding', [])

            if not emb:
                return None

            results = self.qdrant.query_points("cache", query=emb, limit=1, score_threshold=self.threshold).points
            if results:
                cached = self.cache.get(results[0].payload.get("hash"))
                if cached:
                    print(f"    [CACHE HIT] similarity={results[0].score:.3f}")
                    return cached
        except Exception:
            pass
        return None

    async def set(self, query: str, result: ResearchResult):
        """Store result in cache."""
        if not self.qdrant:
            return

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    'https://api.jina.ai/v1/embeddings',
                    headers={'Authorization': f'Bearer {self.jina_key}', 'Content-Type': 'application/json'},
                    json={'model': 'jina-embeddings-v3', 'input': [query]}
                )
                emb = r.json().get('data', [{}])[0].get('embedding', [])

            if emb:
                from qdrant_client.models import PointStruct
                query_hash = hashlib.md5(query.encode()).hexdigest()
                self.cache[query_hash] = result
                point_id = len(self.cache)
                self.qdrant.upsert("cache", [PointStruct(id=point_id, vector=emb, payload={"hash": query_hash})])
        except Exception:
            pass


# ============================================================================
# QUERY ROUTER
# ============================================================================

class QueryRouter:
    """Routes queries to optimal SDKs based on type."""

    ROUTING_TABLE = {
        QueryType.FACTUAL: {
            "primary": ["tavily", "perplexity"],
            "depth": ResearchDepth.QUICK,
        },
        QueryType.CODE: {
            "primary": ["exa", "tavily"],
            "depth": ResearchDepth.STANDARD,
        },
        QueryType.COMPARISON: {
            "primary": ["perplexity", "tavily", "exa"],
            "depth": ResearchDepth.DEEP,
        },
        QueryType.RESEARCH: {
            "primary": ["exa", "tavily", "perplexity", "jina"],
            "depth": ResearchDepth.COMPREHENSIVE,
        },
        QueryType.DEBUG: {
            "primary": ["exa", "tavily"],
            "depth": ResearchDepth.STANDARD,
        },
        QueryType.CREATIVE: {
            "primary": ["perplexity"],
            "depth": ResearchDepth.QUICK,
        },
    }

    @classmethod
    def classify(cls, query: str) -> QueryType:
        """Classify query intent."""
        q = query.lower()

        if any(k in q for k in ["compare", "vs", "versus", "difference", "better"]):
            return QueryType.COMPARISON
        if any(k in q for k in ["write", "code", "function", "implement", "script"]):
            return QueryType.CODE
        if any(k in q for k in ["fix", "error", "bug", "debug", "issue"]):
            return QueryType.DEBUG
        if any(k in q for k in ["what is", "who is", "when", "where", "define"]):
            return QueryType.FACTUAL
        if any(k in q for k in ["research", "deep dive", "analyze", "architecture", "pattern"]):
            return QueryType.RESEARCH
        if any(k in q for k in ["create", "generate", "write a story", "imagine"]):
            return QueryType.CREATIVE

        return QueryType.RESEARCH  # Default

    @classmethod
    def get_route(cls, query_type: QueryType) -> dict:
        """Get routing config for query type."""
        return cls.ROUTING_TABLE.get(query_type, cls.ROUTING_TABLE[QueryType.RESEARCH])


# ============================================================================
# SDK ADAPTERS
# ============================================================================

class SDKAdapter:
    """Base SDK adapter."""

    def __init__(self, keys: dict):
        self.keys = keys

    async def search(self, query: str, **kwargs) -> dict:
        raise NotImplementedError


class ExaAdapter(SDKAdapter):
    async def search(self, query: str, num_results: int = 5, **kwargs) -> dict:
        try:
            from exa_py import Exa
            exa = Exa(self.keys.get("exa"))
            results = exa.search_and_contents(query, type="auto", num_results=num_results, text=True, highlights=True)
            sources = [{"title": r.title, "url": r.url, "text": r.text[:500] if r.text else "", "sdk": "exa"} for r in results.results]
            findings = [f"[exa] {r.title}" for r in results.results[:3]]
            return {"sources": sources, "findings": findings}
        except Exception as e:
            return {"sources": [], "findings": [f"[exa] Error: {str(e)[:40]}"]}


class TavilyAdapter(SDKAdapter):
    async def search(self, query: str, depth: str = "advanced", **kwargs) -> dict:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post('https://api.tavily.com/search', json={
                    'api_key': self.keys.get("tavily"),
                    'query': query,
                    'search_depth': depth,
                    'include_answer': True,
                    'max_results': 5
                })
                data = r.json()
                sources = [{"title": s.get("title", ""), "url": s.get("url", ""), "text": s.get("content", "")[:500], "sdk": "tavily"} for s in data.get("results", [])]
                findings = [f"[tavily] {data.get('answer', '')[:200]}"] if data.get("answer") else []
                return {"sources": sources, "findings": findings}
        except Exception as e:
            return {"sources": [], "findings": []}


class PerplexityAdapter(SDKAdapter):
    async def search(self, query: str, model: str = "sonar-pro", **kwargs) -> dict:
        try:
            async with httpx.AsyncClient(timeout=90) as client:
                r = await client.post(
                    'https://api.perplexity.ai/chat/completions',
                    headers={'Authorization': f'Bearer {self.keys.get("perplexity")}', 'Content-Type': 'application/json'},
                    json={
                        'model': model,
                        'messages': [{'role': 'user', 'content': f"Provide technical analysis: {query}"}],
                        'return_citations': True
                    }
                )
                data = r.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                citations = data.get('citations', [])
                sources = [{"title": "Citation", "url": c, "text": "", "sdk": "perplexity"} for c in citations[:5]]
                findings = [f"[perplexity] {content[:250]}..."] if content else []
                return {"sources": sources, "findings": findings}
        except Exception as e:
            return {"sources": [], "findings": []}


class JinaAdapter(SDKAdapter):
    async def embed(self, texts: list) -> list:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    'https://api.jina.ai/v1/embeddings',
                    headers={'Authorization': f'Bearer {self.keys.get("jina")}', 'Content-Type': 'application/json'},
                    json={'model': 'jina-embeddings-v3', 'input': texts[:15]}
                )
                return [d.get('embedding', []) for d in r.json().get('data', [])]
        except Exception:
            return []

    async def rerank(self, query: str, documents: list) -> list:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    'https://api.jina.ai/v1/rerank',
                    headers={'Authorization': f'Bearer {self.keys.get("jina")}', 'Content-Type': 'application/json'},
                    json={'model': 'jina-reranker-v2-base-multilingual', 'query': query, 'documents': documents[:10], 'top_n': 5}
                )
                results = r.json().get('results', [])
                return [documents[r.get('index', 0)] for r in results]
        except Exception:
            return documents[:5]


# ============================================================================
# SELF-IMPROVING AGENT
# ============================================================================

class SelfImprovingAgent:
    """Agent that learns from research patterns."""

    def __init__(self):
        self.memory = AgentMemory()
        self.performance_history = []

    def learn_from_result(self, result: ResearchResult):
        """Extract learnings from research result."""
        # Track query patterns
        pattern_key = f"{result.query_type.value}_{result.depth.value}"
        if pattern_key not in self.memory.patterns:
            self.memory.patterns[pattern_key] = {"count": 0, "avg_sources": 0, "avg_latency": 0}

        pattern = self.memory.patterns[pattern_key]
        pattern["count"] += 1
        pattern["avg_sources"] = (pattern["avg_sources"] * (pattern["count"] - 1) + len(result.sources)) / pattern["count"]
        pattern["avg_latency"] = (pattern["avg_latency"] * (pattern["count"] - 1) + result.latency_s) / pattern["count"]

        # Track SDK performance
        for sdk in result.sdks_used:
            if sdk not in self.memory.performance:
                self.memory.performance[sdk] = {"calls": 0, "avg_sources": 0}
            perf = self.memory.performance[sdk]
            perf["calls"] += 1

        # Store high-confidence findings
        if result.confidence > 0.9:
            for finding in result.findings[:2]:
                self.memory.facts.append({"finding": finding, "confidence": result.confidence, "timestamp": result.timestamp})

    def get_recommendations(self, query_type: QueryType) -> dict:
        """Get optimized recommendations based on learning."""
        pattern_key = f"{query_type.value}_"
        relevant_patterns = {k: v for k, v in self.memory.patterns.items() if k.startswith(pattern_key)}

        # Find best performing depth
        best_depth = ResearchDepth.STANDARD
        best_score = 0
        for k, v in relevant_patterns.items():
            score = v["avg_sources"] / max(v["avg_latency"], 1)  # Sources per second
            if score > best_score:
                best_score = score
                best_depth = ResearchDepth(k.split("_")[1])

        # Find best SDKs
        sorted_sdks = sorted(self.memory.performance.items(), key=lambda x: x[1]["calls"], reverse=True)
        recommended_sdks = [sdk for sdk, _ in sorted_sdks[:3]]

        return {"depth": best_depth, "sdks": recommended_sdks}

    def reflect(self) -> list:
        """Reflect on performance and generate insights."""
        insights = []

        # Analyze patterns
        if self.memory.patterns:
            most_common = max(self.memory.patterns.items(), key=lambda x: x[1]["count"])
            insights.append(f"Most common pattern: {most_common[0]} ({most_common[1]['count']} queries)")

        # Analyze SDK performance
        if self.memory.performance:
            best_sdk = max(self.memory.performance.items(), key=lambda x: x[1]["calls"])
            insights.append(f"Most used SDK: {best_sdk[0]} ({best_sdk[1]['calls']} calls)")

        # Analyze facts learned
        if self.memory.facts:
            insights.append(f"Facts learned: {len(self.memory.facts)}")

        return insights


# ============================================================================
# UNIFIED RESEARCH ORCHESTRATOR
# ============================================================================

class UnifiedResearchOrchestrator:
    """Main orchestrator combining all components."""

    def __init__(self):
        self.keys = {
            "exa": os.getenv("EXA_API_KEY"),
            "tavily": os.getenv("TAVILY_API_KEY"),
            "jina": os.getenv("JINA_API_KEY"),
            "perplexity": os.getenv("PERPLEXITY_API_KEY"),
        }

        # Adapters
        self.adapters = {
            "exa": ExaAdapter(self.keys),
            "tavily": TavilyAdapter(self.keys),
            "perplexity": PerplexityAdapter(self.keys),
        }
        self.jina = JinaAdapter(self.keys)

        # Components
        self.cache = SemanticCache(self.keys["jina"])
        self.agent = SelfImprovingAgent()
        self.router = QueryRouter()

        # Vector store
        self.qdrant = None

    async def initialize(self):
        """Initialize all components."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        await self.cache.init()
        self.qdrant = QdrantClient(":memory:")
        self.qdrant.create_collection("knowledge", vectors_config=VectorParams(size=1024, distance=Distance.COSINE))
        print("[OK] Unified Research Orchestrator initialized")

    async def research(self, query: str, force_depth: Optional[ResearchDepth] = None) -> ResearchResult:
        """Execute research with full pipeline."""
        start = time.time()

        # Check cache first
        cached = await self.cache.get(query)
        if cached:
            return cached

        # Classify and route
        query_type = self.router.classify(query)
        route = self.router.get_route(query_type)

        # Get agent recommendations
        recommendations = self.agent.get_recommendations(query_type)
        depth = force_depth or recommendations.get("depth", route["depth"])
        sdks_to_use = route["primary"]

        print(f"    Query Type: {query_type.value}")
        print(f"    Depth: {depth.value}")
        print(f"    SDKs: {sdks_to_use}")

        # Execute parallel research
        all_sources = []
        all_findings = []
        sdks_used = []

        async def run_adapter(sdk_name: str):
            if sdk_name in self.adapters:
                result = await self.adapters[sdk_name].search(query)
                return sdk_name, result
            return sdk_name, {"sources": [], "findings": []}

        tasks = [run_adapter(sdk) for sdk in sdks_to_use]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, tuple):
                sdk_name, data = r
                if data.get("sources"):
                    all_sources.extend(data["sources"])
                    sdks_used.append(sdk_name)
                all_findings.extend(data.get("findings", []))

        # Embed and store
        vectors_stored = 0
        texts = [s.get("text", "") for s in all_sources if s.get("text")]
        if texts:
            embeddings = await self.jina.embed(texts)
            if embeddings and embeddings[0]:
                from qdrant_client.models import PointStruct
                base_id = int(time.time() * 1000) % 1000000  # Unique timestamp-based ID
                points = [PointStruct(id=base_id + i, vector=emb, payload={"text": texts[i][:300]}) for i, emb in enumerate(embeddings)]
                self.qdrant.upsert("knowledge", points=points)
                vectors_stored = len(points)

        # Rerank findings
        if len(all_findings) > 3:
            finding_texts = [f[:200] for f in all_findings]
            reranked = await self.jina.rerank(query, finding_texts)
            all_findings = reranked[:5]

        # Calculate confidence
        confidence = min(0.5 + len(all_sources) * 0.05 + len(sdks_used) * 0.1, 0.99)

        # Build result
        result = ResearchResult(
            query=query,
            query_type=query_type,
            depth=depth,
            sources=all_sources,
            findings=all_findings[:10],
            insights=[f[:150] for f in all_findings[:5]],
            vectors_stored=vectors_stored,
            confidence=confidence,
            latency_s=time.time() - start,
            sdks_used=sdks_used,
        )

        # Learn from result
        self.agent.learn_from_result(result)

        # Cache result
        await self.cache.set(query, result)

        return result

    def get_agent_insights(self) -> list:
        """Get agent's learned insights."""
        return self.agent.reflect()


# ============================================================================
# MULTI-AGENT SWARM
# ============================================================================

class ResearchSwarm:
    """Multi-agent research coordination."""

    def __init__(self, orchestrator: UnifiedResearchOrchestrator):
        self.orchestrator = orchestrator
        self.agents = []

    async def spawn_agents(self, count: int = 3):
        """Spawn research agents."""
        self.agents = [SelfImprovingAgent() for _ in range(count)]
        print(f"[OK] Spawned {count} research agents")

    async def parallel_research(self, queries: list) -> list:
        """Execute parallel research across agents."""
        tasks = [self.orchestrator.research(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = []
        for r in results:
            if isinstance(r, ResearchResult):
                valid_results.append(r)

        return valid_results

    async def synthesize(self, results: list) -> dict:
        """Synthesize findings from multiple research results."""
        all_sources = []
        all_findings = []
        total_vectors = 0

        for r in results:
            all_sources.extend(r.sources)
            all_findings.extend(r.findings)
            total_vectors += r.vectors_stored

        # Deduplicate
        seen = set()
        unique_findings = []
        for f in all_findings:
            key = f[:50].lower()
            if key not in seen:
                seen.add(key)
                unique_findings.append(f)

        return {
            "total_queries": len(results),
            "total_sources": len(all_sources),
            "unique_findings": len(unique_findings),
            "total_vectors": total_vectors,
            "top_findings": unique_findings[:10],
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    print("="*70)
    print("UNIFIED RESEARCH SYSTEM - FULL IMPLEMENTATION")
    print("="*70)
    print(f"Start: {datetime.now().isoformat()}")
    print("="*70)

    # Initialize orchestrator
    orchestrator = UnifiedResearchOrchestrator()
    await orchestrator.initialize()

    # Initialize swarm
    swarm = ResearchSwarm(orchestrator)
    await swarm.spawn_agents(3)

    # Test queries across all types
    test_queries = [
        # Factual
        "What is the context window size for Claude 3.5 Sonnet?",

        # Comparison
        "Compare LangGraph vs CrewAI vs AutoGen for multi-agent systems",

        # Code
        "Best practices for implementing RAG with LangChain",

        # Research
        "Deep dive into transformer attention mechanisms and optimization",

        # Debug
        "How to fix CUDA out of memory errors in PyTorch",
    ]

    print("\n" + "="*70)
    print("PHASE 1: SEQUENTIAL RESEARCH (Learning)")
    print("="*70)

    all_results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] {query[:50]}...")
        result = await orchestrator.research(query)
        all_results.append(result)
        print(f"    Sources: {len(result.sources)}, Vectors: {result.vectors_stored}, Confidence: {result.confidence:.0%}")
        print(f"    SDKs: {result.sdks_used}, Latency: {result.latency_s:.1f}s")
        if result.findings:
            f = result.findings[0].encode('ascii', 'replace').decode('ascii')
            print(f"    Finding: {f[:60]}...")

    print("\n" + "="*70)
    print("PHASE 2: AGENT REFLECTION")
    print("="*70)

    insights = orchestrator.get_agent_insights()
    print("\n  Agent Learnings:")
    for insight in insights:
        print(f"    - {insight}")

    print("\n" + "="*70)
    print("PHASE 3: PARALLEL SWARM RESEARCH")
    print("="*70)

    swarm_queries = [
        "State-of-the-art embedding models for RAG 2026",
        "Production vector database deployment patterns",
        "LLM inference optimization techniques",
    ]

    print(f"\n  Executing {len(swarm_queries)} queries in parallel...")
    swarm_results = await swarm.parallel_research(swarm_queries)
    synthesis = await swarm.synthesize(swarm_results)

    print(f"\n  Swarm Results:")
    print(f"    Queries: {synthesis['total_queries']}")
    print(f"    Sources: {synthesis['total_sources']}")
    print(f"    Findings: {synthesis['unique_findings']}")
    print(f"    Vectors: {synthesis['total_vectors']}")

    print("\n  Top Findings:")
    for f in synthesis['top_findings'][:5]:
        clean = f.encode('ascii', 'replace').decode('ascii')
        print(f"    - {clean[:60]}...")

    # Summary
    print("\n" + "="*70)
    print("UNIFIED SYSTEM SUMMARY")
    print("="*70)

    total_sources = sum(len(r.sources) for r in all_results) + synthesis['total_sources']
    total_vectors = sum(r.vectors_stored for r in all_results) + synthesis['total_vectors']
    avg_confidence = sum(r.confidence for r in all_results) / len(all_results)
    avg_latency = sum(r.latency_s for r in all_results) / len(all_results)

    print(f"\n  Total Queries: {len(test_queries) + len(swarm_queries)}")
    print(f"  Total Sources: {total_sources}")
    print(f"  Total Vectors: {total_vectors}")
    print(f"  Avg Confidence: {avg_confidence:.0%}")
    print(f"  Avg Latency: {avg_latency:.1f}s")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "queries": len(test_queries) + len(swarm_queries),
            "sources": total_sources,
            "vectors": total_vectors,
            "confidence": avg_confidence,
            "latency": avg_latency,
        },
        "agent_insights": insights,
        "swarm_synthesis": synthesis,
    }

    with open("unified_system_results.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n[OK] Results saved to unified_system_results.json")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
