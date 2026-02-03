"""
FULL ITERATION SUITE - Comprehensive SDK Execution with Gap Fixes
==================================================================
Combines all research, execution, and gap resolution in iteration loops.
"""

import asyncio
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import httpx

load_dotenv('.config/.env')

# ============================================================================
# FULL ITERATION TOPICS
# ============================================================================

ITERATION_TOPICS = [
    # Advanced RAG Patterns
    {"topic": "Adaptive RAG: dynamically adjusting retrieval based on query complexity", "cat": "RAG"},
    {"topic": "RAG fusion: combining multiple retrieval strategies with RRF", "cat": "RAG"},
    {"topic": "Contextual compression: reducing context size while preserving relevance", "cat": "RAG"},

    # Agent Architecture
    {"topic": "Plan-and-execute agents: separating planning from execution", "cat": "Agent"},
    {"topic": "Tool-use optimization: selecting optimal tools for agent tasks", "cat": "Agent"},
    {"topic": "Agent handoffs: transferring context between specialized agents", "cat": "Agent"},

    # Memory & State
    {"topic": "Hierarchical memory: combining short-term and long-term memory", "cat": "Memory"},
    {"topic": "Memory consolidation: summarizing and compacting agent memory", "cat": "Memory"},
    {"topic": "Episodic memory retrieval: recalling relevant past experiences", "cat": "Memory"},

    # Production Optimization
    {"topic": "Prompt compression: reducing token count while preserving meaning", "cat": "Production"},
    {"topic": "Speculative decoding: accelerating inference with draft models", "cat": "Production"},
    {"topic": "Continuous batching: optimizing throughput for concurrent requests", "cat": "Production"},

    # Emerging Techniques
    {"topic": "Constitutional AI: building value-aligned language models", "cat": "Emerging"},
    {"topic": "Chain-of-verification: self-checking for reduced hallucination", "cat": "Emerging"},
    {"topic": "Retrieval-augmented thought: combining reasoning with retrieval", "cat": "Emerging"},
]


class FullIterationExecutor:
    """Execute comprehensive iterations with all SDKs."""

    def __init__(self):
        self.exa = None
        self.keys = {
            "tavily": os.getenv("TAVILY_API_KEY"),
            "jina": os.getenv("JINA_API_KEY"),
            "perplexity": os.getenv("PERPLEXITY_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
        }
        self.stats = {"sources": 0, "vectors": 0, "insights": 0}

    async def init(self):
        from exa_py import Exa
        self.exa = Exa(os.getenv("EXA_API_KEY"))
        print("[OK] Full Iteration Executor initialized")

    async def research_and_execute(self, topic: str) -> dict:
        """Research topic and execute with all SDKs."""
        result = {"sources": [], "findings": [], "vectors": 0}

        async with httpx.AsyncClient(timeout=90) as client:
            # Parallel research
            tasks = [
                self._exa_research(topic),
                self._tavily_research(client, topic),
                self._perplexity_research(client, topic),
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for r in responses:
                if isinstance(r, dict):
                    result["sources"].extend(r.get("sources", []))
                    result["findings"].extend(r.get("findings", []))

            # Embed and store
            texts = [s.get("text", "")[:500] for s in result["sources"] if s.get("text")]
            if texts:
                embed_result = await self._embed_and_store(client, texts)
                result["vectors"] = embed_result.get("stored", 0)

        return result

    async def _exa_research(self, topic: str) -> dict:
        try:
            search = self.exa.search_and_contents(topic, type="auto", num_results=5, text=True)
            sources = [{"title": r.title, "url": r.url, "text": r.text[:400] if r.text else "", "src": "exa"} for r in search.results]
            findings = [f"[exa] {r.title}" for r in search.results[:2]]
            return {"sources": sources, "findings": findings}
        except Exception as e:
            return {"sources": [], "findings": [f"[exa] Error: {str(e)[:40]}"]}

    async def _tavily_research(self, client: httpx.AsyncClient, topic: str) -> dict:
        try:
            r = await client.post('https://api.tavily.com/search', json={
                'api_key': self.keys["tavily"],
                'query': topic,
                'search_depth': 'advanced',
                'include_answer': True,
                'max_results': 5
            })
            data = r.json()
            sources = [{"title": s.get("title", ""), "url": s.get("url", ""), "text": s.get("content", "")[:400], "src": "tavily"} for s in data.get("results", [])]
            findings = [f"[tavily] {data.get('answer', '')[:150]}"] if data.get("answer") else []
            return {"sources": sources, "findings": findings}
        except Exception as e:
            return {"sources": [], "findings": []}

    async def _perplexity_research(self, client: httpx.AsyncClient, topic: str) -> dict:
        try:
            r = await client.post(
                'https://api.perplexity.ai/chat/completions',
                headers={'Authorization': f'Bearer {self.keys["perplexity"]}', 'Content-Type': 'application/json'},
                json={
                    'model': 'sonar-pro',
                    'messages': [{'role': 'user', 'content': f"Technical deep-dive: {topic}. Provide implementation patterns."}],
                    'return_citations': True
                }
            )
            data = r.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            citations = data.get('citations', [])
            sources = [{"title": "Citation", "url": c, "text": "", "src": "perplexity"} for c in citations[:3]]
            findings = [f"[perplexity] {content[:200]}..."] if content else []
            return {"sources": sources, "findings": findings}
        except Exception as e:
            return {"sources": [], "findings": []}

    async def _embed_and_store(self, client: httpx.AsyncClient, texts: list) -> dict:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct

            # Get embeddings
            r = await client.post(
                'https://api.jina.ai/v1/embeddings',
                headers={'Authorization': f'Bearer {self.keys["jina"]}', 'Content-Type': 'application/json'},
                json={'model': 'jina-embeddings-v3', 'input': texts[:10]}
            )
            embeddings = [d.get('embedding', []) for d in r.json().get('data', [])]

            if not embeddings or not embeddings[0]:
                return {"stored": 0}

            # Store in Qdrant
            qdrant = QdrantClient(":memory:")
            dim = len(embeddings[0])
            qdrant.create_collection("iteration", vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
            points = [PointStruct(id=i, vector=emb, payload={"text": texts[i][:200]}) for i, emb in enumerate(embeddings)]
            qdrant.upsert("iteration", points=points)

            return {"stored": len(points)}
        except Exception as e:
            return {"stored": 0}

    async def run_iteration(self, topic: dict, i: int) -> dict:
        """Run single iteration."""
        print(f"\n[{i:02d}] {topic['cat']}: {topic['topic'][:45]}...")

        start = time.time()
        result = await self.research_and_execute(topic["topic"])

        self.stats["sources"] += len(result["sources"])
        self.stats["vectors"] += result["vectors"]
        self.stats["insights"] += len(result["findings"])

        print(f"     Sources: {len(result['sources'])}, Vectors: {result['vectors']}, Findings: {len(result['findings'])}")

        return {
            "iteration": i,
            "category": topic["cat"],
            "topic": topic["topic"],
            "sources": len(result["sources"]),
            "vectors": result["vectors"],
            "findings": result["findings"][:3],
            "latency": time.time() - start
        }


async def main():
    print("="*70)
    print("FULL ITERATION SUITE - COMPREHENSIVE SDK EXECUTION")
    print("="*70)
    print(f"Topics: {len(ITERATION_TOPICS)}")
    print(f"Start: {datetime.now().isoformat()}")
    print("="*70)

    executor = FullIterationExecutor()
    await executor.init()

    results = []

    for i, topic in enumerate(ITERATION_TOPICS, 1):
        try:
            result = await executor.run_iteration(topic, i)
            results.append(result)
        except Exception as e:
            print(f"     [ERROR] {str(e)[:50]}")
            results.append({"iteration": i, "error": str(e)[:100]})

        if i < len(ITERATION_TOPICS):
            await asyncio.sleep(0.2)

    # Summary
    print("\n" + "="*70)
    print("ITERATION SUITE COMPLETE")
    print("="*70)

    successful = [r for r in results if "error" not in r]
    print(f"\n  Iterations: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Total Sources: {executor.stats['sources']}")
    print(f"  Total Vectors: {executor.stats['vectors']}")
    print(f"  Total Insights: {executor.stats['insights']}")

    if successful:
        avg = sum(r.get("latency", 0) for r in successful) / len(successful)
        print(f"  Avg Latency: {avg:.1f}s")

    # Categories
    print("\n  By Category:")
    cats = {}
    for r in successful:
        c = r.get("category", "Unknown")
        cats[c] = cats.get(c, 0) + 1
    for c, n in sorted(cats.items()):
        print(f"      {c}: {n}")

    # Key findings
    print("\n  Sample Findings:")
    for r in successful[:5]:
        if r.get("findings"):
            finding = r["findings"][0]
            clean = finding.encode('ascii', 'replace').decode('ascii')
            print(f"      {r['category']}: {clean[:55]}...")

    # Save
    with open("full_iteration_results.json", 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": executor.stats,
            "results": results
        }, f, indent=2)

    print(f"\n[OK] Results saved to full_iteration_results.json")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
