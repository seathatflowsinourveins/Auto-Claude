"""
CUTTING EDGE ITERATIONS - Next-Gen AI Patterns
===============================================
Implements and tests the most advanced AI patterns.
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
# CUTTING EDGE TOPICS
# ============================================================================

CUTTING_EDGE_TOPICS = [
    # Self-Improving Systems
    {"topic": "Self-RAG: models that critique and refine their own retrieval", "cat": "Self-Improve"},
    {"topic": "Reflection agents: learning from mistakes and improving", "cat": "Self-Improve"},
    {"topic": "Meta-learning for few-shot adaptation in agents", "cat": "Self-Improve"},

    # Advanced Reasoning
    {"topic": "Tree of Thoughts: exploring multiple reasoning paths", "cat": "Reasoning"},
    {"topic": "Graph of Thoughts: non-linear reasoning with backtracking", "cat": "Reasoning"},
    {"topic": "Chain-of-verification: reducing hallucination through self-check", "cat": "Reasoning"},

    # Multi-Agent Systems
    {"topic": "Society of Mind: emergent intelligence from agent collaboration", "cat": "Multi-Agent"},
    {"topic": "Agent debate: improving answers through adversarial discussion", "cat": "Multi-Agent"},
    {"topic": "Swarm intelligence: collective problem-solving patterns", "cat": "Multi-Agent"},

    # Knowledge Systems
    {"topic": "Neural-symbolic integration: combining LLMs with knowledge graphs", "cat": "Knowledge"},
    {"topic": "Continuous learning: updating knowledge without catastrophic forgetting", "cat": "Knowledge"},
    {"topic": "Knowledge distillation: transferring reasoning to smaller models", "cat": "Knowledge"},

    # Production Innovation
    {"topic": "Inference-time compute scaling: thinking longer for better answers", "cat": "Production"},
    {"topic": "Dynamic routing: selecting models based on query complexity", "cat": "Production"},
    {"topic": "Cascading inference: starting small, escalating when needed", "cat": "Production"},

    # Emerging Frontiers
    {"topic": "World models: agents that simulate before acting", "cat": "Frontier"},
    {"topic": "Embodied AI: language models controlling physical systems", "cat": "Frontier"},
    {"topic": "Neuromorphic computing: brain-inspired AI architectures", "cat": "Frontier"},
]


class CuttingEdgeExecutor:
    """Execute cutting-edge AI pattern research."""

    def __init__(self):
        self.exa = None
        self.keys = {
            "tavily": os.getenv("TAVILY_API_KEY"),
            "jina": os.getenv("JINA_API_KEY"),
            "perplexity": os.getenv("PERPLEXITY_API_KEY"),
        }
        self.totals = {"sources": 0, "vectors": 0, "insights": 0}

    async def init(self):
        from exa_py import Exa
        self.exa = Exa(os.getenv("EXA_API_KEY"))
        print("[OK] Cutting Edge Executor initialized")

    async def deep_research(self, topic: str) -> dict:
        """Deep research with all SDKs."""
        result = {"sources": [], "findings": []}

        async with httpx.AsyncClient(timeout=90) as client:
            # Parallel research
            tasks = [
                self._exa(topic),
                self._tavily(client, topic),
                self._perplexity(client, topic),
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for r in responses:
                if isinstance(r, dict):
                    result["sources"].extend(r.get("sources", []))
                    result["findings"].extend(r.get("findings", []))

            # Embed research
            texts = [s.get("text", "") for s in result["sources"] if s.get("text")]
            if texts:
                result["vectors"] = await self._embed(client, texts)
            else:
                result["vectors"] = 0

        return result

    async def _exa(self, topic: str) -> dict:
        try:
            search = self.exa.search_and_contents(topic, type="auto", num_results=5, text=True, highlights=True)
            sources = [{"title": r.title, "url": r.url, "text": r.text[:400] if r.text else ""} for r in search.results]
            findings = [f"[exa] {r.title}" for r in search.results[:2]]
            # Add highlights as extra findings
            for r in search.results[:2]:
                if hasattr(r, 'highlights') and r.highlights:
                    findings.extend([f"[exa-h] {h[:100]}" for h in r.highlights[:1]])
            return {"sources": sources, "findings": findings}
        except Exception as e:
            return {"sources": [], "findings": []}

    async def _tavily(self, client: httpx.AsyncClient, topic: str) -> dict:
        try:
            r = await client.post('https://api.tavily.com/search', json={
                'api_key': self.keys["tavily"],
                'query': topic,
                'search_depth': 'advanced',
                'include_answer': True,
                'max_results': 5
            })
            data = r.json()
            sources = [{"title": s.get("title", ""), "url": s.get("url", ""), "text": s.get("content", "")[:400]} for s in data.get("results", [])]
            findings = []
            if data.get("answer"):
                findings.append(f"[tavily] {data['answer'][:180]}")
            return {"sources": sources, "findings": findings}
        except Exception as e:
            return {"sources": [], "findings": []}

    async def _perplexity(self, client: httpx.AsyncClient, topic: str) -> dict:
        try:
            r = await client.post(
                'https://api.perplexity.ai/chat/completions',
                headers={'Authorization': f'Bearer {self.keys["perplexity"]}', 'Content-Type': 'application/json'},
                json={
                    'model': 'sonar-pro',
                    'messages': [{'role': 'user', 'content': f"Deep technical analysis of: {topic}. Include architecture, implementation, and real-world applications."}],
                    'return_citations': True
                }
            )
            data = r.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            citations = data.get('citations', [])
            sources = [{"title": "Citation", "url": c, "text": ""} for c in citations[:3]]
            findings = [f"[perplexity] {content[:220]}..."] if content else []
            return {"sources": sources, "findings": findings}
        except Exception as e:
            return {"sources": [], "findings": []}

    async def _embed(self, client: httpx.AsyncClient, texts: list) -> int:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct

            r = await client.post(
                'https://api.jina.ai/v1/embeddings',
                headers={'Authorization': f'Bearer {self.keys["jina"]}', 'Content-Type': 'application/json'},
                json={'model': 'jina-embeddings-v3', 'input': texts[:10]}
            )
            embeddings = [d.get('embedding', []) for d in r.json().get('data', [])]

            if not embeddings or not embeddings[0]:
                return 0

            qdrant = QdrantClient(":memory:")
            dim = len(embeddings[0])
            qdrant.create_collection("cutting_edge", vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
            points = [PointStruct(id=i, vector=emb, payload={"text": texts[i][:200]}) for i, emb in enumerate(embeddings)]
            qdrant.upsert("cutting_edge", points=points)

            return len(points)
        except Exception as e:
            return 0

    async def run_iteration(self, topic: dict, i: int) -> dict:
        """Execute single iteration."""
        print(f"\n[{i:02d}] {topic['cat']}: {topic['topic'][:42]}...")

        start = time.time()
        result = await self.deep_research(topic["topic"])

        self.totals["sources"] += len(result["sources"])
        self.totals["vectors"] += result.get("vectors", 0)
        self.totals["insights"] += len(result["findings"])

        print(f"     Src:{len(result['sources'])} Vec:{result.get('vectors', 0)} Find:{len(result['findings'])}")

        # Show key finding
        if result["findings"]:
            f = result["findings"][0]
            clean = f.encode('ascii', 'replace').decode('ascii')
            print(f"     -> {clean[:55]}...")

        return {
            "i": i,
            "cat": topic["cat"],
            "topic": topic["topic"],
            "sources": len(result["sources"]),
            "vectors": result.get("vectors", 0),
            "findings": result["findings"][:3],
            "latency": time.time() - start
        }


async def main():
    print("="*70)
    print("CUTTING EDGE ITERATIONS - NEXT-GEN AI PATTERNS")
    print("="*70)
    print(f"Topics: {len(CUTTING_EDGE_TOPICS)}")
    print(f"Start: {datetime.now().isoformat()}")
    print("="*70)

    executor = CuttingEdgeExecutor()
    await executor.init()

    results = []

    for i, topic in enumerate(CUTTING_EDGE_TOPICS, 1):
        try:
            result = await executor.run_iteration(topic, i)
            results.append(result)
        except Exception as e:
            print(f"     [ERR] {str(e)[:45]}")
            results.append({"i": i, "error": str(e)[:80]})

        if i < len(CUTTING_EDGE_TOPICS):
            await asyncio.sleep(0.2)

    # Summary
    print("\n" + "="*70)
    print("CUTTING EDGE COMPLETE")
    print("="*70)

    ok = [r for r in results if "error" not in r]
    print(f"\n  Iterations: {len(results)}")
    print(f"  Successful: {len(ok)}")
    print(f"  Sources: {executor.totals['sources']}")
    print(f"  Vectors: {executor.totals['vectors']}")
    print(f"  Insights: {executor.totals['insights']}")

    if ok:
        avg = sum(r.get("latency", 0) for r in ok) / len(ok)
        print(f"  Latency: {avg:.1f}s avg")

    # Categories
    print("\n  Categories:")
    cats = {}
    for r in ok:
        c = r.get("cat", "?")
        cats[c] = cats.get(c, 0) + 1
    for c, n in sorted(cats.items()):
        print(f"      {c}: {n}")

    # Top findings per category
    print("\n  Key Findings by Category:")
    seen_cats = set()
    for r in ok:
        cat = r.get("cat")
        if cat not in seen_cats and r.get("findings"):
            seen_cats.add(cat)
            f = r["findings"][0]
            clean = f.encode('ascii', 'replace').decode('ascii')
            print(f"      [{cat}] {clean[:50]}...")

    # Save
    with open("cutting_edge_results.json", 'w') as f:
        json.dump({
            "ts": datetime.now().isoformat(),
            "totals": executor.totals,
            "results": results
        }, f, indent=2)

    print(f"\n[OK] Saved to cutting_edge_results.json")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
