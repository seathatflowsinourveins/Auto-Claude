"""
LLMOps & Emerging Patterns - Production ML Systems
===================================================
Battle-tested SDK execution for LLMOps and emerging AI patterns.
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
# LLMOPS & EMERGING PATTERNS
# ============================================================================

LLMOPS_TOPICS = [
    # LLMOps Core
    {"topic": "LLM deployment patterns: serverless vs container vs edge inference", "category": "Deployment"},
    {"topic": "Model versioning and A/B testing for LLM applications", "category": "Versioning"},
    {"topic": "LLM monitoring: latency, token usage, cost tracking in production", "category": "Monitoring"},
    {"topic": "Prompt management: versioning, testing, and deployment workflows", "category": "Prompts"},

    # Evaluation
    {"topic": "LLM evaluation frameworks: RAGAS, DeepEval, promptfoo comparison", "category": "Evaluation"},
    {"topic": "Human-in-the-loop evaluation: RLHF, DPO, and preference learning", "category": "HITL"},
    {"topic": "Automated red teaming and adversarial testing for LLMs", "category": "Red Team"},
    {"topic": "Hallucination detection and factual grounding verification", "category": "Hallucination"},

    # Optimization
    {"topic": "Inference optimization: vLLM, TensorRT-LLM, Triton comparison", "category": "Inference"},
    {"topic": "Quantization strategies: GPTQ, AWQ, GGML for production deployment", "category": "Quantization"},
    {"topic": "KV cache optimization: PagedAttention, vLLM memory management", "category": "KV Cache"},
    {"topic": "Batching strategies: continuous batching, dynamic batching patterns", "category": "Batching"},

    # Emerging
    {"topic": "Mixture of Experts: Mixtral, DeepSeek, GPT-4 sparse architectures", "category": "MoE"},
    {"topic": "State Space Models: Mamba, RWKV vs Transformer comparison", "category": "SSM"},
    {"topic": "Long context models: 200K+ context handling and retrieval tradeoffs", "category": "Long Context"},
    {"topic": "Multi-modal LLMs: GPT-4V, Claude vision, Gemini production patterns", "category": "Multi-Modal"},

    # Orchestration
    {"topic": "LLM gateways: LiteLLM, Portkey, RouteLLM for multi-provider routing", "category": "Gateway"},
    {"topic": "Semantic caching: GPTCache, Redis semantic cache patterns", "category": "Caching"},
    {"topic": "Rate limiting and cost control for LLM API usage", "category": "Cost Control"},
    {"topic": "Observability stack: LangSmith, Langfuse, Helicone, Arize comparison", "category": "Observability"},
]


class LLMOpsResearcher:
    """Execute LLMOps research with production SDKs."""

    def __init__(self):
        self.exa = None
        self.keys = {
            "tavily": os.getenv("TAVILY_API_KEY"),
            "jina": os.getenv("JINA_API_KEY"),
            "perplexity": os.getenv("PERPLEXITY_API_KEY"),
        }

    async def init(self):
        from exa_py import Exa
        self.exa = Exa(os.getenv("EXA_API_KEY"))
        print("[OK] LLMOps Researcher initialized")

    async def research(self, topic: str) -> dict:
        """Multi-source research with parallel execution."""
        async with httpx.AsyncClient(timeout=90) as client:
            tasks = [
                self._exa(topic),
                self._tavily(client, topic),
                self._perplexity(client, topic),
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        combined = {"sources": [], "findings": [], "insights": []}
        for r in results:
            if isinstance(r, dict):
                combined["sources"].extend(r.get("sources", []))
                combined["findings"].extend(r.get("findings", []))
                combined["insights"].extend(r.get("insights", []))

        return combined

    async def _exa(self, topic: str) -> dict:
        try:
            search = self.exa.search_and_contents(topic, type="auto", num_results=5, text=True, highlights=True)
            sources = [{"title": r.title, "url": r.url, "text": r.text[:400] if r.text else ""} for r in search.results]
            findings = [f"[exa] {r.title}" for r in search.results[:3]]
            insights = []
            for r in search.results:
                if hasattr(r, 'highlights') and r.highlights:
                    insights.extend([f"[exa] {h[:120]}" for h in r.highlights[:2]])
            return {"sources": sources, "findings": findings, "insights": insights}
        except Exception as e:
            return {"sources": [], "findings": [f"[exa] Error: {str(e)[:50]}"], "insights": []}

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
            findings = [f"[tavily] {data.get('answer', 'No answer')[:200]}"] if data.get("answer") else []
            insights = [f"[tavily] {s.strip()}" for s in data.get("answer", "").split(". ")[:3] if len(s) > 30]
            return {"sources": sources, "findings": findings, "insights": insights}
        except Exception as e:
            return {"sources": [], "findings": [], "insights": []}

    async def _perplexity(self, client: httpx.AsyncClient, topic: str) -> dict:
        try:
            r = await client.post(
                'https://api.perplexity.ai/chat/completions',
                headers={'Authorization': f'Bearer {self.keys["perplexity"]}', 'Content-Type': 'application/json'},
                json={
                    'model': 'sonar-pro',
                    'messages': [{'role': 'user', 'content': f"Provide production-ready technical analysis: {topic}. Include implementation patterns and best practices."}],
                    'return_citations': True
                }
            )
            data = r.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            citations = data.get('citations', [])
            sources = [{"title": "Citation", "url": c, "text": ""} for c in citations[:5]]
            findings = [f"[perplexity] {content[:250]}..."]
            insights = []
            for line in content.split('\n'):
                line = line.strip()
                if line and len(line) > 30 and (line[0].isdigit() or line.startswith('-')):
                    insights.append(f"[perplexity] {line[:100]}")
            return {"sources": sources, "findings": findings, "insights": insights[:5]}
        except Exception as e:
            return {"sources": [], "findings": [], "insights": []}

    async def embed_store(self, texts: list[str]) -> dict:
        """Embed with Jina, store in Qdrant."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct

        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                'https://api.jina.ai/v1/embeddings',
                headers={'Authorization': f'Bearer {self.keys["jina"]}', 'Content-Type': 'application/json'},
                json={'model': 'jina-embeddings-v3', 'input': texts[:15]}
            )
            data = r.json()
            embeddings = [d.get('embedding', []) for d in data.get('data', [])]

        if not embeddings or not embeddings[0]:
            return {"stored": 0}

        qdrant = QdrantClient(":memory:")
        dim = len(embeddings[0])
        qdrant.create_collection("llmops", vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        points = [PointStruct(id=i, vector=emb, payload={"text": texts[i][:300]}) for i, emb in enumerate(embeddings)]
        qdrant.upsert("llmops", points=points)

        return {"stored": len(points), "dims": dim}


async def run_topic(researcher: LLMOpsResearcher, topic: dict, i: int) -> dict:
    """Execute single topic research."""
    print(f"\n{'='*70}")
    print(f"[{i}] {topic['category']}: {topic['topic'][:50]}...")
    print("="*70)

    result = {"iteration": i, "category": topic["category"], "topic": topic["topic"]}
    start = time.time()

    # Research
    print("  [RESEARCH]...")
    research = await researcher.research(topic["topic"])
    result["sources"] = len(research["sources"])
    result["findings"] = len(research["findings"])
    result["insights_raw"] = research["insights"][:5]
    print(f"      Sources: {result['sources']}, Findings: {result['findings']}")

    # Embed
    print("  [EMBED+STORE]...")
    texts = [s.get("text", "") for s in research["sources"] if s.get("text")]
    if texts:
        store = await researcher.embed_store(texts)
        result["stored"] = store.get("stored", 0)
        print(f"      Stored: {result['stored']} vectors")

    # Key insights
    if research["insights"]:
        print("  [INSIGHTS]:")
        for insight in research["insights"][:2]:
            clean = insight.encode('ascii', 'replace').decode('ascii')
            print(f"      {clean[:60]}...")

    result["latency"] = time.time() - start
    print(f"  [DONE] {result['latency']:.1f}s")
    return result


async def main():
    print("="*70)
    print("LLMOPS & EMERGING PATTERNS - PRODUCTION SDK EXECUTION")
    print("="*70)
    print(f"Topics: {len(LLMOPS_TOPICS)}")
    print(f"Start: {datetime.now().isoformat()}")
    print("="*70)

    researcher = LLMOpsResearcher()
    await researcher.init()

    results = []
    totals = {"sources": 0, "stored": 0, "insights": 0}

    for i, topic in enumerate(LLMOPS_TOPICS, 1):
        try:
            r = await run_topic(researcher, topic, i)
            results.append(r)
            totals["sources"] += r.get("sources", 0)
            totals["stored"] += r.get("stored", 0)
            totals["insights"] += len(r.get("insights_raw", []))
        except Exception as e:
            print(f"  [ERROR] {str(e)[:60]}")
            results.append({"iteration": i, "error": str(e)[:100]})

        if i < len(LLMOPS_TOPICS):
            await asyncio.sleep(0.3)

    # Summary
    print("\n" + "="*70)
    print("LLMOPS & EMERGING PATTERNS COMPLETE")
    print("="*70)
    successful = [r for r in results if "error" not in r]
    print(f"\n  Iterations: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Total Sources: {totals['sources']}")
    print(f"  Total Stored: {totals['stored']}")
    print(f"  Total Insights: {totals['insights']}")

    if successful:
        avg = sum(r.get("latency", 0) for r in successful) / len(successful)
        print(f"  Avg Latency: {avg:.1f}s")

    # Categories
    print("\n  Categories:")
    cats = {}
    for r in successful:
        c = r.get("category", "Unknown")
        cats[c] = cats.get(c, 0) + 1
    for c, n in sorted(cats.items()):
        print(f"      {c}: {n}")

    # Save
    with open("llmops_results.json", 'w') as f:
        json.dump({"timestamp": datetime.now().isoformat(), "summary": totals, "results": results}, f, indent=2)
    print("\n[OK] Results saved to llmops_results.json")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
