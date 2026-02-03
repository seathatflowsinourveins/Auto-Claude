"""
EXTENDED TOPICS ITERATIONS - Deep Research on Advanced Areas
==============================================================
Comprehensive coverage of cutting-edge AI/ML patterns
"""

import asyncio
import os
import json
import time
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv
import httpx

load_dotenv('.config/.env')

# ============================================================================
# EXTENDED RESEARCH TOPICS
# ============================================================================

EXTENDED_TOPICS = [
    # Advanced RAG Patterns
    {"topic": "GraphRAG: knowledge graph enhanced retrieval augmented generation", "area": "rag"},
    {"topic": "RAPTOR: recursive abstractive processing for tree-organized retrieval", "area": "rag"},
    {"topic": "HyDE: hypothetical document embeddings for zero-shot retrieval", "area": "rag"},
    {"topic": "ColBERT: late interaction for efficient retrieval", "area": "rag"},
    {"topic": "Multi-vector retrieval: dense, sparse, and late interaction fusion", "area": "rag"},

    # Agent Frameworks Deep Dive
    {"topic": "LangGraph 0.3: command interface, interrupt patterns, human-in-loop", "area": "agents"},
    {"topic": "CrewAI flows: sequential, hierarchical, and consensus processes", "area": "agents"},
    {"topic": "AutoGen 0.4: AssistantAgent, UserProxyAgent, GroupChat patterns", "area": "agents"},
    {"topic": "DSPy 2.6: optimizers, teleprompters, and signature constraints", "area": "agents"},
    {"topic": "Semantic Kernel: planners, plugins, and memory connectors", "area": "agents"},

    # LLM Optimization
    {"topic": "Speculative decoding: draft models for faster inference", "area": "optimization"},
    {"topic": "KV cache optimization: paged attention, prefix caching", "area": "optimization"},
    {"topic": "Quantization: GPTQ, AWQ, GGUF formats comparison", "area": "optimization"},
    {"topic": "Flash Attention 2: memory-efficient attention computation", "area": "optimization"},
    {"topic": "Continuous batching: vLLM, TGI batching strategies", "area": "optimization"},

    # Embeddings & Vector Search
    {"topic": "Matryoshka embeddings: variable dimension representations", "area": "embeddings"},
    {"topic": "Late interaction models: ColBERT, PLAID, ColBERTv2", "area": "embeddings"},
    {"topic": "Binary embeddings: 32x compression with minimal quality loss", "area": "embeddings"},
    {"topic": "Multi-modal embeddings: CLIP, ImageBind, unified representations", "area": "embeddings"},
    {"topic": "Embedding fine-tuning: contrastive learning, hard negatives", "area": "embeddings"},

    # Prompt Engineering
    {"topic": "System prompts: persona, constraints, output formatting", "area": "prompts"},
    {"topic": "Few-shot prompting: example selection, ordering, formatting", "area": "prompts"},
    {"topic": "Chain-of-thought: step-by-step reasoning elicitation", "area": "prompts"},
    {"topic": "ReAct: reasoning and acting interleaved prompting", "area": "prompts"},
    {"topic": "Constitutional AI: self-critique and revision prompts", "area": "prompts"},

    # Evaluation & Testing
    {"topic": "LLM-as-judge: using models to evaluate model outputs", "area": "eval"},
    {"topic": "Semantic similarity metrics: BERTScore, BLEURT, MAUVE", "area": "eval"},
    {"topic": "Factuality evaluation: claim verification, attribution", "area": "eval"},
    {"topic": "Red teaming: adversarial testing for safety", "area": "eval"},
    {"topic": "A/B testing LLMs: statistical significance, guardrails", "area": "eval"},

    # Emerging Patterns
    {"topic": "MCP (Model Context Protocol): tool integration standard", "area": "emerging"},
    {"topic": "Function calling: structured outputs, tool use patterns", "area": "emerging"},
    {"topic": "Streaming: SSE, WebSocket, chunked transfer for LLMs", "area": "emerging"},
    {"topic": "Structured generation: JSON mode, grammar constraints", "area": "emerging"},
    {"topic": "Multi-modal agents: vision, audio, code execution", "area": "emerging"},
]


@dataclass
class ExtendedResult:
    topic: str
    area: str
    sources: list
    findings: list
    vectors: int
    latency: float


class ExtendedTopicsExecutor:
    """Execute extended deep research iterations."""

    def __init__(self):
        self.exa = None
        self.qdrant = None
        self.keys = {
            "tavily": os.getenv("TAVILY_API_KEY"),
            "jina": os.getenv("JINA_API_KEY"),
            "perplexity": os.getenv("PERPLEXITY_API_KEY"),
        }
        self.stats = {"sources": 0, "vectors": 0, "findings": 0}

    async def initialize(self):
        from exa_py import Exa
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.exa = Exa(os.getenv("EXA_API_KEY"))
        self.qdrant = QdrantClient(":memory:")

        for area in ["rag", "agents", "optimization", "embeddings", "prompts", "eval", "emerging"]:
            try:
                self.qdrant.create_collection(area, vectors_config=VectorParams(size=1024, distance=Distance.COSINE))
            except:
                pass

        print("[OK] Extended Topics Executor initialized")
        print(f"    Collections: rag, agents, optimization, embeddings, prompts, eval, emerging")

    async def deep_research(self, topic: str, area: str) -> dict:
        """Deep research with all SDKs."""
        result = {"sources": [], "findings": [], "vectors": 0}

        async with httpx.AsyncClient(timeout=90) as client:
            tasks = [
                self._exa_search(topic),
                self._tavily_search(client, topic),
                self._perplexity_deep(client, topic, area),
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for r in responses:
                if isinstance(r, dict):
                    result["sources"].extend(r.get("sources", []))
                    result["findings"].extend(r.get("findings", []))

            # Embed to area-specific collection
            if result["sources"]:
                texts = [s.get("text", "")[:500] for s in result["sources"] if s.get("text")]
                if texts:
                    result["vectors"] = await self._embed(client, texts, area)

        return result

    async def _exa_search(self, topic: str) -> dict:
        try:
            search = self.exa.search_and_contents(topic, type="neural", num_results=5, text=True, highlights=True)
            sources = [{"title": r.title, "url": r.url, "text": r.text[:500] if r.text else ""} for r in search.results]
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
            sources = [{"title": s.get("title", ""), "url": s.get("url", ""), "text": s.get("content", "")[:500]} for s in data.get("results", [])]
            findings = [f"[tavily] {data.get('answer', '')[:200]}"] if data.get("answer") else []
            return {"sources": sources, "findings": findings}
        except:
            return {"sources": [], "findings": []}

    async def _perplexity_deep(self, client: httpx.AsyncClient, topic: str, area: str) -> dict:
        try:
            r = await client.post(
                'https://api.perplexity.ai/chat/completions',
                headers={'Authorization': f'Bearer {self.keys["perplexity"]}'},
                json={
                    'model': 'sonar-pro',
                    'messages': [{'role': 'user', 'content': f"Deep technical analysis of {topic} in the context of {area}. Include implementation details, best practices, and latest developments."}],
                    'return_citations': True
                }
            )
            data = r.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            citations = data.get('citations', [])
            sources = [{"title": f"Citation {i+1}", "url": c, "text": ""} for i, c in enumerate(citations[:3])]
            findings = [f"[perplexity] {content[:220]}"] if content else []
            return {"sources": sources, "findings": findings}
        except:
            return {"sources": [], "findings": []}

    async def _embed(self, client: httpx.AsyncClient, texts: list, area: str) -> int:
        try:
            from qdrant_client.models import PointStruct

            r = await client.post(
                'https://api.jina.ai/v1/embeddings',
                headers={'Authorization': f'Bearer {self.keys["jina"]}'},
                json={'model': 'jina-embeddings-v3', 'input': texts[:10], 'task': 'retrieval.passage'}
            )
            embeddings = [d.get('embedding', []) for d in r.json().get('data', [])]

            if not embeddings or not embeddings[0]:
                return 0

            base_id = int(time.time() * 1000) % 1000000
            points = [PointStruct(id=base_id + i, vector=emb, payload={"text": texts[i][:300], "area": area}) for i, emb in enumerate(embeddings)]
            self.qdrant.upsert(area, points=points)

            return len(points)
        except:
            return 0

    async def run_iteration(self, topic_data: dict, index: int) -> ExtendedResult:
        topic = topic_data["topic"]
        area = topic_data["area"]

        print(f"\n[{index:02d}] [{area}] {topic[:50]}...")

        start = time.time()
        result = await self.deep_research(topic, area)
        latency = time.time() - start

        self.stats["sources"] += len(result["sources"])
        self.stats["vectors"] += result["vectors"]
        self.stats["findings"] += len(result["findings"])

        print(f"    Src:{len(result['sources'])} Vec:{result['vectors']} Find:{len(result['findings'])} [{latency:.1f}s]")

        if result["findings"]:
            f = result["findings"][0]
            clean = f.encode('ascii', 'replace').decode('ascii')
            print(f"    -> {clean[:60]}...")

        return ExtendedResult(
            topic=topic,
            area=area,
            sources=result["sources"],
            findings=result["findings"],
            vectors=result["vectors"],
            latency=latency
        )


async def main():
    print("="*70)
    print("EXTENDED TOPICS ITERATIONS - DEEP RESEARCH")
    print("="*70)
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Topics: {len(EXTENDED_TOPICS)}")
    print("="*70)

    executor = ExtendedTopicsExecutor()
    await executor.initialize()

    all_results = []

    # Group by area
    by_area = {}
    for t in EXTENDED_TOPICS:
        by_area.setdefault(t["area"], []).append(t)

    iteration = 0
    for area, topics in by_area.items():
        print(f"\n{'='*70}")
        print(f"AREA: {area.upper()}")
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
    print("EXTENDED TOPICS COMPLETE")
    print("="*70)

    print(f"\n  Iterations: {len(all_results)}")
    print(f"  Total Sources: {executor.stats['sources']}")
    print(f"  Total Vectors: {executor.stats['vectors']}")
    print(f"  Total Findings: {executor.stats['findings']}")

    avg_latency = sum(r.latency for r in all_results) / len(all_results) if all_results else 0
    print(f"  Avg Latency: {avg_latency:.1f}s")

    # Vector collections
    print("\n  VECTOR COLLECTIONS:")
    for area in ["rag", "agents", "optimization", "embeddings", "prompts", "eval", "emerging"]:
        try:
            count = executor.qdrant.count(area).count
            print(f"    {area}: {count} vectors")
        except:
            pass

    print("\n  BY AREA:")
    area_stats = {}
    for r in all_results:
        area_stats[r.area] = area_stats.get(r.area, 0) + 1
    for area, count in sorted(area_stats.items()):
        print(f"    {area}: {count}")

    # Key findings per area
    print("\n  KEY FINDINGS:")
    seen = set()
    for r in all_results:
        if r.area not in seen and r.findings:
            seen.add(r.area)
            f = r.findings[0]
            clean = f.encode('ascii', 'replace').decode('ascii')
            print(f"    [{r.area}] {clean[:55]}...")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "stats": executor.stats,
        "results": [
            {
                "topic": r.topic,
                "area": r.area,
                "sources": len(r.sources),
                "vectors": r.vectors,
                "findings": r.findings[:3],
                "latency": r.latency
            }
            for r in all_results
        ]
    }

    with open("extended_topics_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Results saved to extended_topics_results.json")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
