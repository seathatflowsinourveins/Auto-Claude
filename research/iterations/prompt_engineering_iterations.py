"""
PROMPT ENGINEERING ITERATIONS - Advanced Prompting Techniques
=============================================================
System prompts, few-shot, chain-of-thought, structured outputs
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

PROMPT_TOPICS = [
    # System Prompts
    {"topic": "System prompt design: persona, constraints, output format", "area": "system"},
    {"topic": "Role-playing prompts: expert personas, domain knowledge injection", "area": "system"},
    {"topic": "Instruction hierarchy: system vs user prompt precedence", "area": "system"},
    {"topic": "Meta-prompting: prompts that generate prompts", "area": "system"},

    # Few-Shot Learning
    {"topic": "Few-shot example selection: similarity-based, diverse sampling", "area": "fewshot"},
    {"topic": "Dynamic few-shot: runtime example retrieval from vector DB", "area": "fewshot"},
    {"topic": "Example formatting: input-output pairs, chain-of-thought examples", "area": "fewshot"},
    {"topic": "Zero-shot vs few-shot: task complexity thresholds", "area": "fewshot"},

    # Reasoning Techniques
    {"topic": "Chain-of-thought prompting: step-by-step reasoning elicitation", "area": "reasoning"},
    {"topic": "Self-consistency: multiple reasoning paths with voting", "area": "reasoning"},
    {"topic": "Tree of thoughts: branching exploration, backtracking", "area": "reasoning"},
    {"topic": "ReAct: reasoning and acting interleaved", "area": "reasoning"},

    # Structured Outputs
    {"topic": "JSON mode: schema enforcement, nested structures", "area": "structured"},
    {"topic": "Function calling: tool definitions, parameter extraction", "area": "structured"},
    {"topic": "Pydantic output parsing: type validation, error handling", "area": "structured"},
    {"topic": "XML and markdown structured outputs: parsing strategies", "area": "structured"},

    # Advanced Techniques
    {"topic": "Prompt compression: LLMLingua, context distillation", "area": "advanced"},
    {"topic": "Prompt injection defense: delimiters, input validation", "area": "advanced"},
    {"topic": "Multi-turn prompt design: context management, summarization", "area": "advanced"},
    {"topic": "Prompt versioning and testing: A/B testing, regression", "area": "advanced"},
]

@dataclass
class PromptResult:
    topic: str
    area: str
    sources: list
    findings: list
    vectors: int
    latency: float

class PromptExecutor:
    def __init__(self):
        self.exa = None
        self.qdrant = None
        self.keys = {"tavily": os.getenv("TAVILY_API_KEY"), "jina": os.getenv("JINA_API_KEY"), "perplexity": os.getenv("PERPLEXITY_API_KEY")}
        self.stats = {"sources": 0, "vectors": 0, "findings": 0}

    async def initialize(self):
        from exa_py import Exa
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        self.exa = Exa(os.getenv("EXA_API_KEY"))
        self.qdrant = QdrantClient(":memory:")
        self.qdrant.create_collection("prompts", vectors_config=VectorParams(size=1024, distance=Distance.COSINE))
        print("[OK] Prompt Engineering Executor initialized")

    async def research(self, topic: str, area: str) -> dict:
        result = {"sources": [], "findings": [], "vectors": 0}
        async with httpx.AsyncClient(timeout=90) as client:
            tasks = [self._exa(topic), self._tavily(client, topic), self._perplexity(client, topic, area)]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for r in responses:
                if isinstance(r, dict):
                    result["sources"].extend(r.get("sources", []))
                    result["findings"].extend(r.get("findings", []))
            if result["sources"]:
                texts = [s.get("text", "")[:500] for s in result["sources"] if s.get("text")]
                if texts: result["vectors"] = await self._embed(client, texts)
        return result

    async def _exa(self, topic: str) -> dict:
        try:
            search = self.exa.search_and_contents(topic, type="neural", num_results=5, text=True, highlights=True)
            sources = [{"title": r.title, "url": r.url, "text": r.text[:500] if r.text else ""} for r in search.results]
            findings = [f"[exa] {r.title}" for r in search.results[:2]]
            if search.results and hasattr(search.results[0], 'highlights') and search.results[0].highlights:
                findings.append(f"[exa-h] {search.results[0].highlights[0][:120]}")
            return {"sources": sources, "findings": findings}
        except: return {"sources": [], "findings": []}

    async def _tavily(self, client, topic: str) -> dict:
        try:
            r = await client.post('https://api.tavily.com/search', json={'api_key': self.keys["tavily"], 'query': topic, 'search_depth': 'advanced', 'include_answer': True, 'max_results': 5})
            data = r.json()
            sources = [{"title": s.get("title", ""), "url": s.get("url", ""), "text": s.get("content", "")[:500]} for s in data.get("results", [])]
            findings = [f"[tavily] {data.get('answer', '')[:200]}"] if data.get("answer") else []
            return {"sources": sources, "findings": findings}
        except: return {"sources": [], "findings": []}

    async def _perplexity(self, client, topic: str, area: str) -> dict:
        try:
            r = await client.post('https://api.perplexity.ai/chat/completions', headers={'Authorization': f'Bearer {self.keys["perplexity"]}'}, json={'model': 'sonar-pro', 'messages': [{'role': 'user', 'content': f"Prompt engineering technique: {topic}"}], 'return_citations': True})
            data = r.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            findings = [f"[perplexity] {content[:220]}"] if content else []
            return {"sources": [], "findings": findings}
        except: return {"sources": [], "findings": []}

    async def _embed(self, client, texts: list) -> int:
        try:
            from qdrant_client.models import PointStruct
            r = await client.post('https://api.jina.ai/v1/embeddings', headers={'Authorization': f'Bearer {self.keys["jina"]}'}, json={'model': 'jina-embeddings-v3', 'input': texts[:10], 'task': 'retrieval.passage'})
            embeddings = [d.get('embedding', []) for d in r.json().get('data', [])]
            if not embeddings or not embeddings[0]: return 0
            base_id = int(time.time() * 1000) % 1000000
            points = [PointStruct(id=base_id + i, vector=emb, payload={"text": texts[i][:300]}) for i, emb in enumerate(embeddings)]
            self.qdrant.upsert("prompts", points=points)
            return len(points)
        except: return 0

    async def run_iteration(self, topic_data: dict, index: int) -> PromptResult:
        topic, area = topic_data["topic"], topic_data["area"]
        print(f"\n[{index:02d}] [{area}] {topic[:50]}...")
        start = time.time()
        result = await self.research(topic, area)
        latency = time.time() - start
        self.stats["sources"] += len(result["sources"])
        self.stats["vectors"] += result["vectors"]
        self.stats["findings"] += len(result["findings"])
        print(f"    Src:{len(result['sources'])} Vec:{result['vectors']} Find:{len(result['findings'])} [{latency:.1f}s]")
        if result["findings"]:
            clean = result["findings"][0].encode('ascii', 'replace').decode('ascii')
            print(f"    -> {clean[:60]}...")
        return PromptResult(topic=topic, area=area, sources=result["sources"], findings=result["findings"], vectors=result["vectors"], latency=latency)

async def main():
    print("="*70)
    print("PROMPT ENGINEERING ITERATIONS")
    print("="*70)
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Topics: {len(PROMPT_TOPICS)}")
    print("="*70)

    executor = PromptExecutor()
    await executor.initialize()
    all_results = []

    for i, topic_data in enumerate(PROMPT_TOPICS, 1):
        try:
            result = await executor.run_iteration(topic_data, i)
            all_results.append(result)
        except Exception as e:
            print(f"    [ERR] {str(e)[:50]}")
        await asyncio.sleep(0.3)

    print("\n" + "="*70)
    print("PROMPT ENGINEERING COMPLETE")
    print("="*70)
    print(f"\n  Iterations: {len(all_results)}")
    print(f"  Sources: {executor.stats['sources']}")
    print(f"  Vectors: {executor.stats['vectors']}")
    print(f"  Findings: {executor.stats['findings']}")
    avg = sum(r.latency for r in all_results) / len(all_results) if all_results else 0
    print(f"  Avg Latency: {avg:.1f}s")

    with open("prompt_engineering_results.json", 'w') as f:
        json.dump({"timestamp": datetime.now().isoformat(), "stats": executor.stats, "results": [{"topic": r.topic, "area": r.area, "sources": len(r.sources), "vectors": r.vectors, "findings": r.findings[:3], "latency": r.latency} for r in all_results]}, f, indent=2)
    print(f"\n[OK] Saved to prompt_engineering_results.json")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())
