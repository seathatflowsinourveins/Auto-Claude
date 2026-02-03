"""
BEYOND RAG DEEP DIVE - Agentic Systems & Advanced Patterns
===========================================================
Research and execute advanced AI patterns beyond basic RAG.
"""

import asyncio
import os
import sys
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import httpx

load_dotenv('.config/.env')

# ============================================================================
# BEYOND RAG TOPICS - ADVANCED PATTERNS
# ============================================================================

BEYOND_RAG_TOPICS = [
    # Agentic RAG
    {
        "topic": "Agentic RAG: routing queries to specialized retrievers with LangGraph",
        "category": "Agentic RAG"
    },
    {
        "topic": "Tool-augmented RAG: when to retrieve vs when to call external APIs",
        "category": "Tool Augmentation"
    },
    {
        "topic": "ReAct pattern with RAG: reasoning and acting with retrieved context",
        "category": "ReAct RAG"
    },

    # Memory Systems
    {
        "topic": "Long-term memory for agents: episodic vs semantic vs procedural memory",
        "category": "Agent Memory"
    },
    {
        "topic": "Mem0 vs Letta vs MemGPT: production memory architectures comparison",
        "category": "Memory Systems"
    },
    {
        "topic": "Conversation memory management: sliding window vs summarization vs compression",
        "category": "Conversation Memory"
    },

    # Multi-Agent Systems
    {
        "topic": "Multi-agent orchestration: CrewAI vs AutoGen vs LangGraph comparison",
        "category": "Multi-Agent"
    },
    {
        "topic": "Agent communication patterns: shared memory vs message passing vs blackboard",
        "category": "Agent Communication"
    },
    {
        "topic": "Hierarchical agent architectures: supervisor vs peer-to-peer patterns",
        "category": "Agent Hierarchy"
    },

    # Advanced Retrieval
    {
        "topic": "Late interaction models: ColBERT vs PLAID vs ColBERTv2 for retrieval",
        "category": "Late Interaction"
    },
    {
        "topic": "Dense passage retrieval vs sparse retrieval: ANCE, DPR, SPLADE comparison",
        "category": "Dense Retrieval"
    },
    {
        "topic": "Cross-encoder fine-tuning for domain-specific reranking",
        "category": "Fine-tuning"
    },

    # Production Patterns
    {
        "topic": "RAG at scale: sharding, replication, and distributed vector search",
        "category": "Scale"
    },
    {
        "topic": "Real-time RAG: streaming updates and incremental indexing",
        "category": "Real-time"
    },
    {
        "topic": "RAG cost optimization: embedding caching, query deduplication, batching",
        "category": "Cost"
    },

    # Emerging Patterns
    {
        "topic": "Speculative RAG: parallel retrieval with early termination",
        "category": "Speculative"
    },
    {
        "topic": "Constitutional RAG: alignment and safety in retrieval systems",
        "category": "Safety"
    },
    {
        "topic": "RAG with structured data: SQL, knowledge graphs, and APIs",
        "category": "Structured Data"
    },
]


class AdvancedResearchExecutor:
    """Execute advanced research with all SDKs."""

    def __init__(self):
        self.exa = None
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        self.jina_key = os.getenv("JINA_API_KEY")
        self.perplexity_key = os.getenv("PERPLEXITY_API_KEY")

    async def initialize(self):
        from exa_py import Exa
        self.exa = Exa(os.getenv("EXA_API_KEY"))
        print("[OK] Advanced Research Executor initialized")

    async def deep_research(self, topic: str) -> dict:
        """Deep research using all APIs with execution."""
        results = {
            "topic": topic,
            "sources": [],
            "findings": [],
            "code_examples": [],
            "key_insights": [],
        }

        async with httpx.AsyncClient(timeout=90) as client:
            # Parallel API calls
            tasks = [
                self._exa_deep_search(topic),
                self._tavily_advanced(client, topic),
                self._perplexity_pro(client, topic),
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        for r in responses:
            if isinstance(r, dict):
                results["sources"].extend(r.get("sources", []))
                results["findings"].extend(r.get("findings", []))
                results["key_insights"].extend(r.get("insights", []))

        # Post-process: Extract unique insights
        seen = set()
        unique_findings = []
        for f in results["findings"]:
            key = f[:50].lower()
            if key not in seen:
                seen.add(key)
                unique_findings.append(f)
        results["findings"] = unique_findings[:10]

        return results

    async def _exa_deep_search(self, topic: str) -> dict:
        """Exa search with contents and highlights."""
        try:
            # Search for technical content
            search = self.exa.search_and_contents(
                topic,
                type="auto",
                num_results=5,
                text=True,
                highlights=True
            )

            sources = []
            findings = []
            insights = []

            for r in search.results:
                sources.append({
                    "title": r.title,
                    "url": r.url,
                    "text": r.text[:400] if r.text else "",
                    "source": "exa"
                })
                findings.append(f"[exa] {r.title}")

                # Extract highlights as insights
                if hasattr(r, 'highlights') and r.highlights:
                    for h in r.highlights[:2]:
                        insights.append(f"[exa-highlight] {h[:150]}")

            return {"sources": sources, "findings": findings, "insights": insights}
        except Exception as e:
            return {"sources": [], "findings": [f"[exa] Error: {str(e)[:50]}"], "insights": []}

    async def _tavily_advanced(self, client: httpx.AsyncClient, topic: str) -> dict:
        """Tavily advanced search with answer synthesis."""
        try:
            r = await client.post('https://api.tavily.com/search', json={
                'api_key': self.tavily_key,
                'query': topic,
                'search_depth': 'advanced',
                'include_answer': True,
                'include_raw_content': False,
                'max_results': 5
            })
            data = r.json()

            sources = [{
                "title": s.get("title", ""),
                "url": s.get("url", ""),
                "text": s.get("content", "")[:400],
                "source": "tavily"
            } for s in data.get("results", [])]

            findings = []
            insights = []

            if data.get("answer"):
                findings.append(f"[tavily] {data['answer'][:250]}")
                # Extract key sentences as insights
                sentences = data["answer"].split(". ")
                for s in sentences[:3]:
                    if len(s) > 30:
                        insights.append(f"[tavily-insight] {s.strip()}")

            return {"sources": sources, "findings": findings, "insights": insights}
        except Exception as e:
            return {"sources": [], "findings": [f"[tavily] Error: {str(e)[:50]}"], "insights": []}

    async def _perplexity_pro(self, client: httpx.AsyncClient, topic: str) -> dict:
        """Perplexity Sonar Pro for deep analysis."""
        try:
            r = await client.post(
                'https://api.perplexity.ai/chat/completions',
                headers={'Authorization': f'Bearer {self.perplexity_key}', 'Content-Type': 'application/json'},
                json={
                    'model': 'sonar-pro',
                    'messages': [{
                        'role': 'user',
                        'content': f"""Provide a technical deep-dive on: {topic}

Include:
1. Key concepts and definitions
2. Implementation patterns
3. Best practices
4. Common pitfalls to avoid
5. Code examples or pseudocode if applicable"""
                    }],
                    'return_citations': True
                }
            )
            data = r.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            citations = data.get('citations', [])

            sources = [{"title": f"Citation", "url": c, "text": "", "source": "perplexity"} for c in citations[:5]]

            # Extract structured insights from content
            findings = [f"[perplexity] {content[:300]}..."]
            insights = []

            # Parse numbered points
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                    if len(line) > 20:
                        insights.append(f"[perplexity-point] {line[:150]}")

            return {"sources": sources, "findings": findings, "insights": insights[:5]}
        except Exception as e:
            return {"sources": [], "findings": [f"[perplexity] Error: {str(e)[:50]}"], "insights": []}

    async def embed_and_index(self, texts: list[str]) -> dict:
        """Embed with Jina and index in Qdrant."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct

        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                'https://api.jina.ai/v1/embeddings',
                headers={'Authorization': f'Bearer {self.jina_key}', 'Content-Type': 'application/json'},
                json={'model': 'jina-embeddings-v3', 'input': texts[:15]}
            )
            data = r.json()
            embeddings = [d.get('embedding', []) for d in data.get('data', [])]

        if not embeddings or not embeddings[0]:
            return {"indexed": 0}

        qdrant = QdrantClient(":memory:")
        dim = len(embeddings[0])
        qdrant.create_collection(
            collection_name="beyond_rag",
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

        points = [
            PointStruct(id=i, vector=emb, payload={"text": texts[i][:300]})
            for i, emb in enumerate(embeddings)
        ]
        qdrant.upsert(collection_name="beyond_rag", points=points)

        return {"indexed": len(points), "dimensions": dim}


async def execute_deep_dive(executor: AdvancedResearchExecutor, topic: dict, iteration: int) -> dict:
    """Execute deep dive for a single topic."""
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration}: {topic['category']}")
    print(f"Topic: {topic['topic'][:55]}...")
    print("="*70)

    result = {
        "iteration": iteration,
        "category": topic["category"],
        "topic": topic["topic"],
        "timestamp": datetime.now().isoformat(),
    }

    start = time.time()

    # Phase 1: Deep research
    print("\n  [RESEARCH] Multi-Source Deep Dive...")
    research = await executor.deep_research(topic["topic"])
    result["sources"] = len(research["sources"])
    result["findings"] = research["findings"][:5]
    result["insights"] = research["key_insights"][:5]
    print(f"      Sources: {result['sources']}")
    print(f"      Findings: {len(research['findings'])}")
    print(f"      Insights: {len(research['key_insights'])}")

    # Phase 2: Embed and index
    print("\n  [INDEX] Jina Embed + Qdrant Store...")
    texts = [s.get("text", "") for s in research["sources"] if s.get("text")]
    if texts:
        index_result = await executor.embed_and_index(texts)
        result["indexed"] = index_result.get("indexed", 0)
        result["dimensions"] = index_result.get("dimensions", 0)
        print(f"      Indexed: {result['indexed']} vectors ({result.get('dimensions', 0)} dims)")

    # Phase 3: Show key insights
    if research["key_insights"]:
        print("\n  [INSIGHTS] Key Takeaways:")
        for insight in research["key_insights"][:3]:
            # Clean for console output
            clean = insight.encode('ascii', 'replace').decode('ascii')
            print(f"      - {clean[:65]}...")

    result["latency_s"] = time.time() - start
    print(f"\n  [DONE] {result['latency_s']:.1f}s")

    return result


async def main():
    """Main beyond RAG deep dive."""
    print("="*70)
    print("BEYOND RAG DEEP DIVE - AGENTIC SYSTEMS & ADVANCED PATTERNS")
    print("="*70)
    print(f"Topics: {len(BEYOND_RAG_TOPICS)}")
    print(f"Start: {datetime.now().isoformat()}")
    print("="*70)

    executor = AdvancedResearchExecutor()
    await executor.initialize()

    all_results = []
    total_sources = 0
    total_indexed = 0
    total_insights = 0

    for i, topic in enumerate(BEYOND_RAG_TOPICS, 1):
        try:
            result = await execute_deep_dive(executor, topic, i)
            all_results.append(result)
            total_sources += result.get("sources", 0)
            total_indexed += result.get("indexed", 0)
            total_insights += len(result.get("insights", []))
        except Exception as e:
            print(f"  [ERROR] {str(e)[:60]}")
            all_results.append({"iteration": i, "error": str(e)[:100]})

        if i < len(BEYOND_RAG_TOPICS):
            await asyncio.sleep(0.3)

    # Summary
    print("\n" + "="*70)
    print("BEYOND RAG DEEP DIVE COMPLETE")
    print("="*70)

    successful = [r for r in all_results if "error" not in r]
    print(f"\n  Iterations: {len(all_results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Total Sources: {total_sources}")
    print(f"  Total Indexed: {total_indexed}")
    print(f"  Total Insights: {total_insights}")

    if successful:
        avg_latency = sum(r.get("latency_s", 0) for r in successful) / len(successful)
        print(f"  Avg Latency: {avg_latency:.1f}s")

    # Category summary
    print("\n  Categories Covered:")
    categories = {}
    for r in successful:
        cat = r.get("category", "Unknown")
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"      {cat}: {count}")

    # Save
    output_file = "beyond_rag_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": len(BEYOND_RAG_TOPICS),
                "successful": len(successful),
                "sources": total_sources,
                "indexed": total_indexed,
                "insights": total_insights
            },
            "results": all_results
        }, f, indent=2)

    print(f"\n[OK] Results saved to {output_file}")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
