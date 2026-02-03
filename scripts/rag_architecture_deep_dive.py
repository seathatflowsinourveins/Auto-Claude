"""
RAG ARCHITECTURE DEEP DIVE - Battle-Tested SDK Execution
=========================================================
Uses all production SDKs to research AND execute RAG patterns.
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
# SDK CLIENTS
# ============================================================================

class ResearchExecutor:
    """Executes research using all available SDKs."""

    def __init__(self):
        self.exa = None
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        self.jina_key = os.getenv("JINA_API_KEY")
        self.perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

    async def initialize(self):
        """Initialize all SDK clients."""
        from exa_py import Exa
        self.exa = Exa(os.getenv("EXA_API_KEY"))
        print("[OK] All SDK clients initialized")

    async def multi_source_research(self, query: str) -> dict:
        """Research using Exa + Tavily + Perplexity in parallel."""
        results = {"query": query, "sources": [], "findings": []}

        async with httpx.AsyncClient(timeout=60) as client:
            # Parallel execution
            tasks = [
                self._exa_search(query),
                self._tavily_search(client, query),
                self._perplexity_research(client, query),
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        for r in responses:
            if isinstance(r, dict) and not isinstance(r, Exception):
                results["sources"].extend(r.get("sources", []))
                results["findings"].extend(r.get("findings", []))

        return results

    async def _exa_search(self, query: str) -> dict:
        """Exa neural search with contents."""
        try:
            search = self.exa.search_and_contents(query, type="auto", num_results=5, text=True)
            sources = [{"title": r.title, "url": r.url, "text": r.text[:500] if r.text else ""} for r in search.results]
            return {"sources": sources, "findings": [f"[exa] {r.title}" for r in search.results[:3]]}
        except Exception as e:
            return {"sources": [], "findings": [f"[exa] Error: {str(e)[:50]}"]}

    async def _tavily_search(self, client: httpx.AsyncClient, query: str) -> dict:
        """Tavily AI search with answer synthesis."""
        try:
            r = await client.post('https://api.tavily.com/search', json={
                'api_key': self.tavily_key,
                'query': query,
                'search_depth': 'advanced',
                'include_answer': True,
                'max_results': 5
            })
            data = r.json()
            sources = [{"title": s.get("title", ""), "url": s.get("url", ""), "text": s.get("content", "")[:500]} for s in data.get("results", [])]
            findings = [f"[tavily] {data.get('answer', 'No answer')[:200]}"]
            return {"sources": sources, "findings": findings}
        except Exception as e:
            return {"sources": [], "findings": [f"[tavily] Error: {str(e)[:50]}"]}

    async def _perplexity_research(self, client: httpx.AsyncClient, query: str) -> dict:
        """Perplexity Sonar Pro for deep research."""
        try:
            r = await client.post(
                'https://api.perplexity.ai/chat/completions',
                headers={'Authorization': f'Bearer {self.perplexity_key}', 'Content-Type': 'application/json'},
                json={
                    'model': 'sonar-pro',
                    'messages': [{'role': 'user', 'content': f"Provide detailed technical analysis: {query}"}],
                    'return_citations': True
                }
            )
            data = r.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            citations = data.get('citations', [])
            sources = [{"title": f"Citation {i+1}", "url": c, "text": ""} for i, c in enumerate(citations[:5])]
            findings = [f"[perplexity] {content[:300]}..."]
            return {"sources": sources, "findings": findings}
        except Exception as e:
            return {"sources": [], "findings": [f"[perplexity] Error: {str(e)[:50]}"]}

    async def embed_and_store(self, texts: list[str]) -> dict:
        """Embed texts using Jina and store in Qdrant."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct

        async with httpx.AsyncClient(timeout=60) as client:
            # Get Jina embeddings
            r = await client.post(
                'https://api.jina.ai/v1/embeddings',
                headers={'Authorization': f'Bearer {self.jina_key}', 'Content-Type': 'application/json'},
                json={'model': 'jina-embeddings-v3', 'input': texts[:10]}
            )
            data = r.json()
            embeddings = [d.get('embedding', []) for d in data.get('data', [])]

        if not embeddings or not embeddings[0]:
            return {"stored": 0, "error": "No embeddings generated"}

        # Store in Qdrant
        qdrant = QdrantClient(":memory:")
        dim = len(embeddings[0])
        qdrant.create_collection(
            collection_name="rag_research",
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

        points = [
            PointStruct(id=i, vector=emb, payload={"text": texts[i][:500]})
            for i, emb in enumerate(embeddings)
        ]
        qdrant.upsert(collection_name="rag_research", points=points)

        return {"stored": len(points), "dimensions": dim}

    async def rerank_results(self, query: str, documents: list[str]) -> list[str]:
        """Rerank documents using Jina reranker."""
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                'https://api.jina.ai/v1/rerank',
                headers={'Authorization': f'Bearer {self.jina_key}', 'Content-Type': 'application/json'},
                json={
                    'model': 'jina-reranker-v2-base-multilingual',
                    'query': query,
                    'documents': documents[:10],
                    'top_n': 5
                }
            )
            data = r.json()
            results = data.get('results', [])
            return [documents[r.get('index', 0)] for r in results]


# ============================================================================
# RAG ARCHITECTURE RESEARCH TOPICS
# ============================================================================

RAG_TOPICS = [
    # Core RAG Patterns
    {
        "topic": "Naive RAG vs Advanced RAG vs Modular RAG architecture comparison",
        "execute": True,
        "category": "Core Architecture"
    },
    {
        "topic": "Chunking strategies: fixed-size vs semantic vs agentic chunking production patterns",
        "execute": True,
        "category": "Ingestion"
    },
    {
        "topic": "Embedding model selection: OpenAI vs Cohere vs Jina vs BGE benchmarks 2026",
        "execute": True,
        "category": "Embeddings"
    },
    {
        "topic": "Vector database indexing: HNSW vs IVF vs flat index tradeoffs",
        "execute": True,
        "category": "Storage"
    },
    {
        "topic": "Hybrid search: combining BM25 keyword with dense vector retrieval",
        "execute": True,
        "category": "Retrieval"
    },

    # Advanced RAG
    {
        "topic": "Query transformation: HyDE vs query expansion vs multi-query retrieval",
        "execute": True,
        "category": "Query Processing"
    },
    {
        "topic": "Reranking strategies: cross-encoder vs ColBERT vs Cohere rerank comparison",
        "execute": True,
        "category": "Reranking"
    },
    {
        "topic": "Context compression and summarization for long document RAG",
        "execute": True,
        "category": "Context"
    },
    {
        "topic": "Self-RAG and Corrective RAG implementation patterns",
        "execute": True,
        "category": "Self-Improvement"
    },
    {
        "topic": "GraphRAG: combining knowledge graphs with vector retrieval",
        "execute": True,
        "category": "Graph RAG"
    },

    # Production RAG
    {
        "topic": "RAG evaluation metrics: faithfulness, relevance, answer correctness",
        "execute": True,
        "category": "Evaluation"
    },
    {
        "topic": "RAG caching strategies: semantic cache vs query cache patterns",
        "execute": True,
        "category": "Caching"
    },
    {
        "topic": "Multi-modal RAG: images, tables, and code in retrieval",
        "execute": True,
        "category": "Multi-Modal"
    },
    {
        "topic": "RAG observability: tracing, logging, and debugging pipelines",
        "execute": True,
        "category": "Observability"
    },
    {
        "topic": "RAG security: prompt injection, data poisoning, access control",
        "execute": True,
        "category": "Security"
    },
]


# ============================================================================
# EXECUTION LOOP
# ============================================================================

async def research_and_execute(executor: ResearchExecutor, topic: dict, iteration: int) -> dict:
    """Research a topic and execute with SDKs."""
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration}: {topic['category']}")
    print(f"Topic: {topic['topic'][:60]}...")
    print("="*70)

    result = {
        "iteration": iteration,
        "category": topic["category"],
        "topic": topic["topic"],
        "timestamp": datetime.now().isoformat(),
    }

    start_time = time.time()

    # Phase 1: Multi-source research
    print("\n  [PHASE 1] Multi-Source Research...")
    research = await executor.multi_source_research(topic["topic"])
    result["sources_count"] = len(research["sources"])
    result["findings"] = research["findings"][:5]
    print(f"      [OK] Sources: {len(research['sources'])}")
    for f in research["findings"][:2]:
        print(f"          {f[:70]}...")

    # Phase 2: Embed and store findings
    if topic.get("execute") and research["sources"]:
        print("\n  [PHASE 2] Embed & Store (Jina + Qdrant)...")
        texts = [s.get("text", s.get("title", "")) for s in research["sources"] if s.get("text") or s.get("title")]
        if texts:
            store_result = await executor.embed_and_store(texts[:10])
            result["embedded"] = store_result.get("stored", 0)
            result["dimensions"] = store_result.get("dimensions", 0)
            print(f"      [OK] Stored: {result['embedded']} vectors ({result['dimensions']} dims)")

    # Phase 3: Rerank for relevance
    if topic.get("execute") and research["sources"]:
        print("\n  [PHASE 3] Rerank Results (Jina Reranker)...")
        texts = [s.get("text", s.get("title", ""))[:200] for s in research["sources"] if s.get("text") or s.get("title")]
        if texts:
            reranked = await executor.rerank_results(topic["topic"], texts[:10])
            result["reranked_top"] = reranked[:3]
            print(f"      [OK] Top reranked: {len(reranked)}")
            for r in reranked[:2]:
                print(f"          - {r[:60]}...")

    result["latency_s"] = time.time() - start_time
    print(f"\n  [COMPLETE] Latency: {result['latency_s']:.1f}s")

    return result


async def main():
    """Main RAG architecture deep dive loop."""
    print("="*70)
    print("RAG ARCHITECTURE DEEP DIVE - BATTLE-TESTED SDK EXECUTION")
    print("="*70)
    print(f"Topics: {len(RAG_TOPICS)}")
    print(f"Start Time: {datetime.now().isoformat()}")
    print("="*70)

    # Initialize executor
    executor = ResearchExecutor()
    await executor.initialize()

    # Execute all topics
    all_results = []
    total_sources = 0
    total_embedded = 0

    for i, topic in enumerate(RAG_TOPICS, 1):
        try:
            result = await research_and_execute(executor, topic, i)
            all_results.append(result)
            total_sources += result.get("sources_count", 0)
            total_embedded += result.get("embedded", 0)
        except Exception as e:
            print(f"  [ERROR] {e}")
            all_results.append({
                "iteration": i,
                "category": topic["category"],
                "error": str(e)
            })

        # Brief pause
        if i < len(RAG_TOPICS):
            await asyncio.sleep(0.5)

    # Summary
    print("\n" + "="*70)
    print("RAG ARCHITECTURE DEEP DIVE COMPLETE")
    print("="*70)

    successful = [r for r in all_results if "error" not in r]
    print(f"\n  Iterations: {len(all_results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Total Sources: {total_sources}")
    print(f"  Total Embedded: {total_embedded}")

    if successful:
        avg_latency = sum(r.get("latency_s", 0) for r in successful) / len(successful)
        print(f"  Avg Latency: {avg_latency:.1f}s")

    # Category breakdown
    print("\n  By Category:")
    categories = {}
    for r in successful:
        cat = r.get("category", "Unknown")
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in categories.items():
        print(f"      {cat}: {count} topics")

    # Save results
    output_file = "rag_architecture_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_topics": len(RAG_TOPICS),
                "successful": len(successful),
                "total_sources": total_sources,
                "total_embedded": total_embedded,
            },
            "results": all_results
        }, f, indent=2)

    print(f"\n[OK] Results saved to {output_file}")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
