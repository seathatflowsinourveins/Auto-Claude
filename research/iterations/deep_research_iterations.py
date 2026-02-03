"""
DEEP RESEARCH ITERATIONS - Context-Aware SDK Feature Selection
================================================================
Utilizes different features from different research tools based on context
"""

import asyncio
import os
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from dotenv import load_dotenv
import httpx

load_dotenv('.config/.env')

# ============================================================================
# RESEARCH CONTEXT TYPES - Determines SDK Feature Selection
# ============================================================================

class ResearchContext(Enum):
    TECHNICAL_DOCS = "technical_docs"      # Use Exa neural + highlights
    COMPARISON = "comparison"               # Use Perplexity multi-step + Tavily answer
    IMPLEMENTATION = "implementation"       # Use Exa code search + Jina reader
    TROUBLESHOOTING = "troubleshooting"     # Use Tavily advanced + Exa recent
    ARCHITECTURE = "architecture"           # Use Perplexity deep + Exa neural
    BEST_PRACTICES = "best_practices"       # Use all SDKs with synthesis


# SDK Feature Matrix - What features to use for each context
SDK_FEATURES = {
    ResearchContext.TECHNICAL_DOCS: {
        "exa": {"type": "neural", "highlights": True, "num_results": 7},
        "tavily": {"search_depth": "basic", "include_answer": False},
        "perplexity": {"enabled": False},
        "jina": {"task": "retrieval.passage"}
    },
    ResearchContext.COMPARISON: {
        "exa": {"type": "auto", "highlights": True, "num_results": 5},
        "tavily": {"search_depth": "advanced", "include_answer": True},
        "perplexity": {"enabled": True, "multi_turn": True},
        "jina": {"task": "retrieval.query"}
    },
    ResearchContext.IMPLEMENTATION: {
        "exa": {"type": "keyword", "highlights": True, "num_results": 8},
        "tavily": {"search_depth": "advanced", "include_answer": True, "include_raw_content": True},
        "perplexity": {"enabled": True, "focus": "code"},
        "jina": {"task": "retrieval.passage", "reader": True}
    },
    ResearchContext.TROUBLESHOOTING: {
        "exa": {"type": "auto", "highlights": True, "num_results": 10},
        "tavily": {"search_depth": "advanced", "include_answer": True},
        "perplexity": {"enabled": True, "focus": "solutions"},
        "jina": {"task": "retrieval.query"}
    },
    ResearchContext.ARCHITECTURE: {
        "exa": {"type": "neural", "highlights": True, "num_results": 6},
        "tavily": {"search_depth": "advanced", "include_answer": True},
        "perplexity": {"enabled": True, "deep": True},
        "jina": {"task": "retrieval.passage"}
    },
    ResearchContext.BEST_PRACTICES: {
        "exa": {"type": "auto", "highlights": True, "num_results": 5},
        "tavily": {"search_depth": "advanced", "include_answer": True},
        "perplexity": {"enabled": True},
        "jina": {"task": "retrieval.passage"}
    },
}


# ============================================================================
# DEEP RESEARCH TOPICS WITH CONTEXT
# ============================================================================

DEEP_RESEARCH_TOPICS = [
    # Technical Documentation Deep Dives
    {"topic": "LangGraph checkpointing: SqliteSaver vs MemorySaver implementation details", "context": ResearchContext.TECHNICAL_DOCS, "gap": None},
    {"topic": "DSPy 2.6 TypedPredictor: signature constraints and validation", "context": ResearchContext.TECHNICAL_DOCS, "gap": None},
    {"topic": "Qdrant sparse vectors: BM25 implementation and hybrid search", "context": ResearchContext.TECHNICAL_DOCS, "gap": None},
    {"topic": "Letta agent state: core_memory vs archival_memory implementation", "context": ResearchContext.TECHNICAL_DOCS, "gap": None},

    # Comparison Research
    {"topic": "Vector database comparison 2026: Qdrant vs Pinecone vs Weaviate vs Milvus", "context": ResearchContext.COMPARISON, "gap": None},
    {"topic": "Embedding model comparison: OpenAI vs Cohere vs Jina vs Voyage", "context": ResearchContext.COMPARISON, "gap": None},
    {"topic": "Agent framework comparison: LangGraph vs CrewAI vs AutoGen vs DSPy", "context": ResearchContext.COMPARISON, "gap": None},
    {"topic": "RAG evaluation comparison: RAGAS vs DeepEval vs TruLens metrics", "context": ResearchContext.COMPARISON, "gap": None},

    # Implementation Patterns
    {"topic": "Implementing semantic router with LangChain and custom embeddings", "context": ResearchContext.IMPLEMENTATION, "gap": "routing"},
    {"topic": "Building multi-turn conversation memory with LangGraph persistence", "context": ResearchContext.IMPLEMENTATION, "gap": "memory"},
    {"topic": "Implementing tool-use agents with function calling and error recovery", "context": ResearchContext.IMPLEMENTATION, "gap": "tool_use"},
    {"topic": "Building hybrid RAG with dense and sparse retrieval fusion", "context": ResearchContext.IMPLEMENTATION, "gap": "hybrid_rag"},

    # Troubleshooting Patterns
    {"topic": "Debugging LLM hallucinations: detection, prevention, and mitigation", "context": ResearchContext.TROUBLESHOOTING, "gap": "hallucination"},
    {"topic": "Fixing context window overflow: chunking, compression, summarization", "context": ResearchContext.TROUBLESHOOTING, "gap": "context_overflow"},
    {"topic": "Resolving embedding dimension mismatch in vector stores", "context": ResearchContext.TROUBLESHOOTING, "gap": "dimension_mismatch"},
    {"topic": "Handling rate limits: retry strategies, queue management, fallbacks", "context": ResearchContext.TROUBLESHOOTING, "gap": "rate_limits"},

    # Architecture Patterns
    {"topic": "Event-driven agent architecture: message queues and async processing", "context": ResearchContext.ARCHITECTURE, "gap": None},
    {"topic": "Microservices architecture for LLM applications: service boundaries", "context": ResearchContext.ARCHITECTURE, "gap": None},
    {"topic": "Multi-tenant RAG architecture: isolation, sharing, resource management", "context": ResearchContext.ARCHITECTURE, "gap": None},
    {"topic": "Serverless LLM architecture: cold starts, scaling, cost optimization", "context": ResearchContext.ARCHITECTURE, "gap": None},

    # Best Practices
    {"topic": "Prompt engineering best practices: templates, few-shot, chain-of-thought", "context": ResearchContext.BEST_PRACTICES, "gap": None},
    {"topic": "RAG best practices: chunking, retrieval, reranking, generation", "context": ResearchContext.BEST_PRACTICES, "gap": None},
    {"topic": "Agent orchestration best practices: planning, execution, error handling", "context": ResearchContext.BEST_PRACTICES, "gap": None},
    {"topic": "Production LLM best practices: monitoring, testing, deployment", "context": ResearchContext.BEST_PRACTICES, "gap": None},
]


# ============================================================================
# GAP RESOLUTIONS
# ============================================================================

GAP_RESOLUTIONS = {
    "routing": {
        "pattern": "SemanticRouter",
        "implementation": """
class SemanticRouter:
    def __init__(self, routes: list[dict], embedder, threshold=0.82):
        self.routes = routes
        self.embedder = embedder
        self.threshold = threshold
        self.route_vectors = None

    async def initialize(self):
        texts = [r['description'] for r in self.routes]
        self.route_vectors = await self.embedder.embed_batch(texts)

    async def route(self, query: str) -> Optional[str]:
        query_vec = await self.embedder.embed(query)
        best_score, best_route = 0, None
        for i, route_vec in enumerate(self.route_vectors):
            score = cosine_similarity(query_vec, route_vec)
            if score > best_score:
                best_score, best_route = score, self.routes[i]['name']
        return best_route if best_score > self.threshold else 'default'
""",
        "config": {"threshold": 0.82, "fallback": "default", "cache_routes": True}
    },

    "memory": {
        "pattern": "ConversationMemoryManager",
        "implementation": """
class ConversationMemoryManager:
    def __init__(self, max_turns=20, summary_threshold=10):
        self.max_turns = max_turns
        self.summary_threshold = summary_threshold
        self.messages = []
        self.summary = ""

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content, "ts": time.time()})
        if len(self.messages) > self.max_turns:
            self._consolidate()

    def _consolidate(self):
        old = self.messages[:self.summary_threshold]
        self.summary = self._summarize(old)
        self.messages = self.messages[self.summary_threshold:]

    def get_context(self) -> list:
        context = []
        if self.summary:
            context.append({"role": "system", "content": f"Previous conversation summary: {self.summary}"})
        context.extend(self.messages)
        return context
""",
        "config": {"max_turns": 20, "summary_threshold": 10, "summarizer": "gpt-4o-mini"}
    },

    "tool_use": {
        "pattern": "ResilientToolExecutor",
        "implementation": """
class ResilientToolExecutor:
    def __init__(self, tools: dict, max_retries=3):
        self.tools = tools
        self.max_retries = max_retries
        self.execution_history = []

    async def execute(self, tool_name: str, params: dict) -> dict:
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}

        for attempt in range(self.max_retries):
            try:
                result = await self.tools[tool_name](**params)
                self.execution_history.append({
                    "tool": tool_name, "params": params,
                    "result": "success", "attempt": attempt + 1
                })
                return {"success": True, "result": result}
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {"error": str(e), "attempts": attempt + 1}
                await asyncio.sleep(2 ** attempt)
        return {"error": "Max retries exceeded"}
""",
        "config": {"max_retries": 3, "timeout": 30, "parallel": True}
    },

    "hybrid_rag": {
        "pattern": "HybridRetriever",
        "implementation": """
class HybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, alpha=0.5):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.alpha = alpha  # Weight for dense scores

    async def retrieve(self, query: str, k: int = 10) -> list:
        dense_results = await self.dense.search(query, k * 2)
        sparse_results = await self.sparse.search(query, k * 2)

        # Reciprocal Rank Fusion
        scores = {}
        for rank, doc in enumerate(dense_results):
            scores[doc.id] = scores.get(doc.id, 0) + self.alpha / (rank + 60)
        for rank, doc in enumerate(sparse_results):
            scores[doc.id] = scores.get(doc.id, 0) + (1 - self.alpha) / (rank + 60)

        sorted_ids = sorted(scores.keys(), key=lambda x: -scores[x])
        return [self._get_doc(id) for id in sorted_ids[:k]]
""",
        "config": {"alpha": 0.5, "k_constant": 60, "normalize": True}
    },

    "hallucination": {
        "pattern": "HallucinationDetector",
        "implementation": """
class HallucinationDetector:
    def __init__(self, fact_checker, threshold=0.7):
        self.fact_checker = fact_checker
        self.threshold = threshold

    async def check(self, response: str, context: list) -> dict:
        claims = self._extract_claims(response)
        results = []
        for claim in claims:
            score = await self.fact_checker.verify(claim, context)
            results.append({"claim": claim, "score": score, "grounded": score > self.threshold})

        grounded_ratio = sum(1 for r in results if r['grounded']) / len(results) if results else 1.0
        return {
            "grounded_ratio": grounded_ratio,
            "claims": results,
            "is_hallucination": grounded_ratio < 0.8
        }
""",
        "config": {"threshold": 0.7, "min_grounded_ratio": 0.8, "extract_method": "spacy"}
    },

    "context_overflow": {
        "pattern": "ContextCompressor",
        "implementation": """
class ContextCompressor:
    def __init__(self, max_tokens=8000, summarizer=None):
        self.max_tokens = max_tokens
        self.summarizer = summarizer

    async def compress(self, context: list) -> list:
        total_tokens = sum(self._count_tokens(m['content']) for m in context)
        if total_tokens <= self.max_tokens:
            return context

        # Strategy 1: Summarize older messages
        midpoint = len(context) // 2
        older = context[:midpoint]
        newer = context[midpoint:]

        summary = await self.summarizer.summarize([m['content'] for m in older])
        compressed = [{"role": "system", "content": f"Earlier context: {summary}"}]
        compressed.extend(newer)
        return compressed
""",
        "config": {"max_tokens": 8000, "compression_ratio": 0.5, "preserve_recent": 5}
    },

    "dimension_mismatch": {
        "pattern": "EmbeddingAdapter",
        "implementation": """
class EmbeddingAdapter:
    def __init__(self, source_dim: int, target_dim: int, method='projection'):
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.method = method
        self.projection_matrix = None

    def initialize(self):
        if self.method == 'projection':
            # Random projection (preserves distances approximately)
            self.projection_matrix = np.random.randn(self.source_dim, self.target_dim)
            self.projection_matrix /= np.linalg.norm(self.projection_matrix, axis=0)
        elif self.method == 'truncate':
            pass  # Just take first target_dim dimensions

    def adapt(self, embedding: np.ndarray) -> np.ndarray:
        if self.method == 'projection':
            return embedding @ self.projection_matrix
        elif self.method == 'truncate':
            return embedding[:self.target_dim]
        elif self.method == 'pad':
            return np.pad(embedding, (0, self.target_dim - self.source_dim))
""",
        "config": {"methods": ["projection", "truncate", "pad"], "default": "projection"}
    },

    "rate_limits": {
        "pattern": "AdaptiveRateLimiter",
        "implementation": """
class AdaptiveRateLimiter:
    def __init__(self, base_rpm=60, burst_multiplier=1.5):
        self.base_rpm = base_rpm
        self.burst_multiplier = burst_multiplier
        self.request_times = []
        self.backoff_until = 0

    async def acquire(self):
        now = time.time()

        # Check backoff
        if now < self.backoff_until:
            wait_time = self.backoff_until - now
            await asyncio.sleep(wait_time)

        # Clean old requests
        self.request_times = [t for t in self.request_times if now - t < 60]

        # Check rate
        if len(self.request_times) >= self.base_rpm:
            wait_time = 60 - (now - self.request_times[0])
            await asyncio.sleep(wait_time)

        self.request_times.append(time.time())

    def report_rate_limit(self, retry_after: int = 60):
        self.backoff_until = time.time() + retry_after
""",
        "config": {"base_rpm": 60, "burst_multiplier": 1.5, "retry_after_default": 60}
    }
}


@dataclass
class DeepResearchResult:
    topic: str
    context: ResearchContext
    sources: list
    findings: list
    vectors: int
    latency: float
    sdks_used: dict
    gap_resolution: Optional[dict] = None


class DeepResearchExecutor:
    """Execute deep research with context-aware SDK feature selection."""

    def __init__(self):
        self.exa = None
        self.qdrant = None
        self.keys = {
            "tavily": os.getenv("TAVILY_API_KEY"),
            "jina": os.getenv("JINA_API_KEY"),
            "perplexity": os.getenv("PERPLEXITY_API_KEY"),
        }
        self.stats = {"sources": 0, "vectors": 0, "insights": 0, "gaps_resolved": 0}
        self.sdk_usage = {"exa": 0, "tavily": 0, "perplexity": 0, "jina": 0}

    async def initialize(self):
        from exa_py import Exa
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.exa = Exa(os.getenv("EXA_API_KEY"))
        self.qdrant = QdrantClient(":memory:")

        for name in ["deep_research", "gap_resolutions"]:
            try:
                self.qdrant.create_collection(name, vectors_config=VectorParams(size=1024, distance=Distance.COSINE))
            except:
                pass

        print("[OK] Deep Research Executor initialized")
        print("    Context-aware SDK feature selection enabled")

    async def research(self, topic: str, context: ResearchContext) -> dict:
        """Execute research with context-specific SDK features."""
        features = SDK_FEATURES[context]
        result = {"sources": [], "findings": [], "vectors": 0, "sdks_used": {}}

        async with httpx.AsyncClient(timeout=90) as client:
            tasks = []

            # Exa with context-specific settings
            exa_config = features["exa"]
            tasks.append(self._exa_search(topic, exa_config))
            result["sdks_used"]["exa"] = exa_config

            # Tavily with context-specific settings
            tavily_config = features["tavily"]
            tasks.append(self._tavily_search(client, topic, tavily_config))
            result["sdks_used"]["tavily"] = tavily_config

            # Perplexity if enabled
            if features["perplexity"].get("enabled", True):
                perplexity_config = features["perplexity"]
                tasks.append(self._perplexity_search(client, topic, context, perplexity_config))
                result["sdks_used"]["perplexity"] = perplexity_config

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for r in responses:
                if isinstance(r, dict):
                    result["sources"].extend(r.get("sources", []))
                    result["findings"].extend(r.get("findings", []))

            # Embed with Jina using context-specific task
            if result["sources"]:
                texts = [s.get("text", "")[:500] for s in result["sources"] if s.get("text")]
                if texts:
                    jina_config = features["jina"]
                    result["vectors"] = await self._embed(client, texts, jina_config)
                    result["sdks_used"]["jina"] = jina_config

        return result

    async def _exa_search(self, topic: str, config: dict) -> dict:
        try:
            self.sdk_usage["exa"] += 1
            search = self.exa.search_and_contents(
                topic,
                type=config.get("type", "auto"),
                num_results=config.get("num_results", 5),
                text=True,
                highlights=config.get("highlights", True)
            )

            sources = [{
                "title": r.title, "url": r.url,
                "text": r.text[:500] if r.text else "",
                "sdk": "exa",
                "search_type": config.get("type", "auto")
            } for r in search.results]

            findings = [f"[exa-{config.get('type', 'auto')}] {r.title}" for r in search.results[:2]]

            # Add highlights for technical docs
            if config.get("highlights") and search.results:
                for r in search.results[:2]:
                    if hasattr(r, 'highlights') and r.highlights:
                        findings.append(f"[exa-highlight] {r.highlights[0][:150]}")

            return {"sources": sources, "findings": findings}
        except Exception as e:
            return {"sources": [], "findings": []}

    async def _tavily_search(self, client: httpx.AsyncClient, topic: str, config: dict) -> dict:
        try:
            self.sdk_usage["tavily"] += 1
            payload = {
                'api_key': self.keys["tavily"],
                'query': topic,
                'search_depth': config.get("search_depth", "basic"),
                'include_answer': config.get("include_answer", True),
                'max_results': 5
            }

            if config.get("include_raw_content"):
                payload["include_raw_content"] = True

            r = await client.post('https://api.tavily.com/search', json=payload)
            data = r.json()

            sources = [{
                "title": s.get("title", ""),
                "url": s.get("url", ""),
                "text": s.get("content", "")[:500],
                "sdk": "tavily",
                "depth": config.get("search_depth", "basic")
            } for s in data.get("results", [])]

            findings = []
            if config.get("include_answer") and data.get("answer"):
                findings.append(f"[tavily-answer] {data['answer'][:200]}")

            return {"sources": sources, "findings": findings}
        except Exception as e:
            return {"sources": [], "findings": []}

    async def _perplexity_search(self, client: httpx.AsyncClient, topic: str, context: ResearchContext, config: dict) -> dict:
        try:
            self.sdk_usage["perplexity"] += 1

            # Build context-specific prompt
            focus = config.get("focus", "")
            deep = config.get("deep", False)

            if context == ResearchContext.COMPARISON:
                prompt = f"Compare and contrast: {topic}. Provide pros/cons and recommendations."
            elif context == ResearchContext.IMPLEMENTATION:
                prompt = f"Implementation guide for: {topic}. Include code examples and best practices."
            elif context == ResearchContext.TROUBLESHOOTING:
                prompt = f"Troubleshooting guide: {topic}. Include common issues and solutions."
            elif context == ResearchContext.ARCHITECTURE:
                prompt = f"Architecture deep dive: {topic}. Include design patterns and trade-offs."
            else:
                prompt = f"Technical analysis: {topic}"

            if deep:
                prompt += " Provide comprehensive analysis with citations."

            r = await client.post(
                'https://api.perplexity.ai/chat/completions',
                headers={'Authorization': f'Bearer {self.keys["perplexity"]}', 'Content-Type': 'application/json'},
                json={'model': 'sonar-pro', 'messages': [{'role': 'user', 'content': prompt}], 'return_citations': True}
            )
            data = r.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            citations = data.get('citations', [])

            sources = [{"title": f"Citation {i+1}", "url": c, "text": "", "sdk": "perplexity"} for i, c in enumerate(citations[:3])]
            findings = [f"[perplexity-{context.value}] {content[:250]}"] if content else []

            return {"sources": sources, "findings": findings}
        except Exception as e:
            return {"sources": [], "findings": []}

    async def _embed(self, client: httpx.AsyncClient, texts: list, config: dict) -> int:
        try:
            self.sdk_usage["jina"] += 1
            from qdrant_client.models import PointStruct

            r = await client.post(
                'https://api.jina.ai/v1/embeddings',
                headers={'Authorization': f'Bearer {self.keys["jina"]}', 'Content-Type': 'application/json'},
                json={
                    'model': 'jina-embeddings-v3',
                    'input': texts[:10],
                    'task': config.get("task", "retrieval.passage")
                }
            )
            embeddings = [d.get('embedding', []) for d in r.json().get('data', [])]

            if not embeddings or not embeddings[0]:
                return 0

            base_id = int(time.time() * 1000) % 1000000
            points = [PointStruct(id=base_id + i, vector=emb, payload={"text": texts[i][:200]}) for i, emb in enumerate(embeddings)]
            self.qdrant.upsert("deep_research", points=points)

            return len(points)
        except Exception as e:
            return 0

    def resolve_gap(self, gap_name: str) -> Optional[dict]:
        """Get gap resolution implementation."""
        if gap_name and gap_name in GAP_RESOLUTIONS:
            self.stats["gaps_resolved"] += 1
            return GAP_RESOLUTIONS[gap_name]
        return None

    async def run_iteration(self, topic_data: dict, index: int) -> DeepResearchResult:
        topic = topic_data["topic"]
        context = topic_data["context"]
        gap = topic_data.get("gap")

        print(f"\n[{index:02d}] [{context.value}] {topic[:50]}...")

        start = time.time()
        result = await self.research(topic, context)

        # Resolve gap if present
        gap_resolution = None
        if gap:
            gap_resolution = self.resolve_gap(gap)
            if gap_resolution:
                result["findings"].append(f"[gap-resolved] {gap}: {gap_resolution['pattern']}")

        latency = time.time() - start

        self.stats["sources"] += len(result["sources"])
        self.stats["vectors"] += result["vectors"]
        self.stats["insights"] += len(result["findings"])

        # Show SDK usage for this query
        sdks = list(result["sdks_used"].keys())
        print(f"    SDKs: {sdks}")
        print(f"    Src:{len(result['sources'])} Vec:{result['vectors']} Find:{len(result['findings'])} [{latency:.1f}s]")

        if result["findings"]:
            f = result["findings"][0]
            clean = f.encode('ascii', 'replace').decode('ascii')
            print(f"    -> {clean[:60]}...")

        return DeepResearchResult(
            topic=topic,
            context=context,
            sources=result["sources"],
            findings=result["findings"],
            vectors=result["vectors"],
            latency=latency,
            sdks_used=result["sdks_used"],
            gap_resolution=gap_resolution
        )


async def main():
    print("="*70)
    print("DEEP RESEARCH ITERATIONS - CONTEXT-AWARE SDK FEATURES")
    print("="*70)
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Topics: {len(DEEP_RESEARCH_TOPICS)}")
    print(f"Gap Resolutions Available: {len(GAP_RESOLUTIONS)}")
    print("="*70)

    executor = DeepResearchExecutor()
    await executor.initialize()

    all_results = []

    # Group by context for better organization
    by_context = {}
    for t in DEEP_RESEARCH_TOPICS:
        ctx = t["context"].value
        by_context.setdefault(ctx, []).append(t)

    iteration = 0
    for context_name, topics in by_context.items():
        print(f"\n{'='*70}")
        print(f"CONTEXT: {context_name.upper()}")
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
    print("DEEP RESEARCH COMPLETE")
    print("="*70)

    print(f"\n  Iterations: {len(all_results)}")
    print(f"  Total Sources: {executor.stats['sources']}")
    print(f"  Total Vectors: {executor.stats['vectors']}")
    print(f"  Total Insights: {executor.stats['insights']}")
    print(f"  Gaps Resolved: {executor.stats['gaps_resolved']}")

    avg_latency = sum(r.latency for r in all_results) / len(all_results) if all_results else 0
    print(f"  Avg Latency: {avg_latency:.1f}s")

    print("\n  SDK USAGE:")
    for sdk, count in sorted(executor.sdk_usage.items(), key=lambda x: -x[1]):
        print(f"    {sdk}: {count} calls")

    print("\n  BY CONTEXT:")
    ctx_stats = {}
    for r in all_results:
        ctx_stats[r.context.value] = ctx_stats.get(r.context.value, 0) + 1
    for ctx, count in sorted(ctx_stats.items()):
        print(f"    {ctx}: {count}")

    print("\n  GAP RESOLUTIONS:")
    gaps = [r for r in all_results if r.gap_resolution]
    for r in gaps:
        print(f"    - {r.gap_resolution['pattern']}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "stats": executor.stats,
        "sdk_usage": executor.sdk_usage,
        "gap_resolutions": [
            {"gap": r.topic, "pattern": r.gap_resolution["pattern"], "config": r.gap_resolution.get("config", {})}
            for r in gaps
        ],
        "results": [
            {
                "topic": r.topic,
                "context": r.context.value,
                "sources": len(r.sources),
                "vectors": r.vectors,
                "findings": r.findings[:3],
                "latency": r.latency,
                "sdks_used": list(r.sdks_used.keys())
            }
            for r in all_results
        ]
    }

    with open("deep_research_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Results saved to deep_research_results.json")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
