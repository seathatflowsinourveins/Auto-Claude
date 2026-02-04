"""
Unified RAG Pipeline: Orchestrates All RAG Components

This module provides a unified pipeline that integrates all RAG components:
- Multiple retrieval strategies (Self-RAG, CRAG, HyDE, RAPTOR, Agentic, ColBERT, GraphRAG)
- Multiple retrievers (Exa, Tavily, Memory, GraphRAG, ColBERT)
- Reranking with RRF fusion and ColBERT late interaction
- Context management with token budgets
- Entity-relationship retrieval via GraphRAG for multi-hop reasoning
- Comprehensive metrics and logging

Usage:
    from core.rag.pipeline import RAGPipeline, PipelineConfig, QueryType

    pipeline = RAGPipeline(
        retrievers=[exa, tavily, memory, colbert_retriever],
        reranker=cross_encoder,  # Or use ColBERTReranker for late interaction
        strategies={QueryType.RESEARCH: "agentic", QueryType.FACTUAL: "self_rag"}
    )
    result = await pipeline.run("Complex research question")

    # ColBERT-specific usage:
    from core.rag.colbert_retriever import ColBERTRetriever, ColBERTReranker

    # As retriever
    colbert = ColBERTRetriever.from_index(".ragatouille/indexes/my_index")
    pipeline.add_retriever(colbert)

    # As reranker (late interaction scoring)
    colbert_reranker = ColBERTReranker()
    pipeline = RAGPipeline(llm=my_llm, retrievers=[...], reranker=colbert_reranker)

    # GraphRAG for entity-relationship queries:
    from core.rag.graph_rag import GraphRAG, GraphRAGConfig

    # Initialize GraphRAG
    graph_config = GraphRAGConfig(db_path="knowledge_graph.db")
    graph_rag = GraphRAG(llm=my_llm, embedder=my_embedder, config=graph_config)

    # Create pipeline with GraphRAG enabled
    pipeline = RAGPipeline(
        llm=my_llm,
        retrievers=[exa, tavily],
        graph_rag=graph_rag,
        config=PipelineConfig(enable_graph_rag=True, graph_weight=1.5),
        strategies={QueryType.MULTI_HOP: "graph_rag", QueryType.FACTUAL: "graph_rag"}
    )

    # Ingest documents to build knowledge graph
    await pipeline.ingest_to_graph("Alice is CEO of TechCorp in San Francisco.")

    # Query with entity-relationship context
    result = await pipeline.run("Who leads TechCorp?")
"""

from __future__ import annotations

import asyncio
import logging
import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Protocol, Tuple, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .cache_warmer import CacheWarmer, CacheWarmerConfig, WarmingStats
    from .result_cache import ResultCache, ResultCacheConfig, CacheHit, CacheStats
    from .streaming import (
        StreamingConfig,
        StreamEvent,
        CancellationToken,
        StreamingRAGResponse,
    )
    from .colbert_retriever import (
        ColBERTRetriever,
        ColBERTReranker,
        ColBERTConfig,
        ColBERTDocument,
    )
    from .graph_rag import (
        GraphRAG,
        GraphRAGConfig,
        GraphRAGTool,
        GraphSearchResult,
    )
    from .hallucination_guard import (
        HallucinationGuard,
        GuardConfig,
        HallucinationResult,
    )

# Import query rewriter components
from .query_rewriter import (
    HybridQueryRewriter,
    RuleBasedQueryRewriter,
    QueryRewriterConfig,
    RewriteResult,
    QueryIntent,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PROTOCOLS
# =============================================================================

class LLMProvider(Protocol):
    """Protocol for LLM providers."""
    async def generate(self, prompt: str, max_tokens: int = 2048,
                       temperature: float = 0.7, **kwargs) -> str: ...


class RetrieverProtocol(Protocol):
    """Protocol for retrievers."""
    @property
    def name(self) -> str: ...
    async def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]: ...


class RerankerProtocol(Protocol):
    """Protocol for rerankers."""
    async def rerank(self, query: str, documents: List[Any],
                     top_k: int = 10) -> List[Any]: ...


class EvaluatorProtocol(Protocol):
    """Protocol for evaluators."""
    async def evaluate_single(self, question: str, contexts: List[str],
                              answer: str, ground_truth: Optional[str] = None) -> Any: ...


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

class QueryType(str, Enum):
    """Query types for strategy selection."""
    FACTUAL = "factual"
    RESEARCH = "research"
    CODE = "code"
    NEWS = "news"
    MULTI_HOP = "multi_hop"
    COMPARISON = "comparison"
    EXPLANATION = "explanation"
    GENERAL = "general"


class PipelineStage(str, Enum):
    """Stages in the RAG pipeline."""
    QUERY_REWRITE = "query_rewrite"
    ENTITY_EXTRACTION = "entity_extraction"
    RETRIEVE = "retrieve"
    GRAPH_EXPAND = "graph_expand"
    RERANK = "rerank"
    GENERATE = "generate"
    HALLUCINATION_CHECK = "hallucination_check"
    EVALUATE = "evaluate"


class StrategyType(str, Enum):
    """Available RAG strategies."""
    SELF_RAG = "self_rag"
    CRAG = "crag"
    HYDE = "hyde"
    RAPTOR = "raptor"
    AGENTIC = "agentic"
    COLBERT = "colbert"
    GRAPH_RAG = "graph_rag"
    BASIC = "basic"


@dataclass
class PipelineConfig:
    """Configuration for the RAG pipeline.

    Attributes:
        max_token_budget: Maximum tokens for context (default: 4000)
        top_k_retrieve: Documents to retrieve per source (default: 10)
        top_k_final: Final documents after reranking (default: 5)
        enable_query_rewrite: Enable query rewriting (default: True)
        enable_reranking: Enable reranking stage (default: True)
        enable_evaluation: Enable evaluation stage (default: True)
        rrf_k: RRF fusion constant (default: 60)
        compression_threshold: Token count to trigger compression (default: 3000)
        timeout_seconds: Overall pipeline timeout (default: 60)
        temperature: Generation temperature (default: 0.7)
        max_tokens: Max tokens for generation (default: 2048)
        confidence_threshold: Min confidence for early stopping (default: 0.8)
        default_strategy: Default RAG strategy (default: BASIC)
        batch_qps: Queries per second limit for batch operations (default: 10.0)
        batch_max_concurrency: Maximum concurrent queries in batch (default: 10)
        batch_similarity_threshold: Threshold for query deduplication (default: 0.85)
        batch_share_retrieval: Share retrieval results for similar queries (default: True)
        enable_graph_rag: Enable GraphRAG for entity-relationship queries (default: False)
        graph_expand_hops: Number of hops for graph neighborhood expansion (default: 2)
        graph_weight: Weight for graph results in RRF fusion (default: 1.5)
        enable_entity_extraction: Extract entities from query for graph lookup (default: True)
    """
    max_token_budget: int = 4000
    top_k_retrieve: int = 10
    top_k_final: int = 5
    enable_query_rewrite: bool = True
    enable_reranking: bool = True
    enable_evaluation: bool = True
    rrf_k: int = 60
    compression_threshold: int = 3000
    timeout_seconds: float = 60.0
    temperature: float = 0.7
    max_tokens: int = 2048
    confidence_threshold: float = 0.8
    default_strategy: StrategyType = StrategyType.BASIC
    # Batch operation settings
    batch_qps: float = 10.0
    batch_max_concurrency: int = 10
    batch_similarity_threshold: float = 0.85
    batch_share_retrieval: bool = True
    # GraphRAG settings
    enable_graph_rag: bool = False
    graph_expand_hops: int = 2
    graph_weight: float = 1.5
    enable_entity_extraction: bool = True
    # Hallucination Guard settings
    enable_hallucination_guard: bool = False
    hallucination_confidence_threshold: float = 0.7
    hallucination_strict_mode: bool = False
    # Adaptive top-k settings (adjusts retrieval depth based on query complexity)
    enable_adaptive_top_k: bool = True
    adaptive_top_k_low: int = 5       # Simple queries: fewer docs to avoid noise
    adaptive_top_k_medium: int = 10   # Standard queries
    adaptive_top_k_high: int = 15     # Complex queries: more context needed
    adaptive_top_k_very_high: int = 20  # Multi-hop/analytical: maximum context


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RetrievedDocument:
    """A retrieved document with metadata."""
    content: str
    score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_estimate: int = 0

    def __post_init__(self):
        if self.token_estimate == 0:
            self.token_estimate = len(self.content.split()) * 1.3  # Rough estimate


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""
    stage: PipelineStage
    latency_ms: float
    input_count: int = 0
    output_count: int = 0
    decision: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result from RAG pipeline execution.

    Attributes:
        response: Generated response
        confidence: Confidence score (0-1)
        contexts_used: Contexts used for generation
        documents_retrieved: Total documents retrieved
        strategy_used: RAG strategy that was used
        stage_metrics: Metrics per pipeline stage
        total_latency_ms: Total execution time
        evaluation: Optional evaluation results
        hallucination_check: Optional hallucination guard results
        metadata: Additional metadata
    """
    response: str
    confidence: float = 0.0
    contexts_used: List[str] = field(default_factory=list)
    documents_retrieved: int = 0
    strategy_used: StrategyType = StrategyType.BASIC
    stage_metrics: List[StageMetrics] = field(default_factory=list)
    total_latency_ms: float = 0.0
    evaluation: Optional[Dict[str, Any]] = None
    hallucination_check: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result from batch RAG pipeline execution.

    Attributes:
        results: List of individual pipeline results
        total_queries: Total number of queries processed
        unique_queries: Number of unique queries after deduplication
        shared_retrievals: Number of queries that shared retrieval results
        total_latency_ms: Total batch execution time
        throughput_qps: Actual queries per second achieved
        deduplication_stats: Statistics about query deduplication
    """
    results: List[PipelineResult] = field(default_factory=list)
    total_queries: int = 0
    unique_queries: int = 0
    shared_retrievals: int = 0
    total_latency_ms: float = 0.0
    throughput_qps: float = 0.0
    deduplication_stats: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# QUERY CLASSIFIER
# =============================================================================

class QueryClassifier:
    """Classifies queries to determine optimal strategy."""

    def __init__(self, llm: Optional[LLMProvider] = None):
        self.llm = llm
        self._cache: Dict[str, QueryType] = {}

    async def classify(self, query: str) -> QueryType:
        """Classify query type."""
        cache_key = hashlib.md5(query.encode()).hexdigest()[:12]
        if cache_key in self._cache:
            return self._cache[cache_key]

        query_type = self._rule_based_classify(query)

        if self.llm and query_type == QueryType.GENERAL:
            query_type = await self._llm_classify(query)

        self._cache[cache_key] = query_type
        return query_type

    def _rule_based_classify(self, query: str) -> QueryType:
        """Rule-based query classification."""
        q = query.lower()

        if any(w in q for w in ['code', 'function', 'implement', 'syntax', 'api', 'bug']):
            return QueryType.CODE
        if any(w in q for w in ['latest', 'recent', 'news', 'today', 'current', '2024', '2025']):
            return QueryType.NEWS
        if any(w in q for w in ['compare', 'difference', 'versus', 'vs', 'better than']):
            return QueryType.COMPARISON
        if any(w in q for w in ['explain', 'why', 'how does', 'what causes']):
            return QueryType.EXPLANATION
        if any(w in q for w in ['research', 'study', 'analysis', 'investigate', 'deep dive']):
            return QueryType.RESEARCH
        if any(w in q for w in ['who', 'what is', 'when', 'where', 'how many']):
            return QueryType.FACTUAL
        if ' and ' in q and ('?' in q or any(w in q for w in ['also', 'then', 'after'])):
            return QueryType.MULTI_HOP

        return QueryType.GENERAL

    async def _llm_classify(self, query: str) -> QueryType:
        """LLM-based query classification."""
        prompt = f"""Classify this query into one category:
- factual: Simple fact lookup
- research: Deep research needed
- code: Code/programming related
- news: Current events
- multi_hop: Requires multiple reasoning steps
- comparison: Compare/contrast items
- explanation: Explain a concept
- general: General knowledge

Query: {query}

Output only the category name:"""

        try:
            response = await self.llm.generate(prompt, max_tokens=20, temperature=0)
            category = response.strip().lower()
            return QueryType(category) if category in [t.value for t in QueryType] else QueryType.GENERAL
        except Exception:
            return QueryType.GENERAL


# =============================================================================
# QUERY REWRITER
# =============================================================================

class QueryRewriter:
    """
    Advanced query rewriter using HybridQueryRewriter for multi-query expansion.

    Features:
    - Rule-based fast path for simple queries (no LLM calls)
    - LLM-enhanced rewriting for complex queries
    - Synonym and acronym expansion
    - Step-back prompting for broader context
    - Query decomposition for complex multi-part questions
    - Semantic optimization for vector search
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: Optional[QueryRewriterConfig] = None,
        prefer_llm: bool = False,
    ):
        self.llm = llm
        self.config = config or QueryRewriterConfig(
            num_expansions=4,
            enable_synonym_expansion=True,
            enable_step_back=True,
            enable_decomposition=True,
            max_sub_queries=5,
            extract_keywords=True,
        )
        self._hybrid_rewriter = HybridQueryRewriter(
            llm=llm,
            config=self.config,
            prefer_llm=prefer_llm,
        )
        self._rule_based = RuleBasedQueryRewriter(self.config)

    async def rewrite(self, query: str, query_type: QueryType) -> List[str]:
        """
        Rewrite query into multiple variants for better recall using HybridQueryRewriter.

        Returns expanded queries including:
        - Original query
        - Expanded variants with synonym/acronym expansion
        - Semantic-optimized version
        - Step-back query (for broader context)
        - Sub-queries (for complex decomposed questions)

        Args:
            query: The original user query
            query_type: Classification of the query type

        Returns:
            List of query variants for multi-query retrieval
        """
        try:
            result: RewriteResult = await self._hybrid_rewriter.rewrite(query)
            return self._build_query_list(result, query_type)
        except Exception as e:
            logger.warning(f"Hybrid query rewrite failed: {e}, falling back to rule-based")
            try:
                result = await self._rule_based.rewrite(query)
                return self._build_query_list(result, query_type)
            except Exception as e2:
                logger.warning(f"Rule-based rewrite also failed: {e2}")
                return [query]

    async def rewrite_full(self, query: str) -> RewriteResult:
        """
        Get full rewrite result with all metadata.

        Returns the complete RewriteResult including intent, keywords,
        entities, constraints, and sub-queries for advanced pipeline usage.
        """
        try:
            return await self._hybrid_rewriter.rewrite(query)
        except Exception as e:
            logger.warning(f"Full rewrite failed: {e}")
            return await self._rule_based.rewrite(query)

    def _build_query_list(self, result: RewriteResult, query_type: QueryType) -> List[str]:
        """Build deduplicated list of queries from rewrite result."""
        queries = [result.original_query]

        # Add expanded queries
        for q in result.expanded_queries:
            if q and q not in queries:
                queries.append(q)

        # Add semantic query if different
        if result.semantic_query and result.semantic_query not in queries:
            queries.append(result.semantic_query)

        # Add step-back query for broader context retrieval
        if result.step_back_query and result.step_back_query not in queries:
            queries.append(result.step_back_query)

        # For complex queries, add sub-queries
        if result.is_complex and result.sub_queries:
            for sub_q in result.sub_queries[:3]:  # Limit sub-queries
                if sub_q.query and sub_q.query not in queries:
                    queries.append(sub_q.query)

        # Limit total queries to avoid excessive API calls
        return queries[:8]


# =============================================================================
# CONTEXT MANAGER
# =============================================================================

class ContextManager:
    """Manages context within token budget."""

    def __init__(self, max_tokens: int = 4000, compression_threshold: int = 3000):
        self.max_tokens = max_tokens
        self.compression_threshold = compression_threshold

    def select_contexts(
        self,
        documents: List[RetrievedDocument],
        token_budget: Optional[int] = None
    ) -> List[RetrievedDocument]:
        """Select documents within token budget by priority."""
        budget = token_budget or self.max_tokens
        selected: List[RetrievedDocument] = []
        current_tokens = 0

        sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)

        for doc in sorted_docs:
            if current_tokens + doc.token_estimate <= budget:
                selected.append(doc)
                current_tokens += doc.token_estimate

        return selected

    async def compress_context(
        self,
        contexts: List[str],
        llm: LLMProvider,
        query: str
    ) -> str:
        """Compress contexts if they exceed threshold."""
        combined = "\n\n".join(contexts)
        token_est = len(combined.split()) * 1.3

        if token_est <= self.compression_threshold:
            return combined

        prompt = f"""Compress the following context while retaining all information relevant to the query.
Remove redundancy but keep key facts.

Query: {query}

Context:
{combined[:6000]}

Compressed context:"""

        try:
            compressed = await llm.generate(prompt, max_tokens=1500, temperature=0.3)
            return compressed.strip()
        except Exception as e:
            logger.warning(f"Compression failed: {e}, using truncated context")
            return combined[:4000]

    def deduplicate(self, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Remove duplicate documents by content similarity."""
        seen: set = set()
        unique: List[RetrievedDocument] = []

        for doc in documents:
            key = doc.content[:200].lower()
            if key not in seen:
                seen.add(key)
                unique.append(doc)

        return unique


# =============================================================================
# RRF FUSION
# =============================================================================

class RRFFusion:
    """Reciprocal Rank Fusion for combining retrieval results."""

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        result_lists: List[List[RetrievedDocument]],
        weights: Optional[List[float]] = None
    ) -> List[RetrievedDocument]:
        """Fuse multiple ranked lists using RRF."""
        if not result_lists:
            return []

        if weights is None:
            weights = [1.0] * len(result_lists)

        scores: Dict[str, Tuple[RetrievedDocument, float]] = {}

        for results, weight in zip(result_lists, weights):
            for rank, doc in enumerate(results, start=1):
                content_key = doc.content[:200]
                rrf_score = weight / (self.k + rank)

                if content_key in scores:
                    existing_doc, existing_score = scores[content_key]
                    scores[content_key] = (existing_doc, existing_score + rrf_score)
                else:
                    scores[content_key] = (doc, rrf_score)

        sorted_items = sorted(scores.values(), key=lambda x: x[1], reverse=True)
        return [RetrievedDocument(
            content=doc.content,
            score=score,
            source=doc.source,
            metadata=doc.metadata,
            token_estimate=doc.token_estimate
        ) for doc, score in sorted_items]


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """Token bucket rate limiter for controlling query throughput.

    Implements a token bucket algorithm that allows burst capacity while
    maintaining an average rate limit over time.
    """

    def __init__(self, qps: float = 10.0, burst_multiplier: float = 2.0):
        """Initialize the rate limiter.

        Args:
            qps: Queries per second limit
            burst_multiplier: Allow burst up to qps * burst_multiplier
        """
        self.qps = qps
        self.burst_capacity = qps * burst_multiplier
        self.tokens = self.burst_capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Wait time in seconds (0 if no wait was needed)
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.burst_capacity, self.tokens + elapsed * self.qps)
            self.last_refill = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            wait_time = (tokens - self.tokens) / self.qps
            await asyncio.sleep(wait_time)
            self.tokens = 0
            self.last_refill = time.time()
            return wait_time

    async def acquire_batch(self, count: int) -> float:
        """Acquire tokens for a batch of operations.

        Args:
            count: Number of operations in the batch

        Returns:
            Total wait time in seconds
        """
        total_wait = 0.0
        for _ in range(count):
            wait = await self.acquire(1)
            total_wait += wait
        return total_wait


# =============================================================================
# QUERY DEDUPLICATOR
# =============================================================================

class QueryDeduplicator:
    """Deduplicates similar queries for batch processing.

    Uses n-gram similarity to identify semantically similar queries
    and group them together to share retrieval results.
    """

    def __init__(self, similarity_threshold: float = 0.85, ngram_size: int = 3):
        """Initialize the deduplicator.

        Args:
            similarity_threshold: Minimum similarity to consider duplicates (0-1)
            ngram_size: Size of n-grams for similarity computation
        """
        self.similarity_threshold = similarity_threshold
        self.ngram_size = ngram_size

    def _get_ngrams(self, text: str) -> set:
        """Extract n-grams from text."""
        text = text.lower().strip()
        if len(text) < self.ngram_size:
            return {text}
        return {text[i:i + self.ngram_size] for i in range(len(text) - self.ngram_size + 1)}

    def _compute_similarity(self, query1: str, query2: str) -> float:
        """Compute Jaccard similarity between two queries using n-grams."""
        ngrams1 = self._get_ngrams(query1)
        ngrams2 = self._get_ngrams(query2)

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        return intersection / union if union > 0 else 0.0

    def deduplicate(
        self,
        queries: List[str]
    ) -> Tuple[List[str], Dict[str, List[int]], Dict[str, Any]]:
        """Deduplicate queries and return mapping.

        Args:
            queries: List of queries to deduplicate

        Returns:
            Tuple of:
                - List of unique/canonical queries
                - Mapping from canonical query to list of original indices
                - Statistics about deduplication
        """
        if not queries:
            return [], {}, {"total": 0, "unique": 0, "duplicates_found": 0}

        canonical_queries: List[str] = []
        query_to_indices: Dict[str, List[int]] = {}

        for idx, query in enumerate(queries):
            found_match = False

            for canonical in canonical_queries:
                similarity = self._compute_similarity(query, canonical)
                if similarity >= self.similarity_threshold:
                    query_to_indices[canonical].append(idx)
                    found_match = True
                    break

            if not found_match:
                canonical_queries.append(query)
                query_to_indices[query] = [idx]

        stats = {
            "total": len(queries),
            "unique": len(canonical_queries),
            "duplicates_found": len(queries) - len(canonical_queries),
            "deduplication_ratio": 1 - (len(canonical_queries) / len(queries)) if queries else 0
        }

        return canonical_queries, query_to_indices, stats

    def find_similar_groups(
        self,
        queries: List[str]
    ) -> List[List[int]]:
        """Group queries by similarity for shared retrieval.

        Args:
            queries: List of queries

        Returns:
            List of groups, where each group is a list of query indices
        """
        n = len(queries)
        if n == 0:
            return []
        if n == 1:
            return [[0]]

        # Build similarity matrix (sparse, only above threshold)
        groups: List[List[int]] = []
        assigned = set()

        for i in range(n):
            if i in assigned:
                continue

            group = [i]
            assigned.add(i)

            for j in range(i + 1, n):
                if j in assigned:
                    continue

                if self._compute_similarity(queries[i], queries[j]) >= self.similarity_threshold:
                    group.append(j)
                    assigned.add(j)

            groups.append(group)

        return groups


# =============================================================================
# UNIFIED RAG PIPELINE
# =============================================================================

class RAGPipeline:
    """
    Unified RAG Pipeline that orchestrates all components.

    Stages:
        Query -> Rewrite -> Retrieve -> Rerank -> Generate -> Evaluate

    Example:
        >>> pipeline = RAGPipeline(
        ...     llm=my_llm,
        ...     retrievers=[exa, tavily, memory],
        ...     reranker=cross_encoder,
        ...     strategies={QueryType.RESEARCH: "agentic"}
        ... )
        >>> result = await pipeline.run("Complex research question")
        >>> print(result.response)
    """

    def __init__(
        self,
        llm: LLMProvider,
        retrievers: Optional[List[RetrieverProtocol]] = None,
        reranker: Optional[RerankerProtocol] = None,
        evaluator: Optional[EvaluatorProtocol] = None,
        strategies: Optional[Dict[QueryType, str]] = None,
        config: Optional[PipelineConfig] = None,
        rag_implementations: Optional[Dict[str, Any]] = None,
        graph_rag: Optional[Any] = None,
        colbert_reranker: Optional[Any] = None,
        enable_colbert_reranking: bool = False,
    ):
        """Initialize the RAG pipeline.

        Args:
            llm: LLM provider for generation and reasoning
            retrievers: List of retriever implementations (can include ColBERTRetriever)
            reranker: Optional reranker for result fusion (cross-encoder or SemanticReranker)
            evaluator: Optional evaluator for quality metrics
            strategies: Mapping of query types to strategy names
            config: Pipeline configuration
            rag_implementations: Dict of strategy name to implementation
            graph_rag: Optional GraphRAG instance for entity-relationship retrieval
            colbert_reranker: Optional ColBERTReranker for late-interaction reranking
            enable_colbert_reranking: Enable ColBERT reranking stage after RRF fusion
        """
        self.llm = llm
        self.retrievers = retrievers or []
        self.reranker = reranker
        self.evaluator = evaluator
        self.strategies = strategies or {}
        self.config = config or PipelineConfig()
        self.rag_implementations = rag_implementations or {}

        # ColBERT integration for late-interaction reranking
        self._colbert_reranker = colbert_reranker
        self._enable_colbert_reranking = enable_colbert_reranking

        self.classifier = QueryClassifier(llm)
        # Initialize advanced query rewriter with HybridQueryRewriter
        self.rewriter = QueryRewriter(
            llm=llm,
            config=QueryRewriterConfig(
                num_expansions=self.config.top_k_retrieve // 2,  # Scale with retrieve count
                enable_synonym_expansion=True,
                enable_step_back=True,
                enable_decomposition=True,
                max_sub_queries=5,
                extract_keywords=True,
            ),
            prefer_llm=False,  # Use rule-based first, LLM for complex queries
        )
        self.context_manager = ContextManager(
            self.config.max_token_budget,
            self.config.compression_threshold
        )
        self.fusion = RRFFusion(self.config.rrf_k)

        # GraphRAG support
        self._graph_rag: Optional[Any] = graph_rag
        self._graph_rag_tool: Optional[Any] = None
        if graph_rag is not None:
            self._init_graph_rag(graph_rag)

        # Cache warming support
        self._cache_warmer: Optional["CacheWarmer"] = None
        self._cache_warming_enabled: bool = False

        # Result caching support
        self._result_cache: Optional["ResultCache"] = None
        self._result_cache_enabled: bool = False

        # Hallucination guard support
        self._hallucination_guard: Optional["HallucinationGuard"] = None
        self._hallucination_guard_enabled: bool = self.config.enable_hallucination_guard

        # Adaptive top-k complexity analyzer
        self._complexity_analyzer: Optional[Any] = None
        if self.config.enable_adaptive_top_k:
            self._init_complexity_analyzer()

    def _init_complexity_analyzer(self) -> None:
        """Initialize the complexity analyzer for adaptive top-k retrieval."""
        try:
            from .complexity_analyzer import (
                QueryComplexityAnalyzer,
                AnalyzerConfig,
                AdaptiveTopKConfig,
            )
            adaptive_config = AdaptiveTopKConfig(
                low_k=self.config.adaptive_top_k_low,
                medium_k=self.config.adaptive_top_k_medium,
                high_k=self.config.adaptive_top_k_high,
                very_high_k=self.config.adaptive_top_k_very_high,
                enabled=self.config.enable_adaptive_top_k,
            )
            analyzer_config = AnalyzerConfig(adaptive_top_k=adaptive_config)
            self._complexity_analyzer = QueryComplexityAnalyzer(config=analyzer_config)
            logger.info(
                f"Adaptive top-k: LOW={adaptive_config.low_k}, MEDIUM={adaptive_config.medium_k}, "
                f"HIGH={adaptive_config.high_k}, VERY_HIGH={adaptive_config.very_high_k}"
            )
        except ImportError:
            logger.warning("Complexity analyzer unavailable, using fixed top_k_retrieve")
            self._complexity_analyzer = None

    def get_adaptive_top_k(self, query: str) -> int:
        """Get adaptive top-k value based on query complexity."""
        if not self.config.enable_adaptive_top_k or self._complexity_analyzer is None:
            return self.config.top_k_retrieve
        try:
            return self._complexity_analyzer.get_adaptive_top_k(query)
        except Exception as e:
            logger.warning(f"Adaptive top-k failed: {e}")
            return self.config.top_k_retrieve

    def _init_graph_rag(self, graph_rag: Any) -> None:
        """Initialize GraphRAG as a retriever tool.

        Args:
            graph_rag: GraphRAG instance to wrap as retriever
        """
        try:
            from .graph_rag import GraphRAGTool
            self._graph_rag_tool = GraphRAGTool(graph_rag)
            # Register as strategy implementation
            self.rag_implementations["graph_rag"] = graph_rag
            logger.info("GraphRAG initialized and available as retriever")
        except ImportError:
            logger.warning("GraphRAG module not available")
            self._graph_rag = None
            self._graph_rag_tool = None

    def init_graph_rag(
        self,
        graph_rag: Optional[Any] = None,
        db_path: str = ":memory:",
        embedder: Optional[Any] = None,
    ) -> None:
        """Initialize or configure GraphRAG for entity-relationship retrieval.

        This method allows lazy initialization of GraphRAG after pipeline creation.
        It creates a GraphRAG instance and wraps it as a retrieval tool.

        Args:
            graph_rag: Optional pre-configured GraphRAG instance
            db_path: Path to SQLite database for graph storage (default: in-memory)
            embedder: Optional embedding provider for entity embeddings

        Example:
            >>> pipeline.init_graph_rag(db_path="knowledge_graph.db")
            >>> # Or with existing GraphRAG:
            >>> pipeline.init_graph_rag(graph_rag=my_graph_rag)
        """
        if graph_rag is not None:
            self._graph_rag = graph_rag
            self._init_graph_rag(graph_rag)
            return

        try:
            from .graph_rag import GraphRAG, GraphRAGConfig

            config = GraphRAGConfig(
                db_path=db_path,
                max_hops=self.config.graph_expand_hops,
                enable_embeddings=embedder is not None,
            )
            self._graph_rag = GraphRAG(
                llm=self.llm,
                embedder=embedder,
                config=config,
            )
            self._init_graph_rag(self._graph_rag)
            # Update config to enable GraphRAG
            self.config.enable_graph_rag = True
            logger.info(f"GraphRAG initialized with db_path={db_path}")
        except ImportError as e:
            logger.warning(f"Failed to initialize GraphRAG: {e}")

    async def ingest_to_graph(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        """Ingest text into the knowledge graph for entity extraction.

        This method extracts entities and relationships from text and stores
        them in the GraphRAG knowledge graph for later retrieval.

        Args:
            text: Text to process and extract entities from
            metadata: Optional metadata to attach to entities

        Returns:
            Dict with counts of entities and relationships added

        Raises:
            RuntimeError: If GraphRAG is not initialized

        Example:
            >>> stats = await pipeline.ingest_to_graph(
            ...     "Alice is the CEO of TechCorp. TechCorp is headquartered in San Francisco.",
            ...     metadata={"source": "company_docs"}
            ... )
            >>> print(f"Added {stats['entities']} entities, {stats['relationships']} relationships")
        """
        if self._graph_rag is None:
            raise RuntimeError("GraphRAG not initialized. Call init_graph_rag() first.")

        return await self._graph_rag.ingest(text, metadata)

    async def _retrieve_from_graph(
        self,
        query: str,
        top_k: int = 10
    ) -> List[RetrievedDocument]:
        """Retrieve documents using GraphRAG entity-relationship search.

        Args:
            query: Search query
            top_k: Maximum results to return

        Returns:
            List of RetrievedDocument with graph context
        """
        if self._graph_rag_tool is None:
            return []

        try:
            docs = await self._graph_rag_tool.retrieve(
                query=query,
                top_k=top_k,
                expand_hops=self.config.graph_expand_hops,
            )
            return [
                RetrievedDocument(
                    content=doc.get("content", ""),
                    score=doc.get("score", 0.8),
                    source="graph_rag",
                    metadata=doc.get("metadata", {}),
                )
                for doc in docs
            ]
        except Exception as e:
            logger.warning(f"GraphRAG retrieval error: {e}")
            return []

    def get_graph_stats(self) -> Optional[Dict[str, Any]]:
        """Get GraphRAG statistics.

        Returns:
            Dict with entity and relationship counts, or None if not initialized
        """
        if self._graph_rag is None:
            return None
        return self._graph_rag.get_stats()

    async def init_cache_warming(
        self,
        config: Optional["CacheWarmerConfig"] = None,
        custom_queries: Optional[List[str]] = None,
        warm_on_init: bool = True,
        start_background_refresh: bool = True,
    ) -> "WarmingStats":
        """Initialize cache warming for this pipeline.

        Pre-populates cache with frequently accessed queries and optionally
        starts background refresh to keep cache warm.

        Args:
            config: Optional cache warmer configuration
            custom_queries: Optional list of custom queries to pre-warm
            warm_on_init: Whether to warm cache immediately (default: True)
            start_background_refresh: Whether to start background refresh (default: True)

        Returns:
            WarmingStats with results of initial warming

        Example:
            >>> stats = await pipeline.init_cache_warming(
            ...     custom_queries=["What is RAG?", "How does vector search work?"],
            ...     warm_on_init=True,
            ...     start_background_refresh=True,
            ... )
            >>> print(f"Warmed {stats.successful} queries")
        """
        from .cache_warmer import CacheWarmer, CacheWarmerConfig, WarmingStats

        warmer_config = config or CacheWarmerConfig()
        self._cache_warmer = CacheWarmer(
            pipeline=self,
            retrievers=self.retrievers,
            config=warmer_config,
            custom_queries=custom_queries or [],
        )
        self._cache_warming_enabled = True

        stats = WarmingStats()

        if warm_on_init:
            stats = await self._cache_warmer.warm_startup()

        if start_background_refresh:
            self._cache_warmer.start_background_refresh()

        logger.info(
            f"Cache warming initialized: warmed={stats.successful}, "
            f"background_refresh={start_background_refresh}"
        )

        return stats

    def track_query_access(
        self,
        query: str,
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track a query access for pattern learning.

        This method records query access patterns that the cache warmer
        uses to identify frequently accessed content for pre-warming.

        Args:
            query: The query that was accessed
            latency_ms: Latency of the query execution in milliseconds
            metadata: Optional metadata about the access
        """
        if self._cache_warmer and self._cache_warming_enabled:
            self._cache_warmer.record_access(query, latency_ms, metadata)

    def stop_cache_warming(self) -> None:
        """Stop cache warming and background refresh."""
        if self._cache_warmer:
            self._cache_warmer.stop_background_refresh()
            self._cache_warming_enabled = False
            logger.info("Cache warming stopped")

    @property
    def cache_warming_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache warming statistics.

        Returns:
            Dictionary with cache warming metrics or None if not initialized
        """
        if self._cache_warmer:
            return self._cache_warmer.stats
        return None

    async def warm_queries(self, queries: List[str]) -> "WarmingStats":
        """Manually warm specific queries.

        Args:
            queries: List of queries to warm

        Returns:
            WarmingStats with results

        Raises:
            RuntimeError: If cache warming is not initialized
        """
        if not self._cache_warmer:
            raise RuntimeError("Cache warming not initialized. Call init_cache_warming() first.")

        return await self._cache_warmer.refresh_queries(queries)

    def init_result_cache(
        self,
        config: Optional["ResultCacheConfig"] = None,
        embedding_provider: Optional[Any] = None,
    ) -> None:
        """Initialize result caching for this pipeline.

        Enables two-level caching for pipeline results:
        - L1: Exact query match (hash-based, O(1) lookup)
        - L2: Semantic similarity match (embedding-based, >0.95 threshold)

        Args:
            config: Optional result cache configuration
            embedding_provider: Optional embedding provider for semantic cache

        Example:
            >>> from core.rag.result_cache import ResultCacheConfig
            >>> config = ResultCacheConfig(
            ...     max_entries=5000,
            ...     memory_budget_mb=256,
            ...     semantic_threshold=0.95,
            ... )
            >>> pipeline.init_result_cache(config=config, embedding_provider=my_embedder)
        """
        from .result_cache import ResultCache, ResultCacheConfig

        cache_config = config or ResultCacheConfig()
        self._result_cache = ResultCache(
            config=cache_config,
            embedding_provider=embedding_provider,
        )
        self._result_cache_enabled = True
        logger.info("Result caching initialized for pipeline")

    def disable_result_cache(self) -> None:
        """Disable result caching."""
        self._result_cache_enabled = False
        logger.info("Result caching disabled")

    def enable_result_cache(self) -> None:
        """Enable result caching (must be initialized first)."""
        if self._result_cache is None:
            raise RuntimeError("Result cache not initialized. Call init_result_cache() first.")
        self._result_cache_enabled = True
        logger.info("Result caching enabled")

    @property
    def result_cache_stats(self) -> Optional["CacheStats"]:
        """Get result cache statistics.

        Returns:
            CacheStats if cache is initialized, None otherwise
        """
        if self._result_cache:
            return self._result_cache.stats
        return None

    def invalidate_result_cache(
        self,
        pattern: Optional[str] = None,
        max_age_seconds: Optional[float] = None,
        query_type: Optional[str] = None,
    ) -> int:
        """Invalidate result cache entries.

        Args:
            pattern: Optional pattern with wildcards (e.g., "RAG*")
            max_age_seconds: Optional age threshold
            query_type: Optional query type to invalidate

        Returns:
            Number of entries invalidated
        """
        if not self._result_cache:
            return 0

        total = 0
        if pattern:
            total += self._result_cache.invalidate_by_pattern(pattern)
        if max_age_seconds:
            total += self._result_cache.invalidate_by_age(max_age_seconds)
        if query_type:
            total += self._result_cache.invalidate_by_query_type(query_type)

        # If no specific criteria, invalidate expired
        if not pattern and not max_age_seconds and not query_type:
            result = self._result_cache.invalidate_expired()
            total = result.get("l1", 0) + result.get("l2", 0)

        return total

    def init_hallucination_guard(
        self,
        guard: Optional["HallucinationGuard"] = None,
        judge_llm: Optional[LLMProvider] = None,
        config: Optional["GuardConfig"] = None,
    ) -> None:
        """Initialize hallucination guard for post-generation verification.

        The hallucination guard extracts claims from generated responses and
        verifies each claim against the retrieved context to detect hallucinated
        content in real-time.

        Args:
            guard: Pre-configured HallucinationGuard instance
            judge_llm: LLM provider for claim verification (defaults to pipeline LLM)
            config: Guard configuration options

        Example:
            >>> # With default LLM
            >>> pipeline.init_hallucination_guard()
            >>>
            >>> # With custom judge LLM
            >>> pipeline.init_hallucination_guard(judge_llm=gpt4)
            >>>
            >>> # With pre-configured guard
            >>> guard = HallucinationGuard(llm=judge_llm, config=config)
            >>> pipeline.init_hallucination_guard(guard=guard)
        """
        from .hallucination_guard import HallucinationGuard, GuardConfig

        if guard is not None:
            self._hallucination_guard = guard
        else:
            guard_config = config or GuardConfig(
                confidence_threshold=self.config.hallucination_confidence_threshold,
                strict_mode=self.config.hallucination_strict_mode,
            )
            llm = judge_llm or self.llm
            self._hallucination_guard = HallucinationGuard(llm=llm, config=guard_config)

        self._hallucination_guard_enabled = True
        logger.info("Hallucination guard initialized for pipeline")

    def enable_hallucination_guard(self) -> None:
        """Enable hallucination guard (must be initialized first)."""
        if self._hallucination_guard is None:
            # Auto-initialize with default config
            self.init_hallucination_guard()
        self._hallucination_guard_enabled = True
        logger.info("Hallucination guard enabled")

    def disable_hallucination_guard(self) -> None:
        """Disable hallucination guard."""
        self._hallucination_guard_enabled = False
        logger.info("Hallucination guard disabled")

    async def check_hallucination(
        self,
        response: str,
        contexts: List[str],
        question: Optional[str] = None,
    ) -> "HallucinationResult":
        """Check a response for hallucinations.

        This method can be called manually to verify any response against context,
        independent of the pipeline run.

        Args:
            response: Generated response to check
            contexts: Context chunks used for generation
            question: Optional original question for context

        Returns:
            HallucinationResult with detection details

        Raises:
            RuntimeError: If guard not initialized
        """
        if self._hallucination_guard is None:
            self.init_hallucination_guard()

        return await self._hallucination_guard.detect(response, contexts, question)

    @property
    def hallucination_guard_enabled(self) -> bool:
        """Check if hallucination guard is enabled."""
        return self._hallucination_guard_enabled

    async def run(
        self,
        query: str,
        query_type: Optional[QueryType] = None,
        strategy: Optional[StrategyType] = None,
        skip_cache: bool = False,
        **kwargs
    ) -> PipelineResult:
        """Execute the RAG pipeline.

        Args:
            query: User query
            query_type: Optional query type override
            strategy: Optional strategy override
            skip_cache: Skip result cache lookup (default: False)
            **kwargs: Additional arguments passed to stages

        Returns:
            PipelineResult with response and metrics
        """
        start_time = time.time()
        stage_metrics: List[StageMetrics] = []

        # Check result cache first (if enabled and not skipped)
        if self._result_cache_enabled and self._result_cache and not skip_cache:
            cache_hit = self._result_cache.get(query)
            if cache_hit is not None:
                cached_result = cache_hit.value
                # Add cache metadata to result
                if isinstance(cached_result, PipelineResult):
                    cached_result.metadata["cache_hit"] = True
                    cached_result.metadata["cache_level"] = cache_hit.cache_level.value
                    cached_result.metadata["cache_similarity"] = cache_hit.similarity_score
                    cached_result.metadata["cache_lookup_ms"] = cache_hit.lookup_time_ms
                    logger.debug(
                        f"Cache hit ({cache_hit.cache_level.value}): "
                        f"similarity={cache_hit.similarity_score:.3f}"
                    )
                    return cached_result

        try:
            # Stage 1: Classify query
            if query_type is None:
                query_type = await self.classifier.classify(query)
            logger.debug(f"Query classified as: {query_type.value}")

            # Determine strategy
            if strategy is None:
                strategy_name = self.strategies.get(query_type, self.config.default_strategy.value)
                strategy = StrategyType(strategy_name) if strategy_name in [s.value for s in StrategyType] else self.config.default_strategy

            # Check if we have a specialized implementation
            if strategy.value in self.rag_implementations:
                return await self._run_specialized_strategy(
                    query, strategy, stage_metrics, start_time, **kwargs
                )

            # Stage 2: Query Rewrite
            queries = [query]
            if self.config.enable_query_rewrite:
                stage_start = time.time()
                queries = await self.rewriter.rewrite(query, query_type)
                stage_metrics.append(StageMetrics(
                    stage=PipelineStage.QUERY_REWRITE,
                    latency_ms=(time.time() - stage_start) * 1000,
                    input_count=1,
                    output_count=len(queries),
                    details={"variants": queries}
                ))

            # Stage 3: Retrieve with adaptive top-k
            stage_start = time.time()
            # Calculate adaptive top-k based on query complexity
            adaptive_top_k = self.get_adaptive_top_k(query)
            all_documents = await self._retrieve_all(queries, top_k=adaptive_top_k)
            stage_metrics.append(StageMetrics(
                stage=PipelineStage.RETRIEVE,
                latency_ms=(time.time() - stage_start) * 1000,
                input_count=len(queries),
                output_count=len(all_documents),
                details={
                    "sources": list(set(d.source for d in all_documents)),
                    "adaptive_top_k": adaptive_top_k,
                    "adaptive_enabled": self.config.enable_adaptive_top_k,
                }
            ))

            if not all_documents:
                return PipelineResult(
                    response="I could not find relevant information to answer your question.",
                    confidence=0.0,
                    strategy_used=strategy,
                    stage_metrics=stage_metrics,
                    total_latency_ms=(time.time() - start_time) * 1000
                )

            # Stage 4: Rerank
            if self.config.enable_reranking and self.reranker:
                stage_start = time.time()
                all_documents = await self._rerank(query, all_documents)
                stage_metrics.append(StageMetrics(
                    stage=PipelineStage.RERANK,
                    latency_ms=(time.time() - stage_start) * 1000,
                    input_count=len(all_documents),
                    output_count=min(len(all_documents), self.config.top_k_final),
                    decision="reranked"
                ))

            # Select within token budget
            selected_docs = self.context_manager.select_contexts(
                all_documents[:self.config.top_k_final],
                self.config.max_token_budget
            )

            # Stage 5: Generate
            stage_start = time.time()
            contexts = [d.content for d in selected_docs]
            response, confidence = await self._generate(query, contexts, **kwargs)
            stage_metrics.append(StageMetrics(
                stage=PipelineStage.GENERATE,
                latency_ms=(time.time() - stage_start) * 1000,
                input_count=len(contexts),
                output_count=1,
                details={"confidence": confidence}
            ))

            # Stage 6: Hallucination Check (optional, post-generation)
            hallucination_check = None
            if self._hallucination_guard_enabled and self._hallucination_guard:
                stage_start = time.time()
                try:
                    guard_result = await self._hallucination_guard.detect(
                        response=response,
                        context=contexts,
                        question=query
                    )
                    hallucination_check = guard_result.to_dict()
                    stage_metrics.append(StageMetrics(
                        stage=PipelineStage.HALLUCINATION_CHECK,
                        latency_ms=(time.time() - stage_start) * 1000,
                        input_count=guard_result.total_claims,
                        output_count=guard_result.unsupported_claims,
                        decision="flagged" if guard_result.has_hallucination else "passed",
                        details={
                            "confidence": guard_result.confidence,
                            "severity": guard_result.severity.value,
                            "flagged_count": len(guard_result.flagged_claims),
                        }
                    ))

                    # Optionally adjust response confidence based on hallucination check
                    if guard_result.has_hallucination:
                        confidence = min(confidence, guard_result.confidence)
                        logger.warning(
                            f"Hallucination detected: severity={guard_result.severity.value}, "
                            f"flagged_claims={len(guard_result.flagged_claims)}"
                        )
                except Exception as e:
                    logger.warning(f"Hallucination check failed: {e}")

            # Stage 7: Evaluate (optional)
            evaluation = None
            if self.config.enable_evaluation and self.evaluator:
                stage_start = time.time()
                try:
                    eval_result = await self.evaluator.evaluate_single(
                        question=query,
                        contexts=contexts,
                        answer=response
                    )
                    evaluation = {
                        "faithfulness": getattr(eval_result, 'faithfulness', 0),
                        "answer_relevancy": getattr(eval_result, 'answer_relevancy', 0),
                        "passed": getattr(eval_result, 'passed', True)
                    }
                    stage_metrics.append(StageMetrics(
                        stage=PipelineStage.EVALUATE,
                        latency_ms=(time.time() - stage_start) * 1000,
                        details=evaluation
                    ))
                except Exception as e:
                    logger.warning(f"Evaluation failed: {e}")

            total_latency = (time.time() - start_time) * 1000

            # Track query access for cache warming
            self.track_query_access(
                query=query,
                latency_ms=total_latency,
                metadata={"query_type": query_type.value, "strategy": strategy.value}
            )

            result = PipelineResult(
                response=response,
                confidence=confidence,
                contexts_used=contexts,
                documents_retrieved=len(all_documents),
                strategy_used=strategy,
                stage_metrics=stage_metrics,
                total_latency_ms=total_latency,
                evaluation=evaluation,
                hallucination_check=hallucination_check,
                metadata={"query_type": query_type.value, "queries_used": len(queries)}
            )

            # Store in result cache (if enabled)
            if self._result_cache_enabled and self._result_cache and not skip_cache:
                self._result_cache.put(
                    query=query,
                    value=result,
                    query_type=query_type.value if query_type else "general",
                )

            return result

        except asyncio.TimeoutError:
            logger.error("Pipeline timeout")
            return PipelineResult(
                response="Request timed out. Please try again.",
                confidence=0.0,
                stage_metrics=stage_metrics,
                total_latency_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return PipelineResult(
                response=f"An error occurred: {str(e)}",
                confidence=0.0,
                stage_metrics=stage_metrics,
                total_latency_ms=(time.time() - start_time) * 1000
            )

    async def batch_run(
        self,
        queries: List[str],
        query_types: Optional[List[QueryType]] = None,
        strategy: Optional[StrategyType] = None,
        **kwargs
    ) -> BatchResult:
        """Execute the RAG pipeline on a batch of queries with optimized throughput.

        This method provides up to 10x throughput improvement for batch queries by:
        1. Deduplicating similar queries
        2. Sharing retrieval results across similar queries
        3. Processing queries concurrently with rate limiting
        4. Using a semaphore to control maximum concurrency

        Args:
            queries: List of user queries to process
            query_types: Optional list of query type overrides (one per query)
            strategy: Optional strategy override for all queries
            **kwargs: Additional arguments passed to individual run calls

        Returns:
            BatchResult with all individual results and batch metrics

        Example:
            >>> queries = ["What is Python?", "What's Python?", "Explain Docker"]
            >>> batch_result = await pipeline.batch_run(queries)
            >>> for result in batch_result.results:
            ...     print(result.response[:100])
        """
        start_time = time.time()

        if not queries:
            return BatchResult(
                results=[],
                total_queries=0,
                unique_queries=0,
                total_latency_ms=0.0,
                throughput_qps=0.0
            )

        # Initialize rate limiter and deduplicator
        rate_limiter = RateLimiter(
            qps=self.config.batch_qps,
            burst_multiplier=2.0
        )
        deduplicator = QueryDeduplicator(
            similarity_threshold=self.config.batch_similarity_threshold
        )

        # Deduplicate queries
        unique_queries, query_mapping, dedup_stats = deduplicator.deduplicate(queries)
        logger.info(f"Batch processing: {len(queries)} queries -> {len(unique_queries)} unique")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.batch_max_concurrency)

        # Cache for retrieval results to share across similar queries
        retrieval_cache: Dict[str, List[RetrievedDocument]] = {}
        cache_lock = asyncio.Lock()

        async def process_query_with_shared_retrieval(
            query: str,
            query_idx: int,
            query_type: Optional[QueryType] = None
        ) -> PipelineResult:
            """Process a single query with shared retrieval and rate limiting."""
            async with semaphore:
                await rate_limiter.acquire()

                # Check if we can reuse retrieval results
                cache_key = None
                cached_docs = None

                if self.config.batch_share_retrieval:
                    async with cache_lock:
                        for cached_query, docs in retrieval_cache.items():
                            if deduplicator._compute_similarity(query, cached_query) >= self.config.batch_similarity_threshold:
                                cached_docs = docs
                                cache_key = cached_query
                                break

                if cached_docs is not None:
                    # Use cached retrieval results
                    return await self._run_with_cached_retrieval(
                        query=query,
                        cached_documents=cached_docs,
                        query_type=query_type,
                        strategy=strategy,
                        **kwargs
                    )
                else:
                    # Run full pipeline and cache results
                    result = await self.run(
                        query=query,
                        query_type=query_type,
                        strategy=strategy,
                        **kwargs
                    )

                    # Cache the retrieval results for this query
                    if self.config.batch_share_retrieval and result.documents_retrieved > 0:
                        async with cache_lock:
                            # Store as RetrievedDocument objects
                            retrieval_cache[query] = [
                                RetrievedDocument(
                                    content=ctx,
                                    score=1.0 - (i * 0.1),  # Approximate scores
                                    source="cached",
                                    metadata={"original_query": query}
                                )
                                for i, ctx in enumerate(result.contexts_used)
                            ]

                    return result

        # Build task list with proper query type mapping
        tasks = []
        for i, query in enumerate(unique_queries):
            qt = None
            if query_types:
                # Find the original index for this unique query
                original_indices = query_mapping[query]
                if original_indices and original_indices[0] < len(query_types):
                    qt = query_types[original_indices[0]]

            tasks.append(process_query_with_shared_retrieval(query, i, qt))

        # Execute all queries concurrently
        unique_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Map results back to original query order
        results: List[PipelineResult] = [None] * len(queries)  # type: ignore
        shared_retrieval_count = 0

        for unique_query, unique_result in zip(unique_queries, unique_results):
            original_indices = query_mapping[unique_query]

            if isinstance(unique_result, Exception):
                error_result = PipelineResult(
                    response=f"Error processing query: {str(unique_result)}",
                    confidence=0.0,
                    metadata={"error": str(unique_result)}
                )
                for idx in original_indices:
                    results[idx] = error_result
            else:
                # First result is the primary, rest share the retrieval
                for i, idx in enumerate(original_indices):
                    if i == 0:
                        results[idx] = unique_result
                    else:
                        # Create a copy with shared retrieval flag
                        shared_result = PipelineResult(
                            response=unique_result.response,
                            confidence=unique_result.confidence,
                            contexts_used=unique_result.contexts_used,
                            documents_retrieved=unique_result.documents_retrieved,
                            strategy_used=unique_result.strategy_used,
                            stage_metrics=unique_result.stage_metrics,
                            total_latency_ms=unique_result.total_latency_ms,
                            evaluation=unique_result.evaluation,
                            metadata={
                                **unique_result.metadata,
                                "shared_from_query": unique_query,
                                "shared_retrieval": True
                            }
                        )
                        results[idx] = shared_result
                        shared_retrieval_count += 1

        total_latency_ms = (time.time() - start_time) * 1000
        throughput_qps = (len(queries) / (total_latency_ms / 1000)) if total_latency_ms > 0 else 0

        return BatchResult(
            results=results,
            total_queries=len(queries),
            unique_queries=len(unique_queries),
            shared_retrievals=shared_retrieval_count,
            total_latency_ms=total_latency_ms,
            throughput_qps=throughput_qps,
            deduplication_stats=dedup_stats
        )

    async def _run_with_cached_retrieval(
        self,
        query: str,
        cached_documents: List[RetrievedDocument],
        query_type: Optional[QueryType] = None,
        strategy: Optional[StrategyType] = None,
        **kwargs
    ) -> PipelineResult:
        """Run pipeline stages using cached retrieval results.

        This enables sharing retrieval results across similar queries,
        significantly improving batch throughput.
        """
        start_time = time.time()
        stage_metrics: List[StageMetrics] = []

        try:
            # Classify query if not provided
            if query_type is None:
                query_type = await self.classifier.classify(query)

            # Determine strategy
            if strategy is None:
                strategy_name = self.strategies.get(query_type, self.config.default_strategy.value)
                strategy = StrategyType(strategy_name) if strategy_name in [s.value for s in StrategyType] else self.config.default_strategy

            # Skip retrieval - use cached documents
            stage_metrics.append(StageMetrics(
                stage=PipelineStage.RETRIEVE,
                latency_ms=0.0,
                input_count=1,
                output_count=len(cached_documents),
                decision="cached",
                details={"cache_hit": True}
            ))

            # Rerank cached documents for this specific query
            documents = cached_documents
            if self.config.enable_reranking and self.reranker:
                stage_start = time.time()
                documents = await self._rerank(query, cached_documents)
                stage_metrics.append(StageMetrics(
                    stage=PipelineStage.RERANK,
                    latency_ms=(time.time() - stage_start) * 1000,
                    input_count=len(cached_documents),
                    output_count=min(len(documents), self.config.top_k_final),
                    decision="reranked_cached"
                ))

            # Select within token budget
            selected_docs = self.context_manager.select_contexts(
                documents[:self.config.top_k_final],
                self.config.max_token_budget
            )

            # Generate response
            stage_start = time.time()
            contexts = [d.content for d in selected_docs]
            response, confidence = await self._generate(query, contexts, **kwargs)
            stage_metrics.append(StageMetrics(
                stage=PipelineStage.GENERATE,
                latency_ms=(time.time() - stage_start) * 1000,
                input_count=len(contexts),
                output_count=1,
                details={"confidence": confidence, "used_cache": True}
            ))

            total_latency = (time.time() - start_time) * 1000

            return PipelineResult(
                response=response,
                confidence=confidence,
                contexts_used=contexts,
                documents_retrieved=len(cached_documents),
                strategy_used=strategy,
                stage_metrics=stage_metrics,
                total_latency_ms=total_latency,
                metadata={
                    "query_type": query_type.value,
                    "used_cached_retrieval": True
                }
            )

        except Exception as e:
            logger.error(f"Cached retrieval pipeline error: {e}", exc_info=True)
            return PipelineResult(
                response=f"An error occurred: {str(e)}",
                confidence=0.0,
                stage_metrics=stage_metrics,
                total_latency_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e), "used_cached_retrieval": True}
            )

    async def _run_specialized_strategy(
        self,
        query: str,
        strategy: StrategyType,
        stage_metrics: List[StageMetrics],
        start_time: float,
        **kwargs
    ) -> PipelineResult:
        """Run a specialized RAG strategy implementation."""
        impl = self.rag_implementations[strategy.value]
        stage_start = time.time()

        try:
            if strategy == StrategyType.AGENTIC:
                result = await impl.run(query, **kwargs)
                response = result.response
                confidence = result.confidence
                contexts = [c.content for c in getattr(result, 'contexts', [])] if hasattr(result, 'contexts') else []
            else:
                result = await impl.generate(query, **kwargs)
                response = result.response if hasattr(result, 'response') else str(result)
                confidence = getattr(result, 'confidence', 0.7)
                contexts = []

            stage_metrics.append(StageMetrics(
                stage=PipelineStage.GENERATE,
                latency_ms=(time.time() - stage_start) * 1000,
                decision=f"strategy:{strategy.value}",
                details={"iterations": getattr(result, 'iterations', 1)}
            ))

            return PipelineResult(
                response=response,
                confidence=confidence,
                contexts_used=contexts,
                strategy_used=strategy,
                stage_metrics=stage_metrics,
                total_latency_ms=(time.time() - start_time) * 1000,
                metadata={"strategy_result": True}
            )

        except Exception as e:
            logger.error(f"Specialized strategy {strategy.value} failed: {e}")
            raise

    async def _retrieve_all(
        self,
        queries: List[str],
        top_k: Optional[int] = None
    ) -> List[RetrievedDocument]:
        """Retrieve from all sources using multi-query expansion and fuse with RRF.

        This method implements two-level RRF fusion for multi-query retrieval:
        1. First level: Fuse results from different retrievers for each query
        2. Second level: Fuse results across all expanded queries

        When GraphRAG is enabled, this method also retrieves from the knowledge
        graph and fuses results with higher weight for entity-relationship context.

        Args:
            queries: List of queries (original + expanded variants from query rewriter)
            top_k: Optional top-k override. If None, uses config.top_k_retrieve
                   (or adaptive top-k if enabled)

        Returns:
            List of retrieved documents with RRF-fused scores
        """
        # Use provided top_k or fall back to config
        effective_top_k = top_k if top_k is not None else self.config.top_k_retrieve

        # Store per-query results for two-level RRF fusion
        per_query_results: List[List[RetrievedDocument]] = []
        query_weights: List[float] = []

        # Process each query variant
        for query_idx, query in enumerate(queries):
            retriever_results: List[List[RetrievedDocument]] = []
            retriever_weights: List[float] = []

            # Retrieve from standard retrievers in parallel
            retrieval_tasks = []
            for retriever in self.retrievers:
                retrieval_tasks.append(
                    self._retrieve_from_single_source(
                        retriever, query, effective_top_k
                    )
                )

            # Add GraphRAG retrieval if enabled
            if self.config.enable_graph_rag and self._graph_rag_tool is not None:
                retrieval_tasks.append(
                    self._retrieve_from_graph_async(query, effective_top_k)
                )

            # Execute all retrievals in parallel
            if retrieval_tasks:
                results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

                # Process standard retriever results
                for idx, result in enumerate(results[:len(self.retrievers)]):
                    if isinstance(result, Exception):
                        logger.warning(f"Retrieval error for query '{query[:50]}...': {result}")
                        continue
                    if result:
                        retriever_results.append(result)
                        retriever_weights.append(1.0)

                # Process GraphRAG results (if present)
                if self.config.enable_graph_rag and self._graph_rag_tool is not None:
                    graph_result = results[-1]
                    if not isinstance(graph_result, Exception) and graph_result:
                        retriever_results.append(graph_result)
                        retriever_weights.append(self.config.graph_weight)
                        logger.debug(f"GraphRAG retrieved {len(graph_result)} docs for query variant")

            # First-level RRF: Fuse results from different retrievers for this query
            if retriever_results:
                if len(retriever_results) == 1:
                    query_fused = retriever_results[0]
                else:
                    query_fused = self.fusion.fuse(retriever_results, weights=retriever_weights)

                per_query_results.append(query_fused)

                # Weight queries: original query (index 0) gets higher weight
                # Expanded queries get progressively lower weights
                if query_idx == 0:
                    query_weights.append(1.0)  # Original query - highest weight
                elif query_idx <= 2:
                    query_weights.append(0.8)  # Primary expansions
                else:
                    query_weights.append(0.6)  # Step-back and sub-queries

        if not per_query_results:
            return []

        if len(per_query_results) == 1:
            return self.context_manager.deduplicate(per_query_results[0])

        # Second-level RRF: Fuse results across all query variants
        final_fused = self.fusion.fuse(per_query_results, weights=query_weights)
        logger.debug(
            f"Multi-query RRF fusion: {len(queries)} queries, "
            f"{sum(len(r) for r in per_query_results)} total docs -> {len(final_fused)} fused"
        )

        return self.context_manager.deduplicate(final_fused)

    async def _retrieve_from_single_source(
        self,
        retriever: RetrieverProtocol,
        query: str,
        top_k: int
    ) -> List[RetrievedDocument]:
        """Retrieve from a single source with timeout handling.

        Args:
            retriever: The retriever to use
            query: Search query
            top_k: Number of results to retrieve

        Returns:
            List of retrieved documents
        """
        try:
            timeout = self.config.timeout_seconds / max(len(self.retrievers), 1)
            docs = await asyncio.wait_for(
                retriever.retrieve(query, top_k=top_k),
                timeout=timeout
            )
            return [
                RetrievedDocument(
                    content=doc.get("content", str(doc)),
                    score=doc.get("score", 0.5),
                    source=getattr(retriever, 'name', 'unknown'),
                    metadata=doc.get("metadata", {})
                )
                for doc in docs
            ]
        except asyncio.TimeoutError:
            logger.warning(f"Retriever {getattr(retriever, 'name', 'unknown')} timed out")
            return []
        except Exception as e:
            logger.warning(f"Retrieval error from {getattr(retriever, 'name', 'unknown')}: {e}")
            return []

    async def _retrieve_from_graph_async(
        self,
        query: str,
        top_k: int
    ) -> List[RetrievedDocument]:
        """Async wrapper for GraphRAG retrieval.

        Args:
            query: Search query
            top_k: Number of results to retrieve

        Returns:
            List of retrieved documents from GraphRAG
        """
        try:
            return await self._retrieve_from_graph(query=query, top_k=top_k)
        except Exception as e:
            logger.warning(f"GraphRAG retrieval error: {e}")
            return []

    async def _rerank(
        self,
        query: str,
        documents: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """Rerank documents using the configured reranker and optionally ColBERT.

        If ColBERT reranking is enabled, documents go through two-stage reranking:
        1. First pass with cross-encoder (if configured)
        2. Second pass with ColBERT late-interaction (if enabled)

        Args:
            query: Search query
            documents: Documents to rerank

        Returns:
            Reranked documents sorted by relevance score
        """
        if not documents:
            return documents

        reranked_docs = documents

        # Stage 1: Standard reranker (cross-encoder)
        if self.reranker:
            try:
                from .reranker import Document as RerankerDoc

                reranker_docs = [
                    RerankerDoc(id=str(i), content=d.content, metadata=d.metadata)
                    for i, d in enumerate(reranked_docs)
                ]

                reranked = await self.reranker.rerank(
                    query, reranker_docs, top_k=self.config.top_k_final * 2
                )

                reranked_docs = [
                    RetrievedDocument(
                        content=sd.document.content,
                        score=sd.score,
                        source=documents[int(sd.document.id)].source if sd.document.id.isdigit() else "reranked",
                        metadata=sd.document.metadata
                    )
                    for sd in reranked
                ]
            except ImportError:
                logger.warning("Reranker module not available, skipping standard rerank")
            except Exception as e:
                logger.warning(f"Standard reranking failed: {e}")

        # Stage 2: ColBERT late-interaction reranking (optional)
        if self._enable_colbert_reranking and self._colbert_reranker:
            try:
                colbert_results = await self._colbert_reranker.rerank(
                    query,
                    [{"content": d.content, "metadata": d.metadata} for d in reranked_docs],
                    top_k=self.config.top_k_final * 2
                )

                reranked_docs = [
                    RetrievedDocument(
                        content=r.content,
                        score=r.score,
                        source="colbert_reranked",
                        metadata=getattr(r, 'metadata', {})
                    )
                    for r in colbert_results
                ]
                logger.debug(f"ColBERT reranked {len(colbert_results)} documents")
            except Exception as e:
                logger.warning(f"ColBERT reranking failed: {e}")

        return reranked_docs

    async def _generate(
        self,
        query: str,
        contexts: List[str],
        **kwargs
    ) -> Tuple[str, float]:
        """Generate response using LLM."""
        if not contexts:
            context_text = "No relevant context available."
        else:
            context_text = "\n\n---\n\n".join(contexts)

        prompt = f"""Answer the following question using the provided context.
Be accurate, comprehensive, and cite the context when relevant.
If the context is insufficient, acknowledge the limitations.

Question: {query}

Context:
{context_text}

Answer:"""

        response = await self.llm.generate(
            prompt,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature)
        )

        confidence = 0.7 if contexts else 0.3
        return response.strip(), confidence

    def add_retriever(self, retriever: RetrieverProtocol) -> None:
        """Add a retriever to the pipeline."""
        self.retrievers.append(retriever)

    def set_strategy(self, query_type: QueryType, strategy: str) -> None:
        """Set strategy for a query type."""
        self.strategies[query_type] = strategy

    def register_implementation(self, name: str, implementation: Any) -> None:
        """Register a RAG strategy implementation."""
        self.rag_implementations[name] = implementation

    def set_colbert_reranker(self, reranker: Any, enable: bool = True) -> None:
        """Set or replace the ColBERT reranker for late-interaction reranking.

        ColBERT provides superior reranking quality through token-level
        late interaction matching (MaxSim operations).

        Args:
            reranker: ColBERTReranker instance
            enable: Whether to enable ColBERT reranking (default: True)

        Example:
            from core.rag.colbert_retriever import ColBERTReranker
            colbert_reranker = ColBERTReranker()
            pipeline.set_colbert_reranker(colbert_reranker)
        """
        self._colbert_reranker = reranker
        self._enable_colbert_reranking = enable
        if enable:
            logger.info("ColBERT reranking enabled")

    def enable_colbert_reranking(self) -> None:
        """Enable ColBERT reranking stage (requires colbert_reranker to be set)."""
        if self._colbert_reranker is None:
            raise RuntimeError(
                "ColBERT reranker not set. Use set_colbert_reranker() first."
            )
        self._enable_colbert_reranking = True
        logger.info("ColBERT reranking enabled")

    def disable_colbert_reranking(self) -> None:
        """Disable ColBERT reranking stage."""
        self._enable_colbert_reranking = False
        logger.info("ColBERT reranking disabled")

    @property
    def colbert_available(self) -> bool:
        """Check if ColBERT reranking is available and enabled."""
        return (
            self._colbert_reranker is not None
            and self._enable_colbert_reranking
            and getattr(self._colbert_reranker, 'is_available', lambda: True)()
        )

    def get_metrics_summary(self, result: PipelineResult) -> Dict[str, Any]:
        """Get a summary of pipeline metrics."""
        return {
            "total_latency_ms": result.total_latency_ms,
            "stages": {
                m.stage.value: {
                    "latency_ms": m.latency_ms,
                    "input_count": m.input_count,
                    "output_count": m.output_count
                }
                for m in result.stage_metrics
            },
            "documents_retrieved": result.documents_retrieved,
            "contexts_used": len(result.contexts_used),
            "confidence": result.confidence,
            "strategy": result.strategy_used.value,
            "evaluation": result.evaluation
        }

    async def run_streaming(
        self,
        query: str,
        streaming_config: Optional["StreamingConfig"] = None,
        cancellation_token: Optional["CancellationToken"] = None,
        query_type: Optional[QueryType] = None,
        strategy: Optional[StrategyType] = None,
        **kwargs
    ) -> AsyncGenerator["StreamEvent", None]:
        """Execute RAG pipeline with streaming response.

        This method provides streaming support for the RAG pipeline, yielding
        events as each stage progresses. This enables:
        - Real-time UI updates during retrieval
        - Progressive display of reranked results (top-first)
        - Chunk-by-chunk LLM generation streaming
        - Early termination via cancellation token

        Args:
            query: User query
            streaming_config: Configuration for streaming behavior
            cancellation_token: Token for early termination
            query_type: Optional query type override
            strategy: Optional strategy override
            **kwargs: Additional arguments passed to generation

        Yields:
            StreamEvent objects for each pipeline event

        Example:
            >>> async for event in pipeline.run_streaming("What is RAG?"):
            ...     if event.event_type == StreamEventType.GENERATION_CHUNK:
            ...         print(event.chunk, end="", flush=True)
            ...     elif event.event_type == StreamEventType.COMPLETE:
            ...         print(f"\\nDone in {event.total_latency_ms}ms")
        """
        from .streaming import (
            StreamingConfig,
            StreamEvent,
            StreamEventType,
            StreamingStage,
            StreamingRAGResponse,
            CancellationToken,
            CancellationError,
            stream_multi_source_retrieval,
            stream_reranking_top_first,
            stream_llm_generation,
        )

        config = streaming_config or StreamingConfig()
        response = StreamingRAGResponse(config, cancellation_token or CancellationToken())

        # Start streaming
        yield response.start()

        try:
            # Check for cancellation
            if response.cancellation_token.is_cancelled:
                yield response.cancel(response.cancellation_token.reason or "Cancelled")
                return

            # Stage 1: Query Classification and Rewrite
            yield response.start_stage(StreamingStage.QUERY_REWRITE)

            if query_type is None:
                query_type = await self.classifier.classify(query)

            # Determine strategy
            if strategy is None:
                strategy_name = self.strategies.get(query_type, self.config.default_strategy.value)
                strategy = StrategyType(strategy_name) if strategy_name in [s.value for s in StrategyType] else self.config.default_strategy

            queries = [query]
            if self.config.enable_query_rewrite:
                queries = await self.rewriter.rewrite(query, query_type)

            yield response.complete_stage(
                StreamingStage.QUERY_REWRITE,
                {"original_query": query, "rewritten_queries": queries, "query_type": query_type.value}
            )

            # Stage 2: Retrieval (streaming)
            if response.cancellation_token.is_cancelled:
                yield response.cancel(response.cancellation_token.reason or "Cancelled")
                return

            all_documents: List[RetrievedDocument] = []

            if self.retrievers:
                sources = [getattr(r, 'name', type(r).__name__) for r in self.retrievers]
                yield response.start_retrieval(sources)

                # Stream from all retrievers
                for retriever in self.retrievers:
                    retriever_name = getattr(retriever, 'name', type(retriever).__name__)

                    for q in queries:
                        try:
                            docs = await asyncio.wait_for(
                                retriever.retrieve(q, top_k=self.config.top_k_retrieve),
                                timeout=self.config.timeout_seconds / len(self.retrievers)
                            )

                            batch_docs = []
                            for doc in docs:
                                retrieved_doc = RetrievedDocument(
                                    content=doc.get("content", str(doc)),
                                    score=doc.get("score", 0.5),
                                    source=retriever_name,
                                    metadata=doc.get("metadata", {})
                                )
                                all_documents.append(retrieved_doc)
                                batch_docs.append({
                                    "content": retrieved_doc.content,
                                    "score": retrieved_doc.score,
                                    "source": retrieved_doc.source,
                                    "metadata": retrieved_doc.metadata,
                                })

                            if batch_docs:
                                yield response.emit_retrieval_batch(batch_docs, retriever_name)

                        except asyncio.TimeoutError:
                            logger.warning(f"Retriever {retriever_name} timed out")
                        except Exception as e:
                            logger.warning(f"Retrieval error from {retriever_name}: {e}")

                    # Progress update
                    response.emit_progress(
                        StreamingStage.RETRIEVAL,
                        percent=min(100, (self.retrievers.index(retriever) + 1) / len(self.retrievers) * 100),
                        message=f"Retrieved from {retriever_name}",
                        items_processed=len(all_documents),
                    )

                # Deduplicate
                all_documents = self.context_manager.deduplicate(all_documents)
                yield response.complete_retrieval(len(all_documents))

            if not all_documents:
                yield response.complete(
                    "I could not find relevant information to answer your question.",
                    confidence=0.0
                )
                return

            # Stage 3: Reranking (streaming top-first)
            if response.cancellation_token.is_cancelled:
                yield response.cancel(response.cancellation_token.reason or "Cancelled")
                return

            reranked_docs = all_documents

            if self.config.enable_reranking and self.reranker:
                yield response.start_reranking(len(all_documents))

                try:
                    from .reranker import Document as RerankerDoc

                    reranker_docs = [
                        RerankerDoc(id=str(i), content=d.content, metadata=d.metadata)
                        for i, d in enumerate(all_documents)
                    ]

                    reranked = await self.reranker.rerank(
                        query, reranker_docs, top_k=self.config.top_k_final * 2
                    )

                    reranked_docs = []
                    for rank, sd in enumerate(reranked, start=1):
                        original_idx = int(sd.document.id) if sd.document.id.isdigit() else 0
                        original_doc = all_documents[original_idx] if original_idx < len(all_documents) else all_documents[0]

                        new_doc = RetrievedDocument(
                            content=sd.document.content,
                            score=sd.score,
                            source=original_doc.source,
                            metadata=sd.document.metadata
                        )
                        reranked_docs.append(new_doc)

                        # Stream top results as they're determined
                        if config.stream_reranking and rank <= self.config.top_k_final:
                            yield response.emit_rerank_result(
                                {"content": new_doc.content, "score": new_doc.score, "source": new_doc.source},
                                sd.score,
                                rank
                            )

                except ImportError:
                    logger.warning("Reranker module not available")
                except Exception as e:
                    logger.warning(f"Reranking failed: {e}")

                yield response.complete_reranking([
                    {"content": d.content, "score": d.score, "source": d.source}
                    for d in reranked_docs[:self.config.top_k_final]
                ])

            # Select within token budget
            selected_docs = self.context_manager.select_contexts(
                reranked_docs[:self.config.top_k_final],
                self.config.max_token_budget
            )

            # Stage 4: Generation (streaming)
            if response.cancellation_token.is_cancelled:
                yield response.cancel(response.cancellation_token.reason or "Cancelled")
                return

            contexts = [d.content for d in selected_docs]
            yield response.start_generation(contexts)

            # Build prompt
            context_text = "\n\n---\n\n".join(contexts) if contexts else "No relevant context available."
            prompt = f"""Answer the following question using the provided context.
Be accurate, comprehensive, and cite the context when relevant.
If the context is insufficient, acknowledge the limitations.

Question: {query}

Context:
{context_text}

Answer:"""

            # Check for streaming LLM support
            full_response = ""
            if hasattr(self.llm, 'generate_stream'):
                try:
                    async for chunk in self.llm.generate_stream(
                        prompt,
                        max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                        temperature=kwargs.get("temperature", self.config.temperature)
                    ):
                        full_response += chunk
                        if config.stream_generation:
                            yield response.emit_generation_chunk(chunk)
                except Exception as e:
                    logger.warning(f"Streaming generation failed, falling back: {e}")
                    full_response = await self.llm.generate(
                        prompt,
                        max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                        temperature=kwargs.get("temperature", self.config.temperature)
                    )
                    if config.stream_generation:
                        # Emit in chunks for UI consistency
                        chunk_size = config.chunk_size
                        for i in range(0, len(full_response), chunk_size):
                            yield response.emit_generation_chunk(full_response[i:i + chunk_size])
            else:
                # Non-streaming fallback
                full_response = await self.llm.generate(
                    prompt,
                    max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                    temperature=kwargs.get("temperature", self.config.temperature)
                )
                if config.stream_generation:
                    chunk_size = config.chunk_size
                    for i in range(0, len(full_response), chunk_size):
                        yield response.emit_generation_chunk(full_response[i:i + chunk_size])

            confidence = 0.7 if contexts else 0.3
            yield response.complete_generation(full_response.strip(), confidence)

            # Track query access for cache warming
            self.track_query_access(
                query=query,
                latency_ms=response.elapsed_ms,
                metadata={"query_type": query_type.value, "strategy": strategy.value, "streaming": True}
            )

            yield response.complete(full_response.strip(), confidence, {
                "query_type": query_type.value,
                "strategy": strategy.value,
                "documents_retrieved": len(all_documents),
                "contexts_used": len(contexts),
            })

        except CancellationError as e:
            yield response.cancel(str(e))

        except asyncio.TimeoutError:
            yield response.error("Pipeline timeout")

        except Exception as e:
            logger.error(f"Streaming pipeline error: {e}", exc_info=True)
            yield response.error(str(e))


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_pipeline(
    llm: LLMProvider,
    retrievers: Optional[List[RetrieverProtocol]] = None,
    reranker: Optional[RerankerProtocol] = None,
    evaluator: Optional[EvaluatorProtocol] = None,
    self_rag: Optional[Any] = None,
    crag: Optional[Any] = None,
    hyde: Optional[Any] = None,
    agentic_rag: Optional[Any] = None,
    colbert_retriever: Optional[Any] = None,
    colbert_reranker: Optional[Any] = None,
    enable_colbert_reranking: bool = False,
    graph_rag: Optional[Any] = None,
    config: Optional[PipelineConfig] = None,
    **strategy_mappings
) -> RAGPipeline:
    """Factory function to create a configured RAGPipeline.

    Args:
        llm: LLM provider
        retrievers: List of retriever implementations (can include ColBERTRetriever)
        reranker: Optional reranker (cross-encoder or SemanticReranker)
        evaluator: Optional evaluator
        self_rag: Optional Self-RAG implementation
        crag: Optional CRAG implementation
        hyde: Optional HyDE implementation
        agentic_rag: Optional Agentic RAG implementation
        colbert_retriever: Optional ColBERT/RAGatouille retriever for late interaction
        colbert_reranker: Optional ColBERTReranker for late-interaction reranking
        enable_colbert_reranking: Enable ColBERT reranking stage (default: False)
        graph_rag: Optional GraphRAG for entity-relationship retrieval
        config: Pipeline configuration
        **strategy_mappings: Query type to strategy name mappings

    Returns:
        Configured RAGPipeline instance

    Example:
        >>> from core.rag.colbert_retriever import ColBERTRetriever, ColBERTReranker
        >>> colbert = ColBERTRetriever.from_index(".ragatouille/indexes/my_index")
        >>> colbert_reranker = ColBERTReranker()
        >>> pipeline = create_pipeline(
        ...     llm=my_llm,
        ...     retrievers=[exa, tavily, colbert],  # ColBERT as retriever
        ...     colbert_reranker=colbert_reranker,  # ColBERT for reranking
        ...     enable_colbert_reranking=True,
        ...     research="colbert",  # Use ColBERT strategy for research queries
        ... )

        >>> # Or with GraphRAG
        >>> from core.rag.graph_rag import GraphRAG
        >>> graph_rag = GraphRAG(llm=my_llm, embedder=my_embedder)
        >>> pipeline = create_pipeline(
        ...     llm=my_llm,
        ...     retrievers=[exa, tavily],
        ...     graph_rag=graph_rag,
        ...     config=PipelineConfig(enable_graph_rag=True),
        ...     factual="graph_rag",
        ... )
    """
    implementations: Dict[str, Any] = {}
    if self_rag:
        implementations["self_rag"] = self_rag
    if crag:
        implementations["crag"] = crag
    if hyde:
        implementations["hyde"] = hyde
    if agentic_rag:
        implementations["agentic"] = agentic_rag
    if colbert_retriever:
        implementations["colbert"] = colbert_retriever
    if graph_rag:
        implementations["graph_rag"] = graph_rag

    strategies: Dict[QueryType, str] = {}
    for query_type_str, strategy_name in strategy_mappings.items():
        try:
            qt = QueryType(query_type_str)
            strategies[qt] = strategy_name
        except ValueError:
            logger.warning(f"Unknown query type: {query_type_str}")

    # Enable GraphRAG in config if provided
    if graph_rag and config:
        config.enable_graph_rag = True
    elif graph_rag and not config:
        config = PipelineConfig(enable_graph_rag=True)

    return RAGPipeline(
        llm=llm,
        retrievers=retrievers,
        reranker=reranker,
        evaluator=evaluator,
        strategies=strategies,
        config=config,
        rag_implementations=implementations,
        graph_rag=graph_rag,
        colbert_reranker=colbert_reranker,
        enable_colbert_reranking=enable_colbert_reranking,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main pipeline
    "RAGPipeline",
    "create_pipeline",
    # Configuration
    "PipelineConfig",
    # Result types
    "PipelineResult",
    "BatchResult",
    "RetrievedDocument",
    "StageMetrics",
    # Enums
    "QueryType",
    "PipelineStage",
    "StrategyType",
    # Components
    "QueryClassifier",
    "QueryRewriter",
    "ContextManager",
    "RRFFusion",
    "RateLimiter",
    "QueryDeduplicator",
    # Query Rewriter (from query_rewriter module)
    "HybridQueryRewriter",
    "RuleBasedQueryRewriter",
    "QueryRewriterConfig",
    "RewriteResult",
    "QueryIntent",
    # Protocols
    "LLMProvider",
    "RetrieverProtocol",
    "RerankerProtocol",
    "EvaluatorProtocol",
]
