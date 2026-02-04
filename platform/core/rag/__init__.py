"""
RAG (Retrieval-Augmented Generation) Module - V50 Architecture

This package provides production-ready components for RAG pipelines.
All exports are lazy-loaded for optimal startup performance.

Exported Components:
--------------------

Core RAG Classes:
- AgenticRAG: Iterative retrieval-generation loop with tool augmentation
- RAGEvaluator: Ragas-style comprehensive evaluation metrics
- QueryRewriter: Query transformation for improved retrieval (from CorrectiveRAG)
- ContextManager: Context selection and compression within token budgets
- RAGMetrics: See RankingMetrics for evaluation metrics
- GraphRAG: Entity-relationship knowledge graph for multi-hop reasoning
- RAGPipeline: Unified pipeline orchestrating all RAG components

Core Components:
- SemanticChunker: Intelligent text chunking with semantic boundary detection
- SemanticReranker: Cross-encoder reranking with ms-marco-MiniLM-L-6-v2
- RRF Fusion: Reciprocal Rank Fusion for combining multiple retrievers
- Diversity Reranking: MMR-style diversity to reduce redundancy
- Caching: Query-aware caching for repeated requests

Advanced RAG Patterns (V39+):
- Self-RAG: Self-reflective retrieval with reflection tokens
- Corrective RAG: Document grading with web search fallback
- HyDE: Hypothetical Document Embeddings for improved recall
- Multi-Query Expansion: Generate query variations for better recall
- RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
- Agentic RAG: Iterative retrieval-generation loop with tool augmentation
- Contextual Retrieval: Anthropic's context prepending pattern for improved accuracy

Evaluation (V40+):
- RAGEvaluator: Ragas-style comprehensive evaluation metrics
- Context Metrics: Precision, Recall, Relevancy
- Answer Metrics: Relevancy, Correctness, Faithfulness
- Hallucination Detection: Claims extraction and verification
- Ranking Metrics: NDCG@k, MRR, F1

V50 Additions:
- HallucinationGuard: Post-generation claim verification with ClaimVerification
- ContextualRetriever: Anthropic's context prepending with ContextualConfig/ContextualChunk
- DualLevelRetriever: LightRAG-style dual retrieval with DualLevelConfig/DualLevelResult

Semantic Chunking:
    from core.rag import SemanticChunker, Chunk

    chunker = SemanticChunker(max_chunk_size=512, overlap=50)
    chunks = chunker.chunk(document_text, metadata={"source": "api_docs"})

    for chunk in chunks:
        print(f"Chunk: {chunk.token_estimate} tokens, type: {chunk.content_type}")

Reranking:
    from core.rag import SemanticReranker, Document, ScoredDocument

    # Create reranker
    reranker = SemanticReranker()

    # Rerank documents
    documents = [
        Document(id="1", content="Document 1 content..."),
        Document(id="2", content="Document 2 content..."),
    ]
    results = await reranker.rerank("query", documents, top_k=10)

    # RRF fusion
    fused = await reranker.rrf_fusion([list1, list2, list3])

    # Diversity reranking
    diverse = await reranker.diversity_rerank(results, lambda_diversity=0.3)

Self-RAG (Self-Reflective RAG):
    from core.rag import SelfRAG, SelfRAGConfig

    config = SelfRAGConfig(max_iterations=3, top_k=5)
    self_rag = SelfRAG(llm=my_llm, retriever=my_retriever, config=config)

    result = await self_rag.generate("What is microservices architecture?")
    print(f"Response: {result.response}")
    print(f"Confidence: {result.confidence}")
    print(f"Support: {result.support_grade}")

Corrective RAG (CRAG):
    from core.rag import CorrectiveRAG, CRAGConfig

    config = CRAGConfig(correct_threshold=0.7, enable_web_fallback=True)
    crag = CorrectiveRAG(
        llm=my_llm,
        retriever=my_retriever,
        web_search=tavily_adapter
    )

    result = await crag.generate("What are the latest AI developments?")
    print(f"Response: {result.response}")
    print(f"Source: {result.source_type}")
    print(f"Grading: {result.grading_decision}")

HyDE (Hypothetical Document Embeddings):
    from core.rag import HyDERetriever, HyDEConfig

    config = HyDEConfig(n_hypothetical=5, hypothesis_type=HypotheticalDocumentType.ANSWER)
    hyde = HyDERetriever(
        llm=my_llm,
        embedder=my_embedder,
        vector_store=my_store,
        config=config
    )

    result = await hyde.retrieve("What are the benefits of microservices?")
    for doc in result.documents:
        print(doc["content"][:100])

Multi-Query Expansion:
    from core.rag import MultiQueryRetriever, MultiQueryConfig

    config = MultiQueryConfig(n_queries=3, fusion_method='rrf')
    retriever = MultiQueryRetriever(
        llm=my_llm,
        base_retriever=my_retriever,
        config=config
    )

    results = await retriever.retrieve("What is RAG?", top_k=10)

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval):
    from core.rag import RAPTOR, RAPTORConfig, RAPTORNode

    config = RAPTORConfig(chunk_size=100, cluster_method=ClusterMethod.GMM)
    raptor = RAPTOR(summarizer=my_llm, embedder=my_embedder, config=config)

    # Build tree from documents
    await raptor.build_tree(["doc1", "doc2", "doc3"])

    # Retrieve with collapsed tree method (search all levels)
    result = await raptor.retrieve("query", method="collapsed", top_k=5)

    # Or use tree traversal (top-down search)
    result = await raptor.retrieve("query", method="tree_traversal", top_k=5)

    # Simple high-level interface with RAPTORIndex
    index = RAPTORIndex(summarizer=my_llm, embedder=my_embedder)
    await index.add(documents)
    results = await index.search("query", k=5)

Agentic RAG (Iterative Retrieval-Generation Loop):
    from core.rag import AgenticRAG, AgenticRAGConfig, create_agentic_rag

    # Configure agentic RAG with tools
    config = AgenticRAGConfig(
        max_iterations=5,
        confidence_threshold=0.8,
        enable_query_decomposition=True,
        enable_reflection=True,
    )

    # Create with factory (includes Exa, Tavily, memory tools)
    agentic = create_agentic_rag(
        llm=my_llm,
        exa_adapter=exa_adapter,
        tavily_adapter=tavily_adapter,
        memory_backend=memory_backend,
        config=config
    )

    # Run iterative loop
    result = await agentic.run("Complex multi-part question")
    print(f"Response: {result.response}")
    print(f"Confidence: {result.confidence}")
    print(f"Iterations: {result.iterations}")
    print(f"Sources: {result.retrieval_sources}")
    print(f"States: {result.states_visited}")

    # With sub-query decomposition
    if result.sub_queries:
        for sq in result.sub_queries:
            print(f"Sub-query: {sq.query} (type: {sq.query_type})")

RAG Evaluation (Ragas-style):
    from core.rag import RAGEvaluator, EvaluationConfig

    config = EvaluationConfig(faithfulness_threshold=0.7)
    evaluator = RAGEvaluator(llm=judge_llm, config=config)

    results = await evaluator.evaluate(
        questions=["What is RAG?"],
        retrieved_contexts=[["RAG is a technique..."]],
        generated_answers=["RAG stands for..."],
        ground_truth=["Retrieval-Augmented Generation..."]
    )

    print(f"Faithfulness: {results.faithfulness:.2f}")
    print(f"Answer Relevancy: {results.answer_relevancy:.2f}")
    print(f"Context Precision: {results.context_precision:.2f}")
    print(f"Hallucination: {results.hallucination_score:.2f}")
    print(f"Overall: {'PASS' if results.overall_passed else 'FAIL'}")

    # Quick evaluation without LLM (for CI/CD)
    from core.rag import QuickEvaluator
    quick = QuickEvaluator()
    scores = quick.evaluate_all(question, contexts, answer)

Memory Backend Integration:
    from core.rag import MemoryReranker, ChunkMemoryIntegration
    from core.memory.backends.sqlite import SQLiteTierBackend

    # Chunking and storage
    backend = SQLiteTierBackend()
    integration = ChunkMemoryIntegration()
    chunk_ids = await integration.store_document("doc-1", content, backend=backend)

    # Search and rerank
    reranker = MemoryReranker()
    results = await reranker.search_and_rerank(
        backend=backend,
        query="authentication patterns",
        initial_limit=50,
        final_top_k=10,
    )

Contextual Retrieval (Anthropic's Pattern):
    from core.rag import ContextualRetriever, ContextualRetrievalConfig

    # Configure with LLM for context generation
    config = ContextualRetrievalConfig(
        strategy=ContextGenerationStrategy.HYBRID,  # LLM with rule-based fallback
        max_context_tokens=100,
        cache_enabled=True,
    )
    retriever = ContextualRetriever(
        llm=my_llm,
        embedding_provider=embedder,
        config=config
    )

    # Process document - each chunk gets prepended context
    chunks = await retriever.contextualize_document(
        document=document_text,
        doc_title="API Documentation",
        doc_metadata={"source": "docs"}
    )

    for chunk in chunks:
        # Pattern: "This chunk is from [title] about [topic]. It discusses: [context]"
        print(f"Context: {chunk.prepended_context}")
        print(f"Content: {chunk.original_content[:100]}...")
        # Use contextualized embedding for better retrieval accuracy
        store_embedding(chunk.contextualized_embedding)

    # Or use the integrated chunker
    from core.rag import ContextualSemanticChunker

    chunker = ContextualSemanticChunker(llm=my_llm)
    contextualized_chunks = await chunker.chunk_with_context(
        text=document_text,
        doc_title="User Guide"
    )

Dual-Level Retrieval (LightRAG-style):
    from core.rag import (
        DualLevelRetriever,
        DualLevelConfig,
        FusionStrategy,
        RetrievalLevel,
        create_dual_level_retriever,
    )

    # Create with RAPTOR as high-level retriever
    config = DualLevelConfig(
        fusion_strategy=FusionStrategy.SCORE_WEIGHTED,
        low_level_weight=0.6,
        high_level_weight=0.4,
        parallel_retrieval=True,
    )

    retriever = create_dual_level_retriever(
        low_level=exa_adapter,     # Fine-grained chunk retrieval
        raptor=raptor_instance,     # RAPTOR for high-level summaries
        fusion_strategy=FusionStrategy.SCORE_WEIGHTED,
    )

    # Retrieve with both levels
    result = await retriever.retrieve(
        "Compare microservices architecture with monolithic design",
        top_k=10,
        level=RetrievalLevel.BOTH,  # Use both low and high-level
    )

    print(f"Low-level docs: {len(result.low_level_documents)}")
    print(f"High-level docs: {len(result.high_level_documents)}")
    print(f"Fused docs: {len(result.fused_documents)}")
    print(f"Query complexity: {result.query_complexity}")

    # The fused context includes [Detail] and [Summary] tags
    for doc in result.fused_documents:
        level_tag = "[Summary]" if doc.level == RetrievalLevel.HIGH else "[Detail]"
        print(f"{level_tag}: {doc.content[:100]}...")

    # Integrate with RAGPipeline as a strategy
    from core.rag import create_dual_level_strategy

    strategy = create_dual_level_strategy(retriever, llm=my_llm)
    pipeline.register_implementation("dual_level", strategy)
    pipeline.set_strategy(QueryType.COMPARISON, "dual_level")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy imports for better startup performance
if TYPE_CHECKING:
    # Reranker module
    from .reranker import (
        Document,
        ScoredDocument,
        EmbeddingProvider,
        RerankerCache,
        TFIDFScorer,
        CrossEncoderReranker,
        SemanticReranker,
        MemoryReranker,
        MultiQueryConfig,
        MultiQueryExpander,
        MultiQueryRetriever,
        MultiQueryReranker,
        create_reranker,
        create_memory_reranker,
    )
    # Semantic chunker module
    from .semantic_chunker import (
        SemanticChunker,
        Chunk,
        ChunkingStats,
        ContentType,
        ContentTypeDetector,
        SentenceSplitter,
        EmbeddingProvider as ChunkerEmbeddingProvider,
        SemanticBoundaryDetector,
        ChunkMemoryIntegration,
    )
    # Self-RAG module
    from .self_rag import (
        SelfRAG,
        SelfRAGWithReranker,
        SelfRAGConfig,
        SelfRAGResult,
        ReflectionResult,
        RetrievedDocument,
        GenerationAttempt,
        ReflectionToken,
        RetrievalDecision,
        RelevanceGrade,
        SupportGrade,
        UsefulnessGrade,
        SelfRAGPrompts,
    )
    # Corrective RAG module
    from .corrective_rag import (
        CorrectiveRAG,
        CRAGWithReranker,
        CRAGConfig,
        CRAGResult,
        GradedDocument,
        GradingResult,
        DocumentGrader,
        KnowledgeRefiner,
        QueryRewriter,
        GradingDecision,
        TavilySearchAdapter,
        CRAGPrompts,
    )
    # HyDE module
    from .hyde import (
        HyDERetriever,
        HyDEWithReranker,
        MultiHyDE,
        HyDEConfig,
        HyDEResult,
        HypotheticalDocument,
        HypotheticalDocumentType,
        EmbeddingCache,
        HyDEPrompts,
    )
    # RAPTOR module
    from .raptor import (
        RAPTOR,
        RAPTORWithChunker,
        RAPTORIndex,
        RAPTORNode,
        RAPTORTree,
        RAPTORResult,
        RAPTORConfig,
        TreeBuildStats,
        ClusterMethod,
        RetrievalMethod,
        Clusterer,
        GMMClusterer,
        KMeansClusterer,
        LLMSummarizer,
    )
    # Evaluation module
    from .evaluation import (
        RAGEvaluator,
        QuickEvaluator,
        EvaluationConfig,
        EvaluationResult,
        SampleEvaluation,
        MetricResult,
        Claim,
        ContextRelevance,
        FaithfulnessCalculator,
        ContextRelevanceCalculator,
        AnswerRelevancyCalculator,
        AnswerCorrectnessCalculator,
        HallucinationDetector,
        ContextPrecisionCalculator,
        ContextRecallCalculator,
        ClaimsExtractor,
        ClaimsVerifier,
        RankingMetrics,
        EvaluationPrompts,
        create_evaluator,
    )
    # Agentic RAG module
    from .agentic_rag import (
        AgenticRAG,
        AgenticRAGConfig,
        AgenticRAGResult,
        AgentState,
        QueryType,
        EvaluationDecision,
        SubQuery,
        RetrievedContext,
        RetrievalResult,
        GenerationResult,
        QueryDecomposer,
        ToolSelector,
        ResponseEvaluator,
        ResultFusion,
        RetrievalTool,
        BaseRetrievalTool,
        ExaSearchTool,
        TavilySearchTool,
        MemorySearchTool,
        CodeSearchTool,
        AgenticRAGPrompts,
        create_agentic_rag,
    )
    # GraphRAG module
    from .graph_rag import (
        GraphRAG,
        GraphRAGConfig,
        Entity,
        Relationship,
        EntityType,
        RelationshipType,
        GraphSearchResult,
        EntityExtractor,
        GraphStorage,
        GraphRAGTool,
    )
    # Metrics module
    from .metrics import (
        RAGMetricsConfig,
        RetrievalSource,
        LatencyPercentiles,
        RetrievalMetrics,
        QualityMetrics,
        UsageMetrics,
        SlowQueryRecord,
        RAGMetricsCollector,
        RetrievalTracker,
        PrometheusExporter,
        JSONExporter,
        StructuredLogger,
        DegradationAlert,
        RAGAnalyzer,
        RAGMetricsIntegration,
        get_rag_metrics_collector,
        create_rag_metrics_integration,
    )
    # Query Rewriter module
    from .query_rewriter import (
        RuleBasedQueryRewriter,
        LLMQueryRewriter,
        HybridQueryRewriter,
        QueryRewriterConfig,
        RewriteResult,
        QueryIntent,
        QueryRewriterPrompts,
    )
    # Pipeline module
    from .pipeline import (
        RAGPipeline,
        create_pipeline,
        PipelineConfig,
        PipelineResult,
        PipelineStage,
        StrategyType,
        StageMetrics,
        QueryClassifier,
        RRFFusion,
    )
    # Context Manager module
    from .context_manager import (
        ContextManager,
        ContextSelector,
        ContextFormatter,
        ContextCompressor,
        TokenCounter,
        ContextConfig,
        ContextItem,
        FormattedContext,
        FormatStyle,
        ContentType as ContextContentType,
        create_context_manager,
        create_context_item,
        count_tokens,
    )
    # ColBERT/RAGatouille module
    from .colbert_retriever import (
        ColBERTConfig,
        ColBERTDocument,
        ColBERTResult,
        ColBERTRetriever,
        ColBERTReranker,
        create_colbert_retriever,
        create_colbert_reranker,
    )
    # Cache Warmer module
    from .cache_warmer import (
        CacheWarmer,
        PatternAnalyzer,
        CacheWarmingMixin,
        CacheWarmerConfig,
        WarmingStrategy,
        QueryAccess,
        QueryPattern,
        WarmingResult,
        WarmingStats,
        create_cache_warmer,
        warm_pipeline_cache,
    )
    # Result Cache module
    from .result_cache import (
        ResultCache,
        CachedRAGPipelineMixin,
        ResultCacheConfig,
        CacheEntry,
        CacheHit,
        CacheStats,
        CacheLevel,
        InvalidationStrategy,
        QueryTypeTTL,
        ExactMatchCache,
        SemanticCache,
        MemoryBudgetManager,
        create_result_cache,
    )
    # Complexity Analyzer module
    from .complexity_analyzer import (
        QueryComplexityAnalyzer,
        ComplexityAwareRouter,
        AnalyzerConfig,
        AdaptiveTopKConfig,
        QueryAnalysis,
        ComplexityFactors,
        CostEstimate,
        RoutingRecommendation,
        EntityInfo,
        ComplexityLevel,
        QuestionType as AnalyzerQuestionType,
        DomainType,
        ModelTier,
        EntityDetector,
        QuestionClassifier,
        DomainDetector,
    )
    # Streaming module
    from .streaming import (
        StreamingRAGResponse,
        StreamingConfig,
        StreamProgress,
        StreamEvent,
        StreamEventType,
        StreamingStage,
        CancellationToken,
        CancellationError,
        StreamingLLMProvider,
        StreamingRetrieverProtocol,
        StreamingPipelineMixin,
        stream_multi_source_retrieval,
        stream_reranking_top_first,
        stream_llm_generation,
        create_streaming_config,
        create_cancellation_token,
    )
    # Hallucination Guard module
    from .hallucination_guard import (
        HallucinationGuard,
        GuardConfig,
        HallucinationResult,
        VerifiedClaim,
        VerifiedClaim as ClaimVerification,  # V50 alias
        ExtractedClaim,
        ClaimVerdict,
        HallucinationSeverity,
        ClaimsExtractor,
        ClaimVerifier,
        ConfidenceScorer,
        GuardPrompts,
        GuardedPipelineMixin,
        create_hallucination_guard,
        quick_hallucination_check,
    )
    # Contextual Retrieval module (Anthropic's pattern)
    from .contextual_retrieval import (
        ContextualRetriever,
        ContextualSemanticChunker,
        ContextualRetrievalConfig,
        ContextualRetrievalConfig as ContextualConfig,  # V50 alias
        ContextGenerationStrategy,
        ContextualizedChunk,
        ContextualizedChunk as ContextualChunk,  # V50 alias
        DocumentContext,
        ContextualizationStats,
        LLMContextGenerator,
        RuleBasedContextGenerator,
        ContextCache,
    )
    # Dual-Level Retrieval module (LightRAG-style)
    from .dual_level_retrieval import (
        DualLevelRetriever,
        DualLevelConfig,
        DualLevelDocument,
        DualLevelResult,
        DualLevelStrategy,
        RetrievalLevel,
        FusionStrategy,
        QueryComplexity as DualLevelQueryComplexity,
        EntityContext,
        ResultFuser,
        InterleaveFuser,
        ScoreWeightedFuser,
        RRFFuser,
        LevelFirstFuser,
        LevelLastFuser,
        get_fuser,
        QueryComplexityAnalyzer as DualLevelComplexityAnalyzer,
        SimpleEntityExtractor,
        RAPTORHighLevelRetriever,
        create_dual_level_retriever,
        create_dual_level_strategy,
    )


_GETATTR_LOADING = set()


def __getattr__(name: str):
    """Lazy load module contents."""
    if name in _GETATTR_LOADING:
        raise AttributeError(f"module 'core.rag' has no attribute '{name}'")
    _GETATTR_LOADING.add(name)
    try:
        return _lazy_resolve(name)
    finally:
        _GETATTR_LOADING.discard(name)


def _lazy_resolve(name: str):
    """Resolve attribute from submodules without recursion."""
    # Try reranker module first
    try:
        from . import reranker
        if hasattr(reranker, name):
            return getattr(reranker, name)
    except (ImportError, RecursionError):
        pass

    _submodules = [
        'semantic_chunker', 'self_rag', 'corrective_rag', 'hyde', 'raptor',
        'evaluation', 'agentic_rag', 'graph_rag', 'metrics', 'query_rewriter',
        'pipeline', 'context_manager', 'colbert_retriever', 'cache_warmer',
        'result_cache', 'complexity_analyzer', 'streaming', 'hallucination_guard',
        'contextual_retrieval', 'dual_level_retrieval',
    ]
    import importlib
    for mod_name in _submodules:
        try:
            mod = importlib.import_module(f'.{mod_name}', __name__)
            if hasattr(mod, name):
                return getattr(mod, name)
        except (ImportError, RecursionError):
            pass

    raise AttributeError(f"module 'core.rag' has no attribute '{name}'")


def __dir__():
    """Return all available names."""
    return list(__all__)


__all__ = [
    # === Semantic Chunking ===
    "SemanticChunker",
    "Chunk",
    "ChunkingStats",
    "ContentType",
    "ContentTypeDetector",
    "SentenceSplitter",
    "SemanticBoundaryDetector",
    "ChunkMemoryIntegration",

    # === Reranking ===
    # Types
    "Document",
    "ScoredDocument",
    "EmbeddingProvider",
    # Cache
    "RerankerCache",
    # Scorers
    "TFIDFScorer",
    "CrossEncoderReranker",
    # Main reranker
    "SemanticReranker",
    # Memory integration
    "MemoryReranker",
    # Multi-Query Expansion
    "MultiQueryConfig",
    "MultiQueryExpander",
    "MultiQueryRetriever",
    "MultiQueryReranker",
    # Factory functions
    "create_reranker",
    "create_memory_reranker",

    # === Self-RAG ===
    "SelfRAG",
    "SelfRAGWithReranker",
    "SelfRAGConfig",
    "SelfRAGResult",
    "ReflectionResult",
    "RetrievedDocument",
    "GenerationAttempt",
    "ReflectionToken",
    "RetrievalDecision",
    "RelevanceGrade",
    "SupportGrade",
    "UsefulnessGrade",
    "SelfRAGPrompts",

    # === Corrective RAG ===
    "CorrectiveRAG",
    "CRAGWithReranker",
    "CRAGConfig",
    "CRAGResult",
    "GradedDocument",
    "GradingResult",
    "DocumentGrader",
    "KnowledgeRefiner",
    "QueryRewriter",
    "GradingDecision",
    "TavilySearchAdapter",
    "CRAGPrompts",

    # === HyDE ===
    "HyDERetriever",
    "HyDEWithReranker",
    "MultiHyDE",
    "HyDEConfig",
    "HyDEResult",
    "HypotheticalDocument",
    "HypotheticalDocumentType",
    "EmbeddingCache",
    "HyDEPrompts",

    # === RAPTOR ===
    "RAPTOR",
    "RAPTORWithChunker",
    "RAPTORIndex",
    "RAPTORNode",
    "RAPTORTree",
    "RAPTORResult",
    "RAPTORConfig",
    "TreeBuildStats",
    "ClusterMethod",
    "RetrievalMethod",
    "Clusterer",
    "GMMClusterer",
    "KMeansClusterer",
    "LLMSummarizer",

    # === RAG Evaluation ===
    "RAGEvaluator",
    "QuickEvaluator",
    "EvaluationConfig",
    "EvaluationResult",
    "SampleEvaluation",
    "MetricResult",
    "Claim",
    "ContextRelevance",
    "FaithfulnessCalculator",
    "ContextRelevanceCalculator",
    "AnswerRelevancyCalculator",
    "AnswerCorrectnessCalculator",
    "HallucinationDetector",
    "ContextPrecisionCalculator",
    "ContextRecallCalculator",
    "ClaimsExtractor",
    "ClaimsVerifier",
    "RankingMetrics",
    "EvaluationPrompts",
    "create_evaluator",

    # === Agentic RAG ===
    "AgenticRAG",
    "AgenticRAGConfig",
    "AgenticRAGResult",
    "AgentState",
    "QueryType",
    "EvaluationDecision",
    "SubQuery",
    "RetrievedContext",
    "RetrievalResult",
    "GenerationResult",
    "QueryDecomposer",
    "ToolSelector",
    "ResponseEvaluator",
    "ResultFusion",
    "RetrievalTool",
    "BaseRetrievalTool",
    "ExaSearchTool",
    "TavilySearchTool",
    "MemorySearchTool",
    "CodeSearchTool",
    "AgenticRAGPrompts",
    "create_agentic_rag",

    # === GraphRAG ===
    "GraphRAG",
    "GraphRAGConfig",
    "Entity",
    "Relationship",
    "EntityType",
    "RelationshipType",
    "GraphSearchResult",
    "EntityExtractor",
    "GraphStorage",
    "GraphRAGTool",

    # === RAG Metrics ===
    "RAGMetricsConfig",
    "RetrievalSource",
    "LatencyPercentiles",
    "RetrievalMetrics",
    "QualityMetrics",
    "UsageMetrics",
    "SlowQueryRecord",
    "RAGMetricsCollector",
    "RetrievalTracker",
    "PrometheusExporter",
    "JSONExporter",
    "StructuredLogger",
    "DegradationAlert",
    "RAGAnalyzer",
    "RAGMetricsIntegration",
    "get_rag_metrics_collector",
    "create_rag_metrics_integration",

    # === Query Rewriter ===
    "RuleBasedQueryRewriter",
    "LLMQueryRewriter",
    "HybridQueryRewriter",
    "QueryRewriterConfig",
    "RewriteResult",
    "QueryIntent",
    "QueryRewriterPrompts",

    # === Unified Pipeline ===
    "RAGPipeline",
    "create_pipeline",
    "PipelineConfig",
    "PipelineResult",
    "PipelineStage",
    "StrategyType",
    "StageMetrics",
    "QueryClassifier",
    "RRFFusion",

    # === Context Manager ===
    "ContextManager",
    "ContextSelector",
    "ContextFormatter",
    "ContextCompressor",
    "TokenCounter",
    "ContextConfig",
    "ContextItem",
    "FormattedContext",
    "FormatStyle",
    "create_context_manager",
    "create_context_item",
    "count_tokens",

    # === ColBERT/RAGatouille ===
    "ColBERTConfig",
    "ColBERTDocument",
    "ColBERTResult",
    "ColBERTRetriever",
    "ColBERTReranker",
    "create_colbert_retriever",
    "create_colbert_reranker",

    # === Cache Warming ===
    "CacheWarmer",
    "PatternAnalyzer",
    "CacheWarmingMixin",
    "CacheWarmerConfig",
    "WarmingStrategy",
    "QueryAccess",
    "QueryPattern",
    "WarmingResult",
    "WarmingStats",
    "create_cache_warmer",
    "warm_pipeline_cache",

    # === Result Cache ===
    "ResultCache",
    "CachedRAGPipelineMixin",
    "ResultCacheConfig",
    "CacheEntry",
    "CacheHit",
    "CacheStats",
    "CacheLevel",
    "InvalidationStrategy",
    "QueryTypeTTL",
    "ExactMatchCache",
    "SemanticCache",
    "MemoryBudgetManager",
    "create_result_cache",

    # === Streaming ===
    "StreamingRAGResponse",
    "StreamingConfig",
    "StreamProgress",
    "StreamEvent",
    "StreamEventType",
    "StreamingStage",
    "CancellationToken",
    "CancellationError",
    "StreamingLLMProvider",
    "StreamingRetrieverProtocol",
    "StreamingPipelineMixin",
    "stream_multi_source_retrieval",
    "stream_reranking_top_first",
    "stream_llm_generation",
    "create_streaming_config",
    "create_cancellation_token",

    # === Complexity Analyzer ===
    "QueryComplexityAnalyzer",
    "ComplexityAwareRouter",
    "AnalyzerConfig",
    "AdaptiveTopKConfig",
    "QueryAnalysis",
    "ComplexityFactors",
    "CostEstimate",
    "RoutingRecommendation",
    "EntityInfo",
    "ComplexityLevel",
    "DomainType",
    "ModelTier",
    "EntityDetector",
    "QuestionClassifier",
    "DomainDetector",

    # === Hallucination Guard ===
    "HallucinationGuard",
    "GuardConfig",
    "HallucinationResult",
    "VerifiedClaim",
    "ClaimVerification",  # V50 alias for VerifiedClaim
    "ExtractedClaim",
    "ClaimVerdict",
    "HallucinationSeverity",
    "ClaimsExtractor",
    "ClaimVerifier",
    "ConfidenceScorer",
    "GuardPrompts",
    "GuardedPipelineMixin",
    "create_hallucination_guard",
    "quick_hallucination_check",

    # === Contextual Retrieval (Anthropic's Pattern) ===
    "ContextualRetriever",
    "ContextualSemanticChunker",
    "ContextualRetrievalConfig",
    "ContextualConfig",  # V50 alias for ContextualRetrievalConfig
    "ContextGenerationStrategy",
    "ContextualizedChunk",
    "ContextualChunk",  # V50 alias for ContextualizedChunk
    "DocumentContext",
    "ContextualizationStats",
    "LLMContextGenerator",
    "RuleBasedContextGenerator",
    "ContextCache",

    # === Dual-Level Retrieval (LightRAG-style) ===
    "DualLevelRetriever",
    "DualLevelConfig",
    "DualLevelDocument",
    "DualLevelResult",
    "DualLevelStrategy",
    "RetrievalLevel",
    "FusionStrategy",
    "DualLevelQueryComplexity",
    "EntityContext",
    "ResultFuser",
    "InterleaveFuser",
    "ScoreWeightedFuser",
    "RRFFuser",
    "LevelFirstFuser",
    "LevelLastFuser",
    "get_fuser",
    "DualLevelComplexityAnalyzer",
    "SimpleEntityExtractor",
    "RAPTORHighLevelRetriever",
    "create_dual_level_retriever",
    "create_dual_level_strategy",
]
