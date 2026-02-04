"""
LightRAG-Style Dual-Level Retrieval

This module implements dual-level retrieval inspired by LightRAG, combining:
- Low-level: Fine-grained chunk retrieval for specific facts and details
- High-level: RAPTOR summaries or entity-level retrieval for themes and concepts

The dual approach enables comprehensive answers for complex and comparative queries
by capturing both granular information and broader context.

Key Features:
- Parallel low and high-level retrieval paths
- Configurable fusion strategies (interleave, score-weighted, level-first)
- Integration with existing RAGPipeline as a strategy option
- Entity-aware high-level retrieval for relationship queries
- Automatic level selection based on query complexity

Reference: LightRAG paper concepts adapted for production use

Usage:
    from core.rag.dual_level_retrieval import (
        DualLevelRetriever,
        DualLevelConfig,
        RetrievalLevel,
    )

    retriever = DualLevelRetriever(
        low_level_retriever=chunk_retriever,
        high_level_retriever=raptor_retriever,
        config=DualLevelConfig(fusion_strategy=FusionStrategy.SCORE_WEIGHTED),
    )

    result = await retriever.retrieve("Compare microservices vs monolith", top_k=10)
    print(f"Low-level docs: {len(result.low_level_documents)}")
    print(f"High-level docs: {len(result.high_level_documents)}")
    print(f"Fused context: {result.context}")

Version: V1.0.0 (2026-02-04)
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from .raptor import RAPTOR, RAPTORResult, RAPTORNode
    from .pipeline import RAGPipeline, PipelineResult

logger = logging.getLogger(__name__)


# =============================================================================
# PROTOCOLS AND TYPES
# =============================================================================

class LowLevelRetrieverProtocol(Protocol):
    """Protocol for low-level (chunk) retrievers."""

    @property
    def name(self) -> str:
        """Retriever name."""
        ...

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Retrieve fine-grained chunks."""
        ...


class HighLevelRetrieverProtocol(Protocol):
    """Protocol for high-level (summary/entity) retrievers."""

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> Any:
        """Retrieve high-level summaries or entities."""
        ...


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def encode(self, texts: Union[str, List[str]]) -> Any:
        """Encode text(s) to embeddings."""
        ...


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

class RetrievalLevel(str, Enum):
    """Retrieval levels for dual-level architecture."""
    LOW = "low"           # Fine-grained chunks
    HIGH = "high"         # Summaries/entities
    BOTH = "both"         # Both levels (default)
    ADAPTIVE = "adaptive" # Automatic selection based on query


class FusionStrategy(str, Enum):
    """Strategies for fusing low and high-level results."""
    INTERLEAVE = "interleave"         # Alternate between levels
    SCORE_WEIGHTED = "score_weighted"  # Combine by normalized scores
    LEVEL_FIRST = "level_first"        # Low-level first, then high-level
    LEVEL_LAST = "level_last"          # High-level first, then low-level
    RRF = "rrf"                        # Reciprocal Rank Fusion


class QueryComplexity(str, Enum):
    """Query complexity levels for adaptive retrieval."""
    SIMPLE = "simple"           # Single fact lookup
    MODERATE = "moderate"       # Multi-fact or explanation
    COMPLEX = "complex"         # Comparative, analytical
    MULTI_HOP = "multi_hop"     # Requires reasoning chains


@dataclass
class DualLevelConfig:
    """Configuration for dual-level retrieval.

    Attributes:
        low_level_top_k: Number of chunks to retrieve at low level
        high_level_top_k: Number of summaries to retrieve at high level
        final_top_k: Final number of documents after fusion
        fusion_strategy: How to combine low and high-level results
        low_level_weight: Weight for low-level results (0-1)
        high_level_weight: Weight for high-level results (0-1)
        enable_entity_retrieval: Enable entity-based high-level retrieval
        entity_context_window: Context window around entities
        adaptive_threshold: Complexity score threshold for adaptive mode
        rrf_k: K parameter for RRF fusion
        parallel_retrieval: Retrieve both levels in parallel
        include_metadata: Include level metadata in results
    """
    low_level_top_k: int = 10
    high_level_top_k: int = 5
    final_top_k: int = 10
    fusion_strategy: FusionStrategy = FusionStrategy.SCORE_WEIGHTED
    low_level_weight: float = 0.6
    high_level_weight: float = 0.4
    enable_entity_retrieval: bool = True
    entity_context_window: int = 2
    adaptive_threshold: float = 0.5
    rrf_k: int = 60
    parallel_retrieval: bool = True
    include_metadata: bool = True


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DualLevelDocument:
    """A document from dual-level retrieval with level metadata.

    Attributes:
        id: Unique document identifier
        content: Document content
        score: Relevance score
        level: Retrieval level (low or high)
        source: Source retriever name
        metadata: Additional metadata
        entities: Extracted entities (if applicable)
    """
    id: str
    content: str
    score: float
    level: RetrievalLevel
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    entities: List[str] = field(default_factory=list)

    @property
    def token_estimate(self) -> int:
        """Estimate token count (rough: ~4 chars per token)."""
        return len(self.content) // 4


@dataclass
class DualLevelResult:
    """Result from dual-level retrieval.

    Attributes:
        query: Original query
        low_level_documents: Documents from low-level retrieval
        high_level_documents: Documents from high-level retrieval
        fused_documents: Final fused document list
        context: Combined context string
        query_complexity: Detected query complexity
        levels_used: Which levels were actually used
        latency_ms: Total retrieval latency
        metadata: Additional result metadata
    """
    query: str
    low_level_documents: List[DualLevelDocument] = field(default_factory=list)
    high_level_documents: List[DualLevelDocument] = field(default_factory=list)
    fused_documents: List[DualLevelDocument] = field(default_factory=list)
    context: str = ""
    query_complexity: QueryComplexity = QueryComplexity.MODERATE
    levels_used: List[RetrievalLevel] = field(default_factory=list)
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_documents(self) -> int:
        """Total documents across both levels."""
        return len(self.low_level_documents) + len(self.high_level_documents)

    @property
    def fused_count(self) -> int:
        """Number of documents after fusion."""
        return len(self.fused_documents)


@dataclass
class EntityContext:
    """Context extracted around an entity.

    Attributes:
        entity: Entity name or identifier
        entity_type: Type of entity
        context: Text context around the entity
        relationships: Related entities
        score: Relevance score
    """
    entity: str
    entity_type: str
    context: str
    relationships: List[str] = field(default_factory=list)
    score: float = 0.0


# =============================================================================
# QUERY COMPLEXITY ANALYZER
# =============================================================================

class QueryComplexityAnalyzer:
    """Analyzes query complexity to determine optimal retrieval levels.

    Uses rule-based heuristics and optional embedding similarity to classify
    queries into complexity categories.
    """

    # Complexity indicators
    COMPLEX_INDICATORS = [
        "compare", "difference", "versus", "vs", "better",
        "contrast", "similarities", "analyze", "evaluate",
    ]
    MULTI_HOP_INDICATORS = [
        "and then", "after", "because of", "leads to",
        "causes", "results in", "how does", "why does",
    ]
    SIMPLE_INDICATORS = [
        "what is", "who is", "when", "where", "define",
    ]

    def __init__(
        self,
        embedder: Optional[EmbeddingProvider] = None,
        threshold: float = 0.5,
    ):
        """Initialize complexity analyzer.

        Args:
            embedder: Optional embedding provider for semantic analysis
            threshold: Complexity score threshold
        """
        self.embedder = embedder
        self.threshold = threshold

    def analyze(self, query: str) -> Tuple[QueryComplexity, float]:
        """Analyze query complexity.

        Args:
            query: Query to analyze

        Returns:
            Tuple of (complexity level, confidence score)
        """
        query_lower = query.lower()
        score = 0.5  # Base score

        # Check for simple indicators
        for indicator in self.SIMPLE_INDICATORS:
            if indicator in query_lower:
                score -= 0.2
                break

        # Check for complex indicators
        complex_count = sum(
            1 for ind in self.COMPLEX_INDICATORS
            if ind in query_lower
        )
        score += complex_count * 0.15

        # Check for multi-hop indicators
        multi_hop_count = sum(
            1 for ind in self.MULTI_HOP_INDICATORS
            if ind in query_lower
        )
        score += multi_hop_count * 0.2

        # Length-based adjustment
        word_count = len(query.split())
        if word_count > 15:
            score += 0.1
        elif word_count < 5:
            score -= 0.1

        # Question mark count (multiple questions = complex)
        question_count = query.count("?")
        if question_count > 1:
            score += 0.15

        # Clamp score
        score = max(0.0, min(1.0, score))

        # Determine complexity level
        if score < 0.3:
            complexity = QueryComplexity.SIMPLE
        elif score < 0.5:
            complexity = QueryComplexity.MODERATE
        elif score < 0.7:
            complexity = QueryComplexity.COMPLEX
        else:
            complexity = QueryComplexity.MULTI_HOP

        return complexity, score


# =============================================================================
# ENTITY EXTRACTOR
# =============================================================================

class SimpleEntityExtractor:
    """Simple entity extraction using pattern matching.

    For production use, consider integrating with spaCy or other NER models.
    """

    def __init__(self):
        """Initialize entity extractor."""
        self._cache: Dict[str, List[str]] = {}

    def extract(self, text: str) -> List[str]:
        """Extract entities from text.

        Args:
            text: Text to extract entities from

        Returns:
            List of extracted entity strings
        """
        # Cache lookup
        cache_key = text[:100]
        if cache_key in self._cache:
            return self._cache[cache_key]

        entities = []

        # Simple heuristic: capitalized words/phrases
        words = text.split()
        current_entity = []

        for word in words:
            # Skip sentence starters
            if word.endswith((".", "!", "?", ":")):
                if current_entity:
                    entity = " ".join(current_entity)
                    if len(entity) > 2:
                        entities.append(entity)
                    current_entity = []
                continue

            # Check if word is capitalized (potential entity)
            if word and word[0].isupper() and not word.isupper():
                current_entity.append(word)
            else:
                if current_entity:
                    entity = " ".join(current_entity)
                    if len(entity) > 2:
                        entities.append(entity)
                    current_entity = []

        # Handle remaining
        if current_entity:
            entity = " ".join(current_entity)
            if len(entity) > 2:
                entities.append(entity)

        # Deduplicate
        entities = list(dict.fromkeys(entities))

        # Cache result
        self._cache[cache_key] = entities

        return entities


# =============================================================================
# RESULT FUSERS
# =============================================================================

class ResultFuser(ABC):
    """Abstract base for result fusion strategies."""

    @abstractmethod
    def fuse(
        self,
        low_level: List[DualLevelDocument],
        high_level: List[DualLevelDocument],
        top_k: int,
    ) -> List[DualLevelDocument]:
        """Fuse low and high-level results.

        Args:
            low_level: Low-level documents
            high_level: High-level documents
            top_k: Number of results to return

        Returns:
            Fused document list
        """
        ...


class InterleaveFuser(ResultFuser):
    """Interleave results from both levels."""

    def fuse(
        self,
        low_level: List[DualLevelDocument],
        high_level: List[DualLevelDocument],
        top_k: int,
    ) -> List[DualLevelDocument]:
        """Interleave low and high-level results."""
        fused = []
        low_idx = 0
        high_idx = 0

        while len(fused) < top_k and (low_idx < len(low_level) or high_idx < len(high_level)):
            # Alternate: low, high, low, high...
            if low_idx < len(low_level):
                fused.append(low_level[low_idx])
                low_idx += 1

            if len(fused) >= top_k:
                break

            if high_idx < len(high_level):
                fused.append(high_level[high_idx])
                high_idx += 1

        return fused[:top_k]


class ScoreWeightedFuser(ResultFuser):
    """Fuse by normalized and weighted scores."""

    def __init__(
        self,
        low_weight: float = 0.6,
        high_weight: float = 0.4,
    ):
        """Initialize score-weighted fuser.

        Args:
            low_weight: Weight for low-level results
            high_weight: Weight for high-level results
        """
        self.low_weight = low_weight
        self.high_weight = high_weight

    def fuse(
        self,
        low_level: List[DualLevelDocument],
        high_level: List[DualLevelDocument],
        top_k: int,
    ) -> List[DualLevelDocument]:
        """Fuse by weighted scores."""
        # Normalize scores within each level
        low_normalized = self._normalize_scores(low_level, self.low_weight)
        high_normalized = self._normalize_scores(high_level, self.high_weight)

        # Combine and sort by adjusted score
        all_docs = low_normalized + high_normalized
        all_docs.sort(key=lambda d: d.score, reverse=True)

        return all_docs[:top_k]

    def _normalize_scores(
        self,
        docs: List[DualLevelDocument],
        weight: float,
    ) -> List[DualLevelDocument]:
        """Normalize scores to 0-1 range and apply weight."""
        if not docs:
            return []

        max_score = max(d.score for d in docs)
        min_score = min(d.score for d in docs)
        score_range = max_score - min_score if max_score > min_score else 1.0

        normalized = []
        for doc in docs:
            norm_score = ((doc.score - min_score) / score_range) * weight
            normalized.append(DualLevelDocument(
                id=doc.id,
                content=doc.content,
                score=norm_score,
                level=doc.level,
                source=doc.source,
                metadata={
                    **doc.metadata,
                    "original_score": doc.score,
                    "weight_applied": weight,
                },
                entities=doc.entities,
            ))

        return normalized


class RRFFuser(ResultFuser):
    """Reciprocal Rank Fusion for combining result lists."""

    def __init__(self, k: int = 60):
        """Initialize RRF fuser.

        Args:
            k: RRF constant (typically 60)
        """
        self.k = k

    def fuse(
        self,
        low_level: List[DualLevelDocument],
        high_level: List[DualLevelDocument],
        top_k: int,
    ) -> List[DualLevelDocument]:
        """Fuse using RRF."""
        scores: Dict[str, Tuple[DualLevelDocument, float]] = {}

        # Process low-level
        for rank, doc in enumerate(low_level, start=1):
            rrf_score = 1.0 / (self.k + rank)
            key = doc.content[:100]  # Content-based dedup
            if key in scores:
                existing_doc, existing_score = scores[key]
                scores[key] = (existing_doc, existing_score + rrf_score)
            else:
                scores[key] = (doc, rrf_score)

        # Process high-level
        for rank, doc in enumerate(high_level, start=1):
            rrf_score = 1.0 / (self.k + rank)
            key = doc.content[:100]
            if key in scores:
                existing_doc, existing_score = scores[key]
                scores[key] = (existing_doc, existing_score + rrf_score)
            else:
                scores[key] = (doc, rrf_score)

        # Sort by RRF score
        sorted_items = sorted(scores.values(), key=lambda x: x[1], reverse=True)

        fused = []
        for doc, rrf_score in sorted_items[:top_k]:
            fused.append(DualLevelDocument(
                id=doc.id,
                content=doc.content,
                score=rrf_score,
                level=doc.level,
                source=doc.source,
                metadata={**doc.metadata, "rrf_score": rrf_score},
                entities=doc.entities,
            ))

        return fused


class LevelFirstFuser(ResultFuser):
    """Low-level first, then high-level."""

    def fuse(
        self,
        low_level: List[DualLevelDocument],
        high_level: List[DualLevelDocument],
        top_k: int,
    ) -> List[DualLevelDocument]:
        """Return low-level first, then high-level."""
        return (low_level + high_level)[:top_k]


class LevelLastFuser(ResultFuser):
    """High-level first, then low-level."""

    def fuse(
        self,
        low_level: List[DualLevelDocument],
        high_level: List[DualLevelDocument],
        top_k: int,
    ) -> List[DualLevelDocument]:
        """Return high-level first, then low-level."""
        return (high_level + low_level)[:top_k]


def get_fuser(
    strategy: FusionStrategy,
    config: DualLevelConfig,
) -> ResultFuser:
    """Factory function for result fusers.

    Args:
        strategy: Fusion strategy
        config: Dual-level configuration

    Returns:
        Appropriate ResultFuser instance
    """
    if strategy == FusionStrategy.INTERLEAVE:
        return InterleaveFuser()
    elif strategy == FusionStrategy.SCORE_WEIGHTED:
        return ScoreWeightedFuser(
            low_weight=config.low_level_weight,
            high_weight=config.high_level_weight,
        )
    elif strategy == FusionStrategy.RRF:
        return RRFFuser(k=config.rrf_k)
    elif strategy == FusionStrategy.LEVEL_FIRST:
        return LevelFirstFuser()
    elif strategy == FusionStrategy.LEVEL_LAST:
        return LevelLastFuser()
    else:
        return ScoreWeightedFuser()


# =============================================================================
# DUAL LEVEL RETRIEVER
# =============================================================================

class DualLevelRetriever:
    """
    LightRAG-style dual-level retrieval combining low and high-level approaches.

    This retriever enables comprehensive answers for complex queries by:
    1. Low-level: Fine-grained chunk retrieval for specific facts and details
    2. High-level: RAPTOR summaries or entity-level retrieval for themes

    The results are fused using configurable strategies to provide optimal
    context for answer generation.

    Example:
        >>> from core.rag.dual_level_retrieval import (
        ...     DualLevelRetriever,
        ...     DualLevelConfig,
        ...     FusionStrategy,
        ... )
        >>>
        >>> retriever = DualLevelRetriever(
        ...     low_level_retriever=exa_adapter,
        ...     high_level_retriever=raptor,
        ...     config=DualLevelConfig(fusion_strategy=FusionStrategy.SCORE_WEIGHTED),
        ... )
        >>>
        >>> result = await retriever.retrieve(
        ...     "Compare microservices architecture with monolithic design",
        ...     top_k=10,
        ... )
        >>> print(f"Context: {result.context[:500]}...")

    Integration with RAGPipeline:
        >>> from core.rag import RAGPipeline, StrategyType
        >>>
        >>> pipeline = RAGPipeline(llm=my_llm)
        >>> pipeline.register_implementation("dual_level", dual_level_retriever)
        >>> pipeline.set_strategy(QueryType.COMPARISON, "dual_level")
    """

    def __init__(
        self,
        low_level_retriever: Optional[LowLevelRetrieverProtocol] = None,
        high_level_retriever: Optional[HighLevelRetrieverProtocol] = None,
        config: Optional[DualLevelConfig] = None,
        embedder: Optional[EmbeddingProvider] = None,
    ):
        """Initialize dual-level retriever.

        Args:
            low_level_retriever: Retriever for fine-grained chunks
            high_level_retriever: Retriever for summaries/entities (e.g., RAPTOR)
            config: Configuration options
            embedder: Optional embedding provider for similarity calculations
        """
        self.low_level_retriever = low_level_retriever
        self.high_level_retriever = high_level_retriever
        self.config = config or DualLevelConfig()
        self.embedder = embedder

        # Initialize components
        self.complexity_analyzer = QueryComplexityAnalyzer(
            embedder=embedder,
            threshold=self.config.adaptive_threshold,
        )
        self.entity_extractor = SimpleEntityExtractor()
        self.fuser = get_fuser(self.config.fusion_strategy, self.config)

    @property
    def name(self) -> str:
        """Retriever name for pipeline identification."""
        return "dual_level"

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        level: RetrievalLevel = RetrievalLevel.BOTH,
        **kwargs
    ) -> DualLevelResult:
        """Retrieve using dual-level approach.

        Args:
            query: Search query
            top_k: Number of results to return
            level: Which levels to use
            **kwargs: Additional retriever arguments

        Returns:
            DualLevelResult with documents from both levels
        """
        start_time = time.time()
        top_k = top_k or self.config.final_top_k

        # Analyze query complexity
        complexity, complexity_score = self.complexity_analyzer.analyze(query)

        # Determine which levels to use
        if level == RetrievalLevel.ADAPTIVE:
            level = self._adaptive_level_selection(complexity, complexity_score)

        # Initialize result
        result = DualLevelResult(
            query=query,
            query_complexity=complexity,
            levels_used=[],
        )

        # Retrieve from appropriate levels
        if level in (RetrievalLevel.LOW, RetrievalLevel.BOTH):
            result.levels_used.append(RetrievalLevel.LOW)
        if level in (RetrievalLevel.HIGH, RetrievalLevel.BOTH):
            result.levels_used.append(RetrievalLevel.HIGH)

        if self.config.parallel_retrieval and level == RetrievalLevel.BOTH:
            # Parallel retrieval
            low_task = self._retrieve_low_level(query, **kwargs)
            high_task = self._retrieve_high_level(query, **kwargs)
            low_docs, high_docs = await asyncio.gather(low_task, high_task)
            result.low_level_documents = low_docs
            result.high_level_documents = high_docs
        else:
            # Sequential retrieval
            if RetrievalLevel.LOW in result.levels_used:
                result.low_level_documents = await self._retrieve_low_level(
                    query, **kwargs
                )
            if RetrievalLevel.HIGH in result.levels_used:
                result.high_level_documents = await self._retrieve_high_level(
                    query, **kwargs
                )

        # Fuse results
        result.fused_documents = self.fuser.fuse(
            result.low_level_documents,
            result.high_level_documents,
            top_k,
        )

        # Build context
        result.context = self._build_context(result.fused_documents)

        # Calculate latency
        result.latency_ms = (time.time() - start_time) * 1000

        # Add metadata
        result.metadata = {
            "complexity_score": complexity_score,
            "fusion_strategy": self.config.fusion_strategy.value,
            "low_level_count": len(result.low_level_documents),
            "high_level_count": len(result.high_level_documents),
            "fused_count": len(result.fused_documents),
        }

        return result

    async def _retrieve_low_level(
        self,
        query: str,
        **kwargs
    ) -> List[DualLevelDocument]:
        """Retrieve fine-grained chunks.

        Args:
            query: Search query
            **kwargs: Additional arguments

        Returns:
            List of low-level documents
        """
        if not self.low_level_retriever:
            logger.debug("No low-level retriever configured")
            return []

        try:
            results = await self.low_level_retriever.retrieve(
                query,
                top_k=self.config.low_level_top_k,
                **kwargs
            )

            documents = []
            for i, result in enumerate(results):
                if isinstance(result, dict):
                    content = result.get("content", str(result))
                    score = result.get("score", 1.0 / (i + 1))
                    metadata = result.get("metadata", {})
                else:
                    content = str(result)
                    score = 1.0 / (i + 1)
                    metadata = {}

                # Extract entities if enabled
                entities = []
                if self.config.enable_entity_retrieval:
                    entities = self.entity_extractor.extract(content)

                doc = DualLevelDocument(
                    id=f"low_{i}",
                    content=content,
                    score=score,
                    level=RetrievalLevel.LOW,
                    source=getattr(self.low_level_retriever, "name", "low_level"),
                    metadata=metadata,
                    entities=entities,
                )
                documents.append(doc)

            return documents

        except Exception as e:
            logger.warning(f"Low-level retrieval failed: {e}")
            return []

    async def _retrieve_high_level(
        self,
        query: str,
        **kwargs
    ) -> List[DualLevelDocument]:
        """Retrieve summaries or entity-level context.

        Args:
            query: Search query
            **kwargs: Additional arguments

        Returns:
            List of high-level documents
        """
        if not self.high_level_retriever:
            logger.debug("No high-level retriever configured")
            return []

        try:
            # Check if retriever is RAPTOR
            if hasattr(self.high_level_retriever, "retrieve"):
                result = await self.high_level_retriever.retrieve(
                    query,
                    top_k=self.config.high_level_top_k,
                    **kwargs
                )

                # Handle RAPTOR result
                if hasattr(result, "nodes"):
                    return self._convert_raptor_result(result)

                # Handle list of documents
                if isinstance(result, list):
                    return self._convert_list_result(result)

                # Handle dict result
                if isinstance(result, dict):
                    return self._convert_dict_result(result)

            return []

        except Exception as e:
            logger.warning(f"High-level retrieval failed: {e}")
            return []

    def _convert_raptor_result(self, result: Any) -> List[DualLevelDocument]:
        """Convert RAPTOR result to DualLevelDocuments."""
        documents = []

        for i, (node, score) in enumerate(result.nodes):
            doc = DualLevelDocument(
                id=f"high_{node.id}" if hasattr(node, "id") else f"high_{i}",
                content=node.content if hasattr(node, "content") else str(node),
                score=score,
                level=RetrievalLevel.HIGH,
                source="raptor",
                metadata={
                    "tree_level": node.level if hasattr(node, "level") else 0,
                    "is_summary": node.is_summary if hasattr(node, "is_summary") else True,
                },
            )
            documents.append(doc)

        return documents

    def _convert_list_result(self, results: List[Any]) -> List[DualLevelDocument]:
        """Convert list result to DualLevelDocuments."""
        documents = []

        for i, result in enumerate(results):
            if isinstance(result, dict):
                content = result.get("content", str(result))
                score = result.get("score", 1.0 / (i + 1))
                metadata = result.get("metadata", {})
            else:
                content = str(result)
                score = 1.0 / (i + 1)
                metadata = {}

            doc = DualLevelDocument(
                id=f"high_{i}",
                content=content,
                score=score,
                level=RetrievalLevel.HIGH,
                source="high_level",
                metadata=metadata,
            )
            documents.append(doc)

        return documents

    def _convert_dict_result(self, result: Dict[str, Any]) -> List[DualLevelDocument]:
        """Convert dict result to DualLevelDocuments."""
        documents = []

        # Check for common patterns
        if "documents" in result:
            return self._convert_list_result(result["documents"])
        if "nodes" in result:
            return self._convert_list_result(result["nodes"])
        if "context" in result:
            doc = DualLevelDocument(
                id="high_0",
                content=result["context"],
                score=result.get("score", 1.0),
                level=RetrievalLevel.HIGH,
                source="high_level",
                metadata=result.get("metadata", {}),
            )
            return [doc]

        return documents

    def _adaptive_level_selection(
        self,
        complexity: QueryComplexity,
        score: float,
    ) -> RetrievalLevel:
        """Select retrieval level based on query complexity.

        Args:
            complexity: Detected complexity level
            score: Complexity score

        Returns:
            Appropriate retrieval level
        """
        if complexity == QueryComplexity.SIMPLE:
            return RetrievalLevel.LOW
        elif complexity == QueryComplexity.MODERATE:
            return RetrievalLevel.BOTH
        else:
            # Complex or multi-hop - favor high-level but include both
            return RetrievalLevel.BOTH

    def _build_context(self, documents: List[DualLevelDocument]) -> str:
        """Build context string from fused documents.

        Args:
            documents: Fused documents

        Returns:
            Combined context string
        """
        if not documents:
            return ""

        context_parts = []
        for doc in documents:
            level_tag = "[Summary]" if doc.level == RetrievalLevel.HIGH else "[Detail]"
            if self.config.include_metadata:
                context_parts.append(f"{level_tag}\n{doc.content}")
            else:
                context_parts.append(doc.content)

        return "\n\n---\n\n".join(context_parts)

    def set_low_level_retriever(
        self,
        retriever: LowLevelRetrieverProtocol
    ) -> None:
        """Set the low-level retriever.

        Args:
            retriever: Low-level retriever implementation
        """
        self.low_level_retriever = retriever

    def set_high_level_retriever(
        self,
        retriever: HighLevelRetrieverProtocol
    ) -> None:
        """Set the high-level retriever.

        Args:
            retriever: High-level retriever implementation (e.g., RAPTOR)
        """
        self.high_level_retriever = retriever

    def update_fusion_strategy(self, strategy: FusionStrategy) -> None:
        """Update the fusion strategy.

        Args:
            strategy: New fusion strategy
        """
        self.config.fusion_strategy = strategy
        self.fuser = get_fuser(strategy, self.config)


# =============================================================================
# RAPTOR INTEGRATION
# =============================================================================

class RAPTORHighLevelRetriever:
    """Wrapper for RAPTOR as a high-level retriever.

    Provides seamless integration of RAPTOR with the dual-level architecture.
    """

    def __init__(
        self,
        raptor: Any,  # RAPTOR instance
        method: str = "collapsed",
    ):
        """Initialize RAPTOR wrapper.

        Args:
            raptor: RAPTOR instance
            method: Retrieval method (collapsed, tree_traversal, hybrid)
        """
        self.raptor = raptor
        self.method = method

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> Any:
        """Retrieve from RAPTOR tree.

        Args:
            query: Search query
            top_k: Number of results
            **kwargs: Additional arguments

        Returns:
            RAPTORResult
        """
        return await self.raptor.retrieve(
            query,
            method=self.method,
            top_k=top_k,
            **kwargs
        )


# =============================================================================
# PIPELINE INTEGRATION
# =============================================================================

class DualLevelStrategy:
    """Strategy implementation for RAGPipeline integration.

    Implements the strategy interface expected by RAGPipeline.
    """

    def __init__(
        self,
        dual_level_retriever: DualLevelRetriever,
        llm: Optional[Any] = None,
    ):
        """Initialize dual-level strategy.

        Args:
            dual_level_retriever: DualLevelRetriever instance
            llm: Optional LLM for generation
        """
        self.retriever = dual_level_retriever
        self.llm = llm

    async def run(
        self,
        query: str,
        **kwargs
    ) -> DualLevelResult:
        """Run dual-level retrieval.

        Args:
            query: Search query
            **kwargs: Additional arguments

        Returns:
            DualLevelResult
        """
        return await self.retriever.retrieve(query, **kwargs)

    async def generate(
        self,
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using dual-level context.

        Args:
            query: User query
            **kwargs: Additional arguments

        Returns:
            Generation result dict
        """
        result = await self.retriever.retrieve(query, **kwargs)

        if self.llm:
            prompt = f"""Answer the question using the provided context.
The context includes both detailed information [Detail] and summaries [Summary].

Question: {query}

Context:
{result.context}

Answer:"""

            response = await self.llm.generate(prompt)
            return {
                "response": response,
                "context": result.context,
                "documents": result.fused_documents,
                "confidence": 0.8 if result.fused_documents else 0.3,
            }

        return {
            "response": result.context,
            "documents": result.fused_documents,
            "confidence": 0.5,
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_dual_level_retriever(
    low_level: Optional[LowLevelRetrieverProtocol] = None,
    high_level: Optional[HighLevelRetrieverProtocol] = None,
    raptor: Optional[Any] = None,
    fusion_strategy: FusionStrategy = FusionStrategy.SCORE_WEIGHTED,
    low_level_weight: float = 0.6,
    high_level_weight: float = 0.4,
    **kwargs
) -> DualLevelRetriever:
    """Factory function to create a dual-level retriever.

    Args:
        low_level: Low-level (chunk) retriever
        high_level: High-level retriever (overrides raptor if both provided)
        raptor: RAPTOR instance for high-level retrieval
        fusion_strategy: How to combine results
        low_level_weight: Weight for low-level results
        high_level_weight: Weight for high-level results
        **kwargs: Additional config parameters

    Returns:
        Configured DualLevelRetriever
    """
    config = DualLevelConfig(
        fusion_strategy=fusion_strategy,
        low_level_weight=low_level_weight,
        high_level_weight=high_level_weight,
        **kwargs
    )

    # Use RAPTOR as high-level if provided and no explicit high_level
    if raptor is not None and high_level is None:
        high_level = RAPTORHighLevelRetriever(raptor)

    return DualLevelRetriever(
        low_level_retriever=low_level,
        high_level_retriever=high_level,
        config=config,
    )


def create_dual_level_strategy(
    dual_level_retriever: DualLevelRetriever,
    llm: Optional[Any] = None,
) -> DualLevelStrategy:
    """Factory function to create a dual-level strategy for RAGPipeline.

    Args:
        dual_level_retriever: DualLevelRetriever instance
        llm: Optional LLM for generation

    Returns:
        DualLevelStrategy for pipeline integration
    """
    return DualLevelStrategy(dual_level_retriever, llm)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "DualLevelConfig",
    # Enums
    "RetrievalLevel",
    "FusionStrategy",
    "QueryComplexity",
    # Data types
    "DualLevelDocument",
    "DualLevelResult",
    "EntityContext",
    # Main retriever
    "DualLevelRetriever",
    # Fusion strategies
    "ResultFuser",
    "InterleaveFuser",
    "ScoreWeightedFuser",
    "RRFFuser",
    "LevelFirstFuser",
    "LevelLastFuser",
    "get_fuser",
    # Components
    "QueryComplexityAnalyzer",
    "SimpleEntityExtractor",
    # Integrations
    "RAPTORHighLevelRetriever",
    "DualLevelStrategy",
    # Factory functions
    "create_dual_level_retriever",
    "create_dual_level_strategy",
    # Protocols
    "LowLevelRetrieverProtocol",
    "HighLevelRetrieverProtocol",
]
