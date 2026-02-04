"""
Semantic Memory Compression System - V36 Architecture

This module provides intelligent compression for long-term memory storage,
enabling efficient storage while preserving key information and relationships.

Features:
- Multiple compression strategies (extractive, abstractive, hierarchical, clustering)
- Semantic summarization preserving facts and relationships
- Original memory linking for drill-down access
- Lazy decompression with caching
- Background compression scheduler
- Compression quality metrics

Integration:
    from core.memory.compression import (
        MemoryCompressor,
        CompressionStrategy,
        CompressionConfig,
        CompressionResult,
        CompressionMetrics,
        BackgroundCompressor,
    )

Usage:
    compressor = MemoryCompressor(backend)

    # Compress a group of related memories
    result = await compressor.compress_memories(memory_ids)

    # Decompress on access
    memories = await compressor.decompress(compressed_id)

    # Run background compression
    scheduler = BackgroundCompressor(compressor)
    await scheduler.start()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .backends.sqlite import SQLiteTierBackend
    from .backends.base import MemoryEntry

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

class CompressionStrategy(str, Enum):
    """Compression strategies for memory consolidation."""
    EXTRACTIVE = "extractive"      # Select key sentences from originals
    ABSTRACTIVE = "abstractive"    # Generate new summary via LLM
    HIERARCHICAL = "hierarchical"  # Multi-level compression tree
    CLUSTERING = "clustering"      # Group by semantic similarity


class CompressionTrigger(str, Enum):
    """Triggers for compression operations."""
    AGE = "age"                    # Memory age threshold
    ACCESS = "access"              # Low access frequency
    SIMILARITY = "similarity"      # High semantic overlap
    MEMORY_PRESSURE = "pressure"   # Storage constraints
    SCHEDULED = "scheduled"        # Scheduled job
    MANUAL = "manual"              # User-initiated


@dataclass
class CompressionConfig:
    """Configuration for the compression system."""
    # Strategy selection
    default_strategy: CompressionStrategy = CompressionStrategy.EXTRACTIVE

    # Candidate selection thresholds
    min_age_days: float = 7.0              # Minimum age for compression candidates
    max_access_count: int = 5              # Max accesses to be candidate
    min_access_age_days: float = 3.0       # Days since last access
    similarity_threshold: float = 0.7      # Semantic similarity for grouping

    # Compression settings
    min_group_size: int = 3                # Minimum memories to compress together
    max_group_size: int = 50               # Maximum memories per compressed unit
    target_compression_ratio: float = 0.3  # Target 70% reduction
    preserve_high_importance: bool = True  # Keep important memories uncompressed
    importance_threshold: float = 0.8      # Importance threshold to preserve

    # Extractive settings
    max_key_sentences: int = 5             # Max sentences for extractive
    sentence_scoring_method: str = "tfidf" # tfidf, position, frequency

    # Hierarchical settings
    hierarchy_levels: int = 3              # Levels in hierarchy
    level_compression_ratio: float = 0.5   # Compression per level

    # Decompression settings
    cache_decompressed: bool = True        # Cache decompressed content
    cache_ttl_seconds: int = 300           # 5 minute cache
    lazy_load_depth: int = 1               # How deep to lazy load

    # Background scheduler settings
    schedule_interval_hours: float = 24.0  # Run every 24 hours
    memory_pressure_threshold: float = 0.8 # Trigger at 80% capacity
    max_compression_batch: int = 100       # Max memories per batch

    # Quality settings
    min_retention_score: float = 0.7       # Minimum information retention


@dataclass
class CompressionResult:
    """Result of a compression operation."""
    compressed_id: str
    original_ids: List[str]
    strategy: CompressionStrategy

    # Compression details
    original_token_count: int
    compressed_token_count: int
    compression_ratio: float

    # Quality metrics
    retention_score: float           # 0-1, how much info preserved
    coherence_score: float           # 0-1, how coherent is summary
    coverage_score: float            # 0-1, topic coverage

    # Metadata
    compressed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trigger: CompressionTrigger = CompressionTrigger.MANUAL

    # Links to originals (for drill-down)
    archived_locations: Dict[str, str] = field(default_factory=dict)  # id -> location

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "compressed_id": self.compressed_id,
            "original_ids": self.original_ids,
            "strategy": self.strategy.value,
            "original_token_count": self.original_token_count,
            "compressed_token_count": self.compressed_token_count,
            "compression_ratio": round(self.compression_ratio, 4),
            "retention_score": round(self.retention_score, 4),
            "coherence_score": round(self.coherence_score, 4),
            "coverage_score": round(self.coverage_score, 4),
            "compressed_at": self.compressed_at.isoformat(),
            "trigger": self.trigger.value,
            "archived_locations": self.archived_locations,
        }


@dataclass
class CompressionMetrics:
    """Metrics for compression system performance."""
    total_compressions: int = 0
    total_original_tokens: int = 0
    total_compressed_tokens: int = 0
    average_compression_ratio: float = 0.0
    average_retention_score: float = 0.0

    # By strategy breakdown
    compressions_by_strategy: Dict[str, int] = field(default_factory=dict)

    # Decompression stats
    total_decompressions: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0

    # Timing
    average_compression_time_ms: float = 0.0
    average_decompression_time_ms: float = 0.0

    # Quality distribution
    quality_distribution: Dict[str, int] = field(default_factory=dict)  # excellent/good/fair/poor

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_compressions": self.total_compressions,
            "total_original_tokens": self.total_original_tokens,
            "total_compressed_tokens": self.total_compressed_tokens,
            "average_compression_ratio": round(self.average_compression_ratio, 4),
            "average_retention_score": round(self.average_retention_score, 4),
            "compressions_by_strategy": self.compressions_by_strategy,
            "total_decompressions": self.total_decompressions,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "average_compression_time_ms": round(self.average_compression_time_ms, 2),
            "average_decompression_time_ms": round(self.average_decompression_time_ms, 2),
            "quality_distribution": self.quality_distribution,
        }


@dataclass
class CompressedMemory:
    """A compressed memory unit containing summarized content."""
    id: str
    summary: str
    key_facts: List[str]
    key_relationships: List[Dict[str, str]]

    # Original references
    original_ids: List[str]
    original_hashes: Dict[str, str]  # id -> content_hash for verification

    # Hierarchy (for hierarchical compression)
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    level: int = 0

    # Metadata
    strategy: CompressionStrategy = CompressionStrategy.EXTRACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    token_count: int = 0

    # Tags aggregated from originals
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# COMPRESSION STRATEGIES (ABSTRACT BASE)
# =============================================================================

class CompressionStrategyBase(ABC):
    """Abstract base class for compression strategies."""

    @abstractmethod
    async def compress(
        self,
        memories: List["MemoryEntry"],
        config: CompressionConfig
    ) -> Tuple[str, List[str], List[Dict[str, str]]]:
        """
        Compress a list of memories.

        Args:
            memories: List of memory entries to compress
            config: Compression configuration

        Returns:
            Tuple of (summary_text, key_facts, key_relationships)
        """
        pass

    @abstractmethod
    def score_retention(
        self,
        original_memories: List["MemoryEntry"],
        compressed_summary: str,
        key_facts: List[str]
    ) -> float:
        """
        Score how much information was retained.

        Returns:
            Score from 0.0 to 1.0
        """
        pass


class ExtractiveStrategy(CompressionStrategyBase):
    """
    Extractive compression: Select key sentences from originals.

    Fast, preserves original wording, good for factual content.
    """

    async def compress(
        self,
        memories: List["MemoryEntry"],
        config: CompressionConfig
    ) -> Tuple[str, List[str], List[Dict[str, str]]]:
        """Extract key sentences from memories."""
        # Collect all sentences
        sentences: List[Tuple[str, float, str]] = []  # (sentence, score, memory_id)

        for memory in memories:
            memory_sentences = self._split_sentences(memory.content)
            for i, sentence in enumerate(memory_sentences):
                score = self._score_sentence(
                    sentence,
                    i,
                    len(memory_sentences),
                    memory.access_count,
                    config.sentence_scoring_method
                )
                sentences.append((sentence, score, memory.id))

        # Sort by score and select top N
        sentences.sort(key=lambda x: x[1], reverse=True)
        selected = sentences[:config.max_key_sentences]

        # Build summary
        summary_parts = [s[0] for s in selected]
        summary = " ".join(summary_parts)

        # Extract key facts (sentences with highest scores)
        key_facts = [s[0] for s in selected[:min(5, len(selected))]]

        # Extract relationships (simple pattern matching)
        relationships = self._extract_relationships(summary)

        return summary, key_facts, relationships

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s) > 10]

    def _score_sentence(
        self,
        sentence: str,
        position: int,
        total_sentences: int,
        access_count: int,
        method: str
    ) -> float:
        """Score a sentence for importance."""
        score = 0.0

        # Position score (first and last sentences often important)
        if position == 0:
            score += 0.3
        elif position == total_sentences - 1:
            score += 0.2
        else:
            score += 0.1 * (1 - position / total_sentences)

        # Length score (moderate length preferred)
        words = len(sentence.split())
        if 10 <= words <= 30:
            score += 0.2
        elif words < 10:
            score += 0.1
        else:
            score += 0.05

        # Keyword score (presence of important terms)
        important_terms = [
            "important", "key", "critical", "decision", "learned",
            "always", "never", "must", "should", "remember",
            "because", "therefore", "conclusion", "result"
        ]
        term_count = sum(1 for term in important_terms if term.lower() in sentence.lower())
        score += min(0.3, term_count * 0.1)

        # Access boost
        score += min(0.2, access_count * 0.02)

        return score

    def _extract_relationships(self, text: str) -> List[Dict[str, str]]:
        """Extract entity relationships from text."""
        relationships = []

        # Simple patterns for relationships
        patterns = [
            r"(\w+)\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*(\w+)",
            r"(\w+)\s+(?:uses?|requires?|depends?\s+on)\s+(\w+)",
            r"(\w+)\s+(?:connects?\s+to|links?\s+to)\s+(\w+)",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    relationships.append({
                        "subject": match.group(1),
                        "predicate": "related_to",
                        "object": match.group(2)
                    })

        # Deduplicate
        seen = set()
        unique_rels = []
        for rel in relationships:
            key = f"{rel['subject']}-{rel['object']}"
            if key not in seen:
                seen.add(key)
                unique_rels.append(rel)

        return unique_rels[:10]  # Limit to 10 relationships

    def score_retention(
        self,
        original_memories: List["MemoryEntry"],
        compressed_summary: str,
        key_facts: List[str]
    ) -> float:
        """Score information retention using term overlap."""
        # Collect all terms from originals
        original_terms: Set[str] = set()
        for memory in original_memories:
            words = re.findall(r'\b\w+\b', memory.content.lower())
            # Filter stopwords
            stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                        "being", "have", "has", "had", "do", "does", "did", "will",
                        "would", "could", "should", "may", "might", "must", "can",
                        "to", "of", "in", "for", "on", "with", "at", "by", "from",
                        "as", "into", "through", "during", "before", "after", "above",
                        "below", "between", "under", "and", "but", "or", "nor", "so",
                        "yet", "both", "either", "neither", "not", "only", "own", "same",
                        "than", "too", "very", "just", "also", "now", "here", "there",
                        "when", "where", "why", "how", "all", "each", "every", "both",
                        "few", "more", "most", "other", "some", "such", "no", "any"}
            meaningful = {w for w in words if len(w) > 2 and w not in stopwords}
            original_terms.update(meaningful)

        if not original_terms:
            return 1.0

        # Check coverage in summary
        summary_terms = set(re.findall(r'\b\w+\b', compressed_summary.lower()))
        facts_terms = set()
        for fact in key_facts:
            facts_terms.update(re.findall(r'\b\w+\b', fact.lower()))

        compressed_terms = summary_terms | facts_terms

        # Calculate overlap
        overlap = len(original_terms & compressed_terms)
        retention = overlap / len(original_terms)

        return min(1.0, retention * 1.2)  # Slight boost for extractive


class AbstractiveStrategy(CompressionStrategyBase):
    """
    Abstractive compression: Generate new summary using LLM.

    Produces more coherent summaries but requires LLM access.
    Falls back to extractive if no LLM available.
    """

    def __init__(self, llm_provider: Optional[Callable[[str], str]] = None):
        """
        Initialize with optional LLM provider.

        Args:
            llm_provider: Function that takes prompt and returns summary
        """
        self.llm_provider = llm_provider
        self._fallback = ExtractiveStrategy()

    async def compress(
        self,
        memories: List["MemoryEntry"],
        config: CompressionConfig
    ) -> Tuple[str, List[str], List[Dict[str, str]]]:
        """Generate abstractive summary."""
        if not self.llm_provider:
            logger.warning("No LLM provider, falling back to extractive")
            return await self._fallback.compress(memories, config)

        # Prepare prompt
        combined_content = "\n\n".join([
            f"Memory {i+1}: {m.content}"
            for i, m in enumerate(memories)
        ])

        prompt = f"""Summarize the following {len(memories)} related memories into a coherent summary.
Preserve key facts, decisions, and relationships.
Keep the summary concise but comprehensive.

Memories:
{combined_content}

Provide:
1. A concise summary (2-3 sentences)
2. Key facts (bullet points)
3. Important relationships (entity -> relationship -> entity)

Summary:"""

        try:
            response = self.llm_provider(prompt)
            summary, key_facts, relationships = self._parse_response(response)
            return summary, key_facts, relationships
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            return await self._fallback.compress(memories, config)

    def _parse_response(self, response: str) -> Tuple[str, List[str], List[Dict[str, str]]]:
        """Parse LLM response into components."""
        # Simple parsing - real implementation would be more robust
        lines = response.strip().split('\n')

        summary = ""
        key_facts = []
        relationships = []

        current_section = "summary"

        for line in lines:
            line = line.strip()
            if not line:
                continue

            lower = line.lower()
            if "key fact" in lower or "bullet" in lower:
                current_section = "facts"
                continue
            elif "relationship" in lower:
                current_section = "relationships"
                continue

            if current_section == "summary":
                summary += " " + line
            elif current_section == "facts":
                if line.startswith(("-", "*", "+")):
                    key_facts.append(line[1:].strip())
                else:
                    key_facts.append(line)
            elif current_section == "relationships":
                # Try to parse "A -> B" or "A relates to B"
                if "->" in line:
                    parts = line.split("->")
                    if len(parts) >= 2:
                        relationships.append({
                            "subject": parts[0].strip(),
                            "predicate": "relates_to",
                            "object": parts[-1].strip()
                        })

        return summary.strip(), key_facts, relationships

    def score_retention(
        self,
        original_memories: List["MemoryEntry"],
        compressed_summary: str,
        key_facts: List[str]
    ) -> float:
        """Score retention for abstractive compression."""
        # Use same method as extractive
        extractor = ExtractiveStrategy()
        return extractor.score_retention(original_memories, compressed_summary, key_facts)


class HierarchicalStrategy(CompressionStrategyBase):
    """
    Hierarchical compression: Multi-level compression tree.

    Creates a tree structure where leaf nodes contain detailed info
    and higher levels contain progressively more abstract summaries.
    """

    def __init__(self, base_strategy: Optional[CompressionStrategyBase] = None):
        """
        Initialize with base strategy for leaf compression.

        Args:
            base_strategy: Strategy to use at each level (default: extractive)
        """
        self.base_strategy = base_strategy or ExtractiveStrategy()

    async def compress(
        self,
        memories: List["MemoryEntry"],
        config: CompressionConfig
    ) -> Tuple[str, List[str], List[Dict[str, str]]]:
        """Build hierarchical compression."""
        if len(memories) < config.min_group_size:
            return await self.base_strategy.compress(memories, config)

        # Split into chunks for first level
        chunk_size = max(
            config.min_group_size,
            len(memories) // config.hierarchy_levels
        )
        chunks = [
            memories[i:i + chunk_size]
            for i in range(0, len(memories), chunk_size)
        ]

        # Compress each chunk
        level_summaries = []
        all_facts = []
        all_relationships = []

        for chunk in chunks:
            summary, facts, rels = await self.base_strategy.compress(chunk, config)
            level_summaries.append(summary)
            all_facts.extend(facts)
            all_relationships.extend(rels)

        # If we have multiple summaries, compress again
        if len(level_summaries) > 1:
            # Create pseudo-memories from summaries
            from .backends.base import MemoryEntry as ME
            pseudo_memories = [
                ME(
                    id=f"level1_{i}",
                    content=summary,
                    access_count=0
                )
                for i, summary in enumerate(level_summaries)
            ]

            # Compress the summaries
            final_summary, more_facts, more_rels = await self.base_strategy.compress(
                pseudo_memories, config
            )
            all_facts = list(set(all_facts + more_facts))[:10]
            all_relationships = self._dedupe_relationships(all_relationships + more_rels)
        else:
            final_summary = level_summaries[0] if level_summaries else ""

        return final_summary, all_facts, all_relationships

    def _dedupe_relationships(
        self,
        relationships: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Deduplicate relationships."""
        seen = set()
        unique = []
        for rel in relationships:
            key = f"{rel.get('subject', '')}-{rel.get('object', '')}"
            if key not in seen:
                seen.add(key)
                unique.append(rel)
        return unique[:10]

    def score_retention(
        self,
        original_memories: List["MemoryEntry"],
        compressed_summary: str,
        key_facts: List[str]
    ) -> float:
        """Score retention for hierarchical compression."""
        return self.base_strategy.score_retention(
            original_memories, compressed_summary, key_facts
        )


class ClusteringStrategy(CompressionStrategyBase):
    """
    Clustering compression: Group similar memories before compression.

    Uses semantic similarity to group related memories, then compresses
    each cluster separately.
    """

    def __init__(
        self,
        base_strategy: Optional[CompressionStrategyBase] = None,
        embedding_provider: Optional[Callable[[str], List[float]]] = None
    ):
        """
        Initialize with base strategy and embedding provider.

        Args:
            base_strategy: Strategy to compress each cluster
            embedding_provider: Function to generate embeddings
        """
        self.base_strategy = base_strategy or ExtractiveStrategy()
        self.embedding_provider = embedding_provider

    async def compress(
        self,
        memories: List["MemoryEntry"],
        config: CompressionConfig
    ) -> Tuple[str, List[str], List[Dict[str, str]]]:
        """Cluster and compress memories."""
        if len(memories) < config.min_group_size:
            return await self.base_strategy.compress(memories, config)

        # Cluster memories by similarity
        clusters = await self._cluster_memories(memories, config)

        # Compress each cluster
        cluster_summaries = []
        all_facts = []
        all_relationships = []

        for cluster in clusters:
            if len(cluster) > 0:
                summary, facts, rels = await self.base_strategy.compress(
                    cluster, config
                )
                cluster_summaries.append(summary)
                all_facts.extend(facts)
                all_relationships.extend(rels)

        # Combine cluster summaries
        final_summary = " ".join(cluster_summaries)

        # Deduplicate
        all_facts = list(set(all_facts))[:10]
        seen_rels = set()
        unique_rels = []
        for rel in all_relationships:
            key = f"{rel.get('subject', '')}-{rel.get('object', '')}"
            if key not in seen_rels:
                seen_rels.add(key)
                unique_rels.append(rel)

        return final_summary, all_facts, unique_rels[:10]

    async def _cluster_memories(
        self,
        memories: List["MemoryEntry"],
        config: CompressionConfig
    ) -> List[List["MemoryEntry"]]:
        """Cluster memories by semantic similarity."""
        if not self.embedding_provider:
            # Fall back to simple content-based clustering
            return self._cluster_by_keywords(memories, config)

        # Get embeddings
        embeddings = []
        for memory in memories:
            if memory.embedding:
                embeddings.append(memory.embedding)
            else:
                try:
                    emb = self.embedding_provider(memory.content)
                    embeddings.append(emb)
                except Exception:
                    embeddings.append(None)

        # Simple clustering: group by similarity threshold
        clusters: List[List["MemoryEntry"]] = []
        used = set()

        for i, memory in enumerate(memories):
            if i in used:
                continue

            cluster = [memory]
            used.add(i)

            if embeddings[i] is not None:
                for j, other in enumerate(memories):
                    if j in used or j == i:
                        continue

                    if embeddings[j] is not None:
                        sim = self._cosine_similarity(embeddings[i], embeddings[j])
                        if sim >= config.similarity_threshold:
                            cluster.append(other)
                            used.add(j)

            clusters.append(cluster)

        return clusters

    def _cluster_by_keywords(
        self,
        memories: List["MemoryEntry"],
        config: CompressionConfig
    ) -> List[List["MemoryEntry"]]:
        """Simple keyword-based clustering."""
        # Extract keywords per memory
        keywords_per_memory = []
        for memory in memories:
            words = set(re.findall(r'\b\w{4,}\b', memory.content.lower()))
            keywords_per_memory.append(words)

        # Cluster by keyword overlap
        clusters: List[List["MemoryEntry"]] = []
        used = set()

        for i, memory in enumerate(memories):
            if i in used:
                continue

            cluster = [memory]
            used.add(i)

            for j, other in enumerate(memories):
                if j in used:
                    continue

                # Calculate Jaccard similarity
                intersection = len(keywords_per_memory[i] & keywords_per_memory[j])
                union = len(keywords_per_memory[i] | keywords_per_memory[j])

                if union > 0:
                    similarity = intersection / union
                    if similarity >= config.similarity_threshold:
                        cluster.append(other)
                        used.add(j)

            clusters.append(cluster)

        return clusters

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity."""
        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def score_retention(
        self,
        original_memories: List["MemoryEntry"],
        compressed_summary: str,
        key_facts: List[str]
    ) -> float:
        """Score retention for clustering compression."""
        return self.base_strategy.score_retention(
            original_memories, compressed_summary, key_facts
        )


# =============================================================================
# MAIN COMPRESSOR CLASS
# =============================================================================

class MemoryCompressor:
    """
    Main memory compression system.

    Provides:
    - Multiple compression strategies
    - Candidate identification
    - Compression with quality metrics
    - Decompression with caching
    - Original memory archival
    """

    def __init__(
        self,
        backend: Optional["SQLiteTierBackend"] = None,
        config: Optional[CompressionConfig] = None,
        llm_provider: Optional[Callable[[str], str]] = None,
        embedding_provider: Optional[Callable[[str], List[float]]] = None
    ) -> None:
        """
        Initialize the memory compressor.

        Args:
            backend: Memory storage backend
            config: Compression configuration
            llm_provider: Function for LLM summarization
            embedding_provider: Function for generating embeddings
        """
        self._backend = backend
        self._config = config or CompressionConfig()
        self._llm_provider = llm_provider
        self._embedding_provider = embedding_provider

        # Initialize strategies
        self._strategies: Dict[CompressionStrategy, CompressionStrategyBase] = {
            CompressionStrategy.EXTRACTIVE: ExtractiveStrategy(),
            CompressionStrategy.ABSTRACTIVE: AbstractiveStrategy(llm_provider),
            CompressionStrategy.HIERARCHICAL: HierarchicalStrategy(ExtractiveStrategy()),
            CompressionStrategy.CLUSTERING: ClusteringStrategy(
                ExtractiveStrategy(), embedding_provider
            ),
        }

        # Decompression cache
        self._cache: Dict[str, Tuple[List["MemoryEntry"], datetime]] = {}

        # Metrics tracking
        self._metrics = CompressionMetrics()
        self._compression_times: List[float] = []
        self._decompression_times: List[float] = []

    @property
    def backend(self) -> "SQLiteTierBackend":
        """Get the backend, initializing if needed."""
        if self._backend is None:
            from .backends.sqlite import get_sqlite_backend
            self._backend = get_sqlite_backend()
        return self._backend

    @property
    def config(self) -> CompressionConfig:
        """Get compression configuration."""
        return self._config

    # =========================================================================
    # CANDIDATE IDENTIFICATION
    # =========================================================================

    async def identify_candidates(
        self,
        trigger: CompressionTrigger = CompressionTrigger.SCHEDULED
    ) -> List[List[str]]:
        """
        Identify groups of memories that are candidates for compression.

        Returns:
            List of memory ID groups, each group suitable for compression together
        """
        all_memories = await self.backend.list_all()
        now = datetime.now(timezone.utc)

        # Filter candidates based on criteria
        candidates: List["MemoryEntry"] = []

        for memory in all_memories:
            # Skip important memories if configured
            if self._config.preserve_high_importance:
                importance = memory.metadata.get("importance", 0.5)
                if importance >= self._config.importance_threshold:
                    continue

            # Skip already compressed memories
            if memory.metadata.get("is_compressed"):
                continue

            # Check age
            if memory.created_at:
                age_days = (now - memory.created_at).total_seconds() / 86400
                if age_days < self._config.min_age_days:
                    continue

            # Check access patterns
            if memory.access_count > self._config.max_access_count:
                continue

            if memory.last_accessed:
                access_age = (now - memory.last_accessed).total_seconds() / 86400
                if access_age < self._config.min_access_age_days:
                    continue

            candidates.append(memory)

        if not candidates:
            return []

        # Group candidates by similarity
        groups = await self._group_by_similarity(candidates)

        # Filter groups by size
        valid_groups = [
            [m.id for m in group]
            for group in groups
            if self._config.min_group_size <= len(group) <= self._config.max_group_size
        ]

        logger.info(f"Identified {len(valid_groups)} compression candidate groups")
        return valid_groups

    async def _group_by_similarity(
        self,
        memories: List["MemoryEntry"]
    ) -> List[List["MemoryEntry"]]:
        """Group memories by semantic similarity."""
        clustering = ClusteringStrategy(
            embedding_provider=self._embedding_provider
        )
        return await clustering._cluster_memories(memories, self._config)

    # =========================================================================
    # COMPRESSION
    # =========================================================================

    async def compress_memories(
        self,
        memory_ids: List[str],
        strategy: Optional[CompressionStrategy] = None,
        trigger: CompressionTrigger = CompressionTrigger.MANUAL
    ) -> CompressionResult:
        """
        Compress a group of related memories.

        Args:
            memory_ids: List of memory IDs to compress
            strategy: Compression strategy (uses default if not specified)
            trigger: What triggered this compression

        Returns:
            CompressionResult with metrics and links to originals
        """
        start_time = time.time()
        strategy = strategy or self._config.default_strategy

        # Fetch memories
        memories: List["MemoryEntry"] = []
        for mem_id in memory_ids:
            memory = await self.backend.get(mem_id)
            if memory:
                memories.append(memory)

        if len(memories) < self._config.min_group_size:
            raise ValueError(
                f"Not enough memories to compress: {len(memories)} < {self._config.min_group_size}"
            )

        # Calculate original token count
        original_tokens = sum(
            len(m.content) // 4  # Rough estimate
            for m in memories
        )

        # Get strategy implementation
        strategy_impl = self._strategies.get(strategy)
        if not strategy_impl:
            raise ValueError(f"Unknown compression strategy: {strategy}")

        # Compress
        summary, key_facts, relationships = await strategy_impl.compress(
            memories, self._config
        )

        # Score retention
        retention_score = strategy_impl.score_retention(memories, summary, key_facts)

        # Create compressed memory
        compressed_id = self._generate_compressed_id(memory_ids)
        compressed_tokens = len(summary) // 4

        compressed = CompressedMemory(
            id=compressed_id,
            summary=summary,
            key_facts=key_facts,
            key_relationships=relationships,
            original_ids=memory_ids,
            original_hashes={m.id: m.content_hash() for m in memories},
            strategy=strategy,
            token_count=compressed_tokens,
            tags=list(set(tag for m in memories for tag in m.tags)),
            metadata={
                "compression_trigger": trigger.value,
                "original_count": len(memories),
                "original_tokens": original_tokens,
                "retention_score": retention_score,
            }
        )

        # Store compressed memory
        await self._store_compressed(compressed)

        # Archive originals
        archived_locations = await self._archive_originals(memories, compressed_id)

        # Calculate metrics
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        elapsed_ms = (time.time() - start_time) * 1000

        # Update metrics
        self._update_metrics(
            strategy, original_tokens, compressed_tokens,
            retention_score, elapsed_ms
        )

        result = CompressionResult(
            compressed_id=compressed_id,
            original_ids=memory_ids,
            strategy=strategy,
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            compression_ratio=compression_ratio,
            retention_score=retention_score,
            coherence_score=self._score_coherence(summary),
            coverage_score=self._score_coverage(memories, key_facts),
            trigger=trigger,
            archived_locations=archived_locations,
        )

        logger.info(
            f"Compressed {len(memories)} memories into {compressed_id} "
            f"(ratio: {compression_ratio:.2%}, retention: {retention_score:.2%})"
        )

        return result

    def _generate_compressed_id(self, memory_ids: List[str]) -> str:
        """Generate a unique ID for compressed memory."""
        combined = ":".join(sorted(memory_ids))
        hash_val = hashlib.md5(combined.encode()).hexdigest()[:12]
        return f"compressed_{hash_val}"

    async def _store_compressed(self, compressed: CompressedMemory) -> None:
        """Store the compressed memory."""
        from .backends.base import MemoryEntry, MemoryTier, MemoryPriority

        entry = MemoryEntry(
            id=compressed.id,
            content=compressed.summary,
            tier=MemoryTier.ARCHIVAL_MEMORY,
            priority=MemoryPriority.NORMAL,
            tags=compressed.tags,
            metadata={
                "is_compressed": True,
                "key_facts": compressed.key_facts,
                "key_relationships": compressed.key_relationships,
                "original_ids": compressed.original_ids,
                "original_hashes": compressed.original_hashes,
                "strategy": compressed.strategy.value,
                **compressed.metadata
            }
        )

        await self.backend.put(compressed.id, entry)

    async def _archive_originals(
        self,
        memories: List["MemoryEntry"],
        compressed_id: str
    ) -> Dict[str, str]:
        """Archive original memories and mark as compressed."""
        archived_locations = {}

        for memory in memories:
            # Update metadata to mark as archived
            memory.metadata["archived_to"] = compressed_id
            memory.metadata["archived_at"] = datetime.now(timezone.utc).isoformat()

            # Move to archival tier
            from .backends.base import MemoryTier
            memory.tier = MemoryTier.ARCHIVAL_MEMORY

            # Store updated version
            await self.backend.put(memory.id, memory)
            archived_locations[memory.id] = f"archival:{memory.id}"

        return archived_locations

    def _score_coherence(self, summary: str) -> float:
        """Score the coherence of a summary."""
        if not summary:
            return 0.0

        # Simple heuristics
        sentences = re.split(r'[.!?]', summary)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        score = 0.5  # Base score

        # Bonus for proper sentence structure
        if len(sentences) >= 2:
            score += 0.2

        # Bonus for reasonable length
        avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
        if 10 <= avg_sentence_len <= 30:
            score += 0.2

        # Penalty for very short summary
        if len(summary) < 50:
            score -= 0.3

        return min(1.0, max(0.0, score))

    def _score_coverage(
        self,
        memories: List["MemoryEntry"],
        key_facts: List[str]
    ) -> float:
        """Score how well key facts cover the original memories."""
        if not memories or not key_facts:
            return 0.0

        # Check if each memory is represented in facts
        coverage_count = 0

        for memory in memories:
            memory_words = set(re.findall(r'\b\w{4,}\b', memory.content.lower()))

            for fact in key_facts:
                fact_words = set(re.findall(r'\b\w{4,}\b', fact.lower()))
                overlap = len(memory_words & fact_words)

                if overlap >= 2:  # At least 2 significant words overlap
                    coverage_count += 1
                    break

        return coverage_count / len(memories)

    def _update_metrics(
        self,
        strategy: CompressionStrategy,
        original_tokens: int,
        compressed_tokens: int,
        retention_score: float,
        elapsed_ms: float
    ) -> None:
        """Update compression metrics."""
        self._metrics.total_compressions += 1
        self._metrics.total_original_tokens += original_tokens
        self._metrics.total_compressed_tokens += compressed_tokens

        # Update strategy counts
        strategy_key = strategy.value
        self._metrics.compressions_by_strategy[strategy_key] = \
            self._metrics.compressions_by_strategy.get(strategy_key, 0) + 1

        # Update averages
        if self._metrics.total_original_tokens > 0:
            self._metrics.average_compression_ratio = (
                self._metrics.total_compressed_tokens /
                self._metrics.total_original_tokens
            )

        # Update timing
        self._compression_times.append(elapsed_ms)
        self._metrics.average_compression_time_ms = (
            sum(self._compression_times) / len(self._compression_times)
        )

        # Update retention average (rolling)
        if self._metrics.average_retention_score == 0:
            self._metrics.average_retention_score = retention_score
        else:
            # Exponential moving average
            self._metrics.average_retention_score = (
                0.9 * self._metrics.average_retention_score + 0.1 * retention_score
            )

        # Update quality distribution
        if retention_score >= 0.9:
            key = "excellent"
        elif retention_score >= 0.7:
            key = "good"
        elif retention_score >= 0.5:
            key = "fair"
        else:
            key = "poor"

        self._metrics.quality_distribution[key] = \
            self._metrics.quality_distribution.get(key, 0) + 1

    # =========================================================================
    # DECOMPRESSION
    # =========================================================================

    async def decompress(
        self,
        compressed_id: str,
        load_depth: int = 1
    ) -> List["MemoryEntry"]:
        """
        Expand a compressed memory back to its original components.

        Args:
            compressed_id: ID of the compressed memory
            load_depth: How deep to load in hierarchy (1 = immediate originals)

        Returns:
            List of original memory entries
        """
        start_time = time.time()

        # Check cache
        if self._config.cache_decompressed and compressed_id in self._cache:
            cached, cache_time = self._cache[compressed_id]
            age = (datetime.now(timezone.utc) - cache_time).total_seconds()

            if age < self._config.cache_ttl_seconds:
                self._metrics.cache_hits += 1
                self._update_cache_rate()
                return cached

        self._metrics.cache_misses += 1
        self._update_cache_rate()

        # Fetch compressed memory
        compressed = await self.backend.get(compressed_id)
        if not compressed:
            raise ValueError(f"Compressed memory not found: {compressed_id}")

        # Get original IDs from metadata
        original_ids = compressed.metadata.get("original_ids", [])
        if not original_ids:
            raise ValueError(f"No original IDs in compressed memory: {compressed_id}")

        # Fetch originals
        originals: List["MemoryEntry"] = []
        for orig_id in original_ids:
            original = await self.backend.get(orig_id)
            if original:
                originals.append(original)

        # Cache result
        if self._config.cache_decompressed:
            self._cache[compressed_id] = (originals, datetime.now(timezone.utc))

        # Update metrics
        elapsed_ms = (time.time() - start_time) * 1000
        self._metrics.total_decompressions += 1
        self._decompression_times.append(elapsed_ms)
        self._metrics.average_decompression_time_ms = (
            sum(self._decompression_times) / len(self._decompression_times)
        )

        return originals

    def _update_cache_rate(self) -> None:
        """Update cache hit rate."""
        total = self._metrics.cache_hits + self._metrics.cache_misses
        if total > 0:
            self._metrics.cache_hit_rate = self._metrics.cache_hits / total

    async def get_summary_with_details(
        self,
        compressed_id: str
    ) -> Dict[str, Any]:
        """
        Get compressed summary with option to drill down into details.

        Returns:
            Dict with summary, key_facts, and links to originals
        """
        compressed = await self.backend.get(compressed_id)
        if not compressed:
            raise ValueError(f"Compressed memory not found: {compressed_id}")

        return {
            "id": compressed_id,
            "summary": compressed.content,
            "key_facts": compressed.metadata.get("key_facts", []),
            "key_relationships": compressed.metadata.get("key_relationships", []),
            "original_count": len(compressed.metadata.get("original_ids", [])),
            "original_ids": compressed.metadata.get("original_ids", []),
            "strategy": compressed.metadata.get("strategy", "unknown"),
            "retention_score": compressed.metadata.get("retention_score", 0),
            "can_decompress": True,
        }

    # =========================================================================
    # METRICS
    # =========================================================================

    def get_metrics(self) -> CompressionMetrics:
        """Get compression system metrics."""
        return self._metrics

    def clear_cache(self) -> None:
        """Clear the decompression cache."""
        self._cache.clear()
        logger.info("Cleared decompression cache")


# =============================================================================
# BACKGROUND COMPRESSOR
# =============================================================================

class BackgroundCompressor:
    """
    Background scheduler for automatic memory compression.

    Runs compression jobs based on:
    - Scheduled intervals
    - Memory pressure thresholds
    - Age-based triggers
    """

    def __init__(
        self,
        compressor: MemoryCompressor,
        config: Optional[CompressionConfig] = None
    ) -> None:
        """
        Initialize background compressor.

        Args:
            compressor: Memory compressor instance
            config: Configuration (uses compressor's config if not specified)
        """
        self.compressor = compressor
        self.config = config or compressor.config

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_run: Optional[datetime] = None
        self._run_count = 0
        self._total_compressed = 0

    async def start(self) -> None:
        """Start the background compression scheduler."""
        if self._running:
            logger.warning("Background compressor already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Started background compressor")

    async def stop(self) -> None:
        """Stop the background compression scheduler."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Stopped background compressor")

    async def _run_loop(self) -> None:
        """Main compression loop."""
        while self._running:
            try:
                # Check if compression needed
                should_run = await self._should_run()

                if should_run:
                    await self._run_compression()

                # Wait for next interval
                interval_seconds = self.config.schedule_interval_hours * 3600
                await asyncio.sleep(interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background compression error: {e}")
                # Wait before retrying
                await asyncio.sleep(60)

    async def _should_run(self) -> bool:
        """Check if compression should run."""
        # Always run if never ran
        if self._last_run is None:
            return True

        # Check memory pressure
        stats = await self.compressor.backend.get_stats()
        total_memories = stats.get("total_memories", 0)

        # Estimate pressure (simplified)
        # Real implementation would check actual storage limits
        estimated_pressure = total_memories / 10000  # Assume 10K limit

        if estimated_pressure >= self.config.memory_pressure_threshold:
            logger.info(f"Memory pressure trigger: {estimated_pressure:.1%}")
            return True

        # Check time since last run
        elapsed = datetime.now(timezone.utc) - self._last_run
        interval = timedelta(hours=self.config.schedule_interval_hours)

        return elapsed >= interval

    async def _run_compression(self) -> None:
        """Run a compression cycle."""
        logger.info("Starting background compression cycle")
        self._run_count += 1
        self._last_run = datetime.now(timezone.utc)

        try:
            # Identify candidates
            candidate_groups = await self.compressor.identify_candidates(
                trigger=CompressionTrigger.SCHEDULED
            )

            if not candidate_groups:
                logger.info("No compression candidates found")
                return

            # Limit batch size
            groups_to_process = candidate_groups[:self.config.max_compression_batch]

            compressed_count = 0
            for group in groups_to_process:
                try:
                    result = await self.compressor.compress_memories(
                        group,
                        trigger=CompressionTrigger.SCHEDULED
                    )
                    compressed_count += len(result.original_ids)
                except Exception as e:
                    logger.warning(f"Failed to compress group: {e}")

            self._total_compressed += compressed_count
            logger.info(
                f"Compression cycle complete: {compressed_count} memories compressed "
                f"({len(groups_to_process)} groups)"
            )

        except Exception as e:
            logger.error(f"Compression cycle failed: {e}")

    async def run_once(self) -> Dict[str, Any]:
        """Run compression once (manual trigger)."""
        await self._run_compression()

        return {
            "run_count": self._run_count,
            "total_compressed": self._total_compressed,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "metrics": self.compressor.get_metrics().to_dict()
        }

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            "running": self._running,
            "run_count": self._run_count,
            "total_compressed": self._total_compressed,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "next_run_in_hours": self.config.schedule_interval_hours,
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

_compressor_instance: Optional[MemoryCompressor] = None
_background_compressor: Optional[BackgroundCompressor] = None


def get_compressor(
    backend: Optional["SQLiteTierBackend"] = None,
    config: Optional[CompressionConfig] = None,
    llm_provider: Optional[Callable[[str], str]] = None,
    embedding_provider: Optional[Callable[[str], List[float]]] = None
) -> MemoryCompressor:
    """
    Get or create the singleton memory compressor.

    Args:
        backend: Optional backend (uses singleton if None)
        config: Optional config (uses defaults if None)
        llm_provider: Optional LLM for abstractive compression
        embedding_provider: Optional embedding function

    Returns:
        MemoryCompressor instance
    """
    global _compressor_instance
    if _compressor_instance is None:
        _compressor_instance = MemoryCompressor(
            backend, config, llm_provider, embedding_provider
        )
    return _compressor_instance


def get_background_compressor(
    compressor: Optional[MemoryCompressor] = None
) -> BackgroundCompressor:
    """
    Get or create the singleton background compressor.

    Args:
        compressor: Optional compressor (creates one if None)

    Returns:
        BackgroundCompressor instance
    """
    global _background_compressor
    if _background_compressor is None:
        if compressor is None:
            compressor = get_compressor()
        _background_compressor = BackgroundCompressor(compressor)
    return _background_compressor


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def compress_old_memories(
    min_age_days: float = 7.0,
    strategy: CompressionStrategy = CompressionStrategy.EXTRACTIVE
) -> List[CompressionResult]:
    """
    Convenience function to compress old memories.

    Args:
        min_age_days: Minimum age in days for candidates
        strategy: Compression strategy to use

    Returns:
        List of CompressionResult objects
    """
    compressor = get_compressor()

    # Temporarily adjust config
    original_age = compressor.config.min_age_days
    compressor.config.min_age_days = min_age_days

    try:
        groups = await compressor.identify_candidates(
            trigger=CompressionTrigger.AGE
        )

        results = []
        for group in groups:
            result = await compressor.compress_memories(
                group, strategy, CompressionTrigger.AGE
            )
            results.append(result)

        return results
    finally:
        compressor.config.min_age_days = original_age


async def get_compression_stats() -> Dict[str, Any]:
    """
    Get compression system statistics.

    Returns:
        Dictionary with compression metrics
    """
    compressor = get_compressor()
    return compressor.get_metrics().to_dict()


async def decompress_memory(compressed_id: str) -> List[Dict[str, Any]]:
    """
    Decompress a memory and return as dictionaries.

    Args:
        compressed_id: ID of compressed memory

    Returns:
        List of memory dictionaries
    """
    compressor = get_compressor()
    memories = await compressor.decompress(compressed_id)

    return [
        {
            "id": m.id,
            "content": m.content,
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "tags": m.tags,
            "access_count": m.access_count,
        }
        for m in memories
    ]


__all__ = [
    # Enums
    "CompressionStrategy",
    "CompressionTrigger",

    # Configuration
    "CompressionConfig",

    # Result types
    "CompressionResult",
    "CompressionMetrics",
    "CompressedMemory",

    # Strategy implementations
    "CompressionStrategyBase",
    "ExtractiveStrategy",
    "AbstractiveStrategy",
    "HierarchicalStrategy",
    "ClusteringStrategy",

    # Main classes
    "MemoryCompressor",
    "BackgroundCompressor",

    # Factory functions
    "get_compressor",
    "get_background_compressor",

    # Convenience functions
    "compress_old_memories",
    "get_compression_stats",
    "decompress_memory",
]
