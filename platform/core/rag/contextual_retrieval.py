#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.24.0",
# ]
# ///
"""
Contextual Retrieval - Anthropic's Context Prepending Pattern

This module implements Anthropic's contextual retrieval pattern, which prepends
document-level context to each chunk before embedding. This dramatically improves
retrieval accuracy by ensuring chunks retain their broader document context.

Key Features:
- LLM-generated chunk-specific context
- Caching of context generation results (disk + memory)
- Integration with SemanticChunker
- Batch processing for efficiency
- Fallback to rule-based context when LLM unavailable

Pattern: "This chunk is from [doc title] about [topic]. It discusses: [context]"

Reference: Anthropic's "Contextual Retrieval" blog post

Usage:
    from core.rag.contextual_retrieval import ContextualRetriever

    retriever = ContextualRetriever(llm=my_llm, embedding_provider=embedder)

    # Process document
    contextualized_chunks = await retriever.contextualize_document(
        document=document_text,
        doc_title="API Documentation",
        doc_metadata={"source": "docs", "version": "v2"}
    )

    # Each chunk now has prepended context for better embedding
    for chunk in contextualized_chunks:
        print(f"Context: {chunk.prepended_context[:100]}...")
        embedding = chunk.contextualized_embedding

Architecture:
    Document -> Chunks -> Context Generation (LLM) -> Context Prepending -> Embedding
                                    |
                              Cache Lookup/Store
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .semantic_chunker import SemanticChunker, Chunk

logger = logging.getLogger(__name__)


# =============================================================================
# PROTOCOLS
# =============================================================================

class LLMProvider(Protocol):
    """Protocol for LLM providers used in context generation."""
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """Generate text completion."""
        ...


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    def embed(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text."""
        ...

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts."""
        ...


# =============================================================================
# CONFIGURATION
# =============================================================================

class ContextGenerationStrategy(str, Enum):
    """Strategies for generating chunk context."""
    LLM = "llm"                    # Use LLM to generate context
    RULE_BASED = "rule_based"     # Use rule-based extraction
    HYBRID = "hybrid"              # Try LLM, fallback to rule-based


@dataclass
class ContextualRetrievalConfig:
    """Configuration for contextual retrieval.

    Attributes:
        strategy: Context generation strategy (default: HYBRID)
        max_context_tokens: Maximum tokens for prepended context (default: 100)
        batch_size: Batch size for LLM context generation (default: 5)
        cache_enabled: Enable context caching (default: True)
        cache_dir: Directory for cache storage (default: .contextual_cache)
        memory_cache_size: Max entries in memory cache (default: 10000)
        context_template: Template for prepended context
        parallel_requests: Max parallel LLM requests (default: 3)
        timeout_seconds: Timeout per context generation (default: 30)
    """
    strategy: ContextGenerationStrategy = ContextGenerationStrategy.HYBRID
    max_context_tokens: int = 100
    batch_size: int = 5
    cache_enabled: bool = True
    cache_dir: str = ".contextual_cache"
    memory_cache_size: int = 10000
    context_template: str = "This chunk is from {doc_title} about {topic}. It discusses: {context}"
    parallel_requests: int = 3
    timeout_seconds: float = 30.0


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DocumentContext:
    """Context information about a document.

    Attributes:
        doc_id: Unique document identifier
        title: Document title
        summary: Brief document summary
        topics: Main topics covered
        doc_type: Type of document (article, code, api_docs, etc.)
        metadata: Additional metadata
    """
    doc_id: str
    title: str
    summary: str = ""
    topics: List[str] = field(default_factory=list)
    doc_type: str = "document"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextualizedChunk:
    """A chunk with prepended contextual information.

    Attributes:
        chunk_id: Unique chunk identifier
        original_content: Original chunk content
        prepended_context: LLM-generated or rule-based context
        contextualized_content: Context + original content combined
        contextualized_embedding: Embedding of contextualized content
        original_embedding: Embedding of original content (for comparison)
        document_context: Reference to parent document context
        chunk_metadata: Chunk-specific metadata
        context_generation_time_ms: Time to generate context
        cache_hit: Whether context was retrieved from cache
    """
    chunk_id: str
    original_content: str
    prepended_context: str
    contextualized_content: str
    contextualized_embedding: Optional[List[float]] = None
    original_embedding: Optional[List[float]] = None
    document_context: Optional[DocumentContext] = None
    chunk_metadata: Dict[str, Any] = field(default_factory=dict)
    context_generation_time_ms: float = 0.0
    cache_hit: bool = False

    @property
    def token_estimate(self) -> int:
        """Estimate token count of contextualized content."""
        return len(self.contextualized_content) // 4


@dataclass
class ContextualizationStats:
    """Statistics about contextualization operation.

    Attributes:
        total_chunks: Total chunks processed
        cache_hits: Number of cache hits
        llm_generations: Number of LLM generations
        rule_based_fallbacks: Number of rule-based fallbacks
        total_time_ms: Total processing time
        avg_context_tokens: Average context tokens prepended
    """
    total_chunks: int = 0
    cache_hits: int = 0
    llm_generations: int = 0
    rule_based_fallbacks: int = 0
    total_time_ms: float = 0.0
    avg_context_tokens: float = 0.0


# =============================================================================
# CONTEXT CACHE
# =============================================================================

class ContextCache:
    """Two-tier cache for generated contexts (memory + disk).

    Uses SQLite for persistent storage and LRU dict for fast memory access.
    """

    def __init__(
        self,
        cache_dir: str = ".contextual_cache",
        memory_size: int = 10000
    ):
        """Initialize the context cache.

        Args:
            cache_dir: Directory for SQLite database
            memory_size: Maximum entries in memory cache
        """
        self.cache_dir = Path(cache_dir)
        self.memory_size = memory_size
        self._memory_cache: Dict[str, str] = {}
        self._access_order: List[str] = []
        self._db: Optional[sqlite3.Connection] = None
        self._initialized = False

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        if self._initialized:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        db_path = self.cache_dir / "context_cache.db"

        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS contexts (
                chunk_hash TEXT PRIMARY KEY,
                context TEXT NOT NULL,
                doc_id TEXT,
                created_at REAL,
                access_count INTEGER DEFAULT 1
            )
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_doc_id ON contexts(doc_id)
        """)
        self._db.commit()
        self._initialized = True
        logger.info(f"Context cache initialized at {db_path}")

    def _get_cache_key(self, chunk_content: str, doc_id: str) -> str:
        """Generate cache key from chunk content and document ID."""
        key_input = f"{doc_id}:{chunk_content[:500]}"
        return hashlib.sha256(key_input.encode()).hexdigest()

    def get(self, chunk_content: str, doc_id: str) -> Optional[str]:
        """Get cached context for a chunk.

        Args:
            chunk_content: The chunk content
            doc_id: Document identifier

        Returns:
            Cached context or None if not found
        """
        self._init_db()
        cache_key = self._get_cache_key(chunk_content, doc_id)

        # Check memory cache first
        if cache_key in self._memory_cache:
            # Update access order (LRU)
            if cache_key in self._access_order:
                self._access_order.remove(cache_key)
            self._access_order.append(cache_key)
            return self._memory_cache[cache_key]

        # Check disk cache
        if self._db:
            cursor = self._db.execute(
                "SELECT context FROM contexts WHERE chunk_hash = ?",
                (cache_key,)
            )
            row = cursor.fetchone()
            if row:
                context = row[0]
                # Promote to memory cache
                self._memory_cache[cache_key] = context
                self._access_order.append(cache_key)
                self._evict_if_needed()
                # Update access count
                self._db.execute(
                    "UPDATE contexts SET access_count = access_count + 1 WHERE chunk_hash = ?",
                    (cache_key,)
                )
                return context

        return None

    def put(self, chunk_content: str, doc_id: str, context: str) -> None:
        """Store context in cache.

        Args:
            chunk_content: The chunk content
            doc_id: Document identifier
            context: Generated context to cache
        """
        self._init_db()
        cache_key = self._get_cache_key(chunk_content, doc_id)

        # Store in memory cache
        self._memory_cache[cache_key] = context
        self._access_order.append(cache_key)
        self._evict_if_needed()

        # Store in disk cache
        if self._db:
            self._db.execute("""
                INSERT OR REPLACE INTO contexts (chunk_hash, context, doc_id, created_at)
                VALUES (?, ?, ?, ?)
            """, (cache_key, context, doc_id, time.time()))
            self._db.commit()

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if memory cache exceeds size."""
        while len(self._memory_cache) > self.memory_size:
            if self._access_order:
                oldest_key = self._access_order.pop(0)
                self._memory_cache.pop(oldest_key, None)

    def invalidate_document(self, doc_id: str) -> int:
        """Invalidate all cached contexts for a document.

        Args:
            doc_id: Document identifier

        Returns:
            Number of entries invalidated
        """
        self._init_db()
        count = 0

        if self._db:
            cursor = self._db.execute(
                "DELETE FROM contexts WHERE doc_id = ?",
                (doc_id,)
            )
            count = cursor.rowcount
            self._db.commit()

        # Also clear from memory cache (inefficient but rare operation)
        keys_to_remove = [
            k for k, v in self._memory_cache.items()
            if doc_id in k  # Approximate check
        ]
        for key in keys_to_remove:
            self._memory_cache.pop(key, None)
            if key in self._access_order:
                self._access_order.remove(key)

        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        self._init_db()

        stats = {
            "memory_entries": len(self._memory_cache),
            "memory_capacity": self.memory_size,
        }

        if self._db:
            cursor = self._db.execute("SELECT COUNT(*) FROM contexts")
            stats["disk_entries"] = cursor.fetchone()[0]

            cursor = self._db.execute("SELECT SUM(access_count) FROM contexts")
            stats["total_accesses"] = cursor.fetchone()[0] or 0

        return stats

    def close(self) -> None:
        """Close database connection."""
        if self._db:
            self._db.close()
            self._db = None
            self._initialized = False


# =============================================================================
# CONTEXT GENERATORS
# =============================================================================

class ContextPrompts:
    """Prompts for context generation."""

    DOCUMENT_ANALYSIS = """Analyze this document and provide:
1. A brief summary (1-2 sentences)
2. Main topics covered (comma-separated list)
3. Document type (article, code, api_docs, tutorial, etc.)

Document:
{document_preview}

Output as JSON:
{{"summary": "...", "topics": ["...", "..."], "doc_type": "..."}}"""

    CHUNK_CONTEXT = """<document>
{document_content}
</document>

Here is the chunk we want to situate within the document:
<chunk>
{chunk_content}
</chunk>

Please provide a short succinct context (2-3 sentences max) to situate this chunk within the overall document. Focus on:
- What topic or section this chunk belongs to
- Key entities, concepts, or APIs mentioned
- How it relates to the document's main subject

The context will be prepended to the chunk for better retrieval. Answer only with the context, nothing else."""


class RuleBasedContextGenerator:
    """Rule-based context generation (no LLM required).

    Uses heuristics to extract context from document structure:
    - Headers and section titles
    - Code comments and docstrings
    - Key phrases and named entities
    """

    def __init__(self):
        """Initialize the rule-based generator."""
        import re
        self._header_pattern = re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE)
        self._class_pattern = re.compile(r'class\s+(\w+)', re.MULTILINE)
        self._function_pattern = re.compile(r'(?:def|function|func)\s+(\w+)', re.MULTILINE)
        self._docstring_pattern = re.compile(r'"""(.+?)"""', re.DOTALL)

    def extract_context(
        self,
        chunk_content: str,
        document_content: str,
        doc_context: DocumentContext
    ) -> str:
        """Extract context using rules and heuristics.

        Args:
            chunk_content: The chunk to contextualize
            document_content: Full document content
            doc_context: Document-level context

        Returns:
            Generated context string
        """
        context_parts = []

        # Start with document title
        if doc_context.title:
            context_parts.append(f"from '{doc_context.title}'")

        # Add document type if available
        if doc_context.doc_type and doc_context.doc_type != "document":
            context_parts.append(f"({doc_context.doc_type})")

        # Find nearest header in document before this chunk
        chunk_pos = document_content.find(chunk_content[:100])
        if chunk_pos > 0:
            doc_before_chunk = document_content[:chunk_pos]
            headers = self._header_pattern.findall(doc_before_chunk)
            if headers:
                # Get last (nearest) header
                nearest_header = headers[-1].strip()
                if len(nearest_header) < 100:
                    context_parts.append(f"in section '{nearest_header}'")

        # Extract code entities from chunk
        classes = self._class_pattern.findall(chunk_content)
        functions = self._function_pattern.findall(chunk_content)

        if classes:
            context_parts.append(f"defining class {classes[0]}")
        elif functions:
            context_parts.append(f"defining function {functions[0]}")

        # Extract first docstring if present
        docstrings = self._docstring_pattern.findall(chunk_content)
        if docstrings:
            doc_preview = docstrings[0][:100].strip()
            if doc_preview:
                context_parts.append(f"which {doc_preview}")

        # Add topics if available
        if doc_context.topics:
            topics_str = ", ".join(doc_context.topics[:3])
            context_parts.append(f"covering {topics_str}")

        # Combine parts
        if context_parts:
            context = "This chunk is " + " ".join(context_parts) + "."
        else:
            context = f"This chunk is from a document titled '{doc_context.title or 'Untitled'}'."

        return context


class LLMContextGenerator:
    """LLM-based context generation using Anthropic's pattern."""

    def __init__(
        self,
        llm: LLMProvider,
        config: ContextualRetrievalConfig
    ):
        """Initialize the LLM context generator.

        Args:
            llm: LLM provider for generation
            config: Configuration options
        """
        self.llm = llm
        self.config = config
        self._semaphore = asyncio.Semaphore(config.parallel_requests)

    async def analyze_document(
        self,
        document_content: str,
        doc_title: str = ""
    ) -> DocumentContext:
        """Analyze document to extract high-level context.

        Args:
            document_content: Full document text
            doc_title: Optional document title

        Returns:
            DocumentContext with extracted information
        """
        # Use first ~3000 chars for analysis
        preview = document_content[:3000]

        prompt = ContextPrompts.DOCUMENT_ANALYSIS.format(
            document_preview=preview
        )

        try:
            response = await self.llm.generate(
                prompt,
                max_tokens=200,
                temperature=0.0
            )

            # Parse JSON response
            import re
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return DocumentContext(
                    doc_id=hashlib.md5(document_content[:1000].encode()).hexdigest()[:16],
                    title=doc_title or data.get("summary", "")[:50],
                    summary=data.get("summary", ""),
                    topics=data.get("topics", []),
                    doc_type=data.get("doc_type", "document")
                )
        except Exception as e:
            logger.warning(f"Document analysis failed: {e}")

        # Fallback
        return DocumentContext(
            doc_id=hashlib.md5(document_content[:1000].encode()).hexdigest()[:16],
            title=doc_title or "Untitled Document",
            summary="",
            topics=[],
            doc_type="document"
        )

    async def generate_chunk_context(
        self,
        chunk_content: str,
        document_content: str,
        doc_context: DocumentContext
    ) -> str:
        """Generate context for a single chunk using LLM.

        Args:
            chunk_content: The chunk to contextualize
            document_content: Full document for reference
            doc_context: Document-level context

        Returns:
            Generated context string
        """
        # Truncate document if too long (keep ~4000 chars around chunk)
        chunk_pos = document_content.find(chunk_content[:100])
        if len(document_content) > 6000:
            start = max(0, chunk_pos - 2000) if chunk_pos > 0 else 0
            end = min(len(document_content), start + 6000)
            doc_truncated = document_content[start:end]
        else:
            doc_truncated = document_content

        prompt = ContextPrompts.CHUNK_CONTEXT.format(
            document_content=doc_truncated,
            chunk_content=chunk_content
        )

        async with self._semaphore:
            try:
                context = await asyncio.wait_for(
                    self.llm.generate(
                        prompt,
                        max_tokens=self.config.max_context_tokens * 2,
                        temperature=0.0
                    ),
                    timeout=self.config.timeout_seconds
                )
                return context.strip()
            except asyncio.TimeoutError:
                logger.warning("Context generation timed out")
                raise
            except Exception as e:
                logger.warning(f"Context generation failed: {e}")
                raise


# =============================================================================
# CONTEXTUAL RETRIEVER
# =============================================================================

class ContextualRetriever:
    """Main class for Anthropic's contextual retrieval pattern.

    Prepends document-level context to chunks before embedding to improve
    retrieval accuracy. Supports LLM, rule-based, and hybrid strategies.

    Example:
        >>> retriever = ContextualRetriever(llm=my_llm, embedding_provider=embedder)
        >>> chunks = await retriever.contextualize_document(
        ...     document=doc_text,
        ...     doc_title="API Reference"
        ... )
        >>> for chunk in chunks:
        ...     print(f"Context: {chunk.prepended_context}")
        ...     store_embedding(chunk.contextualized_embedding)
    """

    def __init__(
        self,
        llm: Optional[LLMProvider] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        chunker: Optional["SemanticChunker"] = None,
        config: Optional[ContextualRetrievalConfig] = None
    ):
        """Initialize the contextual retriever.

        Args:
            llm: LLM provider for context generation (optional if rule-based)
            embedding_provider: Provider for generating embeddings
            chunker: SemanticChunker instance (creates default if None)
            config: Configuration options
        """
        self.llm = llm
        self.embedding_provider = embedding_provider
        self.config = config or ContextualRetrievalConfig()

        # Initialize chunker
        if chunker is None:
            from .semantic_chunker import SemanticChunker
            self.chunker = SemanticChunker(embed_chunks=False)
        else:
            self.chunker = chunker

        # Initialize generators
        self._rule_generator = RuleBasedContextGenerator()
        self._llm_generator = LLMContextGenerator(llm, self.config) if llm else None

        # Initialize cache
        self._cache = ContextCache(
            cache_dir=self.config.cache_dir,
            memory_size=self.config.memory_cache_size
        ) if self.config.cache_enabled else None

    async def contextualize_document(
        self,
        document: str,
        doc_title: str = "",
        doc_metadata: Optional[Dict[str, Any]] = None,
        existing_chunks: Optional[List["Chunk"]] = None
    ) -> List[ContextualizedChunk]:
        """Process a document with contextual retrieval.

        This is the main entry point. It:
        1. Chunks the document (or uses existing chunks)
        2. Analyzes document for high-level context
        3. Generates chunk-specific contexts (with caching)
        4. Prepends contexts and generates embeddings

        Args:
            document: Full document text
            doc_title: Document title
            doc_metadata: Optional metadata
            existing_chunks: Pre-chunked content (skips chunking step)

        Returns:
            List of ContextualizedChunk objects
        """
        start_time = time.time()
        stats = ContextualizationStats()

        # Step 1: Chunk document if not already chunked
        if existing_chunks is None:
            chunks = self.chunker.chunk(document, metadata=doc_metadata)
        else:
            chunks = existing_chunks

        stats.total_chunks = len(chunks)

        if not chunks:
            return []

        # Step 2: Analyze document for high-level context
        doc_context = await self._get_document_context(document, doc_title, doc_metadata)

        # Step 3: Generate context for each chunk
        contextualized_chunks: List[ContextualizedChunk] = []

        # Process in batches
        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i:i + self.config.batch_size]
            batch_results = await self._process_chunk_batch(
                batch, document, doc_context, stats
            )
            contextualized_chunks.extend(batch_results)

        # Step 4: Generate embeddings for all chunks
        if self.embedding_provider:
            await self._embed_chunks(contextualized_chunks)

        stats.total_time_ms = (time.time() - start_time) * 1000

        # Calculate average context tokens
        if contextualized_chunks:
            total_context_tokens = sum(
                len(c.prepended_context) // 4 for c in contextualized_chunks
            )
            stats.avg_context_tokens = total_context_tokens / len(contextualized_chunks)

        logger.info(
            f"Contextualized {stats.total_chunks} chunks in {stats.total_time_ms:.0f}ms "
            f"(cache_hits={stats.cache_hits}, llm_gen={stats.llm_generations}, "
            f"rule_fallback={stats.rule_based_fallbacks})"
        )

        return contextualized_chunks

    async def contextualize_chunk(
        self,
        chunk_content: str,
        document: str,
        doc_context: Optional[DocumentContext] = None,
        doc_title: str = ""
    ) -> ContextualizedChunk:
        """Contextualize a single chunk.

        Args:
            chunk_content: The chunk text
            document: Full document for context
            doc_context: Optional pre-computed document context
            doc_title: Document title

        Returns:
            ContextualizedChunk with prepended context
        """
        if doc_context is None:
            doc_context = await self._get_document_context(document, doc_title, None)

        stats = ContextualizationStats(total_chunks=1)
        results = await self._process_chunk_batch(
            [type('Chunk', (), {'content': chunk_content, 'chunk_id': 'single'})()],
            document,
            doc_context,
            stats
        )

        if results:
            result = results[0]
            if self.embedding_provider:
                await self._embed_chunks([result])
            return result
        else:
            # Fallback to basic chunk
            return ContextualizedChunk(
                chunk_id="single",
                original_content=chunk_content,
                prepended_context=f"From document: {doc_title}",
                contextualized_content=f"From document: {doc_title}\n\n{chunk_content}",
                document_context=doc_context
            )

    async def _get_document_context(
        self,
        document: str,
        doc_title: str,
        metadata: Optional[Dict[str, Any]]
    ) -> DocumentContext:
        """Get or generate document-level context."""
        if self._llm_generator and self.config.strategy in (
            ContextGenerationStrategy.LLM,
            ContextGenerationStrategy.HYBRID
        ):
            try:
                return await self._llm_generator.analyze_document(document, doc_title)
            except Exception as e:
                logger.warning(f"Document analysis failed: {e}")

        # Fallback to basic context
        return DocumentContext(
            doc_id=hashlib.md5(document[:1000].encode()).hexdigest()[:16],
            title=doc_title or "Untitled",
            metadata=metadata or {}
        )

    async def _process_chunk_batch(
        self,
        chunks: List,
        document: str,
        doc_context: DocumentContext,
        stats: ContextualizationStats
    ) -> List[ContextualizedChunk]:
        """Process a batch of chunks for contextualization."""
        results: List[ContextualizedChunk] = []
        tasks: List[asyncio.Task] = []

        for chunk in chunks:
            chunk_content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            chunk_id = getattr(chunk, 'chunk_id', None) or hashlib.md5(
                chunk_content[:100].encode()
            ).hexdigest()[:12]

            # Check cache first
            if self._cache:
                cached_context = self._cache.get(chunk_content, doc_context.doc_id)
                if cached_context:
                    stats.cache_hits += 1
                    results.append(self._create_contextualized_chunk(
                        chunk_id=chunk_id,
                        chunk_content=chunk_content,
                        context=cached_context,
                        doc_context=doc_context,
                        cache_hit=True
                    ))
                    continue

            # Schedule context generation
            task = asyncio.create_task(
                self._generate_context(chunk_id, chunk_content, document, doc_context, stats)
            )
            tasks.append(task)

        # Wait for all tasks
        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in task_results:
                if isinstance(result, ContextualizedChunk):
                    results.append(result)
                    # Cache the result
                    if self._cache and not result.cache_hit:
                        self._cache.put(
                            result.original_content,
                            doc_context.doc_id,
                            result.prepended_context
                        )
                elif isinstance(result, Exception):
                    logger.error(f"Chunk contextualization failed: {result}")

        return results

    async def _generate_context(
        self,
        chunk_id: str,
        chunk_content: str,
        document: str,
        doc_context: DocumentContext,
        stats: ContextualizationStats
    ) -> ContextualizedChunk:
        """Generate context for a single chunk."""
        start_time = time.time()
        context = ""

        strategy = self.config.strategy

        # Try LLM first if configured
        if strategy in (ContextGenerationStrategy.LLM, ContextGenerationStrategy.HYBRID):
            if self._llm_generator:
                try:
                    context = await self._llm_generator.generate_chunk_context(
                        chunk_content, document, doc_context
                    )
                    stats.llm_generations += 1
                except Exception as e:
                    logger.warning(f"LLM context generation failed: {e}")
                    if strategy == ContextGenerationStrategy.LLM:
                        raise

        # Fallback to rule-based if needed
        if not context and strategy in (
            ContextGenerationStrategy.RULE_BASED,
            ContextGenerationStrategy.HYBRID
        ):
            context = self._rule_generator.extract_context(
                chunk_content, document, doc_context
            )
            stats.rule_based_fallbacks += 1

        # Final fallback
        if not context:
            context = f"This chunk is from '{doc_context.title}'."

        generation_time = (time.time() - start_time) * 1000

        return self._create_contextualized_chunk(
            chunk_id=chunk_id,
            chunk_content=chunk_content,
            context=context,
            doc_context=doc_context,
            cache_hit=False,
            generation_time_ms=generation_time
        )

    def _create_contextualized_chunk(
        self,
        chunk_id: str,
        chunk_content: str,
        context: str,
        doc_context: DocumentContext,
        cache_hit: bool,
        generation_time_ms: float = 0.0
    ) -> ContextualizedChunk:
        """Create a ContextualizedChunk instance."""
        # Combine context with content
        contextualized_content = f"{context}\n\n{chunk_content}"

        return ContextualizedChunk(
            chunk_id=chunk_id,
            original_content=chunk_content,
            prepended_context=context,
            contextualized_content=contextualized_content,
            document_context=doc_context,
            context_generation_time_ms=generation_time_ms,
            cache_hit=cache_hit
        )

    async def _embed_chunks(
        self,
        chunks: List[ContextualizedChunk]
    ) -> None:
        """Generate embeddings for contextualized chunks."""
        if not self.embedding_provider:
            return

        # Embed contextualized content
        contextualized_texts = [c.contextualized_content for c in chunks]
        contextualized_embeddings = self.embedding_provider.embed_batch(contextualized_texts)

        # Optionally embed original content for comparison
        original_texts = [c.original_content for c in chunks]
        original_embeddings = self.embedding_provider.embed_batch(original_texts)

        for chunk, ctx_emb, orig_emb in zip(
            chunks, contextualized_embeddings, original_embeddings
        ):
            chunk.contextualized_embedding = ctx_emb
            chunk.original_embedding = orig_emb

    def invalidate_cache(self, doc_id: str) -> int:
        """Invalidate cached contexts for a document.

        Args:
            doc_id: Document identifier

        Returns:
            Number of entries invalidated
        """
        if self._cache:
            return self._cache.invalidate_document(doc_id)
        return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self._cache:
            return self._cache.get_stats()
        return {"enabled": False}

    def close(self) -> None:
        """Close resources."""
        if self._cache:
            self._cache.close()


# =============================================================================
# SEMANTIC CHUNKER INTEGRATION
# =============================================================================

class ContextualSemanticChunker:
    """SemanticChunker with integrated contextual retrieval.

    This class combines semantic chunking with Anthropic's contextual
    retrieval pattern for optimal RAG performance.

    Example:
        >>> chunker = ContextualSemanticChunker(llm=my_llm)
        >>> chunks = await chunker.chunk_with_context(
        ...     document_text,
        ...     doc_title="User Guide"
        ... )
    """

    def __init__(
        self,
        llm: Optional[LLMProvider] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        chunker_config: Optional[Dict[str, Any]] = None,
        retrieval_config: Optional[ContextualRetrievalConfig] = None
    ):
        """Initialize the contextual semantic chunker.

        Args:
            llm: LLM provider for context generation
            embedding_provider: Provider for embeddings
            chunker_config: Config kwargs for SemanticChunker
            retrieval_config: Config for contextual retrieval
        """
        from .semantic_chunker import SemanticChunker

        chunker_kwargs = chunker_config or {}
        chunker_kwargs['embed_chunks'] = False  # We'll embed after contextualization

        self.chunker = SemanticChunker(**chunker_kwargs)
        self.retriever = ContextualRetriever(
            llm=llm,
            embedding_provider=embedding_provider,
            chunker=self.chunker,
            config=retrieval_config
        )

    async def chunk_with_context(
        self,
        text: str,
        doc_title: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[ContextualizedChunk]:
        """Chunk text and add contextual information.

        Args:
            text: Document text to chunk
            doc_title: Document title
            metadata: Optional metadata

        Returns:
            List of ContextualizedChunk objects
        """
        return await self.retriever.contextualize_document(
            document=text,
            doc_title=doc_title,
            doc_metadata=metadata
        )

    def get_stats(self, chunks: List[ContextualizedChunk]) -> Dict[str, Any]:
        """Get statistics about contextualized chunks."""
        if not chunks:
            return {}

        return {
            "total_chunks": len(chunks),
            "cache_hits": sum(1 for c in chunks if c.cache_hit),
            "avg_context_length": sum(len(c.prepended_context) for c in chunks) / len(chunks),
            "avg_total_length": sum(len(c.contextualized_content) for c in chunks) / len(chunks),
            "has_embeddings": all(c.contextualized_embedding is not None for c in chunks)
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    "ContextualRetriever",
    "ContextualSemanticChunker",
    # Configuration
    "ContextualRetrievalConfig",
    "ContextGenerationStrategy",
    # Data structures
    "ContextualizedChunk",
    "DocumentContext",
    "ContextualizationStats",
    # Generators
    "LLMContextGenerator",
    "RuleBasedContextGenerator",
    # Cache
    "ContextCache",
    # Protocols
    "LLMProvider",
    "EmbeddingProvider",
]
