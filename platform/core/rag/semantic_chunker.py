#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.24.0",
# ]
# ///
"""
Semantic Chunking System for Improved RAG

This module provides intelligent text chunking that respects semantic boundaries
rather than arbitrary character counts. It uses sentence embeddings to find
natural breakpoints where topic or context shifts occur.

Key Features:
- Semantic-aware text splitting using embedding similarity
- Content-type detection (code, markdown, plain text)
- Overlap management for context preservation
- Metadata preservation through chunks
- Fallback to simple sentence splitting when embeddings unavailable

Architecture:
    Text -> Sentences -> Embeddings -> Similarity Scores -> Boundary Detection -> Chunks

Integration:
    from core.rag.semantic_chunker import SemanticChunker, Chunk

    chunker = SemanticChunker(max_chunk_size=512, overlap=50)
    chunks = chunker.chunk(document_text, metadata={"source": "api_docs"})

    for chunk in chunks:
        store_in_memory(chunk.content, chunk.embedding, chunk.metadata)
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

class ContentType(str, Enum):
    """Detected content types for specialized chunking."""
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    CODE_PYTHON = "code_python"
    CODE_JAVASCRIPT = "code_javascript"
    CODE_TYPESCRIPT = "code_typescript"
    CODE_GO = "code_go"
    CODE_RUST = "code_rust"
    CODE_GENERIC = "code_generic"
    JSON = "json"
    YAML = "yaml"
    MIXED = "mixed"


@dataclass
class Chunk:
    """A semantically coherent chunk of text with metadata.

    Attributes:
        content: The actual text content of the chunk
        start_idx: Character start position in original text
        end_idx: Character end position in original text
        metadata: Preserved metadata plus chunk-specific info
        embedding: Optional pre-computed embedding vector
        chunk_id: Unique identifier for this chunk
        content_type: Detected content type
        sentences: Number of sentences in this chunk
    """
    content: str
    start_idx: int
    end_idx: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    chunk_id: Optional[str] = None
    content_type: ContentType = ContentType.PLAIN_TEXT
    sentences: int = 0

    def __post_init__(self):
        if self.chunk_id is None:
            self.chunk_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate a deterministic chunk ID from content."""
        hash_input = f"{self.content[:100]}:{self.start_idx}:{self.end_idx}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    @property
    def token_estimate(self) -> int:
        """Estimate token count (rough: ~4 chars per token)."""
        return len(self.content) // 4

    @property
    def word_count(self) -> int:
        """Get word count."""
        return len(self.content.split())


@dataclass
class ChunkingStats:
    """Statistics about the chunking operation."""
    total_chunks: int = 0
    total_characters: int = 0
    avg_chunk_size: float = 0.0
    min_chunk_size: int = 0
    max_chunk_size: int = 0
    semantic_boundaries_found: int = 0
    fallback_splits: int = 0
    merged_chunks: int = 0
    content_type: ContentType = ContentType.PLAIN_TEXT


# =============================================================================
# CONTENT TYPE DETECTION
# =============================================================================

class ContentTypeDetector:
    """Detect content type from text for specialized chunking strategies."""

    # Language-specific patterns
    PYTHON_PATTERNS = [
        r'^def\s+\w+\s*\(',
        r'^class\s+\w+',
        r'^import\s+\w+',
        r'^from\s+\w+\s+import',
        r'^\s*@\w+',  # decorators
        r'if\s+__name__\s*==',
    ]

    JAVASCRIPT_PATTERNS = [
        r'\bfunction\s+\w+\s*\(',
        r'\bconst\s+\w+\s*=',
        r'\blet\s+\w+\s*=',
        r'\bvar\s+\w+\s*=',
        r'=>',  # arrow functions
        r'\bexport\s+(default\s+)?',
        r'\brequire\s*\(',
    ]

    TYPESCRIPT_PATTERNS = [
        r':\s*(string|number|boolean|any|void)\b',
        r'\binterface\s+\w+',
        r'\btype\s+\w+\s*=',
        r'<\w+>',  # generics
    ]

    GO_PATTERNS = [
        r'^package\s+\w+',
        r'^func\s+(\(\w+\s+\*?\w+\)\s+)?\w+',
        r'^type\s+\w+\s+(struct|interface)',
        r'^import\s+\(',
    ]

    RUST_PATTERNS = [
        r'^fn\s+\w+',
        r'^struct\s+\w+',
        r'^impl\s+',
        r'^use\s+',
        r'^pub\s+(fn|struct|mod|use)',
        r'let\s+mut\s+',
    ]

    MARKDOWN_PATTERNS = [
        r'^#{1,6}\s+',  # headers
        r'^\*\*[^*]+\*\*',  # bold
        r'^\*[^*]+\*',  # italic
        r'^\[.+\]\(.+\)',  # links
        r'^```',  # code blocks
        r'^\s*[-*+]\s+',  # lists
        r'^\s*\d+\.\s+',  # numbered lists
    ]

    @classmethod
    def detect(cls, text: str) -> ContentType:
        """Detect content type from text sample.

        Args:
            text: Text to analyze (first ~2000 chars)

        Returns:
            Detected ContentType
        """
        sample = text[:2000]
        lines = sample.split('\n')

        # Count matches for each type
        scores: Dict[ContentType, int] = {
            ContentType.CODE_PYTHON: 0,
            ContentType.CODE_JAVASCRIPT: 0,
            ContentType.CODE_TYPESCRIPT: 0,
            ContentType.CODE_GO: 0,
            ContentType.CODE_RUST: 0,
            ContentType.MARKDOWN: 0,
            ContentType.JSON: 0,
            ContentType.YAML: 0,
        }

        # Check for JSON
        stripped = sample.strip()
        if stripped.startswith('{') or stripped.startswith('['):
            try:
                import json
                json.loads(text)
                return ContentType.JSON
            except (json.JSONDecodeError, ValueError):
                pass

        # Check for YAML
        if ':' in sample and not stripped.startswith('{'):
            yaml_like_lines = sum(1 for line in lines if re.match(r'^\w+:\s', line))
            if yaml_like_lines > 2:
                scores[ContentType.YAML] = yaml_like_lines * 2

        # Score each language
        for line in lines:
            line = line.rstrip()

            for pattern in cls.PYTHON_PATTERNS:
                if re.search(pattern, line, re.MULTILINE):
                    scores[ContentType.CODE_PYTHON] += 2

            for pattern in cls.JAVASCRIPT_PATTERNS:
                if re.search(pattern, line):
                    scores[ContentType.CODE_JAVASCRIPT] += 2

            for pattern in cls.TYPESCRIPT_PATTERNS:
                if re.search(pattern, line):
                    scores[ContentType.CODE_TYPESCRIPT] += 3

            for pattern in cls.GO_PATTERNS:
                if re.search(pattern, line, re.MULTILINE):
                    scores[ContentType.CODE_GO] += 2

            for pattern in cls.RUST_PATTERNS:
                if re.search(pattern, line, re.MULTILINE):
                    scores[ContentType.CODE_RUST] += 2

            for pattern in cls.MARKDOWN_PATTERNS:
                if re.search(pattern, line, re.MULTILINE):
                    scores[ContentType.MARKDOWN] += 1

        # TypeScript is JS superset - if TS detected, upgrade from JS
        if scores[ContentType.CODE_TYPESCRIPT] > 0 and scores[ContentType.CODE_JAVASCRIPT] > 0:
            scores[ContentType.CODE_TYPESCRIPT] += scores[ContentType.CODE_JAVASCRIPT]
            scores[ContentType.CODE_JAVASCRIPT] = 0

        # Find max score
        max_score = max(scores.values())
        if max_score < 3:
            return ContentType.PLAIN_TEXT

        # Return highest scoring type
        for content_type, score in scores.items():
            if score == max_score:
                return content_type

        return ContentType.PLAIN_TEXT


# =============================================================================
# SENTENCE SPLITTER
# =============================================================================

class SentenceSplitter:
    """Split text into sentences with content-type awareness."""

    # Standard sentence-ending patterns
    SENTENCE_ENDINGS = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z])|'  # Standard sentence end
        r'(?<=[.!?])\s*\n+|'  # Sentence end at newline
        r'\n{2,}'  # Paragraph breaks
    )

    # Code-specific split points
    CODE_SPLIT_PATTERNS = {
        ContentType.CODE_PYTHON: [
            r'\n(?=def\s)',  # function definitions
            r'\n(?=class\s)',  # class definitions
            r'\n(?=@)',  # decorators
            r'\n{2,}',  # blank lines
        ],
        ContentType.CODE_JAVASCRIPT: [
            r'\n(?=function\s)',
            r'\n(?=const\s+\w+\s*=\s*(?:async\s+)?\()',  # arrow functions
            r'\n(?=export\s)',
            r'\n{2,}',
        ],
        ContentType.CODE_TYPESCRIPT: [
            r'\n(?=function\s)',
            r'\n(?=const\s+\w+\s*=\s*(?:async\s+)?\()',
            r'\n(?=export\s)',
            r'\n(?=interface\s)',
            r'\n(?=type\s)',
            r'\n{2,}',
        ],
        ContentType.CODE_GO: [
            r'\n(?=func\s)',
            r'\n(?=type\s)',
            r'\n{2,}',
        ],
        ContentType.CODE_RUST: [
            r'\n(?=fn\s)',
            r'\n(?=pub\s)',
            r'\n(?=impl\s)',
            r'\n(?=struct\s)',
            r'\n{2,}',
        ],
    }

    # Markdown split points
    MARKDOWN_SPLITS = [
        r'\n(?=#{1,6}\s)',  # headers
        r'\n(?=```)',  # code blocks
        r'\n{2,}',  # paragraph breaks
    ]

    @classmethod
    def split(
        cls,
        text: str,
        content_type: ContentType = ContentType.PLAIN_TEXT
    ) -> List[Tuple[str, int, int]]:
        """Split text into sentences/segments.

        Args:
            text: Text to split
            content_type: Detected content type

        Returns:
            List of (sentence, start_idx, end_idx) tuples
        """
        if content_type == ContentType.MARKDOWN:
            return cls._split_markdown(text)
        elif content_type.value.startswith('code_'):
            return cls._split_code(text, content_type)
        elif content_type in (ContentType.JSON, ContentType.YAML):
            return cls._split_structured(text)
        else:
            return cls._split_prose(text)

    @classmethod
    def _split_prose(cls, text: str) -> List[Tuple[str, int, int]]:
        """Split prose text into sentences."""
        sentences = []
        last_end = 0

        for match in cls.SENTENCE_ENDINGS.finditer(text):
            sentence = text[last_end:match.start()].strip()
            if sentence:
                sentences.append((sentence, last_end, match.start()))
            last_end = match.end()

        # Don't forget the last segment
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                sentences.append((remaining, last_end, len(text)))

        # If no splits found, return entire text
        if not sentences:
            sentences.append((text.strip(), 0, len(text)))

        return sentences

    @classmethod
    def _split_code(
        cls,
        text: str,
        content_type: ContentType
    ) -> List[Tuple[str, int, int]]:
        """Split code into logical units (functions, classes, etc.)."""
        patterns = cls.CODE_SPLIT_PATTERNS.get(
            content_type,
            [r'\n{2,}']  # default: split on blank lines
        )

        # Combine patterns
        combined = '|'.join(f'({p})' for p in patterns)
        split_regex = re.compile(combined)

        segments = []
        last_end = 0

        for match in split_regex.finditer(text):
            segment = text[last_end:match.start()].strip()
            if segment:
                segments.append((segment, last_end, match.start()))
            last_end = match.start()

        # Last segment
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                segments.append((remaining, last_end, len(text)))

        if not segments:
            segments.append((text.strip(), 0, len(text)))

        return segments

    @classmethod
    def _split_markdown(cls, text: str) -> List[Tuple[str, int, int]]:
        """Split markdown into logical sections."""
        combined = '|'.join(f'({p})' for p in cls.MARKDOWN_SPLITS)
        split_regex = re.compile(combined)

        segments = []
        last_end = 0

        for match in split_regex.finditer(text):
            segment = text[last_end:match.start()].strip()
            if segment:
                segments.append((segment, last_end, match.start()))
            last_end = match.start()

        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                segments.append((remaining, last_end, len(text)))

        if not segments:
            segments.append((text.strip(), 0, len(text)))

        return segments

    @classmethod
    def _split_structured(cls, text: str) -> List[Tuple[str, int, int]]:
        """Split JSON/YAML into logical blocks."""
        # For JSON/YAML, split on top-level objects/arrays
        segments = []
        lines = text.split('\n')
        current_segment = []
        current_start = 0
        char_pos = 0

        for i, line in enumerate(lines):
            current_segment.append(line)

            # Check for natural break points (empty lines, top-level keys)
            is_break = (
                not line.strip() or
                (i > 0 and re.match(r'^["\w]', line) and not line.startswith(' '))
            )

            if is_break and len('\n'.join(current_segment)) > 50:
                segment_text = '\n'.join(current_segment[:-1]).strip()
                if segment_text:
                    segments.append((segment_text, current_start, char_pos))
                current_segment = [line] if line.strip() else []
                current_start = char_pos

            char_pos += len(line) + 1  # +1 for newline

        # Final segment
        if current_segment:
            segment_text = '\n'.join(current_segment).strip()
            if segment_text:
                segments.append((segment_text, current_start, len(text)))

        if not segments:
            segments.append((text.strip(), 0, len(text)))

        return segments


# =============================================================================
# EMBEDDING PROVIDER
# =============================================================================

class EmbeddingProvider:
    """Manages embedding generation with fallback support."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        custom_provider: Optional[Callable[[str], List[float]]] = None
    ):
        """Initialize embedding provider.

        Args:
            model_name: Name of sentence-transformers model
            custom_provider: Optional custom embedding function
        """
        self.model_name = model_name
        self.custom_provider = custom_provider
        self._model = None
        self._available = None

    @property
    def available(self) -> bool:
        """Check if embeddings are available."""
        if self._available is None:
            if self.custom_provider is not None:
                self._available = True
            else:
                try:
                    self._load_model()
                    self._available = True
                except Exception as e:
                    logger.warning(f"Embeddings unavailable: {e}")
                    self._available = False
        return self._available

    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None and self.custom_provider is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )

    def embed(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if unavailable
        """
        if not self.available:
            return None

        try:
            if self.custom_provider is not None:
                return self.custom_provider(text)
            else:
                self._load_model()
                # Truncate to model's max length
                truncated = text[:5000]
                embedding = self._model.encode(truncated)
                return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (None for failures)
        """
        if not self.available:
            return [None] * len(texts)

        try:
            if self.custom_provider is not None:
                return [self.custom_provider(t) for t in texts]
            else:
                self._load_model()
                truncated = [t[:5000] for t in texts]
                embeddings = self._model.encode(truncated)
                return [e.tolist() for e in embeddings]
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return [None] * len(texts)


# =============================================================================
# SEMANTIC BOUNDARY DETECTION
# =============================================================================

class SemanticBoundaryDetector:
    """Detect semantic boundaries using embedding similarity."""

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        window_size: int = 3
    ):
        """Initialize boundary detector.

        Args:
            similarity_threshold: Cosine similarity below which indicates boundary
            window_size: Number of sentences to consider for smoothing
        """
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size

    def find_boundaries(
        self,
        sentences: List[str],
        embeddings: List[Optional[List[float]]]
    ) -> List[int]:
        """Find semantic boundary indices.

        Args:
            sentences: List of sentences
            embeddings: Corresponding embeddings

        Returns:
            List of indices where boundaries should be placed
        """
        if len(sentences) < 2:
            return []

        # Filter out None embeddings
        valid_indices = [i for i, e in enumerate(embeddings) if e is not None]

        if len(valid_indices) < 2:
            # Not enough embeddings, return empty (no semantic boundaries)
            return []

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(valid_indices) - 1):
            idx1, idx2 = valid_indices[i], valid_indices[i + 1]
            sim = self._cosine_similarity(
                embeddings[idx1],  # type: ignore
                embeddings[idx2]   # type: ignore
            )
            similarities.append((idx1, idx2, sim))

        # Apply smoothing window
        smoothed = self._smooth_similarities(similarities)

        # Find boundaries (low similarity points)
        boundaries = []
        for i, (idx1, idx2, sim) in enumerate(smoothed):
            if sim < self.similarity_threshold:
                # Boundary after idx1
                boundaries.append(idx1 + 1)

        return boundaries

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a_arr = np.array(a)
        b_arr = np.array(b)

        dot = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot / (norm_a * norm_b))

    def _smooth_similarities(
        self,
        similarities: List[Tuple[int, int, float]]
    ) -> List[Tuple[int, int, float]]:
        """Apply smoothing window to similarities."""
        if len(similarities) < self.window_size:
            return similarities

        smoothed = []
        half_window = self.window_size // 2

        for i, (idx1, idx2, _) in enumerate(similarities):
            # Get window of similarities
            start = max(0, i - half_window)
            end = min(len(similarities), i + half_window + 1)
            window_sims = [s[2] for s in similarities[start:end]]
            avg_sim = sum(window_sims) / len(window_sims)
            smoothed.append((idx1, idx2, avg_sim))

        return smoothed


# =============================================================================
# SEMANTIC CHUNKER
# =============================================================================

class SemanticChunker:
    """Chunks text based on semantic boundaries.

    This chunker uses sentence embeddings to find natural breakpoints
    where the topic or context shifts, creating more coherent chunks
    for RAG applications.

    Example:
        >>> chunker = SemanticChunker(max_chunk_size=512, overlap=50)
        >>> chunks = chunker.chunk(document_text)
        >>> for chunk in chunks:
        ...     print(f"Chunk {chunk.chunk_id}: {len(chunk.content)} chars")
    """

    def __init__(
        self,
        max_chunk_size: int = 512,
        min_chunk_size: int = 100,
        overlap: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        embed_chunks: bool = True,
        custom_embedding_provider: Optional[Callable[[str], List[float]]] = None
    ):
        """Initialize the semantic chunker.

        Args:
            max_chunk_size: Maximum tokens per chunk (~4 chars per token)
            min_chunk_size: Minimum tokens per chunk
            overlap: Tokens of overlap between chunks for context
            embedding_model: Sentence-transformers model name
            similarity_threshold: Similarity below which indicates boundary
            embed_chunks: Whether to compute embeddings for output chunks
            custom_embedding_provider: Optional custom embedding function
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
        self.similarity_threshold = similarity_threshold
        self.embed_chunks = embed_chunks

        # Characters per token estimate
        self.chars_per_token = 4
        self.max_chars = max_chunk_size * self.chars_per_token
        self.min_chars = min_chunk_size * self.chars_per_token
        self.overlap_chars = overlap * self.chars_per_token

        # Initialize components
        self.embedding_provider = EmbeddingProvider(
            model_name=embedding_model,
            custom_provider=custom_embedding_provider
        )
        self.boundary_detector = SemanticBoundaryDetector(
            similarity_threshold=similarity_threshold
        )
        self.content_detector = ContentTypeDetector()
        self.sentence_splitter = SentenceSplitter()

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Split text into semantically coherent chunks.

        Args:
            text: Text to chunk
            metadata: Optional metadata to preserve in chunks

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}

        # Detect content type
        content_type = self.content_detector.detect(text)
        logger.debug(f"Detected content type: {content_type}")

        # Split into sentences/segments
        segments = self.sentence_splitter.split(text, content_type)
        logger.debug(f"Split into {len(segments)} segments")

        # Try semantic boundary detection
        chunks = self._chunk_with_embeddings(segments, text, content_type, metadata)

        if not chunks:
            # Fallback to simple chunking
            logger.debug("Falling back to simple chunking")
            chunks = self._chunk_simple(segments, text, content_type, metadata)

        # Merge small chunks
        chunks = self._merge_small_chunks(chunks)

        # Add overlap
        chunks = self._add_overlap(chunks, text)

        # Compute final embeddings if requested
        if self.embed_chunks and self.embedding_provider.available:
            chunks = self._embed_chunks(chunks)

        return chunks

    def _chunk_with_embeddings(
        self,
        segments: List[Tuple[str, int, int]],
        full_text: str,
        content_type: ContentType,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Create chunks using semantic boundary detection."""
        if not self.embedding_provider.available:
            return []

        # Generate embeddings for segments
        segment_texts = [s[0] for s in segments]
        embeddings = self.embedding_provider.embed_batch(segment_texts)

        # Find semantic boundaries
        boundaries = self.boundary_detector.find_boundaries(segment_texts, embeddings)

        if not boundaries:
            # No semantic boundaries found, return empty for fallback
            return []

        # Create chunks based on boundaries
        chunks = []
        current_start = 0

        for boundary_idx in boundaries:
            # Collect segments up to boundary
            chunk_segments = segments[current_start:boundary_idx]
            if chunk_segments:
                chunk = self._create_chunk_from_segments(
                    chunk_segments, content_type, metadata
                )
                chunks.append(chunk)
            current_start = boundary_idx

        # Handle remaining segments
        if current_start < len(segments):
            chunk_segments = segments[current_start:]
            if chunk_segments:
                chunk = self._create_chunk_from_segments(
                    chunk_segments, content_type, metadata
                )
                chunks.append(chunk)

        # Split any chunks that are too large
        final_chunks = []
        for chunk in chunks:
            if len(chunk.content) > self.max_chars:
                split_chunks = self._split_large_chunk(chunk)
                final_chunks.extend(split_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _chunk_simple(
        self,
        segments: List[Tuple[str, int, int]],
        full_text: str,
        content_type: ContentType,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Create chunks using simple size-based splitting (fallback)."""
        chunks = []
        current_segments: List[Tuple[str, int, int]] = []
        current_size = 0

        for segment in segments:
            segment_text, start_idx, end_idx = segment
            segment_size = len(segment_text)

            # If adding this segment exceeds max, create chunk
            if current_size + segment_size > self.max_chars and current_segments:
                chunk = self._create_chunk_from_segments(
                    current_segments, content_type, metadata
                )
                chunks.append(chunk)
                current_segments = []
                current_size = 0

            current_segments.append(segment)
            current_size += segment_size

        # Final chunk
        if current_segments:
            chunk = self._create_chunk_from_segments(
                current_segments, content_type, metadata
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk_from_segments(
        self,
        segments: List[Tuple[str, int, int]],
        content_type: ContentType,
        metadata: Dict[str, Any]
    ) -> Chunk:
        """Create a Chunk from a list of segments."""
        if not segments:
            raise ValueError("Cannot create chunk from empty segments")

        # Combine segment texts
        combined_text = '\n'.join(s[0] for s in segments)
        start_idx = segments[0][1]
        end_idx = segments[-1][2]

        # Create chunk metadata
        chunk_metadata = {
            **metadata,
            'segment_count': len(segments),
        }

        return Chunk(
            content=combined_text,
            start_idx=start_idx,
            end_idx=end_idx,
            metadata=chunk_metadata,
            content_type=content_type,
            sentences=len(segments)
        )

    def _split_large_chunk(self, chunk: Chunk) -> List[Chunk]:
        """Split a chunk that exceeds max size."""
        text = chunk.content
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.max_chars

            # Try to find a good break point (sentence end, newline)
            if end < len(text):
                # Look for sentence end
                search_start = max(start, end - 200)
                best_break = end

                # Check for sentence endings
                for pattern in ['. ', '.\n', '\n\n', '\n']:
                    pos = text.rfind(pattern, search_start, end)
                    if pos > start:
                        best_break = pos + len(pattern)
                        break

                end = best_break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    content=chunk_text,
                    start_idx=chunk.start_idx + start,
                    end_idx=chunk.start_idx + end,
                    metadata=chunk.metadata.copy(),
                    content_type=chunk.content_type,
                    sentences=chunk_text.count('. ') + 1
                ))

            start = end

        return chunks

    def _merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge chunks that are too small."""
        if not chunks:
            return chunks

        merged = []
        current: Optional[Chunk] = None

        for chunk in chunks:
            if current is None:
                current = chunk
            elif len(current.content) + len(chunk.content) < self.min_chars:
                # Merge with current
                current = Chunk(
                    content=current.content + '\n' + chunk.content,
                    start_idx=current.start_idx,
                    end_idx=chunk.end_idx,
                    metadata={**current.metadata, **chunk.metadata},
                    content_type=current.content_type,
                    sentences=current.sentences + chunk.sentences
                )
            else:
                if len(current.content) >= self.min_chars:
                    merged.append(current)
                    current = chunk
                else:
                    # Current too small, merge with next
                    current = Chunk(
                        content=current.content + '\n' + chunk.content,
                        start_idx=current.start_idx,
                        end_idx=chunk.end_idx,
                        metadata={**current.metadata, **chunk.metadata},
                        content_type=current.content_type,
                        sentences=current.sentences + chunk.sentences
                    )

        if current is not None:
            merged.append(current)

        return merged

    def _add_overlap(self, chunks: List[Chunk], full_text: str) -> List[Chunk]:
        """Add overlap from previous chunk for context preservation."""
        if not chunks or self.overlap_chars <= 0:
            return chunks

        overlapped = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]

            # Get overlap text from end of previous chunk
            overlap_start = max(0, len(prev_chunk.content) - self.overlap_chars)
            overlap_text = prev_chunk.content[overlap_start:]

            # Find a clean break point in overlap
            for sep in ['. ', '.\n', '\n', ' ']:
                pos = overlap_text.find(sep)
                if pos >= 0:
                    overlap_text = overlap_text[pos + len(sep):]
                    break

            if overlap_text.strip():
                # Prepend overlap to current chunk
                new_content = f"[...] {overlap_text}\n{curr_chunk.content}"
                curr_chunk = Chunk(
                    content=new_content,
                    start_idx=curr_chunk.start_idx - len(overlap_text),
                    end_idx=curr_chunk.end_idx,
                    metadata={**curr_chunk.metadata, 'has_overlap': True},
                    content_type=curr_chunk.content_type,
                    sentences=curr_chunk.sentences
                )

            overlapped.append(curr_chunk)

        return overlapped

    def _embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Compute embeddings for all chunks."""
        texts = [c.content for c in chunks]
        embeddings = self.embedding_provider.embed_batch(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        return chunks

    def _find_semantic_boundaries(self, sentences: List[str]) -> List[int]:
        """Find natural breakpoints using embedding similarity.

        This is the public API method that wraps the boundary detector.

        Args:
            sentences: List of sentences to analyze

        Returns:
            List of indices where semantic boundaries occur
        """
        if not self.embedding_provider.available:
            return []

        embeddings = self.embedding_provider.embed_batch(sentences)
        return self.boundary_detector.find_boundaries(sentences, embeddings)

    def get_stats(self, chunks: List[Chunk]) -> ChunkingStats:
        """Get statistics about chunking results.

        Args:
            chunks: List of chunks to analyze

        Returns:
            ChunkingStats with metrics
        """
        if not chunks:
            return ChunkingStats()

        sizes = [len(c.content) for c in chunks]

        return ChunkingStats(
            total_chunks=len(chunks),
            total_characters=sum(sizes),
            avg_chunk_size=sum(sizes) / len(sizes),
            min_chunk_size=min(sizes),
            max_chunk_size=max(sizes),
            content_type=chunks[0].content_type if chunks else ContentType.PLAIN_TEXT
        )


# =============================================================================
# MEMORY STORAGE INTEGRATION
# =============================================================================

class ChunkMemoryIntegration:
    """Integration layer for storing chunks in the memory system."""

    def __init__(self, chunker: Optional[SemanticChunker] = None):
        """Initialize integration.

        Args:
            chunker: SemanticChunker instance (creates default if None)
        """
        self.chunker = chunker or SemanticChunker()

    async def store_document(
        self,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        backend=None
    ) -> List[str]:
        """Chunk and store a document in memory.

        Args:
            document_id: Unique identifier for the document
            content: Document content to chunk
            metadata: Optional metadata to attach
            backend: Memory backend (SQLiteTierBackend)

        Returns:
            List of stored chunk IDs
        """
        # Chunk the document
        chunks = self.chunker.chunk(content, metadata={
            **(metadata or {}),
            'document_id': document_id
        })

        stored_ids = []

        if backend is None:
            # Try to import the default backend
            try:
                from core.memory.backends.sqlite import get_sqlite_backend
                backend = get_sqlite_backend()
            except ImportError:
                logger.warning("SQLite backend not available")
                return [c.chunk_id for c in chunks if c.chunk_id]

        # Store each chunk
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **chunk.metadata,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'content_type': chunk.content_type.value,
                'token_estimate': chunk.token_estimate,
            }

            try:
                from core.memory.backends.base import (
                    MemoryEntry, MemoryTier, MemoryPriority, MemoryNamespace
                )

                entry = MemoryEntry(
                    id=chunk.chunk_id or f"{document_id}_chunk_{i}",
                    content=chunk.content,
                    tier=MemoryTier.ARCHIVAL_MEMORY,
                    priority=MemoryPriority.NORMAL,
                    namespace=MemoryNamespace.CONTEXT,
                    embedding=chunk.embedding,
                    metadata=chunk_metadata,
                    tags=[document_id, f"chunk_{i}"]
                )

                await backend.put(entry.id, entry)
                stored_ids.append(entry.id)

            except Exception as e:
                logger.error(f"Failed to store chunk {i}: {e}")

        return stored_ids

    async def search_chunks(
        self,
        query: str,
        limit: int = 10,
        backend=None
    ) -> List[Chunk]:
        """Search for relevant chunks.

        Args:
            query: Search query
            limit: Maximum results
            backend: Memory backend

        Returns:
            List of matching Chunks
        """
        if backend is None:
            try:
                from core.memory.backends.sqlite import get_sqlite_backend
                backend = get_sqlite_backend()
            except ImportError:
                logger.warning("SQLite backend not available")
                return []

        results = await backend.search(query, limit)

        chunks = []
        for entry in results:
            chunk = Chunk(
                content=entry.content,
                start_idx=entry.metadata.get('start_idx', 0),
                end_idx=entry.metadata.get('end_idx', len(entry.content)),
                metadata=entry.metadata,
                embedding=entry.embedding,
                chunk_id=entry.id,
                content_type=ContentType(
                    entry.metadata.get('content_type', 'plain_text')
                )
            )
            chunks.append(chunk)

        return chunks


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    'SemanticChunker',
    'Chunk',
    'ChunkingStats',
    'ContentType',
    # Components
    'ContentTypeDetector',
    'SentenceSplitter',
    'EmbeddingProvider',
    'SemanticBoundaryDetector',
    # Integration
    'ChunkMemoryIntegration',
]


# =============================================================================
# CLI / DEMO
# =============================================================================

def main():
    """Demo the semantic chunker."""
    import sys

    # Sample text for demonstration
    sample_text = """
    # Introduction to Machine Learning

    Machine learning is a subset of artificial intelligence that enables systems
    to learn and improve from experience without being explicitly programmed.
    It focuses on developing computer programs that can access data and use it
    to learn for themselves.

    ## Types of Machine Learning

    There are three main types of machine learning:

    ### Supervised Learning

    In supervised learning, the algorithm learns from labeled training data.
    The model makes predictions based on the input features and compares them
    to the known correct outputs. Common algorithms include linear regression,
    decision trees, and neural networks.

    ### Unsupervised Learning

    Unsupervised learning works with unlabeled data. The algorithm tries to
    find hidden patterns or intrinsic structures in the input data. Clustering
    and dimensionality reduction are common unsupervised techniques.

    ### Reinforcement Learning

    Reinforcement learning is about taking actions to maximize reward in a
    particular situation. The agent learns through trial and error, receiving
    feedback from its actions. This approach is used in game playing,
    robotics, and autonomous systems.

    ## Applications

    Machine learning has numerous applications across industries:
    - Healthcare: Disease diagnosis and drug discovery
    - Finance: Fraud detection and algorithmic trading
    - Transportation: Self-driving cars and route optimization
    - Retail: Recommendation systems and demand forecasting
    """

    print("=" * 60)
    print("SEMANTIC CHUNKER DEMO")
    print("=" * 60)
    print()

    # Create chunker
    chunker = SemanticChunker(
        max_chunk_size=200,  # Smaller for demo
        min_chunk_size=50,
        overlap=20,
        embed_chunks=False  # Skip embeddings for demo
    )

    # Chunk the text
    chunks = chunker.chunk(sample_text, metadata={"source": "demo"})

    print(f"Input text: {len(sample_text)} characters")
    print(f"Generated {len(chunks)} chunks")
    print()

    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i + 1} ({len(chunk.content)} chars) ---")
        print(f"Content type: {chunk.content_type.value}")
        print(f"Sentences: {chunk.sentences}")
        print(f"Token estimate: {chunk.token_estimate}")
        print()
        # Show first 200 chars
        preview = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
        print(preview)
        print()

    # Show stats
    stats = chunker.get_stats(chunks)
    print("=" * 60)
    print("CHUNKING STATISTICS")
    print("=" * 60)
    print(f"Total chunks: {stats.total_chunks}")
    print(f"Total characters: {stats.total_characters}")
    print(f"Average chunk size: {stats.avg_chunk_size:.1f}")
    print(f"Min chunk size: {stats.min_chunk_size}")
    print(f"Max chunk size: {stats.max_chunk_size}")
    print(f"Content type: {stats.content_type.value}")


if __name__ == "__main__":
    main()
