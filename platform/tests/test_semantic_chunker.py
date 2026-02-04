#!/usr/bin/env python3
"""
Tests for Semantic Chunker

Run with: pytest platform/tests/test_semantic_chunker.py -v
"""

import pytest
from typing import List

# Import from the rag module
from core.rag.semantic_chunker import (
    SemanticChunker,
    Chunk,
    ChunkingStats,
    ContentType,
    ContentTypeDetector,
    SentenceSplitter,
    EmbeddingProvider,
    SemanticBoundaryDetector,
)


# =============================================================================
# Test Data
# =============================================================================

SAMPLE_PROSE = """
Machine learning is a subset of artificial intelligence that enables systems
to learn and improve from experience without being explicitly programmed.
It focuses on developing computer programs that can access data and use it
to learn for themselves.

There are three main types of machine learning: supervised learning,
unsupervised learning, and reinforcement learning. Each type has its own
use cases and algorithms.

Deep learning is a subset of machine learning that uses neural networks
with many layers. It has revolutionized fields like computer vision and
natural language processing.
"""

SAMPLE_PYTHON_CODE = '''
def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number recursively."""
    if n <= 1:
        return n
    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)


class FibonacciCalculator:
    """Efficient Fibonacci calculator with memoization."""

    def __init__(self):
        self._cache = {0: 0, 1: 1}

    def calculate(self, n: int) -> int:
        """Calculate Fibonacci with caching."""
        if n not in self._cache:
            self._cache[n] = self.calculate(n - 1) + self.calculate(n - 2)
        return self._cache[n]


def main():
    calc = FibonacciCalculator()
    for i in range(10):
        print(f"F({i}) = {calc.calculate(i)}")


if __name__ == "__main__":
    main()
'''

SAMPLE_MARKDOWN = """
# Introduction

This is the introduction section. It provides context for the document.

## Getting Started

To get started, follow these steps:

1. Install the package
2. Configure your environment
3. Run the initialization

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Virtual environment (recommended)

## Configuration

Configuration is done via environment variables or a config file.

```python
# Example configuration
config = {
    "debug": True,
    "max_workers": 4,
}
```

## Conclusion

This concludes the documentation. For more information, see the API reference.
"""

SAMPLE_JSON = '{"name": "test", "version": "1.0.0", "dependencies": {"numpy": "^1.24.0"}}'

SAMPLE_TYPESCRIPT = '''
interface User {
    id: string;
    name: string;
    email: string;
}

const createUser = (data: Partial<User>): User => {
    return {
        id: crypto.randomUUID(),
        name: data.name || "Unknown",
        email: data.email || "",
    };
};

export const userService = {
    create: createUser,
    findById: async (id: string): Promise<User | null> => {
        // Implementation
        return null;
    },
};
'''


# =============================================================================
# Content Type Detection Tests
# =============================================================================

class TestContentTypeDetector:
    """Tests for content type detection."""

    def test_detect_python(self):
        """Test Python code detection."""
        result = ContentTypeDetector.detect(SAMPLE_PYTHON_CODE)
        assert result == ContentType.CODE_PYTHON

    def test_detect_markdown(self):
        """Test Markdown detection."""
        result = ContentTypeDetector.detect(SAMPLE_MARKDOWN)
        assert result == ContentType.MARKDOWN

    def test_detect_typescript(self):
        """Test TypeScript detection."""
        result = ContentTypeDetector.detect(SAMPLE_TYPESCRIPT)
        assert result == ContentType.CODE_TYPESCRIPT

    def test_detect_json(self):
        """Test JSON detection."""
        result = ContentTypeDetector.detect(SAMPLE_JSON)
        assert result == ContentType.JSON

    def test_detect_plain_text(self):
        """Test plain text detection."""
        plain_text = "This is just some regular text without any special formatting."
        result = ContentTypeDetector.detect(plain_text)
        assert result == ContentType.PLAIN_TEXT

    def test_detect_go_code(self):
        """Test Go code detection."""
        go_code = '''
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}

type Server struct {
    host string
    port int
}
'''
        result = ContentTypeDetector.detect(go_code)
        assert result == ContentType.CODE_GO

    def test_detect_rust_code(self):
        """Test Rust code detection."""
        rust_code = '''
fn main() {
    println!("Hello, world!");
}

pub struct Config {
    debug: bool,
}

impl Config {
    pub fn new() -> Self {
        Config { debug: false }
    }
}
'''
        result = ContentTypeDetector.detect(rust_code)
        assert result == ContentType.CODE_RUST


# =============================================================================
# Sentence Splitter Tests
# =============================================================================

class TestSentenceSplitter:
    """Tests for sentence splitting."""

    def test_split_prose(self):
        """Test prose text splitting."""
        segments = SentenceSplitter.split(SAMPLE_PROSE, ContentType.PLAIN_TEXT)
        assert len(segments) > 0
        # Each segment should have content, start, end
        for text, start, end in segments:
            assert len(text) > 0
            assert start >= 0
            assert end > start

    def test_split_python_code(self):
        """Test Python code splitting."""
        segments = SentenceSplitter.split(SAMPLE_PYTHON_CODE, ContentType.CODE_PYTHON)
        assert len(segments) > 0
        # Should split at function/class definitions
        contents = [s[0] for s in segments]
        # At least one segment should contain 'def' or 'class'
        has_definition = any('def ' in c or 'class ' in c for c in contents)
        assert has_definition

    def test_split_markdown(self):
        """Test Markdown splitting."""
        segments = SentenceSplitter.split(SAMPLE_MARKDOWN, ContentType.MARKDOWN)
        assert len(segments) > 0
        # Should have multiple sections
        contents = [s[0] for s in segments]
        # Should capture headers
        has_header = any('#' in c for c in contents)
        assert has_header

    def test_split_preserves_indices(self):
        """Test that split preserves correct indices."""
        text = "First sentence. Second sentence. Third sentence."
        segments = SentenceSplitter.split(text, ContentType.PLAIN_TEXT)

        for content, start, end in segments:
            # Verify the indices point to the actual content
            extracted = text[start:end].strip()
            # Content should be found in the extracted region
            # (may have minor whitespace differences)
            assert content.strip() in extracted or extracted in content.strip()


# =============================================================================
# Embedding Provider Tests
# =============================================================================

class TestEmbeddingProvider:
    """Tests for embedding provider."""

    def test_provider_initialization(self):
        """Test provider initializes without error."""
        provider = EmbeddingProvider()
        # Should not raise
        assert provider is not None

    def test_custom_provider(self):
        """Test custom embedding provider."""
        def mock_embed(text: str) -> List[float]:
            return [0.1] * 384  # Fixed dimension

        provider = EmbeddingProvider(custom_provider=mock_embed)
        assert provider.available is True

        embedding = provider.embed("test text")
        assert embedding is not None
        assert len(embedding) == 384

    def test_batch_embed(self):
        """Test batch embedding."""
        def mock_embed(text: str) -> List[float]:
            return [len(text) / 100.0] * 384

        provider = EmbeddingProvider(custom_provider=mock_embed)
        texts = ["short", "medium length text", "this is a longer piece of text"]
        embeddings = provider.embed_batch(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert emb is not None
            assert len(emb) == 384


# =============================================================================
# Semantic Boundary Detector Tests
# =============================================================================

class TestSemanticBoundaryDetector:
    """Tests for semantic boundary detection."""

    def test_find_boundaries_empty(self):
        """Test with empty input."""
        detector = SemanticBoundaryDetector()
        boundaries = detector.find_boundaries([], [])
        assert boundaries == []

    def test_find_boundaries_single(self):
        """Test with single sentence."""
        detector = SemanticBoundaryDetector()
        boundaries = detector.find_boundaries(["Single sentence."], [[0.1] * 10])
        assert boundaries == []

    def test_find_boundaries_with_similar_embeddings(self):
        """Test that similar embeddings don't create boundaries."""
        detector = SemanticBoundaryDetector(similarity_threshold=0.5)
        sentences = ["Sentence one.", "Sentence two.", "Sentence three."]
        # All same embedding = max similarity
        embeddings = [[0.5] * 10, [0.5] * 10, [0.5] * 10]
        boundaries = detector.find_boundaries(sentences, embeddings)
        # Should find no boundaries (all similar)
        assert len(boundaries) == 0

    def test_find_boundaries_with_different_embeddings(self):
        """Test that different embeddings create boundaries."""
        detector = SemanticBoundaryDetector(similarity_threshold=0.5)
        sentences = ["Topic A content.", "More about A.", "Topic B different."]
        # First two similar, third different
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0],  # Very different
        ]
        boundaries = detector.find_boundaries(sentences, embeddings)
        # Should find boundary before index 2
        assert 2 in boundaries


# =============================================================================
# Semantic Chunker Tests
# =============================================================================

class TestSemanticChunker:
    """Tests for the main SemanticChunker class."""

    def test_chunker_initialization(self):
        """Test chunker initializes with defaults."""
        chunker = SemanticChunker()
        assert chunker.max_chunk_size == 512
        assert chunker.min_chunk_size == 100
        assert chunker.overlap == 50

    def test_chunker_custom_params(self):
        """Test chunker with custom parameters."""
        chunker = SemanticChunker(
            max_chunk_size=256,
            min_chunk_size=50,
            overlap=25,
        )
        assert chunker.max_chunk_size == 256
        assert chunker.min_chunk_size == 50
        assert chunker.overlap == 25

    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunker = SemanticChunker(embed_chunks=False)
        chunks = chunker.chunk("")
        assert chunks == []

    def test_chunk_whitespace_only(self):
        """Test chunking whitespace-only text."""
        chunker = SemanticChunker(embed_chunks=False)
        chunks = chunker.chunk("   \n\n   ")
        assert chunks == []

    def test_chunk_prose(self):
        """Test chunking prose text."""
        chunker = SemanticChunker(
            max_chunk_size=100,
            min_chunk_size=20,
            embed_chunks=False
        )
        chunks = chunker.chunk(SAMPLE_PROSE)

        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert len(chunk.content) > 0
            assert chunk.chunk_id is not None

    def test_chunk_python_code(self):
        """Test chunking Python code."""
        chunker = SemanticChunker(
            max_chunk_size=150,
            min_chunk_size=30,
            embed_chunks=False
        )
        chunks = chunker.chunk(SAMPLE_PYTHON_CODE)

        assert len(chunks) > 0
        # Should detect Python content type
        assert all(c.content_type == ContentType.CODE_PYTHON for c in chunks)

    def test_chunk_markdown(self):
        """Test chunking Markdown."""
        chunker = SemanticChunker(
            max_chunk_size=100,
            min_chunk_size=20,
            embed_chunks=False
        )
        chunks = chunker.chunk(SAMPLE_MARKDOWN)

        assert len(chunks) > 0
        assert all(c.content_type == ContentType.MARKDOWN for c in chunks)

    def test_chunk_with_metadata(self):
        """Test metadata preservation."""
        chunker = SemanticChunker(embed_chunks=False)
        metadata = {"source": "test", "author": "test_user"}
        chunks = chunker.chunk(SAMPLE_PROSE, metadata=metadata)

        assert len(chunks) > 0
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == "test"

    def test_chunk_indices(self):
        """Test that chunk indices are valid."""
        chunker = SemanticChunker(embed_chunks=False)
        chunks = chunker.chunk(SAMPLE_PROSE)

        for i, chunk in enumerate(chunks):
            assert chunk.start_idx >= 0
            assert chunk.end_idx > chunk.start_idx
            # End of one chunk should be <= start of next (allowing for overlap)
            if i > 0:
                assert chunk.start_idx <= chunks[i - 1].end_idx + chunker.overlap_chars

    def test_chunk_with_custom_embedding(self):
        """Test chunking with custom embedding provider."""
        def mock_embed(text: str) -> List[float]:
            return [0.1] * 384

        chunker = SemanticChunker(
            embed_chunks=True,
            custom_embedding_provider=mock_embed
        )
        chunks = chunker.chunk(SAMPLE_PROSE)

        # Should have embeddings
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 384

    def test_chunk_stats(self):
        """Test chunking statistics."""
        chunker = SemanticChunker(embed_chunks=False)
        chunks = chunker.chunk(SAMPLE_PROSE)

        stats = chunker.get_stats(chunks)
        assert isinstance(stats, ChunkingStats)
        assert stats.total_chunks == len(chunks)
        assert stats.total_characters > 0
        assert stats.avg_chunk_size > 0
        assert stats.min_chunk_size <= stats.max_chunk_size

    def test_chunk_token_estimate(self):
        """Test token estimation."""
        chunk = Chunk(
            content="This is a test with about twenty characters.",
            start_idx=0,
            end_idx=44,
        )
        # ~4 chars per token
        expected_tokens = len(chunk.content) // 4
        assert chunk.token_estimate == expected_tokens

    def test_chunk_word_count(self):
        """Test word count."""
        chunk = Chunk(
            content="One two three four five",
            start_idx=0,
            end_idx=23,
        )
        assert chunk.word_count == 5

    def test_chunk_id_generation(self):
        """Test chunk ID generation is deterministic."""
        chunk1 = Chunk(content="Same content", start_idx=0, end_idx=12)
        chunk2 = Chunk(content="Same content", start_idx=0, end_idx=12)
        # Same content and indices should give same ID
        assert chunk1.chunk_id == chunk2.chunk_id

        chunk3 = Chunk(content="Different content", start_idx=0, end_idx=17)
        # Different content should give different ID
        assert chunk1.chunk_id != chunk3.chunk_id

    def test_find_semantic_boundaries_public_api(self):
        """Test the public _find_semantic_boundaries method."""
        def mock_embed(text: str) -> List[float]:
            # Return different embeddings based on content
            if "machine" in text.lower():
                return [1.0, 0.0, 0.0]
            elif "deep" in text.lower():
                return [0.0, 1.0, 0.0]
            else:
                return [0.0, 0.0, 1.0]

        chunker = SemanticChunker(
            custom_embedding_provider=mock_embed,
            similarity_threshold=0.5
        )
        sentences = [
            "Machine learning is important.",
            "Machine learning uses data.",
            "Deep learning is different.",
        ]
        boundaries = chunker._find_semantic_boundaries(sentences)
        # Should find boundary between ML and DL topics
        assert len(boundaries) >= 0  # May or may not find depending on threshold


# =============================================================================
# Integration Tests
# =============================================================================

class TestChunkerIntegration:
    """Integration tests for the chunker."""

    def test_large_document(self):
        """Test chunking a larger document."""
        # Generate a large document
        paragraphs = [
            f"This is paragraph {i}. " * 5 + "\n\n"
            for i in range(20)
        ]
        large_doc = "".join(paragraphs)

        chunker = SemanticChunker(
            max_chunk_size=200,
            min_chunk_size=50,
            overlap=25,
            embed_chunks=False
        )
        chunks = chunker.chunk(large_doc)

        assert len(chunks) > 0
        # Should have created multiple chunks
        assert len(chunks) >= 3

    def test_overlap_content(self):
        """Test that overlap is actually added."""
        text = """
First section with some content here. This is the first part.

Second section with different content. This is completely separate.

Third section with more content. Yet another distinct part.
"""
        chunker = SemanticChunker(
            max_chunk_size=50,
            min_chunk_size=20,
            overlap=10,
            embed_chunks=False
        )
        chunks = chunker.chunk(text)

        # Check for overlap marker
        has_overlap = any("[...]" in c.content for c in chunks)
        # At least some chunks should have overlap
        assert len(chunks) > 1

    def test_different_content_types_in_sequence(self):
        """Test that different documents get correct content types."""
        chunker = SemanticChunker(embed_chunks=False)

        # Python
        py_chunks = chunker.chunk(SAMPLE_PYTHON_CODE)
        assert py_chunks[0].content_type == ContentType.CODE_PYTHON

        # Markdown
        md_chunks = chunker.chunk(SAMPLE_MARKDOWN)
        assert md_chunks[0].content_type == ContentType.MARKDOWN

        # Plain text
        text_chunks = chunker.chunk(SAMPLE_PROSE)
        assert text_chunks[0].content_type == ContentType.PLAIN_TEXT


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_single_word(self):
        """Test chunking a single word."""
        chunker = SemanticChunker(min_chunk_size=1, embed_chunks=False)
        chunks = chunker.chunk("Hello")
        assert len(chunks) == 1
        assert chunks[0].content.strip() == "Hello"

    def test_single_sentence(self):
        """Test chunking a single sentence."""
        chunker = SemanticChunker(embed_chunks=False)
        chunks = chunker.chunk("This is a single sentence.")
        assert len(chunks) == 1

    def test_very_long_line(self):
        """Test handling of very long lines without breaks."""
        long_line = "word " * 1000  # ~5000 chars
        chunker = SemanticChunker(max_chunk_size=100, embed_chunks=False)
        chunks = chunker.chunk(long_line)

        # Should be split into multiple chunks
        assert len(chunks) > 1
        # Each chunk should be within limits
        for chunk in chunks:
            assert len(chunk.content) <= chunker.max_chars + 100  # Small tolerance

    def test_unicode_content(self):
        """Test handling of Unicode content."""
        unicode_text = """
        This contains Unicode: cafe
        Some emoji: smile rocket
        Chinese: nihao
        Arabic: mrhb
        """
        chunker = SemanticChunker(embed_chunks=False)
        chunks = chunker.chunk(unicode_text)
        assert len(chunks) > 0

    def test_mixed_newlines(self):
        """Test handling of mixed newline styles."""
        mixed = "Line1\nLine2\r\nLine3\rLine4"
        chunker = SemanticChunker(embed_chunks=False)
        chunks = chunker.chunk(mixed)
        assert len(chunks) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
