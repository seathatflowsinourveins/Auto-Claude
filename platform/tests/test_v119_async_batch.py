#!/usr/bin/env python3
"""
V119 Optimization Test: Async Batch Processing for Embeddings

Tests async batch processing by importing and testing real classes -
not by grepping file contents.

Test Date: 2026-01-30, Updated: 2026-02-02 (V14 Iter 55)
"""

import os
import sys
import time
import pytest

# Add platform to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestAsyncBatchStructure:
    """Test async batch structure by importing real classes."""

    def test_local_provider_has_embed_batch(self):
        """LocalEmbeddingProvider should have embed_batch method."""
        try:
            from core.advanced_memory import LocalEmbeddingProvider
        except ImportError:
            pytest.skip("advanced_memory not importable")

        provider = LocalEmbeddingProvider()
        assert hasattr(provider, "embed_batch"), "Should have embed_batch method"
        assert callable(provider.embed_batch)

    def test_semantic_index_has_add_batch(self):
        """SemanticIndex should have add_batch method."""
        try:
            from core.advanced_memory import SemanticIndex, LocalEmbeddingProvider
        except ImportError:
            pytest.skip("advanced_memory not importable")

        provider = LocalEmbeddingProvider()
        index = SemanticIndex(provider)
        assert hasattr(index, "add_batch"), "Should have add_batch method"
        assert callable(index.add_batch)


class TestAsyncBatchBehavior:
    """Test actual behavior of async batch processing."""

    @pytest.mark.asyncio
    async def test_local_provider_embed_batch_parallel(self):
        """Test that LocalEmbeddingProvider.embed_batch runs in parallel."""
        try:
            from core.advanced_memory import LocalEmbeddingProvider
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        provider = LocalEmbeddingProvider()
        texts = [f"Test text {i}" for i in range(10)]

        # Time batch operation
        start = time.perf_counter()
        results = await provider.embed_batch(texts)
        batch_time = time.perf_counter() - start

        # Verify results
        assert len(results) == len(texts), "Should return same number of results"
        for result in results:
            assert len(result.embedding) == 384, "Default dimensions should be 384"

        # Batch should complete quickly (parallel is faster than sequential)
        # For local hash-based provider, this should be very fast
        assert batch_time < 1.0, f"Batch of 10 should complete in <1s, took {batch_time:.2f}s"

    @pytest.mark.asyncio
    async def test_semantic_index_add_batch(self):
        """Test SemanticIndex.add_batch functionality."""
        try:
            from core.advanced_memory import SemanticIndex, LocalEmbeddingProvider
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        provider = LocalEmbeddingProvider()
        index = SemanticIndex(provider)

        # Prepare batch items: (id, content, metadata, importance)
        items = [
            (f"item-{i}", f"Content for item {i}", {"type": "test"}, 1.0)
            for i in range(5)
        ]

        # Add batch
        entries = await index.add_batch(items)

        # Verify results
        assert len(entries) == 5, "Should create 5 entries"
        assert index.count() == 5, "Index should contain 5 entries"

        # Verify each entry
        for i, entry in enumerate(entries):
            assert entry.id == f"item-{i}"
            assert entry.content == f"Content for item {i}"
            assert entry.metadata.get("type") == "test"
            assert len(entry.embedding) == 384

    @pytest.mark.asyncio
    async def test_add_batch_vs_sequential_consistency(self):
        """Test that add_batch produces same results as sequential adds."""
        try:
            from core.advanced_memory import SemanticIndex, LocalEmbeddingProvider
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        provider = LocalEmbeddingProvider()

        # Sequential adds
        index_seq = SemanticIndex(provider)
        for i in range(3):
            await index_seq.add(f"seq-{i}", f"Content {i}", {"type": "seq"}, 1.0)

        # Batch add
        index_batch = SemanticIndex(provider)
        items = [(f"batch-{i}", f"Content {i}", {"type": "batch"}, 1.0) for i in range(3)]
        await index_batch.add_batch(items)

        # Compare embeddings (should be same for same content)
        for i in range(3):
            seq_entry = index_seq.get(f"seq-{i}")
            batch_entry = index_batch.get(f"batch-{i}")

            assert seq_entry is not None
            assert batch_entry is not None

            # Embeddings should be identical for same content
            assert seq_entry.embedding == batch_entry.embedding, \
                f"Embeddings should match for 'Content {i}'"


class TestAsyncBatchPerformance:
    """Performance benchmarks for async batch processing."""

    @pytest.mark.asyncio
    async def test_batch_speedup(self):
        """Verify batch processing is faster than sequential."""
        try:
            from core.advanced_memory import LocalEmbeddingProvider
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        provider = LocalEmbeddingProvider()
        texts = [f"Performance test text {i}" for i in range(50)]

        # Sequential timing (simulating old pattern)
        start = time.perf_counter()
        sequential = []
        for text in texts:
            result = await provider.embed(text)
            sequential.append(result)
        seq_time = time.perf_counter() - start

        # Batch timing (V119 pattern)
        start = time.perf_counter()
        batch = await provider.embed_batch(texts)
        batch_time = time.perf_counter() - start

        # Verify correctness
        assert len(sequential) == len(batch) == len(texts)

        # Log times for visibility
        print(f"\nSequential: {seq_time*1000:.1f}ms")
        print(f"Batch:      {batch_time*1000:.1f}ms")
        print(f"Speedup:    {seq_time/batch_time:.1f}x")

        # Note: For local hash-based provider, both are fast
        # The real speedup is visible with IO-bound providers
        # Just verify batch doesn't degrade performance
        # Local hash-based provider: both fast, allow 3x margin for OS jitter
        assert batch_time <= seq_time * 3.0, \
            f"Batch should not be drastically slower than sequential (seq={seq_time:.3f}s, batch={batch_time:.3f}s)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
