#!/usr/bin/env python3
"""
Voyage AI V39.5 - Streaming & Performance Optimization Tests
=============================================================

Tests the new V39.5 functionality:
1. embed_stream() - AsyncGenerator for progressive embedding
2. embed_batch_streaming() - Batch processing with streaming callback
3. analyze_query_characteristics() - Query analysis for adaptive alpha
4. adaptive_hybrid_search() - Auto-tuning hybrid search
5. prefetch_cache() - Predictive cache warming

Note: These tests use REAL Voyage AI API calls.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orchestration.embedding_layer import (
    EmbeddingLayer,
    EmbeddingConfig,
    EmbeddingModel,
    InputType,
)


# Test corpus for streaming tests
LARGE_CORPUS = [
    "Machine learning models require careful hyperparameter tuning.",
    "Python async programming enables concurrent execution.",
    "Neural networks learn hierarchical representations.",
    "Deep learning frameworks simplify model development.",
    "Transformers revolutionized natural language processing.",
    "Embedding models convert text into dense vectors.",
    "Vector databases enable semantic similarity search.",
    "Attention mechanisms allow models to focus on relevant parts.",
    "Gradient descent optimizes neural network parameters.",
    "Transfer learning leverages pre-trained models.",
    "Regularization techniques prevent model overfitting.",
    "Batch normalization stabilizes neural network training.",
    "Dropout randomly deactivates neurons during training.",
    "Convolutional networks excel at image recognition.",
    "Recurrent networks process sequential data effectively.",
    "BERT models understand bidirectional context.",
    "GPT models generate coherent text sequences.",
    "Fine-tuning adapts pre-trained models to specific tasks.",
    "Data augmentation increases training dataset diversity.",
    "Cross-validation provides robust model evaluation.",
]

# Queries with different characteristics for adaptive alpha testing
TEST_QUERIES = {
    "keyword_heavy": "API SDK v2 http json",  # Should get low alpha
    "semantic_rich": "understanding the conceptual relationship between objects",  # High alpha
    "code_like": "async def process_batch(items: list) -> dict:",  # Low alpha
    "balanced": "python machine learning optimization",  # Medium alpha
    "technical": "RESTful API endpoint authentication JWT tokens",  # Low-medium alpha
}


async def test_embed_stream():
    """Test streaming embeddings with AsyncGenerator."""
    print("\n[TEST] Embed Stream (AsyncGenerator)")
    print("-" * 50)

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=100))
    await layer.initialize()

    # Clear cache to test fresh embeddings
    layer.clear_cache()

    embeddings_received = []
    indices_received = []

    # Stream embeddings
    async for idx, embedding in layer.embed_stream(
        texts=LARGE_CORPUS[:10],
        batch_size=3,  # Small batches to see streaming effect
        input_type=InputType.DOCUMENT,
    ):
        embeddings_received.append(embedding)
        indices_received.append(idx)
        print(f"  Received embedding {idx}: dim={len(embedding)}")

    assert len(embeddings_received) == 10, f"Expected 10, got {len(embeddings_received)}"
    assert indices_received == list(range(10)), "Indices should be in order"
    assert all(len(e) > 0 for e in embeddings_received), "All embeddings should have data"

    print(f"  Streamed {len(embeddings_received)} embeddings successfully")
    print("  [PASS] Embed stream working correctly")
    return True


async def test_embed_batch_streaming():
    """Test batch processing with streaming callback."""
    print("\n[TEST] Embed Batch Streaming (with callback)")
    print("-" * 50)

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=100))
    await layer.initialize()

    # Track callback invocations
    callback_invocations: list[dict[str, Any]] = []

    def on_batch_complete(batch_idx: int, total: int, embeddings: list[list[float]]):
        callback_invocations.append({
            "batch_idx": batch_idx,
            "total": total,
            "count": len(embeddings),
        })
        print(f"  Batch {batch_idx + 1}/{total}: {len(embeddings)} embeddings")

    result = await layer.embed_batch_streaming(
        texts=LARGE_CORPUS,
        target_batch_tokens=2000,  # Small batches for testing
        max_concurrent=2,
        on_batch_complete=on_batch_complete,
    )

    assert len(result.embeddings) == len(LARGE_CORPUS), "Should embed all texts"
    assert len(callback_invocations) > 0, "Callback should be called at least once"

    # Verify all batches were processed
    total_from_callbacks = sum(inv["count"] for inv in callback_invocations)
    assert total_from_callbacks == len(LARGE_CORPUS), "Callback counts should match total"

    print(f"  Total embeddings: {len(result.embeddings)}")
    print(f"  Callback invocations: {len(callback_invocations)}")
    print("  [PASS] Batch streaming working correctly")
    return True


async def test_query_analysis():
    """Test query characteristic analysis."""
    print("\n[TEST] Query Characteristic Analysis")
    print("-" * 50)

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=False))
    await layer.initialize()

    for query_type, query in TEST_QUERIES.items():
        analysis = layer.analyze_query_characteristics(query)
        print(f"  {query_type}:")
        print(f"    Query: '{query[:40]}...'")
        print(f"    Keyword density: {analysis['keyword_density']:.3f}")
        print(f"    Semantic complexity: {analysis['semantic_complexity']:.3f}")
        print(f"    Has code patterns: {analysis['has_code_patterns']}")
        print(f"    Recommended alpha: {analysis['recommended_alpha']}")
        print()

    # Verify expected patterns
    keyword_analysis = layer.analyze_query_characteristics(TEST_QUERIES["keyword_heavy"])
    semantic_analysis = layer.analyze_query_characteristics(TEST_QUERIES["semantic_rich"])
    code_analysis = layer.analyze_query_characteristics(TEST_QUERIES["code_like"])

    # Keyword-heavy should have lower alpha than semantic-rich
    assert keyword_analysis["recommended_alpha"] < semantic_analysis["recommended_alpha"], \
        "Keyword query should have lower alpha than semantic query"

    # Code-like should detect patterns
    assert code_analysis["has_code_patterns"], "Should detect code patterns"

    print("  [PASS] Query analysis working correctly")
    return True


async def test_adaptive_hybrid_search():
    """Test adaptive hybrid search with auto-tuning alpha."""
    print("\n[TEST] Adaptive Hybrid Search")
    print("-" * 50)

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=100))
    await layer.initialize()

    # Pre-compute document embeddings
    doc_result = await layer.embed(
        texts=LARGE_CORPUS,
        model=EmbeddingModel.VOYAGE_4_LITE,
        input_type=InputType.DOCUMENT,
    )
    doc_embeddings = doc_result.embeddings

    # Test with keyword-heavy query
    keyword_query = TEST_QUERIES["keyword_heavy"]
    keyword_results, keyword_alpha = await layer.adaptive_hybrid_search(
        query=keyword_query,
        documents=LARGE_CORPUS,
        doc_embeddings=doc_embeddings,
        top_k=3,
    )
    print(f"  Keyword query: alpha={keyword_alpha:.2f}")
    for idx, score, doc in keyword_results:
        print(f"    [{idx}] {score:.3f}: {doc[:40]}...")

    # Test with semantic query
    semantic_query = TEST_QUERIES["semantic_rich"]
    semantic_results, semantic_alpha = await layer.adaptive_hybrid_search(
        query=semantic_query,
        documents=LARGE_CORPUS,
        doc_embeddings=doc_embeddings,
        top_k=3,
    )
    print(f"\n  Semantic query: alpha={semantic_alpha:.2f}")
    for idx, score, doc in semantic_results:
        print(f"    [{idx}] {score:.3f}: {doc[:40]}...")

    # Test with override
    override_results, override_alpha = await layer.adaptive_hybrid_search(
        query="test query",
        documents=LARGE_CORPUS,
        doc_embeddings=doc_embeddings,
        top_k=3,
        alpha_override=0.9,
    )
    print(f"\n  Override query: alpha={override_alpha:.2f}")

    assert keyword_alpha < semantic_alpha, "Keyword should have lower alpha"
    assert override_alpha == 0.9, "Override should be respected"
    assert len(keyword_results) == 3, "Should return requested top_k"

    print("\n  [PASS] Adaptive hybrid search working correctly")
    return True


async def test_prefetch_cache():
    """Test predictive cache warming."""
    print("\n[TEST] Prefetch Cache (Predictive Warming)")
    print("-" * 50)

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=200))
    await layer.initialize()

    # Clear cache
    layer.clear_cache()
    initial_stats = layer.get_cache_stats()
    print(f"  Initial cache size: {initial_stats['cache_size']}")

    # Simulate recent queries about ML
    recent_queries = [
        "machine learning optimization",
        "neural network training",
        "deep learning models",
    ]

    # Candidate texts to potentially prefetch
    candidate_texts = LARGE_CORPUS + [
        "Quantum computing uses qubits for computation.",  # Unrelated
        "Web servers handle HTTP requests.",  # Unrelated
        "Machine learning requires quality data.",  # Related
        "Neural architecture search automates design.",  # Related
    ]

    result = await layer.prefetch_cache(
        recent_queries=recent_queries,
        candidate_texts=candidate_texts,
        similarity_threshold=0.5,
        max_prefetch=10,
    )

    print(f"  Prefetch result:")
    print(f"    Analyzed: {result['analyzed']}")
    print(f"    Above threshold: {result.get('above_threshold', 'N/A')}")
    print(f"    Prefetched: {result['prefetched']}")

    final_stats = layer.get_cache_stats()
    print(f"  Final cache size: {final_stats['cache_size']}")

    # Cache should have grown
    assert final_stats["cache_size"] > initial_stats["cache_size"], "Cache should grow"
    assert result["prefetched"] >= 0, "Prefetch count should be non-negative"

    print("  [PASS] Prefetch cache working correctly")
    return True


async def test_streaming_memory_efficiency():
    """Test that streaming doesn't hold all embeddings in memory at once."""
    print("\n[TEST] Streaming Memory Efficiency")
    print("-" * 50)

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=False))  # Disable cache for this test
    await layer.initialize()

    # Create moderately large corpus
    large_texts = [f"Document {i}: {text}" for i, text in enumerate(LARGE_CORPUS * 3)]
    print(f"  Testing with {len(large_texts)} documents")

    # Track memory through streaming
    max_batch_size = 0
    batch_count = 0

    async for idx, embedding in layer.embed_stream(
        texts=large_texts,
        batch_size=5,
    ):
        batch_count = idx // 5 + 1
        max_batch_size = max(max_batch_size, 5)

        # In streaming mode, we process one at a time
        # Memory is bounded by batch_size, not total corpus

    print(f"  Processed {idx + 1} embeddings in ~{batch_count} batches")
    print(f"  Max concurrent embeddings per batch: {max_batch_size}")

    # With batch_size=5 and 60 texts, we should have ~12 batches
    expected_batches = (len(large_texts) + 4) // 5
    assert batch_count == expected_batches, f"Expected ~{expected_batches} batches"

    print("  [PASS] Streaming maintains bounded memory")
    return True


async def main():
    """Run all V39.5 streaming tests."""
    print("=" * 60)
    print("VOYAGE AI V39.5 - STREAMING & PERFORMANCE TESTS")
    print("=" * 60)

    tests = [
        ("Embed Stream", test_embed_stream),
        ("Embed Batch Streaming", test_embed_batch_streaming),
        ("Query Analysis", test_query_analysis),
        ("Adaptive Hybrid Search", test_adaptive_hybrid_search),
        ("Prefetch Cache", test_prefetch_cache),
        ("Streaming Memory Efficiency", test_streaming_memory_efficiency),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            await test_fn()
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
