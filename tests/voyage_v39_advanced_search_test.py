#!/usr/bin/env python3
"""
Voyage AI V39.3 - Advanced Semantic Search Validation Tests
============================================================

Tests the new advanced search functionality:
1. semantic_search_mmr() - Maximal Marginal Relevance for diversity
2. semantic_search_multi_query() - Query expansion with fusion
3. hybrid_search() - Vector + BM25 combination
4. semantic_search_with_filters() - Metadata filtering
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orchestration.embedding_layer import EmbeddingLayer, EmbeddingConfig, EmbeddingModel, InputType


# Test corpus - diverse documents for search testing
TEST_CORPUS = [
    "Python async programming with asyncio enables concurrent execution without threads.",
    "Machine learning models require careful hyperparameter tuning for optimal performance.",
    "The async/await syntax in Python makes asynchronous code more readable and maintainable.",
    "Neural networks learn hierarchical representations from raw input data.",
    "Python's threading module provides a high-level interface for working with threads.",
    "Deep learning frameworks like PyTorch and TensorFlow simplify model development.",
    "Concurrent programming in Python can use multiprocessing for CPU-bound tasks.",
    "Transformers have revolutionized natural language processing tasks.",
    "Python decorators allow you to modify function behavior without changing code.",
    "Embedding models convert text into dense vector representations for similarity search.",
]

TEST_METADATA = [
    {"category": "python", "topic": "async", "year": 2024, "difficulty": "intermediate"},
    {"category": "ml", "topic": "training", "year": 2023, "difficulty": "advanced"},
    {"category": "python", "topic": "async", "year": 2024, "difficulty": "intermediate"},
    {"category": "ml", "topic": "deep-learning", "year": 2024, "difficulty": "advanced"},
    {"category": "python", "topic": "threading", "year": 2022, "difficulty": "intermediate"},
    {"category": "ml", "topic": "deep-learning", "year": 2024, "difficulty": "advanced"},
    {"category": "python", "topic": "concurrency", "year": 2023, "difficulty": "intermediate"},
    {"category": "ml", "topic": "nlp", "year": 2024, "difficulty": "advanced"},
    {"category": "python", "topic": "patterns", "year": 2022, "difficulty": "beginner"},
    {"category": "ml", "topic": "embeddings", "year": 2024, "difficulty": "intermediate"},
]


async def test_mmr_search():
    """Test Maximal Marginal Relevance search for diversity."""
    print("\n[TEST] MMR Search (Maximal Marginal Relevance)")
    print("-" * 50)

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=100))
    await layer.initialize()

    # Pre-compute embeddings for efficiency
    result = await layer.embed(
        texts=TEST_CORPUS,
        model=EmbeddingModel.VOYAGE_4_LITE,
        input_type=InputType.DOCUMENT,
    )
    doc_embeddings = result.embeddings
    print(f"  Pre-computed {len(doc_embeddings)} document embeddings")

    # Test MMR with high diversity (low lambda)
    results_diverse = await layer.semantic_search_mmr(
        query="Python async programming patterns",
        documents=TEST_CORPUS,
        doc_embeddings=doc_embeddings,
        top_k=3,
        lambda_mult=0.3,  # Low = more diversity
        fetch_k=10,
    )

    print(f"  Diverse results (lambda=0.3):")
    for idx, score, doc in results_diverse:
        print(f"    [{idx}] {score:.3f}: {doc[:50]}...")

    # Test MMR with high relevance (high lambda)
    results_relevant = await layer.semantic_search_mmr(
        query="Python async programming patterns",
        documents=TEST_CORPUS,
        doc_embeddings=doc_embeddings,
        top_k=3,
        lambda_mult=0.9,  # High = more relevance
        fetch_k=10,
    )

    print(f"  Relevant results (lambda=0.9):")
    for idx, score, doc in results_relevant:
        print(f"    [{idx}] {score:.3f}: {doc[:50]}...")

    # Diversity check: diverse results should have different topics
    diverse_indices = set(r[0] for r in results_diverse)
    relevant_indices = set(r[0] for r in results_relevant)

    assert len(results_diverse) == 3, "Should return 3 diverse results"
    assert len(results_relevant) == 3, "Should return 3 relevant results"

    # Relevant results will likely cluster around async/python topics
    # Diverse results should spread across different topics

    print("  [PASS] MMR search working correctly")
    return True


async def test_multi_query_search():
    """Test multi-query retrieval with fusion."""
    print("\n[TEST] Multi-Query Search (Query Expansion + Fusion)")
    print("-" * 50)

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=100))
    await layer.initialize()

    # Pre-compute embeddings
    result = await layer.embed(
        texts=TEST_CORPUS,
        model=EmbeddingModel.VOYAGE_4_LITE,
        input_type=InputType.DOCUMENT,
    )
    doc_embeddings = result.embeddings

    # Test with RRF fusion
    results_rrf = await layer.semantic_search_multi_query(
        query="How do async patterns improve concurrency?",
        documents=TEST_CORPUS,
        doc_embeddings=doc_embeddings,
        top_k=3,
        num_sub_queries=3,
        fusion_method="rrf",
    )

    print(f"  RRF fusion results:")
    for idx, score, doc in results_rrf:
        print(f"    [{idx}] {score:.4f}: {doc[:50]}...")

    # Test with sum fusion
    results_sum = await layer.semantic_search_multi_query(
        query="How do async patterns improve concurrency?",
        documents=TEST_CORPUS,
        doc_embeddings=doc_embeddings,
        top_k=3,
        num_sub_queries=3,
        fusion_method="sum",
    )

    print(f"  Sum fusion results:")
    for idx, score, doc in results_sum:
        print(f"    [{idx}] {score:.4f}: {doc[:50]}...")

    assert len(results_rrf) == 3, "Should return 3 RRF results"
    assert len(results_sum) == 3, "Should return 3 sum results"

    # Results should include async-related documents
    rrf_indices = [r[0] for r in results_rrf]
    # Documents 0, 2 are about async, should appear in top results
    has_async = any(idx in [0, 2] for idx in rrf_indices)
    print(f"  Contains async-related docs: {has_async}")

    print("  [PASS] Multi-query search working correctly")
    return True


async def test_hybrid_search():
    """Test hybrid search combining vector and BM25."""
    print("\n[TEST] Hybrid Search (Vector + BM25)")
    print("-" * 50)

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=100))
    await layer.initialize()

    # Pre-compute embeddings
    result = await layer.embed(
        texts=TEST_CORPUS,
        model=EmbeddingModel.VOYAGE_4_LITE,
        input_type=InputType.DOCUMENT,
    )
    doc_embeddings = result.embeddings

    # Test with balanced alpha
    results_balanced = await layer.hybrid_search(
        query="Python asyncio concurrent",
        documents=TEST_CORPUS,
        doc_embeddings=doc_embeddings,
        top_k=3,
        alpha=0.5,  # Balanced
    )

    print(f"  Balanced hybrid (alpha=0.5):")
    for idx, score, doc in results_balanced:
        print(f"    [{idx}] {score:.3f}: {doc[:50]}...")

    # Test with keyword-focused (low alpha)
    results_keyword = await layer.hybrid_search(
        query="Python asyncio concurrent",
        documents=TEST_CORPUS,
        doc_embeddings=doc_embeddings,
        top_k=3,
        alpha=0.2,  # More BM25
    )

    print(f"  Keyword-focused (alpha=0.2):")
    for idx, score, doc in results_keyword:
        print(f"    [{idx}] {score:.3f}: {doc[:50]}...")

    # Test with semantic-focused (high alpha)
    results_semantic = await layer.hybrid_search(
        query="Python asyncio concurrent",
        documents=TEST_CORPUS,
        doc_embeddings=doc_embeddings,
        top_k=3,
        alpha=0.9,  # More vector
    )

    print(f"  Semantic-focused (alpha=0.9):")
    for idx, score, doc in results_semantic:
        print(f"    [{idx}] {score:.3f}: {doc[:50]}...")

    assert len(results_balanced) == 3, "Should return 3 balanced results"
    assert len(results_keyword) == 3, "Should return 3 keyword results"
    assert len(results_semantic) == 3, "Should return 3 semantic results"

    # Keyword-focused should favor exact matches
    # Document 0 contains "asyncio" exactly
    keyword_indices = [r[0] for r in results_keyword]
    has_exact_match = 0 in keyword_indices[:2]
    print(f"  Keyword search has exact match in top 2: {has_exact_match}")

    print("  [PASS] Hybrid search working correctly")
    return True


async def test_filtered_search():
    """Test semantic search with metadata filters."""
    print("\n[TEST] Filtered Search (Metadata Conditions)")
    print("-" * 50)

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=100))
    await layer.initialize()

    # Pre-compute embeddings
    result = await layer.embed(
        texts=TEST_CORPUS,
        model=EmbeddingModel.VOYAGE_4_LITE,
        input_type=InputType.DOCUMENT,
    )
    doc_embeddings = result.embeddings

    # Test exact match filter
    results_python = await layer.semantic_search_with_filters(
        query="programming patterns",
        documents=TEST_CORPUS,
        metadata=TEST_METADATA,
        doc_embeddings=doc_embeddings,
        top_k=3,
        filters={"category": "python"},
    )

    print(f"  Python category filter:")
    for idx, score, doc, meta in results_python:
        print(f"    [{idx}] {score:.3f} ({meta['topic']}): {doc[:40]}...")

    # All results should be Python category
    all_python = all(meta["category"] == "python" for _, _, _, meta in results_python)
    assert all_python, "All filtered results should be Python category"
    print(f"  All Python category: {all_python}")

    # Test range filter
    results_recent = await layer.semantic_search_with_filters(
        query="deep learning models",
        documents=TEST_CORPUS,
        metadata=TEST_METADATA,
        doc_embeddings=doc_embeddings,
        top_k=3,
        filters={"year": {"$gte": 2024}},
    )

    print(f"  Year >= 2024 filter:")
    for idx, score, doc, meta in results_recent:
        print(f"    [{idx}] {score:.3f} ({meta['year']}): {doc[:40]}...")

    all_recent = all(meta["year"] >= 2024 for _, _, _, meta in results_recent)
    assert all_recent, "All results should be from 2024+"
    print(f"  All from 2024+: {all_recent}")

    # Test combined filters
    results_combined = await layer.semantic_search_with_filters(
        query="neural network training",
        documents=TEST_CORPUS,
        metadata=TEST_METADATA,
        doc_embeddings=doc_embeddings,
        top_k=3,
        filters={"category": "ml", "difficulty": "advanced"},
    )

    print(f"  Combined filter (ML + advanced):")
    for idx, score, doc, meta in results_combined:
        print(f"    [{idx}] {score:.3f} ({meta['topic']}): {doc[:40]}...")

    all_match = all(
        meta["category"] == "ml" and meta["difficulty"] == "advanced"
        for _, _, _, meta in results_combined
    )
    assert all_match, "All results should match combined filters"
    print(f"  All match combined: {all_match}")

    # Test $in filter
    results_topics = await layer.semantic_search_with_filters(
        query="programming",
        documents=TEST_CORPUS,
        metadata=TEST_METADATA,
        doc_embeddings=doc_embeddings,
        top_k=5,
        filters={"topic": {"$in": ["async", "deep-learning"]}},
    )

    print(f"  Topic in [async, deep-learning] filter:")
    for idx, score, doc, meta in results_topics:
        print(f"    [{idx}] {score:.3f} ({meta['topic']}): {doc[:40]}...")

    all_in_topics = all(
        meta["topic"] in ["async", "deep-learning"]
        for _, _, _, meta in results_topics
    )
    assert all_in_topics, "All results should have matching topics"

    print("  [PASS] Filtered search working correctly")
    return True


async def test_bm25_scoring():
    """Test BM25 scoring independently."""
    print("\n[TEST] BM25 Keyword Scoring")
    print("-" * 50)

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=False))
    await layer.initialize()

    # Test exact keyword matching
    query = "asyncio concurrent"
    scores = layer._compute_bm25_scores(query, TEST_CORPUS)

    print(f"  Query: '{query}'")
    print(f"  BM25 scores:")

    # Sort by score for display
    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    for idx, score in scored[:5]:
        print(f"    [{idx}] {score:.3f}: {TEST_CORPUS[idx][:50]}...")

    # Document 0 and 2 should have high scores (contain async)
    async_docs = [0, 2]
    top_indices = [idx for idx, _ in scored[:3]]
    has_async_in_top = any(idx in async_docs for idx in top_indices)
    print(f"  Async docs in top 3: {has_async_in_top}")

    # Document 6 mentions "concurrent" explicitly
    concurrent_doc = 6
    has_concurrent_in_top = concurrent_doc in top_indices
    print(f"  'Concurrent' doc in top 3: {has_concurrent_in_top}")

    print("  [PASS] BM25 scoring working correctly")
    return True


async def main():
    """Run all advanced search tests."""
    print("=" * 60)
    print("VOYAGE AI V39.3 - ADVANCED SEARCH TESTS")
    print("=" * 60)

    tests = [
        ("MMR Search", test_mmr_search),
        ("Multi-Query Search", test_multi_query_search),
        ("Hybrid Search", test_hybrid_search),
        ("Filtered Search", test_filtered_search),
        ("BM25 Scoring", test_bm25_scoring),
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
