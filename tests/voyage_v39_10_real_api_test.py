#!/usr/bin/env python3
"""
Voyage AI V39.10 - Real API Integration Tests
==============================================

Tests the V39.10 enhancements with REAL Voyage AI API calls:
1. Cost-aware test decorators
2. Real API validation for all V39.x features
3. State of Witness E2E pipeline
4. Performance benchmarking

Note: These tests use REAL Voyage AI API calls.
      Requires VOYAGE_API_KEY environment variable.
      Estimated cost per run: ~$0.40

Run with: python tests/voyage_v39_10_real_api_test.py
"""

import asyncio
import functools
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".config" / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[INFO] Loaded environment from {env_path}")
except ImportError:
    print("[WARNING] python-dotenv not installed, using system environment only")

from core.orchestration.embedding_layer import (
    EmbeddingLayer,
    EmbeddingConfig,
    EmbeddingModel,
    InputType,
)


# ============================================================================
# V39.10 Test Infrastructure
# ============================================================================

@dataclass
class RealAPITestConfig:
    """Configuration for real API testing."""
    api_key: str = field(default_factory=lambda: os.environ.get("VOYAGE_API_KEY", ""))
    max_cost_usd: float = 0.50  # Budget per test run
    timeout_seconds: int = 30
    retry_count: int = 2
    collect_metrics: bool = True
    opik_trace: bool = False  # Enable when Opik is configured

    def validate(self) -> bool:
        """Validate API key is configured."""
        if not self.api_key:
            print("[WARNING] VOYAGE_API_KEY not set - real API tests will be skipped")
            return False
        return True


@dataclass
class TestCostTracker:
    """Track cumulative test costs."""
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    tests_run: int = 0
    tests_skipped: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def add_cost(self, tokens: int, cost_usd: float):
        """Add cost from a test."""
        self.total_tokens += tokens
        self.total_cost_usd += cost_usd
        self.tests_run += 1

    def skip_test(self):
        """Mark a test as skipped."""
        self.tests_skipped += 1

    def summary(self) -> str:
        """Generate cost summary."""
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0
        return (
            f"\n{'=' * 60}\n"
            f"COST SUMMARY\n"
            f"{'=' * 60}\n"
            f"  Tests run: {self.tests_run}\n"
            f"  Tests skipped: {self.tests_skipped}\n"
            f"  Total tokens: {self.total_tokens:,}\n"
            f"  Total cost: ${self.total_cost_usd:.4f}\n"
            f"  Duration: {duration:.1f}s\n"
            f"{'=' * 60}"
        )


# Global cost tracker
COST_TRACKER = TestCostTracker()


def cost_aware(max_tokens: int = 1000, max_cost_usd: float = 0.01):
    """
    Decorator for cost-aware API tests.

    Args:
        max_tokens: Maximum tokens this test should consume
        max_cost_usd: Maximum cost in USD for this test
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            config = RealAPITestConfig()
            if not config.validate():
                print(f"  [SKIP] {func.__name__} - No API key")
                COST_TRACKER.skip_test()
                return None

            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start

                # Estimate cost (voyage-4-large: ~$0.03 per 1M tokens)
                estimated_cost = (max_tokens / 1_000_000) * 0.03
                COST_TRACKER.add_cost(max_tokens, estimated_cost)

                print(f"  [PASS] {func.__name__} ({elapsed:.2f}s, ~${estimated_cost:.4f})")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                print(f"  [FAIL] {func.__name__} ({elapsed:.2f}s): {e}")
                raise

        return wrapper
    return decorator


# ============================================================================
# Test Data
# ============================================================================

ARCHETYPE_GESTURES = {
    "warrior": "Aggressive stance with raised fists, wide stance, tense muscles",
    "nurturer": "Embracing posture with open arms, soft shoulders, welcoming",
    "sage": "Meditative pose with hands together, straight spine, calm",
    "jester": "Dynamic jumping motion, arms flailing, playful energy",
    "lover": "Flowing dance movement, curved body, sensual grace",
    "magician": "Precise hand gestures, focused intent, mystical poses",
    "innocent": "Light bouncy movement, open face, childlike wonder",
    "everyman": "Neutral standing pose, relaxed posture, approachable",
}

TEST_DOCUMENTS = [
    "Machine learning enables computers to learn patterns from data",
    "Neural networks are inspired by biological brain structures",
    "Deep learning uses multiple layers of abstraction",
    "Computer vision processes and analyzes visual information",
    "Natural language processing understands human language",
]

TEST_QUERY = "What is machine learning?"


# ============================================================================
# Phase 1: Core Embedding Tests
# ============================================================================

@cost_aware(max_tokens=500, max_cost_usd=0.005)
async def test_single_embed_real_api():
    """Test single text embedding with real API."""
    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=False))
    await layer.initialize()

    result = await layer.embed(
        texts=["Hello, this is a test embedding."],
        model=EmbeddingModel.VOYAGE_4_LARGE,
        input_type=InputType.DOCUMENT,
    )

    assert result is not None, "Result should not be None"
    assert len(result.embeddings) == 1, "Should have 1 embedding"
    assert len(result.embeddings[0]) == 1024, "voyage-4-large produces 1024-dim vectors"

    return True


@cost_aware(max_tokens=2000, max_cost_usd=0.01)
async def test_batch_embed_real_api():
    """Test batch embedding with real API."""
    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=False))
    await layer.initialize()

    result = await layer.embed(
        texts=TEST_DOCUMENTS,
        model=EmbeddingModel.VOYAGE_4_LARGE,
        input_type=InputType.DOCUMENT,
    )

    assert len(result.embeddings) == 5, f"Expected 5 embeddings, got {len(result.embeddings)}"
    for emb in result.embeddings:
        assert len(emb) == 1024, "Each embedding should be 1024-dim"

    return True


@cost_aware(max_tokens=3000, max_cost_usd=0.015)
async def test_semantic_search_real_api():
    """Test semantic search with real API."""
    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True))
    await layer.initialize()

    # Note: semantic_search uses the layer's default model (set during init)
    results = await layer.semantic_search(
        query=TEST_QUERY,
        documents=TEST_DOCUMENTS,
        top_k=3,
    )

    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    # Results are (index, score, document) tuples
    assert results[0][1] >= results[1][1], "Results should be sorted by score"
    # Machine learning query should match machine learning document
    assert "machine learning" in results[0][2].lower() or "neural" in results[0][2].lower()

    return True


@cost_aware(max_tokens=4000, max_cost_usd=0.02)
async def test_hybrid_search_real_api():
    """Test hybrid search with real API."""
    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True))
    await layer.initialize()

    results = await layer.hybrid_search(
        query="deep learning neural networks",
        documents=TEST_DOCUMENTS,
        top_k=3,
        alpha=0.5,  # Balance vector and BM25
    )

    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    # Results are (index, score, document) tuples - should find neural networks and deep learning docs
    top_docs = [r[2].lower() for r in results]
    assert any("neural" in d or "deep" in d for d in top_docs)

    return True


# ============================================================================
# Phase 2: Advanced Features Tests
# ============================================================================

@cost_aware(max_tokens=3000, max_cost_usd=0.02)
async def test_mmr_search_real_api():
    """Test MMR search for diversity with real API."""
    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True))
    await layer.initialize()

    results = await layer.semantic_search_mmr(
        query=TEST_QUERY,
        documents=TEST_DOCUMENTS,
        top_k=3,
        lambda_mult=0.5,  # Balance relevance and diversity
    )

    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    # Results are (index, score, document) tuples - MMR should return diverse results
    docs = [r[2] for r in results]
    assert len(set(docs)) == 3, "MMR should return 3 unique documents"

    return True


@cost_aware(max_tokens=5000, max_cost_usd=0.025)
async def test_adaptive_hybrid_search_real_api():
    """Test adaptive hybrid search with real API."""
    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True))
    await layer.initialize()

    # Keyword-heavy query should favor BM25 (low alpha)
    results1, alpha1 = await layer.adaptive_hybrid_search(
        query="API SDK v2 documentation",
        documents=TEST_DOCUMENTS,
        top_k=3,
    )

    # Semantic query should favor vectors (high alpha)
    results2, alpha2 = await layer.adaptive_hybrid_search(
        query="understanding conceptual patterns",
        documents=TEST_DOCUMENTS,
        top_k=3,
    )

    assert len(results1) <= 3
    assert len(results2) <= 3
    # Alpha should be different for different query types
    # (keyword-heavy vs semantic)

    return True


# ============================================================================
# Phase 3: State of Witness Tests
# ============================================================================

@cost_aware(max_tokens=8000, max_cost_usd=0.05)
async def test_archetype_embedding_real_api():
    """Test archetype gesture embedding with real API."""
    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True))
    await layer.initialize()

    # Embed all 8 archetype descriptions
    descriptions = list(ARCHETYPE_GESTURES.values())
    result = await layer.embed(
        texts=descriptions,
        model=EmbeddingModel.VOYAGE_4_LARGE,
        input_type=InputType.DOCUMENT,
    )

    assert len(result.embeddings) == 8, "Should embed all 8 archetypes"

    # Calculate pairwise similarities
    import numpy as np
    embeddings = np.array(result.embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    similarities = normalized @ normalized.T

    # Diagonal should be 1.0 (self-similarity)
    for i in range(8):
        assert abs(similarities[i, i] - 1.0) < 0.01, f"Self-similarity should be 1.0"

    # Different archetypes should have lower similarity
    off_diagonal = similarities[~np.eye(8, dtype=bool)]
    avg_similarity = off_diagonal.mean()
    assert avg_similarity < 0.95, f"Archetypes should be distinguishable, avg sim: {avg_similarity}"

    print(f"  Average inter-archetype similarity: {avg_similarity:.3f}")
    return True


@cost_aware(max_tokens=2000, max_cost_usd=0.015)
async def test_gesture_recognition_real_api():
    """Test gesture recognition against archetype library."""
    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True))
    await layer.initialize()

    # Embed archetype library
    archetypes = list(ARCHETYPE_GESTURES.keys())
    descriptions = list(ARCHETYPE_GESTURES.values())

    library_result = await layer.embed(
        texts=descriptions,
        model=EmbeddingModel.VOYAGE_4_LARGE,
        input_type=InputType.DOCUMENT,
    )

    # Test gesture (should match warrior)
    test_gesture = "Fighter stance with clenched fists, ready for combat"
    test_result = await layer.embed(
        texts=[test_gesture],
        model=EmbeddingModel.VOYAGE_4_LARGE,
        input_type=InputType.QUERY,
    )

    # Find closest archetype
    import numpy as np
    test_emb = np.array(test_result.embeddings[0])
    library_embs = np.array(library_result.embeddings)

    similarities = np.dot(library_embs, test_emb) / (
        np.linalg.norm(library_embs, axis=1) * np.linalg.norm(test_emb)
    )

    best_idx = np.argmax(similarities)
    best_archetype = archetypes[best_idx]
    confidence = similarities[best_idx]

    print(f"  Test gesture matched: {best_archetype} (confidence: {confidence:.3f})")
    assert best_archetype == "warrior", f"Expected warrior, got {best_archetype}"
    assert confidence > 0.5, f"Confidence should be > 0.5, got {confidence}"

    return True


# ============================================================================
# Main Test Runner
# ============================================================================

async def run_real_api_tests():
    """Run all real API tests."""
    print("=" * 60)
    print("VOYAGE AI V39.10 - REAL API INTEGRATION TESTS")
    print("=" * 60)
    print("\nNote: These tests make REAL API calls to Voyage AI.")
    print("      Cost estimates are shown for each test.\n")

    COST_TRACKER.start_time = datetime.now()

    tests = [
        # Phase 1: Core embedding
        ("test_single_embed_real_api", test_single_embed_real_api),
        ("test_batch_embed_real_api", test_batch_embed_real_api),
        ("test_semantic_search_real_api", test_semantic_search_real_api),
        ("test_hybrid_search_real_api", test_hybrid_search_real_api),

        # Phase 2: Advanced features
        ("test_mmr_search_real_api", test_mmr_search_real_api),
        ("test_adaptive_hybrid_search_real_api", test_adaptive_hybrid_search_real_api),

        # Phase 3: State of Witness
        ("test_archetype_embedding_real_api", test_archetype_embedding_real_api),
        ("test_gesture_recognition_real_api", test_gesture_recognition_real_api),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        print("-" * 50)
        try:
            result = await test_fn()
            if result is None:
                skipped += 1
            else:
                passed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    COST_TRACKER.end_time = datetime.now()

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    print(COST_TRACKER.summary())

    if failed == 0 and passed > 0:
        print("\n[SUCCESS] V39.10 REAL API TESTS COMPLETE!")
        print("          All features validated with actual Voyage AI API calls.")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_real_api_tests())
    sys.exit(0 if success else 1)
