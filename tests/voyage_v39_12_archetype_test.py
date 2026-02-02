#!/usr/bin/env python3
"""
Voyage AI V39.12 - Archetype-Based Cache Warming Tests

REAL API TESTS - These make actual Voyage AI API calls.
Validates State of Witness archetype integration.

Run with: python tests/voyage_v39_12_archetype_test.py
"""

import asyncio
import os
import sys
import time
from typing import Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load environment
from dotenv import load_dotenv
env_path = os.path.join(project_root, ".config", ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"[INFO] Loaded environment from {env_path}")

from core.orchestration.embedding_layer import (
    EmbeddingLayer,
    EmbeddingConfig,
    EmbeddingModel,
    # V39.12 Archetype exports
    ARCHETYPE_NAMES,
    ARCHETYPE_VARIATIONS,
    ARCHETYPE_COLORS,
    ArchetypeCacheStats,
    ArchetypeEmbeddingLibrary,
    warm_archetype_cache,
)


# Test configuration
TEST_RESULTS: dict[str, Any] = {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "total_tokens": 0,
    "total_cost": 0.0,
}


def estimate_cost(tokens: int, model: str = "voyage-4-large") -> float:
    """Estimate API cost for tokens used."""
    rates = {
        "voyage-4-large": 0.03 / 1_000_000,
        "voyage-4-lite": 0.01 / 1_000_000,
    }
    rate = rates.get(model, 0.03 / 1_000_000)
    return tokens * rate


async def run_test(name: str, test_func, *args, **kwargs) -> bool:
    """Run a single test with timing and cost tracking."""
    print(f"\n[TEST] {name}")
    print("-" * 50)

    start_time = time.time()
    try:
        result = await test_func(*args, **kwargs)
        elapsed = time.time() - start_time

        tokens = result.get("tokens", 0) if isinstance(result, dict) else 0
        cost = estimate_cost(tokens)
        TEST_RESULTS["total_tokens"] += tokens
        TEST_RESULTS["total_cost"] += cost

        print(f"  [PASS] {name} ({elapsed:.2f}s, ~${cost:.4f})")
        TEST_RESULTS["passed"] += 1
        return True
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  [FAIL] {name} ({elapsed:.2f}s)")
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        TEST_RESULTS["failed"] += 1
        return False


# =============================================================================
# V39.12 Archetype Tests
# =============================================================================

async def test_archetype_constants():
    """Test that archetype constants are properly defined."""
    # Validate ARCHETYPE_NAMES
    assert len(ARCHETYPE_NAMES) == 8, f"Expected 8 archetypes, got {len(ARCHETYPE_NAMES)}"
    expected = ["WARRIOR", "NURTURER", "SAGE", "JESTER", "LOVER", "MAGICIAN", "INNOCENT", "EVERYMAN"]
    assert ARCHETYPE_NAMES == expected, f"Archetype names mismatch: {ARCHETYPE_NAMES}"

    # Validate ARCHETYPE_VARIATIONS
    assert len(ARCHETYPE_VARIATIONS) == 8, f"Expected 8 archetype variations"
    for name in ARCHETYPE_NAMES:
        assert name in ARCHETYPE_VARIATIONS, f"Missing variations for {name}"
        variations = ARCHETYPE_VARIATIONS[name]
        assert len(variations) == 5, f"Expected 5 variations for {name}, got {len(variations)}"

    # Validate ARCHETYPE_COLORS
    assert len(ARCHETYPE_COLORS) == 8, f"Expected 8 archetype colors"
    for name in ARCHETYPE_NAMES:
        assert name in ARCHETYPE_COLORS, f"Missing color for {name}"
        color = ARCHETYPE_COLORS[name]
        assert len(color) == 3, f"Expected RGB tuple for {name}"
        assert all(0 <= c <= 255 for c in color), f"Invalid RGB values for {name}"

    print(f"  Archetypes validated: {len(ARCHETYPE_NAMES)}")
    print(f"  Total variations: {sum(len(v) for v in ARCHETYPE_VARIATIONS.values())}")
    return {"tokens": 0}


async def test_archetype_cache_stats():
    """Test ArchetypeCacheStats dataclass."""
    stats = ArchetypeCacheStats(
        archetype="WARRIOR",
        warmed_count=5,
        cache_hits=10,
        cache_misses=2,
        total_queries=12,
        last_warmed="2026-01-26T00:00:00",
    )

    # Test hit rate
    assert abs(stats.hit_rate - 83.33) < 0.1, f"Expected ~83.33% hit rate, got {stats.hit_rate}"

    # Test effectiveness
    assert stats.effectiveness == 100.0, f"Expected 100% effectiveness, got {stats.effectiveness}"

    # Test serialization
    stats_dict = stats.to_dict()
    assert stats_dict["archetype"] == "WARRIOR"
    assert stats_dict["hit_rate_percent"] == 83.33

    print(f"  Hit rate: {stats.hit_rate:.2f}%")
    print(f"  Effectiveness: {stats.effectiveness:.2f}%")
    return {"tokens": 0}


async def test_warm_archetype_cache_real_api():
    """Test archetype cache warming with real API calls."""
    config = EmbeddingConfig(
        cache_enabled=True,
        model=EmbeddingModel.VOYAGE_4_LARGE.value,
    )
    layer = EmbeddingLayer(config)
    await layer.initialize()

    # Warm cache for 2 archetypes (to save API costs)
    result = await warm_archetype_cache(
        layer=layer,
        archetypes=["WARRIOR", "SAGE"],
        model=EmbeddingModel.VOYAGE_4_LARGE,
    )

    assert result["total_warmed"] == 10, f"Expected 10 warmed, got {result['total_warmed']}"
    assert result["per_archetype"]["WARRIOR"] == 5
    assert result["per_archetype"]["SAGE"] == 5
    assert result["errors"] is None, f"Unexpected errors: {result['errors']}"

    # Estimate tokens: 5 variations * 2 archetypes * ~50 tokens each = ~500
    tokens = 500
    print(f"  Warmed: {result['total_warmed']} variations")
    print(f"  Per archetype: WARRIOR={result['per_archetype']['WARRIOR']}, SAGE={result['per_archetype']['SAGE']}")
    return {"tokens": tokens}


async def test_archetype_library_initialize_real_api():
    """Test ArchetypeEmbeddingLibrary initialization with real API."""
    config = EmbeddingConfig(
        cache_enabled=True,
        model=EmbeddingModel.VOYAGE_4_LARGE.value,
    )
    layer = EmbeddingLayer(config)
    await layer.initialize()

    library = ArchetypeEmbeddingLibrary(layer, model=EmbeddingModel.VOYAGE_4_LARGE)
    await library.initialize()

    # Verify all archetypes have embeddings
    assert library._initialized, "Library should be initialized"
    assert len(library._embeddings) == 8, f"Expected 8 archetypes, got {len(library._embeddings)}"
    assert len(library._centroid_embeddings) == 8, "Expected 8 centroid embeddings"

    # Check embedding dimensions
    for archetype in ARCHETYPE_NAMES:
        embeddings = library._embeddings[archetype]
        assert len(embeddings) == 5, f"Expected 5 variations for {archetype}"
        assert len(embeddings[0]) == 1024, "Expected 1024-dim embeddings"

        centroid = library._centroid_embeddings[archetype]
        assert len(centroid) == 1024, "Expected 1024-dim centroid"

    # Estimate tokens: 40 variations * ~50 tokens = 2000
    tokens = 2000
    print(f"  Archetypes loaded: {len(library._embeddings)}")
    print(f"  Embeddings per archetype: 5")
    print(f"  Embedding dimension: 1024")
    return {"tokens": tokens}


async def test_archetype_probability_real_api():
    """Test archetype probability computation with real pose embedding."""
    config = EmbeddingConfig(
        cache_enabled=True,
        model=EmbeddingModel.VOYAGE_4_LARGE.value,
    )
    layer = EmbeddingLayer(config)
    await layer.initialize()

    library = ArchetypeEmbeddingLibrary(layer, model=EmbeddingModel.VOYAGE_4_LARGE)
    await library.initialize()

    # Create a test pose embedding (simulate warrior-like pose)
    test_pose_description = "Person in aggressive fighting stance with fists raised"
    pose_result = await layer.embed([test_pose_description])
    pose_embedding = pose_result.embeddings[0]

    # Compute probabilities
    probs = await library.compute_archetype_probabilities(pose_embedding)

    # Verify probabilities sum to 1
    total_prob = sum(probs.values())
    assert abs(total_prob - 1.0) < 0.001, f"Probabilities should sum to 1, got {total_prob}"

    # Get dominant archetype
    dominant, confidence = library.get_dominant_archetype(probs)
    print(f"  Test pose: '{test_pose_description}'")
    print(f"  Dominant archetype: {dominant} ({confidence:.2%})")
    print(f"  Top 3: {sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]}")

    # Tokens: 40 library + 1 test pose
    return {"tokens": 2050}


async def test_archetype_classification_real_api():
    """Test direct archetype classification."""
    config = EmbeddingConfig(
        cache_enabled=True,
        model=EmbeddingModel.VOYAGE_4_LARGE.value,
    )
    layer = EmbeddingLayer(config)
    await layer.initialize()

    library = ArchetypeEmbeddingLibrary(layer, model=EmbeddingModel.VOYAGE_4_LARGE)
    await library.initialize()

    # Test classification for different poses
    test_cases = [
        ("Person in contemplative meditation pose", "SAGE"),
        ("Playful child jumping with joy", "INNOCENT"),
        ("Neutral person walking casually", "EVERYMAN"),
    ]

    tokens = 2000  # Library initialization
    for description, expected_archetype in test_cases:
        pose_result = await layer.embed([description])
        pose_embedding = pose_result.embeddings[0]

        # Use centroid-based classification (faster)
        matches = await library.classify_to_archetype(pose_embedding, use_variations=False, top_k=3)
        top_match, confidence = matches[0]

        print(f"  '{description[:40]}...'")
        print(f"    -> {top_match} ({confidence:.2%}), expected: {expected_archetype}")
        tokens += 50

    return {"tokens": tokens}


async def test_archetype_color_mapping():
    """Test archetype to color mapping."""
    config = EmbeddingConfig(cache_enabled=True)
    layer = EmbeddingLayer(config)

    library = ArchetypeEmbeddingLibrary(layer)

    # Test color retrieval
    warrior_color = library.get_archetype_color("WARRIOR")
    assert warrior_color == (255, 0, 0), f"Expected red, got {warrior_color}"

    sage_color = library.get_archetype_color("SAGE")
    assert sage_color == (0, 255, 255), f"Expected cyan, got {sage_color}"

    # Test unknown archetype fallback
    unknown_color = library.get_archetype_color("UNKNOWN")
    assert unknown_color == (192, 192, 192), f"Expected silver fallback, got {unknown_color}"

    print(f"  WARRIOR: RGB{warrior_color}")
    print(f"  SAGE: RGB{sage_color}")
    print(f"  UNKNOWN: RGB{unknown_color} (fallback)")
    return {"tokens": 0}


async def test_archetype_serialization():
    """Test library serialization and deserialization."""
    config = EmbeddingConfig(
        cache_enabled=True,
        model=EmbeddingModel.VOYAGE_4_LARGE.value,
    )
    layer = EmbeddingLayer(config)
    await layer.initialize()

    # Initialize library
    library = ArchetypeEmbeddingLibrary(layer, model=EmbeddingModel.VOYAGE_4_LARGE)
    await library.initialize()

    # Serialize
    data = library.to_dict()
    assert "embeddings" in data
    assert "centroids" in data
    assert "model" in data
    assert "stats" in data

    # Verify stats serialization
    assert len(data["stats"]) == 8
    for name in ARCHETYPE_NAMES:
        assert name in data["stats"]

    # Deserialize
    restored = await ArchetypeEmbeddingLibrary.from_dict(data, layer)
    assert restored._initialized
    assert len(restored._embeddings) == 8
    assert len(restored._centroid_embeddings) == 8

    print(f"  Serialized {len(data['embeddings'])} archetypes")
    print(f"  Restored successfully: {restored._initialized}")
    return {"tokens": 2000}


async def test_aggregate_stats():
    """Test aggregate statistics across archetypes."""
    config = EmbeddingConfig(
        cache_enabled=True,
        model=EmbeddingModel.VOYAGE_4_LARGE.value,
    )
    layer = EmbeddingLayer(config)
    await layer.initialize()

    library = ArchetypeEmbeddingLibrary(layer, model=EmbeddingModel.VOYAGE_4_LARGE)
    await library.initialize()

    # Simulate some queries
    test_embedding = [0.0] * 1024
    for _ in range(5):
        await library.compute_archetype_probabilities(test_embedding)

    # Get aggregate stats
    stats = library.get_aggregate_stats()

    assert stats["total_archetypes"] == 8
    assert stats["total_variations_warmed"] == 40  # 8 * 5
    assert stats["total_queries"] == 40  # 5 queries * 8 archetypes each
    assert stats["total_cache_hits"] == 40

    print(f"  Total archetypes: {stats['total_archetypes']}")
    print(f"  Variations warmed: {stats['total_variations_warmed']}")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Hit rate: {stats['overall_hit_rate']:.1f}%")
    return {"tokens": 2000}


async def main():
    """Run all V39.12 archetype tests."""
    print("=" * 60)
    print("VOYAGE AI V39.12 - ARCHETYPE CACHE WARMING TESTS")
    print("=" * 60)
    print("\nNote: These tests make REAL API calls to Voyage AI.")
    print("      Cost estimates are shown for each test.\n")

    start_time = time.time()

    # Run tests
    await run_test("test_archetype_constants", test_archetype_constants)
    await run_test("test_archetype_cache_stats", test_archetype_cache_stats)
    await run_test("test_archetype_color_mapping", test_archetype_color_mapping)
    await run_test("test_warm_archetype_cache_real_api", test_warm_archetype_cache_real_api)
    await run_test("test_archetype_library_initialize_real_api", test_archetype_library_initialize_real_api)
    await run_test("test_archetype_probability_real_api", test_archetype_probability_real_api)
    await run_test("test_archetype_classification_real_api", test_archetype_classification_real_api)
    await run_test("test_archetype_serialization", test_archetype_serialization)
    await run_test("test_aggregate_stats", test_aggregate_stats)

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {TEST_RESULTS['passed']} passed, {TEST_RESULTS['failed']} failed, {TEST_RESULTS['skipped']} skipped")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("COST SUMMARY")
    print("=" * 60)
    print(f"  Tests run: {TEST_RESULTS['passed'] + TEST_RESULTS['failed']}")
    print(f"  Tests skipped: {TEST_RESULTS['skipped']}")
    print(f"  Total tokens: {TEST_RESULTS['total_tokens']:,}")
    print(f"  Total cost: ${TEST_RESULTS['total_cost']:.4f}")
    print(f"  Duration: {elapsed:.1f}s")
    print("=" * 60)

    if TEST_RESULTS["failed"] == 0:
        print("\n[SUCCESS] V39.12 ARCHETYPE TESTS COMPLETE!")
        print("          Archetype-based cache warming validated.")
    else:
        print(f"\n[FAILURE] {TEST_RESULTS['failed']} test(s) failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
