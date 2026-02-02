#!/usr/bin/env python3
"""
Voyage AI V39.1 - Cache Enhancement Validation Tests
=====================================================

Tests the new cache functionality:
1. get_cache_stats() method
2. warm_cache() method
3. get_cache_efficiency_report() method
4. export_cache() / import_cache() methods
5. save/load_cache_to_file() methods
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orchestration.embedding_layer import EmbeddingLayer, EmbeddingConfig, EmbeddingModel, InputType


async def test_cache_stats():
    """Test get_cache_stats() method."""
    print("\n[TEST] Cache Stats Method")
    print("-" * 40)

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=100))
    await layer.initialize()

    # Initial stats - should be empty
    stats = layer.get_cache_stats()
    assert stats["hits"] == 0, "Expected 0 hits initially"
    assert stats["misses"] == 0, "Expected 0 misses initially"
    assert stats["cache_size"] == 0, "Expected empty cache initially"
    assert "memory_mb" in stats, "Should include memory_mb"
    print(f"  Initial stats: {stats}")

    # Embed something to populate cache
    result = await layer.embed(
        texts=["Test document for caching"],
        model=EmbeddingModel.VOYAGE_4_LITE,
        input_type=InputType.DOCUMENT,
    )

    # Stats should now show activity
    stats = layer.get_cache_stats()
    assert stats["misses"] == 1, "Expected 1 miss after first embed"
    assert stats["cache_size"] == 1, "Expected 1 entry in cache"
    print(f"  After embed: {stats}")

    # Embed same text again - should hit cache
    result2 = await layer.embed(
        texts=["Test document for caching"],
        model=EmbeddingModel.VOYAGE_4_LITE,
        input_type=InputType.DOCUMENT,
    )

    stats = layer.get_cache_stats()
    assert stats["hits"] == 1, "Expected 1 hit after re-embed"
    print(f"  After cache hit: {stats}")

    print("  [PASS] Cache stats working correctly")
    return True


async def test_cache_warming():
    """Test warm_cache() method."""
    print("\n[TEST] Cache Warming")
    print("-" * 40)

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=100))
    await layer.initialize()

    texts_to_warm = [
        "First document to pre-cache",
        "Second document to pre-cache",
        "Third document to pre-cache",
    ]

    # Warm cache
    result = await layer.warm_cache(
        texts=texts_to_warm,
        model=EmbeddingModel.VOYAGE_4_LITE,
        input_type=InputType.DOCUMENT,
    )

    assert result["warmed"] == 3, f"Expected 3 warmed, got {result['warmed']}"
    assert result["already_cached"] == 0, "Expected 0 already cached"
    print(f"  Warming result: {result}")

    # Try warming same texts again - should show already cached
    result2 = await layer.warm_cache(
        texts=texts_to_warm,
        model=EmbeddingModel.VOYAGE_4_LITE,
        input_type=InputType.DOCUMENT,
    )

    assert result2["already_cached"] == 3, f"Expected 3 already cached, got {result2['already_cached']}"
    assert result2["warmed"] == 0, "Expected 0 newly warmed"
    print(f"  Re-warming result: {result2}")

    print("  [PASS] Cache warming working correctly")
    return True


async def test_efficiency_report():
    """Test get_cache_efficiency_report() method."""
    print("\n[TEST] Cache Efficiency Report")
    print("-" * 40)

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=10))
    await layer.initialize()

    # Generate some cache activity
    for i in range(5):
        await layer.embed(
            texts=[f"Document {i}"],
            model=EmbeddingModel.VOYAGE_4_LITE,
            input_type=InputType.DOCUMENT,
        )

    # Generate some hits
    for i in range(3):
        await layer.embed(
            texts=[f"Document {i}"],  # Already cached
            model=EmbeddingModel.VOYAGE_4_LITE,
            input_type=InputType.DOCUMENT,
        )

    report = layer.get_cache_efficiency_report()
    print(f"  Efficiency score: {report['efficiency_score']}")
    print(f"  Utilization: {report['utilization_percent']}%")
    print(f"  Capacity: {report['capacity']}")
    print(f"  Recommendations: {report['recommendations']}")

    assert "efficiency_score" in report, "Should include efficiency_score"
    assert "utilization_percent" in report, "Should include utilization_percent"
    assert "recommendations" in report, "Should include recommendations"

    print("  [PASS] Efficiency report working correctly")
    return True


async def test_cache_export_import():
    """Test export_cache() and import_cache() methods."""
    print("\n[TEST] Cache Export/Import")
    print("-" * 40)

    # Create and populate cache
    layer1 = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=100))
    await layer1.initialize()

    await layer1.embed(
        texts=["Export test doc 1", "Export test doc 2"],
        model=EmbeddingModel.VOYAGE_4_LITE,
        input_type=InputType.DOCUMENT,
    )

    # Export
    exported = layer1.export_cache()
    print(f"  Exported {len(exported['entries'])} entries")
    assert len(exported["entries"]) == 2, "Expected 2 entries"

    # Create new layer and import
    layer2 = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=100))
    await layer2.initialize()

    result = layer2.import_cache(exported, validate_ttl=False)
    print(f"  Imported: {result['imported']}, Skipped: {result['skipped']}")

    assert result["imported"] == 2, f"Expected 2 imported, got {result['imported']}"

    # Verify cache works - should hit
    await layer2.embed(
        texts=["Export test doc 1"],
        model=EmbeddingModel.VOYAGE_4_LITE,
        input_type=InputType.DOCUMENT,
    )

    stats = layer2.get_cache_stats()
    assert stats["hits"] >= 1, "Expected at least 1 hit from imported cache"

    print("  [PASS] Cache export/import working correctly")
    return True


async def test_cache_file_persistence():
    """Test save_cache_to_file() and load_cache_from_file() methods."""
    print("\n[TEST] Cache File Persistence")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "test_cache.json"

        # Create and populate cache
        layer1 = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=100))
        await layer1.initialize()

        await layer1.embed(
            texts=["Persistence test doc"],
            model=EmbeddingModel.VOYAGE_4_LITE,
            input_type=InputType.DOCUMENT,
        )

        # Save to file
        save_result = await layer1.save_cache_to_file(str(cache_file))
        print(f"  Save result: {save_result}")
        assert save_result["success"], "Save should succeed"
        assert cache_file.exists(), "Cache file should exist"

        # Load into new layer
        layer2 = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=100))
        await layer2.initialize()

        load_result = await layer2.load_cache_from_file(str(cache_file), validate_ttl=False)
        print(f"  Load result: {load_result}")
        assert load_result["success"], "Load should succeed"
        assert load_result["imported"] == 1, "Should import 1 entry"

        # Verify loaded cache works
        stats = layer2.get_cache_stats()
        assert stats["cache_size"] == 1, "Cache should have 1 entry"

    print("  [PASS] Cache file persistence working correctly")
    return True


async def main():
    """Run all cache tests."""
    print("=" * 60)
    print("VOYAGE AI V39.1 - CACHE ENHANCEMENT TESTS")
    print("=" * 60)

    tests = [
        ("Cache Stats", test_cache_stats),
        ("Cache Warming", test_cache_warming),
        ("Efficiency Report", test_efficiency_report),
        ("Export/Import", test_cache_export_import),
        ("File Persistence", test_cache_file_persistence),
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
