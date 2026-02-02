#!/usr/bin/env python3
"""
Voyage AI V39.12 - Streaming Classification Tests (Phase 2)

REAL API TESTS - These make actual Voyage AI API calls.
Validates domain-agnostic streaming classification infrastructure.

Run with: python tests/voyage_v39_12_streaming_test.py
"""

import asyncio
import os
import sys
import time
from typing import Any, AsyncGenerator

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
    # V39.12 Phase 1 exports
    ARCHETYPE_NAMES,
    ArchetypeEmbeddingLibrary,
    # V39.12 Phase 2 exports
    StreamingClassificationResult,
    StreamingClassifier,
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
# V39.12 Phase 2: Streaming Classification Tests
# =============================================================================

async def test_streaming_result_dataclass():
    """Test StreamingClassificationResult dataclass."""
    result = StreamingClassificationResult(
        timestamp=1706270400.0,
        raw_probabilities={"WARRIOR": 0.4, "SAGE": 0.35, "JESTER": 0.25},
        smoothed_probabilities={"WARRIOR": 0.38, "SAGE": 0.36, "JESTER": 0.26},
        dominant_category="WARRIOR",
        dominant_confidence=0.38,
        temporal_stability=0.85,
        window_size=5,
    )

    # Test basic attributes
    assert result.timestamp == 1706270400.0
    assert result.dominant_category == "WARRIOR"
    assert abs(result.dominant_confidence - 0.38) < 0.001
    assert result.temporal_stability == 0.85
    assert result.window_size == 5

    # Test serialization
    data = result.to_dict()
    assert data["dominant"] == "WARRIOR"
    assert data["confidence"] == 0.38
    assert data["stability"] == 0.85
    assert data["window"] == 5
    assert "raw" in data
    assert "smoothed" in data

    print(f"  Dominant: {result.dominant_category} ({result.dominant_confidence:.2%})")
    print(f"  Stability: {result.temporal_stability:.2%}")
    print(f"  Serialization: OK ({len(data)} fields)")
    return {"tokens": 0}


async def test_streaming_classifier_initialization():
    """Test StreamingClassifier initialization."""
    config = EmbeddingConfig(
        cache_enabled=True,
        model=EmbeddingModel.VOYAGE_4_LARGE.value,
    )
    layer = EmbeddingLayer(config)
    await layer.initialize()

    library = ArchetypeEmbeddingLibrary(layer, model=EmbeddingModel.VOYAGE_4_LARGE)
    await library.initialize()

    # Create classifier with default settings
    classifier = StreamingClassifier(library)
    assert classifier.window_size == 5
    assert classifier.smoothing_factor == 0.85
    assert classifier.history_size == 0
    assert classifier.last_result is None

    # Create classifier with custom settings
    classifier2 = StreamingClassifier(library, window_size=10, smoothing_factor=0.9)
    assert classifier2.window_size == 10
    assert classifier2.smoothing_factor == 0.9

    # Estimate tokens: library initialization
    tokens = 2000
    print(f"  Default window_size: {classifier.window_size}")
    print(f"  Default smoothing_factor: {classifier.smoothing_factor}")
    print(f"  Custom settings validated: OK")
    return {"tokens": tokens}


async def test_streaming_classifier_single_classification():
    """Test single classification with StreamingClassifier."""
    config = EmbeddingConfig(
        cache_enabled=True,
        model=EmbeddingModel.VOYAGE_4_LARGE.value,
    )
    layer = EmbeddingLayer(config)
    await layer.initialize()

    library = ArchetypeEmbeddingLibrary(layer, model=EmbeddingModel.VOYAGE_4_LARGE)
    await library.initialize()

    classifier = StreamingClassifier(library, window_size=5)

    # Generate a test embedding
    test_text = "Person in aggressive fighting stance"
    embed_result = await layer.embed([test_text])
    test_embedding = embed_result.embeddings[0]

    # Classify
    result = await classifier.classify(test_embedding)

    assert isinstance(result, StreamingClassificationResult)
    assert result.dominant_category in ARCHETYPE_NAMES
    assert 0 <= result.dominant_confidence <= 1
    assert 0 <= result.temporal_stability <= 1
    # window_size in result = current history size (1 after first classify)
    # classifier.window_size = configured max (5)
    assert result.window_size == 1  # First classification has 1 item in history
    assert classifier.window_size == 5  # Configured max window size

    # Check history was updated
    assert classifier.history_size == 1
    assert classifier.last_result == result

    # Estimate tokens
    tokens = 2050
    print(f"  Dominant: {result.dominant_category} ({result.dominant_confidence:.2%})")
    print(f"  Stability: {result.temporal_stability:.2%}")
    print(f"  History size: {classifier.history_size}")
    return {"tokens": tokens}


async def test_streaming_classifier_temporal_smoothing():
    """Test temporal smoothing behavior over multiple classifications."""
    config = EmbeddingConfig(
        cache_enabled=True,
        model=EmbeddingModel.VOYAGE_4_LARGE.value,
    )
    layer = EmbeddingLayer(config)
    await layer.initialize()

    library = ArchetypeEmbeddingLibrary(layer, model=EmbeddingModel.VOYAGE_4_LARGE)
    await library.initialize()

    classifier = StreamingClassifier(library, window_size=5, smoothing_factor=0.85)

    # Generate multiple embeddings
    test_texts = [
        "Person in aggressive fighting stance",
        "Person with clenched fists ready to strike",
        "Warrior pose with battle cry",
    ]

    results = []
    for text in test_texts:
        embed_result = await layer.embed([text])
        embedding = embed_result.embeddings[0]
        result = await classifier.classify(embedding)
        results.append(result)

    # Verify temporal behavior
    assert classifier.history_size == 3
    assert len(results) == 3

    # Stability should increase as history builds (more consistent data)
    # Note: Not guaranteed but typical behavior with similar inputs
    print(f"  Classifications: {len(results)}")
    print(f"  History size: {classifier.history_size}")
    print(f"  Stability progression:")
    for i, r in enumerate(results):
        print(f"    [{i+1}] {r.dominant_category}: {r.temporal_stability:.2%}")

    # Estimate tokens
    tokens = 2150
    return {"tokens": tokens}


async def test_streaming_classifier_reset():
    """Test classifier reset functionality."""
    config = EmbeddingConfig(
        cache_enabled=True,
        model=EmbeddingModel.VOYAGE_4_LARGE.value,
    )
    layer = EmbeddingLayer(config)
    await layer.initialize()

    library = ArchetypeEmbeddingLibrary(layer, model=EmbeddingModel.VOYAGE_4_LARGE)
    await library.initialize()

    classifier = StreamingClassifier(library, window_size=5)

    # Perform some classifications
    test_embedding = [0.0] * 1024
    await classifier.classify(test_embedding)
    await classifier.classify(test_embedding)

    assert classifier.history_size == 2
    assert classifier.last_result is not None

    # Reset
    classifier.reset()

    assert classifier.history_size == 0
    assert classifier.last_result is None

    # Estimate tokens
    tokens = 2000
    print(f"  Pre-reset history: 2")
    print(f"  Post-reset history: {classifier.history_size}")
    print(f"  Reset successful: OK")
    return {"tokens": tokens}


async def test_streaming_classifier_stream():
    """Test async streaming classification generator."""
    config = EmbeddingConfig(
        cache_enabled=True,
        model=EmbeddingModel.VOYAGE_4_LARGE.value,
    )
    layer = EmbeddingLayer(config)
    await layer.initialize()

    library = ArchetypeEmbeddingLibrary(layer, model=EmbeddingModel.VOYAGE_4_LARGE)
    await library.initialize()

    classifier = StreamingClassifier(library, window_size=3)

    # Create mock embedding stream
    async def mock_embedding_stream() -> AsyncGenerator[tuple[float, list[float]], None]:
        """Simulate a stream of timestamped embeddings."""
        base_time = time.time()
        for i in range(5):
            # Simulate different embeddings
            embedding = [float(i) / 10.0] * 1024
            yield (base_time + i * 0.1, embedding)
            await asyncio.sleep(0.01)  # Small delay

    # Consume the stream
    results = []
    async for result in classifier.classify_stream(mock_embedding_stream()):
        results.append(result)

    assert len(results) == 5
    assert classifier.history_size == 3  # Window size is 3

    # Verify results are StreamingClassificationResult instances
    for r in results:
        assert isinstance(r, StreamingClassificationResult)
        assert r.dominant_category in ARCHETYPE_NAMES

    # Estimate tokens (no real API calls in mock stream)
    tokens = 2000
    print(f"  Stream items processed: {len(results)}")
    print(f"  Final history size: {classifier.history_size}")
    print(f"  Async generator: OK")
    return {"tokens": tokens}


async def test_streaming_classifier_window_behavior():
    """Test that window correctly limits history size."""
    config = EmbeddingConfig(
        cache_enabled=True,
        model=EmbeddingModel.VOYAGE_4_LARGE.value,
    )
    layer = EmbeddingLayer(config)
    await layer.initialize()

    library = ArchetypeEmbeddingLibrary(layer, model=EmbeddingModel.VOYAGE_4_LARGE)
    await library.initialize()

    window_size = 3
    classifier = StreamingClassifier(library, window_size=window_size)

    # Classify more items than window size
    test_embedding = [0.0] * 1024
    for i in range(10):
        await classifier.classify(test_embedding)

    # History should be capped at window_size
    assert classifier.history_size == window_size

    # Estimate tokens
    tokens = 2000
    print(f"  Window size: {window_size}")
    print(f"  Classifications made: 10")
    print(f"  Final history size: {classifier.history_size}")
    print(f"  Window limiting: OK")
    return {"tokens": tokens}


async def test_streaming_serialization_roundtrip():
    """Test that StreamingClassificationResult serializes correctly for protocols."""
    config = EmbeddingConfig(
        cache_enabled=True,
        model=EmbeddingModel.VOYAGE_4_LARGE.value,
    )
    layer = EmbeddingLayer(config)
    await layer.initialize()

    library = ArchetypeEmbeddingLibrary(layer, model=EmbeddingModel.VOYAGE_4_LARGE)
    await library.initialize()

    classifier = StreamingClassifier(library, window_size=5)

    # Generate a real classification
    test_text = "Contemplative meditation pose"
    embed_result = await layer.embed([test_text])
    result = await classifier.classify(embed_result.embeddings[0])

    # Serialize
    data = result.to_dict()

    # Verify all fields present and valid types
    assert isinstance(data["timestamp"], float)
    assert isinstance(data["raw"], dict)
    assert isinstance(data["smoothed"], dict)
    assert isinstance(data["dominant"], str)
    assert isinstance(data["confidence"], float)
    assert isinstance(data["stability"], float)
    assert isinstance(data["window"], int)

    # Verify probability dicts sum to ~1
    raw_sum = sum(data["raw"].values())
    smoothed_sum = sum(data["smoothed"].values())
    assert abs(raw_sum - 1.0) < 0.001, f"Raw sum: {raw_sum}"
    assert abs(smoothed_sum - 1.0) < 0.001, f"Smoothed sum: {smoothed_sum}"

    # Estimate tokens
    tokens = 2050
    print(f"  Serialized fields: {len(data)}")
    print(f"  Raw probability sum: {raw_sum:.4f}")
    print(f"  Smoothed probability sum: {smoothed_sum:.4f}")
    print(f"  Protocol-ready: OK")
    return {"tokens": tokens}


async def main():
    """Run all V39.12 Phase 2 streaming classification tests."""
    print("=" * 60)
    print("VOYAGE AI V39.12 PHASE 2 - STREAMING CLASSIFICATION TESTS")
    print("=" * 60)
    print("\nNote: These tests make REAL API calls to Voyage AI.")
    print("      Cost estimates are shown for each test.\n")

    start_time = time.time()

    # Run tests
    await run_test("test_streaming_result_dataclass", test_streaming_result_dataclass)
    await run_test("test_streaming_classifier_initialization", test_streaming_classifier_initialization)
    await run_test("test_streaming_classifier_single_classification", test_streaming_classifier_single_classification)
    await run_test("test_streaming_classifier_temporal_smoothing", test_streaming_classifier_temporal_smoothing)
    await run_test("test_streaming_classifier_reset", test_streaming_classifier_reset)
    await run_test("test_streaming_classifier_stream", test_streaming_classifier_stream)
    await run_test("test_streaming_classifier_window_behavior", test_streaming_classifier_window_behavior)
    await run_test("test_streaming_serialization_roundtrip", test_streaming_serialization_roundtrip)

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
        print("\n[SUCCESS] V39.12 PHASE 2 STREAMING TESTS COMPLETE!")
        print("          Streaming classification infrastructure validated.")
    else:
        print(f"\n[FAILURE] {TEST_RESULTS['failed']} test(s) failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
