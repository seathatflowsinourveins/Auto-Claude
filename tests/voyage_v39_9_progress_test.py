#!/usr/bin/env python3
"""
Voyage AI V39.9 - Batch Progress Streaming Tests
=================================================

Tests the V39.9 enhancements:
1. BatchProgressMetrics - rate and ETA calculation
2. BatchProgressEvent - progress event dataclass
3. stream_batch_progress() - AsyncGenerator method
4. wait_for_batch_completion_with_progress() - callback method

Note: These tests validate signatures and local logic.
      Actual batch API streaming tests require a running batch job.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orchestration.embedding_layer import (
    EmbeddingLayer,
    EmbeddingConfig,
    BatchProgressMetrics,
    BatchProgressEvent,
    BatchStatus,
    BatchRequestCounts,
    BatchJob,
)


async def test_batch_progress_metrics_creation():
    """Test BatchProgressMetrics dataclass creation."""
    print("\n[TEST] BatchProgressMetrics Creation")
    print("-" * 50)

    metrics = BatchProgressMetrics(start_time=datetime.now())

    assert metrics.start_time is not None, "start_time should be set"
    assert metrics.samples == [], "samples should be empty initially"
    assert metrics.window_size == 10, "default window_size should be 10"

    print(f"  start_time: {metrics.start_time}")
    print(f"  samples: {metrics.samples}")
    print(f"  window_size: {metrics.window_size}")

    print("  [PASS] BatchProgressMetrics created successfully")
    return True


async def test_batch_progress_metrics_add_sample():
    """Test adding samples to BatchProgressMetrics."""
    print("\n[TEST] BatchProgressMetrics add_sample()")
    print("-" * 50)

    metrics = BatchProgressMetrics(start_time=datetime.now())

    # Add some samples
    metrics.add_sample(10)
    metrics.add_sample(20)
    metrics.add_sample(30)

    assert len(metrics.samples) == 3, "Should have 3 samples"
    assert metrics.samples[0][1] == 10, "First sample should be 10"
    assert metrics.samples[2][1] == 30, "Last sample should be 30"

    print(f"  Added 3 samples: {[s[1] for s in metrics.samples]}")

    # Test window size limiting
    for i in range(15):
        metrics.add_sample(40 + i)

    assert len(metrics.samples) == 10, f"Should be limited to window_size=10, got {len(metrics.samples)}"

    print(f"  After adding 15 more: {len(metrics.samples)} samples (window limited)")
    print("  [PASS] Sample management working correctly")
    return True


async def test_batch_progress_metrics_rate_calculation():
    """Test rate calculation in BatchProgressMetrics."""
    print("\n[TEST] BatchProgressMetrics calculate_rate()")
    print("-" * 50)

    metrics = BatchProgressMetrics(start_time=datetime.now())

    # No samples - should return 0
    rate = metrics.calculate_rate()
    assert rate == 0.0, "Rate with no samples should be 0"
    print(f"  Rate with 0 samples: {rate}")

    # One sample - should return 0
    metrics.add_sample(10)
    rate = metrics.calculate_rate()
    assert rate == 0.0, "Rate with 1 sample should be 0"
    print(f"  Rate with 1 sample: {rate}")

    # Simulate time passing with samples
    # Manually set samples with known timestamps for predictable rate
    now = datetime.now()
    metrics.samples = [
        (now - timedelta(seconds=10), 0),
        (now, 100),
    ]
    rate = metrics.calculate_rate()
    expected_rate = 10.0  # 100 items in 10 seconds = 10/sec
    assert abs(rate - expected_rate) < 0.1, f"Expected rate ~{expected_rate}, got {rate}"

    print(f"  Rate with 100 items over 10s: {rate:.1f}/sec")
    print("  [PASS] Rate calculation working correctly")
    return True


async def test_batch_progress_metrics_eta_calculation():
    """Test ETA calculation in BatchProgressMetrics."""
    print("\n[TEST] BatchProgressMetrics calculate_eta()")
    print("-" * 50)

    metrics = BatchProgressMetrics(start_time=datetime.now())

    # No rate - should return infinity
    eta = metrics.calculate_eta(total=100, completed=50)
    assert eta == float('inf'), "ETA with no rate should be infinity"
    print(f"  ETA with 0 rate: {eta}")

    # Set up known rate
    now = datetime.now()
    metrics.samples = [
        (now - timedelta(seconds=10), 0),
        (now, 50),
    ]
    # Rate = 5/sec, 50 remaining, ETA = 10 seconds
    eta = metrics.calculate_eta(total=100, completed=50)
    expected_eta = 10.0
    assert abs(eta - expected_eta) < 0.5, f"Expected ETA ~{expected_eta}, got {eta}"

    print(f"  ETA with 50/100 at 5/sec: {eta:.1f}s")
    print("  [PASS] ETA calculation working correctly")
    return True


async def test_batch_progress_event_creation():
    """Test BatchProgressEvent dataclass creation."""
    print("\n[TEST] BatchProgressEvent Creation")
    print("-" * 50)

    event = BatchProgressEvent(
        batch_id="batch-test-123",
        status=BatchStatus.IN_PROGRESS,
        total=100,
        completed=50,
        failed=2,
        percent=50.0,
        rate=5.0,
        eta_seconds=10.0,
        is_complete=False,
        is_failed=False,
        timestamp=datetime.now(),
    )

    assert event.batch_id == "batch-test-123"
    assert event.status == BatchStatus.IN_PROGRESS
    assert event.total == 100
    assert event.completed == 50
    assert event.percent == 50.0
    assert not event.is_complete
    assert not event.is_failed

    print(f"  batch_id: {event.batch_id}")
    print(f"  status: {event.status.value}")
    print(f"  progress: {event.completed}/{event.total} ({event.percent}%)")
    print(f"  rate: {event.rate}/sec, ETA: {event.eta_seconds}s")
    print(f"  repr: {event}")

    print("  [PASS] BatchProgressEvent created successfully")
    return True


async def test_batch_progress_event_from_batch_job():
    """Test BatchProgressEvent.from_batch_job() factory method."""
    print("\n[TEST] BatchProgressEvent.from_batch_job()")
    print("-" * 50)

    # Create a mock BatchJob
    job = BatchJob(
        id="batch-factory-test",
        status=BatchStatus.IN_PROGRESS,
        endpoint="/v1/embeddings",
        input_file_id="file-input",
        output_file_id=None,
        error_file_id=None,
        created_at="2026-01-26T12:00:00Z",
        expected_completion_at="2026-01-26T14:00:00Z",
        request_counts=BatchRequestCounts(total=200, completed=100, failed=5),
        metadata={"test": "true"},
    )

    # Create event without metrics
    event = BatchProgressEvent.from_batch_job(job)

    assert event.batch_id == "batch-factory-test"
    assert event.total == 200
    assert event.completed == 100
    assert event.failed == 5
    assert event.percent == 50.0
    assert event.rate == 0.0  # No metrics
    assert event.eta_seconds == float('inf')  # No metrics
    assert not event.is_complete
    assert not event.is_failed

    print(f"  From BatchJob without metrics:")
    print(f"    progress: {event.completed}/{event.total} ({event.percent}%)")
    print(f"    rate: {event.rate}, ETA: {event.eta_seconds}")

    # Create event with metrics
    now = datetime.now()
    metrics = BatchProgressMetrics(start_time=now - timedelta(seconds=20))
    metrics.samples = [
        (now - timedelta(seconds=10), 50),
        (now, 100),
    ]
    # Rate = 5/sec

    event_with_metrics = BatchProgressEvent.from_batch_job(job, metrics)
    assert event_with_metrics.rate == 5.0
    assert abs(event_with_metrics.eta_seconds - 20.0) < 1.0  # 100 remaining at 5/sec

    print(f"  From BatchJob with metrics:")
    print(f"    rate: {event_with_metrics.rate}/sec")
    print(f"    ETA: {event_with_metrics.eta_seconds:.1f}s")

    print("  [PASS] from_batch_job() factory working correctly")
    return True


async def test_stream_batch_progress_signature():
    """Test stream_batch_progress() method signature."""
    print("\n[TEST] stream_batch_progress() Signature")
    print("-" * 50)

    import inspect

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True))
    await layer.initialize()

    # Check method exists
    assert hasattr(layer, "stream_batch_progress"), "stream_batch_progress missing"

    # Check signature
    sig = inspect.signature(layer.stream_batch_progress)
    params = sig.parameters

    assert "batch_id" in params, "batch_id parameter missing"
    assert "poll_interval" in params, "poll_interval parameter missing"
    assert "include_metrics" in params, "include_metrics parameter missing"

    print(f"  batch_id: required")
    print(f"  poll_interval default: {params['poll_interval'].default}")
    print(f"  include_metrics default: {params['include_metrics'].default}")

    assert params["poll_interval"].default == 5.0
    assert params["include_metrics"].default is True

    # Check return type annotation
    return_annotation = sig.return_annotation
    print(f"  return type: {return_annotation}")

    print("  [PASS] V39.9 stream_batch_progress signature correct")
    return True


async def test_wait_for_batch_completion_with_progress_signature():
    """Test wait_for_batch_completion_with_progress() method signature."""
    print("\n[TEST] wait_for_batch_completion_with_progress() Signature")
    print("-" * 50)

    import inspect

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True))
    await layer.initialize()

    # Check method exists
    assert hasattr(layer, "wait_for_batch_completion_with_progress"), "wait_for_batch_completion_with_progress missing"

    # Check signature
    sig = inspect.signature(layer.wait_for_batch_completion_with_progress)
    params = sig.parameters

    assert "batch_id" in params, "batch_id parameter missing"
    assert "on_progress" in params, "on_progress parameter missing"
    assert "poll_interval" in params, "poll_interval parameter missing"
    assert "max_wait" in params, "max_wait parameter missing"

    print(f"  batch_id: required")
    print(f"  on_progress: required (async callback)")
    print(f"  poll_interval default: {params['poll_interval'].default}")
    print(f"  max_wait default: {params['max_wait'].default}")

    assert params["poll_interval"].default == 5.0
    assert params["max_wait"].default == 43200  # 12 hours

    print("  [PASS] V39.9 wait_for_batch_completion_with_progress signature correct")
    return True


async def test_batch_progress_event_repr():
    """Test BatchProgressEvent __repr__ formatting."""
    print("\n[TEST] BatchProgressEvent __repr__()")
    print("-" * 50)

    # Normal case
    event = BatchProgressEvent(
        batch_id="test",
        status=BatchStatus.IN_PROGRESS,
        total=100,
        completed=50,
        failed=0,
        percent=50.0,
        rate=10.5,
        eta_seconds=5.0,
        is_complete=False,
        is_failed=False,
        timestamp=datetime.now(),
    )
    repr_str = repr(event)
    assert "50/100" in repr_str
    assert "50.0%" in repr_str
    assert "10.5/s" in repr_str
    assert "5s" in repr_str

    print(f"  Normal: {repr_str}")

    # Infinity ETA case
    event_inf = BatchProgressEvent(
        batch_id="test",
        status=BatchStatus.IN_PROGRESS,
        total=100,
        completed=0,
        failed=0,
        percent=0.0,
        rate=0.0,
        eta_seconds=float('inf'),
        is_complete=False,
        is_failed=False,
        timestamp=datetime.now(),
    )
    repr_inf = repr(event_inf)
    assert "inf" in repr_inf

    print(f"  Infinite ETA: {repr_inf}")
    print("  [PASS] __repr__ formatting correct")
    return True


async def main():
    """Run all V39.9 progress streaming tests."""
    print("=" * 60)
    print("VOYAGE AI V39.9 - BATCH PROGRESS STREAMING TESTS")
    print("=" * 60)

    print("\nNote: These tests validate V39.9 progress streaming features.")
    print("      Signature and calculation tests run locally.\n")

    tests = [
        ("test_batch_progress_metrics_creation", test_batch_progress_metrics_creation),
        ("test_batch_progress_metrics_add_sample", test_batch_progress_metrics_add_sample),
        ("test_batch_progress_metrics_rate_calculation", test_batch_progress_metrics_rate_calculation),
        ("test_batch_progress_metrics_eta_calculation", test_batch_progress_metrics_eta_calculation),
        ("test_batch_progress_event_creation", test_batch_progress_event_creation),
        ("test_batch_progress_event_from_batch_job", test_batch_progress_event_from_batch_job),
        ("test_stream_batch_progress_signature", test_stream_batch_progress_signature),
        ("test_wait_for_batch_completion_with_progress_signature", test_wait_for_batch_completion_with_progress_signature),
        ("test_batch_progress_event_repr", test_batch_progress_event_repr),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            result = await test_fn()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n[SUCCESS] V39.9 PHASE 1 COMPLETE - Progress streaming implemented!")
        print("          - BatchProgressMetrics with rate/ETA calculation")
        print("          - BatchProgressEvent dataclass with factory method")
        print("          - stream_batch_progress() AsyncGenerator")
        print("          - wait_for_batch_completion_with_progress() callback method")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
