# Voyage AI V39.9 Implementation Plan
## Batch Progress Streaming & WebSocket Integration

**Status**: ✅ PHASE 1 IMPLEMENTED (2026-01-26)
**Target**: Real-time batch job progress visibility
**Dependencies**: V39.8 Batch API implementation
**Priority**: LOW (quality-of-life optimization)
**Phase 1**: ✅ COMPLETE (core streaming + callback)

---

## Gap Analysis

| Current Limitation | V39.9 Solution |
|-------------------|----------------|
| Batch jobs require polling | AsyncGenerator streaming |
| No progress callbacks during wait | Event-driven progress updates |
| Manual status checking | Automatic progress notifications |
| No WebSocket support | Optional WebSocket streaming |

---

## V39.9 Features

### 1. Batch Progress Streaming (AsyncGenerator)

```python
async def stream_batch_progress(
    self,
    batch_id: str,
    poll_interval: float = 5.0,
    include_metrics: bool = True,
) -> AsyncGenerator[BatchProgressEvent, None]:
    """
    Stream batch progress updates using async generator.

    V39.9 Feature: Real-time progress visibility.

    Args:
        batch_id: The batch job ID to monitor
        poll_interval: Seconds between status checks
        include_metrics: Include processing rate, ETA

    Yields:
        BatchProgressEvent with progress, rate, ETA

    Example:
        async for event in layer.stream_batch_progress(job.id):
            print(f"Progress: {event.completed}/{event.total} ({event.percent:.1f}%)")
            print(f"Rate: {event.rate:.1f} embeds/sec, ETA: {event.eta_seconds:.0f}s")
            if event.is_complete:
                break
    """
```

### 2. Progress Event Dataclass

```python
@dataclass
class BatchProgressEvent:
    batch_id: str
    status: BatchStatus
    total: int
    completed: int
    failed: int
    percent: float  # 0.0 to 100.0
    rate: float  # embeddings per second
    eta_seconds: float  # estimated time remaining
    is_complete: bool
    is_failed: bool
    timestamp: datetime
    error_message: Optional[str] = None
```

### 3. Callback-Based Progress

```python
async def wait_for_batch_completion_with_progress(
    self,
    batch_id: str,
    on_progress: Callable[[BatchProgressEvent], Awaitable[None]],
    poll_interval: float = 5.0,
    max_wait: int = 43200,  # 12 hours
) -> BatchJob:
    """
    Wait for batch completion with progress callbacks.

    V39.9 Feature: Progress-aware batch waiting.

    Args:
        batch_id: The batch job ID
        on_progress: Async callback for each progress update
        poll_interval: Seconds between status checks
        max_wait: Maximum seconds to wait

    Example:
        async def log_progress(event: BatchProgressEvent):
            print(f"[{event.percent:.1f}%] {event.completed}/{event.total}")

        job = await layer.wait_for_batch_completion_with_progress(
            batch_id=job.id,
            on_progress=log_progress,
        )
    """
```

### 4. Progress Metrics Calculation

```python
@dataclass
class BatchProgressMetrics:
    """Metrics for batch progress tracking."""
    start_time: datetime
    samples: list[tuple[datetime, int]]  # (timestamp, completed)
    window_size: int = 10  # Moving average window

    def calculate_rate(self) -> float:
        """Calculate current processing rate (items/sec)."""
        ...

    def calculate_eta(self, total: int, completed: int) -> float:
        """Estimate remaining time in seconds."""
        ...
```

---

## State of Witness Use Cases

### 1. Session Recording Batch Monitoring
```python
# Monitor large session recording embedding
async for event in layer.stream_batch_progress(job.id):
    # Update TouchDesigner progress display
    await update_td_progress(event.percent, event.eta_seconds)
    if event.is_complete:
        await notify_td_complete()
```

### 2. Gesture Library Initialization Progress
```python
# Show progress during gesture library batch init
library = GestureEmbeddingLibrary(layer)
job_id = await library.initialize(use_batch=True, batch_wait=False)

async for event in layer.stream_batch_progress(job_id):
    print(f"Loading gestures: {event.completed}/15 ({event.percent:.0f}%)")
```

### 3. Archetype Training Progress
```python
# Track archetype training batch
async def on_training_progress(event: BatchProgressEvent):
    if event.percent % 10 == 0:  # Log every 10%
        logger.info(f"Training: {event.percent:.0f}% complete, ETA: {event.eta_seconds:.0f}s")

await layer.wait_for_batch_completion_with_progress(
    batch_id=training_job.id,
    on_progress=on_training_progress,
)
```

---

## Implementation Order

### Phase 1: Core Streaming (Priority: HIGH) ✅ COMPLETE
1. ✅ Implement `BatchProgressEvent` dataclass
2. ✅ Implement `BatchProgressMetrics` for rate/ETA calculation
3. ✅ Implement `stream_batch_progress()` AsyncGenerator
4. ✅ Add tests for streaming progress

### Phase 2: Callback Integration (Priority: MEDIUM) ✅ COMPLETE
5. ✅ Implement `wait_for_batch_completion_with_progress()`
6. Add progress hooks to GestureEmbeddingLibrary batch mode (deferred)
7. ✅ Add tests for callback-based progress

### Phase 3: Optional WebSocket (Priority: LOW)
8. Research Voyage AI WebSocket capabilities
9. Implement WebSocket progress streaming (if API supports)
10. Add fallback to polling if WebSocket unavailable

---

## Cost Analysis

V39.9 has minimal cost impact:
- No additional API calls (uses existing status polling)
- Configurable poll interval (default 5s)
- Optional metrics calculation (CPU only)

---

## Success Criteria

- [x] `BatchProgressEvent` dataclass implemented ✅ Lines 763-834
- [x] `stream_batch_progress()` AsyncGenerator working ✅ Lines 5333-5384
- [x] Rate and ETA calculations accurate ✅ BatchProgressMetrics Lines 688-761
- [x] `wait_for_batch_completion_with_progress()` implemented ✅ Lines 5386-5440
- [x] Tests for all progress features ✅ 9/9 tests passing
- [ ] Documentation updated (feature summary pending)

---

## Implementation Details

### Phase 1-2: Core Streaming & Callbacks

**Dataclasses added to EmbeddingLayer:**
- `BatchProgressMetrics` (lines 688-761) - Rate/ETA calculation with sliding window
- `BatchProgressEvent` (lines 763-834) - Progress event with factory method

**Methods added to EmbeddingLayer:**
- `stream_batch_progress(batch_id, poll_interval, include_metrics)` - AsyncGenerator for real-time progress
- `wait_for_batch_completion_with_progress(batch_id, on_progress, poll_interval, max_wait)` - Callback-based waiting

**Tests:**
- `tests/voyage_v39_9_progress_test.py` - 9 tests all passing

---

Document Version: 1.1
Created: 2026-01-26
Updated: 2026-01-26
Author: Claude (Ralph Loop V39.9 Planning)
Phase 1-2: Claude (Ralph Loop V39.9 Implementation)
