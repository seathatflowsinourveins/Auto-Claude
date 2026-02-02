# V121 Optimization: Circuit Breaker for API Failures

> **Date**: 2026-01-30
> **Priority**: P1 (Reliability)
> **Status**: IMPLEMENTED ✅

---

## Problem Statement

API failures can cascade and cause system-wide degradation:

```python
# ❌ VULNERABLE - No protection against API failures
async def embed(self, text: str) -> EmbeddingResult:
    response = await client.post(...)  # What if this keeps failing?
    # Keeps retrying, overwhelming the failing service
    # Blocking other operations
    # Wasting resources on doomed requests
```

**Impact**:
- Cascading failures when API is down
- Wasted resources on failed requests
- Poor user experience during outages
- No visibility into failure patterns

---

## Solution

### 1. Circuit Breaker Pattern (Netflix Hystrix-style)

```
┌─────────────────────────────────────────────────────┐
│                CIRCUIT BREAKER STATES               │
├─────────────────────────────────────────────────────┤
│                                                     │
│   CLOSED ──────────────────────────────► OPEN      │
│   (normal)     5 consecutive failures    (failing  │
│   ▲                                       fast)    │
│   │                                        │       │
│   │    2 successes      30s timeout       │       │
│   │                         │             │       │
│   └──── HALF_OPEN ◄─────────┘             ▼       │
│         (testing)                                  │
│                                                    │
└─────────────────────────────────────────────────────┘
```

### 2. Integration with OpenAIEmbeddingProvider

```python
# ✅ PROTECTED - Circuit breaker integration (V121)
class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Provider with connection pooling (V118), caching (V120), and circuit breaker (V121)."""

    # V121: Circuit breaker for API resilience (shared across all instances)
    _circuit_breaker: CircuitBreaker = CircuitBreaker(
        failure_threshold=5,       # Open after 5 consecutive failures
        success_threshold=2,       # Close after 2 successes in half-open
        recovery_timeout=30.0,     # Try again after 30 seconds
        half_open_max_calls=3,     # Allow 3 test calls in half-open
    )

    async def embed(self, text: str) -> EmbeddingResult:
        """V120+V121: Generate embedding with caching and circuit breaker."""
        # V120: Check cache first (bypasses circuit breaker)
        cached = _embedding_cache.get(text, self._model)
        if cached is not None:
            return EmbeddingResult(embedding=cached, ...)

        # Cache miss - call API with circuit breaker (V121)
        async with self._circuit_breaker:
            response = await client.post(...)
            if response.status_code != 200:
                raise RuntimeError(f"API error: {response.text}")
            data = response.json()

        # Cache and return result
        ...
```

### 3. Observability

```python
# Get circuit breaker statistics
stats = OpenAIEmbeddingProvider.get_circuit_stats()
print(f"State: {stats['state']}")           # "closed", "open", or "half_open"
print(f"Failure rate: {stats['failure_rate']:.1%}")
print(f"Rejected calls: {stats['rejected_calls']}")
print(f"Time in OPEN state: {stats['time_in_open']:.1f}s")
```

---

## Files Modified (1 Python File)

| File | Class | Change | Status |
|------|-------|--------|--------|
| `platform/core/advanced_memory.py` | `OpenAIEmbeddingProvider` | Added circuit breaker | ✅ Added |

---

## Circuit Breaker Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `failure_threshold` | 5 | Open after 5 failures - enough to detect real outage |
| `success_threshold` | 2 | Close after 2 successes - confirm recovery |
| `recovery_timeout` | 30.0s | Test recovery after 30s - balance between retry and rest |
| `half_open_max_calls` | 3 | Allow 3 test calls - statistical confidence |

---

## Behavior by State

### CLOSED (Normal Operation)
- All requests pass through
- Failures increment counter
- After 5 consecutive failures → OPEN

### OPEN (Failing Fast)
- Requests immediately rejected with `CircuitOpenError`
- No API calls made (resource preservation)
- After 30 seconds → HALF_OPEN

### HALF_OPEN (Testing Recovery)
- Limited requests (3 max) allowed through
- Success → back to CLOSED
- Any failure → back to OPEN

---

## Error Handling

```python
from platform.core.resilience import CircuitOpenError

try:
    result = await provider.embed("test text")
except CircuitOpenError as e:
    # Circuit is open - API is failing
    # Use fallback: cached result, local model, or graceful error
    logger.warning(f"Circuit open: {e}")
    return fallback_result
except RuntimeError as e:
    # API error (will count toward circuit breaker)
    logger.error(f"API error: {e}")
    raise
```

---

## Integration with V118-V120

```
Request Flow:
┌──────────────┐    ┌─────────────┐    ┌─────────────────┐    ┌─────────────┐
│   Request    │───►│ V120 Cache  │───►│ V121 Circuit    │───►│ V118 Pool   │
│              │    │   Check     │    │    Breaker      │    │   Client    │
└──────────────┘    └──────┬──────┘    └────────┬────────┘    └──────┬──────┘
                          │                    │                    │
                   Cache Hit?             Circuit OK?          API Call
                     │    │                 │    │                │
                  Yes│    │No            Yes│    │No              │
                     ▼    │                 │    ▼                ▼
                 Return   └─────────────────┘  Fail Fast     Response
```

**Benefits of Combined Optimizations**:
- V120 Cache reduces circuit breaker load (cache hits bypass breaker)
- V121 Circuit Breaker prevents cascading failures
- V118 Connection Pool efficient when circuit allows traffic

---

## Quantified Expected Gains

### Reliability Improvements

| Metric | Before | After |
|--------|--------|-------|
| Cascade failure risk | High | Eliminated |
| Resource waste during outage | 100% | ~0% (fail fast) |
| Recovery detection | None | Automatic (30s test) |
| Failure visibility | Limited | Full stats |

### Performance During Outages

| Scenario | Before | After |
|----------|--------|-------|
| Response time (API down) | ~30s (timeout) | ~0ms (fail fast) |
| System load during outage | High (retries) | Minimal |
| Partial recovery handling | None | Graceful (half-open) |

---

## Monitoring

### Recommended Alerts

```python
# Alert if circuit opens
if stats['state'] == 'open':
    alert("Embedding API circuit open - API may be down")

# Alert on high failure rate
if stats['failure_rate'] > 0.2:
    alert(f"High embedding API failure rate: {stats['failure_rate']:.1%}")

# Track time in open state
if stats['time_in_open'] > 300:  # 5 minutes
    escalate("Embedding API down for 5+ minutes")
```

### Dashboard Metrics

```python
{
    "circuit_state": stats['state'],
    "calls_total": stats['total_calls'],
    "calls_success": stats['successful_calls'],
    "calls_failed": stats['failed_calls'],
    "calls_rejected": stats['rejected_calls'],
    "failure_rate": stats['failure_rate'],
    "time_in_open_seconds": stats['time_in_open'],
}
```

---

## Verification

### Test Command
```bash
cd "Z:\insider\AUTO CLAUDE\unleash"
pytest platform/tests/test_v121_circuit_breaker.py -v
```

### Manual Testing
```python
import asyncio
from platform.core.advanced_memory import OpenAIEmbeddingProvider

async def test_circuit():
    provider = OpenAIEmbeddingProvider(api_key="invalid-key")  # Will fail

    for i in range(10):
        try:
            await provider.embed(f"test {i}")
        except Exception as e:
            print(f"Attempt {i}: {type(e).__name__}: {e}")
        print(f"Stats: {provider.get_circuit_stats()}")

asyncio.run(test_circuit())
```

---

## Related Optimizations

- V115: Letta Cloud initialization fix
- V116: Sleep-time agent configuration fix
- V117: Deprecated token= parameter fix
- V118: Connection pooling for HTTP clients
- V119: Async batch processing
- V120: Embedding cache with TTL
- **V121**: Circuit breaker for API failures (this document)

---

## Future Improvements

1. **V122**: Metrics collection for observability (integrate with Langfuse)
2. **Adaptive Thresholds**: Adjust based on historical failure rates
3. **Per-Model Breakers**: Separate breakers for different models
4. **Fallback Providers**: Auto-switch to backup API on circuit open

---

*Optimization completed 2026-01-30 as part of autonomous system optimization iteration.*
