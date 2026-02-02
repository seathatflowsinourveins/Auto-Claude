# Code Patterns

## Async Context Manager
```python
async def __aenter__(self):
    await self.connect()
    return self

async def __aexit__(self, *args):
    await self.disconnect()
```

## Circuit Breaker
```python
@circuit_breaker(failure_threshold=5, reset_timeout=60)
async def call_external_service():
    ...
```

## Retry with Backoff
```python
@with_retry(max_attempts=3, backoff_factor=2.0)
async def flaky_operation():
    ...
```

## Structured Logging
```python
logger.info("operation_completed",
    operation="create_session",
    duration_ms=elapsed,
    result="success")
```
