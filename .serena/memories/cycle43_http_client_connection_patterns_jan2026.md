# Cycle 43: HTTP Client & Connection Patterns (January 25, 2026)

## Overview

HTTP client configuration is often overlooked but critically impacts production reliability. This memory covers httpx (the 2026 standard), connection pooling, timeout strategies, retry patterns, and performance optimization.

## HTTP Client Selection (2026)

| Library | Best For | Async | HTTP/2 | Performance |
|---------|----------|-------|--------|-------------|
| **httpx** | Modern apps, FastAPI | Yes + Sync | Yes | 5-10x faster than requests |
| **aiohttp** | High-concurrency (10k+) | Yes only | No (ext) | Best raw throughput |
| **requests** | Simple scripts | No | No | Baseline |

**2026 Recommendation**: httpx for new projects (sync+async, HTTP/2, modern API).

## httpx Production Setup

### Basic AsyncClient Pattern
```python
import httpx

# CRITICAL: Reuse client across requests (connection pooling)
async with httpx.AsyncClient() as client:
    response = await client.get("https://api.example.com/data")
    
# For application lifetime (FastAPI)
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, connect=5.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    yield
    await app.state.http_client.aclose()
```

### Timeout Configuration (Granular)
```python
# httpx.Timeout(timeout, *, connect, read, write, pool)
timeout = httpx.Timeout(
    10.0,           # Total timeout
    connect=5.0,    # TCP connection
    read=10.0,      # Receive response
    write=5.0,      # Send request body
    pool=5.0,       # Wait for available connection
)

client = httpx.AsyncClient(timeout=timeout)
```

### Connection Limits
```python
limits = httpx.Limits(
    max_connections=100,           # Total pool size
    max_keepalive_connections=20,  # Keep-alive pool
    keepalive_expiry=30.0,         # Seconds before closing idle
)

client = httpx.AsyncClient(limits=limits)
```

### HTTP/2 Support
```python
# HTTP/2 multiplexing - single connection, multiple streams
client = httpx.AsyncClient(http2=True)

# Benefits:
# - Single TCP connection per host
# - Header compression (HPACK)
# - Stream prioritization
# - ~30% latency reduction for multiple requests
```

## Retry Patterns with Tenacity

### Basic Retry with Exponential Backoff
```python
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log
)
import httpx
import logging

logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def fetch_with_retry(client: httpx.AsyncClient, url: str) -> httpx.Response:
    response = await client.get(url)
    response.raise_for_status()
    return response
```

### Retry on Specific Status Codes
```python
from tenacity import retry_if_result

def is_retryable_status(response: httpx.Response) -> bool:
    return response.status_code in {429, 500, 502, 503, 504}

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_result(is_retryable_status),
)
async def fetch_with_status_retry(client: httpx.AsyncClient, url: str) -> httpx.Response:
    return await client.get(url)
```

### Rate Limit Handling (429)
```python
import asyncio

async def fetch_with_rate_limit_handling(
    client: httpx.AsyncClient, 
    url: str,
    max_retries: int = 3
) -> httpx.Response:
    for attempt in range(max_retries):
        response = await client.get(url)
        
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            await asyncio.sleep(retry_after)
            continue
            
        response.raise_for_status()
        return response
    
    raise httpx.HTTPStatusError("Rate limit exceeded", request=response.request, response=response)
```

## Circuit Breaker Pattern

### Using pybreaker with httpx
```python
import pybreaker
import httpx

# Circuit breaker: CLOSED → OPEN after 5 failures
# OPEN → HALF-OPEN after 30 seconds
breaker = pybreaker.CircuitBreaker(
    fail_max=5,
    reset_timeout=30,
    exclude=[httpx.HTTPStatusError],  # Don't trip on 4xx
)

@breaker
async def protected_request(client: httpx.AsyncClient, url: str) -> httpx.Response:
    response = await client.get(url, timeout=5.0)
    response.raise_for_status()
    return response

# Usage with fallback
async def fetch_with_fallback(client: httpx.AsyncClient, url: str):
    try:
        return await protected_request(client, url)
    except pybreaker.CircuitBreakerError:
        return get_cached_response(url)  # Graceful degradation
```

## Performance Optimization

### uvloop for 2-4x Faster Event Loop
```python
import uvloop
uvloop.install()  # Before any asyncio code

# Or in Python 3.12+
import asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
```

### Connection Reuse Pattern
```python
# BAD: New connection per request
async def bad_fetch(url: str):
    async with httpx.AsyncClient() as client:
        return await client.get(url)

# GOOD: Reuse client (connection pool)
class APIClient:
    def __init__(self):
        self._client: httpx.AsyncClient | None = None
    
    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_connections=100),
                http2=True,
            )
        return self._client
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
```

### Concurrent Requests with Semaphore
```python
import asyncio

async def fetch_all(urls: list[str], max_concurrent: int = 10) -> list[httpx.Response]:
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_one(client: httpx.AsyncClient, url: str) -> httpx.Response:
        async with semaphore:
            return await client.get(url)
    
    async with httpx.AsyncClient() as client:
        tasks = [fetch_one(client, url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)
```

## aiohttp for Extreme Concurrency

When httpx isn't enough (10k+ concurrent connections):

```python
import aiohttp

# aiohttp handles massive concurrency better
connector = aiohttp.TCPConnector(
    limit=1000,              # Total connections
    limit_per_host=100,      # Per-host limit
    ttl_dns_cache=300,       # DNS cache TTL
    enable_cleanup_closed=True,
)

async with aiohttp.ClientSession(connector=connector) as session:
    async with session.get(url) as response:
        data = await response.json()
```

**When to use aiohttp over httpx**:
- 10,000+ concurrent connections
- WebSocket-heavy applications
- Maximum raw throughput needed
- Don't need sync API or HTTP/2

## FastAPI Integration Pattern

```python
from fastapi import FastAPI, Depends, Request
from contextlib import asynccontextmanager
import httpx

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create shared HTTP client
    app.state.http_client = httpx.AsyncClient(
        base_url="https://api.external.com",
        timeout=httpx.Timeout(30.0, connect=5.0),
        limits=httpx.Limits(max_connections=50),
        http2=True,
    )
    yield
    # Shutdown: Close client
    await app.state.http_client.aclose()

app = FastAPI(lifespan=lifespan)

def get_http_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.http_client

@app.get("/proxy")
async def proxy_endpoint(client: httpx.AsyncClient = Depends(get_http_client)):
    response = await client.get("/external-data")
    return response.json()
```

## Timeout Strategy Matrix

| Scenario | Connect | Read | Write | Total |
|----------|---------|------|-------|-------|
| Fast API | 2s | 5s | 2s | 10s |
| File Upload | 5s | 60s | 60s | 120s |
| Streaming | 5s | None | 5s | None |
| Internal Service | 1s | 3s | 1s | 5s |

## Anti-Patterns to Avoid

1. **Creating client per request** → Connection overhead, socket exhaustion
2. **No timeout configuration** → Requests hang forever
3. **Sync client in async code** → Blocks event loop
4. **Ignoring connection limits** → Socket exhaustion under load
5. **Not closing client** → Resource leaks
6. **Retry without backoff** → Thundering herd, rate limiting
7. **Retry on all errors** → Retrying client errors (4xx) wastes resources

## Key Libraries

| Library | Purpose |
|---------|---------|
| **httpx** | Modern async HTTP client |
| **tenacity** | Retry with backoff |
| **pybreaker** | Circuit breaker |
| **uvloop** | Fast event loop |
| **aiohttp** | High-concurrency alternative |

---

*Research Date: January 25, 2026*
*Sources: httpx docs, Medium articles on uvloop+httpx, aiohttp vs httpx comparisons, tenacity docs*
