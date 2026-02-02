# Cycle 47: API Rate Limiting & Throttling Patterns (January 2026)

## Official Documentation Sources
- **Redis Official**: https://redis.io/glossary/rate-limiting/
- **SlowAPI Official**: https://slowapi.readthedocs.io/
- **FastAPI Middleware**: https://fastapi.tiangolo.com/tutorial/middleware/

---

## Rate Limiting Algorithms (Redis Official)

### 1. Fixed Window Counter
Simplest approach - count requests in fixed time windows.

```python
import redis
from datetime import datetime

async def fixed_window_limit(
    redis_client: redis.Redis,
    key: str,
    limit: int,
    window_seconds: int = 60
) -> tuple[bool, int]:
    """Fixed window rate limiting from Redis official docs."""
    window = int(datetime.now().timestamp()) // window_seconds
    rate_key = f"ratelimit:{key}:{window}"
    
    # Atomic increment + expire in pipeline
    pipe = redis_client.pipeline()
    pipe.incr(rate_key)
    pipe.expire(rate_key, window_seconds)
    results = pipe.execute()
    
    current_count = results[0]
    remaining = max(0, limit - current_count)
    
    return current_count <= limit, remaining
```

**Pros**: Simple, low memory
**Cons**: Burst at window boundaries (2x limit possible)

### 2. Sliding Window Log
Stores timestamp of each request - most accurate but memory intensive.

```python
import time

async def sliding_window_log(
    redis_client: redis.Redis,
    key: str,
    limit: int,
    window_seconds: int = 60
) -> tuple[bool, int]:
    """Sliding window with request log (Redis ZSET)."""
    now = time.time()
    window_start = now - window_seconds
    rate_key = f"ratelimit:log:{key}"
    
    pipe = redis_client.pipeline()
    # Remove old entries
    pipe.zremrangebyscore(rate_key, 0, window_start)
    # Add current request
    pipe.zadd(rate_key, {str(now): now})
    # Count requests in window
    pipe.zcard(rate_key)
    # Set expiry
    pipe.expire(rate_key, window_seconds)
    results = pipe.execute()
    
    current_count = results[2]
    return current_count <= limit, max(0, limit - current_count)
```

**Pros**: Precise, no boundary issues
**Cons**: O(n) memory per user

### 3. Sliding Window Counter (Recommended)
Hybrid approach - weighted average of current and previous windows.

```python
async def sliding_window_counter(
    redis_client: redis.Redis,
    key: str,
    limit: int,
    window_seconds: int = 60
) -> tuple[bool, int]:
    """Sliding window counter - best balance of accuracy and efficiency."""
    now = time.time()
    current_window = int(now) // window_seconds
    previous_window = current_window - 1
    
    # Position in current window (0.0 to 1.0)
    window_position = (now % window_seconds) / window_seconds
    
    current_key = f"ratelimit:{key}:{current_window}"
    previous_key = f"ratelimit:{key}:{previous_window}"
    
    pipe = redis_client.pipeline()
    pipe.get(previous_key)
    pipe.incr(current_key)
    pipe.expire(current_key, window_seconds * 2)
    results = pipe.execute()
    
    previous_count = int(results[0] or 0)
    current_count = results[1]
    
    # Weighted count: previous * (1 - position) + current
    weighted_count = previous_count * (1 - window_position) + current_count
    
    return weighted_count <= limit, max(0, int(limit - weighted_count))
```

**Pros**: Smooth rate limiting, O(1) memory
**Cons**: Slightly less accurate than log

### 4. Token Bucket
Tokens regenerate over time - allows controlled bursts.

```python
import time
import redis

class TokenBucket:
    """Token bucket from Redis official patterns."""
    
    def __init__(
        self,
        redis_client: redis.Redis,
        key: str,
        capacity: int,
        refill_rate: float  # tokens per second
    ):
        self.redis = redis_client
        self.key = f"bucket:{key}"
        self.capacity = capacity
        self.refill_rate = refill_rate
    
    async def consume(self, tokens: int = 1) -> tuple[bool, float]:
        """Try to consume tokens. Returns (allowed, tokens_remaining)."""
        now = time.time()
        
        # Lua script for atomic token bucket
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local requested = tonumber(ARGV[4])
        
        local bucket = redis.call('HMGET', key, 'tokens', 'last_update')
        local tokens = tonumber(bucket[1]) or capacity
        local last_update = tonumber(bucket[2]) or now
        
        -- Refill tokens
        local elapsed = now - last_update
        tokens = math.min(capacity, tokens + elapsed * refill_rate)
        
        local allowed = 0
        if tokens >= requested then
            tokens = tokens - requested
            allowed = 1
        end
        
        redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
        redis.call('EXPIRE', key, math.ceil(capacity / refill_rate) * 2)
        
        return {allowed, tokens}
        """
        
        result = self.redis.eval(
            lua_script, 1, self.key,
            self.capacity, self.refill_rate, now, tokens
        )
        return bool(result[0]), float(result[1])
```

**Pros**: Allows bursts, smooth long-term rate
**Cons**: More complex, requires Lua for atomicity

### 5. Leaky Bucket
Requests processed at constant rate - queue overflow rejected.

```python
class LeakyBucket:
    """Leaky bucket - constant output rate."""
    
    def __init__(
        self,
        redis_client: redis.Redis,
        key: str,
        capacity: int,
        leak_rate: float  # requests per second
    ):
        self.redis = redis_client
        self.key = f"leaky:{key}"
        self.capacity = capacity
        self.leak_rate = leak_rate
    
    async def add(self) -> tuple[bool, int]:
        """Add request to bucket. Returns (allowed, queue_size)."""
        now = time.time()
        
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local leak_rate = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        
        local bucket = redis.call('HMGET', key, 'water', 'last_leak')
        local water = tonumber(bucket[1]) or 0
        local last_leak = tonumber(bucket[2]) or now
        
        -- Leak water
        local elapsed = now - last_leak
        water = math.max(0, water - elapsed * leak_rate)
        
        local allowed = 0
        if water < capacity then
            water = water + 1
            allowed = 1
        end
        
        redis.call('HMSET', key, 'water', water, 'last_leak', now)
        redis.call('EXPIRE', key, math.ceil(capacity / leak_rate) * 2)
        
        return {allowed, water}
        """
        
        result = self.redis.eval(
            lua_script, 1, self.key,
            self.capacity, self.leak_rate, now
        )
        return bool(result[0]), int(result[1])
```

**Pros**: Guaranteed constant processing rate
**Cons**: Adds latency (queue), complex

---

## SlowAPI Integration (Official Docs)

### Basic Setup with FastAPI

```python
from fastapi import FastAPI, Request, Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Initialize limiter with key function
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Optional: Add as middleware for global limiting
app.add_middleware(SlowAPIMiddleware)
```

### Decorator-Based Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/api/search")
@limiter.limit("5/minute")  # 5 requests per minute
async def search(request: Request, q: str):
    return {"results": [...]}

@app.get("/api/export")
@limiter.limit("1/hour")  # Expensive operation
async def export_data(request: Request):
    return {"data": [...]}

@app.post("/api/upload")
@limiter.limit("10/minute", key_func=lambda req: req.state.user_id)
async def upload(request: Request):
    """Custom key function for user-based limiting."""
    return {"status": "uploaded"}
```

### Multiple Rate Limits

```python
@app.get("/api/resource")
@limiter.limit("100/minute")  # Short-term burst protection
@limiter.limit("1000/hour")   # Long-term abuse prevention
@limiter.limit("10000/day")   # Daily quota
async def get_resource(request: Request):
    return {"data": "..."}
```

### Dynamic Rate Limits

```python
def get_rate_limit_from_user(request: Request) -> str:
    """Dynamic limits based on user tier."""
    user = request.state.user
    limits = {
        "free": "10/minute",
        "pro": "100/minute",
        "enterprise": "1000/minute"
    }
    return limits.get(user.tier, "10/minute")

@app.get("/api/data")
@limiter.limit(get_rate_limit_from_user)
async def get_data(request: Request):
    return {"data": "..."}
```

### Custom Key Functions

```python
from slowapi.util import get_remote_address

def get_api_key(request: Request) -> str:
    """Rate limit by API key."""
    return request.headers.get("X-API-Key", get_remote_address(request))

def get_user_id(request: Request) -> str:
    """Rate limit by authenticated user."""
    if hasattr(request.state, "user"):
        return f"user:{request.state.user.id}"
    return get_remote_address(request)

def get_endpoint_key(request: Request) -> str:
    """Rate limit by endpoint + IP."""
    return f"{request.url.path}:{get_remote_address(request)}"
```

### Redis Backend for Distributed Systems

```python
from slowapi import Limiter
from slowapi.util import get_remote_address
import redis

# Production: Use Redis for distributed rate limiting
redis_client = redis.from_url("redis://localhost:6379")

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="redis://localhost:6379",
    # Or pass client directly:
    # storage=redis_client
)
```

---

## Production Rate Limiting Middleware

### Complete FastAPI Implementation

```python
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import redis.asyncio as redis
import time
from typing import Optional
import hashlib

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Production-grade distributed rate limiting."""
    
    def __init__(
        self,
        app,
        redis_url: str = "redis://localhost:6379",
        default_limit: int = 100,
        window_seconds: int = 60,
        key_prefix: str = "ratelimit"
    ):
        super().__init__(app)
        self.redis = redis.from_url(redis_url)
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix
        
        # Endpoint-specific limits
        self.endpoint_limits = {
            "/api/search": (30, 60),      # 30/min
            "/api/export": (5, 3600),     # 5/hour
            "/api/upload": (10, 60),      # 10/min
            "/api/webhook": (1000, 60),   # 1000/min (high-volume)
        }
    
    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier."""
        # Priority: API Key > User ID > IP
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"apikey:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        
        if hasattr(request.state, "user"):
            return f"user:{request.state.user.id}"
        
        # Fallback to IP (handle proxies)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        return f"ip:{request.client.host}"
    
    def _get_limits(self, path: str) -> tuple[int, int]:
        """Get rate limit for endpoint."""
        # Check exact match first
        if path in self.endpoint_limits:
            return self.endpoint_limits[path]
        
        # Check prefix matches
        for endpoint, limits in self.endpoint_limits.items():
            if path.startswith(endpoint):
                return limits
        
        return self.default_limit, self.window_seconds
    
    async def dispatch(self, request: Request, call_next):
        # Skip health checks and static files
        if request.url.path in ["/health", "/metrics", "/favicon.ico"]:
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        limit, window = self._get_limits(request.url.path)
        
        # Sliding window counter
        now = time.time()
        current_window = int(now) // window
        window_position = (now % window) / window
        
        current_key = f"{self.key_prefix}:{client_id}:{request.url.path}:{current_window}"
        previous_key = f"{self.key_prefix}:{client_id}:{request.url.path}:{current_window - 1}"
        
        pipe = self.redis.pipeline()
        pipe.get(previous_key)
        pipe.incr(current_key)
        pipe.expire(current_key, window * 2)
        results = await pipe.execute()
        
        previous_count = int(results[0] or 0)
        current_count = results[1]
        weighted_count = previous_count * (1 - window_position) + current_count
        
        remaining = max(0, int(limit - weighted_count))
        reset_time = (current_window + 1) * window
        
        # Set rate limit headers
        headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(reset_time)),
            "X-RateLimit-Window": str(window),
        }
        
        if weighted_count > limit:
            retry_after = reset_time - now
            headers["Retry-After"] = str(int(retry_after))
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit of {limit} requests per {window}s exceeded",
                    "retry_after": int(retry_after)
                },
                headers=headers
            )
        
        response = await call_next(request)
        
        # Add headers to successful response
        for key, value in headers.items():
            response.headers[key] = value
        
        return response


# Usage
app = FastAPI()
app.add_middleware(
    RateLimitMiddleware,
    redis_url="redis://localhost:6379",
    default_limit=100,
    window_seconds=60
)
```

---

## Rate Limiting Strategies by Use Case

### 1. API Gateway Level
```yaml
# Kong/nginx rate limiting config
rate_limiting:
  policy: redis
  redis_host: redis-cluster
  limits:
    - second: 10
    - minute: 100
    - hour: 1000
```

### 2. User Tier Limits
```python
TIER_LIMITS = {
    "free": {"requests": 100, "window": 3600},      # 100/hour
    "starter": {"requests": 1000, "window": 3600},  # 1000/hour
    "pro": {"requests": 10000, "window": 3600},     # 10000/hour
    "enterprise": {"requests": 100000, "window": 3600},  # Unlimited effectively
}
```

### 3. Endpoint Criticality
```python
ENDPOINT_LIMITS = {
    # Read operations - higher limits
    "GET /api/users": (1000, 60),
    "GET /api/products": (500, 60),
    
    # Write operations - lower limits
    "POST /api/orders": (10, 60),
    "POST /api/payments": (5, 60),
    
    # Expensive operations - very low
    "POST /api/reports/generate": (1, 300),
    "POST /api/export": (2, 3600),
}
```

---

## Response Headers (Standard)

```
X-RateLimit-Limit: 100          # Max requests allowed
X-RateLimit-Remaining: 45       # Requests remaining
X-RateLimit-Reset: 1706234567   # Unix timestamp when limit resets
Retry-After: 30                 # Seconds until retry (on 429)
```

---

## Algorithm Selection Guide

| Algorithm | Memory | Accuracy | Burst | Use Case |
|-----------|--------|----------|-------|----------|
| Fixed Window | O(1) | Low | 2x possible | Simple APIs |
| Sliding Log | O(n) | Perfect | None | High-value endpoints |
| Sliding Counter | O(1) | High | Minimal | **Recommended default** |
| Token Bucket | O(1) | N/A | Controlled | APIs with burst needs |
| Leaky Bucket | O(1) | N/A | Queue | Constant-rate processing |

---

## Quick Reference

```python
# SlowAPI Basic
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.get("/api")
@limiter.limit("100/minute")
async def api(request: Request): ...

# Redis Sliding Window (Production)
current = redis.incr(f"rate:{key}:{window}")
redis.expire(f"rate:{key}:{window}", window_seconds)
allowed = current <= limit

# Token Bucket (Burst-Tolerant)
bucket = TokenBucket(redis, "api", capacity=100, refill_rate=1.67)
allowed, remaining = await bucket.consume(1)
```

---

*Official sources: Redis.io, SlowAPI docs, FastAPI middleware patterns*
*Cycle 47 - January 2026*
