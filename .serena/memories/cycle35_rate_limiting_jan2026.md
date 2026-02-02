# Cycle 35: Rate Limiting & Throttling Patterns (January 2026)

## Overview
Production-grade rate limiting for Python APIs, covering algorithms, FastAPI integration, and distributed systems with Redis.

---

## Rate Limiting Algorithm Comparison

| Algorithm | How It Works | Best For | Drawbacks |
|-----------|--------------|----------|-----------|
| **Token Bucket** | Tokens refill at fixed rate; request consumes token | Bursty traffic with average control | Memory per client |
| **Leaky Bucket** | Queue drains at constant rate | Smooth output rate | Delays under load |
| **Fixed Window** | Count resets at interval boundaries | Simple implementation | Edge bursts (2x at boundary) |
| **Sliding Window Log** | Track exact timestamps | Precise limiting | Memory-heavy |
| **Sliding Window Counter** | Weighted previous + current window | Balance of precision/memory | Slight approximation |

### Token Bucket (Preferred for APIs)
```python
import time
from dataclasses import dataclass

@dataclass
class TokenBucket:
    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = None
    last_refill: float = None
    
    def __post_init__(self):
        self.tokens = self.capacity
        self.last_refill = time.monotonic()
    
    def consume(self, tokens: int = 1) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

# Usage
bucket = TokenBucket(capacity=100, refill_rate=10)  # 100 burst, 10/sec sustained
```

---

## SlowAPI for FastAPI (Production Standard)

### Installation
```bash
pip install slowapi
```

### Basic Setup
```python
from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/api/resource")
@limiter.limit("100/minute")
async def get_resource(request: Request):
    return {"data": "value"}
```

### Advanced Key Functions
```python
from fastapi import Depends

# By API key
def get_api_key(request: Request) -> str:
    return request.headers.get("X-API-Key", get_remote_address(request))

# By user ID (authenticated)
async def get_user_id(request: Request) -> str:
    user = await get_current_user(request)
    return str(user.id) if user else get_remote_address(request)

# Tiered limits by subscription
@app.get("/api/premium")
@limiter.limit("1000/minute", key_func=get_api_key)
@limiter.limit("100/minute", key_func=get_remote_address)  # Fallback
async def premium_endpoint(request: Request):
    pass
```

### Rate Limit Headers (RFC 6585 + draft-ietf-httpapi-ratelimit-headers)
```python
from slowapi.middleware import SlowAPIMiddleware

app.add_middleware(SlowAPIMiddleware)

# Response headers:
# X-RateLimit-Limit: 100
# X-RateLimit-Remaining: 95
# X-RateLimit-Reset: 1706234567
# Retry-After: 60 (when limited)
```

---

## Redis Backend for Distributed Rate Limiting

### SlowAPI with Redis
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="redis://localhost:6379",
    strategy="fixed-window",  # or "moving-window"
)
```

### Redis + Lua Atomic Rate Limiting
```python
import redis
from typing import Tuple

RATE_LIMIT_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

local current = redis.call('GET', key)
if current and tonumber(current) >= limit then
    local ttl = redis.call('TTL', key)
    return {0, ttl}
end

current = redis.call('INCR', key)
if current == 1 then
    redis.call('EXPIRE', key, window)
end

return {1, limit - current}
"""

class RedisRateLimiter:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.script = self.redis.register_script(RATE_LIMIT_LUA)
    
    def is_allowed(
        self, 
        key: str, 
        limit: int, 
        window_seconds: int
    ) -> Tuple[bool, int]:
        """Returns (allowed, remaining)"""
        import time
        result = self.script(
            keys=[f"ratelimit:{key}"],
            args=[limit, window_seconds, int(time.time())]
        )
        return bool(result[0]), result[1]
```

### Sliding Window with Redis Sorted Sets
```python
SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local request_id = ARGV[4]

-- Remove old entries
redis.call('ZREMRANGEBYSCORE', key, 0, now - window_ms)

-- Count current window
local count = redis.call('ZCARD', key)

if count < limit then
    redis.call('ZADD', key, now, request_id)
    redis.call('PEXPIRE', key, window_ms)
    return {1, limit - count - 1}
end

return {0, 0}
"""
```

---

## Tiered Rate Limiting

```python
from enum import Enum
from typing import Dict

class Tier(str, Enum):
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"

TIER_LIMITS: Dict[Tier, str] = {
    Tier.FREE: "10/minute",
    Tier.BASIC: "100/minute",
    Tier.PRO: "1000/minute",
    Tier.ENTERPRISE: "10000/minute",
}

def get_tier_limit(request: Request) -> str:
    user = get_current_user(request)
    tier = user.subscription_tier if user else Tier.FREE
    return TIER_LIMITS[tier]

@app.get("/api/data")
@limiter.limit(limit_value=get_tier_limit)
async def get_data(request: Request):
    pass
```

---

## DDoS Protection Patterns

### Multi-Layer Defense
```python
# Layer 1: Connection rate (nginx/HAProxy)
# limit_conn_zone $binary_remote_addr zone=conn:10m;
# limit_conn conn 100;

# Layer 2: Request rate (application)
@limiter.limit("1000/minute")  # General limit
@limiter.limit("10/second")    # Burst protection
async def endpoint(request: Request):
    pass

# Layer 3: Expensive operation limits
@limiter.limit("5/minute")
async def search(request: Request, query: str):
    # CPU-intensive operation
    pass

# Layer 4: Circuit breaker for downstream
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=30)
async def call_external_api():
    pass
```

### Adaptive Rate Limiting
```python
import psutil

def adaptive_limit(request: Request) -> str:
    cpu_percent = psutil.cpu_percent()
    if cpu_percent > 80:
        return "10/minute"  # Heavily throttle
    elif cpu_percent > 60:
        return "50/minute"  # Moderate throttle
    return "200/minute"  # Normal operation

@limiter.limit(limit_value=adaptive_limit)
async def adaptive_endpoint(request: Request):
    pass
```

---

## FastAPI Middleware Approach

```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limiter: RedisRateLimiter):
        super().__init__(app)
        self.limiter = limiter
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        allowed, remaining = self.limiter.is_allowed(
            key=client_ip,
            limit=100,
            window_seconds=60
        )
        
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"error": "Too Many Requests"},
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": "100",
                    "X-RateLimit-Remaining": "0",
                }
            )
        
        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
```

---

## Decision Matrix

| Scenario | Solution |
|----------|----------|
| Single server, simple needs | SlowAPI + in-memory |
| Multi-server, same limits | SlowAPI + Redis |
| Different limits per tier | Dynamic key_func + tier lookup |
| Protect expensive operations | Lower limit + circuit breaker |
| API monetization | Per-API-key + quota tracking |
| DDoS protection | Multi-layer + adaptive + upstream |

---

## Production Checklist

- [ ] Redis cluster for distributed limiting
- [ ] Graceful degradation if Redis unavailable
- [ ] Rate limit headers in responses
- [ ] Monitoring: rejection rate, latency impact
- [ ] Alerting on unusual rejection spikes
- [ ] Client documentation with limits
- [ ] Bypass for health checks / internal services
- [ ] Consider X-Forwarded-For behind proxies

---

## Anti-Patterns

| Anti-Pattern | Problem | Solution |
|--------------|---------|----------|
| In-memory only multi-server | Limits per server, not total | Redis backend |
| Fixed window only | 2x burst at boundaries | Sliding window or token bucket |
| No headers | Clients can't self-regulate | Always include X-RateLimit-* |
| Same limit everywhere | Expensive ops DoS'd | Tiered by endpoint cost |
| No bypass | Monitoring breaks | Allowlist internal IPs |
| Blocking on limit check | Added latency | Async Redis, timeout fallback |

---

*Cycle 35 Complete - Rate Limiting & Throttling Patterns*
