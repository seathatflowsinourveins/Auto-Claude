# Cycle 20: Caching Strategies & Patterns (January 2026)

## Research Focus
Production caching patterns for Python systems - from in-memory memoization to distributed Redis architectures.

---

## 1. In-Memory Caching (functools)

### lru_cache - Pure Function Memoization
```python
from functools import lru_cache, cache

# Bounded LRU cache (evicts least-recently-used)
@lru_cache(maxsize=1024)
def expensive_computation(n: int) -> int:
    return sum(range(n))

# Unbounded cache (Python 3.9+) - simpler, faster
@cache
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Cache info inspection
print(fibonacci.cache_info())
# CacheInfo(hits=97, misses=100, maxsize=None, currsize=100)

# Clear cache
fibonacci.cache_clear()
```

### Key Constraints
- **Only hashable arguments** - no lists, dicts, sets
- **Memory-bound** - lives in process memory
- **Not process-safe** - each worker has own cache
- **No TTL** - entries never expire automatically

### typed=True for Type Distinction
```python
@lru_cache(maxsize=128, typed=True)
def process(value):
    return value * 2

# These are cached separately:
process(3)    # int
process(3.0)  # float
```

---

## 2. Persistent Caching (diskcache)

### SQLite-Backed Cache
```python
from diskcache import Cache, FanoutCache

# Simple disk cache
cache = Cache('/tmp/mycache')
cache.set('key', {'data': 'value'}, expire=3600)
value = cache.get('key', default=None)

# Decorator pattern
@cache.memoize(expire=300)
def expensive_query(user_id: int) -> dict:
    return database.get_user(user_id)

# FanoutCache for high concurrency
cache = FanoutCache('/tmp/mycache', shards=8)
```

### Advantages
- **Survives restarts** - SQLite persistence
- **Cross-process safe** - file locking
- **TTL support** - automatic expiration
- **Size limits** - configurable eviction

---

## 3. Redis Caching Patterns

### Pattern 1: Cache-Aside (Lazy Loading)
```python
import redis
import json

r = redis.Redis(decode_responses=True)

def get_user(user_id: str) -> dict:
    # Try cache first
    cached = r.get(f"user:{user_id}")
    if cached:
        return json.loads(cached)
    
    # Cache miss - fetch from DB
    user = database.get_user(user_id)
    
    # Populate cache with TTL
    r.setex(f"user:{user_id}", 3600, json.dumps(user))
    return user
```

**Pros**: Only caches what's needed, simple  
**Cons**: Cache miss penalty, potential stale data

### Pattern 2: Write-Through
```python
def update_user(user_id: str, data: dict) -> dict:
    # Update DB first
    user = database.update_user(user_id, data)
    
    # Immediately update cache
    r.setex(f"user:{user_id}", 3600, json.dumps(user))
    return user
```

**Pros**: Cache always consistent with DB  
**Cons**: Write latency increased

### Pattern 3: Write-Behind (Write-Back)
```python
import asyncio
from collections import deque

write_buffer = deque(maxsize=1000)

async def update_user_async(user_id: str, data: dict):
    # Update cache immediately
    r.setex(f"user:{user_id}", 3600, json.dumps(data))
    
    # Queue for async DB write
    write_buffer.append((user_id, data))

async def flush_writes():
    while True:
        await asyncio.sleep(1)  # Batch every second
        batch = []
        while write_buffer:
            batch.append(write_buffer.popleft())
        if batch:
            await database.bulk_update(batch)
```

**Pros**: Fast writes, reduced DB load  
**Cons**: Data loss risk, complexity

### Pattern 4: Read-Through
```python
# Typically implemented via Redis modules or proxy
# Application only talks to cache, cache handles DB

# Conceptual implementation
class ReadThroughCache:
    def __init__(self, redis_client, db_loader):
        self.r = redis_client
        self.loader = db_loader
    
    def get(self, key: str) -> dict:
        cached = self.r.get(key)
        if cached:
            return json.loads(cached)
        
        # Cache loads from DB automatically
        value = self.loader(key)
        self.r.setex(key, 3600, json.dumps(value))
        return value
```

---

## 4. Cache Invalidation Strategies

### TTL-Based (Time-To-Live)
```python
# Simple but may serve stale data
r.setex("key", 300, "value")  # 5 minute TTL

# Sliding expiration - reset on access
def get_with_sliding_ttl(key: str, ttl: int = 300):
    value = r.get(key)
    if value:
        r.expire(key, ttl)  # Reset TTL
    return value
```

### Event-Based Invalidation
```python
# Pub/Sub for distributed invalidation
def invalidate_user_cache(user_id: str):
    r.delete(f"user:{user_id}")
    r.publish("cache:invalidate", f"user:{user_id}")

# Subscriber in each service
async def cache_invalidation_listener():
    pubsub = r.pubsub()
    pubsub.subscribe("cache:invalidate")
    for message in pubsub.listen():
        if message["type"] == "message":
            key = message["data"]
            local_cache.pop(key, None)
```

### Version-Based (Cache Tags)
```python
def get_user_with_version(user_id: str) -> dict:
    # Get current version
    version = r.get(f"user:{user_id}:version") or "v0"
    
    cached = r.get(f"user:{user_id}:{version}")
    if cached:
        return json.loads(cached)
    
    user = database.get_user(user_id)
    r.setex(f"user:{user_id}:{version}", 3600, json.dumps(user))
    return user

def invalidate_user(user_id: str):
    # Increment version - old cache becomes orphan
    r.incr(f"user:{user_id}:version")
```

---

## 5. Cache Stampede Prevention

### Problem
When cache expires, N concurrent requests all hit DB simultaneously.

### Solution 1: Single-Flight Pattern
```python
import asyncio
from asyncio import Lock

_locks: dict[str, Lock] = {}

async def get_with_single_flight(key: str, loader) -> dict:
    cached = r.get(key)
    if cached:
        return json.loads(cached)
    
    # Get or create lock for this key
    if key not in _locks:
        _locks[key] = Lock()
    
    async with _locks[key]:
        # Double-check after acquiring lock
        cached = r.get(key)
        if cached:
            return json.loads(cached)
        
        # Only one request loads
        value = await loader()
        r.setex(key, 3600, json.dumps(value))
        return value
```

### Solution 2: Probabilistic Early Expiration
```python
import random
import time

def get_with_early_refresh(key: str, ttl: int = 3600):
    cached = r.get(key)
    remaining_ttl = r.ttl(key)
    
    if cached:
        # Probabilistically refresh before expiry
        # Higher chance as TTL approaches 0
        if remaining_ttl < ttl * 0.1:  # Last 10% of TTL
            if random.random() < 0.1:  # 10% chance
                refresh_in_background(key)
        return json.loads(cached)
    
    return fetch_and_cache(key, ttl)
```

### Solution 3: Locking with Stale Fallback
```python
async def get_with_stale_fallback(key: str):
    cached = r.get(key)
    stale = r.get(f"{key}:stale")
    
    if cached:
        return json.loads(cached)
    
    # Try to acquire refresh lock
    if r.set(f"{key}:lock", "1", nx=True, ex=10):
        try:
            value = await fetch_fresh()
            r.setex(key, 300, json.dumps(value))
            r.setex(f"{key}:stale", 3600, json.dumps(value))
            return value
        finally:
            r.delete(f"{key}:lock")
    
    # Return stale while someone else refreshes
    if stale:
        return json.loads(stale)
    
    # No stale data - wait for lock holder
    await asyncio.sleep(0.1)
    return await get_with_stale_fallback(key)
```

---

## 6. Microservices Caching Architecture

### Cache-Per-Service Pattern
```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   User Service  │  │  Order Service  │  │ Product Service │
│  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │
│  │ L1: Local │  │  │  │ L1: Local │  │  │  │ L1: Local │  │
│  │  (memory) │  │  │  │  (memory) │  │  │  │  (memory) │  │
│  └─────┬─────┘  │  │  └─────┬─────┘  │  │  └─────┬─────┘  │
│  ┌─────┴─────┐  │  │  ┌─────┴─────┐  │  │  ┌─────┴─────┐  │
│  │ L2: Redis │  │  │  │ L2: Redis │  │  │  │ L2: Redis │  │
│  │ (cluster) │  │  │  │ (cluster) │  │  │  │ (cluster) │  │
│  └───────────┘  │  │  └───────────┘  │  │  └───────────┘  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                   │                    │
         └───────────────────┼────────────────────┘
                             │
                    ┌────────┴────────┐
                    │  Shared Redis   │
                    │    Cluster      │
                    └─────────────────┘
```

### Multi-Tier Cache Implementation
```python
from functools import lru_cache
import redis

class TieredCache:
    def __init__(self):
        self.l1 = {}  # Process-local dict
        self.l2 = redis.Redis()  # Redis
    
    def get(self, key: str) -> dict | None:
        # L1: Memory (fastest)
        if key in self.l1:
            return self.l1[key]
        
        # L2: Redis
        cached = self.l2.get(key)
        if cached:
            value = json.loads(cached)
            self.l1[key] = value  # Promote to L1
            return value
        
        return None
    
    def set(self, key: str, value: dict, ttl: int = 3600):
        self.l1[key] = value
        self.l2.setex(key, ttl, json.dumps(value))
    
    def invalidate(self, key: str):
        self.l1.pop(key, None)
        self.l2.delete(key)
        # Broadcast to other instances
        self.l2.publish("cache:invalidate", key)
```

---

## 7. Anti-Patterns to Avoid

### ❌ Caching Everything
```python
# BAD: Caching rarely-accessed data wastes memory
@cache
def get_user_by_random_field(field_value):
    ...
```

### ❌ No Cache Size Limits
```python
# BAD: Unbounded cache = OOM
cache = {}  # Grows forever
```

### ❌ Ignoring Cache Coherence
```python
# BAD: Update DB but forget cache
def update_user(user_id, data):
    database.update(user_id, data)
    # Forgot to invalidate cache!
```

### ❌ Too-Long TTLs
```python
# BAD: 24-hour cache for frequently-changing data
r.setex("stock_price", 86400, price)  # Stale for hours
```

---

## Quick Reference

| Pattern | Use Case | Pros | Cons |
|---------|----------|------|------|
| lru_cache | Pure functions | Fast, simple | Memory-only, no TTL |
| diskcache | Persistent memoization | Survives restart | Disk I/O |
| Cache-Aside | Read-heavy workloads | Simple, lazy | Miss penalty |
| Write-Through | Consistency critical | Always fresh | Write latency |
| Write-Behind | Write-heavy | Fast writes | Complexity, data loss risk |
| Single-Flight | Prevent stampede | No thundering herd | Lock overhead |

---

## Sources
- Python functools documentation (2026)
- Redis caching patterns - Official Redis docs
- diskcache documentation
- Martin Fowler - Patterns of Enterprise Application Architecture
- High Scalability blog - caching strategies

*Researched: January 2026 | Cycle 20*
