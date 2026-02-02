# Cycle 46: Caching Strategies & Invalidation Patterns (January 2026)

## Overview
Production patterns for distributed caching in Python, covering Redis integration, caching strategies, invalidation patterns, and stampede prevention.

---

## 1. Caching Strategies

### Cache-Aside (Lazy Loading)
Most common pattern - application manages cache explicitly.
```python
import redis
from functools import wraps
import json

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

def cache_aside(ttl: int = 300, prefix: str = "cache"):
    """Cache-aside decorator with JSON serialization."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key = f"{prefix}:{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Try cache first
            cached = redis_client.get(key)
            if cached:
                return json.loads(cached)
            
            # Cache miss - fetch from source
            result = await func(*args, **kwargs)
            
            # Store in cache
            redis_client.setex(key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator

@cache_aside(ttl=600, prefix="user")
async def get_user_profile(user_id: int) -> dict:
    return await db.fetch_one("SELECT * FROM users WHERE id = $1", user_id)
```

### Write-Through
Write to cache and database simultaneously.
```python
class WriteThroughCache:
    def __init__(self, redis_client, db_session):
        self.redis = redis_client
        self.db = db_session
    
    async def set(self, key: str, value: dict, ttl: int = 300):
        """Write to both cache and database atomically."""
        async with self.db.begin():
            # Write to database first
            await self.db.execute(
                "INSERT INTO cache_store (key, value) VALUES ($1, $2) "
                "ON CONFLICT (key) DO UPDATE SET value = $2",
                key, json.dumps(value)
            )
            # Then update cache
            self.redis.setex(key, ttl, json.dumps(value))
    
    async def get(self, key: str) -> dict | None:
        # Read from cache (always fresh due to write-through)
        cached = self.redis.get(key)
        if cached:
            return json.loads(cached)
        
        # Fallback to DB (cache miss after restart)
        row = await self.db.fetch_one(
            "SELECT value FROM cache_store WHERE key = $1", key
        )
        if row:
            value = json.loads(row["value"])
            self.redis.setex(key, 300, row["value"])  # Repopulate cache
            return value
        return None
```

### Write-Behind (Write-Back)
Async write to database, immediate cache update.
```python
from celery import Celery

app = Celery("cache_tasks", broker="redis://localhost:6379/1")

class WriteBehindCache:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def set(self, key: str, value: dict, ttl: int = 300):
        """Update cache immediately, persist to DB asynchronously."""
        self.redis.setex(key, ttl, json.dumps(value))
        # Queue database write (non-blocking)
        persist_to_database.delay(key, value)
    
    def get(self, key: str) -> dict | None:
        cached = self.redis.get(key)
        return json.loads(cached) if cached else None

@app.task(bind=True, max_retries=3)
def persist_to_database(self, key: str, value: dict):
    """Async task to persist cache to database."""
    try:
        db.execute(
            "INSERT INTO cache_store (key, value, updated_at) "
            "VALUES ($1, $2, NOW()) ON CONFLICT (key) DO UPDATE SET value = $2",
            key, json.dumps(value)
        )
    except Exception as exc:
        raise self.retry(exc=exc, countdown=5)
```

---

## 2. FastAPI + Redis Integration

### Connection Pool with Lifespan
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
import redis.asyncio as aioredis

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create Redis connection pool
    app.state.redis = aioredis.ConnectionPool.from_url(
        "redis://localhost:6379",
        max_connections=50,
        decode_responses=True,
    )
    yield
    # Shutdown: Close pool
    await app.state.redis.disconnect()

app = FastAPI(lifespan=lifespan)

async def get_redis(request) -> aioredis.Redis:
    return aioredis.Redis(connection_pool=request.app.state.redis)

@app.get("/users/{user_id}")
async def get_user(user_id: int, redis: aioredis.Redis = Depends(get_redis)):
    cache_key = f"user:{user_id}"
    
    # Try cache
    if cached := await redis.get(cache_key):
        return {"source": "cache", "data": json.loads(cached)}
    
    # Fetch from DB
    user = await db.get_user(user_id)
    await redis.setex(cache_key, 300, json.dumps(user))
    return {"source": "database", "data": user}
```

### Caching Decorator for FastAPI
```python
from fastapi import Request
from functools import wraps

def cached(ttl: int = 300, key_builder=None):
    """FastAPI route caching decorator."""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            redis = await get_redis(request)
            
            # Build cache key
            if key_builder:
                cache_key = key_builder(request, *args, **kwargs)
            else:
                cache_key = f"route:{request.url.path}:{request.query_params}"
            
            # Check cache
            if cached := await redis.get(cache_key):
                return json.loads(cached)
            
            # Execute handler
            result = await func(request, *args, **kwargs)
            
            # Cache result
            await redis.setex(cache_key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator
```

---

## 3. Cache Invalidation Strategies

### Time-Based (TTL)
Simplest approach - data expires after fixed time.
```python
# Simple TTL
redis.setex("user:123", 300, user_json)  # 5 minutes

# Sliding expiration (extend TTL on access)
async def get_with_sliding_ttl(key: str, ttl: int = 300):
    value = await redis.get(key)
    if value:
        await redis.expire(key, ttl)  # Reset TTL on hit
    return value
```

### Event-Driven Invalidation
Invalidate cache when source data changes.
```python
from fastapi import BackgroundTasks

class CacheInvalidator:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def invalidate_user(self, user_id: int):
        """Invalidate all user-related cache keys."""
        pattern = f"user:{user_id}:*"
        keys = []
        async for key in self.redis.scan_iter(pattern):
            keys.append(key)
        if keys:
            await self.redis.delete(*keys)
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern."""
        keys = [key async for key in self.redis.scan_iter(pattern)]
        if keys:
            await self.redis.delete(*keys)

@app.put("/users/{user_id}")
async def update_user(
    user_id: int,
    data: UserUpdate,
    background_tasks: BackgroundTasks,
    invalidator: CacheInvalidator = Depends()
):
    # Update database
    user = await db.update_user(user_id, data)
    
    # Invalidate cache in background
    background_tasks.add_task(invalidator.invalidate_user, user_id)
    
    return user
```

### CDC with PostgreSQL LISTEN/NOTIFY
Real-time invalidation without polling.
```python
import asyncpg
import asyncio

class PostgresCDCInvalidator:
    """Real-time cache invalidation via PostgreSQL NOTIFY."""
    
    def __init__(self, redis_client, pg_dsn: str):
        self.redis = redis_client
        self.pg_dsn = pg_dsn
    
    async def start_listener(self):
        conn = await asyncpg.connect(self.pg_dsn)
        await conn.add_listener("cache_invalidate", self._handle_notification)
        # Keep connection alive
        while True:
            await asyncio.sleep(60)
    
    async def _handle_notification(self, conn, pid, channel, payload):
        """Handle PostgreSQL NOTIFY message."""
        data = json.loads(payload)
        table = data["table"]
        operation = data["operation"]  # INSERT, UPDATE, DELETE
        record_id = data["id"]
        
        # Invalidate cache based on table
        cache_key = f"{table}:{record_id}"
        await self.redis.delete(cache_key)
        print(f"Invalidated {cache_key} due to {operation}")

# PostgreSQL Trigger (run once to set up)
"""
CREATE OR REPLACE FUNCTION notify_cache_invalidation()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify('cache_invalidate', json_build_object(
        'table', TG_TABLE_NAME,
        'operation', TG_OP,
        'id', COALESCE(NEW.id, OLD.id)
    )::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER users_cache_trigger
AFTER INSERT OR UPDATE OR DELETE ON users
FOR EACH ROW EXECUTE FUNCTION notify_cache_invalidation();
"""
```

---

## 4. Cache Stampede Prevention

### The Problem
When a hot key expires, N concurrent requests all miss cache and hit database simultaneously.
```
Stampede Formula: L = R × T
- L = leaked requests to DB
- R = requests per second (e.g., 10,000)
- T = DB query time (e.g., 0.3s)
- Result: 3,000 simultaneous DB queries!
```

### Solution 1: Distributed Mutex Lock
Only one process rebuilds cache; others wait.
```python
import uuid

class LockedCache:
    LOCK_TTL = 10  # seconds
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def get_or_rebuild(
        self,
        key: str,
        rebuild_fn,
        ttl: int = 300,
        wait_timeout: float = 5.0
    ):
        """Get from cache or rebuild with distributed lock."""
        # Try cache first
        if value := await self.redis.get(key):
            return json.loads(value)
        
        # Try to acquire lock
        lock_key = f"lock:{key}"
        lock_id = str(uuid.uuid4())
        
        if await self.redis.set(lock_key, lock_id, nx=True, ex=self.LOCK_TTL):
            # We got the lock - rebuild cache
            try:
                value = await rebuild_fn()
                await self.redis.setex(key, ttl, json.dumps(value))
                return value
            finally:
                # Release lock (only if we still own it)
                await self._release_lock(lock_key, lock_id)
        else:
            # Someone else is rebuilding - wait for result
            for _ in range(int(wait_timeout * 10)):
                await asyncio.sleep(0.1)
                if value := await self.redis.get(key):
                    return json.loads(value)
            
            # Timeout - try rebuild ourselves
            return await rebuild_fn()
    
    async def _release_lock(self, key: str, lock_id: str):
        """Release lock only if we own it (Lua script for atomicity)."""
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        await self.redis.eval(script, 1, key, lock_id)
```

### Solution 2: Probabilistic Early Expiration (XFetch)
Randomly refresh before expiration to prevent synchronized expiry.
```python
import random
import math

class ProbabilisticCache:
    """
    XFetch algorithm: probabilistically refresh cache before expiration.
    Paper: "Optimal Probabilistic Cache Stampede Prevention"
    """
    
    def __init__(self, redis_client, beta: float = 1.0):
        self.redis = redis_client
        self.beta = beta  # Controls early refresh probability
    
    async def get_or_refresh(
        self,
        key: str,
        rebuild_fn,
        ttl: int = 300,
        delta: float = 1.0  # Estimated rebuild time in seconds
    ):
        """Get with probabilistic early refresh."""
        # Get value with metadata
        data = await self.redis.hgetall(f"xfetch:{key}")
        
        if data:
            value = json.loads(data["value"])
            expiry = float(data["expiry"])
            now = time.time()
            
            # XFetch formula: should we refresh early?
            # P(refresh) increases as we approach expiry
            ttl_remaining = expiry - now
            
            # Random early expiration check
            # -delta * beta * log(random()) gives exponential distribution
            threshold = -delta * self.beta * math.log(random.random())
            
            if ttl_remaining > threshold:
                # Still fresh enough, return cached value
                return value
            
            # Probabilistically chosen to refresh
            # (but return stale value while refreshing in background)
            asyncio.create_task(self._background_refresh(key, rebuild_fn, ttl))
            return value
        
        # Cache miss - rebuild synchronously
        return await self._rebuild(key, rebuild_fn, ttl)
    
    async def _rebuild(self, key: str, rebuild_fn, ttl: int):
        value = await rebuild_fn()
        expiry = time.time() + ttl
        await self.redis.hset(f"xfetch:{key}", mapping={
            "value": json.dumps(value),
            "expiry": str(expiry),
        })
        await self.redis.expire(f"xfetch:{key}", ttl + 60)  # Buffer for stale reads
        return value
    
    async def _background_refresh(self, key: str, rebuild_fn, ttl: int):
        """Refresh cache in background."""
        try:
            await self._rebuild(key, rebuild_fn, ttl)
        except Exception as e:
            logger.warning(f"Background refresh failed for {key}: {e}")
```

### Solution 3: Stale-While-Revalidate
Serve stale data while refreshing in background.
```python
class StaleWhileRevalidateCache:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def get(
        self,
        key: str,
        rebuild_fn,
        ttl: int = 300,
        stale_ttl: int = 60  # Serve stale for extra 60s while revalidating
    ):
        """Get with stale-while-revalidate semantics."""
        data = await self.redis.hgetall(f"swr:{key}")
        
        if not data:
            # Cache miss - synchronous rebuild
            return await self._rebuild(key, rebuild_fn, ttl, stale_ttl)
        
        value = json.loads(data["value"])
        soft_expiry = float(data["soft_expiry"])
        now = time.time()
        
        if now < soft_expiry:
            # Fresh - return immediately
            return value
        
        # Stale but usable - return stale and refresh in background
        asyncio.create_task(self._rebuild(key, rebuild_fn, ttl, stale_ttl))
        return value  # Return stale immediately
    
    async def _rebuild(self, key: str, rebuild_fn, ttl: int, stale_ttl: int):
        value = await rebuild_fn()
        now = time.time()
        await self.redis.hset(f"swr:{key}", mapping={
            "value": json.dumps(value),
            "soft_expiry": str(now + ttl),  # Soft: after this, refresh in bg
        })
        # Hard expiry: soft_ttl + stale_ttl
        await self.redis.expire(f"swr:{key}", ttl + stale_ttl)
        return value
```

---

## 5. Multi-Level Caching

### L1 (In-Process) + L2 (Redis) Pattern
```python
from cachetools import TTLCache
import threading

class MultiLevelCache:
    def __init__(self, redis_client, l1_maxsize: int = 1000, l1_ttl: int = 60):
        self.redis = redis_client
        self.l1 = TTLCache(maxsize=l1_maxsize, ttl=l1_ttl)
        self.l1_lock = threading.Lock()
    
    async def get(self, key: str, rebuild_fn, l2_ttl: int = 300):
        """Get from L1, then L2, then rebuild."""
        # L1: In-memory (fastest)
        with self.l1_lock:
            if key in self.l1:
                return self.l1[key]
        
        # L2: Redis
        if cached := await self.redis.get(key):
            value = json.loads(cached)
            with self.l1_lock:
                self.l1[key] = value
            return value
        
        # Rebuild
        value = await rebuild_fn()
        
        # Populate both levels
        await self.redis.setex(key, l2_ttl, json.dumps(value))
        with self.l1_lock:
            self.l1[key] = value
        
        return value
    
    async def invalidate(self, key: str):
        """Invalidate from all levels."""
        with self.l1_lock:
            self.l1.pop(key, None)
        await self.redis.delete(key)
```

---

## 6. Redis Data Structures for Caching

### Sorted Sets for Leaderboards/Rankings
```python
async def update_leaderboard(user_id: str, score: float):
    await redis.zadd("leaderboard:global", {user_id: score})

async def get_top_players(limit: int = 10) -> list:
    return await redis.zrevrange("leaderboard:global", 0, limit - 1, withscores=True)

async def get_user_rank(user_id: str) -> int:
    rank = await redis.zrevrank("leaderboard:global", user_id)
    return rank + 1 if rank is not None else None
```

### Hashes for Object Caching
```python
async def cache_user(user: dict):
    await redis.hset(f"user:{user['id']}", mapping=user)
    await redis.expire(f"user:{user['id']}", 300)

async def get_user_field(user_id: int, field: str) -> str:
    return await redis.hget(f"user:{user_id}", field)

async def update_user_field(user_id: int, field: str, value: str):
    await redis.hset(f"user:{user_id}", field, value)
```

### HyperLogLog for Unique Counts
```python
async def track_unique_visitor(page: str, visitor_id: str):
    await redis.pfadd(f"visitors:{page}:{date.today()}", visitor_id)

async def get_unique_visitors(page: str) -> int:
    return await redis.pfcount(f"visitors:{page}:{date.today()}")
```

---

## Quick Reference

### Strategy Selection Guide
| Scenario | Strategy | Why |
|----------|----------|-----|
| Read-heavy, rare writes | Cache-aside | Simple, lazy loading |
| Strong consistency | Write-through | DB always in sync |
| Write-heavy, eventual OK | Write-behind | Fast writes, async persist |
| Hot keys | Probabilistic expiry | Prevents stampede |
| Real-time data | CDC invalidation | Instant freshness |

### TTL Guidelines
| Data Type | Recommended TTL |
|-----------|-----------------|
| Session data | 30 min - 24 hours |
| User profiles | 5-15 minutes |
| Product catalog | 1-24 hours |
| Configuration | 5-60 minutes |
| Real-time data | 10-60 seconds |

### Anti-Patterns
- **No TTL**: Memory leaks, stale data forever
- **Cache everything**: Memory exhaustion, low hit ratio
- **Large objects**: Serialization overhead, memory waste
- **Ignoring stampede**: One expired key can crash your DB
- **No monitoring**: Blind to hit ratio, memory usage, evictions

### Monitoring Metrics
```python
# Key metrics to track
info = await redis.info("stats")
hit_rate = info["keyspace_hits"] / (info["keyspace_hits"] + info["keyspace_misses"])
memory_used = await redis.info("memory")["used_memory_human"]
evictions = info["evicted_keys"]
```

---

*Cycle 46 Complete - Caching Strategies & Invalidation Patterns*
*Date: January 2026*
