# Cycle 44: Database Connection Pooling & Async Patterns (January 2026)

## Overview
Production patterns for database connection management in Python async applications. Focus on SQLAlchemy 2.0, asyncpg, PgBouncer, and FastAPI integration.

---

## SQLAlchemy 2.0 Async Engine

### Engine Creation Pattern
```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

# Production async engine configuration
engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost:5432/db",
    pool_size=20,           # Base connections
    max_overflow=10,        # Extra connections under load
    pool_pre_ping=True,     # Validate connections before use
    pool_recycle=3600,      # Recycle after 1 hour
    echo=False,             # Disable SQL logging in prod
    connect_args={
        "server_settings": {
            "jit": "off",   # Faster for OLTP
            "statement_timeout": "30000",  # 30s timeout
        },
        "prepared_statement_cache_size": 100,  # Reduce latency
    },
)

# Session factory
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Prevent lazy loading issues
    autoflush=False,         # Explicit control
)
```

### Session Context Manager
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def get_session() -> AsyncSession:
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

---

## Pool Sizing Formula

### Rule of Thumb
```python
# Optimal pool size calculation
connections = (cpu_cores * 2) + effective_spindle_count

# For SSD (spindle_count ≈ 1):
# 8-core server → (8 * 2) + 1 = 17 connections

# Conservative formula for async apps:
pool_size = min(cpu_cores * 4, 50)
max_overflow = pool_size // 2
```

### Why Small Pools Win
- PostgreSQL handles ~300 connections max efficiently
- Connection overhead: ~5-10MB RAM per connection
- Context switching dominates at high connection counts
- **Pattern**: 20-30 pool_size handles 10,000+ concurrent requests

---

## asyncpg Direct Pooling

### When to Use asyncpg Directly
- Maximum performance required
- Simple queries without ORM
- Bulk data operations

```python
import asyncpg

# Create connection pool
pool = await asyncpg.create_pool(
    dsn="postgresql://user:pass@localhost:5432/db",
    min_size=10,
    max_size=50,
    max_inactive_connection_lifetime=300,  # 5 min idle timeout
    command_timeout=30,
    statement_cache_size=100,
)

# Use pool
async with pool.acquire() as conn:
    rows = await conn.fetch("SELECT * FROM users WHERE id = $1", user_id)
```

### Prepared Statements (Latency Reduction)
```python
# Prepared statement - 20-40% faster for repeated queries
async with pool.acquire() as conn:
    stmt = await conn.prepare("SELECT * FROM orders WHERE user_id = $1")
    orders = await stmt.fetch(user_id)
```

---

## PgBouncer Configuration

### Transaction Mode (Recommended for Web Apps)
```ini
[pgbouncer]
pool_mode = transaction      # Release on transaction end
max_client_conn = 1000       # Max frontend connections
default_pool_size = 25       # Backend connections per user/db
reserve_pool_size = 5        # Extra connections for burst
reserve_pool_timeout = 3     # Wait before using reserve
server_idle_timeout = 600    # Close idle backend connections
```

### Session Mode (When Needed)
Use when:
- LISTEN/NOTIFY required
- Session-level temp tables
- SET commands that must persist

```ini
pool_mode = session
# Higher backend requirement
default_pool_size = 100
```

### Application Connection String
```python
# Connect to PgBouncer, not PostgreSQL directly
DATABASE_URL = "postgresql://user:pass@pgbouncer:6432/db"
```

---

## FastAPI Lifespan Integration

### Modern Lifespan Pattern (FastAPI 0.95+)
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create engine and pool
    engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
    app.state.engine = engine
    app.state.async_session = async_sessionmaker(engine, class_=AsyncSession)
    
    yield  # Application runs here
    
    # Shutdown: Dispose connections
    await engine.dispose()

app = FastAPI(lifespan=lifespan)

# Dependency injection
async def get_db(request: Request) -> AsyncSession:
    async with request.app.state.async_session() as session:
        yield session

@app.get("/users/{user_id}")
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()
```

---

## Connection Health Monitoring

### Pool Statistics
```python
# SQLAlchemy pool status
pool = engine.pool
stats = {
    "checked_in": pool.checkedin(),
    "checked_out": pool.checkedout(),
    "overflow": pool.overflow(),
    "size": pool.size(),
}

# Prometheus metrics integration
from prometheus_client import Gauge

db_pool_in_use = Gauge('db_pool_connections_in_use', 'Active DB connections')
db_pool_in_use.set(pool.checkedout())
```

### Connection Validation
```python
# pool_pre_ping validates on checkout (recommended)
engine = create_async_engine(url, pool_pre_ping=True)

# Custom health check
async def check_db_health():
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
```

---

## Anti-Patterns to Avoid

### ❌ Creating Engine Per Request
```python
# WRONG: New engine per request
@app.get("/")
async def bad_handler():
    engine = create_async_engine(URL)  # Creates new pool!
    async with engine.begin() as conn:
        ...
```

### ❌ Unbounded Connection Growth
```python
# WRONG: No max_overflow limit
engine = create_async_engine(
    url,
    pool_size=20,
    max_overflow=-1,  # Unlimited - will exhaust PostgreSQL
)
```

### ❌ Long-Running Transactions
```python
# WRONG: Holding connection during external calls
async with session.begin():
    await slow_external_api()  # Blocks connection for seconds
    await session.execute(...)
```

---

## Production Checklist

- [ ] pool_pre_ping enabled for stale connection detection
- [ ] pool_recycle set (1-4 hours) for long-running apps
- [ ] max_overflow bounded (typically pool_size / 2)
- [ ] Connection timeout configured (30-60s)
- [ ] PgBouncer for high connection count (>100 concurrent)
- [ ] Metrics exported for pool utilization
- [ ] Health check endpoint validates DB connectivity
- [ ] Graceful shutdown disposes connections properly

---

## Quick Reference

| Setting | SQLAlchemy | asyncpg | PgBouncer |
|---------|------------|---------|-----------|
| Pool Size | `pool_size=20` | `min_size=10, max_size=50` | `default_pool_size=25` |
| Overflow | `max_overflow=10` | N/A | `reserve_pool_size=5` |
| Idle Timeout | `pool_recycle=3600` | `max_inactive_connection_lifetime=300` | `server_idle_timeout=600` |
| Validation | `pool_pre_ping=True` | Default on | `server_check_query` |

---

*Cycle 44 - Database Connection Pooling & Async - January 2026*
