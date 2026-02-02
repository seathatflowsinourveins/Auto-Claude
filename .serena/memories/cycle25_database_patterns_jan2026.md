# Cycle 25: Database Patterns - January 2026

## Research Focus
SQLAlchemy 2.0 async patterns, Alembic migrations, connection pooling, query optimization.

---

## 1. SQLAlchemy 2.0 Async Patterns

### Core Async Setup
```python
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Engine with async driver
engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    pool_size=20,
    max_overflow=10,
    pool_recycle=3600,  # Recycle connections after 1 hour
    pool_pre_ping=True,  # Verify connection health
    echo=False,  # Disable SQL logging in production
)

# Session factory (NOT scoped_session for async!)
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Prevent lazy load after commit
)

# Declarative base with 2.0 style
class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(unique=True, index=True)
    name: Mapped[str | None]  # Optional field
```

### Async Session Dependency (FastAPI)
```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator

@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

# FastAPI dependency
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with get_session() as session:
        yield session

# Usage in endpoint
@app.get("/users/{user_id}")
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()
```

### Eager Loading (Prevent N+1)
```python
from sqlalchemy.orm import selectinload, joinedload

# selectinload: Separate SELECT for related objects (best for collections)
stmt = (
    select(User)
    .options(selectinload(User.orders))
    .where(User.id == user_id)
)

# joinedload: LEFT OUTER JOIN (best for single objects)
stmt = (
    select(Order)
    .options(joinedload(Order.user))
    .where(Order.id == order_id)
)

# Multiple levels
stmt = (
    select(User)
    .options(
        selectinload(User.orders).selectinload(Order.items)
    )
)
```

### Bulk Operations
```python
from sqlalchemy import insert, update

# Bulk insert (no ORM overhead)
await session.execute(
    insert(User),
    [
        {"email": "a@example.com", "name": "Alice"},
        {"email": "b@example.com", "name": "Bob"},
    ]
)

# Bulk update
await session.execute(
    update(User)
    .where(User.status == "pending")
    .values(status="active")
)
await session.commit()
```

---

## 2. Alembic Migrations

### Project Setup
```bash
# Initialize Alembic
alembic init alembic

# Directory structure
alembic/
├── env.py           # Environment configuration
├── script.py.mako   # Migration template
└── versions/        # Migration scripts
```

### Async env.py Configuration
```python
# alembic/env.py
from logging.config import fileConfig
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy import pool
from alembic import context

from app.models import Base  # Import your models
from app.config import settings

config = context.config

# Set database URL from settings
config.set_main_option("sqlalchemy.url", settings.database_url)

target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()

async def run_async_migrations() -> None:
    """Run migrations with async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    
    await connectable.dispose()

def do_run_migrations(connection):
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    import asyncio
    asyncio.run(run_async_migrations())

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### Migration Commands
```bash
# Generate migration from model changes
alembic revision --autogenerate -m "add users table"

# Apply all pending migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# Show current revision
alembic current

# Show migration history
alembic history --verbose
```

### Migration Best Practices
```python
# alembic/versions/001_add_users_table.py
"""add users table

Revision ID: 001abc
Revises: 
Create Date: 2026-01-25

"""
from alembic import op
import sqlalchemy as sa

revision = '001abc'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index('ix_users_email', 'users', ['email'], unique=True)

def downgrade() -> None:
    op.drop_index('ix_users_email', 'users')
    op.drop_table('users')
```

### Data Migrations (Safe Pattern)
```python
def upgrade() -> None:
    # 1. Add new column as nullable
    op.add_column('users', sa.Column('full_name', sa.String(255), nullable=True))
    
    # 2. Migrate data
    op.execute("""
        UPDATE users 
        SET full_name = first_name || ' ' || last_name
        WHERE full_name IS NULL
    """)
    
    # 3. Make non-nullable (after data migration)
    op.alter_column('users', 'full_name', nullable=False)
    
    # 4. Drop old columns (in separate migration!)
    # op.drop_column('users', 'first_name')
    # op.drop_column('users', 'last_name')
```

---

## 3. Connection Pooling

### PgBouncer Configuration
```ini
# pgbouncer.ini
[databases]
mydb = host=127.0.0.1 port=5432 dbname=mydb

[pgbouncer]
listen_addr = 127.0.0.1
listen_port = 6432
auth_type = scram-sha-256
auth_file = /etc/pgbouncer/userlist.txt

# Pool modes:
# session: Connection per client session (default)
# transaction: Connection per transaction (recommended)
# statement: Connection per statement (limited use)
pool_mode = transaction

# Pool sizing
default_pool_size = 20
min_pool_size = 5
max_client_conn = 1000
reserve_pool_size = 5

# Timeouts
server_idle_timeout = 600
query_timeout = 30
client_idle_timeout = 0
```

### SQLAlchemy Pool Configuration
```python
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import AsyncAdaptedQueuePool

engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost:6432/db",  # Via PgBouncer
    
    # Pool settings
    poolclass=AsyncAdaptedQueuePool,
    pool_size=20,          # Steady-state connections
    max_overflow=10,       # Additional connections under load
    pool_timeout=30,       # Wait for connection (seconds)
    pool_recycle=3600,     # Recycle after 1 hour (avoid stale)
    pool_pre_ping=True,    # Health check before use
    
    # Connection args
    connect_args={
        "server_settings": {
            "application_name": "myapp",
            "statement_timeout": "30000",  # 30s query timeout
        }
    }
)
```

### Connection Pool Monitoring
```python
from sqlalchemy import event

@event.listens_for(engine.sync_engine, "checkout")
def receive_checkout(dbapi_connection, connection_record, connection_proxy):
    """Log when connection is checked out from pool."""
    logger.debug("Connection checked out", 
                 pool_size=engine.pool.size(),
                 checked_out=engine.pool.checkedout())

@event.listens_for(engine.sync_engine, "checkin")
def receive_checkin(dbapi_connection, connection_record):
    """Log when connection is returned to pool."""
    logger.debug("Connection returned",
                 pool_size=engine.pool.size())
```

---

## 4. Query Optimization

### EXPLAIN ANALYZE
```python
from sqlalchemy import text

async def explain_query(session: AsyncSession, stmt):
    """Analyze query execution plan."""
    # Get compiled SQL
    compiled = stmt.compile(
        dialect=session.bind.dialect,
        compile_kwargs={"literal_binds": True}
    )
    
    # Run EXPLAIN ANALYZE
    result = await session.execute(
        text(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {compiled}")
    )
    plan = result.scalar_one()
    
    return {
        "execution_time_ms": plan[0]["Execution Time"],
        "planning_time_ms": plan[0]["Planning Time"],
        "plan": plan[0]["Plan"],
    }
```

### Indexing Strategies
```python
from sqlalchemy import Index
from sqlalchemy.dialects.postgresql import JSONB

class Order(Base):
    __tablename__ = "orders"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(index=True)  # FK index
    status: Mapped[str] = mapped_column(index=True)
    total: Mapped[Decimal]
    metadata: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(index=True)
    
    __table_args__ = (
        # Composite index for common query pattern
        Index('ix_orders_user_status', 'user_id', 'status'),
        
        # Partial index (only active orders)
        Index('ix_orders_active', 'created_at',
              postgresql_where=text("status = 'active'")),
        
        # GIN index for JSONB
        Index('ix_orders_metadata', 'metadata',
              postgresql_using='gin'),
    )
```

### N+1 Detection
```python
import structlog
from sqlalchemy import event

logger = structlog.get_logger()

# Track queries per request
_query_count = contextvars.ContextVar('query_count', default=0)

@event.listens_for(engine.sync_engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    count = _query_count.get()
    _query_count.set(count + 1)
    
    # Warn on potential N+1 (more than 10 queries)
    if count > 10:
        logger.warning("Potential N+1 detected",
                      query_count=count,
                      statement=statement[:100])
```

### Batch Loading Pattern
```python
from typing import Sequence

async def get_users_with_orders(
    session: AsyncSession, 
    user_ids: Sequence[int]
) -> list[User]:
    """Efficiently load users with their orders in 2 queries."""
    
    # Query 1: Load users
    result = await session.execute(
        select(User)
        .where(User.id.in_(user_ids))
        .options(selectinload(User.orders))  # Query 2: Load orders
    )
    
    return result.scalars().all()

# Pagination with cursor
async def paginate_orders(
    session: AsyncSession,
    cursor: int | None = None,
    limit: int = 20
) -> tuple[list[Order], int | None]:
    """Cursor-based pagination (more efficient than OFFSET)."""
    
    stmt = select(Order).order_by(Order.id.desc()).limit(limit + 1)
    
    if cursor:
        stmt = stmt.where(Order.id < cursor)
    
    result = await session.execute(stmt)
    orders = result.scalars().all()
    
    # Check if there's a next page
    next_cursor = orders[-1].id if len(orders) > limit else None
    
    return orders[:limit], next_cursor
```

---

## 5. Transaction Patterns

### Nested Transactions (Savepoints)
```python
async def transfer_funds(
    session: AsyncSession,
    from_id: int,
    to_id: int,
    amount: Decimal
) -> bool:
    """Transfer with rollback on failure."""
    
    async with session.begin_nested():  # Creates savepoint
        try:
            # Debit source account
            await session.execute(
                update(Account)
                .where(Account.id == from_id)
                .values(balance=Account.balance - amount)
            )
            
            # Credit destination account
            await session.execute(
                update(Account)
                .where(Account.id == to_id)
                .values(balance=Account.balance + amount)
            )
            
            return True
            
        except Exception as e:
            # Savepoint rolled back, outer transaction continues
            logger.error("Transfer failed", error=str(e))
            return False
```

### Optimistic Locking
```python
from sqlalchemy.orm import Mapped, mapped_column

class Order(Base):
    __tablename__ = "orders"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str]
    version: Mapped[int] = mapped_column(default=1)
    
    __mapper_args__ = {
        "version_id_col": version
    }

# Update with version check
async def update_order_status(
    session: AsyncSession,
    order_id: int,
    new_status: str
) -> Order:
    result = await session.execute(
        select(Order).where(Order.id == order_id)
    )
    order = result.scalar_one()
    
    order.status = new_status  # version auto-increments
    
    try:
        await session.commit()
        return order
    except StaleDataError:
        raise ConcurrentModificationError("Order was modified by another process")
```

---

## Quick Reference

| Pattern | When to Use |
|---------|-------------|
| `async_sessionmaker` | All async SQLAlchemy apps |
| `selectinload` | Loading collections (1-to-many) |
| `joinedload` | Loading single objects (many-to-1) |
| `pool_pre_ping=True` | Long-running apps, cloud DBs |
| `pool_recycle=3600` | When using PgBouncer |
| Cursor pagination | Large datasets (avoid OFFSET) |
| Partial indexes | Filtering on status/type columns |
| GIN indexes | JSONB columns |
| `begin_nested()` | Savepoints within transactions |
| `version_id_col` | Optimistic locking |

---

*Cycle 25 Complete - Database Patterns Documented*
*Next: Cycle 26 - API Design Patterns*
