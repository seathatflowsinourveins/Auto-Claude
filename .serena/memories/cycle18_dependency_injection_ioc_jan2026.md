# Cycle 18: Dependency Injection & Inversion of Control (January 2026)

## CORE CONCEPTS

### Why DI in Python?
Python's dynamic nature provides built-in flexibility, but DI frameworks add:
- **Explicit dependency graphs** - visible, auditable relationships
- **Lifecycle management** - singleton, transient, scoped lifetimes
- **Testing** - easy mocking/stubbing without monkey-patching
- **Configuration** - centralized wiring in one place

### The DI Principle
```python
# WITHOUT DI - tightly coupled
class UserService:
    def __init__(self):
        self.db = PostgresDatabase()  # Hard dependency
        
# WITH DI - loosely coupled
class UserService:
    def __init__(self, db: Database):  # Injected
        self.db = db
```

## PYTHON DI LIBRARIES (2026)

### 1. dependency-injector (Most Popular)
**4.48.3** - Full-featured IoC container

```python
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    
    # Singleton - one instance forever
    database = providers.Singleton(
        Database,
        connection_string=config.db.connection_string
    )
    
    # Factory - new instance each time
    user_repository = providers.Factory(
        UserRepository,
        database=database
    )
    
    # Resource - with cleanup (async context manager)
    http_client = providers.Resource(
        init_http_client,
        timeout=config.http.timeout
    )

# Usage
container = Container()
container.config.from_yaml("config.yaml")

# Wiring - auto-inject into modules
container.wire(modules=[__name__])

@inject
def get_users(repo: UserRepository = Provide[Container.user_repository]):
    return repo.find_all()
```

**Provider Types**:
| Provider | Lifetime | Use Case |
|----------|----------|----------|
| `Singleton` | Application | DB pools, config |
| `Factory` | Per-call | Request handlers |
| `Resource` | Managed | Connections with cleanup |
| `Callable` | Per-call | Functions |
| `Object` | Static | Constants |

### 2. punq (Minimalist)
**No decorators, no magic**

```python
import punq

container = punq.Container()

# Register with interface
container.register(Database, PostgresDatabase)
container.register(UserRepository)  # Auto-resolves Database

# Resolve
repo = container.resolve(UserRepository)
```

### 3. kink (Decorator-Based)
```python
from kink import di, inject

# Register
di[Database] = PostgresDatabase(conn_string)

# Auto-inject
@inject
class UserService:
    def __init__(self, db: Database):
        self.db = db  # Automatically injected
```

### 4. svcs (Service Locator + Health Checks)
```python
from svcs import Container, Registry

registry = Registry()

# Register factory with cleanup
@registry.register_factory(Database)
async def create_database():
    db = await Database.connect()
    yield db
    await db.close()  # Cleanup on container close

# Health checks built-in
registry.register_factory(
    HttpClient,
    factory=create_client,
    ping=lambda client: client.get("/health")
)

# Check all services
await container.aping_all()  # Returns health status
```

### 5. yedi (Type-Safe, Lightweight)
```python
from yedi import container

@container.provide()  # Auto-register
class DatabaseService:
    pass

@container.provide()
class UserService:
    def __init__(self, db: DatabaseService):  # Auto-resolved
        self.db = db

@container.inject  # Auto-inject parameters
def process(user_service: UserService, data: str):
    return user_service.process(data)

result = process(data="test")  # user_service injected
```

## FASTAPI DEPENDENCY INJECTION (2026)

### Built-in Depends System
```python
from fastapi import Depends, FastAPI
from typing import Annotated

app = FastAPI()

# Simple dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Type alias for reusability
DbDep = Annotated[Session, Depends(get_db)]

@app.get("/users")
async def get_users(db: DbDep):
    return db.query(User).all()
```

### Modern Lifespan Management (2026 Pattern)
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.db_pool = await create_pool()
    app.state.redis = await aioredis.create_redis_pool()
    
    yield  # Application runs
    
    # Shutdown
    await app.state.db_pool.close()
    await app.state.redis.close()

app = FastAPI(lifespan=lifespan)

# Access in dependencies
def get_db_pool(request: Request):
    return request.app.state.db_pool
```

### Dependency Trees at Scale
```python
# Base dependencies
def get_settings() -> Settings:
    return Settings()

def get_db_pool(settings: Annotated[Settings, Depends(get_settings)]):
    return create_pool(settings.database_url)

# Composed dependencies
def get_user_repository(
    pool: Annotated[Pool, Depends(get_db_pool)]
) -> UserRepository:
    return UserRepository(pool)

def get_auth_service(
    users: Annotated[UserRepository, Depends(get_user_repository)],
    settings: Annotated[Settings, Depends(get_settings)]
) -> AuthService:
    return AuthService(users, settings.jwt_secret)

# Route uses composed tree
@app.post("/login")
async def login(
    auth: Annotated[AuthService, Depends(get_auth_service)],
    credentials: LoginRequest
):
    return auth.authenticate(credentials)
```

### Scoped Dependencies
```python
from fastapi import Depends
from functools import lru_cache

# Request-scoped (default) - new per request
def get_request_id():
    return uuid.uuid4()

# Application-scoped (singleton via lru_cache)
@lru_cache
def get_settings():
    return Settings()

# Session-scoped (custom)
class ScopedContainer:
    _instances: dict = {}
    
    @classmethod
    def get(cls, key, factory):
        if key not in cls._instances:
            cls._instances[key] = factory()
        return cls._instances[key]
```

## CLEAN ARCHITECTURE WITH DI

### Layered Dependencies (Inward Only)
```
┌─────────────────────────────────────────────┐
│ Frameworks & Drivers (FastAPI, SQLAlchemy)  │
├─────────────────────────────────────────────┤
│ Interface Adapters (Controllers, Repos)     │
├─────────────────────────────────────────────┤
│ Application (Use Cases)                      │
├─────────────────────────────────────────────┤
│ Domain (Entities, Value Objects)             │
└─────────────────────────────────────────────┘
     Dependencies point INWARD only ↑
```

### Dependency Inversion
```python
# Domain layer - defines interface
from abc import ABC, abstractmethod

class UserRepository(ABC):
    @abstractmethod
    async def find_by_id(self, user_id: str) -> User: ...

# Infrastructure layer - implements interface
class PostgresUserRepository(UserRepository):
    def __init__(self, pool: Pool):
        self.pool = pool
    
    async def find_by_id(self, user_id: str) -> User:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
            return User(**row)

# Wire in container
container.register(UserRepository, PostgresUserRepository)
```

## TESTING WITH DI

### Override Dependencies
```python
# dependency-injector
with container.user_repository.override(MockUserRepository()):
    result = get_users()  # Uses mock

# FastAPI
app.dependency_overrides[get_db] = lambda: mock_db

# punq
test_container = punq.Container()
test_container.register(Database, MockDatabase)
```

### Fixture-Based Testing
```python
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def mock_db():
    return MockDatabase()

@pytest.fixture
def client(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db
    yield TestClient(app)
    app.dependency_overrides.clear()

def test_get_users(client):
    response = client.get("/users")
    assert response.status_code == 200
```

## ANTI-PATTERNS TO AVOID

### 1. Service Locator Abuse
```python
# BAD - hidden dependencies
class UserService:
    def get_user(self, id):
        db = Container.get(Database)  # Hidden, hard to test
        
# GOOD - explicit injection
class UserService:
    def __init__(self, db: Database):
        self.db = db
```

### 2. Circular Dependencies
```python
# BAD - A needs B, B needs A
class A:
    def __init__(self, b: B): ...
    
class B:
    def __init__(self, a: A): ...  # Circular!

# GOOD - break with interface or lazy loading
class A:
    def __init__(self, b_factory: Callable[[], B]):
        self._b_factory = b_factory
```

### 3. Over-Injection
```python
# BAD - too many dependencies
class MegaService:
    def __init__(self, a, b, c, d, e, f, g, h): ...  # 8+ deps = code smell

# GOOD - compose smaller services
class UserService:
    def __init__(self, repo: UserRepository): ...
```

## LIBRARY SELECTION GUIDE

| Need | Recommended |
|------|-------------|
| Full-featured IoC | dependency-injector |
| Minimal, no magic | punq |
| FastAPI native | Depends + Annotated |
| Health checks | svcs |
| Type-safe decorators | yedi, kink |

## KEY PATTERNS SUMMARY

1. **Constructor Injection** - Dependencies via `__init__`
2. **Interface Segregation** - Depend on abstractions (ABC)
3. **Composition Root** - Wire everything in one place
4. **Scoped Lifetimes** - Singleton, Transient, Request-scoped
5. **Override for Testing** - Swap implementations easily
6. **Lifespan Management** - Async context managers for cleanup

---
*Cycle 18 - Dependency Injection & IoC - January 2026*
