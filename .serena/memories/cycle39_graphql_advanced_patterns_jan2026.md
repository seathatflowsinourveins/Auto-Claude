# Cycle 39: GraphQL Advanced Patterns (January 2026)

## Overview
Production patterns for GraphQL APIs in Python using Strawberry, DataLoader for N+1 optimization, subscriptions for real-time data, and federation for distributed schemas.

---

## 1. Strawberry GraphQL - Modern Python GraphQL

### Why Strawberry Over Graphene
- **Type hints**: Native Python 3.10+ type annotations
- **Async first**: Built for async/await from the ground up
- **Code-first**: Schema derived from Python classes
- **Federation**: Native Apollo Federation v2 support
- **DataLoader**: Built-in integration

### Basic Schema Definition
```python
import strawberry
from typing import Optional
from datetime import datetime

@strawberry.type
class User:
    id: strawberry.ID
    email: str
    name: str
    created_at: datetime
    
@strawberry.type
class Post:
    id: strawberry.ID
    title: str
    content: str
    author_id: strawberry.ID
    published: bool = False

@strawberry.type
class Query:
    @strawberry.field
    async def user(self, id: strawberry.ID, info: strawberry.Info) -> Optional[User]:
        # Access context for database/services
        db = info.context["db"]
        return await db.get_user(id)
    
    @strawberry.field
    async def posts(self, limit: int = 10, offset: int = 0) -> list[Post]:
        return await db.get_posts(limit=limit, offset=offset)

schema = strawberry.Schema(query=Query)
```

### FastAPI Integration
```python
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter

app = FastAPI()

async def get_context():
    return {
        "db": DatabaseConnection(),
        "user": get_current_user(),
        "dataloaders": create_dataloaders(),
    }

graphql_router = GraphQLRouter(
    schema=schema,
    context_getter=get_context,
)

app.include_router(graphql_router, prefix="/graphql")
```

---

## 2. DataLoader - Solving N+1 Problem

### The N+1 Problem Explained
```
Query: { posts { author { name } } }

Without DataLoader (N+1):
  1 query: SELECT * FROM posts           → 10 posts
  10 queries: SELECT * FROM users WHERE id = ?  (one per post)
  Total: 11 queries ❌

With DataLoader (batched):
  1 query: SELECT * FROM posts           → 10 posts  
  1 query: SELECT * FROM users WHERE id IN (1,2,3...)
  Total: 2 queries ✅
```

### Strawberry DataLoader Implementation
```python
from strawberry.dataloader import DataLoader
from typing import List

# Step 1: Define batch loading function
async def load_users_batch(keys: List[int]) -> List[User]:
    """
    CRITICAL: Return order MUST match input keys order!
    If user not found, return None at that position.
    """
    users = await db.fetch_users_by_ids(keys)
    
    # Create lookup dict for O(1) access
    user_map = {u.id: u for u in users}
    
    # Return in EXACT order of input keys
    return [user_map.get(key) for key in keys]

# Step 2: Create DataLoader instance (per-request!)
def create_dataloaders():
    return {
        "users": DataLoader(load_fn=load_users_batch),
        "posts": DataLoader(load_fn=load_posts_batch),
    }

# Step 3: Use in resolver
@strawberry.type
class Post:
    id: strawberry.ID
    title: str
    author_id: strawberry.ID
    
    @strawberry.field
    async def author(self, info: strawberry.Info) -> User:
        loader = info.context["dataloaders"]["users"]
        return await loader.load(self.author_id)
```

### DataLoader Best Practices

```python
# ✅ CORRECT: One DataLoader per request (in context)
async def get_context():
    return {"user_loader": DataLoader(load_users_batch)}

# ❌ WRONG: Shared DataLoader across requests (cache pollution)
global_loader = DataLoader(load_users_batch)  # Don't do this!

# ✅ CORRECT: Handle missing items
async def load_users_batch(keys: List[int]) -> List[Optional[User]]:
    users = await db.fetch_users_by_ids(keys)
    user_map = {u.id: u for u in users}
    return [user_map.get(key) for key in keys]  # None for missing

# ✅ CORRECT: Prime cache for known data
loader = info.context["user_loader"]
loader.prime(user.id, user)  # Pre-populate cache

# ✅ CORRECT: Clear specific key
loader.clear(user_id)  # After mutation
```

### SQLAlchemy DataLoader Pattern
```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

async def load_users_batch(keys: List[int], session: AsyncSession) -> List[User]:
    stmt = select(UserModel).where(UserModel.id.in_(keys))
    result = await session.execute(stmt)
    users = result.scalars().all()
    
    user_map = {u.id: User.from_orm(u) for u in users}
    return [user_map.get(key) for key in keys]

# Factory that captures session
def create_user_loader(session: AsyncSession):
    async def loader(keys: List[int]) -> List[User]:
        return await load_users_batch(keys, session)
    return DataLoader(load_fn=loader)
```

---

## 3. GraphQL Subscriptions - Real-Time Data

### Strawberry Subscription Definition
```python
import asyncio
from typing import AsyncGenerator

@strawberry.type
class Subscription:
    @strawberry.subscription
    async def post_created(self) -> AsyncGenerator[Post, None]:
        """Stream new posts as they're created."""
        pubsub = get_pubsub()
        async for message in pubsub.subscribe("posts:created"):
            yield Post(**message)
    
    @strawberry.subscription
    async def count(self, target: int = 100) -> AsyncGenerator[int, None]:
        """Simple counter subscription."""
        for i in range(target):
            yield i
            await asyncio.sleep(1)

schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription,
)
```

### WebSocket Transport (FastAPI)
```python
from strawberry.fastapi import GraphQLRouter
from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL, GRAPHQL_WS_PROTOCOL

graphql_router = GraphQLRouter(
    schema=schema,
    subscription_protocols=[
        GRAPHQL_TRANSPORT_WS_PROTOCOL,  # graphql-ws (recommended)
        GRAPHQL_WS_PROTOCOL,            # subscriptions-transport-ws (legacy)
    ],
)
```

### Redis PubSub for Scalable Subscriptions
```python
import aioredis
from typing import AsyncGenerator

class RedisPubSub:
    def __init__(self, redis_url: str = "redis://localhost"):
        self.redis = aioredis.from_url(redis_url)
        self.pubsub = self.redis.pubsub()
    
    async def publish(self, channel: str, message: dict):
        await self.redis.publish(channel, json.dumps(message))
    
    async def subscribe(self, channel: str) -> AsyncGenerator[dict, None]:
        await self.pubsub.subscribe(channel)
        async for message in self.pubsub.listen():
            if message["type"] == "message":
                yield json.loads(message["data"])
    
    async def close(self):
        await self.pubsub.unsubscribe()
        await self.redis.close()

# Usage in subscription
@strawberry.subscription
async def order_updates(self, order_id: str) -> AsyncGenerator[Order, None]:
    pubsub = RedisPubSub()
    try:
        async for data in pubsub.subscribe(f"orders:{order_id}"):
            yield Order(**data)
    finally:
        await pubsub.close()
```

---

## 4. Apollo Federation - Distributed Schemas

### Strawberry Federation Subgraph
```python
import strawberry
from strawberry.federation import Schema

@strawberry.federation.type(keys=["id"])
class User:
    id: strawberry.ID = strawberry.federation.field(external=True)
    
    @classmethod
    def resolve_reference(cls, id: strawberry.ID, info: strawberry.Info) -> "User":
        # Called by router to resolve entity references
        return info.context["db"].get_user(id)

@strawberry.federation.type(keys=["id"])
class Post:
    id: strawberry.ID
    title: str
    content: str
    
    @strawberry.field
    def author(self) -> User:
        # Returns reference, actual User resolved by Users subgraph
        return User(id=self.author_id)

@strawberry.type
class Query:
    @strawberry.field
    def posts(self) -> list[Post]:
        return get_all_posts()

# Federation-aware schema
schema = Schema(query=Query, enable_federation_2=True)
```

### Entity Resolution Pattern
```python
@strawberry.federation.type(keys=["id", "sku"])
class Product:
    id: strawberry.ID
    sku: str
    name: str
    price: float
    
    @classmethod
    def resolve_reference(
        cls, 
        info: strawberry.Info,
        id: strawberry.ID = None,
        sku: str = None,
    ) -> "Product":
        """
        Federation router calls this with key fields.
        Support multiple key combinations.
        """
        db = info.context["db"]
        if id:
            return db.get_product_by_id(id)
        elif sku:
            return db.get_product_by_sku(sku)
        raise ValueError("Must provide id or sku")
```

### Federated Subscriptions (Apollo Router)
```yaml
# router.yaml
supergraph:
  listen: 0.0.0.0:4000

subscription:
  enabled: true
  mode:
    callback:
      # HTTP callback protocol (scalable)
      public_url: https://router.example.com
      listen: 0.0.0.0:4001
      path: /callback
    # Or WebSocket passthrough
    passthrough:
      subgraphs:
        - notifications  # Route to notifications subgraph
```

---

## 5. Performance Patterns

### Query Complexity Limiting
```python
from strawberry.extensions import QueryDepthLimiter
from strawberry.extensions.query_depth_limiter import should_ignore_field

schema = strawberry.Schema(
    query=Query,
    extensions=[
        QueryDepthLimiter(max_depth=10),
    ],
)

# Custom complexity calculation
@strawberry.type
class Query:
    @strawberry.field(
        extensions=[
            strawberry.extensions.FieldExtension(
                complexity=lambda info, args: args.get("limit", 10) * 2
            )
        ]
    )
    def posts(self, limit: int = 10) -> list[Post]:
        return get_posts(limit=limit)
```

### Persisted Queries
```python
from strawberry.extensions import PersistedQueriesExtension

# Client sends hash instead of full query
persisted_queries = {
    "abc123": "query GetUser($id: ID!) { user(id: $id) { name } }",
    "def456": "query GetPosts { posts { title } }",
}

schema = strawberry.Schema(
    query=Query,
    extensions=[
        PersistedQueriesExtension(
            query_map=persisted_queries,
            # Reject non-persisted queries in production
            reject_unknown=True,
        ),
    ],
)
```

### Response Caching
```python
from strawberry.extensions import ResponseCacheExtension

schema = strawberry.Schema(
    query=Query,
    extensions=[
        ResponseCacheExtension(
            max_size=1000,
            ttl=300,  # 5 minutes
        ),
    ],
)

# Field-level cache hints
@strawberry.type
class Query:
    @strawberry.field(cache_control=strawberry.CacheControl(max_age=3600))
    def static_content(self) -> str:
        return get_static_content()
```

---

## 6. Error Handling

### Typed Errors with Union Types
```python
@strawberry.type
class UserNotFound:
    message: str = "User not found"
    user_id: strawberry.ID

@strawberry.type
class PermissionDenied:
    message: str = "Permission denied"
    required_role: str

@strawberry.type  
class UserSuccess:
    user: User

UserResult = strawberry.union("UserResult", [UserSuccess, UserNotFound, PermissionDenied])

@strawberry.type
class Query:
    @strawberry.field
    async def user(self, id: strawberry.ID, info: strawberry.Info) -> UserResult:
        current_user = info.context["user"]
        if not current_user.can_view_users:
            return PermissionDenied(required_role="admin")
        
        user = await get_user(id)
        if not user:
            return UserNotFound(user_id=id)
        
        return UserSuccess(user=user)
```

### Error Extensions
```python
from strawberry.types import Info
from graphql import GraphQLError

class NotFoundError(Exception):
    def __init__(self, resource: str, id: str):
        self.resource = resource
        self.id = id
        super().__init__(f"{resource} {id} not found")

# Custom error formatter
def format_graphql_error(error: GraphQLError, debug: bool) -> dict:
    formatted = error.formatted
    if isinstance(error.original_error, NotFoundError):
        formatted["extensions"] = {
            "code": "NOT_FOUND",
            "resource": error.original_error.resource,
            "id": error.original_error.id,
        }
    return formatted

schema = strawberry.Schema(
    query=Query,
    config=strawberry.SchemaConfig(
        format_error=format_graphql_error,
    ),
)
```

---

## 7. Testing GraphQL APIs

### Strawberry Test Client
```python
import pytest
from strawberry.test import GraphQLTestClient

@pytest.fixture
def graphql_client():
    return GraphQLTestClient(schema)

async def test_get_user(graphql_client):
    query = """
        query GetUser($id: ID!) {
            user(id: $id) {
                id
                name
                email
            }
        }
    """
    result = await graphql_client.query(query, variables={"id": "1"})
    
    assert result.errors is None
    assert result.data["user"]["name"] == "John Doe"

async def test_user_not_found(graphql_client):
    query = """
        query GetUser($id: ID!) {
            user(id: $id) {
                ... on UserNotFound {
                    message
                    userId
                }
                ... on UserSuccess {
                    user { name }
                }
            }
        }
    """
    result = await graphql_client.query(query, variables={"id": "999"})
    
    assert result.data["user"]["__typename"] == "UserNotFound"
```

### Subscription Testing
```python
async def test_subscription():
    async with graphql_client.subscribe(
        """
        subscription {
            postCreated {
                id
                title
            }
        }
        """
    ) as subscription:
        # Trigger event
        await create_post(title="Test Post")
        
        # Receive update
        result = await subscription.__anext__()
        assert result.data["postCreated"]["title"] == "Test Post"
```

---

## Decision Matrix

| Scenario | Pattern |
|----------|---------|
| Python 3.10+ new project | Strawberry (type-first) |
| Legacy Python project | Graphene or Ariadne |
| Nested relations (N+1) | DataLoader per-request |
| Real-time updates | Subscriptions + Redis PubSub |
| Multi-team/microservices | Apollo Federation |
| Public API | Persisted queries + complexity limits |
| High-traffic | Response caching + CDN |

## Anti-Patterns

❌ **Global DataLoader** - Causes cache pollution across requests
❌ **Synchronous resolvers** - Blocks event loop, kills performance
❌ **Unbounded queries** - No depth/complexity limits = DoS vector
❌ **N+1 in nested fields** - Always use DataLoader for relations
❌ **Monolithic schema** - Use federation for large teams/services
❌ **String error messages** - Use typed union errors for clients

---

*Cycle 39 Complete - GraphQL patterns for production Python APIs*
*Strawberry + DataLoader + Subscriptions + Federation*
