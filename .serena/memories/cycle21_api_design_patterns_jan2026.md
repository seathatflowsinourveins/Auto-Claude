# Cycle 21: API Design Patterns - REST, GraphQL, gRPC (January 2026)

## Research Focus
Production API design patterns across the three dominant paradigms: REST, GraphQL, and gRPC.

---

## 1. REST API Best Practices (2026)

### Resource-Oriented Design
```
# GOOD: Nouns for resources, HTTP verbs for actions
GET    /users           # List users
GET    /users/123       # Get user 123
POST   /users           # Create user
PUT    /users/123       # Replace user 123
PATCH  /users/123       # Partial update
DELETE /users/123       # Delete user 123

# BAD: Verbs in URLs
POST /createUser
GET  /getUserById/123
```

### HTTP Status Codes (Use Correctly)
```python
# Success
200  # OK - GET, PUT, PATCH success
201  # Created - POST success
204  # No Content - DELETE success

# Client Errors
400  # Bad Request - validation failed
401  # Unauthorized - not authenticated
403  # Forbidden - authenticated but not authorized
404  # Not Found - resource doesn't exist
409  # Conflict - duplicate or state conflict
422  # Unprocessable Entity - semantic errors

# Server Errors
500  # Internal Server Error
502  # Bad Gateway
503  # Service Unavailable
504  # Gateway Timeout
```

### API Versioning Strategies
```python
# 1. URL Path (Most common, explicit)
GET /v1/users
GET /v2/users

# 2. Header-based (Cleaner URLs)
GET /users
Accept: application/vnd.api+json; version=2

# 3. Query parameter (Simple, cacheable)
GET /users?version=2

# Recommendation: URL path for public APIs, headers for internal
```

### Pagination Patterns

**Offset-Based** (Simple, but performance degrades):
```json
GET /users?limit=20&offset=100

{
  "data": [...],
  "meta": {
    "total": 1000,
    "limit": 20,
    "offset": 100,
    "next": "/users?limit=20&offset=120"
  }
}
```

**Cursor-Based** (Stable, performant):
```json
GET /users?limit=20&cursor=eyJpZCI6MTIzfQ

{
  "data": [...],
  "meta": {
    "next_cursor": "eyJpZCI6MTQzfQ",
    "has_more": true
  }
}
```

**Keyset-Based** (Best performance):
```sql
-- Instead of OFFSET (slow)
SELECT * FROM users LIMIT 20 OFFSET 1000

-- Use keyset (fast, uses index)
SELECT * FROM users WHERE id > 1000 LIMIT 20
```

### Idempotency (Critical for POST)
```python
# Client sends idempotency key
POST /payments
Idempotency-Key: abc-123-unique
Content-Type: application/json

{"amount": 100, "currency": "USD"}

# Server implementation
async def create_payment(request):
    key = request.headers.get("Idempotency-Key")
    
    # Check if already processed
    existing = await cache.get(f"idempotency:{key}")
    if existing:
        return existing  # Return cached response
    
    # Process payment
    result = await process_payment(request.json)
    
    # Cache for 24 hours
    await cache.setex(f"idempotency:{key}", 86400, result)
    return result
```

### OpenAPI Specification (40% Integration Time Reduction)
```yaml
openapi: 3.1.0
info:
  title: User API
  version: 1.0.0

paths:
  /users/{id}:
    get:
      summary: Get user by ID
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: User found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '404':
          description: User not found
```

---

## 2. GraphQL Best Practices (2026)

### Relay-Style Connections (Pagination Standard)
```graphql
type Query {
  users(first: Int, after: String, last: Int, before: String): UserConnection!
}

type UserConnection {
  edges: [UserEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type UserEdge {
  node: User!
  cursor: String!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}
```

### Demand-Oriented Schema Design
```graphql
# BAD: Database-oriented (exposes internal structure)
type User {
  user_id: Int!
  first_name: String
  last_name: String
  address_id: Int
}

# GOOD: Demand-oriented (what clients need)
type User {
  id: ID!
  fullName: String!
  displayName: String!
  address: Address
}
```

### Global Object Identification
```graphql
# Every node has globally unique ID
interface Node {
  id: ID!
}

type User implements Node {
  id: ID!  # e.g., "User:123" base64 encoded
  name: String!
}

type Query {
  node(id: ID!): Node  # Fetch any object by ID
}
```

### N+1 Prevention with DataLoader
```python
from aiodataloader import DataLoader

# BAD: N+1 queries
async def resolve_posts(user):
    return await db.get_posts(user.id)  # Called N times!

# GOOD: DataLoader batches
user_posts_loader = DataLoader(
    lambda user_ids: db.get_posts_batch(user_ids)
)

async def resolve_posts(user, info):
    return await info.context.loaders.user_posts.load(user.id)
```

### Federation (Apollo 2026)
```graphql
# Users subgraph
type User @key(fields: "id") {
  id: ID!
  name: String!
}

# Orders subgraph (references User)
type Order {
  id: ID!
  user: User!  # Resolved via federation
}

extend type User @key(fields: "id") {
  id: ID! @external
  orders: [Order!]!
}
```

### Security Best Practices
```python
# 1. Query depth limiting
from graphene import Schema
schema = Schema(query=Query, max_depth=10)

# 2. Query complexity analysis
@complexity(lambda: 1)  # Base cost
def resolve_user(root, info, id):
    return get_user(id)

@complexity(lambda first: first * 2)  # Cost scales with pagination
def resolve_users(root, info, first):
    return get_users(first)

# 3. Persisted queries (block arbitrary queries)
ALLOWED_QUERIES = {
    "abc123": "query GetUser($id: ID!) { user(id: $id) { name } }"
}
```

---

## 3. gRPC Best Practices (2026)

### Protobuf Service Definition
```protobuf
syntax = "proto3";

package user.v1;

service UserService {
  // Unary RPC
  rpc GetUser(GetUserRequest) returns (User);
  
  // Server streaming
  rpc ListUsers(ListUsersRequest) returns (stream User);
  
  // Client streaming
  rpc UploadUsers(stream User) returns (UploadResponse);
  
  // Bidirectional streaming
  rpc Chat(stream Message) returns (stream Message);
}

message User {
  string id = 1;
  string name = 2;
  string email = 3;
}
```

### Python gRPC Client
```python
import grpc
from user_pb2 import GetUserRequest
from user_pb2_grpc import UserServiceStub

# Reuse channel and stub (CRITICAL for performance)
channel = grpc.insecure_channel('localhost:50051')
stub = UserServiceStub(channel)

# Unary call
response = stub.GetUser(GetUserRequest(id="123"))

# With timeout
response = stub.GetUser(
    GetUserRequest(id="123"),
    timeout=5.0  # seconds
)

# Async client
async with grpc.aio.insecure_channel('localhost:50051') as channel:
    stub = UserServiceStub(channel)
    response = await stub.GetUser(GetUserRequest(id="123"))
```

### Interceptors (Middleware)
```python
class LoggingInterceptor(grpc.UnaryUnaryClientInterceptor):
    def intercept_unary_unary(self, continuation, client_call_details, request):
        start = time.time()
        response = continuation(client_call_details, request)
        duration = time.time() - start
        logger.info(f"{client_call_details.method} took {duration:.3f}s")
        return response

# Apply interceptor
channel = grpc.intercept_channel(
    grpc.insecure_channel('localhost:50051'),
    LoggingInterceptor()
)
```

### Error Handling
```python
from grpc import StatusCode

# Server-side
def GetUser(self, request, context):
    user = db.get_user(request.id)
    if not user:
        context.set_code(StatusCode.NOT_FOUND)
        context.set_details(f"User {request.id} not found")
        return User()
    return user

# Client-side
try:
    response = stub.GetUser(GetUserRequest(id="123"))
except grpc.RpcError as e:
    if e.code() == StatusCode.NOT_FOUND:
        print(f"User not found: {e.details()}")
    elif e.code() == StatusCode.UNAVAILABLE:
        print("Service unavailable, retry later")
    else:
        raise
```

### gRPC Status Codes
| Code | Name | Use Case |
|------|------|----------|
| 0 | OK | Success |
| 1 | CANCELLED | Client cancelled |
| 3 | INVALID_ARGUMENT | Bad request |
| 5 | NOT_FOUND | Resource missing |
| 7 | PERMISSION_DENIED | Not authorized |
| 8 | RESOURCE_EXHAUSTED | Rate limited |
| 13 | INTERNAL | Server error |
| 14 | UNAVAILABLE | Service down (retry) |

### Retry Configuration
```python
# Service config with retry policy
service_config = {
    "methodConfig": [{
        "name": [{"service": "user.v1.UserService"}],
        "retryPolicy": {
            "maxAttempts": 3,
            "initialBackoff": "0.1s",
            "maxBackoff": "1s",
            "backoffMultiplier": 2,
            "retryableStatusCodes": ["UNAVAILABLE"]
        }
    }]
}

channel = grpc.insecure_channel(
    'localhost:50051',
    options=[('grpc.service_config', json.dumps(service_config))]
)
```

### Keepalive for Long-Lived Connections
```python
channel = grpc.insecure_channel(
    'localhost:50051',
    options=[
        ('grpc.keepalive_time_ms', 30000),        # Ping every 30s
        ('grpc.keepalive_timeout_ms', 5000),       # Wait 5s for pong
        ('grpc.keepalive_permit_without_calls', True),  # Ping even when idle
    ]
)
```

---

## 4. Protocol Selection Guide

| Criterion | REST | GraphQL | gRPC |
|-----------|------|---------|------|
| **Use When** | Public APIs, simple CRUD | Flexible queries, mobile apps | Internal microservices |
| **Strengths** | Universal, cacheable | No over-fetching, typed | Fast, streaming, typed |
| **Weaknesses** | Over-fetching, versioning | N+1, complexity | Not browser-native |
| **Caching** | HTTP native | Requires effort | Manual |
| **File Upload** | Multipart | Separate endpoint | Streaming |
| **Browser** | Native fetch | Native fetch | gRPC-Web required |

---

## 5. Anti-Patterns to Avoid

### REST
1. Verbs in URLs (`/createUser`)
2. Ignoring HTTP status codes
3. No pagination on list endpoints
4. Breaking changes without versioning

### GraphQL
1. Exposing database schema directly
2. No query depth/complexity limits
3. Missing DataLoader (N+1)
4. Mutations without proper error types

### gRPC
1. Creating new channels per request
2. No timeout on calls
3. Ignoring keepalive for long connections
4. Not handling specific error codes

---

## Key Statistics (2026)
- OpenAPI reduces integration time by **40%**
- **83%** of API-first companies see faster development
- Proper API design reduces security vulnerabilities by **65%**
- gRPC is **2-10x faster** than REST for internal services

---

## Sources
- graphql.org official best practices (2026)
- grpc.io performance guide
- IBM API Design Tutorial (Jan 2026)
- Apollo GraphQL Federation docs
- Postman gRPC error codes guide

*Researched: January 2026 | Cycle 21*
