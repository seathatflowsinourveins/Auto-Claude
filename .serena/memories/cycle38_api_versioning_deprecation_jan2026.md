# Cycle 38: API Versioning & Deprecation Strategies (January 2026)

## Overview
Comprehensive patterns for API versioning and deprecation in production systems. Covers versioning strategies (URL, header, query), RFC 9745 Deprecation header, Sunset planning, backward compatibility, and migration strategies.

---

## 1. Versioning Strategy Comparison

| Strategy | Example | Pros | Cons |
|----------|---------|------|------|
| **URL Path** | `/v1/users` | Visible, easy routing, cacheable | URL pollution, hard redirects |
| **Header** | `Accept-Version: v1` | Clean URLs, flexible | Hidden, harder debugging |
| **Query Param** | `/users?version=1` | Easy to test | Pollutes query, caching issues |
| **Media Type** | `Accept: application/vnd.api.v1+json` | RESTful, content negotiation | Complex, harder to implement |

### Recommendation: URL Path for Public APIs
```python
# FastAPI URL Path Versioning (Production Standard)
from fastapi import APIRouter, FastAPI

app = FastAPI()

# Version 1
v1_router = APIRouter(prefix="/v1")

@v1_router.get("/users/{user_id}")
async def get_user_v1(user_id: int):
    return {"id": user_id, "name": "Alice"}

# Version 2 (new response structure)
v2_router = APIRouter(prefix="/v2")

@v2_router.get("/users/{user_id}")
async def get_user_v2(user_id: int):
    return {
        "data": {"id": user_id, "name": "Alice"},
        "meta": {"version": "2.0"}
    }

app.include_router(v1_router)
app.include_router(v2_router)
```

### Header-Based Versioning (Internal APIs)
```python
from fastapi import Request, HTTPException

async def get_api_version(request: Request) -> str:
    version = request.headers.get("X-API-Version", "1")
    if version not in ["1", "2"]:
        raise HTTPException(400, f"Unsupported API version: {version}")
    return version

@app.get("/users/{user_id}")
async def get_user(user_id: int, request: Request):
    version = await get_api_version(request)
    
    if version == "1":
        return {"id": user_id, "name": "Alice"}
    else:
        return {"data": {"id": user_id, "name": "Alice"}}
```

---

## 2. RFC 9745: Deprecation HTTP Header (March 2025 Standard)

### Official Deprecation Header
```http
HTTP/1.1 200 OK
Deprecation: Sat, 01 Feb 2026 00:00:00 GMT
Link: <https://api.example.com/docs/deprecation>; rel="deprecation"
```

### Python Implementation
```python
from datetime import datetime
from fastapi import Response

def add_deprecation_headers(
    response: Response,
    deprecation_date: datetime,
    sunset_date: datetime,
    docs_url: str
):
    """Add RFC 9745 compliant deprecation headers"""
    # Deprecation date (when marked deprecated)
    response.headers["Deprecation"] = deprecation_date.strftime(
        "%a, %d %b %Y %H:%M:%S GMT"
    )
    
    # Sunset date (when it will stop working)
    response.headers["Sunset"] = sunset_date.strftime(
        "%a, %d %b %Y %H:%M:%S GMT"
    )
    
    # Link to deprecation docs
    response.headers["Link"] = f'<{docs_url}>; rel="deprecation"'

# Usage in endpoint
@app.get("/v1/legacy-endpoint")
async def legacy_endpoint(response: Response):
    add_deprecation_headers(
        response,
        deprecation_date=datetime(2026, 1, 1),
        sunset_date=datetime(2026, 7, 1),
        docs_url="https://api.example.com/docs/v1-deprecation"
    )
    return {"data": "legacy response"}
```

### Middleware for Deprecated Endpoints
```python
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime

DEPRECATED_ENDPOINTS = {
    "/v1/users": {
        "deprecation": datetime(2026, 1, 1),
        "sunset": datetime(2026, 7, 1),
        "replacement": "/v2/users"
    }
}

class DeprecationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        
        path = request.url.path
        if path in DEPRECATED_ENDPOINTS:
            info = DEPRECATED_ENDPOINTS[path]
            response.headers["Deprecation"] = info["deprecation"].strftime(
                "%a, %d %b %Y %H:%M:%S GMT"
            )
            response.headers["Sunset"] = info["sunset"].strftime(
                "%a, %d %b %Y %H:%M:%S GMT"
            )
            response.headers["Link"] = (
                f'<{info["replacement"]}>; rel="successor-version"'
            )
        
        return response
```

---

## 3. Breaking vs Non-Breaking Changes

### Non-Breaking (Safe to Deploy)
```python
# ✅ Adding optional fields
class UserResponse(BaseModel):
    id: int
    name: str
    email: str | None = None  # NEW optional field - safe

# ✅ Adding new endpoints
@app.get("/v1/users/{id}/preferences")  # NEW endpoint - safe

# ✅ Widening accepted input types
def process(value: int | str):  # Was int only - safe

# ✅ Adding optional query parameters
@app.get("/users")
async def list_users(
    limit: int = 100,
    include_deleted: bool = False  # NEW optional param - safe
): ...
```

### Breaking Changes (Require New Version)
```python
# ❌ Removing fields
class UserResponseV2(BaseModel):
    id: int
    # name removed - BREAKING

# ❌ Changing field types
class UserResponseV2(BaseModel):
    id: str  # Was int - BREAKING

# ❌ Renaming fields
class UserResponseV2(BaseModel):
    user_id: int  # Was "id" - BREAKING

# ❌ Changing response structure
# v1: {"id": 1, "name": "Alice"}
# v2: {"data": {"id": 1, "name": "Alice"}}  # BREAKING

# ❌ Changing authentication method
# v1: API Key header
# v2: OAuth 2.0 only - BREAKING

# ❌ Changing error response format
# v1: {"error": "message"}
# v2: {"errors": [{"code": "...", "message": "..."}]} - BREAKING
```

---

## 4. Semantic Versioning for APIs

### SemVer Pattern: MAJOR.MINOR.PATCH
```
MAJOR: Breaking changes (increment for v1 → v2)
MINOR: New features, backward compatible
PATCH: Bug fixes, no API changes
```

### Version Header Implementation
```python
from pydantic import BaseModel

class APIVersion(BaseModel):
    major: int
    minor: int
    patch: int
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def is_compatible_with(self, other: "APIVersion") -> bool:
        """Check if versions are compatible (same major)"""
        return self.major == other.major

CURRENT_VERSION = APIVersion(major=2, minor=3, patch=1)

@app.get("/version")
async def get_version():
    return {
        "version": str(CURRENT_VERSION),
        "major": CURRENT_VERSION.major,
        "supported_versions": ["2.x", "1.x (deprecated)"]
    }
```

---

## 5. Deprecation Timeline Best Practices

### Standard Timeline
```
T-12 months: Announce deprecation, new version available
T-6 months:  Add deprecation headers, warnings in logs
T-3 months:  Send reminder emails, increase warning visibility
T-1 month:   Final warning, begin brownouts (scheduled outages)
T-0:         Sunset - return HTTP 410 Gone
T+3 months:  Remove code from codebase
```

### Brownout Pattern (Gradual Shutdown)
```python
import random
from datetime import datetime

BROWNOUT_SCHEDULE = [
    # (start_date, end_date, probability_of_503)
    (datetime(2026, 5, 1), datetime(2026, 5, 2), 0.1),   # 10% errors
    (datetime(2026, 5, 15), datetime(2026, 5, 16), 0.25), # 25% errors
    (datetime(2026, 6, 1), datetime(2026, 6, 2), 0.5),   # 50% errors
]

class BrownoutMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path.startswith("/v1/"):
            now = datetime.now()
            for start, end, probability in BROWNOUT_SCHEDULE:
                if start <= now <= end:
                    if random.random() < probability:
                        return JSONResponse(
                            status_code=503,
                            content={
                                "error": "Service temporarily unavailable",
                                "message": "v1 API is being deprecated. Please migrate to v2.",
                                "migration_guide": "https://api.example.com/migrate"
                            },
                            headers={"Retry-After": "3600"}
                        )
        return await call_next(request)
```

---

## 6. HTTP 410 Gone After Sunset

### Post-Deprecation Response
```python
from fastapi import HTTPException
from datetime import datetime

SUNSET_ENDPOINTS = {
    "/v0/users": datetime(2025, 1, 1),  # Already sunset
    "/v1/legacy": datetime(2026, 7, 1),  # Future sunset
}

@app.middleware("http")
async def sunset_check(request, call_next):
    path = request.url.path
    if path in SUNSET_ENDPOINTS:
        sunset_date = SUNSET_ENDPOINTS[path]
        if datetime.now() >= sunset_date:
            return JSONResponse(
                status_code=410,
                content={
                    "error": "Gone",
                    "message": f"This endpoint was sunset on {sunset_date.date()}",
                    "successor": "/v2/users",
                    "documentation": "https://api.example.com/docs/migration"
                }
            )
    return await call_next(request)
```

---

## 7. API Gateway Version Routing

### Kong/NGINX Version Routing
```yaml
# Kong declarative config
services:
  - name: api-v1
    url: http://api-v1-service:8000
    routes:
      - name: v1-route
        paths:
          - /v1
        
  - name: api-v2
    url: http://api-v2-service:8000
    routes:
      - name: v2-route
        paths:
          - /v2

# Header-based routing
plugins:
  - name: request-transformer
    config:
      add:
        headers:
          - "X-API-Version:$(headers.Accept-Version)"
```

### FastAPI with Versioned Routers
```python
# app/versions/v1/__init__.py
from fastapi import APIRouter
router = APIRouter(prefix="/v1", tags=["v1"])

# app/versions/v2/__init__.py
from fastapi import APIRouter
router = APIRouter(prefix="/v2", tags=["v2"])

# main.py
from app.versions import v1, v2

app = FastAPI()
app.include_router(v1.router)
app.include_router(v2.router)
```

---

## 8. Backward Compatibility Patterns

### Additive Changes with Defaults
```python
from pydantic import BaseModel, Field

class UserCreateV1(BaseModel):
    name: str
    email: str

class UserCreateV2(UserCreateV1):
    """V2 adds optional fields - backward compatible"""
    phone: str | None = None
    preferences: dict = Field(default_factory=dict)

# V1 requests still work with V2 endpoint
@app.post("/v2/users")
async def create_user(user: UserCreateV2):
    # phone and preferences have defaults
    return {"id": 1, **user.model_dump()}
```

### Response Envelope Evolution
```python
# V1 Response (bare)
{"id": 1, "name": "Alice"}

# V2 Response (enveloped) - BREAKING if mandatory
{"data": {"id": 1, "name": "Alice"}, "meta": {"version": "2"}}

# Transition: Accept header determines format
@app.get("/users/{id}")
async def get_user(id: int, request: Request):
    user = {"id": id, "name": "Alice"}
    
    accept = request.headers.get("Accept", "")
    if "vnd.api.v2" in accept:
        return {"data": user, "meta": {"version": "2"}}
    return user  # V1 format (default)
```

### Alias Pattern for Renamed Fields
```python
from pydantic import BaseModel, Field

class UserResponse(BaseModel):
    user_id: int = Field(alias="id")  # New name
    id: int | None = None  # Keep old name for backward compat
    
    class Config:
        populate_by_name = True
    
    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        # Include both old and new field names
        data["id"] = data["user_id"]
        return data
```

---

## 9. Version Discovery & Documentation

### OpenAPI Version Tags
```python
from fastapi import FastAPI

app = FastAPI(
    title="My API",
    version="2.3.1",
    openapi_tags=[
        {"name": "v2", "description": "Current stable version"},
        {"name": "v1", "description": "Deprecated - sunset July 2026"},
    ]
)

# Separate OpenAPI specs per version
@app.get("/openapi-v1.json")
async def openapi_v1():
    return generate_openapi_for_version("v1")

@app.get("/openapi-v2.json")  
async def openapi_v2():
    return generate_openapi_for_version("v2")
```

### Version Negotiation Endpoint
```python
@app.get("/")
async def api_root():
    return {
        "versions": {
            "v2": {
                "status": "current",
                "url": "/v2",
                "docs": "/docs#/v2"
            },
            "v1": {
                "status": "deprecated",
                "sunset": "2026-07-01",
                "url": "/v1",
                "migration_guide": "/docs/v1-to-v2"
            }
        },
        "current_version": "v2"
    }
```

---

## 10. Client Communication Strategy

### Deprecation Notice Channels
1. **HTTP Headers**: Deprecation, Sunset, Link (machine-readable)
2. **Response Body**: Warning field in JSON responses
3. **Email**: Direct notification to API key owners
4. **Developer Portal**: Banner, changelog, migration guides
5. **Webhooks**: Deprecation events for automated monitoring

### Warning in Response Body
```python
def add_deprecation_warning(response: dict, endpoint: str) -> dict:
    if endpoint in DEPRECATED_ENDPOINTS:
        info = DEPRECATED_ENDPOINTS[endpoint]
        response["_warnings"] = [{
            "code": "DEPRECATED_ENDPOINT",
            "message": f"This endpoint is deprecated and will be removed on {info['sunset'].date()}",
            "migration": info["replacement"],
            "docs": f"https://api.example.com/docs/migrate-{endpoint}"
        }]
    return response
```

---

## 11. Testing Version Compatibility

### Contract Testing with Schemathesis
```python
# Test that V2 is backward compatible with V1 requests
import schemathesis

schema_v1 = schemathesis.from_uri("http://localhost:8000/openapi-v1.json")
schema_v2 = schemathesis.from_uri("http://localhost:8000/openapi-v2.json")

@schema_v1.parametrize()
def test_v1_requests_work_on_v2(case):
    """V1 request format should work on V2 endpoints"""
    # Modify URL to hit V2
    case.path = case.path.replace("/v1/", "/v2/")
    response = case.call()
    assert response.status_code < 500
```

### Version Comparison Tests
```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_v1_v2_response_compatibility():
    async with AsyncClient(base_url="http://localhost:8000") as client:
        v1_resp = await client.get("/v1/users/1")
        v2_resp = await client.get("/v2/users/1")
        
        v1_data = v1_resp.json()
        v2_data = v2_resp.json()
        
        # V2 envelope contains V1 fields
        assert v1_data["id"] == v2_data["data"]["id"]
        assert v1_data["name"] == v2_data["data"]["name"]
```

---

## 12. Decision Matrix

| Scenario | Strategy |
|----------|----------|
| Public API, many clients | URL path versioning (/v1/, /v2/) |
| Internal microservices | Header versioning (less URL churn) |
| Single breaking change | New version + 6-month deprecation |
| Minor additions | Add to current version (non-breaking) |
| Major rewrite | New major version, parallel operation |
| Emergency security fix | Patch current version, force upgrade |

---

## 13. Anti-Patterns to Avoid

1. **No version from start** - Always version from v1
2. **Immediate deprecation** - Minimum 6 months notice
3. **Silent breaking changes** - Always communicate changes
4. **Too many active versions** - Max 2-3 concurrently
5. **No migration path** - Provide clear upgrade guides
6. **Version in response body only** - Use headers too (machine-readable)
7. **Deprecating without successor** - New version must exist first
8. **No deprecation metrics** - Track usage before sunsetting
9. **Hard cutoff without brownout** - Gradual degradation helps
10. **Forgetting client SDKs** - SDKs need version updates too

---

## 14. Production Checklist

- [ ] Version strategy chosen (URL path recommended for public)
- [ ] RFC 9745 Deprecation headers implemented
- [ ] Sunset header with specific date
- [ ] Link header to migration documentation
- [ ] Changelog per version maintained
- [ ] Migration guides for each breaking change
- [ ] Brownout schedule planned
- [ ] Client notification system (email, portal)
- [ ] Version usage metrics tracked
- [ ] Contract tests for compatibility
- [ ] SDK updates planned
- [ ] HTTP 410 response after sunset

---

*Cycle 38 - API Versioning & Deprecation | January 2026*
*RFC 9745 Deprecation Header (March 2025) integrated*
