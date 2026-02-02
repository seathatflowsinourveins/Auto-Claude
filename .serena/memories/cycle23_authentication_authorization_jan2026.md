# Cycle 23: Authentication & Authorization Patterns (January 2026)

## OAuth 2.1 - The New Standard

### Key Changes from OAuth 2.0
- **Implicit Flow DEPRECATED**: Responsible for 23% of OAuth breaches (Auth0 2025 report)
- **PKCE MANDATORY**: Required for ALL clients (public AND confidential)
- **Password Grant (ROPC) REMOVED**: No more username/password grants
- **Enforcement**: Google, Microsoft, Okta deadline Q2 2026

### Authorization Code + PKCE Flow

```python
import hashlib
import base64
import secrets
from urllib.parse import urlencode

# Step 1: Generate PKCE verifier and challenge
def generate_pkce():
    code_verifier = secrets.token_urlsafe(32)  # 43-128 chars
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).rstrip(b'=').decode()
    return code_verifier, code_challenge

# Step 2: Authorization request
verifier, challenge = generate_pkce()
auth_params = {
    'response_type': 'code',
    'client_id': CLIENT_ID,
    'redirect_uri': REDIRECT_URI,
    'scope': 'openid profile email',
    'state': secrets.token_urlsafe(16),
    'code_challenge': challenge,
    'code_challenge_method': 'S256'  # MUST be S256, not plain
}
auth_url = f"{AUTH_SERVER}/authorize?{urlencode(auth_params)}"

# Step 3: Exchange code for tokens (backend)
async def exchange_code(code: str, verifier: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{AUTH_SERVER}/token",
            data={
                'grant_type': 'authorization_code',
                'code': code,
                'redirect_uri': REDIRECT_URI,
                'client_id': CLIENT_ID,
                'client_secret': CLIENT_SECRET,  # For confidential clients
                'code_verifier': verifier  # PKCE proof
            }
        )
        return response.json()
```

### MCP OAuth 2.1 Integration (March 2025+)

```python
# MCP servers now support OAuth 2.1 natively
# Replaces API key authentication

from mcp import MCPClient

client = MCPClient(
    auth_type='oauth2',
    authorization_server='https://auth.example.com',
    client_id=CLIENT_ID,
    scopes=['mcp:read', 'mcp:write']
)
```

---

## JWT Security Best Practices

### Token Structure

```
Header.Payload.Signature

Header: {"alg": "RS256", "typ": "JWT", "kid": "key-id-1"}
Payload: {"sub": "user123", "iss": "api.example.com", "aud": "client-app", 
          "exp": 1706234567, "iat": 1706230967, "nbf": 1706230967}
Signature: RSASSA-PKCS1-v1_5(SHA256(base64(header) + "." + base64(payload)))
```

### Algorithm Selection

| Algorithm | Use Case | Security |
|-----------|----------|----------|
| RS256 | Production APIs | Asymmetric, public key verification |
| ES256 | High-security, modern | ECDSA, smaller keys |
| HS256 | Internal only | Symmetric, shared secret (AVOID in prod) |
| none | NEVER | Algorithm confusion attack vector |

### Validation Checklist (CRITICAL)

```python
import jwt
from datetime import datetime, timezone

def validate_token(token: str, public_key: str) -> dict:
    try:
        payload = jwt.decode(
            token,
            public_key,
            algorithms=['RS256', 'ES256'],  # Explicit allowlist!
            audience='expected-audience',    # MUST validate
            issuer='expected-issuer',        # MUST validate
            options={
                'require': ['exp', 'iat', 'sub', 'iss', 'aud'],
                'verify_exp': True,
                'verify_iat': True,
                'verify_nbf': True
            }
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthError("Token expired")
    except jwt.InvalidAudienceError:
        raise AuthError("Invalid audience")
    except jwt.InvalidIssuerError:
        raise AuthError("Invalid issuer")
    except jwt.InvalidTokenError as e:
        raise AuthError(f"Invalid token: {e}")
```

### Token Lifetime Strategy

```python
# Access Token: Short-lived (15 minutes)
ACCESS_TOKEN_EXPIRY = 15 * 60  # 900 seconds

# Refresh Token: Longer-lived (7 days) + rotation
REFRESH_TOKEN_EXPIRY = 7 * 24 * 60 * 60

# Token rotation on refresh (security best practice)
async def refresh_tokens(refresh_token: str) -> dict:
    # Validate refresh token
    payload = validate_refresh_token(refresh_token)
    
    # Revoke old refresh token (one-time use)
    await revoke_token(refresh_token)
    
    # Issue new token pair
    new_access = create_access_token(payload['sub'])
    new_refresh = create_refresh_token(payload['sub'])
    
    return {'access_token': new_access, 'refresh_token': new_refresh}
```

### Storage Security

```javascript
// WRONG: localStorage (XSS vulnerable)
localStorage.setItem('token', jwt);

// CORRECT: httpOnly cookie (for web apps)
// Set-Cookie: token=jwt; HttpOnly; Secure; SameSite=Strict; Path=/

// For SPAs: Token Handler Pattern (BFF)
// Browser ↔ BFF (cookies) ↔ API (JWT)
```

---

## Authorization Models

### RBAC (Role-Based Access Control)

```python
# Using Casbin for RBAC
import casbin

# Model definition (rbac_model.conf)
"""
[request_definition]
r = sub, obj, act

[policy_definition]
p = sub, obj, act

[role_definition]
g = _, _

[policy_effect]
e = some(where (p.eft == allow))

[matchers]
m = g(r.sub, p.sub) && r.obj == p.obj && r.act == p.act
"""

# Initialize enforcer
enforcer = casbin.Enforcer("rbac_model.conf", "policy.csv")

# Check permission
if enforcer.enforce("alice", "data1", "read"):
    # Access granted
    pass

# Add role assignment
enforcer.add_grouping_policy("alice", "admin")

# Add permission to role
enforcer.add_policy("admin", "data1", "write")
```

### ABAC (Attribute-Based Access Control)

```python
# Casbin ABAC with attributes
"""
[matchers]
m = r.sub.Department == r.obj.Department && 
    r.sub.Level >= r.obj.RequiredLevel &&
    r.act in r.obj.AllowedActions
"""

# Request with attributes
class Subject:
    def __init__(self, department: str, level: int):
        self.Department = department
        self.Level = level

class Resource:
    def __init__(self, department: str, required_level: int, actions: list):
        self.Department = department
        self.RequiredLevel = required_level
        self.AllowedActions = actions

user = Subject("engineering", 5)
doc = Resource("engineering", 3, ["read", "write"])

enforcer.enforce(user, doc, "read")  # True
```

### ReBAC (Relationship-Based Access Control)

```python
# For complex relationships (Google Zanzibar-style)
# Using OpenFGA or SpiceDB

from openfga_sdk import OpenFgaClient

client = OpenFgaClient(api_url="http://localhost:8080")

# Define relationship tuples
await client.write({
    "writes": {
        "tuple_keys": [
            {"user": "user:alice", "relation": "owner", "object": "doc:readme"},
            {"user": "user:bob", "relation": "viewer", "object": "doc:readme"}
        ]
    }
})

# Check permission (relationship traversal)
result = await client.check({
    "tuple_key": {
        "user": "user:alice",
        "relation": "can_edit",
        "object": "doc:readme"
    }
})
# Returns: {"allowed": true}
```

### FastAPI Integration Pattern

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import casbin

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
enforcer = casbin.Enforcer("model.conf", "policy.csv")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = validate_token(token)
    return payload

def require_permission(resource: str, action: str):
    async def permission_checker(user = Depends(get_current_user)):
        if not enforcer.enforce(user['sub'], resource, action):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )
        return user
    return permission_checker

@app.get("/admin/users")
async def list_users(user = Depends(require_permission("users", "read"))):
    return {"users": [...]}

@app.delete("/admin/users/{user_id}")
async def delete_user(
    user_id: str,
    user = Depends(require_permission("users", "delete"))
):
    # Only users with delete permission reach here
    pass
```

---

## Security Anti-Patterns

### Token Security
- **NEVER** use `alg: none` or allow algorithm switching
- **NEVER** store tokens in localStorage (XSS vulnerable)
- **NEVER** include sensitive data in JWT payload (base64 ≠ encryption)
- **NEVER** use HS256 with guessable secrets

### OAuth Security
- **NEVER** use Implicit Flow (deprecated in OAuth 2.1)
- **NEVER** skip state parameter (CSRF protection)
- **NEVER** use plain code_challenge_method (must be S256)
- **NEVER** expose client_secret in frontend code

### Authorization Security
- **NEVER** rely solely on client-side permission checks
- **NEVER** use sequential/guessable resource IDs without authz
- **NEVER** cache authorization decisions without TTL
- **NEVER** skip permission check in "admin" endpoints

---

## Production Checklist

- [ ] PKCE implemented for all OAuth flows
- [ ] JWT validation includes: exp, iss, aud, signature
- [ ] Refresh token rotation enabled
- [ ] Tokens stored in httpOnly cookies (web) or secure storage (mobile)
- [ ] Authorization checks on every protected endpoint
- [ ] Rate limiting on auth endpoints
- [ ] Audit logging for permission changes
- [ ] Regular key rotation schedule

*Research Date: January 25, 2026*
*Sources: Auth0, Curity, Casbin docs, OAuth 2.1 spec, RFC 7519*
