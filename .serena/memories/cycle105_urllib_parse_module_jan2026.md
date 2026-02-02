# urllib.parse Module - Production Patterns (January 2026)

## Quick Reference

```python
from urllib.parse import (
    urlparse, urlsplit, urlunparse, urlunsplit,  # URL parsing
    parse_qs, parse_qsl,                          # Query string parsing
    urljoin,                                       # URL joining
    quote, quote_plus, unquote, unquote_plus,     # URL encoding
    urlencode,                                     # Dict to query string
    urldefrag,                                     # Fragment removal
)
```

## URL Parsing Functions

### urlparse vs urlsplit

```python
from urllib.parse import urlparse, urlsplit

url = "https://user:pass@example.com:8080/path?query=value#fragment"

# urlparse - splits params from path (legacy ; separator)
result = urlparse(url)
# ParseResult(scheme='https', netloc='user:pass@example.com:8080', 
#             path='/path', params='', query='query=value', fragment='fragment')

# urlsplit - no params splitting (faster, recommended for modern URLs)
result = urlsplit(url)
# SplitResult(scheme='https', netloc='user:pass@example.com:8080',
#             path='/path', query='query=value', fragment='fragment')

# Access components
result.scheme      # 'https'
result.netloc      # 'user:pass@example.com:8080'
result.hostname    # 'example.com'
result.port        # 8080
result.username    # 'user'
result.password    # 'pass'
result.path        # '/path'
result.query       # 'query=value'
result.fragment    # 'fragment'

# Convert back to URL
from urllib.parse import urlunparse, urlunsplit
urlunparse(result)  # For ParseResult
urlunsplit(result)  # For SplitResult
```

### Named Tuple Modification

```python
from urllib.parse import urlparse

parsed = urlparse("https://example.com/old-path")

# Create modified URL using _replace
new_parsed = parsed._replace(path="/new-path", query="foo=bar")
new_url = new_parsed.geturl()  # "https://example.com/new-path?foo=bar"
```

## Query String Parsing

### parse_qs - Dict with Lists

```python
from urllib.parse import parse_qs

query = "name=Alice&tags=python&tags=web&empty="

result = parse_qs(query)
# {'name': ['Alice'], 'tags': ['python', 'web']}
# Note: empty values excluded by default

# Include empty values
result = parse_qs(query, keep_blank_values=True)
# {'name': ['Alice'], 'tags': ['python', 'web'], 'empty': ['']}

# Strict parsing (raise on errors)
result = parse_qs(query, strict_parsing=True)
```

### parse_qsl - List of Tuples (Order Preserved)

```python
from urllib.parse import parse_qsl

query = "name=Alice&tags=python&tags=web"

result = parse_qsl(query)
# [('name', 'Alice'), ('tags', 'python'), ('tags', 'web')]

# Useful for preserving parameter order
```

### Max Parameters (Security)

```python
# Python 3.14+: max_num_fields parameter
from urllib.parse import parse_qs

# Prevent DoS attacks with parameter flooding
result = parse_qs(query, max_num_fields=100)  # Raises ValueError if exceeded
```

## URL Encoding/Quoting

### quote vs quote_plus

```python
from urllib.parse import quote, quote_plus

text = "Hello World! @#$%"

# quote - for path components (space -> %20)
quote(text)           # 'Hello%20World%21%20%40%23%24%25'
quote(text, safe='')  # Encode everything including /

# quote_plus - for query values (space -> +)
quote_plus(text)      # 'Hello+World%21+%40%23%24%25'

# Safe characters (not encoded by default)
quote("/path/to/file")        # '/path/to/file' (/ is safe)
quote("/path/to/file", safe='')  # '%2Fpath%2Fto%2Ffile'
```

### unquote vs unquote_plus

```python
from urllib.parse import unquote, unquote_plus

# unquote - decode %XX sequences
unquote("Hello%20World")    # 'Hello World'

# unquote_plus - also decode + as space
unquote_plus("Hello+World")  # 'Hello World'
unquote("Hello+World")       # 'Hello+World' (+ unchanged)
```

### urlencode - Dict to Query String

```python
from urllib.parse import urlencode

params = {
    'name': 'Alice Bob',
    'tags': ['python', 'web'],  # Multiple values
    'page': 1
}

# Standard encoding
urlencode(params, doseq=True)
# 'name=Alice+Bob&tags=python&tags=web&page=1'

# Safe encoding for special characters in values
urlencode(params, doseq=True, quote_via=quote)
# 'name=Alice%20Bob&tags=python&tags=web&page=1'
```

## URL Joining

### urljoin - Combine Base and Relative URLs

```python
from urllib.parse import urljoin

base = "https://example.com/api/v1/"

# Relative paths
urljoin(base, "users")           # 'https://example.com/api/v1/users'
urljoin(base, "./users")         # 'https://example.com/api/v1/users'
urljoin(base, "../v2/users")     # 'https://example.com/api/v2/users'

# Absolute paths (replace entire path)
urljoin(base, "/other")          # 'https://example.com/other'

# Full URLs (completely replace)
urljoin(base, "https://other.com")  # 'https://other.com'

# Query strings
urljoin(base, "?page=2")         # 'https://example.com/api/v1/?page=2'
```

## Fragment Handling

```python
from urllib.parse import urldefrag

url = "https://example.com/page#section"
base_url, fragment = urldefrag(url)
# base_url = 'https://example.com/page'
# fragment = 'section'
```

## Production URL Builder Class

```python
from urllib.parse import urlencode, urljoin, urlparse, urlunparse, quote
from dataclasses import dataclass
from typing import Optional


@dataclass
class URLBuilder:
    """Production-grade URL builder with validation."""
    
    scheme: str = "https"
    host: str = ""
    port: Optional[int] = None
    path: str = ""
    query: dict = None
    fragment: str = ""
    
    def __post_init__(self):
        self.query = self.query or {}
    
    @classmethod
    def from_url(cls, url: str) -> "URLBuilder":
        """Parse existing URL into builder."""
        parsed = urlparse(url)
        from urllib.parse import parse_qs
        return cls(
            scheme=parsed.scheme or "https",
            host=parsed.hostname or "",
            port=parsed.port,
            path=parsed.path,
            query={k: v[0] if len(v) == 1 else v 
                   for k, v in parse_qs(parsed.query).items()},
            fragment=parsed.fragment,
        )
    
    def with_path(self, *segments: str) -> "URLBuilder":
        """Add path segments (auto-quoted)."""
        quoted_segments = [quote(s, safe='') for s in segments]
        new_path = "/" + "/".join(quoted_segments)
        return URLBuilder(
            self.scheme, self.host, self.port,
            new_path, self.query.copy(), self.fragment
        )
    
    def with_query(self, **params) -> "URLBuilder":
        """Add query parameters."""
        new_query = {**self.query, **params}
        return URLBuilder(
            self.scheme, self.host, self.port,
            self.path, new_query, self.fragment
        )
    
    def build(self) -> str:
        """Build final URL string."""
        netloc = self.host
        if self.port:
            netloc = f"{self.host}:{self.port}"
        
        query_string = urlencode(self.query, doseq=True) if self.query else ""
        
        return urlunparse((
            self.scheme,
            netloc,
            self.path,
            "",  # params (legacy)
            query_string,
            self.fragment,
        ))


# Usage
url = (URLBuilder(host="api.example.com")
       .with_path("v1", "users", "alice@example.com")
       .with_query(include="profile", format="json")
       .build())
# 'https://api.example.com/v1/users/alice%40example.com?include=profile&format=json'
```

## Security Patterns

### URL Validation

```python
from urllib.parse import urlparse

def validate_url(url: str, allowed_schemes: set = {"http", "https"}) -> bool:
    """Validate URL for security."""
    try:
        parsed = urlparse(url)
        
        # Check scheme
        if parsed.scheme not in allowed_schemes:
            return False
        
        # Require hostname
        if not parsed.hostname:
            return False
        
        # Reject file:// and javascript:
        if parsed.scheme in ("file", "javascript", "data"):
            return False
        
        return True
    except Exception:
        return False


def sanitize_redirect_url(url: str, allowed_hosts: set) -> Optional[str]:
    """Prevent open redirect vulnerabilities."""
    parsed = urlparse(url)
    
    # Allow relative URLs
    if not parsed.scheme and not parsed.netloc:
        return url
    
    # Check against allowed hosts
    if parsed.hostname in allowed_hosts:
        return url
    
    return None  # Reject external redirects
```

### Safe URL Construction

```python
from urllib.parse import quote, urlencode

def safe_api_url(base: str, path: str, params: dict) -> str:
    """Construct URL with proper encoding."""
    # Quote path segments individually
    safe_path = "/".join(quote(segment, safe='') for segment in path.split("/"))
    
    # Encode query parameters
    query = urlencode(params, doseq=True)
    
    return f"{base.rstrip('/')}/{safe_path}?{query}"
```

## Common Patterns

### Extract Domain

```python
from urllib.parse import urlparse

def get_domain(url: str) -> str:
    """Extract domain from URL."""
    parsed = urlparse(url)
    return parsed.hostname or ""

get_domain("https://sub.example.com:8080/path")  # 'sub.example.com'
```

### Normalize URL

```python
from urllib.parse import urlparse, urlunparse, urlencode, parse_qsl

def normalize_url(url: str) -> str:
    """Normalize URL for comparison/caching."""
    parsed = urlparse(url.lower())
    
    # Remove default ports
    netloc = parsed.hostname or ""
    if parsed.port and parsed.port not in (80, 443):
        netloc = f"{netloc}:{parsed.port}"
    
    # Sort query parameters
    query_params = sorted(parse_qsl(parsed.query))
    query = urlencode(query_params)
    
    # Remove trailing slash (except root)
    path = parsed.path.rstrip('/') or '/'
    
    return urlunparse((parsed.scheme, netloc, path, '', query, ''))
```

### Parse Query with Defaults

```python
from urllib.parse import urlparse, parse_qs

def get_query_param(url: str, key: str, default: str = None) -> str:
    """Get single query parameter with default."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    values = params.get(key, [])
    return values[0] if values else default

get_query_param("https://example.com?page=5", "page", "1")  # '5'
get_query_param("https://example.com", "page", "1")         # '1'
```

## Key Takeaways

1. **urlsplit over urlparse**: Faster, use urlparse only if you need legacy params splitting
2. **quote for paths, quote_plus for query values**: Space encoding differs
3. **parse_qs returns lists**: Even single values are in lists
4. **urljoin behavior**: Absolute paths replace, relative paths join
5. **Always validate user URLs**: Prevent SSRF and open redirects
6. **Use quote(safe='') for path segments**: Encode everything including /
7. **parse_qsl preserves order**: Use for ordered query parameters
8. **max_num_fields (3.14+)**: Prevent query string DoS attacks
