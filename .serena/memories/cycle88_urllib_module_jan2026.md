# Python urllib Module - Production Patterns (Cycle 88)

## Overview

The `urllib` package provides high-level URL handling with four submodules:
- `urllib.request` - Opening URLs (HTTP, HTTPS, FTP, file)
- `urllib.parse` - URL parsing and construction
- `urllib.error` - Exception classes (URLError, HTTPError)
- `urllib.response` - Response wrapper classes

**Note**: For production HTTP clients, prefer `httpx` or `requests`. Use `urllib` for stdlib-only requirements.

## urllib.request Module

### Basic URL Opening

```python
import urllib.request

# Simple GET request
with urllib.request.urlopen('https://api.example.com/data') as response:
    data = response.read()
    print(response.status)       # 200
    print(response.headers)      # Headers object
    print(data.decode('utf-8'))  # Response body

# With timeout
with urllib.request.urlopen('https://api.example.com', timeout=10) as response:
    data = response.read()
```

### Request Object (Custom Headers, Methods)

```python
import urllib.request
import json

# Custom headers
req = urllib.request.Request(
    'https://api.example.com/data',
    headers={
        'User-Agent': 'MyApp/1.0',
        'Authorization': 'Bearer token123',
        'Accept': 'application/json'
    }
)

with urllib.request.urlopen(req) as response:
    data = json.loads(response.read())

# POST request with JSON body
data = json.dumps({'key': 'value'}).encode('utf-8')
req = urllib.request.Request(
    'https://api.example.com/submit',
    data=data,
    headers={'Content-Type': 'application/json'},
    method='POST'
)

with urllib.request.urlopen(req) as response:
    result = response.read()
```

### All HTTP Methods

```python
import urllib.request

# PUT request
req = urllib.request.Request(
    'https://api.example.com/resource/1',
    data=b'{"status": "updated"}',
    method='PUT'
)

# DELETE request
req = urllib.request.Request(
    'https://api.example.com/resource/1',
    method='DELETE'
)

# HEAD request (no body returned)
req = urllib.request.Request(
    'https://api.example.com/resource',
    method='HEAD'
)

# PATCH request
req = urllib.request.Request(
    'https://api.example.com/resource/1',
    data=b'{"field": "new_value"}',
    method='PATCH'
)
```

### Form Data (POST)

```python
import urllib.request
import urllib.parse

# URL-encoded form data
form_data = urllib.parse.urlencode({
    'username': 'john',
    'password': 'secret123',
    'remember': 'true'
}).encode('ascii')

req = urllib.request.Request(
    'https://example.com/login',
    data=form_data,
    headers={'Content-Type': 'application/x-www-form-urlencoded'}
)

with urllib.request.urlopen(req) as response:
    result = response.read()
```

### SSL Context (Custom Certificates)

```python
import urllib.request
import ssl

# Custom SSL context
context = ssl.create_default_context()
context.check_hostname = True
context.verify_mode = ssl.CERT_REQUIRED

# Load custom CA bundle
context.load_verify_locations('/path/to/ca-bundle.crt')

with urllib.request.urlopen('https://api.example.com', context=context) as response:
    data = response.read()

# Client certificate authentication
context = ssl.create_default_context()
context.load_cert_chain('/path/to/client.crt', '/path/to/client.key')
```

### OpenerDirector (Handler Chains)

```python
import urllib.request

# Build custom opener with multiple handlers
opener = urllib.request.build_opener(
    urllib.request.HTTPHandler(),
    urllib.request.HTTPSHandler(),
    urllib.request.ProxyHandler({'http': 'http://proxy:8080'})
)

# Use opener directly
with opener.open('https://api.example.com') as response:
    data = response.read()

# Install as global default
urllib.request.install_opener(opener)
# Now urlopen() uses this opener
```

### Proxy Configuration

```python
import urllib.request

# Explicit proxy
proxy_handler = urllib.request.ProxyHandler({
    'http': 'http://proxy.example.com:8080',
    'https': 'http://proxy.example.com:8080'
})

opener = urllib.request.build_opener(proxy_handler)

# No proxy (override environment)
no_proxy = urllib.request.ProxyHandler({})
opener = urllib.request.build_opener(no_proxy)

# Proxy with authentication
proxy_auth = urllib.request.ProxyBasicAuthHandler()
proxy_auth.add_password('realm', 'proxy.example.com', 'user', 'password')
opener = urllib.request.build_opener(proxy_handler, proxy_auth)
```

### HTTP Authentication

```python
import urllib.request

# Basic authentication
password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
password_mgr.add_password(
    None,  # realm (None = any realm)
    'https://api.example.com/',
    'username',
    'password'
)

auth_handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
opener = urllib.request.build_opener(auth_handler)
urllib.request.install_opener(opener)

# Digest authentication (SHA-256 supported in 3.14+)
digest_handler = urllib.request.HTTPDigestAuthHandler(password_mgr)
opener = urllib.request.build_opener(digest_handler)
```

### Cookie Handling

```python
import urllib.request
import http.cookiejar

# Create cookie jar
cookie_jar = http.cookiejar.CookieJar()
cookie_handler = urllib.request.HTTPCookieProcessor(cookie_jar)

opener = urllib.request.build_opener(cookie_handler)
urllib.request.install_opener(opener)

# Cookies automatically sent on subsequent requests
urllib.request.urlopen('https://example.com/login')
urllib.request.urlopen('https://example.com/dashboard')  # Cookies included

# Persistent cookies (save to file)
cookie_jar = http.cookiejar.MozillaCookieJar('cookies.txt')
try:
    cookie_jar.load()
except FileNotFoundError:
    pass

# ... make requests ...

cookie_jar.save()
```

### File Download with Progress

```python
import urllib.request

def download_progress(block_count, block_size, total_size):
    downloaded = block_count * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        print(f"\rDownloading: {percent:.1f}%", end='')

# Download file with progress callback
local_file, headers = urllib.request.urlretrieve(
    'https://example.com/large-file.zip',
    filename='downloaded.zip',
    reporthook=download_progress
)

print(f"\nSaved to: {local_file}")

# Clean up temporary files
urllib.request.urlcleanup()
```

### Error Handling

```python
import urllib.request
import urllib.error

try:
    with urllib.request.urlopen('https://api.example.com/data') as response:
        data = response.read()
        
except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}: {e.reason}")
    print(f"Headers: {e.headers}")
    error_body = e.read()  # Can read error response body
    
except urllib.error.URLError as e:
    print(f"URL Error: {e.reason}")
    # e.reason may be socket.error for connection issues
    
except TimeoutError:
    print("Request timed out")
```

## urllib.parse Module

### URL Parsing

```python
from urllib.parse import urlparse, urlsplit

# Parse URL into components
url = 'https://user:pass@example.com:8080/path/to/page?key=value#section'
result = urlparse(url)

result.scheme      # 'https'
result.netloc      # 'user:pass@example.com:8080'
result.path        # '/path/to/page'
result.params      # '' (path parameters, rarely used)
result.query       # 'key=value'
result.fragment    # 'section'
result.username    # 'user'
result.password    # 'pass'
result.hostname    # 'example.com'
result.port        # 8080

# urlsplit() - same but doesn't separate params from path
result = urlsplit(url)
# No result.params attribute
```

### Query String Parsing

```python
from urllib.parse import parse_qs, parse_qsl

query = 'name=John&tags=python&tags=web&empty='

# parse_qs - returns dict with lists
params = parse_qs(query)
# {'name': ['John'], 'tags': ['python', 'web']}

# Keep blank values
params = parse_qs(query, keep_blank_values=True)
# {'name': ['John'], 'tags': ['python', 'web'], 'empty': ['']}

# parse_qsl - returns list of tuples
params = parse_qsl(query)
# [('name', 'John'), ('tags', 'python'), ('tags', 'web')]

# With max fields limit (DoS protection)
params = parse_qs(query, max_num_fields=100)
```

### URL Construction

```python
from urllib.parse import urlunparse, urlunsplit, urlencode

# Build URL from components (6-tuple for urlparse)
url = urlunparse((
    'https',           # scheme
    'api.example.com', # netloc
    '/v1/users',       # path
    '',                # params
    'page=1&limit=10', # query
    ''                 # fragment
))
# 'https://api.example.com/v1/users?page=1&limit=10'

# Build URL from components (5-tuple for urlsplit)
url = urlunsplit((
    'https',
    'api.example.com',
    '/v1/users',
    'page=1',
    'section'
))
# 'https://api.example.com/v1/users?page=1#section'
```

### Query String Building

```python
from urllib.parse import urlencode

# From dictionary
params = {'name': 'John Doe', 'age': 30, 'city': 'New York'}
query = urlencode(params)
# 'name=John+Doe&age=30&city=New+York'

# From list of tuples (preserves order, allows duplicates)
params = [('tag', 'python'), ('tag', 'web'), ('sort', 'date')]
query = urlencode(params)
# 'tag=python&tag=web&sort=date'

# With sequence values (doseq=True)
params = {'tags': ['python', 'web', 'api']}
query = urlencode(params, doseq=True)
# 'tags=python&tags=web&tags=api'

# Use quote() instead of quote_plus() for spaces as %20
from urllib.parse import quote
query = urlencode(params, quote_via=quote)
```

### URL Joining

```python
from urllib.parse import urljoin

base = 'https://example.com/path/page.html'

urljoin(base, 'other.html')
# 'https://example.com/path/other.html'

urljoin(base, '/absolute/path')
# 'https://example.com/absolute/path'

urljoin(base, '../sibling/page')
# 'https://example.com/sibling/page'

urljoin(base, '//other.com/path')
# 'https://other.com/path'  # WARNING: Different host!

urljoin(base, 'https://different.com/')
# 'https://different.com/'  # WARNING: Complete override!
```

### URL Quoting (Encoding)

```python
from urllib.parse import quote, quote_plus, quote_from_bytes

# quote() - for path components (/ not encoded by default)
quote('/path/with spaces/file.txt')
# '/path/with%20spaces/file.txt'

quote('/path/special?chars', safe='')  # Encode everything
# '%2Fpath%2Fspecial%3Fchars'

# quote_plus() - for query values (spaces become +)
quote_plus('hello world')
# 'hello+world'

quote_plus('key=value&other')
# 'key%3Dvalue%26other'

# From bytes
quote_from_bytes(b'binary\xffdata')
# 'binary%FFdata'

# Non-ASCII handling
quote('日本語', encoding='utf-8')
# '%E6%97%A5%E6%9C%AC%E8%AA%9E'
```

### URL Unquoting (Decoding)

```python
from urllib.parse import unquote, unquote_plus, unquote_to_bytes

# unquote() - decode %XX escapes
unquote('/path/with%20spaces')
# '/path/with spaces'

unquote('%E6%97%A5%E6%9C%AC%E8%AA%9E')
# '日本語'

# unquote_plus() - also convert + to space
unquote_plus('hello+world')
# 'hello world'

# To bytes
unquote_to_bytes('binary%FFdata')
# b'binary\xffdata'
```

### File Path <-> URL Conversion

```python
from urllib.request import pathname2url, url2pathname

# Path to URL (platform-aware)
pathname2url('/etc/hosts')  # Unix
# '/etc/hosts'

pathname2url('C:\\Users\\name', add_scheme=True)  # Windows (3.14+)
# 'file:///C:/Users/name'

# URL to path
url2pathname('file:///C:/Users/name', require_scheme=True)  # 3.14+
# 'C:\\Users\\name' (on Windows)
```

### URL Fragment Handling

```python
from urllib.parse import urldefrag

url = 'https://example.com/page#section'
result = urldefrag(url)

result.url       # 'https://example.com/page'
result.fragment  # 'section'
```

## Complete Request Example

```python
import urllib.request
import urllib.parse
import urllib.error
import json
import ssl

def api_request(method, url, data=None, headers=None, timeout=30):
    """Production-ready urllib request wrapper."""
    
    # Prepare headers
    default_headers = {
        'User-Agent': 'MyApp/1.0',
        'Accept': 'application/json'
    }
    if headers:
        default_headers.update(headers)
    
    # Prepare data
    body = None
    if data is not None:
        if isinstance(data, dict):
            body = json.dumps(data).encode('utf-8')
            default_headers['Content-Type'] = 'application/json'
        elif isinstance(data, bytes):
            body = data
        else:
            body = str(data).encode('utf-8')
    
    # Create request
    req = urllib.request.Request(
        url,
        data=body,
        headers=default_headers,
        method=method
    )
    
    # SSL context
    context = ssl.create_default_context()
    
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=context) as response:
            return {
                'status': response.status,
                'headers': dict(response.headers),
                'body': response.read().decode('utf-8')
            }
    except urllib.error.HTTPError as e:
        return {
            'status': e.code,
            'headers': dict(e.headers),
            'body': e.read().decode('utf-8'),
            'error': e.reason
        }
    except urllib.error.URLError as e:
        raise ConnectionError(f"Failed to connect: {e.reason}")

# Usage
response = api_request('GET', 'https://api.example.com/users')
response = api_request('POST', 'https://api.example.com/users', 
                       data={'name': 'John'})
```

## Best Practices

1. **Always use context managers** (`with urlopen(...) as response`)
2. **Set timeouts** to prevent hanging connections
3. **Handle both HTTPError and URLError** for complete error coverage
4. **Use SSL context** for certificate verification control
5. **Prefer parse_qs with max_num_fields** for DoS protection
6. **Use quote() for paths, quote_plus() for query values**
7. **Be careful with urljoin()** - it can redirect to different hosts
8. **Consider httpx/requests** for complex HTTP needs
9. **Use urlsplit() over urlparse()** for modern URL handling
10. **Clean up with urlcleanup()** after using urlretrieve()

## Module Quick Reference

| Function | Purpose |
|----------|---------|
| urlopen() | Open URL and return response |
| Request() | Create request with custom headers/method |
| build_opener() | Create custom opener with handlers |
| install_opener() | Set global default opener |
| urlretrieve() | Download file to disk |
| urlparse() | Parse URL into 6 components |
| urlsplit() | Parse URL into 5 components (modern) |
| urljoin() | Combine base URL with relative URL |
| urlencode() | Build query string from dict/tuples |
| parse_qs() | Parse query string to dict |
| quote() | Percent-encode string for URL path |
| quote_plus() | Percent-encode for query values |
| unquote() | Decode percent-encoded string |

---
*Research Date: January 2026*
*Python Version: 3.11+ (with 3.14 features noted)*
