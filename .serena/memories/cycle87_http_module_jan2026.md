# Python http Module - Production Patterns (Cycle 87)

## Overview

The `http` package provides HTTP protocol support with `http.client` for client-side connections and `http.server` for building HTTP servers. New in Python 3.14: native HTTPS server support.

## http.client Module

### HTTPConnection - Basic Usage

```python
import http.client

# Create connection (does not connect yet)
conn = http.client.HTTPConnection("www.example.com", port=80, timeout=30)

# Make request
conn.request("GET", "/path", headers={"User-Agent": "MyApp/1.0"})

# Get response
response = conn.getresponse()
print(response.status, response.reason)  # 200 OK
data = response.read()

# MUST close connection
conn.close()
```

### HTTPSConnection - Secure Connections

```python
import http.client
import ssl

# Default HTTPS (verifies certificates)
conn = http.client.HTTPSConnection("api.example.com")

# With custom SSL context
context = ssl.create_default_context()
context.check_hostname = True
context.verify_mode = ssl.CERT_REQUIRED
conn = http.client.HTTPSConnection("api.example.com", context=context)

# Request with JSON body
import json
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer token123"
}
body = json.dumps({"key": "value"})
conn.request("POST", "/api/data", body=body, headers=headers)

response = conn.getresponse()
result = json.loads(response.read().decode("utf-8"))
conn.close()
```

### Context Manager Pattern (Recommended)

```python
import http.client

# Python 3.9+ context manager support
with http.client.HTTPSConnection("api.example.com") as conn:
    conn.request("GET", "/resource")
    response = conn.getresponse()
    data = response.read()
    # Connection automatically closed
```

### HTTPResponse Object

```python
response = conn.getresponse()

# Status and reason
response.status          # 200, 404, 500, etc.
response.reason          # "OK", "Not Found", etc.
response.version         # 10 for HTTP/1.0, 11 for HTTP/1.1

# Headers
response.headers         # email.message.Message-like object
response.getheader("Content-Type")  # Get single header
response.getheaders()    # List of (name, value) tuples

# Reading body
response.read()          # Read entire body
response.read(100)       # Read up to 100 bytes
response.readline()      # Read one line
response.readinto(buffer)  # Read into pre-allocated buffer

# Streaming large responses
for chunk in response:
    process(chunk)

# Check if readable
response.isclosed()      # True if response body fully read
```

### HTTP Methods

```python
# All standard HTTP methods
conn.request("GET", "/resource")
conn.request("POST", "/resource", body=data)
conn.request("PUT", "/resource", body=data)
conn.request("DELETE", "/resource")
conn.request("PATCH", "/resource", body=data)
conn.request("HEAD", "/resource")  # No body returned
conn.request("OPTIONS", "/resource")
```

### Chunked Transfer Encoding

```python
# Send chunked data (streaming upload)
conn.request("POST", "/upload", body=None)
conn.send(b"5\r\nHello\r\n")  # Chunk: "Hello"
conn.send(b"6\r\n World\r\n")  # Chunk: " World"
conn.send(b"0\r\n\r\n")  # End of chunks

# HTTPResponse handles chunked responses automatically
response = conn.getresponse()
data = response.read()  # Chunks assembled automatically
```

### Connection Persistence (Keep-Alive)

```python
# HTTP/1.1 uses keep-alive by default
conn = http.client.HTTPConnection("example.com")

# Multiple requests on same connection
conn.request("GET", "/page1")
r1 = conn.getresponse()
r1.read()  # MUST fully read before next request

conn.request("GET", "/page2")
r2 = conn.getresponse()
r2.read()

conn.close()
```

### Proxy Tunneling (CONNECT)

```python
import http.client

# Connect through HTTP proxy to HTTPS destination
conn = http.client.HTTPSConnection(
    "proxy.example.com",
    port=8080
)

# Set up tunnel through proxy
conn.set_tunnel("target.example.com", port=443)
conn.connect()

# Now requests go to target via proxy tunnel
conn.request("GET", "/secure-resource")
response = conn.getresponse()
```

### Error Handling

```python
import http.client
import socket

try:
    conn = http.client.HTTPSConnection("api.example.com", timeout=10)
    conn.request("GET", "/data")
    response = conn.getresponse()
    
    if response.status >= 400:
        raise http.client.HTTPException(
            f"HTTP {response.status}: {response.reason}"
        )
    
    data = response.read()
    
except http.client.HTTPException as e:
    print(f"HTTP error: {e}")
except socket.timeout:
    print("Connection timed out")
except ConnectionRefusedError:
    print("Connection refused")
except ssl.SSLCertVerificationError as e:
    print(f"SSL certificate error: {e}")
finally:
    conn.close()
```

## http.server Module

### Basic HTTP Server

```python
from http.server import HTTPServer, BaseHTTPRequestHandler

class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h1>Hello World</h1>")
    
    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "received"}')

server = HTTPServer(("", 8000), MyHandler)
print("Serving on port 8000...")
server.serve_forever()
```

### ThreadingHTTPServer (Concurrent Requests)

```python
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

# Each request handled in separate thread
server = ThreadingHTTPServer(("", 8000), SimpleHTTPRequestHandler)
server.serve_forever()
```

### HTTPSServer (Python 3.14+)

```python
# New in Python 3.14!
from http.server import HTTPSServer, ThreadingHTTPSServer
import ssl

context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain("cert.pem", "key.pem")

# Single-threaded HTTPS
server = HTTPSServer(("", 443), MyHandler, context=context)

# Or multi-threaded HTTPS
server = ThreadingHTTPSServer(("", 443), MyHandler, context=context)
server.serve_forever()
```

### SimpleHTTPRequestHandler (Static Files)

```python
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

class CustomStaticHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory or os.getcwd(), **kwargs)
    
    # Override to add custom headers
    def end_headers(self):
        self.send_header("Cache-Control", "max-age=3600")
        super().end_headers()

# Serve from specific directory
handler = lambda *args: CustomStaticHandler(*args, directory="/var/www")
server = HTTPServer(("", 8000), handler)
server.serve_forever()
```

### BaseHTTPRequestHandler Attributes

```python
class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Request info
        self.path           # "/resource?query=value"
        self.command        # "GET", "POST", etc.
        self.request_version  # "HTTP/1.1"
        self.headers        # email.message.Message object
        self.client_address # ("127.0.0.1", 54321)
        
        # I/O streams
        self.rfile          # Request body (readable file-like)
        self.wfile          # Response body (writable file-like)
        
        # Server reference
        self.server         # The HTTPServer instance
```

### Custom Error Responses

```python
class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/secret":
            self.send_error(403, "Forbidden", "Access denied")
            return
        
        if self.path.startswith("/api/"):
            self.send_error(404, "Not Found", 
                          f"Endpoint {self.path} does not exist")
            return
        
        # Normal response
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")
```

### Logging Control

```python
class QuietHandler(BaseHTTPRequestHandler):
    # Suppress default logging
    def log_message(self, format, *args):
        pass  # Silent
    
    # Or custom logging
    def log_message(self, format, *args):
        import logging
        logging.info(f"{self.client_address[0]} - {format % args}")
```

## Command-Line Interface

### Basic File Server

```bash
# Serve current directory on port 8000
python -m http.server

# Custom port
python -m http.server 9000

# Bind to specific interface
python -m http.server -b 127.0.0.1

# Serve specific directory
python -m http.server -d /var/www/html
```

### HTTPS Server (Python 3.14+)

```bash
# New TLS options in Python 3.14!
python -m http.server --tls-cert cert.pem --tls-key key.pem

# With password-protected key
python -m http.server --tls-cert cert.pem --tls-key key.pem \
    --tls-password-file keypass.txt

# Full example
python -m http.server 443 -b 0.0.0.0 -d /var/www \
    --tls-cert /etc/ssl/cert.pem \
    --tls-key /etc/ssl/key.pem
```

## HTTP Status Codes (http.HTTPStatus)

```python
from http import HTTPStatus

HTTPStatus.OK                    # 200
HTTPStatus.CREATED               # 201
HTTPStatus.NO_CONTENT            # 204
HTTPStatus.MOVED_PERMANENTLY     # 301
HTTPStatus.FOUND                 # 302
HTTPStatus.NOT_MODIFIED          # 304
HTTPStatus.BAD_REQUEST           # 400
HTTPStatus.UNAUTHORIZED          # 401
HTTPStatus.FORBIDDEN             # 403
HTTPStatus.NOT_FOUND             # 404
HTTPStatus.METHOD_NOT_ALLOWED    # 405
HTTPStatus.INTERNAL_SERVER_ERROR # 500
HTTPStatus.BAD_GATEWAY           # 502
HTTPStatus.SERVICE_UNAVAILABLE   # 503

# Access phrase
HTTPStatus.OK.phrase             # "OK"
HTTPStatus.NOT_FOUND.phrase      # "Not Found"

# Check status categories
status = 201
if 200 <= status < 300:
    print("Success")
elif 400 <= status < 500:
    print("Client error")
elif 500 <= status < 600:
    print("Server error")
```

## Production Patterns

### Graceful Shutdown

```python
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import signal
import threading

server = ThreadingHTTPServer(("", 8000), MyHandler)
server_thread = threading.Thread(target=server.serve_forever)
server_thread.start()

def shutdown_handler(signum, frame):
    print("Shutting down...")
    server.shutdown()  # Stop serve_forever()
    server.server_close()  # Release socket

signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)

server_thread.join()
```

### Request Timeout Handling

```python
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler

class TimeoutHTTPServer(HTTPServer):
    timeout = 30  # Socket accept timeout
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.socket.settimeout(self.timeout)

class MyHandler(BaseHTTPRequestHandler):
    timeout = 10  # Per-request timeout
    
    def do_GET(self):
        try:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        except socket.timeout:
            pass  # Client disconnected
```

### Health Check Endpoint

```python
import json
import time

class HealthHandler(BaseHTTPRequestHandler):
    start_time = time.time()
    
    def do_GET(self):
        if self.path == "/health":
            health = {
                "status": "healthy",
                "uptime": time.time() - self.start_time,
                "version": "1.0.0"
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(health).encode())
        else:
            self.send_error(404)
```

## Best Practices

1. **Always use context managers** for HTTPConnection/HTTPSConnection
2. **Read full response body** before making another request on same connection
3. **Use ThreadingHTTPServer** for concurrent request handling
4. **Set appropriate timeouts** to prevent resource exhaustion
5. **Validate Content-Length** before reading request body
6. **Use HTTPSServer (3.14+)** or reverse proxy for production TLS
7. **Implement graceful shutdown** with signal handlers
8. **Log requests** appropriately for monitoring
9. **Use http.HTTPStatus** for readable status codes
10. **Prefer httpx or aiohttp** for production HTTP clients

## Module Quick Reference

| Class/Function | Purpose |
|---------------|---------|
| HTTPConnection | HTTP/1.1 client connection |
| HTTPSConnection | HTTPS client connection |
| HTTPResponse | Response object with status, headers, body |
| HTTPServer | Single-threaded HTTP server |
| ThreadingHTTPServer | Multi-threaded HTTP server |
| HTTPSServer | HTTPS server (3.14+) |
| ThreadingHTTPSServer | Multi-threaded HTTPS server (3.14+) |
| BaseHTTPRequestHandler | Base class for request handlers |
| SimpleHTTPRequestHandler | Static file serving handler |
| HTTPStatus | HTTP status code enum |

---
*Research Date: January 2026*
*Python Version: 3.11+ (with 3.14 HTTPS features noted)*
