# Python base64 Module - Production Patterns (January 2026)

## Overview

The `base64` module provides encoding/decoding for binary data to ASCII text. Supports Base16, Base32, Base64, and Base85 encodings as specified in **RFC 4648**.

## Encoding Comparison

| Encoding | Expansion | Alphabet | Use Case |
|----------|-----------|----------|----------|
| Base16 | 2x | 0-9, A-F | Hex representation |
| Base32 | 1.6x | A-Z, 2-7 | Case-insensitive URLs, OTP secrets |
| Base64 | 1.33x | A-Z, a-z, 0-9, +, / | Email, JWT, general binary |
| URL-safe Base64 | 1.33x | A-Z, a-z, 0-9, -, _ | URLs, filenames |
| Base85 | 1.25x | 85 printable chars | Git diffs, compact encoding |
| Z85 | 1.25x | 85 printable chars | ZeroMQ (3.13+) |

## Base64 Encoding (Most Common)

### Standard Base64
```python
import base64

# Encode bytes to Base64
data = b'Hello, World!'
encoded = base64.b64encode(data)
print(encoded)  # b'SGVsbG8sIFdvcmxkIQ=='

# Decode Base64 to bytes
decoded = base64.b64decode(encoded)
print(decoded)  # b'Hello, World!'

# From string (must encode to bytes first)
text = "Hello, World!"
encoded = base64.b64encode(text.encode('utf-8'))

# To string (decode bytes result)
decoded_text = base64.b64decode(encoded).decode('utf-8')
```

### URL-Safe Base64
```python
import base64

# URL-safe: replaces + with - and / with _
# Safe for URLs and filenames
data = b'\xfb\xff\xfe'  # Contains bytes that become + and /

standard = base64.b64encode(data)      # b'++/+'
urlsafe = base64.urlsafe_b64encode(data)  # b'--_-'

# Decode URL-safe
decoded = base64.urlsafe_b64decode(urlsafe)

# Still has = padding - may need to strip for URLs
token = base64.urlsafe_b64encode(data).rstrip(b'=')
```

### Validation Mode
```python
import base64
import binascii

# Default: silently ignores invalid characters
data = base64.b64decode("SGVs bG8=")  # Spaces ignored

# Strict mode: raises on invalid characters
try:
    data = base64.b64decode("SGVs bG8=", validate=True)
except binascii.Error as e:
    print(f"Invalid Base64: {e}")
```

### Custom Alphabet
```python
import base64

# Replace + and / with custom characters
data = b'Hello!'
# Use . and _ instead of + and /
encoded = base64.b64encode(data, altchars=b'._')
decoded = base64.b64decode(encoded, altchars=b'._')
```

## Base32 Encoding

```python
import base64

data = b'Hello!'

# Standard Base32 (uppercase A-Z, 2-7)
encoded = base64.b32encode(data)
print(encoded)  # b'JBSWY3DPEE======'

decoded = base64.b32decode(encoded)

# Case-insensitive decode
encoded_lower = b'jbswy3dpee======'
decoded = base64.b32decode(encoded_lower, casefold=True)

# Extended Hex alphabet (0-9, A-V) - Python 3.10+
hex_encoded = base64.b32hexencode(data)
hex_decoded = base64.b32hexdecode(hex_encoded)
```

## Base16 Encoding (Hex)

```python
import base64

data = b'\xde\xad\xbe\xef'

# Base16 is just uppercase hex
encoded = base64.b16encode(data)
print(encoded)  # b'DEADBEEF'

# Case-insensitive decode
decoded = base64.b16decode(b'deadbeef', casefold=True)

# For hex, you can also use bytes.hex() / bytes.fromhex()
hex_str = data.hex()  # 'deadbeef'
back = bytes.fromhex(hex_str)
```

## Base85 Encodings

### Ascii85 (Adobe)
```python
import base64

data = b'Hello, World!'

# Standard Ascii85
encoded = base64.a85encode(data)
print(encoded)  # b'87cURD]i,"Ebo80'

decoded = base64.a85decode(encoded)

# Adobe format with markers
adobe = base64.a85encode(data, adobe=True)
print(adobe)  # b'<~87cURD]i,"Ebo80~>'

# With line wrapping
wrapped = base64.a85encode(data, wrapcol=60)

# Fold spaces ('y' for 4 spaces)
with_spaces = base64.a85encode(b'    data', foldspaces=True)
```

### Base85 (Git-style)
```python
import base64

data = b'Hello, World!'

# Used in git binary diffs
encoded = base64.b85encode(data)
decoded = base64.b85decode(encoded)

# With padding to multiple of 4
padded = base64.b85encode(data, pad=True)
```

### Z85 (ZeroMQ) - Python 3.13+
```python
import base64

# Z85 encoding for ZeroMQ messages
# Input must be multiple of 4 bytes
data = b'Hell'  # 4 bytes

encoded = base64.z85encode(data)
decoded = base64.z85decode(encoded)
```

## Legacy Interface (File Objects)

```python
import base64
from io import BytesIO

# For RFC 2045 MIME compatibility
# Adds newlines every 76 characters

# Encode to file
input_file = BytesIO(b'Binary data here')
output_file = BytesIO()
base64.encode(input_file, output_file)

# Decode from file
input_file = BytesIO(b'QmluYXJ5IGRhdGEgaGVyZQ==\n')
output_file = BytesIO()
base64.decode(input_file, output_file)

# Encode bytes with MIME-style newlines
encoded = base64.encodebytes(b'x' * 100)
# Has newlines every 76 chars

decoded = base64.decodebytes(encoded)
```

## Production Pattern: Base64Codec

```python
"""Production Base64 utilities with common patterns."""
from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class EncodingType(Enum):
    """Supported encoding types."""
    BASE64 = "base64"
    BASE64_URL = "base64url"
    BASE32 = "base32"
    BASE16 = "base16"
    BASE85 = "base85"


@dataclass
class Base64Codec:
    """Production-ready Base64 encoding utilities."""
    
    @staticmethod
    def encode(
        data: bytes,
        *,
        url_safe: bool = False,
        strip_padding: bool = False,
    ) -> str:
        """Encode bytes to Base64 string.
        
        Args:
            data: Bytes to encode
            url_safe: Use URL-safe alphabet (- and _ instead of + and /)
            strip_padding: Remove trailing = padding
            
        Returns:
            Base64 encoded string
        """
        if url_safe:
            encoded = base64.urlsafe_b64encode(data)
        else:
            encoded = base64.b64encode(data)
        
        result = encoded.decode('ascii')
        
        if strip_padding:
            result = result.rstrip('=')
        
        return result
    
    @staticmethod
    def decode(
        data: str,
        *,
        url_safe: bool = False,
        validate: bool = True,
    ) -> bytes:
        """Decode Base64 string to bytes.
        
        Automatically handles missing padding.
        
        Args:
            data: Base64 encoded string
            url_safe: Expect URL-safe alphabet
            validate: Raise on invalid characters
            
        Returns:
            Decoded bytes
            
        Raises:
            ValueError: If invalid Base64
        """
        # Restore padding if missing
        padding = 4 - len(data) % 4
        if padding != 4:
            data += '=' * padding
        
        try:
            if url_safe:
                return base64.urlsafe_b64decode(data)
            else:
                return base64.b64decode(data, validate=validate)
        except binascii.Error as e:
            raise ValueError(f"Invalid Base64: {e}") from e
    
    @staticmethod
    def encode_json(obj: Any, *, url_safe: bool = True) -> str:
        """Encode JSON object to Base64 string.
        
        Common pattern for JWT payloads, API tokens, etc.
        """
        json_bytes = json.dumps(obj, separators=(',', ':')).encode('utf-8')
        return Base64Codec.encode(json_bytes, url_safe=url_safe, strip_padding=True)
    
    @staticmethod
    def decode_json(data: str, *, url_safe: bool = True) -> Any:
        """Decode Base64 string to JSON object."""
        decoded = Base64Codec.decode(data, url_safe=url_safe)
        return json.loads(decoded.decode('utf-8'))
    
    @staticmethod
    def encode_for_url(data: bytes) -> str:
        """Encode bytes for safe URL usage.
        
        URL-safe alphabet, no padding (shorter URLs).
        """
        return Base64Codec.encode(data, url_safe=True, strip_padding=True)
    
    @staticmethod
    def decode_from_url(data: str) -> bytes:
        """Decode URL-safe Base64."""
        return Base64Codec.decode(data, url_safe=True)
    
    @staticmethod
    def encode_for_header(data: bytes) -> str:
        """Encode for HTTP header (Authorization, etc.)."""
        return Base64Codec.encode(data, url_safe=False, strip_padding=False)
    
    @staticmethod
    def is_valid(data: str, *, url_safe: bool = False) -> bool:
        """Check if string is valid Base64."""
        try:
            Base64Codec.decode(data, url_safe=url_safe, validate=True)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def to_data_uri(data: bytes, mime_type: str = "application/octet-stream") -> str:
        """Create data URI from bytes.
        
        Example: data:image/png;base64,iVBORw0KGgo...
        """
        encoded = Base64Codec.encode(data)
        return f"data:{mime_type};base64,{encoded}"
    
    @staticmethod
    def from_data_uri(uri: str) -> tuple[str, bytes]:
        """Parse data URI to mime type and bytes.
        
        Returns:
            Tuple of (mime_type, data)
        """
        if not uri.startswith("data:"):
            raise ValueError("Invalid data URI")
        
        # data:mime/type;base64,DATA
        header, encoded = uri.split(",", 1)
        mime_type = header[5:].replace(";base64", "")
        
        return mime_type, Base64Codec.decode(encoded)


# Common patterns
class TokenEncoder:
    """Encode structured tokens (like simplified JWT)."""
    
    @staticmethod
    def encode_token(header: dict, payload: dict, signature: bytes) -> str:
        """Encode token parts to dot-separated Base64."""
        parts = [
            Base64Codec.encode_json(header),
            Base64Codec.encode_json(payload),
            Base64Codec.encode(signature, url_safe=True, strip_padding=True),
        ]
        return ".".join(parts)
    
    @staticmethod
    def decode_token(token: str) -> tuple[dict, dict, bytes]:
        """Decode token to parts."""
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid token format")
        
        header = Base64Codec.decode_json(parts[0])
        payload = Base64Codec.decode_json(parts[1])
        signature = Base64Codec.decode(parts[2], url_safe=True)
        
        return header, payload, signature


# Usage example
if __name__ == "__main__":
    codec = Base64Codec()
    
    # Basic encoding
    data = b"Hello, World!"
    encoded = codec.encode(data)
    print(f"Encoded: {encoded}")
    decoded = codec.decode(encoded)
    print(f"Decoded: {decoded}")
    
    # URL-safe (for query params, paths)
    url_safe = codec.encode_for_url(data)
    print(f"URL-safe: {url_safe}")
    
    # JSON encoding (common for APIs)
    obj = {"user_id": 123, "exp": 1234567890}
    encoded_json = codec.encode_json(obj)
    print(f"JSON encoded: {encoded_json}")
    decoded_obj = codec.decode_json(encoded_json)
    print(f"JSON decoded: {decoded_obj}")
    
    # Data URI (for embedding in HTML/CSS)
    image_bytes = b'\x89PNG\r\n\x1a\n...'  # PNG header
    data_uri = codec.to_data_uri(image_bytes, "image/png")
    print(f"Data URI: {data_uri[:50]}...")
    
    # Validation
    print(f"Valid: {codec.is_valid('SGVsbG8=')}")  # True
    print(f"Valid: {codec.is_valid('!!!invalid')}")  # False
```

## Key Patterns

### 1. HTTP Basic Authentication
```python
import base64

def basic_auth_header(username: str, password: str) -> str:
    """Create Basic Auth header value."""
    credentials = f"{username}:{password}".encode('utf-8')
    encoded = base64.b64encode(credentials).decode('ascii')
    return f"Basic {encoded}"

# Usage
headers = {"Authorization": basic_auth_header("user", "pass")}
```

### 2. JWT-Style Tokens
```python
import base64
import json

def encode_jwt_part(data: dict) -> str:
    """Encode dict as URL-safe Base64 without padding."""
    json_bytes = json.dumps(data, separators=(',', ':')).encode()
    return base64.urlsafe_b64encode(json_bytes).rstrip(b'=').decode()

def decode_jwt_part(data: str) -> dict:
    """Decode URL-safe Base64 to dict, handling missing padding."""
    # Add padding
    padding = 4 - len(data) % 4
    if padding != 4:
        data += '=' * padding
    return json.loads(base64.urlsafe_b64decode(data))
```

### 3. Binary Data in JSON
```python
import base64
import json

def serialize_with_binary(obj: dict) -> str:
    """Serialize dict with binary fields as Base64."""
    def encode_bytes(o):
        if isinstance(o, bytes):
            return {"__bytes__": base64.b64encode(o).decode()}
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")
    
    return json.dumps(obj, default=encode_bytes)

def deserialize_with_binary(data: str) -> dict:
    """Deserialize JSON with Base64-encoded binary fields."""
    def decode_bytes(o):
        if "__bytes__" in o:
            return base64.b64decode(o["__bytes__"])
        return o
    
    return json.loads(data, object_hook=decode_bytes)
```

### 4. TOTP/HOTP Secret Handling
```python
import base64
import secrets

def generate_totp_secret() -> str:
    """Generate Base32-encoded secret for TOTP apps."""
    # 20 bytes = 160 bits (standard for TOTP)
    secret_bytes = secrets.token_bytes(20)
    # Base32 is case-insensitive, no confusing chars
    return base64.b32encode(secret_bytes).decode()

def decode_totp_secret(secret: str) -> bytes:
    """Decode Base32 TOTP secret."""
    # Handle spaces and lowercase
    clean = secret.upper().replace(' ', '')
    return base64.b32decode(clean)
```

### 5. Image Embedding
```python
import base64
from pathlib import Path

def image_to_data_uri(path: Path) -> str:
    """Convert image file to data URI for HTML embedding."""
    suffix_to_mime = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.svg': 'image/svg+xml',
    }
    
    mime = suffix_to_mime.get(path.suffix.lower(), 'application/octet-stream')
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode()
    
    return f"data:{mime};base64,{encoded}"
```

## Security Considerations (RFC 4648 Section 12)

1. **Base64 is NOT encryption** - only encoding
2. **Validate before decoding** - use `validate=True` for untrusted input
3. **Beware of timing attacks** - use `secrets.compare_digest()` for comparisons
4. **Size limits** - decoded data can be 3/4 size of input
5. **Canonicalization** - multiple encodings can produce same output

```python
import base64
import secrets

# WRONG: Direct comparison vulnerable to timing attacks
if encoded_token == expected_token:
    pass

# CORRECT: Constant-time comparison
if secrets.compare_digest(encoded_token, expected_token):
    pass
```

## Source
- Python 3.14 base64 documentation: https://docs.python.org/3/library/base64.html
- RFC 4648 - The Base16, Base32, and Base64 Data Encodings
- Research cycle: Ralph Loop Cycle 104 (January 2026)
