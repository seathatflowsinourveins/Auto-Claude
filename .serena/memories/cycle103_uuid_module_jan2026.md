# Python uuid Module - Production Patterns (January 2026)

## Overview

The `uuid` module provides immutable UUID objects conforming to **RFC 9562** (supersedes RFC 4122). Python 3.14 added UUID versions 6, 7, and 8 for improved database locality and modern use cases.

## UUID Version Selection Guide

| Version | Function | Use Case | Properties |
|---------|----------|----------|------------|
| **v1** | `uuid1()` | Legacy systems | Time + MAC address (privacy risk) |
| **v3** | `uuid3()` | Deterministic from name | MD5 hash (not secure) |
| **v4** | `uuid4()` | General purpose | Cryptographically random |
| **v5** | `uuid5()` | Deterministic from name | SHA-1 hash |
| **v6** | `uuid6()` | Database primary keys | Time-ordered v1 (3.14+) |
| **v7** | `uuid7()` | Modern distributed systems | Time-ordered, monotonic (3.14+) |
| **v8** | `uuid8()` | Custom requirements | User-defined blocks (3.14+) |

**Recommendation**: Use `uuid4()` for general purposes, `uuid7()` for database PKs (Python 3.14+).

## Core API

### uuid4() - Random UUID (Most Common)
```python
import uuid

# Cryptographically secure random UUID
id = uuid.uuid4()
print(id)  # e.g., 'f47ac10b-58cc-4372-a567-0e02b2c3d479'

# String conversion
str(id)     # '12345678-1234-5678-1234-567812345678'
id.hex      # '12345678123456781234567812345678' (no hyphens)
id.bytes    # b'\x12\x34...' (16 bytes)
id.int      # 24197857161011715162171839636988778104 (128-bit int)
id.urn      # 'urn:uuid:12345678-1234-5678-1234-567812345678'
```

### uuid1() - Time-Based (Privacy Consideration)
```python
import uuid

# Contains MAC address - may compromise privacy
id = uuid.uuid1()

# Optionally specify node (MAC) and clock_seq
id = uuid.uuid1(node=0x123456789abc, clock_seq=42)

# Check if generated safely (multiprocessing-safe)
id.is_safe  # SafeUUID.safe, .unsafe, or .unknown
```

### uuid3() / uuid5() - Namespace-Based (Deterministic)
```python
import uuid

# Same inputs always produce same UUID
# uuid3 uses MD5, uuid5 uses SHA-1
id3 = uuid.uuid3(uuid.NAMESPACE_DNS, 'python.org')
id5 = uuid.uuid5(uuid.NAMESPACE_DNS, 'python.org')

# Built-in namespaces
uuid.NAMESPACE_DNS   # Fully qualified domain names
uuid.NAMESPACE_URL   # URLs
uuid.NAMESPACE_OID   # ISO OIDs
uuid.NAMESPACE_X500  # X.500 Distinguished Names

# Custom namespace
MY_NAMESPACE = uuid.UUID('12345678-1234-5678-1234-567812345678')
user_id = uuid.uuid5(MY_NAMESPACE, 'user@example.com')
```

### uuid6() - Database-Optimized (Python 3.14+)
```python
import uuid

# Reordered v1 for better database B-tree locality
# Time bits are in most-significant positions
id = uuid.uuid6()

# Still uses MAC address like v1
id = uuid.uuid6(node=0x123456789abc)
```

### uuid7() - Modern Time-Ordered (Python 3.14+)
```python
import uuid
import datetime as dt

# Best for distributed systems and databases
# 48-bit timestamp (ms) + 42-bit counter for monotonicity
id = uuid.uuid7()

# Extract creation time
timestamp_ms = id.time  # e.g., 1743936859822
creation_time = dt.datetime.fromtimestamp(timestamp_ms / 1000)

# Guaranteed monotonic within same millisecond
ids = [uuid.uuid7() for _ in range(1000)]
assert ids == sorted(ids)  # Always true
```

### uuid8() - Custom Blocks (Python 3.14+)
```python
import uuid

# Define custom 48-bit, 12-bit, and 62-bit blocks
# NOT cryptographically secure by default
id = uuid.uuid8(
    a=0x123456789ABC,  # 48 bits
    b=0xDEF,           # 12 bits  
    c=0x1122334455667788 & ((1 << 62) - 1)  # 62 bits
)

# Omitted args use pseudo-random values
id = uuid.uuid8()  # All random (but not CSPRNG)
```

### Special UUIDs (Python 3.14+)
```python
import uuid

# Nil UUID (all zeros)
uuid.NIL  # UUID('00000000-0000-0000-0000-000000000000')

# Max UUID (all ones)
uuid.MAX  # UUID('ffffffff-ffff-ffff-ffff-ffffffffffff')
```

## UUID Object Attributes

```python
import uuid

id = uuid.uuid4()

# String representations
str(id)         # '12345678-1234-5678-1234-567812345678'
id.hex          # '12345678123456781234567812345678'
id.urn          # 'urn:uuid:12345678-1234-5678-1234-567812345678'

# Binary representations
id.bytes        # 16-byte big-endian
id.bytes_le     # 16-byte little-endian (Windows COM)
id.int          # 128-bit integer

# Metadata
id.version      # 4 (for uuid4)
id.variant      # uuid.RFC_4122
id.is_safe      # SafeUUID enum

# Time-based fields (v1, v6, v7)
id.time         # 60-bit timestamp (v1/v6) or 48-bit ms (v7)
id.clock_seq    # 14-bit sequence (v1/v6)
id.node         # 48-bit MAC address (v1/v6)

# Raw fields tuple
id.fields       # (time_low, time_mid, time_hi_version, 
                #  clock_seq_hi_variant, clock_seq_low, node)
```

## Creating UUID from Various Formats

```python
import uuid

# From string (flexible parsing)
uuid.UUID('{12345678-1234-5678-1234-567812345678}')  # Braces OK
uuid.UUID('12345678-1234-5678-1234-567812345678')    # Standard
uuid.UUID('12345678123456781234567812345678')        # No hyphens
uuid.UUID('urn:uuid:12345678-1234-5678-1234-567812345678')  # URN

# From bytes
uuid.UUID(bytes=b'\x12\x34\x56\x78' * 4)             # Big-endian
uuid.UUID(bytes_le=b'\x78\x56\x34\x12...')           # Little-endian

# From integer
uuid.UUID(int=0x12345678123456781234567812345678)

# From fields tuple
uuid.UUID(fields=(0x12345678, 0x1234, 0x5678, 0x12, 0x34, 0x567812345678))
```

## Command-Line Interface (Python 3.12+)

```bash
# Generate random UUID (default: uuid4)
python -m uuid

# Specific version
python -m uuid -u uuid1
python -m uuid -u uuid7

# Namespace-based (uuid3/uuid5)
python -m uuid -u uuid5 -n @dns -N python.org
python -m uuid -u uuid3 -n @url -N https://example.com

# Generate multiple UUIDs (Python 3.14+)
python -m uuid -C 10
```

## Production Pattern: UUIDFactory

```python
"""Production UUID factory with version selection and validation."""
from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import ClassVar


class UUIDVersion(Enum):
    """Supported UUID versions with use-case guidance."""
    V1 = 1  # Time + MAC (legacy, privacy risk)
    V3 = 3  # MD5 namespace (deterministic, not secure)
    V4 = 4  # Random (general purpose)
    V5 = 5  # SHA-1 namespace (deterministic)
    V6 = 6  # Time-ordered v1 (3.14+, DB-friendly)
    V7 = 7  # Modern time-ordered (3.14+, recommended)
    V8 = 8  # Custom blocks (3.14+)


@dataclass
class UUIDFactory:
    """Factory for generating UUIDs with consistent patterns."""
    
    # Minimum Python version for new UUID versions
    MIN_VERSION_NEW_UUIDS: ClassVar[tuple] = (3, 14)
    
    # Custom namespace for application-specific UUIDs
    app_namespace: uuid.UUID | None = None
    
    def __post_init__(self):
        """Initialize default namespace if not provided."""
        if self.app_namespace is None:
            # Create a stable namespace from app identifier
            self.app_namespace = uuid.uuid5(
                uuid.NAMESPACE_DNS,
                "myapp.example.com"
            )
    
    @property
    def supports_new_versions(self) -> bool:
        """Check if Python version supports uuid6/7/8."""
        return sys.version_info >= self.MIN_VERSION_NEW_UUIDS
    
    def generate(self, version: UUIDVersion = UUIDVersion.V4) -> uuid.UUID:
        """Generate UUID of specified version.
        
        Args:
            version: UUID version to generate
            
        Returns:
            Generated UUID
            
        Raises:
            ValueError: If version not supported on this Python
        """
        match version:
            case UUIDVersion.V1:
                return uuid.uuid1()
            case UUIDVersion.V4:
                return uuid.uuid4()
            case UUIDVersion.V6:
                if not self.supports_new_versions:
                    raise ValueError("uuid6 requires Python 3.14+")
                return uuid.uuid6()
            case UUIDVersion.V7:
                if not self.supports_new_versions:
                    raise ValueError("uuid7 requires Python 3.14+")
                return uuid.uuid7()
            case _:
                raise ValueError(f"Use generate_named() for {version}")
    
    def generate_named(
        self,
        name: str,
        *,
        version: UUIDVersion = UUIDVersion.V5,
        namespace: uuid.UUID | None = None,
    ) -> uuid.UUID:
        """Generate deterministic UUID from name.
        
        Same name always produces same UUID.
        
        Args:
            name: String to hash
            version: V3 (MD5) or V5 (SHA-1)
            namespace: UUID namespace (default: app_namespace)
            
        Returns:
            Deterministic UUID
        """
        ns = namespace or self.app_namespace
        
        if version == UUIDVersion.V3:
            return uuid.uuid3(ns, name)
        elif version == UUIDVersion.V5:
            return uuid.uuid5(ns, name)
        else:
            raise ValueError("generate_named() requires V3 or V5")
    
    def generate_for_database(self) -> uuid.UUID:
        """Generate UUID optimized for database primary keys.
        
        Uses uuid7 on Python 3.14+ (time-ordered, monotonic).
        Falls back to uuid4 on older versions.
        """
        if self.supports_new_versions:
            return uuid.uuid7()
        return uuid.uuid4()
    
    @staticmethod
    def parse(value: str) -> uuid.UUID:
        """Parse UUID from string with validation.
        
        Accepts various formats:
        - Standard: 12345678-1234-5678-1234-567812345678
        - Compact: 12345678123456781234567812345678
        - Braces: {12345678-1234-5678-1234-567812345678}
        - URN: urn:uuid:12345678-1234-5678-1234-567812345678
        
        Raises:
            ValueError: If string is not valid UUID
        """
        try:
            return uuid.UUID(value)
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid UUID: {value}") from e
    
    @staticmethod
    def is_valid(value: str) -> bool:
        """Check if string is valid UUID format."""
        try:
            uuid.UUID(value)
            return True
        except (ValueError, AttributeError):
            return False
    
    @staticmethod
    def extract_timestamp(id: uuid.UUID) -> datetime | None:
        """Extract creation timestamp from time-based UUID.
        
        Works with v1, v6, and v7 UUIDs.
        
        Returns:
            datetime if time-based UUID, None otherwise
        """
        if id.version == 7:
            # v7: timestamp is milliseconds since Unix epoch
            return datetime.fromtimestamp(id.time / 1000)
        elif id.version in (1, 6):
            # v1/v6: 100-nanosecond intervals since 1582-10-15
            # Convert to Unix timestamp
            uuid_epoch = datetime(1582, 10, 15)
            unix_epoch = datetime(1970, 1, 1)
            offset = (unix_epoch - uuid_epoch).total_seconds()
            timestamp = id.time / 1e7 - offset
            return datetime.fromtimestamp(timestamp)
        return None


# Usage example
if __name__ == "__main__":
    factory = UUIDFactory()
    
    # General purpose
    id1 = factory.generate()  # v4 random
    print(f"Random: {id1}")
    
    # Database primary key
    pk = factory.generate_for_database()  # v7 or v4 fallback
    print(f"DB PK: {pk}")
    
    # Deterministic from name (idempotent)
    user_id = factory.generate_named("user@example.com")
    user_id2 = factory.generate_named("user@example.com")
    assert user_id == user_id2  # Same input = same output
    print(f"User ID: {user_id}")
    
    # Parse from string
    parsed = factory.parse("550e8400-e29b-41d4-a716-446655440000")
    print(f"Parsed: {parsed}, version={parsed.version}")
    
    # Validation
    print(f"Valid: {factory.is_valid('not-a-uuid')}")  # False
    
    # Extract timestamp (v7)
    if factory.supports_new_versions:
        v7_id = factory.generate(UUIDVersion.V7)
        created = factory.extract_timestamp(v7_id)
        print(f"v7 created at: {created}")
```

## Key Patterns

### 1. Database Primary Keys
```python
import uuid

# Python 3.14+: Use uuid7 (time-ordered, monotonic)
# Improves B-tree locality, reduces index fragmentation
pk = uuid.uuid7()

# Pre-3.14: Use uuid4 (random)
pk = uuid.uuid4()

# Store as bytes for efficiency (16 bytes vs 36 char string)
pk_bytes = pk.bytes
```

### 2. Idempotent Entity IDs
```python
import uuid

# Same input always produces same UUID
# Useful for deduplication, caching, ETL

ENTITY_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "entities.myapp.com")

def entity_id(entity_type: str, external_id: str) -> uuid.UUID:
    """Generate stable UUID from external identifier."""
    return uuid.uuid5(ENTITY_NAMESPACE, f"{entity_type}:{external_id}")

# Always produces same UUID for same input
user_id = entity_id("user", "john@example.com")
```

### 3. Short URL-Safe IDs
```python
import uuid
import base64

def uuid_to_short(id: uuid.UUID) -> str:
    """Convert UUID to URL-safe 22-character string."""
    return base64.urlsafe_b64encode(id.bytes).rstrip(b'=').decode()

def short_to_uuid(short: str) -> uuid.UUID:
    """Convert short string back to UUID."""
    padding = 4 - len(short) % 4
    if padding != 4:
        short += '=' * padding
    return uuid.UUID(bytes=base64.urlsafe_b64decode(short))

# Example
id = uuid.uuid4()
short = uuid_to_short(id)  # e.g., 'VQ6IAAAAAAAAAAAAAAAA'
restored = short_to_uuid(short)
assert id == restored
```

### 4. Sortable IDs with uuid7
```python
import uuid
import time

# Generate time-ordered IDs
ids = []
for _ in range(5):
    ids.append(uuid.uuid7())
    time.sleep(0.001)  # 1ms between

# IDs are naturally sorted by creation time
assert ids == sorted(ids)

# Extract creation timestamps
for id in ids:
    print(f"{id}: created at {id.time}ms")
```

## Version Comparison

| Feature | v1 | v4 | v5 | v7 |
|---------|----|----|----|----|
| Random | ❌ | ✅ | ❌ | Partial |
| Deterministic | ❌ | ❌ | ✅ | ❌ |
| Time-ordered | ✅ | ❌ | ❌ | ✅ |
| Monotonic | ❌ | ❌ | ❌ | ✅ |
| Privacy-safe | ❌ | ✅ | ✅ | ✅ |
| DB-friendly | ⚠️ | ❌ | ❌ | ✅ |
| Python version | All | All | All | 3.14+ |

## Source
- Python 3.14 uuid documentation: https://docs.python.org/3/library/uuid.html
- RFC 9562 (supersedes RFC 4122)
- Research cycle: Ralph Loop Cycle 103 (January 2026)
