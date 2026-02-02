# Cycle 41: Message Serialization & Data Formats (January 2026)

## Overview
Comprehensive patterns for high-performance data serialization in Python production systems.
Covers JSON alternatives, binary formats, schema-based serialization, and columnar storage.

---

## 1. orjson - Rust-Powered JSON (2-10x Faster)

### Why orjson
- Written in Rust, compiled to native code
- 2-10x faster than stdlib json
- Native support for dataclasses, datetime, numpy, UUID
- Strict RFC 8259 compliance
- Memory-efficient (no intermediate string allocation)

### Basic Usage
```python
import orjson
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Order:
    id: str
    amount: float
    created_at: datetime

order = Order(id="ord_123", amount=99.95, created_at=datetime.now())

# Serialize (returns bytes, not str)
data = orjson.dumps(order)  # b'{"id":"ord_123","amount":99.95,"created_at":"2026-01-25T..."}'

# Deserialize
parsed = orjson.loads(data)

# Options for customization
data = orjson.dumps(
    order,
    option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_UTC_Z
)
```

### Pydantic v2 Integration
```python
from pydantic import BaseModel
import orjson

class Order(BaseModel):
    id: str
    amount: float
    
    model_config = {
        "json_encoders": {
            # Custom encoders if needed
        }
    }
    
    def model_dump_json_bytes(self) -> bytes:
        """Serialize to bytes using orjson."""
        return orjson.dumps(
            self.model_dump(mode="python"),
            option=orjson.OPT_SERIALIZE_NUMPY
        )

# FastAPI integration
from fastapi import Response

@app.get("/order/{id}")
async def get_order(id: str) -> Response:
    order = await fetch_order(id)
    return Response(
        content=orjson.dumps(order.model_dump()),
        media_type="application/json"
    )
```

### Performance Comparison
```
Library    | Serialize | Deserialize | Notes
-----------|-----------|-------------|-------
json       | 1.0x      | 1.0x        | stdlib baseline
ujson      | 2-3x      | 2-3x        | C extension
orjson     | 5-10x     | 2-5x        | Rust, best overall
rapidjson  | 3-4x      | 2-3x        | C++
```

---

## 2. MessagePack - Binary JSON Alternative

### Why MessagePack
- 50-80% smaller than JSON
- Faster serialization/deserialization than JSON
- Type-rich (binary, extension types)
- Cross-language compatibility
- Good for network protocols and caching

### Basic Usage
```python
import msgpack
from datetime import datetime

# Simple serialization
data = {"name": "Alice", "scores": [95, 87, 92], "active": True}
packed = msgpack.packb(data)  # bytes, ~40% smaller than JSON
unpacked = msgpack.unpackb(packed)

# Streaming for large data
from io import BytesIO

buffer = BytesIO()
packer = msgpack.Packer()
for record in large_dataset:
    buffer.write(packer.pack(record))

buffer.seek(0)
unpacker = msgpack.Unpacker(buffer)
for record in unpacker:
    process(record)
```

### Custom Types with Ext
```python
import msgpack
from datetime import datetime

def encode_datetime(obj):
    if isinstance(obj, datetime):
        return msgpack.ExtType(1, obj.isoformat().encode())
    raise TypeError(f"Unknown type: {type(obj)}")

def decode_ext(code, data):
    if code == 1:
        return datetime.fromisoformat(data.decode())
    return msgpack.ExtType(code, data)

# Usage
data = {"timestamp": datetime.now(), "value": 42.5}
packed = msgpack.packb(data, default=encode_datetime)
unpacked = msgpack.unpackb(packed, ext_hook=decode_ext)
```

### Redis Caching Pattern
```python
import redis
import msgpack
from pydantic import BaseModel

class CacheManager:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
    
    async def get(self, key: str, model: type[BaseModel]) -> BaseModel | None:
        data = await self.redis.get(key)
        if data:
            return model.model_validate(msgpack.unpackb(data))
        return None
    
    async def set(self, key: str, obj: BaseModel, ttl: int = 3600):
        packed = msgpack.packb(obj.model_dump(mode="python"))
        await self.redis.setex(key, ttl, packed)
```

---

## 3. Apache Avro with Schema Registry

### Why Avro
- Schema evolution (forward/backward compatibility)
- Compact binary format
- Schema stored separately (no per-record overhead)
- Perfect for Kafka and event streaming
- Strong typing with schema validation

### Schema Definition
```json
{
  "type": "record",
  "name": "OrderEvent",
  "namespace": "trading.events",
  "fields": [
    {"name": "order_id", "type": "string"},
    {"name": "symbol", "type": "string"},
    {"name": "quantity", "type": "int"},
    {"name": "price", "type": "double"},
    {"name": "side", "type": {"type": "enum", "name": "Side", "symbols": ["BUY", "SELL"]}},
    {"name": "timestamp", "type": {"type": "long", "logicalType": "timestamp-millis"}},
    {"name": "metadata", "type": ["null", {"type": "map", "values": "string"}], "default": null}
  ]
}
```

### Confluent Schema Registry Integration
```python
from confluent_kafka import Producer, Consumer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer, AvroDeserializer
from confluent_kafka.serialization import SerializationContext, MessageField

# Schema Registry client
schema_registry = SchemaRegistryClient({"url": "http://localhost:8081"})

# Producer with Avro
class OrderEvent:
    def __init__(self, order_id: str, symbol: str, quantity: int, price: float, side: str):
        self.order_id = order_id
        self.symbol = symbol
        self.quantity = quantity
        self.price = price
        self.side = side

def order_to_dict(order: OrderEvent, ctx) -> dict:
    return {
        "order_id": order.order_id,
        "symbol": order.symbol,
        "quantity": order.quantity,
        "price": order.price,
        "side": order.side,
        "timestamp": int(datetime.now().timestamp() * 1000),
        "metadata": None
    }

avro_serializer = AvroSerializer(
    schema_registry,
    schema_str,  # Avro schema JSON string
    order_to_dict
)

producer = Producer({"bootstrap.servers": "localhost:9092"})

def produce_order(order: OrderEvent):
    producer.produce(
        topic="orders",
        key=order.order_id.encode(),
        value=avro_serializer(order, SerializationContext("orders", MessageField.VALUE))
    )
    producer.flush()
```

### Schema Evolution Rules
```
Change                  | Backward | Forward | Full
------------------------|----------|---------|------
Add field with default  | ✓        | ✓       | ✓
Remove field w/default  | ✓        | ✓       | ✓
Add optional field      | ✓        | ✓       | ✓
Remove optional field   | ✓        | ✓       | ✓
Change field type       | ✗        | ✗       | ✗
Rename field            | ✗        | ✗       | ✗
Add enum value          | ✗        | ✓       | ✗
```

---

## 4. Parquet - Columnar Storage with PyArrow

### Why Parquet
- Columnar format (excellent for analytics)
- Built-in compression (snappy, zstd, gzip)
- Predicate pushdown (read only needed columns)
- Schema evolution support
- Standard for data lakes

### Basic Usage
```python
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

# From Python dicts
data = {
    "symbol": ["AAPL", "GOOGL", "MSFT"],
    "price": [150.25, 2800.50, 380.75],
    "volume": [1000000, 500000, 750000]
}

table = pa.Table.from_pydict(data)
pq.write_table(table, "trades.parquet", compression="zstd")

# Read with column selection (predicate pushdown)
table = pq.read_table(
    "trades.parquet",
    columns=["symbol", "price"],  # Only read needed columns
    filters=[("price", ">", 200)]  # Row group filtering
)
```

### Polars Integration (Fastest)
```python
import polars as pl

# Write
df = pl.DataFrame({
    "timestamp": [...],
    "symbol": [...],
    "price": [...],
    "volume": [...]
})

df.write_parquet(
    "trades.parquet",
    compression="zstd",
    statistics=True,  # Enable predicate pushdown
    row_group_size=100_000
)

# Read with lazy evaluation
df = (
    pl.scan_parquet("trades/*.parquet")
    .filter(pl.col("symbol") == "AAPL")
    .select(["timestamp", "price", "volume"])
    .collect()
)
```

### Partitioned Writes (Data Lake Pattern)
```python
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime

def write_partitioned(df: pl.DataFrame, base_path: str):
    """Write Parquet with date partitioning."""
    table = df.to_arrow()
    
    pq.write_to_dataset(
        table,
        root_path=base_path,
        partition_cols=["date", "symbol"],
        compression="zstd",
        existing_data_behavior="overwrite_or_ignore"
    )

# Results in:
# base_path/
#   date=2026-01-25/
#     symbol=AAPL/
#       part-0.parquet
#     symbol=GOOGL/
#       part-0.parquet
```

---

## 5. Performance Selection Guide

### Use Case Matrix
```
Use Case                    | Recommended    | Why
----------------------------|----------------|-----------------------------
REST API responses          | orjson         | Fastest JSON, web standard
WebSocket messages          | MessagePack    | Compact, fast, bidirectional
Redis/Memcached caching     | MessagePack    | Compact binary, fast decode
Kafka event streaming       | Avro           | Schema registry, evolution
Data lake storage           | Parquet        | Columnar, compression, pushdown
Config files                | JSON (orjson)  | Human readable
Inter-service RPC           | Protobuf/gRPC  | See Cycle 40
ML model features           | Parquet        | Columnar reads, numpy compat
Time series (hot)           | MessagePack    | Fast append/read
Time series (cold)          | Parquet        | Compression, analytics
```

### Size Comparison (1000 records, typical trading data)
```
Format      | Size (KB) | Ratio | Notes
------------|-----------|-------|------------------
JSON        | 245       | 1.0x  | Baseline
orjson      | 245       | 1.0x  | Same output, faster
MessagePack | 156       | 0.64x | Binary efficiency
Avro        | 89        | 0.36x | Schema separate
Parquet     | 42        | 0.17x | Columnar + compression
```

---

## 6. Production Patterns

### Hybrid Serialization Strategy
```python
from enum import Enum
from typing import Protocol
import orjson
import msgpack

class SerializationFormat(Enum):
    JSON = "json"
    MSGPACK = "msgpack"

class Serializer(Protocol):
    def dumps(self, obj: dict) -> bytes: ...
    def loads(self, data: bytes) -> dict: ...

class OrjsonSerializer:
    def dumps(self, obj: dict) -> bytes:
        return orjson.dumps(obj)
    
    def loads(self, data: bytes) -> dict:
        return orjson.loads(data)

class MsgpackSerializer:
    def dumps(self, obj: dict) -> bytes:
        return msgpack.packb(obj)
    
    def loads(self, data: bytes) -> dict:
        return msgpack.unpackb(data)

def get_serializer(format: SerializationFormat) -> Serializer:
    return {
        SerializationFormat.JSON: OrjsonSerializer(),
        SerializationFormat.MSGPACK: MsgpackSerializer(),
    }[format]
```

### Content Negotiation (FastAPI)
```python
from fastapi import FastAPI, Request, Response
import orjson
import msgpack

app = FastAPI()

@app.middleware("http")
async def content_negotiation(request: Request, call_next):
    response = await call_next(request)
    
    accept = request.headers.get("accept", "application/json")
    
    if "application/msgpack" in accept and hasattr(response, "body"):
        body = orjson.loads(response.body)
        return Response(
            content=msgpack.packb(body),
            media_type="application/msgpack"
        )
    
    return response
```

---

## 7. Anti-Patterns

### ❌ Don't: Mix serialization in hot paths
```python
# BAD: Inconsistent serialization
def process_message(msg: bytes):
    try:
        data = json.loads(msg)  # Slow
    except:
        data = msgpack.unpackb(msg)  # Different format?
```

### ✓ Do: Consistent format per channel
```python
# GOOD: Single format per transport
class KafkaHandler:
    def __init__(self, serializer: Serializer):
        self.serializer = serializer  # One format for all Kafka
    
    def process(self, msg: bytes) -> dict:
        return self.serializer.loads(msg)
```

### ❌ Don't: Serialize datetime as strings unnecessarily
```python
# BAD: String datetime in binary format
msgpack.packb({"time": str(datetime.now())})  # Wastes space
```

### ✓ Do: Use appropriate type representations
```python
# GOOD: Epoch timestamp (compact)
msgpack.packb({"time": int(datetime.now().timestamp() * 1000)})

# Or use ExtType for full datetime
```

---

## 8. Benchmarking Template
```python
import time
import orjson
import msgpack
import json

def benchmark_serialization(data: dict, iterations: int = 10000):
    results = {}
    
    # orjson
    start = time.perf_counter()
    for _ in range(iterations):
        orjson.dumps(data)
    results["orjson_serialize"] = time.perf_counter() - start
    
    # msgpack
    start = time.perf_counter()
    for _ in range(iterations):
        msgpack.packb(data)
    results["msgpack_serialize"] = time.perf_counter() - start
    
    # stdlib json
    start = time.perf_counter()
    for _ in range(iterations):
        json.dumps(data)
    results["json_serialize"] = time.perf_counter() - start
    
    return results
```

---

## Key Takeaways

1. **orjson is the JSON default** - 5-10x faster, drop-in replacement
2. **MessagePack for internal binary** - Caching, WebSocket, compact storage
3. **Avro for event streaming** - Schema registry, Kafka, evolution
4. **Parquet for analytics** - Data lakes, columnar queries, ML features
5. **Pick one format per transport** - Don't mix in hot paths
6. **Benchmark your actual data** - Performance varies by structure

---

*Cycle 41 Complete - Message Serialization & Data Formats*
*Next: Cycle 42 - TBD (Quality-Diversity Expansion)*
