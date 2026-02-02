# Cycle 24: Structured Logging & Observability (January 2026)

## structlog - Production Logging

### Core Philosophy
- **Simple**: Functions that take and return dictionaries
- **Powerful**: Full control over processing pipeline
- **Fast**: No legacy design constraints
- Follows Twelve-Factor App methodology: log to stdout, let platform handle rest

### Basic Setup

```python
import structlog

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()  # Production
        # structlog.dev.ConsoleRenderer()    # Development
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger()
```

### Canonical Log Lines Pattern

```python
# BAD: Many small log lines
log.info("Starting request")
log.info("User authenticated")
log.info("Query completed")
log.info("Request finished")

# GOOD: Single rich canonical log line
log.info(
    "request_completed",
    user_id=user.id,
    endpoint="/api/users",
    method="GET",
    duration_ms=45.2,
    status_code=200,
    query_count=3,
    cache_hit=True
)
```

### Context Variables (Request Scoping)

```python
import structlog
from contextvars import ContextVar

# Bind context for entire request
structlog.contextvars.bind_contextvars(
    request_id=request_id,
    user_id=user.id,
    session_id=session.id
)

# All subsequent logs include this context
log.info("processing_order", order_id=123)
# Output: {"event": "processing_order", "request_id": "abc", "user_id": 42, "order_id": 123}

# Clear at end of request
structlog.contextvars.clear_contextvars()
```

### FastAPI Middleware

```python
from fastapi import FastAPI, Request
from uuid import uuid4
import structlog

app = FastAPI()
log = structlog.get_logger()

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path
    )
    
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = (time.perf_counter() - start_time) * 1000
    
    log.info(
        "request_completed",
        status_code=response.status_code,
        duration_ms=round(duration, 2)
    )
    
    response.headers["X-Request-ID"] = request_id
    return response
```

---

## OpenTelemetry - Distributed Tracing

### Installation

```bash
pip install opentelemetry-api opentelemetry-sdk
pip install opentelemetry-instrumentation-fastapi
pip install opentelemetry-instrumentation-requests
pip install opentelemetry-exporter-otlp
```

### Basic Setup

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# Configure provider
resource = Resource.create({
    "service.name": "my-service",
    "service.version": "1.0.0",
    "deployment.environment": "production"
})

provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)
```

### Manual Span Creation

```python
from opentelemetry import trace

tracer = trace.get_tracer("my.tracer")

# Basic span
with tracer.start_as_current_span("process_order") as span:
    span.set_attribute("order.id", order_id)
    span.set_attribute("order.total", total)
    result = process(order)
    span.set_attribute("result.status", result.status)

# Nested spans (parent-child relationship)
with tracer.start_as_current_span("parent_operation"):
    # Work in parent
    with tracer.start_as_current_span("child_operation"):
        # Work in child (automatically linked to parent)
        pass

# Get current span for attribute addition
current_span = trace.get_current_span()
current_span.set_attribute("user.id", user_id)
```

### FastAPI Auto-Instrumentation

```python
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from fastapi import FastAPI

app = FastAPI()
FastAPIInstrumentor.instrument_app(app)

# All endpoints automatically traced
@app.get("/users/{user_id}")
async def get_user(user_id: str):
    # Span created automatically with attributes:
    # http.method, http.url, http.status_code, etc.
    return {"user_id": user_id}
```

### Context Propagation (W3C Trace Context)

```python
from opentelemetry import trace
from opentelemetry.propagate import inject, extract

# Outgoing request: inject context into headers
headers = {}
inject(headers)  # Adds traceparent, tracestate headers

response = requests.get("http://other-service/api", headers=headers)

# Incoming request: extract context from headers
context = extract(request.headers)
with tracer.start_as_current_span("handle_request", context=context):
    # Span linked to upstream trace
    pass
```

---

## Correlation: Structlog + OpenTelemetry

### Unified Trace ID in Logs

```python
import structlog
from opentelemetry import trace

def add_trace_context(logger, method_name, event_dict):
    """Processor that adds trace context to log events."""
    span = trace.get_current_span()
    if span.is_recording():
        ctx = span.get_span_context()
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        add_trace_context,  # Add trace IDs to every log
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
```

### Example Output

```json
{
  "event": "order_processed",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "span_id": "00f067aa0ba902b7",
  "request_id": "req-abc-123",
  "user_id": 42,
  "order_id": 1234,
  "duration_ms": 45.2,
  "timestamp": "2026-01-25T10:30:00.000Z",
  "level": "info"
}
```

---

## Production Patterns

### Log Levels Strategy

| Level | Use Case |
|-------|----------|
| DEBUG | Verbose debugging (disabled in prod) |
| INFO | Normal operations, business events |
| WARNING | Degraded operation, recoverable issues |
| ERROR | Operation failed, needs attention |
| CRITICAL | System failure, immediate action needed |

### Async Logging (Non-Blocking)

```python
import structlog
from structlog.threadlocal import wrap_dict

# For high-throughput, use thread-local
structlog.configure(
    context_class=wrap_dict(dict),
    cache_logger_on_first_use=True,
)

# Or use async logging handler
import logging
from logging.handlers import QueueHandler, QueueListener
import queue

log_queue = queue.Queue(-1)
handler = QueueHandler(log_queue)
listener = QueueListener(log_queue, logging.StreamHandler())
listener.start()
```

### Sampling for High Volume

```python
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

# Sample 10% of traces
sampler = TraceIdRatioBased(0.1)
provider = TracerProvider(sampler=sampler)

# Always sample errors
from opentelemetry.sdk.trace.sampling import ParentBasedTraceIdRatio
sampler = ParentBasedTraceIdRatio(0.1)  # But respects parent decision
```

### Sensitive Data Redaction

```python
def redact_sensitive(logger, method_name, event_dict):
    """Redact PII and secrets from logs."""
    sensitive_keys = {'password', 'token', 'api_key', 'ssn', 'credit_card'}
    for key in event_dict:
        if any(s in key.lower() for s in sensitive_keys):
            event_dict[key] = "[REDACTED]"
    return event_dict

structlog.configure(
    processors=[
        redact_sensitive,
        # ... other processors
    ]
)
```

---

## Three Pillars Integration

### Unified Observability

```
┌─────────────────────────────────────────────────────────┐
│                    Single Request                        │
├─────────────────────────────────────────────────────────┤
│  TRACE: trace_id=abc123                                 │
│    └─ Span: /api/orders (200ms)                         │
│        └─ Span: database.query (50ms)                   │
│        └─ Span: external.payment (100ms)                │
├─────────────────────────────────────────────────────────┤
│  LOGS: All with trace_id=abc123                         │
│    - "order_received" {user_id: 42}                     │
│    - "payment_processed" {amount: 99.99}                │
│    - "order_completed" {order_id: 1234}                 │
├─────────────────────────────────────────────────────────┤
│  METRICS: Tagged with trace exemplars                   │
│    - request_duration_ms: 200                           │
│    - orders_total: +1                                   │
└─────────────────────────────────────────────────────────┘
```

### OTLP Export Configuration

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

# Traces
trace_exporter = OTLPSpanExporter(endpoint="http://otel-collector:4317")

# Metrics
metric_exporter = OTLPMetricExporter(endpoint="http://otel-collector:4317")

# Logs (experimental in Python)
# Use structlog → JSON → log aggregator instead
```

---

## Anti-Patterns to Avoid

1. **String formatting in logs** - Use structured fields
   ```python
   # BAD: log.info(f"User {user_id} created order {order_id}")
   # GOOD: log.info("order_created", user_id=user_id, order_id=order_id)
   ```

2. **Missing correlation IDs** - Always include trace_id

3. **Logging sensitive data** - Use redaction processors

4. **Blocking log calls** - Use async handlers for high throughput

5. **Too many log levels** - Stick to INFO in production, DEBUG off

6. **No span attributes** - Add business context to spans

7. **Ignoring sampling** - Required for high-volume systems

---

## Production Checklist

- [ ] structlog configured with JSON output
- [ ] OpenTelemetry traces enabled
- [ ] Trace ID propagation across services
- [ ] Logs include trace_id and span_id
- [ ] Sensitive data redacted
- [ ] Request-scoped context via contextvars
- [ ] Sampling configured for high volume
- [ ] OTLP export to collector
- [ ] Log aggregation (Grafana Loki, Elasticsearch)

*Research Date: January 25, 2026*
*Sources: structlog docs, OpenTelemetry Python docs, Dash0, Last9*
