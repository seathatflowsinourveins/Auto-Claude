# Unleashed Platform V2 - Complete Overview

## Executive Summary

The Unleashed Platform V2 is a production-ready SDK orchestration system that integrates multiple AI/ML frameworks into a unified, scalable architecture. This document provides a comprehensive overview of all components, their interactions, and usage patterns.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI LAYER                               │
│  unleash status | adapters | pipelines | research | analyze     │
├─────────────────────────────────────────────────────────────────┤
│                     OBSERVABILITY LAYER                         │
│  Metrics | Tracing | Logging | Health Checks | Alerts           │
├─────────────────────────────────────────────────────────────────┤
│                       SECURITY LAYER                            │
│  Input Validation | API Keys | Rate Limiting | Audit Logging    │
├─────────────────────────────────────────────────────────────────┤
│                      PIPELINE LAYER                             │
│  Deep Research | Self-Improvement | Code Analysis | Evolution   │
├─────────────────────────────────────────────────────────────────┤
│                      ADAPTER LAYER                              │
│  DSPy | LangGraph | Mem0 | TextGrad | Aider | Exa | Firecrawl  │
├─────────────────────────────────────────────────────────────────┤
│                        CORE LAYER                               │
│  Async Executor | Caching | Error Handling | Config | Secrets   │
└─────────────────────────────────────────────────────────────────┘
```

## Component Inventory

### 1. SDK Adapters (`platform/adapters/`)

| Adapter | SDK | Purpose | Status |
|---------|-----|---------|--------|
| `dspy_adapter.py` | DSPy | Declarative prompt programming | ✅ Complete |
| `langgraph_adapter.py` | LangGraph | State graph workflows | ✅ Complete |
| `mem0_adapter.py` | Mem0 | Unified memory layer | ✅ Complete |
| `textgrad_adapter.py` | TextGrad | Gradient-based optimization | ✅ Complete |
| `aider_adapter.py` | Aider | AI pair programming | ✅ Complete |

### 2. Pipelines (`platform/pipelines/`)

| Pipeline | Purpose | Dependencies |
|----------|---------|--------------|
| `deep_research_pipeline.py` | Multi-source research | Exa, Firecrawl, Mem0 |
| `self_improvement_pipeline.py` | Prompt evolution | TextGrad, DSPy |
| `code_analysis_pipeline.py` | Code quality analysis | Aider, LangGraph |
| `agent_evolution_pipeline.py` | Genetic + gradient evolution | TextGrad, DSPy |

### 3. Core Infrastructure (`platform/core/`)

#### 3.1 Async Execution (`async_executor.py`)
- **CircuitBreaker**: Fault tolerance with configurable thresholds
- **RateLimiter**: Token bucket rate limiting
- **AsyncExecutor**: Concurrent execution with semaphores
- **TaskQueue**: Priority-based task scheduling

#### 3.2 Parallel Orchestration (`parallel_orchestrator.py`)
- **ParallelOrchestrator**: DAG-based task scheduling
- **AdapterPool**: Load balancing across adapters
- **Fan-out/Fan-in**: Parallel processing patterns

#### 3.3 Caching (`caching.py`)
- **MemoryCache**: LRU cache with TTL
- **FileCache**: Persistent file-based cache
- **RedisCache**: Distributed cache (requires Redis)
- **SemanticCache**: Embedding-based similarity cache
- **TieredCache**: Multi-backend with promotion

#### 3.4 Error Handling (`error_handling.py`)
- **UnleashedError**: Structured exception hierarchy
- **ErrorHandler**: Recovery strategies (Retry, Fallback, Cache)
- **ErrorAggregator**: Error pattern detection

#### 3.5 Monitoring (`monitoring.py`)
- **Counter/Gauge/Histogram/Summary**: Prometheus-style metrics
- **MetricRegistry**: Central metric management
- **Tracer**: Distributed tracing with spans
- **HealthChecker**: Component health monitoring
- **AlertManager**: Rule-based alerting
- **Profiler**: Performance profiling

#### 3.6 Observability (`observability.py`)
- **Observability**: Unified observability facade
- **ContextualLogger**: Structured logging with context
- **MonitoringDashboard**: Aggregated dashboard data

#### 3.7 Adapter Monitoring (`adapter_monitoring.py`)
- **AdapterHealthChecker**: Adapter-specific health checks
- **AdapterMetricsCollector**: Per-adapter metrics
- **AdapterMonitor**: Integrated adapter monitoring

#### 3.8 Security (`security.py`)
- **InputValidator**: SQL injection, XSS, path traversal detection
- **APIKeyManager**: Secure key generation and rotation
- **RateLimiter**: Per-key rate limiting
- **AuditLogger**: Security event logging
- **SecurityManager**: Unified security facade

#### 3.9 Configuration (`config_validation.py`)
- **ConfigSchema**: Schema-based validation
- **Validators**: Required, Range, Pattern, Choice, URL, Path
- **EnvironmentLoader**: Environment variable integration
- **ConfigurationManager**: Hot-reload support

#### 3.10 Secrets (`secrets.py`)
- **SecretBackend**: Pluggable backends (Environment, File, Memory)
- **EncryptedFileSecretBackend**: Encrypted at-rest storage
- **SecretsManager**: Unified secrets access with audit

### 4. CLI (`platform/cli/`)

```bash
# Platform status
unleash status

# List adapters
unleash adapters

# List pipelines
unleash pipelines

# Run research
unleash research "quantum computing applications" --depth comprehensive

# Analyze code
unleash analyze /path/to/code --depth deep

# Evolve prompt
unleash evolve "You are a helpful assistant" --generations 20

# Interactive mode
unleash interactive
```

## Quick Start

### 1. Basic Usage

```python
from platform.core import get_observability, get_security_manager
from platform.adapters import get_dspy_adapter, get_langgraph_adapter
from platform.pipelines import get_deep_research_pipeline

# Initialize observability
obs = get_observability()
logger = obs.get_logger(__name__)

# Initialize security
security = get_security_manager()

# Get adapters
DSPyAdapter = get_dspy_adapter()
if DSPyAdapter:
    dspy = DSPyAdapter(model="gpt-4o")

# Run pipeline
Pipeline = get_deep_research_pipeline()
if Pipeline:
    pipeline = Pipeline()
    result = await pipeline.research("AI safety", depth="comprehensive")
```

### 2. With Monitoring

```python
from platform.core import get_adapter_monitor
from platform.core.monitoring import timed, traced

monitor = get_adapter_monitor()

# Register adapter for monitoring
monitor.register_adapter("dspy", dspy_instance)

# Track calls automatically
result = await monitor.track_call("dspy", "generate", generate_func, prompt="...")

# Or use decorators
@timed("my_operation")
@traced("my_operation")
async def my_function():
    pass
```

### 3. With Caching

```python
from platform.core.caching import TieredCache, MemoryCache, FileCache

# Create tiered cache (memory → file)
cache = TieredCache([
    MemoryCache(max_size=1000),
    FileCache(cache_dir=".cache")
])

# Use cache
result = await cache.get("key")
if not result:
    result = await expensive_operation()
    await cache.set("key", result, ttl=3600)
```

### 4. With Security

```python
from platform.core.security import get_security_manager

security = get_security_manager()

# Generate API key
raw_key, key_info = security.key_manager.generate_key(
    name="my_service",
    permissions=["read", "write"],
    rate_limit=1000
)

# Validate request
allowed, error, key = await security.validate_request(
    api_key=raw_key,
    data=request_data
)
```

## Configuration

### Environment Variables

```bash
# Platform
UNLEASHED_ENV=production
UNLEASHED_LOG_LEVEL=INFO
UNLEASHED_LOG_FORMAT=json

# SDK Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
EXA_API_KEY=...
FIRECRAWL_API_KEY=...

# Features
UNLEASHED_CACHE_ENABLED=true
UNLEASHED_TRACING_ENABLED=true
UNLEASHED_METRICS_ENABLED=true
```

### Configuration File

```json
{
  "environment": "production",
  "log_level": "INFO",
  "security_level": "strict",
  "rate_limit_requests": 1000,
  "max_concurrent_tasks": 10,
  "cache_enabled": true,
  "cache_ttl": 3600
}
```

## Production Deployment

### Health Checks

```python
from platform.core import get_observability

obs = get_observability()
health = await obs.check_health()

# Returns:
# {
#   "adapter:dspy": {"status": "healthy", "duration_ms": 45},
#   "adapter:langgraph": {"status": "healthy", "duration_ms": 32},
#   "cache": {"status": "healthy", "duration_ms": 5},
#   ...
# }
```

### Prometheus Metrics

```python
metrics_output = obs.export_prometheus()
# Returns Prometheus-format metrics
```

### Dashboard Data

```python
dashboard = await obs.get_dashboard_data()
# Returns comprehensive JSON with metrics, traces, alerts, health
```

## Best Practices

### 1. Always Use Observability

```python
from platform.core.observability import observed

@observed(operation_name="process_request", log_args=True)
async def process_request(data):
    # Automatically logged, traced, and metered
    pass
```

### 2. Handle Errors Gracefully

```python
from platform.core.error_handling import ErrorHandler, ErrorContext

handler = ErrorHandler()
result = await handler.wrap(
    risky_function,
    context=ErrorContext(adapter="dspy", operation="generate")
)
```

### 3. Validate All Input

```python
from platform.core.security import get_security_manager

security = get_security_manager()
threat = security.validator.validate_dict(user_input)
if threat.is_threat:
    raise SecurityError(threat.details)
```

### 4. Use Tiered Caching

```python
# Fast memory cache with file fallback
cache = TieredCache([
    MemoryCache(max_size=1000, default_ttl=300),
    FileCache(cache_dir=".cache", default_ttl=3600)
])
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2024-01 | Complete V2 architecture with all layers |
| 1.5.0 | 2023-12 | Added TextGrad and Aider adapters |
| 1.0.0 | 2023-11 | Initial release with DSPy, LangGraph, Mem0 |

## Support

For issues and feature requests, please file tickets in the project repository.
