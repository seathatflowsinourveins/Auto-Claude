# Cycle 32: Performance & Profiling Patterns (January 25, 2026)

## Research Sources
- "Beyond cProfile: 7 Python Profiling Tools" (Dec 2025)
- "High-Performance Python: AsyncIO vs Multiprocessing" (Dec 2025)
- "Python Memory Profiling in 2025" (Dec 2025)
- Bloomberg Memray documentation
- Python 3.14 tracemalloc documentation
- pyinstrument 5.1.2 documentation

---

## 1. CPU PROFILING TOOLS

### Tool Selection Matrix

| Tool | Type | Overhead | Production Safe | Best For |
|------|------|----------|-----------------|----------|
| **cProfile** | Deterministic | High | No | Development only |
| **py-spy** | Sampling | ~1% | YES | Production profiling |
| **Scalene** | Sampling | Low | Yes | CPU + memory + GPU |
| **pyinstrument** | Sampling | Low | Yes | Call stacks, web apps |
| **Austin** | Sampling | Minimal | YES | Long-running processes |

### py-spy (Production Gold Standard)

```bash
# Attach to running process (NO CODE CHANGES)
py-spy record -o profile.svg --pid 12345

# Live top-like view
py-spy top --pid 12345

# Record with sampling rate
py-spy record -r 100 -o profile.svg -- python myapp.py

# Filter to specific thread
py-spy record --native --subprocesses -- python myapp.py
```

**Key Features**:
- Written in Rust, minimal overhead
- No code modification required
- Works in production containers
- Flame graph output (SVG)
- Can profile native extensions

### Scalene (CPU + Memory + GPU)

```bash
# Profile script with all metrics
scalene script.py

# Profile specific function
scalene --profile-only="my_module" script.py

# JSON output for automation
scalene --json script.py > profile.json

# GPU profiling (CUDA)
scalene --gpu script.py
```

**Output Example**:
```
Time   Memory   Copy    GPU
85%    12 MB    2 MB    15%    heavy_function:42
10%    50 MB    0 MB    80%    gpu_compute:78
```

### pyinstrument (Web-Friendly)

```python
from pyinstrument import Profiler

# Context manager
with Profiler() as profiler:
    result = slow_function()

# Output formats
profiler.output_text(unicode=True, color=True)
profiler.output_html()  # Browser-viewable

# Flask/Django middleware
from pyinstrument.middleware import ProfilerMiddleware
app.wsgi_app = ProfilerMiddleware(app.wsgi_app)
```

**Advantages**:
- Statistical profiling (low overhead)
- HTML output with interactive flamegraphs
- Django/Flask/FastAPI middleware built-in
- Async-aware

### cProfile (Development Only)

```python
import cProfile
import pstats
from io import StringIO

# Profile function
profiler = cProfile.Profile()
profiler.enable()
result = my_function()
profiler.disable()

# Analyze
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions

# Command line
# python -m cProfile -o output.prof script.py
# python -m pstats output.prof
```

**Limitations**:
- High overhead (2-3x slowdown)
- Not production safe
- Missing native extension profiling

---

## 2. ASYNC PROFILING

### AsyncIO Event Loop Profiling

```python
import asyncio

# Enable debug mode (development only)
asyncio.run(main(), debug=True)

# Or via environment
# PYTHONASYNCIODEBUG=1 python script.py
```

**Debug Mode Reports**:
- Coroutines that take >100ms
- Unawaited coroutines
- Resource warnings

### Profiling Async Code with cProfile

```python
import cProfile
import asyncio

async def main():
    await some_coroutine()

# Profile the async main
cProfile.run('asyncio.run(main())', 'async_profile.prof')
```

### py-spy with Async

```bash
# py-spy automatically handles async
py-spy record -o async_profile.svg --pid 12345

# With native extension support
py-spy record --native -- python async_app.py
```

### FastAPI Profiling Pattern

```python
from fastapi import FastAPI, Request
from pyinstrument import Profiler
from starlette.middleware.base import BaseHTTPMiddleware

class ProfilingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Only profile if requested
        if request.query_params.get("profile"):
            profiler = Profiler(async_mode="enabled")
            profiler.start()
            response = await call_next(request)
            profiler.stop()
            print(profiler.output_text())
            return response
        return await call_next(request)

app = FastAPI()
app.add_middleware(ProfilingMiddleware)
```

### uvloop Performance

```python
import uvloop

# 2-4x faster event loop
uvloop.install()

# Then run normally
asyncio.run(main())
```

**Benchmark**: uvloop provides 2-4x speedup for I/O-bound async operations.

---

## 3. MEMORY PROFILING

### tracemalloc (Built-in)

```python
import tracemalloc

# Start tracing
tracemalloc.start()

# Your code here
data = process_large_dataset()

# Get memory snapshot
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ Top 10 memory allocations ]")
for stat in top_stats[:10]:
    print(stat)

# Compare two snapshots (detect leaks)
snapshot1 = tracemalloc.take_snapshot()
# ... code that might leak ...
snapshot2 = tracemalloc.take_snapshot()

diff = snapshot2.compare_to(snapshot1, 'lineno')
for stat in diff[:10]:
    print(stat)
```

### Memray (Bloomberg - Production Grade)

```bash
# Install
pip install memray

# Profile script
memray run script.py

# Generate flame graph
memray flamegraph memray-script.py.bin

# Live view
memray run --live script.py

# pytest integration
pytest --memray tests/
```

**Key Features**:
- Tracks native (C/C++) allocations
- Flame graph visualization
- pytest plugin for regression testing
- Low overhead for production
- Temporal allocation tracking

### objgraph (Reference Debugging)

```python
import objgraph

# Most common types
objgraph.show_most_common_types(limit=10)

# Growth between calls (detect leaks)
objgraph.show_growth(limit=10)

# Find what references an object
objgraph.show_backrefs(my_object, filename='refs.png')

# Count specific types
print(objgraph.count('MyClass'))
```

### Memory Leak Detection Pattern

```python
import gc
import tracemalloc

def detect_memory_leak():
    tracemalloc.start()
    gc.collect()  # Clean slate
    
    snapshot1 = tracemalloc.take_snapshot()
    
    # Run suspected leaky code multiple times
    for _ in range(100):
        leaky_function()
    
    gc.collect()  # Force collection
    snapshot2 = tracemalloc.take_snapshot()
    
    # Compare
    diff = snapshot2.compare_to(snapshot1, 'lineno')
    
    print("Memory changes:")
    for stat in diff[:10]:
        if stat.size_diff > 0:  # Growing
            print(f"{stat.traceback.format()[0]}: +{stat.size_diff} bytes")
```

### Common Memory Leak Sources

1. **Circular References**:
```python
# BAD: Circular reference
class Node:
    def __init__(self):
        self.parent = None
        self.children = []
    
    def add_child(self, child):
        self.children.append(child)
        child.parent = self  # Circular!

# GOOD: Use weakref
import weakref

class Node:
    def __init__(self):
        self._parent = None  # weakref stored here
        self.children = []
    
    @property
    def parent(self):
        return self._parent() if self._parent else None
    
    def add_child(self, child):
        self.children.append(child)
        child._parent = weakref.ref(self)
```

2. **Global Caches Without Bounds**:
```python
# BAD: Unbounded cache
_cache = {}

def cached_fetch(key):
    if key not in _cache:
        _cache[key] = expensive_fetch(key)
    return _cache[key]

# GOOD: Bounded LRU cache
from functools import lru_cache

@lru_cache(maxsize=1024)
def cached_fetch(key):
    return expensive_fetch(key)
```

3. **Event Handler Accumulation**:
```python
# BAD: Handlers never removed
def setup():
    signal.connect(handler)  # Called repeatedly

# GOOD: Track and cleanup
def setup():
    cleanup()  # Remove old handlers first
    signal.connect(handler)
```

---

## 4. GARBAGE COLLECTION

### GC Module Patterns

```python
import gc

# Check GC stats
print(gc.get_stats())

# Force collection
collected = gc.collect()
print(f"Collected {collected} objects")

# Disable for performance-critical sections
gc.disable()
try:
    performance_critical_code()
finally:
    gc.enable()

# Debug mode
gc.set_debug(gc.DEBUG_LEAK)  # Report uncollectable objects
```

### GC Tuning

```python
import gc

# Get current thresholds
print(gc.get_threshold())  # Default: (700, 10, 10)

# Tune for long-running processes (less frequent full GC)
gc.set_threshold(50000, 500, 1000)

# For latency-sensitive apps, collect during idle
def idle_gc():
    gc.collect(generation=0)  # Only young generation
```

### Generation Explanation

```
Generation 0: New objects (collected most frequently)
Generation 1: Survived one collection
Generation 2: Long-lived objects (collected rarely)

Threshold (700, 10, 10) means:
- Gen 0 collected after 700 allocations - deallocations
- Gen 1 collected after 10 Gen 0 collections
- Gen 2 collected after 10 Gen 1 collections
```

---

## 5. BENCHMARKING

### timeit (Micro-benchmarks)

```python
import timeit

# Time a statement
result = timeit.timeit('sum(range(1000))', number=10000)
print(f"{result:.4f} seconds for 10000 iterations")

# Time a function
def my_function():
    return sum(range(1000))

result = timeit.timeit(my_function, number=10000)
```

### pytest-benchmark

```python
import pytest

def test_performance(benchmark):
    result = benchmark(my_function, arg1, arg2)
    assert result == expected

# Run with comparison
# pytest --benchmark-compare
```

### Continuous Benchmarking

```yaml
# GitHub Action for performance regression
- name: Run benchmarks
  run: |
    pytest --benchmark-json=benchmark.json tests/benchmarks/
    
- name: Compare with baseline
  uses: benchmark-action/github-action-benchmark@v1
  with:
    tool: 'pytest'
    output-file-path: benchmark.json
    fail-on-alert: true
    alert-threshold: '150%'  # Fail if 50% slower
```

---

## 6. PRODUCTION MONITORING

### Prometheus Metrics

```python
from prometheus_client import Histogram, Counter, Gauge
import time

# Response time histogram
REQUEST_LATENCY = Histogram(
    'request_latency_seconds',
    'Request latency',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

# Memory gauge
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage')

@REQUEST_LATENCY.time()
async def handle_request(request):
    ...
```

### Memory Monitoring Pattern

```python
import psutil
import os

def get_memory_info():
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    return {
        'rss_mb': mem.rss / 1024 / 1024,
        'vms_mb': mem.vms / 1024 / 1024,
        'percent': process.memory_percent()
    }
```

---

## 7. OPTIMIZATION TARGETS

### Performance Hierarchy (Biggest Impact First)

1. **Algorithm complexity** (O(n²) → O(n log n))
2. **I/O patterns** (batching, caching, async)
3. **Memory allocation** (object pooling, generators)
4. **CPU-bound work** (Cython, numba, multiprocessing)
5. **Micro-optimizations** (usually not worth it)

### Quick Wins

```python
# 1. Use generators for large datasets
# BAD
data = [process(x) for x in huge_list]
# GOOD
data = (process(x) for x in huge_list)

# 2. Use __slots__ for many instances
class Point:
    __slots__ = ['x', 'y']  # 40% less memory

# 3. Use local variables in hot loops
# BAD
for i in range(1000000):
    result = math.sqrt(i)
# GOOD
sqrt = math.sqrt
for i in range(1000000):
    result = sqrt(i)

# 4. Use built-in functions
# BAD
total = 0
for x in data:
    total += x
# GOOD
total = sum(data)

# 5. String joining
# BAD
s = ""
for part in parts:
    s += part
# GOOD
s = "".join(parts)
```

---

## Summary: Profiling Workflow

```
1. IDENTIFY: "Is it CPU, I/O, or memory?"
   - CPU-bound: py-spy, Scalene
   - I/O-bound: async profiling, tracing
   - Memory: memray, tracemalloc

2. MEASURE: Get baseline numbers
   - pytest-benchmark for regression detection
   - Production metrics (Prometheus)

3. PROFILE: Find hotspots
   - py-spy for production
   - Scalene for development (CPU + memory)

4. OPTIMIZE: Fix top 3 hotspots only
   - Algorithm first, micro-optimizations last
   - Verify improvement with benchmarks

5. MONITOR: Prevent regression
   - CI benchmark comparison
   - Production alerting
```

---

*Cycle 32 Complete - Performance & Profiling Patterns*
*Research Date: January 25, 2026*
