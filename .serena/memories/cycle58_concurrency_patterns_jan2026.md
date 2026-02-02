# Cycle 58: Concurrency Patterns (Jan 2026)

Production concurrency patterns from official Python documentation.

## concurrent.futures (Python 3.13+)

### ThreadPoolExecutor - I/O-Bound Tasks

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx

def fetch_url(url: str) -> dict:
    """Blocking I/O - perfect for ThreadPoolExecutor."""
    with httpx.Client() as client:
        response = client.get(url)
        return {"url": url, "status": response.status_code}

def fetch_all_urls(urls: list[str], max_workers: int = 10) -> list[dict]:
    """Process multiple URLs concurrently with threads."""
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_url = {executor.submit(fetch_url, url): url for url in urls}
        
        # Process as they complete (not in submission order)
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as e:
                results.append({"url": url, "error": str(e)})
    
    return results
```

### ProcessPoolExecutor - CPU-Bound Tasks

```python
from concurrent.futures import ProcessPoolExecutor
import math

def compute_prime_factors(n: int) -> list[int]:
    """CPU-intensive task - use ProcessPoolExecutor."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def process_numbers(numbers: list[int]) -> dict[int, list[int]]:
    """Bypass GIL with separate processes for CPU work."""
    results = {}
    
    # max_workers defaults to number of CPUs
    with ProcessPoolExecutor() as executor:
        future_to_num = {
            executor.submit(compute_prime_factors, n): n 
            for n in numbers
        }
        
        for future in as_completed(future_to_num):
            num = future_to_num[future]
            results[num] = future.result()
    
    return results
```

### Executor.map() for Simple Cases

```python
from concurrent.futures import ThreadPoolExecutor

def process_item(item: str) -> str:
    return item.upper()

# Simple map - maintains order, simpler than submit()
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_item, ["a", "b", "c"]))
    # results: ["A", "B", "C"] - order preserved
```

## Python 3.13+ GIL Changes (PEP 703)

### Free-Threading Mode (Experimental)

```python
# Check if running in free-threaded mode
import sys
print(sys.flags.nogil)  # True if GIL disabled

# Python 3.13+ can be built with --disable-gil
# This enables true parallel execution in threads

# Install free-threaded Python:
# pyenv install 3.13.0t  (the 't' suffix)
# Or: python3.13t (if installed separately)
```

### When GIL Still Matters

```python
# GIL released during:
# - I/O operations (file, network, sleep)
# - Calling C extensions that release GIL (NumPy, etc.)
# - Waiting on locks/conditions

# GIL held during:
# - Pure Python computation
# - Object allocation/deallocation
# - Reference counting

# Rule of thumb:
# - I/O-bound → ThreadPoolExecutor (GIL released during I/O)
# - CPU-bound pure Python → ProcessPoolExecutor (bypasses GIL)
# - CPU-bound with NumPy → ThreadPoolExecutor (NumPy releases GIL)
```

## asyncio + Threading Integration

### run_in_executor() - Bridge Sync/Async

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import httpx

# Create a dedicated executor for blocking I/O
io_executor = ThreadPoolExecutor(max_workers=20, thread_name_prefix="io-")

def blocking_http_call(url: str) -> str:
    """Blocking call that can't be made async."""
    with httpx.Client() as client:
        return client.get(url).text

async def fetch_with_executor(url: str) -> str:
    """Run blocking code in thread pool from async context."""
    loop = asyncio.get_running_loop()
    
    # Run blocking function in thread pool
    result = await loop.run_in_executor(io_executor, blocking_http_call, url)
    return result

async def main():
    urls = ["https://api1.com", "https://api2.com"]
    
    # Run blocking calls concurrently from async
    tasks = [fetch_with_executor(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results

# Cleanup executor on shutdown
async def shutdown():
    io_executor.shutdown(wait=True)
```

### asyncio.to_thread() - Python 3.9+ Shorthand

```python
import asyncio

def cpu_intensive_task(data: bytes) -> bytes:
    """Blocking CPU work."""
    import hashlib
    return hashlib.sha256(data).digest()

async def process_async(data: bytes) -> bytes:
    """Simpler than run_in_executor for one-off calls."""
    # Uses default ThreadPoolExecutor internally
    result = await asyncio.to_thread(cpu_intensive_task, data)
    return result
```

## Advanced Patterns

### Cancellation and Timeouts

```python
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time

def slow_task(seconds: int) -> str:
    time.sleep(seconds)
    return f"Slept {seconds}s"

with ThreadPoolExecutor() as executor:
    future = executor.submit(slow_task, 10)
    
    try:
        # Wait max 2 seconds for result
        result = future.result(timeout=2)
    except TimeoutError:
        # Task still running, but we stop waiting
        future.cancel()  # Request cancellation (not guaranteed)
        print("Task timed out")
```

### Executor Context Managers

```python
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

@contextmanager
def managed_executor(max_workers: int = 10):
    """Executor with proper cleanup."""
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        yield executor
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

# Usage
with managed_executor(5) as executor:
    futures = [executor.submit(task, i) for i in range(100)]
```

### ProcessPoolExecutor with Initializer

```python
from concurrent.futures import ProcessPoolExecutor
import heavy_module  # Expensive to import

# Global in worker process
_model = None

def init_worker():
    """Run once per worker process."""
    global _model
    _model = heavy_module.load_model()

def process_with_model(data):
    """Use pre-initialized model."""
    return _model.predict(data)

# Workers initialized once, model reused
with ProcessPoolExecutor(
    max_workers=4,
    initializer=init_worker
) as executor:
    results = list(executor.map(process_with_model, dataset))
```

## Anti-Patterns

### ❌ Creating Executor Per Task

```python
# BAD: New executor for each call
def bad_parallel(items):
    results = []
    for item in items:
        with ThreadPoolExecutor() as executor:  # Overhead!
            future = executor.submit(process, item)
            results.append(future.result())
    return results

# GOOD: Reuse executor
def good_parallel(items):
    with ThreadPoolExecutor() as executor:
        return list(executor.map(process, items))
```

### ❌ Mixing Sync/Async Incorrectly

```python
# BAD: Blocking the event loop
async def bad_async():
    result = blocking_function()  # Blocks entire event loop!
    return result

# GOOD: Use run_in_executor
async def good_async():
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, blocking_function)
    return result
```

### ❌ Too Many Workers

```python
# BAD: More workers than useful
with ThreadPoolExecutor(max_workers=1000) as executor:  # Overhead
    ...

# GOOD: Match to workload
# I/O-bound: 20-100 workers (waiting on network)
# CPU-bound: os.cpu_count() workers (using cores)
import os
with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() + 4)) as executor:
    ...
```

## InterpreterPoolExecutor (Python 3.15 Preview)

```python
# Coming in Python 3.15 - per-interpreter GIL
from concurrent.futures import InterpreterPoolExecutor

# Each interpreter has its own GIL
# True parallelism without separate processes
# Lower overhead than ProcessPoolExecutor

with InterpreterPoolExecutor() as executor:
    # Parallel Python execution with shared memory
    results = list(executor.map(cpu_bound_task, data))
```

## Decision Matrix

| Workload Type | Executor | Why |
|--------------|----------|-----|
| Network I/O | ThreadPoolExecutor | GIL released during I/O |
| File I/O | ThreadPoolExecutor | GIL released during I/O |
| NumPy/SciPy | ThreadPoolExecutor | C extensions release GIL |
| Pure Python CPU | ProcessPoolExecutor | Bypasses GIL entirely |
| Database queries | ThreadPoolExecutor | I/O-bound waiting |
| Image processing | ProcessPoolExecutor | CPU-intensive |
| ML inference | ThreadPoolExecutor | Most frameworks release GIL |

## Quick Reference

```python
# ThreadPoolExecutor: I/O, database, network
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=20) as executor:
    results = executor.map(io_task, items)

# ProcessPoolExecutor: CPU-intensive pure Python
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor() as executor:
    results = executor.map(cpu_task, items)

# asyncio bridge: Run sync code in async context
import asyncio
result = await asyncio.to_thread(blocking_func, arg)

# Check for timeout
future.result(timeout=5.0)  # Raises TimeoutError

# Cancel pending futures on shutdown
executor.shutdown(wait=True, cancel_futures=True)
```
