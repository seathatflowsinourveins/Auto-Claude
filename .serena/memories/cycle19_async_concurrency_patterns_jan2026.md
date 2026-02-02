# Cycle 19: Async & Concurrency Patterns (January 2026)

## CONCURRENCY MODEL SELECTION

### The Golden Rule
```
┌─────────────────────────────────────────────────────────────┐
│  I/O-Bound Tasks     → asyncio (single-threaded, cooperative)│
│  CPU-Bound Tasks     → multiprocessing (true parallelism)    │
│  Mixed/Legacy        → threading (simpler, GIL-limited)      │
└─────────────────────────────────────────────────────────────┘
```

### Decision Matrix (2026)
| Task Type | Best Tool | Why |
|-----------|-----------|-----|
| API calls, DB queries | asyncio | Non-blocking I/O, thousands of connections |
| File downloads | asyncio + aiofiles | Async file I/O |
| Image processing | multiprocessing | CPU-intensive, bypasses GIL |
| Web scraping | asyncio + httpx | High concurrency, low memory |
| ML inference | multiprocessing/threading | Depends on library (PyTorch releases GIL) |
| Mixed workloads | asyncio + to_thread() | Bridge sync code in async context |

## STRUCTURED CONCURRENCY (Python 3.11+)

### asyncio.TaskGroup (The Standard)
```python
import asyncio

async def fetch_data(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

async def main():
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(fetch_data("https://api1.example.com"))
        task2 = tg.create_task(fetch_data("https://api2.example.com"))
        task3 = tg.create_task(fetch_data("https://api3.example.com"))
    
    # All tasks complete or ALL cancel on first exception
    results = [task1.result(), task2.result(), task3.result()]
```

**Key benefit**: If ANY task raises an exception, ALL other tasks are cancelled automatically. No zombie tasks.

### TaskGroup Limitations
- Cannot list or cancel individual tasks
- Cannot add tasks after entering context
- AnyIO/Trio offer more control

## ANYIO: STRUCTURED CONCURRENCY DONE RIGHT

### Why AnyIO Over Raw asyncio
1. **Cancel scopes** - fine-grained cancellation control
2. **Task introspection** - list and manage running tasks
3. **Backend agnostic** - works with asyncio and Trio
4. **Better APIs** - Trio-inspired, more intuitive

### AnyIO Patterns
```python
import anyio

async def main():
    # Task group with cancel scope
    async with anyio.create_task_group() as tg:
        tg.start_soon(fetch_data, "url1")
        tg.start_soon(fetch_data, "url2")
        
        # Cancel scope with timeout
        with anyio.move_on_after(5.0) as scope:
            tg.start_soon(slow_operation)
        
        if scope.cancelled_caught:
            print("Operation timed out")

# Memory streams for async producer/consumer
async def pipeline():
    send_stream, receive_stream = anyio.create_memory_object_stream()
    
    async with anyio.create_task_group() as tg:
        tg.start_soon(producer, send_stream)
        tg.start_soon(consumer, receive_stream)
```

### Cancel Scopes (Trio/AnyIO)
```python
import anyio

async def cancellation_safe():
    # Timeout without raising exception
    with anyio.move_on_after(10.0) as scope:
        await long_running_operation()
    
    if scope.cancelled_caught:
        await cleanup()  # Always runs
        return None
    
    return result

# Nested cancel scopes
async def nested_cancellation():
    with anyio.CancelScope() as outer:
        with anyio.move_on_after(5.0) as inner:
            await operation()
        
        if inner.cancelled_caught:
            # Inner timed out, but outer continues
            await fallback_operation()
```

## ASYNCIO BEST PRACTICES (2026)

### 1. Never Block the Event Loop
```python
# BAD - blocks event loop
def cpu_intensive():
    return sum(i * i for i in range(10_000_000))

# GOOD - run in thread pool
async def cpu_intensive_async():
    return await asyncio.to_thread(cpu_intensive)

# BETTER - run in process pool for true parallelism
async def cpu_intensive_parallel():
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as pool:
        return await loop.run_in_executor(pool, cpu_intensive)
```

### 2. Proper Resource Management
```python
# GOOD - async context manager for cleanup
async def fetch_with_cleanup():
    async with httpx.AsyncClient() as client:
        # Client automatically closed on exit/exception
        return await client.get("https://api.example.com")

# For database connections
async def db_operation():
    async with asyncpg.create_pool(dsn) as pool:
        async with pool.acquire() as conn:
            return await conn.fetch("SELECT * FROM users")
```

### 3. Limit Concurrency (Semaphores)
```python
# Prevent overwhelming services
semaphore = asyncio.Semaphore(10)  # Max 10 concurrent

async def rate_limited_fetch(url: str):
    async with semaphore:
        async with httpx.AsyncClient() as client:
            return await client.get(url)

# Process many URLs with bounded concurrency
async def fetch_all(urls: list[str]):
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(rate_limited_fetch(url)) for url in urls]
    return [t.result() for t in tasks]
```

### 4. Graceful Shutdown
```python
import signal

async def graceful_shutdown():
    loop = asyncio.get_running_loop()
    
    async def shutdown(sig):
        print(f"Received {sig.name}")
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        loop.stop()
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s)))
```

## EVENT LOOP OPTIMIZATION

### uvloop (Production Standard)
```python
# 2-4x faster than default asyncio event loop
import uvloop

# Method 1: Set as default policy
uvloop.install()

# Method 2: Use explicitly
async def main():
    ...

asyncio.run(main())  # Uses uvloop after install()
```

### Custom Event Loop Internals
```python
# Access loop internals for advanced tuning
loop = asyncio.get_running_loop()

# Thread-safe scheduling from sync code
loop.call_soon_threadsafe(callback, arg)

# Low-level: schedule at specific time
loop.call_at(loop.time() + 5.0, callback)

# Debug mode for development
asyncio.run(main(), debug=True)
```

## THREADING VS ASYNCIO

### When Threading is Still Appropriate
```python
import threading
from concurrent.futures import ThreadPoolExecutor

# Legacy sync libraries
def blocking_library_call():
    return requests.get("https://api.example.com")  # Sync library

# Thread pool for multiple sync calls
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(blocking_library_call) for _ in range(10)]
    results = [f.result() for f in futures]

# Bridge to asyncio
async def async_wrapper():
    return await asyncio.to_thread(blocking_library_call)
```

### GIL Considerations
```python
# GIL LIMITS threads for CPU work
# But releases during I/O operations

# Some libraries release GIL:
# - NumPy (during array operations)
# - PyTorch (during CUDA operations)
# - Most C extensions (during C code)

# Python 3.13+ free-threaded mode (experimental)
# Compile with: ./configure --disable-gil
```

## MULTIPROCESSING FOR CPU-BOUND

### Process Pool Pattern
```python
from concurrent.futures import ProcessPoolExecutor
import asyncio

def cpu_heavy(n: int) -> int:
    return sum(i * i for i in range(n))

async def parallel_compute(tasks: list[int]) -> list[int]:
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as pool:
        futures = [loop.run_in_executor(pool, cpu_heavy, n) for n in tasks]
        return await asyncio.gather(*futures)

# Results: True parallelism, bypasses GIL
```

### Shared State with Multiprocessing
```python
from multiprocessing import Manager, Pool

def worker(shared_dict, key, value):
    shared_dict[key] = value

if __name__ == "__main__":
    with Manager() as manager:
        shared = manager.dict()
        with Pool(4) as pool:
            pool.starmap(worker, [(shared, i, i*2) for i in range(10)])
        print(dict(shared))
```

## HIGH-THROUGHPUT PATTERNS (2026)

### Async Queue for Producer/Consumer
```python
async def producer(queue: asyncio.Queue):
    for i in range(100):
        await queue.put({"id": i, "data": f"item_{i}"})
    await queue.put(None)  # Sentinel

async def consumer(queue: asyncio.Queue):
    while True:
        item = await queue.get()
        if item is None:
            break
        await process(item)
        queue.task_done()

async def pipeline():
    queue = asyncio.Queue(maxsize=10)  # Backpressure
    async with asyncio.TaskGroup() as tg:
        tg.create_task(producer(queue))
        # Multiple consumers
        for _ in range(3):
            tg.create_task(consumer(queue))
```

### Batching for Efficiency
```python
async def batch_processor(items: list, batch_size: int = 100):
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        await asyncio.gather(*[process(item) for item in batch])
        await asyncio.sleep(0)  # Yield to event loop
```

## COMMON MISTAKES TO AVOID

### 1. Forgetting to await
```python
# BAD - coroutine never runs
async def mistake():
    fetch_data()  # Missing await!

# GOOD
async def correct():
    await fetch_data()
```

### 2. Creating tasks without tracking
```python
# BAD - task may be garbage collected
async def fire_and_forget():
    asyncio.create_task(background_job())  # Untracked!

# GOOD - store reference
async def tracked():
    task = asyncio.create_task(background_job())
    await task  # Or store in a set
```

### 3. Blocking in async functions
```python
# BAD - blocks event loop
async def blocking():
    time.sleep(1)  # Sync sleep!

# GOOD
async def non_blocking():
    await asyncio.sleep(1)  # Async sleep
```

## SUMMARY TABLE

| Pattern | Tool | Use Case |
|---------|------|----------|
| Structured concurrency | TaskGroup / AnyIO | Safe task management |
| Rate limiting | Semaphore | API throttling |
| Graceful shutdown | Signal handlers | Production servers |
| Fast event loop | uvloop | Production performance |
| CPU parallelism | ProcessPoolExecutor | Heavy computation |
| Sync bridge | asyncio.to_thread | Legacy libraries |
| Backpressure | Queue(maxsize) | Producer/consumer |

---
*Cycle 19 - Async & Concurrency Patterns - January 2026*
