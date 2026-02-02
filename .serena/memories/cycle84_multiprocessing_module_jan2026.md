# Python multiprocessing Module - Production Patterns (January 2026)

## Core Module Overview

The `multiprocessing` module provides process-based parallelism, effectively bypassing the GIL by using subprocesses instead of threads. Use for **CPU-bound** tasks where true parallel execution is needed.

```python
import multiprocessing as mp
```

**Key Difference from threading**: Processes have separate memory spaces - data must be explicitly shared or passed via IPC (queues, pipes, shared memory).

---

## Start Methods (Critical for Production)

### Python 3.14+ Defaults Changed!

| Method | Default On | Description |
|--------|-----------|-------------|
| `spawn` | Windows, macOS | Fresh Python interpreter, pickle args, safest |
| `forkserver` | Linux (3.14+) | Single-threaded server forks workers, fast + safe |
| `fork` | None (legacy) | Copy-on-write, fast but unsafe with threads |

```python
import multiprocessing as mp

# Method 1: Global setting (once per program)
if __name__ == '__main__':
    mp.set_start_method('spawn')  # or 'fork', 'forkserver'

# Method 2: Context (allows multiple methods in same program)
ctx = mp.get_context('spawn')
p = ctx.Process(target=worker)
q = ctx.Queue()
```

### Fork vs Spawn Trade-offs

```python
# SPAWN (Recommended for production)
# + Safe with threads and native libraries
# + Cross-platform (Windows, macOS, Linux)
# - Slower startup (new Python interpreter)
# - All data must be picklable

# FORK (Legacy, use with caution)
# + Fast startup (copy-on-write)
# + Inherits file descriptors, globals
# - UNSAFE with threads (deadlocks, crashes)
# - macOS: Deprecated, causes crashes
# - Not available on Windows

# FORKSERVER (Best of both worlds on Linux)
# + Fast like fork
# + Safe (server is single-threaded)
# + No unnecessary resource inheritance
# - Linux only
```

---

## Process Class

### Basic Usage

```python
from multiprocessing import Process
import os

def worker(name: str, value: int) -> None:
    print(f"Worker {name} (PID: {os.getpid()}) processing {value}")

if __name__ == '__main__':
    # Create and start process
    p = Process(target=worker, args=("A", 42), name="WorkerA")
    p.start()
    
    print(f"Parent PID: {os.getpid()}")
    print(f"Child PID: {p.pid}")
    print(f"Child alive: {p.is_alive()}")
    
    p.join()  # Wait for completion
    print(f"Exit code: {p.exitcode}")
```

### Process Attributes and Methods

```python
p = Process(target=func, args=(), kwargs={}, name="Worker", daemon=False)

# Before start()
p.daemon = True  # Dies when parent exits (set before start!)
p.name = "CustomName"

p.start()  # Begin execution

# After start()
p.pid           # Process ID
p.is_alive()    # True if running
p.exitcode      # None while running, 0 = success, >0 = error, <0 = signal

# Control
p.join(timeout=5.0)  # Wait (with optional timeout)
p.terminate()        # SIGTERM (graceful)
p.kill()            # SIGKILL (immediate, Python 3.7+)
p.close()           # Release resources (Python 3.7+)
```

### Subclassing Process

```python
from multiprocessing import Process

class DataProcessor(Process):
    def __init__(self, data: list):
        super().__init__()
        self.data = data
        self.result = None  # Won't be shared! Use Queue/Pipe
    
    def run(self):
        # This runs in child process
        self.result = sum(self.data)
        print(f"Processed: {self.result}")

if __name__ == '__main__':
    p = DataProcessor([1, 2, 3, 4, 5])
    p.start()
    p.join()
    # p.result is None in parent! Separate memory space
```

---

## Pool - Worker Pool Pattern

The most common pattern for parallel processing.

### Basic Pool Usage

```python
from multiprocessing import Pool

def square(x: int) -> int:
    return x * x

if __name__ == '__main__':
    # Context manager ensures cleanup
    with Pool(processes=4) as pool:
        # Blocking methods
        results = pool.map(square, range(10))        # [0, 1, 4, 9, ...]
        result = pool.apply(square, (5,))            # 25
        
        # Non-blocking (async) methods
        async_result = pool.apply_async(square, (5,))
        print(async_result.get(timeout=1.0))         # 25
        
        # Lazy iterators (memory efficient)
        for result in pool.imap(square, range(1000)):
            print(result)
        
        # Unordered (faster, results as completed)
        for result in pool.imap_unordered(square, range(1000)):
            print(result)
```

### Pool with Initializer

```python
import multiprocessing as mp

# Global in worker process
db_connection = None

def init_worker(db_url: str):
    """Called once per worker process."""
    global db_connection
    db_connection = connect_to_database(db_url)

def process_record(record_id: int):
    """Uses initialized connection."""
    return db_connection.query(record_id)

if __name__ == '__main__':
    with mp.Pool(
        processes=4,
        initializer=init_worker,
        initargs=("postgresql://localhost/db",)
    ) as pool:
        results = pool.map(process_record, record_ids)
```

### Pool with maxtasksperchild

```python
# Restart workers after N tasks (prevents memory leaks)
with Pool(processes=4, maxtasksperchild=100) as pool:
    results = pool.map(memory_intensive_func, large_dataset)
```

### AsyncResult Methods

```python
async_result = pool.apply_async(func, args)

async_result.get(timeout=None)  # Block until result (raises on error)
async_result.wait(timeout=None) # Block until ready
async_result.ready()            # True if completed
async_result.successful()       # True if completed without error
```

---

## Queues and Pipes (IPC)

### Queue - Multi-producer, Multi-consumer

```python
from multiprocessing import Process, Queue

def producer(q: Queue, items: list):
    for item in items:
        q.put(item)
    q.put(None)  # Sentinel

def consumer(q: Queue):
    while True:
        item = q.get()
        if item is None:
            break
        print(f"Processing: {item}")

if __name__ == '__main__':
    queue = Queue(maxsize=100)  # Bounded queue
    
    prod = Process(target=producer, args=(queue, [1, 2, 3]))
    cons = Process(target=consumer, args=(queue,))
    
    prod.start()
    cons.start()
    prod.join()
    cons.join()
```

### Queue Methods

```python
q = Queue(maxsize=0)  # 0 = unlimited

q.put(item, block=True, timeout=None)
q.put_nowait(item)  # Raises Full if queue is full

item = q.get(block=True, timeout=None)
item = q.get_nowait()  # Raises Empty if queue is empty

q.empty()   # Approximate, not reliable for sync
q.full()    # Approximate, not reliable for sync
q.qsize()   # Approximate size

q.close()   # No more puts allowed
q.join_thread()  # Wait for background thread
```

### Pipe - Two-way Communication

```python
from multiprocessing import Process, Pipe

def child(conn):
    conn.send("Hello from child")
    msg = conn.recv()
    print(f"Child received: {msg}")
    conn.close()

if __name__ == '__main__':
    parent_conn, child_conn = Pipe()  # duplex=True by default
    
    p = Process(target=child, args=(child_conn,))
    p.start()
    
    print(f"Parent received: {parent_conn.recv()}")
    parent_conn.send("Hello from parent")
    
    p.join()
```

### Pipe Methods

```python
conn1, conn2 = Pipe(duplex=True)  # Two-way
conn1, conn2 = Pipe(duplex=False) # One-way (conn1 recv only, conn2 send only)

conn.send(obj)           # Send picklable object
conn.recv()              # Receive object (blocks)
conn.poll(timeout=None)  # True if data available

conn.send_bytes(buffer)  # Send raw bytes
conn.recv_bytes(maxlength)
conn.recv_bytes_into(buffer)

conn.fileno()            # File descriptor
conn.close()
```

---

## Shared State

### Value and Array (Low-level Shared Memory)

```python
from multiprocessing import Process, Value, Array

def worker(counter: Value, arr: Array):
    with counter.get_lock():  # Explicit locking
        counter.value += 1
    
    for i in range(len(arr)):
        arr[i] = -arr[i]

if __name__ == '__main__':
    # Type codes from array module: 'i'=int, 'd'=double, 'c'=char
    counter = Value('i', 0)           # Shared integer
    arr = Array('d', [1.0, 2.0, 3.0]) # Shared double array
    
    processes = [Process(target=worker, args=(counter, arr)) for _ in range(4)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    print(f"Counter: {counter.value}")  # 4
    print(f"Array: {arr[:]}")            # [-1.0, -2.0, -3.0]
```

### Manager - High-level Shared Objects

```python
from multiprocessing import Process, Manager

def worker(shared_dict, shared_list, lock):
    with lock:
        shared_dict['count'] = shared_dict.get('count', 0) + 1
        shared_list.append(os.getpid())

if __name__ == '__main__':
    with Manager() as manager:
        shared_dict = manager.dict()
        shared_list = manager.list()
        lock = manager.Lock()
        
        processes = [
            Process(target=worker, args=(shared_dict, shared_list, lock))
            for _ in range(4)
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        
        print(dict(shared_dict))  # {'count': 4}
        print(list(shared_list))  # [pid1, pid2, pid3, pid4]
```

### Manager Supported Types

```python
manager = Manager()

# Collections
manager.dict()
manager.list()
manager.set()       # Python 3.8+
manager.Namespace() # Attribute-based storage

# Synchronization
manager.Lock()
manager.RLock()
manager.Semaphore(value=1)
manager.BoundedSemaphore(value=1)
manager.Condition()
manager.Event()
manager.Barrier(parties)
manager.Queue()
manager.Value(typecode, value)
manager.Array(typecode, sequence)
```

---

## SharedMemory (Python 3.8+) - Zero-Copy Sharing

Fastest way to share large data (NumPy arrays, byte buffers).

### Basic SharedMemory

```python
from multiprocessing import shared_memory

# Create shared memory block
shm = shared_memory.SharedMemory(create=True, size=1024, name="my_buffer")

# Write data
shm.buf[0:5] = b"Hello"

# In another process - attach by name
shm2 = shared_memory.SharedMemory(name="my_buffer")
print(bytes(shm2.buf[0:5]))  # b'Hello'

# Cleanup (IMPORTANT!)
shm2.close()  # Detach from this process
shm.close()
shm.unlink()  # Delete the shared memory block
```

### SharedMemory with NumPy Arrays

```python
import numpy as np
from multiprocessing import shared_memory, Process

def worker(shm_name: str, shape: tuple, dtype):
    # Attach to existing shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    
    # Modify array (visible to all processes)
    arr *= 2
    
    shm.close()

if __name__ == '__main__':
    # Create array in shared memory
    original = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    shm = shared_memory.SharedMemory(create=True, size=original.nbytes)
    shared_arr = np.ndarray(original.shape, dtype=original.dtype, buffer=shm.buf)
    shared_arr[:] = original  # Copy data into shared memory
    
    p = Process(target=worker, args=(shm.name, original.shape, original.dtype))
    p.start()
    p.join()
    
    print(shared_arr)  # [2. 4. 6. 8. 10.]
    
    shm.close()
    shm.unlink()
```

### ShareableList (Python 3.8+)

```python
from multiprocessing import shared_memory

# Create shareable list (fixed size, fixed types)
sl = shared_memory.ShareableList([1, 2.0, "hello", None, True])

# Access in another process
sl2 = shared_memory.ShareableList(name=sl.shm.name)
print(sl2[2])  # "hello"

# Cleanup
sl.shm.close()
sl.shm.unlink()
```

---

## Synchronization Primitives

All threading primitives available in multiprocessing:

```python
from multiprocessing import Lock, RLock, Semaphore, BoundedSemaphore
from multiprocessing import Condition, Event, Barrier

lock = Lock()
rlock = RLock()
sem = Semaphore(3)
bsem = BoundedSemaphore(3)
cond = Condition()
event = Event()
barrier = Barrier(4)

# All support context manager protocol
with lock:
    # Critical section
    pass
```

---

## concurrent.futures - High-Level Interface

Prefer `ProcessPoolExecutor` for simpler API:

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

def compute(x: int) -> int:
    return x * x

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Submit individual tasks
        future = executor.submit(compute, 5)
        print(future.result())  # 25
        
        # Map over iterable
        results = list(executor.map(compute, range(10)))
        
        # Process as completed
        futures = [executor.submit(compute, i) for i in range(10)]
        for future in as_completed(futures):
            print(future.result())
```

---

## Best Practices

### 1. Always Use `if __name__ == '__main__'`

```python
# REQUIRED for spawn/forkserver (prevents infinite process spawning)
if __name__ == '__main__':
    main()
```

### 2. Prefer spawn/forkserver Over fork

```python
# Explicit is better than implicit
mp.set_start_method('spawn')  # or 'forkserver' on Linux
```

### 3. Pass Resources Explicitly

```python
# BAD: Relies on fork inheritance
lock = Lock()
def worker():
    with lock: ...

# GOOD: Explicit argument passing
def worker(lock):
    with lock: ...

Process(target=worker, args=(lock,))
```

### 4. Drain Queues Before Join

```python
# Deadlock risk: join before draining queue
q.put("data")
p.join()       # DEADLOCK if queue not drained

# Correct: drain then join
while not q.empty():
    q.get()
p.join()
```

### 5. Use Context Managers

```python
# Ensures cleanup
with Pool(4) as pool:
    results = pool.map(func, data)
# Pool automatically closed and joined

with Manager() as manager:
    d = manager.dict()
# Manager shutdown automatically
```

### 6. Handle Exceptions in Workers

```python
def safe_worker(x):
    try:
        return risky_operation(x)
    except Exception as e:
        return f"Error: {e}"

# Or use pool error callbacks
result = pool.apply_async(
    risky_func,
    args=(x,),
    error_callback=lambda e: print(f"Error: {e}")
)
```

---

## Quick Reference

| Need | Solution |
|------|----------|
| Parallel map over data | `Pool.map()` or `ProcessPoolExecutor.map()` |
| Share large NumPy array | `shared_memory.SharedMemory` + `np.ndarray` |
| Share Python objects | `Manager().dict()`, `Manager().list()` |
| Send data between processes | `Queue` (multi-producer) or `Pipe` (two-party) |
| Share simple values | `Value('i', 0)`, `Array('d', [1.0, 2.0])` |
| Synchronize access | `Lock`, `RLock`, `Semaphore`, etc. |
| Process individual tasks | `Pool.apply_async()` or `executor.submit()` |
| Limit concurrent processes | `Pool(processes=N)` or `Semaphore(N)` |
