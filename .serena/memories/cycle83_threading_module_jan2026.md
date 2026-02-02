# Python threading Module - Production Patterns (January 2026)

## Core Module Overview

The `threading` module provides high-level thread-based parallelism. Important: Due to the Global Interpreter Lock (GIL), threads in CPython do NOT run Python code in parallel on multiple cores. Use threads for I/O-bound tasks; use `multiprocessing` for CPU-bound tasks.

```python
import threading
```

**Python 3.13+ Note**: Experimental free-threaded builds disable the GIL via `--disable-gil` configure option, enabling true parallel execution.

---

## Thread Class

### Creating and Starting Threads

```python
import threading
import time

def worker(name: str, delay: float) -> None:
    """Worker function for thread."""
    print(f"Thread {name} starting")
    time.sleep(delay)
    print(f"Thread {name} finished")

# Method 1: Pass target function
t = threading.Thread(target=worker, args=("A", 2.0))
t.start()
t.join()  # Wait for completion

# Method 2: Subclass Thread
class MyThread(threading.Thread):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
    
    def run(self):
        print(f"Custom thread {self.name} running")

thread = MyThread("Worker-1")
thread.start()
```

### Thread Attributes and Methods

```python
t = threading.Thread(target=worker, args=("B", 1.0), name="WorkerB")

# Before start
print(t.name)        # "WorkerB"
print(t.daemon)      # False (default)
print(t.ident)       # None (not started)
print(t.native_id)   # None (not started)

t.daemon = True      # Set before start()
t.start()

# After start
print(t.is_alive())  # True while running
print(t.ident)       # Thread identifier (int)
print(t.native_id)   # OS-level thread ID

# Wait with timeout
t.join(timeout=5.0)  # Wait max 5 seconds
if t.is_alive():
    print("Thread still running after timeout")
```

### Daemon Threads

```python
# Daemon threads terminate when main thread exits
def background_task():
    while True:
        time.sleep(1)
        print("Background work...")

daemon = threading.Thread(target=background_task, daemon=True)
daemon.start()
# Main thread exits -> daemon thread killed automatically
```

---

## Lock Object

The most basic synchronization primitive. Only one thread can hold a lock at a time.

```python
lock = threading.Lock()

# Method 1: Explicit acquire/release
lock.acquire()
try:
    # Critical section
    shared_resource += 1
finally:
    lock.release()

# Method 2: Context manager (PREFERRED)
with lock:
    # Critical section
    shared_resource += 1

# Non-blocking acquire
if lock.acquire(blocking=False):
    try:
        # Got the lock
        pass
    finally:
        lock.release()
else:
    print("Lock not available")

# Acquire with timeout
if lock.acquire(timeout=2.0):
    try:
        pass
    finally:
        lock.release()
else:
    print("Timeout waiting for lock")

# Check if locked (for debugging only)
print(lock.locked())  # True/False
```

### Thread-Safe Counter Pattern

```python
class ThreadSafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value
    
    def decrement(self) -> int:
        with self._lock:
            self._value -= 1
            return self._value
    
    @property
    def value(self) -> int:
        with self._lock:
            return self._value
```

---

## RLock (Reentrant Lock)

A lock that can be acquired multiple times by the same thread. Must be released the same number of times.

```python
rlock = threading.RLock()

def outer():
    with rlock:
        print("Outer acquired")
        inner()  # Can acquire same lock again

def inner():
    with rlock:  # Same thread, OK!
        print("Inner acquired")

# Use case: methods that call each other
class RecursiveStructure:
    def __init__(self):
        self._lock = threading.RLock()
        self._data = []
    
    def add(self, item):
        with self._lock:
            self._data.append(item)
    
    def add_many(self, items):
        with self._lock:
            for item in items:
                self.add(item)  # Calls add() which also acquires lock
```

---

## Condition Object

Allows threads to wait until notified by another thread. Always used with an underlying lock.

```python
condition = threading.Condition()
queue = []

def producer():
    for i in range(5):
        with condition:
            queue.append(i)
            print(f"Produced {i}")
            condition.notify()  # Wake one waiting thread
        time.sleep(0.5)

def consumer():
    while True:
        with condition:
            while not queue:  # ALWAYS use while, not if
                condition.wait()  # Release lock, wait, reacquire
            item = queue.pop(0)
            print(f"Consumed {item}")

# Start consumer first
c = threading.Thread(target=consumer, daemon=True)
c.start()

# Then producer
p = threading.Thread(target=producer)
p.start()
p.join()
```

### wait_for() - Predicate Version

```python
condition = threading.Condition()
data_ready = False

def wait_for_data():
    with condition:
        # Wait until predicate returns True
        condition.wait_for(lambda: data_ready, timeout=10.0)
        print("Data is ready!")

def signal_data():
    global data_ready
    with condition:
        data_ready = True
        condition.notify_all()  # Wake ALL waiting threads
```

---

## Semaphore and BoundedSemaphore

Controls access to a resource pool with limited capacity.

```python
# Allow max 3 concurrent accesses
semaphore = threading.Semaphore(3)

def access_resource(id: int):
    print(f"Thread {id} waiting...")
    with semaphore:
        print(f"Thread {id} acquired (slots: {semaphore._value})")
        time.sleep(2)
    print(f"Thread {id} released")

# Start 10 threads, only 3 run concurrently
threads = [threading.Thread(target=access_resource, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
```

### BoundedSemaphore - Prevents Over-Release

```python
# BoundedSemaphore raises ValueError if released more than acquired
bounded = threading.BoundedSemaphore(3)

bounded.acquire()
bounded.release()
bounded.release()  # ValueError: Semaphore released too many times
```

### Connection Pool Pattern

```python
class ConnectionPool:
    def __init__(self, max_connections: int = 5):
        self._semaphore = threading.BoundedSemaphore(max_connections)
        self._connections = [self._create_connection() for _ in range(max_connections)]
        self._lock = threading.Lock()
    
    def _create_connection(self):
        return {"id": id(object()), "in_use": False}
    
    def acquire(self):
        self._semaphore.acquire()
        with self._lock:
            for conn in self._connections:
                if not conn["in_use"]:
                    conn["in_use"] = True
                    return conn
    
    def release(self, conn):
        with self._lock:
            conn["in_use"] = False
        self._semaphore.release()
```

---

## Event Object

Simple signaling mechanism. One thread signals, others wait.

```python
event = threading.Event()

def waiter(name: str):
    print(f"{name} waiting for event...")
    event.wait()  # Blocks until set
    print(f"{name} proceeding!")

def setter():
    time.sleep(2)
    print("Setting event!")
    event.set()

# Start waiters
for i in range(3):
    threading.Thread(target=waiter, args=(f"Thread-{i}",)).start()

# Set event after delay
threading.Thread(target=setter).start()

# Check and clear
print(event.is_set())  # True after set()
event.clear()          # Reset to unset state
print(event.is_set())  # False

# Wait with timeout
if event.wait(timeout=5.0):
    print("Event was set")
else:
    print("Timeout waiting for event")
```

### Graceful Shutdown Pattern

```python
shutdown_event = threading.Event()

def worker():
    while not shutdown_event.is_set():
        # Do work
        print("Working...")
        shutdown_event.wait(timeout=1.0)  # Check every second
    print("Worker shutting down gracefully")

thread = threading.Thread(target=worker)
thread.start()

time.sleep(5)
shutdown_event.set()  # Signal shutdown
thread.join()
```

---

## Timer Object

Execute a function after a delay.

```python
def delayed_action():
    print("Timer fired!")

# Execute after 3 seconds
timer = threading.Timer(3.0, delayed_action)
timer.start()

# Can be cancelled before firing
timer.cancel()

# Timer with arguments
def greet(name: str, greeting: str = "Hello"):
    print(f"{greeting}, {name}!")

timer = threading.Timer(2.0, greet, args=("World",), kwargs={"greeting": "Hi"})
timer.start()
```

### Repeating Timer Pattern

```python
class RepeatingTimer:
    def __init__(self, interval: float, function, *args, **kwargs):
        self._interval = interval
        self._function = function
        self._args = args
        self._kwargs = kwargs
        self._timer = None
        self._running = False
    
    def _run(self):
        self._running = False
        self.start()
        self._function(*self._args, **self._kwargs)
    
    def start(self):
        if not self._running:
            self._timer = threading.Timer(self._interval, self._run)
            self._timer.start()
            self._running = True
    
    def stop(self):
        if self._timer:
            self._timer.cancel()
            self._running = False
```

---

## Barrier Object

Synchronize a fixed number of threads at a common point.

```python
# All 3 threads must reach barrier before any can proceed
barrier = threading.Barrier(3)

def worker(id: int):
    print(f"Worker {id} doing phase 1")
    time.sleep(id * 0.5)  # Different work times
    
    print(f"Worker {id} waiting at barrier...")
    barrier.wait()  # Block until all 3 arrive
    
    print(f"Worker {id} doing phase 2")

threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

### Barrier with Action

```python
def barrier_action():
    print("=== All threads synchronized ===")

barrier = threading.Barrier(3, action=barrier_action)

# barrier_action runs once when all threads reach barrier
```

### Barrier with Timeout

```python
barrier = threading.Barrier(3, timeout=5.0)

try:
    barrier.wait()  # Wait max 5 seconds
except threading.BrokenBarrierError:
    print("Barrier broken - not all threads arrived")

# Manually break barrier
barrier.abort()  # All waiting threads get BrokenBarrierError
```

---

## Thread-Local Data

Data that is unique to each thread.

```python
local_data = threading.local()

def worker(value):
    local_data.x = value  # Each thread has its own 'x'
    time.sleep(1)
    print(f"Thread sees x = {local_data.x}")

threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()
# Output: Each thread prints its own value
```

### Database Connection Per Thread

```python
import threading
from contextlib import contextmanager

class ThreadLocalDB:
    _local = threading.local()
    
    @classmethod
    def get_connection(cls):
        if not hasattr(cls._local, 'connection'):
            cls._local.connection = create_db_connection()
        return cls._local.connection
    
    @classmethod
    @contextmanager
    def transaction(cls):
        conn = cls.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
```

---

## Utility Functions

```python
# Current thread
current = threading.current_thread()
print(current.name)  # "MainThread" or custom name

# Main thread reference
main = threading.main_thread()

# Active thread count
count = threading.active_count()

# List all threads
for t in threading.enumerate():
    print(f"{t.name}: alive={t.is_alive()}")

# Get thread by ident (Python 3.13+)
# thread = threading._get_thread_by_id(ident)
```

---

## Best Practices

### 1. Always Use Context Managers
```python
# Good
with lock:
    critical_section()

# Bad - easy to forget release
lock.acquire()
critical_section()
lock.release()
```

### 2. Avoid Deadlocks - Lock Ordering
```python
# Always acquire locks in consistent order
lock_a = threading.Lock()
lock_b = threading.Lock()

def good_pattern():
    with lock_a:
        with lock_b:
            pass

def also_good():
    with lock_a:
        with lock_b:
            pass

# BAD - inconsistent order causes deadlock
def bad_pattern():
    with lock_b:  # Opposite order!
        with lock_a:
            pass
```

### 3. Use while Loop with Conditions
```python
# CORRECT - handles spurious wakeups
with condition:
    while not predicate():
        condition.wait()

# WRONG - race condition
with condition:
    if not predicate():
        condition.wait()
```

### 4. Prefer Higher-Level Abstractions
```python
# For producer-consumer: use queue.Queue
from queue import Queue

# For thread pools: use concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# For parallel map: use ThreadPoolExecutor.map()
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process, items))
```

---

## GIL and Performance Considerations

- **GIL prevents** parallel Python bytecode execution
- **I/O operations release** the GIL (file, network, time.sleep)
- **NumPy/C extensions** may release GIL during computation
- **Python 3.13+** free-threaded builds: `python -X gil=0`

```python
# Check if free-threaded
import sys
if hasattr(sys, "_is_gil_enabled"):
    print(f"GIL enabled: {sys._is_gil_enabled()}")
```

---

## Quick Reference

| Primitive | Use Case | Key Methods |
|-----------|----------|-------------|
| Thread | Parallel execution | start(), join(), is_alive() |
| Lock | Mutual exclusion | acquire(), release(), locked() |
| RLock | Reentrant locking | Same as Lock, allows recursion |
| Condition | Wait/notify pattern | wait(), wait_for(), notify(), notify_all() |
| Semaphore | Resource pool | acquire(), release() |
| Event | Simple signaling | set(), clear(), wait(), is_set() |
| Timer | Delayed execution | start(), cancel() |
| Barrier | Sync N threads | wait(), abort(), reset() |
| local | Thread-local data | Attribute access |
