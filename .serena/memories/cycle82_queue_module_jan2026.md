# Cycle 82: queue Module Patterns (January 2026)

## Overview
The `queue` module provides thread-safe, synchronized queue implementations for multi-producer, multi-consumer scenarios. All locking semantics are handled internally.

## Queue Types

| Class | Order | Use Case |
|-------|-------|----------|
| `Queue` | FIFO | Standard task queue |
| `LifoQueue` | LIFO (stack) | Depth-first processing |
| `PriorityQueue` | By priority (min first) | Prioritized tasks |
| `SimpleQueue` | FIFO, unbounded | Reentrant-safe, simpler API |

## Core Queue Classes

### Queue (FIFO)
```python
from queue import Queue, Empty, Full

# Create queue (maxsize=0 means infinite)
q = Queue(maxsize=100)

# Basic operations
q.put(item)              # Block until slot available
q.put(item, block=False) # Raise Full immediately if full
q.put(item, timeout=5)   # Block up to 5 seconds

item = q.get()              # Block until item available
item = q.get(block=False)   # Raise Empty immediately if empty
item = q.get(timeout=5)     # Block up to 5 seconds

# Non-blocking shortcuts
q.put_nowait(item)  # Same as put(item, block=False)
item = q.get_nowait()  # Same as get(block=False)

# Status (approximate - not reliable for decisions!)
q.qsize()  # Current size (approximate)
q.empty()  # True if empty (not reliable)
q.full()   # True if full (not reliable)
```

### LifoQueue (Stack)
```python
from queue import LifoQueue

stack = LifoQueue()
stack.put(1)
stack.put(2)
stack.put(3)

stack.get()  # Returns 3 (last in, first out)
stack.get()  # Returns 2
stack.get()  # Returns 1
```

### PriorityQueue
```python
from queue import PriorityQueue

pq = PriorityQueue()

# Items must be comparable, or use tuples (priority, data)
pq.put((2, 'medium priority'))
pq.put((1, 'high priority'))
pq.put((3, 'low priority'))

pq.get()  # (1, 'high priority')
pq.get()  # (2, 'medium priority')
pq.get()  # (3, 'low priority')
```

### PriorityQueue with Non-Comparable Items
```python
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any = field(compare=False)  # Excluded from comparison

pq = PriorityQueue()
pq.put(PrioritizedItem(2, {'task': 'email'}))
pq.put(PrioritizedItem(1, {'task': 'urgent'}))

task = pq.get()  # PrioritizedItem(priority=1, item={'task': 'urgent'})
```

### SimpleQueue (Python 3.7+)
```python
from queue import SimpleQueue

# Unbounded, simpler, reentrant-safe
sq = SimpleQueue()

# Same put/get API but:
# - Always unbounded (no maxsize)
# - put() never blocks (always succeeds)
# - No task_done()/join() support
# - Reentrant: safe to use in __del__ and weakref callbacks

sq.put(item)  # Never blocks
item = sq.get()  # Blocks until available
item = sq.get_nowait()  # Raises Empty if empty
```

## Task Tracking Pattern

```python
import threading
from queue import Queue

def worker(q):
    while True:
        item = q.get()
        try:
            process(item)
        finally:
            q.task_done()  # Signal task completion

q = Queue()

# Start worker threads
for _ in range(4):
    t = threading.Thread(target=worker, args=(q,), daemon=True)
    t.start()

# Add tasks
for task in tasks:
    q.put(task)

# Wait for all tasks to complete
q.join()  # Blocks until task_done() called for each put()
print("All tasks completed")
```

## Graceful Shutdown (Python 3.13+)

```python
from queue import Queue, ShutDown

q = Queue()

# Normal shutdown - drain queue first
q.shutdown(immediate=False)
# - Future put() raises ShutDown
# - get() works until queue empty, then raises ShutDown

# Immediate shutdown - discard remaining items
q.shutdown(immediate=True)
# - Queue drained immediately
# - All blocked get()/put() raise ShutDown
# - join() unblocks

# Handling shutdown in workers
def worker(q):
    while True:
        try:
            item = q.get()
            process(item)
            q.task_done()
        except ShutDown:
            break  # Queue shut down, exit gracefully
```

## Exception Handling

```python
from queue import Queue, Empty, Full, ShutDown

q = Queue(maxsize=10)

# Handle Empty
try:
    item = q.get_nowait()
except Empty:
    print("Queue is empty")

# Handle Full
try:
    q.put_nowait(item)
except Full:
    print("Queue is full")

# Handle ShutDown (Python 3.13+)
try:
    item = q.get()
except ShutDown:
    print("Queue was shut down")
```

## Common Patterns

### Producer-Consumer
```python
import threading
from queue import Queue

def producer(q, items):
    for item in items:
        q.put(item)
    q.put(None)  # Sentinel to signal completion

def consumer(q):
    while True:
        item = q.get()
        if item is None:
            break
        process(item)
        q.task_done()

q = Queue()
prod = threading.Thread(target=producer, args=(q, data))
cons = threading.Thread(target=consumer, args=(q,))

prod.start()
cons.start()
prod.join()
cons.join()
```

### Worker Pool with Timeout
```python
import threading
from queue import Queue, Empty

def worker(q, timeout=1.0):
    while True:
        try:
            item = q.get(timeout=timeout)
        except Empty:
            continue  # Check for shutdown signal
        if item is None:
            break
        process(item)
        q.task_done()

# Graceful shutdown
for _ in range(num_workers):
    q.put(None)  # Send stop signal to each worker
```

### Bounded Buffer (Backpressure)
```python
from queue import Queue

# Bounded queue creates backpressure
buffer = Queue(maxsize=100)

def fast_producer(q):
    for item in generate_items():
        q.put(item)  # Blocks when full, slowing producer

def slow_consumer(q):
    while True:
        item = q.get()
        slow_process(item)
        q.task_done()
```

### Pipeline Pattern
```python
from queue import Queue
import threading

def stage1(input_q, output_q):
    while True:
        item = input_q.get()
        if item is None:
            output_q.put(None)
            break
        result = transform1(item)
        output_q.put(result)
        input_q.task_done()

def stage2(input_q, output_q):
    while True:
        item = input_q.get()
        if item is None:
            output_q.put(None)
            break
        result = transform2(item)
        output_q.put(result)
        input_q.task_done()

# Create pipeline
q1, q2, q3 = Queue(), Queue(), Queue()
threading.Thread(target=stage1, args=(q1, q2)).start()
threading.Thread(target=stage2, args=(q2, q3)).start()

# Feed pipeline
for item in data:
    q1.put(item)
q1.put(None)  # Sentinel
```

## Comparison: queue vs collections.deque vs multiprocessing.Queue

| Feature | queue.Queue | collections.deque | multiprocessing.Queue |
|---------|-------------|-------------------|----------------------|
| Thread-safe | ✓ Full locking | ✓ Atomic append/pop | ✓ Process-safe |
| Blocking get/put | ✓ | ✗ | ✓ |
| task_done/join | ✓ | ✗ | ✗ (use JoinableQueue) |
| Bounded size | ✓ | ✓ (maxlen) | ✓ |
| Cross-process | ✗ | ✗ | ✓ |
| Performance | Good | Fastest | Slower (IPC) |
| Reentrant | SimpleQueue only | ✓ | ✗ |

### When to Use Each

```python
# queue.Queue - Thread communication with blocking
from queue import Queue
q = Queue()

# collections.deque - Fast, atomic, no blocking needed
from collections import deque
d = deque(maxlen=100)
d.append(item)      # Thread-safe
item = d.popleft()  # Thread-safe (raises IndexError if empty)

# multiprocessing.Queue - Cross-process communication
from multiprocessing import Queue, JoinableQueue
q = Queue()  # Basic
jq = JoinableQueue()  # With task_done/join support
```

## Performance Tips

1. **Use SimpleQueue** when you don't need task tracking or bounded size
2. **Avoid checking empty()/full()** - use try/except with get_nowait/put_nowait
3. **Set appropriate maxsize** for backpressure control
4. **Use daemon threads** for workers that should terminate with main thread
5. **Consider deque** for single-threaded or atomic-only operations

## Thread Safety Notes

```python
# WRONG: Race condition
if not q.empty():  # Another thread may consume
    item = q.get()  # May block or raise!

# RIGHT: Use exception handling
try:
    item = q.get_nowait()
except Empty:
    handle_empty()

# WRONG: Checking size then acting
if q.qsize() < q.maxsize:
    q.put(item)  # May block!

# RIGHT: Use timeout or non-blocking
try:
    q.put(item, timeout=1.0)
except Full:
    handle_full()
```

## Source
- Python 3.14 official documentation: https://docs.python.org/3/library/queue.html
- Research date: January 2026
