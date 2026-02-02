# Cycle 81: heapq Module Patterns (January 2026)

## Overview
The `heapq` module provides heap-based priority queue algorithms. Min-heaps keep the smallest element at `heap[0]`. Python 3.14 adds native max-heap support.

## Heap Invariant
```python
# Min-heap: heap[k] <= heap[2*k+1] and heap[k] <= heap[2*k+2]
# Max-heap: heap[k] >= heap[2*k+1] and heap[k] >= heap[2*k+2]
# Root is always at index 0
```

## Core Min-Heap Functions

### Basic Operations
```python
import heapq

# heapify(x) - Transform list into heap in-place, O(n)
data = [3, 1, 4, 1, 5, 9, 2, 6]
heapq.heapify(data)  # data is now a valid min-heap
print(data[0])  # 1 (smallest element)

# heappush(heap, item) - Add item, maintain heap, O(log n)
heap = []
heapq.heappush(heap, 5)
heapq.heappush(heap, 3)
heapq.heappush(heap, 7)
# heap[0] == 3

# heappop(heap) - Remove and return smallest, O(log n)
smallest = heapq.heappop(heap)  # Returns 3

# Peek without removing
smallest = heap[0]  # Just access index 0
```

### Combined Operations (More Efficient)
```python
# heappushpop(heap, item) - Push then pop smallest
# More efficient than heappush() followed by heappop()
result = heapq.heappushpop(heap, 4)  # Push 4, pop smallest

# heapreplace(heap, item) - Pop smallest then push
# More efficient than heappop() followed by heappush()
# WARNING: May return value larger than item added
result = heapq.heapreplace(heap, 2)  # Pop smallest, push 2
```

## Max-Heap Functions (Python 3.14+)

```python
# New in Python 3.14 - native max-heap support
data = [3, 1, 4, 1, 5, 9, 2, 6]

heapq.heapify_max(data)       # Transform to max-heap
print(data[0])                 # 9 (largest element)

heapq.heappush_max(data, 10)  # Push maintaining max-heap
largest = heapq.heappop_max(data)  # Pop largest

# Combined operations
result = heapq.heappushpop_max(heap, 8)   # Push then pop largest
result = heapq.heapreplace_max(heap, 5)   # Pop largest then push
```

### Pre-3.14 Max-Heap Workaround
```python
# Negate values for max-heap behavior
max_heap = []
heapq.heappush(max_heap, -5)  # Store as negative
heapq.heappush(max_heap, -3)
heapq.heappush(max_heap, -7)

largest = -heapq.heappop(max_heap)  # Negate on retrieval: 7
```

## Utility Functions

### nlargest and nsmallest
```python
# nlargest(n, iterable, key=None) - Get n largest elements
# nsmallest(n, iterable, key=None) - Get n smallest elements

data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

top3 = heapq.nlargest(3, data)     # [9, 6, 5]
bottom3 = heapq.nsmallest(3, data) # [1, 1, 2]

# With key function
records = [
    {'name': 'Alice', 'score': 85},
    {'name': 'Bob', 'score': 92},
    {'name': 'Charlie', 'score': 78},
]
top2 = heapq.nlargest(2, records, key=lambda x: x['score'])
# [{'name': 'Bob', 'score': 92}, {'name': 'Alice', 'score': 85}]
```

### Performance Guidelines for nlargest/nsmallest
```python
# n == 1: Use min() or max() instead
# n small relative to len(data): Use nlargest/nsmallest
# n large relative to len(data): Use sorted(data)[:n]
# Repeated calls: Convert to heap first
```

### merge - Merge Sorted Iterables
```python
# merge(*iterables, key=None, reverse=False)
# Memory-efficient merge of pre-sorted streams

import heapq

list1 = [1, 3, 5, 7]
list2 = [2, 4, 6, 8]
list3 = [0, 9, 10]

merged = list(heapq.merge(list1, list2, list3))
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Merge with key (Python 3.5+)
words = heapq.merge(['apple', 'cherry'], ['banana', 'date'], key=str.lower)

# Reverse merge (largest first)
merged_desc = heapq.merge([7, 5, 3], [8, 6, 4], reverse=True)
```

## Priority Queue Patterns

### Basic Priority Queue with Tuples
```python
import heapq

pq = []
heapq.heappush(pq, (5, 'write code'))
heapq.heappush(pq, (1, 'write spec'))
heapq.heappush(pq, (3, 'create tests'))

priority, task = heapq.heappop(pq)  # (1, 'write spec')
```

### Stable Priority Queue (FIFO for Equal Priorities)
```python
import heapq
import itertools

class PriorityQueue:
    def __init__(self):
        self._heap = []
        self._counter = itertools.count()
    
    def push(self, item, priority):
        # Counter ensures FIFO order for equal priorities
        count = next(self._counter)
        heapq.heappush(self._heap, (priority, count, item))
    
    def pop(self):
        priority, count, item = heapq.heappop(self._heap)
        return item
    
    def __len__(self):
        return len(self._heap)
```

### Priority Queue with Task Removal
```python
import heapq
import itertools

REMOVED = '<removed>'

class TaskQueue:
    def __init__(self):
        self._heap = []
        self._entry_finder = {}
        self._counter = itertools.count()
    
    def add(self, task, priority=0):
        if task in self._entry_finder:
            self.remove(task)
        count = next(self._counter)
        entry = [priority, count, task]
        self._entry_finder[task] = entry
        heapq.heappush(self._heap, entry)
    
    def remove(self, task):
        entry = self._entry_finder.pop(task)
        entry[-1] = REMOVED
    
    def pop(self):
        while self._heap:
            priority, count, task = heapq.heappop(self._heap)
            if task is not REMOVED:
                del self._entry_finder[task]
                return task
        raise KeyError('pop from empty queue')
```

### Dataclass-Based Priority Items
```python
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any = field(compare=False)  # Excluded from comparison

heap = []
heapq.heappush(heap, PrioritizedItem(2, 'medium'))
heapq.heappush(heap, PrioritizedItem(1, 'urgent'))
heapq.heappush(heap, PrioritizedItem(3, 'low'))

top = heapq.heappop(heap)  # PrioritizedItem(priority=1, item='urgent')
```

## Advanced Patterns

### Running Median (Two-Heap Approach)
```python
import heapq

def running_median(iterable):
    """Yields cumulative median of values seen so far."""
    lo = []  # max-heap (negated values)
    hi = []  # min-heap
    
    for x in iterable:
        if len(lo) == len(hi):
            # Push to hi, move smallest to lo
            heapq.heappush(lo, -heapq.heappushpop(hi, x))
            yield -lo[0]
        else:
            # Push to lo, move largest to hi
            heapq.heappush(hi, -heapq.heappushpop(lo, -x))
            yield (-lo[0] + hi[0]) / 2

# Python 3.14+ with native max-heap:
def running_median_314(iterable):
    lo = []  # max-heap
    hi = []  # min-heap
    for x in iterable:
        if len(lo) == len(hi):
            heapq.heappush_max(lo, heapq.heappushpop(hi, x))
            yield lo[0]
        else:
            heapq.heappush(hi, heapq.heappushpop_max(lo, x))
            yield (lo[0] + hi[0]) / 2
```

### K Closest Points
```python
import heapq
import math

def k_closest(points: list[tuple], k: int) -> list[tuple]:
    """Find k points closest to origin."""
    def distance(p):
        return math.sqrt(p[0]**2 + p[1]**2)
    
    return heapq.nsmallest(k, points, key=distance)

points = [(1, 2), (3, 4), (1, -1), (5, 5)]
closest = k_closest(points, 2)  # [(1, -1), (1, 2)]
```

### Top K Frequent Elements
```python
import heapq
from collections import Counter

def top_k_frequent(nums: list, k: int) -> list:
    """Find k most frequent elements."""
    counts = Counter(nums)
    return heapq.nlargest(k, counts.keys(), key=counts.get)

nums = [1, 1, 1, 2, 2, 3]
top2 = top_k_frequent(nums, 2)  # [1, 2]
```

### Heapsort
```python
import heapq

def heapsort(iterable):
    """Sort using heap operations."""
    h = []
    for value in iterable:
        heapq.heappush(h, value)
    return [heapq.heappop(h) for _ in range(len(h))]

# Note: Not stable (unlike sorted())
```

## Performance Characteristics

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| heapify | O(n) | Linear time in-place |
| heappush | O(log n) | Bubble up |
| heappop | O(log n) | Bubble down |
| heappushpop | O(log n) | More efficient than separate ops |
| heapreplace | O(log n) | More efficient than separate ops |
| nlargest(k, n) | O(n log k) | For small k |
| nsmallest(k, n) | O(n log k) | For small k |
| merge | O(total) | Lazy iteration |
| peek (heap[0]) | O(1) | Direct access |

## heapq vs bisect Comparison

| Use Case | heapq | bisect |
|----------|-------|--------|
| Extract min/max | O(log n) ✓ | O(n) ✗ |
| Insert | O(log n) ✓ | O(n) ✗ |
| Search by value | O(n) ✗ | O(log n) ✓ |
| Sorted iteration | Pop all: O(n log n) | Already sorted: O(n) |
| Memory | In-place | In-place |

**Rule of thumb:**
- Need min/max quickly? → heapq
- Need arbitrary position lookup? → bisect
- Need both? → Consider sortedcontainers

## Thread Safety
```python
# heapq is NOT thread-safe
# Use queue.PriorityQueue for thread-safe priority queue

from queue import PriorityQueue

pq = PriorityQueue()
pq.put((2, 'medium'))
pq.put((1, 'urgent'))
priority, task = pq.get()  # Blocks if empty
```

## Source
- Python 3.14 official documentation: https://docs.python.org/3/library/heapq.html
- Research date: January 2026
