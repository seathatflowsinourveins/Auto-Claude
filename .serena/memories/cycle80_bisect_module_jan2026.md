# Cycle 80: bisect Module Patterns (January 2026)

## Overview
The `bisect` module provides array bisection algorithms for maintaining sorted lists efficiently.
All functions use binary search for O(log n) lookup, though insertion remains O(n) due to list shifting.

## Core Functions

### Finding Insertion Points
```python
import bisect

# bisect_left(a, x, lo=0, hi=len(a), *, key=None)
# Returns leftmost position where x could be inserted maintaining sort order
# If x already exists, returns position BEFORE existing entries
sorted_list = [1, 3, 3, 3, 5, 7]
pos = bisect.bisect_left(sorted_list, 3)  # Returns 1

# bisect_right(a, x, lo=0, hi=len(a), *, key=None) / bisect()
# Returns rightmost position where x could be inserted
# If x already exists, returns position AFTER existing entries
pos = bisect.bisect_right(sorted_list, 3)  # Returns 4
pos = bisect.bisect(sorted_list, 3)        # Alias for bisect_right
```

### Inserting While Maintaining Order
```python
# insort_left(a, x, lo=0, hi=len(a), *, key=None)
# Insert x in a, keeping it sorted (before existing equal values)
data = [1, 3, 5, 7]
bisect.insort_left(data, 4)  # data = [1, 3, 4, 5, 7]

# insort_right(a, x, lo=0, hi=len(a), *, key=None) / insort()
# Insert x in a, keeping it sorted (after existing equal values)
bisect.insort_right(data, 5)  # Inserts after existing 5
bisect.insort(data, 6)        # Alias for insort_right
```

## Key Parameter (Python 3.10+)
```python
from bisect import bisect_left, insort_left
from collections import namedtuple

# Search by extracted key, not raw value
Movie = namedtuple('Movie', ['name', 'year', 'rating'])
movies = [
    Movie('Jaws', 1975, 4.0),
    Movie('Titanic', 1997, 4.5),
    Movie('Avatar', 2009, 3.5),
]

# Find insertion point by year
pos = bisect_left(movies, 1985, key=lambda m: m.year)  # Returns 1

# Insert maintaining year order
new_movie = Movie('Aliens', 1986, 4.5)
insort_left(movies, new_movie, key=lambda m: m.year)
```

## Common Patterns

### Grade Lookup Table
```python
def grade(score, breakpoints=[60, 70, 80, 90], grades='FDCBA'):
    """Convert numeric score to letter grade."""
    i = bisect.bisect(breakpoints, score)
    return grades[i]

# Usage
grade(85)  # 'B'
grade(95)  # 'A'
grade(55)  # 'F'
```

### Searching Sorted Lists
```python
from bisect import bisect_left

def index(a, x):
    """Locate the leftmost value exactly equal to x."""
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError(f'{x} not found')

def find_lt(a, x):
    """Find rightmost value less than x."""
    i = bisect_left(a, x)
    if i:
        return a[i-1]
    raise ValueError(f'No value less than {x}')

def find_le(a, x):
    """Find rightmost value less than or equal to x."""
    i = bisect.bisect_right(a, x)
    if i:
        return a[i-1]
    raise ValueError(f'No value <= {x}')

def find_gt(a, x):
    """Find leftmost value greater than x."""
    i = bisect.bisect_right(a, x)
    if i != len(a):
        return a[i]
    raise ValueError(f'No value greater than {x}')

def find_ge(a, x):
    """Find leftmost value greater than or equal to x."""
    i = bisect_left(a, x)
    if i != len(a):
        return a[i]
    raise ValueError(f'No value >= {x}')
```

### Priority Queue with Sorted Insertion
```python
from bisect import insort_left
from dataclasses import dataclass

@dataclass
class Task:
    priority: int
    name: str
    
    def __lt__(self, other):
        return self.priority < other.priority

tasks = []
insort_left(tasks, Task(3, 'low priority'))
insort_left(tasks, Task(1, 'urgent'))
insort_left(tasks, Task(2, 'normal'))
# tasks sorted by priority: [Task(1, 'urgent'), Task(2, 'normal'), Task(3, 'low')]
```

### Range Queries
```python
def count_in_range(sorted_list, lo, hi):
    """Count elements where lo <= x < hi."""
    return bisect.bisect_left(sorted_list, hi) - bisect.bisect_left(sorted_list, lo)

def slice_in_range(sorted_list, lo, hi):
    """Get elements where lo <= x < hi."""
    left = bisect.bisect_left(sorted_list, lo)
    right = bisect.bisect_left(sorted_list, hi)
    return sorted_list[left:right]

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
count_in_range(data, 3, 7)  # 4 elements: 3, 4, 5, 6
slice_in_range(data, 3, 7)  # [3, 4, 5, 6]
```

### Maintaining Unique Sorted List
```python
def sorted_unique_insert(sorted_list, x):
    """Insert x only if not already present."""
    i = bisect.bisect_left(sorted_list, x)
    if i == len(sorted_list) or sorted_list[i] != x:
        sorted_list.insert(i, x)
        return True
    return False
```

## Performance Characteristics

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| bisect_left/right | O(log n) | Binary search |
| insort_left/right | O(n) | O(log n) search + O(n) insert |
| Slice extraction | O(k) | k = number of elements in slice |

## Best Practices

1. **Pre-sort data**: bisect assumes list is already sorted
2. **Use key for complex objects** instead of maintaining parallel lists
3. **Consider SortedContainers**: For frequent insertions, `sortedcontainers.SortedList` offers O(log n) insert
4. **Bound your searches**: Use `lo` and `hi` to search subarrays efficiently
5. **Choose left vs right**: Use left for "insert before equals", right for "insert after equals"

## When NOT to Use bisect

- **Frequent insertions**: O(n) insertion dominates; use `sortedcontainers` or heap
- **Unsorted data**: bisect requires sorted input
- **Need full sorting**: Just use `sorted()` or `list.sort()`
- **Dictionary-like access**: Use dict or sorted dict implementations

## Integration with heapq
```python
import heapq
import bisect

# heapq: O(log n) insert AND extract-min, but no efficient search
# bisect: O(log n) search, O(n) insert, no extract-min

# Use heapq when you need min/max extraction
# Use bisect when you need arbitrary position lookups
```

## Source
- Python 3.14 official documentation: https://docs.python.org/3/library/bisect.html
- Research date: January 2026
