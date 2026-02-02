# Python timeit Module - Production Patterns (January 2026)

## Overview

The `timeit` module provides precise benchmarking for small code snippets. It avoids common timing pitfalls by disabling garbage collection and using high-resolution timers.

**Key Principle**: Measures wall-clock time with `time.perf_counter` for sub-microsecond accuracy.

## Core Components

### Timer Class

```python
import timeit

# Create a Timer object
timer = timeit.Timer(
    stmt='"-".join(str(n) for n in range(100))',  # Code to time
    setup='pass',  # Setup code (run once)
    globals=None  # Namespace for execution
)

# Execute once and return time
elapsed = timer.timeit(number=1000000)  # Run 1M times

# Execute multiple times, return list of times
times = timer.repeat(repeat=5, number=1000000)  # 5 runs of 1M each

# Auto-determine iterations (Python 3.6+)
count, total_time = timer.autorange()  # Returns (iterations, total_time)

# Debug: Print exception traceback
timer.print_exc()  # Uses linecache for accurate line numbers
```

### Module-Level Convenience Functions

```python
import timeit

# Quick timing (returns float seconds)
elapsed = timeit.timeit(
    stmt='sum(range(100))',
    setup='pass',
    number=1000000,
    globals=None
)

# Multiple runs (returns list of floats)
times = timeit.repeat(
    stmt='sum(range(100))',
    setup='pass',
    repeat=5,
    number=1000000,
    globals=None
)

# Use min() for best time (standard practice)
best_time = min(times)
```

### default_timer

```python
import timeit

# High-resolution timer (platform-specific)
# Python 3.3+: time.perf_counter (nanosecond resolution)
timer = timeit.default_timer
start = timer()
# ... code ...
elapsed = timer() - start
```

## Command-Line Interface

```bash
# Basic usage
python -m timeit '"-".join(str(n) for n in range(100))'

# With setup
python -m timeit -s 'text = "sample string"' 'text.upper()'

# Control iterations
python -m timeit -n 1000000 'sum(range(100))'  # Exactly 1M iterations

# Control repeats
python -m timeit -r 7 'sum(range(100))'  # 7 repeat runs (default: 5)

# Specify time unit
python -m timeit -u usec 'sum(range(100))'  # usec, msec, sec

# Verbose output
python -m timeit -v 'sum(range(100))'

# Process time instead of wall time
python -m timeit -p 'sum(range(100))'  # Uses time.process_time
```

## Production Patterns

### Pattern 1: Benchmark with Local Variables

```python
import timeit

def benchmark_function():
    """Benchmark a function using globals parameter."""
    def my_function(data):
        return sum(x * 2 for x in data)
    
    test_data = list(range(1000))
    
    # Pass locals as globals for name resolution
    elapsed = timeit.timeit(
        stmt='my_function(test_data)',
        globals={'my_function': my_function, 'test_data': test_data},
        number=10000
    )
    
    return elapsed / 10000  # Average per call
```

### Pattern 2: Compare Multiple Implementations

```python
import timeit

def compare_implementations():
    """Compare different implementations side by side."""
    setup = '''
data = list(range(1000))
'''
    
    implementations = {
        'list_comp': '[x * 2 for x in data]',
        'map': 'list(map(lambda x: x * 2, data))',
        'generator': 'list(x * 2 for x in data)',
    }
    
    results = {}
    for name, stmt in implementations.items():
        times = timeit.repeat(stmt, setup, repeat=5, number=10000)
        results[name] = min(times)  # Best of 5 runs
    
    # Report relative performance
    baseline = min(results.values())
    for name, time in sorted(results.items(), key=lambda x: x[1]):
        ratio = time / baseline
        print(f"{name}: {time:.4f}s ({ratio:.2f}x)")
```

### Pattern 3: Autorange for Unknown Duration

```python
import timeit

def adaptive_benchmark(stmt, setup='pass'):
    """Automatically determine optimal iteration count."""
    timer = timeit.Timer(stmt, setup)
    
    # autorange finds iterations where total time >= 0.2 seconds
    iterations, total_time = timer.autorange()
    
    # Now run proper benchmark with known iterations
    times = timer.repeat(repeat=5, number=iterations)
    
    per_iteration = min(times) / iterations
    return {
        'iterations': iterations,
        'best_total': min(times),
        'per_call': per_iteration,
        'per_call_ns': per_iteration * 1e9
    }
```

### Pattern 4: Benchmark Class Methods

```python
import timeit

class MyClass:
    def __init__(self, size):
        self.data = list(range(size))
    
    def process(self):
        return sum(self.data)

def benchmark_method():
    """Benchmark instance methods properly."""
    obj = MyClass(1000)
    
    # Use globals to expose the instance
    elapsed = timeit.timeit(
        stmt='obj.process()',
        globals={'obj': obj},
        number=100000
    )
    
    return elapsed / 100000
```

### Pattern 5: Statistical Benchmarking

```python
import timeit
import statistics

def statistical_benchmark(stmt, setup='pass', runs=10, iterations=10000):
    """Get statistical measures for benchmarks."""
    timer = timeit.Timer(stmt, setup)
    times = timer.repeat(repeat=runs, number=iterations)
    
    # Per-iteration times
    per_iter = [t / iterations for t in times]
    
    return {
        'min': min(per_iter),
        'max': max(per_iter),
        'mean': statistics.mean(per_iter),
        'median': statistics.median(per_iter),
        'stdev': statistics.stdev(per_iter) if runs > 1 else 0,
        'variance_pct': (statistics.stdev(per_iter) / statistics.mean(per_iter) * 100) if runs > 1 else 0
    }
```

## Important Considerations

### Garbage Collection

```python
import timeit
import gc

# timeit disables GC by default
# To include GC time:
timer = timeit.Timer('create_objects()', 'gc.enable()', globals=globals())

# Or explicitly in setup
elapsed = timeit.timeit(
    stmt='create_objects()',
    setup='import gc; gc.enable()',
    number=1000
)
```

### Why min() Not mean()

```python
# Use min() for benchmarks, not mean()
times = timeit.repeat(stmt, repeat=5, number=100000)

# CORRECT: min() gives most consistent measurement
best = min(times)  # Least interference from other processes

# AVOID: mean() includes outliers from OS scheduling
average = sum(times) / len(times)  # Unreliable
```

### Multiline Statements

```python
# Triple-quoted strings for complex code
elapsed = timeit.timeit('''
result = []
for i in range(100):
    result.append(i * 2)
sum(result)
''', number=10000)

# Or semicolon-separated
elapsed = timeit.timeit('x = 1; y = 2; z = x + y', number=1000000)
```

## Version History

| Version | Feature |
|---------|---------|
| 3.0 | Module introduced |
| 3.3 | default_timer changed to perf_counter |
| 3.5 | globals parameter added |
| 3.6 | autorange() method added |
| 3.7 | Line numbers in exceptions improved |

## See Also

- `time.perf_counter` - High-resolution timer
- `time.process_time` - CPU time only
- `cProfile` - Full profiling (function-level)
- `line_profiler` - Line-by-line profiling (third-party)
- `memory_profiler` - Memory benchmarking (third-party)
