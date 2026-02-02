# Cycle 110: random Module - Pseudo-Random Number Generation

**Source**: https://docs.python.org/3/library/random.html (Python 3.14.2)
**Date**: January 2026

## Overview

The `random` module provides pseudo-random number generation using Mersenne Twister (period 2^19937-1). **NOT for cryptography** - use `secrets` module instead.

## Core Functions

### Seeding & State
```python
import random

# Seed for reproducibility
random.seed(42)  # Deterministic sequence
random.seed()    # System time/OS randomness

# Save and restore state
state = random.getstate()
# ... do random operations ...
random.setstate(state)  # Restore to saved point
```

### Basic Random Float
```python
random.random()  # 0.0 <= x < 1.0 (53-bit precision)
```

## Integer Functions

```python
# Random integer in range [a, b] inclusive
random.randint(1, 6)  # Dice roll: 1, 2, 3, 4, 5, or 6

# Random from range (like range() arguments)
random.randrange(10)        # 0-9
random.randrange(1, 10)     # 1-9
random.randrange(0, 100, 2) # Even: 0, 2, 4, ..., 98

# Random bits
random.getrandbits(8)   # 0-255 (8 random bits)
random.getrandbits(256) # Very large random int
```

## Sequence Functions

### Single Selection
```python
# Pick one random element
random.choice(['win', 'lose', 'draw'])  # 'draw'
random.choice('abcdef')  # 'c'

# Empty sequence raises IndexError
```

### Multiple Selections WITH Replacement
```python
# choices() - can pick same element multiple times (3.6+)
random.choices(['a', 'b', 'c'], k=5)
# ['a', 'c', 'a', 'b', 'a']

# Weighted selection
random.choices(
    ['red', 'black', 'green'],
    weights=[18, 18, 2],  # Relative weights
    k=6
)  # Roulette simulation

# Cumulative weights also supported
random.choices(
    ['A', 'B', 'C'],
    cum_weights=[10, 30, 100],  # A=10%, B=20%, C=70%
    k=3
)
```

### Multiple Selections WITHOUT Replacement
```python
# sample() - each element picked at most once
random.sample([10, 20, 30, 40, 50], k=3)
# [40, 10, 50] - unique elements

# Sample from range (memory efficient)
random.sample(range(10_000_000), k=100)

# With counts (3.9+)
random.sample(
    ['red', 'blue'],
    counts=[4, 2],  # 4 reds, 2 blues in pool
    k=5
)
```

### Shuffle In-Place
```python
deck = ['A', '2', '3', '4', '5']
random.shuffle(deck)  # Modifies deck in place
# deck is now shuffled

# For immutable sequences, use sample
original = (1, 2, 3, 4, 5)
shuffled = random.sample(original, k=len(original))
```

## Random Bytes

```python
# Generate random bytes (3.9+)
random.randbytes(16)  # 16 random bytes

# WARNING: NOT for security - use secrets.token_bytes()
```

## Real-Valued Distributions

### Uniform Distributions
```python
# Uniform float in range
random.uniform(2.5, 10.0)  # 2.5 <= x <= 10.0

# Triangular distribution (mode = peak)
random.triangular(0, 10, 3)  # Peak at 3, range 0-10
```

### Normal (Gaussian) Distributions
```python
# Normal distribution
random.gauss(mu=100, sigma=15)  # IQ scores
random.normalvariate(mu=0, sigma=1)  # Thread-safe version

# Log-normal distribution
random.lognormvariate(mu=0, sigma=1)
```

### Exponential & Related
```python
# Exponential (inter-arrival times)
random.expovariate(lambd=1/5)  # Mean interval of 5

# Gamma distribution
random.gammavariate(alpha=2, beta=1)

# Beta distribution (0-1 range)
random.betavariate(alpha=2, beta=5)

# Pareto distribution
random.paretovariate(alpha=3)

# Weibull distribution
random.weibullvariate(alpha=1, beta=1.5)
```

### Angular Distribution
```python
# Von Mises (circular normal)
random.vonmisesvariate(mu=0, kappa=4)  # Radians
```

## Discrete Distributions (3.12+)

```python
# Binomial distribution
random.binomialvariate(n=10, p=0.5)  # Coin flips
# Returns number of successes (0-10)

# Probability of 5+ heads in 7 flips of biased coin (60%)
trials = 10_000
hits = sum(random.binomialvariate(n=7, p=0.6) >= 5 
           for _ in range(trials))
probability = hits / trials  # ~0.42
```

## Random Class & SystemRandom

### Custom Random Instance
```python
# Separate generator (no shared state)
rng = random.Random(42)
rng.random()
rng.choice([1, 2, 3])

# Useful for thread safety
thread_local_rng = random.Random()
```

### SystemRandom (OS Entropy)
```python
# Uses os.urandom() - not reproducible
secure_rng = random.SystemRandom()
secure_rng.random()
secure_rng.randint(1, 100)

# seed(), getstate(), setstate() not available
# Still NOT for cryptography - use secrets module
```

## Reproducibility

```python
# Same seed = same sequence
random.seed(12345)
a = [random.random() for _ in range(5)]

random.seed(12345)
b = [random.random() for _ in range(5)]

assert a == b  # True
```

**Guarantees:**
- `random()` produces same sequence for same seed
- Backward-compatible seeders provided for new methods

**NOT guaranteed across Python versions** for high-level functions.

## Command-Line Interface (3.13+)

```bash
# Random choice
python -m random egg bacon sausage spam
# Output: bacon

# Random integer 1-N
python -m random 6
# Output: 4

# Random float 0-N
python -m random 1.8
# Output: 1.234...

# Explicit options
python -m random --choice egg bacon spam
python -m random --integer 6
python -m random --float 1.8
```

## Production Patterns

### Weighted Random Selection
```python
from random import choices
from collections import Counter

def simulate_ab_test(control_rate: float, test_rate: float, n: int) -> dict:
    """Simulate A/B test outcomes."""
    results = choices(
        ['control_convert', 'control_no', 'test_convert', 'test_no'],
        weights=[
            control_rate, 1 - control_rate,
            test_rate, 1 - test_rate
        ],
        k=n * 2  # n users per group
    )
    return dict(Counter(results))
```

### Reservoir Sampling
```python
from random import randrange

def reservoir_sample(stream, k: int) -> list:
    """Sample k items from stream of unknown length."""
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = randrange(i + 1)
            if j < k:
                reservoir[j] = item
    return reservoir
```

### Statistical Bootstrapping
```python
from random import choices
from statistics import fmean

def bootstrap_mean_ci(data: list, confidence: float = 0.90, n_samples: int = 1000):
    """Bootstrap confidence interval for mean."""
    means = sorted(
        fmean(choices(data, k=len(data)))
        for _ in range(n_samples)
    )
    alpha = (1 - confidence) / 2
    low_idx = int(n_samples * alpha)
    high_idx = int(n_samples * (1 - alpha))
    return {
        'mean': fmean(data),
        'ci_low': means[low_idx],
        'ci_high': means[high_idx],
    }
```

### Permutation Test
```python
from random import shuffle
from statistics import fmean

def permutation_test(group_a: list, group_b: list, n_perms: int = 10_000):
    """Test if difference between groups is significant."""
    observed_diff = fmean(group_a) - fmean(group_b)
    combined = group_a + group_b
    n_a = len(group_a)
    
    count = 0
    for _ in range(n_perms):
        shuffle(combined)
        new_diff = fmean(combined[:n_a]) - fmean(combined[n_a:])
        if new_diff >= observed_diff:
            count += 1
    
    p_value = count / n_perms
    return {'observed_diff': observed_diff, 'p_value': p_value}
```

### Reproducible Test Data
```python
from random import Random
from dataclasses import dataclass

@dataclass
class TestUser:
    id: int
    name: str
    score: float

def generate_test_users(n: int, seed: int = 42) -> list[TestUser]:
    """Generate reproducible test data."""
    rng = Random(seed)
    names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve']
    return [
        TestUser(
            id=i,
            name=rng.choice(names),
            score=rng.gauss(75, 10)
        )
        for i in range(n)
    ]
```

### Queue Simulation
```python
from random import expovariate, gauss
from heapq import heappush, heappop

def simulate_queue(n_customers: int, n_servers: int, 
                   arrival_rate: float, service_mean: float):
    """Simulate M/G/c queue."""
    servers = [0.0] * n_servers  # When each server is free
    waits = []
    arrival_time = 0.0
    
    for _ in range(n_customers):
        arrival_time += expovariate(arrival_rate)
        next_free = min(servers)
        wait = max(0, next_free - arrival_time)
        waits.append(wait)
        
        service_time = max(0, gauss(service_mean, service_mean * 0.2))
        completion = arrival_time + wait + service_time
        
        # Update server
        idx = servers.index(next_free)
        servers[idx] = completion
    
    return waits
```

## Security Warning

```python
# NEVER use random for:
# - Passwords
# - Tokens
# - Cryptographic keys
# - Session IDs
# - Any security-sensitive values

# USE secrets module instead:
import secrets
secrets.token_hex(16)      # Secure token
secrets.token_urlsafe(16)  # URL-safe token
secrets.randbelow(100)     # Secure random int
secrets.choice(items)      # Secure selection
```

## Version History

| Version | Feature |
|---------|---------|
| 3.6 | choices() |
| 3.9 | randbytes(), sample counts, getrandbits(0) |
| 3.11 | gauss/normalvariate defaults, seed type restrictions |
| 3.12 | binomialvariate(), no auto-convert in randrange |
| 3.13 | Command-line interface |
