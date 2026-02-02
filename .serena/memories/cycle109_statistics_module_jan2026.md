# Cycle 109: statistics Module - Mathematical Statistics Functions

**Source**: https://docs.python.org/3/library/statistics.html (Python 3.14.2)
**Date**: January 2026

## Overview

The `statistics` module provides functions for mathematical statistics of numeric data. Supports `int`, `float`, `Decimal`, and `Fraction`. Added in Python 3.4.

## Averages (Central Location)

### Arithmetic Mean
```python
from statistics import mean, fmean

# mean() - preserves type, slower
mean([1, 2, 3, 4, 4])  # 2.8

# Works with Fraction and Decimal
from fractions import Fraction
mean([Fraction(3, 7), Fraction(1, 21)])  # Fraction(13, 21)

from decimal import Decimal
mean([Decimal("0.5"), Decimal("0.75")])  # Decimal('0.625')

# fmean() - faster, always returns float (3.8+)
fmean([3.5, 4.0, 5.25])  # 4.25

# Weighted mean (3.11+)
grades = [85, 92, 83, 91]
weights = [0.20, 0.20, 0.30, 0.30]  # Quiz, HW, Midterm, Final
fmean(grades, weights)  # 87.6
```

### Geometric & Harmonic Mean
```python
from statistics import geometric_mean, harmonic_mean

# Geometric mean - for multiplicative processes (growth rates)
geometric_mean([54, 24, 36])  # 36.0 (approximately)

# Harmonic mean - for rates/ratios (speeds, prices)
# Car travels 10km at 40km/hr, then 10km at 60km/hr
harmonic_mean([40, 60])  # 48.0 (not 50!)

# Weighted harmonic mean (3.10+)
# 5km at 40km/hr, 30km at 60km/hr
harmonic_mean([40, 60], weights=[5, 30])  # 56.0
```

## Medians

```python
from statistics import median, median_low, median_high, median_grouped

# Standard median (interpolates for even count)
median([1, 3, 5])     # 3
median([1, 3, 5, 7])  # 4.0 (average of 3 and 5)

# median_low - returns actual data point (smaller)
median_low([1, 3, 5, 7])  # 3

# median_high - returns actual data point (larger)
median_high([1, 3, 5, 7])  # 5

# median_grouped - for binned/grouped data
from collections import Counter
demographics = Counter({
    25: 172,  # 20-30 years
    35: 484,  # 30-40 years
    45: 387,  # 40-50 years
})
data = list(demographics.elements())
median_grouped(data, interval=10)  # ~37.5 (interpolated within bin)
```

## Mode

```python
from statistics import mode, multimode

# mode - single most common value (works with non-numeric!)
mode([1, 1, 2, 3, 3, 3, 3, 4])  # 3
mode(["red", "blue", "blue", "red", "red"])  # 'red'

# multimode - all modes in order of first occurrence (3.8+)
multimode('aabbbbccddddeeffffgg')  # ['b', 'd', 'f']
multimode([])  # [] (empty for empty input)

# For min/max mode when multiple exist
min(multimode([1, 1, 2, 2, 3]))  # 1
max(multimode([1, 1, 2, 2, 3]))  # 2
```

## Variance & Standard Deviation

```python
from statistics import variance, stdev, pvariance, pstdev

data = [2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5]

# SAMPLE statistics (n-1 degrees of freedom) - for samples
variance(data)  # 1.372... (Bessel's correction)
stdev(data)     # 1.171...

# POPULATION statistics (n degrees of freedom) - for entire population
pvariance(data)  # 1.176...
pstdev(data)     # 1.085...

# Optimization: pass pre-computed mean to avoid recalculation
m = mean(data)
variance(data, xbar=m)   # Sample
pvariance(data, mu=m)    # Population
```

## Quantiles (3.8+)

```python
from statistics import quantiles

data = [105, 129, 87, 86, 111, 111, 89, 81, 108, 92]

# Quartiles (default n=4)
quantiles(data)  # 3 cut points for 4 groups

# Deciles
quantiles(data, n=10)  # 9 cut points

# Percentiles
quantiles(data, n=100)  # 99 cut points

# Methods: 'exclusive' (default) vs 'inclusive'
quantiles(data, method='exclusive')  # For samples with possible extremes
quantiles(data, method='inclusive')  # When data includes all extremes
```

## Relations Between Variables (3.10+)

### Covariance
```python
from statistics import covariance

x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [1, 2, 3, 1, 2, 3, 1, 2, 3]
z = [9, 8, 7, 6, 5, 4, 3, 2, 1]

covariance(x, y)  #  0.75 (weak positive)
covariance(x, z)  # -7.5  (strong negative)
```

### Correlation
```python
from statistics import correlation

# Pearson correlation (linear relationship, default)
correlation(x, y)  # ~0.0 (no linear relationship)
correlation(x, z)  # -1.0 (perfect negative)

# Spearman correlation (monotonic relationship, 3.12+)
orbital_period = [88, 225, 365, 687, 4331, 10756]
dist_from_sun = [58, 108, 150, 228, 778, 1400]
correlation(orbital_period, dist_from_sun, method='ranked')  # 1.0
```

### Linear Regression
```python
from statistics import linear_regression

year = [1971, 1975, 1979, 1982, 1983]
films = [1, 2, 3, 4, 5]

slope, intercept = linear_regression(year, films)
# Predict films by 2019
round(slope * 2019 + intercept)  # 16

# Proportional (through origin, 3.11+)
model = linear_regression(x, y, proportional=True)
# model.intercept is always 0.0
```

## Kernel Density Estimation (3.13+)

```python
from statistics import kde, kde_random

sample = [-2.1, -1.3, -0.4, 1.9, 5.1, 6.2]

# Create probability density function
f_hat = kde(sample, h=1.5)  # h = bandwidth (smoothing)
f_hat(0)  # Probability density at x=0

# Create cumulative distribution function
cdf = kde(sample, h=1.5, cumulative=True)

# Available kernels:
# - normal (gauss) - weight all points
# - logistic, sigmoid
# - rectangular (uniform), triangular
# - parabolic (epanechnikov), quartic (biweight)
# - triweight, cosine

# Random sampling from estimated PDF
rand = kde_random(sample, h=1.5, seed=42)
new_samples = [rand() for _ in range(100)]
```

## NormalDist Class (3.8+)

```python
from statistics import NormalDist

# Create from parameters
nd = NormalDist(mu=100, sigma=15)  # IQ distribution

# Create from data
nd = NormalDist.from_samples([98, 102, 104, 95, 101])

# Properties
nd.mean      # Arithmetic mean (mu)
nd.median    # Same as mean for normal
nd.mode      # Same as mean for normal
nd.stdev     # Standard deviation (sigma)
nd.variance  # sigma^2

# Probability functions
nd.pdf(100)      # Probability density at x
nd.cdf(115)      # P(X <= 115)
nd.inv_cdf(0.5)  # Value at 50th percentile

# Sampling
samples = nd.samples(1000, seed=42)

# Quantiles
nd.quantiles()      # Quartiles
nd.quantiles(n=10)  # Deciles

# Z-score
nd.zscore(130)  # (130 - mean) / stdev

# Distribution overlap
other = NormalDist(110, 15)
nd.overlap(other)  # Area of overlap (0-1)

# Arithmetic operations (translation/scaling)
celsius = NormalDist(20, 5)
fahrenheit = celsius * (9/5) + 32  # NormalDist(mu=68, sigma=9)

# Adding independent distributions
combined = nd + other  # Means add, variances add
```

## Handling NaN Values

```python
from statistics import median
from math import isnan
from itertools import filterfalse

data = [20.7, float('NaN'), 19.2, 18.3, float('NaN'), 14.4]

# NaN breaks sorting and statistics!
median(data)  # Unexpected result!

# Solution: Strip NaN first
clean = list(filterfalse(isnan, data))
median(clean)  # Correct result
```

## Production Patterns

### Descriptive Statistics Report
```python
from statistics import mean, median, stdev, quantiles
from dataclasses import dataclass

@dataclass
class DescriptiveStats:
    count: int
    mean: float
    median: float
    stdev: float
    min: float
    max: float
    q1: float
    q3: float
    
    @classmethod
    def from_data(cls, data: list[float]) -> 'DescriptiveStats':
        data = sorted(data)
        q = quantiles(data)
        return cls(
            count=len(data),
            mean=mean(data),
            median=median(data),
            stdev=stdev(data) if len(data) > 1 else 0.0,
            min=data[0],
            max=data[-1],
            q1=q[0],
            q3=q[2],
        )
    
    @property
    def iqr(self) -> float:
        return self.q3 - self.q1
```

### A/B Test Analysis
```python
from statistics import NormalDist, mean, stdev
from math import sqrt

def ab_test_significance(control: list, treatment: list, alpha=0.05):
    """Two-sample t-test approximation using NormalDist."""
    n1, n2 = len(control), len(treatment)
    m1, m2 = mean(control), mean(treatment)
    s1, s2 = stdev(control), stdev(treatment)
    
    # Pooled standard error
    se = sqrt(s1**2/n1 + s2**2/n2)
    
    # Effect size
    effect = m2 - m1
    
    # Z-score (approximate for large n)
    z = effect / se
    
    # P-value using standard normal
    standard = NormalDist()
    p_value = 2 * (1 - standard.cdf(abs(z)))
    
    return {
        'control_mean': m1,
        'treatment_mean': m2,
        'effect': effect,
        'p_value': p_value,
        'significant': p_value < alpha,
    }
```

### Rolling Statistics
```python
from collections import deque
from statistics import mean, stdev

class RollingStats:
    """Compute rolling mean and stdev over a window."""
    
    def __init__(self, window_size: int):
        self.window = deque(maxlen=window_size)
    
    def push(self, value: float) -> None:
        self.window.append(value)
    
    @property
    def mean(self) -> float:
        return mean(self.window) if self.window else 0.0
    
    @property
    def stdev(self) -> float:
        if len(self.window) < 2:
            return 0.0
        return stdev(self.window)
    
    def zscore(self, value: float) -> float:
        """How many stdevs is value from rolling mean?"""
        s = self.stdev
        return (value - self.mean) / s if s > 0 else 0.0
```

### Outlier Detection
```python
from statistics import mean, stdev, quantiles

def detect_outliers_zscore(data: list, threshold: float = 3.0) -> list:
    """Detect outliers using z-score method."""
    m, s = mean(data), stdev(data)
    return [x for x in data if abs((x - m) / s) > threshold]

def detect_outliers_iqr(data: list, factor: float = 1.5) -> list:
    """Detect outliers using IQR method."""
    q = quantiles(data)
    q1, q3 = q[0], q[2]
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return [x for x in data if x < lower or x > upper]
```

## Version History

| Version | Feature |
|---------|---------|
| 3.4 | Module added |
| 3.6 | harmonic_mean |
| 3.8 | fmean, geometric_mean, multimode, quantiles, NormalDist |
| 3.9 | NormalDist.zscore |
| 3.10 | covariance, correlation, linear_regression, weighted harmonic_mean |
| 3.11 | weighted fmean, linear_regression proportional |
| 3.12 | Spearman correlation |
| 3.13 | kde, kde_random, single-point quantiles |
