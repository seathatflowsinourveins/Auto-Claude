# Cycle 108: fractions Module - Rational Number Arithmetic

**Source**: https://docs.python.org/3/library/fractions.html (Python 3.14.2)
**Date**: January 2026

## Overview

The `fractions` module provides exact rational number arithmetic via the `Fraction` class. Unlike floats, fractions maintain perfect precision for rational numbers.

## Creating Fractions

### From Integers
```python
from fractions import Fraction

# Two integers (numerator, denominator)
Fraction(16, -10)    # Fraction(-8, 5) - auto-normalized
Fraction(123)        # Fraction(123, 1)
Fraction()           # Fraction(0, 1)

# ZeroDivisionError if denominator is 0
Fraction(1, 0)  # Raises ZeroDivisionError
```

### From Strings
```python
Fraction('3/7')          # Fraction(3, 7)
Fraction(' -3/7 ')       # Whitespace OK
Fraction('2 / 3')        # Space around slash OK (3.12+)
Fraction('1.414213')     # Fraction(1414213, 1000000)
Fraction('-.125')        # Fraction(-1, 8)
Fraction('7e-6')         # Fraction(7, 1000000)
Fraction('1_000/3')      # Underscores OK (3.11+)
```

### From Floats (CAUTION!)
```python
# Float precision issues - NOT what you expect!
Fraction(1.1)  # Fraction(2476979795053773, 2251799813685248)
               # NOT Fraction(11, 10)!

Fraction(2.25)  # Fraction(9, 4) - exact for binary fractions

# Use limit_denominator() to recover rational
Fraction(1.1).limit_denominator()  # Fraction(11, 10)
```

### From Decimal (Preferred for precision)
```python
from decimal import Decimal

# Decimal maintains exact decimal representation
Fraction(Decimal('1.1'))  # Fraction(11, 10) - exact!
Fraction(Decimal('0.333'))  # Fraction(333, 1000)
```

### Alternative Constructors
```python
# from_float - explicit float conversion
Fraction.from_float(0.5)  # Fraction(1, 2)

# from_decimal - explicit Decimal conversion
Fraction.from_decimal(Decimal('1.5'))  # Fraction(3, 2)

# from_number (3.14+) - any numeric type with as_integer_ratio()
Fraction.from_number(2.5)  # Works with float, Decimal, Rational
```

## Properties

```python
f = Fraction(8, -12)
f.numerator    # -2 (lowest terms)
f.denominator  # 3  (always positive, lowest terms)
```

## Key Methods

### as_integer_ratio()
```python
Fraction(3, 4).as_integer_ratio()  # (3, 4)
# Compatible with float.as_integer_ratio()
```

### is_integer() (3.12+)
```python
Fraction(6, 3).is_integer()  # True
Fraction(5, 3).is_integer()  # False
```

### limit_denominator() - Critical for Float Recovery
```python
from math import pi, cos

# Recover rational from float representation
Fraction(cos(pi/3))                     # Complex fraction
Fraction(cos(pi/3)).limit_denominator() # Fraction(1, 2)

# Find rational approximation of pi
Fraction('3.1415926535897932').limit_denominator(1000)
# Fraction(355, 113) - famous approximation!

# Custom max denominator
Fraction(pi).limit_denominator(100)  # Fraction(311, 99)
Fraction(pi).limit_denominator(10)   # Fraction(22, 7)
```

## Arithmetic Operations

```python
a = Fraction(1, 3)
b = Fraction(1, 6)

a + b  # Fraction(1, 2)
a - b  # Fraction(1, 6)
a * b  # Fraction(1, 18)
a / b  # Fraction(2, 1)

# Comparison
a > b  # True
a == Fraction(2, 6)  # True (auto-normalized)

# Works with int
a + 1  # Fraction(4, 3)
a * 3  # Fraction(1, 1)
```

## Rounding Operations

```python
from math import floor, ceil

f = Fraction(355, 113)  # ~3.14159

floor(f)   # 3 (uses __floor__)
ceil(f)    # 4 (uses __ceil__)
round(f)   # 3 (uses __round__, half to even)
round(f, 2)  # Fraction(157, 50) = 3.14

int(f)     # 3 (3.11+ implements __int__)
```

## Formatting (3.12+)

### Float-Style Formatting
```python
f = Fraction(355, 113)

format(f, '.6f')   # '3.141593'
format(f, '.6e')   # '3.141593e+00'
format(f, '.6g')   # '3.14159'
format(f, '.2%')   # '314.16%'

# F-string
f"{f:.4f}"  # '3.1416'
f"{f:*>20.6e}"  # '********3.141593e+00'
```

### Fraction-Style Formatting (3.13+)
```python
f = Fraction(103993, 33102)

format(f, '')       # '103993/33102'
format(f, '_')      # '103_993/33_102' (grouping)
format(f, '.^+10')  # '...+1/7...' (fill, align, sign)
format(f, '#')      # '3/1' (force denominator even if 1)
```

## Production Patterns

### Exact Financial Calculations
```python
from fractions import Fraction
from decimal import Decimal

class ExactMoney:
    """Financial calculations with exact rational arithmetic."""
    
    def __init__(self, amount: str):
        # Use Decimal for exact parsing, Fraction for arithmetic
        self._amount = Fraction(Decimal(amount))
    
    def split(self, parts: int) -> list['ExactMoney']:
        """Split amount exactly into n parts."""
        each = self._amount / parts
        return [ExactMoney._from_fraction(each) for _ in range(parts)]
    
    def allocate(self, ratios: list[int]) -> list['ExactMoney']:
        """Allocate by ratios (e.g., [1, 2, 3] = 1/6, 2/6, 3/6)."""
        total = sum(ratios)
        return [
            ExactMoney._from_fraction(self._amount * r / total)
            for r in ratios
        ]
    
    def to_cents(self) -> int:
        """Round to cents for final output."""
        return round(self._amount * 100)
    
    @classmethod
    def _from_fraction(cls, f: Fraction) -> 'ExactMoney':
        obj = cls.__new__(cls)
        obj._amount = f
        return obj
```

### Recipe Scaling
```python
def scale_recipe(ingredients: dict[str, Fraction], factor: Fraction) -> dict:
    """Scale recipe by exact rational factor."""
    return {name: amount * factor for name, amount in ingredients.items()}

# Example: Scale recipe by 3/4
recipe = {
    'flour_cups': Fraction(2, 1),
    'sugar_cups': Fraction(3, 4),
    'butter_tbsp': Fraction(6, 1),
}

scaled = scale_recipe(recipe, Fraction(3, 4))
# {'flour_cups': Fraction(3, 2), 'sugar_cups': Fraction(9, 16), ...}

# Display nicely
for name, amount in scaled.items():
    print(f"{name}: {amount} ({float(amount):.2f})")
```

### Finding Rational Approximations
```python
from fractions import Fraction
from math import pi, e, sqrt

def best_rational(value: float, max_denom: int = 1000) -> Fraction:
    """Find best rational approximation within denominator limit."""
    return Fraction(value).limit_denominator(max_denom)

# Famous approximations
best_rational(pi, 10)    # Fraction(22, 7)
best_rational(pi, 100)   # Fraction(311, 99)
best_rational(pi, 1000)  # Fraction(355, 113) - Milü
best_rational(e, 100)    # Fraction(193, 71)
best_rational(sqrt(2), 100)  # Fraction(99, 70)
```

### Unit Conversion with Exact Ratios
```python
class UnitConverter:
    """Exact unit conversion using rational ratios."""
    
    CONVERSIONS = {
        ('inch', 'cm'): Fraction(254, 100),
        ('lb', 'kg'): Fraction(45359237, 100000000),
        ('gallon', 'liter'): Fraction(3785411784, 1000000000),
        ('mile', 'km'): Fraction(1609344, 1000000),
    }
    
    @classmethod
    def convert(cls, value: Fraction, from_unit: str, to_unit: str) -> Fraction:
        key = (from_unit, to_unit)
        if key in cls.CONVERSIONS:
            return value * cls.CONVERSIONS[key]
        # Try reverse
        reverse = (to_unit, from_unit)
        if reverse in cls.CONVERSIONS:
            return value / cls.CONVERSIONS[reverse]
        raise ValueError(f"Unknown conversion: {from_unit} -> {to_unit}")

# Exact conversion: 5 inches to cm
UnitConverter.convert(Fraction(5), 'inch', 'cm')  # Fraction(127, 10) = 12.7
```

## Key Gotchas

1. **Float Input Loses Precision**: Use `Decimal` or strings for exact input
2. **Always Normalized**: Fraction(4, 8) becomes Fraction(1, 2)
3. **Denominator Always Positive**: Fraction(1, -2) becomes Fraction(-1, 2)
4. **Hashable & Immutable**: Safe for dict keys and sets
5. **Inherits from numbers.Rational**: Works with `isinstance()` checks

## Version History

| Version | Feature |
|---------|---------|
| 3.2 | Accept float/Decimal in constructor |
| 3.8 | as_integer_ratio() added |
| 3.9 | Uses math.gcd() for normalization |
| 3.11 | Underscores in strings, __int__ support |
| 3.12 | Space around slash, is_integer(), float-style formatting |
| 3.13 | Fill/alignment in fraction formatting |
| 3.14 | from_number(), any as_integer_ratio() object |
