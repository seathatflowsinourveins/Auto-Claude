# decimal Module - Production Patterns (January 2026)

## Quick Reference

```python
from decimal import (
    Decimal, getcontext, setcontext, localcontext,
    Context, BasicContext, ExtendedContext,
    ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR,
    ROUND_HALF_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP,
    ROUND_UP, ROUND_05UP,
    InvalidOperation, DivisionByZero, Inexact,
    Rounded, Subnormal, Overflow, Underflow,
    FloatOperation, Clamped,
)
```

## Why Decimal Over Float

### Float Precision Problems

```python
# Float has binary representation issues
>>> 0.1 + 0.1 + 0.1 - 0.3
5.551115123125783e-17  # Not zero!

>>> 1.1 + 2.2
3.3000000000000003  # Not exactly 3.3!

# Decimal solves this
>>> from decimal import Decimal
>>> Decimal('0.1') + Decimal('0.1') + Decimal('0.1') - Decimal('0.3')
Decimal('0.0')  # Exactly zero!

>>> Decimal('1.1') + Decimal('2.2')
Decimal('3.3')  # Exactly 3.3!
```

### When to Use Decimal

- **Financial calculations**: Money, accounting, invoicing
- **Scientific calculations**: When exact precision matters
- **Trading systems**: Price calculations, P&L
- **Tax calculations**: Where rounding rules are legally defined
- **Any equality comparisons**: Where float errors cause bugs

## Creating Decimals

### From Strings (Preferred)

```python
from decimal import Decimal

# Always use strings for exact values
price = Decimal('19.99')
rate = Decimal('0.0725')
quantity = Decimal('3')

# Supports scientific notation
large = Decimal('1.5E+10')
small = Decimal('2.5E-8')
```

### From Integers

```python
# Integers convert exactly
count = Decimal(42)
negative = Decimal(-100)
```

### From Floats (Use with Caution!)

```python
# Float conversion captures float's inexactness
>>> Decimal(0.1)
Decimal('0.1000000000000000055511151231257827021181583404541015625')

# Use string instead!
>>> Decimal('0.1')
Decimal('0.1')

# Or convert via string
>>> Decimal(str(0.1))
Decimal('0.1')
```

### From Tuples

```python
# (sign, digits, exponent)
# sign: 0 = positive, 1 = negative
>>> Decimal((0, (3, 1, 4, 1, 5), -4))
Decimal('3.1415')

>>> Decimal((1, (1, 2, 3), 2))
Decimal('-12300')
```

### Special Values

```python
>>> Decimal('Infinity')
Decimal('Infinity')

>>> Decimal('-Infinity')
Decimal('-Infinity')

>>> Decimal('NaN')
Decimal('NaN')

>>> Decimal('-0')  # Negative zero
Decimal('-0')
```

## Context and Precision

### Getting/Setting Context

```python
from decimal import getcontext, setcontext, Context

# Get current context
ctx = getcontext()
print(ctx.prec)  # Default: 28 digits

# Set precision globally
getcontext().prec = 50

# Set rounding globally
getcontext().rounding = ROUND_HALF_UP
```

### Local Context (Thread-Safe)

```python
from decimal import localcontext, ROUND_DOWN

# Temporary context for a block
with localcontext() as ctx:
    ctx.prec = 6
    ctx.rounding = ROUND_DOWN
    result = Decimal('1') / Decimal('7')
    # Result has 6 digits, rounded down

# Original context restored after block
```

### Context Parameters

```python
from decimal import Context

ctx = Context(
    prec=28,                    # Precision (digits)
    rounding=ROUND_HALF_EVEN,   # Rounding mode
    Emin=-999999,               # Minimum exponent
    Emax=999999,                # Maximum exponent
    capitals=1,                 # E vs e in output
    clamp=0,                    # Clamping mode
    traps=[InvalidOperation, DivisionByZero, Overflow],
)
```

## Rounding Modes

```python
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR
from decimal import ROUND_HALF_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP
from decimal import ROUND_UP, ROUND_05UP

value = Decimal('2.675')
places = Decimal('0.01')

# ROUND_CEILING - toward positive infinity
value.quantize(places, rounding=ROUND_CEILING)   # 2.68

# ROUND_FLOOR - toward negative infinity  
value.quantize(places, rounding=ROUND_FLOOR)     # 2.67

# ROUND_DOWN - toward zero (truncate)
value.quantize(places, rounding=ROUND_DOWN)      # 2.67

# ROUND_UP - away from zero
value.quantize(places, rounding=ROUND_UP)        # 2.68

# ROUND_HALF_UP - round 5 up (common/standard)
value.quantize(places, rounding=ROUND_HALF_UP)   # 2.68

# ROUND_HALF_DOWN - round 5 down
value.quantize(places, rounding=ROUND_HALF_DOWN) # 2.67

# ROUND_HALF_EVEN - banker's rounding (default)
Decimal('2.665').quantize(places, rounding=ROUND_HALF_EVEN)  # 2.66
Decimal('2.675').quantize(places, rounding=ROUND_HALF_EVEN)  # 2.68

# ROUND_05UP - round up if last digit is 0 or 5
value.quantize(places, rounding=ROUND_05UP)      # 2.67
```

## quantize() for Fixed Decimal Places

```python
from decimal import Decimal, ROUND_HALF_UP

# Round to 2 decimal places
price = Decimal('19.995')
rounded = price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
# Decimal('20.00')

# Round to whole number
whole = Decimal('7.5').quantize(Decimal('1'))  # Decimal('8')

# Round to nearest 5 cents
nickel = Decimal('0.05')
Decimal('1.23').quantize(nickel)  # Decimal('1.25')

# Preserve trailing zeros
Decimal('10').quantize(Decimal('0.00'))  # Decimal('10.00')
```

## Mathematical Operations

### Basic Arithmetic

```python
a = Decimal('10.5')
b = Decimal('3')

a + b   # Decimal('13.5')
a - b   # Decimal('7.5')
a * b   # Decimal('31.5')
a / b   # Decimal('3.5')
a // b  # Decimal('3') - integer division
a % b   # Decimal('1.5') - remainder
a ** b  # Decimal('1157.625') - power
```

### Math Methods

```python
from decimal import Decimal

d = Decimal('100')

d.sqrt()           # Decimal('10')
d.ln()             # Natural log
d.log10()          # Base-10 log
d.exp()            # e^x

# Fused multiply-add (no intermediate rounding)
Decimal('2').fma(3, 5)  # 2*3+5 = Decimal('11')
```

### Comparison

```python
a = Decimal('1.0')
b = Decimal('1.00')

a == b              # True (numeric equality)
a.compare(b)        # Decimal('0') - equal
a.compare_total(b)  # Decimal('1') - different representation
```

## Decimal Methods

### Inspection

```python
d = Decimal('-123.45')

d.sign              # -1 (via as_tuple)
d.is_finite()       # True
d.is_infinite()     # False
d.is_nan()          # False
d.is_zero()         # False
d.is_signed()       # True (negative)
d.is_normal()       # True

d.as_tuple()        # DecimalTuple(sign=1, digits=(1,2,3,4,5), exponent=-2)
d.as_integer_ratio()  # (-12345, 100)
```

### Transformation

```python
d = Decimal('-3.14')

d.copy_abs()        # Decimal('3.14')
d.copy_negate()     # Decimal('3.14')
d.copy_sign(Decimal('1'))  # Decimal('3.14')

d.normalize()       # Remove trailing zeros
d.adjusted()        # Adjusted exponent
```

## Signals and Traps

### Signal Types

```python
from decimal import (
    InvalidOperation,  # Invalid operation (0/0, sqrt(-1))
    DivisionByZero,    # Division by zero
    Overflow,          # Result too large
    Underflow,         # Result too small
    Inexact,           # Result is inexact (rounded)
    Rounded,           # Result was rounded
    Subnormal,         # Result is subnormal
    FloatOperation,    # Float mixed with Decimal
    Clamped,           # Exponent clamped
)
```

### Enabling Traps

```python
from decimal import getcontext, FloatOperation, Inexact

ctx = getcontext()

# Trap float mixing (recommended for strict code)
ctx.traps[FloatOperation] = True

>>> Decimal(3.14)  # Raises FloatOperation!

# Trap inexact results
ctx.traps[Inexact] = True

>>> Decimal('1') / Decimal('3')  # Raises Inexact!
```

### Checking Flags

```python
from decimal import getcontext, Inexact

ctx = getcontext()
ctx.clear_flags()  # Reset flags

result = Decimal('1') / Decimal('3')

if ctx.flags[Inexact]:
    print("Result was rounded!")
```

## Production Money Class

```python
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True, slots=True)
class Money:
    """Immutable money type with currency."""
    
    amount: Decimal
    currency: str = "USD"
    
    # Standard quantization for currency
    CENTS = Decimal('0.01')
    
    def __post_init__(self):
        # Ensure amount is Decimal and rounded
        if isinstance(self.amount, float):
            raise TypeError("Use string or Decimal, not float")
        object.__setattr__(
            self, 'amount',
            Decimal(str(self.amount)).quantize(self.CENTS, ROUND_HALF_UP)
        )
    
    @classmethod
    def from_cents(cls, cents: int, currency: str = "USD") -> Self:
        """Create Money from cents (integer)."""
        return cls(Decimal(cents) / 100, currency)
    
    def __add__(self, other: Self) -> Self:
        self._check_currency(other)
        return Money(self.amount + other.amount, self.currency)
    
    def __sub__(self, other: Self) -> Self:
        self._check_currency(other)
        return Money(self.amount - other.amount, self.currency)
    
    def __mul__(self, factor: Decimal | int | str) -> Self:
        """Multiply by a factor (quantity, rate, etc.)."""
        if isinstance(factor, float):
            raise TypeError("Use string or Decimal for factor")
        return Money(self.amount * Decimal(str(factor)), self.currency)
    
    def __truediv__(self, divisor: Decimal | int | str) -> Self:
        """Divide by a factor."""
        if isinstance(divisor, float):
            raise TypeError("Use string or Decimal for divisor")
        return Money(self.amount / Decimal(str(divisor)), self.currency)
    
    def _check_currency(self, other: Self) -> None:
        if self.currency != other.currency:
            raise ValueError(f"Currency mismatch: {self.currency} vs {other.currency}")
    
    def to_cents(self) -> int:
        """Convert to cents (integer)."""
        return int(self.amount * 100)
    
    def __str__(self) -> str:
        return f"{self.currency} {self.amount:,.2f}"
    
    def __repr__(self) -> str:
        return f"Money({self.amount!r}, {self.currency!r})"


# Usage
price = Money("19.99")
quantity = Decimal("3")
total = price * quantity  # Money(Decimal('59.97'), 'USD')

tax_rate = Decimal("0.0825")
tax = total * tax_rate    # Money(Decimal('4.95'), 'USD')

grand_total = total + tax # Money(Decimal('64.92'), 'USD')

print(grand_total)        # "USD 64.92"
```

## Financial Calculation Patterns

### Percentage Calculations

```python
from decimal import Decimal, ROUND_HALF_UP

def apply_percentage(amount: Decimal, percentage: Decimal) -> Decimal:
    """Apply percentage (e.g., tax, discount)."""
    return (amount * percentage / 100).quantize(
        Decimal('0.01'), rounding=ROUND_HALF_UP
    )

# 8.25% tax on $100
tax = apply_percentage(Decimal('100'), Decimal('8.25'))
# Decimal('8.25')
```

### Interest Calculations

```python
def compound_interest(
    principal: Decimal,
    rate: Decimal,  # Annual rate as decimal (0.05 = 5%)
    years: int,
    compounds_per_year: int = 12
) -> Decimal:
    """Calculate compound interest."""
    n = Decimal(compounds_per_year)
    t = Decimal(years)
    
    # A = P(1 + r/n)^(nt)
    factor = (1 + rate / n) ** (n * t)
    return (principal * factor).quantize(Decimal('0.01'), ROUND_HALF_UP)

# $10,000 at 5% for 5 years, monthly compounding
result = compound_interest(Decimal('10000'), Decimal('0.05'), 5)
# Decimal('12833.59')
```

### Currency Conversion

```python
def convert_currency(
    amount: Decimal,
    rate: Decimal,
    target_precision: Decimal = Decimal('0.01')
) -> Decimal:
    """Convert currency at given exchange rate."""
    return (amount * rate).quantize(target_precision, ROUND_HALF_UP)

# Convert $100 USD to EUR at 0.92 rate
eur = convert_currency(Decimal('100'), Decimal('0.92'))
# Decimal('92.00')
```

## Key Takeaways

1. **Always use strings**: `Decimal('0.1')` not `Decimal(0.1)`
2. **quantize() for rounding**: Use it to round to fixed decimal places
3. **ROUND_HALF_UP for money**: Standard financial rounding
4. **ROUND_HALF_EVEN for statistics**: Reduces cumulative bias
5. **localcontext() for thread-safety**: Temporary precision changes
6. **Trap FloatOperation**: Catch accidental float mixing
7. **Immutable by design**: Decimal objects cannot be changed
8. **Default precision is 28**: Sufficient for most financial calculations
9. **Compare with ==**: Decimals compare numerically, not by representation
10. **Use for all money**: Never use float for financial calculations
