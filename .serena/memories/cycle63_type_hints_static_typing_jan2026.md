# Cycle 63: Type Hints & Static Typing Patterns (January 2026)

Production patterns from official Python typing documentation, mypy, pyright, and beartype.

## 1. Core Type Hints (typing module)

### Basic Annotations
```python
# Variables - type inference usually handles these
age: int = 25
name: str = "Alice"
is_active: bool = True

# Collections (Python 3.9+)
items: list[int] = [1, 2, 3]
mapping: dict[str, float] = {"rate": 0.05}
coordinates: tuple[int, int, int] = (10, 20, 30)
unique: set[str] = {"a", "b"}

# Variable-length tuples
args: tuple[int, ...] = (1, 2, 3, 4, 5)

# Union types (Python 3.10+)
value: int | str = "hello"
maybe_name: str | None = None  # Preferred over Optional[str]

# Pre-3.10 syntax
from typing import Union, Optional
value: Union[int, str] = "hello"
maybe_name: Optional[str] = None  # Same as str | None
```

### Function Annotations
```python
from collections.abc import Iterator, Callable, Iterable

def greet(name: str, excitement: int = 1) -> str:
    return f"Hello, {name}{'!' * excitement}"

# No return value
def log(message: str) -> None:
    print(message)

# Callable types
Handler = Callable[[str, int], bool]
def register(callback: Handler) -> None: ...

# Generator functions return Iterator
def count_up(n: int) -> Iterator[int]:
    for i in range(n):
        yield i

# *args and **kwargs
def flexible(*args: str, **kwargs: int) -> None:
    pass  # args: tuple[str, ...], kwargs: dict[str, int]

# Positional-only (/) and keyword-only (*) parameters
def strict(x: int, /, *, y: int) -> int:
    return x + y
# strict(1, y=2)  # OK
# strict(x=1, y=2)  # Error: x is positional-only
```

### TypedDict for Structured Dictionaries
```python
from typing import TypedDict, Required, NotRequired

class UserDict(TypedDict):
    id: int
    name: str
    email: str | None

class ConfigDict(TypedDict, total=False):
    # All keys optional when total=False
    debug: bool
    timeout: int

class MixedDict(TypedDict):
    required_field: Required[str]
    optional_field: NotRequired[int]

# Usage
user: UserDict = {"id": 1, "name": "Alice", "email": None}
config: ConfigDict = {"debug": True}  # timeout not required
```

### Literal Types
```python
from typing import Literal

Mode = Literal["read", "write", "append"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

def open_file(path: str, mode: Mode) -> None: ...
def set_log_level(level: LogLevel) -> None: ...

# Combined with overloads for return type narrowing
from typing import overload

@overload
def fetch(url: str, raw: Literal[True]) -> bytes: ...
@overload
def fetch(url: str, raw: Literal[False]) -> str: ...
def fetch(url: str, raw: bool = False) -> bytes | str:
    ...
```

## 2. TypeVar and Generics

### Basic TypeVar
```python
from typing import TypeVar

T = TypeVar('T')  # Can be any type

def first(items: list[T]) -> T:
    return items[0]

# With constraints
Numeric = TypeVar('Numeric', int, float)

def add(a: Numeric, b: Numeric) -> Numeric:
    return a + b

# With upper bound
from collections.abc import Hashable
HashableT = TypeVar('HashableT', bound=Hashable)

def get_hash(item: HashableT) -> int:
    return hash(item)
```

### Generic Classes
```python
from typing import Generic, TypeVar

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        return self._items.pop()
    
    def empty(self) -> bool:
        return not self._items

# Usage
int_stack: Stack[int] = Stack()
int_stack.push(42)
value: int = int_stack.pop()
```

### Python 3.12+ Syntax (PEP 695)
```python
# New generic syntax - cleaner, no explicit TypeVar needed
def first[T](items: list[T]) -> T:
    return items[0]

class Stack[T]:
    def __init__(self) -> None:
        self._items: list[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)

# Type aliases with 'type' statement
type Vector[T] = list[T]
type ConnectionMap = dict[str, list[str]]

# Bounded generics
def process[T: Hashable](item: T) -> int:
    return hash(item)

# Constrained generics
def format_num[T: (int, float)](value: T) -> str:
    return f"{value:.2f}"
```

## 3. Protocol - Structural Subtyping

### Defining Protocols
```python
from typing import Protocol, runtime_checkable

class SupportsClose(Protocol):
    def close(self) -> None: ...

class SupportsRead(Protocol):
    def read(self, n: int = -1) -> bytes: ...

# Classes don't need to explicitly inherit Protocol
class MyResource:
    def close(self) -> None:
        print("Closed")

def cleanup(resource: SupportsClose) -> None:
    resource.close()

# This works! MyResource implicitly implements SupportsClose
cleanup(MyResource())
```

### Protocol with Attributes
```python
from typing import Protocol

class Named(Protocol):
    name: str  # Required attribute

class Configurable(Protocol):
    timeout: int
    retries: int = 3  # With default

class DatabaseConfig:
    def __init__(self, timeout: int) -> None:
        self.timeout = timeout
        self.retries = 5
        self.name = "main_db"

def setup(config: Named & Configurable) -> None:  # Intersection (3.12+)
    print(f"Setting up {config.name} with timeout {config.timeout}")
```

### Generic Protocols
```python
from typing import Protocol, TypeVar

T_co = TypeVar('T_co', covariant=True)

class SupportsGetItem(Protocol[T_co]):
    def __getitem__(self, key: int) -> T_co: ...

def get_first[T](container: SupportsGetItem[T]) -> T:
    return container[0]

# Works with list, tuple, any indexable
get_first([1, 2, 3])  # Returns int
get_first(("a", "b"))  # Returns str
```

### Runtime Checkable Protocols
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> None: ...

class Circle:
    def draw(self) -> None:
        print("Drawing circle")

# Now isinstance works!
obj = Circle()
if isinstance(obj, Drawable):
    obj.draw()

# Note: Only checks method existence, not signatures
# Use sparingly - static checking is more reliable
```

## 4. mypy Configuration

### pyproject.toml (Recommended)
```toml
[tool.mypy]
python_version = "3.12"
strict = true  # Enables all strict flags

# Strict mode includes:
# warn_unused_configs = true
# disallow_any_generics = true
# disallow_subclassing_any = true
# disallow_untyped_calls = true
# disallow_untyped_defs = true
# disallow_incomplete_defs = true
# check_untyped_defs = true
# disallow_untyped_decorators = true
# warn_redundant_casts = true
# warn_unused_ignores = true
# warn_return_any = true
# no_implicit_reexport = true
# strict_equality = true
# strict_concatenate = true

# Additional recommended settings
warn_unreachable = true
show_error_codes = true
show_column_numbers = true

# Per-module overrides
[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

[[tool.mypy.overrides]]
module = "third_party_lib.*"
ignore_missing_imports = true
```

### Inline Type Ignores
```python
# Suppress specific error
x = some_untyped_function()  # type: ignore[no-untyped-call]

# Multiple error codes
value = risky_operation()  # type: ignore[return-value, arg-type]

# With explanation (recommended)
result = legacy_api()  # type: ignore[no-any-return]  # Legacy API returns Any
```

### reveal_type for Debugging
```python
from typing import reveal_type  # Python 3.11+

x = [1, 2, 3]
reveal_type(x)  # Revealed type is "list[int]"

# Pre-3.11: mypy/pyright provide reveal_type as a magic function
# Just call reveal_type(expr) without importing
```

## 5. pyright vs mypy

### Key Differences
| Feature | mypy | pyright |
|---------|------|---------|
| Language | Python | TypeScript |
| Speed | Slower | ~5x faster |
| IDE Integration | Plugins | Pylance (VS Code) |
| Strictness | Configurable | Stricter by default |
| Type Narrowing | Good | More sophisticated |
| Error Messages | Verbose | Clearer |

### pyright Configuration (pyrightconfig.json)
```json
{
  "include": ["src"],
  "exclude": ["**/node_modules", "**/__pycache__"],
  "typeCheckingMode": "strict",
  "pythonVersion": "3.12",
  "reportMissingImports": true,
  "reportMissingTypeStubs": false,
  "reportUnusedImport": true,
  "reportUnusedVariable": true,
  "reportDuplicateImport": true
}
```

### pyproject.toml for pyright
```toml
[tool.pyright]
include = ["src"]
typeCheckingMode = "strict"
pythonVersion = "3.12"
reportMissingImports = true
reportUnusedVariable = "warning"
```

## 6. beartype - Runtime Type Checking

### Basic Usage
```python
from beartype import beartype

@beartype
def greet(name: str, times: int = 1) -> str:
    return f"Hello, {name}! " * times

greet("Alice", 3)  # OK
greet(123, 3)  # BeartypeCallHintParamViolation at RUNTIME
```

### Package-Wide Type Checking (Recommended)
```python
# In your_package/__init__.py
from beartype.claw import beartype_this_package

beartype_this_package()  # Type-check ALL functions in package

# Now every annotated function is automatically checked
```

### Beartype Validators (Custom Constraints)
```python
from beartype import beartype
from beartype.vale import Is
from typing import Annotated

# Custom validator with lambda
NonEmptyStr = Annotated[str, Is[lambda s: len(s) > 0]]
PositiveInt = Annotated[int, Is[lambda n: n > 0]]
Percentage = Annotated[float, Is[lambda p: 0.0 <= p <= 100.0]]

@beartype
def create_user(name: NonEmptyStr, age: PositiveInt) -> dict:
    return {"name": name, "age": age}

create_user("Alice", 25)  # OK
create_user("", 25)  # BeartypeCallHintParamViolation: empty string
create_user("Bob", -5)  # BeartypeCallHintParamViolation: negative
```

### NumPy Array Validation
```python
from beartype import beartype
from beartype.vale import Is
from typing import Annotated
import numpy as np

# 2D float array validator
Numpy2DFloat = Annotated[
    np.ndarray,
    Is[lambda arr: arr.ndim == 2 and np.issubdtype(arr.dtype, np.floating)]
]

@beartype
def process_matrix(data: Numpy2DFloat) -> float:
    return float(np.mean(data))

process_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))  # OK
process_matrix(np.array([1, 2, 3]))  # Error: not 2D
```

### Runtime Checking Without Decorators
```python
from beartype.door import is_bearable, die_if_unbearable

# Check if value matches type hint
if is_bearable(value, list[str]):
    # value is list[str]
    ...

# Raise exception if not matching
die_if_unbearable(config, dict[str, int])
```

## 7. Best Practices

### Gradual Typing Strategy
```python
# 1. Start with public API boundaries
def public_function(data: list[dict[str, Any]]) -> Result: ...

# 2. Add TypedDict for complex dict structures
class UserData(TypedDict):
    id: int
    name: str
    settings: dict[str, Any]  # Can be refined later

# 3. Use Protocol for dependencies
class Storage(Protocol):
    def save(self, key: str, value: bytes) -> None: ...
    def load(self, key: str) -> bytes | None: ...

# 4. Gradually replace Any with specific types
```

### Common Patterns
```python
from typing import TypeVar, Callable, ParamSpec

# Decorator that preserves signature
P = ParamSpec('P')
R = TypeVar('R')

def logged(func: Callable[P, R]) -> Callable[P, R]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

# Self type for fluent interfaces
from typing import Self

class Builder:
    def with_name(self, name: str) -> Self:
        self.name = name
        return self
    
    def with_value(self, value: int) -> Self:
        self.value = value
        return self
```

### Forward References
```python
from __future__ import annotations  # Postponed evaluation (recommended)

class Node:
    def __init__(self, children: list[Node]) -> None:  # Works!
        self.children = children

# Or use string quotes
class Tree:
    def __init__(self, left: "Tree | None", right: "Tree | None") -> None:
        self.left = left
        self.right = right
```

## Quick Reference

| Concept | Syntax | Use Case |
|---------|--------|----------|
| Union | `int \| str` | Multiple possible types |
| Optional | `str \| None` | Nullable values |
| TypeVar | `T = TypeVar('T')` | Generic functions/classes |
| Protocol | `class P(Protocol)` | Structural subtyping |
| Literal | `Literal["a", "b"]` | Exact value types |
| TypedDict | `class D(TypedDict)` | Typed dictionaries |
| Annotated | `Annotated[int, ...]` | Metadata + validators |
| Self | `-> Self` | Fluent interfaces |
| ParamSpec | `P = ParamSpec('P')` | Decorator typing |

## Sources
- Python typing documentation (docs.python.org/3/library/typing.html)
- typing.python.org specification
- mypy documentation (mypy.readthedocs.io)
- pyright documentation (github.com/microsoft/pyright)
- beartype documentation (beartype.readthedocs.io)
- PEP 544 (Protocols), PEP 695 (Type Parameter Syntax)
