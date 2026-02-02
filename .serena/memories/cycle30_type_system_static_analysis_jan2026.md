# Cycle 30: Type System & Static Analysis Patterns (January 2026)

**Research Date**: 2026-01-25
**Focus**: Python type hints, mypy/pyright, Ruff, advanced typing patterns

---

## 1. Python Typing Survey 2025 Key Findings

From Meta/JetBrains survey (1,241 respondents, 15% increase from 2024):

**Top Reasons for Type Adoption**:
1. Code quality improvement
2. IDE support / code navigation
3. Documentation (self-documenting code)
4. Catching bugs before runtime
5. Flexibility in gradual adoption

**Industry Trend**: Python typing is now a "critical tool for improving code quality, enabling performance optimizations, and supporting large-scale application development."

---

## 2. Type Checker Comparison (2026)

### mypy vs pyright

| Feature | mypy | pyright |
|---------|------|---------|
| Speed | Slower | 10-50x faster |
| IDE Integration | Good | Excellent (VS Code native) |
| Error Messages | Clear | More detailed |
| Strictness | Configurable | Stricter by default |
| Config | mypy.ini / pyproject.toml | pyrightconfig.json / pyproject.toml |

**Best Practice**: Use BOTH type checkers for comprehensive coverage
- Different interpretation of edge cases
- "Multiple typecheckers is the Python equivalent of multiple compilers"

### pyright Configuration (Production)
```json
// pyrightconfig.json
{
  "include": ["src"],
  "exclude": ["**/node_modules", "**/__pycache__"],
  "typeCheckingMode": "strict",
  "pythonVersion": "3.12",
  "reportMissingTypeStubs": "warning",
  "reportUnknownMemberType": "warning"
}
```

### mypy Configuration (Production)
```toml
# pyproject.toml
[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true

[[tool.mypy.overrides]]
module = "third_party.*"
ignore_missing_imports = true
```

---

## 3. Python 3.12+ Type Parameter Syntax (PEP 695)

### Old Style (Pre-3.12)
```python
from typing import TypeVar, Generic

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

class Container(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value

def first(items: list[T]) -> T:
    return items[0]
```

### New Style (Python 3.12+)
```python
# No imports needed! Type parameters declared inline
class Container[T]:
    def __init__(self, value: T) -> None:
        self.value = value

def first[T](items: list[T]) -> T:
    return items[0]

# With bounds
class NumberContainer[T: (int, float)]:
    def __init__(self, value: T) -> None:
        self.value = value

# Type aliases (PEP 695)
type Vector[T] = list[T]
type Matrix[T] = list[list[T]]
type Callback[**P, R] = Callable[P, R]
```

---

## 4. Advanced Typing Patterns

### Protocol (Structural Subtyping)
```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

# Any class with draw() method matches - no inheritance needed
class Circle:
    def draw(self) -> None:
        print("Drawing circle")

def render(shape: Drawable) -> None:
    shape.draw()

render(Circle())  # Works! Circle satisfies Drawable protocol
```

### ParamSpec (Preserving Function Signatures)
```python
from typing import ParamSpec, TypeVar, Callable

P = ParamSpec('P')
R = TypeVar('R')

def with_logging[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@with_logging
def add(a: int, b: int) -> int:
    return a + b

# Type signature preserved: add(a: int, b: int) -> int
```

### TypeVarTuple (Variadic Generics - PEP 646)
```python
from typing import TypeVarTuple, Unpack

Ts = TypeVarTuple('Ts')

def concat[*Ts](items: tuple[*Ts]) -> tuple[*Ts]:
    return items

# Python 3.11+ native syntax
class Array[*Shape]:
    def __init__(self, data: list[float]) -> None:
        self.data = data

# Use case: NumPy-style shape typing
type Tensor3D = Array[int, int, int]
```

### TypeGuard and TypeIs
```python
from typing import TypeGuard, TypeIs

# TypeGuard: Narrows type in True branch only
def is_string_list(val: list[object]) -> TypeGuard[list[str]]:
    return all(isinstance(x, str) for x in val)

# TypeIs (Python 3.13+): Narrows in both branches
def is_str(val: str | int) -> TypeIs[str]:
    return isinstance(val, str)

# With TypeIs: else branch knows val is int
```

### Self Type (Python 3.11+)
```python
from typing import Self

class Builder:
    def set_name(self, name: str) -> Self:
        self.name = name
        return self  # Returns correct subclass type

class AdvancedBuilder(Builder):
    def set_extra(self, extra: str) -> Self:
        self.extra = extra
        return self

# Chaining works with correct types
builder = AdvancedBuilder().set_name("test").set_extra("data")
```

### Annotated (Metadata Attachment)
```python
from typing import Annotated
from pydantic import Field

# Attach validation metadata to types
UserId = Annotated[int, Field(gt=0, description="User ID")]
Email = Annotated[str, Field(pattern=r"^[\w.-]+@[\w.-]+\.\w+$")]

class User:
    id: UserId
    email: Email
```

---

## 5. Ruff Configuration (2026 Production Standard)

### Full Production Configuration
```toml
# pyproject.toml
[tool.ruff]
target-version = "py312"
line-length = 88
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # Pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
    "ARG",    # flake8-unused-arguments
    "SIM",    # flake8-simplify
    "TCH",    # flake8-type-checking
    "PTH",    # flake8-use-pathlib
    "ERA",    # eradicate (commented code)
    "PL",     # Pylint
    "RUF",    # Ruff-specific rules
]
ignore = [
    "E501",   # line-too-long (handled by formatter)
    "PLR0913", # too-many-arguments
]
fixable = ["ALL"]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101", "ARG"]  # Allow assert, unused args in tests
"__init__.py" = ["F401"]           # Allow unused imports in __init__

[tool.ruff.lint.isort]
known-first-party = ["mypackage"]
force-single-line = true

[tool.ruff.lint.flake8-type-checking]
strict = true  # Move imports to TYPE_CHECKING block

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
docstring-code-format = true
```

### Ruff Key Features (2026)
- **Speed**: Written in Rust, 10-100x faster than Flake8
- **Drop-in replacement**: Replaces Flake8 + isort + pydocstyle + pyupgrade + autoflake
- **800+ rules**: Comprehensive rule set
- **Fix capability**: `ruff check --fix` auto-fixes issues
- **Formatting**: `ruff format` replaces Black

### CI Integration
```yaml
# .github/workflows/lint.yml
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/ruff-action@v1
        with:
          args: "check --output-format=github"
      - uses: astral-sh/ruff-action@v1
        with:
          args: "format --check"
```

---

## 6. Type Checking in CI Pipeline

### Comprehensive Type Check Workflow
```yaml
# .github/workflows/type-check.yml
name: Type Checking
on: [push, pull_request]

jobs:
  typecheck:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install mypy pyright
          pip install -e ".[dev]"
      
      - name: Run mypy
        run: mypy src --strict
      
      - name: Run pyright
        run: pyright src
```

### Pre-commit Hook
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.13.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, pydantic]
```

---

## 7. Common Type Patterns

### Optional vs Union with None
```python
# Equivalent - use Optional for single type + None
from typing import Optional

# Preferred (Python 3.10+)
def find(id: int) -> User | None: ...

# Legacy style
def find(id: int) -> Optional[User]: ...
```

### Overload for Different Return Types
```python
from typing import overload, Literal

@overload
def fetch(url: str, as_json: Literal[True]) -> dict: ...
@overload
def fetch(url: str, as_json: Literal[False]) -> str: ...

def fetch(url: str, as_json: bool = False) -> dict | str:
    response = requests.get(url)
    return response.json() if as_json else response.text
```

### NewType for Type Safety
```python
from typing import NewType

UserId = NewType('UserId', int)
OrderId = NewType('OrderId', int)

def get_user(user_id: UserId) -> User: ...
def get_order(order_id: OrderId) -> Order: ...

# Type error! Can't pass OrderId where UserId expected
get_user(OrderId(123))  # Error
get_user(UserId(123))   # OK
```

### Final and ClassVar
```python
from typing import Final, ClassVar

class Config:
    MAX_CONNECTIONS: Final[int] = 100  # Cannot reassign
    instance_count: ClassVar[int] = 0  # Class-level, not instance

    def __init__(self) -> None:
        Config.instance_count += 1
```

---

## 8. Type Stub Management

### Installing Type Stubs
```bash
# Common stubs
pip install types-requests types-redis types-PyYAML

# Find stubs for a package
pip install types-<package-name>

# Generate stubs for untyped library
stubgen -p untyped_library -o stubs/
```

### pyproject.toml Stub Configuration
```toml
[tool.mypy]
mypy_path = "stubs"

[[tool.mypy.overrides]]
module = "untyped_library.*"
ignore_missing_imports = true
```

---

## 9. Runtime Type Checking (When Needed)

### beartype (Fast Runtime Validation)
```python
from beartype import beartype

@beartype
def process(data: list[int]) -> int:
    return sum(data)

process([1, 2, 3])    # OK
process(["a", "b"])   # Raises BeartypeCallHintViolation
```

### typeguard (Comprehensive Runtime Checks)
```python
from typeguard import typechecked

@typechecked
def greet(name: str, times: int) -> list[str]:
    return [f"Hello, {name}!"] * times
```

---

## 10. IDE Integration Best Practices

### VS Code Settings
```json
{
  "python.analysis.typeCheckingMode": "strict",
  "python.analysis.diagnosticMode": "workspace",
  "python.analysis.autoImportCompletions": true,
  "python.languageServer": "Pylance"
}
```

### PyCharm Configuration
- Settings → Editor → Inspections → Python → Type checker compatibility
- Enable "Type checker compatibility" inspection
- Set severity to "Error" for production code

---

## Key Takeaways

1. **Use Python 3.12+ syntax** - Cleaner generics without TypeVar imports
2. **Run both mypy and pyright** - Different edge case interpretations
3. **Ruff replaces multiple tools** - Faster, simpler configuration
4. **Protocol over ABC** - Structural subtyping is more flexible
5. **ParamSpec for decorators** - Preserves function signatures
6. **Annotated for metadata** - Combine types with validation
7. **Strict mode in production** - Catch more errors early
8. **Pre-commit hooks** - Enforce types before commit

---

*Cycle 30 Complete | Type System & Static Analysis Patterns*
