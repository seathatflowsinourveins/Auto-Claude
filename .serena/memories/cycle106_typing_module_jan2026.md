# typing Module - Production Patterns (January 2026)

## Quick Reference

```python
from typing import (
    # Core types
    Any, Never, NoReturn, Self, LiteralString,
    # Generics
    TypeVar, Generic, ParamSpec, Concatenate, TypeVarTuple,
    # Type forms
    Union, Optional, Literal, Final, ClassVar,
    # Structural typing
    Protocol, runtime_checkable,
    # Type aliases
    TypeAlias, TypeAliasType,
    # Containers
    TypedDict, NamedTuple,
    # Utilities
    cast, assert_type, reveal_type, get_type_hints,
    get_origin, get_args, overload, final,
    # Guards
    TypeGuard, TypeIs,
    # Decorators
    dataclass_transform, override,
)
```

## Python 3.12+ Type Parameter Syntax

### Modern Generic Functions

```python
# Python 3.12+ syntax (preferred)
def first[T](items: list[T]) -> T:
    return items[0]

# Equivalent pre-3.12 syntax
from typing import TypeVar
T = TypeVar('T')
def first(items: list[T]) -> T:
    return items[0]
```

### Modern Generic Classes

```python
# Python 3.12+ syntax
class Stack[T]:
    def __init__(self) -> None:
        self._items: list[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        return self._items.pop()

# With bounds
class ComparableStack[T: Comparable]:
    ...

# With constraints
class NumberStack[T: (int, float)]:
    ...
```

### Modern Type Aliases

```python
# Python 3.12+ type statement (preferred)
type Vector = list[float]
type Matrix[T] = list[list[T]]
type ConnectionOptions = dict[str, str]

# Pre-3.12 TypeAlias
from typing import TypeAlias
Vector: TypeAlias = list[float]
```

## TypeVar Patterns

### Basic TypeVar

```python
from typing import TypeVar

T = TypeVar('T')  # Any type
K = TypeVar('K')  # Key type
V = TypeVar('V')  # Value type

def identity[T](x: T) -> T:
    return x
```

### Bounded TypeVar

```python
from typing import TypeVar
from collections.abc import Hashable

# Upper bound - T must be subtype of Hashable
T = TypeVar('T', bound=Hashable)

def get_key[T: Hashable](item: T) -> int:
    return hash(item)
```

### Constrained TypeVar

```python
# T must be exactly one of these types
T = TypeVar('T', int, float, complex)

def add_numbers[T: (int, float, complex)](a: T, b: T) -> T:
    return a + b
```

### Covariant/Contravariant

```python
from typing import TypeVar

# Covariant - can use more specific types (output positions)
T_co = TypeVar('T_co', covariant=True)

# Contravariant - can use more general types (input positions)
T_contra = TypeVar('T_contra', contravariant=True)

# Python 3.12+ syntax
class Producer[out T]:  # Covariant
    def get(self) -> T: ...

class Consumer[in T]:  # Contravariant
    def accept(self, item: T) -> None: ...
```

## Protocol (Structural Subtyping)

### Basic Protocol

```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

# Any class with draw() method satisfies this protocol
class Circle:
    def draw(self) -> None:
        print("Drawing circle")

def render(item: Drawable) -> None:
    item.draw()

render(Circle())  # OK - structural match
```

### Runtime Checkable Protocol

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Sized(Protocol):
    def __len__(self) -> int: ...

# Can use isinstance at runtime
def process(obj: object) -> int:
    if isinstance(obj, Sized):
        return len(obj)
    return 0
```

### Protocol with Generic

```python
from typing import Protocol

class Comparable[T](Protocol):
    def __lt__(self, other: T) -> bool: ...
    def __gt__(self, other: T) -> bool: ...

def max_item[T: Comparable[T]](items: list[T]) -> T:
    return max(items)
```

## TypedDict

### Basic TypedDict

```python
from typing import TypedDict

class User(TypedDict):
    name: str
    age: int
    email: str

# All keys required by default
user: User = {"name": "Alice", "age": 30, "email": "a@b.com"}
```

### Optional Keys

```python
from typing import TypedDict, NotRequired, Required

class User(TypedDict):
    name: str              # Required (default in TypedDict)
    age: int
    email: NotRequired[str]  # Optional key

# Or use total=False
class OptionalUser(TypedDict, total=False):
    name: Required[str]    # Required even with total=False
    age: int               # Optional
    email: str             # Optional
```

### Inheritance

```python
class BaseUser(TypedDict):
    name: str

class AdminUser(BaseUser):
    admin_level: int
```

## Literal and Final

### Literal Types

```python
from typing import Literal

def set_mode(mode: Literal["read", "write", "append"]) -> None:
    ...

# Numeric literals
def set_verbosity(level: Literal[0, 1, 2, 3]) -> None:
    ...

# Boolean literals
def toggle(on: Literal[True, False]) -> None:
    ...
```

### Final (Constants)

```python
from typing import Final

MAX_SIZE: Final = 100           # Cannot be reassigned
MAX_SIZE: Final[int] = 100      # Explicit type

class Config:
    DEBUG: Final = False        # Class constant
```

## Callable and ParamSpec

### Basic Callable

```python
from collections.abc import Callable

# Function taking int, returning str
Processor = Callable[[int], str]

# Function with variable arguments
Handler = Callable[..., None]

def apply(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)
```

### ParamSpec (Preserving Signatures)

```python
from typing import ParamSpec, Callable, TypeVar

P = ParamSpec('P')
R = TypeVar('R')

def logged[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@logged
def add(a: int, b: int) -> int:
    return a + b
```

### Concatenate

```python
from typing import Concatenate, ParamSpec, Callable

P = ParamSpec('P')

# Add first argument to callable signature
def with_session[**P, R](
    func: Callable[Concatenate[Session, P], R]
) -> Callable[P, R]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with Session() as session:
            return func(session, *args, **kwargs)
    return wrapper
```

## TypeGuard and TypeIs

### TypeGuard (Narrowing)

```python
from typing import TypeGuard

def is_string_list(val: list[object]) -> TypeGuard[list[str]]:
    return all(isinstance(x, str) for x in val)

def process(items: list[object]) -> None:
    if is_string_list(items):
        # Type narrowed to list[str]
        for s in items:
            print(s.upper())
```

### TypeIs (Python 3.13+)

```python
from typing import TypeIs

def is_str(val: str | int) -> TypeIs[str]:
    return isinstance(val, str)

# TypeIs is more precise than TypeGuard for narrowing
```

## Self Type

```python
from typing import Self

class Builder:
    def with_name(self, name: str) -> Self:
        self.name = name
        return self
    
    def with_age(self, age: int) -> Self:
        self.age = age
        return self

class AdminBuilder(Builder):
    def with_role(self, role: str) -> Self:
        self.role = role
        return self

# Chaining works correctly with inheritance
admin = AdminBuilder().with_name("Alice").with_age(30).with_role("admin")
```

## Overload

```python
from typing import overload

@overload
def process(x: int) -> int: ...
@overload
def process(x: str) -> str: ...
@overload
def process(x: list[int]) -> list[int]: ...

def process(x: int | str | list[int]) -> int | str | list[int]:
    if isinstance(x, int):
        return x * 2
    elif isinstance(x, str):
        return x.upper()
    else:
        return [i * 2 for i in x]
```

## NewType (Lightweight Types)

```python
from typing import NewType

UserId = NewType('UserId', int)
OrderId = NewType('OrderId', int)

def get_user(user_id: UserId) -> User:
    ...

# Type-safe distinct types
user_id = UserId(123)
order_id = OrderId(456)

get_user(user_id)   # OK
get_user(order_id)  # Type error! OrderId is not UserId
get_user(123)       # Type error! int is not UserId
```

## Annotated (Metadata)

```python
from typing import Annotated

# Attach metadata to types
UserId = Annotated[int, "User ID from database"]
PositiveInt = Annotated[int, "Must be > 0"]

# With validators (e.g., Pydantic)
from pydantic import Field
Age = Annotated[int, Field(ge=0, le=150)]
```

## assert_type and reveal_type

```python
from typing import assert_type, reveal_type

def example() -> None:
    x = [1, 2, 3]
    
    # Verify type at static analysis time
    assert_type(x, list[int])  # Passes
    
    # Show inferred type during type checking
    reveal_type(x)  # Reveals: list[int]
```

## TYPE_CHECKING Guard

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported during type checking, not at runtime
    from expensive_module import HeavyClass

def process(item: "HeavyClass") -> None:  # Forward reference
    ...
```

## Introspection

### get_type_hints

```python
from typing import get_type_hints

class User:
    name: str
    age: int

hints = get_type_hints(User)
# {'name': <class 'str'>, 'age': <class 'int'>}
```

### get_origin and get_args

```python
from typing import get_origin, get_args

# Decompose generic types
get_origin(list[int])           # list
get_args(list[int])             # (int,)

get_origin(dict[str, int])      # dict
get_args(dict[str, int])        # (str, int)

get_origin(int | str)           # types.UnionType
get_args(int | str)             # (int, str)
```

## Production Type System Class

```python
from typing import TypeVar, Generic, Protocol, TypedDict, Self
from dataclasses import dataclass
from abc import abstractmethod


# Domain types
class Entity(Protocol):
    id: int

class Repository[E: Entity](Protocol):
    def get(self, id: int) -> E | None: ...
    def save(self, entity: E) -> E: ...
    def delete(self, id: int) -> bool: ...

@dataclass
class User:
    id: int
    name: str
    email: str

class UserRepository(Repository[User]):
    def __init__(self) -> None:
        self._users: dict[int, User] = {}
    
    def get(self, id: int) -> User | None:
        return self._users.get(id)
    
    def save(self, entity: User) -> User:
        self._users[entity.id] = entity
        return entity
    
    def delete(self, id: int) -> bool:
        return self._users.pop(id, None) is not None


# Result type for error handling
class Result[T, E](Generic[T, E]):
    def __init__(self, value: T | None = None, error: E | None = None) -> None:
        self._value = value
        self._error = error
    
    @classmethod
    def ok(cls, value: T) -> Self:
        return cls(value=value)
    
    @classmethod
    def err(cls, error: E) -> Self:
        return cls(error=error)
    
    def is_ok(self) -> bool:
        return self._value is not None
    
    def unwrap(self) -> T:
        if self._value is None:
            raise ValueError("Result contains error")
        return self._value
```

## Key Takeaways

1. **Python 3.12+ syntax**: Use `def func[T]()` and `class C[T]:` for generics
2. **type statement**: Use `type Alias = ...` for type aliases (3.12+)
3. **Protocol over ABC**: Prefer structural typing with Protocol
4. **Self for fluent APIs**: Use Self for method chaining in subclasses
5. **TypeGuard for narrowing**: Custom type guards for complex conditions
6. **ParamSpec for decorators**: Preserve function signatures in wrappers
7. **NewType for safety**: Create distinct types without runtime cost
8. **TYPE_CHECKING**: Import expensive types only for static analysis
9. **Literal for enums**: Use Literal for exhaustive string/int options
10. **TypedDict for dicts**: Type structured dictionaries properly
