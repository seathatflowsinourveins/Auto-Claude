# Cycle 72: abc and inspect Modules - Production Patterns (January 2026)

## Overview
Research from official Python 3.13 documentation covering abstract base classes (abc module) and runtime introspection (inspect module). These modules are fundamental for framework design, metaprogramming, and building extensible APIs.

---

## ABC Module - Abstract Base Classes

### ABC vs ABCMeta

```python
from abc import ABC, ABCMeta, abstractmethod

# PREFERRED: Inherit from ABC (cleaner syntax)
class MyProtocol(ABC):
    @abstractmethod
    def process(self, data: bytes) -> bytes:
        """Must be implemented by subclasses."""
        ...

# ALTERNATIVE: Use metaclass directly (needed for multiple inheritance)
class MyMixin(metaclass=ABCMeta):
    @abstractmethod
    def validate(self) -> bool:
        ...

# Multiple inheritance with ABC
class Combined(SomeOtherBase, ABC):
    @abstractmethod
    def combine(self) -> None:
        ...
```

### @abstractmethod Decorator

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        """Calculate shape area."""
        ...
    
    @abstractmethod
    def perimeter(self) -> float:
        """Calculate shape perimeter."""
        ...
    
    # Concrete method can call abstract methods
    def describe(self) -> str:
        return f"Area: {self.area()}, Perimeter: {self.perimeter()}"

# CRITICAL: Decorator stacking order
class Service(ABC):
    # @abstractmethod must be INNERMOST
    @classmethod
    @abstractmethod
    def create(cls) -> 'Service':
        ...
    
    @staticmethod
    @abstractmethod
    def validate(data: dict) -> bool:
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        ...
    
    # Abstract setter
    @name.setter
    @abstractmethod
    def name(self, value: str) -> None:
        ...
```

### Virtual Subclasses with register()

```python
from abc import ABC, abstractmethod

class Serializable(ABC):
    @abstractmethod
    def serialize(self) -> bytes:
        ...

# Register existing class as virtual subclass
# Does NOT require implementing abstract methods!
@Serializable.register
class ThirdPartyData:
    def to_bytes(self) -> bytes:  # Different method name
        return b"data"

# Or register imperatively
Serializable.register(dict)

# Virtual subclass passes isinstance/issubclass checks
obj = ThirdPartyData()
assert isinstance(obj, Serializable)  # True!
assert issubclass(ThirdPartyData, Serializable)  # True!

# BUT abstract methods are NOT enforced
# ThirdPartyData().serialize()  # AttributeError!
```

### __subclasshook__ for Custom Subclass Logic

```python
from abc import ABC, abstractmethod

class Iterable(ABC):
    @abstractmethod
    def __iter__(self):
        ...
    
    @classmethod
    def __subclasshook__(cls, C):
        if cls is Iterable:
            # Check if C has __iter__ method
            if hasattr(C, '__iter__'):
                return True
        return NotImplemented  # Defer to normal mechanism

# Now ANY class with __iter__ is considered subclass
class MyCollection:
    def __iter__(self):
        return iter([1, 2, 3])

assert issubclass(MyCollection, Iterable)  # True (via __subclasshook__)
```

### Cache Token and Abstract Method Updates

```python
from abc import ABCMeta, get_cache_token, update_abstractmethods

# get_cache_token() - Returns token that changes when ABCs are modified
token1 = get_cache_token()
class NewABC(metaclass=ABCMeta): pass
token2 = get_cache_token()
assert token1 != token2  # Token changed

# update_abstractmethods() (3.10+) - Recalculate abstract status
class Base(metaclass=ABCMeta):
    @abstractmethod
    def method(self): ...

class Partial(Base):
    pass  # Still abstract

# Dynamically add implementation
Partial.method = lambda self: "implemented"
update_abstractmethods(Partial)  # Now instantiable!
obj = Partial()  # Works after update
```

---

## Inspect Module - Runtime Introspection

### Signature Inspection

```python
import inspect
from inspect import signature, Signature, Parameter

def greet(name: str, *, greeting: str = "Hello", times: int = 1) -> str:
    return (greeting + " " + name + "! ") * times

# Get signature object
sig = signature(greet)
print(sig)  # (name: str, *, greeting: str = 'Hello', times: int = 1) -> str

# Access parameters
for param_name, param in sig.parameters.items():
    print(f"{param_name}: kind={param.kind.name}, default={param.default}, annotation={param.annotation}")
# name: kind=POSITIONAL_OR_KEYWORD, default=<class 'inspect._empty'>, annotation=<class 'str'>
# greeting: kind=KEYWORD_ONLY, default='Hello', annotation=<class 'str'>
# times: kind=KEYWORD_ONLY, default=1, annotation=<class 'int'>

# Return annotation
print(sig.return_annotation)  # <class 'str'>
```

### Parameter Kinds

```python
from inspect import Parameter

# All 5 parameter kinds:
# 1. POSITIONAL_ONLY (before /)
# 2. POSITIONAL_OR_KEYWORD (regular params)
# 3. VAR_POSITIONAL (*args)
# 4. KEYWORD_ONLY (after * or *args)
# 5. VAR_KEYWORD (**kwargs)

def example(pos_only, /, pos_or_kw, *args, kw_only, **kwargs):
    pass

sig = signature(example)
for name, param in sig.parameters.items():
    print(f"{name}: {param.kind.name}")
# pos_only: POSITIONAL_ONLY
# pos_or_kw: POSITIONAL_OR_KEYWORD
# args: VAR_POSITIONAL
# kw_only: KEYWORD_ONLY
# kwargs: VAR_KEYWORD
```

### Binding Arguments

```python
from inspect import signature

def process(a, b, *args, c=10, **kwargs):
    pass

sig = signature(process)

# bind() - Raises TypeError if binding fails
bound = sig.bind(1, 2, 3, 4, c=20, d=30)
print(bound.arguments)  # {'a': 1, 'b': 2, 'args': (3, 4), 'c': 20, 'kwargs': {'d': 30}}

# apply_defaults() - Fill in default values
bound.apply_defaults()
print(bound.arguments)  # Same, but with defaults applied

# bind_partial() - Allows missing required args
partial_bound = sig.bind_partial(1)  # OK, 'b' missing
```

### Type Checking Functions

```python
import inspect
import asyncio

# Module/class/function checks
inspect.ismodule(inspect)      # True
inspect.isclass(str)           # True
inspect.isfunction(len)        # False (built-in)
inspect.isbuiltin(len)         # True
inspect.ismethod(str.upper)    # False (unbound)

class MyClass:
    def method(self): pass
    @classmethod
    def cls_method(cls): pass
    @staticmethod
    def static_method(): pass

obj = MyClass()
inspect.ismethod(obj.method)        # True (bound method)
inspect.isfunction(MyClass.method)  # True (function in class)

# Async checks
async def async_func(): pass
def sync_func(): pass
async def async_gen():
    yield 1

inspect.iscoroutinefunction(async_func)  # True
inspect.iscoroutinefunction(sync_func)   # False
inspect.isasyncgenfunction(async_gen)    # True
inspect.isgeneratorfunction(sync_func)   # False

# Abstract check
from abc import ABC, abstractmethod
class Abstract(ABC):
    @abstractmethod
    def method(self): ...

inspect.isabstract(Abstract)  # True
```

### Getting Members

```python
import inspect

class Example:
    class_var = 42
    
    def __init__(self):
        self.instance_var = "hello"
    
    def method(self):
        pass
    
    @property
    def prop(self):
        return self.instance_var

# getmembers() - Returns list of (name, value) pairs
members = inspect.getmembers(Example)
methods = inspect.getmembers(Example, predicate=inspect.isfunction)
# [('method', <function Example.method at ...>)]

# getmembers_static() (3.11+) - Doesn't invoke descriptors
# Safer for classes with complex __getattr__
static_members = inspect.getmembers_static(Example)

# get_annotations() (3.10+) - Properly resolve annotations
def func(x: 'ForwardRef') -> 'ReturnType':
    pass

annotations = inspect.get_annotations(func, eval_str=True)
# Evaluates string annotations if possible
```

### Source Code Retrieval

```python
import inspect

def my_function():
    """Docstring here."""
    x = 1
    return x + 1

# Get source code
source = inspect.getsource(my_function)
# 'def my_function():\n    """Docstring here."""\n    x = 1\n    return x + 1\n'

# Get source lines with line numbers
lines, start_line = inspect.getsourcelines(my_function)
# (['def my_function():\n', '    """Docstring here."""\n', ...], 5)

# Get file path
file_path = inspect.getfile(my_function)
# '/path/to/module.py'

# Get docstring (cleaned)
doc = inspect.getdoc(my_function)
# 'Docstring here.'

# cleandoc() - Clean up docstring indentation
raw_doc = """
    This is a
    multi-line docstring
    with indentation.
"""
clean = inspect.cleandoc(raw_doc)
# 'This is a\nmulti-line docstring\nwith indentation.'
```

### Stack and Frame Inspection

```python
import inspect

def inner():
    # Get current frame
    frame = inspect.currentframe()
    print(f"Current function: {frame.f_code.co_name}")
    print(f"Local vars: {frame.f_locals}")
    
    # Get full stack
    stack = inspect.stack()
    for frame_info in stack:
        print(f"  {frame_info.function} at {frame_info.filename}:{frame_info.lineno}")
    
    # FrameInfo attributes:
    # .frame, .filename, .lineno, .function, .code_context, .index, .positions

def outer():
    x = 10
    inner()

outer()

# Get outer frames from a frame object
def get_caller_info():
    frame = inspect.currentframe()
    outer_frames = inspect.getouterframes(frame)
    caller = outer_frames[1]  # Immediate caller
    return f"Called from {caller.function} at line {caller.lineno}"

# IMPORTANT: Always delete frame references to avoid reference cycles
frame = inspect.currentframe()
try:
    # Use frame
    pass
finally:
    del frame
```

### Generator and Coroutine State

```python
import inspect

def my_generator():
    yield 1
    yield 2

gen = my_generator()
print(inspect.getgeneratorstate(gen))  # GEN_CREATED
next(gen)
print(inspect.getgeneratorstate(gen))  # GEN_SUSPENDED
list(gen)
print(inspect.getgeneratorstate(gen))  # GEN_CLOSED

# States: GEN_CREATED, GEN_RUNNING, GEN_SUSPENDED, GEN_CLOSED

# For coroutines
async def my_coro():
    await asyncio.sleep(0)

coro = my_coro()
print(inspect.getcoroutinestate(coro))  # CORO_CREATED
# States: CORO_CREATED, CORO_RUNNING, CORO_SUSPENDED, CORO_CLOSED

# Get local variables in generator/coroutine
print(inspect.getgeneratorlocals(gen))  # {} or locals dict
```

### Practical Pattern: Automatic API Documentation

```python
import inspect
from typing import get_type_hints

def document_class(cls) -> str:
    """Generate documentation for a class."""
    lines = [f"# {cls.__name__}", ""]
    
    if cls.__doc__:
        lines.append(inspect.cleandoc(cls.__doc__))
        lines.append("")
    
    # Document methods
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith('_') and name != '__init__':
            continue
        
        sig = inspect.signature(method)
        lines.append(f"## {name}{sig}")
        
        if method.__doc__:
            lines.append(inspect.cleandoc(method.__doc__))
        lines.append("")
    
    return "\n".join(lines)
```

### Practical Pattern: Dependency Injection

```python
import inspect
from typing import get_type_hints

class Container:
    def __init__(self):
        self._services = {}
    
    def register(self, interface, implementation):
        self._services[interface] = implementation
    
    def resolve(self, cls):
        """Automatically inject dependencies based on type hints."""
        sig = inspect.signature(cls.__init__)
        hints = get_type_hints(cls.__init__)
        
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            param_type = hints.get(param_name)
            if param_type and param_type in self._services:
                kwargs[param_name] = self._services[param_type]()
        
        return cls(**kwargs)
```

---

## Production Best Practices

### ABC Design Guidelines

1. **Use ABC for protocols** that require implementation, Protocol from typing for structural subtyping
2. **register() sparingly** - Virtual subclasses bypass type safety
3. **__subclasshook__ for duck typing** - Check capabilities, not inheritance
4. **Combine with dataclasses** for data-carrying abstract types

### Inspect Safety Guidelines

1. **Delete frame references** immediately after use (avoid cycles)
2. **Use getmembers_static()** (3.11+) for untrusted classes
3. **getsource() can fail** - Handle OSError for built-ins, C extensions
4. **Signature works on most callables** - Not on some built-ins

### Performance Considerations

```python
# Cache signatures for repeated use
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_signature(func):
    return inspect.signature(func)

# Avoid repeated getmembers() calls
# Cache results if class structure is stable
```

---

## Version Compatibility

| Feature | Minimum Python |
|---------|---------------|
| ABC class | 3.4+ |
| @abstractmethod with @property | 3.3+ |
| update_abstractmethods() | 3.10+ |
| getmembers_static() | 3.11+ |
| get_annotations() | 3.10+ |
| Signature.from_callable() | 3.5+ |
| iscoroutinefunction() | 3.5+ |
| isasyncgenfunction() | 3.6+ |
| FrameInfo.positions | 3.11+ |

---

*Research conducted: January 2026*
*Sources: docs.python.org/3/library/abc.html, docs.python.org/3/library/inspect.html*
