# Cycle 74: copy and copyreg Modules - Object Copying Patterns (January 2026)

## Overview
Research from official Python 3.14 documentation covering shallow/deep copy operations (copy module) and pickle/copy registration (copyreg module). These modules are essential for duplicating objects correctly and customizing serialization behavior.

---

## copy Module - Shallow and Deep Copy

### Core Concept

Assignment in Python creates references, not copies. The copy module provides actual object duplication.

```python
import copy

# Assignment creates reference (same object)
original = [1, [2, 3], 4]
reference = original
reference[1][0] = "X"
print(original)  # [1, ['X', 3], 4] - Original modified!

# copy.copy() creates shallow copy
original = [1, [2, 3], 4]
shallow = copy.copy(original)
shallow[0] = "A"           # Doesn't affect original
shallow[1][0] = "X"        # DOES affect original (shared nested list)
print(original)  # [1, ['X', 3], 4]

# copy.deepcopy() creates independent copy
original = [1, [2, 3], 4]
deep = copy.deepcopy(original)
deep[1][0] = "X"           # Doesn't affect original
print(original)  # [1, [2, 3], 4] - Unchanged!
```

### Shallow Copy - copy.copy()

```python
import copy

class Container:
    def __init__(self, items):
        self.items = items

original = Container([1, 2, 3])
shallow = copy.copy(original)

# New Container object created
print(original is shallow)  # False

# BUT the items list is the SAME object
print(original.items is shallow.items)  # True

# Modifying nested objects affects both
shallow.items.append(4)
print(original.items)  # [1, 2, 3, 4]
```

### Deep Copy - copy.deepcopy()

```python
import copy

class TreeNode:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []

# Create tree structure
root = TreeNode("root", [
    TreeNode("child1", [TreeNode("grandchild1")]),
    TreeNode("child2")
])

# Deep copy creates complete independent tree
root_copy = copy.deepcopy(root)

# Verify independence
root_copy.children[0].value = "MODIFIED"
print(root.children[0].value)  # "child1" - unchanged
```

### The memo Dictionary

```python
import copy

# deepcopy uses memo to handle:
# 1. Circular references
# 2. Shared objects (copied only once)

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

# Create circular reference
a = Node("A")
b = Node("B")
a.next = b
b.next = a  # Circular!

# deepcopy handles this correctly
a_copy = copy.deepcopy(a)
print(a_copy.next.next is a_copy)  # True - Circular reference preserved

# Custom memo usage
memo = {}
copy1 = copy.deepcopy(original, memo)
# memo now contains id -> copy mappings
```

### copy.replace() (Python 3.13+)

```python
import copy
from dataclasses import dataclass
from typing import NamedTuple

# Works with dataclasses
@dataclass
class Point:
    x: int
    y: int
    z: int = 0

p1 = Point(1, 2, 3)
p2 = copy.replace(p1, y=20, z=30)
print(p2)  # Point(x=1, y=20, z=30)

# Works with named tuples
class Color(NamedTuple):
    red: int
    green: int
    blue: int

c1 = Color(255, 128, 64)
c2 = copy.replace(c1, green=200)
print(c2)  # Color(red=255, green=200, blue=64)
```

### Custom __copy__ and __deepcopy__

```python
import copy

class ManagedResource:
    _instances = []  # Class-level registry
    
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self._id = len(ManagedResource._instances)
        ManagedResource._instances.append(self)
    
    def __copy__(self):
        """Shallow copy: new instance with same data reference."""
        new = ManagedResource.__new__(ManagedResource)
        new.name = self.name + "_copy"
        new.data = self.data  # Same reference
        new._id = len(ManagedResource._instances)
        ManagedResource._instances.append(new)
        return new
    
    def __deepcopy__(self, memo):
        """Deep copy: new instance with copied data."""
        new = ManagedResource.__new__(ManagedResource)
        memo[id(self)] = new  # Register BEFORE copying children
        
        new.name = copy.deepcopy(self.name, memo)
        new.data = copy.deepcopy(self.data, memo)
        new._id = len(ManagedResource._instances)
        ManagedResource._instances.append(new)
        return new

resource = ManagedResource("db", {"host": "localhost", "port": 5432})
shallow = copy.copy(resource)
deep = copy.deepcopy(resource)

print(resource.data is shallow.data)  # True (shared)
print(resource.data is deep.data)     # False (independent)
```

### Custom __replace__ (Python 3.13+)

```python
import copy

class ImmutableConfig:
    __slots__ = ('host', 'port', 'debug')
    
    def __init__(self, host, port, debug=False):
        object.__setattr__(self, 'host', host)
        object.__setattr__(self, 'port', port)
        object.__setattr__(self, 'debug', debug)
    
    def __setattr__(self, name, value):
        raise AttributeError("ImmutableConfig is immutable")
    
    def __replace__(self, **changes):
        """Create new instance with specified changes."""
        return ImmutableConfig(
            host=changes.get('host', self.host),
            port=changes.get('port', self.port),
            debug=changes.get('debug', self.debug)
        )

config = ImmutableConfig("localhost", 8080)
dev_config = copy.replace(config, debug=True)
print(dev_config.debug)  # True
```

### Built-in Copy Methods

```python
# Many built-in types have copy methods

# dict.copy() - shallow copy
d1 = {"a": [1, 2], "b": 3}
d2 = d1.copy()
d2["a"].append(3)  # Affects d1["a"] too!

# list slicing - shallow copy
l1 = [1, [2, 3], 4]
l2 = l1[:]  # or list(l1)
l2[1].append(4)  # Affects l1[1] too!

# set.copy() - shallow copy
s1 = {1, 2, 3}
s2 = s1.copy()

# For deep copies, always use copy.deepcopy()
import copy
d_deep = copy.deepcopy(d1)
l_deep = copy.deepcopy(l1)
```

### What Cannot Be Copied

```python
import copy

# These types return the original object unchanged:
# - modules
# - functions
# - classes
# - methods
# - stack traces
# - stack frames
# - files
# - sockets

def my_func():
    pass

func_copy = copy.copy(my_func)
print(func_copy is my_func)  # True - same object!

class MyClass:
    pass

class_copy = copy.copy(MyClass)
print(class_copy is MyClass)  # True - same object!
```

---

## copyreg Module - Pickle/Copy Registration

### Basic Registration

```python
import copyreg
import copy
import pickle

class LegacyPoint:
    """Class that doesn't define __reduce__ or __getstate__."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

def pickle_legacy_point(point):
    """Reduction function for LegacyPoint."""
    print(f"Pickling point ({point.x}, {point.y})")
    # Return (constructor, args) tuple
    return (LegacyPoint, (point.x, point.y))

# Register the pickle function
copyreg.pickle(LegacyPoint, pickle_legacy_point)

# Now both pickle AND copy use this function
point = LegacyPoint(10, 20)

# copy.copy() triggers pickle_legacy_point
copied = copy.copy(point)  # Prints: Pickling point (10, 20)

# pickle.dumps() also triggers it
data = pickle.dumps(point)  # Prints: Pickling point (10, 20)
```

### Reduction Function Return Values

```python
import copyreg
import pickle

class ComplexObject:
    def __init__(self, value, metadata=None):
        self.value = value
        self.metadata = metadata or {}
        self._cache = {}  # Don't want to pickle this

def reduce_complex(obj):
    """
    Reduction function can return:
    - string: Global variable name
    - 2-tuple: (callable, args)
    - 3-tuple: (callable, args, state)
    - 4-tuple: (callable, args, state, list_items)
    - 5-tuple: (callable, args, state, list_items, dict_items)
    - 6-tuple: (callable, args, state, list_items, dict_items, state_setter)
    """
    # 3-tuple: constructor, args, additional state
    state = {'metadata': obj.metadata}  # Exclude _cache
    return (ComplexObject, (obj.value,), state)

def setstate_complex(obj, state):
    """Custom state restoration."""
    obj.metadata = state.get('metadata', {})
    obj._cache = {}  # Initialize empty cache

# Register with custom state setter (6-tuple style in __reduce__)
copyreg.pickle(ComplexObject, reduce_complex)

obj = ComplexObject(42, {"created": "2026-01-25"})
obj._cache["temp"] = "data"

# Pickle and restore
data = pickle.dumps(obj)
restored = pickle.loads(data)

print(restored.value)     # 42
print(restored.metadata)  # {'created': '2026-01-25'}
print(restored._cache)    # {} - Empty as expected
```

### copyreg.constructor()

```python
import copyreg

def my_factory(x, y):
    """Factory function that creates objects."""
    return {"x": x, "y": y, "sum": x + y}

# Declare as valid constructor for pickle
copyreg.constructor(my_factory)

# Now my_factory can be used in reduction tuples
# (my_factory, (10, 20)) is valid
```

### Extension Registry (Advanced)

```python
import copyreg

# copyreg maintains extension registry for protocol 2+
# Maps (module, name) -> code and vice versa

# Add extension code (rarely needed in application code)
# copyreg.add_extension(module, name, code)

# Remove extension
# copyreg.remove_extension(module, name, code)

# Lookup
# copyreg.extension_registry  # (module, name) -> code
# copyreg.inverted_registry   # code -> (module, name)
```

---

## Pickle Integration: __reduce__ and __reduce_ex__

### Using __reduce__

```python
import pickle

class DatabaseConnection:
    def __init__(self, host, port, database):
        self.host = host
        self.port = port
        self.database = database
        self._connection = self._connect()
    
    def _connect(self):
        return f"Connection to {self.host}:{self.port}/{self.database}"
    
    def __reduce__(self):
        """
        Return tuple for reconstruction.
        Connection will be re-established on unpickle.
        """
        return (
            self.__class__,
            (self.host, self.port, self.database)
        )

conn = DatabaseConnection("localhost", 5432, "mydb")
data = pickle.dumps(conn)
restored = pickle.loads(data)

print(restored._connection)  # Connection to localhost:5432/mydb
```

### Using __reduce_ex__ for Protocol-Specific Behavior

```python
import pickle

class VersionedData:
    def __init__(self, value, version=1):
        self.value = value
        self.version = version
    
    def __reduce_ex__(self, protocol):
        """Protocol-aware reduction."""
        print(f"Using protocol {protocol}")
        
        if protocol >= 4:
            # Use more efficient format for newer protocols
            return (
                self.__class__,
                (self.value, self.version),
                None,  # state
                None,  # list items
                None,  # dict items
            )
        else:
            # Simpler format for older protocols
            return (self.__class__, (self.value,))

obj = VersionedData(42, version=2)

# Different protocols
pickle.dumps(obj, protocol=2)  # Using protocol 2
pickle.dumps(obj, protocol=5)  # Using protocol 5
```

### Using __getstate__ and __setstate__

```python
import pickle

class CachedProcessor:
    def __init__(self, config):
        self.config = config
        self._cache = {}
        self._initialized = True
    
    def __getstate__(self):
        """Return state to pickle (exclude cache)."""
        state = self.__dict__.copy()
        del state['_cache']  # Don't pickle cache
        return state
    
    def __setstate__(self, state):
        """Restore state from pickle."""
        self.__dict__.update(state)
        self._cache = {}  # Reinitialize cache

processor = CachedProcessor({"threads": 4})
processor._cache["key"] = "expensive_result"

data = pickle.dumps(processor)
restored = pickle.loads(data)

print(restored.config)       # {'threads': 4}
print(restored._cache)       # {} - Empty
print(restored._initialized) # True
```

---

## Production Patterns

### Immutable Copy Pattern

```python
import copy
from dataclasses import dataclass, field
from typing import FrozenSet

@dataclass(frozen=True)
class ImmutableRecord:
    id: int
    tags: FrozenSet[str] = field(default_factory=frozenset)
    
    def with_tags(self, *new_tags):
        """Return new record with additional tags."""
        return copy.replace(self, tags=self.tags | set(new_tags))

record = ImmutableRecord(1, frozenset({"important"}))
updated = record.with_tags("urgent", "reviewed")
print(updated.tags)  # frozenset({'important', 'urgent', 'reviewed'})
```

### Prototype Pattern

```python
import copy
from abc import ABC, abstractmethod

class Prototype(ABC):
    @abstractmethod
    def clone(self):
        """Create a copy of this object."""
        pass

class Document(Prototype):
    def __init__(self, title, content, metadata=None):
        self.title = title
        self.content = content
        self.metadata = metadata or {}
    
    def clone(self):
        """Deep clone the document."""
        return copy.deepcopy(self)

# Template document
template = Document(
    "Report Template",
    "Introduction...",
    {"author": "System", "version": 1}
)

# Create copies for different uses
report1 = template.clone()
report1.title = "Q1 Report"

report2 = template.clone()
report2.title = "Q2 Report"
```

### Safe Deep Copy with Exclusions

```python
import copy

class SafeCopyMixin:
    """Mixin that excludes certain attributes from deep copy."""
    
    _nocopy_attrs = set()  # Override in subclass
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        
        for k, v in self.__dict__.items():
            if k in self._nocopy_attrs:
                # Share reference instead of copying
                setattr(new, k, v)
            else:
                setattr(new, k, copy.deepcopy(v, memo))
        
        return new

class Service(SafeCopyMixin):
    _nocopy_attrs = {'_connection', '_lock'}
    
    def __init__(self, config):
        self.config = config
        self._connection = self._create_connection()
        self._lock = threading.Lock()
    
    def _create_connection(self):
        return "shared_connection"
```

---

## Common Pitfalls

```python
import copy

# PITFALL 1: Expecting assignment to copy
a = [1, 2, 3]
b = a  # b IS a, not a copy!
b.append(4)
print(a)  # [1, 2, 3, 4]

# PITFALL 2: Shallow copy with nested mutables
original = {"data": [1, 2, 3]}
shallow = copy.copy(original)
shallow["data"].append(4)
print(original["data"])  # [1, 2, 3, 4] - Oops!

# PITFALL 3: Forgetting to register memo in __deepcopy__
class Bad:
    def __deepcopy__(self, memo):
        new = Bad()
        # WRONG: Forgot memo[id(self)] = new
        # Can cause infinite recursion with circular refs
        return new

class Good:
    def __deepcopy__(self, memo):
        new = Good()
        memo[id(self)] = new  # Register FIRST
        # Then copy children
        return new

# PITFALL 4: Modifying objects during copy
# __copy__/__deepcopy__ should not modify the original
```

---

## Version Compatibility

| Feature | Minimum Python |
|---------|---------------|
| copy.copy, copy.deepcopy | 1.5+ |
| __copy__, __deepcopy__ | 2.0+ |
| copyreg.pickle | 2.0+ |
| copy.replace | 3.13+ |
| __replace__ | 3.13+ |
| pickle protocol 5 | 3.8+ |

---

*Research conducted: January 2026*
*Sources: docs.python.org/3/library/copy.html, docs.python.org/3/library/copyreg.html*
