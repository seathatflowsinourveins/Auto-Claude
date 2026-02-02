# Cycle 73: weakref and gc Modules - Memory Management Patterns (January 2026)

## Overview
Research from official Python 3.14 documentation covering weak references (weakref module) and garbage collection interface (gc module). These modules are essential for advanced memory management, cache implementations, and preventing memory leaks in long-running applications.

---

## weakref Module - Weak References

### Core Concept

A weak reference does NOT increase the reference count of an object. When only weak references remain, the garbage collector is free to destroy the object.

```python
import weakref

class ExpensiveObject:
    def __init__(self, name):
        self.name = name
    def __del__(self):
        print(f"Deleting {self.name}")

# Create object and weak reference
obj = ExpensiveObject("resource")
weak_ref = weakref.ref(obj)

# Access via weak reference (call it like a function)
print(weak_ref())  # <ExpensiveObject object at ...>

# Delete strong reference
del obj  # Prints: "Deleting resource"

# Weak reference now returns None
print(weak_ref())  # None
```

### weakref.ref() with Callback

```python
import weakref

def cleanup_callback(ref):
    """Called when referent is about to be finalized."""
    print(f"Object referenced by {ref} is being collected")

class Resource:
    pass

obj = Resource()
ref = weakref.ref(obj, cleanup_callback)

del obj  # Callback is invoked
# Output: Object referenced by <weakref at ...> is being collected
```

### weakref.proxy() - Transparent Access

```python
import weakref

class Data:
    def __init__(self, value):
        self.value = value
    def process(self):
        return self.value * 2

obj = Data(21)
proxy = weakref.proxy(obj)

# Use proxy like the original object (no need to call it)
print(proxy.value)      # 21
print(proxy.process())  # 42

del obj
# proxy.value  # Raises ReferenceError: weakly-referenced object no longer exists
```

### WeakValueDictionary - Cache Pattern

```python
import weakref

class ImageData:
    def __init__(self, path):
        self.path = path
        self.data = f"pixels_from_{path}"  # Simulated large data

# Cache that doesn't prevent garbage collection
image_cache = weakref.WeakValueDictionary()

def get_image(path):
    """Get image from cache or load it."""
    if path in image_cache:
        print(f"Cache hit: {path}")
        return image_cache[path]
    
    print(f"Loading: {path}")
    img = ImageData(path)
    image_cache[path] = img
    return img

# Usage
img1 = get_image("photo.jpg")  # Loading: photo.jpg
img2 = get_image("photo.jpg")  # Cache hit: photo.jpg

# When all strong references are gone, cache entry is auto-removed
del img1, img2
# image_cache is now empty (entry was automatically removed)
```

### WeakKeyDictionary - Associate Data with Objects

```python
import weakref

# Associate metadata with objects without modifying them
metadata = weakref.WeakKeyDictionary()

class User:
    def __init__(self, name):
        self.name = name

user = User("Alice")
metadata[user] = {"login_count": 5, "last_seen": "2026-01-25"}

print(metadata[user])  # {'login_count': 5, 'last_seen': '2026-01-25'}

del user
# Metadata entry is automatically removed when user is garbage collected
```

### WeakSet - Collection of Weak References

```python
import weakref

class Observer:
    def __init__(self, name):
        self.name = name
    def notify(self, message):
        print(f"{self.name} received: {message}")

class Subject:
    def __init__(self):
        self._observers = weakref.WeakSet()
    
    def attach(self, observer):
        self._observers.add(observer)
    
    def notify_all(self, message):
        for observer in self._observers:
            observer.notify(message)

subject = Subject()
obs1 = Observer("Observer1")
obs2 = Observer("Observer2")

subject.attach(obs1)
subject.attach(obs2)

subject.notify_all("Hello!")
# Observer1 received: Hello!
# Observer2 received: Hello!

del obs1
subject.notify_all("Goodbye!")
# Only Observer2 received: Goodbye!
```

### WeakMethod - Weak Reference to Bound Methods

```python
import weakref

class Handler:
    def handle(self):
        print("Handling event")

handler = Handler()

# Regular weak reference to bound method FAILS
regular_ref = weakref.ref(handler.handle)
print(regular_ref())  # None (bound method was immediately garbage collected!)

# WeakMethod works correctly
weak_method = weakref.WeakMethod(handler.handle)
print(weak_method())  # <bound method Handler.handle of ...>

weak_method()()  # Prints: Handling event

del handler
print(weak_method())  # None
```

### finalize() - Reliable Cleanup

```python
import weakref
import tempfile
import shutil

class TempDirectory:
    """Temporary directory that cleans itself up."""
    
    def __init__(self):
        self.path = tempfile.mkdtemp()
        # Register cleanup - runs when object is GC'd OR at program exit
        self._finalizer = weakref.finalize(
            self, 
            shutil.rmtree, 
            self.path
        )
    
    def remove(self):
        """Manual cleanup."""
        self._finalizer()
    
    @property
    def removed(self):
        return not self._finalizer.alive

# Usage
temp = TempDirectory()
print(temp.path)  # /tmp/xyz123

# Option 1: Manual removal
temp.remove()

# Option 2: Automatic on garbage collection
del temp

# Option 3: Automatic on program exit (if still alive)
```

### finalize() Control Methods

```python
import weakref

def cleanup(name, value):
    print(f"Cleaning up {name} with value {value}")
    return f"cleaned_{name}"

class Resource:
    pass

obj = Resource()
finalizer = weakref.finalize(obj, cleanup, "resource", 42)

# Check if finalizer is still active
print(finalizer.alive)  # True

# Peek at the finalizer's state without calling it
print(finalizer.peek())  # (<Resource object>, <function cleanup>, ('resource', 42), {})

# Detach and get arguments (disables finalizer)
obj_ref, func, args, kwargs = finalizer.detach()
print(finalizer.alive)  # False

# Manually call with detached args
result = func(*args, **kwargs)  # Cleaning up resource with value 42
print(result)  # cleaned_resource

# Control exit behavior
finalizer2 = weakref.finalize(obj, cleanup, "other", 0)
finalizer2.atexit = False  # Won't be called at program exit
```

### Objects That Support Weak References

```python
import weakref

# SUPPORTED: class instances, functions, methods, sets, generators, etc.
class MyClass: pass
weakref.ref(MyClass())  # OK

def my_func(): pass
weakref.ref(my_func)  # OK

# NOT SUPPORTED: built-in types (list, dict, tuple, int, str)
# weakref.ref([1, 2, 3])  # TypeError!
# weakref.ref({"a": 1})   # TypeError!

# WORKAROUND: Subclass built-in types
class WeakableDict(dict):
    pass

d = WeakableDict(a=1, b=2)
weakref.ref(d)  # OK

# With __slots__, explicitly include __weakref__
class Slotted:
    __slots__ = ('value', '__weakref__')  # Must include __weakref__
    def __init__(self, value):
        self.value = value

weakref.ref(Slotted(42))  # OK
```

---

## gc Module - Garbage Collector Interface

### Basic GC Control

```python
import gc

# Check if GC is enabled
print(gc.isenabled())  # True

# Disable automatic collection
gc.disable()

# Enable automatic collection
gc.enable()

# Force a collection
collected = gc.collect()  # Full collection (generation 2)
print(f"Collected {collected} objects")

# Collect specific generation
gc.collect(0)  # Young generation only
gc.collect(1)  # Young + increment of old (3.14+)
gc.collect(2)  # Full collection
```

### Understanding Generations (Python 3.14+)

```python
import gc

# Python 3.14 simplified to 2 generations:
# - Young generation (gen 0): Newly created objects
# - Old generation (gen 2): Objects that survived collection
# - Gen 1 is removed in 3.14

# Get collection thresholds
thresholds = gc.get_threshold()
print(thresholds)  # (700, 10, 10) - default

# threshold0: Allocations - deallocations before young gen collection
# threshold1: Controls fraction of old gen scanned (higher = slower)
# threshold2: Ignored in 3.14+

# Set thresholds
gc.set_threshold(1000, 15, 15)  # Less frequent collection

# Get current counts
counts = gc.get_count()
print(counts)  # (count0, count1, count2)
```

### Debugging Memory Leaks

```python
import gc

# Enable debugging
gc.set_debug(gc.DEBUG_LEAK)  # Full leak debugging

# Debug flag options:
# gc.DEBUG_STATS - Print collection statistics
# gc.DEBUG_COLLECTABLE - Print collectable objects
# gc.DEBUG_UNCOLLECTABLE - Print uncollectable objects
# gc.DEBUG_SAVEALL - Save all unreachable to gc.garbage
# gc.DEBUG_LEAK - Combination for leak detection

# Get current debug flags
print(gc.get_debug())

# After debugging, disable
gc.set_debug(0)

# Check for uncollectable objects
print(gc.garbage)  # List of uncollectable objects (usually empty in 3.4+)
```

### Tracking Object References

```python
import gc

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

# Create reference cycle
a = Node(1)
b = Node(2)
a.next = b
b.next = a  # Cycle!

# Find what references an object
referrers = gc.get_referrers(a)
print(f"Objects referencing 'a': {len(referrers)}")

# Find what an object references
referents = gc.get_referents(a)
print(f"Objects referenced by 'a': {referents}")

# Check if object is tracked by GC
print(gc.is_tracked(a))    # True (container)
print(gc.is_tracked(42))   # False (atomic)
print(gc.is_tracked("x"))  # False (atomic)
print(gc.is_tracked([]))   # True (container)
print(gc.is_tracked({}))   # False (empty dict optimization)
```

### GC Callbacks for Monitoring

```python
import gc
import time

def gc_callback(phase, info):
    """Monitor garbage collection events."""
    if phase == "start":
        print(f"GC starting: generation {info['generation']}")
    else:  # phase == "stop"
        print(f"GC finished: collected {info['collected']}, "
              f"uncollectable {info['uncollectable']}")

# Register callback
gc.callbacks.append(gc_callback)

# Force collection to see callback
gc.collect()
# Output:
# GC starting: generation 2
# GC finished: collected 0, uncollectable 0

# Remove callback when done
gc.callbacks.remove(gc_callback)
```

### freeze() for Fork Optimization

```python
import gc
import os

def fork_optimized():
    """Optimize memory for forking processes."""
    
    # Disable GC before fork to prevent collection during fork
    gc.disable()
    
    # Collect and freeze all objects
    gc.collect()
    gc.freeze()
    
    # Check frozen count
    print(f"Frozen objects: {gc.get_freeze_count()}")
    
    pid = os.fork()
    
    if pid == 0:
        # Child process
        gc.enable()  # Re-enable GC in child
        # Child has copy-on-write sharing of frozen objects
        print("Child process running")
    else:
        # Parent process
        gc.enable()
        os.waitpid(pid, 0)
        
        # Optionally unfreeze later
        gc.unfreeze()

# Note: Only works on Unix-like systems with fork()
```

### Performance Optimization Pattern

```python
import gc

def optimized_batch_processing(items):
    """Process large batches with GC optimization."""
    
    # Disable GC during intensive processing
    gc.disable()
    
    try:
        results = []
        for i, item in enumerate(items):
            result = process(item)
            results.append(result)
            
            # Manual collection every 1000 items
            if i % 1000 == 0:
                gc.collect(0)  # Young generation only (fast)
        
        return results
    finally:
        # Re-enable and do full collection
        gc.enable()
        gc.collect()

def process(item):
    return item * 2
```

### Web Server GC Optimization (from Close.com research)

```python
import gc

def optimize_for_web_traffic():
    """
    Optimize GC for web request handling.
    Can improve p95 latency by 80-100ms.
    """
    # Do full collection and freeze before starting server
    gc.collect(2)
    gc.freeze()
    
    # Increase gen0 threshold from 700 to 50000
    # This reduces GC frequency during request handling
    _, gen1, gen2 = gc.get_threshold()
    gc.set_threshold(50_000, gen1, gen2)

# Call during application initialization, before starting server
optimize_for_web_traffic()
```

### Detecting Finalized Objects

```python
import gc

class Lazarus:
    """Object that resurrects itself."""
    instance = None
    
    def __del__(self):
        # Resurrect by creating new reference
        Lazarus.instance = self
        print("Lazarus resurrected!")

obj = Lazarus()

# Check if finalized
print(gc.is_finalized(obj))  # False

del obj
gc.collect()
# Output: Lazarus resurrected!

# The object is now finalized but still exists
print(gc.is_finalized(Lazarus.instance))  # True
```

### Get All Tracked Objects

```python
import gc

# Get all objects tracked by GC
all_objects = gc.get_objects()
print(f"Total tracked objects: {len(all_objects)}")

# Get objects by generation (3.14+)
young = gc.get_objects(generation=0)
old = gc.get_objects(generation=2)

print(f"Young generation: {len(young)}")
print(f"Old generation: {len(old)}")

# Get collection statistics
stats = gc.get_stats()
for i, gen_stats in enumerate(stats):
    print(f"Generation {i}: {gen_stats}")
# Example output:
# Generation 0: {'collections': 50, 'collected': 1234, 'uncollectable': 0}
```

---

## Production Best Practices

### When to Use Weak References

1. **Caches**: Prevent cache from keeping objects alive unnecessarily
2. **Observer patterns**: Observers shouldn't prevent subjects from being collected
3. **Circular reference breaking**: Especially in parent-child relationships
4. **Resource cleanup**: Use `finalize()` for deterministic cleanup

### When to Control GC

1. **Batch processing**: Disable during intensive loops, manual collect periodically
2. **Real-time systems**: Tune thresholds to reduce pause times
3. **Fork-based multiprocessing**: Use `freeze()` to optimize copy-on-write
4. **Debugging memory leaks**: Use debug flags and `get_referrers()`

### Common Pitfalls

```python
import weakref

# PITFALL 1: Weak reference to temporary object
ref = weakref.ref(SomeClass())  # Object immediately collected!
print(ref())  # None

# PITFALL 2: Bound methods are ephemeral
class C:
    def method(self): pass
c = C()
ref = weakref.ref(c.method)  # Fails! Use WeakMethod instead

# PITFALL 3: finalize callback referencing the object
obj = SomeClass()
# BAD: callback has reference to obj via closure
weakref.finalize(obj, lambda: cleanup(obj))  # obj never collected!
# GOOD: pass only what's needed
weakref.finalize(obj, cleanup, obj.resource_id)
```

---

## Version Compatibility

| Feature | Minimum Python |
|---------|---------------|
| weakref.ref | 2.1+ |
| WeakKeyDictionary | 2.1+ |
| WeakValueDictionary | 2.1+ |
| WeakSet | 2.7+ |
| WeakMethod | 3.4+ |
| finalize | 3.4+ |
| gc.freeze/unfreeze | 3.7+ |
| gc.is_finalized | 3.9+ |
| 2-generation GC | 3.14+ |

---

*Research conducted: January 2026*
*Sources: docs.python.org/3/library/weakref.html, docs.python.org/3/library/gc.html*
