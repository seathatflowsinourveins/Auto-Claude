# Python ast Module - Production Patterns (January 2026)

## Overview

The `ast` module parses Python source code into Abstract Syntax Trees for analysis, transformation, and code generation. Essential for linters, formatters, refactoring tools, and metaprogramming.

**Key Principle**: AST represents code structure, not bytecode - changes between Python versions.

## Core Functions

### parse() - Parse Source to AST

```python
import ast

# Parse source string
source = '''
def greet(name):
    return f"Hello, {name}!"
'''

tree = ast.parse(source)  # Returns ast.Module

# Parse with mode
tree = ast.parse(source, mode='exec')   # Module (default) - statements
tree = ast.parse('x + 1', mode='eval')  # Expression - single expression
tree = ast.parse('x = 1', mode='single')  # Interactive - single statement

# Parse with filename (for error messages)
tree = ast.parse(source, filename='example.py')

# Type comments (Python 3.8+)
tree = ast.parse(source, type_comments=True)

# Feature version (Python 3.8+)
tree = ast.parse(source, feature_version=(3, 10))
```

### compile() - AST to Code Object

```python
import ast

source = 'x = 1 + 2'
tree = ast.parse(source)

# Compile AST to code object
code = compile(tree, filename='<ast>', mode='exec')

# Execute the code
exec(code)
print(x)  # 3

# For expressions
expr_tree = ast.parse('1 + 2', mode='eval')
code = compile(expr_tree, '<ast>', 'eval')
result = eval(code)  # 3
```

### dump() - Pretty Print AST

```python
import ast

tree = ast.parse('x = 1 + 2')

# Basic dump
print(ast.dump(tree))
# Module(body=[Assign(targets=[Name(id='x', ctx=Store())], ...)])

# With indentation (Python 3.9+)
print(ast.dump(tree, indent=2))

# Include attributes (line/col numbers)
print(ast.dump(tree, indent=2, include_attributes=True))

# Annotate fields (Python 3.10+)
print(ast.dump(tree, indent=2, annotate_fields=True))
```

### unparse() - AST to Source (Python 3.9+)

```python
import ast

# Parse, modify, and unparse
tree = ast.parse('x = 1 + 2')
source = ast.unparse(tree)
print(source)  # 'x = 1 + 2'

# Useful for code generation
def make_function(name, body_expr):
    func = ast.FunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[],
            kw_defaults=[], defaults=[]
        ),
        body=[ast.Return(value=body_expr)],
        decorator_list=[],
        returns=None
    )
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[])))

print(make_function('get_value', ast.Constant(value=42)))
# 'def get_value():\n    return 42'
```

### literal_eval() - Safe Evaluation

```python
import ast

# Safely evaluate literal expressions
result = ast.literal_eval('[1, 2, 3]')  # [1, 2, 3]
result = ast.literal_eval('{"a": 1}')   # {'a': 1}
result = ast.literal_eval('(1, 2)')     # (1, 2)
result = ast.literal_eval('True')       # True
result = ast.literal_eval('None')       # None

# Supports: strings, bytes, numbers, tuples, lists, dicts, sets, booleans, None

# Raises ValueError for non-literals
try:
    ast.literal_eval('1 + 2')  # ValueError - not a literal
except ValueError:
    pass

try:
    ast.literal_eval('os.system("rm -rf /")')  # ValueError - function call
except ValueError:
    pass
```

### walk() - Iterate All Nodes

```python
import ast

tree = ast.parse('''
def greet(name):
    print(f"Hello, {name}!")
''')

# Walk all nodes (no guaranteed order)
for node in ast.walk(tree):
    print(type(node).__name__)

# Count node types
from collections import Counter
node_types = Counter(type(n).__name__ for n in ast.walk(tree))
print(node_types)
```

### fix_missing_locations() - Add Line Numbers

```python
import ast

# When building AST nodes programmatically, add locations
expr = ast.BinOp(
    left=ast.Constant(value=1),
    op=ast.Add(),
    right=ast.Constant(value=2)
)

module = ast.Module(body=[ast.Expr(value=expr)], type_ignores=[])

# Fix missing line/column info (required for compile)
ast.fix_missing_locations(module)

# Now can compile
code = compile(module, '<generated>', 'exec')
```

### get_source_segment() (Python 3.8+)

```python
import ast

source = '''
def example():
    x = 1 + 2
    return x
'''

tree = ast.parse(source)

# Get source text for any node
for node in ast.walk(tree):
    if isinstance(node, ast.BinOp):
        segment = ast.get_source_segment(source, node)
        print(f"BinOp source: {segment}")  # "1 + 2"
```

## NodeVisitor Pattern

```python
import ast

class FunctionCounter(ast.NodeVisitor):
    """Count functions and async functions."""
    
    def __init__(self):
        self.sync_count = 0
        self.async_count = 0
    
    def visit_FunctionDef(self, node):
        self.sync_count += 1
        self.generic_visit(node)  # Visit children
    
    def visit_AsyncFunctionDef(self, node):
        self.async_count += 1
        self.generic_visit(node)

# Usage
tree = ast.parse('''
def sync_func(): pass
async def async_func(): pass
def another(): pass
''')

counter = FunctionCounter()
counter.visit(tree)
print(f"Sync: {counter.sync_count}, Async: {counter.async_count}")
```

## NodeTransformer Pattern

```python
import ast

class StringDoubler(ast.NodeTransformer):
    """Double all string constants."""
    
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return ast.Constant(value=node.value * 2)
        return node

# Usage
tree = ast.parse('x = "hello"')
transformer = StringDoubler()
new_tree = transformer.visit(tree)
ast.fix_missing_locations(new_tree)

print(ast.unparse(new_tree))  # x = 'hellohello'
```

## Common AST Node Types

### Statements (stmt)
```python
ast.FunctionDef      # def func(): ...
ast.AsyncFunctionDef # async def func(): ...
ast.ClassDef         # class Foo: ...
ast.Return           # return value
ast.Assign           # x = value
ast.AugAssign        # x += value
ast.AnnAssign        # x: int = value
ast.For              # for x in iter: ...
ast.AsyncFor         # async for x in iter: ...
ast.While            # while cond: ...
ast.If               # if cond: ...
ast.With             # with ctx: ...
ast.AsyncWith        # async with ctx: ...
ast.Raise            # raise exc
ast.Try              # try: ... except: ...
ast.Assert           # assert cond
ast.Import           # import mod
ast.ImportFrom       # from mod import name
ast.Global           # global x
ast.Nonlocal         # nonlocal x
ast.Expr             # expression statement
ast.Pass             # pass
ast.Break            # break
ast.Continue         # continue
ast.Match            # match subject: ... (Python 3.10+)
```

### Expressions (expr)
```python
ast.BoolOp           # and/or operations
ast.NamedExpr        # walrus := (Python 3.8+)
ast.BinOp            # binary operations (+, -, *, /, etc.)
ast.UnaryOp          # unary operations (not, -, +, ~)
ast.Lambda           # lambda x: x
ast.IfExp            # x if cond else y
ast.Dict             # {k: v}
ast.Set              # {x, y}
ast.ListComp         # [x for x in iter]
ast.SetComp          # {x for x in iter}
ast.DictComp         # {k: v for k, v in iter}
ast.GeneratorExp     # (x for x in iter)
ast.Await            # await expr
ast.Yield            # yield expr
ast.YieldFrom        # yield from expr
ast.Compare          # comparisons (==, <, in, etc.)
ast.Call             # func(args)
ast.FormattedValue   # f-string value {x}
ast.JoinedStr        # f-string
ast.Constant         # literals (1, "str", None, True)
ast.Attribute        # obj.attr
ast.Subscript        # obj[key]
ast.Starred          # *args unpacking
ast.Name             # variable names
ast.List             # [a, b]
ast.Tuple            # (a, b)
ast.Slice            # a:b:c
```

## Production Patterns

### Pattern 1: Code Complexity Analyzer

```python
import ast
from dataclasses import dataclass

@dataclass
class Complexity:
    functions: int = 0
    classes: int = 0
    loops: int = 0
    branches: int = 0
    try_blocks: int = 0

class ComplexityAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.metrics = Complexity()
    
    def visit_FunctionDef(self, node):
        self.metrics.functions += 1
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node):
        self.metrics.functions += 1
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        self.metrics.classes += 1
        self.generic_visit(node)
    
    def visit_For(self, node):
        self.metrics.loops += 1
        self.generic_visit(node)
    
    def visit_While(self, node):
        self.metrics.loops += 1
        self.generic_visit(node)
    
    def visit_If(self, node):
        self.metrics.branches += 1
        self.generic_visit(node)
    
    def visit_Try(self, node):
        self.metrics.try_blocks += 1
        self.generic_visit(node)

def analyze_file(filepath: str) -> Complexity:
    with open(filepath) as f:
        tree = ast.parse(f.read(), filename=filepath)
    analyzer = ComplexityAnalyzer()
    analyzer.visit(tree)
    return analyzer.metrics
```

### Pattern 2: Import Collector

```python
import ast
from typing import Set, Tuple

def collect_imports(source: str) -> Tuple[Set[str], Set[str]]:
    """Return (modules, from_imports)."""
    tree = ast.parse(source)
    
    modules = set()
    from_imports = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                from_imports.add(node.module.split('.')[0])
    
    return modules, from_imports
```

### Pattern 3: Dead Code Detector

```python
import ast
from typing import Set

class DefinitionCollector(ast.NodeVisitor):
    """Collect all defined names."""
    
    def __init__(self):
        self.defined: Set[str] = set()
        self.used: Set[str] = set()
    
    def visit_FunctionDef(self, node):
        self.defined.add(node.name)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        self.defined.add(node.name)
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.defined.add(target.id)
        self.generic_visit(node)
    
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used.add(node.id)
    
    def find_unused(self) -> Set[str]:
        # Exclude private/dunder names
        return {
            name for name in self.defined - self.used
            if not name.startswith('_')
        }
```

### Pattern 4: Code Transformer (Add Logging)

```python
import ast

class AddLogging(ast.NodeTransformer):
    """Add print statements at function entry."""
    
    def visit_FunctionDef(self, node):
        # Create print statement
        log_stmt = ast.Expr(
            value=ast.Call(
                func=ast.Name(id='print', ctx=ast.Load()),
                args=[ast.Constant(value=f'Entering {node.name}')],
                keywords=[]
            )
        )
        
        # Insert at beginning of function body
        node.body.insert(0, log_stmt)
        
        # Continue visiting children
        self.generic_visit(node)
        return node

def add_logging(source: str) -> str:
    tree = ast.parse(source)
    transformer = AddLogging()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)
```

### Pattern 5: Safe Config Parser

```python
import ast
from typing import Any, Dict

def parse_config(config_str: str) -> Dict[str, Any]:
    """Safely parse Python-like config files."""
    # Only allow simple assignments
    tree = ast.parse(config_str)
    
    config = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            raise ValueError(f"Only assignments allowed, got {type(node).__name__}")
        
        if len(node.targets) != 1:
            raise ValueError("Multiple assignment targets not allowed")
        
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            raise ValueError("Only simple name assignments allowed")
        
        # Use literal_eval for safe value parsing
        try:
            value = ast.literal_eval(ast.unparse(node.value))
        except ValueError:
            raise ValueError(f"Unsafe value for {target.id}")
        
        config[target.id] = value
    
    return config

# Usage
config = parse_config('''
DEBUG = True
HOST = "localhost"
PORT = 8080
ALLOWED_IPS = ["127.0.0.1", "192.168.1.1"]
''')
```

## Version History

| Version | Feature |
|---------|---------|
| 3.8 | type_comments, feature_version, get_source_segment(), NamedExpr |
| 3.9 | unparse(), indent param in dump() |
| 3.10 | Match statement nodes, annotate_fields in dump() |
| 3.11 | TryStar for except* |
| 3.12 | TypeAlias, TypeVar, ParamSpec, TypeVarTuple nodes |
| 3.14 | Interpolation, Template nodes for t-strings |

## See Also

- `dis` - Bytecode disassembly (lower level)
- `inspect` - Get source from live objects
- `tokenize` - Lexical analysis
- `symtable` - Symbol table access
- `compile()` - Compile AST to code
