# Python dis Module - Production Patterns (January 2026)

## Overview

The `dis` module disassembles CPython bytecode for debugging, optimization analysis, and understanding Python internals.

**Important**: Bytecode is CPython implementation detail - may change between Python versions.

## Core Functions

### dis() - Main Disassembly Function

```python
import dis

def example():
    x = 1
    return x + 2

# Disassemble function
dis.dis(example)
# Output:
#   2           RESUME                   0
#   3           LOAD_CONST               1 (1)
#               STORE_FAST               0 (x)
#   4           LOAD_FAST                0 (x)
#               LOAD_CONST               2 (2)
#               BINARY_OP                0 (+)
#               RETURN_VALUE

# Disassemble class (all methods)
dis.dis(MyClass)

# Disassemble module (all functions)
dis.dis(my_module)

# Disassemble source string
dis.dis('x = 1 + 2')

# Control recursion depth
dis.dis(func, depth=0)  # No recursion into nested code objects

# Show cache entries (Python 3.11+)
dis.dis(func, show_caches=True)

# Show specialized/adaptive bytecode (Python 3.11+)
dis.dis(func, adaptive=True)

# Show instruction offsets (Python 3.13+)
dis.dis(func, show_offsets=True)

# Show source positions (Python 3.14+)
dis.dis(func, show_positions=True)
```

### code_info() and show_code()

```python
import dis

def example(a, b, *args, **kwargs):
    x = a + b
    return x

# Get detailed code object info as string
info = dis.code_info(example)
print(info)
# Name:              example
# Argument count:    2
# Positional-only:   0
# Kw-only arguments: 0
# Number of locals:  4
# Stack size:        2
# Flags:             OPTIMIZED, NEWLOCALS, VARARGS, VARKEYWORDS
# Constants:         (None,)
# Names:             ()
# Variable names:    ('a', 'b', 'args', 'kwargs', 'x')

# Print directly to stdout (or file)
dis.show_code(example)
dis.show_code(example, file=sys.stderr)
```

### get_instructions() - Iterate Over Instructions

```python
import dis

def example():
    return len([1, 2, 3])

# Generator yielding Instruction namedtuples
for instr in dis.get_instructions(example):
    print(f"{instr.opname:20} {instr.argrepr}")
    
# Output:
# RESUME               0
# LOAD_GLOBAL          len + NULL
# BUILD_LIST           3
# CALL                 1
# RETURN_VALUE
```

### distb() - Disassemble Traceback

```python
import dis

try:
    1 / 0
except:
    dis.distb()  # Shows bytecode with --> marking exception location
```

### Bytecode Class (Python 3.4+)

```python
import dis

def example():
    return 42

# Wrap code in Bytecode object
bc = dis.Bytecode(example)

# Iterate over instructions
for instr in bc:
    print(instr.opname, instr.arg, instr.argval)

# Get formatted disassembly
print(bc.dis())

# Get code info
print(bc.info())

# Access code object
print(bc.codeobj)
print(bc.first_line)

# From traceback
try:
    raise ValueError()
except:
    import sys
    bc = dis.Bytecode.from_traceback(sys.exc_info()[2])
```

## Instruction Namedtuple

```python
from dis import Instruction

# Fields available on each Instruction:
instr.opcode        # Numeric opcode (int)
instr.opname        # Human-readable name (str)
instr.arg           # Numeric argument (int or None)
instr.argval        # Resolved argument value
instr.argrepr       # Human-readable arg description (str)
instr.offset        # Byte offset in bytecode
instr.start_offset  # Including EXTENDED_ARG (Python 3.13+)
instr.line_number   # Source line number (Python 3.13+)
instr.starts_line   # True if starts new source line (Python 3.13+)
instr.is_jump_target  # True if jump target
instr.jump_target   # Target offset for jumps (Python 3.13+)
instr.positions     # Source positions (Python 3.11+)
instr.cache_info    # Cache entry info (Python 3.13+)

# Python 3.11+ baseopcode/baseopname for specialized instructions
instr.baseopcode    # Base opcode before specialization
instr.baseopname    # Base name before specialization
```

## Command-Line Interface

```bash
# Disassemble a Python file
python -m dis script.py

# Disassemble from stdin
echo "x = 1 + 2" | python -m dis

# Show options (Python 3.13+)
python -m dis -h

# Show inline caches (Python 3.13+)
python -m dis -C script.py

# Show instruction offsets (Python 3.13+)
python -m dis -O script.py

# Show source positions (Python 3.14+)
python -m dis -P script.py

# Show specialized bytecode (Python 3.14+)
python -m dis -S script.py
```

## Opcode Collections

```python
import dis

# All opcode names (indexed by opcode number)
dis.opname[100]  # 'LOAD_CONST' or similar

# Opcode name to number mapping
dis.opmap['LOAD_FAST']  # Returns opcode number

# Comparison operators
dis.cmp_op  # ('<', '<=', '==', '!=', '>', '>=')

# Opcodes that use their argument (Python 3.12+)
dis.hasarg  # Sequence of opcodes

# Opcodes accessing constants
dis.hasconst  # [LOAD_CONST, ...]

# Opcodes accessing names
dis.hasname  # [LOAD_NAME, STORE_NAME, ...]

# Opcodes accessing locals
dis.haslocal  # [LOAD_FAST, STORE_FAST, ...]

# Opcodes accessing free/closure variables
dis.hasfree  # [LOAD_DEREF, STORE_DEREF, ...]

# Jump opcodes (Python 3.13+)
dis.hasjump  # All jump instructions

# Exception handler opcodes (Python 3.12+)
dis.hasexc  # Exception handling opcodes

# Comparison opcodes
dis.hascompare  # Boolean comparison opcodes
```

## Utility Functions

```python
import dis

# Find all line starts in code
for offset, lineno in dis.findlinestarts(code_obj):
    print(f"Line {lineno} starts at offset {offset}")

# Find all jump targets
labels = dis.findlabels(code_obj.co_code)

# Calculate stack effect of opcode
effect = dis.stack_effect(opcode, oparg)
effect = dis.stack_effect(opcode, oparg, jump=True)   # If jumping
effect = dis.stack_effect(opcode, oparg, jump=False)  # If not jumping
```

## Production Patterns

### Pattern 1: Bytecode Comparison Tool

```python
import dis
from typing import Callable

def compare_bytecode(func1: Callable, func2: Callable) -> dict:
    """Compare bytecode of two functions."""
    bc1 = list(dis.get_instructions(func1))
    bc2 = list(dis.get_instructions(func2))
    
    ops1 = [i.opname for i in bc1]
    ops2 = [i.opname for i in bc2]
    
    return {
        'func1_ops': len(ops1),
        'func2_ops': len(ops2),
        'identical': ops1 == ops2,
        'diff_count': sum(1 for a, b in zip(ops1, ops2) if a != b),
    }

# Compare list comprehension vs generator expression
def with_list():
    return sum([x * 2 for x in range(100)])

def with_gen():
    return sum(x * 2 for x in range(100))

print(compare_bytecode(with_list, with_gen))
```

### Pattern 2: Instruction Counter

```python
import dis
from collections import Counter
from typing import Callable

def count_instructions(func: Callable) -> Counter:
    """Count instruction types in a function."""
    instructions = dis.get_instructions(func)
    return Counter(instr.opname for instr in instructions)

def analyze_complexity(func: Callable) -> dict:
    """Analyze bytecode complexity."""
    counts = count_instructions(func)
    
    return {
        'total_instructions': sum(counts.values()),
        'unique_opcodes': len(counts),
        'calls': counts.get('CALL', 0),
        'loads': sum(v for k, v in counts.items() if k.startswith('LOAD_')),
        'stores': sum(v for k, v in counts.items() if k.startswith('STORE_')),
        'jumps': sum(v for k, v in counts.items() if 'JUMP' in k),
    }
```

### Pattern 3: Find Function Calls

```python
import dis
from typing import Callable, List

def find_function_calls(func: Callable) -> List[str]:
    """Extract all function names called within a function."""
    calls = []
    instructions = list(dis.get_instructions(func))
    
    for i, instr in enumerate(instructions):
        # LOAD_GLOBAL followed by CALL indicates function call
        if instr.opname == 'LOAD_GLOBAL' and instr.argval:
            # Check if followed by CALL
            for next_instr in instructions[i+1:i+5]:
                if next_instr.opname == 'CALL':
                    calls.append(instr.argval)
                    break
                if next_instr.opname.startswith('LOAD_'):
                    continue
                break
    
    return calls
```

### Pattern 4: Bytecode Diff for Debugging

```python
import dis
import difflib

def bytecode_diff(func1, func2):
    """Show diff between two function's bytecode."""
    lines1 = dis.Bytecode(func1).dis().splitlines()
    lines2 = dis.Bytecode(func2).dis().splitlines()
    
    diff = difflib.unified_diff(
        lines1, lines2,
        fromfile=func1.__name__,
        tofile=func2.__name__,
        lineterm=''
    )
    return '\n'.join(diff)
```

### Pattern 5: Detect Potential Performance Issues

```python
import dis
from typing import Callable, List, Tuple

def detect_issues(func: Callable) -> List[Tuple[str, str]]:
    """Detect potential bytecode-level performance issues."""
    issues = []
    instructions = list(dis.get_instructions(func))
    
    for i, instr in enumerate(instructions):
        # Repeated LOAD_GLOBAL for same name
        if instr.opname == 'LOAD_GLOBAL':
            same_loads = sum(
                1 for other in instructions 
                if other.opname == 'LOAD_GLOBAL' 
                and other.argval == instr.argval
            )
            if same_loads > 3:
                issues.append((
                    'repeated_global',
                    f"'{instr.argval}' loaded {same_loads} times - consider local alias"
                ))
        
        # BINARY_OP in loop (look for backward jump)
        if instr.opname == 'BINARY_OP':
            for other in instructions[i:]:
                if other.opname == 'JUMP_BACKWARD':
                    issues.append((
                        'loop_operation',
                        f"Binary operation at offset {instr.offset} inside loop"
                    ))
                    break
    
    return list(set(issues))  # Deduplicate
```

## Version History

| Version | Feature |
|---------|---------|
| 3.4 | Bytecode class, get_instructions(), stack_effect() |
| 3.6 | 2-byte instructions (word-aligned) |
| 3.10 | Jump offsets instead of byte offsets |
| 3.11 | CACHE instructions, show_caches, adaptive params |
| 3.12 | hasexc, hasarg collections |
| 3.13 | show_offsets, hasjump, CLI options -C/-O, Instruction fields |
| 3.14 | show_positions, CLI -P/-S options |

## See Also

- `compile()` - Compile source to code object
- `exec()` / `eval()` - Execute code objects
- `types.CodeType` - Code object type
- `inspect` - Higher-level introspection
- `py_compile` - Compile to .pyc
