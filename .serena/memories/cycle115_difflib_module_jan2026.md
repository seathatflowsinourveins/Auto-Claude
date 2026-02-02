# Python difflib Module - Production Patterns (January 2026)

## Overview
The `difflib` module provides classes and functions for comparing sequences. Essential for showing file differences, fuzzy matching, config comparison, and generating human-readable deltas in various formats (unified, context, HTML).

## Core Functions

### get_close_matches() - Fuzzy String Matching
```python
from difflib import get_close_matches

# Find similar strings
get_close_matches('appel', ['ape', 'apple', 'peach', 'puppy'])
# ['apple', 'ape']

# Typo correction / autocomplete
import keyword
get_close_matches('whle', keyword.kwlist)  # ['while']

# Parameters
get_close_matches(
    word,           # Target string
    possibilities,  # List of candidates
    n=3,            # Max matches to return
    cutoff=0.6      # Minimum similarity (0-1)
)
```

### unified_diff() - Git-Style Diffs
```python
from difflib import unified_diff

s1 = ['bacon\n', 'eggs\n', 'ham\n']
s2 = ['python\n', 'eggy\n', 'hamster\n']

diff = unified_diff(s1, s2, 
    fromfile='before.py', 
    tofile='after.py',
    fromfiledate='2026-01-25',
    tofiledate='2026-01-25',
    n=3,           # Context lines
    lineterm='\n'  # Line terminator
)
print(''.join(diff))
# --- before.py 2026-01-25
# +++ after.py 2026-01-25
# @@ -1,3 +1,3 @@
# -bacon
# -eggs
# -ham
# +python
# +eggy
# +hamster
```

### context_diff() - Traditional Context Diff
```python
from difflib import context_diff

diff = context_diff(s1, s2, 
    fromfile='before.py', 
    tofile='after.py',
    n=3  # Context lines
)
# *** before.py
# --- after.py
# ***************
# *** 1,3 ****
# ! bacon
# ! eggs
# ! ham
# --- 1,3 ----
# ! python
# ! eggy
# ! hamster
```

### ndiff() - Detailed Character-Level Diff
```python
from difflib import ndiff

a = 'one\ntwo\nthree\n'.splitlines(keepends=True)
b = 'ore\ntree\nemu\n'.splitlines(keepends=True)

diff = ndiff(a, b)
print(''.join(diff))
# - one
# ? ^
# + ore
# ? ^
# - two
# - three
# ? -
# + tree
# + emu
```

### diff_bytes() - Binary/Unknown Encoding
```python
from difflib import diff_bytes, unified_diff

# Compare bytes with unknown encoding
a = [b'line1\n', b'line2\n']
b = [b'line1\n', b'changed\n']

diff = diff_bytes(unified_diff, a, b, 
    fromfile=b'old', tofile=b'new')
# Returns bytes iterator
```

## SequenceMatcher - Core Comparison Engine

### Basic Usage
```python
from difflib import SequenceMatcher

# Compare strings
s = SequenceMatcher(None, "abcd", "bcde")
s.ratio()  # 0.75 (similarity score 0-1)

# With junk filtering (ignore spaces)
s = SequenceMatcher(lambda x: x == " ", 
    "private Thread currentThread;",
    "private volatile Thread currentThread;")
s.ratio()  # 0.866
```

### Key Methods
```python
# Similarity ratio (0-1)
s.ratio()           # Expensive, accurate
s.quick_ratio()     # Upper bound, faster
s.real_quick_ratio() # Rougher upper bound, fastest

# Find longest matching block
s.find_longest_match(0, len(a), 0, len(b))
# Returns Match(a=start_a, b=start_b, size=length)

# All matching blocks
s.get_matching_blocks()
# [(0, 0, 2), (3, 2, 2), (5, 4, 0)]  # Last is dummy

# Edit operations (how to transform a→b)
s.get_opcodes()
# [('delete', 0, 1, 0, 0),
#  ('equal', 1, 3, 0, 2),
#  ('replace', 3, 4, 2, 3),
#  ('equal', 4, 6, 3, 5),
#  ('insert', 6, 6, 5, 6)]
```

### Opcode Tags
| Tag | Meaning |
|-----|---------|
| `'replace'` | `a[i1:i2]` → `b[j1:j2]` |
| `'delete'` | Remove `a[i1:i2]` |
| `'insert'` | Insert `b[j1:j2]` at position |
| `'equal'` | Sequences match |

### Junk and Autojunk
```python
# Disable autojunk heuristic (for small sequences)
s = SequenceMatcher(None, a, b, autojunk=False)

# Custom junk function
s = SequenceMatcher(
    isjunk=lambda x: x in " \t",  # Ignore whitespace
    a=seq1, b=seq2
)
```

## HtmlDiff - Visual HTML Comparison

```python
from difflib import HtmlDiff

differ = HtmlDiff(
    tabsize=8,       # Tab width
    wrapcolumn=None, # Wrap at column (None=no wrap)
    linejunk=None,   # Line filter function
    charjunk=None    # Character filter function
)

# Generate complete HTML file
html = differ.make_file(
    fromlines, tolines,
    fromdesc='Original',
    todesc='Modified',
    context=True,    # Show only changes + context
    numlines=5,      # Context lines
    charset='utf-8'  # HTML charset
)

# Generate HTML table only (for embedding)
table = differ.make_table(fromlines, tolines, ...)
```

## Differ - Line-by-Line Comparison

```python
from difflib import Differ

d = Differ(linejunk=None, charjunk=None)
result = list(d.compare(text1, text2))

# Output codes:
# '- ' = unique to text1
# '+ ' = unique to text2
# '  ' = common to both
# '? ' = intraline diff guide
```

## Helper Functions

### restore() - Reconstruct from Diff
```python
from difflib import ndiff, restore

diff = list(ndiff(a, b))

# Reconstruct original sequences
original = list(restore(diff, 1))  # First sequence
modified = list(restore(diff, 2))  # Second sequence
```

### IS_LINE_JUNK / IS_CHARACTER_JUNK
```python
from difflib import IS_LINE_JUNK, IS_CHARACTER_JUNK

IS_LINE_JUNK("   \n")  # True (blank line)
IS_LINE_JUNK("# comment\n")  # True (only has #)
IS_CHARACTER_JUNK(" ")  # True (space)
IS_CHARACTER_JUNK("\t")  # True (tab)
```

## Production Patterns

### Pattern 1: Config Comparison Tool
```python
from difflib import unified_diff
from pathlib import Path
import json

def compare_configs(old_path: Path, new_path: Path) -> str:
    """Generate diff between two config files."""
    old = json.loads(old_path.read_text())
    new = json.loads(new_path.read_text())
    
    # Pretty format for readable diff
    old_lines = json.dumps(old, indent=2, sort_keys=True).splitlines(keepends=True)
    new_lines = json.dumps(new, indent=2, sort_keys=True).splitlines(keepends=True)
    
    diff = unified_diff(
        old_lines, new_lines,
        fromfile=str(old_path),
        tofile=str(new_path),
        lineterm=''
    )
    return '\n'.join(diff)
```

### Pattern 2: Fuzzy Command Matcher
```python
from difflib import get_close_matches

class CommandSuggester:
    """Suggest corrections for mistyped commands."""
    
    def __init__(self, commands: list[str], cutoff: float = 0.6):
        self.commands = commands
        self.cutoff = cutoff
    
    def suggest(self, typo: str) -> list[str]:
        return get_close_matches(typo, self.commands, n=3, cutoff=self.cutoff)
    
    def execute_or_suggest(self, cmd: str) -> str:
        if cmd in self.commands:
            return f"Executing: {cmd}"
        
        suggestions = self.suggest(cmd)
        if suggestions:
            return f"Unknown command '{cmd}'. Did you mean: {', '.join(suggestions)}?"
        return f"Unknown command '{cmd}'"

# Usage
cli = CommandSuggester(['start', 'stop', 'restart', 'status', 'deploy'])
cli.execute_or_suggest('strat')  # "Did you mean: start?"
```

### Pattern 3: Text Similarity Scorer
```python
from difflib import SequenceMatcher

def similarity_score(a: str, b: str, ignore_case: bool = True) -> float:
    """Calculate similarity ratio between two strings."""
    if ignore_case:
        a, b = a.lower(), b.lower()
    return SequenceMatcher(None, a, b).ratio()

def find_duplicates(items: list[str], threshold: float = 0.85) -> list[tuple]:
    """Find near-duplicate strings."""
    duplicates = []
    for i, a in enumerate(items):
        for b in items[i+1:]:
            score = similarity_score(a, b)
            if score >= threshold:
                duplicates.append((a, b, score))
    return duplicates
```

### Pattern 4: Change Highlighter
```python
from difflib import SequenceMatcher

def highlight_changes(old: str, new: str) -> tuple[str, str]:
    """Return strings with change markers."""
    s = SequenceMatcher(None, old, new)
    
    old_parts, new_parts = [], []
    
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'equal':
            old_parts.append(old[i1:i2])
            new_parts.append(new[j1:j2])
        elif tag == 'replace':
            old_parts.append(f"[-{old[i1:i2]}-]")
            new_parts.append(f"[+{new[j1:j2]}+]")
        elif tag == 'delete':
            old_parts.append(f"[-{old[i1:i2]}-]")
        elif tag == 'insert':
            new_parts.append(f"[+{new[j1:j2]}+]")
    
    return ''.join(old_parts), ''.join(new_parts)

# Usage
old, new = highlight_changes("hello world", "hello there")
# old: "hello [-world-]"
# new: "hello [+there+]"
```

### Pattern 5: HTML Diff Report Generator
```python
from difflib import HtmlDiff
from pathlib import Path

def generate_diff_report(
    old_file: Path, 
    new_file: Path, 
    output: Path,
    context: bool = True
) -> None:
    """Generate HTML diff report for code review."""
    old_lines = old_file.read_text().splitlines()
    new_lines = new_file.read_text().splitlines()
    
    differ = HtmlDiff(tabsize=4, wrapcolumn=80)
    
    html = differ.make_file(
        old_lines, new_lines,
        fromdesc=f"Original: {old_file.name}",
        todesc=f"Modified: {new_file.name}",
        context=context,
        numlines=5
    )
    
    output.write_text(html)
```

### Pattern 6: Semantic Version Comparison
```python
from difflib import SequenceMatcher

def compare_versions(old: dict, new: dict, path: str = "") -> list[dict]:
    """Deep compare two dicts, return list of changes."""
    changes = []
    
    all_keys = set(old.keys()) | set(new.keys())
    
    for key in all_keys:
        current_path = f"{path}.{key}" if path else key
        
        if key not in old:
            changes.append({'path': current_path, 'type': 'added', 'value': new[key]})
        elif key not in new:
            changes.append({'path': current_path, 'type': 'removed', 'value': old[key]})
        elif old[key] != new[key]:
            if isinstance(old[key], dict) and isinstance(new[key], dict):
                changes.extend(compare_versions(old[key], new[key], current_path))
            else:
                changes.append({
                    'path': current_path,
                    'type': 'modified',
                    'old': old[key],
                    'new': new[key]
                })
    
    return changes
```

## Performance Tips

1. **Reuse SequenceMatcher** when comparing one sequence against many:
```python
s = SequenceMatcher()
s.set_seq2(reference)  # Set once
for candidate in candidates:
    s.set_seq1(candidate)  # Change only first
    score = s.ratio()
```

2. **Use quick_ratio() for filtering** before expensive ratio():
```python
s = SequenceMatcher(None, a, b)
if s.quick_ratio() > threshold:  # Fast upper bound
    if s.ratio() > threshold:    # Accurate check
        # Process match
```

3. **Disable autojunk for small sequences** (< 200 items):
```python
s = SequenceMatcher(None, a, b, autojunk=False)
```

## Common Pitfalls

```python
# WRONG: ratio() order matters!
SequenceMatcher(None, 'tide', 'diet').ratio()  # 0.25
SequenceMatcher(None, 'diet', 'tide').ratio()  # 0.5

# RIGHT: Normalize comparison
def symmetric_ratio(a, b):
    return max(
        SequenceMatcher(None, a, b).ratio(),
        SequenceMatcher(None, b, a).ratio()
    )

# WRONG: Expecting minimal diff
# difflib prioritizes "looks right" over minimal edits

# RIGHT: Use for human-readable diffs, not minimal patches
```

## Algorithm Notes

- Based on Ratcliff-Obershelp "gestalt pattern matching"
- O(n²) worst case, often better in practice
- Not minimal edit distance (prioritizes locality)
- Autojunk marks items appearing >1% of sequence as junk (for sequences ≥200)
