# Cycle 60: File I/O & Path Patterns (Jan 2026)

Production file handling patterns from official Python documentation.

## pathlib - Modern Path Handling (Python 3.4+)

### Basic Path Operations

```python
from pathlib import Path

# Create Path objects
path = Path("/home/user/project/data.json")
relative = Path("src/main.py")
cwd = Path.cwd()
home = Path.home()

# Path components
path.name          # "data.json"
path.stem          # "data"
path.suffix        # ".json"
path.parent        # Path("/home/user/project")
path.parts         # ("/", "home", "user", "project", "data.json")
path.anchor        # "/" on Unix, "C:\\" on Windows

# Path manipulation
path.with_name("config.json")     # Change filename
path.with_stem("backup")          # Change stem only
path.with_suffix(".yaml")         # Change extension
```

### Joining Paths (Use / Operator)

```python
from pathlib import Path

# GOOD: Use / operator (cross-platform)
base = Path("/home/user")
config = base / "config" / "settings.json"

# GOOD: Using joinpath
config = base.joinpath("config", "settings.json")

# BAD: String concatenation (breaks cross-platform)
config = base + "/config/settings.json"  # Don't do this!
```

### File Discovery with Glob

```python
from pathlib import Path

project = Path("src/")

# Find all Python files in directory
py_files = list(project.glob("*.py"))

# Recursive search (all subdirectories)
all_py = list(project.rglob("*.py"))

# Multiple patterns
configs = list(project.glob("**/*.{json,yaml,toml}"))

# Find specific patterns
tests = list(project.glob("**/test_*.py"))
```

### Reading and Writing Files

```python
from pathlib import Path
import json

path = Path("config.json")

# Read entire file as text
content = path.read_text(encoding="utf-8")

# Read as bytes
binary = path.read_bytes()

# Write text (overwrites)
path.write_text('{"key": "value"}', encoding="utf-8")

# Write bytes
path.write_bytes(b'\x00\x01\x02')

# JSON read/write pattern
data = json.loads(path.read_text())
path.write_text(json.dumps(data, indent=2))
```

### Path Existence and Type Checks

```python
from pathlib import Path

path = Path("some/path")

# Existence checks
path.exists()       # True if path exists
path.is_file()      # True if regular file
path.is_dir()       # True if directory
path.is_symlink()   # True if symbolic link
path.is_absolute()  # True if absolute path

# Get absolute path
absolute = path.resolve()

# Check if path is relative to another
path.is_relative_to("/home/user")  # Python 3.9+
```

### Directory Operations

```python
from pathlib import Path

directory = Path("output/reports/2026")

# Create directory (with parents)
directory.mkdir(parents=True, exist_ok=True)

# List directory contents
for item in directory.iterdir():
    if item.is_file():
        print(f"File: {item.name}")
    elif item.is_dir():
        print(f"Dir: {item.name}")

# Remove empty directory
directory.rmdir()
```

### File Operations

```python
from pathlib import Path
import shutil

source = Path("data.json")
dest = Path("backup/data.json")

# Rename (move within same filesystem)
source.rename(dest)

# Replace (overwrite if exists)
source.replace(dest)

# Delete file
source.unlink(missing_ok=True)  # Python 3.8+

# Copy file (use shutil)
shutil.copy2(source, dest)  # Preserves metadata

# Get file stats
stats = source.stat()
stats.st_size       # Size in bytes
stats.st_mtime      # Modification time
stats.st_ctime      # Creation time (Windows) / metadata change (Unix)
```

### Common Anti-Pattern: Converting to str Too Early

```python
from pathlib import Path

path = Path("data/file.txt")

# BAD: Losing Path benefits
def bad_process(filepath: str):
    # Now you need os.path for everything
    pass

bad_process(str(path))  # Converting too early

# GOOD: Keep as Path throughout
def good_process(filepath: Path):
    content = filepath.read_text()
    parent = filepath.parent
    # Full Path API available

good_process(path)
```

## aiofiles - Async File I/O

### Basic Async Read/Write

```python
import aiofiles
import asyncio

async def read_file(path: str) -> str:
    """Async file read - doesn't block event loop."""
    async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
        return await f.read()

async def write_file(path: str, content: str) -> None:
    """Async file write."""
    async with aiofiles.open(path, mode='w', encoding='utf-8') as f:
        await f.write(content)

async def append_log(path: str, message: str) -> None:
    """Async append to file."""
    async with aiofiles.open(path, mode='a', encoding='utf-8') as f:
        await f.write(f"{message}\n")
```

### Async Line-by-Line Reading

```python
import aiofiles

async def process_large_file(path: str) -> int:
    """Process file line by line without loading into memory."""
    line_count = 0
    async with aiofiles.open(path, mode='r') as f:
        async for line in f:
            line_count += 1
            await process_line(line.strip())
    return line_count

async def process_line(line: str) -> None:
    """Process a single line."""
    # Your async processing logic
    pass
```

### Async JSON Operations

```python
import aiofiles
import json

async def read_json(path: str) -> dict:
    """Async JSON read."""
    async with aiofiles.open(path, mode='r') as f:
        content = await f.read()
        return json.loads(content)

async def write_json(path: str, data: dict) -> None:
    """Async JSON write with pretty printing."""
    async with aiofiles.open(path, mode='w') as f:
        await f.write(json.dumps(data, indent=2))
```

### Async Binary Operations

```python
import aiofiles

async def copy_file_async(source: str, dest: str, chunk_size: int = 64 * 1024) -> None:
    """Async file copy with chunked reading."""
    async with aiofiles.open(source, mode='rb') as src:
        async with aiofiles.open(dest, mode='wb') as dst:
            while True:
                chunk = await src.read(chunk_size)
                if not chunk:
                    break
                await dst.write(chunk)
```

### Concurrent File Operations

```python
import aiofiles
import asyncio
from pathlib import Path

async def read_multiple_files(paths: list[Path]) -> list[str]:
    """Read multiple files concurrently."""
    async def read_one(path: Path) -> str:
        async with aiofiles.open(path, mode='r') as f:
            return await f.read()
    
    # All files read concurrently
    return await asyncio.gather(*[read_one(p) for p in paths])
```

## watchdog - File System Monitoring

### Basic File Watcher

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
import time

class MyHandler(FileSystemEventHandler):
    """Handle file system events."""
    
    def on_created(self, event: FileSystemEvent):
        if not event.is_directory:
            print(f"Created: {event.src_path}")
    
    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory:
            print(f"Modified: {event.src_path}")
    
    def on_deleted(self, event: FileSystemEvent):
        if not event.is_directory:
            print(f"Deleted: {event.src_path}")
    
    def on_moved(self, event: FileSystemEvent):
        print(f"Moved: {event.src_path} -> {event.dest_path}")

def watch_directory(path: str):
    """Watch directory for changes."""
    observer = Observer()
    observer.schedule(MyHandler(), path, recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    watch_directory("./watched_folder")
```

### Pattern Matching Handler

```python
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

class PythonFileHandler(PatternMatchingEventHandler):
    """Watch only Python files."""
    
    def __init__(self):
        super().__init__(
            patterns=["*.py"],
            ignore_patterns=["*/__pycache__/*", "*/.git/*"],
            ignore_directories=True,
            case_sensitive=True
        )
    
    def on_modified(self, event):
        print(f"Python file changed: {event.src_path}")
        # Trigger tests, linting, etc.
```

### Regex Matching Handler

```python
from watchdog.events import RegexMatchingEventHandler

class LogHandler(RegexMatchingEventHandler):
    """Watch log files with regex."""
    
    def __init__(self):
        super().__init__(
            regexes=[r".*\.log$", r".*\.txt$"],
            ignore_regexes=[r".*/\..*"],  # Ignore hidden files
            ignore_directories=True
        )
    
    def on_modified(self, event):
        print(f"Log updated: {event.src_path}")
```

### Async-Compatible Watcher

```python
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from queue import Queue

class AsyncEventHandler(FileSystemEventHandler):
    """Bridge watchdog events to asyncio."""
    
    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self.queue = queue
        self.loop = loop
    
    def on_any_event(self, event):
        # Thread-safe way to put event in async queue
        self.loop.call_soon_threadsafe(
            self.queue.put_nowait, event
        )

async def watch_async(path: str):
    """Async file watcher."""
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    
    handler = AsyncEventHandler(queue, loop)
    observer = Observer()
    observer.schedule(handler, path, recursive=True)
    observer.start()
    
    try:
        while True:
            event = await queue.get()
            print(f"Event: {event.event_type} - {event.src_path}")
            # Process event asynchronously
    finally:
        observer.stop()
        observer.join()
```

### Production Watcher with Debouncing

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from collections import defaultdict
from threading import Timer

class DebouncedHandler(FileSystemEventHandler):
    """Debounce rapid file changes."""
    
    def __init__(self, callback, delay: float = 0.5):
        self.callback = callback
        self.delay = delay
        self.timers: dict[str, Timer] = {}
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        path = event.src_path
        
        # Cancel existing timer for this path
        if path in self.timers:
            self.timers[path].cancel()
        
        # Set new timer
        timer = Timer(self.delay, self._trigger, args=[path])
        self.timers[path] = timer
        timer.start()
    
    def _trigger(self, path: str):
        del self.timers[path]
        self.callback(path)

def on_file_changed(path: str):
    print(f"File stabilized: {path}")
    # Run build, tests, etc.
```

## Temporary Files and Directories

```python
import tempfile
from pathlib import Path

# Temporary file (auto-deleted)
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    f.write('{"temp": true}')
    temp_path = Path(f.name)

# Temporary directory (auto-deleted)
with tempfile.TemporaryDirectory() as tmpdir:
    temp_path = Path(tmpdir) / "data.txt"
    temp_path.write_text("temporary content")
    # Directory and contents deleted after block

# Get temp directory path
temp_dir = Path(tempfile.gettempdir())
```

## Decision Matrix

| Use Case | Tool | Why |
|----------|------|-----|
| Path manipulation | pathlib | Object-oriented, cross-platform |
| Sync file read/write | pathlib | Simple, built-in |
| Async file I/O | aiofiles | Non-blocking in async code |
| File monitoring | watchdog | Cross-platform, event-driven |
| Large file processing | aiofiles + chunks | Memory efficient |
| Temporary files | tempfile | Auto-cleanup, secure |

## Quick Reference

```python
# pathlib essentials
from pathlib import Path
path = Path("dir") / "file.txt"
content = path.read_text()
path.write_text("content")
path.mkdir(parents=True, exist_ok=True)
files = list(path.parent.rglob("*.py"))

# aiofiles essentials
import aiofiles
async with aiofiles.open("file.txt", mode='r') as f:
    content = await f.read()

# watchdog essentials
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
observer = Observer()
observer.schedule(MyHandler(), path, recursive=True)
observer.start()
```
