# Python tempfile Module - Production Patterns (January 2026)

## Overview

The `tempfile` module creates temporary files and directories securely. It provides high-level interfaces with automatic cleanup (context managers) and low-level functions requiring manual cleanup.

## High-Level Interfaces (Auto-Cleanup)

### TemporaryFile - Anonymous Temp File

```python
import tempfile

# Basic usage - auto-deleted on close
with tempfile.TemporaryFile(mode='w+b') as fp:
    fp.write(b'Hello world!')
    fp.seek(0)
    data = fp.read()
# File automatically deleted here

# Text mode
with tempfile.TemporaryFile(mode='w+t', encoding='utf-8') as fp:
    fp.write('Unicode text')
    fp.seek(0)
    text = fp.read()

# With custom location and naming
with tempfile.TemporaryFile(
    mode='w+b',
    suffix='.tmp',
    prefix='myapp_',
    dir='/custom/temp'
) as fp:
    fp.write(b'data')
```

**Key characteristics:**
- Default mode is `'w+b'` (read/write binary)
- On Unix, directory entry removed immediately (invisible file)
- On Windows/other platforms, may have visible name
- Uses `os.O_TMPFILE` on Linux kernel 3.11+ for security

### NamedTemporaryFile - Visible Named File

```python
import tempfile
import os

# Basic - deleted on close (default)
with tempfile.NamedTemporaryFile() as fp:
    print(f"Temp file: {fp.name}")
    fp.write(b'data')
# File deleted here

# Keep file after close (delete=False)
with tempfile.NamedTemporaryFile(delete=False) as fp:
    temp_path = fp.name
    fp.write(b'persistent data')
# File still exists - manual cleanup needed
os.unlink(temp_path)

# Delete on context exit, not on close (Python 3.12+)
with tempfile.NamedTemporaryFile(delete_on_close=False) as fp:
    fp.write(b'data')
    fp.close()
    # File still exists, can reopen by name
    with open(fp.name, 'rb') as f:
        data = f.read()
# File deleted on context exit

# Custom naming
with tempfile.NamedTemporaryFile(
    suffix='.json',
    prefix='config_',
    dir='/tmp',
    delete=True
) as fp:
    fp.write(b'{"key": "value"}')
```

**Key parameters:**
- `delete=True` (default): Delete file when closed
- `delete_on_close=True` (default): Delete immediately on close
- `delete_on_close=False`: Delete on context exit (allows reopen)
- `.name` attribute: Full path to temp file

### SpooledTemporaryFile - Memory-Backed Until Threshold

```python
import tempfile

# Stays in memory until max_size exceeded
with tempfile.SpooledTemporaryFile(max_size=1024*1024) as fp:  # 1MB
    fp.write(b'small data')  # In memory
    fp.seek(0)
    data = fp.read()
# Never hit disk if under 1MB

# Force rollover to disk
with tempfile.SpooledTemporaryFile(max_size=1024) as fp:
    fp.write(b'x' * 2000)  # Exceeds max_size, rolls to disk
    
    # Or force rollover manually
    fp.rollover()
    
    # Check if rolled over
    # fp._file is io.BytesIO (memory) or real file (disk)

# Useful for buffering uploads
def process_upload(stream, max_memory=5*1024*1024):
    with tempfile.SpooledTemporaryFile(max_size=max_memory) as buffer:
        for chunk in stream:
            buffer.write(chunk)
        buffer.seek(0)
        return process_data(buffer)
```

**Use cases:**
- HTTP request body buffering
- Caching data that might be small enough for memory
- Avoiding disk I/O for small operations

### TemporaryDirectory - Temp Directory with Cleanup

```python
import tempfile
import os

# Basic usage
with tempfile.TemporaryDirectory() as tmpdir:
    print(f"Temp dir: {tmpdir}")
    
    # Create files inside
    filepath = os.path.join(tmpdir, 'data.txt')
    with open(filepath, 'w') as f:
        f.write('content')
    
    # Create subdirectories
    subdir = os.path.join(tmpdir, 'subdir')
    os.makedirs(subdir)
# Directory and ALL contents deleted here

# Custom naming
with tempfile.TemporaryDirectory(
    suffix='_work',
    prefix='build_',
    dir='/tmp'
) as tmpdir:
    pass

# Ignore cleanup errors (Python 3.10+)
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
    # Even if files are locked, cleanup won't raise
    pass

# Disable cleanup for debugging (Python 3.12+)
with tempfile.TemporaryDirectory(delete=False) as tmpdir:
    print(f"Debug dir (not deleted): {tmpdir}")
# Directory preserved for inspection

# Manual cleanup
tmpdir = tempfile.TemporaryDirectory()
print(tmpdir.name)
# ... use directory ...
tmpdir.cleanup()  # Explicit cleanup
```

## Low-Level Functions (Manual Cleanup Required)

### mkstemp() - Create Temp File

```python
import tempfile
import os

# Returns (file_descriptor, path) tuple
fd, path = tempfile.mkstemp()
try:
    # Write using os-level operations
    os.write(fd, b'data')
    
    # Or convert to file object
    with os.fdopen(fd, 'w+b') as fp:
        fp.write(b'more data')
        # fd is now managed by file object
finally:
    os.unlink(path)  # Manual cleanup

# With options
fd, path = tempfile.mkstemp(
    suffix='.dat',
    prefix='temp_',
    dir='/custom/temp',
    text=False  # Binary mode (default)
)

# Text mode
fd, path = tempfile.mkstemp(text=True)
os.write(fd, 'text data'.encode())  # Still need bytes for os.write

# Bytes path (suffix=b'' forces bytes return)
fd, path = tempfile.mkstemp(suffix=b'')
# path is bytes, not str
```

**Security guarantees:**
- No race conditions (uses `os.O_EXCL`)
- Readable/writable only by creating user
- Not executable
- File descriptor not inherited by child processes

### mkdtemp() - Create Temp Directory

```python
import tempfile
import os
import shutil

# Returns absolute path string
tmpdir = tempfile.mkdtemp()
try:
    # Use directory
    filepath = os.path.join(tmpdir, 'file.txt')
    with open(filepath, 'w') as f:
        f.write('content')
finally:
    shutil.rmtree(tmpdir)  # Manual cleanup

# With options
tmpdir = tempfile.mkdtemp(
    suffix='_work',
    prefix='build_',
    dir='/custom/temp'
)
```

**Security guarantees:**
- No race conditions
- Accessible only by creating user (rwx------)

## Utility Functions

### gettempdir() - Get Temp Directory Path

```python
import tempfile

# Get default temp directory
tmpdir = tempfile.gettempdir()
# Returns str: /tmp, C:\Users\...\AppData\Local\Temp, etc.

# Bytes version
tmpdir_bytes = tempfile.gettempdirb()

# Directory selection order:
# 1. TMPDIR environment variable
# 2. TEMP environment variable
# 3. TMP environment variable
# 4. Platform-specific:
#    - Windows: C:\TEMP, C:\TMP, \TEMP, \TMP
#    - Unix: /tmp, /var/tmp, /usr/tmp
# 5. Current working directory (last resort)
```

### gettempprefix() - Get Filename Prefix

```python
import tempfile

# Get default prefix (usually 'tmp')
prefix = tempfile.gettempprefix()

# Bytes version
prefix_bytes = tempfile.gettempprefixb()
```

### tempdir Global Variable

```python
import tempfile

# Override default temp directory globally (discouraged)
tempfile.tempdir = '/my/custom/temp'

# Better approach: use dir parameter per-call
with tempfile.TemporaryFile(dir='/my/custom/temp') as fp:
    pass
```

## Security Considerations

### Avoid mktemp() - DEPRECATED

```python
import tempfile
import os

# WRONG - Race condition vulnerability (deprecated since 2.3)
path = tempfile.mktemp()  # DON'T USE
# Time gap here - another process could create file with same name
with open(path, 'w') as f:
    f.write('data')

# CORRECT - Use NamedTemporaryFile
with tempfile.NamedTemporaryFile(delete=False) as f:
    path = f.name
    f.write(b'data')
# No race condition - file created atomically
```

### Secure Temp File Pattern

```python
import tempfile
import os

def secure_temp_write(data: bytes, suffix: str = '') -> str:
    """Write data to secure temp file, return path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, data)
        return path
    except:
        os.unlink(path)
        raise
    finally:
        os.close(fd)

# Usage
path = secure_temp_write(b'secret data', suffix='.key')
try:
    # Use file
    pass
finally:
    os.unlink(path)
```

## Production Patterns

### TempFileManager Class

```python
import tempfile
import os
import shutil
from pathlib import Path
from typing import Optional, Iterator, BinaryIO
from contextlib import contextmanager

class TempFileManager:
    """Production-grade temporary file management."""
    
    def __init__(
        self,
        base_dir: Optional[str] = None,
        prefix: str = 'app_',
        cleanup_on_error: bool = True
    ):
        self.base_dir = base_dir
        self.prefix = prefix
        self.cleanup_on_error = cleanup_on_error
        self._temp_paths: list[str] = []
    
    @contextmanager
    def temp_file(
        self,
        suffix: str = '',
        mode: str = 'w+b',
        delete: bool = True
    ) -> Iterator[BinaryIO]:
        """Context manager for temporary file."""
        fp = tempfile.NamedTemporaryFile(
            mode=mode,
            suffix=suffix,
            prefix=self.prefix,
            dir=self.base_dir,
            delete=False  # We handle deletion
        )
        try:
            if not delete:
                self._temp_paths.append(fp.name)
            yield fp
        except Exception:
            if self.cleanup_on_error:
                self._safe_unlink(fp.name)
            raise
        finally:
            fp.close()
            if delete:
                self._safe_unlink(fp.name)
    
    @contextmanager
    def temp_directory(
        self,
        suffix: str = '',
        delete: bool = True
    ) -> Iterator[str]:
        """Context manager for temporary directory."""
        tmpdir = tempfile.mkdtemp(
            suffix=suffix,
            prefix=self.prefix,
            dir=self.base_dir
        )
        try:
            if not delete:
                self._temp_paths.append(tmpdir)
            yield tmpdir
        except Exception:
            if self.cleanup_on_error:
                self._safe_rmtree(tmpdir)
            raise
        finally:
            if delete:
                self._safe_rmtree(tmpdir)
    
    def create_temp_file(self, suffix: str = '') -> str:
        """Create temp file, return path (caller must cleanup)."""
        fd, path = tempfile.mkstemp(
            suffix=suffix,
            prefix=self.prefix,
            dir=self.base_dir
        )
        os.close(fd)
        self._temp_paths.append(path)
        return path
    
    def create_temp_dir(self, suffix: str = '') -> str:
        """Create temp directory, return path (caller must cleanup)."""
        path = tempfile.mkdtemp(
            suffix=suffix,
            prefix=self.prefix,
            dir=self.base_dir
        )
        self._temp_paths.append(path)
        return path
    
    def cleanup_all(self) -> None:
        """Clean up all tracked temporary paths."""
        for path in self._temp_paths:
            if os.path.isfile(path):
                self._safe_unlink(path)
            elif os.path.isdir(path):
                self._safe_rmtree(path)
        self._temp_paths.clear()
    
    @staticmethod
    def _safe_unlink(path: str) -> None:
        try:
            os.unlink(path)
        except OSError:
            pass
    
    @staticmethod
    def _safe_rmtree(path: str) -> None:
        try:
            shutil.rmtree(path)
        except OSError:
            pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_all()
        return False


# Usage examples
if __name__ == "__main__":
    # As context manager
    with TempFileManager(prefix='myapp_') as tm:
        # Temp file that auto-deletes
        with tm.temp_file(suffix='.json') as fp:
            fp.write(b'{"data": 1}')
        
        # Temp directory
        with tm.temp_directory() as tmpdir:
            Path(tmpdir, 'file.txt').write_text('content')
        
        # Persistent temp file (cleaned on manager exit)
        path = tm.create_temp_file(suffix='.dat')
        Path(path).write_bytes(b'data')
    # All temps cleaned up
    
    # Standalone usage
    tm = TempFileManager()
    try:
        path1 = tm.create_temp_file()
        path2 = tm.create_temp_dir()
        # Use temps...
    finally:
        tm.cleanup_all()
```

### Spooled Upload Handler

```python
import tempfile
from typing import Iterator, BinaryIO

class SpooledUploadHandler:
    """Handle file uploads with memory/disk spillover."""
    
    def __init__(self, max_memory: int = 5 * 1024 * 1024):  # 5MB
        self.max_memory = max_memory
    
    def buffer_stream(
        self,
        stream: Iterator[bytes]
    ) -> tempfile.SpooledTemporaryFile:
        """Buffer stream, spilling to disk if needed."""
        buffer = tempfile.SpooledTemporaryFile(
            max_size=self.max_memory,
            mode='w+b'
        )
        try:
            for chunk in stream:
                buffer.write(chunk)
            buffer.seek(0)
            return buffer
        except:
            buffer.close()
            raise
    
    def is_in_memory(self, fp: tempfile.SpooledTemporaryFile) -> bool:
        """Check if spooled file is still in memory."""
        import io
        return isinstance(fp._file, io.BytesIO)
```

## Common Gotchas

1. **Windows file locking**: Can't delete open files on Windows
2. **NamedTemporaryFile reopen**: Use `delete_on_close=False` to reopen by name
3. **mkstemp returns fd**: Must close fd or use `os.fdopen()` to convert
4. **TemporaryFile invisible**: May not have visible name on Unix
5. **Cleanup order**: Ensure files closed before directory cleanup

## Version History

- Python 3.12: Added `delete` param to TemporaryDirectory, `delete_on_close` to NamedTemporaryFile
- Python 3.11: SpooledTemporaryFile fully implements BufferedIOBase/TextIOBase
- Python 3.10: Added `ignore_cleanup_errors` to TemporaryDirectory
- Python 3.5: `os.O_TMPFILE` flag used on Linux; bytes suffix/prefix support
