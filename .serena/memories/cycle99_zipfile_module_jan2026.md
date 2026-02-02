# Python zipfile Module - Production Patterns (January 2026)

## Overview

The `zipfile` module provides tools to create, read, write, append, and list ZIP archives. Supports ZIP64 extensions (>4GB files), multiple compression methods, and provides both high-level and low-level interfaces.

## Compression Methods

```python
import zipfile

# Available compression constants
zipfile.ZIP_STORED      # No compression (default)
zipfile.ZIP_DEFLATED    # Standard ZIP compression (requires zlib)
zipfile.ZIP_BZIP2       # BZIP2 compression (requires bz2)
zipfile.ZIP_LZMA        # LZMA compression (requires lzma)
zipfile.ZIP_ZSTANDARD   # Zstandard compression (Python 3.14+, requires compression.zstd)
```

## ZipFile Class - Core Operations

### Opening Archives

```python
import zipfile

# Read existing archive
with zipfile.ZipFile('archive.zip', 'r') as zf:
    print(zf.namelist())

# Create new archive (truncates existing)
with zipfile.ZipFile('new.zip', 'w') as zf:
    zf.write('file.txt')

# Append to existing archive
with zipfile.ZipFile('archive.zip', 'a') as zf:
    zf.write('another.txt')

# Exclusive create (error if exists)
with zipfile.ZipFile('unique.zip', 'x') as zf:
    zf.write('data.txt')

# With compression
with zipfile.ZipFile('compressed.zip', 'w', 
                     compression=zipfile.ZIP_DEFLATED,
                     compresslevel=9) as zf:
    zf.write('large_file.txt')

# ZIP64 for large files (enabled by default)
with zipfile.ZipFile('huge.zip', 'w', allowZip64=True) as zf:
    zf.write('multi_gb_file.bin')

# Handle old timestamps (before 1980)
with zipfile.ZipFile('archive.zip', 'w', strict_timestamps=False) as zf:
    zf.write('old_file.txt')  # Timestamp set to 1980-01-01

# Specify metadata encoding for reading
with zipfile.ZipFile('legacy.zip', 'r', metadata_encoding='cp437') as zf:
    print(zf.namelist())
```

### Reading Archives

```python
import zipfile

with zipfile.ZipFile('archive.zip', 'r') as zf:
    # List all members
    names = zf.namelist()  # ['file1.txt', 'dir/file2.txt']
    
    # Get ZipInfo objects for all members
    infos = zf.infolist()
    for info in infos:
        print(f"{info.filename}: {info.file_size} bytes")
    
    # Get info for specific member
    info = zf.getinfo('file1.txt')
    print(f"Compressed: {info.compress_size}, Original: {info.file_size}")
    
    # Read file contents as bytes
    content = zf.read('file1.txt')
    
    # Read with password (encrypted archives)
    content = zf.read('secret.txt', pwd=b'password123')
    
    # Open as file-like object
    with zf.open('file1.txt') as f:
        for line in f:
            print(line.decode('utf-8'))
    
    # Extract single file
    zf.extract('file1.txt', path='output_dir')
    
    # Extract all files
    zf.extractall(path='output_dir')
    
    # Extract with password
    zf.extractall(path='output_dir', pwd=b'password')
    
    # Test archive integrity
    bad_file = zf.testzip()  # Returns None if OK, or first bad filename
    
    # Print table of contents
    zf.printdir()
```

### Writing Archives

```python
import zipfile
import os

with zipfile.ZipFile('output.zip', 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    # Write file from disk
    zf.write('local_file.txt')
    
    # Write with different archive name
    zf.write('local_file.txt', arcname='renamed.txt')
    
    # Write with path stripped
    zf.write('/path/to/file.txt', arcname='file.txt')
    
    # Write with per-file compression settings
    zf.write('image.png', compress_type=zipfile.ZIP_STORED)  # Don't compress
    zf.write('text.txt', compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    
    # Write string/bytes data directly
    zf.writestr('generated.txt', 'Hello, World!')
    zf.writestr('binary.dat', b'\x00\x01\x02\x03')
    
    # Write with ZipInfo for custom metadata
    info = zipfile.ZipInfo('custom.txt')
    info.compress_type = zipfile.ZIP_DEFLATED
    info.date_time = (2026, 1, 25, 12, 0, 0)
    zf.writestr(info, 'Custom metadata file')
    
    # Create directory entry (Python 3.11+)
    zf.mkdir('new_directory/')
    
    # Add comment to archive
    zf.comment = b'This is an archive comment'

# Write entire directory tree
def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            filepath = os.path.join(root, file)
            arcname = os.path.relpath(filepath, path)
            ziph.write(filepath, arcname)

with zipfile.ZipFile('project.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zipdir('my_project/', zf)
```

### Streaming Write (Unknown Size)

```python
import zipfile

with zipfile.ZipFile('streaming.zip', 'w') as zf:
    # For files >2GB with unknown size, use force_zip64
    with zf.open('large_file.dat', 'w', force_zip64=True) as f:
        for chunk in generate_chunks():
            f.write(chunk)
```

## ZipInfo Objects

```python
import zipfile

# Create ZipInfo from filesystem file
info = zipfile.ZipInfo.from_file('document.pdf', arcname='docs/document.pdf')

# Create manually
info = zipfile.ZipInfo(filename='manual.txt', date_time=(2026, 1, 25, 10, 30, 0))

# Key attributes
info.filename        # Name in archive
info.file_size       # Uncompressed size
info.compress_size   # Compressed size
info.compress_type   # Compression method
info.date_time       # (year, month, day, hour, min, sec)
info.CRC             # CRC-32 checksum
info.comment         # File comment (bytes)
info.external_attr   # External file attributes
info.compress_level  # Compression level (Python 3.13+)

# Check if directory
if info.is_dir():
    print(f"{info.filename} is a directory")

# date_time tuple structure:
# (year, month, day, hours, minutes, seconds)
# Year >= 1980, month/day 1-based, hours/mins/secs 0-based
```

## Path Objects (pathlib-like interface)

```python
import zipfile

# Create Path from archive
root = zipfile.Path('archive.zip')

# Navigate using / operator
subpath = root / 'subdir' / 'file.txt'

# Or using joinpath
subpath = root.joinpath('subdir', 'file.txt')

# Check existence and type
if subpath.exists():
    if subpath.is_file():
        content = subpath.read_text()
    elif subpath.is_dir():
        for child in subpath.iterdir():
            print(child.name)

# Read content
text = subpath.read_text(encoding='utf-8')
data = subpath.read_bytes()

# Open as file
with subpath.open('r', encoding='utf-8') as f:
    for line in f:
        print(line)

# Path attributes
print(subpath.name)      # 'file.txt'
print(subpath.suffix)    # '.txt'
print(subpath.stem)      # 'file'
print(subpath.suffixes)  # ['.txt']
```

## PyZipFile - Python Libraries

```python
import zipfile

# Create archive with Python bytecode
with zipfile.PyZipFile('mypackage.zip', 'w') as pzf:
    # Compile and add .py files as .pyc
    pzf.writepy('mypackage/')
    
    # With optimization level
    # 0 = no optimization
    # 1 = remove assert statements
    # 2 = remove docstrings too

# Create optimized archive
with zipfile.PyZipFile('optimized.zip', 'w', optimize=2) as pzf:
    pzf.writepy('mypackage/')

# Filter what gets included
def no_tests(path):
    name = os.path.basename(path)
    return not (name == 'test' or name.startswith('test_'))

with zipfile.PyZipFile('production.zip', 'w') as pzf:
    pzf.writepy('mypackage/', filterfunc=no_tests)
```

## Utility Functions

```python
import zipfile

# Check if file is a valid ZIP
if zipfile.is_zipfile('maybe.zip'):
    print("Valid ZIP file")

# Works with file-like objects too
with open('file.dat', 'rb') as f:
    if zipfile.is_zipfile(f):
        print("It's a ZIP")
```

## Password-Protected Archives

```python
import zipfile

# Read encrypted archive
with zipfile.ZipFile('encrypted.zip', 'r') as zf:
    # Set default password
    zf.setpassword(b'secret123')
    
    # Read all files with default password
    for name in zf.namelist():
        content = zf.read(name)
    
    # Or specify per-file password
    content = zf.read('file.txt', pwd=b'different_password')

# Note: zipfile cannot CREATE encrypted archives
# Decryption is slow (pure Python implementation)
```

## Security Considerations

### Path Traversal Prevention

```python
import zipfile
import os

def safe_extract(zf: zipfile.ZipFile, dest: str) -> None:
    """Extract ZIP safely, preventing path traversal attacks."""
    dest = os.path.abspath(dest)
    
    for member in zf.namelist():
        # Get absolute path of extraction target
        target = os.path.abspath(os.path.join(dest, member))
        
        # Verify it's within destination
        if not target.startswith(dest + os.sep) and target != dest:
            raise ValueError(f"Path traversal detected: {member}")
    
    # Safe to extract
    zf.extractall(dest)

# Usage
with zipfile.ZipFile('untrusted.zip', 'r') as zf:
    safe_extract(zf, 'output/')
```

### Zip Bomb Protection

```python
import zipfile

def safe_extract_with_limits(
    zf: zipfile.ZipFile,
    dest: str,
    max_size: int = 1024 * 1024 * 1024,  # 1GB
    max_files: int = 10000
) -> None:
    """Extract with size and file count limits."""
    total_size = sum(info.file_size for info in zf.infolist())
    file_count = len(zf.namelist())
    
    if total_size > max_size:
        raise ValueError(f"Archive too large: {total_size} bytes")
    
    if file_count > max_files:
        raise ValueError(f"Too many files: {file_count}")
    
    # Check compression ratio (zip bomb indicator)
    for info in zf.infolist():
        if info.compress_size > 0:
            ratio = info.file_size / info.compress_size
            if ratio > 100:  # Suspicious compression ratio
                raise ValueError(f"Suspicious compression ratio for {info.filename}")
    
    zf.extractall(dest)
```

## Production Patterns

### ZipArchiveManager Class

```python
import zipfile
import os
import io
from pathlib import Path
from typing import Optional, Iterator, BinaryIO, Union
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class ArchiveInfo:
    """Summary information about a ZIP archive."""
    file_count: int
    total_size: int
    compressed_size: int
    compression_ratio: float

class ZipArchiveManager:
    """Production-grade ZIP archive management."""
    
    def __init__(
        self,
        compression: int = zipfile.ZIP_DEFLATED,
        compresslevel: int = 6,
        max_extract_size: int = 10 * 1024 * 1024 * 1024,  # 10GB
        max_files: int = 100000
    ):
        self.compression = compression
        self.compresslevel = compresslevel
        self.max_extract_size = max_extract_size
        self.max_files = max_files
    
    def create_archive(
        self,
        archive_path: Union[str, Path],
        source_paths: list[Union[str, Path]],
        base_dir: Optional[Union[str, Path]] = None
    ) -> ArchiveInfo:
        """Create archive from multiple source paths."""
        archive_path = Path(archive_path)
        
        with zipfile.ZipFile(
            archive_path, 'w',
            compression=self.compression,
            compresslevel=self.compresslevel
        ) as zf:
            for source in source_paths:
                source = Path(source)
                if source.is_file():
                    arcname = source.name if base_dir is None else source.relative_to(base_dir)
                    zf.write(source, arcname)
                elif source.is_dir():
                    self._add_directory(zf, source, base_dir)
        
        return self.get_archive_info(archive_path)
    
    def _add_directory(
        self,
        zf: zipfile.ZipFile,
        directory: Path,
        base_dir: Optional[Path]
    ) -> None:
        """Recursively add directory to archive."""
        for root, dirs, files in os.walk(directory):
            root_path = Path(root)
            for file in files:
                filepath = root_path / file
                if base_dir:
                    arcname = str(filepath.relative_to(base_dir))
                else:
                    arcname = str(filepath.relative_to(directory.parent))
                zf.write(filepath, arcname)
    
    def extract_archive(
        self,
        archive_path: Union[str, Path],
        dest_dir: Union[str, Path],
        members: Optional[list[str]] = None
    ) -> list[str]:
        """Safely extract archive with security checks."""
        dest_dir = Path(dest_dir).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(archive_path, 'r') as zf:
            # Security checks
            self._validate_archive(zf)
            self._check_path_traversal(zf, dest_dir)
            
            # Extract
            members_to_extract = members or zf.namelist()
            zf.extractall(dest_dir, members=members_to_extract)
            
            return members_to_extract
    
    def _validate_archive(self, zf: zipfile.ZipFile) -> None:
        """Validate archive against security limits."""
        infos = zf.infolist()
        
        if len(infos) > self.max_files:
            raise ValueError(f"Too many files: {len(infos)} > {self.max_files}")
        
        total_size = sum(info.file_size for info in infos)
        if total_size > self.max_extract_size:
            raise ValueError(f"Archive too large: {total_size} bytes")
        
        # Check for zip bombs
        for info in infos:
            if info.compress_size > 0 and info.file_size > 0:
                ratio = info.file_size / info.compress_size
                if ratio > 1000:
                    raise ValueError(f"Suspicious compression ratio: {ratio}")
    
    def _check_path_traversal(self, zf: zipfile.ZipFile, dest: Path) -> None:
        """Check for path traversal attacks."""
        dest_str = str(dest)
        
        for name in zf.namelist():
            target = (dest / name).resolve()
            if not str(target).startswith(dest_str):
                raise ValueError(f"Path traversal attempt: {name}")
    
    def get_archive_info(self, archive_path: Union[str, Path]) -> ArchiveInfo:
        """Get summary information about archive."""
        with zipfile.ZipFile(archive_path, 'r') as zf:
            infos = zf.infolist()
            total_size = sum(info.file_size for info in infos)
            compressed_size = sum(info.compress_size for info in infos)
            
            return ArchiveInfo(
                file_count=len(infos),
                total_size=total_size,
                compressed_size=compressed_size,
                compression_ratio=total_size / compressed_size if compressed_size > 0 else 1.0
            )
    
    def list_contents(
        self,
        archive_path: Union[str, Path],
        pattern: Optional[str] = None
    ) -> Iterator[zipfile.ZipInfo]:
        """List archive contents, optionally filtered by pattern."""
        import fnmatch
        
        with zipfile.ZipFile(archive_path, 'r') as zf:
            for info in zf.infolist():
                if pattern is None or fnmatch.fnmatch(info.filename, pattern):
                    yield info
    
    @contextmanager
    def open_member(
        self,
        archive_path: Union[str, Path],
        member: str,
        mode: str = 'r'
    ) -> Iterator[BinaryIO]:
        """Open a specific member as file-like object."""
        with zipfile.ZipFile(archive_path, 'r') as zf:
            with zf.open(member, mode) as f:
                yield f
    
    def read_member(
        self,
        archive_path: Union[str, Path],
        member: str
    ) -> bytes:
        """Read member contents as bytes."""
        with zipfile.ZipFile(archive_path, 'r') as zf:
            return zf.read(member)
    
    def add_to_archive(
        self,
        archive_path: Union[str, Path],
        files: dict[str, Union[str, bytes, Path]]
    ) -> None:
        """Add files to existing archive."""
        with zipfile.ZipFile(archive_path, 'a') as zf:
            for arcname, content in files.items():
                if isinstance(content, (str, Path)):
                    zf.write(content, arcname)
                else:
                    zf.writestr(arcname, content)
    
    def create_in_memory(
        self,
        files: dict[str, bytes]
    ) -> bytes:
        """Create ZIP archive in memory."""
        buffer = io.BytesIO()
        
        with zipfile.ZipFile(buffer, 'w', self.compression) as zf:
            for name, content in files.items():
                zf.writestr(name, content)
        
        return buffer.getvalue()


# Usage examples
if __name__ == "__main__":
    manager = ZipArchiveManager(compresslevel=9)
    
    # Create archive
    info = manager.create_archive(
        'backup.zip',
        ['documents/', 'config.json'],
        base_dir=Path.cwd()
    )
    print(f"Created archive: {info.file_count} files, {info.compression_ratio:.1f}x compression")
    
    # Safe extraction
    extracted = manager.extract_archive('backup.zip', 'restored/')
    print(f"Extracted {len(extracted)} files")
    
    # List contents
    for info in manager.list_contents('backup.zip', '*.txt'):
        print(f"{info.filename}: {info.file_size} bytes")
    
    # In-memory archive
    data = manager.create_in_memory({
        'hello.txt': b'Hello, World!',
        'data.json': b'{"key": "value"}'
    })
```

## Command-Line Interface

```bash
# Create archive
python -m zipfile -c archive.zip file1.txt file2.txt directory/

# Extract archive
python -m zipfile -e archive.zip output_dir/

# List contents
python -m zipfile -l archive.zip

# Test archive integrity
python -m zipfile -t archive.zip

# With metadata encoding
python -m zipfile -l archive.zip --metadata-encoding cp437
```

## Common Gotchas

1. **Archive names**: Should be relative, no leading slashes
2. **Null bytes**: Truncate filename at null byte
3. **Encryption**: Can READ encrypted, cannot CREATE encrypted
4. **Decryption speed**: Pure Python, very slow
5. **ZIP64**: Enabled by default for >4GB files
6. **Timestamps**: Limited to 1980-2107 range
7. **Close required**: Must close or use context manager for valid archive

## Version History

- Python 3.14: Added ZIP_ZSTANDARD compression
- Python 3.13: Public compress_level attribute on ZipInfo
- Python 3.12: Added delete parameter to TemporaryDirectory
- Python 3.11: Added ZipFile.mkdir(), metadata_encoding parameter
- Python 3.8: Added strict_timestamps parameter
- Python 3.6: Added ZipFile.open() write mode
