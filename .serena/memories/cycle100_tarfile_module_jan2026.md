# Cycle 100: tarfile Module - Production Patterns (January 2026)

## Overview

The `tarfile` module provides comprehensive support for reading and writing tar archives, including compressed variants (gzip, bzip2, xz, zstd). Python 3.12+ introduced **extraction filters** as a critical security feature.

## Core Classes

### TarFile - Archive Interface

```python
import tarfile
from pathlib import Path

# Mode strings: 'r', 'r:gz', 'r:bz2', 'r:xz', 'r:zst', 'w', 'a', 'x'
# Streaming: 'r|gz', 'w|gz', 'r|bz2', 'w|bz2', 'r|xz', 'w|xz'

# Reading archives
with tarfile.open('archive.tar.gz', 'r:gz') as tar:
    tar.extractall(path='dest/', filter='data')  # ALWAYS use filter!
    members = tar.getmembers()
    names = tar.getnames()

# Writing archives
with tarfile.open('backup.tar.xz', 'w:xz') as tar:
    tar.add('src/', arcname='backup')
    tar.addfile(tarinfo, fileobj=data_stream)

# Streaming (unseekable sources)
with tarfile.open(fileobj=response.raw, mode='r|gz') as tar:
    for member in tar:
        if member.isreg():
            tar.extract(member, path='dest/', filter='data')
```

### TarInfo - Member Metadata

```python
# Create TarInfo manually
info = tarfile.TarInfo(name='data/config.json')
info.size = len(data)
info.mtime = time.time()
info.mode = 0o644
info.type = tarfile.REGTYPE  # Regular file

# Add with custom metadata
with tarfile.open('archive.tar', 'w') as tar:
    tar.addfile(info, io.BytesIO(data))

# Inspect member attributes
for member in tar.getmembers():
    print(f"{member.name}: {member.size} bytes, mode={oct(member.mode)}")
    print(f"  uid={member.uid}, gid={member.gid}")
    print(f"  mtime={member.mtime}, type={member.type}")
```

## Extraction Filters (Python 3.12+ CRITICAL)

### Security Filters

```python
# NEVER use without filter - security risk!
# tar.extractall()  # DEPRECATED - will warn/error in Python 3.14+

# Filter options:
# 'fully_trusted' - No filtering (dangerous, use only for trusted sources)
# 'tar'           - Limits features to common tar behavior
# 'data'          - Maximum security - files only, no special types

# Production extraction
with tarfile.open('archive.tar.gz', 'r:gz') as tar:
    tar.extractall(path='dest/', filter='data')

# Single file extraction
with tarfile.open('archive.tar', 'r') as tar:
    tar.extract('specific/file.txt', path='dest/', filter='data')
```

### Custom Filters

```python
import tarfile
from tarfile import TarInfo, FilterError

def secure_filter(member: TarInfo, dest_path: str) -> TarInfo | None:
    """Custom extraction filter with security checks."""
    # Block absolute paths
    if member.name.startswith('/'):
        raise FilterError(f"Absolute path blocked: {member.name}")
    
    # Block path traversal
    if '..' in member.name:
        raise FilterError(f"Path traversal blocked: {member.name}")
    
    # Block special files (devices, fifos)
    if member.isdev() or member.isfifo():
        return None  # Skip silently
    
    # Block large files
    if member.size > 100 * 1024 * 1024:  # 100MB limit
        raise FilterError(f"File too large: {member.name} ({member.size} bytes)")
    
    # Sanitize permissions
    if member.isdir():
        member.mode = 0o755
    else:
        member.mode = 0o644
    
    # Reset ownership
    member.uid = member.gid = 0
    member.uname = member.gname = ''
    
    return member

# Use custom filter
with tarfile.open('archive.tar.gz', 'r:gz') as tar:
    tar.extractall(path='dest/', filter=secure_filter)
```

## Compression Modes

### All Supported Formats

```python
# Uncompressed
tarfile.open('archive.tar', 'w')          # Write
tarfile.open('archive.tar', 'r')          # Read

# Gzip (most common)
tarfile.open('archive.tar.gz', 'w:gz')    # Write
tarfile.open('archive.tgz', 'r:gz')       # Read

# Bzip2 (better compression, slower)
tarfile.open('archive.tar.bz2', 'w:bz2')  # Write
tarfile.open('archive.tbz2', 'r:bz2')     # Read

# LZMA/XZ (best compression, slowest)
tarfile.open('archive.tar.xz', 'w:xz')    # Write
tarfile.open('archive.txz', 'r:xz')       # Read

# Zstandard (Python 3.14+, fast + good compression)
tarfile.open('archive.tar.zst', 'w:zst')  # Write
tarfile.open('archive.tar.zst', 'r:zst')  # Read

# Auto-detect compression on read
tarfile.open('archive.tar.gz')  # Detects .gz automatically
```

### Compression Levels

```python
import gzip
import tarfile

# Custom compression level via fileobj
with gzip.open('archive.tar.gz', 'wb', compresslevel=9) as gz:
    with tarfile.open(fileobj=gz, mode='w') as tar:
        tar.add('data/')
```

## Tar Formats

```python
# USTAR - POSIX.1-1988 (most compatible, 256 char path limit)
tarfile.open('archive.tar', 'w', format=tarfile.USTAR_FORMAT)

# GNU - Extended GNU format (long names, sparse files)
tarfile.open('archive.tar', 'w', format=tarfile.GNU_FORMAT)

# PAX - POSIX.1-2001 (unlimited paths, Unicode, extended headers)
tarfile.open('archive.tar', 'w', format=tarfile.PAX_FORMAT)  # DEFAULT

# Check archive format
with tarfile.open('archive.tar') as tar:
    print(tar.format)  # One of USTAR_FORMAT, GNU_FORMAT, PAX_FORMAT
```

## Streaming Mode

```python
import tarfile
import requests

def extract_from_url(url: str, dest: Path) -> list[str]:
    """Extract tar.gz from HTTP stream without full download."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    extracted = []
    with tarfile.open(fileobj=response.raw, mode='r|gz') as tar:
        for member in tar:
            if member.isreg() and not member.name.startswith('.'):
                tar.extract(member, path=dest, filter='data')
                extracted.append(member.name)
    
    return extracted

# Pipe-based streaming (stdin)
import sys
with tarfile.open(fileobj=sys.stdin.buffer, mode='r|gz') as tar:
    tar.extractall(filter='data')
```

## Production Pattern: TarArchiveManager

```python
import tarfile
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Iterator, Callable
from dataclasses import dataclass

@dataclass
class ArchiveMember:
    """Metadata for an archive member."""
    name: str
    size: int
    mtime: float
    mode: int
    is_dir: bool
    is_file: bool
    is_link: bool

class TarArchiveManager:
    """Production-grade tar archive management with security defaults."""
    
    # Maximum extraction sizes
    MAX_FILE_SIZE = 500 * 1024 * 1024      # 500MB per file
    MAX_TOTAL_SIZE = 10 * 1024 * 1024 * 1024  # 10GB total
    MAX_MEMBERS = 100_000
    
    def __init__(
        self,
        archive_path: Path,
        mode: str = 'r',
        compression: str | None = None
    ):
        self.archive_path = Path(archive_path)
        self.mode = mode
        self.compression = compression or self._detect_compression()
    
    def _detect_compression(self) -> str:
        """Detect compression from file extension."""
        suffix = self.archive_path.suffix.lower()
        mapping = {
            '.gz': 'gz', '.tgz': 'gz',
            '.bz2': 'bz2', '.tbz2': 'bz2',
            '.xz': 'xz', '.txz': 'xz',
            '.zst': 'zst',
        }
        return mapping.get(suffix, '')
    
    def _get_mode_string(self) -> str:
        """Build tarfile mode string."""
        if self.compression:
            return f"{self.mode}:{self.compression}"
        return self.mode
    
    def _security_filter(
        self,
        member: tarfile.TarInfo,
        dest_path: str
    ) -> tarfile.TarInfo | None:
        """Apply security restrictions to extraction."""
        # Block absolute paths and traversal
        if member.name.startswith('/') or '..' in member.name:
            return None
        
        # Block special file types
        if member.isdev() or member.isfifo():
            return None
        
        # Enforce size limits
        if member.size > self.MAX_FILE_SIZE:
            raise tarfile.FilterError(
                f"File exceeds size limit: {member.name} ({member.size} bytes)"
            )
        
        # Sanitize permissions
        if member.isdir():
            member.mode = 0o755
        elif member.isreg():
            member.mode = 0o644
        
        # Reset ownership
        member.uid = member.gid = 0
        member.uname = member.gname = ''
        
        return member
    
    def list_members(self) -> Iterator[ArchiveMember]:
        """Iterate over archive members with metadata."""
        with tarfile.open(self.archive_path, self._get_mode_string()) as tar:
            for member in tar.getmembers():
                yield ArchiveMember(
                    name=member.name,
                    size=member.size,
                    mtime=member.mtime,
                    mode=member.mode,
                    is_dir=member.isdir(),
                    is_file=member.isreg(),
                    is_link=member.issym() or member.islnk()
                )
    
    def extract_all(
        self,
        dest: Path,
        filter_func: Callable | None = None
    ) -> list[Path]:
        """Extract all members with security filtering."""
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        
        filter_to_use = filter_func or self._security_filter
        extracted = []
        total_size = 0
        
        with tarfile.open(self.archive_path, self._get_mode_string()) as tar:
            members = tar.getmembers()
            
            if len(members) > self.MAX_MEMBERS:
                raise ValueError(f"Archive has too many members: {len(members)}")
            
            for member in members:
                total_size += member.size
                if total_size > self.MAX_TOTAL_SIZE:
                    raise ValueError(f"Total extraction size exceeds limit")
            
            tar.extractall(path=dest, filter=filter_to_use)
            
            for member in members:
                if member.isreg():
                    extracted.append(dest / member.name)
        
        return extracted
    
    def extract_member(
        self,
        member_name: str,
        dest: Path
    ) -> Path | None:
        """Extract a single member by name."""
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        
        with tarfile.open(self.archive_path, self._get_mode_string()) as tar:
            try:
                member = tar.getmember(member_name)
            except KeyError:
                return None
            
            tar.extract(member, path=dest, filter=self._security_filter)
            return dest / member.name
    
    @classmethod
    def create_archive(
        cls,
        archive_path: Path,
        source_paths: list[Path],
        compression: str = 'gz',
        base_dir: Path | None = None
    ) -> 'TarArchiveManager':
        """Create a new tar archive from source paths."""
        mode = f'w:{compression}' if compression else 'w'
        
        with tarfile.open(archive_path, mode) as tar:
            for source in source_paths:
                source = Path(source)
                if base_dir:
                    arcname = source.relative_to(base_dir)
                else:
                    arcname = source.name
                tar.add(source, arcname=str(arcname))
        
        return cls(archive_path, mode='r', compression=compression)
    
    def get_checksum(self, algorithm: str = 'sha256') -> str:
        """Calculate archive checksum."""
        hasher = hashlib.new(algorithm)
        with open(self.archive_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify archive can be fully read without errors."""
        try:
            with tarfile.open(self.archive_path, self._get_mode_string()) as tar:
                for member in tar:
                    if member.isreg():
                        tar.extractfile(member)
            return True
        except (tarfile.TarError, OSError):
            return False

# Usage examples
if __name__ == '__main__':
    # Create archive
    manager = TarArchiveManager.create_archive(
        Path('backup.tar.gz'),
        [Path('src/'), Path('data/')],
        compression='gz'
    )
    
    # List contents
    for member in manager.list_members():
        print(f"{member.name}: {member.size} bytes")
    
    # Safe extraction
    extracted = manager.extract_all(Path('restore/'))
    print(f"Extracted {len(extracted)} files")
    
    # Verify integrity
    if manager.verify_integrity():
        print(f"Checksum: {manager.get_checksum()}")
```

## Command-Line Interface

```bash
# List archive contents
python -m tarfile -l archive.tar.gz

# Extract archive
python -m tarfile -e archive.tar.gz dest/

# Create archive
python -m tarfile -c archive.tar.gz file1.txt file2.txt dir/

# Verbose mode
python -m tarfile -v -l archive.tar.gz
```

## Security Best Practices

1. **Always use extraction filters** - `filter='data'` for untrusted archives
2. **Validate before extraction** - Check sizes, member count, paths
3. **Limit resource consumption** - Cap file size, total size, member count
4. **Sanitize permissions** - Reset mode, uid, gid after extraction
5. **Avoid symlink attacks** - Block or carefully validate symbolic links
6. **Use streaming for large archives** - Avoid loading entire archive in memory

## Key Takeaways

| Feature | Recommendation |
|---------|----------------|
| Read mode | Use `'r:gz'` for explicit, `'r'` for auto-detect |
| Write mode | Use `'w:xz'` for best compression, `'w:gz'` for compatibility |
| Extraction filter | ALWAYS specify - `'data'` for security, custom for flexibility |
| Format | PAX_FORMAT (default) for Unicode and long paths |
| Streaming | Use `'r|gz'` for network streams or pipes |
| Large files | Stream extraction, chunked processing |
