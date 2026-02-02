# Python ftplib Module - Production Patterns (January 2026)

## Overview

The `ftplib` module implements the client side of the FTP protocol (RFC 959) with TLS/SSL support via `FTP_TLS` (RFC 4217). Default encoding is UTF-8 per RFC 2640.

## Core Classes

### FTP Class

```python
from ftplib import FTP

# Context manager pattern (recommended)
with FTP('ftp.example.com') as ftp:
    ftp.login('user', 'password')
    ftp.cwd('/public')
    files = ftp.nlst()
    print(files)

# Manual connection
ftp = FTP()
ftp.connect('ftp.example.com', 21, timeout=30)
ftp.login('user', 'password')
print(ftp.getwelcome())
# ... operations ...
ftp.quit()  # Sends QUIT command
```

### FTP_TLS Class (Secure FTP)

```python
from ftplib import FTP_TLS
import ssl

# Implicit TLS (connect on port 990)
with FTP_TLS('ftp.example.com') as ftps:
    ftps.login('user', 'password')
    ftps.prot_p()  # Switch to protected data connection
    files = ftps.nlst()

# Explicit TLS (STARTTLS on port 21)
ftps = FTP_TLS()
ftps.connect('ftp.example.com', 21)
ftps.auth()  # Upgrade to TLS
ftps.login('user', 'password')
ftps.prot_p()  # Protected data transfers
# ... operations ...
ftps.quit()

# Custom SSL context
context = ssl.create_default_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE  # For self-signed certs (not recommended)

ftps = FTP_TLS(context=context)
ftps.connect('ftp.example.com')
```

## File Transfer Operations

### Binary Transfer (retrbinary/storbinary)

```python
from ftplib import FTP
from pathlib import Path

# Download binary file
with FTP('ftp.example.com') as ftp:
    ftp.login('user', 'password')
    
    # Download to file
    with open('local_file.zip', 'wb') as f:
        ftp.retrbinary('RETR remote_file.zip', f.write)
    
    # Download with callback for progress
    def progress_callback(data: bytes):
        downloaded.append(len(data))
        print(f"Downloaded {sum(downloaded)} bytes")
    
    downloaded = []
    ftp.retrbinary('RETR large_file.bin', progress_callback, blocksize=8192)

# Upload binary file
with FTP('ftp.example.com') as ftp:
    ftp.login('user', 'password')
    
    with open('local_file.zip', 'rb') as f:
        ftp.storbinary('STOR remote_file.zip', f)
    
    # With callback and custom block size
    with open('large_file.bin', 'rb') as f:
        ftp.storbinary('STOR remote_large.bin', f, blocksize=8192, callback=progress_cb)
```

### Text/ASCII Transfer (retrlines/storlines)

```python
from ftplib import FTP

# Download text file line by line
lines = []
with FTP('ftp.example.com') as ftp:
    ftp.login('user', 'password')
    ftp.retrlines('RETR readme.txt', lines.append)

content = '\n'.join(lines)

# Upload text file
with FTP('ftp.example.com') as ftp:
    ftp.login('user', 'password')
    
    with open('readme.txt', 'rb') as f:  # Note: 'rb' mode
        ftp.storlines('STOR readme.txt', f)
```

## Directory Operations

### Navigation and Listing

```python
from ftplib import FTP

with FTP('ftp.example.com') as ftp:
    ftp.login('user', 'password')
    
    # Current directory
    current = ftp.pwd()
    print(f"Current: {current}")
    
    # Change directory
    ftp.cwd('/public/files')
    
    # Go up one level
    ftp.cwd('..')
    
    # Simple file list
    files = ftp.nlst()  # Returns list of filenames
    
    # Detailed listing (like ls -l)
    ftp.dir()  # Prints to stdout
    
    # Detailed listing to list
    listing = []
    ftp.dir(listing.append)
    
    # MLSD - Machine-readable listing (RFC 3659)
    for name, facts in ftp.mlsd():
        print(f"{name}: type={facts.get('type')}, size={facts.get('size')}")
    
    # MLSD with specific path
    for name, facts in ftp.mlsd('/public'):
        if facts.get('type') == 'file':
            print(f"File: {name}, modified: {facts.get('modify')}")
```

### Directory Management

```python
from ftplib import FTP, error_perm

with FTP('ftp.example.com') as ftp:
    ftp.login('user', 'password')
    
    # Create directory
    try:
        ftp.mkd('/new_directory')
    except error_perm as e:
        if '550' in str(e):  # Already exists
            pass
        else:
            raise
    
    # Remove empty directory
    ftp.rmd('/old_directory')
    
    # Create nested directories (helper function)
    def mkd_recursive(ftp: FTP, path: str):
        parts = path.strip('/').split('/')
        current = ''
        for part in parts:
            current = f"{current}/{part}"
            try:
                ftp.mkd(current)
            except error_perm:
                pass  # Already exists
    
    mkd_recursive(ftp, '/a/b/c/d')
```

## File Operations

```python
from ftplib import FTP

with FTP('ftp.example.com') as ftp:
    ftp.login('user', 'password')
    
    # Get file size
    size = ftp.size('file.zip')
    print(f"Size: {size} bytes")
    
    # Rename file
    ftp.rename('old_name.txt', 'new_name.txt')
    
    # Delete file
    ftp.delete('unwanted_file.txt')
    
    # Send arbitrary command
    response = ftp.sendcmd('FEAT')  # List server features
    print(response)
    
    # NOOP (keep-alive)
    ftp.voidcmd('NOOP')
```

## Transfer Mode Control

```python
from ftplib import FTP

with FTP('ftp.example.com') as ftp:
    ftp.login('user', 'password')
    
    # Passive mode (default, works through firewalls)
    ftp.set_pasv(True)
    
    # Active mode (requires open ports)
    ftp.set_pasv(False)
    
    # Low-level transfer command
    # Returns socket for data transfer
    conn = ftp.transfercmd('RETR file.txt')
    data = conn.recv(8192)
    conn.close()
    ftp.voidresp()  # Get server response
    
    # With expected size
    conn, size = ftp.ntransfercmd('RETR file.txt')
    print(f"Expected size: {size}")
```

## Exception Handling

```python
from ftplib import (
    FTP,
    error_reply,    # Unexpected server reply
    error_temp,     # Temporary error (4xx response)
    error_perm,     # Permanent error (5xx response)
    error_proto,    # Protocol error
    all_errors      # Tuple of all FTP exceptions + OSError
)

def safe_ftp_operation():
    try:
        with FTP('ftp.example.com', timeout=30) as ftp:
            ftp.login('user', 'password')
            ftp.retrbinary('RETR file.zip', open('file.zip', 'wb').write)
    
    except error_temp as e:
        # 4xx errors - try again later
        print(f"Temporary error: {e}")
        # Implement retry logic
    
    except error_perm as e:
        # 5xx errors - fix the problem
        code = str(e)[:3]
        if code == '530':
            print("Login failed - check credentials")
        elif code == '550':
            print("File not found or permission denied")
        elif code == '553':
            print("Invalid filename")
        else:
            print(f"Permanent error: {e}")
    
    except error_proto as e:
        print(f"Protocol error: {e}")
    
    except OSError as e:
        print(f"Network error: {e}")
```

## Production FTPClient Class

```python
from ftplib import FTP, FTP_TLS, error_perm, error_temp
from pathlib import Path
from typing import Iterator, Optional, Callable
from contextlib import contextmanager
import time
import logging

logger = logging.getLogger(__name__)

class FTPClient:
    """Production FTP client with retry logic and secure defaults."""
    
    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        *,
        port: int = 21,
        use_tls: bool = True,
        passive: bool = True,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.use_tls = use_tls
        self.passive = passive
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._ftp: Optional[FTP] = None
    
    def connect(self) -> None:
        """Establish FTP connection with optional TLS."""
        if self.use_tls:
            self._ftp = FTP_TLS(timeout=self.timeout)
        else:
            self._ftp = FTP(timeout=self.timeout)
        
        self._ftp.connect(self.host, self.port)
        
        if self.use_tls:
            self._ftp.auth()  # Upgrade to TLS
        
        self._ftp.login(self.user, self.password)
        
        if self.use_tls:
            self._ftp.prot_p()  # Protected data transfers
        
        self._ftp.set_pasv(self.passive)
        logger.info(f"Connected to {self.host}")
    
    def disconnect(self) -> None:
        """Close FTP connection."""
        if self._ftp:
            try:
                self._ftp.quit()
            except Exception:
                self._ftp.close()
            self._ftp = None
            logger.info(f"Disconnected from {self.host}")
    
    def __enter__(self) -> 'FTPClient':
        self.connect()
        return self
    
    def __exit__(self, *args) -> None:
        self.disconnect()
    
    def _with_retry(self, operation: Callable, *args, **kwargs):
        """Execute operation with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except error_temp as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(self.retry_delay * (2 ** attempt))
            except error_perm:
                raise  # Don't retry permanent errors
        
        raise last_error
    
    def download(
        self,
        remote_path: str,
        local_path: Path,
        *,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Download file with optional progress callback."""
        downloaded = 0
        
        def callback(data: bytes):
            nonlocal downloaded
            downloaded += len(data)
            if progress_callback:
                progress_callback(downloaded)
        
        with open(local_path, 'wb') as f:
            def write_and_track(data: bytes):
                f.write(data)
                callback(data)
            
            self._with_retry(
                self._ftp.retrbinary,
                f'RETR {remote_path}',
                write_and_track
            )
        
        logger.info(f"Downloaded {remote_path} -> {local_path} ({downloaded} bytes)")
    
    def upload(
        self,
        local_path: Path,
        remote_path: str,
        *,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Upload file with optional progress callback."""
        uploaded = 0
        
        def callback(data: bytes):
            nonlocal uploaded
            uploaded += len(data)
            if progress_callback:
                progress_callback(uploaded)
        
        with open(local_path, 'rb') as f:
            self._with_retry(
                self._ftp.storbinary,
                f'STOR {remote_path}',
                f,
                callback=callback
            )
        
        logger.info(f"Uploaded {local_path} -> {remote_path} ({uploaded} bytes)")
    
    def list_files(self, path: str = '.') -> list[dict]:
        """List files with metadata using MLSD."""
        results = []
        for name, facts in self._ftp.mlsd(path):
            if name not in ('.', '..'):
                results.append({
                    'name': name,
                    'type': facts.get('type', 'unknown'),
                    'size': int(facts.get('size', 0)),
                    'modify': facts.get('modify'),
                })
        return results
    
    def ensure_directory(self, path: str) -> None:
        """Create directory if it doesn't exist."""
        parts = path.strip('/').split('/')
        current = ''
        for part in parts:
            current = f"{current}/{part}"
            try:
                self._ftp.mkd(current)
            except error_perm:
                pass  # Already exists
    
    def delete_file(self, path: str) -> None:
        """Delete a file."""
        self._with_retry(self._ftp.delete, path)
        logger.info(f"Deleted {path}")
    
    def sync_directory(
        self,
        local_dir: Path,
        remote_dir: str,
        *,
        delete_extra: bool = False,
    ) -> None:
        """Sync local directory to remote."""
        self.ensure_directory(remote_dir)
        
        remote_files = {f['name'] for f in self.list_files(remote_dir)}
        local_files = set()
        
        for local_file in local_dir.iterdir():
            if local_file.is_file():
                local_files.add(local_file.name)
                remote_path = f"{remote_dir}/{local_file.name}"
                self.upload(local_file, remote_path)
        
        if delete_extra:
            for extra in remote_files - local_files:
                self.delete_file(f"{remote_dir}/{extra}")
```

## Usage Example

```python
# Download with progress
def show_progress(bytes_downloaded: int):
    print(f"\rDownloaded: {bytes_downloaded:,} bytes", end='')

with FTPClient('ftp.example.com', 'user', 'pass', use_tls=True) as client:
    client.download('remote/file.zip', Path('local/file.zip'), progress_callback=show_progress)
    print()  # Newline after progress

# Sync local folder to FTP
with FTPClient('ftp.example.com', 'user', 'pass') as client:
    client.sync_directory(Path('./uploads'), '/public/uploads', delete_extra=True)
```

## Key Points

1. **Always use context managers** - Ensures proper connection cleanup
2. **Prefer FTP_TLS** - Plain FTP sends credentials in cleartext
3. **Call prot_p()** - Switches data channel to TLS after auth()
4. **Use passive mode** - Works through NAT/firewalls (default)
5. **Handle error_temp** - Implement retry logic for 4xx responses
6. **Use MLSD over nlst/dir** - Machine-readable format (RFC 3659)
7. **Set timeouts** - Prevent hanging on network issues
8. **Binary vs text** - Use retrbinary/storbinary for non-text files

## Common Response Codes

| Code | Meaning |
|------|---------|
| 125 | Data connection already open, transfer starting |
| 150 | File status okay, opening data connection |
| 226 | Transfer complete |
| 230 | User logged in |
| 250 | Requested action okay |
| 331 | User name okay, need password |
| 421 | Service not available |
| 425 | Can't open data connection |
| 426 | Connection closed, transfer aborted |
| 450 | Requested action not taken |
| 500 | Syntax error |
| 530 | Not logged in |
| 550 | Requested action not taken (file unavailable) |
| 553 | Requested action not taken (filename not allowed) |
