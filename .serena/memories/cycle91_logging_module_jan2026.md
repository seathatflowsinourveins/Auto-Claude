# Python logging Module - Production Patterns (Jan 2026)

## Overview
The `logging` module provides a flexible event logging system for applications. Based on PEP 282, it supports hierarchical loggers, multiple handlers, and customizable formatting.

## Logger Objects

### Getting Loggers
```python
import logging

# Root logger (not recommended for libraries)
logging.warning("Root logger message")

# Named logger (recommended)
logger = logging.getLogger(__name__)  # Use module name for hierarchy

# Logger hierarchy
# "a.b.c" is child of "a.b" which is child of "a"
parent = logging.getLogger("myapp")
child = logging.getLogger("myapp.database")  # Inherits from parent
```

### Logger Configuration
```python
logger = logging.getLogger(__name__)

# Set minimum level (messages below this are ignored)
logger.setLevel(logging.DEBUG)

# Propagation to parent loggers (default True)
logger.propagate = True  # Set False to prevent duplicate logs

# Check if level would be processed
if logger.isEnabledFor(logging.DEBUG):
    logger.debug("Expensive %s", expensive_operation())

# Add/remove handlers
logger.addHandler(handler)
logger.removeHandler(handler)

# Add/remove filters
logger.addFilter(filter_obj)
logger.removeFilter(filter_obj)
```

### Logging Methods
```python
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")

# With exception info
try:
    risky_operation()
except Exception:
    logger.exception("Failed with traceback")  # Adds exc_info automatically
    # Or explicitly:
    logger.error("Failed", exc_info=True)

# Extra fields for LogRecord
logger.info("User action", extra={"user_id": 123, "action": "login"})

# Stack info (useful for async debugging)
logger.warning("Unexpected state", stack_info=True)
```

## Logging Levels

| Level | Numeric | Use Case |
|-------|---------|----------|
| DEBUG | 10 | Detailed diagnostic info |
| INFO | 20 | Confirmation of expected behavior |
| WARNING | 30 | Unexpected but recoverable |
| ERROR | 40 | Serious problem, some functionality failed |
| CRITICAL | 50 | Program may not continue |
| NOTSET | 0 | Inherit from parent |

```python
# Custom levels
TRACE = 5
logging.addLevelName(TRACE, "TRACE")

def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kwargs)

logging.Logger.trace = trace
```

## Handler Classes

### StreamHandler (console output)
```python
import sys

# To stdout (default stderr)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(console)
```

### FileHandler (simple file logging)
```python
file_handler = logging.FileHandler(
    filename="app.log",
    mode="a",           # append (default)
    encoding="utf-8",   # Always specify!
    delay=True          # Don't open until first write
)
```

### RotatingFileHandler (size-based rotation)
```python
from logging.handlers import RotatingFileHandler

rotating = RotatingFileHandler(
    filename="app.log",
    maxBytes=10_000_000,    # 10MB per file
    backupCount=5,          # Keep 5 backup files
    encoding="utf-8"
)
# Creates: app.log, app.log.1, app.log.2, ... app.log.5
```

### TimedRotatingFileHandler (time-based rotation)
```python
from logging.handlers import TimedRotatingFileHandler

timed = TimedRotatingFileHandler(
    filename="app.log",
    when="midnight",        # 'S', 'M', 'H', 'D', 'midnight', 'W0'-'W6'
    interval=1,             # Every 1 unit of 'when'
    backupCount=30,         # Keep 30 days
    encoding="utf-8",
    utc=True                # Use UTC time for rotation
)
# Creates: app.log, app.log.2026-01-24, app.log.2026-01-23, ...
```

### SocketHandler & DatagramHandler (network logging)
```python
from logging.handlers import SocketHandler, DatagramHandler

# TCP (reliable)
tcp_handler = SocketHandler(host="logserver", port=9000)

# UDP (faster, unreliable)
udp_handler = DatagramHandler(host="logserver", port=9000)
```

### SMTPHandler (email alerts)
```python
from logging.handlers import SMTPHandler

mail_handler = SMTPHandler(
    mailhost=("smtp.example.com", 587),
    fromaddr="app@example.com",
    toaddrs=["ops@example.com"],
    subject="Application Error",
    credentials=("user", "password"),
    secure=()  # Enable TLS
)
mail_handler.setLevel(logging.ERROR)  # Only email errors
```

### HTTPHandler (webhook/API logging)
```python
from logging.handlers import HTTPHandler

http_handler = HTTPHandler(
    host="logs.example.com",
    url="/api/logs",
    method="POST",
    secure=True  # HTTPS
)
```

### QueueHandler & QueueListener (async logging)
```python
import queue
from logging.handlers import QueueHandler, QueueListener

# Create queue
log_queue = queue.Queue(-1)  # Unlimited size

# Handler that puts to queue (used by application)
queue_handler = QueueHandler(log_queue)
logger.addHandler(queue_handler)

# Real handlers (run in background thread)
file_handler = logging.FileHandler("app.log")
console_handler = logging.StreamHandler()

# Listener processes queue in separate thread
listener = QueueListener(
    log_queue,
    file_handler,
    console_handler,
    respect_handler_level=True
)
listener.start()

# On shutdown
listener.stop()
```

### MemoryHandler (buffered logging)
```python
from logging.handlers import MemoryHandler

# Buffer until capacity or flushLevel reached
memory_handler = MemoryHandler(
    capacity=100,                    # Buffer 100 records
    flushLevel=logging.ERROR,        # Flush on ERROR
    target=file_handler,             # Send to this handler
    flushOnClose=True
)
```

### WatchedFileHandler (external rotation support)
```python
from logging.handlers import WatchedFileHandler

# Detects when log file is rotated externally (logrotate)
watched = WatchedFileHandler(
    filename="app.log",
    encoding="utf-8"
)
```

### NullHandler (library default)
```python
# In library code - prevents "No handlers" warning
logging.getLogger("mylibrary").addHandler(logging.NullHandler())
```

## Formatter Class

### Basic Formatting
```python
# %-style (default)
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# {}-style (Python 3.2+)
formatter = logging.Formatter(
    fmt="{asctime} - {name} - {levelname} - {message}",
    style="{"
)

# $-style (Template)
formatter = logging.Formatter(
    fmt="$asctime - $name - $levelname - $message",
    style="$"
)
```

### LogRecord Attributes
```python
# Available in format strings:
# %(name)s        - Logger name
# %(levelno)s     - Numeric level (10, 20, 30, 40, 50)
# %(levelname)s   - Text level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
# %(pathname)s    - Full path of source file
# %(filename)s    - Filename portion of pathname
# %(module)s      - Module name
# %(funcName)s    - Function name
# %(lineno)d      - Line number
# %(created)f     - Unix timestamp
# %(asctime)s     - Human-readable time
# %(msecs)d       - Millisecond portion
# %(relativeCreated)d - Milliseconds since logging loaded
# %(thread)d      - Thread ID
# %(threadName)s  - Thread name
# %(process)d     - Process ID
# %(processName)s - Process name
# %(message)s     - The logged message
# %(exc_info)s    - Exception info (or empty)
# %(stack_info)s  - Stack info (or empty)
# %(taskName)s    - asyncio task name (Python 3.12+)
```

### Custom Formatter
```python
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Include exception if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Include extra fields
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and key not in log_data:
                log_data[key] = value
        
        return json.dumps(log_data)
```

## Filter Objects

```python
class ContextFilter(logging.Filter):
    """Add contextual information to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add custom attributes
        record.request_id = getattr(thread_local, "request_id", "N/A")
        record.user_id = getattr(thread_local, "user_id", "anonymous")
        
        # Return True to log, False to suppress
        return True

class LevelRangeFilter(logging.Filter):
    """Only allow records within a level range."""
    
    def __init__(self, min_level: int, max_level: int):
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level
    
    def filter(self, record: logging.LogRecord) -> bool:
        return self.min_level <= record.levelno <= self.max_level

# Usage
info_only = LevelRangeFilter(logging.INFO, logging.INFO)
console_handler.addFilter(info_only)
```

## LoggerAdapter (contextual logging)

```python
class RequestAdapter(logging.LoggerAdapter):
    """Add request context to all log messages."""
    
    def process(self, msg, kwargs):
        # Prepend request_id to message
        request_id = self.extra.get("request_id", "N/A")
        return f"[{request_id}] {msg}", kwargs

# Usage
base_logger = logging.getLogger(__name__)
request_logger = RequestAdapter(base_logger, {"request_id": "abc-123"})
request_logger.info("Processing request")
# Output: [abc-123] Processing request
```

## Configuration Methods

### basicConfig (simple setup)
```python
# Call ONCE at application startup (not in libraries!)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", encoding="utf-8")
    ]
)

# Or with filename shorthand
logging.basicConfig(
    filename="app.log",
    filemode="a",
    encoding="utf-8",
    level=logging.DEBUG
)
```

### dictConfig (recommended for production)
```python
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "json": {
            "()": "myapp.logging.JSONFormatter"  # Custom class
        }
    },
    
    "filters": {
        "context": {
            "()": "myapp.logging.ContextFilter"
        }
    },
    
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "json",
            "filename": "app.log",
            "maxBytes": 10485760,
            "backupCount": 5,
            "encoding": "utf-8"
        },
        "error_file": {
            "class": "logging.FileHandler",
            "level": "ERROR",
            "formatter": "standard",
            "filename": "errors.log",
            "encoding": "utf-8"
        }
    },
    
    "loggers": {
        "": {  # Root logger
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": True
        },
        "myapp": {
            "handlers": ["console", "file", "error_file"],
            "level": "DEBUG",
            "propagate": False
        },
        "urllib3": {
            "level": "WARNING"  # Quiet noisy libraries
        },
        "sqlalchemy.engine": {
            "level": "WARNING"
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
```

### fileConfig (INI-style)
```python
# logging.conf
"""
[loggers]
keys=root,myapp

[handlers]
keys=console,file

[formatters]
keys=standard

[logger_root]
level=DEBUG
handlers=console

[logger_myapp]
level=DEBUG
handlers=console,file
qualname=myapp
propagate=0

[handler_console]
class=StreamHandler
level=INFO
formatter=standard
args=(sys.stdout,)

[handler_file]
class=handlers.RotatingFileHandler
level=DEBUG
formatter=standard
args=('app.log', 'a', 10485760, 5, 'utf-8')

[formatter_standard]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
"""

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
```

## Thread Safety

```python
# logging module is thread-safe by default
# Each handler has a lock (handler.acquire() / handler.release())

# For async applications, use QueueHandler
import asyncio
import queue
from logging.handlers import QueueHandler, QueueListener

def setup_async_logging():
    log_queue = queue.Queue(-1)
    
    # Actual handlers (run in listener thread)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler("app.log", encoding="utf-8")
    ]
    
    # Configure root logger with queue handler only
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(QueueHandler(log_queue))
    
    # Start listener (dequeues and dispatches to real handlers)
    listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
    listener.start()
    
    return listener  # Return so it can be stopped on shutdown
```

## Production Logging Pattern

```python
"""Production-ready logging configuration."""
import logging
import logging.config
import queue
import sys
from pathlib import Path
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler


class ProductionLogger:
    """Thread-safe, async-compatible logging setup."""
    
    def __init__(
        self,
        app_name: str,
        log_dir: Path,
        level: int = logging.INFO,
        max_bytes: int = 10_000_000,
        backup_count: int = 5
    ):
        self.app_name = app_name
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Queue for async logging
        self.log_queue: queue.Queue = queue.Queue(-1)
        self.listener: QueueListener | None = None
        
        self._setup_logging(level, max_bytes, backup_count)
    
    def _setup_logging(self, level: int, max_bytes: int, backup_count: int):
        # Formatters
        detailed = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        simple = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S"
        )
        
        # Handlers
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple)
        
        file_handler = RotatingFileHandler(
            filename=self.log_dir / f"{self.app_name}.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed)
        
        error_handler = RotatingFileHandler(
            filename=self.log_dir / f"{self.app_name}.error.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed)
        
        # Queue handler for async
        queue_handler = QueueHandler(self.log_queue)
        
        # Configure root logger
        root = logging.getLogger()
        root.setLevel(level)
        root.handlers.clear()
        root.addHandler(queue_handler)
        
        # Start listener
        self.listener = QueueListener(
            self.log_queue,
            console_handler,
            file_handler,
            error_handler,
            respect_handler_level=True
        )
        self.listener.start()
        
        # Quiet noisy libraries
        for noisy in ["urllib3", "httpx", "httpcore", "asyncio"]:
            logging.getLogger(noisy).setLevel(logging.WARNING)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a named logger."""
        return logging.getLogger(name)
    
    def shutdown(self):
        """Stop the queue listener."""
        if self.listener:
            self.listener.stop()


# Usage
if __name__ == "__main__":
    prod_logger = ProductionLogger(
        app_name="myapp",
        log_dir=Path("logs"),
        level=logging.DEBUG
    )
    
    logger = prod_logger.get_logger(__name__)
    logger.info("Application started")
    
    try:
        # Application code
        pass
    finally:
        prod_logger.shutdown()
```

## Common Patterns

### Contextual Logging with contextvars
```python
import logging
import contextvars
from typing import Any

# Context variables (async-safe)
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="N/A")
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id", default="anonymous")

class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.user_id = user_id_var.get()
        return True

# Add filter to handler
handler = logging.StreamHandler()
handler.addFilter(ContextFilter())
handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(request_id)s | %(user_id)s | %(message)s"
))

# In request handler
async def handle_request(request):
    request_id_var.set(request.headers.get("X-Request-ID", str(uuid.uuid4())))
    user_id_var.set(request.user.id if request.user else "anonymous")
    
    logger.info("Processing request")  # Automatically includes context
```

### Structured Logging with extra
```python
def log_event(logger: logging.Logger, event: str, **context):
    """Log structured event with context."""
    logger.info(
        event,
        extra={
            "event": event,
            **context
        }
    )

# Usage
log_event(logger, "user_login", user_id=123, ip="1.2.3.4", method="oauth")
```

## Key Takeaways

1. **Use `__name__` for logger names** - Creates proper hierarchy
2. **Libraries: only NullHandler** - Let applications configure
3. **Always use QueueHandler in production** - Non-blocking I/O
4. **Set encoding="utf-8"** - Avoid encoding errors
5. **Use dictConfig** - Most flexible configuration method
6. **Quiet noisy libraries** - Set WARNING level for urllib3, httpx
7. **Use LoggerAdapter or contextvars** - For request-scoped context
8. **Never call basicConfig in libraries** - Only in application entry point
9. **Use exception()** - Automatically includes traceback
10. **isEnabledFor() before expensive formatting** - Performance optimization
