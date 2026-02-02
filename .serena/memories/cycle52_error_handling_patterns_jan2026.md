# Python Error Handling Patterns (January 2026)

Production-grade error handling patterns from official documentation research.

## 1. Tenacity Retry Library (8.5.0+)

### Basic Retry
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3))
def call_api():
    """Retry up to 3 times on any exception."""
    response = requests.get("https://api.example.com")
    response.raise_for_status()
    return response.json()
```

### Stop Conditions
```python
from tenacity import (
    retry, stop_after_attempt, stop_after_delay,
    stop_never, stop_when_event_set
)

# Stop after N attempts
@retry(stop=stop_after_attempt(5))
def max_attempts(): ...

# Stop after N seconds total
@retry(stop=stop_after_delay(30))
def time_limit(): ...

# Combine conditions with OR (|)
@retry(stop=(stop_after_delay(10) | stop_after_attempt(5)))
def either_condition(): ...

# Never stop (use with caution!)
@retry(stop=stop_never)
def infinite_retry(): ...
```

### Wait Strategies
```python
from tenacity import (
    retry, wait_fixed, wait_random, wait_exponential,
    wait_exponential_jitter, wait_chain
)

# Fixed delay between retries
@retry(wait=wait_fixed(2))  # 2 seconds
def fixed_wait(): ...

# Random delay (avoid thundering herd)
@retry(wait=wait_random(min=1, max=3))
def random_wait(): ...

# Exponential backoff (RECOMMENDED for APIs)
@retry(wait=wait_exponential(multiplier=1, min=4, max=60))
def exponential_backoff(): ...

# Exponential with jitter (BEST for distributed systems)
@retry(wait=wait_exponential_jitter(initial=1, max=60))
def jittered_backoff(): ...

# Chain different waits
@retry(wait=wait_chain(
    wait_fixed(1),           # First retry: 1s
    wait_fixed(2),           # Second retry: 2s  
    wait_exponential(max=10) # Rest: exponential
))
def chained_wait(): ...
```

### Retry Conditions
```python
from tenacity import (
    retry, retry_if_exception_type, retry_if_exception_message,
    retry_if_result, retry_if_not_result
)

# Retry only on specific exceptions
@retry(retry=retry_if_exception_type(ConnectionError))
def network_call(): ...

# Retry on multiple exception types
@retry(retry=retry_if_exception_type((IOError, TimeoutError)))
def io_operation(): ...

# Retry based on exception message
@retry(retry=retry_if_exception_message(match="rate limit"))
def rate_limited_call(): ...

# Retry based on result (e.g., None means failure)
@retry(retry=retry_if_result(lambda x: x is None))
def returns_none_on_failure(): ...
```

### Callbacks and Logging
```python
from tenacity import (
    retry, before_log, after_log, before_sleep_log
)
import logging

logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),
    before=before_log(logger, logging.INFO),
    after=after_log(logger, logging.WARNING),
    before_sleep=before_sleep_log(logger, logging.DEBUG)
)
def logged_operation(): ...

# Custom callbacks
def my_before_callback(retry_state):
    print(f"Attempt #{retry_state.attempt_number}")

def my_after_callback(retry_state):
    print(f"Outcome: {retry_state.outcome}")

@retry(
    before=my_before_callback,
    after=my_after_callback
)
def custom_callbacks(): ...
```

### Reraise and Statistics
```python
from tenacity import retry, stop_after_attempt, RetryError

# Reraise original exception (RECOMMENDED)
@retry(stop=stop_after_attempt(3), reraise=True)
def clean_stack_trace():
    """On final failure, raises original exception, not RetryError."""
    raise ValueError("Original error")

# Access retry statistics
@retry(stop=stop_after_attempt(3))
def with_stats():
    raise Exception("fail")

try:
    with_stats()
except Exception:
    print(with_stats.retry.statistics)
    # {'start_time': ..., 'attempt_number': 3, 'idle_for': ...}
```

### Async Retry
```python
from tenacity import retry, stop_after_attempt, AsyncRetrying

# Decorator for async functions
@retry(stop=stop_after_attempt(3))
async def async_api_call():
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

# Context manager for async code blocks
async def manual_async_retry():
    async for attempt in AsyncRetrying(stop=stop_after_attempt(3)):
        with attempt:
            return await some_async_operation()
```

### Context Manager Pattern
```python
from tenacity import Retrying, stop_after_attempt, wait_fixed

# Retry a code block (not just a function)
for attempt in Retrying(stop=stop_after_attempt(3), wait=wait_fixed(1)):
    with attempt:
        # This block will be retried
        result = fragile_operation()
        validate(result)
```

## 2. Returns Library - Result Type (0.23.0+)

### Basic Result Pattern
```python
from returns.result import Result, Success, Failure

def divide(a: int, b: int) -> Result[float, ZeroDivisionError]:
    if b == 0:
        return Failure(ZeroDivisionError("Cannot divide by zero"))
    return Success(a / b)

# Usage
result = divide(10, 2)  # Success(5.0)
result = divide(10, 0)  # Failure(ZeroDivisionError(...))
```

### @safe Decorator (Exception to Result)
```python
from returns.result import safe, ResultE

@safe
def parse_json(data: str) -> dict:
    """Automatically catches exceptions and wraps in Result."""
    return json.loads(data)

# Returns Success({"key": "value"}) or Failure(JSONDecodeError)
result = parse_json('{"key": "value"}')
result = parse_json('invalid json')

# Type alias for convenience
# ResultE[T] = Result[T, Exception]
```

### Pattern Matching (Python 3.10+)
```python
from returns.result import Result, Success, Failure

def process_result(result: Result[int, str]) -> str:
    match result:
        case Success(value):
            return f"Got value: {value}"
        case Failure(error):
            return f"Error: {error}"

# Pre-3.10 alternative
def process_result_old(result):
    if isinstance(result, Success):
        return f"Got value: {result.unwrap()}"
    else:
        return f"Error: {result.failure()}"
```

### Railway Oriented Programming
```python
from returns.result import Result, Success, Failure, safe
from returns.pipeline import flow

@safe
def fetch_user(user_id: int) -> dict:
    return db.users.get(user_id)

@safe  
def validate_user(user: dict) -> dict:
    if not user.get("active"):
        raise ValueError("User inactive")
    return user

@safe
def get_permissions(user: dict) -> list:
    return permissions_service.get(user["id"])

# Chain operations - stops at first failure
def get_user_permissions(user_id: int):
    return (
        fetch_user(user_id)
        .bind(validate_user)
        .bind(get_permissions)
    )

# Alternative with flow()
def get_user_permissions_flow(user_id: int):
    return flow(
        user_id,
        fetch_user,
        validate_user,
        get_permissions,
    )
```

### Result Methods
```python
from returns.result import Success, Failure

result = Success(10)

# .map() - Transform success value
result.map(lambda x: x * 2)  # Success(20)

# .bind() - Chain Result-returning functions
result.bind(lambda x: Success(x + 5))  # Success(15)

# .alt() - Transform failure value
Failure("error").alt(lambda e: f"wrapped: {e}")  # Failure("wrapped: error")

# .lash() - Recover from failure
Failure("oops").lash(lambda e: Success("recovered"))  # Success("recovered")

# .value_or() - Get value or default
Success(10).value_or(0)  # 10
Failure("x").value_or(0)  # 0

# .unwrap() - Get value (raises on Failure!)
Success(10).unwrap()  # 10
Failure("x").unwrap()  # Raises UnwrapFailedError
```

### Maybe Type (Optional handling)
```python
from returns.maybe import Maybe, Some, Nothing

def find_user(user_id: int) -> Maybe[User]:
    user = db.users.find(user_id)
    if user:
        return Some(user)
    return Nothing

# Usage
find_user(1).map(lambda u: u.name).value_or("Unknown")
```

## 3. Exception Handling Best Practices

### Proper try-except-else-finally
```python
def process_file(path: str) -> str:
    file = None
    try:
        file = open(path, 'r')
        content = file.read()
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise
    except PermissionError:
        logger.error(f"Permission denied: {path}")
        raise
    else:
        # Only runs if no exception
        logger.info(f"Successfully read {len(content)} bytes")
        return content
    finally:
        # Always runs (cleanup)
        if file:
            file.close()
```

### Exception Chaining
```python
class PaymentError(Exception):
    """Base payment exception."""
    pass

class PaymentServiceUnavailable(PaymentError):
    """External payment service is down."""
    pass

def process_payment(amount: float):
    try:
        gateway.charge(amount)
    except ConnectionError as e:
        # Chain the original exception with `from e`
        raise PaymentServiceUnavailable(
            f"Payment gateway unavailable"
        ) from e
```

### Exception Handler Decorator
```python
import functools
import logging
from typing import TypeVar, Callable, Any

logger = logging.getLogger(__name__)
T = TypeVar('T')

def handle_exceptions(
    default: T = None,
    exceptions: tuple = (Exception,),
    log_level: int = logging.ERROR
) -> Callable:
    """Decorator for consistent exception handling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger.log(
                    log_level,
                    f"Exception in {func.__name__}: {e}",
                    exc_info=True
                )
                return default
        return wrapper
    return decorator

# Usage
@handle_exceptions(default=[], exceptions=(ValueError, KeyError))
def parse_config(data: dict) -> list:
    return data["items"]
```

### Context Manager for Error Boundaries
```python
from contextlib import contextmanager
from typing import Optional, Type
import logging

logger = logging.getLogger(__name__)

@contextmanager
def error_boundary(
    *exceptions: Type[Exception],
    default: Any = None,
    reraise: bool = False
):
    """Context manager that catches and logs exceptions."""
    try:
        yield
    except exceptions as e:
        logger.exception(f"Error boundary caught: {e}")
        if reraise:
            raise
        return default

# Usage
with error_boundary(ValueError, TypeError, default="fallback"):
    result = risky_operation()
```

### Custom Exception Hierarchy
```python
class ServiceError(Exception):
    """Base exception for service layer."""
    def __init__(self, message: str, code: str = None):
        super().__init__(message)
        self.code = code

class ValidationError(ServiceError):
    """Input validation failed."""
    pass

class NotFoundError(ServiceError):
    """Resource not found."""
    pass

class ConflictError(ServiceError):
    """Resource state conflict."""
    pass

class ExternalServiceError(ServiceError):
    """External dependency failed."""
    def __init__(self, message: str, service: str, original: Exception = None):
        super().__init__(message, code="EXTERNAL_ERROR")
        self.service = service
        self.original = original
```

### Structured Logging with Exceptions
```python
import structlog

logger = structlog.get_logger()

def process_order(order_id: str):
    try:
        order = fetch_order(order_id)
        validate_order(order)
        return fulfill_order(order)
    except ValidationError as e:
        logger.warning(
            "order_validation_failed",
            order_id=order_id,
            error=str(e),
            error_code=e.code
        )
        raise
    except ExternalServiceError as e:
        logger.error(
            "external_service_failed",
            order_id=order_id,
            service=e.service,
            error=str(e),
            exc_info=True
        )
        raise
```

## 4. Combined Patterns

### Tenacity + Structured Logging
```python
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

logger = structlog.get_logger()

def log_attempt(retry_state):
    logger.info(
        "retry_attempt",
        function=retry_state.fn.__name__,
        attempt=retry_state.attempt_number,
        outcome=str(retry_state.outcome) if retry_state.outcome else None
    )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=10),
    after=log_attempt,
    reraise=True
)
def resilient_api_call(endpoint: str):
    response = requests.get(endpoint, timeout=5)
    response.raise_for_status()
    return response.json()
```

### Result Type + Tenacity
```python
from returns.result import Result, Success, Failure, safe
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(3), reraise=True)
@safe
def fetch_with_retry(url: str) -> dict:
    """Retries up to 3 times, returns Result."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()

# Returns Success(data) or Failure(exception)
result = fetch_with_retry("https://api.example.com/data")
```

### Complete Error Handling Pipeline
```python
from tenacity import retry, stop_after_attempt, wait_exponential
from returns.result import Result, Success, Failure, safe
import structlog

logger = structlog.get_logger()

class PaymentProcessor:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=30),
        reraise=True
    )
    async def _charge_gateway(self, amount: float) -> str:
        """Low-level gateway call with retry."""
        return await self.gateway.charge(amount)
    
    @safe
    async def process_payment(
        self, 
        user_id: str, 
        amount: float
    ) -> str:
        """Process payment, returns Result."""
        # Validate
        if amount <= 0:
            raise ValueError("Amount must be positive")
        
        # Get user (may fail)
        user = await self.get_user(user_id)
        if not user.payment_method:
            raise ValueError("No payment method")
        
        # Charge with retry
        transaction_id = await self._charge_gateway(amount)
        
        logger.info(
            "payment_processed",
            user_id=user_id,
            amount=amount,
            transaction_id=transaction_id
        )
        return transaction_id

# Usage
result = await processor.process_payment("user123", 99.99)
match result:
    case Success(txn_id):
        return {"status": "success", "transaction_id": txn_id}
    case Failure(error):
        return {"status": "error", "message": str(error)}
```

## Quick Reference

### When to Use Each Pattern

| Pattern | Use Case |
|---------|----------|
| **Tenacity @retry** | Transient failures (network, rate limits) |
| **Returns Result** | Expected failures (validation, not found) |
| **Custom exceptions** | Domain-specific error hierarchy |
| **Exception chaining** | Wrapping lower-level errors |
| **Error boundary** | Graceful degradation |
| **Structured logging** | Production observability |

### Anti-Patterns to Avoid

```python
# BAD: Bare except
try:
    risky()
except:  # Catches SystemExit, KeyboardInterrupt!
    pass

# BAD: Swallowing exceptions silently
try:
    risky()
except Exception:
    pass  # No logging, no reraise

# BAD: Catching too broadly
try:
    data = json.loads(input_str)
    result = process(data)
    save_to_db(result)
except Exception as e:
    print(f"Error: {e}")  # Which step failed?

# BAD: Using exceptions for flow control
try:
    value = cache.get(key)
except KeyError:
    value = compute(key)
# Better: if key in cache: ...

# BAD: Infinite retry without backoff
@retry()  # Will hammer the server!
def api_call(): ...
```

## Version Reference

- **tenacity**: 8.5.0+ (latest stable)
- **returns**: 0.23.0+ (latest stable)  
- **Python**: 3.10+ (for pattern matching)
