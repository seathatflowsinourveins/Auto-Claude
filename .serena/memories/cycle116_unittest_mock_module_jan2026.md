# Python unittest.mock Module - Production Patterns (January 2026)

## Overview
The `unittest.mock` module provides a powerful framework for replacing parts of your system under test with mock objects. Essential for isolating units, testing edge cases, and verifying interactions without side effects.

## Core Classes

### Mock - Basic Mock Object
```python
from unittest.mock import Mock

# Create a mock
mock = Mock()

# Accessing any attribute returns a new Mock
mock.some_attribute  # Returns Mock
mock.some_method()   # Returns Mock

# Setting return values
mock.method.return_value = 42
mock.method()  # 42

# Nested return values
mock.connection.cursor.return_value.fetchall.return_value = [('row',)]

# Call tracking
mock.method(1, 2, key='value')
mock.method.assert_called_with(1, 2, key='value')
mock.method.call_count  # 1
mock.method.call_args   # call(1, 2, key='value')
mock.method.call_args_list  # [call(1, 2, key='value')]
```

### MagicMock - Mock with Magic Methods
```python
from unittest.mock import MagicMock

# MagicMock pre-configures magic methods
mock = MagicMock()

len(mock)      # Works (returns 0)
mock[0]        # Works (returns MagicMock)
iter(mock)     # Works
bool(mock)     # True

# Configure magic methods
mock.__len__.return_value = 5
len(mock)  # 5

mock.__getitem__.return_value = 'item'
mock[0]  # 'item'
```

### AsyncMock - For Async Code (3.8+)
```python
from unittest.mock import AsyncMock
import asyncio

async_mock = AsyncMock()
async_mock.return_value = 42

# Usage in async code
result = await async_mock()  # 42

# With side_effect
async_mock.side_effect = [1, 2, 3]
await async_mock()  # 1
await async_mock()  # 2
```

## The patch() Decorator/Context Manager

### Basic Patching
```python
from unittest.mock import patch

# As decorator
@patch('module.ClassName')
def test_something(MockClass):
    instance = MockClass.return_value
    instance.method.return_value = 'result'
    # Test code here

# As context manager
with patch('module.function') as mock_func:
    mock_func.return_value = 42
    # Test code here

# Patching multiple targets
@patch('module.func2')
@patch('module.func1')
def test_something(mock_func1, mock_func2):
    # Note: decorators applied bottom-up, so params are reversed
    pass
```

### patch.object() - Patch Attributes
```python
from unittest.mock import patch
import mymodule

# Patch attribute on object
with patch.object(mymodule, 'some_function', return_value=42):
    result = mymodule.some_function()  # 42

# Patch method on class
with patch.object(MyClass, 'method', return_value='mocked'):
    obj = MyClass()
    obj.method()  # 'mocked'
```

### patch.dict() - Patch Dictionaries
```python
from unittest.mock import patch
import os

# Patch os.environ
with patch.dict(os.environ, {'API_KEY': 'test-key'}):
    assert os.environ['API_KEY'] == 'test-key'

# Clear and set
with patch.dict(os.environ, {'ONLY_THIS': 'value'}, clear=True):
    assert 'PATH' not in os.environ
```

### patch.multiple() - Multiple Patches
```python
from unittest.mock import patch, DEFAULT

with patch.multiple('module', 
    func1=DEFAULT, 
    func2=Mock(return_value=42)
) as mocks:
    mocks['func1'].return_value = 10
```

## side_effect - Dynamic Behavior

```python
from unittest.mock import Mock

mock = Mock()

# Raise exception
mock.side_effect = ValueError("error!")
mock()  # Raises ValueError

# Return different values on each call
mock.side_effect = [1, 2, 3]
mock()  # 1
mock()  # 2
mock()  # 3

# Callable for dynamic logic
def dynamic_return(*args, **kwargs):
    if args[0] == 'special':
        return 'special_result'
    return 'default'

mock.side_effect = dynamic_return
mock('special')  # 'special_result'
mock('other')    # 'default'

# Mixed: exception then values
mock.side_effect = [ValueError("first"), 42, 43]
```

## Assertions

### Call Assertions
```python
from unittest.mock import Mock, call

mock = Mock()
mock(1, 2, key='value')
mock(3, 4)

# Assert specific calls
mock.assert_called()              # At least once
mock.assert_called_once()         # Exactly once (fails here)
mock.assert_called_with(3, 4)     # Most recent call
mock.assert_called_once_with(...) # Exactly once with args

# Assert any call matches
mock.assert_any_call(1, 2, key='value')

# Assert call sequence
mock.assert_has_calls([
    call(1, 2, key='value'),
    call(3, 4)
], any_order=False)

# Assert not called
mock.assert_not_called()  # Fails if called
```

### Call Object
```python
from unittest.mock import Mock, call

mock = Mock()
mock.method(1, 2)
mock.method.nested(3, key='val')

# Access call info
mock.method.call_args       # call(1, 2)
mock.method.call_args.args  # (1, 2)
mock.method.call_args.kwargs  # {}

# Build expected calls
expected = [
    call.method(1, 2),
    call.method.nested(3, key='val')
]
mock.assert_has_calls(expected)
```

## Spec and Autospec - Type Safety

### spec - Limit Attributes
```python
from unittest.mock import Mock

class RealClass:
    def method(self, arg):
        return arg

# Mock with spec
mock = Mock(spec=RealClass)
mock.method(1)      # Works
mock.nonexistent()  # Raises AttributeError!

# spec_set is stricter (prevents setting new attrs)
mock = Mock(spec_set=RealClass)
mock.new_attr = 1   # Raises AttributeError!
```

### create_autospec - Full Signature Matching
```python
from unittest.mock import create_autospec

def real_function(a, b, c=None):
    pass

mock_func = create_autospec(real_function)
mock_func(1, 2)           # Works
mock_func(1)              # Raises TypeError (missing b)
mock_func(1, 2, 3, 4)     # Raises TypeError (too many args)

# For classes
mock_class = create_autospec(RealClass)
instance = mock_class.return_value
instance.method.return_value = 'mocked'
```

## Special Mocks

### PropertyMock - Mock Properties
```python
from unittest.mock import PropertyMock, patch

class MyClass:
    @property
    def prop(self):
        return 'real'

# Patch the property
with patch.object(MyClass, 'prop', new_callable=PropertyMock) as mock_prop:
    mock_prop.return_value = 'mocked'
    obj = MyClass()
    assert obj.prop == 'mocked'
```

### sentinel - Unique Objects
```python
from unittest.mock import sentinel

# Create unique objects for testing
mock = Mock(return_value=sentinel.SPECIAL_VALUE)
result = mock()
assert result is sentinel.SPECIAL_VALUE  # Identity check
```

### DEFAULT - Placeholder
```python
from unittest.mock import patch, DEFAULT

# Use DEFAULT to auto-create mocks
with patch.multiple('module', func1=DEFAULT, func2=DEFAULT) as mocks:
    # mocks['func1'] and mocks['func2'] are MagicMocks
    pass
```

## Production Patterns

### Pattern 1: Database Mock
```python
from unittest.mock import Mock, patch, MagicMock

def test_user_repository():
    # Mock database connection
    mock_conn = MagicMock()
    mock_cursor = mock_conn.cursor.return_value
    mock_cursor.fetchone.return_value = {'id': 1, 'name': 'Test'}
    
    with patch('myapp.db.get_connection', return_value=mock_conn):
        user = get_user(1)
        
        mock_cursor.execute.assert_called_once()
        assert user['name'] == 'Test'
```

### Pattern 2: HTTP Client Mock
```python
from unittest.mock import patch, Mock
import json

def test_api_client():
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'data': 'test'}
    mock_response.raise_for_status = Mock()
    
    with patch('requests.get', return_value=mock_response) as mock_get:
        result = fetch_data('https://api.example.com')
        
        mock_get.assert_called_once_with(
            'https://api.example.com',
            headers={'Authorization': 'Bearer test'}
        )
        assert result == {'data': 'test'}
```

### Pattern 3: Time-Based Testing
```python
from unittest.mock import patch
from datetime import datetime

def test_time_sensitive():
    fixed_time = datetime(2026, 1, 25, 12, 0, 0)
    
    with patch('mymodule.datetime') as mock_dt:
        mock_dt.now.return_value = fixed_time
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
        
        result = get_current_hour()
        assert result == 12
```

### Pattern 4: File Operations Mock
```python
from unittest.mock import patch, mock_open

def test_file_reader():
    file_content = "line1\nline2\nline3"
    
    with patch('builtins.open', mock_open(read_data=file_content)):
        result = read_file('any_path.txt')
        assert result == ['line1', 'line2', 'line3']

# Multiple files
def test_multiple_files():
    files = {
        'config.json': '{"key": "value"}',
        'data.txt': 'some data'
    }
    
    def open_side_effect(path, *args, **kwargs):
        return mock_open(read_data=files[path])()
    
    with patch('builtins.open', side_effect=open_side_effect):
        config = read_config('config.json')
        data = read_data('data.txt')
```

### Pattern 5: Async HTTP Mock
```python
from unittest.mock import AsyncMock, patch

async def test_async_api():
    mock_response = AsyncMock()
    mock_response.json = AsyncMock(return_value={'status': 'ok'})
    mock_response.status = 200
    
    mock_session = AsyncMock()
    mock_session.get.return_value.__aenter__.return_value = mock_response
    
    with patch('aiohttp.ClientSession', return_value=mock_session):
        result = await fetch_async_data()
        assert result['status'] == 'ok'
```

### Pattern 6: Context Manager Mock
```python
from unittest.mock import MagicMock, patch

def test_context_manager():
    mock_file = MagicMock()
    mock_file.__enter__.return_value = mock_file
    mock_file.__exit__.return_value = False
    mock_file.read.return_value = 'content'
    
    with patch('builtins.open', return_value=mock_file):
        with open('file.txt') as f:
            assert f.read() == 'content'
```

## Where to Patch

```python
# WRONG: Patch where defined
# mymodule.py: from os import listdir
@patch('os.listdir')  # Won't work!

# RIGHT: Patch where used
@patch('mymodule.listdir')  # Patches the reference in mymodule

# Rule: patch where the name is looked up, not where it's defined
```

## Best Practices

1. **Use spec/autospec** to catch interface changes
2. **Patch at the right location** (where used, not defined)
3. **Keep mocks focused** - mock only what's necessary
4. **Verify calls** - don't just set return values
5. **Use context managers** for cleaner test code
6. **Reset mocks** between tests with `mock.reset_mock()`

## Common Pitfalls

```python
# WRONG: Forgetting return_value for instance methods
mock_class = Mock()
mock_class.method.return_value = 'value'  # On class, not instance

# RIGHT: Access instance via return_value
mock_class.return_value.method.return_value = 'value'

# WRONG: Patching too broadly
@patch('requests')  # Patches entire module

# RIGHT: Patch specific function
@patch('requests.get')

# WRONG: Not resetting between tests
# mock.reset_mock() or use fresh mocks per test
```

## Version History
- 3.8: Added AsyncMock, seal()
- 3.9: Added call_args.args and call_args.kwargs
- 3.12: Performance improvements
