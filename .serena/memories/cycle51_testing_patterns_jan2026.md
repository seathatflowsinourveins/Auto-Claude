# Cycle 51: Testing Patterns - pytest, Hypothesis, unittest.mock
## Research Date: January 2026 | Sources: Official Documentation

---

## 1. PYTEST FIXTURES (docs.pytest.org/en/stable/)

### Basic Fixture Pattern
```python
import pytest

@pytest.fixture
def sample_data():
    """Simple fixture returning test data."""
    return {"id": 1, "name": "test"}

def test_using_fixture(sample_data):
    """Fixtures injected via function argument."""
    assert sample_data["id"] == 1
```

### Fixture Scopes
```python
# Scope options: function (default), class, module, package, session
@pytest.fixture(scope="module")
def db_connection():
    """Expensive setup done once per module."""
    conn = create_database_connection()
    yield conn  # yield = setup/teardown pattern
    conn.close()

@pytest.fixture(scope="session")
def app():
    """App instance shared across all tests in session."""
    return create_app(testing=True)

# Dynamic scope based on config
@pytest.fixture(scope=lambda fixture_name, config: "session" if config.getoption("--reuse-db") else "function")
def dynamic_scope_fixture():
    pass
```

### Yield Fixtures (Recommended Teardown Pattern)
```python
@pytest.fixture
def managed_resource():
    """Setup before yield, teardown after."""
    resource = acquire_resource()
    yield resource
    # Teardown runs even if test fails
    resource.cleanup()

# Multiple resources with safe teardown
@pytest.fixture
def complex_resource():
    resources = []
    try:
        res1 = create_resource_1()
        resources.append(res1)
        res2 = create_resource_2()
        resources.append(res2)
        yield res1, res2
    finally:
        for res in reversed(resources):
            res.close()
```

### Autouse Fixtures
```python
@pytest.fixture(autouse=True)
def setup_logging():
    """Runs automatically for all tests in scope."""
    logging.basicConfig(level=logging.DEBUG)

@pytest.fixture(autouse=True)
def reset_environment():
    """Reset env vars after each test."""
    original = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original)
```

### Parametrized Fixtures
```python
@pytest.fixture(params=["mysql", "postgres", "sqlite"])
def database(request):
    """Run test with each database type."""
    db = create_engine(request.param)
    yield db
    db.dispose()

@pytest.fixture(params=[
    pytest.param("fast", marks=pytest.mark.fast),
    pytest.param("slow", marks=pytest.mark.slow),
])
def config(request):
    return load_config(request.param)
```

### Factory Fixtures
```python
@pytest.fixture
def make_user():
    """Factory pattern for creating test objects."""
    created_users = []
    
    def _make_user(name, email=None):
        user = User(name=name, email=email or f"{name}@test.com")
        created_users.append(user)
        return user
    
    yield _make_user
    
    # Cleanup all created users
    for user in created_users:
        user.delete()

def test_multiple_users(make_user):
    admin = make_user("admin", "admin@test.com")
    guest = make_user("guest")
    assert admin.name != guest.name
```

### Fixtures Requesting Other Fixtures
```python
@pytest.fixture
def user(db_connection):
    """Fixtures can depend on other fixtures."""
    return db_connection.create_user("test")

@pytest.fixture
def authenticated_client(app, user):
    """Chain of dependencies resolved automatically."""
    client = app.test_client()
    client.login(user)
    return client
```

---

## 2. PYTEST PARAMETRIZE (docs.pytest.org/en/stable/how-to/parametrize.html)

### Basic Parametrize
```python
@pytest.mark.parametrize("input,expected", [
    ("3+5", 8),
    ("2+4", 6),
    ("6*9", 54),
])
def test_eval(input, expected):
    assert eval(input) == expected
```

### With pytest.param for Marks
```python
@pytest.mark.parametrize("test_input,expected", [
    ("3+5", 8),
    pytest.param("6*9", 42, marks=pytest.mark.xfail(reason="known bug")),
    pytest.param("slow_op()", 100, marks=pytest.mark.slow),
    pytest.param("", 0, marks=pytest.mark.skip(reason="empty not supported")),
])
def test_eval_with_marks(test_input, expected):
    assert eval(test_input) == expected
```

### Stacked Parametrize (Combinations)
```python
@pytest.mark.parametrize("x", [0, 1, 2])
@pytest.mark.parametrize("y", [10, 20])
def test_combinations(x, y):
    """Runs 6 times: (0,10), (0,20), (1,10), (1,20), (2,10), (2,20)"""
    assert x + y >= x

# Equivalent to:
@pytest.mark.parametrize("x,y", [
    (0, 10), (0, 20), (1, 10), (1, 20), (2, 10), (2, 20)
])
def test_explicit_combinations(x, y):
    pass
```

### IDs for Readable Test Names
```python
@pytest.mark.parametrize("user,expected", [
    pytest.param({"role": "admin"}, True, id="admin-can-access"),
    pytest.param({"role": "guest"}, False, id="guest-denied"),
])
def test_access(user, expected):
    assert check_access(user) == expected
```

### Dynamic Parametrization with Hook
```python
# conftest.py
def pytest_generate_tests(metafunc):
    """Generate test cases dynamically."""
    if "db_url" in metafunc.fixturenames:
        urls = load_database_urls_from_config()
        metafunc.parametrize("db_url", urls)
```

### Class-Level Parametrize
```python
@pytest.mark.parametrize("browser", ["chrome", "firefox"])
class TestBrowser:
    def test_navigate(self, browser):
        pass
    
    def test_click(self, browser):
        pass
    # Both tests run for chrome AND firefox
```

---

## 3. HYPOTHESIS (hypothesis.readthedocs.io 6.150.2)

### Basic Property-Based Testing
```python
from hypothesis import given, assume, settings, example
from hypothesis import strategies as st

@given(st.integers(), st.integers())
def test_addition_commutative(x, y):
    """Hypothesis generates random x, y values."""
    assert x + y == y + x

@given(st.lists(st.integers()))
def test_sorted_is_sorted(lst):
    """Test with randomly generated lists."""
    result = sorted(lst)
    assert all(result[i] <= result[i+1] for i in range(len(result)-1))
```

### Built-in Strategies
```python
from hypothesis import strategies as st

# Primitives
st.integers()                    # Any integer
st.integers(min_value=0, max_value=100)  # Bounded
st.floats(allow_nan=False)       # Floats without NaN
st.booleans()                    # True/False
st.text()                        # Unicode strings
st.text(min_size=1, max_size=100)  # Bounded length
st.binary()                      # Bytes

# Collections
st.lists(st.integers())          # List of ints
st.lists(st.integers(), min_size=1, max_size=10)  # Bounded
st.tuples(st.integers(), st.text())  # Fixed structure
st.dictionaries(st.text(), st.integers())  # Dict
st.frozensets(st.integers())     # Immutable set

# Choice
st.one_of(st.integers(), st.text())  # Either type
st.integers() | st.text()        # Same as one_of
st.sampled_from(["a", "b", "c"]) # Random choice
st.just("fixed")                 # Exact value

# Building objects
st.builds(User, name=st.text(), age=st.integers(0, 150))
st.from_type(MyDataclass)        # From type hints
```

### Filtering and Assuming
```python
# Filter strategy (can slow tests if filter rejects often)
@given(st.integers().filter(lambda n: n % 2 == 0))
def test_even_numbers(n):
    assert n % 2 == 0

# assume() inside test (preferred for complex conditions)
@given(st.integers(), st.integers())
def test_division(x, y):
    assume(y != 0)  # Skip if y is 0
    result = x / y
    assert result * y == pytest.approx(x)
```

### Composite Strategies
```python
@st.composite
def ordered_pairs(draw):
    """Generate (a, b) where a <= b."""
    a = draw(st.integers())
    b = draw(st.integers(min_value=a))
    return (a, b)

@given(ordered_pairs())
def test_order(pair):
    a, b = pair
    assert a <= b

@st.composite
def valid_user(draw):
    """Generate valid User objects."""
    name = draw(st.text(min_size=1, max_size=50))
    email = draw(st.emails())
    age = draw(st.integers(min_value=0, max_value=150))
    return User(name=name, email=email, age=age)
```

### Settings and Examples
```python
from hypothesis import settings, Phase

@settings(
    max_examples=500,           # More test cases
    deadline=None,              # No time limit per test
    suppress_health_check=[],   # Health check config
)
@given(st.text())
def test_with_settings(s):
    pass

# Force specific test cases
@example("")                    # Always test empty string
@example("edge case")           # Always test this value
@given(st.text())
def test_with_examples(s):
    pass
```

### Stateful Testing
```python
from hypothesis.stateful import RuleBasedStateMachine, rule, precondition

class DatabaseStateMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.db = Database()
        self.model = {}  # Python dict as reference
    
    @rule(key=st.text(), value=st.integers())
    def insert(self, key, value):
        self.db.insert(key, value)
        self.model[key] = value
    
    @rule(key=st.text())
    @precondition(lambda self: len(self.model) > 0)
    def get(self, key):
        if key in self.model:
            assert self.db.get(key) == self.model[key]

TestDatabase = DatabaseStateMachine.TestCase
```

---

## 4. UNITTEST.MOCK (Python 3.14.2 stdlib)

### Basic Mock and MagicMock
```python
from unittest.mock import Mock, MagicMock

# Basic mock
mock = Mock()
mock.method(1, 2, 3)
mock.method.assert_called_once_with(1, 2, 3)

# MagicMock includes magic methods (__len__, __iter__, etc.)
magic = MagicMock()
magic.__len__.return_value = 5
assert len(magic) == 5
```

### Return Values and Side Effects
```python
# Simple return value
mock = Mock(return_value=42)
assert mock() == 42

# Different return for each call
mock = Mock(side_effect=[1, 2, 3])
assert mock() == 1
assert mock() == 2
assert mock() == 3

# Raise exception
mock = Mock(side_effect=ValueError("error"))
with pytest.raises(ValueError):
    mock()

# Dynamic side effect function
def dynamic(arg):
    return arg * 2
mock = Mock(side_effect=dynamic)
assert mock(5) == 10
```

### patch() Decorator and Context Manager
```python
from unittest.mock import patch

# As decorator
@patch('module.ClassName')
def test_class(MockClass):
    instance = MockClass.return_value
    instance.method.return_value = 'result'
    # Code under test uses mocked class

# As context manager
def test_context():
    with patch('module.function') as mock_func:
        mock_func.return_value = 'mocked'
        result = module.function()
        assert result == 'mocked'

# Patching object attributes
@patch.object(SomeClass, 'method')
def test_method(mock_method):
    mock_method.return_value = 'mocked'
```

### spec and autospec for Type Safety
```python
from unittest.mock import create_autospec

# Autospec ensures mock matches real object's interface
mock_db = create_autospec(DatabaseConnection)
mock_db.query("SELECT *")  # OK
mock_db.nonexistent_method()  # Raises AttributeError!

# With patch
@patch('module.DatabaseConnection', autospec=True)
def test_with_autospec(MockDB):
    instance = MockDB.return_value
    instance.query.return_value = []
```

### Assertion Methods
```python
mock = Mock()

mock(1, 2, key='value')

mock.assert_called()                      # Called at least once
mock.assert_called_once()                 # Exactly once
mock.assert_called_with(1, 2, key='value')  # Last call args
mock.assert_called_once_with(1, 2, key='value')  # Once with args
mock.assert_not_called()                  # Never called

# Inspect calls
assert mock.call_count == 1
assert mock.call_args == ((1, 2), {'key': 'value'})
assert mock.call_args_list == [((1, 2), {'key': 'value'})]
```

### AsyncMock for Async Functions
```python
from unittest.mock import AsyncMock

async def test_async():
    mock = AsyncMock(return_value=42)
    result = await mock()
    assert result == 42
    mock.assert_awaited_once()

# Patching async function
@patch('module.async_function', new_callable=AsyncMock)
async def test_patched_async(mock_func):
    mock_func.return_value = 'result'
    result = await module.async_function()
    assert result == 'result'
```

### PropertyMock
```python
from unittest.mock import PropertyMock

class MyClass:
    @property
    def name(self):
        return "real"

with patch.object(MyClass, 'name', new_callable=PropertyMock) as mock_name:
    mock_name.return_value = 'mocked'
    obj = MyClass()
    assert obj.name == 'mocked'
```

---

## 5. INTEGRATION PATTERNS

### Hypothesis + pytest Fixtures
```python
@pytest.fixture
def db():
    return Database()

@given(st.text(min_size=1))
def test_with_fixture(db, key):
    """Fixtures work with @given."""
    db.set(key, "value")
    assert db.get(key) == "value"
```

### Hypothesis + pytest.mark.parametrize
```python
@pytest.mark.parametrize("multiplier", [1, 2, 3])
@given(st.integers())
def test_combined(multiplier, n):
    """Parametrize for fixed values, given for generated."""
    result = n * multiplier
    assert result == n * multiplier
```

### Mock + pytest Fixtures
```python
@pytest.fixture
def mock_api():
    with patch('module.api_client') as mock:
        mock.fetch.return_value = {"data": "test"}
        yield mock

def test_with_mock_fixture(mock_api):
    result = service.get_data()
    mock_api.fetch.assert_called_once()
    assert result["data"] == "test"
```

---

## 6. BEST PRACTICES SUMMARY

### Fixture Best Practices
1. **Use yield fixtures** for cleanup instead of finalizers
2. **Scope appropriately** - session for expensive, function for isolation
3. **Factory fixtures** for creating multiple instances
4. **Conftest.py** for shared fixtures across modules
5. **Avoid autouse=True** unless truly needed everywhere

### Parametrize Best Practices
1. **Use IDs** for readable test output
2. **pytest.param with marks** for expected failures
3. **Don't over-parametrize** - keep combinations manageable
4. **Stack decorators** for cartesian products

### Hypothesis Best Practices
1. **Start simple** - basic strategies first
2. **Use assume()** over .filter() for complex conditions
3. **@example()** for known edge cases
4. **Composite strategies** for domain objects
5. **Settings profiles** for CI vs local

### Mock Best Practices
1. **Always use autospec** to catch interface changes
2. **Patch where used**, not where defined
3. **Assert calls explicitly** - don't assume
4. **AsyncMock** for async code
5. **Keep mocks minimal** - only mock what's needed

---

## Quick Reference Table

| Tool | Primary Use | Key Pattern |
|------|-------------|-------------|
| pytest fixtures | Test setup/teardown | `@pytest.fixture` + yield |
| pytest parametrize | Multiple inputs | `@pytest.mark.parametrize` |
| Hypothesis | Property-based | `@given(strategy)` |
| unittest.mock | Isolate dependencies | `@patch` + autospec |

**Version Info:**
- pytest: 8.x (2026)
- Hypothesis: 6.150.2
- unittest.mock: Python 3.14.2 stdlib
