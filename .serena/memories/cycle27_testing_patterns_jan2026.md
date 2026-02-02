# Cycle 27: Testing Patterns - January 2026

## Research Focus
pytest best practices, fixtures, property-based testing (Hypothesis), mocking strategies.

---

## 1. pytest Fundamentals

### Project Structure
```
project/
├── src/
│   └── mypackage/
│       └── module.py
├── tests/
│   ├── conftest.py       # Shared fixtures
│   ├── unit/
│   │   └── test_module.py
│   └── integration/
│       └── test_api.py
└── pyproject.toml
```

### Basic Test Pattern
```python
import pytest
from mypackage.module import calculate

class TestCalculate:
    """Group related tests in classes."""
    
    def test_add_positive_numbers(self):
        assert calculate(2, 3) == 5
    
    def test_add_negative_numbers(self):
        assert calculate(-2, -3) == -5
    
    def test_invalid_input_raises(self):
        with pytest.raises(TypeError, match="must be numeric"):
            calculate("a", 1)
```

### Parametrized Tests
```python
@pytest.mark.parametrize("a,b,expected", [
    (2, 3, 5),
    (-1, 1, 0),
    (0, 0, 0),
    (100, 200, 300),
])
def test_add_various_inputs(a, b, expected):
    assert calculate(a, b) == expected

# Multiple parameter sets
@pytest.mark.parametrize("x", [1, 2, 3])
@pytest.mark.parametrize("y", [10, 20])
def test_matrix(x, y):  # Runs 6 times (3 * 2)
    assert x + y > 0
```

---

## 2. pytest Fixtures

### Fixture Scopes
```python
import pytest

@pytest.fixture(scope="function")  # Default: new per test
def db_connection():
    conn = create_connection()
    yield conn
    conn.close()

@pytest.fixture(scope="class")  # Shared per test class
def expensive_resource():
    return load_expensive_resource()

@pytest.fixture(scope="module")  # Shared per module
def module_setup():
    return setup_module_state()

@pytest.fixture(scope="session")  # Shared across entire test run
def database():
    db = create_database()
    yield db
    db.drop()
```

### Fixture Factories
```python
@pytest.fixture
def make_user():
    """Factory fixture for creating users with custom attributes."""
    created_users = []
    
    def _make_user(name="Test", email=None):
        email = email or f"{name.lower()}@test.com"
        user = User(name=name, email=email)
        created_users.append(user)
        return user
    
    yield _make_user
    
    # Cleanup all created users
    for user in created_users:
        user.delete()

def test_user_creation(make_user):
    user1 = make_user("Alice")
    user2 = make_user("Bob", email="bob@custom.com")
    assert user1.email == "alice@test.com"
```

### Async Fixtures
```python
import pytest
import pytest_asyncio

@pytest_asyncio.fixture
async def async_client():
    async with AsyncClient(app, base_url="http://test") as client:
        yield client

@pytest.mark.asyncio
async def test_async_endpoint(async_client):
    response = await async_client.get("/health")
    assert response.status_code == 200
```

### conftest.py (Shared Fixtures)
```python
# tests/conftest.py
import pytest

@pytest.fixture(autouse=True)
def reset_database():
    """Automatically runs before each test."""
    db.reset()
    yield
    db.cleanup()

@pytest.fixture
def auth_headers(make_user):
    user = make_user("admin")
    token = create_token(user)
    return {"Authorization": f"Bearer {token}"}
```

---

## 3. Hypothesis Property-Based Testing

### Basic Property Tests
```python
from hypothesis import given, strategies as st

@given(st.integers(), st.integers())
def test_addition_commutative(x, y):
    """Property: x + y == y + x for all integers."""
    assert x + y == y + x

@given(st.lists(st.integers()))
def test_sorted_is_idempotent(lst):
    """Property: sorting twice gives same result."""
    assert sorted(sorted(lst)) == sorted(lst)

@given(st.text(min_size=1))
def test_reverse_reverse_identity(s):
    """Property: reversing twice returns original."""
    assert s[::-1][::-1] == s
```

### Built-in Strategies
```python
from hypothesis import strategies as st

# Primitives
st.integers(min_value=0, max_value=100)
st.floats(allow_nan=False, allow_infinity=False)
st.text(alphabet="abc", min_size=1, max_size=10)
st.booleans()
st.none()

# Collections
st.lists(st.integers(), min_size=1, max_size=50)
st.dictionaries(st.text(), st.integers())
st.tuples(st.integers(), st.text())
st.sets(st.integers())

# Complex
st.emails()
st.datetimes()
st.uuids()
st.binary()

# Combining
st.one_of(st.integers(), st.text())  # Union
st.integers() | st.text()             # Same as above
```

### Custom Strategies
```python
from hypothesis import strategies as st
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: str

# Strategy from Pydantic model
user_strategy = st.builds(
    User,
    name=st.text(min_size=1, max_size=50),
    age=st.integers(min_value=0, max_value=150),
    email=st.emails()
)

@given(user_strategy)
def test_user_serialization(user):
    data = user.model_dump()
    reconstructed = User(**data)
    assert reconstructed == user
```

### Stateful Testing (Rule-Based)
```python
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant
import hypothesis.strategies as st

class ShoppingCartMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.cart = ShoppingCart()
        self.items = {}  # Model: item_id -> quantity
    
    @rule(item_id=st.integers(1, 100), quantity=st.integers(1, 10))
    def add_item(self, item_id, quantity):
        self.cart.add(item_id, quantity)
        self.items[item_id] = self.items.get(item_id, 0) + quantity
    
    @rule(item_id=st.integers(1, 100))
    def remove_item(self, item_id):
        if item_id in self.items:
            self.cart.remove(item_id)
            del self.items[item_id]
    
    @invariant()
    def cart_matches_model(self):
        """Invariant: real cart always matches our model."""
        assert self.cart.total_items() == sum(self.items.values())

# Run the state machine
TestShoppingCart = ShoppingCartMachine.TestCase
```

### Hypothesis Settings
```python
from hypothesis import settings, Verbosity, Phase

@settings(
    max_examples=500,           # More thorough testing
    deadline=1000,              # 1 second per example
    verbosity=Verbosity.verbose,
    suppress_health_check=[],
    phases=[Phase.generate, Phase.shrink],  # Skip reuse
)
@given(st.integers())
def test_with_custom_settings(x):
    assert True
```

---

## 4. Mocking with pytest-mock

### Basic Mocking
```python
def test_external_api(mocker):
    # Patch the function
    mock_fetch = mocker.patch('mymodule.fetch_data')
    mock_fetch.return_value = {"status": "ok"}
    
    result = process_data()
    
    # Assertions
    mock_fetch.assert_called_once()
    assert result["status"] == "ok"
```

### Mocking Objects
```python
def test_service(mocker):
    # Mock an object's method
    mock_db = mocker.patch.object(Database, 'query')
    mock_db.return_value = [{"id": 1, "name": "Test"}]
    
    service = UserService(Database())
    users = service.get_all()
    
    assert len(users) == 1
    mock_db.assert_called_once_with("SELECT * FROM users")
```

### Side Effects
```python
def test_retry_logic(mocker):
    # First two calls fail, third succeeds
    mock_api = mocker.patch('mymodule.api_call')
    mock_api.side_effect = [
        ConnectionError("Failed"),
        ConnectionError("Failed again"),
        {"status": "success"}
    ]
    
    result = call_with_retry(max_retries=3)
    
    assert result["status"] == "success"
    assert mock_api.call_count == 3

def test_exception(mocker):
    mock_api = mocker.patch('mymodule.api_call')
    mock_api.side_effect = ValueError("Invalid input")
    
    with pytest.raises(ValueError):
        process()
```

### Spying (Track Real Calls)
```python
def test_spy(mocker):
    # Spy on real method (doesn't replace it)
    spy = mocker.spy(MyClass, 'method')
    
    obj = MyClass()
    result = obj.method(42)  # Real method called
    
    spy.assert_called_once_with(42)
    assert result == expected_real_result
```

### Async Mocking
```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_async_function(mocker):
    mock_fetch = mocker.patch('mymodule.async_fetch', new_callable=AsyncMock)
    mock_fetch.return_value = {"data": "test"}
    
    result = await process_async()
    
    mock_fetch.assert_awaited_once()
    assert result["data"] == "test"
```

### Where to Patch (Critical!)
```python
# mymodule.py
from external import fetch_data  # Imported into mymodule

def process():
    return fetch_data()

# test_mymodule.py
def test_process(mocker):
    # WRONG: Patches where defined
    # mocker.patch('external.fetch_data')
    
    # CORRECT: Patch where USED
    mocker.patch('mymodule.fetch_data')
```

---

## 5. Integration Testing Patterns

### Database Testing
```python
@pytest.fixture
async def db_session():
    """Create isolated database session per test."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with async_session() as session:
        yield session
        await session.rollback()  # Rollback after each test

@pytest.mark.asyncio
async def test_user_crud(db_session):
    user = User(name="Test")
    db_session.add(user)
    await db_session.flush()
    
    result = await db_session.get(User, user.id)
    assert result.name == "Test"
```

### API Testing with httpx
```python
from httpx import AsyncClient

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.mark.asyncio
async def test_create_user(client, auth_headers):
    response = await client.post(
        "/users",
        json={"name": "Test", "email": "test@example.com"},
        headers=auth_headers
    )
    assert response.status_code == 201
    assert response.json()["name"] == "Test"
```

---

## 6. Test Organization Best Practices

### Markers for Categorization
```python
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow",
    "integration: marks integration tests",
    "unit: marks unit tests",
]

# In tests
@pytest.mark.slow
def test_heavy_computation():
    pass

@pytest.mark.integration
def test_database_connection():
    pass

# Run specific markers
# pytest -m "not slow"
# pytest -m "unit"
```

### Coverage Configuration
```toml
# pyproject.toml
[tool.coverage.run]
branch = true
source = ["src"]
omit = ["tests/*", "*/__init__.py"]

[tool.coverage.report]
fail_under = 80
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
```

### Parallel Execution
```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest -n auto  # Auto-detect CPU cores
pytest -n 4     # Use 4 workers
```

---

## Quick Reference

| Pattern | Use Case |
|---------|----------|
| `@pytest.fixture(scope="session")` | Expensive setup (DB) |
| `@pytest.fixture(autouse=True)` | Always run (cleanup) |
| Factory fixtures | Multiple instances with variations |
| `@pytest.mark.parametrize` | Test multiple inputs |
| `@given(st.integers())` | Property-based testing |
| `RuleBasedStateMachine` | Stateful property tests |
| `mocker.patch('where.used')` | Mock dependencies |
| `mocker.spy()` | Track real calls |
| `AsyncMock` | Mock async functions |
| `pytest -n auto` | Parallel execution |

---

*Cycle 27 Complete - Testing Patterns Documented*
*Next: Cycle 28 - CLI & DevOps Patterns*
