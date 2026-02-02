# Cycle 26: Validation & Data Modeling Patterns - January 2026

## Research Focus
Pydantic v2 validation patterns, data contracts, schema evolution, input security.

---

## 1. Pydantic v2 Validator Types

### Validator Hierarchy (Execution Order)
```
1. @field_validator(mode='before')  # Raw input → transform
2. Type coercion (automatic)        # str → int, etc.
3. @field_validator(mode='after')   # Post-coercion validation
4. @model_validator(mode='before')  # Cross-field BEFORE parsing
5. @model_validator(mode='after')   # Cross-field AFTER all fields valid
```

### Field Validators
```python
from pydantic import BaseModel, field_validator, ValidationInfo

class User(BaseModel):
    email: str
    age: int
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()  # Normalize
    
    @field_validator('age', mode='before')
    @classmethod
    def parse_age(cls, v):
        """Transform before type coercion."""
        if isinstance(v, str):
            return int(v.strip())
        return v
    
    @field_validator('age')
    @classmethod
    def validate_age_range(cls, v: int, info: ValidationInfo) -> int:
        """Access other fields via info.data."""
        if v < 0 or v > 150:
            raise ValueError('Age must be 0-150')
        return v
```

### Model Validators (Cross-Field)
```python
from pydantic import BaseModel, model_validator

class DateRange(BaseModel):
    start_date: date
    end_date: date
    
    @model_validator(mode='after')
    def validate_date_order(self) -> 'DateRange':
        """Runs after all fields are validated."""
        if self.end_date < self.start_date:
            raise ValueError('end_date must be >= start_date')
        return self
    
    @model_validator(mode='before')
    @classmethod
    def normalize_input(cls, data: dict) -> dict:
        """Transform raw input before field parsing."""
        if isinstance(data.get('start_date'), str):
            data['start_date'] = parse_date(data['start_date'])
        return data
```

### Computed Fields
```python
from pydantic import BaseModel, computed_field

class Order(BaseModel):
    items: list[Item]
    tax_rate: float = 0.08
    
    @computed_field
    @property
    def subtotal(self) -> float:
        return sum(item.price * item.quantity for item in self.items)
    
    @computed_field
    @property
    def total(self) -> float:
        return self.subtotal * (1 + self.tax_rate)
```

### Annotated Validators (Reusable)
```python
from typing import Annotated
from pydantic import AfterValidator, BeforeValidator

def strip_whitespace(v: str) -> str:
    return v.strip()

def validate_not_empty(v: str) -> str:
    if not v:
        raise ValueError('Cannot be empty')
    return v

# Reusable type with validation
CleanString = Annotated[str, BeforeValidator(strip_whitespace), AfterValidator(validate_not_empty)]

class Profile(BaseModel):
    name: CleanString  # Applies both validators
    bio: CleanString
```

---

## 2. Strict Mode vs Lax Mode

### Default (Lax) Mode
```python
class User(BaseModel):
    age: int

User(age="25")  # OK: "25" → 25 (coerced)
User(age=25.9)  # OK: 25.9 → 25 (truncated)
```

### Strict Mode (No Coercion)
```python
from pydantic import BaseModel, ConfigDict

class StrictUser(BaseModel):
    model_config = ConfigDict(strict=True)
    age: int

StrictUser(age="25")  # ERROR: str not allowed
StrictUser(age=25)    # OK
```

### Per-Field Strict
```python
from pydantic import BaseModel, Field
from typing import Annotated
from pydantic import Strict

class MixedModel(BaseModel):
    loose_int: int                           # Allows coercion
    strict_int: Annotated[int, Strict()]     # No coercion
```

---

## 3. Data Contracts & Schema Evolution

### Contract Definition Pattern
```python
from pydantic import BaseModel, Field
from typing import Literal

class UserContractV1(BaseModel):
    """Data contract - machine-checked agreement."""
    schema_version: Literal["1.0"] = "1.0"
    
    user_id: str = Field(..., min_length=1, max_length=50)
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    created_at: datetime
    
    # Contract metadata
    model_config = ConfigDict(
        extra='forbid',  # Reject unknown fields
        frozen=True,     # Immutable after creation
    )
```

### Schema Evolution Rules (Backward Compatible)
```
SAFE Changes (Backward Compatible):
✅ Add optional field with default
✅ Widen type (int → int | float)
✅ Relax validation (min=5 → min=1)
✅ Add new enum values (consumers ignore unknown)

BREAKING Changes:
❌ Remove required field
❌ Rename field
❌ Narrow type (int | float → int)
❌ Tighten validation (min=1 → min=5)
❌ Change field meaning
```

### Versioned Schemas Pattern
```python
from pydantic import BaseModel
from typing import Union

class UserV1(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    name: str

class UserV2(BaseModel):
    schema_version: Literal["2.0"] = "2.0"
    first_name: str  # Breaking: renamed from 'name'
    last_name: str   # Breaking: new required field

# Discriminated union for multi-version support
User = Annotated[Union[UserV1, UserV2], Field(discriminator='schema_version')]

def parse_user(data: dict) -> UserV1 | UserV2:
    return TypeAdapter(User).validate_python(data)
```

### Contract Enforcement
```python
from pydantic import BaseModel, ConfigDict

class StrictContract(BaseModel):
    """Production contract with enforcement."""
    model_config = ConfigDict(
        extra='forbid',       # No extra fields
        strict=True,          # No type coercion
        validate_default=True,# Validate defaults too
        revalidate_instances='always',
    )
```

---

## 4. FastAPI Integration

### Request Validation
```python
from fastapi import FastAPI, Body, Query, Path
from pydantic import BaseModel, Field

app = FastAPI()

class CreateUser(BaseModel):
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    password: str = Field(..., min_length=8)
    age: int = Field(..., ge=0, le=150)

@app.post("/users")
async def create_user(user: CreateUser):
    # user is already validated
    return {"email": user.email}

@app.get("/users/{user_id}")
async def get_user(
    user_id: int = Path(..., gt=0),
    include_orders: bool = Query(False),
):
    return {"user_id": user_id}
```

### Custom Error Responses
```python
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "details": exc.errors(),  # Structured error info
        }
    )
```

---

## 5. Security-Focused Validation

### SQL Injection Prevention
```python
from pydantic import BaseModel, field_validator
import re

class SearchQuery(BaseModel):
    query: str
    
    @field_validator('query')
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        # Reject SQL injection patterns
        dangerous = re.compile(r"(--|;|'|\"|\bOR\b|\bAND\b|\bUNION\b)", re.I)
        if dangerous.search(v):
            raise ValueError('Invalid characters in query')
        return v

# BETTER: Use parameterized queries (SQLAlchemy does this)
# Never concatenate user input into SQL
```

### Path Traversal Prevention
```python
from pydantic import BaseModel, field_validator
from pathlib import Path

class FileRequest(BaseModel):
    filename: str
    
    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v: str) -> str:
        # Prevent path traversal
        if '..' in v or v.startswith('/'):
            raise ValueError('Invalid filename')
        # Only allow specific extensions
        allowed = {'.txt', '.pdf', '.jpg'}
        if Path(v).suffix.lower() not in allowed:
            raise ValueError(f'Extension not allowed: {Path(v).suffix}')
        return v
```

### XSS Prevention (Output Encoding)
```python
import html
from pydantic import BaseModel, field_validator

class Comment(BaseModel):
    content: str
    
    @field_validator('content')
    @classmethod
    def escape_html(cls, v: str) -> str:
        # Escape HTML entities for safe rendering
        return html.escape(v)
```

### Rate Limit Fields
```python
from pydantic import BaseModel, Field

class PaginatedRequest(BaseModel):
    """Prevent abuse via validation limits."""
    page: int = Field(default=1, ge=1, le=1000)
    page_size: int = Field(default=20, ge=1, le=100)  # Max 100 per page
    
    # Computed property for offset
    @computed_field
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size
```

---

## 6. Advanced Patterns

### Conditional Validation
```python
from pydantic import BaseModel, model_validator

class Payment(BaseModel):
    method: Literal['card', 'bank_transfer']
    card_number: str | None = None
    bank_account: str | None = None
    
    @model_validator(mode='after')
    def validate_payment_details(self) -> 'Payment':
        if self.method == 'card' and not self.card_number:
            raise ValueError('card_number required for card payments')
        if self.method == 'bank_transfer' and not self.bank_account:
            raise ValueError('bank_account required for bank transfers')
        return self
```

### Generic Validated Types
```python
from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

class PositiveInt(int):
    """Custom type with validation."""
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.int_schema(gt=0)

class Order(BaseModel):
    quantity: PositiveInt  # Must be > 0
```

### Validation Context
```python
from pydantic import BaseModel, field_validator, ValidationInfo

class User(BaseModel):
    role: str
    admin_key: str | None = None
    
    @field_validator('admin_key')
    @classmethod
    def validate_admin_key(cls, v, info: ValidationInfo):
        # Access context passed during validation
        context = info.context or {}
        if context.get('require_admin') and not v:
            raise ValueError('Admin key required in this context')
        return v

# Usage with context
user = User.model_validate(
    {'role': 'admin'},
    context={'require_admin': True}
)
```

---

## Quick Reference

| Pattern | Use Case |
|---------|----------|
| `@field_validator(mode='before')` | Transform raw input |
| `@field_validator(mode='after')` | Validate after coercion |
| `@model_validator(mode='after')` | Cross-field validation |
| `Annotated[T, Validator]` | Reusable validation |
| `ConfigDict(strict=True)` | No type coercion |
| `ConfigDict(extra='forbid')` | Reject unknown fields |
| `Field(pattern=...)` | Regex validation |
| `Field(ge=0, le=100)` | Numeric bounds |
| `Literal['a', 'b']` | Enum-like constraints |
| `computed_field` | Derived values |

---

*Cycle 26 Complete - Validation & Data Modeling Documented*
*Next: Cycle 27 - Testing Patterns*
