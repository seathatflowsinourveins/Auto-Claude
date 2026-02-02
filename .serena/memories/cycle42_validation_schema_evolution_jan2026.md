# Cycle 42: Data Validation & Schema Evolution (January 2026)

## Overview
Production patterns for runtime validation, type safety at boundaries, and graceful schema evolution.
Covers Pydantic v2 advanced patterns, Zod for TypeScript, and schema compatibility strategies.

---

## 1. The Validation Boundary Problem

### TypeScript's Runtime Blind Spot
```
Compile-Time Safety ≠ Runtime Safety

TypeScript types DISAPPEAR at runtime:
- API responses can return anything
- User input is untrusted
- External systems don't respect your interfaces
- Database queries return unknown shapes
```

### The Boundary Rule
```
TRUST NOTHING at system boundaries:
├── API responses (external services)
├── User input (forms, uploads)
├── Database results (dynamic queries)
├── Message queue payloads
├── File contents
└── Environment variables

Validate ONCE at boundary, then trust internally.
```

---

## 2. Pydantic v2 Advanced Patterns

### Discriminated Unions (Tagged Unions)
```python
from pydantic import BaseModel, Field
from typing import Literal, Union, Annotated

class CreditCard(BaseModel):
    type: Literal["credit_card"] = "credit_card"
    card_number: str
    cvv: str

class BankTransfer(BaseModel):
    type: Literal["bank_transfer"] = "bank_transfer"
    account_number: str
    routing_number: str

class Crypto(BaseModel):
    type: Literal["crypto"] = "crypto"
    wallet_address: str
    network: str

# Discriminated union - Pydantic uses 'type' field to pick model
PaymentMethod = Annotated[
    Union[CreditCard, BankTransfer, Crypto],
    Field(discriminator="type")
]

class Order(BaseModel):
    id: str
    payment: PaymentMethod  # Efficient O(1) validation

# Usage
order = Order.model_validate({
    "id": "ord_123",
    "payment": {"type": "crypto", "wallet_address": "0x...", "network": "eth"}
})
# Pydantic instantly knows to use Crypto model
```

### Generic Models (Reusable Containers)
```python
from pydantic import BaseModel
from typing import Generic, TypeVar

T = TypeVar("T")

class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: T | None = None
    error: str | None = None
    
class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    per_page: int
    
    @property
    def has_next(self) -> bool:
        return self.page * self.per_page < self.total

# Usage - type-safe containers
class User(BaseModel):
    id: str
    name: str

response: ApiResponse[User] = ApiResponse(
    success=True,
    data=User(id="1", name="Alice")
)

paginated: PaginatedResponse[User] = PaginatedResponse(
    items=[User(id="1", name="Alice")],
    total=100,
    page=1,
    per_page=10
)
```

### Validator Execution Order
```
Pydantic v2 Validator Pipeline:
1. @field_validator(mode='before')  → Raw input (pre-coercion)
2. Type coercion                    → Automatic conversion
3. @field_validator(mode='after')   → Post-coercion validation
4. @model_validator(mode='after')   → Cross-field validation
```

```python
from pydantic import BaseModel, field_validator, model_validator, ValidationInfo

class Transaction(BaseModel):
    amount: float
    currency: str
    source_account: str
    target_account: str
    
    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency(cls, v: str) -> str:
        return v.upper().strip()
    
    @field_validator("amount", mode="after")
    @classmethod
    def validate_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Amount must be positive")
        return v
    
    @model_validator(mode="after")
    def validate_different_accounts(self) -> "Transaction":
        if self.source_account == self.target_account:
            raise ValueError("Source and target must differ")
        return self
```

### Computed Fields (Pydantic v2)
```python
from pydantic import BaseModel, computed_field

class Rectangle(BaseModel):
    width: float
    height: float
    
    @computed_field
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @computed_field
    @property
    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

# area and perimeter included in serialization
rect = Rectangle(width=10, height=5)
print(rect.model_dump())  # {'width': 10, 'height': 5, 'area': 50, 'perimeter': 30}
```

### Strict Mode (No Coercion)
```python
from pydantic import BaseModel, ConfigDict

class StrictUser(BaseModel):
    model_config = ConfigDict(
        strict=True,      # No type coercion
        extra="forbid",   # Reject unknown fields
        frozen=True,      # Immutable after creation
    )
    
    id: int
    name: str

# "123" won't coerce to 123 - raises ValidationError
# StrictUser(id="123", name="Alice")  # Error!
StrictUser(id=123, name="Alice")  # OK
```

---

## 3. Zod for TypeScript (Runtime Validation)

### Basic Pattern
```typescript
import { z } from "zod";

// Define schema (runtime)
const UserSchema = z.object({
  id: z.string().uuid(),
  email: z.string().email(),
  age: z.number().int().min(0).max(150),
  role: z.enum(["admin", "user", "guest"]),
  metadata: z.record(z.string()).optional(),
});

// Infer TypeScript type (compile-time)
type User = z.infer<typeof UserSchema>;

// Validate at boundary
function parseUser(data: unknown): User {
  return UserSchema.parse(data);  // Throws on invalid
}

// Safe parse (no throw)
const result = UserSchema.safeParse(data);
if (result.success) {
  const user = result.data;  // Type-safe User
} else {
  console.error(result.error.issues);
}
```

### Discriminated Unions in Zod
```typescript
const CreditCardSchema = z.object({
  type: z.literal("credit_card"),
  cardNumber: z.string(),
  cvv: z.string().length(3),
});

const BankTransferSchema = z.object({
  type: z.literal("bank_transfer"),
  accountNumber: z.string(),
  routingNumber: z.string(),
});

const PaymentSchema = z.discriminatedUnion("type", [
  CreditCardSchema,
  BankTransferSchema,
]);

type Payment = z.infer<typeof PaymentSchema>;
// TypeScript knows: Payment = CreditCard | BankTransfer
```

### Transform and Preprocessing
```typescript
const DateSchema = z.string().transform((s) => new Date(s));

const TrimmedString = z.string().transform((s) => s.trim().toLowerCase());

// Preprocess for coercion before validation
const NumberFromString = z.preprocess(
  (val) => (typeof val === "string" ? parseInt(val, 10) : val),
  z.number()
);
```

### API Response Validation
```typescript
// Type-safe fetch wrapper
async function fetchUser(id: string): Promise<User> {
  const response = await fetch(`/api/users/${id}`);
  const data = await response.json();
  return UserSchema.parse(data);  // Validated!
}

// With error handling
async function safeFetchUser(id: string): Promise<User | null> {
  const response = await fetch(`/api/users/${id}`);
  const data = await response.json();
  const result = UserSchema.safeParse(data);
  
  if (!result.success) {
    console.error("Invalid API response:", result.error.format());
    return null;
  }
  
  return result.data;
}
```

---

## 4. Schema Evolution Strategies

### Compatibility Types
```
BACKWARD COMPATIBLE (new reads old):
├── Add optional field with default
├── Widen type (int → int | float)
├── Add enum value (careful - reader must handle unknown)
└── Make required field optional

FORWARD COMPATIBLE (old reads new):
├── Ignore unknown fields
├── Use optional fields
└── Provide defaults for missing

FULL COMPATIBLE (both directions):
├── Add optional fields with defaults only
├── Never remove fields
└── Never change types
```

### Pydantic Schema Evolution
```python
from pydantic import BaseModel, Field

# V1: Original schema
class UserV1(BaseModel):
    id: str
    name: str

# V2: Added optional field with default (backward compatible)
class UserV2(BaseModel):
    id: str
    name: str
    email: str | None = None  # New optional field

# V3: Added field with default value
class UserV3(BaseModel):
    id: str
    name: str
    email: str | None = None
    role: str = "user"  # Default for old data

# Migration: V1 data works with V3
old_data = {"id": "1", "name": "Alice"}
user = UserV3.model_validate(old_data)  # Works!
# user.email = None, user.role = "user"
```

### Handling Unknown Fields
```python
from pydantic import BaseModel, ConfigDict

# IGNORE unknown fields (forward compatible)
class FlexibleModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    known_field: str

# FORBID unknown fields (strict API)
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    known_field: str

# ALLOW unknown fields (store them)
class ExtensibleModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    known_field: str
    # Unknown fields accessible via model_extra
```

### Version Field Pattern
```python
from pydantic import BaseModel, field_validator
from typing import Literal

class VersionedEvent(BaseModel):
    version: Literal[1, 2, 3]
    # Common fields...

class EventV1(VersionedEvent):
    version: Literal[1] = 1
    old_field: str

class EventV2(VersionedEvent):
    version: Literal[2] = 2
    new_field: str  # Renamed from old_field

def parse_event(data: dict) -> VersionedEvent:
    version = data.get("version", 1)
    if version == 1:
        return EventV1.model_validate(data)
    elif version == 2:
        return EventV2.model_validate(data)
    raise ValueError(f"Unknown version: {version}")
```

---

## 5. Schema Registry Patterns (Kafka/Avro)

### Compatibility Modes
```
Mode              | Check
------------------|------------------------------------------
BACKWARD          | New schema can read old data
BACKWARD_TRANSITIVE | All previous versions readable
FORWARD           | Old schema can read new data
FORWARD_TRANSITIVE  | All future versions readable
FULL              | Both directions (one version)
FULL_TRANSITIVE   | Both directions (all versions)
NONE              | No compatibility checking
```

### Avro Schema Evolution Rules
```json
// V1
{
  "type": "record",
  "name": "User",
  "fields": [
    {"name": "id", "type": "string"},
    {"name": "name", "type": "string"}
  ]
}

// V2 - Add optional field (backward compatible)
{
  "type": "record",
  "name": "User",
  "fields": [
    {"name": "id", "type": "string"},
    {"name": "name", "type": "string"},
    {"name": "email", "type": ["null", "string"], "default": null}
  ]
}
```

### Breaking vs Non-Breaking
```
✅ SAFE Changes:
- Add field with default
- Add optional field (union with null)
- Add alias to field
- Promote int to long/float/double

❌ BREAKING Changes:
- Remove field without default
- Rename field (use alias instead)
- Change field type incompatibly
- Change field from optional to required
```

---

## 6. Production Validation Patterns

### FastAPI Request Validation
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

class CreateOrder(BaseModel):
    product_id: str
    quantity: int
    
    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError("Quantity must be 1-100")
        return v

@app.post("/orders")
async def create_order(order: CreateOrder):
    # order is already validated by Pydantic
    return {"status": "created", "order": order}
```

### Layered Validation
```python
# Layer 1: Schema validation (Pydantic)
class OrderInput(BaseModel):
    product_id: str
    quantity: int

# Layer 2: Business validation (Service)
class OrderService:
    async def create_order(self, input: OrderInput) -> Order:
        # Business rules
        product = await self.product_repo.get(input.product_id)
        if not product:
            raise BusinessError("Product not found")
        if product.stock < input.quantity:
            raise BusinessError("Insufficient stock")
        # Create order...
```

### Error Response Standardization
```python
from pydantic import BaseModel, ValidationError
from fastapi import Request
from fastapi.responses import JSONResponse

class ValidationErrorResponse(BaseModel):
    type: str = "validation_error"
    errors: list[dict]

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content=ValidationErrorResponse(
            errors=[
                {
                    "field": ".".join(str(loc) for loc in err["loc"]),
                    "message": err["msg"],
                    "type": err["type"],
                }
                for err in exc.errors()
            ]
        ).model_dump()
    )
```

---

## 7. Anti-Patterns

### ❌ Don't: Trust external data without validation
```python
# BAD
response = requests.get(url)
data = response.json()
user = User(**data)  # No validation!

# GOOD
response = requests.get(url)
user = User.model_validate(response.json())  # Validates!
```

### ❌ Don't: Validate deep in call stack
```python
# BAD - Validation scattered everywhere
def process_order(data: dict):
    validate_product(data["product"])  # Here
    validate_quantity(data["quantity"])  # And here
    ...

# GOOD - Validate once at boundary
def process_order(order: ValidatedOrder):  # Already validated
    ...
```

### ❌ Don't: Ignore schema compatibility
```python
# BAD - Breaking change without version
class UserV2(BaseModel):
    user_id: str  # Renamed from 'id' - breaks old clients!

# GOOD - Maintain compatibility
class UserV2(BaseModel):
    user_id: str
    id: str = Field(alias="user_id", deprecated=True)  # Backward compat
```

---

## 8. Key Takeaways

1. **Validate at boundaries** - Trust nothing from external sources
2. **Use discriminated unions** - O(1) validation for polymorphic types
3. **Zod for TypeScript** - Runtime validation with type inference
4. **Schema evolution** - Plan for change from day one
5. **Strict mode for APIs** - Forbid unknown fields, no coercion
6. **Layered validation** - Schema → Business rules → Persistence
7. **Standardize errors** - Consistent error response format

---

*Cycle 42 Complete - Data Validation & Schema Evolution*
*Next: Cycle 43 - TBD*
