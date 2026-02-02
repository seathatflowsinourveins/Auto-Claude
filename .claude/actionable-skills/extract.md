# /extract - Structured Output Extraction Skill

## Description
Apply structured extraction patterns (Instructor/Outlines concepts) to produce schema-validated outputs. I structure my responses to guarantee valid data.

## When to Use
- User needs structured data (JSON, typed objects)
- Extracting information from unstructured text
- Producing API-compatible responses
- Data validation is critical

## Extraction Protocol (What I Actually Do)

### Step 1: Define Schema
```
INTERNAL PROCESS:
- Identify required fields
- Determine data types
- Note constraints (ranges, enums, patterns)
- Consider optional vs required

OUTPUT: Mental schema definition
```

### Step 2: Extract Data
```
INTERNAL PROCESS:
- Parse input text for relevant information
- Map extracted values to schema fields
- Handle missing/ambiguous data explicitly

OUTPUT: Raw extracted values
```

### Step 3: Validate
```
INTERNAL PROCESS:
- Type check all values
- Verify constraints are met
- Handle validation failures:
  - Re-extract if possible
  - Ask user if ambiguous
  - Use sensible defaults if safe

OUTPUT: Validated data or error with specific reason
```

### Step 4: Format Output
```
OUTPUT FORMAT:
- JSON for data interchange
- Python code for typed objects
- Markdown table for human readability
```

## Pattern Examples

### Simple Extraction
```python
# User: "John is 25 years old and lives in NYC"

class User(BaseModel):
    name: str
    age: int
    city: str

# I extract:
{
    "name": "John",
    "age": 25,
    "city": "NYC"
}
```

### Constrained Extraction
```python
# User: "The market is looking bullish today"

class Sentiment(BaseModel):
    assessment: Literal["bullish", "bearish", "neutral"]
    confidence: float  # 0.0 to 1.0

# I extract:
{
    "assessment": "bullish",
    "confidence": 0.85
}
```

### Nested Extraction
```python
# Complex document → Structured output

class TradeSignal(BaseModel):
    symbol: str
    action: Literal["buy", "sell", "hold"]
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str

# I extract with validation:
# - entry_price > 0
# - stop_loss < entry_price (for buy)
# - take_profit > entry_price (for buy)
```

## Validation Retry Pattern

When extraction fails validation:

```
Attempt 1: Extract data
→ Validation Error: age must be positive

Attempt 2: Re-read source, extract with error context
→ Found "age: -1" was typo, context suggests 21

Attempt 3: Return validated or ask user
→ "I found age=-1 in the input. Did you mean 21 based on context?"
```

## Integration with V30.8

This skill implements Section 53 (Structured Output Generation) of CLAUDE_SELF_ENHANCEMENT_V30.md

Key insight: I can apply Instructor concepts without the library - it's about structured thinking and validation, not specific tools.
