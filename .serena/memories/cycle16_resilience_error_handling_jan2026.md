# Cycle 16: Resilience, Fault Tolerance & Error Handling (January 2026)

**Research Date**: 2026-01-25
**Focus**: Circuit breaker, retry patterns, Saga, LLM agent error handling, Temporal durable execution

---

## 1. CORE RESILIENCE PATTERNS

### The Resilience Triad
**Source**: Ali Ali, LinkedIn Oct 2025

```
🔄 RETRY     + ⏱ TIMEOUT    + ⚡ CIRCUIT BREAKER
   ↓              ↓                ↓
Handles      Prevents          Stops
temporary    infinite          cascading
glitches     waiting           failures
```

### Circuit Breaker Pattern (Jan 2026)
**Source**: Bhuman Soni, Medium

**Three States**:
```
┌─────────────────────────────────────────────────────────────┐
│                   CIRCUIT BREAKER STATES                     │
├─────────────────────────────────────────────────────────────┤
│  CLOSED (Normal)                                            │
│  ├── All requests pass through                              │
│  ├── Track failure count                                    │
│  └── If failures > threshold → OPEN                         │
├─────────────────────────────────────────────────────────────┤
│  OPEN (Tripped)                                             │
│  ├── Requests fail immediately (no calls to dependency)     │
│  ├── Prevents cascading failures                            │
│  └── After timeout → HALF-OPEN                              │
├─────────────────────────────────────────────────────────────┤
│  HALF-OPEN (Testing)                                        │
│  ├── Allow limited test requests                            │
│  ├── If success → CLOSED                                    │
│  └── If failure → OPEN                                      │
└─────────────────────────────────────────────────────────────┘
```

**Python Implementation (pybreaker)**:
```python
import pybreaker

# Configure circuit breaker
breaker = pybreaker.CircuitBreaker(
    fail_max=5,              # Open after 5 failures
    reset_timeout=30,        # Try half-open after 30s
    exclude=[ValueError],    # Don't count these as failures
)

@breaker
def call_external_service():
    return requests.get("https://api.example.com/data")
```

### Retry with Tenacity (Python)
**Source**: Amitav Roy, Aug 2025

```python
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log
)
import logging

logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(ConnectionError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def resilient_api_call():
    return requests.post("https://api.example.com/action")
```

**Best Practices**:
- **Exponential backoff**: 1s, 2s, 4s... prevents thundering herd
- **Jitter**: Add randomness to prevent synchronized retries
- **Max retries**: 2-3 for transient, fail fast for permanent errors
- **Idempotency**: Ensure retried operations are safe to repeat

### Dapr Default Resiliency Policies
**Source**: Dapr Docs, Jan 2026

```yaml
# dapr/resiliency.yaml
apiVersion: dapr.io/v1alpha1
kind: Resiliency
spec:
  policies:
    retries:
      DefaultRetryPolicy:
        policy: constant
        maxRetries: 3
        duration: 1s
    
    timeouts:
      DefaultTimeoutPolicy: 30s
    
    circuitBreakers:
      DefaultCircuitBreakerPolicy:
        maxRequests: 1
        interval: 10s
        timeout: 30s
        trip: consecutiveFailures >= 5
```

---

## 2. LLM AGENT ERROR HANDLING

### Multi-Level Error Handling (LangGraph)
**Source**: Sparkco AI, Oct 2025

```python
from langgraph.graph import StateGraph
from typing import TypedDict, Optional

class AgentState(TypedDict):
    messages: list
    error: Optional[str]
    retry_count: int
    fallback_used: bool

def node_with_error_handling(state: AgentState):
    try:
        result = execute_tool(state)
        return {"messages": state["messages"] + [result], "error": None}
    except ToolTimeoutError:
        if state["retry_count"] < 3:
            return {"retry_count": state["retry_count"] + 1}
        return {"error": "Tool timeout after 3 retries", "fallback_used": True}
    except LLMError as e:
        return {"error": str(e), "fallback_used": True}
```

### Instructor Exception Hierarchy
**Source**: python.useinstructor.com

```python
from instructor.core.exceptions import (
    InstructorError,          # Base exception
    IncompleteOutputException, # Partial/truncated response
    InstructorRetryException,  # Retry limit exceeded
    ValidationError,           # Pydantic validation failed
    ProviderError,            # LLM provider error
    ConfigurationError,       # Setup/config issues
    ModeError,                # Invalid mode
)

try:
    result = client.chat.completions.create(...)
except IncompleteOutputException as e:
    # Handle truncated responses
    partial_result = e.last_completion
except ValidationError as e:
    # Handle Pydantic validation failures
    log_validation_error(e.errors())
except InstructorError as e:
    # Catch-all for Instructor errors
    handle_generic_error(e)
```

### Graceful Degradation for Agents
**Source**: Monetizely, Aug 2025

```python
class GracefulAgent:
    """Agent with multi-level fallback strategy"""
    
    async def execute(self, task: str) -> Result:
        strategies = [
            self.try_primary_llm,        # Level 1: Primary model
            self.try_fallback_llm,       # Level 2: Cheaper/faster model
            self.try_cached_response,    # Level 3: Similar past response
            self.try_rule_based,         # Level 4: Deterministic rules
            self.return_safe_default,    # Level 5: Safe default
        ]
        
        for strategy in strategies:
            try:
                return await strategy(task)
            except Exception as e:
                self.log_degradation(strategy.__name__, e)
                continue
        
        raise AllStrategiesFailedError(task)
```

---

## 3. SAGA PATTERN & DISTRIBUTED TRANSACTIONS

### The Saga Pattern (Temporal, Jan 2026)
**Source**: Temporal.io, skyro-tech

**Problem**: In microservices, no single database transaction spans services.
**Solution**: Saga = sequence of local transactions with compensating actions.

```
┌─────────────────────────────────────────────────────────────┐
│                      SAGA EXECUTION                          │
├─────────────────────────────────────────────────────────────┤
│  HAPPY PATH:                                                │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│  │ Reserve │ → │ Payment │ → │ Fulfill │ → │ Confirm │     │
│  │  Seat   │   │ Charge  │   │  Order  │   │  Email  │     │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘     │
├─────────────────────────────────────────────────────────────┤
│  COMPENSATION (on failure at Fulfill):                      │
│  ┌─────────┐   ┌─────────┐                                  │
│  │ Release │ ← │ Refund  │ ← ❌ Fulfill Failed              │
│  │  Seat   │   │ Payment │                                  │
│  └─────────┘   └─────────┘                                  │
└─────────────────────────────────────────────────────────────┘
```

### Saga Types
| Type | Coordination | Best For |
|------|--------------|----------|
| **Choreography** | Events between services | Simple, decoupled systems |
| **Orchestration** | Central coordinator | Complex, visibility needed |

### Temporal Durable Execution
**Tagline**: "What if your code never failed?"

```python
from temporalio import activity, workflow
from datetime import timedelta

@activity.defn
async def reserve_seat(seat_id: str) -> str:
    # Temporal handles retries automatically
    return await booking_service.reserve(seat_id)

@activity.defn
async def charge_payment(amount: float) -> str:
    return await payment_service.charge(amount)

@activity.defn
async def release_seat(seat_id: str) -> None:
    """Compensating action"""
    await booking_service.release(seat_id)

@workflow.defn
class BookingWorkflow:
    @workflow.run
    async def run(self, booking: BookingRequest) -> BookingResult:
        # Temporal tracks state, handles failures, retries
        seat_id = await workflow.execute_activity(
            reserve_seat,
            booking.seat_id,
            start_to_close_timeout=timedelta(seconds=30),
        )
        
        try:
            payment_id = await workflow.execute_activity(
                charge_payment,
                booking.amount,
                start_to_close_timeout=timedelta(seconds=60),
            )
        except Exception:
            # Automatic compensation
            await workflow.execute_activity(release_seat, seat_id)
            raise
        
        return BookingResult(seat_id=seat_id, payment_id=payment_id)
```

### Temporal Retry Policies
```python
from temporalio.common import RetryPolicy

retry_policy = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=1),
    maximum_attempts=5,
    non_retryable_error_types=["InvalidInputError", "BusinessRuleViolation"],
)

await workflow.execute_activity(
    my_activity,
    retry_policy=retry_policy,
)
```

---

## 4. ERROR HANDLING BEST PRACTICES

### Error Classification Matrix
| Category | Retry? | Example | Action |
|----------|--------|---------|--------|
| **Transient** | Yes | Network timeout | Exponential backoff |
| **Recoverable** | Maybe | Rate limit | Wait and retry |
| **Permanent** | No | Invalid input | Fail fast |
| **Unknown** | Once | Unexpected 500 | Log, alert, fail |

### Agent Error Recovery Pattern
```python
class AgentErrorRecovery:
    """Standard recovery pattern for LLM agents"""
    
    ERROR_STRATEGIES = {
        "rate_limit": ("wait", 60),
        "context_length": ("truncate", None),
        "invalid_json": ("retry_with_prompt", 3),
        "tool_not_found": ("fallback_tool", None),
        "timeout": ("retry", 3),
    }
    
    async def recover(self, error: AgentError) -> RecoveryAction:
        strategy, param = self.ERROR_STRATEGIES.get(
            error.type, ("fail", None)
        )
        
        if strategy == "wait":
            await asyncio.sleep(param)
            return RecoveryAction.RETRY
        elif strategy == "truncate":
            self.truncate_context()
            return RecoveryAction.RETRY
        elif strategy == "retry_with_prompt":
            if self.retry_count < param:
                self.enhance_prompt_with_error(error)
                return RecoveryAction.RETRY
        
        return RecoveryAction.FAIL
```

---

## 5. QUICK REFERENCE

```
RESILIENCE TRIAD:
  Retry       → Tenacity (exponential backoff + jitter)
  Timeout     → Fail fast, free resources
  Circuit     → pybreaker (CLOSED → OPEN → HALF-OPEN)

LLM AGENT ERRORS:
  Instructor  → Exception hierarchy for structured output
  LangGraph   → State-driven multi-level handling
  Graceful    → Primary → Fallback → Cache → Rules → Default

SAGA PATTERN:
  Choreography → Event-driven, decoupled
  Orchestration → Central coordinator (Temporal)
  Compensation → Rollback via inverse operations

TEMPORAL DURABLE EXECUTION:
  "Code that never fails"
  ├── Automatic retries with policies
  ├── State persisted across failures
  ├── Compensating transactions built-in
  └── Visibility into workflow state
```

---

*Cycle 16 of Perpetual Enhancement Loops*
*Focus: System architecture, analytical reasoning, auditing frameworks - NOT creative AI*
