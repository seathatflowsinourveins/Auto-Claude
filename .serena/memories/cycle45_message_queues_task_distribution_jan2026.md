# Cycle 45: Message Queues & Task Distribution Patterns (January 2026)

## Overview
Production patterns for distributed task processing in Python, covering Celery 5.6+, broker selection, idempotency, and durable execution with Temporal.

---

## 1. Celery 5.6 Production Configuration

### Core Setup with FastAPI
```python
from celery import Celery
from kombu import Queue, Exchange

# Celery app with RabbitMQ broker
app = Celery(
    "tasks",
    broker="amqp://user:pass@localhost:5672/vhost",
    backend="redis://localhost:6379/0",
    include=["app.tasks.email", "app.tasks.processing"],
)

# Production configuration
app.conf.update(
    # Serialization (msgpack faster than JSON)
    task_serializer="msgpack",
    result_serializer="msgpack",
    accept_content=["msgpack", "json"],
    
    # Reliability
    task_acks_late=True,              # Ack after task completes (not before)
    task_reject_on_worker_lost=True,  # Requeue if worker dies mid-task
    task_time_limit=3600,             # Hard limit: 1 hour
    task_soft_time_limit=3300,        # Soft limit: 55 mins (allows cleanup)
    
    # Prefetch optimization
    worker_prefetch_multiplier=4,     # Fetch 4 tasks per worker at a time
    worker_concurrency=8,             # 8 concurrent workers
    
    # Result backend TTL
    result_expires=86400,             # Results expire after 24 hours
    
    # Broker connection resilience
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,
)
```

### Task Routing by Priority
```python
# Define exchanges and queues
default_exchange = Exchange("default", type="direct")
priority_exchange = Exchange("priority", type="direct")

app.conf.task_queues = (
    Queue("default", default_exchange, routing_key="default"),
    Queue("high_priority", priority_exchange, routing_key="high"),
    Queue("low_priority", default_exchange, routing_key="low"),
    Queue("cpu_intensive", default_exchange, routing_key="cpu"),
)

# Route tasks to queues
app.conf.task_routes = {
    "app.tasks.email.*": {"queue": "high_priority"},
    "app.tasks.reports.*": {"queue": "low_priority"},
    "app.tasks.ml.*": {"queue": "cpu_intensive"},
}

# Or use dynamic routing
@app.task(bind=True, queue="high_priority")
def send_critical_notification(self, user_id: int, message: str):
    ...
```

---

## 2. Broker Selection: RabbitMQ vs Redis

### Decision Matrix
| Factor | RabbitMQ | Redis |
|--------|----------|-------|
| **Reliability** | ✅ AMQP guarantees, persistent queues | ⚠️ At-most-once by default |
| **Performance** | 10-20K msg/sec | 100K+ msg/sec |
| **Complexity** | Higher (exchanges, bindings) | Lower (simple pub/sub) |
| **Memory** | Disk-backed (survives restarts) | In-memory (volatile) |
| **Use Case** | Financial, critical workflows | Cache, real-time, fire-and-forget |
| **Clustering** | Built-in federation | Redis Cluster / Sentinel |

### RabbitMQ for Critical Tasks
```python
# RabbitMQ with publisher confirms
app = Celery(
    "critical_tasks",
    broker="amqp://user:pass@rabbitmq:5672/vhost",
    broker_transport_options={
        "confirm_publish": True,          # Wait for broker ACK
        "max_retries": 5,
        "interval_start": 0,
        "interval_step": 0.5,
        "interval_max": 3,
    },
)

# Enable message persistence
app.conf.task_default_delivery_mode = 2  # Persistent messages
```

### Redis for High-Throughput
```python
# Redis with optimizations
app = Celery(
    "fast_tasks",
    broker="redis://localhost:6379/0",
    broker_transport_options={
        "visibility_timeout": 3600,       # 1 hour visibility
        "fanout_prefix": True,
        "fanout_patterns": True,
        "socket_keepalive": True,
        "retry_on_timeout": True,
    },
)
```

---

## 3. Task Idempotency Patterns

### Idempotency Key Pattern
```python
from celery import Task
from functools import wraps
import hashlib
import redis

redis_client = redis.Redis()

def idempotent_task(ttl: int = 86400):
    """Decorator ensuring task runs exactly once per unique args."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate idempotency key from task name + args
            key_data = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
            idempotency_key = hashlib.sha256(key_data.encode()).hexdigest()
            lock_key = f"task:lock:{idempotency_key}"
            result_key = f"task:result:{idempotency_key}"
            
            # Check if already processed
            if cached := redis_client.get(result_key):
                return cached.decode()
            
            # Acquire distributed lock
            if not redis_client.set(lock_key, "1", nx=True, ex=300):
                raise Exception("Task already in progress")
            
            try:
                result = func(*args, **kwargs)
                redis_client.setex(result_key, ttl, str(result))
                return result
            finally:
                redis_client.delete(lock_key)
        
        return wrapper
    return decorator

@app.task(bind=True)
@idempotent_task(ttl=3600)
def process_payment(self, payment_id: str, amount: float):
    """Guaranteed exactly-once payment processing."""
    # Process payment...
    return {"status": "completed", "payment_id": payment_id}
```

### Database-Level Idempotency
```python
from sqlalchemy import Column, String, DateTime, UniqueConstraint
from sqlalchemy.dialects.postgresql import insert

class ProcessedTask(Base):
    __tablename__ = "processed_tasks"
    
    id = Column(String, primary_key=True)
    task_name = Column(String, nullable=False)
    processed_at = Column(DateTime, default=datetime.utcnow)
    result = Column(JSON)
    
    __table_args__ = (
        UniqueConstraint("id", "task_name", name="uq_task_idempotency"),
    )

@app.task(bind=True)
def idempotent_db_task(self, task_id: str, payload: dict):
    """Use database constraint for idempotency."""
    async with async_session() as session:
        stmt = insert(ProcessedTask).values(
            id=task_id,
            task_name=self.name,
            result=None,
        ).on_conflict_do_nothing()
        
        result = await session.execute(stmt)
        if result.rowcount == 0:
            # Already processed
            existing = await session.get(ProcessedTask, task_id)
            return existing.result
        
        # Process task
        output = do_work(payload)
        
        # Update result
        await session.execute(
            update(ProcessedTask)
            .where(ProcessedTask.id == task_id)
            .values(result=output)
        )
        await session.commit()
        return output
```

---

## 4. Retry Strategies with Exponential Backoff

### Advanced Retry Configuration
```python
from celery import Task
from celery.exceptions import MaxRetriesExceededError
import random

class RetryableTask(Task):
    """Base task with intelligent retry behavior."""
    
    autoretry_for = (ConnectionError, TimeoutError, IOError)
    retry_kwargs = {"max_retries": 5}
    retry_backoff = True           # Exponential backoff
    retry_backoff_max = 600        # Max 10 minutes
    retry_jitter = True            # Add randomness to prevent thundering herd

@app.task(
    bind=True,
    base=RetryableTask,
    autoretry_for=(Exception,),
    retry_backoff=2,               # Base: 2 seconds
    retry_backoff_max=300,         # Max: 5 minutes
    max_retries=5,
)
def fetch_external_api(self, url: str):
    """Task with automatic exponential backoff retry."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()

# Manual retry with custom backoff
@app.task(bind=True, max_retries=10)
def process_with_custom_retry(self, data: dict):
    try:
        return external_service.process(data)
    except RateLimitError as exc:
        # Custom backoff: 2^attempt + jitter
        backoff = (2 ** self.request.retries) + random.uniform(0, 1)
        raise self.retry(exc=exc, countdown=min(backoff, 300))
    except FatalError:
        # Don't retry fatal errors
        raise
```

### Circuit Breaker Pattern
```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
def call_external_service(payload):
    """Circuit breaker prevents cascade failures."""
    return requests.post("https://api.example.com", json=payload)

@app.task(bind=True)
def task_with_circuit_breaker(self, payload: dict):
    try:
        return call_external_service(payload)
    except CircuitBreakerError:
        # Circuit is open, task will be retried later
        raise self.retry(countdown=60)
```

---

## 5. Celery Primitives: Chains, Groups, Chords

### Task Chains (Sequential)
```python
from celery import chain

# Execute tasks sequentially, passing results
workflow = chain(
    fetch_data.s(url="https://api.example.com/data"),
    transform_data.s(),
    store_data.s(destination="s3://bucket/output"),
)
result = workflow.apply_async()
```

### Task Groups (Parallel)
```python
from celery import group

# Execute tasks in parallel
parallel_tasks = group(
    process_chunk.s(chunk) for chunk in data_chunks
)
results = parallel_tasks.apply_async()

# Wait for all to complete
all_results = results.get()  # Blocking
```

### Chords (Parallel + Callback)
```python
from celery import chord

# Parallel tasks with a callback when all complete
workflow = chord(
    [process_item.s(item) for item in items],
    aggregate_results.s()  # Called with list of all results
)
final_result = workflow.apply_async()
```

### Complex Workflows
```python
from celery import chain, group, chord

# ETL pipeline with parallel extraction
etl_workflow = chain(
    # Step 1: Parallel extraction from multiple sources
    group(
        extract_from_db.s(),
        extract_from_api.s(),
        extract_from_file.s(),
    ),
    # Step 2: Merge all extracted data
    merge_data.s(),
    # Step 3: Parallel transformation
    chord(
        [transform_partition.s(i) for i in range(4)],
        combine_partitions.s()
    ),
    # Step 4: Load to destination
    load_to_warehouse.s(),
)
```

---

## 6. Dead Letter Queues & Error Handling

### RabbitMQ Dead Letter Exchange
```python
from kombu import Queue, Exchange

# Dead letter exchange
dlx = Exchange("dead_letter", type="direct")
dlq = Queue("dead_letter_queue", dlx, routing_key="dead")

# Main queue with DLX configuration
main_queue = Queue(
    "main_queue",
    Exchange("main", type="direct"),
    routing_key="main",
    queue_arguments={
        "x-dead-letter-exchange": "dead_letter",
        "x-dead-letter-routing-key": "dead",
        "x-message-ttl": 86400000,  # 24 hours
    },
)

app.conf.task_queues = (main_queue, dlq)
```

### Error Handler with Dead Letter
```python
from celery.signals import task_failure

@task_failure.connect
def handle_task_failure(sender, task_id, exception, args, kwargs, traceback, einfo, **kw):
    """Send failed tasks to dead letter queue for investigation."""
    dead_letter_task.delay(
        task_name=sender.name,
        task_id=task_id,
        exception=str(exception),
        args=args,
        kwargs=kwargs,
        traceback=str(traceback),
    )

@app.task(queue="dead_letter_queue")
def dead_letter_task(task_name, task_id, exception, args, kwargs, traceback):
    """Process failed tasks - log, alert, or retry later."""
    logger.error(f"Task {task_name}[{task_id}] failed: {exception}")
    # Send to monitoring/alerting system
    sentry_sdk.capture_message(f"Dead letter: {task_name}", extra={
        "task_id": task_id,
        "args": args,
        "exception": exception,
    })
```

---

## 7. Temporal for Durable Execution

### When to Use Temporal vs Celery
| Scenario | Celery | Temporal |
|----------|--------|----------|
| Simple background jobs | ✅ | Overkill |
| Fire-and-forget | ✅ | Overkill |
| Long-running workflows | ⚠️ Fragile | ✅ |
| Saga patterns | ⚠️ Manual | ✅ Built-in |
| Human-in-the-loop | ❌ | ✅ |
| Cross-service transactions | ⚠️ Complex | ✅ |

### Temporal Workflow Example
```python
from temporalio import workflow, activity
from temporalio.client import Client
from datetime import timedelta

@activity.defn
async def charge_payment(amount: float, payment_id: str) -> str:
    # Actual payment processing
    return f"charged:{payment_id}"

@activity.defn
async def reserve_inventory(items: list[str]) -> bool:
    # Reserve items in warehouse
    return True

@activity.defn
async def send_confirmation(order_id: str) -> None:
    # Send email
    pass

@workflow.defn
class OrderWorkflow:
    """Durable order processing with automatic compensation."""
    
    @workflow.run
    async def run(self, order: dict) -> str:
        # Step 1: Reserve inventory (compensated on failure)
        reserved = await workflow.execute_activity(
            reserve_inventory,
            order["items"],
            start_to_close_timeout=timedelta(minutes=5),
        )
        
        try:
            # Step 2: Charge payment
            payment_result = await workflow.execute_activity(
                charge_payment,
                order["amount"],
                order["payment_id"],
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=1),
                    backoff_coefficient=2.0,
                ),
            )
        except Exception:
            # Automatic compensation: release inventory
            await workflow.execute_activity(
                release_inventory, order["items"]
            )
            raise
        
        # Step 3: Confirm order
        await workflow.execute_activity(
            send_confirmation,
            order["order_id"],
            start_to_close_timeout=timedelta(minutes=1),
        )
        
        return f"Order {order['order_id']} completed"

# Start workflow
async def main():
    client = await Client.connect("localhost:7233")
    
    result = await client.execute_workflow(
        OrderWorkflow.run,
        {"order_id": "123", "items": ["item1"], "amount": 99.99},
        id="order-123",
        task_queue="order-processing",
    )
```

---

## 8. Monitoring & Observability

### Celery Flower Dashboard
```bash
# Install and run Flower
pip install flower
celery -A app flower --port=5555

# Prometheus metrics endpoint
celery -A app flower --port=5555 --prometheus-port=5556
```

### Custom Metrics with Prometheus
```python
from prometheus_client import Counter, Histogram, start_http_server
from celery.signals import task_prerun, task_postrun, task_failure

TASK_COUNTER = Counter("celery_tasks_total", "Total tasks", ["name", "status"])
TASK_DURATION = Histogram("celery_task_duration_seconds", "Task duration", ["name"])

@task_prerun.connect
def task_prerun_handler(sender, task_id, task, **kwargs):
    task.start_time = time.time()

@task_postrun.connect
def task_postrun_handler(sender, task_id, task, retval, state, **kwargs):
    duration = time.time() - getattr(task, 'start_time', time.time())
    TASK_DURATION.labels(name=sender.name).observe(duration)
    TASK_COUNTER.labels(name=sender.name, status="success").inc()

@task_failure.connect
def task_failure_handler(sender, task_id, **kwargs):
    TASK_COUNTER.labels(name=sender.name, status="failure").inc()

# Start metrics server
start_http_server(8000)
```

---

## Quick Reference

### Celery Production Checklist
- [ ] `task_acks_late=True` for reliability
- [ ] `task_reject_on_worker_lost=True` for auto-requeue
- [ ] Set `task_time_limit` and `task_soft_time_limit`
- [ ] Configure retry with exponential backoff
- [ ] Implement idempotency for critical tasks
- [ ] Set up dead letter queue for failed tasks
- [ ] Monitor with Flower or Prometheus

### Anti-Patterns
- **No idempotency**: Duplicate task execution on retry
- **Missing time limits**: Tasks run forever on failure
- **Ignoring dead letters**: Failed tasks disappear silently
- **Synchronous waits in tasks**: Blocks workers
- **Large payloads in messages**: Use object references instead

### Broker Selection Guide
- **RabbitMQ**: Financial transactions, ordering guarantees, persistence required
- **Redis**: Real-time analytics, caching, high-throughput fire-and-forget
- **SQS**: AWS-native, serverless, managed infrastructure
- **Temporal**: Long-running workflows, sagas, human-in-the-loop

---

*Cycle 45 Complete - Message Queues & Task Distribution Patterns*
*Date: January 2026*
