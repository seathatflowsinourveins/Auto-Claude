# Cycle 34: Background Tasks & Queues (January 2026)

## Task Queue Selection Matrix

| Library | Best For | Broker | Async | Complexity |
|---------|----------|--------|-------|------------|
| **Celery** | Large scale, multi-broker | Redis/RabbitMQ/SQS | No (prefork) | High |
| **ARQ** | FastAPI, async-first | Redis only | YES | Low |
| **RQ** | Simple Redis queue | Redis only | No | Very Low |
| **Dramatiq** | Celery alternative | Redis/RabbitMQ | No | Medium |
| **Huey** | Lightweight | Redis/SQLite | Optional | Low |
| **Temporal** | Durable workflows | Temporal Server | Yes | High |

## FastAPI BackgroundTasks (Built-in)

**Use case**: Lightweight, in-process tasks (email, logging, simple cleanup)

```python
from fastapi import BackgroundTasks

def send_email(email: str, message: str):
    # Runs after response is sent
    smtp.send(email, message)

@app.post("/signup")
async def signup(user: User, background_tasks: BackgroundTasks):
    create_user(user)
    background_tasks.add_task(send_email, user.email, "Welcome!")
    return {"status": "created"}  # Returns immediately
```

**Limitations**:
- No persistence (lost on crash)
- No retries
- No result tracking
- Tied to web process lifecycle

## Celery Production Patterns

### Basic Setup (Celery 5.6)
```python
from celery import Celery

app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

# Configuration
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 min hard limit
    task_soft_time_limit=240,  # 4 min soft limit
)

@app.task(bind=True, max_retries=3)
def process_order(self, order_id: int):
    try:
        result = do_processing(order_id)
        return result
    except TransientError as exc:
        raise self.retry(exc=exc, countdown=60)  # Retry in 60s
```

### Concurrency Options
```bash
# Prefork (default) - CPU-bound, most features
celery -A proj worker --pool=prefork --concurrency=4

# Eventlet/Gevent - IO-bound, high concurrency
celery -A proj worker --pool=eventlet --concurrency=100

# Solo - Single-threaded, debugging
celery -A proj worker --pool=solo
```

### Prefetch Tuning
```python
# Default prefetch_multiplier=4 (worker grabs 4 * concurrency tasks)
# For long tasks, reduce to prevent blocking
app.conf.worker_prefetch_multiplier = 1

# For fair task distribution across workers
app.conf.task_acks_late = True
app.conf.worker_prefetch_multiplier = 1
```

### Task Routing
```python
app.conf.task_routes = {
    'tasks.email.*': {'queue': 'email'},
    'tasks.reports.*': {'queue': 'reports', 'routing_key': 'reports.#'},
}

# Start specialized workers
# celery -A proj worker -Q email -c 2
# celery -A proj worker -Q reports -c 1
```

### Periodic Tasks (Celery Beat)
```python
from celery.schedules import crontab

app.conf.beat_schedule = {
    'cleanup-every-hour': {
        'task': 'tasks.cleanup',
        'schedule': crontab(minute=0),  # Every hour
    },
    'daily-report': {
        'task': 'tasks.generate_report',
        'schedule': crontab(hour=6, minute=0),  # 6 AM
    },
}

# Run beat scheduler
# celery -A proj beat --loglevel=info
```

## ARQ (Async-First for FastAPI)

### Why ARQ over Celery for FastAPI
- **Native async/await** - No blocking, works with asyncio
- **Simpler** - Less configuration, Redis-only
- **Type hints** - Full typing support
- **FastAPI integration** - Natural fit

### Basic Setup
```python
# tasks.py
from arq import create_pool
from arq.connections import RedisSettings

async def send_email(ctx, email: str, subject: str):
    """ARQ task - ctx is injected automatically."""
    await smtp.send_async(email, subject)
    return {"sent": True}

class WorkerSettings:
    functions = [send_email]
    redis_settings = RedisSettings(host='localhost', port=6379)
    max_jobs = 10
    job_timeout = 300
```

### FastAPI Integration
```python
from arq import create_pool
from arq.connections import RedisSettings
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.arq_pool = await create_pool(RedisSettings())
    yield
    await app.state.arq_pool.close()

app = FastAPI(lifespan=lifespan)

@app.post("/send-email")
async def queue_email(email: str, request: Request):
    job = await request.app.state.arq_pool.enqueue_job(
        'send_email', email, "Welcome!"
    )
    return {"job_id": job.job_id}
```

### Retry with Exponential Backoff
```python
from arq import Retry

async def unreliable_task(ctx, data: dict):
    try:
        result = await call_external_api(data)
        return result
    except APIError as e:
        # Retry with exponential backoff
        retry_count = ctx.get('job_try', 1)
        if retry_count < 5:
            delay = 2 ** retry_count  # 2, 4, 8, 16, 32 seconds
            raise Retry(defer=delay)
        raise  # Give up after 5 retries

class WorkerSettings:
    functions = [unreliable_task]
    max_tries = 5
    retry_jobs = True
```

### Running ARQ Worker
```bash
arq tasks.WorkerSettings
```

## RQ (Redis Queue) - Simple Alternative

```python
from redis import Redis
from rq import Queue

redis_conn = Redis()
q = Queue(connection=redis_conn)

# Enqueue task
job = q.enqueue(send_email, 'user@example.com', 'Hello')

# Check result
job.result  # None until complete
job.get_status()  # 'queued', 'started', 'finished', 'failed'

# Worker
# rq worker --with-scheduler
```

## Dramatiq - Celery Alternative

```python
import dramatiq
from dramatiq.brokers.redis import RedisBroker

redis_broker = RedisBroker(host="localhost", port=6379)
dramatiq.set_broker(redis_broker)

@dramatiq.actor(max_retries=3, min_backoff=1000, max_backoff=300000)
def send_email(email: str, message: str):
    smtp.send(email, message)

# Enqueue
send_email.send("user@example.com", "Hello")

# Worker
# dramatiq tasks
```

## APScheduler (Scheduling)

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

scheduler = AsyncIOScheduler()

@scheduler.scheduled_job(CronTrigger(hour=0, minute=0))  # Midnight
async def daily_cleanup():
    await cleanup_old_records()

@scheduler.scheduled_job('interval', hours=4)
async def periodic_sync():
    await sync_external_data()

# Start with FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.start()
    yield
    scheduler.shutdown()
```

## Temporal (Durable Execution)

**Use case**: Complex workflows, long-running processes, exactly-once semantics

```python
from temporalio import workflow, activity
from temporalio.client import Client
from temporalio.worker import Worker

@activity.defn
async def send_email(email: str) -> str:
    await smtp.send_async(email, "Welcome!")
    return "sent"

@activity.defn
async def charge_card(amount: float) -> str:
    result = await payment.charge(amount)
    return result

@workflow.defn
class OnboardingWorkflow:
    @workflow.run
    async def run(self, user_id: str) -> str:
        # Activities are durable - survive crashes
        await workflow.execute_activity(
            send_email,
            user_id,
            start_to_close_timeout=timedelta(seconds=30),
        )
        await workflow.execute_activity(
            charge_card,
            9.99,
            start_to_close_timeout=timedelta(seconds=60),
        )
        return "onboarding_complete"

# Start workflow
async def main():
    client = await Client.connect("localhost:7233")
    await client.execute_workflow(
        OnboardingWorkflow.run,
        "user_123",
        id="onboarding-user-123",
        task_queue="onboarding",
    )
```

## Decision Matrix

### Use FastAPI BackgroundTasks when:
- Simple, fire-and-forget tasks
- No persistence needed
- Low volume (< 100/min)
- Tasks complete quickly (< 30s)

### Use ARQ when:
- FastAPI + async codebase
- Redis available
- Need retries and result tracking
- Moderate scale

### Use Celery when:
- High scale (millions of tasks)
- Need multiple broker options
- Complex routing/priorities
- Periodic tasks (Beat)
- Team already knows Celery

### Use Temporal when:
- Complex multi-step workflows
- Need exactly-once semantics
- Long-running processes (hours/days)
- Saga pattern with compensation
- Human-in-the-loop approvals

## Production Checklist

1. **Monitoring**: Track queue depth, task duration, failure rate
2. **Dead Letter Queue**: Handle permanently failed tasks
3. **Idempotency**: Design tasks to be safely retried
4. **Timeouts**: Always set hard/soft limits
5. **Graceful Shutdown**: Drain queues before restart
6. **Separate Queues**: Isolate fast vs slow tasks
7. **Result Expiry**: Don't keep results forever
8. **Health Checks**: Monitor worker liveness

## Anti-Patterns

1. **Passing large objects** - Pass IDs, fetch in task
2. **No timeouts** - Tasks run forever
3. **Ignoring failures** - No retry strategy
4. **Single queue** - Fast tasks blocked by slow
5. **Sync in async** - Blocking calls in ARQ tasks
6. **No idempotency** - Retries cause duplicates

## Key References

- Celery 5.6 Docs: https://docs.celeryq.dev/
- ARQ Docs: https://arq-docs.helpmanual.io/
- Temporal Python SDK: https://docs.temporal.io/develop/python
- APScheduler: https://apscheduler.readthedocs.io/
