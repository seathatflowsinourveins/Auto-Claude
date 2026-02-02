# Cycle 22: Event-Driven Architecture & Message Queues (January 2026)

## Research Focus
Event-driven architecture patterns, Apache Kafka, RabbitMQ, and Python async messaging.

---

## 1. Event-Driven Architecture Fundamentals

### Core Concepts
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Producer   │ ──► │ Event Broker │ ──► │   Consumer   │
│  (emits)     │     │  (routes)    │     │  (reacts)    │
└──────────────┘     └──────────────┘     └──────────────┘

Key Benefits:
- Loose coupling (producers don't know consumers)
- Independent scaling
- Temporal decoupling (async processing)
- Resilience (queue buffers during failures)
```

### Seven Essential EDA Patterns

**1. Competing Consumers**
```
Multiple consumers share a queue, each processing different messages.
Use for: Scaling out heavy workloads

Queue ──► Consumer 1 (processes message A)
      ──► Consumer 2 (processes message B)
      ──► Consumer 3 (processes message C)
```

**2. Event Sourcing**
```python
# State = replay of all events (not current snapshot)
events = [
    {"type": "AccountCreated", "id": "123", "balance": 0},
    {"type": "Deposited", "amount": 100},
    {"type": "Withdrawn", "amount": 30},
]
# Current balance = 0 + 100 - 30 = 70

# Benefits: Complete audit trail, temporal queries, debugging
```

**3. CQRS (Command Query Responsibility Segregation)**
```python
# Separate read and write models

# Write side (commands)
class CommandHandler:
    async def handle_deposit(self, cmd: DepositCommand):
        event = Deposited(account_id=cmd.id, amount=cmd.amount)
        await self.event_store.append(event)
        await self.event_bus.publish(event)

# Read side (queries) - denormalized for fast reads
class QueryHandler:
    async def get_balance(self, account_id: str) -> float:
        return await self.read_db.get_balance(account_id)
```

**4. Event-Carried State Transfer (ECST)**
```python
# Event contains the data needed, no callback required
event = {
    "type": "OrderShipped",
    "order_id": "123",
    "customer_email": "user@example.com",  # Included!
    "shipping_address": {...}               # Included!
}
# Consumer doesn't need to fetch customer details
```

**5. Choreography (vs Orchestration)**
```
Choreography: Each service reacts to events independently
  OrderService → OrderCreated → PaymentService
                             → InventoryService
                             → NotificationService

Orchestration: Central coordinator directs flow
  Saga Orchestrator → PaymentService
                   → InventoryService
                   → NotificationService
```

**6. Saga Pattern (Distributed Transactions)**
```python
# Compensating transactions for rollback
class OrderSaga:
    steps = [
        ("reserve_inventory", "release_inventory"),
        ("charge_payment", "refund_payment"),
        ("ship_order", "cancel_shipment"),
    ]
    
    async def execute(self, order):
        completed = []
        for action, compensate in self.steps:
            try:
                await getattr(self, action)(order)
                completed.append(compensate)
            except Exception:
                # Rollback in reverse order
                for comp in reversed(completed):
                    await getattr(self, comp)(order)
                raise
```

**7. Outbox Pattern (Reliable Publishing)**
```python
# Atomic write to DB + outbox table
async with db.transaction():
    await db.save(order)
    await db.insert_outbox({
        "aggregate_id": order.id,
        "event_type": "OrderCreated",
        "payload": order.to_json()
    })

# Separate process polls outbox and publishes
async def outbox_processor():
    events = await db.get_pending_outbox_events()
    for event in events:
        await kafka.produce(event)
        await db.mark_outbox_published(event.id)
```

---

## 2. Apache Kafka Patterns

### Producer Best Practices (confluent-kafka)
```python
from confluent_kafka import Producer
import json

# Configuration
config = {
    'bootstrap.servers': 'localhost:9092',
    'acks': 'all',              # Wait for all replicas
    'retries': 3,
    'retry.backoff.ms': 100,
    'linger.ms': 5,             # Batch for 5ms
    'batch.size': 16384,        # 16KB batches
    'compression.type': 'snappy',
}

producer = Producer(config)

def delivery_callback(err, msg):
    if err:
        print(f"Delivery failed: {err}")
    else:
        print(f"Delivered to {msg.topic()}[{msg.partition()}]")

# Produce with callback
producer.produce(
    topic='orders',
    key=order_id.encode(),
    value=json.dumps(order).encode(),
    callback=delivery_callback
)

# CRITICAL: flush() or poll() to actually send
producer.poll(0)  # Trigger callbacks
producer.flush()  # Block until all sent
```

### Consumer Best Practices
```python
from confluent_kafka import Consumer, KafkaError

config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'order-processor',
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': False,  # Manual commit for safety
}

consumer = Consumer(config)
consumer.subscribe(['orders'])

try:
    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            raise KafkaException(msg.error())
        
        # Process message
        process_order(msg.value())
        
        # Manual commit after successful processing
        consumer.commit(asynchronous=False)
finally:
    consumer.close()
```

### Async Kafka (aiokafka)
```python
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import asyncio

# Async Producer
async def produce():
    producer = AIOKafkaProducer(bootstrap_servers='localhost:9092')
    await producer.start()
    try:
        await producer.send_and_wait('topic', b'message')
    finally:
        await producer.stop()

# Async Consumer
async def consume():
    consumer = AIOKafkaConsumer(
        'topic',
        bootstrap_servers='localhost:9092',
        group_id='my-group'
    )
    await consumer.start()
    try:
        async for msg in consumer:
            print(f"Received: {msg.value}")
    finally:
        await consumer.stop()
```

### Consumer Group Scaling
```
Topic: orders (6 partitions)

Consumer Group: order-processors
├── Consumer 1 → Partitions 0, 1
├── Consumer 2 → Partitions 2, 3
└── Consumer 3 → Partitions 4, 5

Rule: Max consumers = number of partitions
      More consumers than partitions = idle consumers
```

### Delivery Guarantees
| Guarantee | Config | Trade-off |
|-----------|--------|-----------|
| At-most-once | auto.commit=True, no retry | Fast, may lose |
| At-least-once | manual commit after process | Safe, may duplicate |
| Exactly-once | Transactional API + idempotent | Slow, complex |

---

## 3. RabbitMQ Patterns

### Exchange Types
```
1. Direct Exchange: Route by exact routing key match
   → Use for: Point-to-point, work queues

2. Fanout Exchange: Broadcast to all bound queues
   → Use for: Pub/sub, notifications

3. Topic Exchange: Route by pattern matching (*.error, logs.#)
   → Use for: Selective routing, log aggregation

4. Headers Exchange: Route by message headers
   → Use for: Complex routing logic
```

### Python with pika (Sync)
```python
import pika
import json

connection = pika.BlockingConnection(
    pika.ConnectionParameters('localhost')
)
channel = connection.channel()

# Declare durable queue (survives restart)
channel.queue_declare(queue='orders', durable=True)

# Publish with persistence
channel.basic_publish(
    exchange='',
    routing_key='orders',
    body=json.dumps(order),
    properties=pika.BasicProperties(
        delivery_mode=2,  # Persistent
        content_type='application/json',
    )
)

# Consumer with manual ack
def callback(ch, method, properties, body):
    try:
        process_order(json.loads(body))
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception:
        # Requeue on failure
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

channel.basic_qos(prefetch_count=10)  # Limit unacked messages
channel.basic_consume(queue='orders', on_message_callback=callback)
channel.start_consuming()
```

### Dead Letter Exchange (DLX)
```python
# Messages go to DLX when:
# 1. Rejected (nack with requeue=False)
# 2. TTL expired
# 3. Queue length exceeded
# 4. Delivery limit exceeded (quorum queues)

# Setup DLX
channel.exchange_declare('dlx', exchange_type='direct')
channel.queue_declare('dead_letters', durable=True)
channel.queue_bind('dead_letters', 'dlx', routing_key='orders')

# Main queue with DLX
channel.queue_declare(
    'orders',
    durable=True,
    arguments={
        'x-dead-letter-exchange': 'dlx',
        'x-dead-letter-routing-key': 'orders',
        'x-message-ttl': 300000,  # 5 min TTL
    }
)
```

### Async RabbitMQ (aio-pika)
```python
import aio_pika
import asyncio

async def main():
    connection = await aio_pika.connect_robust("amqp://localhost/")
    
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=10)
        
        queue = await channel.declare_queue("orders", durable=True)
        
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    print(f"Received: {message.body}")
                    # Auto-ack on context exit
```

### Delay Queue Pattern
```python
# RabbitMQ doesn't have native delay - use TTL + DLX

# 1. Create delay queue (messages expire to main queue)
channel.queue_declare(
    'delay_5min',
    arguments={
        'x-dead-letter-exchange': '',
        'x-dead-letter-routing-key': 'orders',
        'x-message-ttl': 300000,  # 5 minutes
    }
)

# 2. Publish to delay queue
channel.basic_publish(
    exchange='',
    routing_key='delay_5min',
    body=message
)
# After 5 min, message appears in 'orders' queue
```

---

## 4. Choosing Between Kafka and RabbitMQ

| Criterion | Kafka | RabbitMQ |
|-----------|-------|----------|
| **Use Case** | Event streaming, logs | Task queues, RPC |
| **Throughput** | Millions/sec | Thousands/sec |
| **Message Retention** | Configurable (days/weeks) | Until consumed |
| **Ordering** | Per partition | Per queue |
| **Consumer Model** | Pull (poll) | Push (callback) |
| **Replay** | Yes (offset reset) | No (consumed = gone) |
| **Complexity** | Higher (ZK/KRaft) | Lower |
| **Best For** | Analytics, CDC, logs | Microservices, tasks |

---

## 5. Anti-Patterns to Avoid

### General EDA
1. **God Event** - Event with too much data
2. **Missing Idempotency** - Duplicate processing
3. **Sync over Async** - Blocking for responses defeats purpose
4. **No Dead Letter Handling** - Lost messages

### Kafka
1. **Too Many Partitions** - Increases latency, ZK load
2. **Missing flush()** - Messages never sent
3. **Auto-commit without processing** - Data loss
4. **One consumer per message** - Use consumer groups

### RabbitMQ
1. **No Ack** - Message loss on crash
2. **Immediate Requeue on Failure** - Infinite loop
3. **No Prefetch Limit** - Memory exhaustion
4. **Forgetting Durability** - Lost on restart

---

## 6. Production Checklist

- [ ] Dead letter queue configured
- [ ] Idempotent consumers (handle duplicates)
- [ ] Monitoring on queue depth/lag
- [ ] Retry with exponential backoff
- [ ] Schema registry for message contracts
- [ ] Consumer group health checks
- [ ] Poison message handling
- [ ] Graceful shutdown (drain queue)

---

## Sources
- Solace EDA Patterns Guide (2026)
- Confluent Kafka Python Documentation
- RabbitMQ Dead Letter Exchange docs
- "Kafka: The Definitive Guide" - Neha Narkhede
- System Design Codex - EDA Patterns

*Researched: January 2026 | Cycle 22*
