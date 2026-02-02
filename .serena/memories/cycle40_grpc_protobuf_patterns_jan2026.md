# Cycle 40: gRPC & Protocol Buffers Production Patterns (January 2026)

## Overview
Comprehensive gRPC and Protocol Buffers patterns for high-performance Python microservices communication. Covers AsyncIO API, interceptors, streaming, schema evolution, and production deployment patterns.

---

## 1. Protocol Buffers Schema Design

### Basic Service Definition
```protobuf
syntax = "proto3";

package trading.v1;

option python_generic_services = true;

// Import common types
import "google/protobuf/timestamp.proto";
import "google/protobuf/empty.proto";

// Service definition
service OrderService {
  // Unary RPC
  rpc CreateOrder(CreateOrderRequest) returns (Order);
  
  // Server streaming
  rpc StreamOrderUpdates(OrderFilter) returns (stream OrderUpdate);
  
  // Client streaming
  rpc BatchCreateOrders(stream CreateOrderRequest) returns (BatchResult);
  
  // Bidirectional streaming
  rpc TradeStream(stream TradeRequest) returns (stream TradeResponse);
}

message Order {
  string id = 1;
  string symbol = 2;
  OrderSide side = 3;
  double quantity = 4;
  double price = 5;
  OrderStatus status = 6;
  google.protobuf.Timestamp created_at = 7;
}

enum OrderSide {
  ORDER_SIDE_UNSPECIFIED = 0;
  ORDER_SIDE_BUY = 1;
  ORDER_SIDE_SELL = 2;
}

enum OrderStatus {
  ORDER_STATUS_UNSPECIFIED = 0;
  ORDER_STATUS_PENDING = 1;
  ORDER_STATUS_FILLED = 2;
  ORDER_STATUS_CANCELLED = 3;
}
```

### Schema Evolution Best Practices
```protobuf
message Order {
  // RULE 1: Never reuse tag numbers
  string id = 1;
  string symbol = 2;
  
  // RULE 2: Reserve removed fields
  reserved 3, 4;  // Previously: old_field, deprecated_field
  reserved "old_field", "deprecated_field";
  
  // RULE 3: Use optional for nullable fields (proto3)
  optional double stop_price = 5;
  
  // RULE 4: Use wrapper types for nullable primitives
  google.protobuf.DoubleValue limit_price = 6;
  
  // RULE 5: Default values - be explicit about meaning
  int32 retry_count = 7;  // Default 0 means "no retries"
  
  // RULE 6: Oneof for mutually exclusive fields
  oneof execution_type {
    MarketExecution market = 10;
    LimitExecution limit = 11;
    StopExecution stop = 12;
  }
}

// RULE 7: Use nested messages for logical grouping
message CreateOrderRequest {
  message OrderParams {
    string symbol = 1;
    double quantity = 2;
  }
  OrderParams params = 1;
  string client_order_id = 2;
}
```

### Tag Number Management
```protobuf
// Tag numbers 1-15: 1 byte (use for frequent fields)
// Tag numbers 16-2047: 2 bytes
// Tag numbers 2048-262143: 3 bytes

message HighFrequencyMessage {
  // Most accessed fields get 1-15
  string id = 1;
  int64 timestamp = 2;
  double value = 3;
  
  // Less frequent fields get higher numbers
  string description = 16;
  map<string, string> metadata = 17;
}
```

---

## 2. gRPC AsyncIO Server Implementation

### Basic Async Server
```python
import grpc
from grpc import aio
from concurrent import futures
import asyncio

from trading.v1 import order_pb2, order_pb2_grpc

class OrderServiceServicer(order_pb2_grpc.OrderServiceServicer):
    """Async gRPC service implementation."""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
    
    async def CreateOrder(
        self,
        request: order_pb2.CreateOrderRequest,
        context: grpc.aio.ServicerContext
    ) -> order_pb2.Order:
        """Unary RPC - single request, single response."""
        # Access metadata
        metadata = dict(context.invocation_metadata())
        client_id = metadata.get("x-client-id", "unknown")
        
        # Create order in database
        async with self.db_pool.acquire() as conn:
            order = await conn.fetchrow(
                "INSERT INTO orders (symbol, quantity, price) VALUES ($1, $2, $3) RETURNING *",
                request.params.symbol,
                request.params.quantity,
                request.params.price
            )
        
        # Set response metadata
        await context.send_initial_metadata([
            ("x-order-id", order["id"]),
            ("x-request-id", metadata.get("x-request-id", "")),
        ])
        
        return order_pb2.Order(
            id=order["id"],
            symbol=order["symbol"],
            quantity=order["quantity"],
            status=order_pb2.ORDER_STATUS_PENDING
        )
    
    async def StreamOrderUpdates(
        self,
        request: order_pb2.OrderFilter,
        context: grpc.aio.ServicerContext
    ):
        """Server streaming - single request, stream of responses."""
        async for update in self._subscribe_to_updates(request.order_ids):
            if context.cancelled():
                break
            yield order_pb2.OrderUpdate(
                order_id=update["order_id"],
                status=update["status"],
                filled_quantity=update["filled_qty"]
            )
    
    async def BatchCreateOrders(
        self,
        request_iterator,
        context: grpc.aio.ServicerContext
    ) -> order_pb2.BatchResult:
        """Client streaming - stream of requests, single response."""
        orders_created = 0
        errors = []
        
        async for request in request_iterator:
            try:
                await self._create_single_order(request)
                orders_created += 1
            except Exception as e:
                errors.append(str(e))
        
        return order_pb2.BatchResult(
            success_count=orders_created,
            error_count=len(errors),
            errors=errors
        )
    
    async def TradeStream(
        self,
        request_iterator,
        context: grpc.aio.ServicerContext
    ):
        """Bidirectional streaming - stream both ways."""
        async def process_requests():
            async for request in request_iterator:
                yield await self._process_trade(request)
        
        async for response in process_requests():
            if context.cancelled():
                break
            yield response


async def serve():
    """Start the async gRPC server."""
    server = aio.server(
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 10000),
            ("grpc.keepalive_timeout_ms", 5000),
            ("grpc.keepalive_permit_without_calls", True),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.http2.min_time_between_pings_ms", 10000),
        ]
    )
    
    # Add service
    order_pb2_grpc.add_OrderServiceServicer_to_server(
        OrderServiceServicer(db_pool), server
    )
    
    # Add health checking
    from grpc_health.v1 import health, health_pb2_grpc
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("trading.v1.OrderService", health.HealthCheckResponse.SERVING)
    
    # Add reflection for debugging
    from grpc_reflection.v1alpha import reflection
    SERVICE_NAMES = (
        order_pb2.DESCRIPTOR.services_by_name['OrderService'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    
    listen_addr = "[::]:50051"
    server.add_insecure_port(listen_addr)
    
    await server.start()
    print(f"Server started on {listen_addr}")
    
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
```

---

## 3. gRPC AsyncIO Client Implementation

### Async Client with Connection Management
```python
import grpc
from grpc import aio
from contextlib import asynccontextmanager
from typing import AsyncIterator
import asyncio

from trading.v1 import order_pb2, order_pb2_grpc


class OrderClient:
    """Async gRPC client with connection management."""
    
    def __init__(self, target: str = "localhost:50051"):
        self.target = target
        self._channel: aio.Channel | None = None
        self._stub: order_pb2_grpc.OrderServiceStub | None = None
    
    async def connect(self):
        """Establish channel connection."""
        self._channel = aio.insecure_channel(
            self.target,
            options=[
                ("grpc.keepalive_time_ms", 10000),
                ("grpc.keepalive_timeout_ms", 5000),
                ("grpc.keepalive_permit_without_calls", True),
                ("grpc.enable_retries", 1),
                ("grpc.service_config", '''{
                    "methodConfig": [{
                        "name": [{"service": "trading.v1.OrderService"}],
                        "retryPolicy": {
                            "maxAttempts": 3,
                            "initialBackoff": "0.1s",
                            "maxBackoff": "1s",
                            "backoffMultiplier": 2,
                            "retryableStatusCodes": ["UNAVAILABLE", "DEADLINE_EXCEEDED"]
                        }
                    }]
                }''')
            ]
        )
        self._stub = order_pb2_grpc.OrderServiceStub(self._channel)
    
    async def close(self):
        """Close channel."""
        if self._channel:
            await self._channel.close()
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, *args):
        await self.close()
    
    async def create_order(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timeout: float = 5.0
    ) -> order_pb2.Order:
        """Create order with timeout and metadata."""
        request = order_pb2.CreateOrderRequest(
            params=order_pb2.CreateOrderRequest.OrderParams(
                symbol=symbol,
                quantity=quantity
            )
        )
        
        # Add metadata
        metadata = [
            ("x-client-id", "trading-bot-1"),
            ("x-request-id", str(uuid.uuid4())),
        ]
        
        try:
            response = await self._stub.CreateOrder(
                request,
                timeout=timeout,
                metadata=metadata
            )
            return response
        except aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                raise TimeoutError(f"Order creation timed out after {timeout}s")
            elif e.code() == grpc.StatusCode.UNAVAILABLE:
                raise ConnectionError("gRPC server unavailable")
            raise
    
    async def stream_updates(
        self,
        order_ids: list[str]
    ) -> AsyncIterator[order_pb2.OrderUpdate]:
        """Stream order updates (server streaming)."""
        request = order_pb2.OrderFilter(order_ids=order_ids)
        
        async for update in self._stub.StreamOrderUpdates(request):
            yield update
    
    async def batch_create(
        self,
        orders: list[dict]
    ) -> order_pb2.BatchResult:
        """Batch create orders (client streaming)."""
        async def request_generator():
            for order in orders:
                yield order_pb2.CreateOrderRequest(
                    params=order_pb2.CreateOrderRequest.OrderParams(**order)
                )
        
        return await self._stub.BatchCreateOrders(request_generator())
    
    async def trade_stream(
        self,
        trades: AsyncIterator[dict]
    ) -> AsyncIterator[order_pb2.TradeResponse]:
        """Bidirectional trade stream."""
        async def request_generator():
            async for trade in trades:
                yield order_pb2.TradeRequest(**trade)
        
        async for response in self._stub.TradeStream(request_generator()):
            yield response


# Usage
async def main():
    async with OrderClient("localhost:50051") as client:
        # Unary call
        order = await client.create_order("AAPL", 100, 150.00)
        print(f"Created: {order.id}")
        
        # Server streaming
        async for update in client.stream_updates([order.id]):
            print(f"Update: {update.status}")
```

---

## 4. Interceptors (Cross-Cutting Concerns)

### Server Interceptors
```python
from grpc import aio
import time
import structlog
from typing import Callable, Any

logger = structlog.get_logger()


class LoggingInterceptor(aio.ServerInterceptor):
    """Log all RPC calls with timing."""
    
    async def intercept_service(
        self,
        continuation: Callable[[Any], Any],
        handler_call_details: grpc.HandlerCallDetails
    ):
        start_time = time.perf_counter()
        method = handler_call_details.method
        
        logger.info("rpc_started", method=method)
        
        try:
            response = await continuation(handler_call_details)
            duration = time.perf_counter() - start_time
            logger.info("rpc_completed", method=method, duration_ms=duration * 1000)
            return response
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error("rpc_failed", method=method, duration_ms=duration * 1000, error=str(e))
            raise


class AuthInterceptor(aio.ServerInterceptor):
    """Validate authentication tokens."""
    
    def __init__(self, auth_service):
        self.auth_service = auth_service
    
    async def intercept_service(
        self,
        continuation: Callable[[Any], Any],
        handler_call_details: grpc.HandlerCallDetails
    ):
        # Extract metadata
        metadata = dict(handler_call_details.invocation_metadata or [])
        token = metadata.get("authorization", "")
        
        if not token.startswith("Bearer "):
            raise grpc.RpcError(
                grpc.StatusCode.UNAUTHENTICATED,
                "Missing or invalid authorization header"
            )
        
        # Validate token
        try:
            user = await self.auth_service.validate(token[7:])
            # Inject user into context (via metadata)
            handler_call_details.invocation_metadata.append(
                ("x-user-id", user.id)
            )
        except Exception:
            raise grpc.RpcError(
                grpc.StatusCode.UNAUTHENTICATED,
                "Invalid token"
            )
        
        return await continuation(handler_call_details)


class RateLimitInterceptor(aio.ServerInterceptor):
    """Rate limit by client ID."""
    
    def __init__(self, redis_client, limit: int = 100, window: int = 60):
        self.redis = redis_client
        self.limit = limit
        self.window = window
    
    async def intercept_service(
        self,
        continuation: Callable[[Any], Any],
        handler_call_details: grpc.HandlerCallDetails
    ):
        metadata = dict(handler_call_details.invocation_metadata or [])
        client_id = metadata.get("x-client-id", "anonymous")
        
        key = f"ratelimit:{client_id}"
        current = await self.redis.incr(key)
        if current == 1:
            await self.redis.expire(key, self.window)
        
        if current > self.limit:
            raise grpc.RpcError(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f"Rate limit exceeded: {self.limit} requests per {self.window}s"
            )
        
        return await continuation(handler_call_details)


# Apply interceptors to server
server = aio.server(
    interceptors=[
        LoggingInterceptor(),
        AuthInterceptor(auth_service),
        RateLimitInterceptor(redis_client),
    ]
)
```

### Client Interceptors
```python
from grpc import aio
import time


class ClientLoggingInterceptor(aio.UnaryUnaryClientInterceptor):
    """Log client-side RPC calls."""
    
    async def intercept_unary_unary(
        self,
        continuation,
        client_call_details,
        request
    ):
        start = time.perf_counter()
        method = client_call_details.method
        
        try:
            response = await continuation(client_call_details, request)
            duration = time.perf_counter() - start
            logger.info("client_rpc_success", method=method, duration_ms=duration * 1000)
            return response
        except aio.AioRpcError as e:
            duration = time.perf_counter() - start
            logger.error("client_rpc_error", method=method, code=e.code().name, duration_ms=duration * 1000)
            raise


class RetryInterceptor(aio.UnaryUnaryClientInterceptor):
    """Retry failed calls with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 0.1):
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    async def intercept_unary_unary(
        self,
        continuation,
        client_call_details,
        request
    ):
        retries = 0
        last_error = None
        
        while retries <= self.max_retries:
            try:
                return await continuation(client_call_details, request)
            except aio.AioRpcError as e:
                if e.code() not in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
                    raise
                
                last_error = e
                retries += 1
                if retries <= self.max_retries:
                    delay = self.base_delay * (2 ** (retries - 1))
                    await asyncio.sleep(delay)
        
        raise last_error


# Apply to channel
channel = aio.insecure_channel(
    "localhost:50051",
    interceptors=[
        ClientLoggingInterceptor(),
        RetryInterceptor(max_retries=3),
    ]
)
```

### Using grpc-interceptor Library (Simplified)
```python
from grpc_interceptor import ServerInterceptor
from grpc_interceptor.exceptions import GrpcException


class SimpleAuthInterceptor(ServerInterceptor):
    """Simplified interceptor using grpc-interceptor library."""
    
    def intercept(self, method, request, context, method_name):
        metadata = dict(context.invocation_metadata())
        
        if "authorization" not in metadata:
            raise GrpcException(
                status_code=grpc.StatusCode.UNAUTHENTICATED,
                details="Missing authorization"
            )
        
        # Call the actual method
        return method(request, context)
```

---

## 5. Health Checking Implementation

### Server-Side Health Service
```python
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
import asyncio


class HealthManager:
    """Manage service health status."""
    
    def __init__(self, health_servicer: health.HealthServicer):
        self.servicer = health_servicer
        self._checks: dict[str, Callable[[], bool]] = {}
    
    def register_check(self, service_name: str, check_fn: Callable[[], bool]):
        """Register a health check function."""
        self._checks[service_name] = check_fn
    
    async def run_checks(self):
        """Periodically run health checks."""
        while True:
            for service_name, check_fn in self._checks.items():
                try:
                    is_healthy = await asyncio.to_thread(check_fn)
                    status = (
                        health_pb2.HealthCheckResponse.SERVING
                        if is_healthy
                        else health_pb2.HealthCheckResponse.NOT_SERVING
                    )
                except Exception:
                    status = health_pb2.HealthCheckResponse.NOT_SERVING
                
                self.servicer.set(service_name, status)
            
            await asyncio.sleep(10)  # Check every 10 seconds


# Usage in server setup
health_servicer = health.HealthServicer()
health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

health_manager = HealthManager(health_servicer)
health_manager.register_check("trading.v1.OrderService", lambda: db_pool.is_connected())
health_manager.register_check("", lambda: True)  # Overall server health

asyncio.create_task(health_manager.run_checks())
```

### Client-Side Health Checking
```python
from grpc_health.v1 import health_pb2, health_pb2_grpc


async def check_server_health(channel: aio.Channel, service: str = "") -> bool:
    """Check if a service is healthy."""
    stub = health_pb2_grpc.HealthStub(channel)
    
    try:
        response = await stub.Check(
            health_pb2.HealthCheckRequest(service=service),
            timeout=5.0
        )
        return response.status == health_pb2.HealthCheckResponse.SERVING
    except aio.AioRpcError:
        return False


async def wait_for_healthy(channel: aio.Channel, timeout: float = 30.0):
    """Wait for server to become healthy."""
    stub = health_pb2_grpc.HealthStub(channel)
    deadline = time.time() + timeout
    
    while time.time() < deadline:
        try:
            response = await stub.Check(
                health_pb2.HealthCheckRequest(service=""),
                timeout=2.0
            )
            if response.status == health_pb2.HealthCheckResponse.SERVING:
                return True
        except aio.AioRpcError:
            pass
        
        await asyncio.sleep(0.5)
    
    raise TimeoutError(f"Server not healthy after {timeout}s")
```

---

## 6. Bidirectional Streaming Patterns

### Real-Time Price Feed
```python
class PriceFeedServicer(price_pb2_grpc.PriceFeedServicer):
    """Bidirectional streaming for real-time prices."""
    
    async def StreamPrices(self, request_iterator, context):
        """Client sends subscriptions, server sends price updates."""
        subscriptions: set[str] = set()
        
        async def handle_subscriptions():
            """Process incoming subscription changes."""
            async for request in request_iterator:
                if request.action == price_pb2.SUBSCRIBE:
                    subscriptions.add(request.symbol)
                elif request.action == price_pb2.UNSUBSCRIBE:
                    subscriptions.discard(request.symbol)
        
        async def send_prices():
            """Send prices for subscribed symbols."""
            async for price in self.price_feed.subscribe_all():
                if price.symbol in subscriptions:
                    yield price_pb2.PriceUpdate(
                        symbol=price.symbol,
                        bid=price.bid,
                        ask=price.ask,
                        timestamp=price.timestamp
                    )
        
        # Run both concurrently
        subscription_task = asyncio.create_task(handle_subscriptions())
        
        try:
            async for update in send_prices():
                if context.cancelled():
                    break
                yield update
        finally:
            subscription_task.cancel()
```

### Chat/Messaging Pattern
```python
class ChatServicer(chat_pb2_grpc.ChatServicer):
    """Bidirectional chat streaming."""
    
    def __init__(self):
        self.rooms: dict[str, asyncio.Queue] = {}
    
    async def Chat(self, request_iterator, context):
        """Bidirectional chat messages."""
        user_id = self._get_user_id(context)
        room_id = None
        
        async def receive_messages():
            nonlocal room_id
            async for msg in request_iterator:
                if msg.HasField("join"):
                    room_id = msg.join.room_id
                    await self._join_room(room_id, user_id)
                elif msg.HasField("message"):
                    await self._broadcast(room_id, user_id, msg.message.text)
                elif msg.HasField("leave"):
                    await self._leave_room(room_id, user_id)
        
        async def send_messages():
            while True:
                if room_id and room_id in self.rooms:
                    try:
                        msg = await asyncio.wait_for(
                            self.rooms[room_id].get(),
                            timeout=1.0
                        )
                        yield msg
                    except asyncio.TimeoutError:
                        continue
                else:
                    await asyncio.sleep(0.1)
        
        receive_task = asyncio.create_task(receive_messages())
        
        try:
            async for msg in send_messages():
                if context.cancelled():
                    break
                yield msg
        finally:
            receive_task.cancel()
            if room_id:
                await self._leave_room(room_id, user_id)
```

---

## 7. Production Deployment Patterns

### Load Balancing Configuration
```python
# Client-side load balancing
channel = aio.insecure_channel(
    "dns:///my-service.default.svc.cluster.local:50051",
    options=[
        ("grpc.lb_policy_name", "round_robin"),
        # Or use xDS for dynamic config
        # ("grpc.lb_policy_name", "xds"),
    ]
)
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: order-service
          image: order-service:latest
          ports:
            - containerPort: 50051
              name: grpc
          readinessProbe:
            grpc:
              port: 50051
              service: ""  # Overall health
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            grpc:
              port: 50051
            initialDelaySeconds: 15
            periodSeconds: 20
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: order-service
spec:
  ports:
    - port: 50051
      targetPort: 50051
      name: grpc
  selector:
    app: order-service
```

### Graceful Shutdown
```python
import signal


async def serve_with_graceful_shutdown():
    server = aio.server()
    # ... add services ...
    
    await server.start()
    
    async def shutdown(sig):
        print(f"Received {sig.name}, shutting down...")
        # Stop accepting new connections
        await server.stop(grace=10)  # 10 second grace period
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s)))
    
    await server.wait_for_termination()
```

---

## 8. Error Handling Patterns

### Structured Error Responses
```python
from google.rpc import status_pb2, error_details_pb2
from grpc_status import rpc_status


def create_detailed_error(
    code: grpc.StatusCode,
    message: str,
    field_violations: list[tuple[str, str]] = None
) -> grpc.RpcError:
    """Create error with rich details."""
    detail = status_pb2.Status(
        code=code.value[0],
        message=message
    )
    
    if field_violations:
        bad_request = error_details_pb2.BadRequest()
        for field, description in field_violations:
            violation = bad_request.field_violations.add()
            violation.field = field
            violation.description = description
        detail.details.add().Pack(bad_request)
    
    return rpc_status.to_status(detail).exception()


# Usage in servicer
async def CreateOrder(self, request, context):
    errors = self.validate_order(request)
    if errors:
        raise create_detailed_error(
            grpc.StatusCode.INVALID_ARGUMENT,
            "Order validation failed",
            field_violations=errors
        )
```

### Client-Side Error Handling
```python
from grpc_status import rpc_status
from google.rpc import error_details_pb2


async def create_order_with_error_handling(client, request):
    try:
        return await client.CreateOrder(request)
    except aio.AioRpcError as e:
        # Extract rich error details
        status = rpc_status.from_call(e)
        
        for detail in status.details:
            if detail.Is(error_details_pb2.BadRequest.DESCRIPTOR):
                bad_request = error_details_pb2.BadRequest()
                detail.Unpack(bad_request)
                
                for violation in bad_request.field_violations:
                    print(f"Field {violation.field}: {violation.description}")
        
        raise
```

---

## 9. Testing Patterns

### Unit Testing with Mock
```python
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_db_pool():
    pool = AsyncMock()
    pool.acquire.return_value.__aenter__.return_value = AsyncMock()
    return pool


@pytest.mark.asyncio
async def test_create_order(mock_db_pool):
    # Arrange
    servicer = OrderServiceServicer(mock_db_pool)
    context = MagicMock(spec=grpc.aio.ServicerContext)
    context.invocation_metadata.return_value = [("x-client-id", "test")]
    context.send_initial_metadata = AsyncMock()
    
    request = order_pb2.CreateOrderRequest(
        params=order_pb2.CreateOrderRequest.OrderParams(
            symbol="AAPL",
            quantity=100
        )
    )
    
    # Mock DB response
    mock_db_pool.acquire.return_value.__aenter__.return_value.fetchrow.return_value = {
        "id": "order-123",
        "symbol": "AAPL",
        "quantity": 100
    }
    
    # Act
    result = await servicer.CreateOrder(request, context)
    
    # Assert
    assert result.id == "order-123"
    assert result.symbol == "AAPL"
```

### Integration Testing
```python
import pytest
from grpc import aio


@pytest.fixture
async def grpc_server():
    server = aio.server()
    order_pb2_grpc.add_OrderServiceServicer_to_server(
        OrderServiceServicer(real_db_pool), server
    )
    port = server.add_insecure_port("[::]:0")
    await server.start()
    yield f"localhost:{port}"
    await server.stop(0)


@pytest.fixture
async def grpc_client(grpc_server):
    async with OrderClient(grpc_server) as client:
        yield client


@pytest.mark.asyncio
async def test_end_to_end_order_flow(grpc_client):
    # Create order
    order = await grpc_client.create_order("AAPL", 100, 150.0)
    assert order.status == order_pb2.ORDER_STATUS_PENDING
    
    # Stream updates
    updates = []
    async for update in grpc_client.stream_updates([order.id]):
        updates.append(update)
        if update.status == order_pb2.ORDER_STATUS_FILLED:
            break
    
    assert len(updates) > 0
```

---

## Anti-Patterns to Avoid

### 1. Blocking in Async Context
```python
# WRONG: Blocking call in async servicer
async def CreateOrder(self, request, context):
    result = self.db.execute(...)  # Blocks event loop!

# CORRECT: Use async or run in executor
async def CreateOrder(self, request, context):
    result = await self.db.execute(...)
    # Or: result = await asyncio.to_thread(self.db.execute, ...)
```

### 2. Missing Deadline Propagation
```python
# WRONG: Ignoring incoming deadline
async def CreateOrder(self, request, context):
    await self.slow_operation()  # Ignores client deadline

# CORRECT: Respect and propagate deadlines
async def CreateOrder(self, request, context):
    remaining = context.time_remaining()
    if remaining < 1.0:
        raise grpc.RpcError(grpc.StatusCode.DEADLINE_EXCEEDED, "Insufficient time")
    
    await asyncio.wait_for(self.slow_operation(), timeout=remaining - 0.5)
```

### 3. Resource Leaks in Streaming
```python
# WRONG: Not handling cancellation
async def StreamUpdates(self, request, context):
    async for update in self.updates():
        yield update  # Never checks if client disconnected

# CORRECT: Check cancellation
async def StreamUpdates(self, request, context):
    async for update in self.updates():
        if context.cancelled():
            return
        yield update
```

---

*Cycle 40 Complete: gRPC & Protocol Buffers patterns for high-performance Python microservices*
