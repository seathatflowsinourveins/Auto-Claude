# Cycle 59: WebSocket & Real-time Patterns (Jan 2026)

Production real-time communication patterns from official documentation.

## Python websockets Library (v16.0)

### Consumer Pattern - Receiving Messages

```python
import websockets

async def consumer_handler(websocket):
    """Process incoming messages from WebSocket."""
    async for message in websocket:
        await consume(message)  # Your business logic
    # Iteration terminates when client disconnects

async def consume(message: str) -> None:
    """Process a single message."""
    data = json.loads(message)
    # Handle the message
```

### Producer Pattern - Sending Messages

```python
from websockets.exceptions import ConnectionClosed

async def producer_handler(websocket):
    """Send messages to WebSocket."""
    while True:
        try:
            message = await produce()  # Your business logic
            await websocket.send(message)
        except ConnectionClosed:
            break  # Client disconnected

async def produce() -> str:
    """Generate next message to send."""
    # Your message generation logic
    return json.dumps({"type": "update", "data": {...}})
```

### Consumer + Producer (Bidirectional)

```python
import asyncio

async def handler(websocket):
    """Handle both sending and receiving concurrently."""
    consumer_task = asyncio.create_task(consumer_handler(websocket))
    producer_task = asyncio.create_task(producer_handler(websocket))
    
    # Wait for first task to complete, cancel the other
    done, pending = await asyncio.wait(
        [consumer_task, producer_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    
    for task in pending:
        task.cancel()
```

### Connection Registration Pattern

```python
connected: set[websockets.WebSocketServerProtocol] = set()

async def handler(websocket):
    """Track connected clients."""
    # Register
    connected.add(websocket)
    try:
        # Broadcast to all connected clients
        websockets.broadcast(connected, "Hello!")
        await asyncio.sleep(10)
    finally:
        # Unregister on disconnect
        connected.remove(websocket)
```

### Server Setup

```python
import websockets

async def main():
    async with websockets.serve(handler, "localhost", 8765) as server:
        await server.serve_forever()

# With custom configuration
async def main_configured():
    async with websockets.serve(
        handler,
        "localhost",
        8765,
        ping_interval=20,      # Keepalive ping every 20s
        ping_timeout=10,       # Wait 10s for pong
        close_timeout=5,       # Wait 5s for close handshake
        max_size=2**20,        # 1MB max message size
        compression=None,      # Disable compression for low latency
    ) as server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
```

## FastAPI WebSocket Patterns

### Basic WebSocket Endpoint

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Echo: {data}")
```

### ConnectionManager for Broadcasting

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

class ConnectionManager:
    """Manage WebSocket connections for broadcasting."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal(f"You: {data}", websocket)
            await manager.broadcast(f"Client #{client_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{client_id} left")
```

### WebSocket with Dependencies

```python
from typing import Annotated
from fastapi import (
    Cookie, Depends, Query, WebSocket, 
    WebSocketException, status
)

async def get_token(
    websocket: WebSocket,
    session: Annotated[str | None, Cookie()] = None,
    token: Annotated[str | None, Query()] = None,
) -> str:
    """Authenticate WebSocket connection."""
    if session is None and token is None:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
    return session or token

@app.websocket("/ws/{item_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    item_id: str,
    token: Annotated[str, Depends(get_token)],
):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Item {item_id}, Token {token}: {data}")
```

### Room-Based Broadcasting

```python
from collections import defaultdict

class RoomManager:
    """Manage WebSocket connections by room."""
    
    def __init__(self):
        self.rooms: dict[str, set[WebSocket]] = defaultdict(set)
    
    async def join(self, room: str, websocket: WebSocket):
        await websocket.accept()
        self.rooms[room].add(websocket)
    
    def leave(self, room: str, websocket: WebSocket):
        self.rooms[room].discard(websocket)
        if not self.rooms[room]:
            del self.rooms[room]
    
    async def broadcast_to_room(self, room: str, message: str):
        for ws in self.rooms.get(room, []):
            await ws.send_text(message)

rooms = RoomManager()

@app.websocket("/ws/room/{room_id}")
async def room_websocket(websocket: WebSocket, room_id: str):
    await rooms.join(room_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await rooms.broadcast_to_room(room_id, data)
    except WebSocketDisconnect:
        rooms.leave(room_id, websocket)
```

## Server-Sent Events (SSE)

### When to Use SSE vs WebSocket

| Feature | SSE | WebSocket |
|---------|-----|-----------|
| Direction | Server → Client only | Bidirectional |
| Protocol | HTTP | WS (upgrade from HTTP) |
| Reconnection | Automatic | Manual |
| Complexity | Simple | More complex |
| Firewall | HTTP-friendly | May be blocked |
| Use case | Notifications, feeds | Chat, gaming |

### FastAPI SSE with StreamingResponse

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

async def event_generator():
    """Generate SSE events."""
    counter = 0
    while True:
        counter += 1
        # SSE format: "data: <message>\n\n"
        yield f"data: Event {counter}\n\n"
        await asyncio.sleep(1)

@app.get("/events")
async def stream_events():
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
```

### SSE with sse-starlette Library

```python
# pip install sse-starlette
from sse_starlette.sse import EventSourceResponse
from fastapi import FastAPI, Request
import asyncio

app = FastAPI()

async def event_publisher(request: Request):
    """Publish events, checking for client disconnect."""
    counter = 0
    while True:
        if await request.is_disconnected():
            break
        
        counter += 1
        yield {
            "event": "message",
            "id": str(counter),
            "data": f"Counter: {counter}"
        }
        await asyncio.sleep(1)

@app.get("/sse")
async def sse_endpoint(request: Request):
    return EventSourceResponse(event_publisher(request))
```

### SSE with Named Events

```python
async def multi_event_generator():
    """Generate different event types."""
    while True:
        # Named event
        yield {
            "event": "heartbeat",
            "data": "ping"
        }
        await asyncio.sleep(5)
        
        # Data event (default)
        yield {
            "event": "update",
            "data": json.dumps({"status": "active"})
        }
        await asyncio.sleep(1)
```

### Client-Side SSE (JavaScript)

```javascript
// Browser client for SSE
const eventSource = new EventSource('/sse');

eventSource.onmessage = (event) => {
    console.log('Message:', event.data);
};

eventSource.addEventListener('heartbeat', (event) => {
    console.log('Heartbeat:', event.data);
});

eventSource.onerror = (error) => {
    console.error('SSE Error:', error);
    eventSource.close();
};
```

## Scaling WebSockets with Redis Pub/Sub

```python
import redis.asyncio as redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import json

app = FastAPI()
redis_client = redis.from_url("redis://localhost:6379")

class PubSubManager:
    """Scale WebSockets across multiple processes with Redis."""
    
    def __init__(self):
        self.local_connections: dict[str, set[WebSocket]] = {}
    
    async def connect(self, channel: str, websocket: WebSocket):
        await websocket.accept()
        if channel not in self.local_connections:
            self.local_connections[channel] = set()
            # Start Redis subscriber for this channel
            asyncio.create_task(self._subscribe(channel))
        self.local_connections[channel].add(websocket)
    
    async def _subscribe(self, channel: str):
        """Listen to Redis channel and broadcast to local connections."""
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel)
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = message["data"].decode()
                await self._broadcast_local(channel, data)
    
    async def _broadcast_local(self, channel: str, message: str):
        """Broadcast to local WebSocket connections."""
        for ws in self.local_connections.get(channel, []):
            await ws.send_text(message)
    
    async def publish(self, channel: str, message: str):
        """Publish to Redis (reaches all processes)."""
        await redis_client.publish(channel, message)

pubsub = PubSubManager()

@app.websocket("/ws/channel/{channel}")
async def channel_websocket(websocket: WebSocket, channel: str):
    await pubsub.connect(channel, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await pubsub.publish(channel, data)
    except WebSocketDisconnect:
        pubsub.local_connections[channel].discard(websocket)
```

## Anti-Patterns

### ❌ No Graceful Disconnect Handling

```python
# BAD: No disconnect handling
@app.websocket("/ws")
async def bad_ws(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()  # Crashes on disconnect!

# GOOD: Handle disconnects
@app.websocket("/ws")
async def good_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        # Cleanup
        pass
```

### ❌ In-Memory State Without Process Awareness

```python
# BAD: Won't work with multiple workers
connections = []  # Lost when process restarts

# GOOD: Use Redis or database for multi-process
# See PubSubManager pattern above
```

### ❌ Blocking the Event Loop

```python
# BAD: Blocking call in async handler
@app.websocket("/ws")
async def bad_ws(websocket: WebSocket):
    await websocket.accept()
    result = slow_sync_function()  # Blocks all connections!

# GOOD: Use run_in_executor
import asyncio

@app.websocket("/ws")
async def good_ws(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, slow_sync_function)
```

## Decision Matrix

| Use Case | Technology | Why |
|----------|------------|-----|
| Chat/messaging | WebSocket | Bidirectional, low latency |
| Live notifications | SSE | Simpler, HTTP-native |
| Live dashboard | SSE or WebSocket | SSE if server-push only |
| Gaming | WebSocket | Bidirectional, binary support |
| Stock ticker | SSE | Server-push, auto-reconnect |
| Collaborative editing | WebSocket | Real-time sync needed |
| File upload progress | SSE | Server → client updates |

## Quick Reference

```python
# websockets library - server
import websockets
async with websockets.serve(handler, "localhost", 8765):
    await asyncio.Future()  # Run forever

# FastAPI WebSocket
from fastapi import WebSocket, WebSocketDisconnect
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        pass

# FastAPI SSE
from fastapi.responses import StreamingResponse
async def events():
    yield "data: hello\n\n"
return StreamingResponse(events(), media_type="text/event-stream")

# sse-starlette
from sse_starlette.sse import EventSourceResponse
return EventSourceResponse(event_generator(request))
```
