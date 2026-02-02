# Cycle 36: WebSockets & Real-Time Patterns (January 2026)

## Overview
Production-grade WebSocket patterns for Python, covering FastAPI/Starlette, python-socketio, horizontal scaling with Redis pub/sub, and connection management.

---

## FastAPI WebSocket Fundamentals

### Basic WebSocket Endpoint
```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        print("Client disconnected")
```

### Connection Manager Pattern
```python
from typing import Dict, Set
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.rooms: Dict[str, Set[str]] = {}  # room -> user_ids
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
    
    def disconnect(self, user_id: str):
        self.active_connections.pop(user_id, None)
        # Remove from all rooms
        for room in self.rooms.values():
            room.discard(user_id)
    
    async def send_personal(self, user_id: str, message: dict):
        if ws := self.active_connections.get(user_id):
            await ws.send_json(message)
    
    async def broadcast(self, message: dict, exclude: str = None):
        for user_id, ws in self.active_connections.items():
            if user_id != exclude:
                await ws.send_json(message)
    
    async def broadcast_to_room(self, room: str, message: dict):
        for user_id in self.rooms.get(room, set()):
            await self.send_personal(user_id, message)
    
    def join_room(self, user_id: str, room: str):
        self.rooms.setdefault(room, set()).add(user_id)
    
    def leave_room(self, user_id: str, room: str):
        if room in self.rooms:
            self.rooms[room].discard(user_id)

manager = ConnectionManager()
```

---

## WebSocket Authentication

### Token-Based Auth (Query Parameter)
```python
from fastapi import WebSocket, Query, HTTPException, status

async def get_current_user(token: str = Query(...)):
    user = await verify_token(token)
    if not user:
        raise HTTPException(status_code=status.WS_1008_POLICY_VIOLATION)
    return user

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(...)
):
    try:
        user = await get_current_user(token)
    except HTTPException:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    await manager.connect(websocket, user.id)
    # ...
```

### First-Message Auth Pattern
```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # First message must be auth
    try:
        auth_data = await asyncio.wait_for(
            websocket.receive_json(),
            timeout=5.0  # 5 second auth timeout
        )
        user = await verify_token(auth_data.get("token"))
        if not user:
            await websocket.close(code=4001)  # Custom auth failure code
            return
    except asyncio.TimeoutError:
        await websocket.close(code=4002)  # Auth timeout
        return
    
    # Authenticated - proceed with connection
    await websocket.send_json({"type": "auth_success", "user_id": user.id})
```

---

## Heartbeat & Keepalive

### Server-Side Heartbeat
```python
import asyncio

async def heartbeat(websocket: WebSocket, interval: int = 30):
    """Send periodic pings to detect dead connections."""
    while True:
        try:
            await asyncio.sleep(interval)
            await websocket.send_json({"type": "ping", "ts": time.time()})
        except Exception:
            break

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Start heartbeat task
    heartbeat_task = asyncio.create_task(heartbeat(websocket))
    
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "pong":
                continue  # Heartbeat response
            # Handle other messages...
    finally:
        heartbeat_task.cancel()
```

### Connection Timeout Detection
```python
async def receive_with_timeout(websocket: WebSocket, timeout: float = 60.0):
    """Disconnect if no message received within timeout."""
    try:
        return await asyncio.wait_for(websocket.receive_json(), timeout=timeout)
    except asyncio.TimeoutError:
        await websocket.close(code=4003)  # Idle timeout
        raise WebSocketDisconnect()
```

---

## Horizontal Scaling with Redis Pub/Sub

### The Problem
Single-server WebSocket: messages only reach clients on same server.
Multi-server: need Redis pub/sub to broadcast across all instances.

### Redis Pub/Sub Manager
```python
import aioredis
import json
from contextlib import asynccontextmanager

class RedisPubSubManager:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.pubsub = None
        self.redis = None
    
    async def connect(self):
        self.redis = await aioredis.from_url(self.redis_url)
        self.pubsub = self.redis.pubsub()
    
    async def disconnect(self):
        if self.pubsub:
            await self.pubsub.close()
        if self.redis:
            await self.redis.close()
    
    async def publish(self, channel: str, message: dict):
        await self.redis.publish(channel, json.dumps(message))
    
    async def subscribe(self, channel: str):
        await self.pubsub.subscribe(channel)
    
    async def listen(self):
        async for message in self.pubsub.listen():
            if message["type"] == "message":
                yield json.loads(message["data"])

pubsub_manager = RedisPubSubManager()
```

### Integrated Connection Manager with Redis
```python
class ScalableConnectionManager:
    def __init__(self, pubsub: RedisPubSubManager):
        self.local_connections: Dict[str, WebSocket] = {}
        self.pubsub = pubsub
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.local_connections[user_id] = websocket
    
    async def broadcast_to_room(self, room: str, message: dict, via_redis: bool = True):
        if via_redis:
            # Publish to Redis - all servers receive this
            await self.pubsub.publish(f"room:{room}", message)
        else:
            # Local only (for messages originating from Redis subscription)
            for user_id, ws in self.local_connections.items():
                if user_id in self.get_room_members(room):
                    await ws.send_json(message)
    
    async def redis_listener(self, room: str):
        """Background task to receive Redis messages."""
        await self.pubsub.subscribe(f"room:{room}")
        async for message in self.pubsub.listen():
            await self.broadcast_to_room(room, message, via_redis=False)
```

### FastAPI Lifespan Integration
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await pubsub_manager.connect()
    asyncio.create_task(redis_listener_task())
    yield
    # Shutdown
    await pubsub_manager.disconnect()

app = FastAPI(lifespan=lifespan)
```

---

## python-socketio (Alternative to Raw WebSockets)

### Server Setup with FastAPI
```python
import socketio

# Create Socket.IO server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    logger=True
)

# Wrap with ASGI app
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

@sio.event
async def connect(sid, environ, auth):
    # auth contains client-provided auth data
    if not await verify_auth(auth):
        raise socketio.exceptions.ConnectionRefusedError('Authentication failed')
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

@sio.event
async def message(sid, data):
    # Handle incoming message
    await sio.emit('response', {'echo': data}, to=sid)

@sio.event
async def join_room(sid, room):
    sio.enter_room(sid, room)
    await sio.emit('room_joined', {'room': room}, to=sid)

@sio.event
async def leave_room(sid, room):
    sio.leave_room(sid, room)
```

### Rooms and Broadcasting
```python
# Broadcast to room
await sio.emit('notification', data, room='general')

# Broadcast to all except sender
await sio.emit('message', data, room='chat', skip_sid=sid)

# Namespaces for logical separation
@sio.on('message', namespace='/chat')
async def chat_message(sid, data):
    await sio.emit('message', data, namespace='/chat', room='general')

@sio.on('message', namespace='/notifications')
async def notification(sid, data):
    await sio.emit('alert', data, namespace='/notifications')
```

### Redis Adapter for Scaling
```python
# Automatic cross-server broadcasting via Redis
mgr = socketio.AsyncRedisManager('redis://localhost:6379')
sio = socketio.AsyncServer(
    async_mode='asgi',
    client_manager=mgr  # Enables multi-server support
)
```

---

## Backpressure Handling

### Slow Client Detection
```python
import asyncio
from collections import deque

class BackpressureManager:
    def __init__(self, max_queue: int = 100):
        self.queues: Dict[str, deque] = {}
        self.max_queue = max_queue
    
    async def send_with_backpressure(
        self, 
        websocket: WebSocket, 
        user_id: str, 
        message: dict
    ):
        queue = self.queues.setdefault(user_id, deque(maxlen=self.max_queue))
        
        if len(queue) >= self.max_queue:
            # Client too slow - drop oldest messages or disconnect
            queue.popleft()
            # Optional: disconnect slow client
            # await websocket.close(code=4004)
        
        queue.append(message)
        
        try:
            await asyncio.wait_for(
                websocket.send_json(message),
                timeout=5.0
            )
            queue.popleft()  # Sent successfully
        except asyncio.TimeoutError:
            # Message queued but not sent
            pass
```

### Fan-Out with Semaphore
```python
async def broadcast_with_limit(connections: Dict[str, WebSocket], message: dict):
    """Limit concurrent sends to prevent overwhelming the event loop."""
    semaphore = asyncio.Semaphore(50)  # Max 50 concurrent sends
    
    async def send_one(ws: WebSocket):
        async with semaphore:
            try:
                await asyncio.wait_for(ws.send_json(message), timeout=5.0)
            except Exception:
                pass
    
    await asyncio.gather(*[send_one(ws) for ws in connections.values()])
```

---

## Production Patterns

### Graceful Shutdown
```python
import signal

shutdown_event = asyncio.Event()

def handle_shutdown(sig, frame):
    shutdown_event.set()

signal.signal(signal.SIGTERM, handle_shutdown)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while not shutdown_event.is_set():
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=1.0  # Check shutdown every second
                )
                # Process message
            except asyncio.TimeoutError:
                continue
    finally:
        await websocket.close(code=1001)  # Going away
```

### Sticky Sessions (Load Balancer)
```nginx
# Nginx upstream with sticky sessions
upstream websocket_backend {
    ip_hash;  # Sticky by client IP
    server backend1:8000;
    server backend2:8000;
}

server {
    location /ws {
        proxy_pass http://websocket_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;  # 24 hours
    }
}
```

### Health Check Endpoint
```python
@app.get("/health/ws")
async def websocket_health():
    return {
        "status": "healthy",
        "active_connections": len(manager.active_connections),
        "rooms": len(manager.rooms),
        "redis_connected": pubsub_manager.redis is not None
    }
```

---

## Decision Matrix

| Requirement | Solution |
|-------------|----------|
| Simple bidirectional | FastAPI WebSocket |
| Rooms + namespaces | python-socketio |
| Multi-server scaling | Redis pub/sub adapter |
| Auto-reconnect client | Socket.IO client |
| Raw performance | Starlette WebSocket |
| Browser + native | WebSocket (universal) |

---

## Production Checklist

- [ ] Authentication on connect (token or first-message)
- [ ] Heartbeat/ping-pong for dead connection detection
- [ ] Graceful disconnect handling
- [ ] Redis pub/sub for horizontal scaling
- [ ] Backpressure handling for slow clients
- [ ] Sticky sessions if not using Redis
- [ ] Connection limits per user
- [ ] Rate limiting on incoming messages
- [ ] Metrics: connection count, message rate, latency
- [ ] Graceful shutdown with client notification

---

## Anti-Patterns

| Anti-Pattern | Problem | Solution |
|--------------|---------|----------|
| In-memory only multi-server | Messages don't cross servers | Redis pub/sub adapter |
| No heartbeat | Ghost connections consume resources | Ping/pong every 30s |
| Blocking event loop | One slow client blocks all | asyncio.wait_for timeouts |
| Unlimited connections | Memory exhaustion | Per-user connection limits |
| No auth timeout | DoS via pending connections | 5-second auth deadline |
| Sync operations | Event loop blocked | Always use async I/O |

---

*Cycle 36 Complete - WebSockets & Real-Time Patterns*
