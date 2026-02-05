"""
Message Bus - Event Routing and Pub/Sub

V65 Modular Decomposition - Extracted from ultimate_orchestrator.py

This module provides a lightweight message bus for event-driven
communication between orchestration components.

Features:
- Async event publishing
- Priority-based event handling
- Event filtering and routing
- Dead letter queue for failed handlers
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Union,
)

logger = logging.getLogger(__name__)

# Type aliases
EventHandler = Callable[[Any], Coroutine[Any, Any, None]]
SyncEventHandler = Callable[[Any], None]
AnyEventHandler = Union[EventHandler, SyncEventHandler]


@dataclass
class MessageBusConfig:
    """Configuration for the MessageBus."""
    max_queue_size: int = 10000
    processing_timeout_ms: float = 5000.0
    enable_dead_letter: bool = True
    max_dead_letter_size: int = 1000
    backpressure_threshold: float = 0.8


@dataclass
class DeadLetterEntry:
    """Entry in the dead letter queue."""
    event: Any
    handler: str
    error: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retry_count: int = 0


class MessageBus:
    """
    Event-driven message bus for orchestration.

    Provides async event publishing with:
    - Priority-based handling
    - Backpressure management
    - Dead letter queue for failures
    - Event filtering
    """

    def __init__(self, config: Optional[MessageBusConfig] = None):
        self.config = config or MessageBusConfig()

        # Subscription management
        self._handlers: Dict[str, List[AnyEventHandler]] = {}
        self._async_handlers: Dict[str, List[EventHandler]] = {}
        self._sync_handlers: Dict[str, List[SyncEventHandler]] = {}

        # Event queue for async processing
        self._queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.max_queue_size
        )
        self._processing = False
        self._worker_task: Optional[asyncio.Task] = None

        # Dead letter queue
        self._dead_letter: List[DeadLetterEntry] = []

        # Metrics
        self._published_count = 0
        self._processed_count = 0
        self._failed_count = 0
        self._dropped_count = 0

    def subscribe(
        self,
        event_type: str,
        handler: AnyEventHandler
    ) -> None:
        """
        Subscribe to an event type.

        Handlers can be async or sync functions.
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
            self._async_handlers[event_type] = []
            self._sync_handlers[event_type] = []

        self._handlers[event_type].append(handler)

        # Classify handler
        if asyncio.iscoroutinefunction(handler):
            self._async_handlers[event_type].append(handler)  # type: ignore
        else:
            self._sync_handlers[event_type].append(handler)  # type: ignore

    def unsubscribe(
        self,
        event_type: str,
        handler: AnyEventHandler
    ) -> bool:
        """
        Unsubscribe from an event type.

        Returns True if handler was found and removed.
        """
        if event_type not in self._handlers:
            return False

        try:
            self._handlers[event_type].remove(handler)

            if asyncio.iscoroutinefunction(handler):
                self._async_handlers[event_type].remove(handler)  # type: ignore
            else:
                self._sync_handlers[event_type].remove(handler)  # type: ignore

            return True
        except ValueError:
            return False

    async def publish(
        self,
        event: Any,
        event_type: Optional[str] = None,
        priority: int = 3
    ) -> bool:
        """
        Publish an event to all subscribers.

        Args:
            event: The event to publish
            event_type: Optional event type (defaults to event.event_type or class name)
            priority: Priority level (1=highest, 5=lowest)

        Returns:
            True if event was queued, False if backpressure rejected it
        """
        # Determine event type
        if event_type is None:
            event_type = getattr(event, "event_type", type(event).__name__)

        # Check backpressure
        current_size = self._queue.qsize()
        threshold = int(self.config.max_queue_size * self.config.backpressure_threshold)
        if current_size >= threshold:
            self._dropped_count += 1
            logger.warning(f"Event dropped due to backpressure: {event_type}")
            return False

        # Queue for async processing
        try:
            self._queue.put_nowait((event, event_type, priority))
            self._published_count += 1
            return True
        except asyncio.QueueFull:
            self._dropped_count += 1
            return False

    def publish_sync(
        self,
        event: Any,
        event_type: Optional[str] = None
    ) -> int:
        """
        Publish an event synchronously (blocking).

        Calls all sync handlers immediately.
        Returns number of handlers called.
        """
        if event_type is None:
            event_type = getattr(event, "event_type", type(event).__name__)

        handlers = self._sync_handlers.get(event_type, [])
        called = 0

        for handler in handlers:
            try:
                handler(event)
                called += 1
            except Exception as e:
                self._handle_failure(event, handler.__name__, str(e))

        self._processed_count += called
        return called

    async def process_one(self) -> bool:
        """
        Process a single event from the queue.

        Returns True if an event was processed.
        """
        try:
            event, event_type, priority = await asyncio.wait_for(
                self._queue.get(),
                timeout=0.1
            )
        except asyncio.TimeoutError:
            return False

        await self._dispatch(event, event_type)
        return True

    async def process_all(self) -> int:
        """
        Process all events in the queue.

        Returns number of events processed.
        """
        processed = 0
        while not self._queue.empty():
            try:
                event, event_type, priority = self._queue.get_nowait()
                await self._dispatch(event, event_type)
                processed += 1
            except asyncio.QueueEmpty:
                break
        return processed

    async def _dispatch(self, event: Any, event_type: str) -> None:
        """Dispatch event to all handlers."""
        handlers = self._handlers.get(event_type, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await asyncio.wait_for(
                        handler(event),
                        timeout=self.config.processing_timeout_ms / 1000
                    )
                else:
                    handler(event)
                self._processed_count += 1
            except asyncio.TimeoutError:
                self._handle_failure(
                    event,
                    handler.__name__,
                    f"Handler timed out after {self.config.processing_timeout_ms}ms"
                )
            except Exception as e:
                self._handle_failure(event, handler.__name__, str(e))

    def _handle_failure(
        self,
        event: Any,
        handler_name: str,
        error: str
    ) -> None:
        """Handle a failed event handler."""
        self._failed_count += 1

        if not self.config.enable_dead_letter:
            return

        entry = DeadLetterEntry(
            event=event,
            handler=handler_name,
            error=error
        )

        self._dead_letter.append(entry)

        # Trim if exceeds max size
        if len(self._dead_letter) > self.config.max_dead_letter_size:
            self._dead_letter = self._dead_letter[-self.config.max_dead_letter_size:]

        logger.warning(f"Event handler failed: {handler_name} - {error}")

    async def start_worker(self) -> None:
        """Start background worker for processing events."""
        if self._processing:
            return

        self._processing = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.debug("Message bus worker started")

    async def stop_worker(self) -> None:
        """Stop the background worker."""
        self._processing = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        logger.debug("Message bus worker stopped")

    async def _worker_loop(self) -> None:
        """Background worker loop."""
        while self._processing:
            try:
                await self.process_one()
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(0.1)

    def get_dead_letter(
        self,
        limit: int = 10
    ) -> List[DeadLetterEntry]:
        """Get recent dead letter entries."""
        return self._dead_letter[-limit:]

    def clear_dead_letter(self) -> int:
        """Clear dead letter queue. Returns count cleared."""
        count = len(self._dead_letter)
        self._dead_letter.clear()
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        return {
            "queue_size": self._queue.qsize(),
            "max_queue_size": self.config.max_queue_size,
            "published_count": self._published_count,
            "processed_count": self._processed_count,
            "failed_count": self._failed_count,
            "dropped_count": self._dropped_count,
            "dead_letter_size": len(self._dead_letter),
            "subscribed_types": list(self._handlers.keys()),
            "handler_count": sum(len(h) for h in self._handlers.values()),
            "processing": self._processing
        }


# Singleton instance
_default_bus: Optional[MessageBus] = None


def get_message_bus() -> MessageBus:
    """Get or create the default message bus."""
    global _default_bus
    if _default_bus is None:
        _default_bus = MessageBus()
    return _default_bus


async def publish(event: Any, event_type: Optional[str] = None) -> bool:
    """Publish to the default message bus."""
    bus = get_message_bus()
    return await bus.publish(event, event_type)


def subscribe(event_type: str, handler: AnyEventHandler) -> None:
    """Subscribe to the default message bus."""
    bus = get_message_bus()
    bus.subscribe(event_type, handler)
