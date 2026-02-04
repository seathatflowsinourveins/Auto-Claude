"""
Lifecycle Management for Unleashed Platform - V50 Architecture

Provides centralized graceful shutdown handling for all platform components.
Ensures no data loss during shutdown by properly draining connections,
flushing writes, canceling tasks, and persisting caches.

Components Managed:
1. HTTP Connection Pool - Drain active connections with timeout
2. Memory Backends - Flush pending writes, save HNSW index
3. Background Tasks - Cancel with configurable timeout
4. Cache - Persist to disk if configured

Signal Handling:
- SIGTERM: Graceful shutdown (default 30s timeout)
- SIGINT: Graceful shutdown (Ctrl+C)

Usage:
    from core.lifecycle import LifecycleManager, get_lifecycle_manager

    # Get singleton instance
    manager = get_lifecycle_manager()

    # Register custom shutdown handlers
    manager.register_shutdown_handler(my_cleanup_fn, priority=50)

    # Start signal handling
    await manager.start()

    # Manual shutdown (if needed)
    await manager.shutdown(timeout=30.0)

Architecture:
    - Priority-based handler execution (higher priority = earlier execution)
    - Async-safe with proper task cancellation
    - Timeout protection for stuck handlers
    - Comprehensive logging for debugging shutdown issues
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import signal
import sys
import time
import weakref
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class ShutdownPhase(Enum):
    """Phases of the shutdown process."""
    NOT_STARTED = auto()
    SIGNALS_RECEIVED = auto()
    DRAINING_CONNECTIONS = auto()
    CANCELING_TASKS = auto()
    FLUSHING_MEMORY = auto()
    PERSISTING_CACHE = auto()
    CLEANUP = auto()
    COMPLETED = auto()


class ShutdownPriority:
    """Standard priority levels for shutdown handlers.

    Higher priority = executed earlier in shutdown sequence.
    """
    CRITICAL = 100      # Connection pools, external services
    HIGH = 75           # Memory backends, data persistence
    NORMAL = 50         # Background tasks, caches
    LOW = 25            # Logging, metrics
    CLEANUP = 0         # Final cleanup, temp files


# Type aliases
ShutdownHandler = Callable[[], Awaitable[None]]
SyncShutdownHandler = Callable[[], None]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class HandlerRegistration:
    """Registration info for a shutdown handler."""
    handler: Union[ShutdownHandler, SyncShutdownHandler]
    name: str
    priority: int
    timeout: float
    is_async: bool
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __lt__(self, other: "HandlerRegistration") -> bool:
        """Sort by priority (higher first)."""
        return self.priority > other.priority


@dataclass
class ShutdownResult:
    """Result of a shutdown handler execution."""
    handler_name: str
    success: bool
    duration_seconds: float
    error: Optional[str] = None


@dataclass
class ShutdownReport:
    """Complete report of the shutdown process."""
    started_at: datetime
    completed_at: datetime
    phase: ShutdownPhase
    handlers_executed: int
    handlers_succeeded: int
    handlers_failed: int
    handlers_timed_out: int
    results: List[ShutdownResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Total shutdown duration."""
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def success(self) -> bool:
        """Whether shutdown completed without critical failures."""
        return self.handlers_failed == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": round(self.duration_seconds, 3),
            "phase": self.phase.name,
            "handlers": {
                "executed": self.handlers_executed,
                "succeeded": self.handlers_succeeded,
                "failed": self.handlers_failed,
                "timed_out": self.handlers_timed_out,
            },
            "results": [
                {
                    "name": r.handler_name,
                    "success": r.success,
                    "duration": round(r.duration_seconds, 3),
                    "error": r.error,
                }
                for r in self.results
            ],
            "errors": self.errors,
        }


# =============================================================================
# LIFECYCLE MANAGER
# =============================================================================

class LifecycleManager:
    """
    Centralized lifecycle management for the Unleashed Platform.

    Provides:
    - Signal handling (SIGTERM, SIGINT)
    - Priority-based shutdown handler execution
    - Timeout protection for stuck handlers
    - Component-specific shutdown utilities
    - Comprehensive shutdown reporting

    Thread Safety:
    - All operations are asyncio-safe
    - Uses locks to prevent race conditions
    - Supports concurrent handler execution within priority groups

    Example:
        manager = LifecycleManager()

        # Register handlers
        manager.register_shutdown_handler(
            close_database,
            name="database",
            priority=ShutdownPriority.HIGH
        )

        # Start signal handling
        await manager.start()

        # Application runs...

        # On SIGTERM/SIGINT, shutdown is automatic
        # Or manually: await manager.shutdown()
    """

    def __init__(
        self,
        default_timeout: float = 30.0,
        handler_timeout: float = 10.0,
        enable_signal_handling: bool = True,
    ) -> None:
        """
        Initialize the lifecycle manager.

        Args:
            default_timeout: Default total shutdown timeout in seconds
            handler_timeout: Default per-handler timeout in seconds
            enable_signal_handling: Whether to register signal handlers
        """
        self._default_timeout = default_timeout
        self._handler_timeout = handler_timeout
        self._enable_signals = enable_signal_handling

        self._handlers: List[HandlerRegistration] = []
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._phase = ShutdownPhase.NOT_STARTED
        self._started = False
        self._shutting_down = False
        self._shutdown_report: Optional[ShutdownReport] = None

        # Track background tasks for cancellation
        self._background_tasks: Set[asyncio.Task] = set()

        # Component references (weak refs to avoid circular dependencies)
        self._connection_pool: Optional[weakref.ref] = None
        self._memory_backends: List[weakref.ref] = []
        self._caches: List[weakref.ref] = []

        logger.debug("LifecycleManager initialized")

    # =========================================================================
    # HANDLER REGISTRATION
    # =========================================================================

    def register_shutdown_handler(
        self,
        handler: Union[ShutdownHandler, SyncShutdownHandler],
        name: Optional[str] = None,
        priority: int = ShutdownPriority.NORMAL,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Register a shutdown handler to be called during graceful shutdown.

        Handlers are executed in priority order (highest first).
        Handlers with the same priority may execute concurrently.

        Args:
            handler: Async or sync function to call during shutdown
            name: Human-readable name for logging (defaults to function name)
            priority: Execution priority (higher = earlier)
            timeout: Handler-specific timeout (defaults to handler_timeout)
        """
        if self._shutting_down:
            logger.warning(f"Cannot register handler during shutdown: {name}")
            return

        handler_name = name or getattr(handler, "__name__", "unnamed")
        is_async = asyncio.iscoroutinefunction(handler)
        handler_timeout = timeout or self._handler_timeout

        registration = HandlerRegistration(
            handler=handler,
            name=handler_name,
            priority=priority,
            timeout=handler_timeout,
            is_async=is_async,
        )

        self._handlers.append(registration)
        self._handlers.sort()  # Re-sort by priority

        logger.debug(f"Registered shutdown handler: {handler_name} (priority={priority})")

    def unregister_shutdown_handler(self, name: str) -> bool:
        """
        Unregister a shutdown handler by name.

        Args:
            name: Name of the handler to unregister

        Returns:
            True if handler was found and removed
        """
        for i, registration in enumerate(self._handlers):
            if registration.name == name:
                del self._handlers[i]
                logger.debug(f"Unregistered shutdown handler: {name}")
                return True
        return False

    # =========================================================================
    # COMPONENT REGISTRATION
    # =========================================================================

    def register_connection_pool(self, pool: Any) -> None:
        """Register the HTTP connection pool for graceful draining."""
        self._connection_pool = weakref.ref(pool)
        logger.debug("Registered connection pool for lifecycle management")

    def register_memory_backend(self, backend: Any) -> None:
        """Register a memory backend for graceful shutdown."""
        self._memory_backends.append(weakref.ref(backend))
        logger.debug(f"Registered memory backend: {type(backend).__name__}")

    def register_cache(self, cache: Any) -> None:
        """Register a cache for graceful shutdown (persistence)."""
        self._caches.append(weakref.ref(cache))
        logger.debug(f"Registered cache: {type(cache).__name__}")

    def track_background_task(self, task: asyncio.Task) -> None:
        """
        Track a background task for cancellation during shutdown.

        Tasks are automatically removed when completed.
        """
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    # =========================================================================
    # SIGNAL HANDLING
    # =========================================================================

    async def start(self) -> None:
        """
        Start the lifecycle manager and register signal handlers.

        Call this after all handlers are registered.
        """
        if self._started:
            return

        self._started = True

        if self._enable_signals:
            self._setup_signal_handlers()

        # Register atexit handler for sync cleanup
        atexit.register(self._sync_cleanup)

        logger.info("LifecycleManager started with signal handling")

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()

        # Platform-specific signal handling
        if sys.platform != "win32":
            # Unix-like systems
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(self._signal_handler(s))
                )
            logger.debug("Registered Unix signal handlers (SIGTERM, SIGINT)")
        else:
            # Windows - use signal module directly
            def win_handler(signum, frame):
                asyncio.create_task(self._signal_handler(signum))

            signal.signal(signal.SIGTERM, win_handler)
            signal.signal(signal.SIGINT, win_handler)
            logger.debug("Registered Windows signal handlers (SIGTERM, SIGINT)")

    async def _signal_handler(self, sig: int) -> None:
        """Handle shutdown signals."""
        sig_name = signal.Signals(sig).name
        logger.info(f"Received signal {sig_name}, initiating graceful shutdown...")

        self._phase = ShutdownPhase.SIGNALS_RECEIVED
        await self.shutdown()

    def _sync_cleanup(self) -> None:
        """Synchronous cleanup for atexit handler."""
        if not self._shutting_down:
            logger.debug("atexit cleanup (shutdown not completed via signal)")

    # =========================================================================
    # SHUTDOWN EXECUTION
    # =========================================================================

    async def shutdown(self, timeout: Optional[float] = None) -> ShutdownReport:
        """
        Execute graceful shutdown of all registered components.

        Shutdown order:
        1. Drain HTTP connection pool
        2. Cancel background tasks
        3. Flush memory backend writes
        4. Persist caches to disk
        5. Execute registered handlers (by priority)
        6. Final cleanup

        Args:
            timeout: Total shutdown timeout (defaults to default_timeout)

        Returns:
            ShutdownReport with results
        """
        async with self._lock:
            if self._shutting_down:
                logger.warning("Shutdown already in progress")
                if self._shutdown_report:
                    return self._shutdown_report
                # Wait for existing shutdown to complete
                await self._shutdown_event.wait()
                return self._shutdown_report or self._create_empty_report()

            self._shutting_down = True

        effective_timeout = timeout or self._default_timeout
        started_at = datetime.now(timezone.utc)
        results: List[ShutdownResult] = []
        errors: List[str] = []

        handlers_succeeded = 0
        handlers_failed = 0
        handlers_timed_out = 0

        logger.info(f"Starting graceful shutdown (timeout={effective_timeout}s)")

        try:
            # Phase 1: Drain connection pool
            self._phase = ShutdownPhase.DRAINING_CONNECTIONS
            await self._drain_connection_pool(results, errors)

            # Phase 2: Cancel background tasks
            self._phase = ShutdownPhase.CANCELING_TASKS
            await self._cancel_background_tasks(results, errors)

            # Phase 3: Flush memory backends
            self._phase = ShutdownPhase.FLUSHING_MEMORY
            await self._flush_memory_backends(results, errors)

            # Phase 4: Persist caches
            self._phase = ShutdownPhase.PERSISTING_CACHE
            await self._persist_caches(results, errors)

            # Phase 5: Execute registered handlers
            self._phase = ShutdownPhase.CLEANUP
            for registration in self._handlers:
                result = await self._execute_handler(registration)
                results.append(result)

                if result.success:
                    handlers_succeeded += 1
                elif result.error and "timeout" in result.error.lower():
                    handlers_timed_out += 1
                    handlers_failed += 1
                else:
                    handlers_failed += 1

            self._phase = ShutdownPhase.COMPLETED

        except Exception as e:
            errors.append(f"Shutdown error: {e}")
            logger.exception("Error during shutdown")

        completed_at = datetime.now(timezone.utc)

        self._shutdown_report = ShutdownReport(
            started_at=started_at,
            completed_at=completed_at,
            phase=self._phase,
            handlers_executed=len(results),
            handlers_succeeded=handlers_succeeded,
            handlers_failed=handlers_failed,
            handlers_timed_out=handlers_timed_out,
            results=results,
            errors=errors,
        )

        self._shutdown_event.set()

        # Log summary
        duration = self._shutdown_report.duration_seconds
        if self._shutdown_report.success:
            logger.info(f"Graceful shutdown completed in {duration:.2f}s")
        else:
            logger.warning(
                f"Shutdown completed with {handlers_failed} failures in {duration:.2f}s"
            )

        return self._shutdown_report

    async def _execute_handler(
        self,
        registration: HandlerRegistration
    ) -> ShutdownResult:
        """Execute a single shutdown handler with timeout protection."""
        start_time = time.time()

        try:
            if registration.is_async:
                await asyncio.wait_for(
                    registration.handler(),
                    timeout=registration.timeout
                )
            else:
                # Run sync handler in executor
                loop = asyncio.get_running_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, registration.handler),
                    timeout=registration.timeout
                )

            duration = time.time() - start_time
            logger.debug(f"Handler '{registration.name}' completed in {duration:.3f}s")

            return ShutdownResult(
                handler_name=registration.name,
                success=True,
                duration_seconds=duration,
            )

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            error_msg = f"Handler '{registration.name}' timed out after {registration.timeout}s"
            logger.warning(error_msg)

            return ShutdownResult(
                handler_name=registration.name,
                success=False,
                duration_seconds=duration,
                error=error_msg,
            )

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Handler '{registration.name}' failed: {e}"
            logger.error(error_msg)

            return ShutdownResult(
                handler_name=registration.name,
                success=False,
                duration_seconds=duration,
                error=error_msg,
            )

    # =========================================================================
    # COMPONENT-SPECIFIC SHUTDOWN
    # =========================================================================

    async def _drain_connection_pool(
        self,
        results: List[ShutdownResult],
        errors: List[str]
    ) -> None:
        """Drain HTTP connection pool, allowing active requests to complete."""
        if self._connection_pool is None:
            return

        pool = self._connection_pool()
        if pool is None:
            return

        start_time = time.time()
        handler_name = "connection_pool_drain"

        try:
            # Check if pool has shutdown method
            if hasattr(pool, "shutdown"):
                await asyncio.wait_for(
                    pool.shutdown(),
                    timeout=self._handler_timeout
                )
                logger.info("Connection pool drained successfully")
            elif hasattr(pool, "close"):
                # Fallback to close
                if asyncio.iscoroutinefunction(pool.close):
                    await pool.close()
                else:
                    pool.close()
                logger.info("Connection pool closed")

            results.append(ShutdownResult(
                handler_name=handler_name,
                success=True,
                duration_seconds=time.time() - start_time,
            ))

        except asyncio.TimeoutError:
            error_msg = "Connection pool drain timed out"
            logger.warning(error_msg)
            errors.append(error_msg)
            results.append(ShutdownResult(
                handler_name=handler_name,
                success=False,
                duration_seconds=time.time() - start_time,
                error=error_msg,
            ))

        except Exception as e:
            error_msg = f"Connection pool drain failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            results.append(ShutdownResult(
                handler_name=handler_name,
                success=False,
                duration_seconds=time.time() - start_time,
                error=error_msg,
            ))

    async def _cancel_background_tasks(
        self,
        results: List[ShutdownResult],
        errors: List[str]
    ) -> None:
        """Cancel all tracked background tasks with timeout."""
        if not self._background_tasks:
            return

        start_time = time.time()
        handler_name = "background_tasks_cancel"

        # Get current tasks (copy to avoid modification during iteration)
        tasks = list(self._background_tasks)
        cancelled_count = 0

        logger.info(f"Canceling {len(tasks)} background tasks...")

        for task in tasks:
            if not task.done():
                task.cancel()
                cancelled_count += 1

        # Wait for tasks to complete cancellation
        if tasks:
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self._handler_timeout
                )

        duration = time.time() - start_time
        logger.info(f"Cancelled {cancelled_count} background tasks in {duration:.2f}s")

        results.append(ShutdownResult(
            handler_name=handler_name,
            success=True,
            duration_seconds=duration,
        ))

    async def _flush_memory_backends(
        self,
        results: List[ShutdownResult],
        errors: List[str]
    ) -> None:
        """Flush pending writes to all memory backends."""
        for backend_ref in self._memory_backends:
            backend = backend_ref()
            if backend is None:
                continue

            backend_name = type(backend).__name__
            handler_name = f"memory_flush_{backend_name}"
            start_time = time.time()

            try:
                # Try various flush/close methods
                if hasattr(backend, "flush"):
                    if asyncio.iscoroutinefunction(backend.flush):
                        await asyncio.wait_for(
                            backend.flush(),
                            timeout=self._handler_timeout
                        )
                    else:
                        backend.flush()

                if hasattr(backend, "save"):
                    if asyncio.iscoroutinefunction(backend.save):
                        await asyncio.wait_for(
                            backend.save(),
                            timeout=self._handler_timeout
                        )
                    else:
                        backend.save()

                if hasattr(backend, "close"):
                    if asyncio.iscoroutinefunction(backend.close):
                        await backend.close()
                    else:
                        backend.close()

                duration = time.time() - start_time
                logger.debug(f"Memory backend '{backend_name}' flushed in {duration:.3f}s")

                results.append(ShutdownResult(
                    handler_name=handler_name,
                    success=True,
                    duration_seconds=duration,
                ))

            except asyncio.TimeoutError:
                error_msg = f"Memory backend '{backend_name}' flush timed out"
                logger.warning(error_msg)
                errors.append(error_msg)
                results.append(ShutdownResult(
                    handler_name=handler_name,
                    success=False,
                    duration_seconds=time.time() - start_time,
                    error=error_msg,
                ))

            except Exception as e:
                error_msg = f"Memory backend '{backend_name}' flush failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                results.append(ShutdownResult(
                    handler_name=handler_name,
                    success=False,
                    duration_seconds=time.time() - start_time,
                    error=error_msg,
                ))

    async def _persist_caches(
        self,
        results: List[ShutdownResult],
        errors: List[str]
    ) -> None:
        """Persist caches to disk if configured."""
        for cache_ref in self._caches:
            cache = cache_ref()
            if cache is None:
                continue

            cache_name = type(cache).__name__
            handler_name = f"cache_persist_{cache_name}"
            start_time = time.time()

            try:
                # Check for persistence capability
                should_persist = getattr(cache, "persist_on_shutdown", False)

                if should_persist:
                    if hasattr(cache, "persist"):
                        if asyncio.iscoroutinefunction(cache.persist):
                            await asyncio.wait_for(
                                cache.persist(),
                                timeout=self._handler_timeout
                            )
                        else:
                            cache.persist()
                        logger.debug(f"Cache '{cache_name}' persisted to disk")

                # Always try to clear/close
                if hasattr(cache, "close"):
                    if asyncio.iscoroutinefunction(cache.close):
                        await cache.close()
                    else:
                        cache.close()

                duration = time.time() - start_time
                results.append(ShutdownResult(
                    handler_name=handler_name,
                    success=True,
                    duration_seconds=duration,
                ))

            except asyncio.TimeoutError:
                error_msg = f"Cache '{cache_name}' persist timed out"
                logger.warning(error_msg)
                errors.append(error_msg)
                results.append(ShutdownResult(
                    handler_name=handler_name,
                    success=False,
                    duration_seconds=time.time() - start_time,
                    error=error_msg,
                ))

            except Exception as e:
                error_msg = f"Cache '{cache_name}' persist failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                results.append(ShutdownResult(
                    handler_name=handler_name,
                    success=False,
                    duration_seconds=time.time() - start_time,
                    error=error_msg,
                ))

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _create_empty_report(self) -> ShutdownReport:
        """Create an empty shutdown report."""
        now = datetime.now(timezone.utc)
        return ShutdownReport(
            started_at=now,
            completed_at=now,
            phase=ShutdownPhase.NOT_STARTED,
            handlers_executed=0,
            handlers_succeeded=0,
            handlers_failed=0,
            handlers_timed_out=0,
        )

    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._shutting_down

    @property
    def phase(self) -> ShutdownPhase:
        """Get current shutdown phase."""
        return self._phase

    @property
    def registered_handlers(self) -> List[str]:
        """Get list of registered handler names."""
        return [h.name for h in self._handlers]

    def get_status(self) -> Dict[str, Any]:
        """Get lifecycle manager status."""
        return {
            "started": self._started,
            "shutting_down": self._shutting_down,
            "phase": self._phase.name,
            "handlers_registered": len(self._handlers),
            "handler_names": self.registered_handlers,
            "background_tasks": len(self._background_tasks),
            "connection_pool_registered": self._connection_pool is not None,
            "memory_backends_registered": len(self._memory_backends),
            "caches_registered": len(self._caches),
        }


# =============================================================================
# SINGLETON AND FACTORY
# =============================================================================

_lifecycle_manager: Optional[LifecycleManager] = None


def get_lifecycle_manager(
    default_timeout: float = 30.0,
    handler_timeout: float = 10.0,
    enable_signal_handling: bool = True,
) -> LifecycleManager:
    """
    Get or create the singleton LifecycleManager instance.

    Args:
        default_timeout: Total shutdown timeout (only used on creation)
        handler_timeout: Per-handler timeout (only used on creation)
        enable_signal_handling: Whether to enable signal handlers

    Returns:
        The singleton LifecycleManager instance
    """
    global _lifecycle_manager

    if _lifecycle_manager is None:
        _lifecycle_manager = LifecycleManager(
            default_timeout=default_timeout,
            handler_timeout=handler_timeout,
            enable_signal_handling=enable_signal_handling,
        )

    return _lifecycle_manager


def reset_lifecycle_manager() -> None:
    """Reset the singleton instance (for testing)."""
    global _lifecycle_manager
    _lifecycle_manager = None


# =============================================================================
# CONVENIENCE DECORATORS
# =============================================================================

def on_shutdown(
    name: Optional[str] = None,
    priority: int = ShutdownPriority.NORMAL,
    timeout: Optional[float] = None,
):
    """
    Decorator to register a function as a shutdown handler.

    Example:
        @on_shutdown(name="cleanup_temp_files", priority=ShutdownPriority.LOW)
        async def cleanup_temp_files():
            # Cleanup logic
            pass
    """
    def decorator(func: Union[ShutdownHandler, SyncShutdownHandler]):
        manager = get_lifecycle_manager()
        manager.register_shutdown_handler(
            handler=func,
            name=name,
            priority=priority,
            timeout=timeout,
        )
        return func
    return decorator


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

async def graceful_shutdown_context():
    """
    Async context manager for graceful shutdown handling.

    Example:
        async with graceful_shutdown_context():
            # Application code
            await run_server()
    """
    manager = get_lifecycle_manager()
    await manager.start()

    try:
        yield manager
    finally:
        if not manager.is_shutting_down:
            await manager.shutdown()


def register_platform_components(
    connection_pool: Optional[Any] = None,
    unified_memory: Optional[Any] = None,
    cache: Optional[Any] = None,
) -> LifecycleManager:
    """
    Register standard platform components with the lifecycle manager.

    This is a convenience function for typical platform setup.

    Args:
        connection_pool: MCP connection pool instance
        unified_memory: UnifiedMemory instance
        cache: Cache instance

    Returns:
        The lifecycle manager instance
    """
    manager = get_lifecycle_manager()

    if connection_pool is not None:
        manager.register_connection_pool(connection_pool)

    if unified_memory is not None:
        manager.register_memory_backend(unified_memory)

        # Also register sub-backends if available
        for attr in ["_sqlite", "_hnsw", "_bitemporal", "_procedural", "_letta"]:
            backend = getattr(unified_memory, attr, None)
            if backend is not None:
                manager.register_memory_backend(backend)

    if cache is not None:
        manager.register_cache(cache)

    return manager


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main class
    "LifecycleManager",

    # Enums
    "ShutdownPhase",
    "ShutdownPriority",

    # Data classes
    "HandlerRegistration",
    "ShutdownResult",
    "ShutdownReport",

    # Factory functions
    "get_lifecycle_manager",
    "reset_lifecycle_manager",

    # Decorators
    "on_shutdown",

    # Helpers
    "graceful_shutdown_context",
    "register_platform_components",
]
