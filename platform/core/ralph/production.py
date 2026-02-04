"""
Ralph Loop Production Enhancements - Unleashed Platform V11+

This module provides production-ready enhancements for the Ralph Loop:
- Structured logging with correlation IDs
- Prometheus metrics integration
- Graceful shutdown handling
- Rate limiting for API calls
- State checkpointing for crash recovery
- Enhanced V11 feature implementations

Based on ADR-026 guidelines and production best practices.
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import json
import logging
import os
import signal
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

# ============================================================================
# Correlation ID Context Management
# ============================================================================

_correlation_context = threading.local()


def get_correlation_id() -> str:
    """Get the current correlation ID or generate a new one."""
    if not hasattr(_correlation_context, "correlation_id"):
        _correlation_context.correlation_id = str(uuid.uuid4())[:12]
    return _correlation_context.correlation_id


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current context."""
    _correlation_context.correlation_id = correlation_id


@contextmanager
def correlation_context(correlation_id: Optional[str] = None):
    """Context manager for correlation ID scope."""
    old_id = getattr(_correlation_context, "correlation_id", None)
    _correlation_context.correlation_id = correlation_id or str(uuid.uuid4())[:12]
    try:
        yield _correlation_context.correlation_id
    finally:
        if old_id:
            _correlation_context.correlation_id = old_id
        else:
            delattr(_correlation_context, "correlation_id")


# ============================================================================
# Structured Logging with Correlation IDs
# ============================================================================

class StructuredLogFormatter(logging.Formatter):
    """JSON formatter for structured logging with correlation IDs."""

    def __init__(self, service_name: str = "ralph_loop"):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self.service_name,
            "correlation_id": get_correlation_id(),
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add loop context if available
        if hasattr(record, "loop_id"):
            log_entry["loop_id"] = record.loop_id
        if hasattr(record, "iteration"):
            log_entry["iteration"] = record.iteration
        if hasattr(record, "strategy"):
            log_entry["strategy"] = record.strategy

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            log_entry["exception_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry)


class RalphLogger:
    """Enhanced logger with correlation ID and context support."""

    def __init__(self, name: str = "ralph_loop", loop_id: Optional[str] = None):
        self._logger = logging.getLogger(name)
        self.loop_id = loop_id
        self._current_iteration: Optional[int] = None
        self._current_strategy: Optional[str] = None

    def set_context(
        self,
        loop_id: Optional[str] = None,
        iteration: Optional[int] = None,
        strategy: Optional[str] = None
    ) -> None:
        """Set logging context."""
        if loop_id:
            self.loop_id = loop_id
        self._current_iteration = iteration
        self._current_strategy = strategy

    def _log(self, level: int, msg: str, **extra: Any) -> None:
        """Internal log method with context injection."""
        record = self._logger.makeRecord(
            self._logger.name, level, "", 0, msg, (), None
        )
        record.loop_id = self.loop_id
        record.iteration = self._current_iteration
        record.strategy = self._current_strategy
        record.extra_fields = extra
        self._logger.handle(record)

    def debug(self, msg: str, **extra: Any) -> None:
        self._log(logging.DEBUG, msg, **extra)

    def info(self, msg: str, **extra: Any) -> None:
        self._log(logging.INFO, msg, **extra)

    def warning(self, msg: str, **extra: Any) -> None:
        self._log(logging.WARNING, msg, **extra)

    def error(self, msg: str, **extra: Any) -> None:
        self._log(logging.ERROR, msg, **extra)

    def critical(self, msg: str, **extra: Any) -> None:
        self._log(logging.CRITICAL, msg, **extra)


def configure_production_logging(
    log_level: str = "INFO",
    service_name: str = "ralph_loop",
    log_file: Optional[Path] = None
) -> RalphLogger:
    """Configure production logging with structured JSON output."""
    logger = logging.getLogger("ralph_loop")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler with structured JSON
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(StructuredLogFormatter(service_name))
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredLogFormatter(service_name))
        logger.addHandler(file_handler)

    return RalphLogger("ralph_loop")


# ============================================================================
# Prometheus Metrics Integration
# ============================================================================

@dataclass
class MetricValue:
    """A metric value with labels."""
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class RalphMetrics:
    """Prometheus-compatible metrics for Ralph Loop."""

    # Default histogram buckets for latency
    LATENCY_BUCKETS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]

    def __init__(self, namespace: str = "ralph_loop"):
        self.namespace = namespace
        self._counters: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._histograms: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self._lock = threading.Lock()

        # Pre-register core metrics
        self._metric_definitions: Dict[str, MetricType] = {
            "iterations_total": MetricType.COUNTER,
            "improvements_total": MetricType.COUNTER,
            "failures_total": MetricType.COUNTER,
            "api_calls_total": MetricType.COUNTER,
            "rate_limit_hits_total": MetricType.COUNTER,
            "checkpoints_saved_total": MetricType.COUNTER,
            "hypothesis_generated_total": MetricType.COUNTER,
            "hypothesis_verified_total": MetricType.COUNTER,
            "hypothesis_rejected_total": MetricType.COUNTER,
            "rag_retrievals_total": MetricType.COUNTER,
            "reward_hacking_detected_total": MetricType.COUNTER,
            "current_fitness": MetricType.GAUGE,
            "current_iteration": MetricType.GAUGE,
            "active_loops": MetricType.GAUGE,
            "buffer_size": MetricType.GAUGE,
            "speculation_acceptance_rate": MetricType.GAUGE,
            "token_compression_ratio": MetricType.GAUGE,
            "iteration_latency_seconds": MetricType.HISTOGRAM,
            "api_call_latency_seconds": MetricType.HISTOGRAM,
            "verification_latency_seconds": MetricType.HISTOGRAM,
            "checkpoint_size_bytes": MetricType.HISTOGRAM,
        }

    def _labels_to_key(self, labels: Dict[str, str]) -> str:
        """Convert labels dict to a string key."""
        return json.dumps(sorted(labels.items())) if labels else ""

    def inc_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        full_name = f"{self.namespace}_{name}"
        key = self._labels_to_key(labels or {})
        with self._lock:
            self._counters[full_name][key] += value

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value."""
        full_name = f"{self.namespace}_{name}"
        key = self._labels_to_key(labels or {})
        with self._lock:
            self._gauges[full_name][key] = value

    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram observation."""
        full_name = f"{self.namespace}_{name}"
        key = self._labels_to_key(labels or {})
        with self._lock:
            self._histograms[full_name][key].append(value)
            # Keep only last 10000 observations to bound memory
            if len(self._histograms[full_name][key]) > 10000:
                self._histograms[full_name][key] = self._histograms[full_name][key][-10000:]

    @contextmanager
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.observe_histogram(name, duration, labels)

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        with self._lock:
            # Export counters
            for name, values in self._counters.items():
                lines.append(f"# TYPE {name} counter")
                for key, value in values.items():
                    labels_str = self._format_labels(key)
                    lines.append(f"{name}{labels_str} {value}")

            # Export gauges
            for name, values in self._gauges.items():
                lines.append(f"# TYPE {name} gauge")
                for key, value in values.items():
                    labels_str = self._format_labels(key)
                    lines.append(f"{name}{labels_str} {value}")

            # Export histograms
            for name, values in self._histograms.items():
                lines.append(f"# TYPE {name} histogram")
                for key, observations in values.items():
                    labels_str = self._format_labels(key)
                    if observations:
                        # Calculate buckets
                        sorted_obs = sorted(observations)
                        count = len(sorted_obs)
                        total = sum(sorted_obs)

                        for bucket in self.LATENCY_BUCKETS:
                            bucket_count = sum(1 for o in sorted_obs if o <= bucket)
                            lines.append(f'{name}_bucket{{le="{bucket}"{labels_str[1:-1] if labels_str else ""}}} {bucket_count}')
                        lines.append(f'{name}_bucket{{le="+Inf"{labels_str[1:-1] if labels_str else ""}}} {count}')
                        lines.append(f'{name}_sum{labels_str} {total}')
                        lines.append(f'{name}_count{labels_str} {count}')

        return "\n".join(lines)

    def _format_labels(self, key: str) -> str:
        """Format labels string from key."""
        if not key:
            return ""
        labels = dict(json.loads(key))
        label_pairs = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ",".join(label_pairs) + "}"

    def get_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of all metrics."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    name: {
                        key: {
                            "count": len(values),
                            "sum": sum(values) if values else 0,
                            "min": min(values) if values else 0,
                            "max": max(values) if values else 0,
                            "avg": sum(values) / len(values) if values else 0,
                        }
                        for key, values in obs.items()
                    }
                    for name, obs in self._histograms.items()
                },
            }


# Global metrics instance
_metrics: Optional[RalphMetrics] = None


def get_metrics(namespace: str = "ralph_loop") -> RalphMetrics:
    """Get or create the global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = RalphMetrics(namespace)
    return _metrics


# ============================================================================
# Rate Limiting for API Calls
# ============================================================================

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: float = 10.0
    burst_size: int = 20
    retry_after_seconds: float = 1.0
    max_retries: int = 3


class TokenBucket:
    """Token bucket rate limiter implementation."""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity  # max burst size
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> Tuple[bool, float]:
        """
        Try to acquire tokens.
        Returns (success, wait_time) where wait_time is how long to wait if not successful.
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now

            # Add tokens based on elapsed time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True, 0.0
            else:
                # Calculate wait time
                wait_time = (tokens - self.tokens) / self.rate
                return False, wait_time


class RateLimiter:
    """Rate limiter with multiple buckets for different operation types."""

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()
        self._metrics = get_metrics()

    def _get_bucket(self, key: str) -> TokenBucket:
        """Get or create a token bucket for the given key."""
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = TokenBucket(
                    self.config.requests_per_second,
                    self.config.burst_size
                )
            return self._buckets[key]

    async def acquire(self, key: str = "default", tokens: int = 1) -> bool:
        """Acquire tokens with async waiting."""
        bucket = self._get_bucket(key)

        for attempt in range(self.config.max_retries):
            success, wait_time = bucket.acquire(tokens)
            if success:
                return True

            self._metrics.inc_counter("rate_limit_hits_total", labels={"key": key})
            await asyncio.sleep(wait_time)

        return False

    def acquire_sync(self, key: str = "default", tokens: int = 1) -> bool:
        """Synchronous token acquisition."""
        bucket = self._get_bucket(key)

        for attempt in range(self.config.max_retries):
            success, wait_time = bucket.acquire(tokens)
            if success:
                return True

            self._metrics.inc_counter("rate_limit_hits_total", labels={"key": key})
            time.sleep(wait_time)

        return False

    @asynccontextmanager
    async def limit(self, key: str = "default"):
        """Async context manager for rate-limited operations."""
        await self.acquire(key)
        try:
            yield
        finally:
            pass  # Token already consumed


def rate_limited(key: str = "default"):
    """Decorator for rate-limited async functions."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            limiter = get_rate_limiter()
            await limiter.acquire(key)
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Global rate limiter
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(config: Optional[RateLimitConfig] = None) -> RateLimiter:
    """Get or create the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(config)
    return _rate_limiter


# ============================================================================
# Graceful Shutdown Handling
# ============================================================================

class ShutdownHandler:
    """Handles graceful shutdown of Ralph Loop operations."""

    def __init__(self, timeout_seconds: float = 30.0):
        self.timeout_seconds = timeout_seconds
        self._shutdown_event = asyncio.Event()
        self._sync_shutdown_event = threading.Event()
        self._registered_tasks: List[asyncio.Task] = []
        self._cleanup_callbacks: List[Callable] = []
        self._is_shutting_down = False
        self._lock = threading.Lock()
        self._logger = RalphLogger("shutdown_handler")

        # Register signal handlers
        self._register_signals()

    def _register_signals(self) -> None:
        """Register signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except (ValueError, OSError):
            # Signals not available (e.g., in non-main thread)
            pass

        # Register atexit handler
        atexit.register(self._atexit_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        sig_name = signal.Signals(signum).name
        self._logger.info(f"Received {sig_name}, initiating graceful shutdown")
        self.initiate_shutdown()

    def _atexit_handler(self) -> None:
        """Handle atexit callback."""
        if not self._is_shutting_down:
            self.initiate_shutdown()

    def initiate_shutdown(self) -> None:
        """Initiate graceful shutdown."""
        with self._lock:
            if self._is_shutting_down:
                return
            self._is_shutting_down = True

        self._logger.info("Initiating graceful shutdown")
        self._shutdown_event.set()
        self._sync_shutdown_event.set()

        # Run cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                self._logger.error(f"Cleanup callback failed: {e}")

    def register_cleanup(self, callback: Callable) -> None:
        """Register a cleanup callback to run on shutdown."""
        self._cleanup_callbacks.append(callback)

    def register_task(self, task: asyncio.Task) -> None:
        """Register an async task to be cancelled on shutdown."""
        self._registered_tasks.append(task)

    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown has been initiated."""
        return self._is_shutting_down

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()

    async def shutdown_tasks(self) -> None:
        """Cancel and wait for all registered tasks."""
        if not self._registered_tasks:
            return

        self._logger.info(f"Cancelling {len(self._registered_tasks)} tasks")

        # Cancel all tasks
        for task in self._registered_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._registered_tasks, return_exceptions=True),
                timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            self._logger.warning(f"Some tasks did not complete within {self.timeout_seconds}s")


# Global shutdown handler
_shutdown_handler: Optional[ShutdownHandler] = None


def get_shutdown_handler(timeout_seconds: float = 30.0) -> ShutdownHandler:
    """Get or create the global shutdown handler."""
    global _shutdown_handler
    if _shutdown_handler is None:
        _shutdown_handler = ShutdownHandler(timeout_seconds)
    return _shutdown_handler


# ============================================================================
# State Checkpointing for Crash Recovery
# ============================================================================

@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    checkpoint_id: str
    loop_id: str
    iteration: int
    timestamp: str
    fitness: float
    status: str
    size_bytes: int
    checksum: str


class CheckpointManager:
    """Manages state checkpointing for crash recovery."""

    def __init__(
        self,
        checkpoint_dir: Path,
        max_checkpoints: int = 10,
        auto_save_interval: int = 5
    ):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.auto_save_interval = auto_save_interval
        self._metrics = get_metrics()
        self._logger = RalphLogger("checkpoint_manager")
        self._lock = threading.Lock()

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, loop_id: str, checkpoint_id: str) -> Path:
        """Get the path for a checkpoint file."""
        loop_dir = self.checkpoint_dir / loop_id
        loop_dir.mkdir(parents=True, exist_ok=True)
        return loop_dir / f"{checkpoint_id}.json"

    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate MD5 checksum of data."""
        return hashlib.md5(data).hexdigest()

    def save_checkpoint(
        self,
        loop_id: str,
        state_dict: Dict[str, Any],
        iteration: int,
        fitness: float,
        status: str
    ) -> CheckpointMetadata:
        """Save a checkpoint."""
        start_time = time.perf_counter()

        checkpoint_id = f"checkpoint_{iteration}_{int(time.time())}"
        checkpoint_path = self._get_checkpoint_path(loop_id, checkpoint_id)

        # Serialize state
        state_json = json.dumps(state_dict, default=str, indent=2)
        state_bytes = state_json.encode("utf-8")
        checksum = self._calculate_checksum(state_bytes)

        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            loop_id=loop_id,
            iteration=iteration,
            timestamp=datetime.now(timezone.utc).isoformat(),
            fitness=fitness,
            status=status,
            size_bytes=len(state_bytes),
            checksum=checksum
        )

        # Write checkpoint with metadata
        checkpoint_data = {
            "metadata": {
                "checkpoint_id": metadata.checkpoint_id,
                "loop_id": metadata.loop_id,
                "iteration": metadata.iteration,
                "timestamp": metadata.timestamp,
                "fitness": metadata.fitness,
                "status": metadata.status,
                "size_bytes": metadata.size_bytes,
                "checksum": metadata.checksum
            },
            "state": state_dict
        }

        with self._lock:
            # Write atomically using temp file + rename
            temp_path = checkpoint_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(checkpoint_data, f, default=str, indent=2)
            temp_path.rename(checkpoint_path)

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints(loop_id)

        # Record metrics
        duration = time.perf_counter() - start_time
        self._metrics.inc_counter("checkpoints_saved_total", labels={"loop_id": loop_id})
        self._metrics.observe_histogram("checkpoint_size_bytes", metadata.size_bytes, labels={"loop_id": loop_id})
        self._logger.info(
            f"Checkpoint saved",
            checkpoint_id=checkpoint_id,
            size_bytes=metadata.size_bytes,
            duration_ms=duration * 1000
        )

        return metadata

    def _cleanup_old_checkpoints(self, loop_id: str) -> None:
        """Remove old checkpoints keeping only the most recent ones."""
        loop_dir = self.checkpoint_dir / loop_id
        if not loop_dir.exists():
            return

        checkpoints = sorted(
            loop_dir.glob("checkpoint_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        # Remove excess checkpoints
        for checkpoint in checkpoints[self.max_checkpoints:]:
            try:
                checkpoint.unlink()
                self._logger.debug(f"Removed old checkpoint: {checkpoint.name}")
            except OSError as e:
                self._logger.warning(f"Failed to remove checkpoint: {e}")

    def load_checkpoint(self, loop_id: str, checkpoint_id: Optional[str] = None) -> Optional[Tuple[CheckpointMetadata, Dict[str, Any]]]:
        """Load a checkpoint. If no checkpoint_id, load the most recent."""
        loop_dir = self.checkpoint_dir / loop_id

        if checkpoint_id:
            checkpoint_path = self._get_checkpoint_path(loop_id, checkpoint_id)
        else:
            # Find most recent checkpoint
            checkpoints = list(loop_dir.glob("checkpoint_*.json")) if loop_dir.exists() else []
            if not checkpoints:
                return None
            checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, "r") as f:
                checkpoint_data = json.load(f)

            # Verify checksum
            state_bytes = json.dumps(checkpoint_data["state"], default=str).encode("utf-8")
            calculated_checksum = self._calculate_checksum(state_bytes)

            meta = checkpoint_data["metadata"]
            if meta.get("checksum") and meta["checksum"] != calculated_checksum:
                self._logger.warning(f"Checkpoint checksum mismatch: {checkpoint_path.name}")

            metadata = CheckpointMetadata(
                checkpoint_id=meta["checkpoint_id"],
                loop_id=meta["loop_id"],
                iteration=meta["iteration"],
                timestamp=meta["timestamp"],
                fitness=meta["fitness"],
                status=meta["status"],
                size_bytes=meta["size_bytes"],
                checksum=meta.get("checksum", "")
            )

            self._logger.info(
                f"Checkpoint loaded",
                checkpoint_id=metadata.checkpoint_id,
                iteration=metadata.iteration
            )

            return metadata, checkpoint_data["state"]

        except (json.JSONDecodeError, KeyError) as e:
            self._logger.error(f"Failed to load checkpoint: {e}")
            return None

    def list_checkpoints(self, loop_id: str) -> List[CheckpointMetadata]:
        """List all checkpoints for a loop."""
        loop_dir = self.checkpoint_dir / loop_id
        if not loop_dir.exists():
            return []

        checkpoints = []
        for checkpoint_path in loop_dir.glob("checkpoint_*.json"):
            try:
                with open(checkpoint_path, "r") as f:
                    data = json.load(f)
                    meta = data["metadata"]
                    checkpoints.append(CheckpointMetadata(
                        checkpoint_id=meta["checkpoint_id"],
                        loop_id=meta["loop_id"],
                        iteration=meta["iteration"],
                        timestamp=meta["timestamp"],
                        fitness=meta["fitness"],
                        status=meta["status"],
                        size_bytes=meta["size_bytes"],
                        checksum=meta.get("checksum", "")
                    ))
            except (json.JSONDecodeError, KeyError):
                continue

        return sorted(checkpoints, key=lambda c: c.iteration, reverse=True)


# ============================================================================
# Enhanced V11 Feature Implementations
# ============================================================================

@dataclass
class HypothesisTracker:
    """
    Production-ready hypothesis tracking for Speculative Decoding.

    Tracks hypothesis generation, verification, and statistics with proper
    metrics and logging for production observability.
    """
    hypothesis_id: str
    content: str
    confidence: float
    generation_cost: int
    generated_at: str
    verification_status: str = "pending"  # pending, verified, rejected
    verification_result: Optional[bool] = None
    verification_reasoning: str = ""
    verification_latency_ms: float = 0.0
    batch_id: Optional[str] = None
    parent_hypothesis_id: Optional[str] = None


class SpeculativeDecodingManager:
    """
    Production-ready Speculative Decoding manager.

    Implements PEARL-style (ICLR 2025) parallel hypothesis generation
    with proper tracking, verification, and metrics.
    """

    def __init__(
        self,
        max_parallel_hypotheses: int = 4,
        verification_timeout_seconds: float = 30.0,
        min_confidence_threshold: float = 0.3
    ):
        self.max_parallel_hypotheses = max_parallel_hypotheses
        self.verification_timeout = verification_timeout_seconds
        self.min_confidence_threshold = min_confidence_threshold
        self._hypotheses: Dict[str, HypothesisTracker] = {}
        self._batch_counter = 0
        self._metrics = get_metrics()
        self._logger = RalphLogger("speculative_decoding")
        self._lock = threading.Lock()

        # Statistics
        self._total_generated = 0
        self._total_verified = 0
        self._total_rejected = 0

    def create_batch_id(self) -> str:
        """Create a new batch ID for parallel hypotheses."""
        self._batch_counter += 1
        return f"batch_{self._batch_counter}_{int(time.time())}"

    async def generate_hypotheses(
        self,
        context: str,
        generator_func: Callable[[str], Any],
        num_hypotheses: Optional[int] = None
    ) -> List[HypothesisTracker]:
        """Generate multiple hypotheses in parallel."""
        batch_id = self.create_batch_id()
        num = num_hypotheses or self.max_parallel_hypotheses

        self._logger.info(
            f"Generating {num} speculative hypotheses",
            batch_id=batch_id
        )

        # Generate hypotheses in parallel
        hypotheses = []
        start_time = time.perf_counter()

        try:
            tasks = [
                asyncio.create_task(self._generate_single(context, generator_func, batch_id, i))
                for i in range(num)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, HypothesisTracker):
                    hypotheses.append(result)
                    with self._lock:
                        self._hypotheses[result.hypothesis_id] = result
                elif isinstance(result, Exception):
                    self._logger.warning(f"Hypothesis generation failed: {result}")

        except Exception as e:
            self._logger.error(f"Batch hypothesis generation failed: {e}")

        duration_ms = (time.perf_counter() - start_time) * 1000
        self._total_generated += len(hypotheses)
        self._metrics.inc_counter("hypothesis_generated_total", len(hypotheses))
        self._logger.info(
            f"Generated {len(hypotheses)} hypotheses",
            batch_id=batch_id,
            duration_ms=duration_ms
        )

        return hypotheses

    async def _generate_single(
        self,
        context: str,
        generator_func: Callable[[str], Any],
        batch_id: str,
        index: int
    ) -> HypothesisTracker:
        """Generate a single hypothesis."""
        hypothesis_id = f"hyp_{batch_id}_{index}"
        start_time = time.perf_counter()

        try:
            result = await generator_func(context)
            content = str(result.get("content", result) if isinstance(result, dict) else result)
            confidence = float(result.get("confidence", 0.5) if isinstance(result, dict) else 0.5)
            cost = int(result.get("tokens", 0) if isinstance(result, dict) else 0)
        except Exception as e:
            content = f"[Generation failed: {e}]"
            confidence = 0.0
            cost = 0

        return HypothesisTracker(
            hypothesis_id=hypothesis_id,
            content=content,
            confidence=confidence,
            generation_cost=cost,
            generated_at=datetime.now(timezone.utc).isoformat(),
            batch_id=batch_id
        )

    async def verify_hypothesis(
        self,
        hypothesis: HypothesisTracker,
        verifier_func: Callable[[str], Any],
        timeout: Optional[float] = None
    ) -> HypothesisTracker:
        """Verify a single hypothesis."""
        start_time = time.perf_counter()

        try:
            result = await asyncio.wait_for(
                verifier_func(hypothesis.content),
                timeout=timeout or self.verification_timeout
            )

            if isinstance(result, dict):
                verified = result.get("verified", False)
                reasoning = result.get("reasoning", "")
            else:
                verified = bool(result)
                reasoning = str(result) if result else ""

            hypothesis.verification_status = "verified" if verified else "rejected"
            hypothesis.verification_result = verified
            hypothesis.verification_reasoning = reasoning[:500]

        except asyncio.TimeoutError:
            hypothesis.verification_status = "rejected"
            hypothesis.verification_result = False
            hypothesis.verification_reasoning = "Verification timeout"
        except Exception as e:
            hypothesis.verification_status = "rejected"
            hypothesis.verification_result = False
            hypothesis.verification_reasoning = f"Error: {e}"

        hypothesis.verification_latency_ms = (time.perf_counter() - start_time) * 1000

        # Update statistics and metrics
        if hypothesis.verification_result:
            self._total_verified += 1
            self._metrics.inc_counter("hypothesis_verified_total")
        else:
            self._total_rejected += 1
            self._metrics.inc_counter("hypothesis_rejected_total")

        self._metrics.observe_histogram(
            "verification_latency_seconds",
            hypothesis.verification_latency_ms / 1000
        )

        return hypothesis

    async def verify_batch(
        self,
        hypotheses: List[HypothesisTracker],
        verifier_func: Callable[[str], Any]
    ) -> List[HypothesisTracker]:
        """Verify multiple hypotheses in parallel."""
        tasks = [
            asyncio.create_task(self.verify_hypothesis(h, verifier_func))
            for h in hypotheses
        ]
        return await asyncio.gather(*tasks)

    def get_best_hypothesis(self, hypotheses: List[HypothesisTracker]) -> Optional[HypothesisTracker]:
        """Get the best verified hypothesis by confidence."""
        verified = [h for h in hypotheses if h.verification_result]
        if not verified:
            return None
        return max(verified, key=lambda h: h.confidence)

    @property
    def acceptance_rate(self) -> float:
        """Calculate overall acceptance rate."""
        total = self._total_verified + self._total_rejected
        return self._total_verified / total if total > 0 else 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get speculation statistics."""
        return {
            "total_generated": self._total_generated,
            "total_verified": self._total_verified,
            "total_rejected": self._total_rejected,
            "acceptance_rate": self.acceptance_rate,
            "active_hypotheses": len(self._hypotheses)
        }


class ChainOfDraftManager:
    """
    Production-ready Chain-of-Draft manager for token compression.

    Implements arxiv 2502.18600 Chain-of-Draft reasoning with 92% token compression.
    """

    def __init__(
        self,
        max_tokens_per_step: int = 5,
        target_compression_ratio: float = 0.92
    ):
        self.max_tokens_per_step = max_tokens_per_step
        self.target_compression_ratio = target_compression_ratio
        self._metrics = get_metrics()
        self._logger = RalphLogger("chain_of_draft")

        # Statistics
        self._total_draft_tokens = 0
        self._total_equivalent_cot_tokens = 0

    async def compress_reasoning(
        self,
        full_reasoning: str,
        compressor_func: Callable[[str], Any]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Compress full reasoning into minimal-token draft steps.

        Returns (draft_steps, statistics)
        """
        start_time = time.perf_counter()

        try:
            result = await compressor_func(full_reasoning)

            if isinstance(result, dict):
                draft_steps = result.get("steps", [full_reasoning])
                original_tokens = result.get("original_tokens", len(full_reasoning.split()))
            else:
                draft_steps = [str(result)]
                original_tokens = len(full_reasoning.split())

        except Exception as e:
            self._logger.error(f"Compression failed: {e}")
            draft_steps = [full_reasoning]
            original_tokens = len(full_reasoning.split())

        # Calculate compression statistics
        draft_tokens = sum(len(step.split()) for step in draft_steps)
        compression_ratio = 1 - (draft_tokens / max(1, original_tokens))

        self._total_draft_tokens += draft_tokens
        self._total_equivalent_cot_tokens += original_tokens

        duration_ms = (time.perf_counter() - start_time) * 1000

        self._metrics.set_gauge("token_compression_ratio", compression_ratio)

        stats = {
            "draft_steps": len(draft_steps),
            "draft_tokens": draft_tokens,
            "original_tokens": original_tokens,
            "compression_ratio": compression_ratio,
            "duration_ms": duration_ms
        }

        self._logger.debug(
            f"Compressed reasoning: {original_tokens} -> {draft_tokens} tokens ({compression_ratio:.1%})"
        )

        return draft_steps, stats

    def get_overall_compression(self) -> float:
        """Get overall compression ratio across all operations."""
        if self._total_equivalent_cot_tokens == 0:
            return 0.0
        return 1 - (self._total_draft_tokens / self._total_equivalent_cot_tokens)


class AdaptiveRAGManager:
    """
    Production-ready Adaptive RAG manager.

    Implements INKER/DynamicRAG-style adaptive retrieval based on
    confidence and novelty thresholds.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        novelty_threshold: float = 0.5,
        max_retrievals_per_query: int = 5
    ):
        self.confidence_threshold = confidence_threshold
        self.novelty_threshold = novelty_threshold
        self.max_retrievals = max_retrievals_per_query
        self._metrics = get_metrics()
        self._logger = RalphLogger("adaptive_rag")

        # Statistics
        self._total_queries = 0
        self._retrievals_performed = 0
        self._retrievals_skipped = 0
        self._successful_retrievals = 0

    def should_retrieve(
        self,
        confidence: float,
        novelty: float,
        context: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Determine whether to perform retrieval based on confidence and novelty.

        Returns (should_retrieve, reasoning)
        """
        self._total_queries += 1

        # Low confidence -> retrieve for verification
        if confidence < self.confidence_threshold:
            self._logger.debug(f"Retrieval recommended: low confidence ({confidence:.2f})")
            return True, f"Low confidence ({confidence:.2f}) requires external knowledge"

        # High novelty -> retrieve for new information
        if novelty > self.novelty_threshold:
            self._logger.debug(f"Retrieval recommended: high novelty ({novelty:.2f})")
            return True, f"High novelty ({novelty:.2f}) suggests unseen information"

        self._retrievals_skipped += 1
        return False, f"Internal knowledge sufficient (conf={confidence:.2f}, novelty={novelty:.2f})"

    async def perform_retrieval(
        self,
        query: str,
        retriever_func: Callable[[str], Any],
        confidence: float,
        novelty: float
    ) -> Dict[str, Any]:
        """
        Perform adaptive retrieval if needed.

        Returns retrieval results with metadata.
        """
        start_time = time.perf_counter()

        should_retrieve, reasoning = self.should_retrieve(confidence, novelty, query)

        if not should_retrieve:
            return {
                "retrieved": False,
                "reason": reasoning,
                "results": [],
                "latency_ms": 0
            }

        try:
            self._retrievals_performed += 1
            self._metrics.inc_counter("rag_retrievals_total")

            results = await retriever_func(query)

            if isinstance(results, dict):
                documents = results.get("documents", [])
                scores = results.get("scores", [])
            else:
                documents = results if isinstance(results, list) else [str(results)]
                scores = []

            # Track success
            if documents:
                self._successful_retrievals += 1

            latency_ms = (time.perf_counter() - start_time) * 1000

            return {
                "retrieved": True,
                "reason": reasoning,
                "results": documents[:self.max_retrievals],
                "scores": scores[:self.max_retrievals] if scores else [],
                "latency_ms": latency_ms,
                "count": len(documents)
            }

        except Exception as e:
            self._logger.error(f"Retrieval failed: {e}")
            return {
                "retrieved": True,
                "reason": reasoning,
                "results": [],
                "error": str(e),
                "latency_ms": (time.perf_counter() - start_time) * 1000
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG statistics."""
        return {
            "total_queries": self._total_queries,
            "retrievals_performed": self._retrievals_performed,
            "retrievals_skipped": self._retrievals_skipped,
            "successful_retrievals": self._successful_retrievals,
            "retrieval_rate": self._retrievals_performed / max(1, self._total_queries),
            "success_rate": self._successful_retrievals / max(1, self._retrievals_performed)
        }


@dataclass
class RewardHackingSignal:
    """A detected reward hacking signal."""
    signal_id: str
    signal_type: str  # proxy_divergence, specification_gaming, reward_tampering, suspicious_pattern
    severity: float  # 0.0 to 1.0
    description: str
    evidence: Dict[str, Any]
    detected_at: str
    mitigation_applied: bool = False
    mitigation_action: Optional[str] = None


class RewardHackingDetector:
    """
    Production-ready reward hacking detection.

    Implements detection strategies from A2RM and APRM (ICLR 2026):
    - Proxy divergence detection
    - Specification gaming patterns
    - Reward tampering signals
    - Statistical anomaly detection
    """

    def __init__(
        self,
        proxy_divergence_threshold: float = 0.3,
        suspicious_improvement_threshold: float = 0.5,
        history_window: int = 50
    ):
        self.proxy_divergence_threshold = proxy_divergence_threshold
        self.suspicious_improvement_threshold = suspicious_improvement_threshold
        self.history_window = history_window
        self._metrics = get_metrics()
        self._logger = RalphLogger("reward_hacking_detector")

        # History tracking
        self._reward_history: List[float] = []
        self._proxy_reward_history: List[float] = []
        self._detected_signals: List[RewardHackingSignal] = []
        self._signal_counter = 0
        self._lock = threading.Lock()

    def record_rewards(self, true_reward: float, proxy_reward: float) -> None:
        """Record reward observations for divergence tracking."""
        with self._lock:
            self._reward_history.append(true_reward)
            self._proxy_reward_history.append(proxy_reward)

            # Maintain window size
            if len(self._reward_history) > self.history_window:
                self._reward_history = self._reward_history[-self.history_window:]
                self._proxy_reward_history = self._proxy_reward_history[-self.history_window:]

    def check_proxy_divergence(self, true_reward: float, proxy_reward: float) -> Optional[RewardHackingSignal]:
        """
        Check for proxy-true reward divergence (Goodhart's Law).

        When proxy reward increases but true reward stagnates or decreases,
        this indicates potential reward hacking.
        """
        if len(self._reward_history) < 5:
            return None

        # Calculate recent trends
        recent_true = self._reward_history[-5:]
        recent_proxy = self._proxy_reward_history[-5:]

        true_trend = recent_true[-1] - recent_true[0]
        proxy_trend = recent_proxy[-1] - recent_proxy[0]

        # Divergence: proxy improving but true stagnating/declining
        if proxy_trend > 0.1 and true_trend < 0.01:
            divergence = proxy_trend - true_trend
            if divergence > self.proxy_divergence_threshold:
                signal = self._create_signal(
                    signal_type="proxy_divergence",
                    severity=min(1.0, divergence / self.proxy_divergence_threshold),
                    description=f"Proxy reward improving (+{proxy_trend:.3f}) while true reward stagnant ({true_trend:+.3f})",
                    evidence={
                        "true_trend": true_trend,
                        "proxy_trend": proxy_trend,
                        "divergence": divergence,
                        "recent_true": recent_true,
                        "recent_proxy": recent_proxy
                    }
                )
                return signal

        return None

    def check_suspicious_improvement(self, improvement: float) -> Optional[RewardHackingSignal]:
        """
        Check for suspiciously large single-step improvements.

        Sudden large gains may indicate gaming rather than genuine improvement.
        """
        if len(self._reward_history) < 10:
            return None

        # Calculate historical improvement statistics
        improvements = [
            self._reward_history[i] - self._reward_history[i-1]
            for i in range(1, len(self._reward_history))
        ]

        if not improvements:
            return None

        avg_improvement = sum(improvements) / len(improvements)
        std_improvement = (sum((x - avg_improvement) ** 2 for x in improvements) / len(improvements)) ** 0.5

        # Check if current improvement is anomalous (>3 std devs)
        if std_improvement > 0 and abs(improvement - avg_improvement) > 3 * std_improvement:
            if improvement > self.suspicious_improvement_threshold:
                signal = self._create_signal(
                    signal_type="suspicious_pattern",
                    severity=min(1.0, improvement / self.suspicious_improvement_threshold),
                    description=f"Anomalously large improvement: {improvement:.3f} (avg={avg_improvement:.3f}, std={std_improvement:.3f})",
                    evidence={
                        "improvement": improvement,
                        "avg_improvement": avg_improvement,
                        "std_improvement": std_improvement,
                        "z_score": (improvement - avg_improvement) / std_improvement
                    }
                )
                return signal

        return None

    def check_specification_gaming(
        self,
        solution: str,
        previous_solution: str,
        fitness_change: float
    ) -> Optional[RewardHackingSignal]:
        """
        Check for specification gaming patterns.

        Detects solutions that technically satisfy metrics but violate intent.
        """
        # Simple heuristic checks (in production, use more sophisticated analysis)
        gaming_patterns = [
            ("trivial", solution.strip() == "" or len(solution) < 10),
            ("repetition", self._check_repetition(solution)),
            ("length_gaming", len(solution) > len(previous_solution) * 10 and fitness_change > 0),
        ]

        for pattern_name, detected in gaming_patterns:
            if detected:
                signal = self._create_signal(
                    signal_type="specification_gaming",
                    severity=0.6,
                    description=f"Potential specification gaming detected: {pattern_name}",
                    evidence={
                        "pattern": pattern_name,
                        "solution_length": len(solution),
                        "fitness_change": fitness_change
                    }
                )
                return signal

        return None

    def _check_repetition(self, text: str, threshold: float = 0.7) -> bool:
        """Check if text contains excessive repetition."""
        if len(text) < 100:
            return False

        words = text.lower().split()
        if len(words) < 10:
            return False

        unique_ratio = len(set(words)) / len(words)
        return unique_ratio < (1 - threshold)

    def _create_signal(
        self,
        signal_type: str,
        severity: float,
        description: str,
        evidence: Dict[str, Any]
    ) -> RewardHackingSignal:
        """Create and register a reward hacking signal."""
        self._signal_counter += 1

        signal = RewardHackingSignal(
            signal_id=f"rh_signal_{self._signal_counter}",
            signal_type=signal_type,
            severity=severity,
            description=description,
            evidence=evidence,
            detected_at=datetime.now(timezone.utc).isoformat()
        )

        with self._lock:
            self._detected_signals.append(signal)
            if len(self._detected_signals) > 100:
                self._detected_signals = self._detected_signals[-100:]

        self._metrics.inc_counter(
            "reward_hacking_detected_total",
            labels={"type": signal_type}
        )

        self._logger.warning(
            f"Reward hacking signal detected: {signal_type}",
            severity=severity,
            signal_id=signal.signal_id
        )

        return signal

    def apply_mitigation(self, signal: RewardHackingSignal, action: str) -> None:
        """Record mitigation action for a signal."""
        signal.mitigation_applied = True
        signal.mitigation_action = action
        self._logger.info(f"Mitigation applied: {action}", signal_id=signal.signal_id)

    def get_signals(self, min_severity: float = 0.0) -> List[RewardHackingSignal]:
        """Get detected signals above minimum severity."""
        with self._lock:
            return [s for s in self._detected_signals if s.severity >= min_severity]

    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        with self._lock:
            signals_by_type = defaultdict(int)
            for signal in self._detected_signals:
                signals_by_type[signal.signal_type] += 1

            return {
                "total_signals": len(self._detected_signals),
                "signals_by_type": dict(signals_by_type),
                "history_size": len(self._reward_history),
                "avg_severity": sum(s.severity for s in self._detected_signals) / max(1, len(self._detected_signals))
            }


# ============================================================================
# Production Configuration
# ============================================================================

@dataclass
class ProductionConfig:
    """Configuration for production Ralph Loop deployment."""

    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    service_name: str = "ralph_loop"

    # Metrics
    metrics_namespace: str = "ralph_loop"
    metrics_export_interval: int = 60

    # Rate limiting
    api_requests_per_second: float = 10.0
    api_burst_size: int = 20

    # Checkpointing
    checkpoint_dir: Path = field(default_factory=lambda: Path.home() / ".ralph" / "checkpoints")
    max_checkpoints: int = 10
    auto_save_interval: int = 5

    # Shutdown
    shutdown_timeout_seconds: float = 30.0

    # V11 Features
    speculative_max_hypotheses: int = 4
    speculative_verification_timeout: float = 30.0
    chain_of_draft_max_tokens: int = 5
    rag_confidence_threshold: float = 0.7
    rag_novelty_threshold: float = 0.5
    reward_hacking_proxy_threshold: float = 0.3

    @classmethod
    def from_env(cls) -> ProductionConfig:
        """Load configuration from environment variables."""
        return cls(
            log_level=os.environ.get("RALPH_LOG_LEVEL", "INFO"),
            log_file=Path(os.environ["RALPH_LOG_FILE"]) if "RALPH_LOG_FILE" in os.environ else None,
            service_name=os.environ.get("RALPH_SERVICE_NAME", "ralph_loop"),
            api_requests_per_second=float(os.environ.get("RALPH_API_RPS", "10.0")),
            api_burst_size=int(os.environ.get("RALPH_API_BURST", "20")),
            checkpoint_dir=Path(os.environ.get("RALPH_CHECKPOINT_DIR", str(Path.home() / ".ralph" / "checkpoints"))),
            max_checkpoints=int(os.environ.get("RALPH_MAX_CHECKPOINTS", "10")),
            shutdown_timeout_seconds=float(os.environ.get("RALPH_SHUTDOWN_TIMEOUT", "30.0")),
        )


def initialize_production(config: Optional[ProductionConfig] = None) -> Dict[str, Any]:
    """
    Initialize all production components.

    Returns a dict of initialized components.
    """
    config = config or ProductionConfig()

    # Configure logging
    logger = configure_production_logging(
        log_level=config.log_level,
        service_name=config.service_name,
        log_file=config.log_file
    )

    # Initialize metrics
    metrics = get_metrics(config.metrics_namespace)

    # Initialize rate limiter
    rate_limiter = get_rate_limiter(RateLimitConfig(
        requests_per_second=config.api_requests_per_second,
        burst_size=config.api_burst_size
    ))

    # Initialize shutdown handler
    shutdown_handler = get_shutdown_handler(config.shutdown_timeout_seconds)

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config.checkpoint_dir,
        max_checkpoints=config.max_checkpoints,
        auto_save_interval=config.auto_save_interval
    )

    # Initialize V11 managers
    speculative_manager = SpeculativeDecodingManager(
        max_parallel_hypotheses=config.speculative_max_hypotheses,
        verification_timeout_seconds=config.speculative_verification_timeout
    )

    cod_manager = ChainOfDraftManager(
        max_tokens_per_step=config.chain_of_draft_max_tokens
    )

    rag_manager = AdaptiveRAGManager(
        confidence_threshold=config.rag_confidence_threshold,
        novelty_threshold=config.rag_novelty_threshold
    )

    reward_detector = RewardHackingDetector(
        proxy_divergence_threshold=config.reward_hacking_proxy_threshold
    )

    logger.info(
        "Production components initialized",
        config=config.service_name,
        checkpoint_dir=str(config.checkpoint_dir)
    )

    return {
        "config": config,
        "logger": logger,
        "metrics": metrics,
        "rate_limiter": rate_limiter,
        "shutdown_handler": shutdown_handler,
        "checkpoint_manager": checkpoint_manager,
        "speculative_manager": speculative_manager,
        "cod_manager": cod_manager,
        "rag_manager": rag_manager,
        "reward_detector": reward_detector
    }
