"""
Adapter Registry - Centralized Adapter Management with Health Monitoring
=========================================================================

Provides a centralized registry for all SDK adapters with:
- Explicit registration and discovery
- Health check monitoring for all adapters
- Startup status logging
- Version validation for SDK dependencies
- Graceful degradation on import failures

Usage:
    from adapters.registry import AdapterRegistry, get_registry

    # Get the singleton registry
    registry = get_registry()

    # Register adapters (done automatically on import)
    registry.register_adapter("exa", ExaAdapter, sdk_name="exa_py", min_version="0.1.0")

    # Get an adapter
    adapter_cls = registry.get_adapter("exa")
    if adapter_cls:
        adapter = adapter_cls()
        await adapter.initialize({})

    # List all available adapters
    available = registry.list_available()

    # Health check all adapters
    health = await registry.health_check_all()

    # Get detailed status report
    status = registry.get_status_report()
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AdapterLoadStatus(str, Enum):
    """Status of adapter loading."""
    NOT_LOADED = "not_loaded"
    LOADED = "loaded"
    SDK_MISSING = "sdk_missing"
    SDK_VERSION_MISMATCH = "sdk_version_mismatch"
    IMPORT_ERROR = "import_error"
    INITIALIZATION_ERROR = "initialization_error"


@dataclass
class AdapterInfo:
    """Information about a registered adapter."""
    name: str
    adapter_class: Optional[Type] = None
    sdk_name: Optional[str] = None
    sdk_version: Optional[str] = None
    min_version: Optional[str] = None
    status: AdapterLoadStatus = AdapterLoadStatus.NOT_LOADED
    error_message: Optional[str] = None
    load_time_ms: float = 0.0
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_health_check: Optional[datetime] = None
    is_healthy: Optional[bool] = None
    features: List[str] = field(default_factory=list)
    layer: Optional[str] = None
    priority: int = 0


@dataclass
class HealthCheckResult:
    """Result of an adapter health check."""
    adapter_name: str
    is_healthy: bool
    status: str
    latency_ms: float
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.utcnow)


class AdapterRegistry:
    """
    Centralized registry for SDK adapters.

    Provides:
    - Adapter registration with metadata
    - SDK dependency validation
    - Health monitoring
    - Startup logging
    - Graceful error handling
    """

    _instance: Optional["AdapterRegistry"] = None

    def __new__(cls) -> "AdapterRegistry":
        """Singleton pattern for global registry access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._adapters: Dict[str, AdapterInfo] = {}
        self._adapter_instances: Dict[str, Any] = {}
        self._load_callbacks: List[Callable[[str, AdapterInfo], None]] = []
        self._initialized = True
        self._startup_logged = False

    def register_adapter(
        self,
        name: str,
        adapter_class: Optional[Type] = None,
        sdk_name: Optional[str] = None,
        min_version: Optional[str] = None,
        layer: Optional[str] = None,
        priority: int = 0,
        features: Optional[List[str]] = None,
    ) -> None:
        """
        Register an adapter with the registry.

        Args:
            name: Unique adapter name (e.g., "exa", "tavily")
            adapter_class: The adapter class to register
            sdk_name: Name of the SDK package (for version checking)
            min_version: Minimum required SDK version
            layer: SDK layer (e.g., "RESEARCH", "KNOWLEDGE")
            priority: Priority for ordering (higher = more preferred)
            features: List of supported features/operations
        """
        start_time = time.time()

        info = AdapterInfo(
            name=name,
            adapter_class=adapter_class,
            sdk_name=sdk_name,
            min_version=min_version,
            layer=layer,
            priority=priority,
            features=features or [],
        )

        # Validate SDK if specified
        if sdk_name:
            sdk_available, sdk_version, sdk_error = self._check_sdk_availability(
                sdk_name, min_version
            )
            info.sdk_version = sdk_version

            if not sdk_available:
                if "not installed" in (sdk_error or "").lower():
                    info.status = AdapterLoadStatus.SDK_MISSING
                elif "version" in (sdk_error or "").lower():
                    info.status = AdapterLoadStatus.SDK_VERSION_MISMATCH
                else:
                    info.status = AdapterLoadStatus.IMPORT_ERROR
                info.error_message = sdk_error
                logger.warning(
                    f"Adapter '{name}' SDK unavailable: {sdk_error}"
                )
            else:
                info.status = AdapterLoadStatus.LOADED
                logger.debug(
                    f"Adapter '{name}' registered with SDK {sdk_name}=={sdk_version}"
                )
        elif adapter_class:
            info.status = AdapterLoadStatus.LOADED
            logger.debug(f"Adapter '{name}' registered (no SDK dependency)")
        else:
            info.status = AdapterLoadStatus.NOT_LOADED
            info.error_message = "No adapter class provided"

        info.load_time_ms = (time.time() - start_time) * 1000
        self._adapters[name] = info

        # Notify callbacks
        for callback in self._load_callbacks:
            try:
                callback(name, info)
            except Exception as e:
                logger.warning(f"Load callback error for '{name}': {e}")

    def _check_sdk_availability(
        self,
        sdk_name: str,
        min_version: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if an SDK is available and meets version requirements.

        Returns:
            Tuple of (available, version, error_message)
        """
        # Map SDK names to import names (handle cases like exa_py -> exa)
        import_name_map = {
            "exa_py": "exa_py",
            "exa-py": "exa_py",
            "tavily-python": "tavily",
            "tavily_python": "tavily",
            "firecrawl-py": "firecrawl",
            "firecrawl_py": "firecrawl",
        }

        import_name = import_name_map.get(sdk_name, sdk_name.replace("-", "_"))

        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", None)

            if version is None:
                # Try common version attribute locations
                for attr in ["VERSION", "version", "__VERSION__"]:
                    version = getattr(module, attr, None)
                    if version:
                        break

            # Version comparison if required
            if min_version and version:
                if not self._version_meets_minimum(version, min_version):
                    return (
                        False,
                        version,
                        f"SDK {sdk_name} version {version} < required {min_version}",
                    )

            return True, version, None

        except ImportError as e:
            return False, None, f"SDK {sdk_name} not installed: {e}"
        except Exception as e:
            return False, None, f"Error checking SDK {sdk_name}: {e}"

    def _version_meets_minimum(
        self, version: str, min_version: str
    ) -> bool:
        """
        Check if a version string meets minimum requirements.

        Handles semver-like versions (1.2.3, 1.2.3a1, etc.)
        """
        def parse_version(v: str) -> Tuple[int, ...]:
            """Parse version string to comparable tuple."""
            # Remove common prefixes and clean up
            v = v.lstrip("v").split("+")[0].split("-")[0]
            # Extract numeric parts
            parts = []
            for part in v.split("."):
                # Handle versions like "1.2.3a1"
                num = ""
                for char in part:
                    if char.isdigit():
                        num += char
                    else:
                        break
                parts.append(int(num) if num else 0)
            return tuple(parts)

        try:
            return parse_version(version) >= parse_version(min_version)
        except Exception:
            # If parsing fails, assume it's acceptable
            return True

    def get_adapter(self, name: str) -> Optional[Type]:
        """
        Get an adapter class by name.

        Args:
            name: Adapter name

        Returns:
            Adapter class if available and loaded, None otherwise
        """
        info = self._adapters.get(name)
        if info and info.status == AdapterLoadStatus.LOADED:
            return info.adapter_class
        return None

    def get_adapter_info(self, name: str) -> Optional[AdapterInfo]:
        """
        Get full adapter information by name.

        Args:
            name: Adapter name

        Returns:
            AdapterInfo if registered, None otherwise
        """
        return self._adapters.get(name)

    def list_available(self) -> List[str]:
        """
        List all available (loaded) adapter names.

        Returns:
            List of adapter names that are ready to use
        """
        return [
            name
            for name, info in self._adapters.items()
            if info.status == AdapterLoadStatus.LOADED
        ]

    def list_all(self) -> List[str]:
        """
        List all registered adapter names (including unavailable).

        Returns:
            List of all registered adapter names
        """
        return list(self._adapters.keys())

    def list_by_layer(self, layer: str) -> List[str]:
        """
        List adapters by SDK layer.

        Args:
            layer: Layer name (e.g., "RESEARCH", "KNOWLEDGE")

        Returns:
            List of adapter names in the specified layer
        """
        return [
            name
            for name, info in self._adapters.items()
            if info.layer == layer and info.status == AdapterLoadStatus.LOADED
        ]

    def list_unavailable(self) -> List[Tuple[str, str]]:
        """
        List unavailable adapters with their error messages.

        Returns:
            List of (name, error_message) tuples
        """
        return [
            (name, info.error_message or "Unknown error")
            for name, info in self._adapters.items()
            if info.status != AdapterLoadStatus.LOADED
        ]

    async def health_check_all(
        self,
        timeout: float = 30.0,
    ) -> Dict[str, HealthCheckResult]:
        """
        Run health checks on all available adapters.

        Args:
            timeout: Maximum time for all health checks

        Returns:
            Dict mapping adapter names to HealthCheckResult
        """
        results: Dict[str, HealthCheckResult] = {}
        available = self.list_available()

        if not available:
            logger.warning("No adapters available for health check")
            return results

        async def check_adapter(name: str) -> Tuple[str, HealthCheckResult]:
            """Check a single adapter's health."""
            start_time = time.time()
            info = self._adapters.get(name)

            if not info or not info.adapter_class:
                return name, HealthCheckResult(
                    adapter_name=name,
                    is_healthy=False,
                    status="not_loaded",
                    latency_ms=0,
                    error="Adapter not loaded",
                )

            try:
                # Create instance if not cached
                if name not in self._adapter_instances:
                    adapter = info.adapter_class()
                    # Initialize with empty config for health check
                    await asyncio.wait_for(
                        adapter.initialize({}),
                        timeout=10.0,
                    )
                    self._adapter_instances[name] = adapter

                adapter = self._adapter_instances[name]

                # Run health check if available
                if hasattr(adapter, "health_check"):
                    result = await asyncio.wait_for(
                        adapter.health_check(),
                        timeout=10.0,
                    )
                    is_healthy = result.success if hasattr(result, "success") else True
                    details = result.data if hasattr(result, "data") else {}
                    error = result.error if hasattr(result, "error") else None
                else:
                    is_healthy = True
                    details = {"message": "No health_check method"}
                    error = None

                latency_ms = (time.time() - start_time) * 1000

                # Update adapter info
                info.last_health_check = datetime.utcnow()
                info.is_healthy = is_healthy

                return name, HealthCheckResult(
                    adapter_name=name,
                    is_healthy=is_healthy,
                    status="healthy" if is_healthy else "unhealthy",
                    latency_ms=latency_ms,
                    error=error,
                    details=details or {},
                )

            except asyncio.TimeoutError:
                return name, HealthCheckResult(
                    adapter_name=name,
                    is_healthy=False,
                    status="timeout",
                    latency_ms=(time.time() - start_time) * 1000,
                    error="Health check timed out",
                )
            except Exception as e:
                return name, HealthCheckResult(
                    adapter_name=name,
                    is_healthy=False,
                    status="error",
                    latency_ms=(time.time() - start_time) * 1000,
                    error=str(e),
                )

        # Run health checks concurrently
        try:
            tasks = [check_adapter(name) for name in available]
            completed = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )

            for item in completed:
                if isinstance(item, tuple):
                    name, result = item
                    results[name] = result
                elif isinstance(item, Exception):
                    logger.error(f"Health check task failed: {item}")

        except asyncio.TimeoutError:
            logger.error(f"Health check timed out after {timeout}s")

        return results

    def get_status_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive status report for all adapters.

        Returns:
            Dict with status summary and details
        """
        loaded = []
        sdk_missing = []
        import_errors = []
        version_mismatches = []

        for name, info in self._adapters.items():
            entry = {
                "name": name,
                "sdk": info.sdk_name,
                "version": info.sdk_version,
                "layer": info.layer,
                "priority": info.priority,
                "features": info.features,
                "load_time_ms": info.load_time_ms,
            }

            if info.status == AdapterLoadStatus.LOADED:
                loaded.append(entry)
            elif info.status == AdapterLoadStatus.SDK_MISSING:
                entry["error"] = info.error_message
                sdk_missing.append(entry)
            elif info.status == AdapterLoadStatus.SDK_VERSION_MISMATCH:
                entry["error"] = info.error_message
                entry["min_version"] = info.min_version
                version_mismatches.append(entry)
            else:
                entry["error"] = info.error_message
                import_errors.append(entry)

        return {
            "summary": {
                "total_registered": len(self._adapters),
                "loaded": len(loaded),
                "sdk_missing": len(sdk_missing),
                "version_mismatches": len(version_mismatches),
                "import_errors": len(import_errors),
            },
            "loaded_adapters": sorted(loaded, key=lambda x: -x.get("priority", 0)),
            "sdk_missing": sdk_missing,
            "version_mismatches": version_mismatches,
            "import_errors": import_errors,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def log_startup_status(self, level: int = logging.INFO) -> None:
        """
        Log adapter status at startup.

        Args:
            level: Logging level to use
        """
        if self._startup_logged:
            return

        report = self.get_status_report()
        summary = report["summary"]

        # Main status line
        logger.log(
            level,
            f"Adapter Registry: {summary['loaded']}/{summary['total_registered']} adapters loaded"
        )

        # List loaded adapters
        if report["loaded_adapters"]:
            loaded_names = [a["name"] for a in report["loaded_adapters"]]
            logger.log(level, f"  Available: {', '.join(loaded_names)}")

        # Warn about missing SDKs
        if report["sdk_missing"]:
            missing = [f"{a['name']} ({a['sdk']})" for a in report["sdk_missing"]]
            logger.warning(f"  SDK missing: {', '.join(missing)}")
            logger.warning("  Install with: pip install <sdk_name>")

        # Warn about version mismatches
        if report["version_mismatches"]:
            for a in report["version_mismatches"]:
                logger.warning(
                    f"  Version mismatch: {a['name']} - {a['error']}"
                )

        # Error about import failures
        if report["import_errors"]:
            for a in report["import_errors"]:
                logger.error(f"  Import error: {a['name']} - {a.get('error', 'Unknown')}")

        self._startup_logged = True

    def add_load_callback(
        self,
        callback: Callable[[str, AdapterInfo], None],
    ) -> None:
        """
        Add a callback to be called when adapters are loaded.

        Args:
            callback: Function(adapter_name, adapter_info)
        """
        self._load_callbacks.append(callback)

    async def shutdown_all(self) -> Dict[str, bool]:
        """
        Shutdown all cached adapter instances.

        Returns:
            Dict mapping adapter names to shutdown success status
        """
        results = {}

        for name, adapter in self._adapter_instances.items():
            try:
                if hasattr(adapter, "shutdown"):
                    await adapter.shutdown()
                results[name] = True
            except Exception as e:
                logger.error(f"Error shutting down adapter '{name}': {e}")
                results[name] = False

        self._adapter_instances.clear()
        return results

    def clear(self) -> None:
        """Clear all registered adapters and cached instances."""
        self._adapters.clear()
        self._adapter_instances.clear()
        self._startup_logged = False

    async def execute_with_rate_limit(
        self,
        adapter_name: str,
        operation: str,
        wait: bool = True,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Any:
        """
        Execute an adapter operation with rate limiting.

        This is a convenience method that combines:
        1. Getting the adapter from registry
        2. Acquiring rate limit permission
        3. Executing the operation
        4. Handling rate limit responses (429s)

        Args:
            adapter_name: Name of the adapter
            operation: Operation to execute
            wait: If True, wait for rate limit. If False, raise exception.
            timeout: Maximum time to wait for rate limit (if wait=True)
            **kwargs: Arguments passed to adapter.execute()

        Returns:
            Result from adapter.execute()

        Raises:
            ValueError: If adapter not found
            RateLimitExceeded: If rate limit exceeded and wait=False
        """
        # Import rate limiter
        try:
            from core.rate_limiter import get_rate_limiter, RateLimitExceeded
        except ImportError:
            # Rate limiter not available, execute without rate limiting
            adapter_cls = self.get_adapter(adapter_name)
            if not adapter_cls:
                raise ValueError(f"Adapter '{adapter_name}' not found or not loaded")

            if adapter_name not in self._adapter_instances:
                adapter = adapter_cls()
                await adapter.initialize({})
                self._adapter_instances[adapter_name] = adapter

            adapter = self._adapter_instances[adapter_name]
            return await adapter.execute(operation, **kwargs)

        limiter = get_rate_limiter()

        # Acquire rate limit permission
        acquired = await limiter.acquire(adapter_name, wait=wait, timeout=timeout)
        if not acquired:
            raise RateLimitExceeded(
                f"Rate limit exceeded for adapter '{adapter_name}'"
            )

        # Get or create adapter instance
        adapter_cls = self.get_adapter(adapter_name)
        if not adapter_cls:
            raise ValueError(f"Adapter '{adapter_name}' not found or not loaded")

        if adapter_name not in self._adapter_instances:
            adapter = adapter_cls()
            await adapter.initialize({})
            self._adapter_instances[adapter_name] = adapter

        adapter = self._adapter_instances[adapter_name]

        try:
            result = await adapter.execute(operation, **kwargs)

            # Check for rate limit response and update limiter
            if hasattr(result, "metadata") and result.metadata:
                if result.metadata.get("status_code") == 429:
                    retry_after = result.metadata.get("retry_after")
                    limiter.on_rate_limit(adapter_name, retry_after)

            # Parse rate limit headers if available
            if hasattr(result, "headers") and result.headers:
                limiter.update_from_headers(adapter_name, result.headers)

            return result

        except Exception as e:
            # Check for 429 response
            if hasattr(e, "status_code") and getattr(e, "status_code") == 429:
                retry_after = None
                if hasattr(e, "headers"):
                    headers = getattr(e, "headers", {})
                    retry_after_str = headers.get("Retry-After")
                    if retry_after_str and str(retry_after_str).isdigit():
                        retry_after = float(retry_after_str)
                limiter.on_rate_limit(adapter_name, retry_after)
            raise


# Global registry instance
_registry: Optional[AdapterRegistry] = None


def get_registry() -> AdapterRegistry:
    """
    Get the global adapter registry instance.

    Returns:
        The singleton AdapterRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = AdapterRegistry()
    return _registry


def register_adapter(
    name: str,
    adapter_class: Optional[Type] = None,
    sdk_name: Optional[str] = None,
    min_version: Optional[str] = None,
    layer: Optional[str] = None,
    priority: int = 0,
    features: Optional[List[str]] = None,
) -> None:
    """
    Register an adapter with the global registry.

    Convenience function that wraps get_registry().register_adapter().
    """
    get_registry().register_adapter(
        name=name,
        adapter_class=adapter_class,
        sdk_name=sdk_name,
        min_version=min_version,
        layer=layer,
        priority=priority,
        features=features,
    )


# Export public API
__all__ = [
    "AdapterRegistry",
    "AdapterInfo",
    "AdapterLoadStatus",
    "HealthCheckResult",
    "get_registry",
    "register_adapter",
]
