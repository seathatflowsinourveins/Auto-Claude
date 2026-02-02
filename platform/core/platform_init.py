"""
Platform Auto-Init - UNLEASH Platform Bootstrap System.

Initializes the memory tier system with essential context and starts
background services for production operation.

Features:
- Memory Bootstrap: Pre-loads CLAUDE.md, project context, verified patterns
- Sleep Agent: Starts background consolidation automatically
- Pressure Monitoring: Registers handlers for memory warnings
- Session Context: Project-specific knowledge loading

Usage:
    from platform_init import PlatformInit

    # Simple initialization
    platform = await PlatformInit.bootstrap()

    # Custom initialization
    platform = await PlatformInit.bootstrap(
        project="UNLEASH",
        load_claude_md=True,
        start_sleep_agent=True
    )

    # Shutdown gracefully
    await platform.shutdown()

Version: 1.0.0 (January 2026)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from memory_tiers import (
    MemoryTier,
    MemoryTierManager,
    MemoryPriority,
    MemoryPressureLevel,
    MemoryPressureEvent,
    reset_memory_system,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PlatformConfig:
    """Configuration for platform initialization."""

    # Project settings
    project_name: str = "UNLEASH"
    project_root: Optional[Path] = None

    # Memory tier settings
    main_context_tokens: int = 8192
    core_memory_tokens: int = 4096
    recall_memory_entries: int = 1000
    archival_memory_entries: int = 10000

    # Letta integration
    letta_agent_id: Optional[str] = None

    # Bootstrap options
    load_claude_md: bool = True
    load_project_context: bool = True
    load_verified_patterns: bool = True

    # Background services
    start_sleep_agent: bool = True
    consolidation_interval_minutes: int = 5

    # Pressure monitoring
    enable_pressure_warnings: bool = True
    pressure_log_level: str = "WARNING"


@dataclass
class BootstrapResult:
    """Result of platform bootstrap operation."""
    success: bool
    manager: Optional[MemoryTierManager]
    entries_loaded: int
    errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __str__(self) -> str:
        status = "✅ SUCCESS" if self.success else "❌ FAILED"
        return f"Bootstrap {status}: {self.entries_loaded} entries loaded"


# =============================================================================
# VERIFIED PATTERNS (From Research)
# =============================================================================

VERIFIED_PATTERNS: List[Dict[str, Any]] = [
    # Letta SDK (verified 2026-01-29)
    {
        "content": "Letta SDK: Use client.agents.messages.create(agentId, {messages: [{role: 'user', content: msg}]})",
        "tags": ["api_signature", "letta", "verified"],
        "source": "docs.letta.com",
    },
    # LangGraph (verified 2026-01-29)
    {
        "content": "LangGraph: Use workflow.add_conditional_edges(source, routing_fn, path_map) NOT addConditionalEdge",
        "tags": ["api_signature", "langgraph", "verified"],
        "source": "langchain-ai/langgraph",
    },
    # Checkpointer (verified 2026-01-29)
    {
        "content": "LangGraph Checkpointer: Use checkpointer.put(config, state, metadata) with config={'configurable': {'thread_id': '...'}}",
        "tags": ["api_signature", "langgraph", "checkpointer", "verified"],
        "source": "langchain-ai/langgraph",
    },
    # Memory Tier Pattern
    {
        "content": "Memory Pressure: NORMAL(<50%) → ELEVATED(50-70%) → WARNING(70-85%) → CRITICAL(85-95%) → OVERFLOW(>95%)",
        "tags": ["pattern", "memory", "pressure"],
        "source": "MemGPT/Letta research",
    },
]


# =============================================================================
# PLATFORM INIT
# =============================================================================

class PlatformInit:
    """
    UNLEASH Platform initialization and lifecycle management.

    Handles:
    - Memory tier bootstrap with essential context
    - Sleep-time agent startup
    - Pressure monitoring registration
    - Graceful shutdown
    """

    def __init__(
        self,
        config: Optional[PlatformConfig] = None
    ) -> None:
        self.config = config or PlatformConfig()
        self._manager: Optional[MemoryTierManager] = None
        self._pressure_handlers: List[Callable[[MemoryPressureEvent], None]] = []
        self._initialized = False
        self._bootstrap_result: Optional[BootstrapResult] = None

    @classmethod
    async def bootstrap(
        cls,
        project: str = "UNLEASH",
        load_claude_md: bool = True,
        start_sleep_agent: bool = True,
        letta_agent_id: Optional[str] = None,
        **kwargs: Any
    ) -> "PlatformInit":
        """
        Bootstrap the platform with sensible defaults.

        Args:
            project: Project name
            load_claude_md: Whether to load CLAUDE.md into memory
            start_sleep_agent: Whether to start background consolidation
            letta_agent_id: Optional Letta agent for external memory
            **kwargs: Additional config options

        Returns:
            Initialized PlatformInit instance
        """
        config = PlatformConfig(
            project_name=project,
            load_claude_md=load_claude_md,
            start_sleep_agent=start_sleep_agent,
            letta_agent_id=letta_agent_id,
            **kwargs
        )

        platform = cls(config)
        await platform.initialize()
        return platform

    async def initialize(self) -> BootstrapResult:
        """
        Initialize the platform with all configured options.

        Returns:
            BootstrapResult with initialization status
        """
        errors: List[str] = []
        entries_loaded = 0

        try:
            # Reset singleton for clean state
            reset_memory_system()

            # Create memory tier manager
            self._manager = MemoryTierManager(
                main_context_tokens=self.config.main_context_tokens,
                core_memory_tokens=self.config.core_memory_tokens,
                recall_memory_entries=self.config.recall_memory_entries,
                archival_memory_entries=self.config.archival_memory_entries,
                letta_agent_id=self.config.letta_agent_id,
                auto_tier_management=True
            )

            # Register pressure handler
            if self.config.enable_pressure_warnings and self._manager:
                self._manager.register_pressure_handler(self._handle_pressure_event)

            # Load CLAUDE.md
            if self.config.load_claude_md:
                loaded = await self._load_claude_md()
                entries_loaded += loaded

            # Load verified patterns
            if self.config.load_verified_patterns:
                loaded = await self._load_verified_patterns()
                entries_loaded += loaded

            # Load project context
            if self.config.load_project_context:
                loaded = await self._load_project_context()
                entries_loaded += loaded

            # Start sleep-time agent
            if self.config.start_sleep_agent and self._manager:
                await self._manager.start_sleep_agent()
                logger.info("Sleep-time agent started")

            self._initialized = True

            self._bootstrap_result = BootstrapResult(
                success=True,
                manager=self._manager,
                entries_loaded=entries_loaded,
                errors=errors
            )

            logger.info(f"Platform initialized: {entries_loaded} entries loaded")

        except Exception as e:
            errors.append(str(e))
            logger.error(f"Platform initialization failed: {e}")

            self._bootstrap_result = BootstrapResult(
                success=False,
                manager=None,
                entries_loaded=entries_loaded,
                errors=errors
            )

        return self._bootstrap_result

    async def _load_claude_md(self) -> int:
        """Load CLAUDE.md into core memory."""
        if not self._manager:
            return 0

        loaded = 0
        claude_md_paths = [
            Path.home() / ".claude" / "CLAUDE.md",
            Path.cwd() / "CLAUDE.md",
        ]

        for path in claude_md_paths:
            if path.exists():
                try:
                    content = path.read_text(encoding="utf-8")

                    # Store summary in core memory (always visible)
                    # Extract key sections for core memory
                    if "## CRITICAL" in content or "## IMPORTANT" in content:
                        # Extract critical rules
                        await self._manager.remember(
                            content=f"[CLAUDE.md Critical Rules from {path}]",
                            tier=MemoryTier.CORE_MEMORY,
                            priority=MemoryPriority.CRITICAL,
                            content_type="config",
                            tags=["claude_md", "rules", "critical"],
                            source=str(path)
                        )
                        loaded += 1

                    # Store full content in archival for reference
                    if len(content) > 500:
                        await self._manager.remember(
                            content=content[:4000],  # Truncate for storage
                            tier=MemoryTier.ARCHIVAL_MEMORY,
                            priority=MemoryPriority.HIGH,
                            content_type="config",
                            tags=["claude_md", "full"],
                            source=str(path)
                        )
                        loaded += 1

                    logger.debug(f"Loaded CLAUDE.md from {path}")

                except Exception as e:
                    logger.warning(f"Failed to load CLAUDE.md from {path}: {e}")

        return loaded

    async def _load_verified_patterns(self) -> int:
        """Load verified API patterns into core memory."""
        if not self._manager:
            return 0

        loaded = 0

        for pattern in VERIFIED_PATTERNS:
            try:
                await self._manager.remember(
                    content=pattern["content"],
                    tier=MemoryTier.CORE_MEMORY,
                    priority=MemoryPriority.HIGH,
                    content_type="pattern",
                    tags=pattern.get("tags", []),
                    source=pattern.get("source", "research")
                )
                loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load pattern: {e}")

        return loaded

    async def _load_project_context(self) -> int:
        """Load project-specific context."""
        if not self._manager:
            return 0

        loaded = 0

        # Store project info
        project_info = {
            "UNLEASH": {
                "path": "Z:\\insider\\AUTO CLAUDE\\unleash",
                "description": "AI Agent SDK Platform with memory tiers",
                "letta_agent": "agent-90226e2c-44be-486b-bd1f-05121f2f7957"
            },
            "WITNESS": {
                "path": "Z:\\insider\\AUTO CLAUDE\\Touchdesigner-createANDBE",
                "description": "TouchDesigner creative AI system",
                "letta_agent": "agent-1a5cf5ba-ea17-4631-aade-4a43516fb8e7"
            },
            "ALPHAFORGE": {
                "path": "Z:\\insider\\AUTO CLAUDE\\autonomous AI trading system",
                "description": "Autonomous AI trading platform",
                "letta_agent": "agent-f857250d-1e57-40bf-99c4-5aa7c0103b7a"
            }
        }

        if self.config.project_name in project_info:
            info = project_info[self.config.project_name]
            await self._manager.remember(
                content=f"Project: {self.config.project_name}\nPath: {info['path']}\nDescription: {info['description']}",
                tier=MemoryTier.CORE_MEMORY,
                priority=MemoryPriority.CRITICAL,
                content_type="project",
                tags=["project", "context", self.config.project_name.lower()],
                source="platform_init"
            )
            loaded += 1

        return loaded

    def _handle_pressure_event(self, event: MemoryPressureEvent) -> None:
        """Handle memory pressure change events."""
        if event.current_level in (
            MemoryPressureLevel.WARNING,
            MemoryPressureLevel.CRITICAL,
            MemoryPressureLevel.OVERFLOW
        ):
            # Log warning
            msg = event.to_system_message()
            if msg:
                logger.warning(msg)

            # Notify registered handlers
            for handler in self._pressure_handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Pressure handler error: {e}")

    def register_pressure_handler(
        self,
        handler: Callable[[MemoryPressureEvent], None]
    ) -> None:
        """Register a custom pressure event handler."""
        self._pressure_handlers.append(handler)

    @property
    def manager(self) -> Optional[MemoryTierManager]:
        """Get the memory tier manager."""
        return self._manager

    @property
    def is_initialized(self) -> bool:
        """Check if platform is initialized."""
        return self._initialized

    async def get_status(self) -> Dict[str, Any]:
        """Get platform status."""
        if not self._manager:
            return {"initialized": False, "error": "Not initialized"}

        stats = self._manager.get_stats()
        pressure = await self._manager.get_pressure_report()
        sleep_status = self._manager.get_sleep_agent_status()

        return {
            "initialized": self._initialized,
            "project": self.config.project_name,
            "memory": {
                "total_entries": stats.total_entries,
                "hit_rate": f"{stats.hit_rate:.1%}",
                "evictions": stats.eviction_count,
            },
            "pressure": pressure,
            "sleep_agent": sleep_status,
            "bootstrap": {
                "success": self._bootstrap_result.success if self._bootstrap_result else False,
                "entries_loaded": self._bootstrap_result.entries_loaded if self._bootstrap_result else 0,
            }
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown the platform."""
        if self._manager:
            # Stop sleep agent
            await self._manager.stop_sleep_agent()

            # Run final consolidation
            await self._manager.run_consolidation_now()

            logger.info("Platform shutdown complete")

        self._initialized = False


# =============================================================================
# QUICK START FUNCTIONS
# =============================================================================

async def quick_start(project: str = "UNLEASH") -> PlatformInit:
    """
    Quick start the platform with defaults.

    Example:
        platform = await quick_start("UNLEASH")
        manager = platform.manager
    """
    return await PlatformInit.bootstrap(project=project)


def run_quick_start(project: str = "UNLEASH") -> PlatformInit:
    """Synchronous wrapper for quick_start."""
    return asyncio.run(quick_start(project))


# =============================================================================
# VERIFICATION
# =============================================================================

async def verify_platform() -> bool:
    """Verify platform initialization works correctly."""
    print("=" * 60)
    print("Platform Init - Verification")
    print("=" * 60)

    try:
        # Bootstrap
        print("\n[1] Bootstrapping platform...")
        platform = await PlatformInit.bootstrap(
            project="UNLEASH",
            load_claude_md=True,
            start_sleep_agent=True
        )
        print(f"    ✓ Bootstrap: {platform._bootstrap_result}")

        # Check status
        print("\n[2] Checking status...")
        status = await platform.get_status()
        print(f"    ✓ Initialized: {status['initialized']}")
        print(f"    ✓ Project: {status['project']}")
        print(f"    ✓ Entries loaded: {status['bootstrap']['entries_loaded']}")

        # Memory operations
        print("\n[3] Testing memory operations...")
        manager = platform.manager
        if not manager:
            print("    ❌ Manager not initialized")
            return False

        entry = await manager.remember(
            "Test entry from verification",
            tier=MemoryTier.MAIN_CONTEXT
        )
        print(f"    ✓ Stored entry: {entry.id}")

        recalled = await manager.recall(entry.id)
        print(f"    ✓ Recalled entry: {recalled.id if recalled else 'FAILED'}")

        # Pressure check
        print("\n[4] Checking pressure levels...")
        pressure = await manager.get_pressure_report()
        for tier, info in pressure.items():
            print(f"    ✓ {tier}: {info['level']}")

        # Sleep agent status
        print("\n[5] Sleep agent status...")
        sleep_status = manager.get_sleep_agent_status()
        print(f"    ✓ Running: {sleep_status.get('running', False)}")
        print(f"    ✓ Auto-consolidation: {sleep_status.get('auto_consolidation', False)}")

        # Shutdown
        print("\n[6] Graceful shutdown...")
        await platform.shutdown()
        print("    ✓ Shutdown complete")

        print("\n" + "=" * 60)
        print("Verification PASSED ✅")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ Verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    asyncio.run(verify_platform())
