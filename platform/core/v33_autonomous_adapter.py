"""
V33 Autonomous Adapter - Integration for UnifiedOrchestratorFacade
===================================================================

Wraps V33ProductionLoop for integration with UNLEASH orchestration architecture.
Provides the OrchestratorProtocol interface for unified routing.

Features:
- Direction monitoring with chi-squared drift detection
- GOAP correction planning (A* pathfinding)
- Factory Signals compound learning
- Circuit breaker production reliability
- Letta cross-session memory

Version: V33 (2026-01-31)
"""

import os
import sys
import asyncio
import logging
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Add integrations path
INTEGRATIONS_PATH = os.path.expanduser("~/.claude/integrations")
if INTEGRATIONS_PATH not in sys.path:
    sys.path.insert(0, INTEGRATIONS_PATH)


@dataclass
class V33AdapterConfig:
    """Configuration for V33 autonomous adapter."""

    # Loop settings
    max_iterations: int = 100
    check_interval: int = 5
    completion_promise: str = "<promise>COMPLETE</promise>"

    # Direction monitoring
    drift_window_size: int = 10
    drift_threshold_low: float = 0.05
    drift_threshold_medium: float = 0.01
    drift_threshold_high: float = 0.005
    drift_threshold_critical: float = 0.001

    # Letta settings
    letta_enabled: bool = True
    letta_agent_id: Optional[str] = None

    # Circuit breaker
    failure_threshold: int = 5
    recovery_timeout: float = 30.0

    # Executor settings
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 16000

    # Performance
    parallel_execution: bool = True
    max_parallel_tasks: int = 6


class V33AutonomousAdapter:
    """
    V33 Autonomous Adapter for UnifiedOrchestratorFacade.

    Implements OrchestratorProtocol interface for unified routing.
    Wraps V33ProductionLoop for direction-monitored autonomous execution.

    Usage in UnifiedOrchestratorFacade:
        adapter = V33AutonomousAdapter()
        result = await adapter.execute("Goal description", max_iterations=50)
    """

    def __init__(self, config: Optional[V33AdapterConfig] = None):
        self.config = config or V33AdapterConfig()
        self._v33_loop = None
        self._initialized = False
        self._execution_count = 0
        self._total_iterations = 0
        self._drift_corrections = 0

    async def _lazy_init(self) -> None:
        """Lazy initialization of V33 components."""
        if self._initialized:
            return

        try:
            # Import V33 production loop
            from v33_production_loop import (
                V33ProductionLoop,
                V33LoopConfig,
                create_v33_loop,
                create_monitored_loop,
                create_learning_loop,
            )

            self._v33_module = {
                'V33ProductionLoop': V33ProductionLoop,
                'V33LoopConfig': V33LoopConfig,
                'create_v33_loop': create_v33_loop,
                'create_monitored_loop': create_monitored_loop,
                'create_learning_loop': create_learning_loop,
            }

            self._initialized = True
            logger.info("V33 autonomous adapter initialized")

        except ImportError as e:
            logger.warning(f"V33ProductionLoop not available: {e}")
            self._v33_module = None
            self._initialized = True  # Mark as initialized to avoid retry

    async def execute(
        self,
        goal: str,
        max_iterations: Optional[int] = None,
        letta_agent_id: Optional[str] = None,
        executor: Optional[Any] = None,
        enable_direction_monitoring: bool = True,
        enable_factory_signals: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute autonomous loop with V33 direction monitoring.

        Args:
            goal: Goal/objective to achieve
            max_iterations: Override max iterations
            letta_agent_id: Letta agent for cross-session memory
            executor: Custom executor (or uses default)
            enable_direction_monitoring: Enable chi-squared drift detection
            enable_factory_signals: Enable compound learning
            **kwargs: Additional arguments

        Returns:
            Dict with execution results, metrics, and drift corrections
        """
        await self._lazy_init()

        if not self._v33_module:
            return {
                "success": False,
                "error": "V33ProductionLoop not available",
                "status": "unavailable",
            }

        start_time = datetime.now(timezone.utc)
        self._execution_count += 1

        try:
            # Create loop config
            V33LoopConfig = self._v33_module['V33LoopConfig']

            loop_config = V33LoopConfig(
                max_iterations=max_iterations or self.config.max_iterations,
                completion_promise=self.config.completion_promise,
                check_interval=self.config.check_interval,
                drift_window_size=self.config.drift_window_size,
                enable_direction_monitoring=enable_direction_monitoring,
                enable_factory_signals=enable_factory_signals,
                letta_agent_id=letta_agent_id or self.config.letta_agent_id,
            )

            # Choose factory function based on settings
            if enable_direction_monitoring and letta_agent_id:
                create_fn = self._v33_module['create_learning_loop']
                loop = create_fn(
                    goal=goal,
                    executor=executor,
                    letta_agent_id=letta_agent_id or self.config.letta_agent_id,
                    max_iterations=max_iterations or self.config.max_iterations,
                )
            elif enable_direction_monitoring:
                create_fn = self._v33_module['create_monitored_loop']
                loop = create_fn(
                    goal=goal,
                    executor=executor,
                    max_iterations=max_iterations or self.config.max_iterations,
                )
            else:
                create_fn = self._v33_module['create_v33_loop']
                loop = create_fn(
                    goal=goal,
                    executor=executor,
                    config=loop_config,
                )

            # Run the loop
            result = await loop.run()

            # Update stats
            self._total_iterations += result.get('iterations', 0)
            self._drift_corrections += result.get('drift_corrections', 0)

            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            return {
                "success": result.get('status') == 'complete',
                "status": result.get('status'),
                "iterations": result.get('iterations', 0),
                "exit_reason": result.get('exit_reason'),
                "drift_corrections": result.get('drift_corrections', 0),
                "duration_ms": duration_ms,
                "direction_metrics": result.get('direction_metrics', {}),
                "factory_signals": result.get('factory_signals', {}),
                "letta_passages": result.get('letta_passages', []),
                "result": result.get('final_output'),
                "adapter": "v33_autonomous",
            }

        except Exception as e:
            logger.error(f"V33 execution failed: {e}")
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            return {
                "success": False,
                "error": str(e),
                "status": "error",
                "duration_ms": duration_ms,
                "adapter": "v33_autonomous",
            }

    async def run(self, task: str, **kwargs) -> Dict[str, Any]:
        """Alias for execute() to match OrchestratorProtocol."""
        return await self.execute(task, **kwargs)

    async def process(self, task: str, **kwargs) -> Dict[str, Any]:
        """Alias for execute() to match OrchestratorProtocol."""
        return await self.execute(task, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            "executions": self._execution_count,
            "total_iterations": self._total_iterations,
            "drift_corrections": self._drift_corrections,
            "initialized": self._initialized,
            "v33_available": self._v33_module is not None,
        }


# Convenience factory functions

def create_v33_adapter(
    letta_agent_id: Optional[str] = None,
    max_iterations: int = 100,
    **kwargs
) -> V33AutonomousAdapter:
    """Create V33 adapter with default settings."""
    config = V33AdapterConfig(
        max_iterations=max_iterations,
        letta_agent_id=letta_agent_id,
        **kwargs
    )
    return V33AutonomousAdapter(config)


def create_monitored_adapter(
    max_iterations: int = 50,
    check_interval: int = 3,
    **kwargs
) -> V33AutonomousAdapter:
    """Create V33 adapter optimized for direction monitoring."""
    config = V33AdapterConfig(
        max_iterations=max_iterations,
        check_interval=check_interval,
        **kwargs
    )
    return V33AutonomousAdapter(config)


def create_learning_adapter(
    letta_agent_id: str,
    max_iterations: int = 100,
    **kwargs
) -> V33AutonomousAdapter:
    """Create V33 adapter with Letta cross-session learning."""
    config = V33AdapterConfig(
        letta_agent_id=letta_agent_id,
        max_iterations=max_iterations,
        letta_enabled=True,
        **kwargs
    )
    return V33AutonomousAdapter(config)


# For direct testing
async def main():
    """Test V33 adapter."""
    adapter = create_v33_adapter()

    result = await adapter.execute(
        "Test V33 adapter execution",
        max_iterations=3,
        enable_direction_monitoring=True,
    )

    print(f"Result: {result}")
    print(f"Stats: {adapter.get_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
