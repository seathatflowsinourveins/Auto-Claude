"""
Production-Ready Ralph Loop Integration

This module provides a production-ready wrapper for the Ralph Loop
that integrates all production enhancements:
- Structured logging with correlation IDs
- Prometheus metrics
- Graceful shutdown
- Rate limiting
- State checkpointing
- Enhanced V11 features

Usage:
    from core.ralph.production_loop import ProductionRalphLoop

    loop = ProductionRalphLoop(
        task="Improve the code quality",
        max_iterations=100
    )

    # Run with production features
    result = await loop.run()

    # Get metrics
    print(loop.metrics.export_prometheus())
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .production import (
    AdaptiveRAGManager,
    ChainOfDraftManager,
    CheckpointManager,
    CheckpointMetadata,
    HypothesisTracker,
    ProductionConfig,
    RalphLogger,
    RalphMetrics,
    RateLimitConfig,
    RateLimiter,
    RewardHackingDetector,
    RewardHackingSignal,
    ShutdownHandler,
    SpeculativeDecodingManager,
    configure_production_logging,
    correlation_context,
    get_correlation_id,
    get_metrics,
    get_rate_limiter,
    get_shutdown_handler,
    initialize_production,
    rate_limited,
)

# Import the base RalphLoop
try:
    from ..ralph_loop import RalphLoop, LoopState, IterationResult
except ImportError:
    # Fallback for testing
    RalphLoop = None
    LoopState = None
    IterationResult = None


@dataclass
class ProductionIterationResult:
    """Enhanced iteration result with production metadata."""
    iteration: int
    started_at: str
    completed_at: str
    latency_ms: float
    fitness_score: float
    improvements: List[str]
    artifacts_created: List[str]
    errors: List[str]
    metadata: Dict[str, Any]

    # Production enhancements
    correlation_id: str = ""
    strategy_used: str = ""
    api_calls_made: int = 0
    rate_limited: bool = False
    checkpoint_saved: bool = False
    hypotheses_generated: int = 0
    hypotheses_verified: int = 0
    rag_retrievals: int = 0
    token_compression_ratio: float = 0.0
    reward_hacking_signals: List[str] = field(default_factory=list)


class ProductionRalphLoop:
    """
    Production-ready Ralph Loop with all enhancements.

    This class wraps the base RalphLoop and adds:
    - Correlation ID tracking for distributed tracing
    - Prometheus-compatible metrics
    - Graceful shutdown handling
    - Rate limiting for API calls
    - Automatic checkpointing with crash recovery
    - Enhanced V11 feature implementations
    """

    def __init__(
        self,
        task: str,
        max_iterations: int = 100,
        config: Optional[ProductionConfig] = None,
        checkpoint_dir: Optional[Path] = None
    ):
        self.task = task
        self.max_iterations = max_iterations
        self.config = config or ProductionConfig()

        if checkpoint_dir:
            self.config.checkpoint_dir = checkpoint_dir

        # Initialize production components
        components = initialize_production(self.config)

        self.logger: RalphLogger = components["logger"]
        self.metrics: RalphMetrics = components["metrics"]
        self.rate_limiter: RateLimiter = components["rate_limiter"]
        self.shutdown_handler: ShutdownHandler = components["shutdown_handler"]
        self.checkpoint_manager: CheckpointManager = components["checkpoint_manager"]
        self.speculative_manager: SpeculativeDecodingManager = components["speculative_manager"]
        self.cod_manager: ChainOfDraftManager = components["cod_manager"]
        self.rag_manager: AdaptiveRAGManager = components["rag_manager"]
        self.reward_detector: RewardHackingDetector = components["reward_detector"]

        # Initialize base loop if available
        self._base_loop: Optional[RalphLoop] = None
        if RalphLoop:
            self._base_loop = RalphLoop(
                task=task,
                max_iterations=max_iterations,
                checkpoint_dir=self.config.checkpoint_dir
            )

        # Internal state
        self._current_iteration = 0
        self._current_correlation_id: Optional[str] = None
        self._is_running = False
        self._state: Optional[LoopState] = None
        self._iteration_history: List[ProductionIterationResult] = []

        # Register cleanup on shutdown
        self.shutdown_handler.register_cleanup(self._emergency_checkpoint)

        # Track active loops
        self.metrics.set_gauge("active_loops", 1, {"loop_id": self.loop_id})

    @property
    def loop_id(self) -> str:
        """Get the loop ID."""
        if self._base_loop:
            return self._base_loop.loop_id
        return f"prod_loop_{id(self)}"

    @property
    def state(self) -> Optional[LoopState]:
        """Get the current loop state."""
        if self._base_loop:
            return self._base_loop.state
        return self._state

    def _emergency_checkpoint(self) -> None:
        """Save emergency checkpoint on shutdown."""
        if self._is_running and self.state:
            self.logger.info("Saving emergency checkpoint on shutdown")
            try:
                self._save_checkpoint(emergency=True)
            except Exception as e:
                self.logger.error(f"Emergency checkpoint failed: {e}")

    def _save_checkpoint(self, emergency: bool = False) -> Optional[CheckpointMetadata]:
        """Save a checkpoint."""
        if not self.state:
            return None

        state_dict = self.state.to_dict() if hasattr(self.state, "to_dict") else {}

        metadata = self.checkpoint_manager.save_checkpoint(
            loop_id=self.loop_id,
            state_dict=state_dict,
            iteration=self._current_iteration,
            fitness=self.state.best_fitness if self.state else 0.0,
            status="emergency" if emergency else self.state.status if self.state else "unknown"
        )

        self.metrics.inc_counter("checkpoints_saved_total", labels={"emergency": str(emergency)})
        return metadata

    async def _rate_limited_api_call(
        self,
        api_func: Callable,
        key: str = "default",
        *args,
        **kwargs
    ) -> Any:
        """Execute an API call with rate limiting."""
        await self.rate_limiter.acquire(key)

        start_time = time.perf_counter()
        try:
            result = await api_func(*args, **kwargs)
            duration = time.perf_counter() - start_time

            self.metrics.inc_counter("api_calls_total", labels={"key": key, "status": "success"})
            self.metrics.observe_histogram("api_call_latency_seconds", duration, labels={"key": key})

            return result
        except Exception as e:
            duration = time.perf_counter() - start_time

            self.metrics.inc_counter("api_calls_total", labels={"key": key, "status": "error"})
            self.metrics.observe_histogram("api_call_latency_seconds", duration, labels={"key": key})

            raise

    async def _run_speculative_iteration(
        self,
        context: str,
        generator_func: Callable[[str], Any],
        verifier_func: Callable[[str], Any]
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Run a speculative decoding iteration with production tracking.

        Returns (best_result, statistics)
        """
        # Generate hypotheses
        hypotheses = await self.speculative_manager.generate_hypotheses(
            context,
            generator_func
        )

        if not hypotheses:
            return None, {"status": "no_hypotheses"}

        # Verify hypotheses
        verified = await self.speculative_manager.verify_batch(hypotheses, verifier_func)

        # Get best result
        best = self.speculative_manager.get_best_hypothesis(verified)

        stats = self.speculative_manager.get_statistics()
        stats["best_confidence"] = best.confidence if best else 0.0
        stats["best_hypothesis_id"] = best.hypothesis_id if best else None

        return (best.content if best else None), stats

    async def _run_compressed_reasoning(
        self,
        full_reasoning: str,
        compressor_func: Callable[[str], Any]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Run Chain-of-Draft compression with tracking."""
        return await self.cod_manager.compress_reasoning(full_reasoning, compressor_func)

    async def _adaptive_retrieve(
        self,
        query: str,
        retriever_func: Callable[[str], Any],
        confidence: float,
        novelty: float
    ) -> Dict[str, Any]:
        """Run adaptive RAG retrieval."""
        return await self.rag_manager.perform_retrieval(
            query, retriever_func, confidence, novelty
        )

    def _check_reward_hacking(
        self,
        true_reward: float,
        proxy_reward: float,
        solution: str,
        previous_solution: str,
        improvement: float
    ) -> List[RewardHackingSignal]:
        """Check for reward hacking signals."""
        signals = []

        # Record rewards for trend analysis
        self.reward_detector.record_rewards(true_reward, proxy_reward)

        # Check for proxy divergence
        divergence_signal = self.reward_detector.check_proxy_divergence(true_reward, proxy_reward)
        if divergence_signal:
            signals.append(divergence_signal)

        # Check for suspicious improvement
        suspicious_signal = self.reward_detector.check_suspicious_improvement(improvement)
        if suspicious_signal:
            signals.append(suspicious_signal)

        # Check for specification gaming
        gaming_signal = self.reward_detector.check_specification_gaming(
            solution, previous_solution, improvement
        )
        if gaming_signal:
            signals.append(gaming_signal)

        return signals

    async def run_iteration(self) -> ProductionIterationResult:
        """
        Run a single production-enhanced iteration.

        Returns enhanced iteration result with all production metadata.
        """
        with correlation_context() as correlation_id:
            self._current_correlation_id = correlation_id
            self.logger.set_context(
                loop_id=self.loop_id,
                iteration=self._current_iteration,
                strategy=self.state.current_strategy if self.state else "unknown"
            )

            iteration_start = time.perf_counter()
            started_at = datetime.now(timezone.utc).isoformat()

            improvements = []
            artifacts = []
            errors = []
            api_calls = 0
            was_rate_limited = False
            checkpoint_saved = False
            hyp_generated = 0
            hyp_verified = 0
            rag_retrievals = 0
            compression_ratio = 0.0
            hacking_signals: List[str] = []

            try:
                # Check for shutdown
                if self.shutdown_handler.is_shutting_down:
                    self.logger.info("Shutdown requested, stopping iteration")
                    return self._create_result(
                        started_at, ["Shutdown requested"], [], ["iteration_cancelled"],
                        0, 0.0, correlation_id
                    )

                # Run the base iteration if available
                if self._base_loop:
                    with self.metrics.timer("iteration_latency_seconds"):
                        base_result = await self._base_loop.run_iteration()

                    if isinstance(base_result, dict):
                        improvements = base_result.get("improvements", [])
                        artifacts = base_result.get("artifacts_created", [])
                        errors = base_result.get("errors", [])
                    elif hasattr(base_result, "improvements"):
                        improvements = base_result.improvements
                        artifacts = base_result.artifacts_created
                        errors = base_result.errors

                # Update metrics
                self.metrics.inc_counter("iterations_total")
                if improvements:
                    self.metrics.inc_counter("improvements_total")
                if errors:
                    self.metrics.inc_counter("failures_total")

                self.metrics.set_gauge("current_iteration", self._current_iteration)
                if self.state:
                    self.metrics.set_gauge("current_fitness", self.state.best_fitness)

                # Update speculative decoding metrics
                spec_stats = self.speculative_manager.get_statistics()
                hyp_generated = spec_stats["total_generated"]
                hyp_verified = spec_stats["total_verified"]
                self.metrics.set_gauge("speculation_acceptance_rate", spec_stats["acceptance_rate"])

                # Update RAG metrics
                rag_stats = self.rag_manager.get_statistics()
                rag_retrievals = rag_stats["retrievals_performed"]

                # Update compression metrics
                compression_ratio = self.cod_manager.get_overall_compression()
                self.metrics.set_gauge("token_compression_ratio", compression_ratio)

                # Checkpoint if needed
                if self._current_iteration % self.config.auto_save_interval == 0:
                    self._save_checkpoint()
                    checkpoint_saved = True

                self._current_iteration += 1

            except asyncio.CancelledError:
                self.logger.warning("Iteration cancelled")
                errors.append("iteration_cancelled")
                raise

            except Exception as e:
                self.logger.error(f"Iteration failed: {e}")
                errors.append(str(e))
                self.metrics.inc_counter("failures_total")

            latency_ms = (time.perf_counter() - iteration_start) * 1000

            result = ProductionIterationResult(
                iteration=self._current_iteration - 1,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc).isoformat(),
                latency_ms=latency_ms,
                fitness_score=self.state.best_fitness if self.state else 0.0,
                improvements=improvements,
                artifacts_created=artifacts,
                errors=errors,
                metadata={
                    "correlation_id": correlation_id,
                    "spec_stats": self.speculative_manager.get_statistics(),
                    "rag_stats": self.rag_manager.get_statistics(),
                    "reward_stats": self.reward_detector.get_statistics()
                },
                correlation_id=correlation_id,
                strategy_used=self.state.current_strategy if self.state else "",
                api_calls_made=api_calls,
                rate_limited=was_rate_limited,
                checkpoint_saved=checkpoint_saved,
                hypotheses_generated=hyp_generated,
                hypotheses_verified=hyp_verified,
                rag_retrievals=rag_retrievals,
                token_compression_ratio=compression_ratio,
                reward_hacking_signals=hacking_signals
            )

            self._iteration_history.append(result)
            return result

    def _create_result(
        self,
        started_at: str,
        improvements: List[str],
        artifacts: List[str],
        errors: List[str],
        fitness: float,
        latency_ms: float,
        correlation_id: str
    ) -> ProductionIterationResult:
        """Create a production iteration result."""
        return ProductionIterationResult(
            iteration=self._current_iteration,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            latency_ms=latency_ms,
            fitness_score=fitness,
            improvements=improvements,
            artifacts_created=artifacts,
            errors=errors,
            metadata={},
            correlation_id=correlation_id
        )

    async def run(
        self,
        initial_solution: Any = None,
        resume_from: Optional[str] = None
    ) -> LoopState:
        """
        Run the production Ralph Loop.

        Args:
            initial_solution: Starting solution to improve
            resume_from: Checkpoint ID or loop ID to resume from
        """
        with correlation_context() as correlation_id:
            self.logger.info(
                f"Starting production Ralph Loop",
                task=self.task,
                max_iterations=self.max_iterations,
                correlation_id=correlation_id
            )

            self._is_running = True

            try:
                # Try to resume from checkpoint
                if resume_from:
                    checkpoint = self.checkpoint_manager.load_checkpoint(resume_from)
                    if checkpoint:
                        metadata, state_dict = checkpoint
                        self._current_iteration = metadata.iteration
                        self.logger.info(
                            f"Resumed from checkpoint",
                            checkpoint_id=metadata.checkpoint_id,
                            iteration=metadata.iteration
                        )

                # Run base loop if available
                if self._base_loop:
                    self._state = await self._base_loop.run(initial_solution, resume_from)
                else:
                    # Minimal standalone execution
                    while self._current_iteration < self.max_iterations:
                        if self.shutdown_handler.is_shutting_down:
                            break

                        await self.run_iteration()

                        # Log progress
                        if self._current_iteration % 10 == 0:
                            self.logger.info(
                                f"Progress: iteration {self._current_iteration}/{self.max_iterations}"
                            )

            except asyncio.CancelledError:
                self.logger.warning("Loop cancelled")
                self._save_checkpoint(emergency=True)
                raise

            except Exception as e:
                self.logger.error(f"Loop failed: {e}")
                self._save_checkpoint(emergency=True)
                raise

            finally:
                self._is_running = False
                self.metrics.set_gauge("active_loops", 0, {"loop_id": self.loop_id})

            # Final checkpoint
            self._save_checkpoint()

            self.logger.info(
                f"Loop completed",
                iterations=self._current_iteration,
                final_fitness=self.state.best_fitness if self.state else 0.0
            )

            return self.state

    def pause(self) -> None:
        """Pause the loop and save checkpoint."""
        self.logger.info("Pausing loop")
        self._save_checkpoint()
        if self._base_loop:
            self._base_loop.pause()

    def get_progress(self) -> Dict[str, Any]:
        """Get detailed progress with production metrics."""
        base_progress = {}
        if self._base_loop:
            base_progress = self._base_loop.get_progress()

        return {
            **base_progress,
            "production": {
                "correlation_id": self._current_correlation_id,
                "is_running": self._is_running,
                "checkpoints_saved": self.checkpoint_manager.list_checkpoints(self.loop_id),
                "speculation_stats": self.speculative_manager.get_statistics(),
                "rag_stats": self.rag_manager.get_statistics(),
                "compression_ratio": self.cod_manager.get_overall_compression(),
                "reward_hacking_stats": self.reward_detector.get_statistics(),
                "metrics_snapshot": self.metrics.get_snapshot()
            }
        }

    def get_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        return self.metrics.export_prometheus()

    def get_iteration_history(self) -> List[ProductionIterationResult]:
        """Get the iteration history."""
        return self._iteration_history


# Factory function for easy instantiation
def create_production_loop(
    task: str,
    max_iterations: int = 100,
    **config_kwargs
) -> ProductionRalphLoop:
    """
    Create a production Ralph Loop with custom configuration.

    Args:
        task: The task to optimize
        max_iterations: Maximum iterations to run
        **config_kwargs: Additional ProductionConfig parameters

    Returns:
        Configured ProductionRalphLoop instance
    """
    config = ProductionConfig(**config_kwargs)
    return ProductionRalphLoop(task, max_iterations, config)
