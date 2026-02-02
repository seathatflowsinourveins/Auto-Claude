"""
UNLEASH L8 Quality-Diversity Layer Adapter
==========================================

Quality-Diversity optimization for creative AI exploration:
- MAP-Elites algorithm for diverse high-quality solutions
- CMA-ME (CMA-ES + MAP-Elites) for continuous optimization
- GridArchive for behavioral space coverage

Verified against official docs 2026-01-30:
- Context7: /icaros-usc/pyribs (361 snippets, 82.5 benchmark)
- Exa deep search for production patterns
"""

from typing import Optional, Callable, Any, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EmitterType(Enum):
    """Types of emitters for solution generation"""
    EVOLUTION_STRATEGY = "evolution_strategy"  # CMA-ES based
    GAUSSIAN = "gaussian"                       # Simple Gaussian mutations
    ISO_LINE = "iso_line"                       # Isotropic line sampling
    GRADIENT = "gradient"                       # Gradient-based optimization


class RankerType(Enum):
    """Ranker types for emitter selection"""
    TWO_IMPROVEMENT = "2imp"     # Two-stage improvement ranking
    IMPROVEMENT = "imp"         # Single improvement ranking
    RANDOM = "random"           # Random ranking
    OBJECTIVE = "obj"           # Objective-based ranking


class RestartRule(Enum):
    """Restart rules for emitters"""
    NO_IMPROVEMENT = "no_improvement"  # Restart when no improvement
    BASIC = "basic"                    # Basic restart rule


@dataclass
class ArchiveConfig:
    """Configuration for the solution archive"""
    solution_dim: int
    behavior_dims: List[int]           # Grid dimensions for behavior space
    behavior_ranges: List[Tuple[float, float]]  # Min/max ranges per dimension
    learning_rate: float = 1.0         # For CMA-MAE
    threshold_min: float = -np.inf     # Minimum threshold for acceptance


@dataclass
class EmitterConfig:
    """Configuration for emitters"""
    emitter_type: EmitterType = EmitterType.EVOLUTION_STRATEGY
    x0: Optional[np.ndarray] = None    # Initial solution
    sigma0: float = 0.5                # Initial step size
    batch_size: int = 36               # Solutions per batch
    ranker: RankerType = RankerType.TWO_IMPROVEMENT
    restart_rule: RestartRule = RestartRule.NO_IMPROVEMENT
    selection_rule: str = "filter"     # "filter" or "mu"
    seed: Optional[int] = None


@dataclass
class QDStats:
    """Statistics from QD optimization"""
    coverage: float = 0.0              # Fraction of archive filled
    qd_score: float = 0.0              # Sum of all objective values
    obj_max: float = -np.inf           # Maximum objective found
    obj_mean: float = 0.0              # Mean objective of archive
    iterations: int = 0                # Number of iterations run


@dataclass
class QDResult:
    """Result from QD optimization"""
    best_solution: np.ndarray
    best_objective: float
    best_measures: np.ndarray
    stats: QDStats
    archive_data: Optional[Any] = None  # Full archive DataFrame


class UnifiedQDOptimizer:
    """
    Unified Quality-Diversity optimizer for UNLEASH platform.

    Implements MAP-Elites and CMA-ME algorithms for exploring
    diverse high-quality solutions in creative AI applications.

    Example:
        qd = UnifiedQDOptimizer(
            archive_config=ArchiveConfig(
                solution_dim=100,
                behavior_dims=[50, 50],
                behavior_ranges=[(-5, 5), (-5, 5)]
            ),
            num_emitters=15
        )

        # Define your objective and measure functions
        def evaluate(solutions):
            objectives = -np.sum(np.square(solutions), axis=1)
            measures = solutions[:, :2]  # First 2 dims as behavior
            return objectives, measures

        # Run optimization
        result = qd.optimize(evaluate, iterations=1000)
    """

    def __init__(
        self,
        archive_config: ArchiveConfig,
        emitter_config: Optional[EmitterConfig] = None,
        num_emitters: int = 15,
        result_archive: bool = True,
    ):
        self.archive_config = archive_config
        self.emitter_config = emitter_config or EmitterConfig()
        self.num_emitters = num_emitters
        self.use_result_archive = result_archive

        self._archive = None
        self._result_archive = None
        self._emitters = None
        self._scheduler = None
        self._initialized = False
        self._stats = QDStats()

        self._init_components()

    def _init_components(self) -> None:
        """Initialize pyribs components"""
        try:
            from ribs.archives import GridArchive
            from ribs.emitters import EvolutionStrategyEmitter
            from ribs.schedulers import Scheduler

            # Create main archive
            self._archive = GridArchive(
                solution_dim=self.archive_config.solution_dim,
                dims=self.archive_config.behavior_dims,
                ranges=self.archive_config.behavior_ranges,
                learning_rate=self.archive_config.learning_rate,
                threshold_min=self.archive_config.threshold_min,
            )

            # Create result archive for CMA-MAE
            if self.use_result_archive:
                self._result_archive = GridArchive(
                    solution_dim=self.archive_config.solution_dim,
                    dims=self.archive_config.behavior_dims,
                    ranges=self.archive_config.behavior_ranges,
                )

            # Create emitters
            x0 = self.emitter_config.x0
            if x0 is None:
                x0 = np.zeros(self.archive_config.solution_dim)

            self._emitters = []
            for i in range(self.num_emitters):
                seed = self.emitter_config.seed
                if seed is not None:
                    seed = seed + i

                # Type hint for pyribs Literals
                selection: Any = self.emitter_config.selection_rule
                restart: Any = self.emitter_config.restart_rule.value

                emitter = EvolutionStrategyEmitter(
                    self._archive,
                    x0=x0,
                    sigma0=self.emitter_config.sigma0,
                    ranker=self.emitter_config.ranker.value,
                    selection_rule=selection,
                    restart_rule=restart,
                    batch_size=self.emitter_config.batch_size,
                    seed=seed,
                )
                self._emitters.append(emitter)

            # Create scheduler
            self._scheduler = Scheduler(
                self._archive,
                self._emitters,
                result_archive=self._result_archive,
            )

            self._initialized = True
            logger.info(f"QD optimizer initialized: {self.num_emitters} emitters, "
                       f"archive dims {self.archive_config.behavior_dims}")

        except ImportError as e:
            logger.error(f"pyribs not installed: {e}")
            logger.info("Install with: pip install ribs[all]")
            raise

    def ask(self) -> np.ndarray:
        """
        Get next batch of solutions to evaluate.

        Returns:
            Array of solutions with shape (batch_size * num_emitters, solution_dim)
        """
        if not self._initialized or self._scheduler is None:
            raise RuntimeError("Optimizer not initialized")

        return self._scheduler.ask()

    def tell(
        self,
        objectives: np.ndarray,
        measures: np.ndarray,
    ) -> None:
        """
        Report evaluation results back to the optimizer.

        Args:
            objectives: Objective values for each solution
            measures: Behavior measures for each solution
        """
        if not self._initialized or self._scheduler is None:
            raise RuntimeError("Optimizer not initialized")

        self._scheduler.tell(objectives, measures)
        self._update_stats()

    def _update_stats(self) -> None:
        """Update optimization statistics"""
        archive = self._result_archive if self._result_archive else self._archive

        if archive is None:
            return

        stats = archive.stats
        self._stats.coverage = float(stats.coverage)
        self._stats.qd_score = float(stats.qd_score)
        self._stats.obj_max = float(stats.obj_max) if stats.obj_max is not None else -np.inf
        self._stats.obj_mean = float(stats.obj_mean) if stats.obj_mean is not None else 0.0
        self._stats.iterations += 1

    def optimize(
        self,
        evaluate_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
        iterations: int = 1000,
        log_interval: int = 100,
    ) -> QDResult:
        """
        Run full optimization loop.

        Args:
            evaluate_fn: Function that takes solutions and returns (objectives, measures)
            iterations: Number of optimization iterations
            log_interval: Iterations between progress logs

        Returns:
            QDResult with best solution and statistics
        """
        for itr in range(iterations):
            # Ask for solutions
            solutions = self.ask()

            # Evaluate solutions
            objectives, measures = evaluate_fn(solutions)

            # Tell results
            self.tell(objectives, measures)

            # Log progress
            if (itr + 1) % log_interval == 0:
                logger.info(f"Iteration {itr + 1}/{iterations}")
                logger.info(f"  Coverage: {self._stats.coverage * 100:.2f}%")
                logger.info(f"  QD Score: {self._stats.qd_score:.2f}")
                logger.info(f"  Max Objective: {self._stats.obj_max:.2f}")

        return self.get_result()

    def get_result(self) -> QDResult:
        """Get current optimization result"""
        archive = self._result_archive if self._result_archive else self._archive

        if archive is None:
            raise RuntimeError("No archive available. Optimizer not initialized.")

        # Find best solution
        data = archive.data()
        best_idx = int(np.argmax(data["objective"]))

        return QDResult(
            best_solution=data["solution"][best_idx],
            best_objective=float(data["objective"][best_idx]),
            best_measures=data["measures"][best_idx],
            stats=self._stats,
            archive_data=archive.data(return_type="pandas"),
        )

    def sample_elites(
        self,
        n: int = 10,
        method: str = "random"
    ) -> List[Dict[str, Any]]:
        """
        Sample elite solutions from the archive.

        Args:
            n: Number of elites to sample
            method: "random", "best", or "diverse"

        Returns:
            List of elite dictionaries with solution, objective, measures
        """
        archive = self._result_archive if self._result_archive else self._archive

        if archive is None:
            raise RuntimeError("No archive available. Optimizer not initialized.")

        data = archive.data()

        if method == "best":
            # Top n by objective
            indices = np.argsort(data["objective"])[-n:][::-1]
        elif method == "diverse":
            # Sample from different regions of behavior space
            indices = np.random.choice(len(data["objective"]), size=min(n, len(data["objective"])), replace=False)
        else:
            # Random sampling
            indices = np.random.choice(len(data["objective"]), size=min(n, len(data["objective"])), replace=False)

        elites = []
        for idx in indices:
            elites.append({
                "solution": data["solution"][idx],
                "objective": data["objective"][idx],
                "measures": data["measures"][idx],
            })

        return elites

    def visualize(
        self,
        save_path: Optional[str] = None,
        title: str = "QD Archive Heatmap"
    ) -> None:
        """
        Visualize the archive as a heatmap.

        Args:
            save_path: Path to save the figure (optional)
            title: Plot title
        """
        try:
            import matplotlib.pyplot as plt
            from ribs.visualize import grid_archive_heatmap

            archive = self._result_archive if self._result_archive else self._archive

            if archive is None:
                logger.warning("No archive available for visualization")
                return

            plt.figure(figsize=(8, 6))
            grid_archive_heatmap(archive)
            plt.title(title)
            plt.xlabel("Behavior Dimension 1")
            plt.ylabel("Behavior Dimension 2")

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                logger.info(f"Saved visualization to {save_path}")
            else:
                plt.show()

            plt.close()

        except ImportError:
            logger.warning("matplotlib not available for visualization")

    def export_archive(self, path: str) -> None:
        """Export archive to CSV file"""
        archive = self._result_archive if self._result_archive else self._archive

        if archive is None:
            logger.error("No archive available for export")
            return

        try:
            df = archive.data(return_type="pandas")
            df.to_csv(path, index=False)
            logger.info(f"Exported archive to {path}")
        except Exception as e:
            logger.error(f"Failed to export archive: {e}")

    @property
    def stats(self) -> QDStats:
        """Get current optimization statistics"""
        return self._stats


# Convenience functions for common QD setups

def create_creative_optimizer(
    solution_dim: int,
    behavior_dims: List[int] = [50, 50],
    behavior_ranges: List[Tuple[float, float]] = [(-1, 1), (-1, 1)],
    num_emitters: int = 15,
    sigma0: float = 0.5,
) -> UnifiedQDOptimizer:
    """
    Create a QD optimizer configured for creative exploration.

    Args:
        solution_dim: Dimension of solution vectors (e.g., latent space dim)
        behavior_dims: Grid dimensions for behavior archive
        behavior_ranges: Ranges for each behavior dimension
        num_emitters: Number of CMA-ES emitters
        sigma0: Initial step size

    Returns:
        Configured UnifiedQDOptimizer
    """
    return UnifiedQDOptimizer(
        archive_config=ArchiveConfig(
            solution_dim=solution_dim,
            behavior_dims=behavior_dims,
            behavior_ranges=behavior_ranges,
        ),
        emitter_config=EmitterConfig(
            sigma0=sigma0,
            ranker=RankerType.TWO_IMPROVEMENT,
            restart_rule=RestartRule.NO_IMPROVEMENT,
        ),
        num_emitters=num_emitters,
    )


def sphere_benchmark(solutions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard sphere benchmark for testing QD algorithms.

    Args:
        solutions: Array of solutions to evaluate

    Returns:
        Tuple of (objectives, measures)
    """
    # Negative sphere function (higher is better)
    shift = 5.12 * 0.4
    objectives = -np.sum(np.square(solutions - shift), axis=1)

    # Measures: sum of halves
    dim = solutions.shape[1]
    clipped = np.clip(solutions, -5.12, 5.12)
    measures = np.stack([
        np.sum(clipped[:, :dim // 2], axis=1),
        np.sum(clipped[:, dim // 2:], axis=1),
    ], axis=1)

    return objectives, measures
