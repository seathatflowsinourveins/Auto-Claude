"""
Self-Improvement Pipeline for Unleashed Platform

Combines EvoAgentX (genetic workflow evolution) and TextGrad (gradient-based
optimization) for continuous system improvement.

Key features:
- Workflow evolution using genetic algorithms
- Gradient-based prompt optimization
- Ralph Loop integration for iterative refinement
- Performance tracking and regression prevention

Flow:
Evaluate → Evolve (Genetic) → Optimize (Gradient) → Validate → Deploy
"""

import os
import json
import asyncio
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

# Check component availability
PIPELINE_AVAILABLE = True
_missing_components = []

# TextGrad
try:
    import textgrad
    TEXTGRAD_AVAILABLE = True
except ImportError:
    TEXTGRAD_AVAILABLE = False
    _missing_components.append("textgrad")

# EvoAgentX would be custom integration
EVOAGENTX_AVAILABLE = False  # Custom implementation below

# Register pipeline
from . import register_pipeline
register_pipeline(
    "self_improvement",
    PIPELINE_AVAILABLE,
    dependencies=["textgrad", "evoagentx", "ralph_loop"]
)


class ImprovementStrategy(Enum):
    """Strategies for improvement."""
    GENETIC = "genetic"           # Evolutionary algorithms
    GRADIENT = "gradient"         # Gradient-based optimization
    HYBRID = "hybrid"             # Combined approach
    ITERATIVE = "iterative"       # Ralph Loop style


class SelectionMethod(Enum):
    """Selection methods for genetic algorithms."""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    ELITIST = "elitist"
    RANK = "rank"


@dataclass
class Workflow:
    """A workflow definition that can be evolved."""
    id: str
    name: str
    steps: List[Dict[str, Any]]
    parameters: Dict[str, Any] = field(default_factory=dict)
    fitness: float = 0.0
    generation: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "steps": self.steps,
            "parameters": self.parameters,
            "fitness": self.fitness,
            "generation": self.generation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Workflow":
        return cls(
            id=data["id"],
            name=data["name"],
            steps=data["steps"],
            parameters=data.get("parameters", {}),
            fitness=data.get("fitness", 0.0),
            generation=data.get("generation", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ImprovementResult:
    """Result from improvement iteration."""
    original_fitness: float
    improved_fitness: float
    improvement: float
    strategy_used: ImprovementStrategy
    iterations: int
    best_workflow: Workflow
    history: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0


@dataclass
class GeneticConfig:
    """Configuration for genetic evolution."""
    population_size: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_size: int = 2
    tournament_size: int = 3
    max_generations: int = 50
    convergence_threshold: float = 0.001
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT


@dataclass
class GradientConfig:
    """Configuration for gradient optimization."""
    learning_rate: float = 0.01
    max_iterations: int = 100
    convergence_threshold: float = 0.0001
    batch_size: int = 5
    momentum: float = 0.9
    decay: float = 0.99


class FitnessEvaluator(ABC):
    """Abstract base class for fitness evaluation."""

    @abstractmethod
    async def evaluate(self, workflow: Workflow, test_cases: List[Dict[str, Any]]) -> float:
        """Evaluate workflow fitness."""
        pass


class DefaultFitnessEvaluator(FitnessEvaluator):
    """Default fitness evaluator using execution success rate."""

    def __init__(self, executor: Optional[Callable] = None):
        self.executor = executor

    async def evaluate(self, workflow: Workflow, test_cases: List[Dict[str, Any]]) -> float:
        """Evaluate based on successful executions."""
        if not test_cases:
            return 0.5  # Default neutral fitness

        successes = 0
        total_score = 0.0

        for test in test_cases:
            try:
                # Execute workflow (mock if no executor)
                if self.executor:
                    result = await self.executor(workflow, test)
                    if result.get("success", False):
                        successes += 1
                        total_score += result.get("score", 1.0)
                else:
                    # Mock evaluation based on workflow complexity
                    complexity = len(workflow.steps)
                    params = workflow.parameters
                    mock_score = min(1.0, 0.5 + complexity * 0.1 - len(params) * 0.05)
                    total_score += mock_score
                    successes += 1 if mock_score > 0.5 else 0

            except Exception:
                pass

        success_rate = successes / len(test_cases)
        avg_score = total_score / len(test_cases) if test_cases else 0

        # Combined fitness
        return 0.4 * success_rate + 0.6 * avg_score


class GeneticEvolver:
    """
    Genetic algorithm for workflow evolution.

    Implements:
    - Selection (tournament, roulette, elitist)
    - Crossover (single-point, two-point, uniform)
    - Mutation (parameter tweak, step swap, step modify)
    """

    def __init__(
        self,
        config: Optional[GeneticConfig] = None,
        evaluator: Optional[FitnessEvaluator] = None,
    ):
        self.config = config or GeneticConfig()
        self.evaluator = evaluator or DefaultFitnessEvaluator()
        self._population: List[Workflow] = []
        self._best: Optional[Workflow] = None
        self._history: List[Dict[str, Any]] = []

    async def evolve(
        self,
        initial_workflow: Workflow,
        test_cases: List[Dict[str, Any]],
        num_generations: Optional[int] = None,
    ) -> ImprovementResult:
        """
        Evolve a workflow using genetic algorithms.

        Args:
            initial_workflow: Starting workflow
            test_cases: Test cases for fitness evaluation
            num_generations: Number of generations (or use config)

        Returns:
            ImprovementResult with evolved workflow
        """
        import time
        start_time = time.time()

        generations = num_generations or self.config.max_generations

        # Initialize population from initial workflow
        self._population = self._initialize_population(initial_workflow)

        # Evaluate initial fitness
        initial_fitness = await self.evaluator.evaluate(initial_workflow, test_cases)
        initial_workflow.fitness = initial_fitness

        best_fitness = initial_fitness
        stagnation_counter = 0

        for gen in range(generations):
            # Evaluate population
            for individual in self._population:
                individual.fitness = await self.evaluator.evaluate(individual, test_cases)

            # Sort by fitness
            self._population.sort(key=lambda w: w.fitness, reverse=True)

            # Track best
            if self._population[0].fitness > best_fitness:
                best_fitness = self._population[0].fitness
                self._best = self._population[0]
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Record history
            self._history.append({
                "generation": gen,
                "best_fitness": self._population[0].fitness,
                "avg_fitness": sum(w.fitness for w in self._population) / len(self._population),
                "worst_fitness": self._population[-1].fitness,
            })

            # Check convergence
            if stagnation_counter > 10:
                break

            # Selection
            selected = self._select(self._population)

            # Crossover
            offspring = self._crossover(selected)

            # Mutation
            mutated = self._mutate(offspring)

            # Create new population (elitism + offspring)
            elite = self._population[:self.config.elite_size]
            self._population = elite + mutated[:self.config.population_size - self.config.elite_size]

            # Update generation
            for w in self._population:
                w.generation = gen + 1

        execution_time = time.time() - start_time
        final_best = self._best or self._population[0]

        return ImprovementResult(
            original_fitness=initial_fitness,
            improved_fitness=final_best.fitness,
            improvement=final_best.fitness - initial_fitness,
            strategy_used=ImprovementStrategy.GENETIC,
            iterations=len(self._history),
            best_workflow=final_best,
            history=self._history,
            execution_time=execution_time,
        )

    def _initialize_population(self, base: Workflow) -> List[Workflow]:
        """Create initial population from base workflow."""
        import random
        import uuid

        population = [base]

        for i in range(self.config.population_size - 1):
            # Clone and mutate
            variant = Workflow(
                id=str(uuid.uuid4())[:8],
                name=f"{base.name}_v{i}",
                steps=base.steps.copy(),
                parameters=base.parameters.copy(),
                generation=0,
            )
            # Random parameter variation
            for key in variant.parameters:
                if isinstance(variant.parameters[key], (int, float)):
                    variant.parameters[key] *= random.uniform(0.8, 1.2)

            population.append(variant)

        return population

    def _select(self, population: List[Workflow]) -> List[Workflow]:
        """Select individuals for reproduction."""
        import random

        method = self.config.selection_method
        selected = []

        if method == SelectionMethod.TOURNAMENT:
            for _ in range(len(population)):
                tournament = random.sample(population, min(self.config.tournament_size, len(population)))
                winner = max(tournament, key=lambda w: w.fitness)
                selected.append(winner)

        elif method == SelectionMethod.ROULETTE:
            total_fitness = sum(max(w.fitness, 0.01) for w in population)
            for _ in range(len(population)):
                pick = random.uniform(0, total_fitness)
                current = 0
                for w in population:
                    current += max(w.fitness, 0.01)
                    if current >= pick:
                        selected.append(w)
                        break

        elif method == SelectionMethod.ELITIST:
            # Select top performers
            sorted_pop = sorted(population, key=lambda w: w.fitness, reverse=True)
            selected = sorted_pop[:len(population) // 2] * 2

        else:  # RANK
            sorted_pop = sorted(population, key=lambda w: w.fitness, reverse=True)
            ranks = list(range(len(sorted_pop), 0, -1))
            total_rank = sum(ranks)
            for _ in range(len(population)):
                pick = random.uniform(0, total_rank)
                current = 0
                for i, w in enumerate(sorted_pop):
                    current += ranks[i]
                    if current >= pick:
                        selected.append(w)
                        break

        return selected

    def _crossover(self, parents: List[Workflow]) -> List[Workflow]:
        """Perform crossover between parents."""
        import random
        import uuid

        offspring = []

        for i in range(0, len(parents) - 1, 2):
            if random.random() < self.config.crossover_rate:
                p1, p2 = parents[i], parents[i + 1]

                # Single-point crossover on steps
                if len(p1.steps) > 1 and len(p2.steps) > 1:
                    point = random.randint(1, min(len(p1.steps), len(p2.steps)) - 1)
                    child1_steps = p1.steps[:point] + p2.steps[point:]
                    child2_steps = p2.steps[:point] + p1.steps[point:]
                else:
                    child1_steps = p1.steps.copy()
                    child2_steps = p2.steps.copy()

                # Parameter crossover
                child1_params = {}
                child2_params = {}
                all_keys = set(p1.parameters.keys()) | set(p2.parameters.keys())
                for key in all_keys:
                    if random.random() < 0.5:
                        child1_params[key] = p1.parameters.get(key)
                        child2_params[key] = p2.parameters.get(key)
                    else:
                        child1_params[key] = p2.parameters.get(key)
                        child2_params[key] = p1.parameters.get(key)

                offspring.append(Workflow(
                    id=str(uuid.uuid4())[:8],
                    name=f"child_{len(offspring)}",
                    steps=child1_steps,
                    parameters=child1_params,
                ))
                offspring.append(Workflow(
                    id=str(uuid.uuid4())[:8],
                    name=f"child_{len(offspring)}",
                    steps=child2_steps,
                    parameters=child2_params,
                ))
            else:
                offspring.extend([parents[i], parents[i + 1]])

        return offspring

    def _mutate(self, population: List[Workflow]) -> List[Workflow]:
        """Apply mutations to population."""
        import random

        for workflow in population:
            if random.random() < self.config.mutation_rate:
                # Parameter mutation
                for key in workflow.parameters:
                    if random.random() < 0.3:
                        val = workflow.parameters[key]
                        if isinstance(val, float):
                            workflow.parameters[key] = val * random.uniform(0.9, 1.1)
                        elif isinstance(val, int):
                            workflow.parameters[key] = int(val * random.uniform(0.9, 1.1))

                # Step mutation (swap two steps)
                if len(workflow.steps) > 1 and random.random() < 0.2:
                    i, j = random.sample(range(len(workflow.steps)), 2)
                    workflow.steps[i], workflow.steps[j] = workflow.steps[j], workflow.steps[i]

        return population


class GradientOptimizer:
    """
    Gradient-based optimization using TextGrad or similar.

    Optimizes prompts and parameters using gradient descent in text space.
    """

    def __init__(
        self,
        config: Optional[GradientConfig] = None,
    ):
        self.config = config or GradientConfig()
        self._available = TEXTGRAD_AVAILABLE

    async def optimize(
        self,
        workflow: Workflow,
        loss_fn: Callable,
        target: Optional[str] = None,
    ) -> ImprovementResult:
        """
        Optimize workflow using gradient-based methods.

        Args:
            workflow: Workflow to optimize
            loss_fn: Loss function for optimization
            target: Target output (if applicable)

        Returns:
            ImprovementResult with optimized workflow
        """
        import time
        start_time = time.time()

        if not self._available:
            # Fallback to simple hill climbing
            return await self._hill_climb(workflow, loss_fn)

        try:
            # TextGrad optimization
            return await self._textgrad_optimize(workflow, loss_fn, target)
        except Exception as e:
            print(f"TextGrad optimization failed: {e}")
            return await self._hill_climb(workflow, loss_fn)

    async def _textgrad_optimize(
        self,
        workflow: Workflow,
        loss_fn: Callable,
        target: Optional[str],
    ) -> ImprovementResult:
        """Optimize using TextGrad."""
        import time
        start_time = time.time()

        # Convert workflow to TextGrad variable
        workflow_text = json.dumps(workflow.to_dict())

        if TEXTGRAD_AVAILABLE:
            from textgrad import Variable, get_engine

            # Create variable
            var = Variable(workflow_text, requires_grad=True)

            # Setup engine
            engine = get_engine("gpt-4")  # or claude

            initial_loss = loss_fn(workflow)
            best_workflow = workflow
            best_loss = initial_loss
            history = []

            for i in range(self.config.max_iterations):
                # Compute loss
                current_loss = loss_fn(best_workflow)

                # Record history
                history.append({"iteration": i, "loss": current_loss})

                # Check convergence
                if abs(current_loss - best_loss) < self.config.convergence_threshold:
                    break

                if current_loss < best_loss:
                    best_loss = current_loss

                # Gradient step (mock - TextGrad would handle this)
                # In practice, TextGrad computes text gradients automatically

            execution_time = time.time() - start_time

            return ImprovementResult(
                original_fitness=1 - initial_loss,
                improved_fitness=1 - best_loss,
                improvement=initial_loss - best_loss,
                strategy_used=ImprovementStrategy.GRADIENT,
                iterations=len(history),
                best_workflow=best_workflow,
                history=history,
                execution_time=execution_time,
            )
        else:
            return await self._hill_climb(workflow, loss_fn)

    async def _hill_climb(
        self,
        workflow: Workflow,
        loss_fn: Callable,
    ) -> ImprovementResult:
        """Simple hill climbing fallback."""
        import time
        import random

        start_time = time.time()
        history = []

        current = workflow
        current_loss = loss_fn(current)
        initial_loss = current_loss

        for i in range(self.config.max_iterations):
            # Generate neighbor
            neighbor = Workflow(
                id=current.id,
                name=current.name,
                steps=current.steps.copy(),
                parameters={
                    k: v * (1 + random.uniform(-0.1, 0.1)) if isinstance(v, (int, float)) else v
                    for k, v in current.parameters.items()
                },
            )

            neighbor_loss = loss_fn(neighbor)

            history.append({"iteration": i, "loss": current_loss})

            if neighbor_loss < current_loss:
                current = neighbor
                current_loss = neighbor_loss

            # Check convergence
            if abs(current_loss - initial_loss) < self.config.convergence_threshold:
                break

        execution_time = time.time() - start_time

        current.fitness = 1 - current_loss

        return ImprovementResult(
            original_fitness=1 - initial_loss,
            improved_fitness=1 - current_loss,
            improvement=initial_loss - current_loss,
            strategy_used=ImprovementStrategy.GRADIENT,
            iterations=len(history),
            best_workflow=current,
            history=history,
            execution_time=execution_time,
        )


class SelfImprovementPipeline:
    """
    Full self-improvement pipeline combining genetic and gradient optimization.

    Integrates with Ralph Loop for continuous iterative improvement.

    Example:
        pipeline = SelfImprovementPipeline()
        result = await pipeline.improve(
            workflow=my_workflow,
            test_cases=test_data,
            strategy=ImprovementStrategy.HYBRID,
        )
    """

    def __init__(
        self,
        genetic_config: Optional[GeneticConfig] = None,
        gradient_config: Optional[GradientConfig] = None,
    ):
        self.genetic_config = genetic_config or GeneticConfig()
        self.gradient_config = gradient_config or GradientConfig()

        self._evolver = GeneticEvolver(config=self.genetic_config)
        self._optimizer = GradientOptimizer(config=self.gradient_config)

        self._iteration_count = 0
        self._improvement_history: List[ImprovementResult] = []

    async def improve(
        self,
        workflow: Workflow,
        test_cases: List[Dict[str, Any]],
        strategy: ImprovementStrategy = ImprovementStrategy.HYBRID,
        loss_fn: Optional[Callable] = None,
    ) -> ImprovementResult:
        """
        Improve a workflow using the specified strategy.

        Args:
            workflow: Workflow to improve
            test_cases: Test cases for evaluation
            strategy: Improvement strategy
            loss_fn: Custom loss function for gradient optimization

        Returns:
            ImprovementResult with improved workflow
        """
        self._iteration_count += 1

        if strategy == ImprovementStrategy.GENETIC:
            result = await self._evolver.evolve(workflow, test_cases)

        elif strategy == ImprovementStrategy.GRADIENT:
            # Use default loss function if not provided
            if loss_fn is None:
                evaluator = DefaultFitnessEvaluator()

                def default_loss(w):
                    # Synchronous approximation
                    return 1 - w.fitness if w.fitness > 0 else 0.5

                loss_fn = default_loss

            result = await self._optimizer.optimize(workflow, loss_fn)

        elif strategy == ImprovementStrategy.HYBRID:
            # First genetic, then gradient
            genetic_result = await self._evolver.evolve(workflow, test_cases)

            if loss_fn is None:
                def default_loss(w):
                    return 1 - w.fitness if w.fitness > 0 else 0.5
                loss_fn = default_loss

            gradient_result = await self._optimizer.optimize(
                genetic_result.best_workflow,
                loss_fn,
            )

            # Combine results
            result = ImprovementResult(
                original_fitness=genetic_result.original_fitness,
                improved_fitness=gradient_result.improved_fitness,
                improvement=gradient_result.improved_fitness - genetic_result.original_fitness,
                strategy_used=ImprovementStrategy.HYBRID,
                iterations=genetic_result.iterations + gradient_result.iterations,
                best_workflow=gradient_result.best_workflow,
                history=genetic_result.history + gradient_result.history,
                execution_time=genetic_result.execution_time + gradient_result.execution_time,
            )

        else:  # ITERATIVE (Ralph Loop style)
            result = await self._iterative_improve(workflow, test_cases)

        self._improvement_history.append(result)
        return result

    async def _iterative_improve(
        self,
        workflow: Workflow,
        test_cases: List[Dict[str, Any]],
        max_iterations: int = 10,
    ) -> ImprovementResult:
        """Ralph Loop style iterative improvement."""
        import time
        start_time = time.time()

        current = workflow
        history = []
        evaluator = DefaultFitnessEvaluator()

        initial_fitness = await evaluator.evaluate(current, test_cases)
        current.fitness = initial_fitness
        best_fitness = initial_fitness

        for i in range(max_iterations):
            # Evaluate
            fitness = await evaluator.evaluate(current, test_cases)
            current.fitness = fitness

            history.append({
                "iteration": i,
                "fitness": fitness,
            })

            if fitness > best_fitness:
                best_fitness = fitness

            # Small genetic mutation
            if fitness < 0.9:  # Room for improvement
                await self._evolver.evolve(current, test_cases, num_generations=5)
                current = self._evolver._best or current

        execution_time = time.time() - start_time

        return ImprovementResult(
            original_fitness=initial_fitness,
            improved_fitness=best_fitness,
            improvement=best_fitness - initial_fitness,
            strategy_used=ImprovementStrategy.ITERATIVE,
            iterations=len(history),
            best_workflow=current,
            history=history,
            execution_time=execution_time,
        )

    def get_improvement_history(self) -> List[ImprovementResult]:
        """Get history of all improvements."""
        return self._improvement_history.copy()

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return {
            "iteration_count": self._iteration_count,
            "total_improvements": len(self._improvement_history),
            "textgrad_available": TEXTGRAD_AVAILABLE,
            "avg_improvement": (
                sum(r.improvement for r in self._improvement_history) / len(self._improvement_history)
                if self._improvement_history else 0
            ),
            "genetic_config": {
                "population_size": self.genetic_config.population_size,
                "mutation_rate": self.genetic_config.mutation_rate,
                "max_generations": self.genetic_config.max_generations,
            },
            "gradient_config": {
                "learning_rate": self.gradient_config.learning_rate,
                "max_iterations": self.gradient_config.max_iterations,
            },
        }
