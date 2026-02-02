"""
Agent Evolution Pipeline for Unleashed Platform

This pipeline combines multiple optimization strategies to evolve AI agents:
1. Genetic Evolution (EvoAgentX) - Population-based agent mutation
2. Gradient Optimization (TextGrad) - Differentiable prompt refinement
3. Quality-Diversity (MAP-Elites) - Diverse solution exploration
4. Memory-Guided (Mem0) - Experience-based improvement

The pipeline enables continuous agent self-improvement through
iterative evaluation, selection, and refinement cycles.
"""

import asyncio
import random
import hashlib
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

# Pipeline availability flag
PIPELINE_AVAILABLE = True

# Register pipeline
from . import register_pipeline
register_pipeline(
    "agent_evolution",
    PIPELINE_AVAILABLE,
    ["textgrad", "evoagentx", "mem0"]
)


class EvolutionStrategy(Enum):
    """Evolution strategies available."""
    GENETIC = "genetic"           # Pure genetic algorithm
    GRADIENT = "gradient"         # Pure gradient descent
    MAP_ELITES = "map_elites"     # Quality-diversity optimization
    HYBRID = "hybrid"             # Combined approach
    MEMORY_GUIDED = "memory"      # Experience-based evolution


class FitnessMetric(Enum):
    """Metrics for evaluating agent fitness."""
    TASK_SUCCESS = "task_success"      # Binary success rate
    QUALITY_SCORE = "quality_score"    # Quality of outputs
    EFFICIENCY = "efficiency"          # Speed and resource usage
    DIVERSITY = "diversity"            # Behavioral diversity
    COMPOSITE = "composite"            # Weighted combination


@dataclass
class AgentGenome:
    """
    Represents an agent's evolvable configuration.

    The genome encodes all parameters that can be mutated
    during evolution, including prompts, parameters, and behaviors.
    """
    id: str
    generation: int
    system_prompt: str
    task_prompts: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    behaviors: List[str] = field(default_factory=list)
    fitness: float = 0.0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary."""
        return {
            "id": self.id,
            "generation": self.generation,
            "system_prompt": self.system_prompt,
            "task_prompts": self.task_prompts,
            "parameters": self.parameters,
            "behaviors": self.behaviors,
            "fitness": self.fitness,
            "parent_ids": self.parent_ids,
            "mutation_history": self.mutation_history,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentGenome":
        """Create genome from dictionary."""
        return cls(**data)


@dataclass
class EvolutionResult:
    """Result from an evolution run."""
    best_genome: AgentGenome
    population: List[AgentGenome]
    generations_completed: int
    fitness_history: List[float]
    diversity_score: float
    convergence_generation: Optional[int]
    total_evaluations: int
    evolution_time_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result from evaluating a single genome."""
    genome_id: str
    fitness: float
    task_scores: Dict[str, float] = field(default_factory=dict)
    behavioral_features: List[float] = field(default_factory=list)
    execution_time: float = 0.0
    errors: List[str] = field(default_factory=list)


class PopulationManager:
    """Manages a population of agent genomes."""

    def __init__(
        self,
        population_size: int = 20,
        elite_ratio: float = 0.1,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
    ):
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population: List[AgentGenome] = []
        self.archive: List[AgentGenome] = []  # Hall of fame

    def initialize_population(
        self,
        base_prompt: str,
        task_prompts: Dict[str, str],
        variation_fn: Optional[Callable[[str], str]] = None,
    ) -> List[AgentGenome]:
        """
        Initialize population with variations of base genome.

        Args:
            base_prompt: Starting system prompt
            task_prompts: Task-specific prompts
            variation_fn: Function to create variations

        Returns:
            Initial population
        """
        self.population = []

        for i in range(self.population_size):
            # Create genome ID
            genome_id = hashlib.md5(
                f"{base_prompt}{i}{datetime.now()}".encode()
            ).hexdigest()[:12]

            # Apply variation if provided
            if variation_fn and i > 0:
                system_prompt = variation_fn(base_prompt)
            else:
                system_prompt = base_prompt

            genome = AgentGenome(
                id=genome_id,
                generation=0,
                system_prompt=system_prompt,
                task_prompts=task_prompts.copy(),
                parameters={
                    "temperature": random.uniform(0.3, 0.9),
                    "max_tokens": random.choice([512, 1024, 2048]),
                    "top_p": random.uniform(0.8, 1.0),
                },
                behaviors=[],
                metadata={"created_at": datetime.now().isoformat()},
            )
            self.population.append(genome)

        return self.population

    def select_parents(
        self,
        tournament_size: int = 3,
    ) -> Tuple[AgentGenome, AgentGenome]:
        """
        Select two parents using tournament selection.

        Args:
            tournament_size: Number of candidates per tournament

        Returns:
            Two parent genomes
        """
        def tournament() -> AgentGenome:
            candidates = random.sample(
                self.population,
                min(tournament_size, len(self.population))
            )
            return max(candidates, key=lambda g: g.fitness)

        return tournament(), tournament()

    def crossover(
        self,
        parent1: AgentGenome,
        parent2: AgentGenome,
        generation: int,
    ) -> AgentGenome:
        """
        Create offspring through crossover.

        Args:
            parent1: First parent genome
            parent2: Second parent genome
            generation: Current generation number

        Returns:
            Offspring genome
        """
        # Generate new ID
        child_id = hashlib.md5(
            f"{parent1.id}{parent2.id}{generation}".encode()
        ).hexdigest()[:12]

        # Crossover system prompts (take segments from each)
        words1 = parent1.system_prompt.split()
        words2 = parent2.system_prompt.split()
        crossover_point = random.randint(1, min(len(words1), len(words2)) - 1)

        if random.random() < 0.5:
            system_prompt = " ".join(words1[:crossover_point] + words2[crossover_point:])
        else:
            system_prompt = " ".join(words2[:crossover_point] + words1[crossover_point:])

        # Crossover task prompts
        task_prompts = {}
        for key in set(parent1.task_prompts) | set(parent2.task_prompts):
            if key in parent1.task_prompts and key in parent2.task_prompts:
                task_prompts[key] = random.choice([
                    parent1.task_prompts[key],
                    parent2.task_prompts[key]
                ])
            elif key in parent1.task_prompts:
                task_prompts[key] = parent1.task_prompts[key]
            else:
                task_prompts[key] = parent2.task_prompts[key]

        # Crossover parameters (average)
        parameters = {}
        for key in set(parent1.parameters) | set(parent2.parameters):
            val1 = parent1.parameters.get(key)
            val2 = parent2.parameters.get(key)
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                parameters[key] = (val1 + val2) / 2
            else:
                parameters[key] = random.choice([val1, val2])

        return AgentGenome(
            id=child_id,
            generation=generation,
            system_prompt=system_prompt,
            task_prompts=task_prompts,
            parameters=parameters,
            behaviors=list(set(parent1.behaviors + parent2.behaviors)),
            parent_ids=[parent1.id, parent2.id],
            mutation_history=["crossover"],
        )

    def mutate(self, genome: AgentGenome, mutation_strength: float = 0.1) -> AgentGenome:
        """
        Apply mutation to a genome.

        Args:
            genome: Genome to mutate
            mutation_strength: How much to mutate (0-1)

        Returns:
            Mutated genome (copy)
        """
        # Create copy
        mutated = AgentGenome(
            id=genome.id,
            generation=genome.generation,
            system_prompt=genome.system_prompt,
            task_prompts=genome.task_prompts.copy(),
            parameters=genome.parameters.copy(),
            behaviors=genome.behaviors.copy(),
            parent_ids=genome.parent_ids.copy(),
            mutation_history=genome.mutation_history.copy(),
        )

        mutations_applied = []

        # Mutate system prompt (word-level changes)
        if random.random() < self.mutation_rate:
            words = mutated.system_prompt.split()
            if words:
                # Insert, delete, or modify a word
                mutation_type = random.choice(["insert", "delete", "modify"])
                pos = random.randint(0, len(words) - 1)

                if mutation_type == "insert":
                    modifiers = ["carefully", "thoroughly", "efficiently",
                                "clearly", "precisely", "step by step"]
                    words.insert(pos, random.choice(modifiers))
                elif mutation_type == "delete" and len(words) > 5:
                    words.pop(pos)
                else:  # modify
                    synonyms = {
                        "help": ["assist", "support", "aid"],
                        "create": ["generate", "produce", "develop"],
                        "analyze": ["examine", "investigate", "study"],
                    }
                    word = words[pos].lower()
                    if word in synonyms:
                        words[pos] = random.choice(synonyms[word])

                mutated.system_prompt = " ".join(words)
                mutations_applied.append(f"prompt_{mutation_type}")

        # Mutate parameters
        if random.random() < self.mutation_rate:
            for param, value in mutated.parameters.items():
                if isinstance(value, float):
                    # Gaussian mutation
                    mutated.parameters[param] = max(0.0, min(1.0,
                        value + random.gauss(0, mutation_strength)
                    ))
                elif isinstance(value, int):
                    # Integer mutation
                    mutated.parameters[param] = max(1,
                        value + random.randint(-100, 100)
                    )
            mutations_applied.append("parameters")

        mutated.mutation_history.extend(mutations_applied)
        return mutated

    def evolve_generation(
        self,
        generation: int,
    ) -> List[AgentGenome]:
        """
        Evolve the population by one generation.

        Args:
            generation: Current generation number

        Returns:
            New population
        """
        # Sort by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)

        # Elite selection (keep top performers)
        num_elites = max(1, int(self.population_size * self.elite_ratio))
        new_population = self.population[:num_elites]

        # Archive best genome
        if self.population[0].fitness > 0:
            self.archive.append(self.population[0])

        # Generate offspring
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2, generation)
            else:
                # Clone and mutate
                parent = self.select_parents()[0]
                child = AgentGenome(
                    id=hashlib.md5(f"{parent.id}{generation}{random.random()}".encode()).hexdigest()[:12],
                    generation=generation,
                    system_prompt=parent.system_prompt,
                    task_prompts=parent.task_prompts.copy(),
                    parameters=parent.parameters.copy(),
                    behaviors=parent.behaviors.copy(),
                    parent_ids=[parent.id],
                    mutation_history=["clone"],
                )

            # Apply mutation
            child = self.mutate(child)
            child.generation = generation
            new_population.append(child)

        self.population = new_population
        return self.population


class GradientOptimizer:
    """
    Gradient-based prompt optimization using TextGrad-style approach.

    Instead of genetic mutations, uses LLM feedback to compute
    "gradients" and improve prompts in a directed manner.
    """

    def __init__(self, evaluation_prompt: Optional[str] = None):
        self.evaluation_prompt = evaluation_prompt or """
Evaluate this agent prompt and provide specific feedback for improvement:

PROMPT:
{prompt}

TASK PERFORMANCE:
{performance}

Provide:
1. What works well (keep these elements)
2. What needs improvement (specific issues)
3. Suggested changes (concrete modifications)
"""

    async def compute_gradient(
        self,
        genome: AgentGenome,
        task_results: Dict[str, Any],
        llm_fn: Callable,
    ) -> Dict[str, str]:
        """
        Compute text gradients for prompt improvement.

        Args:
            genome: Current genome
            task_results: Results from task evaluation
            llm_fn: Function to call LLM

        Returns:
            Gradients (improvement suggestions) for each component
        """
        prompt = self.evaluation_prompt.format(
            prompt=genome.system_prompt,
            performance=json.dumps(task_results, indent=2),
        )

        # Get LLM feedback (this is the "gradient")
        gradient = await llm_fn(prompt)

        return {
            "system_prompt": gradient,
            "suggestions": gradient,
        }

    async def apply_gradient(
        self,
        genome: AgentGenome,
        gradients: Dict[str, str],
        llm_fn: Callable,
        learning_rate: float = 1.0,
    ) -> AgentGenome:
        """
        Apply gradients to improve the genome.

        Args:
            genome: Current genome
            gradients: Improvement suggestions
            llm_fn: Function to call LLM
            learning_rate: How aggressively to apply changes

        Returns:
            Improved genome
        """
        improvement_prompt = f"""
Based on this feedback, improve the following prompt.
Make the changes suggested while keeping what works well.
Learning rate: {learning_rate} (1.0 = full changes, 0.5 = half changes)

CURRENT PROMPT:
{genome.system_prompt}

FEEDBACK:
{gradients.get('suggestions', '')}

Respond with ONLY the improved prompt, nothing else.
"""

        improved_prompt = await llm_fn(improvement_prompt)

        return AgentGenome(
            id=hashlib.md5(f"{genome.id}_grad".encode()).hexdigest()[:12],
            generation=genome.generation + 1,
            system_prompt=improved_prompt.strip(),
            task_prompts=genome.task_prompts.copy(),
            parameters=genome.parameters.copy(),
            behaviors=genome.behaviors.copy(),
            parent_ids=[genome.id],
            mutation_history=genome.mutation_history + ["gradient_step"],
        )


class AgentEvolutionPipeline:
    """
    Main pipeline for evolving AI agents.

    Combines genetic algorithms, gradient optimization,
    and memory-guided improvement for agent evolution.
    """

    def __init__(
        self,
        population_size: int = 20,
        strategy: EvolutionStrategy = EvolutionStrategy.HYBRID,
        fitness_metric: FitnessMetric = FitnessMetric.COMPOSITE,
        max_generations: int = 50,
        convergence_threshold: float = 0.95,
    ):
        """
        Initialize evolution pipeline.

        Args:
            population_size: Number of agents in population
            strategy: Evolution strategy to use
            fitness_metric: How to evaluate fitness
            max_generations: Maximum generations to run
            convergence_threshold: Fitness threshold for early stopping
        """
        self.population_size = population_size
        self.strategy = strategy
        self.fitness_metric = fitness_metric
        self.max_generations = max_generations
        self.convergence_threshold = convergence_threshold

        self.population_manager = PopulationManager(
            population_size=population_size,
        )
        self.gradient_optimizer = GradientOptimizer()

        # Statistics
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []

    async def evaluate_genome(
        self,
        genome: AgentGenome,
        tasks: List[Dict[str, Any]],
        evaluation_fn: Callable,
    ) -> EvaluationResult:
        """
        Evaluate a genome on a set of tasks.

        Args:
            genome: Genome to evaluate
            tasks: List of task specifications
            evaluation_fn: Function to evaluate task performance

        Returns:
            Evaluation result with fitness score
        """
        task_scores = {}
        total_score = 0.0
        errors = []

        for task in tasks:
            try:
                score = await evaluation_fn(genome, task)
                task_scores[task.get("name", "unnamed")] = score
                total_score += score
            except Exception as e:
                errors.append(f"Task {task.get('name')}: {str(e)}")
                task_scores[task.get("name", "unnamed")] = 0.0

        fitness = total_score / len(tasks) if tasks else 0.0

        return EvaluationResult(
            genome_id=genome.id,
            fitness=fitness,
            task_scores=task_scores,
            errors=errors,
        )

    def calculate_diversity(self, population: List[AgentGenome]) -> float:
        """
        Calculate population diversity.

        Uses prompt similarity to measure diversity.
        """
        if len(population) < 2:
            return 1.0

        # Simple diversity: unique prompt prefixes
        prefixes = set()
        for genome in population:
            prefix = genome.system_prompt[:50] if genome.system_prompt else ""
            prefixes.add(prefix)

        return len(prefixes) / len(population)

    async def evolve(
        self,
        base_prompt: str,
        task_prompts: Dict[str, str],
        tasks: List[Dict[str, Any]],
        evaluation_fn: Callable,
        llm_fn: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None,
    ) -> EvolutionResult:
        """
        Run the full evolution process.

        Args:
            base_prompt: Starting system prompt
            task_prompts: Task-specific prompts
            tasks: List of evaluation tasks
            evaluation_fn: Function to evaluate task performance
            llm_fn: Function to call LLM (for gradient optimization)
            progress_callback: Called after each generation

        Returns:
            Evolution result with best genome
        """
        start_time = datetime.now()
        total_evaluations = 0
        convergence_generation = None

        # Initialize population
        population = self.population_manager.initialize_population(
            base_prompt=base_prompt,
            task_prompts=task_prompts,
        )

        # Evolution loop
        for generation in range(self.max_generations):
            # Evaluate all genomes
            for genome in population:
                result = await self.evaluate_genome(genome, tasks, evaluation_fn)
                genome.fitness = result.fitness
                total_evaluations += 1

            # Track best fitness
            best_fitness = max(g.fitness for g in population)
            self.fitness_history.append(best_fitness)

            # Track diversity
            diversity = self.calculate_diversity(population)
            self.diversity_history.append(diversity)

            # Progress callback
            if progress_callback:
                progress_callback({
                    "generation": generation,
                    "best_fitness": best_fitness,
                    "diversity": diversity,
                    "population_size": len(population),
                })

            # Check convergence
            if best_fitness >= self.convergence_threshold:
                convergence_generation = generation
                break

            # Evolution step based on strategy
            if self.strategy == EvolutionStrategy.GENETIC:
                population = self.population_manager.evolve_generation(generation + 1)

            elif self.strategy == EvolutionStrategy.GRADIENT and llm_fn:
                # Gradient optimization on top performers
                top_genomes = sorted(population, key=lambda g: g.fitness, reverse=True)[:5]
                improved = []
                for genome in top_genomes:
                    gradients = await self.gradient_optimizer.compute_gradient(
                        genome,
                        {"fitness": genome.fitness},
                        llm_fn,
                    )
                    improved_genome = await self.gradient_optimizer.apply_gradient(
                        genome, gradients, llm_fn
                    )
                    improved.append(improved_genome)
                population = improved + population[5:]

            elif self.strategy == EvolutionStrategy.HYBRID and llm_fn:
                # Genetic evolution
                population = self.population_manager.evolve_generation(generation + 1)

                # Gradient refinement on elite
                elite = population[0]
                gradients = await self.gradient_optimizer.compute_gradient(
                    elite,
                    {"fitness": elite.fitness},
                    llm_fn,
                )
                refined = await self.gradient_optimizer.apply_gradient(
                    elite, gradients, llm_fn, learning_rate=0.5
                )
                population[0] = refined

            else:
                # Default to genetic
                population = self.population_manager.evolve_generation(generation + 1)

        # Final evaluation
        for genome in population:
            if genome.fitness == 0:
                result = await self.evaluate_genome(genome, tasks, evaluation_fn)
                genome.fitness = result.fitness
                total_evaluations += 1

        # Find best genome
        best_genome = max(population, key=lambda g: g.fitness)

        evolution_time = (datetime.now() - start_time).total_seconds()

        return EvolutionResult(
            best_genome=best_genome,
            population=population,
            generations_completed=generation + 1,
            fitness_history=self.fitness_history,
            diversity_score=self.calculate_diversity(population),
            convergence_generation=convergence_generation,
            total_evaluations=total_evaluations,
            evolution_time_seconds=evolution_time,
            metadata={
                "strategy": self.strategy.value,
                "fitness_metric": self.fitness_metric.value,
                "population_size": self.population_size,
            },
        )

    async def evolve_from_archive(
        self,
        archive_path: str,
        tasks: List[Dict[str, Any]],
        evaluation_fn: Callable,
    ) -> EvolutionResult:
        """
        Continue evolution from a saved archive.

        Args:
            archive_path: Path to saved population
            tasks: Evaluation tasks
            evaluation_fn: Evaluation function

        Returns:
            Evolution result
        """
        # Load archive
        with open(archive_path, "r") as f:
            data = json.load(f)

        population = [AgentGenome.from_dict(g) for g in data["population"]]
        self.population_manager.population = population

        # Get base prompt from best genome
        best = max(population, key=lambda g: g.fitness)

        return await self.evolve(
            base_prompt=best.system_prompt,
            task_prompts=best.task_prompts,
            tasks=tasks,
            evaluation_fn=evaluation_fn,
        )

    def save_population(self, path: str):
        """Save current population to file."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "fitness_history": self.fitness_history,
            "population": [g.to_dict() for g in self.population_manager.population],
            "archive": [g.to_dict() for g in self.population_manager.archive],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def get_evolution_pipeline(**kwargs) -> AgentEvolutionPipeline:
    """Get configured evolution pipeline."""
    return AgentEvolutionPipeline(**kwargs)
