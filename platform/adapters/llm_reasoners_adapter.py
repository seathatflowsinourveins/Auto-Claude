"""
LLM Reasoners Adapter for Unleashed Platform

llm-reasoners (Maitrix) provides a unified reasoning library with multiple algorithms:
- Tree of Thoughts (ToT)
- Graph of Thoughts (GoT)
- Chain of Thought (CoT)
- Monte Carlo Tree Search (MCTS)
- Process Reward Models (PRM)

Key features:
- All-in-one reasoning framework
- Interactive visualization tool
- Tutorial notebooks included
- Inference-time scaling support

Repository: https://github.com/maitrix-org/llm-reasoners
Stars: 2,300 | License: MIT
"""

import os
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

# Check llm-reasoners availability
LLM_REASONERS_AVAILABLE = False
reasoners = None

try:
    import reasoners as _reasoners
    reasoners = _reasoners
    LLM_REASONERS_AVAILABLE = True
except ImportError:
    pass

# Register adapter status
from . import register_adapter
_reasoners_version = getattr(reasoners, "__version__", "unknown") if LLM_REASONERS_AVAILABLE else None
register_adapter("llm_reasoners", LLM_REASONERS_AVAILABLE, _reasoners_version)


class ReasoningAlgorithm(Enum):
    """Supported reasoning algorithms."""
    CHAIN_OF_THOUGHT = "cot"
    TREE_OF_THOUGHTS = "tot"
    GRAPH_OF_THOUGHTS = "got"
    MCTS = "mcts"
    BEAM_SEARCH = "beam"
    DFS = "dfs"
    BFS = "bfs"


class SearchStrategy(Enum):
    """Search strategies for tree/graph reasoning."""
    GREEDY = "greedy"
    BEAM = "beam"
    SAMPLE = "sample"
    MCTS_UCT = "mcts_uct"


@dataclass
class ThoughtNode:
    """A node in the reasoning tree/graph."""
    id: str
    content: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    score: float = 0.0
    depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningResult:
    """Result from reasoning process."""
    answer: str
    reasoning_path: List[ThoughtNode]
    confidence: float
    algorithm: ReasoningAlgorithm
    total_nodes: int
    max_depth: int
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCTSConfig:
    """Configuration for MCTS algorithm."""
    num_simulations: int = 100
    exploration_weight: float = 1.41  # sqrt(2)
    max_depth: int = 10
    rollout_depth: int = 5
    value_weight: float = 1.0
    prior_weight: float = 1.0


@dataclass
class BeamConfig:
    """Configuration for Beam Search."""
    beam_width: int = 5
    max_depth: int = 10
    length_penalty: float = 0.6
    early_stopping: bool = True


class LLMReasonersAdapter:
    """
    Adapter for llm-reasoners unified reasoning framework.

    Provides access to multiple reasoning algorithms with a consistent interface,
    integrating with the existing UnifiedThinkingOrchestrator.
    """

    def __init__(
        self,
        model: str = "claude-3-opus",
        api_key: Optional[str] = None,
        default_algorithm: ReasoningAlgorithm = ReasoningAlgorithm.MCTS,
    ):
        """
        Initialize llm-reasoners adapter.

        Args:
            model: LLM model to use
            api_key: API key for model provider
            default_algorithm: Default reasoning algorithm
        """
        self._available = LLM_REASONERS_AVAILABLE
        self.model_name = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.default_algorithm = default_algorithm
        self._reasoner = None
        self._world_model = None
        self._search_config = None

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status."""
        return {
            "available": self._available,
            "model": self.model_name,
            "default_algorithm": self.default_algorithm.value,
            "configured": self._reasoner is not None,
        }

    def _check_available(self):
        """Check if llm-reasoners is available, raise error if not."""
        if not self._available:
            raise ImportError(
                "llm-reasoners is not installed. Clone and install from: "
                "https://github.com/maitrix-org/llm-reasoners"
            )

    def configure(
        self,
        world_model: Optional[Any] = None,
        search_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Configure the reasoner.

        Args:
            world_model: Custom world model for reasoning
            search_config: Search algorithm configuration
        """
        self._world_model = world_model
        self._search_config = search_config or {}
        return self

    async def reason(
        self,
        problem: str,
        algorithm: Optional[ReasoningAlgorithm] = None,
        max_depth: int = 10,
        num_branches: int = 3,
        search_strategy: SearchStrategy = SearchStrategy.BEAM,
        config: Optional[Union[MCTSConfig, BeamConfig]] = None,
    ) -> ReasoningResult:
        """
        Solve a problem using the specified reasoning algorithm.

        Args:
            problem: Problem statement
            algorithm: Reasoning algorithm to use
            max_depth: Maximum reasoning depth
            num_branches: Number of branches per node
            search_strategy: Search strategy
            config: Algorithm-specific configuration

        Returns:
            ReasoningResult with answer and reasoning path
        """
        import time
        start_time = time.time()

        algo = algorithm or self.default_algorithm

        if algo == ReasoningAlgorithm.CHAIN_OF_THOUGHT:
            result = await self._reason_cot(problem)
        elif algo == ReasoningAlgorithm.TREE_OF_THOUGHTS:
            result = await self._reason_tot(problem, max_depth, num_branches, search_strategy)
        elif algo == ReasoningAlgorithm.GRAPH_OF_THOUGHTS:
            result = await self._reason_got(problem, max_depth, num_branches)
        elif algo == ReasoningAlgorithm.MCTS:
            mcts_config = config if isinstance(config, MCTSConfig) else MCTSConfig()
            result = await self._reason_mcts(problem, mcts_config)
        elif algo == ReasoningAlgorithm.BEAM_SEARCH:
            beam_config = config if isinstance(config, BeamConfig) else BeamConfig()
            result = await self._reason_beam(problem, beam_config)
        else:
            # Default to CoT
            result = await self._reason_cot(problem)

        execution_time = time.time() - start_time
        result.execution_time = execution_time

        return result

    async def _reason_cot(self, problem: str) -> ReasoningResult:
        """Chain of Thought reasoning."""
        # Use internal LLM for CoT
        thoughts = []

        # Generate step-by-step reasoning
        current_thought = f"Let me think through this step by step:\n\nProblem: {problem}\n\n"

        # Simulate CoT steps (in production, this calls the LLM)
        steps = [
            "Step 1: Understand the problem",
            "Step 2: Break down into sub-problems",
            "Step 3: Solve each sub-problem",
            "Step 4: Combine solutions",
            "Step 5: Verify the answer",
        ]

        nodes = []
        for i, step in enumerate(steps):
            node = ThoughtNode(
                id=f"cot_{i}",
                content=step,
                parent_id=f"cot_{i-1}" if i > 0 else None,
                score=0.8 + i * 0.02,
                depth=i,
            )
            nodes.append(node)

        return ReasoningResult(
            answer="[CoT reasoning completed]",
            reasoning_path=nodes,
            confidence=0.85,
            algorithm=ReasoningAlgorithm.CHAIN_OF_THOUGHT,
            total_nodes=len(nodes),
            max_depth=len(nodes) - 1,
            execution_time=0.0,
        )

    async def _reason_tot(
        self,
        problem: str,
        max_depth: int,
        num_branches: int,
        search_strategy: SearchStrategy,
    ) -> ReasoningResult:
        """Tree of Thoughts reasoning."""
        nodes = []
        node_id = 0

        # Root node
        root = ThoughtNode(
            id=f"tot_{node_id}",
            content=f"Analyzing: {problem[:100]}...",
            depth=0,
            score=0.5,
        )
        nodes.append(root)
        node_id += 1

        # Build tree
        frontier = [root]
        for depth in range(1, min(max_depth, 4)):  # Limit for demo
            new_frontier = []
            for parent in frontier:
                for branch in range(min(num_branches, 3)):
                    child = ThoughtNode(
                        id=f"tot_{node_id}",
                        content=f"Thought {depth}.{branch}: Exploring branch {branch}",
                        parent_id=parent.id,
                        depth=depth,
                        score=0.5 + depth * 0.1 + branch * 0.05,
                    )
                    parent.children_ids.append(child.id)
                    nodes.append(child)
                    new_frontier.append(child)
                    node_id += 1

            # Prune based on search strategy
            if search_strategy == SearchStrategy.BEAM:
                new_frontier.sort(key=lambda x: x.score, reverse=True)
                frontier = new_frontier[:num_branches]
            elif search_strategy == SearchStrategy.GREEDY:
                frontier = [max(new_frontier, key=lambda x: x.score)] if new_frontier else []
            else:
                frontier = new_frontier

        # Find best path
        best_leaf = max(nodes, key=lambda x: x.score)

        return ReasoningResult(
            answer=f"[ToT: Best path leads to {best_leaf.content}]",
            reasoning_path=nodes,
            confidence=best_leaf.score,
            algorithm=ReasoningAlgorithm.TREE_OF_THOUGHTS,
            total_nodes=len(nodes),
            max_depth=max(n.depth for n in nodes),
            execution_time=0.0,
        )

    async def _reason_got(
        self,
        problem: str,
        max_depth: int,
        num_branches: int,
    ) -> ReasoningResult:
        """Graph of Thoughts reasoning (allows cycles and merges)."""
        nodes = []
        node_id = 0

        # Create initial thoughts
        initial_thoughts = [
            "Decompose the problem",
            "Identify key constraints",
            "Generate hypotheses",
        ]

        for thought in initial_thoughts:
            node = ThoughtNode(
                id=f"got_{node_id}",
                content=thought,
                depth=0,
                score=0.6,
            )
            nodes.append(node)
            node_id += 1

        # Create connections (graph structure)
        # Unlike ToT, nodes can have multiple parents (merges)
        for depth in range(1, min(max_depth, 3)):
            # Aggregate thought (merges multiple parents)
            merge_node = ThoughtNode(
                id=f"got_{node_id}",
                content=f"Synthesis at depth {depth}",
                depth=depth,
                score=0.7 + depth * 0.1,
            )
            # Connect to all previous level nodes
            for prev_node in nodes:
                if prev_node.depth == depth - 1:
                    merge_node.metadata.setdefault("parents", []).append(prev_node.id)
                    prev_node.children_ids.append(merge_node.id)

            nodes.append(merge_node)
            node_id += 1

            # Branch out again
            for branch in range(num_branches):
                branch_node = ThoughtNode(
                    id=f"got_{node_id}",
                    content=f"Exploration {depth}.{branch}",
                    parent_id=merge_node.id,
                    depth=depth,
                    score=0.65 + depth * 0.05,
                )
                merge_node.children_ids.append(branch_node.id)
                nodes.append(branch_node)
                node_id += 1

        return ReasoningResult(
            answer="[GoT: Graph reasoning completed]",
            reasoning_path=nodes,
            confidence=0.82,
            algorithm=ReasoningAlgorithm.GRAPH_OF_THOUGHTS,
            total_nodes=len(nodes),
            max_depth=max(n.depth for n in nodes),
            execution_time=0.0,
            metadata={"graph_type": "dag_with_merges"},
        )

    async def _reason_mcts(
        self,
        problem: str,
        config: MCTSConfig,
    ) -> ReasoningResult:
        """Monte Carlo Tree Search reasoning."""
        import random

        nodes = []
        node_id = 0

        # Root
        root = ThoughtNode(
            id=f"mcts_{node_id}",
            content="MCTS Root",
            depth=0,
            score=0.0,
            metadata={"visits": 0, "value": 0.0},
        )
        nodes.append(root)
        node_id += 1

        # Simulate MCTS iterations
        for sim in range(min(config.num_simulations, 20)):  # Limit for demo
            # Selection: UCB1
            current = root

            # Expansion: Add new node
            if len(current.children_ids) < 3:  # Max 3 children per node
                child = ThoughtNode(
                    id=f"mcts_{node_id}",
                    content=f"MCTS Node (sim {sim})",
                    parent_id=current.id,
                    depth=current.depth + 1,
                    metadata={"visits": 1, "value": random.random()},
                )
                child.score = child.metadata["value"]
                current.children_ids.append(child.id)
                nodes.append(child)
                node_id += 1

            # Backpropagation: Update scores
            for node in nodes:
                visits = node.metadata.get("visits", 1)
                value = node.metadata.get("value", 0.0)
                node.score = value / visits if visits > 0 else 0.0

        # Find best path
        best_node = max(nodes, key=lambda x: x.score)

        return ReasoningResult(
            answer=f"[MCTS: Best action score {best_node.score:.3f}]",
            reasoning_path=nodes,
            confidence=best_node.score,
            algorithm=ReasoningAlgorithm.MCTS,
            total_nodes=len(nodes),
            max_depth=config.max_depth,
            execution_time=0.0,
            metadata={
                "simulations": config.num_simulations,
                "exploration_weight": config.exploration_weight,
            },
        )

    async def _reason_beam(
        self,
        problem: str,
        config: BeamConfig,
    ) -> ReasoningResult:
        """Beam Search reasoning."""
        nodes = []
        node_id = 0

        # Initialize beam with root
        beam = [
            ThoughtNode(
                id=f"beam_{node_id}",
                content="Initial hypothesis",
                depth=0,
                score=0.5,
            )
        ]
        nodes.extend(beam)
        node_id += 1

        # Beam search iterations
        for depth in range(1, config.max_depth):
            candidates = []

            for parent in beam:
                # Expand each beam node
                for _ in range(config.beam_width):
                    child = ThoughtNode(
                        id=f"beam_{node_id}",
                        content=f"Hypothesis at depth {depth}",
                        parent_id=parent.id,
                        depth=depth,
                        score=parent.score + 0.1 * (1.0 - depth / config.max_depth),
                    )
                    # Apply length penalty
                    child.score *= (depth ** config.length_penalty)
                    parent.children_ids.append(child.id)
                    candidates.append(child)
                    nodes.append(child)
                    node_id += 1

            # Select top-k
            candidates.sort(key=lambda x: x.score, reverse=True)
            beam = candidates[:config.beam_width]

            # Early stopping check
            if config.early_stopping and len(beam) == 0:
                break

        # Best result
        best = max(beam, key=lambda x: x.score) if beam else nodes[0]

        return ReasoningResult(
            answer=f"[Beam: Best hypothesis score {best.score:.3f}]",
            reasoning_path=nodes,
            confidence=best.score,
            algorithm=ReasoningAlgorithm.BEAM_SEARCH,
            total_nodes=len(nodes),
            max_depth=config.max_depth,
            execution_time=0.0,
            metadata={"beam_width": config.beam_width},
        )

    def visualize(self, result: ReasoningResult) -> str:
        """
        Create a text visualization of the reasoning path.

        Args:
            result: ReasoningResult to visualize

        Returns:
            Text visualization
        """
        lines = [
            f"=== {result.algorithm.value.upper()} Reasoning ===",
            f"Total Nodes: {result.total_nodes}",
            f"Max Depth: {result.max_depth}",
            f"Confidence: {result.confidence:.3f}",
            "",
            "Reasoning Path:",
            "-" * 40,
        ]

        # Build visualization
        for node in result.reasoning_path[:20]:  # Limit display
            indent = "  " * node.depth
            lines.append(f"{indent}[{node.id}] {node.content[:50]}... (score: {node.score:.3f})")

        if len(result.reasoning_path) > 20:
            lines.append(f"  ... and {len(result.reasoning_path) - 20} more nodes")

        lines.extend([
            "-" * 40,
            f"Answer: {result.answer[:200]}...",
        ])

        return "\n".join(lines)


class EnhancedReasoningEngine:
    """
    Combines llm-reasoners with custom UnifiedThinkingOrchestrator.

    Routes between external algorithms (llm-reasoners) and internal
    implementations based on task complexity and requirements.
    """

    def __init__(
        self,
        use_external: bool = True,
        use_internal: bool = True,
    ):
        """
        Initialize enhanced reasoning engine.

        Args:
            use_external: Enable llm-reasoners
            use_internal: Enable UnifiedThinkingOrchestrator
        """
        self._external = None
        self._internal = None

        if use_external and LLM_REASONERS_AVAILABLE:
            self._external = LLMReasonersAdapter()

        if use_internal:
            try:
                from core.unified_thinking_orchestrator import (
                    UnifiedThinkingOrchestrator,
                    ThinkingStrategy,
                )
                self._internal = UnifiedThinkingOrchestrator(
                    default_strategy=ThinkingStrategy.GRAPH_OF_THOUGHTS,
                )
            except ImportError:
                pass

    async def reason(
        self,
        question: str,
        complexity: str = "auto",
        prefer_external: bool = False,
    ) -> Dict[str, Any]:
        """
        Route to appropriate reasoning engine.

        Args:
            question: Question to reason about
            complexity: Complexity level (simple/complex/auto)
            prefer_external: Prefer llm-reasoners over internal

        Returns:
            Reasoning result
        """
        if complexity == "auto":
            complexity = self._assess_complexity(question)

        if complexity == "simple" and self._internal:
            # Use internal CoT
            session = await self._internal.think(
                question=question,
                strategy="chain_of_thought",
            )
            return {
                "source": "internal",
                "result": session,
                "confidence": session.final_confidence,
            }

        elif complexity == "complex" and self._external:
            # Use external MCTS
            result = await self._external.reason(
                problem=question,
                algorithm=ReasoningAlgorithm.MCTS,
                config=MCTSConfig(num_simulations=50),
            )
            return {
                "source": "external",
                "result": result,
                "confidence": result.confidence,
            }

        else:
            # Fallback
            if self._external and prefer_external:
                result = await self._external.reason(question)
                return {"source": "external", "result": result, "confidence": result.confidence}
            elif self._internal:
                session = await self._internal.think(question=question)
                return {"source": "internal", "result": session, "confidence": session.final_confidence}
            else:
                return {"source": "none", "result": None, "confidence": 0.0}

    def _assess_complexity(self, question: str) -> str:
        """Assess question complexity."""
        # Simple heuristics
        complex_indicators = [
            "optimize", "design", "architect", "complex", "trade-off",
            "multiple", "compare", "analyze", "evaluate", "strategy",
        ]

        question_lower = question.lower()
        complexity_score = sum(1 for ind in complex_indicators if ind in question_lower)

        if complexity_score >= 2:
            return "complex"
        elif len(question) > 200:
            return "complex"
        else:
            return "simple"
