"""
V8 MCTS & Multi-Agent Self-Play

Extracted from ralph_loop.py V8 enhancements.
Implements Monte Carlo Tree Search and Multi-Agent Self-Play.

Classes:
- MCTSNode: MCTS node with UCB1/PUCT selection (SRA-MCTS, IJCAI 2025)
- MCTSState: Overall MCTS state management
- SelfPlayAgent: Agent perspective for self-play (MARSHAL pattern)
- SelfPlayState: Multi-agent competitive improvement
- StrategistState: Bi-level MCTS for strategy optimization

References:
- SRA-MCTS (IJCAI 2025)
- AlphaZero PUCT (Silver et al.)
- MARSHAL: Multi-Agent Reasoning via Self-Play
- Strategist: Bi-Level MCTS
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MCTSNode:
    """
    V8: Monte Carlo Tree Search node for exploration.

    Based on SRA-MCTS (IJCAI 2025) and Strategist (bi-level MCTS).
    Uses UCB1 formula for selection: value/visits + c * sqrt(ln(parent_visits) / visits)

    Attributes:
        node_id: Unique identifier for this node
        state: Serialized state representation
        parent_id: ID of parent node (None for root)
        children_ids: IDs of child nodes
        visits: Number of times node was visited
        total_value: Sum of backpropagated values
        prior_probability: Prior from policy network or heuristic
        depth: Depth in tree (root = 0)
        action_taken: Action that led to this state
        is_terminal: Whether this is a terminal state
        created_at: ISO timestamp of creation

    Methods:
        q_value: Average value (exploitation term)
        ucb1_score: UCB1 selection score
        puct_score: AlphaZero-style PUCT score

    Example:
        node = MCTSNode(node_id="root", state="{...}")
        score = node.ucb1_score(parent_visits=100)
    """
    node_id: str
    state: str  # Serialized state representation
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    prior_probability: float = 0.5  # From policy network or heuristic
    depth: int = 0
    action_taken: str = ""  # Action that led to this state
    is_terminal: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def q_value(self) -> float:
        """Average value (exploitation term)."""
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def ucb1_score(self, parent_visits: int, exploration_constant: float = 1.414) -> float:
        """
        Calculate UCB1 score for node selection.

        UCB1 = Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))
        where Q is average value, N(s) is parent visits, N(s,a) is this node visits

        Args:
            parent_visits: Visit count of parent node
            exploration_constant: Exploration parameter (default sqrt(2))

        Returns:
            UCB1 score (higher = should be explored)
        """
        if self.visits == 0:
            return float('inf')  # Always explore unvisited nodes

        exploitation = self.q_value
        exploration = exploration_constant * math.sqrt(math.log(parent_visits + 1) / self.visits)
        return exploitation + exploration

    def puct_score(self, parent_visits: int, c_puct: float = 1.5) -> float:
        """
        Calculate PUCT score (AlphaZero-style with prior).

        PUCT = Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Args:
            parent_visits: Visit count of parent node
            c_puct: PUCT exploration constant

        Returns:
            PUCT score (higher = should be explored)
        """
        if self.visits == 0:
            return float('inf')

        exploitation = self.q_value
        exploration = c_puct * self.prior_probability * math.sqrt(parent_visits) / (1 + self.visits)
        return exploitation + exploration


@dataclass
class MCTSState:
    """
    V8: Overall MCTS state management for tree search exploration.

    Implements:
    - Tree construction and traversal
    - Selection, expansion, simulation, backpropagation
    - Progressive widening for continuous action spaces

    Attributes:
        root_id: ID of root node
        nodes: Dict mapping node IDs to nodes
        max_depth: Maximum tree depth
        max_simulations: Maximum simulations to run
        simulations_done: Simulations completed so far
        exploration_constant: UCB1 exploration parameter
        progressive_widening_alpha: For continuous action spaces
        best_path: Best path found so far
        best_value: Best value achieved

    Example:
        mcts = MCTSState(max_simulations=100)
        root = MCTSNode(node_id="root", state="initial")
        mcts.add_node(root)
    """
    root_id: str = ""
    nodes: Dict[str, MCTSNode] = field(default_factory=dict)
    max_depth: int = 10
    max_simulations: int = 50
    simulations_done: int = 0
    exploration_constant: float = 1.414
    progressive_widening_alpha: float = 0.5  # For continuous action spaces
    best_path: List[str] = field(default_factory=list)
    best_value: float = 0.0

    def add_node(self, node: MCTSNode) -> None:
        """Add a node to the tree."""
        self.nodes[node.node_id] = node
        if not self.root_id:
            self.root_id = node.node_id

    def get_node(self, node_id: str) -> Optional[MCTSNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def select_child(self, node_id: str) -> Optional[str]:
        """
        Select best child using UCB1.

        Args:
            node_id: ID of parent node

        Returns:
            ID of selected child, or None if no children
        """
        node = self.nodes.get(node_id)
        if not node or not node.children_ids:
            return None

        best_score = float('-inf')
        best_child = None

        for child_id in node.children_ids:
            child = self.nodes.get(child_id)
            if child:
                score = child.ucb1_score(node.visits, self.exploration_constant)
                if score > best_score:
                    best_score = score
                    best_child = child_id

        return best_child

    def backpropagate(self, node_id: str, value: float) -> None:
        """
        Backpropagate value up the tree.

        Args:
            node_id: Starting node for backpropagation
            value: Value to propagate
        """
        current_id = node_id
        while current_id:
            node = self.nodes.get(current_id)
            if not node:
                break
            node.visits += 1
            node.total_value += value
            current_id = node.parent_id

    def get_best_action(self) -> Tuple[str, float]:
        """
        Get the best action from root based on visit counts.

        Returns:
            Tuple of (action_taken, q_value) for best child
        """
        root = self.nodes.get(self.root_id)
        if not root or not root.children_ids:
            return ("", 0.0)

        best_visits = -1
        best_child = None

        for child_id in root.children_ids:
            child = self.nodes.get(child_id)
            if child and child.visits > best_visits:
                best_visits = child.visits
                best_child = child

        if best_child:
            return (best_child.action_taken, best_child.q_value)
        return ("", 0.0)

    def get_tree_stats(self) -> Dict[str, Any]:
        """Get statistics about the search tree."""
        depths = [n.depth for n in self.nodes.values()]
        visits = [n.visits for n in self.nodes.values()]
        return {
            "total_nodes": len(self.nodes),
            "max_depth_reached": max(depths) if depths else 0,
            "total_simulations": self.simulations_done,
            "avg_visits_per_node": sum(visits) / len(visits) if visits else 0,
            "best_action": self.get_best_action()[0],
            "best_value": self.best_value
        }


@dataclass
class SelfPlayAgent:
    """
    V8: An agent perspective for multi-agent self-play (MARSHAL pattern).

    Agents with different perspectives compete to find the best strategy.
    Uses Elo ratings to track relative performance.

    Attributes:
        agent_id: Unique identifier
        perspective: Strategy type ("exploiter", "explorer", "conservative", "aggressive")
        strategy: Description of strategy
        fitness_achieved: Latest fitness score
        rounds_played: Total rounds participated
        wins: Total wins
        elo_rating: Chess-style Elo rating (starts at 1500)

    Example:
        agent = SelfPlayAgent(
            agent_id="agent_1",
            perspective="explorer",
            strategy="Try novel combinations"
        )
        agent.update_elo(opponent_elo=1600, won=True)
    """
    agent_id: str
    perspective: str  # "exploiter", "explorer", "conservative", "aggressive"
    strategy: str
    fitness_achieved: float = 0.0
    rounds_played: int = 0
    wins: int = 0
    elo_rating: float = 1500.0  # Chess-style Elo rating

    def update_elo(self, opponent_elo: float, won: bool, k_factor: float = 32.0) -> None:
        """
        Update Elo rating after a match.

        Args:
            opponent_elo: Opponent's Elo rating
            won: Whether this agent won
            k_factor: Elo update sensitivity (default 32)
        """
        expected = 1 / (1 + 10 ** ((opponent_elo - self.elo_rating) / 400))
        actual = 1.0 if won else 0.0
        self.elo_rating += k_factor * (actual - expected)

    @property
    def win_rate(self) -> float:
        """Win rate over all rounds played."""
        return self.wins / self.rounds_played if self.rounds_played > 0 else 0.0


@dataclass
class SelfPlayState:
    """
    V8: Multi-Agent Self-Play state for competitive improvement.

    Based on MARSHAL (Multi-Agent Reasoning via Self-Play) and
    Strategist bi-level MCTS improvement patterns.

    Agents with different strategies compete, winner's approach is reinforced.

    Attributes:
        agents: List of competing agents
        rounds_completed: Total tournament rounds
        max_rounds: Maximum rounds to play
        tournament_history: History of round results
        best_strategy_found: Current best strategy
        population_diversity: Measure of strategy diversity

    Example:
        state = SelfPlayState()
        state.initialize_agents()
        result = state.run_tournament_round({"agent_0": 0.8, "agent_1": 0.6})
    """
    agents: List[SelfPlayAgent] = field(default_factory=list)
    rounds_completed: int = 0
    max_rounds: int = 10
    tournament_history: List[Dict[str, Any]] = field(default_factory=list)
    best_strategy_found: str = ""
    population_diversity: float = 1.0  # Measure of strategy diversity

    def initialize_agents(self) -> None:
        """Initialize a diverse set of competing agents."""
        perspectives = [
            ("exploiter", "Focus on known good strategies, optimize incrementally"),
            ("explorer", "Try novel combinations, high variance approaches"),
            ("conservative", "Minimize risk, prefer stable improvements"),
            ("aggressive", "Push boundaries, accept higher failure rate for breakthroughs")
        ]

        self.agents = []
        for i, (perspective, strategy) in enumerate(perspectives):
            self.agents.append(SelfPlayAgent(
                agent_id=f"agent_{i}",
                perspective=perspective,
                strategy=strategy
            ))

    def run_tournament_round(self, fitness_results: Dict[str, float]) -> Dict[str, Any]:
        """
        Run a tournament round and update agent ratings.

        Args:
            fitness_results: Dict mapping agent_id to fitness score

        Returns:
            Dict with round results including winner and Elo rankings
        """
        self.rounds_completed += 1

        # Update agent fitness scores
        for agent in self.agents:
            if agent.agent_id in fitness_results:
                agent.fitness_achieved = fitness_results[agent.agent_id]
                agent.rounds_played += 1

        # Determine round winner
        if self.agents:
            sorted_agents = sorted(self.agents, key=lambda a: a.fitness_achieved, reverse=True)
            winner = sorted_agents[0]
            winner.wins += 1

            # Update Elo ratings (pairwise)
            for i, agent_a in enumerate(sorted_agents):
                for agent_b in sorted_agents[i+1:]:
                    agent_a.update_elo(agent_b.elo_rating, won=True)
                    agent_b.update_elo(agent_a.elo_rating, won=False)

            self.best_strategy_found = winner.strategy

            round_result = {
                "round": self.rounds_completed,
                "winner": winner.agent_id,
                "winner_perspective": winner.perspective,
                "winner_fitness": winner.fitness_achieved,
                "elo_rankings": [(a.agent_id, a.elo_rating) for a in sorted(self.agents, key=lambda x: x.elo_rating, reverse=True)]
            }
            self.tournament_history.append(round_result)
            return round_result

        return {"round": self.rounds_completed, "winner": None}

    def get_consensus_strategy(self) -> str:
        """
        Get the consensus strategy weighted by Elo ratings.

        Returns:
            String describing top strategies weighted by Elo
        """
        if not self.agents:
            return "balanced exploration and exploitation"

        # Weight strategies by Elo rating
        total_elo = sum(a.elo_rating for a in self.agents)
        weighted_strategies = []
        for agent in sorted(self.agents, key=lambda a: a.elo_rating, reverse=True):
            weight = agent.elo_rating / total_elo if total_elo > 0 else 0.25
            weighted_strategies.append(f"{agent.perspective} ({weight:.1%}): {agent.strategy}")

        return " | ".join(weighted_strategies[:2])  # Top 2 strategies

    def calculate_diversity(self) -> float:
        """
        Calculate diversity of strategies in the population.

        Returns:
            Diversity score (0.0 to 1.0, higher = more diverse)
        """
        if len(self.agents) < 2:
            return 1.0

        # Simple diversity: variance in Elo ratings normalized
        elos = [a.elo_rating for a in self.agents]
        mean_elo = sum(elos) / len(elos)
        variance = sum((e - mean_elo) ** 2 for e in elos) / len(elos)
        # Normalize: high variance = low diversity (one dominant strategy)
        self.population_diversity = 1.0 / (1.0 + variance / 10000)
        return self.population_diversity


@dataclass
class StrategistState:
    """
    V8: Bi-Level MCTS state for Strategist pattern.

    Based on Strategist paper: Optimizes BOTH the solution AND the search strategy.

    Level 1 (Inner): MCTS for solution exploration
    Level 2 (Outer): Meta-MCTS for search strategy optimization

    Attributes:
        inner_mcts: Solution-level search state
        outer_mcts: Strategy-level search state
        current_search_params: Current search parameters
        param_history: History of (params, score) pairs
        meta_iterations: Number of meta-level iterations

    Example:
        strategist = StrategistState()
        strategist.initialize()
        params = strategist.get_recommended_params()
    """
    inner_mcts: Optional[MCTSState] = None  # Solution-level search
    outer_mcts: Optional[MCTSState] = None  # Strategy-level search
    current_search_params: Dict[str, float] = field(default_factory=lambda: {
        "exploration_constant": 1.414,
        "max_depth": 10,
        "simulation_budget": 50,
        "progressive_widening": 0.5
    })
    param_history: List[Tuple[Dict[str, float], float]] = field(default_factory=list)
    meta_iterations: int = 0

    def initialize(self) -> None:
        """Initialize both MCTS levels."""
        self.inner_mcts = MCTSState(
            exploration_constant=self.current_search_params["exploration_constant"],
            max_depth=int(self.current_search_params["max_depth"]),
            max_simulations=int(self.current_search_params["simulation_budget"]),
            progressive_widening_alpha=self.current_search_params["progressive_widening"]
        )
        self.outer_mcts = MCTSState(
            exploration_constant=1.0,  # More exploitation at meta-level
            max_depth=5,
            max_simulations=10
        )

    def update_from_inner_result(self, inner_value: float) -> None:
        """
        Update outer MCTS based on inner search performance.

        Args:
            inner_value: Value achieved by inner search
        """
        self.param_history.append((self.current_search_params.copy(), inner_value))
        self.meta_iterations += 1

        # Outer MCTS would suggest new params (simplified here)
        if len(self.param_history) >= 3:
            recent = self.param_history[-3:]
            best_params, best_value = max(recent, key=lambda x: x[1])

            # Slightly mutate best parameters
            self.current_search_params = {
                k: v * random.uniform(0.9, 1.1)
                for k, v in best_params.items()
            }
            # Clamp values
            self.current_search_params["exploration_constant"] = max(0.5, min(3.0, self.current_search_params["exploration_constant"]))
            self.current_search_params["max_depth"] = max(3, min(20, int(self.current_search_params["max_depth"])))

    def get_recommended_params(self) -> Dict[str, float]:
        """Get current recommended search parameters."""
        return self.current_search_params.copy()
