#!/usr/bin/env python3
"""
Knowledge Layer - Graph RAG & Quality-Diversity Optimization
Part of the V33 Architecture (Layer 8) - Phase 9 Production Fix.

Provides unified access to two knowledge SDKs:
- graphrag: Microsoft's Graph-based RAG for knowledge extraction
- pyribs: Quality-Diversity optimization with MAP-Elites

NO STUBS: All SDKs must be explicitly installed and configured.
Missing SDKs raise SDKNotAvailableError with install instructions.
Misconfigured SDKs raise SDKConfigurationError with missing config.

Usage:
    from core.knowledge import (
        # Exceptions
        SDKNotAvailableError,
        SDKConfigurationError,

        # GraphRAG
        get_graphrag_client,
        GraphRAGClient,
        GRAPHRAG_AVAILABLE,

        # PyRibs
        get_pyribs_archive,
        PyRibsClient,
        PYRIBS_AVAILABLE,

        # Factory
        KnowledgeFactory,
    )

    # Quick start with explicit error handling
    try:
        graph = get_graphrag_client()
    except SDKNotAvailableError as e:
        print(f"Install: {e.install_cmd}")
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Optional, List, Dict, Tuple, Callable, Union
from dataclasses import dataclass, field
import numpy as np

# Import exceptions from observability layer
from core.observability import (
    SDKNotAvailableError,
    SDKConfigurationError,
)


# ============================================================================
# SDK Availability Checks - Import-time validation
# ============================================================================

# GraphRAG - Knowledge Graph RAG
GRAPHRAG_AVAILABLE = False
GRAPHRAG_ERROR = None
try:
    from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
    from graphrag.query.indexer_adapters import (
        read_indexer_entities,
        read_indexer_relationships,
        read_indexer_reports,
        read_indexer_text_units,
    )
    from graphrag.query.llm.oai.chat_openai import ChatOpenAI
    from graphrag.query.llm.oai.typing import OpenaiApiType
    from graphrag.query.structured_search.global_search.community_context import (
        GlobalCommunityContext,
    )
    from graphrag.query.structured_search.global_search.search import GlobalSearch
    from graphrag.query.structured_search.local_search.mixed_context import (
        LocalSearchMixedContext,
    )
    from graphrag.query.structured_search.local_search.search import LocalSearch
    GRAPHRAG_AVAILABLE = True
except Exception as e:
    GRAPHRAG_ERROR = str(e)

# PyRibs - Quality-Diversity
PYRIBS_AVAILABLE = False
PYRIBS_ERROR = None
try:
    from ribs.archives import GridArchive, CVTArchive
    from ribs.emitters import EvolutionStrategyEmitter, GaussianEmitter
    from ribs.schedulers import Scheduler
    from ribs.visualize import grid_archive_heatmap, cvt_archive_heatmap
    PYRIBS_AVAILABLE = True
except Exception as e:
    PYRIBS_ERROR = str(e)


# ============================================================================
# GraphRAG Types and Implementation
# ============================================================================

class SearchMode(str, Enum):
    """GraphRAG search modes."""
    LOCAL = "local"      # Entity-centric search
    GLOBAL = "global"    # Community-based search
    HYBRID = "hybrid"    # Combined approach


@dataclass
class GraphRAGConfig:
    """Configuration for GraphRAG."""
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    index_path: Optional[str] = None
    community_level: int = 2
    max_tokens: int = 4096


@dataclass
class Entity:
    """A knowledge graph entity."""
    id: str
    name: str
    type: str
    description: str
    rank: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    """A knowledge graph relationship."""
    source: str
    target: str
    type: str
    description: str
    weight: float = 1.0


@dataclass
class SearchResult:
    """Result from a GraphRAG search."""
    answer: str
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    context_text: str = ""
    sources: List[str] = field(default_factory=list)
    score: float = 0.0


class GraphRAGClient:
    """
    GraphRAG client for knowledge graph-based RAG.

    Provides entity extraction, community detection,
    and structured search over knowledge graphs.
    """

    def __init__(
        self,
        config: Optional[GraphRAGConfig] = None,
        **kwargs: Any,
    ):
        """
        Initialize GraphRAG client.

        Args:
            config: Optional GraphRAGConfig
            **kwargs: Override config values
        """
        if not GRAPHRAG_AVAILABLE:
            raise SDKNotAvailableError(
                sdk_name="graphrag",
                install_cmd="pip install graphrag>=0.3.0",
                docs_url="https://microsoft.github.io/graphrag/"
            )

        self.config = config or GraphRAGConfig()
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self._llm = None
        self._local_search = None
        self._global_search = None
        self._entities = []
        self._relationships = []

        self._init_llm()

    def _init_llm(self) -> None:
        """Initialize the LLM for GraphRAG."""
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SDKConfigurationError(
                sdk_name="graphrag",
                missing_config=["OPENAI_API_KEY"],
                example="OPENAI_API_KEY=sk-..."
            )

        self._llm = ChatOpenAI(
            api_key=api_key,
            api_base=self.config.api_base,
            model=self.config.model,
            api_type=OpenaiApiType.OpenAI,
            max_tokens=self.config.max_tokens,
        )

    def load_index(self, index_path: str) -> None:
        """
        Load a pre-built GraphRAG index.

        Args:
            index_path: Path to index directory
        """
        import pandas as pd

        # Load entities
        entity_df = pd.read_parquet(f"{index_path}/create_final_entities.parquet")
        self._entities = read_indexer_entities(entity_df, None, 2)

        # Load relationships
        rel_df = pd.read_parquet(f"{index_path}/create_final_relationships.parquet")
        self._relationships = read_indexer_relationships(rel_df)

        # Load text units
        text_df = pd.read_parquet(f"{index_path}/create_final_text_units.parquet")
        text_units = read_indexer_text_units(text_df)

        # Load community reports for global search
        report_df = pd.read_parquet(f"{index_path}/create_final_community_reports.parquet")
        reports = read_indexer_reports(report_df, None, self.config.community_level)

        # Initialize local search
        local_context = LocalSearchMixedContext(
            entities=self._entities,
            relationships=self._relationships,
            text_units=text_units,
            entity_text_embeddings=None,  # Would need embedding store
        )
        self._local_search = LocalSearch(
            llm=self._llm,
            context_builder=local_context,
            token_encoder=None,
        )

        # Initialize global search
        global_context = GlobalCommunityContext(
            community_reports=reports,
            entities=self._entities,
        )
        self._global_search = GlobalSearch(
            llm=self._llm,
            context_builder=global_context,
            token_encoder=None,
        )

    def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.LOCAL,
        top_k: int = 10,
    ) -> SearchResult:
        """
        Search the knowledge graph.

        Args:
            query: Search query
            mode: Search mode (local, global, hybrid)
            top_k: Number of results

        Returns:
            SearchResult with answer and context
        """
        if mode == SearchMode.LOCAL and self._local_search:
            result = self._local_search.search(query)
        elif mode == SearchMode.GLOBAL and self._global_search:
            result = self._global_search.search(query)
        else:
            # Fallback to local if available
            if self._local_search:
                result = self._local_search.search(query)
            else:
                return SearchResult(
                    answer="No search index loaded. Call load_index() first.",
                )

        return SearchResult(
            answer=result.response,
            context_text=result.context_text if hasattr(result, "context_text") else "",
            sources=result.sources if hasattr(result, "sources") else [],
        )

    def get_entities(
        self,
        entity_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Entity]:
        """
        Get entities from the knowledge graph.

        Args:
            entity_type: Filter by entity type
            limit: Maximum entities to return

        Returns:
            List of Entity objects
        """
        entities = []
        for e in self._entities[:limit]:
            if entity_type and e.type != entity_type:
                continue
            entities.append(Entity(
                id=str(e.id),
                name=e.title,
                type=e.type or "unknown",
                description=e.description or "",
                rank=getattr(e, "rank", 0.0),
            ))
        return entities

    def get_relationships(
        self,
        source_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Relationship]:
        """
        Get relationships from the knowledge graph.

        Args:
            source_id: Filter by source entity
            limit: Maximum relationships to return

        Returns:
            List of Relationship objects
        """
        relationships = []
        for r in self._relationships[:limit]:
            if source_id and str(r.source) != source_id:
                continue
            relationships.append(Relationship(
                source=str(r.source),
                target=str(r.target),
                type=r.type or "related_to",
                description=r.description or "",
                weight=getattr(r, "weight", 1.0),
            ))
        return relationships


# ============================================================================
# PyRibs Types and Implementation
# ============================================================================

class ArchiveType(str, Enum):
    """PyRibs archive types."""
    GRID = "grid"    # Regular grid discretization
    CVT = "cvt"      # Centroidal Voronoi Tessellation


class EmitterType(str, Enum):
    """PyRibs emitter types."""
    GAUSSIAN = "gaussian"  # Gaussian perturbation
    ES = "es"              # Evolution Strategy (CMA-ES)
    ISO_LINE = "iso_line"  # Isotropic line sampling


@dataclass
class PyRibsConfig:
    """Configuration for PyRibs."""
    archive_type: ArchiveType = ArchiveType.GRID
    solution_dim: int = 10
    behavior_dim: int = 2
    grid_dims: Tuple[int, ...] = (100, 100)
    behavior_ranges: Tuple[Tuple[float, float], ...] = ((0, 1), (0, 1))
    seed: int = 42


@dataclass
class Solution:
    """A solution in the archive."""
    id: int
    parameters: np.ndarray
    objective: float
    behavior: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchiveStats:
    """Statistics about the archive."""
    num_elites: int
    coverage: float
    qd_score: float
    max_objective: float
    mean_objective: float


class PyRibsClient:
    """
    PyRibs client for Quality-Diversity optimization.

    Implements MAP-Elites and related algorithms for
    discovering diverse, high-quality solutions.
    """

    def __init__(
        self,
        config: Optional[PyRibsConfig] = None,
        **kwargs: Any,
    ):
        """
        Initialize PyRibs client.

        Args:
            config: Optional PyRibsConfig
            **kwargs: Override config values
        """
        if not PYRIBS_AVAILABLE:
            raise SDKNotAvailableError(
                sdk_name="pyribs",
                install_cmd="pip install ribs>=0.7.0",
                docs_url="https://docs.pyribs.org/"
            )

        self.config = config or PyRibsConfig()
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self._archive = None
        self._emitters = []
        self._scheduler = None

        self._init_archive()

    def _init_archive(self) -> None:
        """Initialize the archive based on config."""
        if self.config.archive_type == ArchiveType.GRID:
            self._archive = GridArchive(
                solution_dim=self.config.solution_dim,
                dims=self.config.grid_dims,
                ranges=self.config.behavior_ranges,
                seed=self.config.seed,
            )
        else:  # CVT
            # Calculate cells from grid dims
            num_cells = 1
            for d in self.config.grid_dims:
                num_cells *= d

            self._archive = CVTArchive(
                solution_dim=self.config.solution_dim,
                cells=num_cells,
                ranges=self.config.behavior_ranges,
                seed=self.config.seed,
            )

    def add_emitter(
        self,
        emitter_type: EmitterType = EmitterType.GAUSSIAN,
        sigma: float = 0.1,
        batch_size: int = 32,
    ) -> None:
        """
        Add an emitter to the optimization.

        Args:
            emitter_type: Type of emitter
            sigma: Mutation strength
            batch_size: Solutions per batch
        """
        if emitter_type == EmitterType.GAUSSIAN:
            emitter = GaussianEmitter(
                self._archive,
                sigma=sigma,
                batch_size=batch_size,
            )
        else:  # ES
            emitter = EvolutionStrategyEmitter(
                self._archive,
                sigma0=sigma,
                batch_size=batch_size,
            )

        self._emitters.append(emitter)

        # Update scheduler
        self._scheduler = Scheduler(self._archive, self._emitters)

    def ask(self) -> np.ndarray:
        """
        Get solutions to evaluate.

        Returns:
            Array of solution parameters [batch_size, solution_dim]
        """
        if self._scheduler is None:
            raise ValueError("No emitters added. Call add_emitter() first.")

        return self._scheduler.ask()

    def tell(
        self,
        objectives: np.ndarray,
        behaviors: np.ndarray,
    ) -> None:
        """
        Report evaluation results.

        Args:
            objectives: Objective values [batch_size]
            behaviors: Behavior values [batch_size, behavior_dim]
        """
        if self._scheduler is None:
            raise ValueError("No emitters added. Call add_emitter() first.")

        self._scheduler.tell(objectives, behaviors)

    def optimize(
        self,
        objective_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
        iterations: int = 100,
        batch_size: int = 32,
    ) -> ArchiveStats:
        """
        Run the full optimization loop.

        Args:
            objective_fn: Function that takes solutions and returns (objectives, behaviors)
            iterations: Number of optimization iterations
            batch_size: Solutions per batch

        Returns:
            Final archive statistics
        """
        # Add default emitter if none added
        if not self._emitters:
            self.add_emitter(batch_size=batch_size)

        for _ in range(iterations):
            solutions = self.ask()
            objectives, behaviors = objective_fn(solutions)
            self.tell(objectives, behaviors)

        return self.get_stats()

    def get_stats(self) -> ArchiveStats:
        """Get current archive statistics."""
        stats = self._archive.stats

        # Calculate coverage (percentage of cells filled)
        if hasattr(self._archive, "cells"):
            total_cells = self._archive.cells
        else:
            total_cells = 1
            for d in self.config.grid_dims:
                total_cells *= d

        coverage = len(self._archive) / total_cells

        return ArchiveStats(
            num_elites=len(self._archive),
            coverage=coverage,
            qd_score=stats.qd_score,
            max_objective=stats.obj_max,
            mean_objective=stats.obj_mean,
        )

    def get_elites(self, top_k: Optional[int] = None) -> List[Solution]:
        """
        Get elite solutions from the archive.

        Args:
            top_k: Number of top elites to return

        Returns:
            List of Solution objects
        """
        # Get all elites
        df = self._archive.as_pandas()

        if top_k:
            df = df.nlargest(top_k, "objective")

        solutions = []
        for idx, row in df.iterrows():
            # Extract solution parameters
            param_cols = [c for c in df.columns if c.startswith("solution_")]
            params = np.array([row[c] for c in param_cols])

            # Extract behaviors
            behavior_cols = [c for c in df.columns if c.startswith("measure_")]
            behaviors = np.array([row[c] for c in behavior_cols])

            solutions.append(Solution(
                id=idx,
                parameters=params,
                objective=row["objective"],
                behavior=behaviors,
            ))

        return solutions

    def get_elite_at(self, behavior: np.ndarray) -> Optional[Solution]:
        """
        Get the elite at a specific behavior location.

        Args:
            behavior: Target behavior coordinates

        Returns:
            Solution if cell is occupied, None otherwise
        """
        # Find the cell index for this behavior
        index = self._archive.index_of_single(behavior)

        if self._archive.occupied[index]:
            row = self._archive.retrieve_single(behavior)
            return Solution(
                id=int(index),
                parameters=row[0],
                objective=row[1],
                behavior=behavior,
            )

        return None

    def visualize(
        self,
        output_path: Optional[str] = None,
        cmap: str = "viridis",
    ) -> Any:
        """
        Visualize the archive as a heatmap.

        Args:
            output_path: Optional path to save figure
            cmap: Colormap name

        Returns:
            Matplotlib figure
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))

        if self.config.archive_type == ArchiveType.GRID:
            grid_archive_heatmap(self._archive, ax=ax, cmap=cmap)
        else:
            cvt_archive_heatmap(self._archive, ax=ax, cmap=cmap)

        ax.set_xlabel("Behavior 1")
        ax.set_ylabel("Behavior 2")
        ax.set_title(f"QD Archive ({len(self._archive)} elites)")

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")

        return fig


# ============================================================================
# Explicit Getter Functions - Raise SDKNotAvailableError if unavailable
# ============================================================================

def get_graphrag_client(
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    **kwargs: Any,
) -> GraphRAGClient:
    """
    Get a GraphRAG client.

    Args:
        api_key: OpenAI API key
        model: LLM model to use
        **kwargs: Additional configuration

    Returns:
        GraphRAGClient instance

    Raises:
        SDKNotAvailableError: If graphrag is not installed
        SDKConfigurationError: If API key is missing
    """
    return GraphRAGClient(config=GraphRAGConfig(api_key=api_key, model=model, **kwargs))


def get_pyribs_archive(
    solution_dim: int = 10,
    behavior_dim: int = 2,
    archive_type: ArchiveType = ArchiveType.GRID,
    **kwargs: Any,
) -> PyRibsClient:
    """
    Get a PyRibs QD archive client.

    Args:
        solution_dim: Dimension of solutions
        behavior_dim: Dimension of behavior space
        archive_type: Type of archive
        **kwargs: Additional configuration

    Returns:
        PyRibsClient instance

    Raises:
        SDKNotAvailableError: If pyribs is not installed
    """
    # Build behavior ranges based on dimension
    behavior_ranges = tuple((0.0, 1.0) for _ in range(behavior_dim))

    # Build grid dims
    cells_per_dim = int(100 ** (1 / behavior_dim))
    grid_dims = tuple(cells_per_dim for _ in range(behavior_dim))

    config = PyRibsConfig(
        archive_type=archive_type,
        solution_dim=solution_dim,
        behavior_dim=behavior_dim,
        grid_dims=grid_dims,
        behavior_ranges=behavior_ranges,
        **kwargs,
    )
    return PyRibsClient(config=config)


# ============================================================================
# Unified Factory
# ============================================================================

class KnowledgeFactory:
    """
    Unified factory for creating knowledge clients.

    Provides a single entry point for GraphRAG and PyRibs with
    consistent configuration and V33 integration.
    """

    def __init__(self):
        """Initialize the factory."""
        self._graphrag: Optional[GraphRAGClient] = None
        self._pyribs: Optional[PyRibsClient] = None

    def get_availability(self) -> Dict[str, bool]:
        """Get availability status of all SDKs."""
        return {
            "graphrag": GRAPHRAG_AVAILABLE,
            "pyribs": PYRIBS_AVAILABLE,
        }

    def create_graphrag(self, **kwargs: Any) -> GraphRAGClient:
        """Create a GraphRAG client."""
        self._graphrag = GraphRAGClient(**kwargs)
        return self._graphrag

    def create_pyribs(self, **kwargs: Any) -> PyRibsClient:
        """Create a PyRibs client."""
        self._pyribs = PyRibsClient(**kwargs)
        return self._pyribs

    def get_graphrag(self) -> Optional[GraphRAGClient]:
        """Get the cached GraphRAG client."""
        return self._graphrag

    def get_pyribs(self) -> Optional[PyRibsClient]:
        """Get the cached PyRibs client."""
        return self._pyribs


# ============================================================================
# Module-level availability
# ============================================================================

KNOWLEDGE_AVAILABLE = GRAPHRAG_AVAILABLE or PYRIBS_AVAILABLE


def get_available_sdks() -> Dict[str, bool]:
    """Get availability status of all knowledge SDKs."""
    return {
        "graphrag": GRAPHRAG_AVAILABLE,
        "pyribs": PYRIBS_AVAILABLE,
    }


# ============================================================================
# All Exports
# ============================================================================

__all__ = [
    # Exceptions (re-exported from observability)
    "SDKNotAvailableError",
    "SDKConfigurationError",

    # Availability flags
    "GRAPHRAG_AVAILABLE",
    "PYRIBS_AVAILABLE",
    "KNOWLEDGE_AVAILABLE",

    # Getter functions (raise on unavailable)
    "get_graphrag_client",
    "get_pyribs_archive",

    # GraphRAG
    "GraphRAGClient",
    "GraphRAGConfig",
    "SearchMode",
    "Entity",
    "Relationship",
    "SearchResult",

    # PyRibs
    "PyRibsClient",
    "PyRibsConfig",
    "ArchiveType",
    "EmitterType",
    "Solution",
    "ArchiveStats",

    # Factory
    "KnowledgeFactory",
    "get_available_sdks",
]
