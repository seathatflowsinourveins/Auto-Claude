"""
UNLEASH L7 Knowledge Layer Adapter
===================================

Unified knowledge graph adapter supporting GraphRAG for enhanced RAG:
- Global search for dataset-wide themes and patterns
- Local search for entity-focused queries
- Dynamic community selection for optimal context

Verified against official docs 2026-01-30:
- Context7: /microsoft/graphrag (488 snippets, 73.9 benchmark)
- Exa deep search for production patterns
"""

from typing import Optional, Any, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class SearchMethod(Enum):
    """GraphRAG search methods"""
    GLOBAL = "global"    # High-level themes via community summaries
    LOCAL = "local"      # Entity-focused via graph traversal
    HYBRID = "hybrid"    # Both methods combined


class ResponseType(Enum):
    """Response format types"""
    SINGLE_PARAGRAPH = "Single Paragraph"
    MULTIPLE_PARAGRAPHS = "Multiple Paragraphs"
    BULLET_POINTS = "List of 5 bullet points"
    DETAILED = "Detailed response with examples"


@dataclass
class SearchContext:
    """Context returned from search operations"""
    entities_used: List[str] = field(default_factory=list)
    communities_used: List[int] = field(default_factory=list)
    reports_used: List[str] = field(default_factory=list)
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Result from knowledge graph search"""
    response: str
    context: SearchContext
    method: SearchMethod
    community_level: Optional[int] = None
    token_count: Optional[int] = None


@dataclass
class GraphRAGConfig:
    """Configuration for GraphRAG"""
    project_directory: str
    community_level: int = 2
    dynamic_community_selection: bool = False
    response_type: ResponseType = ResponseType.MULTIPLE_PARAGRAPHS
    streaming: bool = False


class UnifiedKnowledgeGraph:
    """
    Unified knowledge graph layer for UNLEASH platform.

    Provides GraphRAG-based search over indexed documents with
    support for both global (theme) and local (entity) queries.

    Example:
        kg = UnifiedKnowledgeGraph(
            config=GraphRAGConfig(
                project_directory="./my_project",
                community_level=2
            )
        )

        # Global search for themes
        result = await kg.global_search("What are the main themes?")

        # Local search for entities
        result = await kg.local_search("Who is the main character?")
    """

    def __init__(
        self,
        config: GraphRAGConfig,
        auto_load: bool = True,
    ):
        self.config = config
        self.project_dir = Path(config.project_directory)

        # DataFrames for index data
        self._entities_df: Optional[pd.DataFrame] = None
        self._communities_df: Optional[pd.DataFrame] = None
        self._community_reports_df: Optional[pd.DataFrame] = None
        self._text_units_df: Optional[pd.DataFrame] = None
        self._relationships_df: Optional[pd.DataFrame] = None

        # GraphRAG API and config
        self._graphrag_config: Optional[Any] = None
        self._initialized = False

        if auto_load:
            self._load_index()

    def _load_index(self) -> None:
        """Load GraphRAG index files from parquet"""
        output_dir = self.project_dir / "output"

        if not output_dir.exists():
            logger.warning(f"Output directory not found: {output_dir}")
            logger.info("Run 'graphrag index' first to create the index")
            return

        try:
            # Load core index files
            entities_path = output_dir / "entities.parquet"
            communities_path = output_dir / "communities.parquet"
            reports_path = output_dir / "community_reports.parquet"

            if entities_path.exists():
                self._entities_df = pd.read_parquet(entities_path)
                logger.info(f"Loaded {len(self._entities_df)} entities")

            if communities_path.exists():
                self._communities_df = pd.read_parquet(communities_path)
                logger.info(f"Loaded {len(self._communities_df)} communities")

            if reports_path.exists():
                self._community_reports_df = pd.read_parquet(reports_path)
                logger.info(f"Loaded {len(self._community_reports_df)} community reports")

            # Load optional files
            text_units_path = output_dir / "text_units.parquet"
            relationships_path = output_dir / "relationships.parquet"

            if text_units_path.exists():
                self._text_units_df = pd.read_parquet(text_units_path)

            if relationships_path.exists():
                self._relationships_df = pd.read_parquet(relationships_path)

            # Load GraphRAG config
            self._load_graphrag_config()

            self._initialized = True
            logger.info("GraphRAG index loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load GraphRAG index: {e}")
            raise

    def _load_graphrag_config(self) -> None:
        """Load GraphRAG configuration"""
        try:
            from graphrag.config import GraphRagConfig  # type: ignore[import-not-found]

            settings_path = self.project_dir / "settings.yaml"
            if settings_path.exists():
                self._graphrag_config = GraphRagConfig.from_file(str(settings_path))
            else:
                # Use default config
                self._graphrag_config = GraphRagConfig()

            logger.info("GraphRAG configuration loaded")
        except ImportError:
            logger.warning("GraphRAG not installed, using mock config")
            self._graphrag_config = None

    async def global_search(
        self,
        query: str,
        community_level: Optional[int] = None,
        dynamic_selection: Optional[bool] = None,
        response_type: Optional[ResponseType] = None,
    ) -> SearchResult:
        """
        Perform global search for dataset-wide themes and patterns.

        Global search uses community summaries to answer high-level
        questions about the entire dataset.

        Args:
            query: The search query
            community_level: Override default community level (higher = smaller communities)
            dynamic_selection: Override dynamic community selection
            response_type: Override response format

        Returns:
            SearchResult with response and context metadata
        """
        if not self._initialized:
            raise RuntimeError("Index not loaded. Run index loading first.")

        level = community_level or self.config.community_level
        dynamic = dynamic_selection if dynamic_selection is not None else self.config.dynamic_community_selection
        resp_type = response_type or self.config.response_type

        try:
            from graphrag import api  # type: ignore[import-not-found]

            response, context = await api.global_search(
                config=self._graphrag_config,
                entities=self._entities_df,
                communities=self._communities_df,
                community_reports=self._community_reports_df,
                community_level=level,
                dynamic_community_selection=dynamic,
                response_type=resp_type.value,
                query=query,
            )

            return SearchResult(
                response=response,
                context=self._parse_context(context),
                method=SearchMethod.GLOBAL,
                community_level=level,
            )

        except ImportError:
            # Fallback for when graphrag is not installed
            return await self._mock_global_search(query, level)

    async def local_search(
        self,
        query: str,
        community_level: Optional[int] = None,
        response_type: Optional[ResponseType] = None,
    ) -> SearchResult:
        """
        Perform local search for entity-focused queries.

        Local search uses graph traversal to find specific entities
        and their relationships.

        Args:
            query: The search query
            community_level: Override default community level
            response_type: Override response format

        Returns:
            SearchResult with response and context metadata
        """
        if not self._initialized:
            raise RuntimeError("Index not loaded. Run index loading first.")

        level = community_level or self.config.community_level
        resp_type = response_type or self.config.response_type

        try:
            from graphrag import api  # type: ignore[import-not-found]

            response, context = await api.local_search(
                config=self._graphrag_config,
                entities=self._entities_df,
                communities=self._communities_df,
                community_reports=self._community_reports_df,
                text_units=self._text_units_df,
                relationships=self._relationships_df,
                community_level=level,
                response_type=resp_type.value,
                query=query,
            )

            return SearchResult(
                response=response,
                context=self._parse_context(context),
                method=SearchMethod.LOCAL,
                community_level=level,
            )

        except ImportError:
            return await self._mock_local_search(query, level)

    async def hybrid_search(
        self,
        query: str,
        community_level: Optional[int] = None,
    ) -> Tuple[SearchResult, SearchResult]:
        """
        Perform both global and local search for comprehensive results.

        Useful when you need both high-level themes and specific
        entity information.

        Args:
            query: The search query
            community_level: Override default community level

        Returns:
            Tuple of (global_result, local_result)
        """
        import asyncio

        global_task = self.global_search(query, community_level)
        local_task = self.local_search(query, community_level)

        global_result, local_result = await asyncio.gather(
            global_task, local_task
        )

        return global_result, local_result

    def _parse_context(self, raw_context: Any) -> SearchContext:
        """Parse raw context from GraphRAG into SearchContext"""
        if raw_context is None:
            return SearchContext()

        context = SearchContext()

        if isinstance(raw_context, dict):
            context.entities_used = raw_context.get("entities", [])
            context.communities_used = raw_context.get("communities", [])
            context.reports_used = raw_context.get("reports", [])
            context.relevance_scores = raw_context.get("scores", {})
            context.metadata = raw_context

        return context

    async def _mock_global_search(self, query: str, level: int) -> SearchResult:
        """Mock global search when GraphRAG not installed"""
        logger.warning("Using mock global search - install graphrag for real search")

        # Simulate search using loaded DataFrames
        response = f"[Mock Global Search] Query: {query}\n"

        if self._community_reports_df is not None:
            report_count = len(self._community_reports_df)
            response += f"Found {report_count} community reports.\n"
            response += "Sample themes from community reports..."

        return SearchResult(
            response=response,
            context=SearchContext(metadata={"mock": True}),
            method=SearchMethod.GLOBAL,
            community_level=level,
        )

    async def _mock_local_search(self, query: str, level: int) -> SearchResult:
        """Mock local search when GraphRAG not installed"""
        logger.warning("Using mock local search - install graphrag for real search")

        response = f"[Mock Local Search] Query: {query}\n"

        if self._entities_df is not None:
            response += f"Found {len(self._entities_df)} entities in graph.\n"

        return SearchResult(
            response=response,
            context=SearchContext(metadata={"mock": True}),
            method=SearchMethod.LOCAL,
            community_level=level,
        )

    def get_entity(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Get entity by name from the knowledge graph"""
        if self._entities_df is None:
            return None

        matches = self._entities_df[
            self._entities_df["name"].str.lower() == entity_name.lower()
        ]

        if len(matches) == 0:
            return None

        return matches.iloc[0].to_dict()

    def get_community(self, community_id: int) -> Optional[Dict[str, Any]]:
        """Get community by ID"""
        if self._communities_df is None:
            return None

        matches = self._communities_df[
            self._communities_df["id"] == community_id
        ]

        if len(matches) == 0:
            return None

        return matches.iloc[0].to_dict()

    def list_entities(
        self,
        limit: int = 100,
        entity_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List entities from the knowledge graph"""
        if self._entities_df is None:
            return []

        df = self._entities_df

        if entity_type:
            df = df[df["type"] == entity_type]

        return df.head(limit).to_dict(orient="records")  # type: ignore[call-overload]

    def list_communities(
        self,
        level: Optional[int] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List communities from the knowledge graph"""
        if self._communities_df is None:
            return []

        df = self._communities_df

        if level is not None:
            df = df[df["level"] == level]

        return df.head(limit).to_dict(orient="records")  # type: ignore[call-overload]

    @property
    def stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded knowledge graph"""
        return {
            "entities_count": len(self._entities_df) if self._entities_df is not None else 0,
            "communities_count": len(self._communities_df) if self._communities_df is not None else 0,
            "reports_count": len(self._community_reports_df) if self._community_reports_df is not None else 0,
            "text_units_count": len(self._text_units_df) if self._text_units_df is not None else 0,
            "relationships_count": len(self._relationships_df) if self._relationships_df is not None else 0,
            "initialized": self._initialized,
            "project_directory": str(self.project_dir),
        }


# Singleton instance for global access
_default_kg: Optional[UnifiedKnowledgeGraph] = None


def get_knowledge_graph() -> Optional[UnifiedKnowledgeGraph]:
    """Get the default knowledge graph instance"""
    return _default_kg


def configure_knowledge_graph(
    project_directory: str,
    community_level: int = 2,
    dynamic_community_selection: bool = False,
    **kwargs
) -> UnifiedKnowledgeGraph:
    """
    Configure global knowledge graph settings.

    Args:
        project_directory: Path to GraphRAG project
        community_level: Default community level for searches
        dynamic_community_selection: Enable dynamic selection
        **kwargs: Additional configuration options

    Returns:
        Configured UnifiedKnowledgeGraph instance
    """
    global _default_kg

    config = GraphRAGConfig(
        project_directory=project_directory,
        community_level=community_level,
        dynamic_community_selection=dynamic_community_selection,
        **kwargs
    )

    _default_kg = UnifiedKnowledgeGraph(config)
    return _default_kg
