"""
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

This module implements RAPTOR, a hierarchical retrieval method that builds
tree structures from documents through recursive clustering and summarization.
This enables retrieval at multiple levels of abstraction - from detailed chunks
to high-level summaries.

Key Features:
- Tree construction through recursive clustering and summarization
- Multiple retrieval modes: tree traversal and collapsed tree
- Incremental updates without full rebuilds
- Integration with SemanticChunker and existing embedding providers

Architecture:
    Documents -> Chunks -> Cluster (GMM/K-Means) -> Summarize -> Repeat
                    |                                    |
                 Level 0 -------- Level 1 -------- Level N (root)

Reference: https://arxiv.org/abs/2401.18059

Integration:
    from core.rag.raptor import RAPTOR, RAPTORConfig, RAPTORNode

    raptor = RAPTOR(summarizer=my_llm, embedder=my_embedder)
    await raptor.build_tree(documents)
    results = await raptor.retrieve("query", method="collapsed")
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union, Set

# Optional numpy import with fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

# Optional sklearn for clustering
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    GaussianMixture = None  # type: ignore
    KMeans = None  # type: ignore
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# PROTOCOLS AND TYPES
# =============================================================================

class SummarizerProvider(Protocol):
    """Protocol for LLM summarization providers."""

    async def summarize(
        self,
        text: str,
        max_length: int = 256,
        **kwargs
    ) -> str:
        """Summarize text. Returns concise summary."""
        ...


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def encode(self, texts: Union[str, List[str]]) -> Any:
        """Encode text(s) to embedding(s). Returns numpy array or list."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        ...


class ClusterMethod(str, Enum):
    """Clustering methods for tree construction."""
    GMM = "gmm"                    # Gaussian Mixture Model (soft clustering)
    KMEANS = "kmeans"              # K-Means (hard clustering)
    HIERARCHICAL = "hierarchical"  # Agglomerative hierarchical


class RetrievalMethod(str, Enum):
    """Retrieval methods for RAPTOR tree."""
    COLLAPSED = "collapsed"        # Flatten all nodes, standard vector search
    TREE_TRAVERSAL = "tree_traversal"  # Top-down traversal
    HYBRID = "hybrid"              # Combine both methods


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RAPTORConfig:
    """Configuration for RAPTOR tree construction and retrieval.

    Attributes:
        chunk_size: Size of leaf chunks in tokens (default: 100)
        min_cluster_size: Minimum documents per cluster (default: 2)
        max_cluster_size: Maximum documents per cluster (default: 10)
        cluster_method: Clustering algorithm (default: GMM)
        n_clusters_ratio: Ratio for determining cluster count (default: 0.5)
        summary_max_tokens: Maximum tokens for summaries (default: 256)
        max_tree_depth: Maximum tree depth (default: 5)
        similarity_threshold: Threshold for traversal (default: 0.5)
        top_k: Default number of results (default: 5)
        embed_summaries: Whether to embed summaries (default: True)
        traversal_beam_width: Beam width for tree traversal (default: 3)
    """
    chunk_size: int = 100
    min_cluster_size: int = 2
    max_cluster_size: int = 10
    cluster_method: ClusterMethod = ClusterMethod.GMM
    n_clusters_ratio: float = 0.5
    summary_max_tokens: int = 256
    max_tree_depth: int = 5
    similarity_threshold: float = 0.5
    top_k: int = 5
    embed_summaries: bool = True
    traversal_beam_width: int = 3


@dataclass
class RAPTORNode:
    """A node in the RAPTOR tree.

    Attributes:
        id: Unique node identifier
        content: Text content (original chunk or summary)
        embedding: Vector embedding of content
        level: Tree level (0 = leaf/original, higher = summaries)
        children: List of child node IDs
        parent: Parent node ID (None for root)
        metadata: Additional metadata
        is_summary: Whether this is a summary node
    """
    id: str
    content: str
    embedding: Optional[List[float]] = None
    level: int = 0
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_summary: bool = False

    def __post_init__(self):
        if self.id is None:
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique node ID."""
        hash_input = f"{self.content[:100]}:{self.level}:{id(self)}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    @property
    def token_estimate(self) -> int:
        """Estimate token count (rough: ~4 chars per token)."""
        return len(self.content) // 4

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RAPTORNode):
            return False
        return self.id == other.id


@dataclass
class RAPTORTree:
    """The complete RAPTOR tree structure.

    Attributes:
        nodes: All nodes indexed by ID
        levels: Nodes organized by level
        root_ids: IDs of root nodes
        leaf_ids: IDs of leaf nodes
        metadata: Tree metadata (build time, stats, etc.)
    """
    nodes: Dict[str, RAPTORNode] = field(default_factory=dict)
    levels: Dict[int, List[str]] = field(default_factory=dict)
    root_ids: List[str] = field(default_factory=list)
    leaf_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: RAPTORNode) -> None:
        """Add a node to the tree."""
        self.nodes[node.id] = node
        if node.level not in self.levels:
            self.levels[node.level] = []
        self.levels[node.level].append(node.id)

        # Track leaves and roots
        if node.level == 0:
            self.leaf_ids.append(node.id)
        if node.parent is None and node.level > 0:
            # Will update when tree is complete
            pass

    def get_node(self, node_id: str) -> Optional[RAPTORNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> List[RAPTORNode]:
        """Get child nodes."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.children if cid in self.nodes]

    def get_parent(self, node_id: str) -> Optional[RAPTORNode]:
        """Get parent node."""
        node = self.nodes.get(node_id)
        if not node or not node.parent:
            return None
        return self.nodes.get(node.parent)

    def get_level(self, level: int) -> List[RAPTORNode]:
        """Get all nodes at a level."""
        if level not in self.levels:
            return []
        return [self.nodes[nid] for nid in self.levels[level] if nid in self.nodes]

    def get_all_nodes(self) -> List[RAPTORNode]:
        """Get all nodes in the tree."""
        return list(self.nodes.values())

    @property
    def max_level(self) -> int:
        """Get the maximum tree level (root level)."""
        if not self.levels:
            return 0
        return max(self.levels.keys())

    @property
    def total_nodes(self) -> int:
        """Get total number of nodes."""
        return len(self.nodes)


@dataclass
class RAPTORResult:
    """Result from RAPTOR retrieval.

    Attributes:
        nodes: Retrieved nodes with scores
        context: Combined context from retrieved nodes
        levels_searched: Which tree levels were searched
        retrieval_method: Method used for retrieval
    """
    nodes: List[Tuple[RAPTORNode, float]]  # (node, score)
    context: str
    levels_searched: List[int] = field(default_factory=list)
    retrieval_method: RetrievalMethod = RetrievalMethod.COLLAPSED


@dataclass
class TreeBuildStats:
    """Statistics from tree building."""
    total_documents: int = 0
    total_chunks: int = 0
    total_summaries: int = 0
    tree_depth: int = 0
    build_time_seconds: float = 0.0
    levels: Dict[int, int] = field(default_factory=dict)  # level -> node count


# =============================================================================
# CLUSTERING IMPLEMENTATIONS
# =============================================================================

class Clusterer(ABC):
    """Abstract base class for clustering algorithms."""

    @abstractmethod
    def cluster(
        self,
        embeddings: Any,
        n_clusters: int
    ) -> List[List[int]]:
        """Cluster embeddings into groups.

        Args:
            embeddings: Embedding matrix (n_samples x n_dims)
            n_clusters: Target number of clusters

        Returns:
            List of clusters, each containing document indices
        """
        ...


class GMMClusterer(Clusterer):
    """Gaussian Mixture Model clustering (soft clustering)."""

    def cluster(
        self,
        embeddings: Any,
        n_clusters: int
    ) -> List[List[int]]:
        """Cluster using GMM."""
        if not SKLEARN_AVAILABLE or not NUMPY_AVAILABLE:
            logger.warning("sklearn/numpy not available, falling back to simple clustering")
            return self._simple_cluster(embeddings, n_clusters)

        n_samples = len(embeddings)
        if n_samples < n_clusters:
            n_clusters = max(1, n_samples)

        try:
            gmm = GaussianMixture(
                n_components=n_clusters,
                covariance_type='full',
                random_state=42,
                n_init=3
            )
            gmm.fit(embeddings)
            labels = gmm.predict(embeddings)

            # Group indices by cluster
            clusters: Dict[int, List[int]] = {}
            for idx, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(idx)

            return list(clusters.values())

        except Exception as e:
            logger.warning(f"GMM clustering failed: {e}, using simple clustering")
            return self._simple_cluster(embeddings, n_clusters)

    def _simple_cluster(
        self,
        embeddings: Any,
        n_clusters: int
    ) -> List[List[int]]:
        """Simple uniform clustering fallback."""
        n_samples = len(embeddings)
        cluster_size = max(1, n_samples // n_clusters)
        clusters = []

        for i in range(0, n_samples, cluster_size):
            cluster = list(range(i, min(i + cluster_size, n_samples)))
            if cluster:
                clusters.append(cluster)

        return clusters


class KMeansClusterer(Clusterer):
    """K-Means clustering (hard clustering)."""

    def cluster(
        self,
        embeddings: Any,
        n_clusters: int
    ) -> List[List[int]]:
        """Cluster using K-Means."""
        if not SKLEARN_AVAILABLE or not NUMPY_AVAILABLE:
            logger.warning("sklearn/numpy not available, falling back to simple clustering")
            return self._simple_cluster(embeddings, n_clusters)

        n_samples = len(embeddings)
        if n_samples < n_clusters:
            n_clusters = max(1, n_samples)

        try:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
            labels = kmeans.fit_predict(embeddings)

            # Group indices by cluster
            clusters: Dict[int, List[int]] = {}
            for idx, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(idx)

            return list(clusters.values())

        except Exception as e:
            logger.warning(f"K-Means clustering failed: {e}, using simple clustering")
            return self._simple_cluster(embeddings, n_clusters)

    def _simple_cluster(
        self,
        embeddings: Any,
        n_clusters: int
    ) -> List[List[int]]:
        """Simple uniform clustering fallback."""
        n_samples = len(embeddings)
        cluster_size = max(1, n_samples // n_clusters)
        clusters = []

        for i in range(0, n_samples, cluster_size):
            cluster = list(range(i, min(i + cluster_size, n_samples)))
            if cluster:
                clusters.append(cluster)

        return clusters


def get_clusterer(method: ClusterMethod) -> Clusterer:
    """Factory function for clusterers."""
    if method == ClusterMethod.GMM:
        return GMMClusterer()
    elif method == ClusterMethod.KMEANS:
        return KMeansClusterer()
    else:
        # Default to GMM
        return GMMClusterer()


# =============================================================================
# SUMMARIZATION WRAPPER
# =============================================================================

class LLMSummarizer:
    """Wrapper for LLM-based summarization."""

    SUMMARIZE_PROMPT = """Summarize the following text passages into a single coherent summary.
Preserve key information, main concepts, and important details.
Keep the summary concise but comprehensive.

Text passages:
{passages}

Summary:"""

    def __init__(self, llm: SummarizerProvider):
        """Initialize summarizer.

        Args:
            llm: LLM provider with summarize method
        """
        self.llm = llm

    async def summarize_cluster(
        self,
        texts: List[str],
        max_tokens: int = 256
    ) -> str:
        """Summarize a cluster of texts.

        Args:
            texts: List of text passages to summarize
            max_tokens: Maximum summary length

        Returns:
            Summarized text
        """
        if not texts:
            return ""

        if len(texts) == 1:
            # Single text, just return truncated version
            return texts[0][:max_tokens * 4]

        # Combine texts with separators
        combined = "\n\n---\n\n".join(texts)

        # Truncate if too long (rough limit)
        max_input_chars = 8000  # Approx 2000 tokens
        if len(combined) > max_input_chars:
            combined = combined[:max_input_chars]

        prompt = self.SUMMARIZE_PROMPT.format(passages=combined)

        try:
            summary = await self.llm.summarize(prompt, max_length=max_tokens)
            return summary.strip()
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback: return first passage truncated
            return texts[0][:max_tokens * 4]


# =============================================================================
# RAPTOR IMPLEMENTATION
# =============================================================================

class RAPTOR:
    """
    RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.

    Builds hierarchical document trees through recursive clustering and
    summarization, enabling retrieval at multiple levels of abstraction.

    Key Benefits:
    - Multi-level context: From detailed chunks to high-level summaries
    - Improved accuracy on long documents and complex queries
    - State-of-the-art on QuALITY benchmark (+20% over standard RAG)

    Example:
        >>> from core.rag.raptor import RAPTOR, RAPTORConfig
        >>>
        >>> config = RAPTORConfig(chunk_size=100, cluster_method=ClusterMethod.GMM)
        >>> raptor = RAPTOR(summarizer=my_llm, embedder=my_embedder, config=config)
        >>>
        >>> await raptor.build_tree(documents)
        >>> result = await raptor.retrieve("What is the main theme?", method="collapsed")
        >>> print(result.context)
    """

    def __init__(
        self,
        summarizer: SummarizerProvider,
        embedder: EmbeddingProvider,
        config: Optional[RAPTORConfig] = None,
        chunker: Optional[Any] = None,  # SemanticChunker
    ):
        """Initialize RAPTOR.

        Args:
            summarizer: LLM provider for summarization
            embedder: Embedding provider for vectorization
            config: Configuration options
            chunker: Optional SemanticChunker for document chunking
        """
        self.summarizer_provider = summarizer
        self.embedder = embedder
        self.config = config or RAPTORConfig()
        self.chunker = chunker

        # Initialize components
        self.summarizer = LLMSummarizer(summarizer)
        self.clusterer = get_clusterer(self.config.cluster_method)

        # Tree storage
        self.tree: Optional[RAPTORTree] = None
        self._build_stats: Optional[TreeBuildStats] = None

    async def build_tree(
        self,
        documents: List[str],
        document_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> RAPTORTree:
        """Build the RAPTOR tree from documents.

        Args:
            documents: List of document texts
            document_metadata: Optional metadata for each document

        Returns:
            Built RAPTORTree
        """
        import time
        start_time = time.time()

        logger.info(f"Building RAPTOR tree from {len(documents)} documents")

        self.tree = RAPTORTree()
        self._build_stats = TreeBuildStats(total_documents=len(documents))

        document_metadata = document_metadata or [{} for _ in documents]

        # Step 1: Chunk documents into leaf nodes
        leaf_nodes = await self._create_leaf_nodes(documents, document_metadata)
        logger.info(f"Created {len(leaf_nodes)} leaf nodes")
        self._build_stats.total_chunks = len(leaf_nodes)

        if not leaf_nodes:
            logger.warning("No leaf nodes created, returning empty tree")
            return self.tree

        # Add leaves to tree
        for node in leaf_nodes:
            self.tree.add_node(node)

        # Step 2: Build tree levels recursively
        current_level_nodes = leaf_nodes
        current_level = 0

        while (len(current_level_nodes) > 1 and
               current_level < self.config.max_tree_depth):

            logger.info(f"Building level {current_level + 1} from {len(current_level_nodes)} nodes")

            # Cluster current level
            next_level_nodes = await self._build_next_level(
                current_level_nodes,
                current_level + 1
            )

            if not next_level_nodes or len(next_level_nodes) >= len(current_level_nodes):
                # No reduction, stop building
                logger.info(f"Tree construction complete at level {current_level}")
                break

            # Add new level nodes to tree
            for node in next_level_nodes:
                self.tree.add_node(node)
                self._build_stats.total_summaries += 1

            self._build_stats.levels[current_level + 1] = len(next_level_nodes)
            current_level_nodes = next_level_nodes
            current_level += 1

        # Mark root nodes
        self.tree.root_ids = [n.id for n in current_level_nodes]
        self._build_stats.tree_depth = current_level + 1
        self._build_stats.build_time_seconds = time.time() - start_time

        logger.info(
            f"RAPTOR tree built: {self.tree.total_nodes} nodes, "
            f"{self._build_stats.tree_depth} levels, "
            f"{self._build_stats.build_time_seconds:.2f}s"
        )

        # Update tree metadata
        self.tree.metadata = {
            "total_nodes": self.tree.total_nodes,
            "tree_depth": self._build_stats.tree_depth,
            "total_summaries": self._build_stats.total_summaries,
            "build_time": self._build_stats.build_time_seconds,
            "config": {
                "cluster_method": self.config.cluster_method.value,
                "chunk_size": self.config.chunk_size,
            }
        }

        return self.tree

    async def _create_leaf_nodes(
        self,
        documents: List[str],
        metadata: List[Dict[str, Any]]
    ) -> List[RAPTORNode]:
        """Create leaf nodes from documents."""
        leaf_nodes: List[RAPTORNode] = []

        for doc_idx, (doc, meta) in enumerate(zip(documents, metadata)):
            if not doc.strip():
                continue

            # Chunk document if chunker available
            if self.chunker is not None:
                try:
                    chunks = self.chunker.chunk(doc, metadata=meta)
                    for chunk_idx, chunk in enumerate(chunks):
                        node = RAPTORNode(
                            id=f"leaf_{doc_idx}_{chunk_idx}_{uuid.uuid4().hex[:8]}",
                            content=chunk.content,
                            level=0,
                            metadata={
                                **meta,
                                "doc_index": doc_idx,
                                "chunk_index": chunk_idx,
                                "source": "chunk"
                            },
                            is_summary=False
                        )
                        leaf_nodes.append(node)
                except Exception as e:
                    logger.warning(f"Chunking failed for doc {doc_idx}: {e}")
                    # Fallback to simple chunking
                    leaf_nodes.extend(
                        self._simple_chunk(doc, doc_idx, meta)
                    )
            else:
                # Simple chunking based on config
                leaf_nodes.extend(
                    self._simple_chunk(doc, doc_idx, meta)
                )

        # Embed all leaf nodes
        if leaf_nodes and self.config.embed_summaries:
            await self._embed_nodes(leaf_nodes)

        return leaf_nodes

    def _simple_chunk(
        self,
        text: str,
        doc_idx: int,
        metadata: Dict[str, Any]
    ) -> List[RAPTORNode]:
        """Simple chunking by approximate token count."""
        chunk_size_chars = self.config.chunk_size * 4  # ~4 chars per token
        chunks: List[RAPTORNode] = []

        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            if current_size + para_size > chunk_size_chars and current_chunk:
                # Create chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(RAPTORNode(
                    id=f"leaf_{doc_idx}_{len(chunks)}_{uuid.uuid4().hex[:8]}",
                    content=chunk_text,
                    level=0,
                    metadata={
                        **metadata,
                        "doc_index": doc_idx,
                        "chunk_index": len(chunks),
                        "source": "simple_chunk"
                    },
                    is_summary=False
                ))
                current_chunk = []
                current_size = 0

            current_chunk.append(para)
            current_size += para_size

        # Handle remaining
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(RAPTORNode(
                id=f"leaf_{doc_idx}_{len(chunks)}_{uuid.uuid4().hex[:8]}",
                content=chunk_text,
                level=0,
                metadata={
                    **metadata,
                    "doc_index": doc_idx,
                    "chunk_index": len(chunks),
                    "source": "simple_chunk"
                },
                is_summary=False
            ))

        return chunks

    async def _build_next_level(
        self,
        current_nodes: List[RAPTORNode],
        next_level: int
    ) -> List[RAPTORNode]:
        """Build the next level of the tree by clustering and summarizing."""
        if len(current_nodes) <= 1:
            return current_nodes

        # Get embeddings for clustering
        embeddings = self._get_node_embeddings(current_nodes)

        if embeddings is None or len(embeddings) == 0:
            # Can't cluster without embeddings
            logger.warning("No embeddings for clustering, stopping tree build")
            return current_nodes

        # Calculate number of clusters
        n_clusters = max(1, int(len(current_nodes) * self.config.n_clusters_ratio))
        n_clusters = min(n_clusters, len(current_nodes) - 1)  # At least some reduction

        if n_clusters <= 1:
            # Would create single summary of everything
            n_clusters = max(1, len(current_nodes) // 2)

        # Cluster nodes
        clusters = self.clusterer.cluster(embeddings, n_clusters)
        logger.debug(f"Created {len(clusters)} clusters for level {next_level}")

        # Create summary nodes for each cluster
        summary_nodes: List[RAPTORNode] = []

        summarize_tasks = []
        cluster_data = []

        for cluster_idx, cluster_indices in enumerate(clusters):
            if not cluster_indices:
                continue

            cluster_nodes = [current_nodes[i] for i in cluster_indices]

            # Enforce cluster size limits
            if len(cluster_nodes) < self.config.min_cluster_size:
                # Too small, skip (nodes will be orphaned or merged)
                continue

            if len(cluster_nodes) > self.config.max_cluster_size:
                # Too large, truncate
                cluster_nodes = cluster_nodes[:self.config.max_cluster_size]

            cluster_data.append((cluster_idx, cluster_nodes))
            cluster_texts = [n.content for n in cluster_nodes]
            summarize_tasks.append(
                self.summarizer.summarize_cluster(
                    cluster_texts,
                    max_tokens=self.config.summary_max_tokens
                )
            )

        # Run summarization in parallel
        summaries = await asyncio.gather(*summarize_tasks, return_exceptions=True)

        for (cluster_idx, cluster_nodes), summary in zip(cluster_data, summaries):
            if isinstance(summary, Exception):
                logger.warning(f"Summarization failed for cluster {cluster_idx}: {summary}")
                continue

            if not summary or not summary.strip():
                continue

            # Create summary node
            child_ids = [n.id for n in cluster_nodes]
            summary_node = RAPTORNode(
                id=f"summary_{next_level}_{cluster_idx}_{uuid.uuid4().hex[:8]}",
                content=summary,
                level=next_level,
                children=child_ids,
                metadata={
                    "cluster_size": len(cluster_nodes),
                    "cluster_index": cluster_idx,
                    "source": "summary"
                },
                is_summary=True
            )

            # Update parent references in children
            for child_node in cluster_nodes:
                child_node.parent = summary_node.id

            summary_nodes.append(summary_node)

        # Embed summary nodes
        if summary_nodes and self.config.embed_summaries:
            await self._embed_nodes(summary_nodes)

        return summary_nodes

    def _get_node_embeddings(
        self,
        nodes: List[RAPTORNode]
    ) -> Optional[Any]:
        """Get embeddings for nodes as numpy array."""
        embeddings = []
        for node in nodes:
            if node.embedding is not None:
                embeddings.append(node.embedding)
            else:
                # Need to embed
                return None

        if not embeddings:
            return None

        if NUMPY_AVAILABLE:
            return np.array(embeddings)
        return embeddings

    async def _embed_nodes(self, nodes: List[RAPTORNode]) -> None:
        """Embed multiple nodes."""
        texts = [n.content for n in nodes]

        try:
            embeddings = self.embedder.encode(texts)

            # Convert to list format, handling different return shapes
            if NUMPY_AVAILABLE and hasattr(embeddings, 'shape'):
                # Numpy array
                if len(embeddings.shape) == 1:
                    # Single embedding returned as 1D array
                    embeddings_list = [embeddings.tolist()]
                else:
                    # Multiple embeddings as 2D array
                    embeddings_list = embeddings.tolist()
            elif NUMPY_AVAILABLE and hasattr(embeddings, 'tolist'):
                result = embeddings.tolist()
                # Check if it's nested (2D) or flat (1D)
                if result and isinstance(result[0], (int, float)):
                    embeddings_list = [result]
                else:
                    embeddings_list = result
            elif isinstance(embeddings, list) and len(embeddings) > 0:
                if hasattr(embeddings[0], 'tolist'):
                    embeddings_list = [e.tolist() for e in embeddings]
                elif isinstance(embeddings[0], (int, float)):
                    # Single embedding as flat list
                    embeddings_list = [embeddings]
                else:
                    embeddings_list = embeddings
            else:
                embeddings_list = [list(e) for e in embeddings]

            # Ensure we have the right number of embeddings
            if len(embeddings_list) != len(nodes):
                # If single embedding was returned for batch, broadcast
                if len(embeddings_list) == 1:
                    logger.warning(
                        f"Embedder returned 1 embedding for {len(nodes)} texts, "
                        "embedding each separately"
                    )
                    # Re-embed each text separately
                    embeddings_list = []
                    for text in texts:
                        emb = self.embedder.encode(text)
                        if hasattr(emb, 'tolist'):
                            emb = emb.tolist()
                        if isinstance(emb, list) and emb and isinstance(emb[0], (int, float)):
                            embeddings_list.append(emb)
                        else:
                            embeddings_list.append(list(emb))

            # Assign embeddings
            for node, emb in zip(nodes, embeddings_list):
                node.embedding = emb

        except Exception as e:
            logger.error(f"Failed to embed nodes: {e}")

    async def retrieve(
        self,
        query: str,
        method: Union[RetrievalMethod, str] = RetrievalMethod.COLLAPSED,
        top_k: Optional[int] = None,
        **kwargs
    ) -> RAPTORResult:
        """Retrieve relevant context from the RAPTOR tree.

        Args:
            query: Search query
            method: Retrieval method (collapsed, tree_traversal, hybrid)
            top_k: Number of results to return
            **kwargs: Additional arguments

        Returns:
            RAPTORResult with retrieved nodes and context
        """
        if self.tree is None or self.tree.total_nodes == 0:
            logger.warning("RAPTOR tree is empty, no retrieval possible")
            return RAPTORResult(
                nodes=[],
                context="",
                levels_searched=[],
                retrieval_method=RetrievalMethod(method) if isinstance(method, str) else method
            )

        top_k = top_k or self.config.top_k

        if isinstance(method, str):
            method = RetrievalMethod(method)

        if method == RetrievalMethod.COLLAPSED:
            return await self._retrieve_collapsed(query, top_k, **kwargs)
        elif method == RetrievalMethod.TREE_TRAVERSAL:
            return await self._retrieve_tree_traversal(query, top_k, **kwargs)
        elif method == RetrievalMethod.HYBRID:
            return await self._retrieve_hybrid(query, top_k, **kwargs)
        else:
            # Default to collapsed
            return await self._retrieve_collapsed(query, top_k, **kwargs)

    async def _retrieve_collapsed(
        self,
        query: str,
        top_k: int,
        **kwargs
    ) -> RAPTORResult:
        """Collapsed tree retrieval: flatten all nodes and do standard vector search."""
        # Embed query
        query_embedding = self._embed_query(query)
        if query_embedding is None:
            return RAPTORResult(
                nodes=[],
                context="",
                levels_searched=list(self.tree.levels.keys()) if self.tree else [],
                retrieval_method=RetrievalMethod.COLLAPSED
            )

        # Score all nodes
        scored_nodes: List[Tuple[RAPTORNode, float]] = []

        for node in self.tree.get_all_nodes():
            if node.embedding is None:
                continue

            score = self._cosine_similarity(query_embedding, node.embedding)
            scored_nodes.append((node, score))

        # Sort by score and take top_k
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        top_nodes = scored_nodes[:top_k]

        # Build context
        context = self._build_context([n for n, _ in top_nodes])

        return RAPTORResult(
            nodes=top_nodes,
            context=context,
            levels_searched=list(self.tree.levels.keys()),
            retrieval_method=RetrievalMethod.COLLAPSED
        )

    async def _retrieve_tree_traversal(
        self,
        query: str,
        top_k: int,
        **kwargs
    ) -> RAPTORResult:
        """Tree traversal retrieval: start at root and traverse down relevant branches."""
        query_embedding = self._embed_query(query)
        if query_embedding is None:
            return RAPTORResult(
                nodes=[],
                context="",
                levels_searched=[],
                retrieval_method=RetrievalMethod.TREE_TRAVERSAL
            )

        beam_width = self.config.traversal_beam_width
        collected_nodes: List[Tuple[RAPTORNode, float]] = []
        levels_searched: Set[int] = set()

        # Start from roots
        if not self.tree.root_ids:
            # If no roots, use highest level
            max_level = self.tree.max_level
            current_nodes = self.tree.get_level(max_level)
        else:
            current_nodes = [self.tree.get_node(rid) for rid in self.tree.root_ids]
            current_nodes = [n for n in current_nodes if n is not None]

        while current_nodes:
            # Score current level nodes
            scored = []
            for node in current_nodes:
                if node.embedding is not None:
                    score = self._cosine_similarity(query_embedding, node.embedding)
                    scored.append((node, score))
                    levels_searched.add(node.level)

            # Select top beam_width nodes
            scored.sort(key=lambda x: x[1], reverse=True)
            selected = scored[:beam_width]

            # Add selected to collected
            for node, score in selected:
                if score >= self.config.similarity_threshold:
                    collected_nodes.append((node, score))

            # Get children of selected nodes
            next_nodes = []
            for node, _ in selected:
                children = self.tree.get_children(node.id)
                next_nodes.extend(children)

            if not next_nodes:
                break

            current_nodes = next_nodes

        # Sort collected and take top_k
        collected_nodes.sort(key=lambda x: x[1], reverse=True)
        top_nodes = collected_nodes[:top_k]

        # Build context
        context = self._build_context([n for n, _ in top_nodes])

        return RAPTORResult(
            nodes=top_nodes,
            context=context,
            levels_searched=sorted(levels_searched),
            retrieval_method=RetrievalMethod.TREE_TRAVERSAL
        )

    async def _retrieve_hybrid(
        self,
        query: str,
        top_k: int,
        **kwargs
    ) -> RAPTORResult:
        """Hybrid retrieval: combine collapsed and tree traversal results."""
        # Run both methods
        collapsed_result = await self._retrieve_collapsed(query, top_k, **kwargs)
        traversal_result = await self._retrieve_tree_traversal(query, top_k, **kwargs)

        # Combine and deduplicate
        seen_ids: Set[str] = set()
        combined: List[Tuple[RAPTORNode, float]] = []

        for node, score in collapsed_result.nodes + traversal_result.nodes:
            if node.id not in seen_ids:
                seen_ids.add(node.id)
                combined.append((node, score))

        # Sort by score
        combined.sort(key=lambda x: x[1], reverse=True)
        top_nodes = combined[:top_k]

        # Combine levels searched
        all_levels = set(collapsed_result.levels_searched + traversal_result.levels_searched)

        # Build context
        context = self._build_context([n for n, _ in top_nodes])

        return RAPTORResult(
            nodes=top_nodes,
            context=context,
            levels_searched=sorted(all_levels),
            retrieval_method=RetrievalMethod.HYBRID
        )

    def _embed_query(self, query: str) -> Optional[List[float]]:
        """Embed a query string."""
        try:
            embedding = self.embedder.encode(query)
            if NUMPY_AVAILABLE and hasattr(embedding, 'tolist'):
                # Handle 2D output (batch of 1)
                if len(embedding.shape) > 1:
                    return embedding[0].tolist()
                return embedding.tolist()
            elif isinstance(embedding, list):
                if embedding and isinstance(embedding[0], list):
                    return embedding[0]
                return embedding
            return list(embedding)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return None

    def _cosine_similarity(
        self,
        a: List[float],
        b: List[float]
    ) -> float:
        """Compute cosine similarity between two vectors."""
        if NUMPY_AVAILABLE:
            a_arr = np.array(a).flatten()  # Ensure 1D
            b_arr = np.array(b).flatten()  # Ensure 1D
            dot = np.dot(a_arr, b_arr)
            norm_a = np.linalg.norm(a_arr)
            norm_b = np.linalg.norm(b_arr)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            result = dot / (norm_a * norm_b)
            # Handle numpy scalar vs array
            if hasattr(result, 'item'):
                return float(result.item())
            return float(result)
        else:
            # Pure Python fallback
            # Flatten if nested
            if a and isinstance(a[0], (list, tuple)):
                a = [x for sublist in a for x in sublist]
            if b and isinstance(b[0], (list, tuple)):
                b = [x for sublist in b for x in sublist]
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x ** 2 for x in a) ** 0.5
            norm_b = sum(x ** 2 for x in b) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

    def _build_context(self, nodes: List[RAPTORNode]) -> str:
        """Build context string from nodes."""
        if not nodes:
            return ""

        # Sort by level (lower levels = more specific)
        sorted_nodes = sorted(nodes, key=lambda n: n.level)

        # Build context with level indicators
        context_parts = []
        for node in sorted_nodes:
            level_indicator = f"[Level {node.level}]" if node.is_summary else "[Source]"
            context_parts.append(f"{level_indicator}\n{node.content}")

        return "\n\n---\n\n".join(context_parts)

    # ==========================================================================
    # INCREMENTAL UPDATE METHODS
    # ==========================================================================

    async def add_documents(
        self,
        documents: List[str],
        document_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """Add new documents to existing tree (incremental update).

        Args:
            documents: New documents to add
            document_metadata: Optional metadata for documents

        Returns:
            Number of new nodes added
        """
        if self.tree is None:
            # No tree exists, build from scratch
            await self.build_tree(documents, document_metadata)
            return self.tree.total_nodes if self.tree else 0

        document_metadata = document_metadata or [{} for _ in documents]

        # Create new leaf nodes
        new_leaves = await self._create_leaf_nodes(documents, document_metadata)
        if not new_leaves:
            return 0

        # Add to tree
        for node in new_leaves:
            self.tree.add_node(node)

        # Re-cluster affected branches
        # For simplicity, we re-cluster the entire lowest level that has unassigned nodes
        await self._recluster_level(0)

        return len(new_leaves)

    async def _recluster_level(self, level: int) -> None:
        """Re-cluster nodes at a specific level and update parent summaries."""
        level_nodes = self.tree.get_level(level)

        # Find orphaned nodes (no parent)
        orphaned = [n for n in level_nodes if n.parent is None and n.level == level]

        if len(orphaned) < self.config.min_cluster_size:
            return

        # Build next level from orphaned nodes
        next_level = level + 1
        new_summaries = await self._build_next_level(orphaned, next_level)

        for node in new_summaries:
            self.tree.add_node(node)

        # Recursively update higher levels if needed
        if len(new_summaries) > 1:
            await self._recluster_level(next_level)

    def clear_tree(self) -> None:
        """Clear the current tree."""
        self.tree = None
        self._build_stats = None

    @property
    def stats(self) -> Optional[TreeBuildStats]:
        """Get tree building statistics."""
        return self._build_stats


# =============================================================================
# INTEGRATION WITH SEMANTIC CHUNKER
# =============================================================================

class RAPTORWithChunker:
    """
    RAPTOR integrated with SemanticChunker for improved leaf node creation.

    Example:
        >>> from core.rag.raptor import RAPTORWithChunker
        >>> from core.rag.semantic_chunker import SemanticChunker
        >>>
        >>> chunker = SemanticChunker(max_chunk_size=100)
        >>> raptor = RAPTORWithChunker(
        ...     summarizer=my_llm,
        ...     embedder=my_embedder,
        ...     chunker=chunker
        ... )
        >>> await raptor.build_tree(documents)
    """

    def __init__(
        self,
        summarizer: SummarizerProvider,
        embedder: EmbeddingProvider,
        chunker: Any,  # SemanticChunker
        config: Optional[RAPTORConfig] = None,
    ):
        """Initialize RAPTOR with semantic chunker.

        Args:
            summarizer: LLM provider for summarization
            embedder: Embedding provider
            chunker: SemanticChunker instance
            config: Configuration
        """
        self.raptor = RAPTOR(
            summarizer=summarizer,
            embedder=embedder,
            config=config,
            chunker=chunker
        )

    async def build_tree(
        self,
        documents: List[str],
        document_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> RAPTORTree:
        """Build tree using semantic chunking."""
        return await self.raptor.build_tree(documents, document_metadata)

    async def retrieve(
        self,
        query: str,
        method: Union[RetrievalMethod, str] = RetrievalMethod.COLLAPSED,
        top_k: Optional[int] = None,
        **kwargs
    ) -> RAPTORResult:
        """Retrieve from tree."""
        return await self.raptor.retrieve(query, method, top_k, **kwargs)

    async def add_documents(
        self,
        documents: List[str],
        document_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """Add documents incrementally."""
        return await self.raptor.add_documents(documents, document_metadata)


# =============================================================================
# RAPTOR INDEX (VECTOR STORE WRAPPER)
# =============================================================================

class RAPTORIndex:
    """
    High-level RAPTOR index that wraps tree operations with a simple interface.

    Example:
        >>> index = RAPTORIndex(summarizer=my_llm, embedder=my_embedder)
        >>> await index.add(["doc1", "doc2", "doc3"])
        >>> results = await index.search("query", k=5)
        >>> for node, score in results:
        ...     print(f"{score:.3f}: {node.content[:50]}...")
    """

    def __init__(
        self,
        summarizer: SummarizerProvider,
        embedder: EmbeddingProvider,
        config: Optional[RAPTORConfig] = None,
    ):
        """Initialize RAPTOR index.

        Args:
            summarizer: LLM provider
            embedder: Embedding provider
            config: Configuration
        """
        self.raptor = RAPTOR(
            summarizer=summarizer,
            embedder=embedder,
            config=config
        )
        self._documents: List[str] = []

    async def add(
        self,
        documents: Union[str, List[str]],
        metadata: Optional[Union[Dict, List[Dict]]] = None
    ) -> int:
        """Add documents to the index.

        Args:
            documents: Single document or list of documents
            metadata: Optional metadata

        Returns:
            Number of nodes created
        """
        if isinstance(documents, str):
            documents = [documents]

        if metadata is None:
            metadata = [{}] * len(documents)
        elif isinstance(metadata, dict):
            metadata = [metadata]

        self._documents.extend(documents)

        if self.raptor.tree is None:
            await self.raptor.build_tree(self._documents, metadata)
            return self.raptor.tree.total_nodes if self.raptor.tree else 0
        else:
            return await self.raptor.add_documents(documents, metadata)

    async def search(
        self,
        query: str,
        k: int = 5,
        method: str = "collapsed"
    ) -> List[Tuple[RAPTORNode, float]]:
        """Search the index.

        Args:
            query: Search query
            k: Number of results
            method: Retrieval method

        Returns:
            List of (node, score) tuples
        """
        result = await self.raptor.retrieve(query, method=method, top_k=k)
        return result.nodes

    def clear(self) -> None:
        """Clear the index."""
        self.raptor.clear_tree()
        self._documents.clear()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    "RAPTOR",
    "RAPTORWithChunker",
    "RAPTORIndex",
    # Data classes
    "RAPTORNode",
    "RAPTORTree",
    "RAPTORResult",
    "RAPTORConfig",
    "TreeBuildStats",
    # Enums
    "ClusterMethod",
    "RetrievalMethod",
    # Clustering
    "Clusterer",
    "GMMClusterer",
    "KMeansClusterer",
    "get_clusterer",
    # Summarization
    "LLMSummarizer",
    # Protocols
    "SummarizerProvider",
    "EmbeddingProvider",
]
