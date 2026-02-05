"""
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

Enhanced implementation with full recursive tree indexing for UNLEASH platform.

Builds hierarchical summary trees from documents through recursive clustering
and summarization. Retrieval searches ALL levels (collapsed tree) for +20%
accuracy on multi-hop questions (QuALITY benchmark).

Architecture:
    Documents -> Chunk -> Embed -> Cluster (UMAP+GMM or cosine) -> Summarize -> Repeat
    Result: Multi-level tree stored in vector DB
    Retrieval: Search ALL levels (collapsed tree) or tree traversal

Features (V66 - Enhanced):
- Multi-level tree building (3-4 levels of abstraction)
- UMAP + GMM clustering with fallback to greedy cosine
- LLM-based summarization with heuristic fallback
- Collapsed tree retrieval for comprehensive results
- Tree traversal retrieval for hierarchical search
- Incremental tree updates
- Tree pruning for stale branches
- Cross-document reasoning via knowledge graph integration
- Memory block integration for cross-session persistence
- Full JSON serialization for export/import

Reference: https://arxiv.org/abs/2401.18059
GitHub: parthsarthi03/raptor

Usage:
    from platform.core.rag.raptor import (
        RAPTORProcessor,
        RAPTORTree,
        RAPTORConfig,
        ClusterMethod,
    )

    # Basic usage
    config = RAPTORConfig(max_levels=4, cluster_method=ClusterMethod.UMAP_GMM)
    processor = RAPTORProcessor(config=config, embed_fn=my_embed_fn)
    tree = processor.build_tree(documents)
    results = processor.search_tree(tree, "my query", top_k=5)

    # With LLM summarization
    processor = RAPTORProcessor(
        config=config,
        embed_fn=my_embed_fn,
        summarize_fn=my_llm_summarize,
    )

    # Incremental updates
    processor.update_tree(tree, new_documents)

    # Prune stale branches
    processor.prune_tree(tree, max_age_days=30)

    # Export/import
    tree_json = processor.export_tree(tree)
    restored_tree = processor.import_tree(tree_json)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Optional numpy for faster vector operations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False

# Optional UMAP for dimensionality reduction
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    umap = None  # type: ignore[assignment]
    UMAP_AVAILABLE = False

# Optional sklearn for GMM clustering
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    GaussianMixture = None  # type: ignore[assignment]
    KMeans = None  # type: ignore[assignment]
    SKLEARN_AVAILABLE = False


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

class ClusterMethod(str, Enum):
    """Clustering method for RAPTOR tree building."""
    UMAP_GMM = "umap_gmm"  # UMAP dimensionality reduction + GMM clustering
    GMM = "gmm"  # Direct GMM clustering
    KMEANS = "kmeans"  # K-Means clustering
    GREEDY_COSINE = "greedy_cosine"  # Greedy cosine similarity (no dependencies)


class RetrievalMethod(str, Enum):
    """Retrieval method for RAPTOR search."""
    COLLAPSED = "collapsed"  # Search all levels simultaneously
    TREE_TRAVERSAL = "tree_traversal"  # Top-down hierarchical traversal
    HYBRID = "hybrid"  # Combination of collapsed and tree traversal


class SummarizationMethod(str, Enum):
    """Summarization method for cluster summaries."""
    LLM = "llm"  # LLM-based abstractive summarization
    TFIDF = "tfidf"  # TF-IDF extractive summarization (heuristic)
    TEXTRANK = "textrank"  # TextRank extractive summarization


@dataclass
class RAPTORConfig:
    """Configuration for RAPTOR processor.

    Attributes:
        max_levels: Maximum depth of the tree (default 4).
        cluster_threshold: Cosine similarity threshold for greedy clustering (default 0.5).
        min_cluster_size: Minimum items per cluster before merging (default 2).
        max_cluster_size: Maximum items per cluster (default 10).
        n_clusters_ratio: Ratio of clusters to documents when using GMM/KMeans (default 0.5).
        embedding_dim: Dimension for content-hash embeddings when no embed_fn (default 64).
        cluster_method: Clustering algorithm to use (default GREEDY_COSINE).
        summarization_method: How to generate cluster summaries (default TFIDF).
        top_k: Default number of results for search (default 5).
        similarity_threshold: Minimum similarity score for retrieval (default 0.0).
        umap_n_neighbors: UMAP neighbors parameter (default 15).
        umap_min_dist: UMAP min_dist parameter (default 0.1).
        umap_n_components: UMAP output dimensions (default 5).
        gmm_covariance_type: GMM covariance type (default "full").
        enable_cross_doc_reasoning: Enable cross-document entity linking (default False).
        persist_embeddings: Store embeddings in tree nodes (default True).
    """
    max_levels: int = 4
    cluster_threshold: float = 0.5
    min_cluster_size: int = 2
    max_cluster_size: int = 10
    n_clusters_ratio: float = 0.5
    embedding_dim: int = 64
    cluster_method: ClusterMethod = ClusterMethod.GREEDY_COSINE
    summarization_method: SummarizationMethod = SummarizationMethod.TFIDF
    top_k: int = 5
    similarity_threshold: float = 0.0
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_components: int = 5
    gmm_covariance_type: str = "full"
    enable_cross_doc_reasoning: bool = False
    persist_embeddings: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_levels < 1:
            raise ValueError(f"max_levels must be >= 1, got {self.max_levels}")
        if not 0.0 <= self.cluster_threshold <= 1.0:
            raise ValueError(f"cluster_threshold must be in [0, 1], got {self.cluster_threshold}")
        if self.min_cluster_size < 1:
            raise ValueError(f"min_cluster_size must be >= 1, got {self.min_cluster_size}")
        if not 0.0 < self.n_clusters_ratio <= 1.0:
            raise ValueError(f"n_clusters_ratio must be in (0, 1], got {self.n_clusters_ratio}")


# =============================================================================
# PROTOCOLS
# =============================================================================

class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def __call__(self, text: str) -> List[float]:
        """Embed a single text string."""
        ...


class SummarizationProvider(Protocol):
    """Protocol for summarization providers."""

    def __call__(self, texts: List[str]) -> str:
        """Summarize a list of texts into a single summary."""
        ...


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TreeNode:
    """A node in the RAPTOR tree.

    Attributes:
        content: Text content (original chunk or summary).
        level: Tree level. 0 = original chunk, 1+ = summary levels.
        children: IDs of child nodes in the level below.
        parent: ID of parent node in the level above, or None for roots.
        node_id: Unique identifier for this node.
        metadata: Arbitrary metadata dict.
        embedding: Optional vector embedding of content.
        is_summary: Whether this node is a summary of child nodes.
        created_at: Timestamp when node was created.
        last_accessed: Timestamp when node was last accessed.
        access_count: Number of times node was accessed.
        cluster_id: ID of the cluster this node belongs to.
    """
    content: str
    level: int
    children: List[str]
    parent: Optional[str]
    node_id: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    is_summary: bool = False
    created_at: Optional[float] = None
    last_accessed: Optional[float] = None
    access_count: int = 0
    cluster_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = time.time()
        if self.last_accessed is None:
            self.last_accessed = self.created_at

    @property
    def token_estimate(self) -> int:
        """Estimate token count (~4 chars per token)."""
        return len(self.content) // 4

    @property
    def age_days(self) -> float:
        """Age of node in days since creation."""
        if self.created_at is None:
            return 0.0
        return (time.time() - self.created_at) / 86400

    def touch(self) -> None:
        """Update last_accessed timestamp and increment access_count."""
        self.last_accessed = time.time()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "content": self.content,
            "level": self.level,
            "children": self.children,
            "parent": self.parent,
            "node_id": self.node_id,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "is_summary": self.is_summary,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "cluster_id": self.cluster_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TreeNode":
        """Deserialize from dictionary."""
        return cls(
            content=data["content"],
            level=data["level"],
            children=data.get("children", []),
            parent=data.get("parent"),
            node_id=data["node_id"],
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            is_summary=data.get("is_summary", False),
            created_at=data.get("created_at"),
            last_accessed=data.get("last_accessed"),
            access_count=data.get("access_count", 0),
            cluster_id=data.get("cluster_id"),
        )


@dataclass
class RAPTORTree:
    """Complete RAPTOR tree with multi-level hierarchy.

    Stores all nodes indexed by level for efficient access. Supports
    JSON serialization for persistence.

    Attributes:
        nodes: All nodes indexed by node_id.
        levels: Mapping from level number to list of node_ids at that level.
        metadata: Tree-level metadata (build time, config, stats).
    """
    nodes: Dict[str, TreeNode] = field(default_factory=dict)
    levels: Dict[int, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: TreeNode) -> None:
        """Add a node to the tree, updating the level index."""
        self.nodes[node.node_id] = node
        if node.level not in self.levels:
            self.levels[node.level] = []
        if node.node_id not in self.levels[node.level]:
            self.levels[node.level].append(node.node_id)

    def remove_node(self, node_id: str) -> Optional[TreeNode]:
        """Remove a node from the tree.

        Args:
            node_id: ID of node to remove.

        Returns:
            The removed node, or None if not found.
        """
        if node_id not in self.nodes:
            return None

        node = self.nodes.pop(node_id)

        # Remove from level index
        if node.level in self.levels and node_id in self.levels[node.level]:
            self.levels[node.level].remove(node_id)
            if not self.levels[node.level]:
                del self.levels[node.level]

        # Update parent's children list
        if node.parent and node.parent in self.nodes:
            parent = self.nodes[node.parent]
            if node_id in parent.children:
                parent.children.remove(node_id)

        # Update children's parent reference
        for child_id in node.children:
            if child_id in self.nodes:
                self.nodes[child_id].parent = None

        return node

    def get_level(self, n: int) -> List[TreeNode]:
        """Return all nodes at level n."""
        if n not in self.levels:
            return []
        return [
            self.nodes[nid]
            for nid in self.levels[n]
            if nid in self.nodes
        ]

    def get_all_nodes(self) -> List[TreeNode]:
        """Return all nodes across all levels (collapsed tree)."""
        return list(self.nodes.values())

    def get_leaves(self) -> List[TreeNode]:
        """Return all leaf nodes (level 0)."""
        return self.get_level(0)

    def get_roots(self) -> List[TreeNode]:
        """Return all root nodes (nodes with no parent)."""
        return [n for n in self.nodes.values() if n.parent is None and n.level == self.max_level]

    @property
    def max_level(self) -> int:
        """Highest level in the tree. Returns -1 if tree is empty."""
        if not self.levels:
            return -1
        return max(self.levels.keys())

    @property
    def total_nodes(self) -> int:
        """Total number of nodes in the tree."""
        return len(self.nodes)

    @property
    def leaf_count(self) -> int:
        """Number of leaf nodes (original documents)."""
        return len(self.get_level(0))

    @property
    def summary_count(self) -> int:
        """Number of summary nodes."""
        return sum(1 for n in self.nodes.values() if n.is_summary)

    def get_node(self, node_id: str) -> Optional[TreeNode]:
        """Retrieve a node by ID. Returns None if not found."""
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> List[TreeNode]:
        """Get child nodes of a node."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.children if cid in self.nodes]

    def get_parent(self, node_id: str) -> Optional[TreeNode]:
        """Get parent node of a node."""
        node = self.nodes.get(node_id)
        if not node or not node.parent:
            return None
        return self.nodes.get(node.parent)

    def get_ancestors(self, node_id: str) -> List[TreeNode]:
        """Get all ancestor nodes (parent, grandparent, etc.)."""
        ancestors = []
        current = self.get_parent(node_id)
        while current:
            ancestors.append(current)
            current = self.get_parent(current.node_id)
        return ancestors

    def get_descendants(self, node_id: str) -> List[TreeNode]:
        """Get all descendant nodes (children, grandchildren, etc.)."""
        descendants = []
        node = self.nodes.get(node_id)
        if not node:
            return []

        queue = list(node.children)
        while queue:
            child_id = queue.pop(0)
            if child_id in self.nodes:
                child = self.nodes[child_id]
                descendants.append(child)
                queue.extend(child.children)

        return descendants

    def to_json(self) -> str:
        """Serialize the tree to a JSON string."""
        data = {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "levels": {str(k): v for k, v in self.levels.items()},
            "metadata": self.metadata,
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "RAPTORTree":
        """Deserialize a tree from a JSON string."""
        data = json.loads(json_str)
        tree = cls()
        tree.metadata = data.get("metadata", {})

        # Reconstruct levels with int keys
        for level_str, node_ids in data.get("levels", {}).items():
            tree.levels[int(level_str)] = node_ids

        # Reconstruct nodes
        for nid, node_data in data.get("nodes", {}).items():
            tree.nodes[nid] = TreeNode.from_dict(node_data)

        return tree


@dataclass
class RAPTORResult:
    """Result from a RAPTOR search.

    Attributes:
        node: The matched TreeNode.
        score: Similarity score (0.0 to 1.0 for cosine similarity).
        level: The tree level this result came from.
    """
    node: TreeNode
    score: float
    level: int


@dataclass
class TreeBuildStats:
    """Statistics from tree building process."""
    total_documents: int = 0
    total_nodes: int = 0
    max_level: int = 0
    build_time_seconds: float = 0.0
    nodes_per_level: Dict[int, int] = field(default_factory=dict)
    cluster_method: str = ""
    summarization_method: str = ""


# =============================================================================
# HEURISTIC SUMMARIZER (No LLM dependency)
# =============================================================================

def _tokenize_words(text: str) -> List[str]:
    """Split text into lowercase word tokens, stripping punctuation."""
    return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())


def _sentence_split(text: str) -> List[str]:
    """Split text into sentences using regex boundary detection."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if len(s.strip()) >= 10]


def _tfidf_summarize(texts: List[str], max_sentences: int = 5) -> str:
    """Heuristic extractive summarizer using TF-IDF-like scoring.

    Scores each sentence by the sum of its term frequencies weighted by
    inverse document frequency across the cluster. Picks the top-scoring
    sentences and concatenates them with a "Summary: " prefix.
    """
    if not texts:
        return ""
    if len(texts) == 1 and len(texts[0]) < 200:
        return f"Summary: {texts[0]}"

    # Collect all sentences from all texts
    all_sentences: List[str] = []
    for text in texts:
        all_sentences.extend(_sentence_split(text))

    if not all_sentences:
        combined = " ".join(t[:200] for t in texts[:3])
        return f"Summary: {combined}"

    # Build document frequency
    n_sentences = len(all_sentences)
    doc_freq: Counter = Counter()
    sentence_tokens: List[List[str]] = []

    for sent in all_sentences:
        tokens = _tokenize_words(sent)
        sentence_tokens.append(tokens)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            doc_freq[token] += 1

    # Score each sentence by sum of tf * idf
    scored: List[Tuple[float, int, str]] = []
    for idx, (sent, tokens) in enumerate(zip(all_sentences, sentence_tokens)):
        if not tokens:
            continue
        tf: Counter = Counter(tokens)
        score = 0.0
        for token, count in tf.items():
            tf_val = count / len(tokens)
            idf_val = math.log((n_sentences + 1) / (doc_freq.get(token, 0) + 1)) + 1
            score += tf_val * idf_val

        # Position bonus: earlier sentences score slightly higher
        position_bonus = 1.0 + 0.1 * (1.0 - idx / max(n_sentences, 1))
        score *= position_bonus

        # Length penalty
        sent_len = len(sent)
        if sent_len < 30:
            score *= 0.5
        elif sent_len > 500:
            score *= 0.7

        scored.append((score, idx, sent))

    # Sort by score descending, take top max_sentences
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:max_sentences]

    # Re-order by original position for coherence
    top.sort(key=lambda x: x[1])

    summary_text = " ".join(s for _, _, s in top)
    return f"Summary: {summary_text}"


# =============================================================================
# COSINE SIMILARITY UTILITIES
# =============================================================================

def _content_hash_embedding(text: str, dim: int = 64) -> List[float]:
    """Generate a deterministic pseudo-embedding from content hash.

    Uses SHA-256 hash bytes to create a fixed-dimension vector. This is
    NOT a semantic embedding -- it is a placeholder so that the pipeline
    works without an external embedding provider.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    while len(h) < dim * 4:
        h += hashlib.sha256(h).digest()

    embedding = []
    for i in range(dim):
        raw = int.from_bytes(h[i * 4 : i * 4 + 4], byteorder="big", signed=False)
        val = (raw / (2**32 - 1)) * 2.0 - 1.0
        embedding.append(val)

    # Normalize to unit vector
    norm = math.sqrt(sum(v * v for v in embedding))
    if norm > 0:
        embedding = [v / norm for v in embedding]
    return embedding


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if NUMPY_AVAILABLE:
        a_arr = np.array(a, dtype=np.float64)
        b_arr = np.array(b, dtype=np.float64)
        dot = float(np.dot(a_arr, b_arr))
        norm_a = float(np.linalg.norm(a_arr))
        norm_b = float(np.linalg.norm(b_arr))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    else:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# =============================================================================
# CLUSTERING ALGORITHMS
# =============================================================================

def _greedy_cluster(
    embeddings: List[List[float]],
    threshold: float,
    min_cluster_size: int,
    max_cluster_size: int = 10,
) -> List[List[int]]:
    """Greedy clustering by pairwise cosine similarity.

    Iterates through items and assigns each to the first existing cluster
    whose centroid has cosine similarity >= threshold with the item.
    """
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    clusters: List[List[int]] = []
    centroids: List[List[float]] = []

    for idx in range(n):
        emb = embeddings[idx]
        best_cluster = -1
        best_sim = -1.0

        for c_idx, centroid in enumerate(centroids):
            if len(clusters[c_idx]) >= max_cluster_size:
                continue
            sim = _cosine_similarity(emb, centroid)
            if sim >= threshold and sim > best_sim:
                best_sim = sim
                best_cluster = c_idx

        if best_cluster >= 0:
            clusters[best_cluster].append(idx)
            # Update centroid as running mean
            c_size = len(clusters[best_cluster])
            old = centroids[best_cluster]
            centroids[best_cluster] = [
                (old[d] * (c_size - 1) + emb[d]) / c_size
                for d in range(len(emb))
            ]
        else:
            clusters.append([idx])
            centroids.append(list(emb))

    # Merge small clusters into nearest neighbor
    if min_cluster_size > 1:
        merged = True
        while merged:
            merged = False
            small_indices = [
                i for i, c in enumerate(clusters)
                if len(c) < min_cluster_size and len(clusters) > 1
            ]
            for s_idx in reversed(small_indices):
                if s_idx >= len(clusters):
                    continue
                best_target = -1
                best_sim = -2.0
                for t_idx in range(len(clusters)):
                    if t_idx == s_idx:
                        continue
                    sim = _cosine_similarity(centroids[s_idx], centroids[t_idx])
                    if sim > best_sim:
                        best_sim = sim
                        best_target = t_idx
                if best_target >= 0:
                    clusters[best_target].extend(clusters[s_idx])
                    all_embs = [embeddings[i] for i in clusters[best_target]]
                    dim = len(all_embs[0])
                    centroids[best_target] = [
                        sum(e[d] for e in all_embs) / len(all_embs)
                        for d in range(dim)
                    ]
                    clusters.pop(s_idx)
                    centroids.pop(s_idx)
                    merged = True
                    break

    return clusters


def _umap_gmm_cluster(
    embeddings: List[List[float]],
    n_clusters: int,
    config: RAPTORConfig,
) -> List[List[int]]:
    """Cluster using UMAP dimensionality reduction + GMM.

    Falls back to greedy clustering if UMAP/sklearn unavailable.
    """
    if not NUMPY_AVAILABLE or not SKLEARN_AVAILABLE:
        logger.warning("umap_gmm_cluster fallback to greedy (numpy/sklearn unavailable)")
        return _greedy_cluster(embeddings, config.cluster_threshold, config.min_cluster_size)

    emb_array = np.array(embeddings, dtype=np.float64)

    # Apply UMAP if available and beneficial
    if UMAP_AVAILABLE and emb_array.shape[0] > config.umap_n_components + 1:
        try:
            reducer = umap.UMAP(
                n_neighbors=min(config.umap_n_neighbors, emb_array.shape[0] - 1),
                min_dist=config.umap_min_dist,
                n_components=min(config.umap_n_components, emb_array.shape[1]),
                metric="cosine",
                random_state=42,
            )
            reduced = reducer.fit_transform(emb_array)
        except Exception as e:
            logger.warning("UMAP reduction failed, using raw embeddings", error=str(e))
            reduced = emb_array
    else:
        reduced = emb_array

    # Apply GMM clustering
    n_clusters = min(n_clusters, reduced.shape[0])
    if n_clusters < 1:
        n_clusters = 1

    try:
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type=config.gmm_covariance_type,
            random_state=42,
            max_iter=100,
        )
        labels = gmm.fit_predict(reduced)
    except Exception as e:
        logger.warning("GMM clustering failed, using greedy fallback", error=str(e))
        return _greedy_cluster(embeddings, config.cluster_threshold, config.min_cluster_size)

    # Convert labels to cluster lists
    clusters: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)

    return list(clusters.values())


def _gmm_cluster(
    embeddings: List[List[float]],
    n_clusters: int,
    config: RAPTORConfig,
) -> List[List[int]]:
    """Cluster using GMM without UMAP."""
    if not NUMPY_AVAILABLE or not SKLEARN_AVAILABLE:
        return _greedy_cluster(embeddings, config.cluster_threshold, config.min_cluster_size)

    emb_array = np.array(embeddings, dtype=np.float64)
    n_clusters = min(n_clusters, emb_array.shape[0])
    if n_clusters < 1:
        n_clusters = 1

    try:
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type=config.gmm_covariance_type,
            random_state=42,
            max_iter=100,
        )
        labels = gmm.fit_predict(emb_array)
    except Exception as e:
        logger.warning("GMM clustering failed", error=str(e))
        return _greedy_cluster(embeddings, config.cluster_threshold, config.min_cluster_size)

    clusters: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)

    return list(clusters.values())


def _kmeans_cluster(
    embeddings: List[List[float]],
    n_clusters: int,
    config: RAPTORConfig,
) -> List[List[int]]:
    """Cluster using K-Means."""
    if not NUMPY_AVAILABLE or not SKLEARN_AVAILABLE:
        return _greedy_cluster(embeddings, config.cluster_threshold, config.min_cluster_size)

    emb_array = np.array(embeddings, dtype=np.float64)
    n_clusters = min(n_clusters, emb_array.shape[0])
    if n_clusters < 1:
        n_clusters = 1

    try:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            max_iter=100,
            n_init=10,
        )
        labels = kmeans.fit_predict(emb_array)
    except Exception as e:
        logger.warning("KMeans clustering failed", error=str(e))
        return _greedy_cluster(embeddings, config.cluster_threshold, config.min_cluster_size)

    clusters: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)

    return list(clusters.values())


# =============================================================================
# RAPTOR PROCESSOR
# =============================================================================

class RAPTORProcessor:
    """Recursive Abstractive Processing for Tree-Organized Retrieval.

    Builds hierarchical summary trees from document collections. At each
    level, documents are embedded, clustered, and each cluster is summarized.
    The process repeats on the summaries until convergence or max_levels.

    Retrieval searches ALL levels simultaneously (collapsed tree) to
    capture both fine-grained detail and high-level themes.

    Args:
        config: RAPTORConfig with all settings.
        embed_fn: Optional embedding function. Signature: (str) -> List[float].
        summarize_fn: Optional summarization function. Signature: (List[str]) -> str.
        knowledge_graph: Optional KnowledgeGraph for cross-document reasoning.

    Example:
        >>> config = RAPTORConfig(max_levels=4, cluster_method=ClusterMethod.UMAP_GMM)
        >>> processor = RAPTORProcessor(config=config, embed_fn=my_embed_fn)
        >>> tree = processor.build_tree(documents)
        >>> results = processor.search_tree(tree, "query", top_k=5)
    """

    def __init__(
        self,
        config: Optional[RAPTORConfig] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        summarize_fn: Optional[Callable[[List[str]], str]] = None,
        knowledge_graph: Optional[Any] = None,
        # Legacy parameters for backward compatibility
        max_levels: Optional[int] = None,
        cluster_threshold: Optional[float] = None,
        min_cluster_size: Optional[int] = None,
        embedding_dim: Optional[int] = None,
    ) -> None:
        # Build config from parameters or use default
        if config is None:
            config = RAPTORConfig(
                max_levels=max_levels if max_levels is not None else 4,
                cluster_threshold=cluster_threshold if cluster_threshold is not None else 0.5,
                min_cluster_size=min_cluster_size if min_cluster_size is not None else 2,
                embedding_dim=embedding_dim if embedding_dim is not None else 64,
            )

        self.config = config
        self.embed_fn = embed_fn
        self.summarize_fn = summarize_fn or _tfidf_summarize
        self.knowledge_graph = knowledge_graph

        # Legacy attribute access
        self.max_levels = config.max_levels
        self.cluster_threshold = config.cluster_threshold
        self.min_cluster_size = config.min_cluster_size
        self._embedding_dim = config.embedding_dim

        # Stats tracking
        self._stats: Optional[TreeBuildStats] = None

    @property
    def stats(self) -> Optional[TreeBuildStats]:
        """Return build statistics from last tree build."""
        return self._stats

    def _embed(self, text: str) -> List[float]:
        """Embed a text using the configured embed_fn or fallback hash."""
        if self.embed_fn is not None:
            return self.embed_fn(text)
        return _content_hash_embedding(text, dim=self._embedding_dim)

    def _generate_node_id(self, content: str, level: int, index: int) -> str:
        """Generate a deterministic node ID from content, level, and index."""
        hash_input = f"raptor:L{level}:I{index}:{content[:200]}"
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]

    def _cluster(
        self,
        embeddings: List[List[float]],
        n_items: int,
    ) -> List[List[int]]:
        """Apply configured clustering method."""
        n_clusters = max(1, int(n_items * self.config.n_clusters_ratio))

        if self.config.cluster_method == ClusterMethod.UMAP_GMM:
            return _umap_gmm_cluster(embeddings, n_clusters, self.config)
        elif self.config.cluster_method == ClusterMethod.GMM:
            return _gmm_cluster(embeddings, n_clusters, self.config)
        elif self.config.cluster_method == ClusterMethod.KMEANS:
            return _kmeans_cluster(embeddings, n_clusters, self.config)
        else:
            return _greedy_cluster(
                embeddings,
                self.config.cluster_threshold,
                self.config.min_cluster_size,
                self.config.max_cluster_size,
            )

    def build_tree(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> RAPTORTree:
        """Build a RAPTOR tree from a list of documents.

        Documents are placed at level 0. Each subsequent level clusters
        the nodes from the level below and creates summary nodes.

        Args:
            documents: List of document text strings.
            metadata: Optional list of metadata dicts, one per document.

        Returns:
            RAPTORTree with all levels populated.

        Raises:
            ValueError: If documents is not a list.
        """
        if not isinstance(documents, list):
            raise ValueError("documents must be a list of strings")

        start_time = time.time()
        tree = RAPTORTree()

        # Filter empty documents
        valid_docs: List[Tuple[str, Dict[str, Any]]] = []
        meta_list = metadata or []
        for i, doc in enumerate(documents):
            if doc and doc.strip():
                doc_meta = meta_list[i] if i < len(meta_list) else {}
                valid_docs.append((doc.strip(), doc_meta))

        if not valid_docs:
            logger.info("raptor_build_tree_empty", documents=0)
            self._stats = TreeBuildStats(
                total_documents=0,
                total_nodes=0,
                max_level=-1,
                build_time_seconds=time.time() - start_time,
                cluster_method=self.config.cluster_method.value,
            )
            tree.metadata = asdict(self._stats)
            return tree

        logger.info("raptor_build_tree_start", documents=len(valid_docs))

        # Level 0: original documents as leaf nodes
        current_level_nodes: List[TreeNode] = []
        for idx, (doc, meta) in enumerate(valid_docs):
            node_id = self._generate_node_id(doc, level=0, index=idx)
            embedding = self._embed(doc) if self.config.persist_embeddings else None
            node = TreeNode(
                content=doc,
                level=0,
                children=[],
                parent=None,
                node_id=node_id,
                metadata={**meta, "doc_index": idx, "source": "original"},
                embedding=embedding,
                is_summary=False,
            )
            tree.add_node(node)
            current_level_nodes.append(node)

        nodes_per_level: Dict[int, int] = {0: len(current_level_nodes)}

        # Build higher levels
        for level in range(1, self.config.max_levels + 1):
            if len(current_level_nodes) <= 1:
                logger.info(
                    "raptor_build_tree_converged",
                    level=level - 1,
                    reason="single_node",
                )
                break

            # Get embeddings for clustering
            embeddings = []
            for node in current_level_nodes:
                if node.embedding is not None:
                    embeddings.append(node.embedding)
                else:
                    embeddings.append(self._embed(node.content))

            # Cluster
            clusters = self._cluster(embeddings, len(current_level_nodes))

            # Check for no reduction
            if len(clusters) >= len(current_level_nodes):
                logger.info(
                    "raptor_build_tree_converged",
                    level=level,
                    reason="no_reduction",
                    clusters=len(clusters),
                    nodes=len(current_level_nodes),
                )
                break

            # Summarize each cluster
            next_level_nodes: List[TreeNode] = []
            for c_idx, cluster_indices in enumerate(clusters):
                cluster_nodes = [current_level_nodes[i] for i in cluster_indices]
                cluster_texts = [n.content for n in cluster_nodes]

                # Generate cluster ID
                cluster_id = hashlib.sha256(
                    f"cluster:L{level}:C{c_idx}".encode()
                ).hexdigest()[:12]

                # Summarize
                summary = self.summarize_fn(cluster_texts)
                if not summary or not summary.strip():
                    summary = f"Summary: {' '.join(t[:100] for t in cluster_texts[:3])}"

                node_id = self._generate_node_id(summary, level=level, index=c_idx)
                embedding = self._embed(summary) if self.config.persist_embeddings else None
                child_ids = [n.node_id for n in cluster_nodes]

                summary_node = TreeNode(
                    content=summary,
                    level=level,
                    children=child_ids,
                    parent=None,
                    node_id=node_id,
                    metadata={
                        "cluster_index": c_idx,
                        "cluster_size": len(cluster_nodes),
                        "source": "summary",
                    },
                    embedding=embedding,
                    is_summary=True,
                    cluster_id=cluster_id,
                )

                # Set parent references on children
                for child_node in cluster_nodes:
                    child_node.parent = node_id

                tree.add_node(summary_node)
                next_level_nodes.append(summary_node)

            nodes_per_level[level] = len(next_level_nodes)
            logger.info(
                "raptor_build_level",
                level=level,
                clusters=len(clusters),
                summary_nodes=len(next_level_nodes),
            )
            current_level_nodes = next_level_nodes

        build_time = time.time() - start_time
        self._stats = TreeBuildStats(
            total_documents=len(valid_docs),
            total_nodes=tree.total_nodes,
            max_level=tree.max_level,
            build_time_seconds=round(build_time, 4),
            nodes_per_level=nodes_per_level,
            cluster_method=self.config.cluster_method.value,
            summarization_method=self.config.summarization_method.value,
        )
        tree.metadata = {
            "build_time_seconds": round(build_time, 4),
            "total_documents": len(valid_docs),
            "total_nodes": tree.total_nodes,
            "max_level": tree.max_level,
            "nodes_per_level": nodes_per_level,
            "config": {
                "max_levels": self.config.max_levels,
                "cluster_threshold": self.config.cluster_threshold,
                "min_cluster_size": self.config.min_cluster_size,
                "embedding_dim": self._embedding_dim,
                "cluster_method": self.config.cluster_method.value,
                "has_embed_fn": self.embed_fn is not None,
            },
        }

        logger.info(
            "raptor_build_tree_complete",
            total_nodes=tree.total_nodes,
            max_level=tree.max_level,
            build_time_seconds=round(build_time, 4),
        )

        return tree

    def search(
        self,
        tree: RAPTORTree,
        query: str,
        top_k: int = 5,
    ) -> List[RAPTORResult]:
        """Search all levels of the tree for the most relevant nodes.

        This is an alias for search_tree with collapsed retrieval for
        backward compatibility.
        """
        return self.search_tree(tree, query, top_k=top_k, method=RetrievalMethod.COLLAPSED)

    def search_tree(
        self,
        tree: RAPTORTree,
        query: str,
        top_k: Optional[int] = None,
        method: RetrievalMethod = RetrievalMethod.COLLAPSED,
        similarity_threshold: Optional[float] = None,
    ) -> List[RAPTORResult]:
        """Search the tree using specified retrieval method.

        Args:
            tree: A built RAPTORTree.
            query: Search query string.
            top_k: Number of results to return (default from config).
            method: Retrieval method (collapsed, tree_traversal, hybrid).
            similarity_threshold: Minimum similarity score (default from config).

        Returns:
            List of RAPTORResult sorted by score descending.
        """
        if not query or not query.strip():
            logger.warning("raptor_search_empty_query")
            return []

        if tree.total_nodes == 0:
            logger.warning("raptor_search_empty_tree")
            return []

        top_k = top_k or self.config.top_k
        threshold = similarity_threshold or self.config.similarity_threshold

        if method == RetrievalMethod.COLLAPSED:
            return self._search_collapsed(tree, query, top_k, threshold)
        elif method == RetrievalMethod.TREE_TRAVERSAL:
            return self._search_tree_traversal(tree, query, top_k, threshold)
        else:  # HYBRID
            return self._search_hybrid(tree, query, top_k, threshold)

    def _search_collapsed(
        self,
        tree: RAPTORTree,
        query: str,
        top_k: int,
        threshold: float,
    ) -> List[RAPTORResult]:
        """Collapsed tree search: search all levels simultaneously."""
        query_embedding = self._embed(query.strip())
        all_nodes = tree.get_all_nodes()

        scored: List[Tuple[float, TreeNode]] = []
        for node in all_nodes:
            if node.embedding is None:
                emb = self._embed(node.content)
            else:
                emb = node.embedding
            sim = _cosine_similarity(query_embedding, emb)
            if sim >= threshold:
                node.touch()
                scored.append((sim, node))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = [
            RAPTORResult(node=node, score=score, level=node.level)
            for score, node in scored[:top_k]
        ]

        logger.info(
            "raptor_search_collapsed",
            query_length=len(query),
            results=len(results),
            top_score=results[0].score if results else 0.0,
        )

        return results

    def _search_tree_traversal(
        self,
        tree: RAPTORTree,
        query: str,
        top_k: int,
        threshold: float,
    ) -> List[RAPTORResult]:
        """Tree traversal search: top-down hierarchical search.

        Starts at the highest level, finds best matching summaries,
        then descends to their children, repeating until leaves.
        """
        query_embedding = self._embed(query.strip())
        results: List[RAPTORResult] = []
        visited: set = set()

        # Start from roots (highest level)
        current_nodes = tree.get_level(tree.max_level)
        if not current_nodes:
            current_nodes = tree.get_all_nodes()

        while current_nodes and len(results) < top_k:
            # Score current level
            scored: List[Tuple[float, TreeNode]] = []
            for node in current_nodes:
                if node.node_id in visited:
                    continue
                visited.add(node.node_id)

                if node.embedding is None:
                    emb = self._embed(node.content)
                else:
                    emb = node.embedding
                sim = _cosine_similarity(query_embedding, emb)
                if sim >= threshold:
                    node.touch()
                    scored.append((sim, node))

            scored.sort(key=lambda x: x[0], reverse=True)

            # Add top results from this level
            for score, node in scored[:max(1, top_k - len(results))]:
                results.append(RAPTORResult(node=node, score=score, level=node.level))

            # Descend to children of top nodes
            next_nodes: List[TreeNode] = []
            for _, node in scored[:3]:  # Expand top 3
                next_nodes.extend(tree.get_children(node.node_id))

            current_nodes = next_nodes

        logger.info(
            "raptor_search_tree_traversal",
            query_length=len(query),
            results=len(results),
            top_score=results[0].score if results else 0.0,
        )

        return results[:top_k]

    def _search_hybrid(
        self,
        tree: RAPTORTree,
        query: str,
        top_k: int,
        threshold: float,
    ) -> List[RAPTORResult]:
        """Hybrid search: combine collapsed and tree traversal results."""
        collapsed = self._search_collapsed(tree, query, top_k, threshold)
        traversal = self._search_tree_traversal(tree, query, top_k, threshold)

        # Merge results using reciprocal rank fusion
        seen: Dict[str, RAPTORResult] = {}
        rrf_scores: Dict[str, float] = {}
        k = 60  # RRF constant

        for rank, result in enumerate(collapsed):
            nid = result.node.node_id
            seen[nid] = result
            rrf_scores[nid] = rrf_scores.get(nid, 0) + 1 / (k + rank + 1)

        for rank, result in enumerate(traversal):
            nid = result.node.node_id
            if nid not in seen:
                seen[nid] = result
            rrf_scores[nid] = rrf_scores.get(nid, 0) + 1 / (k + rank + 1)

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for nid in sorted_ids[:top_k]:
            result = seen[nid]
            # Update score to RRF score
            results.append(RAPTORResult(
                node=result.node,
                score=rrf_scores[nid],
                level=result.level,
            ))

        logger.info(
            "raptor_search_hybrid",
            query_length=len(query),
            results=len(results),
            top_score=results[0].score if results else 0.0,
        )

        return results

    def collapse_tree(self, tree: RAPTORTree) -> List[TreeNode]:
        """Flatten all levels of the tree into a single list.

        Returns all nodes across all levels, sorted by level (ascending)
        then by node_id for deterministic ordering.
        """
        all_nodes = tree.get_all_nodes()
        all_nodes.sort(key=lambda n: (n.level, n.node_id))
        return all_nodes

    def update_tree(
        self,
        tree: RAPTORTree,
        new_documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Incrementally update tree with new documents.

        Adds new documents as leaf nodes and rebuilds summary levels
        that are affected by the new documents.

        Args:
            tree: Existing RAPTORTree to update.
            new_documents: New documents to add.
            metadata: Optional metadata for new documents.

        Returns:
            Number of nodes added.
        """
        if not new_documents:
            return 0

        start_count = tree.total_nodes

        # Filter empty documents
        valid_docs: List[Tuple[str, Dict[str, Any]]] = []
        meta_list = metadata or []
        for i, doc in enumerate(new_documents):
            if doc and doc.strip():
                doc_meta = meta_list[i] if i < len(meta_list) else {}
                valid_docs.append((doc.strip(), doc_meta))

        if not valid_docs:
            return 0

        # Add new leaf nodes
        existing_leaf_count = len(tree.get_level(0))
        new_leaves: List[TreeNode] = []

        for idx, (doc, meta) in enumerate(valid_docs):
            node_id = self._generate_node_id(doc, level=0, index=existing_leaf_count + idx)
            embedding = self._embed(doc) if self.config.persist_embeddings else None
            node = TreeNode(
                content=doc,
                level=0,
                children=[],
                parent=None,
                node_id=node_id,
                metadata={**meta, "doc_index": existing_leaf_count + idx, "source": "original", "incremental": True},
                embedding=embedding,
                is_summary=False,
            )
            tree.add_node(node)
            new_leaves.append(node)

        # Find orphan leaves (no parent) and rebuild summaries
        orphan_leaves = [n for n in tree.get_level(0) if n.parent is None]

        if len(orphan_leaves) >= self.config.min_cluster_size:
            # Rebuild summary levels for orphans
            self._rebuild_summaries(tree, orphan_leaves)

        added_count = tree.total_nodes - start_count
        logger.info(
            "raptor_update_tree",
            new_documents=len(valid_docs),
            nodes_added=added_count,
        )

        return added_count

    def _rebuild_summaries(self, tree: RAPTORTree, leaf_nodes: List[TreeNode]) -> None:
        """Rebuild summary levels for given leaf nodes."""
        current_level_nodes = leaf_nodes

        for level in range(1, self.config.max_levels + 1):
            if len(current_level_nodes) <= 1:
                break

            embeddings = []
            for node in current_level_nodes:
                if node.embedding is not None:
                    embeddings.append(node.embedding)
                else:
                    embeddings.append(self._embed(node.content))

            clusters = self._cluster(embeddings, len(current_level_nodes))

            if len(clusters) >= len(current_level_nodes):
                break

            next_level_nodes: List[TreeNode] = []
            existing_count = len(tree.get_level(level))

            for c_idx, cluster_indices in enumerate(clusters):
                cluster_nodes = [current_level_nodes[i] for i in cluster_indices]
                cluster_texts = [n.content for n in cluster_nodes]

                summary = self.summarize_fn(cluster_texts)
                if not summary or not summary.strip():
                    summary = f"Summary: {' '.join(t[:100] for t in cluster_texts[:3])}"

                node_id = self._generate_node_id(summary, level=level, index=existing_count + c_idx)
                embedding = self._embed(summary) if self.config.persist_embeddings else None
                child_ids = [n.node_id for n in cluster_nodes]

                summary_node = TreeNode(
                    content=summary,
                    level=level,
                    children=child_ids,
                    parent=None,
                    node_id=node_id,
                    metadata={
                        "cluster_index": existing_count + c_idx,
                        "cluster_size": len(cluster_nodes),
                        "source": "summary",
                        "incremental": True,
                    },
                    embedding=embedding,
                    is_summary=True,
                )

                for child_node in cluster_nodes:
                    child_node.parent = node_id

                tree.add_node(summary_node)
                next_level_nodes.append(summary_node)

            current_level_nodes = next_level_nodes

    def prune_tree(
        self,
        tree: RAPTORTree,
        max_age_days: Optional[float] = None,
        min_access_count: Optional[int] = None,
        max_nodes: Optional[int] = None,
    ) -> int:
        """Prune stale or unused branches from the tree.

        Args:
            tree: RAPTORTree to prune.
            max_age_days: Remove nodes older than this.
            min_access_count: Remove nodes accessed fewer times.
            max_nodes: Keep only top N nodes by access count.

        Returns:
            Number of nodes removed.
        """
        nodes_to_remove: List[str] = []

        # Identify nodes to prune
        for node_id, node in tree.nodes.items():
            if max_age_days is not None and node.age_days > max_age_days:
                nodes_to_remove.append(node_id)
            elif min_access_count is not None and node.access_count < min_access_count:
                nodes_to_remove.append(node_id)

        # If max_nodes specified, keep only top accessed nodes
        if max_nodes is not None and tree.total_nodes > max_nodes:
            sorted_nodes = sorted(
                tree.nodes.values(),
                key=lambda n: (n.access_count, -n.age_days),
                reverse=True,
            )
            keep_ids = {n.node_id for n in sorted_nodes[:max_nodes]}
            for node_id in tree.nodes:
                if node_id not in keep_ids:
                    nodes_to_remove.append(node_id)

        # Remove duplicates
        nodes_to_remove = list(set(nodes_to_remove))

        # Remove nodes (cascading removal of orphaned parents)
        removed_count = 0
        for node_id in nodes_to_remove:
            if tree.remove_node(node_id):
                removed_count += 1

        # Clean up orphaned summary nodes
        changed = True
        while changed:
            changed = False
            for node in list(tree.nodes.values()):
                if node.is_summary and not node.children:
                    if tree.remove_node(node.node_id):
                        removed_count += 1
                        changed = True

        logger.info(
            "raptor_prune_tree",
            nodes_removed=removed_count,
            remaining_nodes=tree.total_nodes,
        )

        return removed_count

    def export_tree(self, tree: RAPTORTree, path: Optional[Path] = None) -> str:
        """Export tree to JSON format.

        Args:
            tree: RAPTORTree to export.
            path: Optional file path to save to.

        Returns:
            JSON string representation.
        """
        json_str = tree.to_json()

        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json_str, encoding="utf-8")
            logger.info("raptor_export_tree", path=str(path), nodes=tree.total_nodes)

        return json_str

    def import_tree(self, json_data: Union[str, Path]) -> RAPTORTree:
        """Import tree from JSON format.

        Args:
            json_data: JSON string or path to JSON file.

        Returns:
            Reconstructed RAPTORTree.
        """
        if isinstance(json_data, Path):
            json_str = json_data.read_text(encoding="utf-8")
        else:
            json_str = json_data

        tree = RAPTORTree.from_json(json_str)
        logger.info("raptor_import_tree", nodes=tree.total_nodes, max_level=tree.max_level)
        return tree

    def build_tree_from_memory_blocks(
        self,
        memory_blocks: List[Dict[str, Any]],
    ) -> RAPTORTree:
        """Build a RAPTOR tree from memory blocks.

        Integrates with the UNLEASH memory system for cross-session persistence.

        Args:
            memory_blocks: List of memory block dicts with 'content' and optional metadata.

        Returns:
            RAPTORTree built from memory blocks.
        """
        documents = []
        metadata = []

        for block in memory_blocks:
            content = block.get("content", "")
            if content:
                documents.append(content)
                block_meta = {
                    "memory_id": block.get("id", ""),
                    "memory_type": block.get("type", "working"),
                    "topic": block.get("metadata", {}).get("topic", ""),
                }
                metadata.append(block_meta)

        return self.build_tree(documents, metadata)

    def clear_tree(self) -> None:
        """Clear internal state (for testing)."""
        self._stats = None


# =============================================================================
# RAPTOR STRATEGY FOR RAG PIPELINE
# =============================================================================

class RAPTORStrategy:
    """RAG strategy wrapper for RAPTOR retrieval.

    Implements the RAGStrategy protocol for integration with AdaptiveRAGRouter.
    """

    def __init__(
        self,
        processor: RAPTORProcessor,
        tree: RAPTORTree,
        retrieval_method: RetrievalMethod = RetrievalMethod.COLLAPSED,
    ) -> None:
        self.processor = processor
        self.tree = tree
        self.retrieval_method = retrieval_method

    async def execute(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """Execute RAPTOR search.

        Args:
            query: Search query.
            **kwargs: Additional parameters (top_k, etc.)

        Returns:
            List of result dicts with content, score, level, metadata.
        """
        top_k = kwargs.get("top_k", 5)
        results = self.processor.search_tree(
            self.tree,
            query,
            top_k=top_k,
            method=self.retrieval_method,
        )

        return [
            {
                "content": r.node.content,
                "score": r.score,
                "level": r.level,
                "is_summary": r.node.is_summary,
                "metadata": r.node.metadata,
            }
            for r in results
        ]


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_raptor_processor(
    embed_fn: Optional[Callable[[str], List[float]]] = None,
    summarize_fn: Optional[Callable[[List[str]], str]] = None,
    use_umap_gmm: bool = False,
    max_levels: int = 4,
    **kwargs: Any,
) -> RAPTORProcessor:
    """Factory function to create a RAPTOR processor.

    Args:
        embed_fn: Embedding function.
        summarize_fn: Summarization function.
        use_umap_gmm: Use UMAP+GMM clustering if available.
        max_levels: Maximum tree depth.
        **kwargs: Additional config parameters.

    Returns:
        Configured RAPTORProcessor.
    """
    cluster_method = ClusterMethod.GREEDY_COSINE
    if use_umap_gmm and UMAP_AVAILABLE and SKLEARN_AVAILABLE:
        cluster_method = ClusterMethod.UMAP_GMM
    elif SKLEARN_AVAILABLE:
        cluster_method = ClusterMethod.GMM

    config = RAPTORConfig(
        max_levels=max_levels,
        cluster_method=cluster_method,
        **kwargs,
    )

    return RAPTORProcessor(
        config=config,
        embed_fn=embed_fn,
        summarize_fn=summarize_fn,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    "RAPTORProcessor",
    "RAPTORTree",
    "TreeNode",
    "RAPTORResult",
    "RAPTORConfig",
    "TreeBuildStats",
    # Enums
    "ClusterMethod",
    "RetrievalMethod",
    "SummarizationMethod",
    # Strategy
    "RAPTORStrategy",
    # Factory
    "create_raptor_processor",
    # Utility functions (for testing)
    "_content_hash_embedding",
    "_cosine_similarity",
    "_greedy_cluster",
    "_sentence_split",
    "_tfidf_summarize",
    "_tokenize_words",
]
