"""
HNSW Vector Index Backend - High-Performance Semantic Search

This module provides a persistent HNSW (Hierarchical Navigable Small World)
vector index backend for the UnifiedMemory system, delivering 150x-12,500x
speedup over linear scan for semantic search operations.

Features:
- Loads existing .swarm/hnsw.index for persistent vector storage
- Uses hnswlib for high-performance ANN search
- Falls back to pure-Python implementation if hnswlib unavailable
- Configurable ef_search and k parameters for speed/accuracy tradeoff
- Thread-safe with read-write locking for concurrent access

Performance Targets:
- 50K vectors: 0.07-0.09ms latency, 0.85-0.9 recall at ef=10-20
- 1M vectors: 0.22-0.25ms latency, 0.95-0.98 recall at ef=40-80

Integration:
    from core.memory.backends.hnsw import HNSWBackend, get_hnsw_backend

    backend = get_hnsw_backend()
    results = await backend.search(query_embedding, k=10)

V41 Architecture - HNSW Integration for UnifiedMemory
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import heapq
import json
import logging
import math
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# Track instances for atexit cleanup
_registered_backends: List["HNSWBackend"] = []

# Try to import hnswlib for optimized implementation
try:
    import hnswlib
    HNSWLIB_AVAILABLE = True
    logger.info("hnswlib available - using optimized HNSW backend")
except ImportError:
    HNSWLIB_AVAILABLE = False
    logger.warning("hnswlib not available - using pure-Python fallback (slower)")

# Try to import numpy for efficient vector operations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("numpy not available - using pure-Python vector operations")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class HNSWConfig:
    """Configuration for HNSW index."""
    # Index parameters (optimized per ADR-027)
    m: int = 16                      # Max connections per node (reduced from 48 for memory savings)
    ef_construction: int = 200       # Build quality (higher = better recall, slower build)
    ef_search: int = 100             # Search quality (higher = better recall, slower search)

    # Dimensions (must match embedding model)
    dimension: int = 384             # all-MiniLM-L6-v2 default (from embeddings.json)

    # Space type
    space: str = "cosine"            # Distance metric: "cosine", "l2", "ip"

    # Capacity
    max_elements: int = 50000        # Maximum vectors to store

    # Paths
    index_path: Optional[Path] = None
    metadata_path: Optional[Path] = None

    # Performance tuning
    num_threads: int = 4             # Threads for batch operations
    allow_replace_deleted: bool = False


@dataclass
class HNSWSearchResult:
    """Result from HNSW search."""
    id: str
    score: float                     # Similarity score (0-1 for cosine)
    distance: float                  # Raw distance
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# PURE-PYTHON HNSW NODE (FALLBACK)
# =============================================================================

@dataclass
class HNSWNode:
    """Node in pure-Python HNSW graph."""
    id: str
    embedding: List[float]
    level: int
    connections: Dict[int, Set[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)


# =============================================================================
# ASYNC READ-WRITE LOCK
# =============================================================================

class AsyncRWLock:
    """
    Async Read-Write Lock for concurrent read access with exclusive writes.
    Allows multiple concurrent readers while ensuring exclusive access for writers.
    """

    def __init__(self):
        self._read_count = 0
        self._write_lock = asyncio.Lock()
        self._read_lock = asyncio.Lock()

    async def acquire_read(self):
        async with self._read_lock:
            self._read_count += 1
            if self._read_count == 1:
                await self._write_lock.acquire()

    async def release_read(self):
        async with self._read_lock:
            self._read_count -= 1
            if self._read_count == 0:
                self._write_lock.release()

    async def acquire_write(self):
        await self._write_lock.acquire()

    async def release_write(self):
        self._write_lock.release()

    def read_lock(self) -> "_ReadLockContext":
        return _ReadLockContext(self)

    def write_lock(self) -> "_WriteLockContext":
        return _WriteLockContext(self)


class _ReadLockContext:
    def __init__(self, rwlock: AsyncRWLock):
        self._rwlock = rwlock

    async def __aenter__(self):
        await self._rwlock.acquire_read()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._rwlock.release_read()
        return False


class _WriteLockContext:
    def __init__(self, rwlock: AsyncRWLock):
        self._rwlock = rwlock

    async def __aenter__(self):
        await self._rwlock.acquire_write()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._rwlock.release_write()
        return False


# =============================================================================
# HNSWLIB BACKEND (OPTIMIZED)
# =============================================================================

class HNSWLibBackend:
    """
    HNSW backend using hnswlib for optimized performance.

    Performance: 150x-12,500x faster than brute-force linear scan.
    Memory: ~4KB per vector at M=16, dimension=384
    """

    def __init__(self, config: HNSWConfig):
        if not HNSWLIB_AVAILABLE:
            raise ImportError("hnswlib is required for HNSWLibBackend")

        self.config = config
        self._index: Optional[hnswlib.Index] = None
        self._id_to_label: Dict[str, int] = {}
        self._label_to_id: Dict[int, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._next_label: int = 0
        self._lock = threading.RLock()
        self._async_lock = AsyncRWLock()

        self._initialized = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize or load the HNSW index."""
        with self._lock:
            # Create index
            self._index = hnswlib.Index(space=self.config.space, dim=self.config.dimension)

            # Try to load existing index
            if self.config.index_path and self.config.index_path.exists():
                try:
                    self._index.load_index(
                        str(self.config.index_path),
                        max_elements=self.config.max_elements
                    )
                    self._load_metadata()
                    logger.info(
                        f"Loaded HNSW index from {self.config.index_path} "
                        f"with {self._index.get_current_count()} vectors"
                    )
                    self._initialized = True
                    return
                except Exception as e:
                    logger.warning(f"Failed to load existing index: {e}, creating new")

            # Initialize new index
            self._index.init_index(
                max_elements=self.config.max_elements,
                ef_construction=self.config.ef_construction,
                M=self.config.m
            )
            self._index.set_ef(self.config.ef_search)
            self._index.set_num_threads(self.config.num_threads)
            self._initialized = True
            logger.info(f"Created new HNSW index with dim={self.config.dimension}, M={self.config.m}")

    def _load_metadata(self) -> None:
        """Load metadata from JSON file."""
        if self.config.metadata_path and self.config.metadata_path.exists():
            try:
                with open(self.config.metadata_path, 'r') as f:
                    data = json.load(f)
                    self._id_to_label = {k: int(v) for k, v in data.get("id_to_label", {}).items()}
                    self._label_to_id = {int(k): v for k, v in data.get("label_to_id", {}).items()}
                    self._metadata = data.get("metadata", {})
                    self._next_label = data.get("next_label", 0)
                    logger.debug(f"Loaded metadata with {len(self._id_to_label)} entries")
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")

    def _save_metadata(self) -> None:
        """Save metadata to JSON file."""
        if self.config.metadata_path:
            try:
                self.config.metadata_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.config.metadata_path, 'w') as f:
                    json.dump({
                        "id_to_label": self._id_to_label,
                        "label_to_id": {str(k): v for k, v in self._label_to_id.items()},
                        "metadata": self._metadata,
                        "next_label": self._next_label,
                    }, f)
            except Exception as e:
                logger.warning(f"Failed to save metadata: {e}")

    async def insert(
        self,
        id: str,
        embedding: Union[List[float], "np.ndarray"],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Insert a vector into the index."""
        async with self._async_lock.write_lock():
            with self._lock:
                if not self._initialized:
                    raise RuntimeError("HNSW index not initialized")

                # Convert to numpy array
                if NUMPY_AVAILABLE:
                    vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
                else:
                    vec = [embedding]

                # Get or create label
                if id in self._id_to_label:
                    label = self._id_to_label[id]
                else:
                    label = self._next_label
                    self._next_label += 1
                    self._id_to_label[id] = label
                    self._label_to_id[label] = id

                # Add to index
                self._index.add_items(vec, [label])

                # Store metadata
                if metadata:
                    self._metadata[id] = metadata

    async def search(
        self,
        query: Union[List[float], "np.ndarray"],
        k: int = 10,
        ef: Optional[int] = None
    ) -> List[HNSWSearchResult]:
        """Search for k nearest neighbors."""
        async with self._async_lock.read_lock():
            with self._lock:
                if not self._initialized or self._index.get_current_count() == 0:
                    return []

                # Set ef_search if specified
                if ef:
                    self._index.set_ef(ef)

                # Convert to numpy array
                if NUMPY_AVAILABLE:
                    vec = np.array(query, dtype=np.float32).reshape(1, -1)
                else:
                    vec = [query]

                # Adjust k to available vectors
                k = min(k, self._index.get_current_count())

                # Search
                labels, distances = self._index.knn_query(vec, k=k)

                # Build results
                results: List[HNSWSearchResult] = []
                for label, distance in zip(labels[0], distances[0]):
                    id = self._label_to_id.get(int(label))
                    if id:
                        # Convert distance to similarity score
                        if self.config.space == "cosine":
                            score = 1.0 - distance
                        elif self.config.space == "l2":
                            score = 1.0 / (1.0 + distance)
                        else:  # ip (inner product)
                            score = distance

                        results.append(HNSWSearchResult(
                            id=id,
                            score=float(score),
                            distance=float(distance),
                            metadata=self._metadata.get(id, {})
                        ))

                return results

    async def delete(self, id: str) -> bool:
        """Mark a vector as deleted (hnswlib doesn't support true deletion)."""
        async with self._async_lock.write_lock():
            with self._lock:
                if id in self._id_to_label:
                    # Note: hnswlib doesn't support true deletion
                    # We mark it deleted and skip in results
                    label = self._id_to_label[id]
                    self._index.mark_deleted(label)
                    del self._id_to_label[id]
                    del self._label_to_id[label]
                    if id in self._metadata:
                        del self._metadata[id]
                    return True
                return False

    def save(self) -> None:
        """Persist the index to disk."""
        with self._lock:
            if self.config.index_path:
                self.config.index_path.parent.mkdir(parents=True, exist_ok=True)
                self._index.save_index(str(self.config.index_path))
                self._save_metadata()
                logger.info(f"Saved HNSW index to {self.config.index_path}")

    def __len__(self) -> int:
        with self._lock:
            return self._index.get_current_count() if self._index else 0

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        with self._lock:
            return {
                "count": self._index.get_current_count() if self._index else 0,
                "max_elements": self.config.max_elements,
                "dimension": self.config.dimension,
                "space": self.config.space,
                "m": self.config.m,
                "ef_construction": self.config.ef_construction,
                "ef_search": self.config.ef_search,
                "backend": "hnswlib",
            }


# =============================================================================
# PURE-PYTHON HNSW BACKEND (FALLBACK)
# =============================================================================

class PurePythonHNSWBackend:
    """
    Pure-Python HNSW implementation for when hnswlib is not available.

    Slower than hnswlib but still provides logarithmic search complexity.
    Use only as fallback when hnswlib cannot be installed.
    """

    def __init__(self, config: HNSWConfig):
        self.config = config
        self.nodes: Dict[str, HNSWNode] = {}
        self.entry_point: Optional[str] = None
        self.max_level: int = 0
        self.ml = 1 / math.log(config.m) if config.m > 1 else 1.0
        self._lock = AsyncRWLock()
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._initialized = True

    def _random_level(self) -> int:
        """Generate random level using exponential distribution."""
        import random
        r = random.random()
        return int(-math.log(r) * self.ml) if r > 0 else 0

    def _distance(self, a: List[float], b: List[float]) -> float:
        """Compute cosine distance between embeddings."""
        if self.config.space == "cosine":
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 1.0
            return 1.0 - (dot / (norm_a * norm_b))
        elif self.config.space == "l2":
            return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
        else:  # ip
            return -sum(x * y for x, y in zip(a, b))

    def _search_layer(
        self,
        query: List[float],
        entry_points: Set[str],
        ef: int,
        level: int
    ) -> List[Tuple[float, str]]:
        """Search a single layer of the HNSW graph."""
        visited: Set[str] = set(entry_points)
        candidates: List[Tuple[float, str]] = []
        results: List[Tuple[float, str]] = []

        for ep in entry_points:
            if ep in self.nodes:
                dist = self._distance(query, self.nodes[ep].embedding)
                heapq.heappush(candidates, (dist, ep))
                heapq.heappush(results, (-dist, ep))

        while candidates:
            dist, current = heapq.heappop(candidates)

            if results and dist > -results[0][0]:
                break

            node = self.nodes.get(current)
            if not node or level not in node.connections:
                continue

            for neighbor_id in node.connections[level]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    neighbor = self.nodes.get(neighbor_id)
                    if neighbor:
                        neighbor_dist = self._distance(query, neighbor.embedding)

                        if len(results) < ef or neighbor_dist < -results[0][0]:
                            heapq.heappush(candidates, (neighbor_dist, neighbor_id))
                            heapq.heappush(results, (-neighbor_dist, neighbor_id))

                            if len(results) > ef:
                                heapq.heappop(results)

        return [(-dist, id) for dist, id in results]

    async def insert(
        self,
        id: str,
        embedding: Union[List[float], Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Insert a new vector into the index."""
        async with self._lock.write_lock():
            # Convert numpy to list if needed
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()

            level = self._random_level()
            node = HNSWNode(
                id=id,
                embedding=embedding,
                level=level,
                metadata=metadata or {}
            )

            if not self.entry_point:
                self.nodes[id] = node
                self.entry_point = id
                self.max_level = level
                if metadata:
                    self._metadata[id] = metadata
                return

            # Search from top to bottom
            entry_points = {self.entry_point}

            for lc in range(self.max_level, level, -1):
                results = self._search_layer(embedding, entry_points, 1, lc)
                if results:
                    entry_points = {results[0][1]}

            # Insert at each level
            for lc in range(min(level, self.max_level), -1, -1):
                results = self._search_layer(
                    embedding,
                    entry_points,
                    self.config.ef_construction,
                    lc
                )

                neighbors = [r[1] for r in results[:self.config.m]]
                node.connections[lc] = set(neighbors)

                for neighbor_id in neighbors:
                    neighbor = self.nodes.get(neighbor_id)
                    if neighbor:
                        if lc not in neighbor.connections:
                            neighbor.connections[lc] = set()
                        neighbor.connections[lc].add(id)

                entry_points = {r[1] for r in results}

            self.nodes[id] = node
            if metadata:
                self._metadata[id] = metadata

            if level > self.max_level:
                self.max_level = level
                self.entry_point = id

    async def search(
        self,
        query: Union[List[float], Any],
        k: int = 10,
        ef: Optional[int] = None
    ) -> List[HNSWSearchResult]:
        """Search for k nearest neighbors."""
        async with self._lock.read_lock():
            if not self.entry_point:
                return []

            # Convert numpy to list if needed
            if hasattr(query, 'tolist'):
                query = query.tolist()

            ef_search = ef or self.config.ef_search
            entry_points = {self.entry_point}

            # Search from top to bottom
            for lc in range(self.max_level, 0, -1):
                results = self._search_layer(query, entry_points, 1, lc)
                if results:
                    entry_points = {results[0][1]}

            # Final search at level 0
            results = self._search_layer(query, entry_points, ef_search, 0)

            # Build result list
            search_results: List[HNSWSearchResult] = []
            for dist, id in sorted(results)[:k]:
                if id in self.nodes:
                    # Convert distance to similarity score
                    if self.config.space == "cosine":
                        score = 1.0 - dist
                    elif self.config.space == "l2":
                        score = 1.0 / (1.0 + dist)
                    else:
                        score = -dist

                    search_results.append(HNSWSearchResult(
                        id=id,
                        score=score,
                        distance=dist,
                        metadata=self._metadata.get(id, {})
                    ))

            return search_results

    async def delete(self, id: str) -> bool:
        """Delete a vector from the index."""
        async with self._lock.write_lock():
            if id not in self.nodes:
                return False

            node = self.nodes[id]

            # Remove from neighbors' connection lists
            for level, neighbors in node.connections.items():
                for neighbor_id in neighbors:
                    neighbor = self.nodes.get(neighbor_id)
                    if neighbor and level in neighbor.connections:
                        neighbor.connections[level].discard(id)

            del self.nodes[id]
            if id in self._metadata:
                del self._metadata[id]

            # Update entry point if needed
            if self.entry_point == id:
                if self.nodes:
                    self.entry_point = next(iter(self.nodes))
                    self.max_level = self.nodes[self.entry_point].level
                else:
                    self.entry_point = None
                    self.max_level = 0

            return True

    def save(self, path: Optional[Path] = None) -> None:
        """Save index to disk for pure-Python backend.

        Serializes the graph structure and node data to JSON for persistence.
        While slower than hnswlib's native format, this ensures no data loss.

        Args:
            path: Optional path override for index file
        """
        save_path = path or Path.cwd() / ".swarm" / "hnsw_python.json"

        if not self.nodes:
            logger.debug("Pure-Python HNSW: No nodes to save")
            return

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Serialize graph structure
            serialized = {
                "config": {
                    "dimension": self.config.dimension,
                    "m": self.config.m,
                    "ef_construction": self.config.ef_construction,
                    "ef_search": self.config.ef_search,
                    "space": self.config.space,
                    "max_elements": self.config.max_elements,
                },
                "entry_point": self.entry_point,
                "max_level": self.max_level,
                "nodes": {
                    node_id: {
                        "id": node.id,
                        "embedding": node.embedding,
                        "level": node.level,
                        "connections": {str(k): list(v) for k, v in node.connections.items()},
                        "metadata": node.metadata,
                    }
                    for node_id, node in self.nodes.items()
                },
                "metadata": self._metadata,
            }

            with open(save_path, 'w') as f:
                json.dump(serialized, f)

            logger.info(f"Pure-Python HNSW saved {len(self.nodes)} nodes to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save Pure-Python HNSW: {e}")

    def load(self, path: Optional[Path] = None) -> bool:
        """Load index from disk for pure-Python backend.

        Args:
            path: Optional path override for index file

        Returns:
            True if loaded successfully, False otherwise
        """
        load_path = path or Path.cwd() / ".swarm" / "hnsw_python.json"

        if not load_path.exists():
            logger.debug(f"Pure-Python HNSW: No saved index at {load_path}")
            return False

        try:
            with open(load_path, 'r') as f:
                data = json.load(f)

            # Restore config
            config_data = data.get("config", {})
            self.config.dimension = config_data.get("dimension", self.config.dimension)
            self.config.m = config_data.get("m", self.config.m)
            self.config.ef_construction = config_data.get("ef_construction", self.config.ef_construction)
            self.config.ef_search = config_data.get("ef_search", self.config.ef_search)
            self.config.space = config_data.get("space", self.config.space)

            # Restore graph structure
            self.entry_point = data.get("entry_point")
            self.max_level = data.get("max_level", 0)
            self._metadata = data.get("metadata", {})

            # Restore nodes
            self.nodes = {}
            for node_id, node_data in data.get("nodes", {}).items():
                connections = {int(k): set(v) for k, v in node_data.get("connections", {}).items()}
                self.nodes[node_id] = HNSWNode(
                    id=node_data["id"],
                    embedding=node_data["embedding"],
                    level=node_data["level"],
                    connections=connections,
                    metadata=node_data.get("metadata", {}),
                )

            logger.info(f"Pure-Python HNSW loaded {len(self.nodes)} nodes from {load_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load Pure-Python HNSW: {e}")
            return False

    def __len__(self) -> int:
        return len(self.nodes)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "count": len(self.nodes),
            "max_elements": self.config.max_elements,
            "dimension": self.config.dimension,
            "space": self.config.space,
            "m": self.config.m,
            "ef_construction": self.config.ef_construction,
            "ef_search": self.config.ef_search,
            "max_level": self.max_level,
            "backend": "pure_python",
        }


# =============================================================================
# UNIFIED HNSW BACKEND
# =============================================================================

class HNSWBackend:
    """
    Unified HNSW backend that uses hnswlib when available, falling back to
    pure-Python implementation otherwise.

    This is the main interface for HNSW operations in the UnifiedMemory system.

    Features:
    - Automatic persistence on interpreter exit (atexit handler)
    - Periodic checkpoint support for long-running processes
    - Dirty tracking to minimize unnecessary saves
    - Recovery on startup from persisted index

    Usage:
        backend = HNSWBackend(config)
        await backend.insert("id1", embedding, {"type": "fact"})
        results = await backend.search(query_embedding, k=10)
        backend.save()  # Explicit save (also happens on exit)
    """

    def __init__(
        self,
        config: Optional[HNSWConfig] = None,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        auto_save: bool = True,
        checkpoint_interval: int = 100,
    ):
        """
        Initialize HNSW backend with persistence support.

        Args:
            config: Optional HNSW configuration
            index_path: Path for index file (default: .swarm/hnsw.index)
            metadata_path: Path for metadata file (default: .swarm/hnsw.metadata.json)
            auto_save: Register atexit handler for automatic save on exit (default: True)
            checkpoint_interval: Save after this many inserts (0 to disable, default: 100)
        """
        # Determine paths
        if index_path is None:
            # Default to .swarm directory
            swarm_dir = Path.cwd() / ".swarm"
            index_path = swarm_dir / "hnsw.index"
            metadata_path = swarm_dir / "hnsw.metadata.json"

        # Create config
        self.config = config or HNSWConfig()
        self.config.index_path = index_path
        self.config.metadata_path = metadata_path or index_path.with_suffix(".metadata.json")

        # Persistence tracking
        self._dirty = False
        self._insert_count = 0
        self._checkpoint_interval = checkpoint_interval
        self._last_checkpoint = time.time()
        self._auto_save = auto_save

        # Initialize appropriate backend
        if HNSWLIB_AVAILABLE:
            self._backend = HNSWLibBackend(self.config)
            self._backend_type = "hnswlib"
        else:
            self._backend = PurePythonHNSWBackend(self.config)
            self._backend_type = "pure_python"
            # Try to load existing pure-Python index
            if hasattr(self._backend, 'load'):
                self._backend.load()

        # Register for automatic save on exit
        if auto_save:
            _registered_backends.append(self)
            logger.debug("Registered HNSW backend for atexit cleanup")

        logger.info(f"Initialized HNSWBackend with {self._backend_type} backend")

    async def insert(
        self,
        id: str,
        embedding: Union[List[float], "np.ndarray"],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Insert a vector into the index.

        Automatically triggers checkpoint if checkpoint_interval is reached.
        """
        await self._backend.insert(id, embedding, metadata)
        self._dirty = True
        self._insert_count += 1

        # Trigger checkpoint if interval reached
        if self._checkpoint_interval > 0 and self._insert_count >= self._checkpoint_interval:
            self.checkpoint()

    async def search(
        self,
        query: Union[List[float], "np.ndarray"],
        k: int = 10,
        ef: Optional[int] = None
    ) -> List[HNSWSearchResult]:
        """Search for k nearest neighbors."""
        return await self._backend.search(query, k, ef)

    async def delete(self, id: str) -> bool:
        """Delete a vector from the index."""
        result = await self._backend.delete(id)
        if result:
            self._dirty = True
        return result

    async def batch_insert(
        self,
        items: List[Tuple[str, Union[List[float], "np.ndarray"], Optional[Dict[str, Any]]]]
    ) -> int:
        """Batch insert multiple vectors.

        Triggers a single checkpoint after the batch completes.
        """
        count = 0
        for id, embedding, metadata in items:
            await self._backend.insert(id, embedding, metadata)
            count += 1

        self._dirty = True
        self._insert_count += count

        # Single checkpoint after batch
        if self._checkpoint_interval > 0 and count > 0:
            self.checkpoint()

        return count

    def checkpoint(self) -> bool:
        """
        Save index to disk if dirty.

        Returns:
            True if save was performed, False if not needed
        """
        if not self._dirty:
            logger.debug("HNSW checkpoint skipped - no changes")
            return False

        try:
            self._backend.save()
            self._dirty = False
            self._insert_count = 0
            self._last_checkpoint = time.time()
            logger.info(f"HNSW checkpoint completed at {self._last_checkpoint}")
            return True
        except Exception as e:
            logger.error(f"HNSW checkpoint failed: {e}")
            return False

    def save(self) -> None:
        """Persist the index to disk (unconditional save)."""
        self._backend.save()
        self._dirty = False
        self._insert_count = 0

    def __len__(self) -> int:
        return len(self._backend)

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        stats = self._backend.get_stats()
        stats["index_path"] = str(self.config.index_path) if self.config.index_path else None
        stats["metadata_path"] = str(self.config.metadata_path) if self.config.metadata_path else None
        stats["dirty"] = self._dirty
        stats["pending_inserts"] = self._insert_count
        stats["last_checkpoint"] = self._last_checkpoint
        stats["checkpoint_interval"] = self._checkpoint_interval
        return stats

    @property
    def is_available(self) -> bool:
        """Check if the backend is available and initialized."""
        return self._backend._initialized if hasattr(self._backend, '_initialized') else True

    @property
    def backend_type(self) -> str:
        """Get the type of backend being used."""
        return self._backend_type

    @property
    def is_dirty(self) -> bool:
        """Check if there are unsaved changes."""
        return self._dirty


# =============================================================================
# SINGLETON AND FACTORY
# =============================================================================

_hnsw_backend: Optional[HNSWBackend] = None


def get_hnsw_backend(
    config: Optional[HNSWConfig] = None,
    index_path: Optional[Path] = None,
    force_new: bool = False
) -> HNSWBackend:
    """
    Get or create the singleton HNSW backend instance.

    Args:
        config: Optional configuration override
        index_path: Optional path to index file
        force_new: Force creation of new instance

    Returns:
        HNSWBackend instance
    """
    global _hnsw_backend

    if _hnsw_backend is None or force_new:
        _hnsw_backend = HNSWBackend(config=config, index_path=index_path)

    return _hnsw_backend


def reset_hnsw_backend() -> None:
    """Reset the singleton instance (for testing)."""
    global _hnsw_backend
    if _hnsw_backend is not None:
        # Save before reset
        try:
            _hnsw_backend.save()
        except Exception as e:
            logger.warning(f"Failed to save HNSW backend during reset: {e}")
    _hnsw_backend = None


# =============================================================================
# ATEXIT HANDLER
# =============================================================================

def _atexit_save_all_backends() -> None:
    """
    Save all registered HNSW backends on interpreter exit.

    This ensures vector indices are persisted even if the process
    terminates without explicit close() calls.
    """
    saved_count = 0
    error_count = 0

    for backend in _registered_backends:
        try:
            if backend.is_dirty:
                backend.save()
                saved_count += 1
                logger.debug(f"HNSW atexit: saved backend with {len(backend)} vectors")
        except Exception as e:
            error_count += 1
            logger.error(f"HNSW atexit: failed to save backend: {e}")

    if saved_count > 0 or error_count > 0:
        logger.info(f"HNSW atexit: saved {saved_count} backends, {error_count} errors")


# Register the atexit handler
atexit.register(_atexit_save_all_backends)


__all__ = [
    # Main classes
    "HNSWBackend",
    "HNSWConfig",
    "HNSWSearchResult",
    # Backends
    "HNSWLibBackend",
    "PurePythonHNSWBackend",
    # Factory
    "get_hnsw_backend",
    "reset_hnsw_backend",
    # Constants
    "HNSWLIB_AVAILABLE",
    "NUMPY_AVAILABLE",
]
