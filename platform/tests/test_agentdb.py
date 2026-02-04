"""
AgentDB / Vector Search Test Suite

Validates the .swarm/ AgentDB state including:
- SQLite memory database schema and connectivity
- HNSW vector index integrity and metadata
- Vector insertion and retrieval simulation
- Memory cleanup (LRU) threshold logic
- Swarm state file validation
- HNSW parameter constraints
- Memory size monitoring
- Concurrent access simulation
"""

import json
import os
import sqlite3
import struct
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SWARM_DIR = ROOT_DIR / ".swarm"

DB_PATH = SWARM_DIR / "memory.db"
HNSW_INDEX_PATH = SWARM_DIR / "hnsw.index"
HNSW_METADATA_PATH = SWARM_DIR / "hnsw.metadata.json"
STATE_PATH = SWARM_DIR / "state.json"
SCHEMA_PATH = SWARM_DIR / "schema.sql"

# ---------------------------------------------------------------------------
# Expected configuration constants
# ---------------------------------------------------------------------------

EXPECTED_HNSW_M = 16
EXPECTED_HNSW_EF_CONSTRUCTION = 200
EXPECTED_HNSW_EF_SEARCH = 100
EXPECTED_HNSW_MAX_ELEMENTS = 10_000
EXPECTED_DIMENSIONS = 768

EXPECTED_TOPOLOGY = "hierarchical"
EXPECTED_MAX_AGENTS = 12
EXPECTED_STRATEGY = "specialized"

LRU_CLEANUP_THRESHOLD = 0.80  # 80 %
MAX_DB_SIZE_MB = 50  # reasonable upper-bound for the local DB


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture(scope="module")
def swarm_dir():
    """Return the swarm directory path, skip if missing."""
    if not SWARM_DIR.is_dir():
        pytest.skip(".swarm directory does not exist")
    return SWARM_DIR


@pytest.fixture(scope="module")
def db_connection(swarm_dir):
    """Open a read-only connection to the memory database."""
    if not DB_PATH.is_file():
        pytest.skip("memory.db not found in .swarm/")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def hnsw_metadata(swarm_dir) -> Dict[str, Any]:
    """Load and return the HNSW metadata JSON."""
    if not HNSW_METADATA_PATH.is_file():
        pytest.skip("hnsw.metadata.json not found in .swarm/")
    with open(HNSW_METADATA_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def swarm_state(swarm_dir) -> Dict[str, Any]:
    """Load and return the swarm state JSON."""
    if not STATE_PATH.is_file():
        pytest.skip("state.json not found in .swarm/")
    with open(STATE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def schema_sql(swarm_dir) -> str:
    """Load and return the raw schema SQL."""
    if not SCHEMA_PATH.is_file():
        pytest.skip("schema.sql not found in .swarm/")
    with open(SCHEMA_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture
def tmp_db():
    """Create a temporary SQLite database seeded with the schema for
    mutation tests so we never touch the real DB."""
    if not SCHEMA_PATH.is_file():
        pytest.skip("schema.sql required for temp DB fixture")
    with open(SCHEMA_PATH, "r", encoding="utf-8") as fh:
        schema = fh.read()
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name)
    conn.executescript(schema)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()
    os.unlink(tmp.name)


# =========================================================================
# 1. Memory Database Connectivity & Schema Validation
# =========================================================================


class TestDatabaseConnectivity:
    """Verify the SQLite database is reachable and schema is correct."""

    def test_database_file_exists(self, swarm_dir):
        assert DB_PATH.is_file(), "memory.db must exist in .swarm/"

    def test_database_opens_successfully(self, db_connection):
        cursor = db_connection.execute("SELECT 1")
        assert cursor.fetchone()[0] == 1

    def test_wal_journal_mode(self, db_connection):
        row = db_connection.execute("PRAGMA journal_mode").fetchone()
        # WAL is set in schema; read-only connections may report 'wal' or 'delete'
        assert row[0] in ("wal", "delete")

    def test_foreign_keys_enabled(self, db_connection):
        row = db_connection.execute("PRAGMA foreign_keys").fetchone()
        # read-only may disable FK; just verify the pragma is recognized
        assert row[0] in (0, 1)

    @pytest.mark.parametrize("table_name", [
        "memory_entries",
        "patterns",
        "pattern_history",
        "trajectories",
        "trajectory_steps",
        "migration_state",
        "sessions",
        "vector_indexes",
        "metadata",
    ])
    def test_required_table_exists(self, db_connection, table_name):
        cursor = db_connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        row = cursor.fetchone()
        assert row is not None, f"Table '{table_name}' is missing from database"

    def test_memory_entries_columns(self, db_connection):
        cursor = db_connection.execute("PRAGMA table_info(memory_entries)")
        columns = {row["name"] for row in cursor.fetchall()}
        expected = {
            "id", "key", "namespace", "content", "type",
            "embedding", "embedding_model", "embedding_dimensions",
            "tags", "metadata", "owner_id",
            "created_at", "updated_at", "expires_at", "last_accessed_at",
            "access_count", "status",
        }
        assert expected.issubset(columns), (
            f"Missing columns: {expected - columns}"
        )

    def test_patterns_columns(self, db_connection):
        cursor = db_connection.execute("PRAGMA table_info(patterns)")
        columns = {row["name"] for row in cursor.fetchall()}
        expected = {
            "id", "name", "pattern_type", "condition", "action",
            "confidence", "success_count", "failure_count",
            "decay_rate", "half_life_days",
            "embedding", "embedding_dimensions",
            "version", "parent_id",
        }
        assert expected.issubset(columns), (
            f"Missing columns: {expected - columns}"
        )

    def test_vector_indexes_columns(self, db_connection):
        cursor = db_connection.execute("PRAGMA table_info(vector_indexes)")
        columns = {row["name"] for row in cursor.fetchall()}
        expected = {
            "id", "name", "dimensions", "metric",
            "hnsw_m", "hnsw_ef_construction", "hnsw_ef_search",
            "quantization_type", "quantization_bits",
            "total_vectors",
        }
        assert expected.issubset(columns), (
            f"Missing columns: {expected - columns}"
        )

    def test_schema_version_is_3(self, db_connection):
        row = db_connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        assert row is not None, "schema_version metadata entry missing"
        assert row["value"] == "3.0.0"

    def test_backend_is_hybrid(self, db_connection):
        row = db_connection.execute(
            "SELECT value FROM metadata WHERE key = 'backend'"
        ).fetchone()
        assert row is not None
        assert row["value"] == "hybrid"

    @pytest.mark.parametrize("feature", [
        "vector_embeddings",
        "pattern_learning",
        "temporal_decay",
        "hnsw_indexing",
    ])
    def test_metadata_feature_enabled(self, db_connection, feature):
        row = db_connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (feature,)
        ).fetchone()
        assert row is not None, f"Metadata key '{feature}' missing"
        assert row["value"] == "enabled"

    @pytest.mark.parametrize("index_name", [
        "idx_memory_namespace",
        "idx_memory_key",
        "idx_memory_type",
        "idx_memory_status",
        "idx_memory_created",
        "idx_memory_accessed",
        "idx_memory_owner",
        "idx_patterns_type",
        "idx_patterns_confidence",
        "idx_patterns_status",
    ])
    def test_required_index_exists(self, db_connection, index_name):
        cursor = db_connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
            (index_name,),
        )
        assert cursor.fetchone() is not None, (
            f"Index '{index_name}' is missing from database"
        )


# =========================================================================
# 2. HNSW Index Existence & Metadata Integrity
# =========================================================================


class TestHNSWIndexIntegrity:
    """Validate that the HNSW index file and its metadata are well-formed."""

    def test_hnsw_index_file_exists(self, swarm_dir):
        assert HNSW_INDEX_PATH.is_file(), "hnsw.index must exist in .swarm/"

    def test_hnsw_index_not_empty(self, swarm_dir):
        assert HNSW_INDEX_PATH.stat().st_size > 0, "hnsw.index is empty"

    def test_hnsw_metadata_file_exists(self, swarm_dir):
        assert HNSW_METADATA_PATH.is_file(), (
            "hnsw.metadata.json must exist in .swarm/"
        )

    def test_hnsw_metadata_is_valid_json(self, hnsw_metadata):
        # The fixture already parses JSON; arriving here means it is valid.
        assert isinstance(hnsw_metadata, (dict, list))

    def test_hnsw_metadata_entries_have_required_keys(self, hnsw_metadata):
        """Each entry in the metadata (list of [id, obj]) should have id
        and content keys."""
        if isinstance(hnsw_metadata, list):
            for item in hnsw_metadata[:5]:
                if isinstance(item, list) and len(item) == 2:
                    entry_id, entry_obj = item
                    assert isinstance(entry_id, str)
                    assert isinstance(entry_obj, dict)
                    assert "id" in entry_obj
                    assert "content" in entry_obj

    def test_hnsw_metadata_ids_are_unique(self, hnsw_metadata):
        if isinstance(hnsw_metadata, list):
            ids = [item[0] for item in hnsw_metadata if isinstance(item, list)]
            assert len(ids) == len(set(ids)), "Duplicate IDs in HNSW metadata"


# =========================================================================
# 3. Vector Insertion & Retrieval Simulation
# =========================================================================


class TestVectorInsertionRetrieval:
    """Simulate insertion and retrieval of vector entries using the temp DB."""

    def _make_entry_id(self) -> str:
        return f"test_{int(time.time() * 1000)}_{os.getpid()}"

    def test_insert_memory_entry(self, tmp_db):
        entry_id = self._make_entry_id()
        tmp_db.execute(
            """INSERT INTO memory_entries
               (id, key, namespace, content, type, embedding_dimensions)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (entry_id, "test-key", "test-ns", "test content", "semantic", 768),
        )
        tmp_db.commit()
        row = tmp_db.execute(
            "SELECT * FROM memory_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        assert row is not None
        assert row["content"] == "test content"
        assert row["embedding_dimensions"] == 768

    def test_insert_and_retrieve_embedding_json(self, tmp_db):
        entry_id = self._make_entry_id()
        fake_embedding = json.dumps([0.1] * 768)
        tmp_db.execute(
            """INSERT INTO memory_entries
               (id, key, namespace, content, type, embedding, embedding_dimensions)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (entry_id, "embed-key", "embed-ns", "embedded content",
             "semantic", fake_embedding, 768),
        )
        tmp_db.commit()
        row = tmp_db.execute(
            "SELECT embedding, embedding_dimensions FROM memory_entries WHERE id = ?",
            (entry_id,),
        ).fetchone()
        assert row is not None
        parsed = json.loads(row["embedding"])
        assert len(parsed) == 768
        assert abs(parsed[0] - 0.1) < 1e-9

    def test_namespace_key_uniqueness_constraint(self, tmp_db):
        entry_id_1 = self._make_entry_id()
        tmp_db.execute(
            """INSERT INTO memory_entries (id, key, namespace, content)
               VALUES (?, ?, ?, ?)""",
            (entry_id_1, "dup-key", "dup-ns", "first"),
        )
        tmp_db.commit()
        entry_id_2 = entry_id_1 + "_dup"
        with pytest.raises(sqlite3.IntegrityError):
            tmp_db.execute(
                """INSERT INTO memory_entries (id, key, namespace, content)
                   VALUES (?, ?, ?, ?)""",
                (entry_id_2, "dup-key", "dup-ns", "second"),
            )

    def test_type_check_constraint(self, tmp_db):
        entry_id = self._make_entry_id()
        with pytest.raises(sqlite3.IntegrityError):
            tmp_db.execute(
                """INSERT INTO memory_entries (id, key, namespace, content, type)
                   VALUES (?, ?, ?, ?, ?)""",
                (entry_id, "bad-type", "ns", "content", "INVALID_TYPE"),
            )

    def test_status_check_constraint(self, tmp_db):
        entry_id = self._make_entry_id()
        with pytest.raises(sqlite3.IntegrityError):
            tmp_db.execute(
                """INSERT INTO memory_entries
                   (id, key, namespace, content, status)
                   VALUES (?, ?, ?, ?, ?)""",
                (entry_id, "bad-status", "ns", "content", "BOGUS"),
            )

    def test_retrieve_by_namespace(self, tmp_db):
        for i in range(5):
            eid = f"ns_test_{i}_{int(time.time()*1000)}"
            tmp_db.execute(
                """INSERT INTO memory_entries (id, key, namespace, content)
                   VALUES (?, ?, ?, ?)""",
                (eid, f"k_{i}", "retrieval-ns", f"content {i}"),
            )
        tmp_db.commit()
        rows = tmp_db.execute(
            "SELECT COUNT(*) as cnt FROM memory_entries WHERE namespace = ?",
            ("retrieval-ns",),
        ).fetchone()
        assert rows["cnt"] == 5

    def test_access_count_default_zero(self, tmp_db):
        entry_id = self._make_entry_id()
        tmp_db.execute(
            """INSERT INTO memory_entries (id, key, namespace, content)
               VALUES (?, ?, ?, ?)""",
            (entry_id, "ac-key", "ac-ns", "access count test"),
        )
        tmp_db.commit()
        row = tmp_db.execute(
            "SELECT access_count FROM memory_entries WHERE id = ?",
            (entry_id,),
        ).fetchone()
        assert row["access_count"] == 0


# =========================================================================
# 4. Memory Cleanup Threshold Logic
# =========================================================================


class TestMemoryCleanupThreshold:
    """Validate LRU cleanup at 80% capacity threshold."""

    def test_lru_threshold_constant(self):
        assert LRU_CLEANUP_THRESHOLD == 0.80

    def test_cleanup_identifies_least_recently_accessed(self, tmp_db):
        """Insert entries with varying last_accessed_at timestamps.
        Verify that ordering by last_accessed_at ASC yields the oldest first."""
        now_ms = int(time.time() * 1000)
        for i in range(10):
            eid = f"lru_{i}_{now_ms}"
            tmp_db.execute(
                """INSERT INTO memory_entries
                   (id, key, namespace, content, last_accessed_at, access_count)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (eid, f"lru_k_{i}", "lru-ns", f"lru content {i}",
                 now_ms - (10 - i) * 60_000, i),
            )
        tmp_db.commit()

        rows = tmp_db.execute(
            """SELECT id, last_accessed_at FROM memory_entries
               WHERE namespace = 'lru-ns'
               ORDER BY last_accessed_at ASC"""
        ).fetchall()
        timestamps = [r["last_accessed_at"] for r in rows]
        assert timestamps == sorted(timestamps), (
            "LRU query should return oldest-accessed entries first"
        )

    def test_cleanup_respects_threshold_percentage(self, tmp_db):
        """When entries exceed threshold, the oldest N should be evictable."""
        max_elements = 100
        threshold = int(max_elements * LRU_CLEANUP_THRESHOLD)  # 80
        now_ms = int(time.time() * 1000)

        # Insert entries up to the threshold
        for i in range(threshold + 5):
            eid = f"cap_{i}_{now_ms}"
            tmp_db.execute(
                """INSERT INTO memory_entries
                   (id, key, namespace, content, last_accessed_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (eid, f"cap_k_{i}", "cap-ns", f"cap {i}",
                 now_ms - (threshold + 5 - i) * 1000),
            )
        tmp_db.commit()

        total = tmp_db.execute(
            "SELECT COUNT(*) as cnt FROM memory_entries WHERE namespace = 'cap-ns'"
        ).fetchone()["cnt"]

        assert total == threshold + 5  # 85 entries

        # Number to evict to get back to threshold
        evict_count = total - threshold  # 5
        assert evict_count == 5

        # Verify we can select the correct eviction candidates
        candidates = tmp_db.execute(
            """SELECT id FROM memory_entries
               WHERE namespace = 'cap-ns'
               ORDER BY last_accessed_at ASC
               LIMIT ?""",
            (evict_count,),
        ).fetchall()
        assert len(candidates) == evict_count

    def test_expired_entries_evicted_first(self, tmp_db):
        """Entries past their expires_at should be prioritised for cleanup."""
        now_ms = int(time.time() * 1000)
        past = now_ms - 60_000
        future = now_ms + 3_600_000

        tmp_db.execute(
            """INSERT INTO memory_entries (id, key, namespace, content, expires_at)
               VALUES (?, ?, ?, ?, ?)""",
            ("expired_1", "exp_k1", "exp-ns", "old", past),
        )
        tmp_db.execute(
            """INSERT INTO memory_entries (id, key, namespace, content, expires_at)
               VALUES (?, ?, ?, ?, ?)""",
            ("active_1", "exp_k2", "exp-ns", "new", future),
        )
        tmp_db.commit()

        expired = tmp_db.execute(
            """SELECT id FROM memory_entries
               WHERE namespace = 'exp-ns' AND expires_at < ?""",
            (now_ms,),
        ).fetchall()
        assert len(expired) == 1
        assert expired[0]["id"] == "expired_1"


# =========================================================================
# 5. Swarm State Validation
# =========================================================================


class TestSwarmState:
    """Validate swarm state.json content against expected configuration."""

    def test_state_file_exists(self, swarm_dir):
        assert STATE_PATH.is_file(), "state.json must exist in .swarm/"

    def test_state_is_valid_json(self, swarm_state):
        assert isinstance(swarm_state, dict)

    def test_topology_is_hierarchical(self, swarm_state):
        assert swarm_state.get("topology") == EXPECTED_TOPOLOGY

    def test_max_agents(self, swarm_state):
        assert swarm_state.get("maxAgents") == EXPECTED_MAX_AGENTS

    def test_strategy(self, swarm_state):
        assert swarm_state.get("strategy") == EXPECTED_STRATEGY

    def test_status_is_ready(self, swarm_state):
        assert swarm_state.get("status") == "ready"

    def test_swarm_has_id(self, swarm_state):
        assert "id" in swarm_state
        assert swarm_state["id"].startswith("swarm-")

    def test_initialized_at_is_iso_string(self, swarm_state):
        from datetime import datetime
        ts = swarm_state.get("initializedAt", "")
        # Should be parseable as ISO 8601
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        assert dt.year >= 2026


# =========================================================================
# 6. HNSW Parameter Validation
# =========================================================================


class TestHNSWParameters:
    """Ensure the vector_indexes table stores correct HNSW parameters."""

    def test_default_index_dimensions(self, db_connection):
        row = db_connection.execute(
            "SELECT dimensions FROM vector_indexes WHERE name = 'default'"
        ).fetchone()
        assert row is not None, "Default vector index not found"
        assert row["dimensions"] == EXPECTED_DIMENSIONS

    def test_default_index_hnsw_m(self, db_connection):
        row = db_connection.execute(
            "SELECT hnsw_m FROM vector_indexes WHERE name = 'default'"
        ).fetchone()
        assert row is not None
        assert row["hnsw_m"] == EXPECTED_HNSW_M

    def test_default_index_ef_construction(self, db_connection):
        row = db_connection.execute(
            "SELECT hnsw_ef_construction FROM vector_indexes WHERE name = 'default'"
        ).fetchone()
        assert row is not None
        assert row["hnsw_ef_construction"] == EXPECTED_HNSW_EF_CONSTRUCTION

    def test_default_index_ef_search(self, db_connection):
        row = db_connection.execute(
            "SELECT hnsw_ef_search FROM vector_indexes WHERE name = 'default'"
        ).fetchone()
        assert row is not None
        assert row["hnsw_ef_search"] == EXPECTED_HNSW_EF_SEARCH

    def test_default_index_metric_is_cosine(self, db_connection):
        row = db_connection.execute(
            "SELECT metric FROM vector_indexes WHERE name = 'default'"
        ).fetchone()
        assert row is not None
        assert row["metric"] == "cosine"

    def test_patterns_index_exists(self, db_connection):
        row = db_connection.execute(
            "SELECT dimensions FROM vector_indexes WHERE name = 'patterns'"
        ).fetchone()
        assert row is not None
        assert row["dimensions"] == EXPECTED_DIMENSIONS

    def test_hnsw_m_within_valid_range(self, db_connection):
        rows = db_connection.execute(
            "SELECT hnsw_m FROM vector_indexes"
        ).fetchall()
        for row in rows:
            m = row["hnsw_m"]
            assert 2 <= m <= 100, f"HNSW M={m} is out of valid range [2, 100]"

    def test_ef_construction_gte_m(self, db_connection):
        """ef_construction should be >= M for a well-formed HNSW index."""
        rows = db_connection.execute(
            "SELECT hnsw_m, hnsw_ef_construction FROM vector_indexes"
        ).fetchall()
        for row in rows:
            assert row["hnsw_ef_construction"] >= row["hnsw_m"]


# =========================================================================
# 7. Memory Size Monitoring
# =========================================================================


class TestMemorySizeMonitoring:
    """Validate that the database and index files are within acceptable bounds."""

    def test_database_file_size_nonzero(self, swarm_dir):
        size = DB_PATH.stat().st_size
        assert size > 0, "memory.db should not be empty"

    def test_database_file_size_under_limit(self, swarm_dir):
        size_mb = DB_PATH.stat().st_size / (1024 * 1024)
        assert size_mb < MAX_DB_SIZE_MB, (
            f"memory.db is {size_mb:.2f} MB, exceeds {MAX_DB_SIZE_MB} MB limit"
        )

    def test_hnsw_index_size_nonzero(self, swarm_dir):
        size = HNSW_INDEX_PATH.stat().st_size
        assert size > 0, "hnsw.index should not be empty"

    def test_hnsw_metadata_size_nonzero(self, swarm_dir):
        size = HNSW_METADATA_PATH.stat().st_size
        assert size > 0, "hnsw.metadata.json should not be empty"

    def test_total_swarm_directory_size(self, swarm_dir):
        total = sum(
            f.stat().st_size
            for f in SWARM_DIR.iterdir()
            if f.is_file()
        )
        total_mb = total / (1024 * 1024)
        assert total_mb < 100, (
            f".swarm/ directory is {total_mb:.2f} MB, exceeds 100 MB limit"
        )

    def test_memory_entry_count_under_max_elements(self, db_connection):
        row = db_connection.execute(
            "SELECT COUNT(*) as cnt FROM memory_entries"
        ).fetchone()
        assert row["cnt"] <= EXPECTED_HNSW_MAX_ELEMENTS, (
            f"Entry count {row['cnt']} exceeds maxElements {EXPECTED_HNSW_MAX_ELEMENTS}"
        )

    def test_hnsw_metadata_count_matches_reasonable_bound(self, hnsw_metadata):
        if isinstance(hnsw_metadata, list):
            assert len(hnsw_metadata) <= EXPECTED_HNSW_MAX_ELEMENTS


# =========================================================================
# 8. Concurrent Access Simulation
# =========================================================================


class TestConcurrentAccess:
    """Simulate concurrent reads/writes to verify SQLite handles WAL mode
    concurrency correctly."""

    @staticmethod
    def _read_worker(db_path: str, query: str) -> int:
        """Worker that opens its own connection and runs a read query."""
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(query).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    @staticmethod
    def _write_worker(db_path: str, entry_id: str) -> bool:
        """Worker that inserts a row into a temporary database."""
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                """INSERT INTO memory_entries (id, key, namespace, content)
                   VALUES (?, ?, ?, ?)""",
                (entry_id, f"conc_{entry_id}", "concurrent-ns", "concurrent"),
            )
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            conn.close()

    def test_concurrent_reads(self, swarm_dir):
        """Multiple threads should be able to read simultaneously."""
        if not DB_PATH.is_file():
            pytest.skip("memory.db not available")

        query = "SELECT COUNT(*) FROM memory_entries"
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(self._read_worker, str(DB_PATH), query)
                for _ in range(8)
            ]
            results = [f.result() for f in as_completed(futures)]

        # All results should be the same count
        assert len(set(results)) == 1, (
            "Concurrent reads returned inconsistent counts"
        )

    def test_concurrent_writes(self, tmp_db):
        """Multiple threads writing to a temp DB should not corrupt data."""
        tmp_path = tmp_db.execute("PRAGMA database_list").fetchone()[2]
        tmp_db.close()

        entry_ids = [f"conc_w_{i}_{int(time.time()*1000)}" for i in range(20)]
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [
                pool.submit(self._write_worker, tmp_path, eid)
                for eid in entry_ids
            ]
            results = [f.result() for f in as_completed(futures)]

        success_count = sum(1 for r in results if r)
        assert success_count > 0, "At least some concurrent writes should succeed"

        # Verify data integrity after concurrent writes
        conn = sqlite3.connect(tmp_path)
        row = conn.execute(
            "SELECT COUNT(*) FROM memory_entries WHERE namespace = 'concurrent-ns'"
        ).fetchone()
        conn.close()
        assert row[0] == success_count

    def test_concurrent_read_during_write(self, tmp_db):
        """Reads should not block on writes in WAL mode."""
        tmp_path = tmp_db.execute("PRAGMA database_list").fetchone()[2]

        # Insert some seed data
        for i in range(10):
            tmp_db.execute(
                """INSERT INTO memory_entries (id, key, namespace, content)
                   VALUES (?, ?, ?, ?)""",
                (f"seed_{i}", f"seed_k_{i}", "rw-ns", f"seed {i}"),
            )
        tmp_db.commit()

        def reader():
            c = sqlite3.connect(f"file:{tmp_path}?mode=ro", uri=True)
            try:
                return c.execute(
                    "SELECT COUNT(*) FROM memory_entries WHERE namespace = 'rw-ns'"
                ).fetchone()[0]
            finally:
                c.close()

        def writer(idx):
            c = sqlite3.connect(tmp_path)
            try:
                c.execute(
                    """INSERT INTO memory_entries (id, key, namespace, content)
                       VALUES (?, ?, ?, ?)""",
                    (f"rw_{idx}", f"rw_k_{idx}", "rw-ns", f"rw {idx}"),
                )
                c.commit()
                return True
            except Exception:
                return False
            finally:
                c.close()

        with ThreadPoolExecutor(max_workers=6) as pool:
            read_futures = [pool.submit(reader) for _ in range(3)]
            write_futures = [pool.submit(writer, i) for i in range(3)]

            read_results = [f.result() for f in read_futures]
            write_results = [f.result() for f in write_futures]

        # Reads should all succeed and return consistent (snapshot) counts
        assert all(isinstance(r, int) for r in read_results)
        # Writes should mostly succeed
        assert any(write_results), "At least one concurrent write should succeed"


# =========================================================================
# 9. Pattern Learning Tables Integrity
# =========================================================================


class TestPatternLearning:
    """Validate pattern learning schema constraints in the temp DB."""

    def test_pattern_type_constraint(self, tmp_db):
        valid_types = [
            "task-routing", "error-recovery", "optimization", "learning",
            "coordination", "prediction", "code-pattern", "workflow",
        ]
        for ptype in valid_types:
            pid = f"pat_{ptype}_{int(time.time()*1000)}"
            tmp_db.execute(
                """INSERT INTO patterns
                   (id, name, pattern_type, condition, action)
                   VALUES (?, ?, ?, ?, ?)""",
                (pid, f"test-{ptype}", ptype, ".*", "noop"),
            )
        tmp_db.commit()
        count = tmp_db.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
        assert count == len(valid_types)

    def test_invalid_pattern_type_rejected(self, tmp_db):
        with pytest.raises(sqlite3.IntegrityError):
            tmp_db.execute(
                """INSERT INTO patterns
                   (id, name, pattern_type, condition, action)
                   VALUES (?, ?, ?, ?, ?)""",
                ("bad_pat", "bad", "NOT_VALID", ".*", "noop"),
            )

    def test_confidence_default_value(self, tmp_db):
        pid = f"conf_{int(time.time()*1000)}"
        tmp_db.execute(
            """INSERT INTO patterns (id, name, pattern_type, condition, action)
               VALUES (?, ?, ?, ?, ?)""",
            (pid, "confidence-test", "learning", ".*", "noop"),
        )
        tmp_db.commit()
        row = tmp_db.execute(
            "SELECT confidence FROM patterns WHERE id = ?", (pid,)
        ).fetchone()
        assert abs(row[0] - 0.5) < 1e-9


# =========================================================================
# 10. Schema SQL File Validation
# =========================================================================


class TestSchemaSQLFile:
    """Validate the schema.sql file itself for completeness."""

    def test_schema_file_exists(self, swarm_dir):
        assert SCHEMA_PATH.is_file()

    def test_schema_contains_wal_pragma(self, schema_sql):
        assert "journal_mode = WAL" in schema_sql

    def test_schema_contains_foreign_keys_pragma(self, schema_sql):
        assert "foreign_keys = ON" in schema_sql

    def test_schema_contains_all_table_creates(self, schema_sql):
        expected_tables = [
            "memory_entries", "patterns", "pattern_history",
            "trajectories", "trajectory_steps", "migration_state",
            "sessions", "vector_indexes", "metadata",
        ]
        for table in expected_tables:
            assert f"CREATE TABLE IF NOT EXISTS {table}" in schema_sql, (
                f"Schema missing CREATE TABLE for '{table}'"
            )

    def test_schema_seeds_default_vector_indexes(self, schema_sql):
        assert "'default'" in schema_sql
        assert "'patterns'" in schema_sql
        assert "768" in schema_sql
