#!/usr/bin/env python3
"""
UNLEASH Code Intelligence Test Suite
=====================================
Part of the Unified Code Intelligence Architecture (5-Layer, 2026)

IMPORTANT: These tests verify FUNCTIONALITY, not just existence.
A test that can pass with a broken system is USELESS.

Tests:
1. Qdrant: Collection exists AND has correct dimensions AND can store/retrieve
2. Embedding: API works AND produces correct-dimensional vectors
3. Search: Query returns relevant results AND ranks correctly
4. Pipeline: End-to-end embedding + search works on real code

Usage:
    pytest platform/tests/test_code_intelligence.py -v
    pytest platform/tests/test_code_intelligence.py -v -k "test_search"

Requirements:
    pip install pytest pytest-asyncio voyageai qdrant-client httpx
    export VOYAGE_API_KEY=your_key
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path("Z:/insider/AUTO CLAUDE/unleash")
sys.path.insert(0, str(PROJECT_ROOT))

# Constants
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "unleash_code"
EXPECTED_DIMENSION = 1024  # Voyage-code-3 default


class TestQdrantFunctionality:
    """
    Test Qdrant vector database FUNCTIONALITY.

    GOOD: Verifies storage, retrieval, and search actually work
    BAD:  Just checking if port is open or collection exists
    """

    @pytest.fixture
    def qdrant_client(self):
        from qdrant_client import QdrantClient
        return QdrantClient(QDRANT_URL)

    def test_qdrant_is_running(self, qdrant_client):
        """Verify Qdrant responds to health check."""
        # This is a basic existence test, but necessary as a prerequisite
        import httpx
        response = httpx.get(f"{QDRANT_URL}/healthz", timeout=5)
        assert response.status_code == 200, "Qdrant is not running"

    def test_collection_has_correct_dimensions(self, qdrant_client):
        """Verify collection exists with correct vector dimensions."""
        info = qdrant_client.get_collection(QDRANT_COLLECTION)

        # Access vector size (handle both dict and object forms)
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, dict):
            actual_dim = vectors_config.get("size", 0)
        else:
            actual_dim = getattr(vectors_config, "size", 0)

        assert actual_dim == EXPECTED_DIMENSION, (
            f"Collection has {actual_dim} dimensions, expected {EXPECTED_DIMENSION}. "
            f"This means embeddings will fail! Recreate with correct dimensions."
        )

    def test_can_insert_and_retrieve_vector(self, qdrant_client):
        """
        FUNCTIONAL TEST: Actually insert a vector and retrieve it.

        This proves Qdrant can store and return data, not just that it's running.
        """
        from qdrant_client.models import PointStruct

        test_id = "test_functional_vector_001"
        test_vector = [0.1] * EXPECTED_DIMENSION
        test_payload = {"file_path": "test.py", "content": "def test(): pass"}

        # Insert
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[PointStruct(
                id=test_id,
                vector=test_vector,
                payload=test_payload,
            )],
        )

        # Retrieve
        results = qdrant_client.retrieve(
            collection_name=QDRANT_COLLECTION,
            ids=[test_id],
            with_vectors=True,
        )

        assert len(results) == 1, "Failed to retrieve inserted vector"
        assert results[0].payload["file_path"] == "test.py", "Payload mismatch"
        assert len(results[0].vector) == EXPECTED_DIMENSION, "Vector dimension mismatch"

        # Cleanup
        qdrant_client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=[test_id],
        )

    def test_vector_search_returns_similar_results(self, qdrant_client):
        """
        FUNCTIONAL TEST: Search returns vectors similar to query.

        This proves semantic search actually works, not just that the endpoint exists.
        """
        from qdrant_client.models import PointStruct

        # Insert test vectors - one similar, one different
        similar_vector = [0.9] * EXPECTED_DIMENSION
        different_vector = [0.1] * EXPECTED_DIMENSION
        query_vector = [0.85] * EXPECTED_DIMENSION  # Similar to similar_vector

        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointStruct(id="similar_001", vector=similar_vector, payload={"type": "similar"}),
                PointStruct(id="different_001", vector=different_vector, payload={"type": "different"}),
            ],
        )

        # Search
        results = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=2,
        )

        assert len(results.points) == 2, "Should return 2 results"
        # The similar vector should rank higher
        assert results.points[0].payload["type"] == "similar", (
            "Vector search is not returning similar vectors first! "
            "This means semantic search is broken."
        )

        # Cleanup
        qdrant_client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=["similar_001", "different_001"],
        )


@pytest.mark.skipif(
    not os.getenv("VOYAGE_API_KEY"),
    reason="VOYAGE_API_KEY not set"
)
class TestEmbeddingFunctionality:
    """
    Test Voyage embedding API FUNCTIONALITY.

    GOOD: Verifies API returns correct-dimensional embeddings
    BAD:  Just checking if the library is installed
    """

    @pytest.fixture
    def voyage_client(self):
        import voyageai  # type: ignore[import-untyped]
        return voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))  # type: ignore[attr-defined]

    def test_embedding_produces_correct_dimensions(self, voyage_client):
        """Verify Voyage API returns vectors with expected dimensions."""
        result = voyage_client.embed(
            ["def hello(): print('world')"],
            model="voyage-code-3",
            input_type="document",
        )

        assert len(result.embeddings) == 1, "Should return 1 embedding"
        assert len(result.embeddings[0]) == EXPECTED_DIMENSION, (
            f"Embedding has {len(result.embeddings[0])} dimensions, expected {EXPECTED_DIMENSION}. "
            f"This means storage in Qdrant will fail!"
        )

    def test_similar_code_has_similar_embeddings(self, voyage_client):
        """
        FUNCTIONAL TEST: Semantically similar code produces similar embeddings.

        This proves the embedding model understands code semantics.
        """
        import numpy as np

        code_a = "def add(x, y): return x + y"
        code_b = "def sum_numbers(a, b): return a + b"  # Similar semantics
        code_c = "def multiply(x, y): return x * y"     # Different semantics

        result = voyage_client.embed(
            [code_a, code_b, code_c],
            model="voyage-code-3",
            input_type="document",
        )

        emb_a = np.array(result.embeddings[0])
        emb_b = np.array(result.embeddings[1])
        emb_c = np.array(result.embeddings[2])

        # Cosine similarity
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_ab = cosine_sim(emb_a, emb_b)  # Should be high (similar code)
        sim_ac = cosine_sim(emb_a, emb_c)  # Should be lower (different operation)

        assert sim_ab > sim_ac, (
            f"Similar code (add/sum) should have higher similarity ({sim_ab:.3f}) "
            f"than different code (add/multiply) ({sim_ac:.3f}). "
            f"This suggests embeddings don't capture code semantics!"
        )

    def test_query_vs_document_embeddings_differ(self, voyage_client):
        """Verify query and document embeddings are different (asymmetric search)."""
        code = "def authenticate(user, password): pass"

        doc_result = voyage_client.embed([code], model="voyage-code-3", input_type="document")
        query_result = voyage_client.embed([code], model="voyage-code-3", input_type="query")

        # They should be different vectors (asymmetric embedding)
        assert doc_result.embeddings[0] != query_result.embeddings[0], (
            "Document and query embeddings should differ for asymmetric search"
        )


@pytest.mark.skipif(
    not os.getenv("VOYAGE_API_KEY"),
    reason="VOYAGE_API_KEY not set"
)
class TestEndToEndSearch:
    """
    Test end-to-end semantic search FUNCTIONALITY.

    This is the ultimate test: can we embed code and find it via natural language?
    """

    def test_search_finds_relevant_code(self):
        """
        FUNCTIONAL TEST: End-to-end semantic search finds relevant code.

        This is the most important test. If this fails, the system is BROKEN.
        """
        # Import search_code directly using importlib to avoid platform module conflict
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "embedding_pipeline",
            PROJECT_ROOT / "platform" / "core" / "embedding_pipeline.py"
        )
        embedding_module = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(embedding_module)  # type: ignore
        search_code = embedding_module.search_code
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct
        import voyageai  # type: ignore[import-untyped]

        # Setup: Embed some test code
        voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))  # type: ignore[attr-defined]
        qdrant = QdrantClient(QDRANT_URL)

        test_codes = [
            ("auth_code", "def authenticate_user(username, password): return check_credentials(username, password)"),
            ("db_code", "def query_database(sql): return connection.execute(sql).fetchall()"),
            ("api_code", "def fetch_api_data(url): return requests.get(url).json()"),
        ]

        # Embed and store test code
        for code_id, code in test_codes:
            result = voyage.embed([code], model="voyage-code-3", input_type="document")
            qdrant.upsert(
                collection_name=QDRANT_COLLECTION,
                points=[PointStruct(
                    id=f"test_{code_id}",
                    vector=result.embeddings[0],
                    payload={"file_path": f"{code_id}.py", "content": code},
                )],
            )

        try:
            # Test: Search for authentication-related code
            results = search_code("how to check user login credentials", limit=3)

            assert len(results) > 0, "Search returned no results"

            # The authentication code should be the top result
            top_result = results[0]
            assert "auth" in top_result["file_path"].lower() or "authenticate" in top_result["content"].lower(), (
                f"Search for 'login credentials' should find authentication code, "
                f"but found: {top_result['file_path']}"
            )

        finally:
            # Cleanup
            qdrant.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=[f"test_{code_id}" for code_id, _ in test_codes],
            )


class TestToolAvailability:
    """
    Test that required tools are installed and respond.

    NOTE: These are existence checks, which are necessary but NOT sufficient.
    The functional tests above are what actually matter.
    """

    def test_pyright_available(self):
        """Check pyright is installed and responds."""
        import subprocess
        result = subprocess.run(
            ["pyright", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, (
            f"pyright not working: {result.stderr}. "
            f"Install with: pip install pyright"
        )

    def test_narsil_available(self):
        """Check narsil-mcp is installed."""
        import shutil
        path = shutil.which("narsil-mcp")
        if path is None:
            # Check cargo bin
            cargo_path = Path.home() / ".cargo" / "bin" / "narsil-mcp.exe"
            if not cargo_path.exists():
                pytest.skip("narsil-mcp not installed")

    def test_mcp_language_server_available(self):
        """Check mcp-language-server is installed."""
        import shutil
        path = shutil.which("mcp-language-server")
        if path is None:
            # Check go bin
            go_path = Path.home() / "go" / "bin" / "mcp-language-server.exe"
            assert go_path.exists(), (
                "mcp-language-server not found. "
                "Install: go install github.com/nickcdryan/mcp-language-server@latest"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
