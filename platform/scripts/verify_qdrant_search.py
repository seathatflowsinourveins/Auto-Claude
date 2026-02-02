#!/usr/bin/env python3
"""
Quick verification that Qdrant semantic search works correctly.
This script inserts test vectors and verifies search returns correct results.

Usage:
    python platform/scripts/verify_qdrant_search.py
"""

import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path("Z:/insider/AUTO CLAUDE/unleash")
sys.path.insert(0, str(PROJECT_ROOT))

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "unleash_code"
DIMENSION = 1024


def main():
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, PointIdsList

    print("=" * 60)
    print("QDRANT SEMANTIC SEARCH VERIFICATION")
    print("=" * 60)

    client = QdrantClient(QDRANT_URL)

    # 1. Verify collection exists
    print("\n[1/4] Checking collection...")
    info = client.get_collection(QDRANT_COLLECTION)
    vectors_config = info.config.params.vectors
    if isinstance(vectors_config, dict):
        actual_dim = vectors_config.get("size", 0)
    else:
        actual_dim = getattr(vectors_config, "size", 0)

    if actual_dim != DIMENSION:
        print(f"  FAIL: Dimension mismatch (expected {DIMENSION}, got {actual_dim})")
        return 1
    print(f"  OK: Collection has correct dimensions ({actual_dim})")

    # 2. Insert test vectors
    print("\n[2/4] Inserting test vectors...")

    # Simulate embeddings with distinct patterns:
    # - auth_code: high values in first 512 dims
    # - db_code: high values in last 512 dims
    # - api_code: high values in middle dims

    auth_vector = [0.9] * 512 + [0.1] * 512
    db_vector = [0.1] * 512 + [0.9] * 512
    api_vector = [0.1] * 256 + [0.9] * 512 + [0.1] * 256

    # Use UUIDs for point IDs
    import uuid
    auth_id = str(uuid.uuid4())
    db_id = str(uuid.uuid4())
    api_id = str(uuid.uuid4())

    test_points = [
        PointStruct(
            id=auth_id,
            vector=auth_vector,
            payload={
                "file_path": "test/auth.py",
                "content": "def authenticate_user(username, password): return check_credentials(username, password)",
                "language": "python",
            }
        ),
        PointStruct(
            id=db_id,
            vector=db_vector,
            payload={
                "file_path": "test/database.py",
                "content": "def query_database(sql): return connection.execute(sql).fetchall()",
                "language": "python",
            }
        ),
        PointStruct(
            id=api_id,
            vector=api_vector,
            payload={
                "file_path": "test/api.py",
                "content": "def fetch_api_data(url): return requests.get(url).json()",
                "language": "python",
            }
        ),
    ]

    test_ids = [auth_id, db_id, api_id]  # Save for cleanup

    client.upsert(collection_name=QDRANT_COLLECTION, points=test_points)
    print(f"  OK: Inserted {len(test_points)} test vectors")

    # 3. Test search functionality
    print("\n[3/4] Testing semantic search...")

    # Query similar to auth_vector should return auth_code first
    query_auth = [0.85] * 512 + [0.15] * 512
    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_auth,
        limit=3,
    )

    if not results.points:
        print("  FAIL: Search returned no results")
        return 1

    top_result = results.points[0]
    top_file_path = top_result.payload.get("file_path", "") if top_result.payload else ""
    if "auth" not in top_file_path:
        print(f"  FAIL: Expected auth.py as top result, got: {top_file_path}")
        return 1

    print(f"  OK: Top result is '{top_file_path}' (score: {top_result.score:.3f})")

    # Query similar to db_vector should return db_code first
    query_db = [0.15] * 512 + [0.85] * 512
    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_db,
        limit=3,
    )

    top_result = results.points[0]
    top_file_path = top_result.payload.get("file_path", "") if top_result.payload else ""
    if "database" not in top_file_path:
        print(f"  FAIL: Expected database.py as top result, got: {top_file_path}")
        return 1

    print(f"  OK: Top result is '{top_file_path}' (score: {top_result.score:.3f})")

    # 4. Cleanup
    print("\n[4/4] Cleaning up test vectors...")
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=PointIdsList(points=test_ids),  # type: ignore[arg-type]
    )
    print("  OK: Test vectors removed")

    print("\n" + "=" * 60)
    print("VERIFICATION PASSED")
    print("=" * 60)
    print("\nQdrant semantic search is working correctly!")
    print("To populate with real embeddings, set VOYAGE_API_KEY and run:")
    print("  python platform/core/embedding_pipeline.py --core-only")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
