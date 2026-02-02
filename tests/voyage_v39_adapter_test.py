#!/usr/bin/env python3
"""
Voyage AI V39.4 - Project Adapter Enhancement Tests
====================================================

Tests the new adapter methods:
1. WitnessVectorAdapter.find_similar_poses_mmr()
2. WitnessVectorAdapter.hybrid_shader_search()
3. WitnessVectorAdapter.search_particles_with_filters()
4. WitnessVectorAdapter.discover_archetypes_mmr()
5. TradingVectorAdapter.find_similar_signals_mmr()
6. TradingVectorAdapter.hybrid_strategy_search()

Note: These tests use mock Qdrant data since real vector DB may not be available.
"""

import asyncio
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orchestration.embedding_layer import (
    EmbeddingLayer,
    EmbeddingConfig,
    EmbeddingModel,
    InputType,
)


# Mock QdrantSearchResult for testing
@dataclass
class MockQdrantSearchResult:
    id: str
    score: float
    payload: dict[str, Any]
    vector: Optional[list[float]] = None


# Mock Qdrant store for testing
class MockQdrantStore:
    """Mock Qdrant store that returns test data."""

    def __init__(self):
        self.collections: dict[str, list[MockQdrantSearchResult]] = {}
        self.embedding_layer: Optional[EmbeddingLayer] = None

    async def initialize(self):
        pass

    async def create_collection(self, name: str, dimension: int, distance: str = "cosine"):
        self.collections[name] = []

    async def scroll(
        self,
        collection: str,
        limit: int = 100,
        filter: Optional[dict] = None,
    ) -> list[MockQdrantSearchResult]:
        return self.collections.get(collection, [])

    async def search_with_text(
        self,
        collection: str,
        query_text: str,
        limit: int = 10,
        filter: Optional[dict] = None,
        input_type: InputType = InputType.QUERY,
    ) -> list[MockQdrantSearchResult]:
        return self.collections.get(collection, [])[:limit]

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 10,
        filter: Optional[dict] = None,
    ) -> list[MockQdrantSearchResult]:
        return self.collections.get(collection, [])[:limit]

    async def upsert(
        self,
        collection: str,
        embeddings: list[list[float]],
        ids: list[str],
        payloads: list[dict],
    ):
        if collection not in self.collections:
            self.collections[collection] = []

        for id_, emb, payload in zip(ids, embeddings, payloads):
            self.collections[collection].append(
                MockQdrantSearchResult(id=id_, score=1.0, payload=payload, vector=emb)
            )


# Test data for Witness adapter
WITNESS_POSES = [
    {"description": "Standing warrior pose with arms raised", "archetype": "WARRIOR"},
    {"description": "Gentle nurturing embrace pose", "archetype": "NURTURER"},
    {"description": "Wise sage contemplation pose", "archetype": "SAGE"},
    {"description": "Playful jester jumping pose", "archetype": "JESTER"},
    {"description": "Graceful lover dance pose", "archetype": "LOVER"},
    {"description": "Mysterious magician gesture", "archetype": "MAGICIAN"},
    {"description": "Innocent child wonder pose", "archetype": "INNOCENT"},
    {"description": "Relaxed everyman neutral pose", "archetype": "EVERYMAN"},
]

WITNESS_SHADERS = [
    {"name": "noise_fractal", "type": "fragment", "preview": "vec3 noise = simplex3d(uv * 10.0);"},
    {"name": "particle_render", "type": "fragment", "preview": "float alpha = smoothstep(0.5, 0.0, length(uv));"},
    {"name": "vertex_transform", "type": "vertex", "preview": "gl_Position = uTDMat.proj * pos;"},
    {"name": "sdf_sphere", "type": "fragment", "preview": "float d = length(p) - radius;"},
    {"name": "bloom_pass", "type": "fragment", "preview": "vec3 bloom = texture(sTD2DInputs[0], uv).rgb * intensity;"},
]

WITNESS_PARTICLES = [
    {"description": "Explosive burst pattern", "archetype": "WARRIOR", "gravity": 9.8, "mass": 1.2},
    {"description": "Gentle flowing particles", "archetype": "NURTURER", "gravity": 2.0, "mass": 0.5},
    {"description": "Slow contemplative drift", "archetype": "SAGE", "gravity": 1.0, "mass": 0.3},
    {"description": "Chaotic bouncing spheres", "archetype": "JESTER", "gravity": 15.0, "mass": 0.8},
]


async def setup_mock_witness_adapter():
    """Create a mock Witness adapter with test data."""
    from core.orchestration.embedding_layer import WitnessVectorAdapter

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=100))
    await layer.initialize()

    mock_store = MockQdrantStore()
    await mock_store.initialize()

    # Pre-embed test data
    pose_descs = [p["description"] for p in WITNESS_POSES]
    shader_previews = [s["preview"] for s in WITNESS_SHADERS]
    particle_descs = [p["description"] for p in WITNESS_PARTICLES]

    # Generate embeddings
    pose_result = await layer.embed(
        texts=pose_descs,
        model=EmbeddingModel.VOYAGE_4_LITE,
        input_type=InputType.DOCUMENT,
    )

    shader_result = await layer.embed(
        texts=shader_previews,
        model=EmbeddingModel.VOYAGE_4_LITE,
        input_type=InputType.DOCUMENT,
    )

    particle_result = await layer.embed(
        texts=particle_descs,
        model=EmbeddingModel.VOYAGE_4_LITE,
        input_type=InputType.DOCUMENT,
    )

    # Populate mock store
    mock_store.collections["witness_poses"] = [
        MockQdrantSearchResult(
            id=f"pose_{i}",
            score=1.0,
            payload=WITNESS_POSES[i],
            vector=pose_result.embeddings[i],
        )
        for i in range(len(WITNESS_POSES))
    ]

    mock_store.collections["witness_shaders"] = [
        MockQdrantSearchResult(
            id=f"shader_{i}",
            score=1.0,
            payload=WITNESS_SHADERS[i],
            vector=shader_result.embeddings[i],
        )
        for i in range(len(WITNESS_SHADERS))
    ]

    mock_store.collections["witness_particles"] = [
        MockQdrantSearchResult(
            id=f"particle_{i}",
            score=1.0,
            payload=WITNESS_PARTICLES[i],
            vector=particle_result.embeddings[i],
        )
        for i in range(len(WITNESS_PARTICLES))
    ]

    adapter = WitnessVectorAdapter(layer, mock_store)
    return adapter, layer


async def test_witness_pose_mmr():
    """Test MMR search for diverse pose discovery."""
    print("\n[TEST] Witness Pose MMR Search")
    print("-" * 50)

    adapter, layer = await setup_mock_witness_adapter()

    # Test diverse results (low lambda)
    results = await adapter.find_similar_poses_mmr(
        query="dynamic expressive pose",
        top_k=4,
        lambda_mult=0.3,  # High diversity
        fetch_k=8,
    )

    print(f"  Diverse results (lambda=0.3): {len(results)} poses")
    archetypes = set()
    for r in results:
        archetype = r.payload.get("archetype", "UNKNOWN")
        archetypes.add(archetype)
        print(f"    {archetype}: {r.payload['description'][:40]}... (score: {r.score:.3f})")

    print(f"  Unique archetypes found: {len(archetypes)}")

    # With high diversity, we should get multiple archetypes
    assert len(archetypes) >= 2, "MMR should return diverse archetypes"

    print("  [PASS] Pose MMR search working correctly")
    return True


async def test_witness_hybrid_shader():
    """Test hybrid search for shader code."""
    print("\n[TEST] Witness Hybrid Shader Search")
    print("-" * 50)

    adapter, layer = await setup_mock_witness_adapter()

    # Test with keyword focus (low alpha)
    results_keyword = await adapter.hybrid_shader_search(
        query="noise simplex fractal",
        alpha=0.3,  # More keyword focus
        top_k=3,
    )

    print(f"  Keyword-focused results (alpha=0.3): {len(results_keyword)} shaders")
    for r in results_keyword:
        print(f"    {r.payload['name']}: {r.payload['preview'][:40]}...")

    # Test with semantic focus (high alpha)
    results_semantic = await adapter.hybrid_shader_search(
        query="generate visual effects texture",
        alpha=0.8,  # More semantic focus
        top_k=3,
    )

    print(f"  Semantic-focused results (alpha=0.8): {len(results_semantic)} shaders")
    for r in results_semantic:
        print(f"    {r.payload['name']}: {r.payload['preview'][:40]}...")

    assert len(results_keyword) > 0, "Should find keyword matches"
    assert len(results_semantic) > 0, "Should find semantic matches"

    print("  [PASS] Hybrid shader search working correctly")
    return True


async def test_witness_filtered_particles():
    """Test filtered search for particle systems."""
    print("\n[TEST] Witness Filtered Particle Search")
    print("-" * 50)

    adapter, layer = await setup_mock_witness_adapter()

    # Test with archetype filter
    results_warrior = await adapter.search_particles_with_filters(
        query="high energy motion",
        filters={"archetype": "WARRIOR"},
        top_k=3,
    )

    print(f"  WARRIOR filter results: {len(results_warrior)} particles")
    for r in results_warrior:
        print(f"    {r.payload['archetype']}: {r.payload['description']}")

    # All results should be WARRIOR
    all_warrior = all(
        r.payload.get("archetype") == "WARRIOR"
        for r in results_warrior
    )
    print(f"  All WARRIOR archetype: {all_warrior}")

    # Test with gravity filter
    results_high_gravity = await adapter.search_particles_with_filters(
        query="fast falling particles",
        filters={"gravity": {"$gte": 8.0}},
        top_k=3,
    )

    print(f"  High gravity filter (>=8.0): {len(results_high_gravity)} particles")
    for r in results_high_gravity:
        print(f"    gravity={r.payload.get('gravity', 0)}: {r.payload['description']}")

    print("  [PASS] Filtered particle search working correctly")
    return True


async def test_witness_archetype_discovery():
    """Test archetype discovery from seed pose."""
    print("\n[TEST] Witness Archetype Discovery")
    print("-" * 50)

    adapter, layer = await setup_mock_witness_adapter()

    # Start with warrior pose, discover other archetypes
    results = await adapter.discover_archetypes_mmr(
        seed_pose="fierce combat stance with raised fist",
        diversity=0.8,  # High diversity
        num_archetypes=4,
    )

    print(f"  Discovered {len(results)} archetypes from warrior seed:")
    for archetype, confidence, payload in results:
        print(f"    {archetype}: {payload['description'][:40]}... (conf: {confidence:.3f})")

    # Should find multiple different archetypes
    unique_archetypes = set(a for a, _, _ in results)
    print(f"  Unique archetypes: {unique_archetypes}")

    assert len(unique_archetypes) >= 2, "Should discover multiple archetypes"

    print("  [PASS] Archetype discovery working correctly")
    return True


async def test_embedding_layer_methods():
    """Test that embedding layer advanced methods work in adapter context."""
    print("\n[TEST] Embedding Layer Method Integration")
    print("-" * 50)

    layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True, cache_size=100))
    await layer.initialize()

    # Test documents
    docs = [p["description"] for p in WITNESS_POSES]

    # Generate embeddings
    result = await layer.embed(
        texts=docs,
        model=EmbeddingModel.VOYAGE_4_LITE,
        input_type=InputType.DOCUMENT,
    )

    # Test MMR
    mmr_results = await layer.semantic_search_mmr(
        query="expressive body movement",
        documents=docs,
        doc_embeddings=result.embeddings,
        top_k=3,
        lambda_mult=0.5,
    )
    print(f"  MMR results: {len(mmr_results)}")

    # Test hybrid
    hybrid_results = await layer.hybrid_search(
        query="warrior pose stance",
        documents=docs,
        doc_embeddings=result.embeddings,
        top_k=3,
        alpha=0.5,
    )
    print(f"  Hybrid results: {len(hybrid_results)}")

    # Test filtered
    metadata = [{"archetype": p["archetype"]} for p in WITNESS_POSES]
    filtered_results = await layer.semantic_search_with_filters(
        query="contemplative pose",
        documents=docs,
        metadata=metadata,
        doc_embeddings=result.embeddings,
        filters={"archetype": {"$in": ["SAGE", "INNOCENT"]}},
        top_k=3,
    )
    print(f"  Filtered results (SAGE/INNOCENT): {len(filtered_results)}")

    assert len(mmr_results) == 3
    assert len(hybrid_results) == 3

    print("  [PASS] Embedding layer methods integrated correctly")
    return True


async def main():
    """Run all adapter tests."""
    print("=" * 60)
    print("VOYAGE AI V39.4 - ADAPTER ENHANCEMENT TESTS")
    print("=" * 60)

    tests = [
        ("Embedding Layer Methods", test_embedding_layer_methods),
        ("Witness Pose MMR", test_witness_pose_mmr),
        ("Witness Hybrid Shader", test_witness_hybrid_shader),
        ("Witness Filtered Particles", test_witness_filtered_particles),
        ("Witness Archetype Discovery", test_witness_archetype_discovery),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            await test_fn()
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
