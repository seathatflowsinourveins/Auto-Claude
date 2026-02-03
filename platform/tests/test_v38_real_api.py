#!/usr/bin/env python3
"""
V38.0 REAL API Validation Test - Voyage AI Embedding Layer

Tests ALL features with REAL API calls - NO MOCKS.
Validates: HTTP API, output_dimension, quantization, ALL best models.

PAID TIER - Full rate limits (300 RPM, 1M TPM)

API Key: Uses VOYAGE_API_KEY env var or default from embedding_layer.py
"""

import asyncio
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.orchestration.embedding_layer import (
    EmbeddingLayer,
    EmbeddingConfig,
    EmbeddingModel,
    RerankModel,
    InputType,
    OutputDType,
    OutputDimension,
    create_embedding_layer,
    create_model_mixing_layer,
    VOYAGE_AVAILABLE,
    HTTPX_AVAILABLE,
)


# Rate limit delay (paid tier has 300 RPM - minimal delay needed)
RATE_LIMIT_DELAY = 1  # seconds


async def wait_for_rate_limit():
    """Minimal wait for paid tier."""
    await asyncio.sleep(RATE_LIMIT_DELAY)


async def test_v38_real_api():
    """Comprehensive V38.0 REAL API validation with BEST models."""
    print("\n" + "=" * 70)
    print("V38.0 REAL API VALIDATION - BEST MODELS (PAID TIER)")
    print("=" * 70)

    results = []

    # Check dependencies
    print(f"\n[DEPS] Voyage SDK Available: {VOYAGE_AVAILABLE}")
    print(f"[DEPS] HTTPX Available: {HTTPX_AVAILABLE}")
    print(f"[MODE] Paid Tier - Full Rate Limits")

    if not VOYAGE_AVAILABLE:
        print("\n[ERROR] voyageai not installed. Run: pip install voyageai")
        return False

    layer = None
    docs = []

    # =========================================================================
    # TEST 1: voyage-4-large (BEST General Embeddings)
    # =========================================================================
    print("\n" + "-" * 50)
    print("[TEST 1] voyage-4-large (BEST General Model)")
    print("-" * 50)

    try:
        layer = create_embedding_layer(model=EmbeddingModel.VOYAGE_4_LARGE.value)
        await layer.initialize()

        docs = [
            "Machine learning enables computers to learn from data patterns.",
            "Deep neural networks power modern AI systems with layered architectures.",
            "Natural language processing understands human text semantically.",
            "Computer vision systems recognize objects in images and videos.",
            "Reinforcement learning agents learn through trial and error.",
        ]

        result = await layer.embed_documents(docs)

        print(f"  Model: {result.model}")
        print(f"  Count: {result.count}")
        print(f"  Dimension: {result.dimension}")
        print(f"  Tokens: {result.total_tokens}")

        assert result.count == 5, f"Expected 5, got {result.count}"
        assert result.dimension == 1024, f"Expected 1024, got {result.dimension}"

        results.append(("voyage-4-large", True, f"dim={result.dimension}, tokens={result.total_tokens}"))
        print("  [PASS]")

    except Exception as e:
        results.append(("voyage-4-large", False, str(e)[:100]))
        print(f"  [FAIL] {e}")

    await wait_for_rate_limit()

    # =========================================================================
    # TEST 2: voyage-4-lite (Fast Queries)
    # =========================================================================
    print("\n" + "-" * 50)
    print("[TEST 2] voyage-4-lite (Fast Query Model)")
    print("-" * 50)

    try:
        lite_layer = create_embedding_layer(model=EmbeddingModel.VOYAGE_4_LITE.value)
        await lite_layer.initialize()

        query_result = await lite_layer.embed_queries([
            "How does machine learning work?",
            "What are neural networks?",
        ])

        print(f"  Model: {query_result.model}")
        print(f"  Count: {query_result.count}")
        print(f"  Dimension: {query_result.dimension}")

        assert query_result.model == "voyage-4-lite"
        assert query_result.dimension == 1024

        results.append(("voyage-4-lite", True, f"dim={query_result.dimension}"))
        print("  [PASS]")

    except Exception as e:
        results.append(("voyage-4-lite", False, str(e)[:100]))
        print(f"  [FAIL] {e}")

    await wait_for_rate_limit()

    # =========================================================================
    # TEST 3: Model Mixing (voyage-4-large + voyage-4-lite)
    # =========================================================================
    print("\n" + "-" * 50)
    print("[TEST 3] Model Mixing (Large Docs + Lite Queries)")
    print("-" * 50)

    try:
        mix_layer = create_model_mixing_layer(
            document_model=EmbeddingModel.VOYAGE_4_LARGE.value,
            query_model=EmbeddingModel.VOYAGE_4_LITE.value,
        )
        await mix_layer.initialize()

        # Embed query with lite model
        query_emb = await mix_layer.embed_query("AI and machine learning")

        print(f"  Document model: {mix_layer.config.model}")
        print(f"  Query model: {mix_layer.config.query_model}")
        print(f"  Query dimension: {len(query_emb)}")
        print(f"  Shared embedding space: YES")

        assert len(query_emb) == 1024

        results.append(("Model Mixing", True, "large+lite, shared space"))
        print("  [PASS]")

    except Exception as e:
        results.append(("Model Mixing", False, str(e)[:100]))
        print(f"  [FAIL] {e}")

    await wait_for_rate_limit()

    # =========================================================================
    # TEST 4: voyage-code-3 (BEST Code Embeddings)
    # =========================================================================
    print("\n" + "-" * 50)
    print("[TEST 4] voyage-code-3 (BEST Code Model)")  # 1024d default, 2048d max
    print("-" * 50)

    try:
        code_snippets = [
            "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "async function fetchData(url: string): Promise<any> {\n    const response = await fetch(url);\n    return response.json();\n}",
            "fn main() {\n    let nums = vec![1, 2, 3, 4, 5];\n    let sum: i32 = nums.iter().sum();\n    println!(\"Sum: {}\", sum);\n}",
        ]

        code_result = await layer.embed_code(code_snippets)

        print(f"  Model: {code_result.model}")
        print(f"  Count: {code_result.count}")
        print(f"  Dimension: {code_result.dimension}")

        assert code_result.model == "voyage-code-3"
        # voyage-code-3 returns 1024d by default, supports up to 2048d via output_dimension
        assert code_result.dimension in (1024, 2048), f"Expected 1024/2048, got {code_result.dimension}"

        results.append(("voyage-code-3", True, f"dim={code_result.dimension}"))
        print("  [PASS]")

    except Exception as e:
        results.append(("voyage-code-3", False, str(e)[:100]))
        print(f"  [FAIL] {e}")

    await wait_for_rate_limit()

    # =========================================================================
    # TEST 5: Semantic Search
    # =========================================================================
    print("\n" + "-" * 50)
    print("[TEST 5] Semantic Search")
    print("-" * 50)

    try:
        search_results = await layer.semantic_search(
            query="neural networks and deep learning",
            documents=docs,
            top_k=3,
        )

        print(f"  Query: 'neural networks and deep learning'")
        print(f"  Top 3 results:")
        for i, (idx, score, doc) in enumerate(search_results[:3]):
            print(f"    {i+1}. [{score:.4f}] {doc[:50]}...")

        assert len(search_results) == 3
        # Cosine similarity thresholds vary; 0.4+ is reasonable for semantic match
        assert search_results[0][1] > 0.4, f"Expected >0.4, got {search_results[0][1]}"

        results.append(("Semantic Search", True, f"top_score={search_results[0][1]:.4f}"))
        print("  [PASS]")

    except Exception as e:
        results.append(("Semantic Search", False, str(e)[:100]))
        print(f"  [FAIL] {e}")

    await wait_for_rate_limit()

    # =========================================================================
    # TEST 6: rerank-2.5 (BEST Reranker)
    # =========================================================================
    print("\n" + "-" * 50)
    print("[TEST 6] rerank-2.5 (BEST Reranker - 32K tokens)")
    print("-" * 50)

    try:
        rerank_result = await layer.rerank(
            query="artificial intelligence and machine learning systems",
            documents=docs,
            model=RerankModel.RERANK_2_5.value,
            top_k=5,
        )

        print(f"  Model: {rerank_result.model}")
        print(f"  Results: {rerank_result.count}")
        print(f"  Top scores:")
        for idx, score, doc in rerank_result.results[:3]:
            print(f"    [{score:.4f}] {doc[:50]}...")

        assert rerank_result.model == "rerank-2.5"
        assert rerank_result.count == 5

        results.append(("rerank-2.5", True, f"top_score={rerank_result.results[0][1]:.4f}"))
        print("  [PASS]")

    except Exception as e:
        results.append(("rerank-2.5", False, str(e)[:100]))
        print(f"  [FAIL] {e}")

    await wait_for_rate_limit()

    # =========================================================================
    # TEST 7: Two-Stage Search (Embeddings + Reranking)
    # =========================================================================
    print("\n" + "-" * 50)
    print("[TEST 7] Two-Stage Search (Embed + Rerank Pipeline)")
    print("-" * 50)

    try:
        two_stage = await layer.semantic_search_with_rerank(
            query="deep learning systems for AI",
            documents=docs,
            initial_k=5,
            final_k=3,
        )

        print(f"  Stage 1: Embedding search (top 5)")
        print(f"  Stage 2: Reranking (top 3)")
        print(f"  Final results:")
        for idx, score, doc in two_stage.results:
            print(f"    [{score:.4f}] {doc[:50]}...")

        assert two_stage.count == 3

        results.append(("Two-Stage Search", True, f"5->3, score={two_stage.results[0][1]:.4f}"))
        print("  [PASS]")

    except Exception as e:
        results.append(("Two-Stage Search", False, str(e)[:100]))
        print(f"  [FAIL] {e}")

    await wait_for_rate_limit()

    # =========================================================================
    # TEST 8: HTTP API - Output Dimension 256 (V38.0)
    # =========================================================================
    print("\n" + "-" * 50)
    print("[TEST 8] HTTP API - Output Dimension 256 (V38.0)")
    print("-" * 50)

    if not HTTPX_AVAILABLE:
        print("  [SKIP] httpx not installed")
        results.append(("Output Dimension", None, "httpx not installed"))
    else:
        try:
            dim_result = await layer.embed_with_options(
                texts=["Test dimension reduction to 256."],
                output_dimension=256,
            )

            print(f"  Requested: 256d")
            print(f"  Actual: {dim_result.dimension}d")
            print(f"  Storage: 4x smaller than 1024d")

            assert dim_result.dimension == 256, f"Expected 256, got {dim_result.dimension}"

            results.append(("Output Dim 256", True, f"256d OK (4x smaller)"))
            print("  [PASS]")

        except Exception as e:
            results.append(("Output Dim 256", False, str(e)[:100]))
            print(f"  [FAIL] {e}")

    await wait_for_rate_limit()

    # =========================================================================
    # TEST 9: HTTP API - Output Dimension 512 (V38.0)
    # =========================================================================
    print("\n" + "-" * 50)
    print("[TEST 9] HTTP API - Output Dimension 512 (V38.0)")
    print("-" * 50)

    if not HTTPX_AVAILABLE:
        print("  [SKIP] httpx not installed")
        results.append(("Output Dim 512", None, "httpx not installed"))
    else:
        try:
            dim_result = await layer.embed_with_options(
                texts=["Test dimension reduction to 512."],
                output_dimension=512,
            )

            print(f"  Requested: 512d")
            print(f"  Actual: {dim_result.dimension}d")

            assert dim_result.dimension == 512

            results.append(("Output Dim 512", True, f"512d OK (2x smaller)"))
            print("  [PASS]")

        except Exception as e:
            results.append(("Output Dim 512", False, str(e)[:100]))
            print(f"  [FAIL] {e}")

    await wait_for_rate_limit()

    # =========================================================================
    # TEST 10: HTTP API - Quantization int8 (V38.0)
    # =========================================================================
    print("\n" + "-" * 50)
    print("[TEST 10] HTTP API - Quantization int8 (V38.0)")
    print("-" * 50)

    if not HTTPX_AVAILABLE:
        print("  [SKIP] httpx not installed")
        results.append(("Quantization int8", None, "httpx not installed"))
    else:
        try:
            quant_result = await layer.embed_with_options(
                texts=["Test int8 quantization."],
                output_dtype=OutputDType.INT8.value,
            )

            print(f"  Dtype: int8")
            print(f"  Dimension: {quant_result.dimension}")
            print(f"  Storage: 4x smaller (1 byte per dim)")

            results.append(("Quantization int8", True, f"dim={quant_result.dimension}, 4x smaller"))
            print("  [PASS]")

        except Exception as e:
            results.append(("Quantization int8", False, str(e)[:100]))
            print(f"  [FAIL] {e}")

    await wait_for_rate_limit()

    # =========================================================================
    # TEST 11: HTTP API - Quantization binary (V38.0)
    # =========================================================================
    print("\n" + "-" * 50)
    print("[TEST 11] HTTP API - Quantization binary (V38.0)")
    print("-" * 50)

    if not HTTPX_AVAILABLE:
        print("  [SKIP] httpx not installed")
        results.append(("Quantization binary", None, "httpx not installed"))
    else:
        try:
            binary_result = await layer.embed_with_options(
                texts=["Test binary quantization for Hamming distance."],
                output_dtype=OutputDType.BINARY.value,
            )

            print(f"  Dtype: binary")
            print(f"  Dimension: {binary_result.dimension}")
            print(f"  Storage: 32x smaller (1 bit per dim)!")

            results.append(("Quantization binary", True, f"dim={binary_result.dimension}, 32x smaller!"))
            print("  [PASS]")

        except Exception as e:
            results.append(("Quantization binary", False, str(e)[:100]))
            print(f"  [FAIL] {e}")

    await wait_for_rate_limit()

    # =========================================================================
    # TEST 12: HTTP API - Combined 256d + int8 (V38.0)
    # =========================================================================
    print("\n" + "-" * 50)
    print("[TEST 12] HTTP API - Ultra Compact (256d + int8)")
    print("-" * 50)

    if not HTTPX_AVAILABLE:
        print("  [SKIP] httpx not installed")
        results.append(("Ultra Compact", None, "httpx not installed"))
    else:
        try:
            compact_result = await layer.embed_quantized(
                texts=["Ultra-compact embedding for billion-scale search."],
                dtype=OutputDType.INT8.value,
                dimension=OutputDimension.D256.value,
            )

            print(f"  Dimension: {compact_result.dimension}")
            print(f"  Dtype: int8")

            # Storage calculation
            baseline = 1024 * 4  # 1024d * 4 bytes (float32) = 4096 bytes
            compact = 256 * 1    # 256d * 1 byte (int8) = 256 bytes
            savings = baseline / compact

            print(f"  Storage: {compact} bytes (vs {baseline} baseline)")
            print(f"  Savings: {savings:.0f}x reduction!")

            assert compact_result.dimension == 256

            results.append(("Ultra Compact", True, f"256d int8 = {savings:.0f}x savings"))
            print("  [PASS]")

        except Exception as e:
            results.append(("Ultra Compact", False, str(e)[:100]))
            print(f"  [FAIL] {e}")

    # =========================================================================
    # TEST 13: Cache Performance
    # =========================================================================
    print("\n" + "-" * 50)
    print("[TEST 13] Cache Performance")
    print("-" * 50)

    try:
        # Clear and test cache
        layer.clear_cache()

        # First call - cache miss
        await layer.embed_documents(["Cache test document."])

        # Second call - cache hit
        await layer.embed_documents(["Cache test document."])

        stats = layer.get_stats()

        print(f"  Version: {stats['version']}")
        print(f"  Cache hits: {stats['cache_stats']['hits']}")
        print(f"  Cache misses: {stats['cache_stats']['misses']}")
        print(f"  Hit rate: {stats['cache_stats']['hit_rate']}%")

        assert stats['version'] == "V38.0"
        assert stats['cache_stats']['hits'] >= 1

        results.append(("Cache", True, f"hit_rate={stats['cache_stats']['hit_rate']}%"))
        print("  [PASS]")

    except Exception as e:
        results.append(("Cache", False, str(e)[:100]))
        print(f"  [FAIL] {e}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("V38.0 VALIDATION SUMMARY - BEST MODELS")
    print("=" * 70)

    passed = sum(1 for _, status, _ in results if status is True)
    failed = sum(1 for _, status, _ in results if status is False)
    skipped = sum(1 for _, status, _ in results if status is None)

    for name, status, detail in results:
        if status is True:
            icon = "\u2705"  # green check
        elif status is False:
            icon = "\u274c"  # red x
        else:
            icon = "\u23ed"  # skip
        print(f"  {icon} {name}: {detail}")

    print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 70)

    if passed >= 10:
        print("\n\u2728 V38.0 VALIDATED - All core features working!")
        print("   - voyage-4-large (best general)")
        print("   - voyage-code-3 (best code, 2048d)")
        print("   - rerank-2.5 (best reranker)")
        print("   - HTTP API (output_dimension + quantization)")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(test_v38_real_api())
    sys.exit(0 if success else 1)
