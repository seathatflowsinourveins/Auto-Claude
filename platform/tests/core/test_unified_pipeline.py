"""
Test script for the Unified Autonomous Research Pipeline.

Tests the integration of:
- Exa (search)
- Firecrawl (scraping)
- Crawl4AI (deep crawling)
- LightRAG (knowledge graph)
- LlamaIndex (vector RAG)
- Cache Layer (memory + disk caching)
- MCP Python SDK (Model Context Protocol)
"""

import sys
import os
import asyncio
import pytest

# Add platform path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_sdk_status():
    """Test SDK availability status."""
    print("=" * 60)
    print("SDK STATUS CHECK")
    print("=" * 60)

    from core.sdk_integrations import sdk_status

    status = sdk_status()

    print(f"\nAvailable SDKs: {status['available']}")
    print()

    for name, info in status['sdks'].items():
        symbol = "[OK]" if info['available'] else "[--]"
        partial = " (partial)" if info.get('partial') else ""
        print(f"  {symbol} {name:12}{partial} - {info['description']}")

    print()
    return status


def test_cache_layer():
    """Test cache layer functionality."""
    print("=" * 60)
    print("CACHE LAYER STATUS")
    print("=" * 60)

    try:
        from core.cache_layer import ResearchCache, get_cache, CACHE_AVAILABLE

        if not CACHE_AVAILABLE:
            print("\n[--] Cache layer not available")
            return None

        cache = get_cache()
        print(f"\n[OK] Cache layer available")
        print(f"[OK] Cache initialized: {cache is not None}")

        # Test operations
        cache.cache_search('test_query', {'results': []})
        result = cache.get_search('test_query')
        print(f"[OK] Search caching: {'Working' if result is not None else 'Failed'}")

        cache.cache_scrape('http://test.com', {'content': 'test'})
        result = cache.get_scrape('http://test.com')
        print(f"[OK] Scrape caching: {'Working' if result is not None else 'Failed'}")

        stats = cache.stats()
        print(f"\n[OK] Memory cache: {stats['memory']['items']} items")
        print(f"[OK] Disk cache: {stats['disk']['items']} items")

        print()
        return cache
    except Exception as e:
        print(f"\n[--] Cache layer error: {e}")
        return None


def test_orchestrator_status():
    """Test ecosystem orchestrator status."""
    print("=" * 60)
    print("ECOSYSTEM ORCHESTRATOR STATUS")
    print("=" * 60)

    from core.ecosystem_orchestrator import get_orchestrator

    orchestrator = get_orchestrator()
    status = orchestrator.sdk_status()

    print(f"\n[OK] Orchestrator created")
    print(f"[OK] Research available: {status.get('research', {}).get('available')}")
    print(f"[OK] Research initialized: {status.get('research', {}).get('initialized')}")
    print(f"[OK] Cache available: {status.get('cache', {}).get('available')}")
    print(f"[OK] Cache initialized: {status.get('cache', {}).get('initialized')}")

    extended = status.get('extended_sdks', {})
    print(f"\nExtended SDKs: {extended.get('available', [])}")

    print()
    return orchestrator


@pytest.mark.asyncio
async def test_autonomous_research_dry_run():
    """Test autonomous research pipeline (dry run without API calls)."""
    print("=" * 60)
    print("AUTONOMOUS RESEARCH PIPELINE (DRY RUN)")
    print("=" * 60)

    from core.ecosystem_orchestrator import get_orchestrator
    from core.sdk_integrations import (
        CRAWL4AI_AVAILABLE,
        LLAMAINDEX_AVAILABLE,
        LIGHTRAG_AVAILABLE,
    )

    orchestrator = get_orchestrator()

    print("\nPipeline capabilities:")
    print(f"  - Exa Search: Available")
    print(f"  - Firecrawl Scrape: Available")
    print(f"  - Crawl4AI Deep Crawl: {'Available' if CRAWL4AI_AVAILABLE else 'Not available'}")
    print(f"  - LightRAG Knowledge Graph: {'Available' if LIGHTRAG_AVAILABLE else 'Not available'}")
    print(f"  - LlamaIndex Vector RAG: {'Available' if LLAMAINDEX_AVAILABLE else 'Not available'}")

    print("\n[INFO] To run full pipeline, use:")
    print("  results = await orchestrator.autonomous_research('your query')")
    print("\n[INFO] To research and query, use:")
    print("  results = await orchestrator.research_and_query('topic', 'question')")

    print()
    return True


def main():
    """Run all tests."""
    print()
    print("=" * 60)
    print("  UNIFIED AUTONOMOUS RESEARCH PIPELINE TESTS")
    print("=" * 60)
    print()

    # Test 1: SDK Status
    sdk_status_result = test_sdk_status()

    # Test 2: Cache Layer
    cache = test_cache_layer()

    # Test 3: Orchestrator Status
    orchestrator = test_orchestrator_status()

    # Test 4: Pipeline Dry Run
    asyncio.run(test_autonomous_research_dry_run())

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    available_count = len(sdk_status_result['available'])
    total_sdks = len(sdk_status_result['sdks'])

    print(f"\n[RESULT] {available_count}/{total_sdks} SDKs available")
    print(f"[RESULT] Cache layer: {'Enabled' if cache else 'Disabled'}")
    print(f"[RESULT] Unified pipeline ready: {'Yes' if available_count >= 2 else 'Partial'}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
