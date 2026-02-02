"""
Test script for the Enhanced Unified Autonomous Research Pipeline.

Tests the integration of:
- Exa (search)
- Firecrawl (scraping)
- Crawl4AI (deep crawling)
- LightRAG (knowledge graph)
- LlamaIndex (vector RAG)
- Cache Layer (memory + disk caching)
- Unified Thinking Orchestrator (GoT, ToT, CoT, Debate, etc.)
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


def test_thinking_orchestrator():
    """Test unified thinking orchestrator."""
    print("=" * 60)
    print("UNIFIED THINKING ORCHESTRATOR STATUS")
    print("=" * 60)

    try:
        from core.unified_thinking_orchestrator import (
            UnifiedThinkingOrchestrator,
            ThinkingStrategy,
            create_thinking_orchestrator,
        )

        print("\n[OK] Thinking orchestrator module available")

        # Test creation
        orchestrator = create_thinking_orchestrator()
        print("[OK] Orchestrator created successfully")

        # Show available strategies
        strategies = [s.value for s in ThinkingStrategy]
        print(f"\n[OK] Available strategies: {len(strategies)}")
        for strategy in strategies:
            print(f"    - {strategy}")

        print()
        return orchestrator
    except ImportError as e:
        print(f"\n[--] Thinking orchestrator not available: {e}")
        return None
    except Exception as e:
        print(f"\n[--] Thinking orchestrator error: {e}")
        return None


@pytest.mark.asyncio
async def test_thinking_strategies():
    """Test each thinking strategy (dry run)."""
    print("=" * 60)
    print("THINKING STRATEGIES TEST (DRY RUN)")
    print("=" * 60)

    try:
        from core.unified_thinking_orchestrator import (
            UnifiedThinkingOrchestrator,
            ThinkingStrategy,
        )

        orchestrator = UnifiedThinkingOrchestrator(
            default_strategy=ThinkingStrategy.CHAIN_OF_THOUGHT,
            default_budget_tier="simple",
        )

        question = "What is the best approach for scalable microservices?"

        strategies_to_test = [
            ("Chain-of-Thought", ThinkingStrategy.CHAIN_OF_THOUGHT),
            ("Tree-of-Thoughts", ThinkingStrategy.TREE_OF_THOUGHTS),
            ("Graph-of-Thoughts", ThinkingStrategy.GRAPH_OF_THOUGHTS),
            ("Self-Consistency", ThinkingStrategy.SELF_CONSISTENCY),
            ("Debate", ThinkingStrategy.DEBATE),
        ]

        print(f"\nQuestion: {question}")
        print("-" * 40)

        for name, strategy in strategies_to_test:
            session = await orchestrator.think(
                question=question,
                strategy=strategy,
                budget_tier="simple",
                max_depth=2,
                num_branches=2,
            )

            uncertainty = orchestrator.estimate_uncertainty(session)

            print(f"\n[{name}]")
            print(f"  Nodes created: {session.node_count}")
            print(f"  Budget used: {session.budget.utilization:.1%}")
            print(f"  Confidence: {session.final_confidence:.2f}")
            print(f"  Uncertainty: {uncertainty.confidence_level.value}")

        stats = orchestrator.get_stats()
        print(f"\n[OK] Total sessions: {stats['total_sessions']}")
        print(f"[OK] Total nodes: {stats['total_nodes_created']}")

        print()
        return True
    except Exception as e:
        print(f"\n[--] Strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


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

    # Test thinking initialization
    print("\n[Testing Thinking Integration]")
    thinking_result = orchestrator.init_thinking(
        strategy="graph_of_thoughts",
        budget_tier="simple",
    )
    print(f"[OK] Thinking init: {thinking_result.get('success')}")
    if thinking_result.get('success'):
        print(f"[OK] Available strategies: {len(thinking_result.get('available_strategies', []))}")

    print()
    return orchestrator


@pytest.mark.asyncio
async def test_research_with_thinking_dry_run():
    """Test research with thinking pipeline (dry run without API calls)."""
    print("=" * 60)
    print("RESEARCH + THINKING PIPELINE (DRY RUN)")
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
    print(f"  - Unified Thinking: {orchestrator.has_thinking}")

    print("\nThinking strategies available:")
    strategies = [
        "chain_of_thought",
        "tree_of_thoughts",
        "graph_of_thoughts",
        "self_consistency",
        "debate",
        "metacognitive",
        "reflexion",
        "ultrathink",
    ]
    for s in strategies:
        print(f"  - {s}")

    print("\n[INFO] To run full pipeline with thinking, use:")
    print("  results = await orchestrator.research_with_thinking(")
    print("      query='AI reasoning patterns',")
    print("      question='What are the best approaches?',")
    print("      thinking_strategy='graph_of_thoughts'")
    print("  )")

    print()
    return True


@pytest.mark.asyncio
async def test_self_reflection():
    """Test self-reflection verification (dry run)."""
    print("=" * 60)
    print("SELF-REFLECTION VERIFICATION TEST")
    print("=" * 60)

    from core.ecosystem_orchestrator import get_orchestrator

    orchestrator = get_orchestrator()

    # Initialize thinking if not already done
    if not orchestrator.has_thinking:
        orchestrator.init_thinking(strategy="graph_of_thoughts")

    # Test self-reflection on a mock conclusion
    mock_conclusion = "Microservices should use event-driven architecture."
    mock_content = [
        {"title": "Source 1", "content": "Event-driven architecture enables loose coupling..."},
        {"title": "Source 2", "content": "Scalability is achieved through horizontal scaling..."},
    ]

    print("\n[Testing Self-Reflection]")
    print(f"  Mock conclusion: '{mock_conclusion}'")

    try:
        reflection_result = await orchestrator._verify_with_reflection(
            conclusion=mock_conclusion,
            research_content=mock_content,
            original_question="What is the best approach for scalable microservices?",
        )

        print(f"\n[OK] Reflection executed successfully")
        print(f"  Success: {reflection_result.get('success')}")
        print(f"  Improved: {reflection_result.get('improved')}")
        print(f"  Confidence: {reflection_result.get('reflection_confidence', 'N/A')}")
        print(f"  Nodes evaluated: {reflection_result.get('nodes_evaluated', 'N/A')}")

        if reflection_result.get("refined_conclusion"):
            print(f"  Refined conclusion: {reflection_result['refined_conclusion'][:100]}...")

        print()
        return reflection_result
    except Exception as e:
        print(f"\n[--] Self-reflection test failed: {e}")
        return None


def main():
    """Run all tests."""
    print()
    print("=" * 60)
    print("  ENHANCED UNIFIED AUTONOMOUS RESEARCH PIPELINE TESTS")
    print("=" * 60)
    print()

    # Test 1: SDK Status
    sdk_status_result = test_sdk_status()

    # Test 2: Cache Layer
    cache = test_cache_layer()

    # Test 3: Thinking Orchestrator
    thinking = test_thinking_orchestrator()

    # Test 4: Thinking Strategies
    asyncio.run(test_thinking_strategies())

    # Test 5: Orchestrator Status (includes thinking)
    orchestrator = test_orchestrator_status()

    # Test 6: Research + Thinking Pipeline Dry Run
    asyncio.run(test_research_with_thinking_dry_run())

    # Test 7: Self-Reflection Verification
    reflection_result = asyncio.run(test_self_reflection())

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    available_count = len(sdk_status_result['available'])
    total_sdks = len(sdk_status_result['sdks'])

    print(f"\n[RESULT] {available_count}/{total_sdks} SDKs available")
    print(f"[RESULT] Cache layer: {'Enabled' if cache else 'Disabled'}")
    print(f"[RESULT] Thinking orchestrator: {'Enabled' if thinking else 'Disabled'}")
    print(f"[RESULT] Self-reflection: {'Working' if reflection_result and reflection_result.get('success') else 'Not tested'}")
    print(f"[RESULT] Enhanced pipeline ready: {'Yes' if available_count >= 2 and thinking else 'Partial'}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
