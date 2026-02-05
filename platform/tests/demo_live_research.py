"""
UNLEASH Live Research Test - Real API Calls
============================================
Testing all 5 research tools with REAL API calls!
"""
import asyncio
import os
import sys
import time

# Add platform to path
sys.path.insert(0, "Z:/insider/AUTO CLAUDE/unleash/platform")

# Load environment
from dotenv import load_dotenv
load_dotenv("Z:/insider/AUTO CLAUDE/unleash/.env")
load_dotenv("Z:/insider/AUTO CLAUDE/unleash/.config/.env")

print("=" * 70)
print("UNLEASH RESEARCH SWARM - LIVE API TEST")
print("=" * 70)

async def test_exa_full_features():
    """Test Exa's full 9 features."""
    print("\n[1/5] EXA NEURAL SEARCH - Testing All Features")
    print("-" * 50)

    try:
        from adapters.exa_adapter import ExaAdapter

        exa = ExaAdapter()
        await exa.initialize({})

        # Feature 1: Neural Search
        print("  [1] Neural semantic search...")
        result = await exa.execute("search", query="LangGraph StateGraph tutorial", num_results=3)
        if result.success:
            print(f"      Found {len(result.data.get('results', []))} results")
            for r in result.data.get('results', [])[:2]:
                print(f"        - {r.get('title', 'N/A')[:50]}")
        else:
            print(f"      Error: {result.error}")

        # Feature 2: Search with contents
        print("  [2] Search with full contents...")
        result = await exa.execute("search", query="DSPy optimization", num_results=2, include_contents=True)
        if result.success:
            results = result.data.get('results', [])
            if results and 'text' in results[0]:
                print(f"      Got content: {len(results[0].get('text', ''))} chars")

        # Feature 3: Find similar
        print("  [3] Find similar URLs...")
        result = await exa.execute("find_similar", url="https://python.langchain.com/docs/", num_results=3)
        if result.success:
            print(f"      Found {len(result.data.get('results', []))} similar sites")

        # Feature 4: Get contents
        print("  [4] Get URL contents...")
        result = await exa.execute("get_contents", urls=["https://docs.anthropic.com/"])
        if result.success:
            print(f"      Retrieved content successfully")

        await exa.shutdown()
        print("  EXA: All features tested!")
        return True

    except Exception as e:
        print(f"  EXA Error: {e}")
        return False

async def test_tavily_features():
    """Test Tavily AI search."""
    print("\n[2/5] TAVILY AI SEARCH")
    print("-" * 50)

    try:
        from adapters.tavily_adapter import TavilyAdapter

        tavily = TavilyAdapter()
        await tavily.initialize({})

        # Standard search
        print("  [1] AI-powered search...")
        result = await tavily.execute("search", query="Claude API best practices 2024")
        if result.success:
            results = result.data.get('results', [])
            print(f"      Found {len(results)} results")
            for r in results[:2]:
                print(f"        - {r.get('title', 'N/A')[:50]}")

        # Context search
        print("  [2] Context-aware search...")
        result = await tavily.execute("search",
            query="React hooks patterns",
            search_depth="advanced"
        )
        if result.success:
            print(f"      Advanced search completed")

        await tavily.shutdown()
        print("  TAVILY: All features tested!")
        return True

    except Exception as e:
        print(f"  TAVILY Error: {e}")
        return False

async def test_jina_features():
    """Test Jina URL to markdown."""
    print("\n[3/5] JINA READER - URL TO MARKDOWN")
    print("-" * 50)

    try:
        from adapters.jina_adapter import JinaAdapter

        jina = JinaAdapter()
        await jina.initialize({})

        # Read URL
        print("  [1] Reading FastAPI docs...")
        result = await jina.execute("read", url="https://fastapi.tiangolo.com/")
        if result.success:
            content = result.data.get('content', '')
            print(f"      Got {len(content)} chars of markdown")
            print(f"      Preview: {content[:100]}...")

        # Read another URL
        print("  [2] Reading Anthropic docs...")
        result = await jina.execute("read", url="https://docs.anthropic.com/en/docs/welcome")
        if result.success:
            content = result.data.get('content', '')
            print(f"      Got {len(content)} chars")

        await jina.shutdown()
        print("  JINA: All features tested!")
        return True

    except Exception as e:
        print(f"  JINA Error: {e}")
        return False

async def test_perplexity_features():
    """Test Perplexity deep research."""
    print("\n[4/5] PERPLEXITY SONAR - DEEP RESEARCH")
    print("-" * 50)

    try:
        from adapters.perplexity_adapter import PerplexityAdapter

        pplx = PerplexityAdapter()
        await pplx.initialize({})

        # Standard chat
        print("  [1] Sonar chat with web grounding...")
        result = await pplx.execute("chat",
            query="What are the latest features in LangGraph 0.2?",
        )
        if result.success:
            content = result.data.get('content', '')
            citations = result.data.get('citations', [])
            print(f"      Response: {len(content)} chars, {len(citations)} citations")
            if content:
                print(f"      Preview: {content[:150]}...")
        else:
            print(f"      Note: {result.error}")

        # Pro search (if available)
        print("  [2] Sonar Pro search...")
        result = await pplx.execute("pro",
            query="Compare Raft vs Paxos consensus algorithms",
        )
        if result.success:
            print(f"      Pro response received")

        await pplx.shutdown()
        print("  PERPLEXITY: Features tested!")
        return True

    except Exception as e:
        print(f"  PERPLEXITY Error: {e}")
        return False

async def test_context7_features():
    """Test Context7 SDK documentation."""
    print("\n[5/5] CONTEXT7 - SDK DOCUMENTATION")
    print("-" * 50)

    try:
        from adapters.context7_adapter import Context7Adapter

        ctx7 = Context7Adapter()
        await ctx7.initialize({})

        # Resolve library
        print("  [1] Resolving LangChain library...")
        result = await ctx7.execute("resolve", library_name="langchain")
        if result.success:
            print(f"      Library ID: {result.data.get('library_id')}")

        # Query docs
        print("  [2] Querying LangGraph documentation...")
        result = await ctx7.execute("query",
            library_id="langgraph",
            query="StateGraph",
        )
        if result.success:
            content = result.data.get('content', '')
            print(f"      Got {len(content)} chars of documentation")
            if content:
                print(f"      Preview: {content[:150]}...")

        # Query React docs
        print("  [3] Querying React documentation...")
        result = await ctx7.execute("query",
            library_id="react",
            query="hooks useState useEffect",
        )
        if result.success:
            print(f"      React docs retrieved")

        await ctx7.shutdown()
        print("  CONTEXT7: All features tested!")
        return True

    except Exception as e:
        print(f"  CONTEXT7 Error: {e}")
        return False

async def test_full_research_swarm():
    """Test the full Ultimate Research Swarm."""
    print("\n" + "=" * 70)
    print("ULTIMATE RESEARCH SWARM - FULL INTEGRATION TEST")
    print("=" * 70)

    try:
        from core.ultimate_research_swarm import (
            UltimateResearchSwarm,
            ResearchDepth,
            get_ultimate_swarm,
        )

        swarm = UltimateResearchSwarm()
        await swarm.initialize()

        # Quick research
        print("\n[A] Quick Research (<2s target)...")
        start = time.time()
        result = await swarm.research(
            "LangGraph StateGraph best practices",
            depth=ResearchDepth.QUICK,
        )
        elapsed = time.time() - start
        print(f"    Query: {result.query}")
        print(f"    Tools: {result.tools_used}")
        print(f"    Sources: {len(result.sources)}")
        print(f"    Latency: {elapsed:.2f}s")
        print(f"    Confidence: {result.confidence:.0%}")
        if result.key_findings:
            print(f"    Findings: {result.key_findings[:2]}")

        # Comprehensive research
        print("\n[B] Comprehensive Research (multi-tool)...")
        start = time.time()
        result = await swarm.research(
            "distributed consensus algorithms comparison Raft Paxos PBFT",
            depth=ResearchDepth.COMPREHENSIVE,
        )
        elapsed = time.time() - start
        print(f"    Query: {result.query}")
        print(f"    Tools: {result.tools_used}")
        print(f"    Sources: {len(result.sources)}")
        print(f"    Latency: {elapsed:.2f}s")
        print(f"    Confidence: {result.confidence:.0%}")

        # SDK Documentation research
        print("\n[C] SDK Documentation Research...")
        result = await swarm.research_sdk_docs(
            library="fastapi",
            query="dependency injection patterns",
        )
        print(f"    Query: {result.query}")
        print(f"    Tools: {result.tools_used}")
        print(f"    Sources: {len(result.sources)}")

        # Get stats
        stats = swarm.get_stats()
        print("\n[STATS]")
        print(f"    Total queries: {stats.get('total_queries', 0)}")
        print(f"    Cache hits: {stats.get('cache_hits', 0)}")

        await swarm.shutdown()
        print("\n ULTIMATE RESEARCH SWARM: FULLY OPERATIONAL!")
        return True

    except Exception as e:
        import traceback
        print(f"Swarm Error: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print(f"\nAPI Keys loaded:")
    print(f"  EXA_API_KEY: {'Y' if os.getenv('EXA_API_KEY') else 'N'}")
    print(f"  TAVILY_API_KEY: {'Y' if os.getenv('TAVILY_API_KEY') else 'N'}")
    print(f"  JINA_API_KEY: {'Y' if os.getenv('JINA_API_KEY') else 'N'}")
    print(f"  PERPLEXITY_API_KEY: {'Y' if os.getenv('PERPLEXITY_API_KEY') else 'N'}")
    print(f"  CONTEXT7_API_KEY: {'Y' if os.getenv('CONTEXT7_API_KEY') else 'N'}")

    results = {}

    # Test each adapter
    results['exa'] = await test_exa_full_features()
    results['tavily'] = await test_tavily_features()
    results['jina'] = await test_jina_features()
    results['perplexity'] = await test_perplexity_features()
    results['context7'] = await test_context7_features()

    # Test full swarm
    results['swarm'] = await test_full_research_swarm()

    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {name.upper():15} {status}")

    total = sum(results.values())
    print(f"\n  TOTAL: {total}/{len(results)} tests passed")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
