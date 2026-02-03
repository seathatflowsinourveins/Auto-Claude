"""
DEEP DIVE RESEARCH - Full Feature Exploration
==============================================
Exploring ALL features of each research tool and fetching
real official documentation from multiple sources.
"""
import asyncio
import os
import sys
import time
import json

sys.path.insert(0, "Z:/insider/AUTO CLAUDE/unleash/platform")

from dotenv import load_dotenv
load_dotenv("Z:/insider/AUTO CLAUDE/unleash/.env")
load_dotenv("Z:/insider/AUTO CLAUDE/unleash/.config/.env")

print("=" * 80)
print("DEEP DIVE RESEARCH - EXPLORING ALL FEATURES")
print("=" * 80)


async def deep_dive_exa():
    """Deep dive into ALL 9 Exa features."""
    print("\n" + "=" * 80)
    print("EXA NEURAL SEARCH - ALL 9 FEATURES")
    print("=" * 80)

    from adapters.exa_adapter import ExaAdapter
    exa = ExaAdapter()
    await exa.initialize({})

    features_tested = 0

    # Feature 1: Neural Semantic Search
    print("\n[1/9] NEURAL SEMANTIC SEARCH")
    print("-" * 40)
    result = await exa.execute("search", query="state-of-the-art transformer architectures 2024", num_results=5)
    if result.success:
        print(f"   Query: 'state-of-the-art transformer architectures 2024'")
        print(f"   Results: {len(result.data.get('results', []))}")
        for r in result.data.get('results', [])[:3]:
            print(f"   - {r['title'][:60]}...")
            print(f"     URL: {r['url'][:70]}...")
        features_tested += 1

    # Feature 2: Keyword Search (Fast)
    print("\n[2/9] KEYWORD SEARCH (FAST MODE)")
    print("-" * 40)
    result = await exa.execute("search", query="LangChain LCEL documentation", type="fast", num_results=5)
    if result.success:
        print(f"   Mode: Keyword (fast)")
        print(f"   Results: {len(result.data.get('results', []))}")
        features_tested += 1
    else:
        print(f"   Error: {result.error}")

    # Feature 3: Auto Mode (Best of Both)
    print("\n[3/9] AUTO MODE SEARCH")
    print("-" * 40)
    result = await exa.execute("search", query="how to implement RAG with vector databases", type="auto", num_results=5)
    if result.success:
        print(f"   Mode: Auto (neural + keyword)")
        print(f"   Results: {len(result.data.get('results', []))}")
        features_tested += 1

    # Feature 4: Search with Full Contents
    print("\n[4/9] SEARCH WITH FULL CONTENTS")
    print("-" * 40)
    result = await exa.execute("search", query="FastAPI dependency injection tutorial", num_results=2, include_contents=True)
    if result.success:
        results = result.data.get('results', [])
        if results:
            content = results[0].get('text', '')
            print(f"   Content retrieved: {len(content)} characters")
            print(f"   Preview: {content[:200]}...")
        features_tested += 1

    # Feature 5: Search with Highlights
    print("\n[5/9] SEARCH WITH HIGHLIGHTS")
    print("-" * 40)
    result = await exa.execute("search", query="Claude API function calling", num_results=3, include_highlights=True)
    if result.success:
        results = result.data.get('results', [])
        if results:
            highlights = results[0].get('highlights', []) or []
            print(f"   Highlights extracted: {len(highlights)}")
            for h in highlights[:2]:
                print(f"   - {str(h)[:100]}...")
        features_tested += 1

    # Feature 6: Find Similar URLs
    print("\n[6/9] FIND SIMILAR URLs")
    print("-" * 40)
    result = await exa.execute("find_similar", url="https://docs.anthropic.com/", num_results=5)
    if result.success:
        print(f"   Similar to: docs.anthropic.com")
        for r in result.data.get('results', [])[:3]:
            print(f"   - {r.get('title', 'N/A')[:50]} ({r.get('url', '')[:50]}...)")
        features_tested += 1

    # Feature 7: Get Contents from URLs
    print("\n[7/9] GET CONTENTS FROM URLs")
    print("-" * 40)
    result = await exa.execute("get_contents", urls=["https://react.dev/learn", "https://fastapi.tiangolo.com/tutorial/"])
    if result.success:
        contents = result.data.get('contents', [])
        print(f"   URLs processed: {len(contents)}")
        for c in contents[:2]:
            print(f"   - {c.get('url', 'N/A')[:50]}: {len(c.get('text', ''))} chars")
        features_tested += 1

    # Feature 8: Domain Filtering
    print("\n[8/9] DOMAIN FILTERING")
    print("-" * 40)
    result = await exa.execute("search",
        query="machine learning tutorial",
        include_domains=["arxiv.org", "pytorch.org", "tensorflow.org"],
        num_results=5
    )
    if result.success:
        print(f"   Filtered to: arxiv.org, pytorch.org, tensorflow.org")
        print(f"   Results: {len(result.data.get('results', []))}")
        features_tested += 1

    # Feature 9: Date Filtering
    print("\n[9/9] DATE FILTERING")
    print("-" * 40)
    result = await exa.execute("search",
        query="GPT-4 improvements",
        start_published_date="2024-01-01",
        num_results=5
    )
    if result.success:
        print(f"   Filter: Published after 2024-01-01")
        print(f"   Results: {len(result.data.get('results', []))}")
        features_tested += 1

    await exa.shutdown()
    print(f"\n   EXA FEATURES TESTED: {features_tested}/9")
    return features_tested


async def fetch_official_documentation():
    """Fetch official documentation from multiple SDK sources."""
    print("\n" + "=" * 80)
    print("OFFICIAL DOCUMENTATION FETCHING")
    print("=" * 80)

    from adapters.jina_adapter import JinaAdapter
    from adapters.context7_adapter import Context7Adapter

    jina = JinaAdapter()
    ctx7 = Context7Adapter()
    await jina.initialize({})
    await ctx7.initialize({})

    docs_fetched = []

    # Official documentation sources
    doc_sources = [
        ("Anthropic Claude", "https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching"),
        ("OpenAI", "https://platform.openai.com/docs/guides/function-calling"),
        ("LangChain", "https://python.langchain.com/docs/concepts/"),
        ("FastAPI", "https://fastapi.tiangolo.com/tutorial/dependencies/"),
        ("React", "https://react.dev/learn/state-a-components-memory"),
        ("PyTorch", "https://pytorch.org/docs/stable/nn.html"),
    ]

    for name, url in doc_sources:
        print(f"\n[{name}]")
        print("-" * 40)
        result = await jina.execute("read", url=url)
        if result.success:
            content = result.data.get('content', '')
            print(f"   URL: {url[:60]}...")
            print(f"   Content: {len(content)} characters")
            print(f"   Preview: {content[:150]}...")
            docs_fetched.append({
                "name": name,
                "url": url,
                "chars": len(content),
                "preview": content[:500]
            })

    await jina.shutdown()
    await ctx7.shutdown()

    print(f"\n   OFFICIAL DOCS FETCHED: {len(docs_fetched)}/{len(doc_sources)}")
    return docs_fetched


async def comprehensive_perplexity_research():
    """Deep research using Perplexity's different models."""
    print("\n" + "=" * 80)
    print("PERPLEXITY DEEP RESEARCH - ALL MODELS")
    print("=" * 80)

    from adapters.perplexity_adapter import PerplexityAdapter

    pplx = PerplexityAdapter()
    await pplx.initialize({})

    research_topics = []

    # Sonar Standard - Quick Research
    print("\n[SONAR STANDARD]")
    print("-" * 40)
    result = await pplx.execute("chat",
        query="What are the key differences between LangChain and LangGraph for building agents?"
    )
    if result.success:
        content = result.data.get('content', '')
        citations = result.data.get('citations', [])
        print(f"   Response: {len(content)} chars")
        print(f"   Citations: {len(citations)}")
        print(f"   Content Preview: {content[:300]}...")
        research_topics.append({
            "model": "sonar",
            "topic": "LangChain vs LangGraph",
            "chars": len(content),
            "citations": len(citations)
        })

    # Sonar Pro - Advanced Research
    print("\n[SONAR PRO]")
    print("-" * 40)
    result = await pplx.execute("pro",
        query="Explain the trade-offs between different distributed consensus algorithms: Raft, Paxos, PBFT, and Tendermint. Include real-world use cases."
    )
    if result.success:
        content = result.data.get('content', '')
        citations = result.data.get('citations', [])
        print(f"   Response: {len(content)} chars")
        print(f"   Citations: {len(citations)}")
        print(f"   Content Preview: {content[:300]}...")
        research_topics.append({
            "model": "sonar-pro",
            "topic": "Consensus Algorithms",
            "chars": len(content),
            "citations": len(citations)
        })

    await pplx.shutdown()

    print(f"\n   PERPLEXITY RESEARCH COMPLETED: {len(research_topics)} topics")
    return research_topics


async def ultimate_swarm_comprehensive():
    """Run comprehensive research using the full Ultimate Research Swarm."""
    print("\n" + "=" * 80)
    print("ULTIMATE RESEARCH SWARM - COMPREHENSIVE QUERIES")
    print("=" * 80)

    from core.ultimate_research_swarm import (
        UltimateResearchSwarm,
        ResearchDepth,
    )

    swarm = UltimateResearchSwarm()
    await swarm.initialize()

    research_queries = [
        ("Quick", "React hooks best practices useState useEffect", ResearchDepth.QUICK),
        ("Standard", "Python async await patterns for high-performance APIs", ResearchDepth.STANDARD),
        ("Comprehensive", "Multi-agent AI systems architecture patterns and frameworks", ResearchDepth.COMPREHENSIVE),
    ]

    results = []

    for name, query, depth in research_queries:
        print(f"\n[{name.upper()} RESEARCH]")
        print("-" * 40)
        print(f"   Query: {query[:50]}...")

        start = time.time()
        result = await swarm.research(query, depth=depth)
        elapsed = time.time() - start

        print(f"   Tools Used: {result.tools_used}")
        print(f"   Sources: {len(result.sources)}")
        print(f"   Confidence: {result.confidence:.0%}")
        print(f"   Latency: {elapsed:.2f}s")

        if result.key_findings:
            print(f"   Key Findings:")
            for finding in result.key_findings[:2]:
                print(f"     - {finding[:80]}...")

        results.append({
            "name": name,
            "query": query,
            "tools": list(result.tools_used),
            "sources": len(result.sources),
            "confidence": result.confidence,
            "latency": elapsed
        })

    # SDK Documentation Research
    print(f"\n[SDK DOCUMENTATION RESEARCH]")
    print("-" * 40)
    sdk_result = await swarm.research_sdk_docs(
        library="anthropic",
        query="prompt caching and function calling"
    )
    print(f"   Library: anthropic")
    print(f"   Tools: {sdk_result.tools_used}")
    print(f"   Sources: {len(sdk_result.sources)}")

    stats = swarm.get_stats()
    print(f"\n[SWARM STATISTICS]")
    print(f"   Total Queries: {stats.get('total_queries', 0)}")
    print(f"   Cache Hits: {stats.get('cache_hits', 0)}")
    print(f"   Average Latency: {stats.get('avg_latency_ms', 0):.0f}ms")

    await swarm.shutdown()

    return results


async def main():
    """Run all deep dive tests."""
    print("\n" + "=" * 80)
    print("UNLEASH DEEP DIVE - COMPREHENSIVE RESEARCH TEST")
    print("=" * 80)
    print(f"\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    total_start = time.time()

    # 1. Exa Deep Dive
    exa_features = await deep_dive_exa()

    # 2. Official Documentation Fetching
    docs = await fetch_official_documentation()

    # 3. Perplexity Deep Research
    perplexity_topics = await comprehensive_perplexity_research()

    # 4. Ultimate Swarm Comprehensive
    swarm_results = await ultimate_swarm_comprehensive()

    total_elapsed = time.time() - total_start

    # Final Summary
    print("\n" + "=" * 80)
    print("DEEP DIVE SUMMARY")
    print("=" * 80)
    print(f"\n   Exa Features Tested: {exa_features}/9")
    print(f"   Official Docs Fetched: {len(docs)}")
    print(f"   Perplexity Topics Researched: {len(perplexity_topics)}")
    print(f"   Swarm Queries Completed: {len(swarm_results)}")
    print(f"\n   Total Time: {total_elapsed:.1f}s")

    print("\n" + "=" * 80)
    print("RESEARCH SWARM FULLY UNLEASHED!")
    print("=" * 80)

    # Save results
    results = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "exa_features": exa_features,
        "docs_fetched": len(docs),
        "perplexity_topics": perplexity_topics,
        "swarm_results": swarm_results,
        "total_time_seconds": total_elapsed
    }

    with open("Z:/insider/AUTO CLAUDE/unleash/deep_dive_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n   Results saved to: deep_dive_results.json")


if __name__ == "__main__":
    asyncio.run(main())
