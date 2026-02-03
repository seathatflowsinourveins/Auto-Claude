"""
FULL UNLEASH - ALL FEATURES OF ALL RESEARCH TOOLS
==================================================
Testing EVERY feature of EVERY adapter with REAL API calls!

TOOLS & FEATURES:
- EXA: search, get_contents, find_similar, search_and_contents (4 ops)
- TAVILY: search, research, extract, qna, get_context (5 ops)
- JINA: read, search, embed, rerank (4 ops)
- PERPLEXITY: chat, pro, research (3 ops)
- CONTEXT7: resolve, query, search (3 ops)

TOTAL: 19 OPERATIONS TO TEST!
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

# Suppress verbose logging
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("core").setLevel(logging.WARNING)

print("=" * 80)
print("FULL UNLEASH - ALL FEATURES OF ALL RESEARCH TOOLS")
print("=" * 80)
print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")


async def unleash_exa():
    """UNLEASH ALL 4 EXA OPERATIONS."""
    print("\n" + "=" * 80)
    print("EXA NEURAL SEARCH - 4 OPERATIONS")
    print("=" * 80)

    from adapters.exa_adapter import ExaAdapter
    exa = ExaAdapter()
    await exa.initialize({})

    results = {}

    # 1. SEARCH - Neural semantic search
    print("\n[1/4] SEARCH - Neural Semantic Search")
    print("-" * 60)
    result = await exa.execute("search",
        query="advanced RAG retrieval augmented generation techniques 2024",
        num_results=5,
        type="neural"
    )
    if result.success:
        data = result.data
        print(f"   Results: {len(data.get('results', []))}")
        for r in data.get('results', [])[:3]:
            print(f"   - {r['title'][:55]}...")
            print(f"     {r['url'][:65]}...")
        results['search'] = {"success": True, "count": len(data.get('results', []))}
    else:
        print(f"   Error: {result.error}")
        results['search'] = {"success": False, "error": result.error}

    # 2. GET_CONTENTS - Fetch full content from URLs
    print("\n[2/4] GET_CONTENTS - Fetch Full Content from URLs")
    print("-" * 60)
    result = await exa.execute("get_contents",
        urls=[
            "https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching",
            "https://platform.openai.com/docs/guides/function-calling",
        ],
        text=True,
        highlights=True
    )
    if result.success:
        contents = result.data.get('contents', [])
        print(f"   URLs processed: {len(contents)}")
        for c in contents:
            text_len = len(c.get('text', ''))
            print(f"   - {c.get('url', 'N/A')[:50]}...")
            print(f"     Content: {text_len} chars")
            if text_len > 0:
                print(f"     Preview: {c.get('text', '')[:100]}...")
        results['get_contents'] = {"success": True, "urls": len(contents)}
    else:
        print(f"   Error: {result.error}")
        results['get_contents'] = {"success": False, "error": result.error}

    # 3. FIND_SIMILAR - Find similar websites
    print("\n[3/4] FIND_SIMILAR - Find Similar Websites")
    print("-" * 60)
    result = await exa.execute("find_similar",
        url="https://langchain.com/",
        num_results=5
    )
    if result.success:
        similar = result.data.get('results', [])
        print(f"   Similar to: langchain.com")
        print(f"   Found: {len(similar)} similar sites")
        for s in similar[:4]:
            print(f"   - {s.get('title', 'N/A')[:40]}")
            print(f"     {s.get('url', '')[:55]}...")
        results['find_similar'] = {"success": True, "count": len(similar)}
    else:
        print(f"   Error: {result.error}")
        results['find_similar'] = {"success": False, "error": result.error}

    # 4. SEARCH_AND_CONTENTS - Combined search + content fetch
    print("\n[4/4] SEARCH_AND_CONTENTS - Search + Fetch Content")
    print("-" * 60)
    result = await exa.execute("search_and_contents",
        query="LangGraph StateGraph tutorial examples",
        num_results=3,
        text=True
    )
    if result.success:
        data = result.data.get('results', [])
        print(f"   Query: LangGraph StateGraph tutorial")
        print(f"   Results with content: {len(data)}")
        for r in data[:2]:
            print(f"   - {r.get('title', 'N/A')[:50]}...")
            text = r.get('text', '')
            print(f"     Content: {len(text)} chars")
        results['search_and_contents'] = {"success": True, "count": len(data)}
    else:
        print(f"   Error: {result.error}")
        results['search_and_contents'] = {"success": False, "error": result.error}

    await exa.shutdown()
    success_count = sum(1 for r in results.values() if r.get('success'))
    print(f"\n   EXA OPERATIONS: {success_count}/4 PASSED")
    return results


async def unleash_tavily():
    """UNLEASH ALL 5 TAVILY OPERATIONS."""
    print("\n" + "=" * 80)
    print("TAVILY AI SEARCH - 5 OPERATIONS")
    print("=" * 80)

    from adapters.tavily_adapter import TavilyAdapter
    tavily = TavilyAdapter()
    await tavily.initialize({})

    results = {}

    # 1. SEARCH - AI-powered web search
    print("\n[1/5] SEARCH - AI-Powered Web Search")
    print("-" * 60)
    result = await tavily.execute("search",
        query="best practices for building AI agents with tool use",
        search_depth="basic",
        max_results=5
    )
    if result.success:
        data = result.data.get('results', [])
        print(f"   Query: AI agents with tool use")
        print(f"   Results: {len(data)}")
        for r in data[:3]:
            print(f"   - {r.get('title', 'N/A')[:55]}...")
        results['search'] = {"success": True, "count": len(data)}
    else:
        print(f"   Error: {result.error}")
        results['search'] = {"success": False, "error": result.error}

    # 2. RESEARCH - Deep multi-step research
    print("\n[2/5] RESEARCH - Deep Multi-Step Research")
    print("-" * 60)
    result = await tavily.execute("research",
        query="Compare vector databases: Pinecone vs Weaviate vs Qdrant vs Chroma",
        search_depth="advanced"
    )
    if result.success:
        data = result.data
        content = data.get('content', '') or data.get('answer', '')
        sources = data.get('sources', []) or data.get('results', [])
        print(f"   Topic: Vector database comparison")
        print(f"   Response: {len(content)} chars")
        print(f"   Sources: {len(sources)}")
        if content:
            print(f"   Preview: {content[:200]}...")
        results['research'] = {"success": True, "chars": len(content), "sources": len(sources)}
    else:
        print(f"   Error: {result.error}")
        results['research'] = {"success": False, "error": result.error}

    # 3. EXTRACT - Extract structured data from URLs
    print("\n[3/5] EXTRACT - Extract Data from URLs")
    print("-" * 60)
    result = await tavily.execute("extract",
        urls=["https://fastapi.tiangolo.com/", "https://react.dev/"]
    )
    if result.success:
        data = result.data.get('results', [])
        print(f"   URLs processed: {len(data)}")
        for r in data[:2]:
            content = r.get('raw_content', '') or r.get('content', '')
            print(f"   - {r.get('url', 'N/A')[:50]}...")
            print(f"     Extracted: {len(content)} chars")
        results['extract'] = {"success": True, "urls": len(data)}
    else:
        print(f"   Error: {result.error}")
        results['extract'] = {"success": False, "error": result.error}

    # 4. QNA - Question answering with citations
    print("\n[4/5] QNA - Question Answering with Citations")
    print("-" * 60)
    result = await tavily.execute("qna",
        query="What is the difference between LCEL and LangGraph in LangChain?"
    )
    if result.success:
        data = result.data
        answer = data.get('answer', '')
        print(f"   Question: LCEL vs LangGraph difference")
        print(f"   Answer: {len(answer)} chars")
        if answer:
            print(f"   Response: {answer[:250]}...")
        results['qna'] = {"success": True, "answer_length": len(answer)}
    else:
        print(f"   Error: {result.error}")
        results['qna'] = {"success": False, "error": result.error}

    # 5. CONTEXT - Get context for RAG
    print("\n[5/5] CONTEXT - Get Context for RAG")
    print("-" * 60)
    result = await tavily.execute("context",
        query="How to implement semantic caching for LLM applications"
    )
    if result.success:
        data = result.data
        context = data.get('context', '') or str(data)
        print(f"   Query: Semantic caching for LLM")
        print(f"   Context: {len(context)} chars")
        if context:
            print(f"   Preview: {context[:200]}...")
        results['get_context'] = {"success": True, "context_length": len(context)}
    else:
        print(f"   Error: {result.error}")
        results['get_context'] = {"success": False, "error": result.error}

    await tavily.shutdown()
    success_count = sum(1 for r in results.values() if r.get('success'))
    print(f"\n   TAVILY OPERATIONS: {success_count}/5 PASSED")
    return results


async def unleash_jina():
    """UNLEASH ALL 4 JINA OPERATIONS."""
    print("\n" + "=" * 80)
    print("JINA AI - 4 OPERATIONS")
    print("=" * 80)

    from adapters.jina_adapter import JinaAdapter
    jina = JinaAdapter()
    await jina.initialize({})

    results = {}

    # 1. READ - Convert URL to markdown
    print("\n[1/4] READ - Convert URL to Markdown")
    print("-" * 60)
    urls_to_read = [
        ("Anthropic Docs", "https://docs.anthropic.com/en/docs/build-with-claude/tool-use"),
        ("OpenAI Docs", "https://platform.openai.com/docs/guides/text-generation"),
        ("LangChain LCEL", "https://python.langchain.com/docs/concepts/lcel/"),
    ]
    read_results = []
    for name, url in urls_to_read:
        result = await jina.execute("read", url=url)
        if result.success:
            content = result.data.get('content', '')
            print(f"   [{name}] {len(content)} chars")
            print(f"     Preview: {content[:80]}...")
            read_results.append({"name": name, "chars": len(content)})
    results['read'] = {"success": len(read_results) > 0, "urls_read": len(read_results)}

    # 2. SEARCH - Semantic web search
    print("\n[2/4] SEARCH - Semantic Web Search")
    print("-" * 60)
    result = await jina.execute("search",
        query="how to build multi-agent systems with LangGraph"
    )
    if result.success:
        data = result.data
        content = data.get('content', '') or data.get('results', '')
        print(f"   Query: Multi-agent systems with LangGraph")
        print(f"   Response: {len(str(content))} chars")
        if content:
            print(f"   Preview: {str(content)[:200]}...")
        results['search'] = {"success": True, "content_length": len(str(content))}
    else:
        print(f"   Error: {result.error}")
        results['search'] = {"success": False, "error": result.error}

    # 3. EMBED - Generate embeddings
    print("\n[3/4] EMBED - Generate Embeddings")
    print("-" * 60)
    texts_to_embed = [
        "LangGraph is a framework for building stateful multi-agent applications",
        "Vector databases store embeddings for semantic similarity search",
        "RAG combines retrieval with generation for grounded responses",
    ]
    result = await jina.execute("embed", texts=texts_to_embed)
    if result.success:
        embeddings = result.data.get('embeddings', [])
        print(f"   Texts embedded: {len(texts_to_embed)}")
        print(f"   Embeddings returned: {len(embeddings)}")
        if embeddings:
            print(f"   Embedding dimension: {len(embeddings[0])}")
            print(f"   First embedding (first 5 dims): {embeddings[0][:5]}")
        results['embed'] = {"success": True, "count": len(embeddings)}
    else:
        print(f"   Error: {result.error}")
        results['embed'] = {"success": False, "error": result.error}

    # 4. RERANK - Rerank documents by relevance
    print("\n[4/4] RERANK - Rerank Documents by Relevance")
    print("-" * 60)
    documents = [
        "LangGraph provides a way to build multi-agent systems with state management",
        "Python is a popular programming language for machine learning",
        "StateGraph is the core abstraction in LangGraph for defining agent workflows",
        "JavaScript is commonly used for web development",
        "LangGraph supports both synchronous and asynchronous execution of agents",
    ]
    result = await jina.execute("rerank",
        query="How to build multi-agent systems with LangGraph StateGraph",
        documents=documents,
        top_n=3
    )
    if result.success:
        reranked = result.data.get('results', [])
        print(f"   Query: Multi-agent systems with LangGraph")
        print(f"   Documents: {len(documents)}")
        print(f"   Top reranked: {len(reranked)}")
        for i, r in enumerate(reranked[:3]):
            score = r.get('relevance_score', r.get('score', 0))
            doc = r.get('document', {})
            text = doc.get('text', str(doc))[:60] if isinstance(doc, dict) else str(doc)[:60]
            print(f"   {i+1}. Score: {score:.4f} - {text}...")
        results['rerank'] = {"success": True, "reranked": len(reranked)}
    else:
        print(f"   Error: {result.error}")
        results['rerank'] = {"success": False, "error": result.error}

    await jina.shutdown()
    success_count = sum(1 for r in results.values() if r.get('success'))
    print(f"\n   JINA OPERATIONS: {success_count}/4 PASSED")
    return results


async def unleash_perplexity():
    """UNLEASH ALL 3 PERPLEXITY OPERATIONS."""
    print("\n" + "=" * 80)
    print("PERPLEXITY SONAR - 3 OPERATIONS")
    print("=" * 80)

    from adapters.perplexity_adapter import PerplexityAdapter
    pplx = PerplexityAdapter()
    await pplx.initialize({})

    results = {}

    # 1. CHAT - Standard Sonar chat with web grounding
    print("\n[1/3] CHAT - Sonar Chat with Web Grounding")
    print("-" * 60)
    result = await pplx.execute("chat",
        query="What are the key features of Claude 3.5 Sonnet and how does it compare to GPT-4?",
        return_citations=True
    )
    if result.success:
        content = result.data.get('content', '')
        citations = result.data.get('citations', [])
        print(f"   Model: sonar")
        print(f"   Response: {len(content)} chars")
        print(f"   Citations: {len(citations)}")
        if content:
            print(f"   Preview: {content[:200]}...")
        if citations:
            print(f"   Sources:")
            for c in citations[:3]:
                print(f"     - {c[:60]}...")
        results['chat'] = {"success": True, "chars": len(content), "citations": len(citations)}
    else:
        print(f"   Error: {result.error}")
        results['chat'] = {"success": False, "error": result.error}

    # 2. PRO - Sonar Pro for complex queries
    print("\n[2/3] PRO - Sonar Pro for Complex Queries")
    print("-" * 60)
    result = await pplx.execute("pro",
        query="Explain the architecture of transformer-based language models, including attention mechanisms, positional encoding, and how they enable in-context learning. Include recent improvements like Flash Attention and Ring Attention."
    )
    if result.success:
        content = result.data.get('content', '')
        citations = result.data.get('citations', [])
        print(f"   Model: sonar-pro")
        print(f"   Response: {len(content)} chars")
        print(f"   Citations: {len(citations)}")
        if content:
            print(f"   Preview: {content[:200]}...")
        results['pro'] = {"success": True, "chars": len(content), "citations": len(citations)}
    else:
        print(f"   Error: {result.error}")
        results['pro'] = {"success": False, "error": result.error}

    # 3. RESEARCH - Deep research with sonar-deep-research
    print("\n[3/3] RESEARCH - Deep Research Mode")
    print("-" * 60)
    result = await pplx.execute("research",
        query="What are the latest developments in AI agent frameworks? Compare AutoGPT, BabyAGI, CrewAI, LangGraph, and Letta for building autonomous agents."
    )
    if result.success:
        content = result.data.get('content', '')
        citations = result.data.get('citations', [])
        print(f"   Model: sonar-deep-research")
        print(f"   Response: {len(content)} chars")
        print(f"   Citations: {len(citations)}")
        if content:
            print(f"   Preview: {content[:250]}...")
        results['research'] = {"success": True, "chars": len(content), "citations": len(citations)}
    else:
        print(f"   Error: {result.error}")
        results['research'] = {"success": False, "error": result.error}

    await pplx.shutdown()
    success_count = sum(1 for r in results.values() if r.get('success'))
    print(f"\n   PERPLEXITY OPERATIONS: {success_count}/3 PASSED")
    return results


async def unleash_context7():
    """UNLEASH ALL 3 CONTEXT7 OPERATIONS."""
    print("\n" + "=" * 80)
    print("CONTEXT7 SDK DOCS - 3 OPERATIONS")
    print("=" * 80)

    from adapters.context7_adapter import Context7Adapter
    ctx7 = Context7Adapter()
    await ctx7.initialize({})

    results = {}

    # 1. RESOLVE - Resolve library name to ID
    print("\n[1/3] RESOLVE - Resolve Library Names")
    print("-" * 60)
    libraries = ["langchain", "react", "fastapi", "pytorch", "anthropic"]
    resolved = []
    for lib in libraries:
        result = await ctx7.execute("resolve", library_name=lib)
        if result.success:
            lib_id = result.data.get('library_id', lib)
            print(f"   {lib} -> {lib_id}")
            resolved.append(lib)
    results['resolve'] = {"success": len(resolved) > 0, "resolved": len(resolved)}

    # 2. QUERY - Query library documentation
    print("\n[2/3] QUERY - Query Library Documentation")
    print("-" * 60)
    queries = [
        ("langchain", "LCEL chains"),
        ("langgraph", "StateGraph nodes edges"),
        ("react", "useState useEffect hooks"),
        ("fastapi", "dependency injection"),
    ]
    queried = []
    for lib, query in queries:
        result = await ctx7.execute("query", library_id=lib, query=query)
        if result.success:
            content = result.data.get('content', '')
            print(f"   [{lib}] {query}")
            print(f"     Content: {len(content)} chars")
            if content:
                print(f"     Preview: {content[:80]}...")
            queried.append({"lib": lib, "chars": len(content)})
    results['query'] = {"success": len(queried) > 0, "queries": len(queried)}

    # 3. SEARCH - Search across all libraries
    print("\n[3/3] SEARCH - Search Across Libraries")
    print("-" * 60)
    result = await ctx7.execute("search", query="async state management patterns")
    if result.success:
        data = result.data
        search_results = data.get('results', [])
        content = data.get('content', '')
        print(f"   Query: async state management patterns")
        print(f"   Results: {len(search_results)}")
        print(f"   Content: {len(content)} chars")
        if search_results:
            for r in search_results[:3]:
                print(f"   - {r.get('title', 'N/A')[:50]}...")
        results['search'] = {"success": True, "results": len(search_results), "content_chars": len(content)}
    else:
        print(f"   Error: {result.error}")
        results['search'] = {"success": False, "error": result.error}

    await ctx7.shutdown()
    success_count = sum(1 for r in results.values() if r.get('success'))
    print(f"\n   CONTEXT7 OPERATIONS: {success_count}/3 PASSED")
    return results


async def unleash_full_swarm():
    """UNLEASH THE FULL ULTIMATE RESEARCH SWARM."""
    print("\n" + "=" * 80)
    print("ULTIMATE RESEARCH SWARM - FULL INTEGRATION")
    print("=" * 80)

    from core.ultimate_research_swarm import (
        UltimateResearchSwarm,
        ResearchDepth,
    )

    swarm = UltimateResearchSwarm()
    await swarm.initialize()

    results = []

    # Test all depth levels
    test_queries = [
        ("QUICK", "React hooks useState useEffect patterns", ResearchDepth.QUICK),
        ("STANDARD", "Python asyncio patterns for high-performance web APIs", ResearchDepth.STANDARD),
        ("COMPREHENSIVE", "Building production-ready LLM applications with RAG and agents", ResearchDepth.COMPREHENSIVE),
        ("DEEP", "Distributed consensus algorithms: Raft vs Paxos vs PBFT implementation trade-offs", ResearchDepth.DEEP),
    ]

    for name, query, depth in test_queries:
        print(f"\n[{name} DEPTH]")
        print("-" * 60)
        print(f"   Query: {query[:50]}...")

        start = time.time()
        result = await swarm.research(query, depth=depth)
        elapsed = time.time() - start

        print(f"   Tools: {result.tools_used}")
        print(f"   Sources: {len(result.sources)}")
        print(f"   Confidence: {result.confidence:.0%}")
        print(f"   Latency: {elapsed:.2f}s")

        if result.key_findings:
            print(f"   Key Findings:")
            for finding in result.key_findings[:2]:
                print(f"     - {finding[:70]}...")

        results.append({
            "depth": name,
            "tools": list(result.tools_used),
            "sources": len(result.sources),
            "confidence": result.confidence,
            "latency": elapsed
        })

    # SDK Documentation Research
    print(f"\n[SDK DOCUMENTATION]")
    print("-" * 60)
    sdk_libs = ["langchain", "anthropic", "openai"]
    for lib in sdk_libs:
        result = await swarm.research_sdk_docs(
            library=lib,
            query="API usage patterns"
        )
        print(f"   [{lib}] Tools: {result.tools_used}, Sources: {len(result.sources)}")

    stats = swarm.get_stats()
    print(f"\n[SWARM STATS]")
    print(f"   Total Queries: {stats.get('total_queries', 0)}")
    print(f"   Avg Latency: {stats.get('avg_latency_ms', 0):.0f}ms")

    await swarm.shutdown()
    return results


async def main():
    """RUN FULL UNLEASH TEST."""
    print("\n" + "=" * 80)
    print("COMMENCING FULL UNLEASH - ALL 19 OPERATIONS")
    print("=" * 80)

    total_start = time.time()
    all_results = {}

    # 1. Unleash Exa (4 ops)
    all_results['exa'] = await unleash_exa()

    # 2. Unleash Tavily (5 ops)
    all_results['tavily'] = await unleash_tavily()

    # 3. Unleash Jina (4 ops)
    all_results['jina'] = await unleash_jina()

    # 4. Unleash Perplexity (3 ops)
    all_results['perplexity'] = await unleash_perplexity()

    # 5. Unleash Context7 (3 ops)
    all_results['context7'] = await unleash_context7()

    # 6. Unleash Full Swarm
    all_results['swarm'] = await unleash_full_swarm()

    total_elapsed = time.time() - total_start

    # FINAL SUMMARY
    print("\n" + "=" * 80)
    print("FULL UNLEASH SUMMARY")
    print("=" * 80)

    total_ops = 0
    passed_ops = 0

    for adapter, results in all_results.items():
        if adapter == 'swarm':
            continue
        ops = len(results)
        passed = sum(1 for r in results.values() if r.get('success'))
        total_ops += ops
        passed_ops += passed
        status = "PASS" if passed == ops else f"{passed}/{ops}"
        print(f"   {adapter.upper():15} {status}")

    print(f"\n   TOTAL OPERATIONS: {passed_ops}/{total_ops} PASSED")
    print(f"   TOTAL TIME: {total_elapsed:.1f}s")

    print("\n" + "=" * 80)
    print("RESEARCH SWARM FULLY UNLEASHED!")
    print("=" * 80)

    # Save results
    with open("Z:/insider/AUTO CLAUDE/unleash/full_unleash_results.json", "w") as f:
        json.dump({
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_operations": total_ops,
            "passed_operations": passed_ops,
            "total_time_seconds": total_elapsed,
            "results": all_results
        }, f, indent=2, default=str)

    print(f"\n   Results saved to: full_unleash_results.json")


if __name__ == "__main__":
    asyncio.run(main())
