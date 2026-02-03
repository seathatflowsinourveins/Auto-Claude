"""
ULTIMATE UNLEASH - ALL ADVANCED FEATURES
=========================================
Testing EVERY advanced feature of ALL research tools!

EXPANDED FEATURES:
- EXA: search, get_contents, find_similar, search_and_contents, answer, research (6 ops)
- TAVILY: search, research, extract, qna, context, map, crawl (7 ops)
- JINA: read, search, embed, rerank, segment, ground, deepsearch, classify (8 ops)
- PERPLEXITY: chat, pro, research, reasoning, search (5 ops)
- CONTEXT7: resolve, query, search (3 ops)

TOTAL: 29 OPERATIONS TO TEST!
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
logging.getLogger("numexpr").setLevel(logging.WARNING)

print("=" * 80)
print("ULTIMATE UNLEASH - ALL ADVANCED FEATURES")
print("=" * 80)
print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")


async def unleash_exa_advanced():
    """UNLEASH ALL 6 EXA OPERATIONS including new Answer & Research."""
    print("\n" + "=" * 80)
    print("EXA NEURAL SEARCH - 6 ADVANCED OPERATIONS")
    print("=" * 80)

    from adapters.exa_adapter import ExaAdapter
    exa = ExaAdapter()
    await exa.initialize({})

    results = {}

    # 1. SEARCH with advanced options
    print("\n[1/6] SEARCH - Neural with Date/Domain Filters")
    print("-" * 60)
    result = await exa.execute("search",
        query="transformer architecture improvements 2024",
        num_results=5,
        type="neural",
        start_published_date="2024-01-01",
        include_domains=["arxiv.org", "openai.com", "anthropic.com"]
    )
    if result.success:
        print(f"   Results: {len(result.data.get('results', []))}")
        for r in result.data.get('results', [])[:2]:
            print(f"   - {r['title'][:50]}...")
        results['search'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['search'] = {"success": False}

    # 2. GET_CONTENTS with highlights
    print("\n[2/6] GET_CONTENTS - With Highlights & Summary")
    print("-" * 60)
    result = await exa.execute("get_contents",
        urls=["https://docs.anthropic.com/en/docs/welcome"],
        text=True,
        highlights=True
    )
    if result.success:
        contents = result.data.get('contents', [])
        print(f"   URLs: {len(contents)}")
        if contents:
            print(f"   Content: {len(contents[0].get('text', ''))} chars")
        results['get_contents'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['get_contents'] = {"success": False}

    # 3. FIND_SIMILAR with exclude source
    print("\n[3/6] FIND_SIMILAR - Exclude Source Domain")
    print("-" * 60)
    result = await exa.execute("find_similar",
        url="https://openai.com/research",
        num_results=5,
        exclude_source_domain=True
    )
    if result.success:
        print(f"   Similar sites: {len(result.data.get('results', []))}")
        results['find_similar'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['find_similar'] = {"success": False}

    # 4. SEARCH_AND_CONTENTS combined
    print("\n[4/6] SEARCH_AND_CONTENTS - Combined Operation")
    print("-" * 60)
    result = await exa.execute("search_and_contents",
        query="RAG retrieval augmented generation tutorial",
        num_results=3,
        text=True
    )
    if result.success:
        print(f"   Results with content: {len(result.data.get('results', []))}")
        results['search_and_contents'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['search_and_contents'] = {"success": False}

    # 5. ANSWER - Question answering (NEW!)
    print("\n[5/6] ANSWER - AI Question Answering")
    print("-" * 60)
    result = await exa.execute("answer",
        query="What is the difference between RAG and fine-tuning for LLMs?"
    )
    if result.success:
        answer = result.data.get('answer', '')
        print(f"   Answer: {len(answer)} chars")
        if answer:
            print(f"   Preview: {answer[:150]}...")
        results['answer'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['answer'] = {"success": False}

    # 6. RESEARCH - Structured research (NEW!)
    print("\n[6/6] RESEARCH - Complex Research with Schema")
    print("-" * 60)
    result = await exa.execute("research",
        instructions="Compare the top 3 vector databases for AI applications"
    )
    if result.success:
        print(f"   Research completed: {len(str(result.data))} chars")
        results['research'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['research'] = {"success": False}

    await exa.shutdown()
    success = sum(1 for r in results.values() if r.get('success'))
    print(f"\n   EXA ADVANCED: {success}/6 PASSED")
    return results


async def unleash_tavily_advanced():
    """UNLEASH ALL 7 TAVILY OPERATIONS including Map & Crawl."""
    print("\n" + "=" * 80)
    print("TAVILY AI SEARCH - 7 ADVANCED OPERATIONS")
    print("=" * 80)

    from adapters.tavily_adapter import TavilyAdapter
    tavily = TavilyAdapter()
    await tavily.initialize({})

    results = {}

    # 1. SEARCH with advanced depth
    print("\n[1/5] SEARCH - Advanced Depth")
    print("-" * 60)
    result = await tavily.execute("search",
        query="multi-agent AI frameworks comparison 2024",
        search_depth="advanced",
        max_results=10
    )
    if result.success:
        print(f"   Results: {len(result.data.get('results', []))}")
        results['search'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['search'] = {"success": False}

    # 2. RESEARCH deep mode
    print("\n[2/5] RESEARCH - Deep Multi-Step")
    print("-" * 60)
    result = await tavily.execute("research",
        query="Best practices for building production LLM applications with RAG",
        search_depth="advanced"
    )
    if result.success:
        sources = result.data.get('sources', result.data.get('results', []))
        print(f"   Sources: {len(sources)}")
        results['research'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['research'] = {"success": False}

    # 3. EXTRACT from URLs
    print("\n[3/5] EXTRACT - Structured Data")
    print("-" * 60)
    result = await tavily.execute("extract",
        urls=["https://langchain.com/", "https://www.anthropic.com/"]
    )
    if result.success:
        extracted = result.data.get('results', [])
        print(f"   Extracted: {len(extracted)} URLs")
        results['extract'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['extract'] = {"success": False}

    # 4. QNA question answering
    print("\n[4/5] QNA - Direct Answer")
    print("-" * 60)
    result = await tavily.execute("qna",
        query="What is LangGraph and how is it different from LangChain?"
    )
    if result.success:
        answer = result.data.get('answer', '')
        print(f"   Answer: {len(answer)} chars")
        if answer:
            print(f"   Response: {answer[:120]}...")
        results['qna'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['qna'] = {"success": False}

    # 5. CONTEXT for RAG
    print("\n[5/7] CONTEXT - RAG Context")
    print("-" * 60)
    result = await tavily.execute("context",
        query="How to implement semantic caching for LLM applications"
    )
    if result.success:
        context = result.data.get('context', str(result.data))
        print(f"   Context: {len(context)} chars")
        results['context'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['context'] = {"success": False}

    # 6. MAP - Website structure (NEW!)
    print("\n[6/7] MAP - Website Structure")
    print("-" * 60)
    result = await tavily.execute("map",
        url="https://docs.anthropic.com",
        max_depth=1,
        limit=10
    )
    if result.success:
        pages = result.data.get('results', [])
        print(f"   Pages discovered: {len(pages)}")
        results['map'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['map'] = {"success": False}

    # 7. CRAWL - Website content extraction (NEW!)
    print("\n[7/7] CRAWL - Website Content Extraction")
    print("-" * 60)
    result = await tavily.execute("crawl",
        url="https://docs.anthropic.com/en/docs/welcome",
        max_depth=1,
        limit=3,
        format="markdown"
    )
    if result.success:
        pages = result.data.get('results', [])
        print(f"   Pages crawled: {len(pages)}")
        results['crawl'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['crawl'] = {"success": False}

    await tavily.shutdown()
    success = sum(1 for r in results.values() if r.get('success'))
    print(f"\n   TAVILY ADVANCED: {success}/7 PASSED")
    return results


async def unleash_jina_advanced():
    """UNLEASH ALL 8 JINA OPERATIONS including DeepSearch & Classify."""
    print("\n" + "=" * 80)
    print("JINA AI - 8 ADVANCED OPERATIONS")
    print("=" * 80)

    from adapters.jina_adapter import JinaAdapter
    jina = JinaAdapter()
    await jina.initialize({})

    results = {}

    # 1. READ with image descriptions
    print("\n[1/6] READ - With Image Descriptions")
    print("-" * 60)
    result = await jina.execute("read",
        url="https://react.dev/learn",
        with_images=True
    )
    if result.success:
        content = result.data.get('content', '')
        print(f"   Content: {len(content)} chars")
        results['read'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['read'] = {"success": False}

    # 2. SEARCH with site filter (NEW!)
    print("\n[2/6] SEARCH - Site-Filtered")
    print("-" * 60)
    result = await jina.execute("search",
        query="StateGraph tutorial",
        site="langchain.com"
    )
    if result.success:
        print(f"   Content: {len(result.data.get('content', ''))} chars")
        results['search'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['search'] = {"success": False}

    # 3. EMBED multiple texts
    print("\n[3/6] EMBED - Multi-Text Embeddings")
    print("-" * 60)
    result = await jina.execute("embed",
        texts=[
            "LangGraph provides stateful multi-agent orchestration",
            "Vector databases enable semantic similarity search",
            "RAG combines retrieval with generation for grounding",
            "Prompt engineering improves LLM output quality",
        ],
        dimensions=1024
    )
    if result.success:
        embeddings = result.data.get('embeddings', [])
        print(f"   Embeddings: {len(embeddings)}")
        if embeddings:
            print(f"   Dimensions: {len(embeddings[0])}")
        results['embed'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['embed'] = {"success": False}

    # 4. RERANK documents
    print("\n[4/6] RERANK - Relevance Scoring")
    print("-" * 60)
    docs = [
        "LangGraph is a framework for building multi-agent systems",
        "Python is a popular programming language",
        "StateGraph defines nodes and edges for agent workflows",
        "Machine learning models require training data",
        "LangGraph supports async execution of agents",
    ]
    result = await jina.execute("rerank",
        query="How to build multi-agent systems with LangGraph",
        documents=docs,
        top_n=3
    )
    if result.success:
        reranked = result.data.get('results', [])
        print(f"   Reranked: {len(reranked)} docs")
        for r in reranked[:2]:
            score = r.get('relevance_score', r.get('score', 0))
            print(f"   - Score {score:.3f}")
        results['rerank'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['rerank'] = {"success": False}

    # 5. SEGMENT long content (NEW!)
    print("\n[5/6] SEGMENT - Content Chunking")
    print("-" * 60)
    long_content = """
    LangGraph is a powerful framework for building stateful, multi-agent applications.
    It extends LangChain with graph-based orchestration capabilities.
    The StateGraph class is the core abstraction that defines nodes and edges.
    Each node represents a function that processes state.
    Edges define transitions between nodes based on conditions.
    This enables complex workflows with loops, branches, and parallel execution.
    """ * 5
    result = await jina.execute("segment",
        content=long_content,
        max_chunk_length=500
    )
    if result.success:
        chunks = result.data.get('chunks', [])
        print(f"   Chunks: {len(chunks)}")
        results['segment'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['segment'] = {"success": False}

    # 6. GROUND fact-checking
    print("\n[6/8] GROUND - Fact Verification")
    print("-" * 60)
    result = await jina.execute("ground",
        statement="LangGraph was released by LangChain as a framework for multi-agent systems"
    )
    if result.success:
        evidence = result.data.get('evidence', '')
        print(f"   Evidence: {len(evidence)} chars")
        results['ground'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['ground'] = {"success": False}

    # 7. DEEPSEARCH - Deep agentic search (NEW!)
    print("\n[7/8] DEEPSEARCH - Agentic Deep Search")
    print("-" * 60)
    result = await jina.execute("deepsearch",
        query="What are the key architectural differences between LangGraph and AutoGen for multi-agent systems?",
        budget_tokens=4000,
        max_attempts=5
    )
    if result.success:
        answer = result.data.get('answer', result.data.get('content', ''))
        sources = result.data.get('sources', [])
        print(f"   Answer: {len(answer)} chars")
        print(f"   Sources: {len(sources)}")
        if answer:
            print(f"   Preview: {answer[:100]}...")
        results['deepsearch'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['deepsearch'] = {"success": False}

    # 8. CLASSIFY - Zero-shot classification (NEW!)
    print("\n[8/8] CLASSIFY - Text Classification")
    print("-" * 60)
    result = await jina.execute("classify",
        text="LangGraph provides a framework for building stateful multi-agent applications with graph-based orchestration",
        labels=["AI/ML Framework", "Database", "Web Development", "DevOps", "Security"]
    )
    if result.success:
        label = result.data.get('label', '')
        score = result.data.get('score', 0)
        print(f"   Label: {label}")
        print(f"   Score: {score:.3f}")
        results['classify'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['classify'] = {"success": False}

    await jina.shutdown()
    success = sum(1 for r in results.values() if r.get('success'))
    print(f"\n   JINA ADVANCED: {success}/8 PASSED")
    return results


async def unleash_perplexity_advanced():
    """UNLEASH ALL 5 PERPLEXITY OPERATIONS including Reasoning & Search."""
    print("\n" + "=" * 80)
    print("PERPLEXITY SONAR - 5 ADVANCED OPERATIONS")
    print("=" * 80)

    from adapters.perplexity_adapter import PerplexityAdapter
    pplx = PerplexityAdapter()
    await pplx.initialize({})

    results = {}

    # 1. CHAT with citations
    print("\n[1/3] CHAT - Web-Grounded with Citations")
    print("-" * 60)
    result = await pplx.execute("chat",
        query="What are the latest advancements in AI agents and autonomous systems in 2024?",
        return_citations=True
    )
    if result.success:
        content = result.data.get('content', '')
        citations = result.data.get('citations', [])
        print(f"   Response: {len(content)} chars")
        print(f"   Citations: {len(citations)}")
        if content:
            print(f"   Preview: {content[:120]}...")
        results['chat'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['chat'] = {"success": False}

    # 2. PRO advanced reasoning
    print("\n[2/3] PRO - Advanced Reasoning")
    print("-" * 60)
    result = await pplx.execute("pro",
        query="Compare and contrast the architectures of GPT-4, Claude 3, and Gemini for enterprise AI applications"
    )
    if result.success:
        content = result.data.get('content', '')
        citations = result.data.get('citations', [])
        print(f"   Response: {len(content)} chars")
        print(f"   Citations: {len(citations)}")
        results['pro'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['pro'] = {"success": False}

    # 3. RESEARCH deep analysis
    print("\n[3/5] RESEARCH - Deep Research")
    print("-" * 60)
    result = await pplx.execute("research",
        query="Analyze the trade-offs between different RAG architectures: naive RAG, advanced RAG with reranking, and agentic RAG"
    )
    if result.success:
        content = result.data.get('content', '')
        citations = result.data.get('citations', [])
        print(f"   Response: {len(content)} chars")
        print(f"   Citations: {len(citations)}")
        if content:
            print(f"   Preview: {content[:150]}...")
        results['research'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['research'] = {"success": False}

    # 4. REASONING - Chain of Thought (NEW!)
    print("\n[4/5] REASONING - Chain of Thought")
    print("-" * 60)
    result = await pplx.execute("reasoning",
        query="What are the key differences between Raft and Paxos consensus algorithms, and when would you use each?",
        reasoning_effort="medium"
    )
    if result.success:
        content = result.data.get('content', '')
        reasoning_steps = result.data.get('reasoning_steps', [])
        citations = result.data.get('citations', [])
        print(f"   Response: {len(content)} chars")
        print(f"   Reasoning steps: {len(reasoning_steps)}")
        print(f"   Citations: {len(citations)}")
        results['reasoning'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['reasoning'] = {"success": False}

    # 5. SEARCH - Raw web search (NEW!)
    print("\n[5/5] SEARCH - Raw Web Search")
    print("-" * 60)
    result = await pplx.execute("search",
        query="LangGraph multi-agent orchestration patterns 2024",
        max_results=5
    )
    if result.success:
        search_results = result.data.get('results', [])
        print(f"   Results: {len(search_results)}")
        results['search'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['search'] = {"success": False}

    await pplx.shutdown()
    success = sum(1 for r in results.values() if r.get('success'))
    print(f"\n   PERPLEXITY ADVANCED: {success}/5 PASSED")
    return results


async def unleash_context7_advanced():
    """UNLEASH ALL 3 CONTEXT7 OPERATIONS."""
    print("\n" + "=" * 80)
    print("CONTEXT7 SDK DOCS - 3 ADVANCED OPERATIONS")
    print("=" * 80)

    from adapters.context7_adapter import Context7Adapter
    ctx7 = Context7Adapter()
    await ctx7.initialize({})

    results = {}

    # 1. RESOLVE multiple libraries
    print("\n[1/3] RESOLVE - Multiple Libraries")
    print("-" * 60)
    libs = ["langchain", "langgraph", "react", "fastapi", "anthropic", "openai"]
    resolved = 0
    for lib in libs:
        result = await ctx7.execute("resolve", library_name=lib)
        if result.success:
            resolved += 1
            print(f"   {lib} -> {result.data.get('library_id')}")
    results['resolve'] = {"success": resolved > 0, "count": resolved}

    # 2. QUERY documentation
    print("\n[2/3] QUERY - Documentation Queries")
    print("-" * 60)
    queries = [
        ("langchain", "LCEL chains and runnables"),
        ("langgraph", "StateGraph checkpointing"),
        ("anthropic", "tool use function calling"),
    ]
    queried = 0
    for lib, query in queries:
        result = await ctx7.execute("query", library_id=lib, query=query)
        if result.success:
            queried += 1
            content = result.data.get('content', '')
            print(f"   [{lib}] {len(content)} chars")
    results['query'] = {"success": queried > 0, "count": queried}

    # 3. SEARCH across libraries
    print("\n[3/3] SEARCH - Cross-Library Search")
    print("-" * 60)
    result = await ctx7.execute("search", query="async state management patterns")
    if result.success:
        search_results = result.data.get('results', [])
        content = result.data.get('content', '')
        print(f"   Results: {len(search_results)}")
        print(f"   Content: {len(content)} chars")
        results['search'] = {"success": True}
    else:
        print(f"   Error: {result.error}")
        results['search'] = {"success": False}

    await ctx7.shutdown()
    success = sum(1 for r in results.values() if r.get('success'))
    print(f"\n   CONTEXT7 ADVANCED: {success}/3 PASSED")
    return results


async def main():
    """RUN ULTIMATE UNLEASH TEST."""
    print("\n" + "=" * 80)
    print("COMMENCING ULTIMATE UNLEASH - 29 ADVANCED OPERATIONS")
    print("=" * 80)

    total_start = time.time()
    all_results = {}

    # Test all adapters with advanced features
    all_results['exa'] = await unleash_exa_advanced()
    all_results['tavily'] = await unleash_tavily_advanced()
    all_results['jina'] = await unleash_jina_advanced()
    all_results['perplexity'] = await unleash_perplexity_advanced()
    all_results['context7'] = await unleash_context7_advanced()

    total_elapsed = time.time() - total_start

    # FINAL SUMMARY
    print("\n" + "=" * 80)
    print("ULTIMATE UNLEASH SUMMARY")
    print("=" * 80)

    total_ops = 0
    passed_ops = 0
    op_counts = {'exa': 6, 'tavily': 7, 'jina': 8, 'perplexity': 5, 'context7': 3}

    for adapter, results in all_results.items():
        ops = op_counts.get(adapter, len(results))
        passed = sum(1 for r in results.values() if r.get('success'))
        total_ops += ops
        passed_ops += passed
        status = "PASS" if passed == ops else f"{passed}/{ops}"
        print(f"   {adapter.upper():15} {status}")

    print(f"\n   TOTAL OPERATIONS: {passed_ops}/{total_ops} PASSED")
    print(f"   SUCCESS RATE: {passed_ops/total_ops*100:.1f}%")
    print(f"   TOTAL TIME: {total_elapsed:.1f}s")

    print("\n" + "=" * 80)
    print("RESEARCH SWARM ULTIMATE UNLEASHED!")
    print("=" * 80)

    # Save results
    with open("Z:/insider/AUTO CLAUDE/unleash/ultimate_unleash_results.json", "w") as f:
        json.dump({
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_operations": total_ops,
            "passed_operations": passed_ops,
            "success_rate": passed_ops/total_ops*100,
            "total_time_seconds": total_elapsed,
            "results": all_results
        }, f, indent=2, default=str)

    print(f"\n   Results saved to: ultimate_unleash_results.json")


if __name__ == "__main__":
    asyncio.run(main())
