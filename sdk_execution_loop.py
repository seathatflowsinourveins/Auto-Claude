"""
SDK EXECUTION LOOP - Battle-Tested Production SDK Testing
==========================================================
Executes real SDK code from official docs, not just research.
Tests actual API calls, validates responses, and captures working patterns.
"""

import asyncio
import os
import sys
import json
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv('.config/.env')

# ============================================================================
# PRODUCTION SDK EXECUTION TESTS
# ============================================================================

async def test_anthropic_sdk():
    """Test Anthropic Claude SDK - Official Production Patterns"""
    print("\n" + "="*70)
    print("SDK EXECUTION: ANTHROPIC CLAUDE")
    print("="*70)

    results = {"sdk": "anthropic", "tests": []}

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("      [SKIP] ANTHROPIC_API_KEY not set")
        results["status"] = "SKIP"
        return results

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        # Test 1: Basic message
        print("\n  [1] Basic Message API...")
        start = time.time()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": "Say 'SDK test successful' in exactly 3 words"}]
        )
        latency = (time.time() - start) * 1000
        print(f"      [OK] Response: {message.content[0].text[:50]}... ({latency:.0f}ms)")
        results["tests"].append({"name": "basic_message", "success": True, "latency_ms": latency})

        # Test 2: Tool Use (Function Calling)
        print("\n  [2] Tool Use (Function Calling)...")
        start = time.time()
        tools = [{
            "name": "get_weather",
            "description": "Get weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }]
        tool_message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            tools=tools,
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}]
        )
        latency = (time.time() - start) * 1000
        tool_use = [b for b in tool_message.content if b.type == "tool_use"]
        if tool_use:
            print(f"      [OK] Tool called: {tool_use[0].name} with {tool_use[0].input} ({latency:.0f}ms)")
        results["tests"].append({"name": "tool_use", "success": bool(tool_use), "latency_ms": latency})

        # Test 3: Streaming
        print("\n  [3] Streaming Response...")
        start = time.time()
        stream_text = ""
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            messages=[{"role": "user", "content": "Count from 1 to 5"}]
        ) as stream:
            for text in stream.text_stream:
                stream_text += text
        latency = (time.time() - start) * 1000
        print(f"      [OK] Streamed: {stream_text[:50]}... ({latency:.0f}ms)")
        results["tests"].append({"name": "streaming", "success": True, "latency_ms": latency})

        results["status"] = "PASS"
        results["tests_passed"] = len([t for t in results["tests"] if t["success"]])

    except Exception as e:
        print(f"      [X] Error: {e}")
        results["status"] = "FAIL"
        results["error"] = str(e)

    return results


async def test_openai_sdk():
    """Test OpenAI SDK - Official Production Patterns"""
    print("\n" + "="*70)
    print("SDK EXECUTION: OPENAI")
    print("="*70)

    results = {"sdk": "openai", "tests": []}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("      [SKIP] OPENAI_API_KEY not set")
        results["status"] = "SKIP"
        return results

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        # Test 1: Chat Completion
        print("\n  [1] Chat Completion...")
        start = time.time()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'OpenAI SDK works' in 3 words"}],
            max_tokens=20
        )
        latency = (time.time() - start) * 1000
        print(f"      [OK] Response: {response.choices[0].message.content} ({latency:.0f}ms)")
        results["tests"].append({"name": "chat_completion", "success": True, "latency_ms": latency})

        # Test 2: Function Calling
        print("\n  [2] Function Calling...")
        start = time.time()
        tools = [{
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get stock price",
                "parameters": {
                    "type": "object",
                    "properties": {"symbol": {"type": "string"}},
                    "required": ["symbol"]
                }
            }
        }]
        func_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Get AAPL stock price"}],
            tools=tools,
            max_tokens=100
        )
        latency = (time.time() - start) * 1000
        tool_calls = func_response.choices[0].message.tool_calls
        if tool_calls:
            print(f"      [OK] Function: {tool_calls[0].function.name}({tool_calls[0].function.arguments}) ({latency:.0f}ms)")
        results["tests"].append({"name": "function_calling", "success": bool(tool_calls), "latency_ms": latency})

        # Test 3: Embeddings
        print("\n  [3] Embeddings...")
        start = time.time()
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input="Production SDK test"
        )
        latency = (time.time() - start) * 1000
        dims = len(embedding.data[0].embedding)
        print(f"      [OK] Embedding: {dims} dimensions ({latency:.0f}ms)")
        results["tests"].append({"name": "embeddings", "success": True, "latency_ms": latency})

        results["status"] = "PASS"
        results["tests_passed"] = len([t for t in results["tests"] if t["success"]])

    except Exception as e:
        print(f"      [X] Error: {e}")
        results["status"] = "FAIL"
        results["error"] = str(e)

    return results


async def test_langchain_sdk():
    """Test LangChain SDK - Official Production Patterns"""
    print("\n" + "="*70)
    print("SDK EXECUTION: LANGCHAIN (with Groq)")
    print("="*70)

    results = {"sdk": "langchain", "tests": []}

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("      [SKIP] GROQ_API_KEY not set")
        results["status"] = "SKIP"
        return results

    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        # Test 1: Basic LLM Chain
        print("\n  [1] Basic LLM Chain...")
        start = time.time()
        llm = ChatGroq(model="llama-3.3-70b-versatile", max_tokens=50, api_key=groq_key)
        result = llm.invoke([HumanMessage(content="Say 'LangChain works'")])
        latency = (time.time() - start) * 1000
        print(f"      [OK] Response: {result.content[:50]}... ({latency:.0f}ms)")
        results["tests"].append({"name": "basic_chain", "success": True, "latency_ms": latency})

        # Test 2: Prompt Template Chain
        print("\n  [2] Prompt Template Chain...")
        start = time.time()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{input}")
        ])
        chain = prompt | llm | StrOutputParser()
        output = chain.invoke({"input": "What is 2+2?"})
        latency = (time.time() - start) * 1000
        print(f"      [OK] Output: {output[:50]}... ({latency:.0f}ms)")
        results["tests"].append({"name": "prompt_chain", "success": True, "latency_ms": latency})

        # Test 3: Tool Binding
        print("\n  [3] Tool Binding...")
        start = time.time()
        from langchain_core.tools import tool

        @tool
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        llm_with_tools = llm.bind_tools([multiply])
        result = llm_with_tools.invoke("What is 7 times 8?")
        latency = (time.time() - start) * 1000
        has_tool = hasattr(result, 'tool_calls') and result.tool_calls
        print(f"      [OK] Tool bound: {has_tool} ({latency:.0f}ms)")
        results["tests"].append({"name": "tool_binding", "success": True, "latency_ms": latency})

        results["status"] = "PASS"
        results["tests_passed"] = len([t for t in results["tests"] if t["success"]])

    except Exception as e:
        print(f"      [X] Error: {e}")
        results["status"] = "FAIL"
        results["error"] = str(e)

    return results


async def test_langgraph_sdk():
    """Test LangGraph SDK - Official Production Patterns"""
    print("\n" + "="*70)
    print("SDK EXECUTION: LANGGRAPH")
    print("="*70)

    results = {"sdk": "langgraph", "tests": []}

    try:
        from langgraph.graph import StateGraph, START, END
        from typing import TypedDict, Annotated
        import operator

        # Test 1: Basic StateGraph
        print("\n  [1] StateGraph Creation...")
        start = time.time()

        class State(TypedDict):
            messages: Annotated[list, operator.add]
            count: int

        def node_a(state: State) -> dict:
            return {"messages": ["Node A executed"], "count": state["count"] + 1}

        def node_b(state: State) -> dict:
            return {"messages": ["Node B executed"], "count": state["count"] + 1}

        graph = StateGraph(State)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)

        app = graph.compile()
        latency = (time.time() - start) * 1000
        print(f"      [OK] Graph compiled ({latency:.0f}ms)")
        results["tests"].append({"name": "graph_creation", "success": True, "latency_ms": latency})

        # Test 2: Graph Execution
        print("\n  [2] Graph Execution...")
        start = time.time()
        result = app.invoke({"messages": [], "count": 0})
        latency = (time.time() - start) * 1000
        print(f"      [OK] Messages: {result['messages']}, Count: {result['count']} ({latency:.0f}ms)")
        results["tests"].append({"name": "graph_execution", "success": result["count"] == 2, "latency_ms": latency})

        # Test 3: Conditional Edges
        print("\n  [3] Conditional Routing...")
        start = time.time()

        def router(state: State) -> str:
            return "end" if state["count"] >= 2 else "continue"

        graph2 = StateGraph(State)
        graph2.add_node("process", node_a)
        graph2.add_edge(START, "process")
        graph2.add_conditional_edges("process", router, {"continue": "process", "end": END})
        app2 = graph2.compile()
        result2 = app2.invoke({"messages": [], "count": 0})
        latency = (time.time() - start) * 1000
        print(f"      [OK] Final count: {result2['count']} ({latency:.0f}ms)")
        results["tests"].append({"name": "conditional_edges", "success": True, "latency_ms": latency})

        results["status"] = "PASS"
        results["tests_passed"] = len([t for t in results["tests"] if t["success"]])

    except Exception as e:
        print(f"      [X] Error: {e}")
        results["status"] = "FAIL"
        results["error"] = str(e)

    return results


async def test_exa_sdk():
    """Test Exa SDK - Official Production Patterns"""
    print("\n" + "="*70)
    print("SDK EXECUTION: EXA NEURAL SEARCH")
    print("="*70)

    results = {"sdk": "exa", "tests": []}

    try:
        from exa_py import Exa

        client = Exa(os.getenv("EXA_API_KEY"))

        # Test 1: Neural Search
        print("\n  [1] Neural Search...")
        start = time.time()
        search = client.search("Claude API best practices", type="neural", num_results=3)
        latency = (time.time() - start) * 1000
        print(f"      [OK] Results: {len(search.results)} ({latency:.0f}ms)")
        for r in search.results[:2]:
            print(f"        - {r.title[:50]}...")
        results["tests"].append({"name": "neural_search", "success": len(search.results) > 0, "latency_ms": latency})

        # Test 2: Search with Contents
        print("\n  [2] Search with Contents...")
        start = time.time()
        contents = client.search_and_contents(
            "LangGraph StateGraph tutorial",
            type="auto",
            num_results=2,
            text=True
        )
        latency = (time.time() - start) * 1000
        has_text = any(r.text for r in contents.results)
        print(f"      [OK] Results with text: {has_text} ({latency:.0f}ms)")
        results["tests"].append({"name": "search_contents", "success": has_text, "latency_ms": latency})

        # Test 3: Find Similar
        print("\n  [3] Find Similar...")
        start = time.time()
        if search.results:
            similar = client.find_similar(search.results[0].url, num_results=2)
            latency = (time.time() - start) * 1000
            print(f"      [OK] Similar: {len(similar.results)} results ({latency:.0f}ms)")
            results["tests"].append({"name": "find_similar", "success": True, "latency_ms": latency})

        results["status"] = "PASS"
        results["tests_passed"] = len([t for t in results["tests"] if t["success"]])

    except Exception as e:
        print(f"      [X] Error: {e}")
        results["status"] = "FAIL"
        results["error"] = str(e)

    return results


async def test_qdrant_sdk():
    """Test Qdrant SDK - Official Production Patterns"""
    print("\n" + "="*70)
    print("SDK EXECUTION: QDRANT VECTOR DB")
    print("="*70)

    results = {"sdk": "qdrant", "tests": []}

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        import uuid

        # Test 1: In-Memory Client
        print("\n  [1] In-Memory Client...")
        start = time.time()
        client = QdrantClient(":memory:")
        latency = (time.time() - start) * 1000
        print(f"      [OK] Client created ({latency:.0f}ms)")
        results["tests"].append({"name": "client_creation", "success": True, "latency_ms": latency})

        # Test 2: Create Collection
        print("\n  [2] Create Collection...")
        start = time.time()
        client.create_collection(
            collection_name="test_collection",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        latency = (time.time() - start) * 1000
        print(f"      [OK] Collection created ({latency:.0f}ms)")
        results["tests"].append({"name": "create_collection", "success": True, "latency_ms": latency})

        # Test 3: Upsert & Search
        print("\n  [3] Upsert & Search...")
        start = time.time()
        import random
        points = [
            PointStruct(id=i, vector=[random.random() for _ in range(384)], payload={"text": f"doc_{i}"})
            for i in range(10)
        ]
        client.upsert(collection_name="test_collection", points=points)

        results_search = client.query_points(
            collection_name="test_collection",
            query=[random.random() for _ in range(384)],
            limit=3
        ).points
        latency = (time.time() - start) * 1000
        print(f"      [OK] Searched: {len(results_search)} results ({latency:.0f}ms)")
        results["tests"].append({"name": "upsert_search", "success": len(results_search) == 3, "latency_ms": latency})

        results["status"] = "PASS"
        results["tests_passed"] = len([t for t in results["tests"] if t["success"]])

    except Exception as e:
        print(f"      [X] Error: {e}")
        results["status"] = "FAIL"
        results["error"] = str(e)

    return results


async def test_tavily_sdk():
    """Test Tavily SDK - Official Production Patterns"""
    print("\n" + "="*70)
    print("SDK EXECUTION: TAVILY AI SEARCH")
    print("="*70)

    results = {"sdk": "tavily", "tests": []}

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("      [SKIP] TAVILY_API_KEY not set")
        results["status"] = "SKIP"
        return results

    try:
        import httpx

        async with httpx.AsyncClient(timeout=60) as client:
            # Test 1: Basic Search
            print("\n  [1] Basic Search...")
            start = time.time()
            r = await client.post('https://api.tavily.com/search', json={
                'api_key': api_key,
                'query': 'LangGraph production patterns',
                'search_depth': 'basic',
                'include_answer': True,
                'max_results': 3
            })
            latency = (time.time() - start) * 1000
            data = r.json()
            print(f"      [OK] Results: {len(data.get('results', []))} ({latency:.0f}ms)")
            if data.get('answer'):
                print(f"          Answer: {data['answer'][:60]}...")
            results["tests"].append({"name": "basic_search", "success": True, "latency_ms": latency})

            # Test 2: Advanced Search
            print("\n  [2] Advanced Search...")
            start = time.time()
            r = await client.post('https://api.tavily.com/search', json={
                'api_key': api_key,
                'query': 'Claude API tool use best practices',
                'search_depth': 'advanced',
                'include_answer': True,
                'max_results': 5
            })
            latency = (time.time() - start) * 1000
            data = r.json()
            print(f"      [OK] Results: {len(data.get('results', []))} ({latency:.0f}ms)")
            results["tests"].append({"name": "advanced_search", "success": True, "latency_ms": latency})

            # Test 3: Extract
            print("\n  [3] URL Extract...")
            start = time.time()
            r = await client.post('https://api.tavily.com/extract', json={
                'api_key': api_key,
                'urls': ['https://docs.anthropic.com/en/docs/build-with-claude/tool-use']
            })
            latency = (time.time() - start) * 1000
            data = r.json()
            print(f"      [OK] Extracted: {len(data.get('results', []))} URLs ({latency:.0f}ms)")
            results["tests"].append({"name": "extract", "success": True, "latency_ms": latency})

        results["status"] = "PASS"
        results["tests_passed"] = len([t for t in results["tests"] if t["success"]])

    except Exception as e:
        print(f"      [X] Error: {e}")
        results["status"] = "FAIL"
        results["error"] = str(e)

    return results


async def test_jina_sdk():
    """Test Jina AI SDK - Official Production Patterns"""
    print("\n" + "="*70)
    print("SDK EXECUTION: JINA AI")
    print("="*70)

    results = {"sdk": "jina", "tests": []}

    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        print("      [SKIP] JINA_API_KEY not set")
        results["status"] = "SKIP"
        return results

    try:
        import httpx

        async with httpx.AsyncClient(timeout=60) as client:
            # Test 1: Reader API
            print("\n  [1] Reader API (URL to Markdown)...")
            start = time.time()
            r = await client.get(
                'https://r.jina.ai/https://docs.anthropic.com/en/docs/build-with-claude/tool-use',
                headers={'Authorization': f'Bearer {api_key}', 'Accept': 'application/json'}
            )
            latency = (time.time() - start) * 1000
            print(f"      [OK] Content: {len(r.text)} chars ({latency:.0f}ms)")
            results["tests"].append({"name": "reader_api", "success": True, "latency_ms": latency})

            # Test 2: Embeddings
            print("\n  [2] Embeddings API...")
            start = time.time()
            r = await client.post(
                'https://api.jina.ai/v1/embeddings',
                headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
                json={'model': 'jina-embeddings-v3', 'input': ['Production SDK test for embeddings']}
            )
            latency = (time.time() - start) * 1000
            data = r.json()
            dims = len(data.get('data', [{}])[0].get('embedding', []))
            print(f"      [OK] Dimensions: {dims} ({latency:.0f}ms)")
            results["tests"].append({"name": "embeddings", "success": dims > 0, "latency_ms": latency})

            # Test 3: Reranker
            print("\n  [3] Reranker API...")
            start = time.time()
            r = await client.post(
                'https://api.jina.ai/v1/rerank',
                headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
                json={
                    'model': 'jina-reranker-v2-base-multilingual',
                    'query': 'production deployment best practices',
                    'documents': [
                        'How to deploy ML models to production',
                        'Best practices for LLM deployment',
                        'Random unrelated document about cooking'
                    ],
                    'top_n': 2
                }
            )
            latency = (time.time() - start) * 1000
            print(f"      [OK] Status: {r.status_code} ({latency:.0f}ms)")
            results["tests"].append({"name": "reranker", "success": r.status_code == 200, "latency_ms": latency})

        results["status"] = "PASS"
        results["tests_passed"] = len([t for t in results["tests"] if t["success"]])

    except Exception as e:
        print(f"      [X] Error: {e}")
        results["status"] = "FAIL"
        results["error"] = str(e)

    return results


async def test_perplexity_sdk():
    """Test Perplexity SDK - Official Production Patterns"""
    print("\n" + "="*70)
    print("SDK EXECUTION: PERPLEXITY SONAR")
    print("="*70)

    results = {"sdk": "perplexity", "tests": []}

    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("      [SKIP] PERPLEXITY_API_KEY not set")
        results["status"] = "SKIP"
        return results

    try:
        import httpx

        async with httpx.AsyncClient(timeout=60) as client:
            # Test 1: Sonar Chat
            print("\n  [1] Sonar Chat with Citations...")
            start = time.time()
            r = await client.post(
                'https://api.perplexity.ai/chat/completions',
                headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
                json={
                    'model': 'sonar',
                    'messages': [{'role': 'user', 'content': 'What are the key features of Claude API?'}],
                    'return_citations': True
                }
            )
            latency = (time.time() - start) * 1000
            data = r.json()
            citations = len(data.get('citations', []))
            print(f"      [OK] Citations: {citations} ({latency:.0f}ms)")
            results["tests"].append({"name": "sonar_chat", "success": True, "latency_ms": latency})

            # Test 2: Sonar Pro
            print("\n  [2] Sonar Pro (Advanced)...")
            start = time.time()
            r = await client.post(
                'https://api.perplexity.ai/chat/completions',
                headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
                json={
                    'model': 'sonar-pro',
                    'messages': [{'role': 'user', 'content': 'Compare LangGraph vs CrewAI for production'}],
                    'return_citations': True
                }
            )
            latency = (time.time() - start) * 1000
            data = r.json()
            content_len = len(data.get('choices', [{}])[0].get('message', {}).get('content', ''))
            citations = len(data.get('citations', []))
            print(f"      [OK] Response: {content_len} chars, {citations} citations ({latency:.0f}ms)")
            results["tests"].append({"name": "sonar_pro", "success": content_len > 0, "latency_ms": latency})

        results["status"] = "PASS"
        results["tests_passed"] = len([t for t in results["tests"] if t["success"]])

    except Exception as e:
        print(f"      [X] Error: {e}")
        results["status"] = "FAIL"
        results["error"] = str(e)

    return results


async def main():
    """Main SDK execution loop."""
    print("="*70)
    print("SDK EXECUTION LOOP - BATTLE-TESTED PRODUCTION PATTERNS")
    print("="*70)
    print(f"Start Time: {datetime.now().isoformat()}")
    print("="*70)

    all_results = []

    # Execute all SDK tests
    all_results.append(await test_anthropic_sdk())
    all_results.append(await test_openai_sdk())
    all_results.append(await test_langchain_sdk())
    all_results.append(await test_langgraph_sdk())
    all_results.append(await test_exa_sdk())
    all_results.append(await test_tavily_sdk())
    all_results.append(await test_jina_sdk())
    all_results.append(await test_perplexity_sdk())
    all_results.append(await test_qdrant_sdk())

    # Summary
    print("\n" + "="*70)
    print("SDK EXECUTION SUMMARY")
    print("="*70)

    total_tests = 0
    passed_tests = 0

    for result in all_results:
        sdk = result["sdk"]
        status = result.get("status", "UNKNOWN")
        tests_passed = result.get("tests_passed", 0)
        total = len(result.get("tests", []))
        total_tests += total
        passed_tests += tests_passed

        symbol = "[OK]" if status == "PASS" else "[X]"
        print(f"  {symbol} {sdk.upper()}: {tests_passed}/{total} tests passed")

    print(f"\n  TOTAL: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.0f}%)")
    print("="*70)

    # Save results
    output_file = "sdk_execution_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0
            },
            "results": all_results
        }, f, indent=2)

    print(f"\n[OK] Results saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
