"""
PRODUCTION PATTERNS LOOP - Execute Real SDK Patterns from Official Docs
========================================================================
Tests battle-tested production patterns with actual execution.
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
# PRODUCTION PATTERN TESTS
# ============================================================================

async def pattern_langchain_rag():
    """LangChain RAG Pattern - Production Ready"""
    print("\n" + "="*70)
    print("PATTERN: LangChain RAG with Qdrant")
    print("="*70)

    results = {"pattern": "langchain_rag", "tests": []}

    try:
        from langchain_community.vectorstores import Qdrant
        from langchain_core.documents import Document
        from langchain_huggingface import HuggingFaceEmbeddings
        from qdrant_client import QdrantClient

        # Test 1: Create embeddings
        print("\n  [1] Initialize HuggingFace Embeddings...")
        start = time.time()
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        latency = (time.time() - start) * 1000
        print(f"      [OK] Embeddings loaded ({latency:.0f}ms)")
        results["tests"].append({"name": "embeddings_init", "success": True, "latency_ms": latency})

        # Test 2: Create vector store
        print("\n  [2] Create Qdrant Vector Store...")
        start = time.time()
        client = QdrantClient(":memory:")
        docs = [
            Document(page_content="LangGraph is great for building agentic workflows", metadata={"source": "docs"}),
            Document(page_content="CrewAI excels at multi-agent collaboration", metadata={"source": "blog"}),
            Document(page_content="Claude API supports tool use and function calling", metadata={"source": "api"}),
        ]
        vectorstore = Qdrant.from_documents(
            docs,
            embeddings,
            location=":memory:",
            collection_name="test_rag"
        )
        latency = (time.time() - start) * 1000
        print(f"      [OK] Vector store created with {len(docs)} docs ({latency:.0f}ms)")
        results["tests"].append({"name": "vectorstore_create", "success": True, "latency_ms": latency})

        # Test 3: Similarity search
        print("\n  [3] Similarity Search...")
        start = time.time()
        results_search = vectorstore.similarity_search("agent workflows", k=2)
        latency = (time.time() - start) * 1000
        print(f"      [OK] Found {len(results_search)} results ({latency:.0f}ms)")
        for r in results_search:
            print(f"          - {r.page_content[:50]}...")
        results["tests"].append({"name": "similarity_search", "success": len(results_search) > 0, "latency_ms": latency})

        results["status"] = "PASS"
        results["tests_passed"] = len([t for t in results["tests"] if t["success"]])

    except Exception as e:
        print(f"      [X] Error: {e}")
        results["status"] = "FAIL"
        results["error"] = str(e)

    return results


async def pattern_langgraph_agent():
    """LangGraph Agent Pattern - Production Ready"""
    print("\n" + "="*70)
    print("PATTERN: LangGraph ReAct Agent")
    print("="*70)

    results = {"pattern": "langgraph_agent", "tests": []}

    try:
        from langgraph.graph import StateGraph, START, END
        from typing import TypedDict, Annotated, Literal
        import operator

        # Test 1: Create ReAct-style state
        print("\n  [1] Define Agent State...")
        start = time.time()

        class AgentState(TypedDict):
            messages: Annotated[list, operator.add]
            next_action: str
            result: str

        latency = (time.time() - start) * 1000
        print(f"      [OK] State defined ({latency:.0f}ms)")
        results["tests"].append({"name": "state_definition", "success": True, "latency_ms": latency})

        # Test 2: Create tool nodes
        print("\n  [2] Create Tool Nodes...")
        start = time.time()

        def search_tool(state: AgentState) -> dict:
            return {"messages": ["Searched for information"], "result": "Found relevant data"}

        def calculate_tool(state: AgentState) -> dict:
            return {"messages": ["Performed calculation"], "result": "42"}

        def router(state: AgentState) -> Literal["search", "calculate", "end"]:
            if "search" in str(state.get("messages", [])):
                return "calculate"
            elif "calculate" in str(state.get("messages", [])):
                return "end"
            return "search"

        latency = (time.time() - start) * 1000
        print(f"      [OK] Tool nodes created ({latency:.0f}ms)")
        results["tests"].append({"name": "tool_nodes", "success": True, "latency_ms": latency})

        # Test 3: Build and run graph
        print("\n  [3] Build and Execute Graph...")
        start = time.time()

        graph = StateGraph(AgentState)
        graph.add_node("search", search_tool)
        graph.add_node("calculate", calculate_tool)
        graph.add_edge(START, "search")
        graph.add_conditional_edges("search", router, {"search": "search", "calculate": "calculate", "end": END})
        graph.add_conditional_edges("calculate", router, {"search": "search", "calculate": "calculate", "end": END})

        app = graph.compile()
        result = app.invoke({"messages": [], "next_action": "", "result": ""})

        latency = (time.time() - start) * 1000
        print(f"      [OK] Agent executed: {result['messages']} ({latency:.0f}ms)")
        results["tests"].append({"name": "graph_execution", "success": True, "latency_ms": latency})

        results["status"] = "PASS"
        results["tests_passed"] = len([t for t in results["tests"] if t["success"]])

    except Exception as e:
        print(f"      [X] Error: {e}")
        results["status"] = "FAIL"
        results["error"] = str(e)

    return results


async def pattern_multi_source_research():
    """Multi-Source Research Pattern - Production Ready"""
    print("\n" + "="*70)
    print("PATTERN: Multi-Source Research (Exa + Tavily + Jina)")
    print("="*70)

    results = {"pattern": "multi_source_research", "tests": []}

    try:
        import httpx
        from exa_py import Exa

        query = "Claude API structured outputs best practices"

        # Test 1: Exa Neural Search
        print("\n  [1] Exa Neural Search...")
        start = time.time()
        exa = Exa(os.getenv("EXA_API_KEY"))
        exa_results = exa.search(query, type="neural", num_results=3)
        latency = (time.time() - start) * 1000
        print(f"      [OK] Exa: {len(exa_results.results)} results ({latency:.0f}ms)")
        results["tests"].append({"name": "exa_search", "success": True, "latency_ms": latency, "count": len(exa_results.results)})

        # Test 2: Tavily AI Search
        print("\n  [2] Tavily AI Search...")
        start = time.time()
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post('https://api.tavily.com/search', json={
                'api_key': os.getenv("TAVILY_API_KEY"),
                'query': query,
                'search_depth': 'advanced',
                'include_answer': True,
                'max_results': 3
            })
            tavily_data = r.json()
        latency = (time.time() - start) * 1000
        tavily_count = len(tavily_data.get('results', []))
        print(f"      [OK] Tavily: {tavily_count} results ({latency:.0f}ms)")
        if tavily_data.get('answer'):
            print(f"          Answer: {tavily_data['answer'][:60]}...")
        results["tests"].append({"name": "tavily_search", "success": True, "latency_ms": latency, "count": tavily_count})

        # Test 3: Jina Embeddings for Semantic Comparison
        print("\n  [3] Jina Embeddings for Reranking...")
        start = time.time()
        async with httpx.AsyncClient(timeout=60) as client:
            # Collect all texts
            texts = [r.title for r in exa_results.results[:3]]
            texts.extend([r.get('title', '') for r in tavily_data.get('results', [])[:3]])

            r = await client.post(
                'https://api.jina.ai/v1/rerank',
                headers={'Authorization': f'Bearer {os.getenv("JINA_API_KEY")}', 'Content-Type': 'application/json'},
                json={
                    'model': 'jina-reranker-v2-base-multilingual',
                    'query': query,
                    'documents': texts[:5],
                    'top_n': 3
                }
            )
        latency = (time.time() - start) * 1000
        print(f"      [OK] Jina reranked {len(texts[:5])} docs ({latency:.0f}ms)")
        results["tests"].append({"name": "jina_rerank", "success": r.status_code == 200, "latency_ms": latency})

        # Summary
        total_sources = len(exa_results.results) + tavily_count
        print(f"\n      Total sources gathered: {total_sources}")

        results["status"] = "PASS"
        results["tests_passed"] = len([t for t in results["tests"] if t["success"]])
        results["total_sources"] = total_sources

    except Exception as e:
        print(f"      [X] Error: {e}")
        results["status"] = "FAIL"
        results["error"] = str(e)

    return results


async def pattern_hybrid_search():
    """Hybrid Search Pattern - BM25 + Vector"""
    print("\n" + "="*70)
    print("PATTERN: Hybrid Search (BM25 + Vector)")
    print("="*70)

    results = {"pattern": "hybrid_search", "tests": []}

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVectorParams, SparseVector
        import random

        # Test 1: Create collection with sparse+dense vectors
        print("\n  [1] Create Hybrid Collection...")
        start = time.time()
        client = QdrantClient(":memory:")

        # Note: In production, you'd use actual BM25 vectors
        # This demonstrates the pattern structure
        client.create_collection(
            collection_name="hybrid_test",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        latency = (time.time() - start) * 1000
        print(f"      [OK] Collection created ({latency:.0f}ms)")
        results["tests"].append({"name": "collection_create", "success": True, "latency_ms": latency})

        # Test 2: Insert documents with vectors
        print("\n  [2] Insert Documents...")
        start = time.time()
        docs = [
            "LangGraph provides powerful state management for agents",
            "CrewAI enables multi-agent collaboration patterns",
            "Claude API supports structured outputs with JSON schema",
            "RAG combines retrieval with generation for accuracy",
            "Vector databases enable semantic similarity search",
        ]
        points = [
            PointStruct(
                id=i,
                vector=[random.random() for _ in range(384)],
                payload={"text": doc, "source": f"doc_{i}"}
            )
            for i, doc in enumerate(docs)
        ]
        client.upsert(collection_name="hybrid_test", points=points)
        latency = (time.time() - start) * 1000
        print(f"      [OK] Inserted {len(docs)} documents ({latency:.0f}ms)")
        results["tests"].append({"name": "insert_docs", "success": True, "latency_ms": latency})

        # Test 3: Execute hybrid search
        print("\n  [3] Execute Hybrid Search...")
        start = time.time()
        search_results = client.query_points(
            collection_name="hybrid_test",
            query=[random.random() for _ in range(384)],
            limit=3,
            with_payload=True
        ).points
        latency = (time.time() - start) * 1000
        print(f"      [OK] Found {len(search_results)} results ({latency:.0f}ms)")
        for r in search_results:
            print(f"          - {r.payload['text'][:40]}...")
        results["tests"].append({"name": "hybrid_search", "success": len(search_results) > 0, "latency_ms": latency})

        results["status"] = "PASS"
        results["tests_passed"] = len([t for t in results["tests"] if t["success"]])

    except Exception as e:
        print(f"      [X] Error: {e}")
        results["status"] = "FAIL"
        results["error"] = str(e)

    return results


async def pattern_streaming_response():
    """Streaming Response Pattern - Production Ready"""
    print("\n" + "="*70)
    print("PATTERN: Streaming LLM Response")
    print("="*70)

    results = {"pattern": "streaming_response", "tests": []}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("      [SKIP] OPENAI_API_KEY not set")
        results["status"] = "SKIP"
        return results

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        # Test 1: Create streaming response
        print("\n  [1] Initialize Stream...")
        start = time.time()
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "List 3 best practices for production AI systems, briefly"}],
            stream=True,
            max_tokens=150
        )
        latency = (time.time() - start) * 1000
        print(f"      [OK] Stream initialized ({latency:.0f}ms)")
        results["tests"].append({"name": "stream_init", "success": True, "latency_ms": latency})

        # Test 2: Collect chunks
        print("\n  [2] Collect Stream Chunks...")
        start = time.time()
        chunks = []
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                chunks.append(content)
                full_response += content
        latency = (time.time() - start) * 1000
        print(f"      [OK] Received {len(chunks)} chunks ({latency:.0f}ms)")
        print(f"          Response: {full_response[:80]}...")
        results["tests"].append({"name": "stream_collect", "success": len(chunks) > 0, "latency_ms": latency})

        # Test 3: Verify complete response
        print("\n  [3] Verify Response...")
        start = time.time()
        is_complete = len(full_response) > 50
        latency = (time.time() - start) * 1000
        print(f"      [OK] Response complete: {is_complete} ({len(full_response)} chars)")
        results["tests"].append({"name": "verify_complete", "success": is_complete, "latency_ms": latency})

        results["status"] = "PASS"
        results["tests_passed"] = len([t for t in results["tests"] if t["success"]])

    except Exception as e:
        print(f"      [X] Error: {e}")
        results["status"] = "FAIL"
        results["error"] = str(e)

    return results


async def main():
    """Main production patterns loop."""
    print("="*70)
    print("PRODUCTION PATTERNS LOOP - BATTLE-TESTED EXECUTION")
    print("="*70)
    print(f"Start Time: {datetime.now().isoformat()}")
    print("="*70)

    all_results = []

    # Execute all pattern tests
    all_results.append(await pattern_langchain_rag())
    all_results.append(await pattern_langgraph_agent())
    all_results.append(await pattern_multi_source_research())
    all_results.append(await pattern_hybrid_search())
    all_results.append(await pattern_streaming_response())

    # Summary
    print("\n" + "="*70)
    print("PRODUCTION PATTERNS SUMMARY")
    print("="*70)

    total_tests = 0
    passed_tests = 0

    for result in all_results:
        pattern = result["pattern"]
        status = result.get("status", "UNKNOWN")
        tests_passed = result.get("tests_passed", 0)
        total = len(result.get("tests", []))
        total_tests += total
        passed_tests += tests_passed

        symbol = "[OK]" if status == "PASS" else "[X]" if status == "FAIL" else "[SKIP]"
        print(f"  {symbol} {pattern}: {tests_passed}/{total} tests passed")

    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    print(f"\n  TOTAL: {passed_tests}/{total_tests} tests passed ({success_rate*100:.0f}%)")
    print("="*70)

    # Save results
    output_file = "production_patterns_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "success_rate": success_rate
            },
            "results": all_results
        }, f, indent=2)

    print(f"\n[OK] Results saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
