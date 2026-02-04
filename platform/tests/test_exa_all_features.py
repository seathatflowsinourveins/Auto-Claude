"""
Exa Adapter - Full Feature Test Suite
======================================

Tests ALL Exa 2.0+ features with real API calls:
1. search - Neural/semantic search (fast, auto, neural, keyword, deep)
2. get_contents - Extract content from URLs
3. find_similar - Find similar content
4. search_and_contents - Combined search + content
5. answer - AI answers with citations
6. stream_answer - Streaming responses
7. research - Async structured research
8. company_research - Company intelligence
9. linkedin_search - People/company discovery
10. code_search - Code snippets & docs

Run: python -m pytest platform/tests/test_exa_all_features.py -v -s
"""

import asyncio
import os
import sys
from pathlib import Path

from adapters.exa_adapter import ExaAdapter, EXA_AVAILABLE

import pytest


# Test configuration
TEST_QUERIES = {
    "general": "Claude AI assistant capabilities 2024",
    "code": "LangGraph StateGraph Python example",
    "company": "anthropic.com",
    "person": "Dario Amodei CEO",
    "github": "https://github.com/anthropics/anthropic-cookbook",
    "research": "Compare vector databases for AI applications",
}


@pytest.fixture
async def adapter():
    """Create and initialize Exa adapter."""
    adapter = ExaAdapter()
    result = await adapter.initialize({})
    yield adapter
    await adapter.shutdown()


class TestExaAllFeatures:
    """Test all Exa features with real API calls."""

    @pytest.mark.asyncio
    async def test_00_initialization(self, adapter):
        """Test adapter initialization and status."""
        print("\n" + "="*60)
        print("TEST 00: INITIALIZATION")
        print("="*60)

        assert adapter.sdk_name == "exa"
        print(f"✓ SDK Name: {adapter.sdk_name}")
        print(f"✓ SDK Available: {adapter.available}")

        if not adapter.available:
            pytest.skip("Exa SDK not installed")

        # Check if API key is set
        api_key = os.getenv("EXA_API_KEY")
        if api_key:
            print(f"✓ API Key: {api_key[:10]}...{api_key[-4:]}")
        else:
            print("⚠ API Key: Not set (running in mock mode)")

    @pytest.mark.asyncio
    async def test_01_search_fast(self, adapter):
        """Test fast search (<350ms latency target)."""
        print("\n" + "="*60)
        print("TEST 01: SEARCH - FAST MODE")
        print("="*60)

        result = await adapter.execute(
            "search",
            query=TEST_QUERIES["general"],
            type="fast",
            num_results=5,
        )

        print(f"Success: {result.success}")
        print(f"Latency: {result.latency_ms:.2f}ms")

        if result.success:
            data = result.data
            print(f"Results count: {data.get('count', len(data.get('results', [])))}")
            print(f"Type: {data.get('type')}")
            print(f"Mock: {data.get('mock', False)}")

            for i, r in enumerate(data.get("results", [])[:3]):
                print(f"\n  Result {i+1}:")
                print(f"    Title: {r.get('title', 'N/A')[:60]}")
                print(f"    URL: {r.get('url', 'N/A')[:60]}")
                print(f"    Score: {r.get('score', 0):.3f}")
        else:
            print(f"Error: {result.error}")

        assert result.success

    @pytest.mark.asyncio
    async def test_02_search_auto(self, adapter):
        """Test auto search (intelligent hybrid)."""
        print("\n" + "="*60)
        print("TEST 02: SEARCH - AUTO MODE")
        print("="*60)

        result = await adapter.execute(
            "search",
            query=TEST_QUERIES["general"],
            type="auto",
            num_results=5,
            use_autoprompt=True,
        )

        print(f"Success: {result.success}")
        print(f"Latency: {result.latency_ms:.2f}ms")

        if result.success:
            data = result.data
            print(f"Results: {data.get('count', len(data.get('results', [])))}")
            print(f"Autoprompt: {data.get('autoprompt_string', 'N/A')}")

            for i, r in enumerate(data.get("results", [])[:2]):
                print(f"\n  Result {i+1}: {r.get('title', 'N/A')[:50]}")

        assert result.success

    @pytest.mark.asyncio
    async def test_03_search_neural(self, adapter):
        """Test pure neural/embeddings search."""
        print("\n" + "="*60)
        print("TEST 03: SEARCH - NEURAL MODE")
        print("="*60)

        result = await adapter.execute(
            "search",
            query="semantic understanding of natural language",
            type="neural",
            num_results=5,
        )

        print(f"Success: {result.success}")
        print(f"Latency: {result.latency_ms:.2f}ms")
        print(f"Results: {len(result.data.get('results', []))}")

        assert result.success

    @pytest.mark.asyncio
    async def test_04_search_deep(self, adapter):
        """Test deep agentic search (highest quality, ~3.5s)."""
        print("\n" + "="*60)
        print("TEST 04: SEARCH - DEEP MODE (Agentic)")
        print("="*60)

        result = await adapter.execute(
            "search",
            query="distributed consensus algorithms comparison Raft Paxos PBFT",
            type="deep",
            num_results=10,
            additional_queries=[
                "Byzantine fault tolerance",
                "leader election protocols",
            ],
        )

        print(f"Success: {result.success}")
        print(f"Latency: {result.latency_ms:.2f}ms (expected ~3500ms)")

        if result.success:
            results = result.data.get("results", [])
            print(f"Results: {len(results)}")
            for i, r in enumerate(results[:3]):
                print(f"  {i+1}. {r.get('title', 'N/A')[:50]}")

        assert result.success

    @pytest.mark.asyncio
    async def test_05_search_with_filters(self, adapter):
        """Test search with domain and date filters."""
        print("\n" + "="*60)
        print("TEST 05: SEARCH - WITH FILTERS")
        print("="*60)

        result = await adapter.execute(
            "search",
            query="machine learning research papers",
            type="auto",
            num_results=10,
            include_domains=["arxiv.org", "openreview.net"],
            category="research paper",
            start_published_date="2024-01-01",
        )

        print(f"Success: {result.success}")
        print(f"Latency: {result.latency_ms:.2f}ms")

        if result.success:
            results = result.data.get("results", [])
            print(f"Results: {len(results)}")
            for r in results[:3]:
                print(f"  - {r.get('url', 'N/A')[:60]}")

        assert result.success

    @pytest.mark.asyncio
    async def test_06_search_github(self, adapter):
        """Test GitHub-focused search."""
        print("\n" + "="*60)
        print("TEST 06: SEARCH - GITHUB CATEGORY")
        print("="*60)

        result = await adapter.execute(
            "search",
            query="LangChain agent examples Python",
            type="auto",
            num_results=10,
            category="github",
            include_domains=["github.com"],
        )

        print(f"Success: {result.success}")
        print(f"Latency: {result.latency_ms:.2f}ms")

        if result.success:
            results = result.data.get("results", [])
            print(f"GitHub results: {len(results)}")
            for r in results[:5]:
                print(f"  - {r.get('title', 'N/A')[:40]} | {r.get('url', '')}")

        assert result.success

    @pytest.mark.asyncio
    async def test_07_get_contents(self, adapter):
        """Test content extraction from URLs."""
        print("\n" + "="*60)
        print("TEST 07: GET CONTENTS")
        print("="*60)

        urls = [
            "https://docs.anthropic.com/en/docs/overview",
            "https://github.com/anthropics/anthropic-cookbook",
        ]

        result = await adapter.execute(
            "get_contents",
            urls=urls,
            text=True,
            highlights=True,
            summary=True,
        )

        print(f"Success: {result.success}")
        print(f"Latency: {result.latency_ms:.2f}ms")

        if result.success:
            contents = result.data.get("contents", [])
            print(f"Contents fetched: {len(contents)}")
            for c in contents:
                print(f"\n  URL: {c.get('url', 'N/A')[:50]}")
                print(f"  Title: {c.get('title', 'N/A')[:40]}")
                text = c.get('text', '')
                print(f"  Text length: {len(text)} chars")
                if c.get('summary'):
                    print(f"  Summary: {c.get('summary')[:100]}...")

        assert result.success

    @pytest.mark.asyncio
    async def test_08_find_similar(self, adapter):
        """Test finding similar content to a URL."""
        print("\n" + "="*60)
        print("TEST 08: FIND SIMILAR")
        print("="*60)

        result = await adapter.execute(
            "find_similar",
            url=TEST_QUERIES["github"],
            num_results=5,
            exclude_source_domain=True,
        )

        print(f"Success: {result.success}")
        print(f"Latency: {result.latency_ms:.2f}ms")

        if result.success:
            results = result.data.get("results", [])
            print(f"Similar pages found: {len(results)}")
            print(f"Source: {result.data.get('source_url', 'N/A')}")
            for r in results[:5]:
                print(f"  - {r.get('title', 'N/A')[:40]} (score: {r.get('score', 0):.3f})")

        assert result.success

    @pytest.mark.asyncio
    async def test_09_search_and_contents(self, adapter):
        """Test combined search + content extraction."""
        print("\n" + "="*60)
        print("TEST 09: SEARCH AND CONTENTS")
        print("="*60)

        result = await adapter.execute(
            "search_and_contents",
            query="Python async patterns best practices",
            num_results=3,
            type="auto",
            text=True,
            highlights=True,
            summary=True,
        )

        print(f"Success: {result.success}")
        print(f"Latency: {result.latency_ms:.2f}ms")

        if result.success:
            results = result.data.get("results", [])
            print(f"Results with contents: {len(results)}")
            for r in results[:2]:
                print(f"\n  Title: {r.get('title', 'N/A')[:50]}")
                print(f"  URL: {r.get('url', 'N/A')[:50]}")
                if r.get('highlights'):
                    print(f"  Highlights: {r.get('highlights')[0][:80]}...")
                if r.get('summary'):
                    print(f"  Summary: {r.get('summary')[:80]}...")

        assert result.success

    @pytest.mark.asyncio
    async def test_10_answer(self, adapter):
        """Test AI-generated answers with citations."""
        print("\n" + "="*60)
        print("TEST 10: ANSWER (AI with Citations)")
        print("="*60)

        result = await adapter.execute(
            "answer",
            query="What is Claude AI and what are its main capabilities?",
        )

        print(f"Success: {result.success}")
        print(f"Latency: {result.latency_ms:.2f}ms")

        if result.success:
            data = result.data
            answer = data.get("answer", "")
            citations = data.get("citations", [])

            print(f"\nAnswer ({len(answer)} chars):")
            print(f"  {answer[:300]}...")
            print(f"\nCitations: {len(citations)}")
            for c in citations[:3]:
                print(f"  - {c}")

        assert result.success

    @pytest.mark.asyncio
    async def test_11_stream_answer(self, adapter):
        """Test streaming answer responses."""
        print("\n" + "="*60)
        print("TEST 11: STREAM ANSWER")
        print("="*60)

        result = await adapter.execute(
            "stream_answer",
            query="Explain the difference between Raft and Paxos consensus",
        )

        print(f"Success: {result.success}")
        print(f"Latency: {result.latency_ms:.2f}ms")

        if result.success:
            data = result.data
            print(f"Streamed: {data.get('streamed', False)}")
            answer = data.get("answer", "")
            print(f"Answer length: {len(answer)} chars")
            print(f"Preview: {answer[:200]}...")

        assert result.success

    @pytest.mark.asyncio
    async def test_12_company_research(self, adapter):
        """Test comprehensive company research."""
        print("\n" + "="*60)
        print("TEST 12: COMPANY RESEARCH")
        print("="*60)

        result = await adapter.execute(
            "company_research",
            domain=TEST_QUERIES["company"],
        )

        print(f"Success: {result.success}")
        print(f"Latency: {result.latency_ms:.2f}ms")

        if result.success:
            data = result.data
            print(f"Company: {data.get('company')}")
            pages = data.get("pages", [])
            search_results = data.get("search_results", [])

            print(f"Pages crawled: {len(pages)}")
            print(f"Search results: {len(search_results)}")

            for p in pages[:2]:
                print(f"\n  Page: {p.get('title', 'N/A')[:40]}")
                if p.get('summary'):
                    print(f"  Summary: {p.get('summary')[:80]}...")

        assert result.success

    @pytest.mark.asyncio
    async def test_13_linkedin_search_people(self, adapter):
        """Test LinkedIn people search."""
        print("\n" + "="*60)
        print("TEST 13: LINKEDIN SEARCH - PEOPLE")
        print("="*60)

        result = await adapter.execute(
            "linkedin_search",
            query=TEST_QUERIES["person"],
            search_type="people",
            num_results=5,
        )

        print(f"Success: {result.success}")
        print(f"Latency: {result.latency_ms:.2f}ms")

        if result.success:
            results = result.data.get("results", [])
            print(f"LinkedIn profiles found: {len(results)}")
            for r in results[:5]:
                print(f"  - {r.get('title', 'N/A')[:50]}")
                print(f"    {r.get('url', 'N/A')}")

        assert result.success

    @pytest.mark.asyncio
    async def test_14_linkedin_search_companies(self, adapter):
        """Test LinkedIn company search."""
        print("\n" + "="*60)
        print("TEST 14: LINKEDIN SEARCH - COMPANIES")
        print("="*60)

        result = await adapter.execute(
            "linkedin_search",
            query="Anthropic AI company",
            search_type="company",
            num_results=5,
        )

        print(f"Success: {result.success}")
        print(f"Latency: {result.latency_ms:.2f}ms")

        if result.success:
            results = result.data.get("results", [])
            print(f"LinkedIn companies found: {len(results)}")
            for r in results[:5]:
                print(f"  - {r.get('title', 'N/A')[:50]}")

        assert result.success

    @pytest.mark.asyncio
    async def test_15_code_search(self, adapter):
        """Test code search on GitHub/StackOverflow/etc."""
        print("\n" + "="*60)
        print("TEST 15: CODE SEARCH")
        print("="*60)

        result = await adapter.execute(
            "code_search",
            query=TEST_QUERIES["code"],
            language="Python",
            num_results=10,
        )

        print(f"Success: {result.success}")
        print(f"Latency: {result.latency_ms:.2f}ms")

        if result.success:
            results = result.data.get("results", [])
            print(f"Code results: {len(results)}")
            for r in results[:5]:
                print(f"\n  {r.get('title', 'N/A')[:50]}")
                print(f"  URL: {r.get('url', 'N/A')[:60]}")
                if r.get('highlights'):
                    print(f"  Code: {r.get('highlights')[0][:80]}...")

        assert result.success

    @pytest.mark.asyncio
    async def test_16_code_search_specific_repos(self, adapter):
        """Test code search with specific repo domains."""
        print("\n" + "="*60)
        print("TEST 16: CODE SEARCH - SPECIFIC REPOS")
        print("="*60)

        result = await adapter.execute(
            "code_search",
            query="StateGraph workflow",
            language="Python",
            num_results=10,
            include_domains=[
                "github.com/langchain-ai",
                "github.com/anthropics",
            ],
        )

        print(f"Success: {result.success}")
        print(f"Latency: {result.latency_ms:.2f}ms")

        if result.success:
            results = result.data.get("results", [])
            print(f"Filtered results: {len(results)}")
            for r in results[:5]:
                print(f"  - {r.get('url', 'N/A')}")

        assert result.success

    @pytest.mark.asyncio
    async def test_17_research_async(self, adapter):
        """Test async research with structured output."""
        print("\n" + "="*60)
        print("TEST 17: RESEARCH (Async Structured)")
        print("="*60)

        result = await adapter.execute(
            "research",
            instructions=TEST_QUERIES["research"],
            output_schema={
                "type": "object",
                "properties": {
                    "databases": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "pros": {"type": "array", "items": {"type": "string"}},
                                "cons": {"type": "array", "items": {"type": "string"}},
                            }
                        }
                    },
                    "recommendation": {"type": "string"},
                }
            },
        )

        print(f"Success: {result.success}")
        print(f"Latency: {result.latency_ms:.2f}ms")

        if result.success:
            data = result.data
            print(f"Task ID: {data.get('task_id', 'N/A')}")
            print(f"Fallback mode: {data.get('fallback', False)}")

            if data.get('result'):
                print(f"Result type: {type(data['result'])}")
                if isinstance(data['result'], dict):
                    print(f"Result keys: {list(data['result'].keys())}")
                else:
                    print(f"Result: {str(data['result'])[:200]}...")

            if data.get('citations'):
                print(f"Citations: {len(data['citations'])}")

        assert result.success

    @pytest.mark.asyncio
    async def test_18_health_check(self, adapter):
        """Test health check endpoint."""
        print("\n" + "="*60)
        print("TEST 18: HEALTH CHECK")
        print("="*60)

        result = await adapter.health_check()

        print(f"Success: {result.success}")

        if result.success:
            data = result.data
            print(f"Status: {data.get('status')}")
            if data.get('stats'):
                stats = data['stats']
                print(f"\nAccumulated Stats:")
                print(f"  Searches: {stats.get('searches', 0)}")
                print(f"  Contents fetched: {stats.get('contents_fetched', 0)}")
                print(f"  Total results: {stats.get('total_results', 0)}")
                print(f"  Company researches: {stats.get('company_researches', 0)}")
                print(f"  Code searches: {stats.get('code_searches', 0)}")
                print(f"  LinkedIn searches: {stats.get('linkedin_searches', 0)}")
                print(f"  Research tasks: {stats.get('research_tasks', 0)}")
        else:
            print(f"Error: {result.error}")

        assert result.success

    @pytest.mark.asyncio
    async def test_19_news_search(self, adapter):
        """Test news-focused search."""
        print("\n" + "="*60)
        print("TEST 19: NEWS SEARCH")
        print("="*60)

        result = await adapter.execute(
            "search",
            query="AI regulation 2024",
            type="auto",
            num_results=10,
            category="news",
            start_published_date="2024-06-01",
        )

        print(f"Success: {result.success}")
        print(f"Latency: {result.latency_ms:.2f}ms")

        if result.success:
            results = result.data.get("results", [])
            print(f"News articles: {len(results)}")
            for r in results[:5]:
                print(f"  - {r.get('title', 'N/A')[:50]}")
                print(f"    Date: {r.get('published_date', 'N/A')}")

        assert result.success

    @pytest.mark.asyncio
    async def test_20_livecrawl_modes(self, adapter):
        """Test different livecrawl modes."""
        print("\n" + "="*60)
        print("TEST 20: LIVECRAWL MODES")
        print("="*60)

        modes = ["never", "fallback", "preferred", "always"]

        for mode in modes:
            result = await adapter.execute(
                "search",
                query="latest Claude API updates",
                type="auto",
                num_results=3,
                livecrawl=mode,
            )
            print(f"\n  Mode '{mode}': {'✓' if result.success else '✗'} ({result.latency_ms:.0f}ms)")

        print("\nAll livecrawl modes tested")


async def run_all_tests():
    """Run all tests manually."""
    print("\n" + "="*70)
    print("EXA ADAPTER - FULL FEATURE TEST SUITE")
    print("="*70)
    print(f"SDK Available: {EXA_AVAILABLE}")
    print(f"API Key Set: {bool(os.getenv('EXA_API_KEY'))}")
    print("="*70)

    adapter = ExaAdapter()
    result = await adapter.initialize({})
    print(f"\nInitialization: {result.success}")
    if result.data:
        print(f"Features: {result.data.get('features', [])}")
        print(f"Search Types: {result.data.get('search_types', [])}")

    # Run test suite
    test_suite = TestExaAllFeatures()

    tests = [
        ("00_initialization", test_suite.test_00_initialization),
        ("01_search_fast", test_suite.test_01_search_fast),
        ("02_search_auto", test_suite.test_02_search_auto),
        ("03_search_neural", test_suite.test_03_search_neural),
        ("04_search_deep", test_suite.test_04_search_deep),
        ("05_search_filters", test_suite.test_05_search_with_filters),
        ("06_search_github", test_suite.test_06_search_github),
        ("07_get_contents", test_suite.test_07_get_contents),
        ("08_find_similar", test_suite.test_08_find_similar),
        ("09_search_and_contents", test_suite.test_09_search_and_contents),
        ("10_answer", test_suite.test_10_answer),
        ("11_stream_answer", test_suite.test_11_stream_answer),
        ("12_company_research", test_suite.test_12_company_research),
        ("13_linkedin_people", test_suite.test_13_linkedin_search_people),
        ("14_linkedin_companies", test_suite.test_14_linkedin_search_companies),
        ("15_code_search", test_suite.test_15_code_search),
        ("16_code_search_repos", test_suite.test_16_code_search_specific_repos),
        ("17_research_async", test_suite.test_17_research_async),
        ("18_health_check", test_suite.test_18_health_check),
        ("19_news_search", test_suite.test_19_news_search),
        ("20_livecrawl_modes", test_suite.test_20_livecrawl_modes),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            await test_func(adapter)
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ FAILED: {name}")
            print(f"   Error: {e}")

    await adapter.shutdown()

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"Total: {passed + failed}")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
