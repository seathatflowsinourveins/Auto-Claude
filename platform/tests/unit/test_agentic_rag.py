"""
Unit Tests for Agentic RAG Loop

Tests cover:
- State machine transitions
- Query decomposition
- Tool selection
- Response evaluation
- Result fusion
- Integration scenarios
"""

from __future__ import annotations

import asyncio
import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch

# Import module under test
from core.rag.agentic_rag import (
    AgenticRAG,
    AgenticRAGConfig,
    AgenticRAGResult,
    AgentState,
    QueryType,
    EvaluationDecision,
    SubQuery,
    RetrievedContext,
    RetrievalResult,
    EvaluationResult,
    GenerationResult,
    QueryDecomposer,
    ToolSelector,
    ResponseEvaluator,
    ResultFusion,
    BaseRetrievalTool,
    ExaSearchTool,
    TavilySearchTool,
    MemorySearchTool,
    create_agentic_rag,
    AgenticRAGPrompts,
)


# =============================================================================
# MOCK PROVIDERS
# =============================================================================

class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.calls: List[Dict[str, Any]] = []
        self.default_response = "Mock response"

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        self.calls.append({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        })

        # Check for specific response patterns
        for key, response in self.responses.items():
            if key in prompt:
                return response

        return self.default_response


class MockRetrievalTool(BaseRetrievalTool):
    """Mock retrieval tool for testing."""

    def __init__(
        self,
        name: str = "mock_tool",
        results: Optional[List[Dict[str, Any]]] = None
    ):
        super().__init__(
            name=name,
            description=f"Mock {name} tool for testing",
            query_types=["general", "factual"]
        )
        self.results = results or []
        self.retrieve_calls: List[Dict[str, Any]] = []

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        self.retrieve_calls.append({
            "query": query,
            "top_k": top_k,
            **kwargs
        })

        if self.results:
            return self.results[:top_k]

        # Default mock results
        return [
            {
                "content": f"Mock content for: {query}",
                "score": 0.9,
                "metadata": {"source": self.name}
            }
        ]


class MockAdapter:
    """Mock adapter for Exa/Tavily testing."""

    def __init__(self, results: Optional[Dict[str, Any]] = None):
        self.results = results or {"results": []}
        self.success = True

    async def execute(self, operation: str, **kwargs) -> MagicMock:
        result = MagicMock()
        result.success = self.success
        result.data = self.results
        result.error = None if self.success else "Mock error"
        return result


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def mock_llm():
    """Create mock LLM provider."""
    return MockLLMProvider({
        "QUERY_TYPE": "QUERY_TYPE: factual\nNEEDS_DECOMPOSITION: no\nREASONING: Simple query\nSUGGESTED_SOURCES: memory",
        "Break down": "1: What is the main concept?\n2: How does it work?",
        "select the best tools": "SELECTED_TOOLS: memory, exa\nREASONING: Good coverage",
        "Evaluate": "RELEVANCE: 0.8\nCOMPLETENESS: 0.85\nACCURACY: 0.9\nCOHERENCE: 0.85\nISSUES: none\nSUGGESTIONS: none\nDECISION: COMPLETE",
    })


@pytest.fixture
def mock_tool():
    """Create mock retrieval tool."""
    return MockRetrievalTool(
        name="test_tool",
        results=[
            {"content": "Test content 1", "score": 0.9, "metadata": {"id": "1"}},
            {"content": "Test content 2", "score": 0.8, "metadata": {"id": "2"}},
        ]
    )


@pytest.fixture
def config():
    """Create test configuration."""
    return AgenticRAGConfig(
        max_iterations=3,
        confidence_threshold=0.7,
        enable_query_decomposition=True,
        enable_reflection=True,
        enable_tool_selection=True,
        top_k_per_source=5,
        max_total_documents=10,
        timeout_seconds=30.0,
    )


@pytest.fixture
def agentic_rag(mock_llm, mock_tool, config):
    """Create AgenticRAG instance for testing."""
    return AgenticRAG(
        llm=mock_llm,
        tools=[mock_tool],
        config=config,
    )


# =============================================================================
# QUERY DECOMPOSER TESTS
# =============================================================================

class TestQueryDecomposer:
    """Tests for QueryDecomposer."""

    @pytest.mark.asyncio
    async def test_analyze_query_simple(self, mock_llm):
        """Test query analysis for simple query."""
        decomposer = QueryDecomposer(mock_llm)

        query_type, needs_decomp, sources = await decomposer.analyze_query(
            "What is machine learning?"
        )

        assert query_type == QueryType.FACTUAL
        assert needs_decomp is False
        assert len(mock_llm.calls) == 1

    @pytest.mark.asyncio
    async def test_analyze_query_complex(self, mock_llm):
        """Test query analysis for complex query."""
        mock_llm.responses["QUERY_TYPE"] = (
            "QUERY_TYPE: multi_hop\n"
            "NEEDS_DECOMPOSITION: yes\n"
            "REASONING: Complex multi-part question\n"
            "SUGGESTED_SOURCES: exa, tavily"
        )

        decomposer = QueryDecomposer(mock_llm)

        query_type, needs_decomp, sources = await decomposer.analyze_query(
            "Compare ML and DL approaches for NLP and explain their trade-offs"
        )

        assert needs_decomp is True

    @pytest.mark.asyncio
    async def test_decompose_query(self, mock_llm):
        """Test query decomposition."""
        decomposer = QueryDecomposer(mock_llm)

        sub_queries = await decomposer.decompose(
            "Compare Python and JavaScript for web development"
        )

        assert len(sub_queries) >= 1
        assert all(isinstance(sq, SubQuery) for sq in sub_queries)

    @pytest.mark.asyncio
    async def test_decompose_with_dependencies(self, mock_llm):
        """Test decomposition with dependencies."""
        mock_llm.responses["Break down"] = (
            "1: What is Python?\n"
            "2: What is JavaScript?\n"
            "[depends on 1,2] 3: How do they compare?"
        )

        decomposer = QueryDecomposer(mock_llm)

        sub_queries = await decomposer.decompose(
            "Compare Python and JavaScript"
        )

        assert len(sub_queries) == 3
        # Third query should have dependencies
        assert len(sub_queries[2].depends_on) > 0

    def test_infer_query_type(self, mock_llm):
        """Test query type inference."""
        decomposer = QueryDecomposer(mock_llm)

        assert decomposer._infer_query_type("How to implement a function") == QueryType.CODE
        assert decomposer._infer_query_type("Latest AI news today") == QueryType.NEWS
        assert decomposer._infer_query_type("Compare React vs Vue") == QueryType.COMPARISON
        assert decomposer._infer_query_type("Explain how RAG works") == QueryType.EXPLANATION


# =============================================================================
# TOOL SELECTOR TESTS
# =============================================================================

class TestToolSelector:
    """Tests for ToolSelector."""

    def test_select_by_query_type(self, mock_llm, mock_tool):
        """Test rule-based tool selection."""
        selector = ToolSelector(mock_llm, [mock_tool])

        selected = selector.select_by_query_type(QueryType.GENERAL)

        assert len(selected) >= 1
        assert mock_tool in selected

    @pytest.mark.asyncio
    async def test_select_with_llm(self, mock_llm, mock_tool):
        """Test LLM-based tool selection."""
        selector = ToolSelector(mock_llm, [mock_tool])

        selected = await selector.select_with_llm(
            "What is machine learning?",
            QueryType.FACTUAL
        )

        assert len(selected) >= 1

    @pytest.mark.asyncio
    async def test_select_fallback_on_error(self, mock_llm, mock_tool):
        """Test fallback to rule-based on LLM error."""
        # Make LLM fail
        async def failing_generate(*args, **kwargs):
            raise Exception("LLM error")

        mock_llm.generate = failing_generate
        selector = ToolSelector(mock_llm, [mock_tool])

        selected = await selector.select_with_llm(
            "Test query",
            QueryType.GENERAL
        )

        # Should fallback to rule-based selection
        assert len(selected) >= 1


# =============================================================================
# RESPONSE EVALUATOR TESTS
# =============================================================================

class TestResponseEvaluator:
    """Tests for ResponseEvaluator."""

    @pytest.mark.asyncio
    async def test_evaluate_good_response(self, mock_llm, config):
        """Test evaluation of a good response."""
        evaluator = ResponseEvaluator(mock_llm)

        result = await evaluator.evaluate(
            query="What is RAG?",
            response="RAG is Retrieval-Augmented Generation...",
            context="Context about RAG...",
            iteration=1,
            config=config
        )

        assert isinstance(result, EvaluationResult)
        assert result.decision == EvaluationDecision.COMPLETE
        assert result.confidence > 0.5

    @pytest.mark.asyncio
    async def test_evaluate_poor_response(self, mock_llm, config):
        """Test evaluation of a poor response."""
        mock_llm.responses["Evaluate"] = (
            "RELEVANCE: 0.2\n"
            "COMPLETENESS: 0.3\n"
            "ACCURACY: 0.4\n"
            "COHERENCE: 0.5\n"
            "ISSUES: Missing key information, off-topic\n"
            "SUGGESTIONS: Focus on the query, add examples\n"
            "DECISION: RETRY_RETRIEVE"
        )

        evaluator = ResponseEvaluator(mock_llm)

        result = await evaluator.evaluate(
            query="What is quantum computing?",
            response="I like coffee.",
            context="Quantum computing...",
            iteration=1,
            config=config
        )

        assert result.relevance_score < 0.5
        assert result.decision == EvaluationDecision.RETRY_RETRIEVE
        assert len(result.issues) > 0

    @pytest.mark.asyncio
    async def test_evaluate_at_max_iterations(self, mock_llm, config):
        """Test evaluation forces complete at max iterations."""
        mock_llm.responses["Evaluate"] = (
            "RELEVANCE: 0.6\n"
            "COMPLETENESS: 0.5\n"
            "ACCURACY: 0.6\n"
            "COHERENCE: 0.6\n"
            "ISSUES: Could be better\n"
            "SUGGESTIONS: Add more detail\n"
            "DECISION: REFINE"
        )

        evaluator = ResponseEvaluator(mock_llm)

        result = await evaluator.evaluate(
            query="Test query",
            response="Test response",
            context="Test context",
            iteration=config.max_iterations,  # At max
            config=config
        )

        # Should force complete at max iterations
        assert result.decision == EvaluationDecision.COMPLETE


# =============================================================================
# RESULT FUSION TESTS
# =============================================================================

class TestResultFusion:
    """Tests for ResultFusion."""

    def test_rrf_fusion_single_source(self):
        """Test RRF fusion with single source."""
        contexts = [
            RetrievedContext(content="Doc 1", source="a", score=0.9),
            RetrievedContext(content="Doc 2", source="a", score=0.8),
        ]

        fused = ResultFusion.rrf_fusion([[contexts[0], contexts[1]]])

        assert len(fused) == 2
        # First should have higher RRF score
        assert fused[0].score > fused[1].score

    def test_rrf_fusion_multiple_sources(self):
        """Test RRF fusion with multiple sources."""
        list1 = [
            RetrievedContext(content="Doc 1", source="a", score=0.9),
            RetrievedContext(content="Doc 2", source="a", score=0.8),
        ]
        list2 = [
            RetrievedContext(content="Doc 3", source="b", score=0.95),
            RetrievedContext(content="Doc 1", source="b", score=0.7),  # Duplicate
        ]

        fused = ResultFusion.rrf_fusion([list1, list2])

        # Should deduplicate
        contents = [f.content for f in fused]
        assert contents.count("Doc 1") == 1

        # Doc 1 should be ranked higher due to appearing in both lists
        assert fused[0].content == "Doc 1"

    def test_rrf_fusion_empty(self):
        """Test RRF fusion with empty input."""
        fused = ResultFusion.rrf_fusion([])
        assert fused == []

    def test_weighted_fusion(self):
        """Test weighted fusion."""
        list1 = [
            RetrievedContext(content="Doc 1", source="a", score=0.8),
        ]
        list2 = [
            RetrievedContext(content="Doc 2", source="b", score=0.9),
        ]

        # Give higher weight to second source
        fused = ResultFusion.weighted_fusion([list1, list2], weights=[1.0, 2.0])

        # Doc 2 should be first due to higher weight
        assert fused[0].content == "Doc 2"


# =============================================================================
# AGENTIC RAG INTEGRATION TESTS
# =============================================================================

class TestAgenticRAGIntegration:
    """Integration tests for AgenticRAG."""

    @pytest.mark.asyncio
    async def test_run_simple_query(self, agentic_rag):
        """Test running a simple query."""
        result = await agentic_rag.run("What is machine learning?")

        assert isinstance(result, AgenticRAGResult)
        assert result.response
        assert result.iterations >= 1
        assert AgentState.PLAN in result.states_visited
        assert AgentState.RETRIEVE in result.states_visited
        assert AgentState.GENERATE in result.states_visited

    @pytest.mark.asyncio
    async def test_run_with_initial_context(self, agentic_rag):
        """Test running with pre-provided context."""
        result = await agentic_rag.run(
            "Summarize this information",
            initial_context=["Context 1: Important fact", "Context 2: Another fact"]
        )

        assert result.documents_retrieved >= 2

    @pytest.mark.asyncio
    async def test_state_transitions(self, agentic_rag):
        """Test state machine transitions."""
        result = await agentic_rag.run("Test query")

        # Check expected state sequence
        states = result.states_visited
        assert states[0] == AgentState.INIT
        assert AgentState.PLAN in states
        assert AgentState.COMPLETE in states or AgentState.ERROR in states

    @pytest.mark.asyncio
    async def test_max_iterations_limit(self, mock_llm, mock_tool):
        """Test that max iterations is respected."""
        # Make evaluation always suggest refine
        mock_llm.responses["Evaluate"] = (
            "RELEVANCE: 0.5\n"
            "COMPLETENESS: 0.5\n"
            "ACCURACY: 0.5\n"
            "COHERENCE: 0.5\n"
            "ISSUES: needs improvement\n"
            "SUGGESTIONS: try again\n"
            "DECISION: REFINE"
        )

        config = AgenticRAGConfig(max_iterations=2)
        rag = AgenticRAG(llm=mock_llm, tools=[mock_tool], config=config)

        result = await rag.run("Test query")

        assert result.iterations <= 2

    @pytest.mark.asyncio
    async def test_confidence_early_stopping(self, mock_llm, mock_tool):
        """Test early stopping when confidence threshold is met."""
        # Make evaluation return high confidence
        mock_llm.responses["Evaluate"] = (
            "RELEVANCE: 0.95\n"
            "COMPLETENESS: 0.95\n"
            "ACCURACY: 0.95\n"
            "COHERENCE: 0.95\n"
            "ISSUES: none\n"
            "SUGGESTIONS: none\n"
            "DECISION: COMPLETE"
        )

        config = AgenticRAGConfig(
            max_iterations=5,
            confidence_threshold=0.8
        )
        rag = AgenticRAG(llm=mock_llm, tools=[mock_tool], config=config)

        result = await rag.run("Test query")

        # Should complete early
        assert result.iterations <= 2
        assert result.confidence >= 0.8

    @pytest.mark.asyncio
    async def test_query_decomposition(self, mock_llm, mock_tool):
        """Test query decomposition for complex queries."""
        # Configure for decomposition
        mock_llm.responses["QUERY_TYPE"] = (
            "QUERY_TYPE: multi_hop\n"
            "NEEDS_DECOMPOSITION: yes\n"
            "REASONING: Complex query\n"
            "SUGGESTED_SOURCES: memory"
        )

        config = AgenticRAGConfig(enable_query_decomposition=True)
        rag = AgenticRAG(llm=mock_llm, tools=[mock_tool], config=config)

        result = await rag.run("Compare A and B, then explain implications")

        assert len(result.sub_queries) >= 1

    @pytest.mark.asyncio
    async def test_tool_retrieval(self, agentic_rag, mock_tool):
        """Test that tools are called during retrieval."""
        await agentic_rag.run("Test query")

        assert len(mock_tool.retrieve_calls) >= 1

    @pytest.mark.asyncio
    async def test_add_tool(self, agentic_rag):
        """Test adding tools dynamically."""
        new_tool = MockRetrievalTool(name="new_tool")

        agentic_rag.add_tool(new_tool)

        assert new_tool in agentic_rag.tools


# =============================================================================
# TOOL WRAPPER TESTS
# =============================================================================

class TestToolWrappers:
    """Tests for tool wrapper classes."""

    @pytest.mark.asyncio
    async def test_exa_search_tool(self):
        """Test ExaSearchTool wrapper."""
        mock_adapter = MockAdapter({
            "results": [
                {"text": "Result 1", "score": 0.9, "url": "http://example.com"},
            ]
        })

        tool = ExaSearchTool(mock_adapter)

        results = await tool.retrieve("test query", top_k=5)

        assert len(results) >= 1
        assert "content" in results[0]
        assert "score" in results[0]

    @pytest.mark.asyncio
    async def test_tavily_search_tool(self):
        """Test TavilySearchTool wrapper."""
        mock_adapter = MockAdapter({
            "answer": "Synthesized answer",
            "results": [
                {"content": "Result 1", "score": 0.8, "url": "http://example.com"},
            ]
        })

        tool = TavilySearchTool(mock_adapter)

        results = await tool.retrieve("test query", top_k=5)

        # Should include synthesized answer
        assert len(results) >= 2
        assert any(r["content"] == "Synthesized answer" for r in results)

    @pytest.mark.asyncio
    async def test_memory_search_tool(self):
        """Test MemorySearchTool wrapper."""
        # Mock memory backend
        mock_backend = AsyncMock()
        mock_backend.search.return_value = [
            MagicMock(content="Memory result", score=0.9, id="1", metadata={})
        ]

        tool = MemorySearchTool(mock_backend)

        results = await tool.retrieve("test query", top_k=5)

        assert len(results) >= 1
        mock_backend.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        """Test tool error handling."""
        mock_adapter = MockAdapter()
        mock_adapter.success = False
        mock_adapter.results = {"error": "API error"}

        tool = ExaSearchTool(mock_adapter)

        results = await tool.retrieve("test query")

        # Should return empty on error
        assert results == []


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactory:
    """Tests for factory function."""

    def test_create_agentic_rag_minimal(self, mock_llm):
        """Test creating AgenticRAG with minimal config."""
        rag = create_agentic_rag(llm=mock_llm)

        assert rag is not None
        assert len(rag.tools) == 0

    def test_create_agentic_rag_with_tools(self, mock_llm):
        """Test creating AgenticRAG with tools."""
        mock_exa = MockAdapter()
        mock_tavily = MockAdapter()

        rag = create_agentic_rag(
            llm=mock_llm,
            exa_adapter=mock_exa,
            tavily_adapter=mock_tavily,
        )

        assert len(rag.tools) == 2

    def test_create_agentic_rag_with_config(self, mock_llm):
        """Test creating AgenticRAG with custom config."""
        config = AgenticRAGConfig(max_iterations=10)

        rag = create_agentic_rag(llm=mock_llm, config=config)

        assert rag.config.max_iterations == 10


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_query(self, agentic_rag):
        """Test handling of empty query."""
        result = await agentic_rag.run("")

        assert result is not None

    @pytest.mark.asyncio
    async def test_no_tools(self, mock_llm, config):
        """Test running without any tools."""
        rag = AgenticRAG(llm=mock_llm, tools=[], config=config)

        result = await rag.run("Test query")

        assert result is not None
        assert result.retrieval_sources == []

    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_llm, mock_tool):
        """Test timeout handling."""
        # Create very short timeout
        config = AgenticRAGConfig(timeout_seconds=0.001)
        rag = AgenticRAG(llm=mock_llm, tools=[mock_tool], config=config)

        # Add delay to LLM
        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(0.1)
            return "Response"

        mock_llm.generate = slow_generate

        result = await rag.run("Test query")

        # Should complete (possibly with partial results) without crashing
        assert result is not None

    @pytest.mark.asyncio
    async def test_llm_error_handling(self, mock_tool, config):
        """Test handling of LLM errors."""
        failing_llm = MockLLMProvider()

        async def failing_generate(*args, **kwargs):
            raise Exception("LLM error")

        failing_llm.generate = failing_generate

        rag = AgenticRAG(llm=failing_llm, tools=[mock_tool], config=config)

        result = await rag.run("Test query")

        # Should handle error gracefully
        assert result is not None
        assert AgentState.ERROR in result.states_visited or result.response == ""

    @pytest.mark.asyncio
    async def test_context_deduplication(self, agentic_rag):
        """Test context deduplication."""
        contexts = [
            RetrievedContext(content="Same content", source="a", score=0.9),
            RetrievedContext(content="Same content", source="b", score=0.8),
            RetrievedContext(content="Different content", source="a", score=0.7),
        ]

        deduped = agentic_rag._deduplicate_contexts(contexts, max_count=10)

        # Should have 2 unique contents
        assert len(deduped) == 2

    @pytest.mark.asyncio
    async def test_context_limit(self, agentic_rag):
        """Test context count limiting."""
        contexts = [
            RetrievedContext(content=f"Content {i}", source="a", score=0.9 - i * 0.1)
            for i in range(20)
        ]

        deduped = agentic_rag._deduplicate_contexts(contexts, max_count=5)

        assert len(deduped) == 5
        # Should keep highest scoring
        assert deduped[0].content == "Content 0"


# =============================================================================
# PROMPT TEMPLATE TESTS
# =============================================================================

class TestPromptTemplates:
    """Tests for prompt templates."""

    def test_query_analysis_prompt(self):
        """Test query analysis prompt formatting."""
        prompt = AgenticRAGPrompts.QUERY_ANALYSIS.format(query="Test query")

        assert "Test query" in prompt
        assert "QUERY_TYPE" in prompt

    def test_decomposition_prompt(self):
        """Test decomposition prompt formatting."""
        prompt = AgenticRAGPrompts.QUERY_DECOMPOSITION.format(
            query="Complex query",
            context="Some context"
        )

        assert "Complex query" in prompt
        assert "Some context" in prompt

    def test_generate_response_prompt(self):
        """Test generation prompt formatting."""
        prompt = AgenticRAGPrompts.GENERATE_RESPONSE.format(
            query="User question",
            context="Retrieved context"
        )

        assert "User question" in prompt
        assert "Retrieved context" in prompt

    def test_evaluate_response_prompt(self):
        """Test evaluation prompt formatting."""
        prompt = AgenticRAGPrompts.EVALUATE_RESPONSE.format(
            query="Query",
            response="Response",
            context="Context"
        )

        assert "RELEVANCE" in prompt
        assert "COMPLETENESS" in prompt
        assert "DECISION" in prompt


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
